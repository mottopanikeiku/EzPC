#!/usr/bin/env python3
"""Produce one fail-closed, source-bound known-zero full ResNet18 graph run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import io
import math
from decimal import Decimal, InvalidOperation
import os
import pathlib
import secrets
import shutil
import stat
import subprocess
import signal
import sys
import time
from typing import Any, NoReturn
import check_resnet18_graph_contract
import graph_build_provenance


ADAPTER_HEADER = (
    "key_source", "stock_key_items", "truncation_items", "remask_edges",
    "raw_key_bytes_per_party", "record_bytes_per_party", "p0_record_digest",
    "p1_record_digest", "total_us", "status",
)
PARTY_HEADER = (
    "party", "total_us", "linear_us", "secure_truncate_us",
    "stock_nonlinear_us", "global_pool_us", "terminal_us",
    "total_protocol_bytes_sent", "party_channel_bytes_sent",
    "stock_bytes_sent", "stock_bytes_received", "truncations", "handoffs",
    "dabits", "edabits", "logical_opened_bits", "triples",
    "stock_raw_key_bytes", "status",
)
CHECKER_HEADER = (
    "source_bindings", "linear_inputs", "linear_outputs",
    "secure_stochastic_truncations", "stock_key_stream", "residual_merges",
    "global_average_pool", "state_links", "terminal_output",
    "trace_contract", "counter_contract", "stock_nonlinear_key_source",
    "scope", "status",
)
CONTROL_HEADER = ("control", "expected_rejection", "no_partial_output", "status")
SCHEMA = "ringlpn-known-zero-full-resnet18-graph-v4"
EXPECTED_LINEAR_SCHEMA = "ringlpn-forward-linear-record-set-v4"
EXPECTED_SOURCE_SCHEMA = "ringlpn-full-linear-execution-v1"
GRAPH_APPROVAL_SCHEMA = graph_build_provenance.APPROVAL_SCHEMA
GRAPH_APPROVAL_SCOPE = graph_build_provenance.APPROVAL_SCOPE
GRAPH_BINARY_FILES = {
    "adapter": "test_stock_nonlinear_full_keygen",
    "runtime": "test_resnet18_full_graph",
    "contract_probe": "test_resnet18_graph_contract",
}
GRAPH_SOURCE_FILES = graph_build_provenance.MINIMUM_APPROVAL_SOURCES
RETAINED_EVIDENCE_PATHS = frozenset({
    "FULL_GRAPH.manifest",
    "adapter.csv",
    "adapter.log",
    "checker.csv",
    "checker.log",
    "controls.csv",
    "linear/private_inputs/binary_approval.json",
    "linear/private_inputs/linear_adapter_build_provenance.json",
    "linear/LINEAR_RECORD_SET.manifest",
    "party0.csv",
    "party0.log",
    "party1.csv",
    "party1.log",
    "private_inputs/graph_binary_approval.json",
    "private_inputs/linear_binary_approval.json",
    "private_inputs/linear_adapter_build_provenance.json",
    "private_inputs/graph_build_provenance.json",
})
ACTIVE_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"full-graph-runner: {message}")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

def snapshot_executable(source: pathlib.Path,
                        destination: pathlib.Path) -> dict[str, Any]:
    try:
        metadata = source.lstat()
    except OSError as error:
        fail(f"cannot inspect executable {source}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or \
            not os.access(source, os.X_OK):
        fail(f"executable source must be an executable regular non-symlink: {source}")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500
    )
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_fd)
        while chunk := os.read(source_fd, 1 << 20):
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short executable snapshot write")
                view = view[written:]
        after = os.fstat(source_fd)
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if (size != before.st_size or
                any(getattr(before, field) != getattr(after, field)
                    for field in stable_fields)):
            fail(f"executable source changed while snapshotting: {source}")
        os.fsync(destination_fd)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    os.chmod(destination, 0o500)
    return {"sha256": digest.hexdigest(), "size": size}

def write_private_bytes(path: pathlib.Path, payload: bytes,
                        mode: int) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, mode)


def require_digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_graph_approval(
        path: pathlib.Path, ringlpn: pathlib.Path,
        provenance_path: pathlib.Path,
) -> tuple[bytes, dict[str, dict[str, Any]], bytes, dict[str, Any]]:
    try:
        _, payload_value = graph_build_provenance.stable_file_binding(
            path, "graph binary approval", payload_required=True
        )
        assert payload_value is not None
        payload = payload_value
        document = graph_build_provenance.load_canonical_document(
            payload, "graph binary approval"
        )
    except graph_build_provenance.ProvenanceError as error:
        fail(str(error))
    if (set(document) != {
            "approval_digest", "binaries", "provenance", "schema", "scope",
            "sources", "validation"} or
            document.get("schema") != GRAPH_APPROVAL_SCHEMA or
            document.get("scope") != GRAPH_APPROVAL_SCOPE):
        fail("unsupported or inexact graph binary approval")
    claimed = require_digest(
        document.get("approval_digest"), "graph approval.approval_digest"
    )
    if self_digest(document, "approval_digest") != claimed:
        fail("invalid graph binary approval self-digest")
    sources = document.get("sources")
    if not isinstance(sources, dict) or \
            not set(GRAPH_SOURCE_FILES).issubset(sources):
        fail("graph binary approval omits a minimum source binding")
    repo = ringlpn.parents[1]
    for relative, binding in sources.items():
        if not isinstance(relative, str):
            fail("graph approval source paths must be strings")
        pure_relative = pathlib.PurePosixPath(relative)
        if (pure_relative.is_absolute() or ".." in pure_relative.parts or
                pure_relative.as_posix() != relative):
            fail(f"invalid graph approval source path: {relative}")
        if not isinstance(binding, dict) or set(binding) != {"sha256", "size"}:
            fail(f"invalid graph approval source binding: {relative}")
        require_digest(
            binding.get("sha256"), f"graph approval.sources.{relative}"
        )
        if (not isinstance(binding.get("size"), int) or
                isinstance(binding.get("size"), bool) or binding["size"] < 0):
            fail(f"invalid graph approval source size: {relative}")
        candidate = ringlpn / pure_relative
        try:
            graph_build_provenance.tracked_file(candidate, repo)
            actual = graph_build_provenance.file_binding(candidate)
        except graph_build_provenance.ProvenanceError as error:
            fail(f"invalid graph approval source {relative}: {error}")
        if actual != binding:
            fail(f"graph approval source binding differs: {relative}")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
            "path", "provenance_digest", "sha256", "size"}:
        fail("graph approval has no exact build provenance binding")
    try:
        declared_path = provenance_path.relative_to(ringlpn).as_posix()
    except ValueError:
        fail("graph build provenance must be under Ring-LPN")
    if provenance.get("path") != declared_path:
        fail("graph approval provenance path differs")
    try:
        provenance_payload, provenance_document = (
            graph_build_provenance.verify_manifest(
                provenance_path, repo,
                ringlpn / "build/graph-libraries",
            )
        )
    except graph_build_provenance.ProvenanceError as error:
        fail(f"graph build provenance rejected: {error}")
    expected_provenance_digest = require_digest(
        provenance.get("provenance_digest"),
        "graph approval.provenance.provenance_digest",
    )
    expected_provenance_sha = require_digest(
        provenance.get("sha256"), "graph approval.provenance.sha256"
    )
    if (not isinstance(provenance.get("size"), int) or
            isinstance(provenance.get("size"), bool) or
            provenance["size"] != len(provenance_payload)):
        fail("graph build provenance size differs from approval")
    if provenance_document["provenance_digest"] != expected_provenance_digest:
        fail("graph build provenance digest differs from approval")
    if hashlib.sha256(provenance_payload).hexdigest() != expected_provenance_sha:
        fail("graph build provenance file differs from approval")

    artifact_digests = {
        name: binding["sha256"]
        for name, binding in sorted(provenance_document["artifacts"].items())
    }
    validation = document.get("validation")
    if not isinstance(validation, dict) or \
            set(validation) != graph_build_provenance.VALIDATION_CHECKS:
        fail("graph approval validation receipt inventory differs")
    for check, receipt in validation.items():
        if not isinstance(receipt, dict) or set(receipt) != {
                "evidence", "receipt", "status"} or \
                receipt.get("status") != "pass":
            fail(f"invalid graph approval validation status: {check}")
        evidence = receipt.get("evidence")
        if (not isinstance(evidence, dict) or set(evidence) != {
                "artifacts", "check", "provenance_digest", "schema", "status"} or
                evidence.get("schema") !=
                graph_build_provenance.VALIDATION_RECEIPT_SCHEMA or
                evidence.get("check") != check or
                evidence.get("status") != "pass" or
                evidence.get("provenance_digest") !=
                provenance_document["provenance_digest"] or
                evidence.get("artifacts") != artifact_digests):
            fail(f"validation receipt does not bind graph provenance: {check}")
        receipt_binding = receipt.get("receipt")
        if not isinstance(receipt_binding, dict) or set(receipt_binding) != {
                "path", "sha256", "size"}:
            fail(f"invalid validation receipt binding: {check}")
        receipt_path = receipt_binding.get("path")
        pure_receipt = (
            pathlib.PurePosixPath(receipt_path)
            if isinstance(receipt_path, str) else None
        )
        if (pure_receipt is None or pure_receipt.is_absolute() or
                ".." in pure_receipt.parts or
                pure_receipt.as_posix() != receipt_path):
            fail(f"invalid validation receipt path: {check}")
        evidence_payload = canonical(evidence) + b"\n"
        if (require_digest(
                receipt_binding.get("sha256"),
                f"validation receipt {check}.sha256",
                ) != hashlib.sha256(evidence_payload).hexdigest() or
                receipt_binding.get("size") != len(evidence_payload)):
            fail(f"validation receipt byte binding differs: {check}")

    binaries = document.get("binaries")
    expected_names = set(GRAPH_BINARY_FILES.values())
    if not isinstance(binaries, dict) or set(binaries) != expected_names:
        fail("graph binary approval executable inventory differs")
    approved: dict[str, dict[str, Any]] = {}
    for label, name in GRAPH_BINARY_FILES.items():
        binding = binaries.get(name)
        if not isinstance(binding, dict) or set(binding) != {
                "path", "sha256", "size"} or binding.get("path") != f"bin/{name}":
            fail(f"invalid graph approval binary binding: {name}")
        require_digest(
            binding.get("sha256"), f"graph approval.binaries.{name}"
        )
        if (not isinstance(binding.get("size"), int) or
                isinstance(binding.get("size"), bool) or binding["size"] < 0):
            fail(f"invalid graph approval binary size: {name}")
        provenance_artifact = provenance_document["artifacts"][name]
        expected_artifact_path = graph_build_provenance.canonical_path_argument(
            str(ringlpn / "bin" / name), repo,
            ringlpn / "build/graph-libraries",
        )
        if (provenance_artifact["path"] != expected_artifact_path or
                binding["sha256"] != provenance_artifact["sha256"] or
                binding["size"] != provenance_artifact["size"]):
            fail(f"graph approval binary is detached from provenance: {name}")
        approved[label] = {
            "sha256": binding["sha256"],
            "size": binding["size"],
        }
    return payload, approved, provenance_payload, provenance_document


def record_digest(path: pathlib.Path) -> str:
    size = path.stat().st_size
    if size <= 32:
        fail(f"record too short: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        remaining = size - 32
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                fail(f"truncated record: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        trailing = source.read(32)
        if len(trailing) != 32 or source.read(1):
            fail(f"invalid record tail: {path}")
    calculated = digest.digest()
    if calculated != trailing:
        fail(f"record digest mismatch: {path}")
    return calculated.hex()


def canonical(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def self_digest(document: dict[str, Any], field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_json_once(path: pathlib.Path,
                   label: str) -> tuple[dict[str, Any], str, bytes]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                fail(f"{label} must be a nonempty regular non-symlink file")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1 << 20))
                if not chunk:
                    fail(f"{label} was shortened while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                fail(f"{label} was lengthened while reading")
            after = os.fstat(descriptor)
            stable_fields = (
                "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
            )
            if any(getattr(before, field) != getattr(after, field)
                   for field in stable_fields):
                fail(f"{label} changed while reading")
        finally:
            os.close(descriptor)
        payload = b"".join(chunks)
        data = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"invalid {label}: {error}")
    if not isinstance(data, dict):
        fail(f"{label} must be an object")
    return data, hashlib.sha256(payload).hexdigest(), payload

def summarize_linear_record_set(document: dict[str, Any]) -> dict[str, Any]:
    layers = document.get("layers")
    if not isinstance(layers, list) or len(layers) != 21:
        fail("linear record set does not contain 21 measured layers")

    def decimal_metric(row: dict[str, Any], field: str,
                       label: str) -> Decimal:
        value = row.get(field)
        try:
            parsed = Decimal(value)
        except (InvalidOperation, TypeError):
            fail(f"invalid {label}.{field} metric")
        if not parsed.is_finite() or parsed < 0:
            fail(f"invalid {label}.{field} metric")
        return parsed

    def integer_metric(row: dict[str, Any], field: str,
                       label: str) -> int:
        value = row.get(field)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            fail(f"invalid {label}.{field} metric")
        if parsed < 0 or str(parsed) != str(value):
            fail(f"invalid {label}.{field} metric")
        return parsed

    p0_total = Decimal(0)
    p1_total = Decimal(0)
    critical_total = Decimal(0)
    dealer_total = Decimal(0)
    online_total = Decimal(0)
    p0_protocol_bytes = 0
    p1_protocol_bytes = 0
    p0_dependency_rounds = 0
    p1_dependency_rounds = 0
    payload_bytes = 0
    peak_host_bytes = 0
    peak_gpu_bytes = 0
    minimum_gpu_free_bytes: int | None = None
    for expected_order, layer in enumerate(layers, 1):
        if not isinstance(layer, dict) or \
                layer.get("linear_order") != expected_order:
            fail("linear record-set layers are not canonically ordered")
        metrics = layer.get("metrics")
        if not isinstance(metrics, dict):
            fail(f"linear layer {expected_order} lacks metrics")
        p0 = metrics.get("party0")
        p1 = metrics.get("party1")
        checker = metrics.get("checker")
        if not all(isinstance(row, dict) for row in (p0, p1, checker)):
            fail(f"linear layer {expected_order} has malformed metrics")
        label = str(layer.get("layer", expected_order))
        p0_us = decimal_metric(p0, "total_us", f"{label}.party0")
        p1_us = decimal_metric(p1, "total_us", f"{label}.party1")
        p0_total += p0_us
        p1_total += p1_us
        critical_total += max(p0_us, p1_us)
        dealer_total += decimal_metric(
            checker, "matched_dealer_keygen_us", f"{label}.checker",
        )
        online_total += decimal_metric(
            checker, "checker_two_share_online_us", f"{label}.checker",
        )
        p0_protocol_bytes += integer_metric(
            p0, "protocol_bytes_sent", f"{label}.party0",
        )
        p1_protocol_bytes += integer_metric(
            p1, "protocol_bytes_sent", f"{label}.party1",
        )
        p0_dependency_rounds += integer_metric(
            p0, "protocol_dependency_rounds", f"{label}.party0",
        )
        p1_dependency_rounds += integer_metric(
            p1, "protocol_dependency_rounds", f"{label}.party1",
        )
        payload_bytes += integer_metric(
            checker, "final_payload_bytes_per_party", f"{label}.checker",
        )
        for role, row in (("party0", p0), ("party1", p1),
                          ("checker", checker)):
            peak_host_bytes = max(
                peak_host_bytes,
                integer_metric(row, "peak_host_rss_bytes", f"{label}.{role}"),
            )
            peak_gpu_bytes = max(
                peak_gpu_bytes,
                integer_metric(row, "peak_gpu_bytes", f"{label}.{role}"),
            )
            free_bytes = integer_metric(
                row, "min_gpu_free_bytes", f"{label}.{role}",
            )
            minimum_gpu_free_bytes = (
                free_bytes if minimum_gpu_free_bytes is None
                else min(minimum_gpu_free_bytes, free_bytes)
            )

    return {
        "layers": len(layers),
        "timing_unit": "microseconds",
        "sum_party0_total_us": str(p0_total),
        "sum_party1_total_us": str(p1_total),
        "sum_layer_critical_path_us": str(critical_total),
        "sum_matched_dealer_keygen_us": str(dealer_total),
        "sum_unchanged_online_us": str(online_total),
        "party0_protocol_bytes_sent": p0_protocol_bytes,
        "party1_protocol_bytes_sent": p1_protocol_bytes,
        "party0_protocol_dependency_rounds": p0_dependency_rounds,
        "party1_protocol_dependency_rounds": p1_dependency_rounds,
        "final_payload_bytes_per_party": payload_bytes,
        "max_observed_host_rss_bytes": peak_host_bytes,
        "max_observed_device_wide_gpu_used_bytes": peak_gpu_bytes,
        "minimum_observed_device_free_bytes": minimum_gpu_free_bytes,
        "aggregation": (
            "arithmetic sums over one retained row per layer; layer critical "
            "path is sum(max(party0.total_us,party1.total_us)); GPU values are "
            "device-wide observations and may include unrelated occupancy"
        ),
    }

LINEAR_PRIVATE_ARTIFACTS = (
    "p0_record", "p1_record", "p0_state", "p1_state",
)
LinearArtifactArgument = tuple[int, str, pathlib.Path, int, str]


def bind_linear_artifacts(
        document: dict[str, Any],
        root: pathlib.Path) -> list[LinearArtifactArgument]:
    layers = document.get("layers")
    if not isinstance(layers, list) or len(layers) != 21:
        fail("linear record set does not contain 21 artifact-bound layers")
    bound: list[LinearArtifactArgument] = []
    for order, layer in enumerate(layers, 1):
        if not isinstance(layer, dict):
            fail(f"linear layer {order} is not an object")
        name = layer.get("layer")
        operator = layer.get("operator")
        if not isinstance(name, str) or not name:
            fail(f"linear layer {order} has invalid name")
        if operator == "conv2d":
            suffix = ".conv"
        elif operator == "fc":
            suffix = ".fc"
        else:
            fail(f"linear layer {order} has invalid operator")
        prefix = f"{order:02d}_{name}"
        expected_paths = {
            "p0_record": f"{prefix}/party0/key_p0{suffix}",
            "p1_record": f"{prefix}/party1/key_p1{suffix}",
            "p0_state": f"{prefix}/party0/mask.state",
            "p1_state": f"{prefix}/party1/mask.state",
        }
        artifacts = layer.get("artifacts")
        if not isinstance(artifacts, dict):
            fail(f"linear layer {order} lacks artifact bindings")
        for label in LINEAR_PRIVATE_ARTIFACTS:
            binding = artifacts.get(label)
            diagnostic = f"linear layer {order} {label}"
            if not isinstance(binding, dict) or \
                    set(binding) != {"path", "bytes", "sha256"}:
                fail(f"{diagnostic} must bind path/bytes/SHA")
            relative_text = binding.get("path")
            if relative_text != expected_paths[label]:
                fail(f"{diagnostic} path mismatch")
            relative = pathlib.PurePosixPath(relative_text)
            if relative.is_absolute() or "." in relative.parts or \
                    ".." in relative.parts:
                fail(f"{diagnostic} has unsafe path")
            byte_count = binding.get("bytes")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or \
                    byte_count <= 0:
                fail(f"{diagnostic} has invalid byte count")
            digest = require_digest(binding.get("sha256"),
                                    f"{diagnostic}.sha256")
            bound.append((
                order, label, root.joinpath(*relative.parts),
                byte_count, digest,
            ))
    return bound


def append_linear_artifact_arguments(
        command: list[str],
        artifacts: list[LinearArtifactArgument]) -> None:
    for order, label, path, byte_count, digest in artifacts:
        command.extend([
            "--linear-artifact", str(order), label, str(path),
            str(byte_count), digest,
        ])


def write_atomic(path: pathlib.Path, data: bytes, mode: int) -> None:
    temporary = pathlib.Path(str(path) + ".tmp")
    if path.exists() or temporary.exists():
        fail(f"refusing stale publication path: {path}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    published = False
    try:
        with os.fdopen(fd, "wb", closefd=True) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
            fsync_directory(path.parent)
        raise

def fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remove_private_scratch(path: pathlib.Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        fail(f"cannot inspect {label} before cleanup: {error}")
    if (stat.S_ISLNK(metadata.st_mode) or
            not stat.S_ISDIR(metadata.st_mode) or
            metadata.st_uid != os.geteuid() or
            stat.S_IMODE(metadata.st_mode) & 0o077):
        fail(f"refusing to clean non-owner-private {label}: {path}")
    shutil.rmtree(path)


def publish_retained_summary(
        summary_root: pathlib.Path,
        retained: dict[str, bytes],
        manifest_digest: str,
        declared_provenance_path: str) -> None:
    if set(retained) != RETAINED_EVIDENCE_PATHS:
        missing = sorted(RETAINED_EVIDENCE_PATHS.difference(retained))
        unexpected = sorted(set(retained).difference(RETAINED_EVIDENCE_PATHS))
        fail(
            "retained evidence allowlist differs"
            f" (missing={missing}, unexpected={unexpected})"
        )
    pure_declared = pathlib.PurePosixPath(declared_provenance_path)
    if (pure_declared.is_absolute() or ".." in pure_declared.parts or
            pure_declared.as_posix() != declared_provenance_path):
        fail("graph approval provenance relocation source is not canonical")
    staging = summary_root.parent / (
        f".{summary_root.name}.staging-{secrets.token_hex(8)}"
    )
    staging.mkdir(mode=0o700)
    os.chmod(staging, 0o700)
    published = False
    try:
        for name, payload in sorted(retained.items()):
            target = staging / pathlib.PurePosixPath(name)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(target.parent, 0o700)
            write_private_bytes(target, payload, 0o400)
            fsync_directory(target.parent)
        index: dict[str, Any] = {
            "schema": "ringlpn-known-zero-full-resnet18-retained-evidence-v2",
            "full_graph_manifest_digest": manifest_digest,
            "files": {
                name: {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(retained.items())
            },
            "relocations": [{
                "declared_path": declared_provenance_path,
                "document": "private_inputs/graph_binary_approval.json",
                "field": "provenance.path",
                "retained_path":
                    "private_inputs/graph_build_provenance.json",
                "sha256": hashlib.sha256(
                    retained[
                        "private_inputs/graph_build_provenance.json"
                    ]
                ).hexdigest(),
            }],
            "status": "pass",
        }
        index["index_digest"] = self_digest(index, "index_digest")
        write_atomic(staging / "INDEX.json", canonical(index) + b"\n", 0o400)
        fsync_directory(staging)
        try:
            summary_root.lstat()
        except FileNotFoundError:
            pass
        else:
            fail(f"refusing stale summary publication path: {summary_root}")
        os.rename(staging, summary_root)
        published = True
        fsync_directory(summary_root.parent)
    except BaseException:
        cleanup = summary_root if published else staging
        remove_private_scratch(cleanup, "summary publication scratch")
        fsync_directory(summary_root.parent)
        raise


def read_bytes_once(path: pathlib.Path, label: str) -> bytes:
    try:
        _, payload_value = graph_build_provenance.stable_file_binding(
            path, label, payload_required=True
        )
    except graph_build_provenance.ProvenanceError as error:
        fail(str(error))
    assert payload_value is not None
    return payload_value


def load_csv(path: pathlib.Path, expected_header: tuple[str, ...],
             label: str) -> tuple[dict[str, str], bytes]:
    try:
        _, payload_value = graph_build_provenance.stable_file_binding(
            path, label, payload_required=True
        )
        assert payload_value is not None
        payload = payload_value
        text = payload.decode("utf-8")
        source = io.StringIO(text, newline="")
        rows = list(csv.DictReader(source))
        source.seek(0)
        header = tuple(next(csv.reader(source)))
    except (graph_build_provenance.ProvenanceError, UnicodeError,
            csv.Error, StopIteration) as error:
        fail(f"invalid {label} CSV: {error}")
    if header != expected_header or len(rows) != 1 or \
            set(rows[0]) != set(expected_header):
        fail(f"unexpected {label} CSV contract")
    if rows[0].get("status") != "pass":
        fail(f"{label} did not pass")
    return rows[0], payload


def load_trace_rejection_csv(path: pathlib.Path) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
            source.seek(0)
            header = tuple(next(csv.reader(source)))
    except (OSError, UnicodeError, csv.Error, StopIteration) as error:
        fail(f"invalid semantic trace-control CSV: {error}")
    if header != CHECKER_HEADER or len(rows) != 1 or \
            set(rows[0]) != set(CHECKER_HEADER):
        fail("unexpected semantic trace-control CSV contract")
    row = rows[0]
    independently_valid = (
        "source_bindings", "secure_stochastic_truncations", "stock_key_stream",
        "residual_merges", "global_average_pool", "state_links",
        "terminal_output", "counter_contract",
    )
    if row.get("trace_contract") != "FAIL" or row.get("status") != "FAIL" or \
            any(row.get(field) != "pass" for field in independently_valid):
        fail("trace mutant did not reach the semantic trace-contract rejection")


def replace_argument(command: list[str], option: str, value: pathlib.Path | str) -> None:
    try:
        command[command.index(option) + 1] = str(value)
    except (ValueError, IndexError):
        fail(f"control command lacks required argument {option}")


def ledger_snapshot(root: pathlib.Path) -> tuple[tuple[str, int, str], ...]:
    snapshot: list[tuple[str, int, str]] = []
    for path in sorted(root.iterdir()):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"graph ledger contains a non-regular entry: {path}")
        snapshot.append((path.name, metadata.st_size, sha256(path)))
    return tuple(snapshot)


def cancel_active_children(signum: int, _frame: Any) -> NoReturn:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    children = tuple(ACTIVE_CHILDREN.values())
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in children:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in children:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        ACTIVE_CHILDREN.pop(process.pid, None)
    raise SystemExit(128 + signum)


def terminate_and_reap(process: subprocess.Popen[bytes] | None) -> int:
    if process is None:
        return 1
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return_code = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return_code = process.wait()
        ACTIVE_CHILDREN.pop(process.pid, None)
        return return_code
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    ACTIVE_CHILDREN.pop(process.pid, None)
    return return_code


def reap_finished_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    ACTIVE_CHILDREN.pop(process.pid, None)

def run_logged(command: list[str], stdout_path: pathlib.Path,
               stderr_path: pathlib.Path, timeout: int,
               env: dict[str, str] | None = None,
               expected: int = 0) -> int:
    process: subprocess.Popen[bytes] | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=env,
                start_new_session=True,
            )
            ACTIVE_CHILDREN[process.pid] = process
            return_code = process.wait(timeout=timeout)
            reap_finished_group(process)
        except subprocess.TimeoutExpired:
            terminate_and_reap(process)
            fail(f"command timed out after {timeout} seconds: {command[0]}")
        except OSError as error:
            fail(f"command failed to execute: {command[0]}: {error}")
    if return_code != expected:
        fail(f"command returned {return_code}, expected {expected}: {command[0]}")
    return return_code


def run_bounded(command: list[str], timeout: int, label: str) -> int:
    process: subprocess.Popen[bytes] | None = None
    try:
        child_env = os.environ.copy()
        child_env["RINGLPN_COOPERATIVE_PROCESS_GROUP"] = "1"
        process = subprocess.Popen(
            command, env=child_env, start_new_session=True
        )
        ACTIVE_CHILDREN[process.pid] = process
        return_code = process.wait(timeout=timeout)
        reap_finished_group(process)
        return return_code
    except subprocess.TimeoutExpired:
        terminate_and_reap(process)
        fail(f"{label} timed out after {timeout} seconds")
    except OSError as error:
        fail(f"{label} failed to execute: {error}")


def remaining_timeout(deadline: float, label: str) -> int:
    remaining = math.ceil(deadline - time.monotonic())
    if remaining <= 0:
        fail(f"{label} deadline expired")
    return remaining


def run_pair(command0: list[str], command1: list[str], env0: dict[str, str],
             env1: dict[str, str], p0_csv: pathlib.Path, p1_csv: pathlib.Path,
             p0_log: pathlib.Path, p1_log: pathlib.Path, timeout: int,
             expected_success: bool,
             wait_claim: pathlib.Path | None = None) -> tuple[int, int]:
    deadline = time.monotonic() + timeout
    process0: subprocess.Popen[bytes] | None = None
    process1: subprocess.Popen[bytes] | None = None
    try:
        with (p0_csv.open("xb") as stdout0, p1_csv.open("xb") as stdout1,
              p0_log.open("xb") as stderr0, p1_log.open("xb") as stderr1):
            process0 = subprocess.Popen(
                command0, stdout=stdout0, stderr=stderr0, env=env0,
                start_new_session=True,
            )
            ACTIVE_CHILDREN[process0.pid] = process0
            if wait_claim is not None:
                while not wait_claim.is_file():
                    if process0.poll() is not None:
                        fail("party 0 exited before durable graph claim")
                    if time.monotonic() >= deadline:
                        fail("party 0 graph claim timed out")
                    time.sleep(0.05)
            process1 = subprocess.Popen(
                command1, stdout=stdout1, stderr=stderr1, env=env1,
                start_new_session=True,
            )
            ACTIVE_CHILDREN[process1.pid] = process1
            while True:
                rc0 = process0.poll()
                rc1 = process1.poll()
                if rc0 is not None and rc1 is not None:
                    break
                if rc0 is not None and rc0 != 0 and rc1 is None:
                    rc1 = terminate_and_reap(process1)
                    break
                if rc1 is not None and rc1 != 0 and rc0 is None:
                    rc0 = terminate_and_reap(process0)
                    break
                if not expected_success and rc0 == 0 and rc1 is None:
                    rc1 = terminate_and_reap(process1)
                    break
                if not expected_success and rc1 == 0 and rc0 is None:
                    rc0 = terminate_and_reap(process0)
                    break
                if time.monotonic() >= deadline:
                    fail("paired graph process timed out")
                time.sleep(0.05)
    except BaseException:
        terminate_and_reap(process1)
        terminate_and_reap(process0)
        raise
    reap_finished_group(process0)
    reap_finished_group(process1)
    if expected_success and (rc0 != 0 or rc1 != 0):
        diagnostics: list[str] = []
        for party, log_path in (("p0", p0_log), ("p1", p1_log)):
            try:
                lines = log_path.read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            markers = [
                line for line in lines
                if line.startswith("[full-graph]") and
                ("rejected" in line or "failed" in line)
            ]
            diagnostics.extend(f"{party}: {line}" for line in markers[-2:])
        detail = f"; {'; '.join(diagnostics)}" if diagnostics else ""
        fail(f"full graph failed (p0={rc0}, p1={rc1}){detail}")
    if not expected_success and (rc0 == 0 or rc1 == 0):
        fail(f"negative control unexpectedly succeeded (p0={rc0}, p1={rc1})")
    return rc0, rc1


def reject_symlink_components(path: pathlib.Path, label: str) -> None:
    current = pathlib.Path(path.anchor) if path.is_absolute() else pathlib.Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current /= part
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            fail(f"{label} contains symlink component: {current}")


def artifact(path: pathlib.Path, root: pathlib.Path,
             digest_override: str | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest_override if digest_override is not None else sha256(path),
        "digest_scope": "content_excluding_trailing_digest" if digest_override else "whole_file",
    }

def portable_path(path: pathlib.Path, source_root: pathlib.Path,
                  output_root: pathlib.Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    for scope, root in (
        ("ringlpn-source-relative", source_root),
        ("run-output-relative", output_root),
    ):
        try:
            relative = resolved.relative_to(root.resolve(strict=True))
        except ValueError:
            continue
        return {"path": relative.as_posix(), "path_scope": scope}
    return {"path": resolved.name, "path_scope": "external-input-basename"}
def paths_overlap(left: pathlib.Path, right: pathlib.Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def reject_linux_ephemeral_overlap(first: int, last: int, label: str) -> None:
    contract = pathlib.Path("/proc/sys/net/ipv4/ip_local_port_range")
    if not contract.is_file():
        return
    try:
        fields = contract.read_text(encoding="ascii").split()
        if len(fields) != 2:
            raise ValueError("expected two bounds")
        low, high = (int(field) for field in fields)
    except (OSError, UnicodeError, ValueError) as error:
        fail(f"cannot parse Linux ephemeral port range: {error}")
    if first <= high and last >= low:
        fail(
            f"{label} {first}..{last} overlaps Linux ephemeral "
            f"client range {low}..{high}"
        )

def linear_lane_port_range(raw: str, index: int) -> tuple[int, int]:
    fields = raw.split(":")
    port_fields = fields[3].split("-") if len(fields) == 4 else []
    if len(fields) != 4 or len(port_fields) != 2:
        fail(
            f"malformed linear lane {index}; expected "
            "P0_GPU:P1_GPU:CHECK_GPU:FIRST_PORT-LAST_PORT"
        )
    try:
        p0_gpu, p1_gpu, check_gpu = (int(field, 10) for field in fields[:3])
        first, last = (int(field, 10) for field in port_fields)
    except ValueError:
        fail(f"malformed linear lane {index}; ordinals and ports must be integers")
    if min(p0_gpu, p1_gpu, check_gpu) < 0 or \
            len({p0_gpu, p1_gpu, check_gpu}) != 3:
        fail(f"linear lane {index} party/checker GPUs must be pairwise distinct")
    if first <= 0 or last > 65535 or last < first + 85:
        fail(f"linear lane {index} port range is invalid or too small")
    return first, last


def parse_args() -> argparse.Namespace:
    script = pathlib.Path(__file__).resolve()
    ringlpn = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--summary-root", type=pathlib.Path)
    parser.add_argument(
        "--record-set-root", type=pathlib.Path,
        help=("TEST-ONLY reuse of an already-consumed, externally verified "
              "stateful linear record set"),
    )
    parser.add_argument("--manifest", type=pathlib.Path, default=
                        ringlpn / "results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json")
    parser.add_argument("--binary-approval", type=pathlib.Path, default=
                        ringlpn / "results/fc/linear_adapter_binary_approval_2026_08_07.json")
    parser.add_argument("--graph-binary-approval", type=pathlib.Path, default=
                        ringlpn / "results/fc/resnet18_full_graph_binary_approval_2026_08_10.json")
    parser.add_argument("--graph-build-provenance", type=pathlib.Path, default=
                        ringlpn / "bin/resnet18_full_graph_build_provenance.json")
    parser.add_argument("--bin-dir", type=pathlib.Path, default=ringlpn / "bin")
    parser.add_argument("--p0-gpu", type=int, default=0)
    parser.add_argument("--p1-gpu", type=int, default=1)
    parser.add_argument("--check-gpu", type=int, default=2)
    parser.add_argument("--trusted-gpu", type=int, default=1)
    parser.add_argument("--linear-base-port", type=int, default=28800)
    parser.add_argument(
        "--linear-lane",
        action="append",
        default=[],
        metavar="P0:P1:CHECK:FIRST-LAST",
        help="exclusive linear-preprocessing lane; repeat to enable LPT scheduling",
    )
    parser.add_argument("--graph-base-port", type=int, default=29000)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--invocation-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_last_port = args.graph_base_port + 3
    signal.signal(signal.SIGINT, cancel_active_children)
    signal.signal(signal.SIGTERM, cancel_active_children)
    signal.signal(signal.SIGHUP, cancel_active_children)
    if args.graph_base_port <= 0 or graph_last_port > 65535:
        fail("graph base port leaves insufficient room")
    if args.record_set_root is None:
        if args.linear_lane:
            linear_ranges = [
                linear_lane_port_range(raw, index)
                for index, raw in enumerate(args.linear_lane)
            ]
            for first, last in linear_ranges:
                if first <= graph_last_port and args.graph_base_port <= last:
                    fail("linear-lane and graph port ranges overlap")
                reject_linux_ephemeral_overlap(
                    first, last, "linear lane port range"
                )
        else:
            linear_last_port = args.linear_base_port + 4 * 21 + 1
            if args.linear_base_port <= 0 or linear_last_port > 65535:
                fail("linear base port leaves insufficient room")
            if (
                args.linear_base_port <= graph_last_port
                and args.graph_base_port <= linear_last_port
            ):
                fail("linear and graph port ranges overlap")
            reject_linux_ephemeral_overlap(
                args.linear_base_port, linear_last_port, "linear port range"
            )
    elif args.linear_lane:
        fail("--linear-lane cannot be used with an external record set")
    reject_linux_ephemeral_overlap(
        args.graph_base_port, graph_last_port, "graph port range"
    )
    if min(args.p0_gpu, args.p1_gpu, args.check_gpu, args.trusted_gpu) < 0 or \
            len({args.p0_gpu, args.p1_gpu, args.check_gpu}) != 3:
        fail(
            "graph party/checker GPUs must be pairwise distinct and every "
            "GPU index nonnegative"
        )
    if args.timeout_seconds <= 0:
        fail("timeout must be positive")
    ringlpn = pathlib.Path(__file__).resolve().parents[1]
    output_root = args.output_root.absolute()
    state_root = args.state_root.absolute()
    for path, label in ((output_root, "output root"), (state_root, "state root")):
        reject_symlink_components(path, label)
        if path.exists():
            fail(f"{label} already exists: {path}")
        if not path.parent.is_dir():
            fail(f"{label} parent does not exist: {path.parent}")
    if paths_overlap(output_root, state_root):
        fail("output and persistent state roots must be ancestor-disjoint")
    summary_root = args.summary_root.absolute() if args.summary_root else None
    if summary_root is not None:
        reject_symlink_components(summary_root, "summary root")
        if summary_root.exists() or not summary_root.parent.is_dir():
            fail("summary root must be absent with an existing parent")
        if paths_overlap(summary_root, state_root) or \
                paths_overlap(summary_root, output_root):
            fail("summary, output, and state roots must be ancestor-disjoint")
    record_set_mode = (
        "generated_in_run" if args.record_set_root is None
        else "test_only_already_consumed_external_reuse"
    )
    external_linear_root: pathlib.Path | None = None
    if args.record_set_root is not None:
        external_candidate = args.record_set_root.absolute()
        reject_symlink_components(
            external_candidate, "external record-set root"
        )
        external_linear_root = external_candidate.resolve(strict=True)
        if paths_overlap(external_linear_root, output_root) or \
                paths_overlap(external_linear_root, state_root) or \
                (summary_root is not None and
                 paths_overlap(external_linear_root, summary_root)):
            fail(
                "external record-set, output, state, and summary roots must "
                "be ancestor-disjoint"
            )


    manifest = args.manifest.resolve(strict=True)
    approval_input = args.binary_approval.absolute()
    reject_symlink_components(approval_input, "linear binary approval")
    approval = approval_input.resolve(strict=True)
    linear_approval_bytes = read_bytes_once(
        approval, "linear binary approval"
    )
    try:
        linear_approval_document = json.loads(linear_approval_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(f"invalid linear binary approval: {error}")
    linear_provenance_binding = linear_approval_document.get(
        "build_provenance"
    )
    if not isinstance(linear_provenance_binding, dict) or \
            not isinstance(linear_provenance_binding.get("path"), str):
        fail("linear binary approval lacks build provenance")
    relative_linear_provenance = pathlib.PurePosixPath(
        linear_provenance_binding["path"]
    )
    if (relative_linear_provenance.is_absolute() or
            "." in relative_linear_provenance.parts or
            ".." in relative_linear_provenance.parts):
        fail("linear binary approval build provenance path is noncanonical")
    linear_provenance_source = ringlpn.joinpath(
        *relative_linear_provenance.parts
    )
    reject_symlink_components(
        linear_provenance_source, "linear adapter build provenance"
    )
    linear_provenance_bytes = read_bytes_once(
        linear_provenance_source, "linear adapter build provenance"
    )
    if (
        linear_provenance_binding.get("sha256")
        != hashlib.sha256(linear_provenance_bytes).hexdigest()
        or linear_provenance_binding.get("size")
        != len(linear_provenance_bytes)
    ):
        fail("linear adapter build provenance differs from approval")
    graph_approval = args.graph_binary_approval.resolve(strict=True)
    graph_provenance = args.graph_build_provenance.absolute()
    try:
        graph_build_provenance.reject_symlink_components(
            graph_provenance, ringlpn.parents[1], "graph build provenance"
        )
    except graph_build_provenance.ProvenanceError as error:
        fail(str(error))
    (graph_approval_bytes, approved_graph_binaries,
     graph_provenance_bytes, graph_provenance_document) = (
        validate_graph_approval(graph_approval, ringlpn, graph_provenance)
    )
    bin_dir = args.bin_dir.resolve(strict=True)
    source_document, manifest_sha, _ = load_json_once(
        manifest, "source execution manifest"
    )
    if manifest_sha != check_resnet18_graph_contract.EXPECTED_MANIFEST_SHA256:
        fail("source execution manifest hash differs from the approved checkpoint")
    if source_document.get("schema") != EXPECTED_SOURCE_SCHEMA:
        fail("source execution manifest schema differs from the approved checkpoint")
    source_plan_digest = require_digest(
        source_document.get("plan_digest"), "source execution plan digest"
    )
    binaries = {
        "linear_runner": ringlpn / "scripts/run_full_linear_record_set.py",
        "adapter": bin_dir / "test_stock_nonlinear_full_keygen",
        "runtime": bin_dir / "test_resnet18_full_graph",
    }
    for label, path in binaries.items():
        if not path.is_file() or (label != "linear_runner" and not os.access(path, os.X_OK)):
            fail(f"missing executable {label}: {path}")

    source_gate = ringlpn / "scripts/run_full_linear_manifest_gate.sh"
    contract_gate = ringlpn / "scripts/run_resnet18_graph_contract_gate.sh"
    for label, gate in (
        ("source execution manifest", source_gate),
        ("compiled graph contract", contract_gate),
    ):
        gate_return_code = run_bounded([str(gate)], 60, f"{label} gate")
        if gate_return_code != 0:
            fail(f"{label} gate rejected the current tree")
    contract_checker = ringlpn / "scripts/check_resnet18_graph_contract.py"
    contract_probe = bin_dir / "test_resnet18_graph_contract"
    if not contract_checker.is_file() or not contract_probe.is_file() or \
            not os.access(contract_probe, os.X_OK):
        fail("compiled graph contract evidence is unavailable after its gate")
    binaries["contract_probe"] = contract_probe
    output_root.mkdir(mode=0o700)
    state_root.mkdir(mode=0o700)
    private_root = output_root / "private_inputs"
    private_root.mkdir(mode=0o700)
    graph_approval_snapshot = private_root / "graph_binary_approval.json"
    write_private_bytes(graph_approval_snapshot, graph_approval_bytes, 0o400)
    linear_approval_snapshot = private_root / "linear_binary_approval.json"
    write_private_bytes(
        linear_approval_snapshot, linear_approval_bytes, 0o400
    )
    graph_provenance_snapshot = private_root / "graph_build_provenance.json"
    write_private_bytes(
        graph_provenance_snapshot, graph_provenance_bytes, 0o400
    )
    linear_provenance_snapshot = (
        private_root / "linear_adapter_build_provenance.json"
    )
    write_private_bytes(
        linear_provenance_snapshot, linear_provenance_bytes, 0o400
    )
    for label, name in GRAPH_BINARY_FILES.items():
        snapshot = private_root / name
        snapshot_binding = snapshot_executable(binaries[label], snapshot)
        if snapshot_binding != approved_graph_binaries[label]:
            fail(f"graph executable snapshot differs from approval: {name}")
        binaries[label] = snapshot
    fsync_directory(private_root)
    linear_ledger = state_root / "linear-ledger"
    graph_ledger = state_root / "graph-ledger"
    linear_ledger.mkdir(mode=0o700)
    graph_ledger.mkdir(mode=0o700)
    logs = output_root / "logs"
    graph_dir = output_root / "graph"
    run_dir = output_root / "run"
    controls_dir = output_root / "controls"
    for directory in (logs, graph_dir, run_dir, controls_dir):
        directory.mkdir(mode=0o700)
    orchestration_started = time.monotonic()

    if args.record_set_root is None:
        linear_root = output_root / "linear"
        command = [
            sys.executable, str(binaries["linear_runner"]),
            "--manifest", str(manifest), "--bin-dir", str(bin_dir),
            "--output-root", str(linear_root),
            "--binary-approval", str(linear_approval_snapshot),
            "--ledger-root", str(linear_ledger),
            "--timeout-seconds", str(args.timeout_seconds),
            "--emit-mask-states",
        ]
        if args.linear_lane:
            for lane in args.linear_lane:
                command.extend(["--lane", lane])
        else:
            command.extend([
                "--p0-gpu", str(args.p0_gpu),
                "--p1-gpu", str(args.p1_gpu),
                "--check-gpu", str(args.check_gpu),
                "--base-port", str(args.linear_base_port),
            ])
        linear_env = os.environ.copy()
        linear_env["RINGLPN_COOPERATIVE_PROCESS_GROUP"] = "1"
        run_logged(
            command, logs / "linear_record_set.stdout.log",
            logs / "linear_record_set.stderr.log", args.timeout_seconds,
            linear_env,
        )
    else:
        if external_linear_root is None:
            fail("external record-set root was not resolved")
        linear_root = external_linear_root
        external_manifest = linear_root / "LINEAR_RECORD_SET.manifest"
        reject_symlink_components(
            external_manifest, "external linear record-set manifest"
        )
        linear_snapshot_root = output_root / "linear"
        linear_snapshot_root.mkdir(mode=0o700)
        linear_manifest = linear_snapshot_root / "LINEAR_RECORD_SET.manifest"
        write_private_bytes(
            linear_manifest,
            read_bytes_once(
                external_manifest, "external linear record-set manifest"
            ),
            0o400,
        )
        fsync_directory(linear_snapshot_root)
        verifier = [
            sys.executable, str(binaries["linear_runner"]),
            "--verify-record-set", str(linear_manifest),
            "--verify-record-set-root", str(linear_root),
        ]
        run_logged(
            verifier, logs / "linear_record_set.stdout.log",
            logs / "linear_record_set.stderr.log", args.timeout_seconds,
        )
    if args.record_set_root is None:
        linear_manifest = linear_root / "LINEAR_RECORD_SET.manifest"
    reject_symlink_components(linear_manifest, "linear record-set manifest")
    linear_document, linear_manifest_sha, linear_manifest_bytes = load_json_once(
        linear_manifest, "linear record-set manifest"
    )
    if linear_document.get("schema") != EXPECTED_LINEAR_SCHEMA:
        fail("full graph requires the stateful linear record-set schema")
    record_linear_approval_bytes = read_bytes_once(
        linear_root / "private_inputs/binary_approval.json",
        "record-set linear binary approval",
    )
    record_linear_provenance_bytes = read_bytes_once(
        linear_root / "private_inputs/linear_adapter_build_provenance.json",
        "record-set linear adapter build provenance",
    )
    if (
        record_linear_approval_bytes != linear_approval_bytes
        or record_linear_provenance_bytes != linear_provenance_bytes
    ):
        fail(
            "record-set approval/provenance differs from the immutable "
            "linear approval snapshot"
        )
    profile = linear_document.get("profile")
    expected_profile = {
        "qbits": 128,
        "bw": 32,
        "ole_n": 262144,
        "ole_c": 2,
        "ole_t": 8,
        "noise": "regular",
    }
    if not isinstance(profile, dict) or any(
            profile.get(field) != value
            for field, value in expected_profile.items()):
        fail("full graph requires the canonical q128/bw32 regular profile")
    linear_digest = linear_document.get("record_set_digest")
    if not isinstance(linear_digest, str) or len(linear_digest) != 64 or \
            self_digest(linear_document, "record_set_digest") != linear_digest:
        fail("invalid linear record-set self digest")
    source_binding = linear_document.get("source_execution_manifest")
    if (
        not isinstance(source_binding, dict)
        or source_binding.get("sha256") != manifest_sha
        or linear_document.get("source_plan_digest") != source_plan_digest
    ):
        fail("linear record set is not bound to the selected source manifest")
    linear_aggregate = summarize_linear_record_set(linear_document)
    linear_artifacts = bind_linear_artifacts(linear_document, linear_root)

    invocation = args.invocation_id or secrets.token_hex(16)
    if (len(invocation) != 32 or invocation == "0" * 32 or
            any(character not in "0123456789abcdef"
                for character in invocation)):
        fail("invocation ID must be nonzero 128-bit lowercase hex")

    controls: list[dict[str, str]] = []
    # Exercise the same bilateral publication primitive without generating keys.
    control_p0 = controls_dir / "publication_p0"
    control_p1 = controls_dir / "publication_p1_blocker"
    control_p1.mkdir(mode=0o700)
    publication_csv = controls_dir / "publication.csv"
    run_logged([
        str(binaries["adapter"]), "--publication-control", "--csv-header",
        "--p0-output", str(control_p0), "--p1-output", str(control_p1),
    ], publication_csv, controls_dir / "publication.log", args.timeout_seconds)
    publication_control, _ = load_csv(
        publication_csv,
        ("control", "p0_absent", "p1_temp_absent", "status"),
        "bilateral publication control",
    )
    if control_p0.exists() or pathlib.Path(str(control_p0) + ".tmp").exists() or \
            pathlib.Path(str(control_p1) + ".tmp").exists():
        fail("bilateral publication control left partial output")
    controls.append({
        "control": publication_control["control"],
        "expected_rejection": "pass",
        "no_partial_output": "pass",
        "status": "pass",
    })

    graph0 = graph_dir / "party0.full"
    graph1 = graph_dir / "party1.full"
    adapter_csv = graph_dir / "adapter.csv"
    adapter_env = os.environ.copy()
    adapter_env["CUDA_VISIBLE_DEVICES"] = str(args.trusted_gpu)
    adapter_command = [
        str(binaries["adapter"]), "--csv-header", "--gpu", "0",
        "--invocation", invocation, "--manifest-digest", manifest_sha,
        "--record-set-digest", linear_digest,
        "--record-set-root", str(linear_root),
        "--p0-output", str(graph0), "--p1-output", str(graph1),
    ]
    append_linear_artifact_arguments(adapter_command, linear_artifacts)
    run_logged(
        adapter_command, adapter_csv, graph_dir / "adapter.log",
        args.timeout_seconds, adapter_env,
    )
    adapter_metrics, adapter_csv_bytes = load_csv(
        adapter_csv, ADAPTER_HEADER, "trusted adapter"
    )
    graph_digests = (record_digest(graph0), record_digest(graph1))
    if graph_digests != (adapter_metrics["p0_record_digest"],
                         adapter_metrics["p1_record_digest"]):
        fail("adapter CSV record digests differ from published records")

    common = [
        "--invocation-id", invocation, "--manifest-digest", manifest_sha,
        "--record-set-digest", linear_digest,
        "--p0-graph-digest", graph_digests[0],
        "--p1-graph-digest", graph_digests[1],
        "--record-set-root", str(linear_root),
        "--port", str(args.graph_base_port), "--csv-header",
    ]
    append_linear_artifact_arguments(common, linear_artifacts)
    output0 = run_dir / "party0.trace"
    output1 = run_dir / "party1.trace"
    command0 = [
        str(binaries["runtime"]), "--party", "0", "--gpu", "0",
        "--ledger", str(graph_ledger), "--graph-record", str(graph0),
        "--output", str(output0), *common,
    ]
    command1 = [
        str(binaries["runtime"]), "--party", "1", "--gpu", "0",
        "--host", "127.0.0.1", "--ledger", str(graph_ledger),
        "--graph-record", str(graph1), "--output", str(output1), *common,
    ]
    env0 = os.environ.copy()
    env1 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = str(args.p0_gpu)
    env1["CUDA_VISIBLE_DEVICES"] = str(args.p1_gpu)
    graph_execution_deadline = time.monotonic() + args.timeout_seconds
    run_pair(command0, command1, env0, env1,
             run_dir / "party0.csv", run_dir / "party1.csv",
             run_dir / "party0.log", run_dir / "party1.log",
             remaining_timeout(
                 graph_execution_deadline, "graph party/checker"
             ), True,
             graph_ledger / f"{invocation}.p0.claim")
    party0, party0_csv_bytes = load_csv(
        run_dir / "party0.csv", PARTY_HEADER, "graph party 0"
    )
    party1, party1_csv_bytes = load_csv(
        run_dir / "party1.csv", PARTY_HEADER, "graph party 1"
    )

    checker_csv = run_dir / "checker.csv"
    checker_command = [
        str(binaries["runtime"]), "--check", "--csv-header",
        "--invocation-id", invocation, "--manifest-digest", manifest_sha,
        "--record-set-digest", linear_digest,
        "--p0-graph-digest", graph_digests[0],
        "--p1-graph-digest", graph_digests[1],
        "--record-set-root", str(linear_root),
        "--p0-graph-record", str(graph0), "--p1-graph-record", str(graph1),
        "--p0-output", str(output0), "--p1-output", str(output1),
    ]
    append_linear_artifact_arguments(checker_command, linear_artifacts)
    checker_env = os.environ.copy()
    checker_env["CUDA_VISIBLE_DEVICES"] = str(args.check_gpu)
    run_logged(
        checker_command, checker_csv, run_dir / "checker.log",
        remaining_timeout(graph_execution_deadline, "graph party/checker"),
        checker_env,
    )
    checker, checker_csv_bytes = load_csv(
        checker_csv, CHECKER_HEADER, "full graph checker"
    )

    duplicate_dir = controls_dir / "duplicate_invocation"
    duplicate_dir.mkdir(mode=0o700)
    duplicate_outputs = (
        duplicate_dir / "party0.trace", duplicate_dir / "party1.trace",
    )
    duplicate_commands = (command0.copy(), command1.copy())
    for command, output in zip(duplicate_commands, duplicate_outputs):
        replace_argument(command, "--output", output)
    positive_output_digests = (sha256(output0), sha256(output1))
    ledger_before_duplicate = ledger_snapshot(graph_ledger)
    for party, (command, environment) in enumerate(
            zip(duplicate_commands, (env0, env1))):
        run_logged(
            command, duplicate_dir / f"p{party}.csv",
            duplicate_dir / f"p{party}.log", args.timeout_seconds,
            environment, expected=2,
        )
    duplicate_logs = tuple(
        (duplicate_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in ("p0.log", "p1.log")
    )
    ledger_before_by_name = {
        name: (size, digest)
        for name, size, digest in ledger_before_duplicate
    }
    ledger_after_by_name = {
        name: (size, digest)
        for name, size, digest in ledger_snapshot(graph_ledger)
    }
    pending_names = {
        f"{invocation}.p0.pending", f"{invocation}.p1.pending",
    }
    expected_ledger_names = set(ledger_before_by_name) | pending_names
    ledger_append_exact = \
        set(ledger_after_by_name) == expected_ledger_names and \
        all(ledger_after_by_name.get(name) == binding
            for name, binding in ledger_before_by_name.items()) and \
        all(ledger_after_by_name.get(f"{invocation}.p{party}.pending") ==
            ledger_before_by_name.get(f"{invocation}.p{party}.claim")
            for party in (0, 1))
    duplicate_clean = \
        not any(path.exists() for output in duplicate_outputs
                for path in (output, pathlib.Path(str(output) + ".tmp"))) and \
        (sha256(output0), sha256(output1)) == positive_output_digests and \
        ledger_append_exact and \
        all("(local=1 claim=0)" in log for log in duplicate_logs)
    controls.append({"control": "reused_invocation",
                     "expected_rejection": "pass",
                     "no_partial_output": "pass" if duplicate_clean else "FAIL",
                     "status": "pass" if duplicate_clean else "FAIL"})

    stale_dir = controls_dir / "stale_output"
    stale_dir.mkdir(mode=0o700)
    stale_ledger = state_root / "stale-output-ledger"
    stale_ledger.mkdir(mode=0o700)
    stale_invocation = secrets.token_hex(16)
    stale_graphs = (
        stale_dir / "party0.full", stale_dir / "party1.full",
    )
    stale_adapter_command = adapter_command.copy()
    replace_argument(stale_adapter_command, "--invocation", stale_invocation)
    replace_argument(stale_adapter_command, "--p0-output", stale_graphs[0])
    replace_argument(stale_adapter_command, "--p1-output", stale_graphs[1])
    stale_adapter_csv = stale_dir / "adapter.csv"
    run_logged(
        stale_adapter_command, stale_adapter_csv, stale_dir / "adapter.log",
        args.timeout_seconds, adapter_env,
    )
    stale_adapter_metrics, _ = load_csv(
        stale_adapter_csv, ADAPTER_HEADER, "stale-output trusted adapter",
    )
    stale_graph_digests = (
        record_digest(stale_graphs[0]), record_digest(stale_graphs[1]),
    )
    if stale_graph_digests != (
            stale_adapter_metrics["p0_record_digest"],
            stale_adapter_metrics["p1_record_digest"]):
        fail("stale-output adapter digests differ from its records")
    stale_outputs = (
        stale_dir / "party0.trace", stale_dir / "party1.trace",
    )
    sentinel = secrets.token_bytes(64)
    for output in stale_outputs:
        write_private_bytes(output, sentinel, 0o400)
    sentinel_digests = tuple(sha256(output) for output in stale_outputs)
    stale_commands = (command0.copy(), command1.copy())
    for party, command in enumerate(stale_commands):
        replace_argument(command, "--invocation-id", stale_invocation)
        replace_argument(command, "--p0-graph-digest", stale_graph_digests[0])
        replace_argument(command, "--p1-graph-digest", stale_graph_digests[1])
        replace_argument(command, "--ledger", stale_ledger)
        replace_argument(command, "--graph-record", stale_graphs[party])
        replace_argument(command, "--output", stale_outputs[party])
    run_pair(stale_commands[0], stale_commands[1], env0, env1,
             stale_dir / "p0.csv", stale_dir / "p1.csv",
             stale_dir / "p0.log", stale_dir / "p1.log",
             args.timeout_seconds, False)
    stale_logs = tuple(
        (stale_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in ("p0.log", "p1.log")
    )
    stale_clean = \
        tuple(sha256(output) for output in stale_outputs) == sentinel_digests and \
        not any(pathlib.Path(str(output) + ".tmp").exists()
                for output in stale_outputs) and \
        not ledger_snapshot(stale_ledger) and \
        all("(output=0 paths=1 graph_opened=1 graph_bound=1 "
            "identity=1 cuda=1)" in log and
            "(local=0 claim=0)" in log for log in stale_logs)
    controls.append({"control": "stale_output",
                     "expected_rejection": "pass",
                     "no_partial_output": "pass" if stale_clean else "FAIL",
                     "status": "pass" if stale_clean else "FAIL"})

    swap_dir = controls_dir / "party_swap"
    swap_dir.mkdir(mode=0o700)
    swap_ledger = state_root / "party-swap-ledger"
    swap_ledger.mkdir(mode=0o700)
    swap0_out = swap_dir / "p0.trace"
    swap1_out = swap_dir / "p1.trace"
    swap0 = command0.copy()
    swap1 = command1.copy()
    for command, ledger, graph_path, output in (
        (swap0, swap_ledger, graph1, swap0_out),
        (swap1, swap_ledger, graph0, swap1_out),
    ):
        command[command.index("--ledger") + 1] = str(ledger)
        command[command.index("--graph-record") + 1] = str(graph_path)
        command[command.index("--output") + 1] = str(output)
    run_pair(swap0, swap1, env0, env1,
             swap_dir / "p0.csv", swap_dir / "p1.csv",
             swap_dir / "p0.log", swap_dir / "p1.log",
             args.timeout_seconds, False)
    swap_clean = not any(path.exists() for path in
                         (swap0_out, swap1_out,
                          pathlib.Path(str(swap0_out) + ".tmp"),
                          pathlib.Path(str(swap1_out) + ".tmp")))
    controls.append({"control": "party_record_swap",
                     "expected_rejection": "pass",
                     "no_partial_output": "pass" if swap_clean else "FAIL",
                     "status": "pass" if swap_clean else "FAIL"})

    corrupt_graph_dir = controls_dir / "corrupt_graph_record"
    corrupt_graph_dir.mkdir(mode=0o700)
    corrupt_graph = corrupt_graph_dir / "party0.truncated"
    with graph0.open("rb") as source, corrupt_graph.open("xb") as target:
        prefix = source.read(1024)
        if len(prefix) != 1024:
            fail("graph corruption control source is too short")
        target.write(prefix)
        target.flush()
        os.fsync(target.fileno())
    os.chmod(corrupt_graph, 0o400)
    corrupt_graph_ledger = state_root / "corrupt-graph-ledger"
    corrupt_graph_ledger.mkdir(mode=0o700)
    corrupt_graph_output = corrupt_graph_dir / "party0.trace"
    corrupt_graph_command = command0.copy()
    corrupt_graph_command[corrupt_graph_command.index("--ledger") + 1] = \
        str(corrupt_graph_ledger)
    corrupt_graph_command[
        corrupt_graph_command.index("--graph-record") + 1
    ] = str(corrupt_graph)
    corrupt_graph_command[corrupt_graph_command.index("--output") + 1] = \
        str(corrupt_graph_output)
    run_logged(
        corrupt_graph_command,
        corrupt_graph_dir / "party0.csv",
        corrupt_graph_dir / "party0.log",
        args.timeout_seconds,
        env0,
        expected=2,
    )
    corrupt_graph_clean = \
        not corrupt_graph_output.exists() and \
        not pathlib.Path(str(corrupt_graph_output) + ".tmp").exists() and \
        not any(corrupt_graph_ledger.iterdir())
    controls.append({
        "control": "truncated_nonlinear_graph_record",
        "expected_rejection": "pass",
        "no_partial_output": "pass" if corrupt_graph_clean else "FAIL",
        "status": "pass" if corrupt_graph_clean else "FAIL",
    })

    payload_corrupt_dir = controls_dir / "nonlinear_payload_corruption"
    payload_corrupt_dir.mkdir(mode=0o700)
    payload_corrupt_graph = payload_corrupt_dir / "party0.full"
    source_graph_size = graph0.stat().st_size
    if source_graph_size <= 4096 + 32:
        fail("nonlinear graph record is too short for payload corruption")
    shutil.copyfile(graph0, payload_corrupt_graph)
    payload_offset = source_graph_size - 33
    with graph0.open("rb") as source:
        source.seek(payload_offset)
        original_payload_byte = source.read(1)
        source.seek(source_graph_size - 32)
        original_trailing_digest = source.read(32)
    with payload_corrupt_graph.open("r+b") as target:
        target.seek(payload_offset)
        copied_payload_byte = target.read(1)
        if len(copied_payload_byte) != 1 or \
                copied_payload_byte != original_payload_byte:
            fail("nonlinear payload corruption copy differs before mutation")
        target.seek(payload_offset)
        target.write(bytes([copied_payload_byte[0] ^ 1]))
        target.flush()
        os.fsync(target.fileno())
    os.chmod(payload_corrupt_graph, 0o400)
    with payload_corrupt_graph.open("rb") as target:
        target.seek(source_graph_size - 32)
        mutated_trailing_digest = target.read(32)
    if payload_corrupt_graph.stat().st_size != source_graph_size or \
            mutated_trailing_digest != original_trailing_digest:
        fail("nonlinear payload mutant changed record length or trailing digest")
    payload_corrupt_ledger = state_root / "payload-corrupt-graph-ledger"
    payload_corrupt_ledger.mkdir(mode=0o700)
    payload_corrupt_output = payload_corrupt_dir / "party0.trace"
    payload_corrupt_command = command0.copy()
    replace_argument(payload_corrupt_command, "--ledger",
                     payload_corrupt_ledger)
    replace_argument(payload_corrupt_command, "--graph-record",
                     payload_corrupt_graph)
    replace_argument(payload_corrupt_command, "--output",
                     payload_corrupt_output)
    run_logged(
        payload_corrupt_command,
        payload_corrupt_dir / "party0.csv",
        payload_corrupt_dir / "party0.log",
        args.timeout_seconds,
        env0,
        expected=2,
    )
    payload_corrupt_log = (payload_corrupt_dir / "party0.log").read_text(
        encoding="utf-8", errors="replace",
    )
    payload_corrupt_clean = \
        not payload_corrupt_output.exists() and \
        not pathlib.Path(str(payload_corrupt_output) + ".tmp").exists() and \
        not ledger_snapshot(payload_corrupt_ledger) and \
        "(output=1 paths=1 graph_opened=0 graph_bound=0 " \
        "identity=1 cuda=1)" in payload_corrupt_log and \
        "(local=0 claim=0)" in payload_corrupt_log
    controls.append({
        "control": "nonlinear_graph_payload_corruption",
        "expected_rejection": "pass",
        "no_partial_output": "pass" if payload_corrupt_clean else "FAIL",
        "status": "pass" if payload_corrupt_clean else "FAIL",
    })

    trace_dir = controls_dir / "semantic_trace_corruption"
    trace_dir.mkdir(mode=0o700)
    corrupted_traces = (
        trace_dir / "party0.trace", trace_dir / "party1.trace",
    )
    positive_trace_digests = (sha256(output0), sha256(output1))
    for source, destination in zip((output0, output1), corrupted_traces):
        mutated = bytearray(source.read_bytes())
        if len(mutated) != 512 + 111 * 32 + 32:
            fail("semantic trace control has an unexpected run-record size")
        previous_digest = bytes(mutated[-32:])
        mutated[600] ^= 1
        mutated[-32:] = hashlib.sha256(mutated[:-32]).digest()
        if bytes(mutated[-32:]) == previous_digest:
            fail("semantic trace mutation did not change its record digest")
        write_private_bytes(destination, bytes(mutated), 0o400)
        record_digest(destination)
    corrupt_checker = checker_command.copy()
    replace_argument(corrupt_checker, "--p0-output", corrupted_traces[0])
    replace_argument(corrupt_checker, "--p1-output", corrupted_traces[1])
    corrupt_checker_csv = trace_dir / "checker.csv"
    run_logged(corrupt_checker, corrupt_checker_csv,
               trace_dir / "checker.log", args.timeout_seconds,
               checker_env, expected=1)
    load_trace_rejection_csv(corrupt_checker_csv)
    trace_clean = (sha256(output0), sha256(output1)) == positive_trace_digests
    controls.append({"control": "digest_valid_trace_corruption",
                     "expected_rejection": "pass",
                     "no_partial_output": "pass" if trace_clean else "FAIL",
                     "status": "pass" if trace_clean else "FAIL"})

    controls_csv = controls_dir / "controls.csv"
    with controls_csv.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CONTROL_HEADER)
        writer.writeheader()
        writer.writerows(controls)
        output.flush()
        os.fsync(output.fileno())
    failed_controls = [
        row for row in controls if row["status"] != "pass"
    ]
    if failed_controls:
        failed_rows = json.dumps(
            failed_controls, sort_keys=True, separators=(",", ":")
        )
        fail(
            "negative controls failed; exact_failed_rows="
            f"{failed_rows}; diagnostics retained at {output_root}"
        )

    linear_schedule = linear_document.get("resource_schedule")
    if isinstance(linear_schedule, dict):
        scheduled_lanes = linear_schedule.get("lanes")
    else:
        scheduled_lanes = None
    if isinstance(scheduled_lanes, list) and scheduled_lanes:
        primary_linear_lane = scheduled_lanes[0]
        linear_p0_gpu = primary_linear_lane.get("p0_gpu")
        linear_p1_gpu = primary_linear_lane.get("p1_gpu")
        linear_check_gpu = primary_linear_lane.get("check_gpu")
        linear_base_port = primary_linear_lane.get("port_first")
    else:
        linear_p0_gpu = linear_document.get("p0_gpu")
        linear_p1_gpu = linear_document.get("p1_gpu")
        linear_check_gpu = linear_document.get("check_gpu")
        layer_rows = linear_document.get("layers")
        linear_base_port = (
            layer_rows[0].get("port")
            if isinstance(layer_rows, list) and layer_rows
            and isinstance(layer_rows[0], dict) else None
        )
        legacy_resources = (
            linear_p0_gpu, linear_p1_gpu, linear_check_gpu, linear_base_port,
        )
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0 for value in legacy_resources):
            fail("linear record set lacks valid legacy resource metadata")


    manifest_document: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_scope": (
            "known-zero exact stock ResNet18 graph choreography; dealerless "
            "Ring-LPN linear preprocessing; TEST-ONLY trusted stock nonlinear keys; "
            "not private/trained inference or a concrete-security claim"
        ),
        "invocation_id": invocation,
        "source_execution_manifest": {
            **portable_path(manifest, ringlpn, output_root),
            "sha256": manifest_sha,
        },
        "linear_record_set": {
            **portable_path(linear_manifest, ringlpn, output_root),
            "record_set_digest": linear_digest,
            "schema": linear_document["schema"],
            "sha256": linear_manifest_sha,
            "aggregate": linear_aggregate,
            "mode": record_set_mode,
        },
        "binary_approval": {
            **portable_path(
                linear_approval_snapshot, ringlpn, output_root
            ),
            "bytes": len(linear_approval_bytes),
            "sha256": hashlib.sha256(linear_approval_bytes).hexdigest(),
        },
        "linear_adapter_build_provenance": {
            **portable_path(
                linear_provenance_snapshot, ringlpn, output_root
            ),
            "bytes": len(linear_provenance_bytes),
            "sha256": hashlib.sha256(
                linear_provenance_bytes
            ).hexdigest(),
        },
        "graph_binary_approval": {
            **portable_path(graph_approval_snapshot, ringlpn, output_root),
            "bytes": len(graph_approval_bytes),
            "sha256": hashlib.sha256(graph_approval_bytes).hexdigest(),
        },
        "graph_build_provenance": {
            **portable_path(
                graph_provenance_snapshot, ringlpn, output_root
            ),
            "provenance_digest":
                graph_provenance_document["provenance_digest"],
            "schema": graph_provenance_document["schema"],
            "bytes": len(graph_provenance_bytes),
            "sha256": hashlib.sha256(graph_provenance_bytes).hexdigest(),
        },
        "binaries": {
            name: {
                **portable_path(path, ringlpn, output_root),
                "sha256": sha256(path),
            }
            for name, path in binaries.items()
        },
        "graph_contract": {
            "checker": {
                **portable_path(contract_checker, ringlpn, output_root),
                "sha256": sha256(contract_checker),
            },
            "source_manifest_gate": {
                **portable_path(source_gate, ringlpn, output_root),
                "sha256": sha256(source_gate),
            },
            "status": "pass",
        },
        "graph_records": {
            "party0": artifact(graph0, output_root, graph_digests[0]),
            "party1": artifact(graph1, output_root, graph_digests[1]),
        },
        "run_records": {
            "party0": artifact(output0, output_root),
            "party1": artifact(output1, output_root),
        },
        "metrics": {
            "adapter": adapter_metrics,
            "party0": party0,
            "party1": party1,
            "checker": checker,
            "orchestrator_wall_us": int(
                (time.monotonic() - orchestration_started) * 1_000_000
            ),
        },
        "controls": controls,
        "gpu_assignment": {"linear_p0": linear_p0_gpu,
                           "linear_p1": linear_p1_gpu,
                           "linear_check": linear_check_gpu,
                           "trusted_adapter": args.trusted_gpu,
                           "graph_p0": args.p0_gpu, "graph_p1": args.p1_gpu},
        "ports": {"linear_base": linear_base_port,
                  "graph_party_channel": args.graph_base_port,
                  "graph_stock_channel": args.graph_base_port + 2},
        "status": "pass",
    }
    manifest_document["manifest_digest"] = self_digest(manifest_document,
                                                       "manifest_digest")
    publication = output_root / "FULL_GRAPH.manifest"
    publication_bytes = canonical(manifest_document) + b"\n"
    write_atomic(publication, publication_bytes, 0o400)

    if summary_root is not None:
        retained = {
            "FULL_GRAPH.manifest": publication_bytes,
            "private_inputs/graph_binary_approval.json":
                graph_approval_bytes,
            "private_inputs/linear_binary_approval.json":
                linear_approval_bytes,
            "private_inputs/linear_adapter_build_provenance.json":
                linear_provenance_bytes,
            "private_inputs/graph_build_provenance.json":
                graph_provenance_bytes,
            "linear/LINEAR_RECORD_SET.manifest": linear_manifest_bytes,
            "linear/private_inputs/binary_approval.json":
                record_linear_approval_bytes,
            "linear/private_inputs/linear_adapter_build_provenance.json":
                record_linear_provenance_bytes,
            "adapter.csv": adapter_csv_bytes,
            "party0.csv": party0_csv_bytes,
            "party1.csv": party1_csv_bytes,
            "checker.csv": checker_csv_bytes,
            "controls.csv": read_bytes_once(
                controls_csv, "retained controls CSV"
            ),
            "adapter.log": read_bytes_once(
                graph_dir / "adapter.log", "retained adapter log"
            ),
            "party0.log": read_bytes_once(
                run_dir / "party0.log", "retained party-0 log"
            ),
            "party1.log": read_bytes_once(
                run_dir / "party1.log", "retained party-1 log"
            ),
            "checker.log": read_bytes_once(
                run_dir / "checker.log", "retained checker log"
            ),
        }
        try:
            declared_provenance_path = graph_provenance.relative_to(
                ringlpn
            ).as_posix()
            publish_retained_summary(
                summary_root, retained, manifest_document["manifest_digest"],
                declared_provenance_path,
            )
        except BaseException as error:
            try:
                publication.unlink(missing_ok=True)
                fsync_directory(output_root)
            except OSError as rollback_error:
                fail(
                    "retained summary publication failed and FULL_GRAPH.manifest "
                    f"rollback failed: {rollback_error}"
                )
            fail(
                "retained summary publication failed; no summary or "
                f"FULL_GRAPH.manifest published: {error}"
            )

    print(f"full-graph-runner: FULL GRAPH PASS {manifest_document['manifest_digest']}")


if __name__ == "__main__":
    main()
