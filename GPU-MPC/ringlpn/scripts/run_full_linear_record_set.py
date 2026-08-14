#!/usr/bin/env python3
"""Validate or produce an isolated ordered set of 21 forward-linear records.

The records are independently produced adapter artifacts.  They do not carry
Orca graph state and do not establish residual, stochastic-truncation, or
full-model execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import secrets
import re
import stat
import subprocess
import signal
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, NoReturn

SOURCE_SCHEMA = "ringlpn-full-linear-execution-v1"
PLAN_SCHEMA = "ringlpn-forward-linear-plan-validation-v2"
RECORD_SET_SCHEMA = "ringlpn-forward-linear-record-set-v2"
STATEFUL_RECORD_SET_SCHEMA = "ringlpn-forward-linear-record-set-v4"
COMPOSITION_STATUS = "isolated_linear_records_no_graph_state"
STATEFUL_COMPOSITION_STATUS = "stateful_linear_records_pending_graph_execution"
ARTIFACT_SCOPE = "ordered_isolated_forward_linear_adapter_records"
STATEFUL_ARTIFACT_SCOPE = (
    "ordered_source_bound_forward_linear_records_with_party_local_mask_state"
)
SECURITY_SCOPE = (
    "static_semi_honest_component_composition_only_no_concrete_security_level"
)
EXECUTION_EXCLUSIONS = [
    "graph_state", "residual_merges", "stochastic_truncation",
    "nonlinear_layers", "full_model_execution",
]
STATEFUL_EXECUTION_EXCLUSIONS = [
    "residual_merges", "stochastic_truncation", "nonlinear_layers",
    "full_model_execution",
]
PLAN_HEADER = (
    "operator", "qbits", "bw", "size_input_words", "size_weight_words",
    "size_output_words", "cross_terms", "ring_batches",
    "ring_application_slots", "ring_bootstrap_slots", "output_height",
    "output_width",
)
BINARY_NAMES = {
    "test_two_party_conv_preprocess",
    "test_two_party_fc_preprocess",
}
OBSERVED_PROVENANCE = "observed_executable_snapshot_not_approved_build_provenance"
VALIDATED_PROVENANCE = "validated_reproducible_build_receipts_snapshot"
BINARY_APPROVAL_SCHEMA = "ringlpn-linear-adapter-binary-approval-v2"
TERMINATION_REQUESTED = threading.Event()
BINARY_APPROVAL_SCOPE = (
    "reproducible_linear_adapters_for_internal_advisor_execution"
)
BINARY_BUILD_PROVENANCE_SCHEMA = (
    "ringlpn-linear-adapter-build-provenance-v2"
)
PROVENANCE_PATH_TOKENS = frozenset({
    "${REPO}", "${CANONICAL_SOURCE}", "${BUILD_ROOT}",
})
PROVENANCE_ENVIRONMENT_KEYS = frozenset({
    "CUDA_ARCH", "CXX", "HOME", "LANG", "LC_ALL", "NVCC", "OBJCOPY",
    "PATH", "PWD", "RINGLPN_CANONICAL_BUILD_ACTIVE",
    "RINGLPN_COMPONENT_DISPATCH_ACTIVE", "RINGLPN_LINEAR_COMMAND_FILE",
    "RINGLPN_LINEAR_DEPFILE", "RINGLPN_LINEAR_ENVIRONMENT_FILE",
    "RINGLPN_LINEAR_KIND", "RINGLPN_LINEAR_LINK_MAP",
    "SOURCE_DATE_EPOCH", "TMPDIR", "TZ", "ZERO_AR_DATE",
})
BINARY_APPROVAL_CLASSIFICATION = "internal/advisor"
PROBE_TIMEOUT_SECONDS = 60
PARTY_METRIC_SUFFIX = (
    "ole_n,ole_c,ole_t,noise,ring_batches,ring_application_slots,"
    "ring_bootstrap_slots,ring_ole_instances,slots_used,dpf_trees,"
    "dpf_string_ots,dpf_bit_triples,dpf_scalar_oles,"
    "dpf_epoch_zero_scalar_oles,dpf_pcg_scalar_oles,"
    "dpf_pcg_oles_reserved,dpf_pcg_oles_discarded,"
    "dpf_pcg_opening_words_sent,dpf_logical_opened_bits,"
    "dpf_meaningful_share_bits,spfss_key_bytes,public_a_seed_words_sent,"
    "derandomization_words_sent,conversions,"
    "conversion_logical_opened_bits,conversion_meaningful_share_bits,"
    "protocol_bytes_sent,protocol_direction_switches,total_us,status,"
    "protocol_dependency_rounds,preflight_us,ot_setup_us,dpf_phase_a_us,"
    "dpf_phase_b_us,dpf_phase_c_us,spfss_grouping_us,"
    "public_polynomial_exchange_us,gpu_ringlpn_expansion_us,"
    "derandomization_openings_us,conversion_us,serialization_us,commit_us,"
    "peak_host_rss_bytes,peak_gpu_bytes,min_gpu_free_bytes,"
    "transport_straight_bytes_sent,transport_straight_bytes_received,"
    "transport_reversed_bytes_sent,transport_reversed_bytes_received,"
    "base_ots,base_ot_setup_bytes_sent,base_ot_setup_bytes_received,"
    "channel_auth_straight_bytes_sent,channel_auth_straight_bytes_received,"
    "channel_auth_reversed_bytes_sent,channel_auth_reversed_bytes_received,"
    "transport_bytes_include_base_ot,base_ot_setup_dependency_rounds,"
    "invocation_id,ledger_digest,ot_backend,ot_backend_revision,"
    "ot_correlation_straight_bytes_sent,"
    "ot_correlation_straight_bytes_received,"
    "ot_correlation_reversed_bytes_sent,"
    "ot_correlation_reversed_bytes_received,ot_adjustment_bytes_sent,"
    "ot_adjustment_bytes_received,ot_ciphertext_bytes_sent,"
    "ot_ciphertext_bytes_received,ot_inventory_straight_declared,"
    "ot_inventory_straight_consumed,ot_inventory_reversed_declared,"
    "ot_inventory_reversed_consumed,ot_backend_review_status,"
    "ring_application_slots_discarded,ot_backend_bridge_sha256,"
    "dpf_breadth_evaluator_calls,dpf_root_to_leaf_evaluator_calls"
).split(",")
FC_PARTY_HEADER = tuple(
    "party,qbits,bw,rows,inner,cols".split(",") + PARTY_METRIC_SUFFIX
)
CONV_PARTY_HEADER = tuple(
    "party,qbits,bw,n,h,w,ci,fh,fw,co,padding,stride,oh,ow".split(",")
    + PARTY_METRIC_SUFFIX
)
FC_CHECKER_HEADER = tuple(
    "qbits,bw,rows,inner,cols,ring_batches,final_payload_bytes_per_party,"
    "matched_dealer_keygen_us,checker_two_share_online_us,"
    "matched_dealer_keygen_contract,key_order,online_contract,status,"
    "checker_us,peak_host_rss_bytes,peak_gpu_bytes,min_gpu_free_bytes,"
    "invocation_id,ledger_digest".split(",")
)
CONV_CHECKER_HEADER = tuple(
    "qbits,bw,n,h,w,ci,fh,fw,co,padding,stride,oh,ow,ring_batches,"
    "final_payload_bytes_per_party,matched_dealer_keygen_us,"
    "checker_two_share_online_us,matched_dealer_keygen_contract,key_order,"
    "unchanged_online_contract,status,checker_us,peak_host_rss_bytes,"
    "peak_gpu_bytes,min_gpu_free_bytes,invocation_id,ledger_digest".split(",")
)
PARTY_TEXT_FIELDS = {
    "party", "noise", "status", "transport_bytes_include_base_ot",
    "invocation_id", "ledger_digest", "ot_backend", "ot_backend_revision",
    "ot_backend_bridge_sha256", "ot_backend_review_status",
}
PARTY_NA_FIELDS = {
    "transport_straight_bytes_received",
    "transport_reversed_bytes_received",
    "base_ot_setup_bytes_received",
    "base_ot_setup_dependency_rounds",
    "ot_correlation_straight_bytes_sent",
    "ot_correlation_straight_bytes_received",
    "ot_correlation_reversed_bytes_sent",
    "ot_correlation_reversed_bytes_received",
    "ot_adjustment_bytes_sent",
    "ot_adjustment_bytes_received",
    "ot_ciphertext_bytes_sent",
    "ot_ciphertext_bytes_received",
    "ot_inventory_straight_declared",
    "ot_inventory_straight_consumed",
    "ot_inventory_reversed_declared",
    "ot_inventory_reversed_consumed",
}
CHECKER_TEXT_FIELDS = {
    "matched_dealer_keygen_contract", "key_order", "online_contract",
    "unchanged_online_contract", "status", "invocation_id", "ledger_digest",
}
ARTIFACT_LABELS = {
    "p0_record", "p1_record", "p0_metrics_csv", "p1_metrics_csv",
    "p0_log", "p1_log", "checker_csv", "checker_log",
}
STATEFUL_ARTIFACT_LABELS = ARTIFACT_LABELS | {"p0_state", "p1_state"}
MANIFEST_BOUND_PRIVATE_ARTIFACTS = {
    "p0_record", "p1_record", "p0_state", "p1_state",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"linear-record-set-runner: {message}")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1 << 20):
                digest.update(chunk)
    except OSError as error:
        fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def request_termination(signum: int, _frame: Any) -> NoReturn:
    TERMINATION_REQUESTED.set()
    raise SystemExit(128 + signum)


def canonical(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"cannot parse {label}: {error}")
    if not isinstance(value, dict):
        fail(f"expected JSON object in {label}")
    return value


def read_bytes_once(path: pathlib.Path, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                fail(f"{label} must be a regular non-symlink file")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1 << 20))
                if not chunk:
                    fail(f"{label} was shortened while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                fail(f"{label} was lengthened while reading")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as error:
        fail(f"cannot read {label} {path}: {error}")


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"invalid {label}")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"invalid {label}")
    return value


def require_hex_digest(value: Any, label: str) -> str:
    text = require_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        fail(f"invalid {label}")
    return text


def fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_private_bytes(path: pathlib.Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, mode)


def write_atomic(path: pathlib.Path, document: dict[str, Any], mode: int = 0o600) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    temporary = path.with_name("." + path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
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


def snapshot_executable(source: pathlib.Path, destination: pathlib.Path) -> str:
    try:
        metadata = source.lstat()
    except OSError as error:
        fail(f"cannot inspect executable {source}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"executable source must be a regular non-symlink: {source}")
    if not os.access(source, os.X_OK):
        fail(f"executable source is not executable: {source}")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(source_fd, 1 << 20):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    os.chmod(destination, 0o500)
    return digest.hexdigest()


def parse_cli_pairs(cli: list[str], label: str) -> dict[str, int]:
    if len(cli) % 2 != 0:
        fail(f"invalid CLI pairs for {label}")
    parsed: dict[str, int] = {}
    for offset in range(0, len(cli), 2):
        option = cli[offset]
        if not option.startswith("--") or option in parsed:
            fail(f"invalid or duplicate CLI option for {label}")
        try:
            parsed[option] = int(cli[offset + 1])
        except ValueError:
            fail(f"non-integer CLI value for {label}.{option}")
    return parsed


def expected_plan_row(layer: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    name = require_string(layer.get("layer"), "layer name")
    operator = layer.get("operator")
    cli = layer.get("cli")
    if not isinstance(cli, list) or not all(isinstance(item, str) for item in cli):
        fail(f"invalid CLI for {name}")
    cli_values = parse_cli_pairs(cli, name)
    if cli_values.pop("--layer-ordinal", None) != require_int(
            layer.get("linear_order"), f"{name}.linear_order", 1):
        fail(f"invalid layer ordinal for {name}")
    common: dict[str, Any] = {
        "operator": "conv" if operator == "conv2d" else "fc",
        "qbits": require_int(profile.get("qbits"), "profile.qbits", 1),
        "bw": require_int(profile.get("bw"), "profile.bw", 3),
        "size_input_words": require_int(layer.get("size_input_words"), f"{name}.size_input_words", 1),
        "size_weight_words": require_int(layer.get("size_weight_words"), f"{name}.size_weight_words", 1),
        "size_output_words": require_int(layer.get("size_output_words"), f"{name}.size_output_words", 1),
        "cross_terms": require_int(layer.get("cross_terms"), f"{name}.cross_terms", 1),
        "ring_batches": require_int(layer.get("ring_batches"), f"{name}.ring_batches", 1),
        "ring_application_slots": require_int(layer.get("ring_application_slots"), f"{name}.ring_application_slots", 1),
        "ring_bootstrap_slots": require_int(layer.get("ring_bootstrap_slots"), f"{name}.ring_bootstrap_slots", 1),
    }
    for field in ("ring_application_slots", "ring_bootstrap_slots"):
        if common[field] != require_int(profile.get(field), f"profile.{field}", 1):
            fail(f"profile/manifest mismatch for {name}.{field}")
    if require_int(layer.get("batch"), f"{name}.batch", 1) != 1 or require_int(
            layer.get("matmul_batch"), f"{name}.matmul_batch", 1) != 1:
        fail(f"unsupported batch contract for {name}")
    if operator == "conv2d":
        contract = layer.get("conv")
        fields = ("n", "h", "w", "ci", "fh", "fw", "co", "padding", "stride")
        if not isinstance(contract, dict) or set(cli_values) != {f"--{field}" for field in fields}:
            fail(f"invalid convolution shape contract for {name}")
        shape = {field: require_int(contract.get(field), f"{name}.conv.{field}", 0) for field in fields}
        if any(cli_values[f"--{field}"] != shape[field] for field in fields):
            fail(f"CLI/manifest convolution mismatch for {name}")
        oh = require_int(contract.get("oh"), f"{name}.conv.oh", 1)
        ow = require_int(contract.get("ow"), f"{name}.conv.ow", 1)
        if shape["n"] != 1 or min(shape["h"], shape["w"], shape["ci"], shape["fh"], shape["fw"], shape["co"], shape["stride"]) < 1:
            fail(f"invalid convolution dimensions for {name}")
        if layer.get("input_shape") != f"{shape['n']}x{shape['h']}x{shape['w']}x{shape['ci']}" or layer.get("output_shape") != f"{shape['n']}x{oh}x{ow}x{shape['co']}":
            fail(f"shape-string mismatch for {name}")
        if common["size_input_words"] != shape["n"] * shape["h"] * shape["w"] * shape["ci"] or common["size_weight_words"] != shape["fh"] * shape["fw"] * shape["ci"] * shape["co"] or common["size_output_words"] != shape["n"] * oh * ow * shape["co"]:
            fail(f"shape/size mismatch for {name}")
        common["output_height"] = oh
        common["output_width"] = ow
    elif operator == "fc":
        contract = layer.get("matmul")
        fields = ("rows", "inner", "cols")
        if not isinstance(contract, dict) or set(cli_values) != {f"--{field}" for field in fields}:
            fail(f"invalid FC shape contract for {name}")
        shape = {field: require_int(contract.get(field), f"{name}.matmul.{field}", 1) for field in fields}
        if any(cli_values[f"--{field}"] != shape[field] for field in fields):
            fail(f"CLI/manifest FC mismatch for {name}")
        if layer.get("input_shape") != f"{shape['rows']}x{shape['inner']}" or layer.get("output_shape") != f"{shape['rows']}x{shape['cols']}":
            fail(f"shape-string mismatch for {name}")
        if common["size_input_words"] != shape["rows"] * shape["inner"] or common["size_weight_words"] != shape["inner"] * shape["cols"] or common["size_output_words"] != shape["rows"] * shape["cols"] or common["cross_terms"] != shape["rows"] * shape["inner"] * shape["cols"]:
            fail(f"shape/size/cross-term mismatch for {name}")
        common["output_height"] = 0
        common["output_width"] = 0
    else:
        fail(f"unsupported operator for {name}")
    return common


def verify_source_manifest(repo_root: pathlib.Path, snapshot_path: pathlib.Path,
                           document: dict[str, Any],
                           probe_timeout: int) -> list[dict[str, Any]]:
    if document.get("schema") != SOURCE_SCHEMA or document.get("model") != "ResNet18":
        fail("unsupported source execution-manifest schema/model")
    source = document.get("source_layer_manifest")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        fail("missing source-layer manifest binding")
    profile = document.get("profile")
    if not isinstance(profile, dict):
        fail("source execution manifest lacks a profile")
    execution_ole_n = require_int(profile.get("ole_n"), "profile.ole_n", 1)
    builder = repo_root / "GPU-MPC/ringlpn/scripts/build_full_linear_model_manifest.py"
    command = [
        sys.executable, str(builder), "--repo-root", str(repo_root),
        "--layer-manifest", str(repo_root / source["path"]),
        "--model", "ResNet18", "--ole-n", str(execution_ole_n),
        "--out", str(snapshot_path), "--check",
    ]
    completed = run_bounded_capture(
        command, "source execution-manifest builder", probe_timeout
    )
    if completed.returncode != 0:
        fail("source execution manifest is stale or source-invalid: " + completed.stderr.strip())
    layers = document.get("layers")
    summary = document.get("summary")
    if not isinstance(layers, list) or not isinstance(profile, dict) or not isinstance(summary, dict):
        fail("source manifest lacks layers/profile/summary")
    if require_int(summary.get("linear_layers"), "linear layer count", 1) != 21 or len(layers) != 21:
        fail("source manifest must contain exactly 21 linear layers")
    if profile.get("qbits") not in (64, 128) or profile.get("noise") not in ("uniform", "regular") or require_int(profile.get("bw"), "profile.bw", 3) > 32:
        fail("unsupported source profile")
    for field in ("ole_n", "ole_c", "ole_t"):
        require_int(profile.get(field), f"profile.{field}", 1)
    expected_profile_scope = {
        "public_ring_vector_scope":
            "one_jointly_seeded_shake256_random_oracle_per_linear_layer_independent_domain_separated_vector_per_ring_ole",
        "ring_lpn_assumption_scope":
            "independent_public_vector_ring_lpn_in_shake256_random_oracle_model_no_concrete_security_claim",
    }
    for field, expected in expected_profile_scope.items():
        if profile.get(field) != expected:
            fail(f"source profile has unsupported {field}")
    seen_names: set[str] = set()
    seen_compatibility: set[str] = set()
    binaries: set[str] = set()
    for index, raw in enumerate(layers, start=1):
        if not isinstance(raw, dict):
            fail(f"layer {index} is not an object")
        name = require_string(raw.get("layer"), f"layer {index}.name")
        compatibility = require_hex_digest(raw.get("compatibility_id"), f"{name}.compatibility_id")
        if name in seen_names or compatibility in seen_compatibility:
            fail(f"duplicate layer contract at position {index}")
        if require_int(raw.get("linear_order"), f"{name}.linear_order", 1) != index:
            fail(f"non-contiguous linear order at {name}")
        expected_binary = "test_two_party_conv_preprocess" if raw.get("operator") == "conv2d" else "test_two_party_fc_preprocess"
        if raw.get("binary") != expected_binary:
            fail(f"wrong operator binary for {name}")
        truncation = raw.get("truncation")
        if not isinstance(truncation, dict):
            fail(f"missing source-only truncation metadata for {name}")
        bin_bits = require_int(truncation.get("bin"), f"{name}.truncation.bin", 3)
        bout = require_int(truncation.get("bout"), f"{name}.truncation.bout", 1)
        shift = require_int(truncation.get("shift"), f"{name}.truncation.shift", 1)
        if bin_bits != profile["bw"] or bout + shift != bin_bits:
            fail(f"invalid source-only truncation metadata for {name}")
        expected_plan_row(raw, profile)
        seen_names.add(name)
        seen_compatibility.add(compatibility)
        binaries.add(expected_binary)
    if binaries != BINARY_NAMES:
        fail("source manifest must use both FC and Conv executables")
    stock_nodes = document.get("stock_state_nodes")
    merges = document.get("residual_merges")
    key_stream = document.get("stock_key_stream")
    if not isinstance(stock_nodes, list) or [
            node.get("node") for node in stock_nodes if isinstance(node, dict)
    ] != [
        "globalaveragepool46", "flatten47", "gemm48_pre_sign_extend",
        "gemm48_terminal_output",
    ]:
        fail("source manifest lacks the exact four stock state nodes")
    if any(node.get("execution") !=
           "not_executed_by_linear_record_artifact" for node in stock_nodes):
        fail("stock state node execution scope is unsafe")
    expected_merges = [
        ("residual_stage1_block1", "conv5", "relu2", "identity"),
        ("residual_stage1_block2", "conv10", "relu7", "identity"),
        ("residual_stage2_block1", "conv15", "conv16", "projection"),
        ("residual_stage2_block2", "conv21", "relu18", "identity"),
        ("residual_stage3_block1", "conv26", "conv27", "projection"),
        ("residual_stage3_block2", "conv32", "relu29", "identity"),
        ("residual_stage4_block1", "conv37", "conv38", "projection"),
        ("residual_stage4_block2", "conv43", "relu40", "identity"),
    ]
    if not isinstance(merges, list) or [
            (merge.get("branch_id"), merge.get("main_operand"),
             merge.get("shortcut_operand"), merge.get("shortcut_kind"))
            for merge in merges if isinstance(merge, dict)
    ] != expected_merges:
        fail("source manifest residual topology mismatch")
    if any(merge.get("execution") !=
           "not_executed_by_linear_record_artifact" for merge in merges):
        fail("residual merge execution scope is unsafe")
    if (not isinstance(key_stream, list) or len(key_stream) != 62 or
            not all(isinstance(item, dict) for item in key_stream)):
        fail("source manifest stock key stream must contain 62 objects")
    if [item.get("stream_position") for item in key_stream
            if isinstance(item, dict)] != list(range(1, 63)):
        fail("source manifest stock key positions are not contiguous")
    if (sum(item.get("key_kind") == "GPUStTRKey" for item in key_stream) != 21
            or sum(item.get("ringlpn") is True for item in key_stream) != 21
            or key_stream[-1].get("key_kind") != "raw_mask[1000_words]"
            or any(item.get("execution") !=
                   "not_executed_by_linear_record_artifact"
                   for item in key_stream)):
        fail("source manifest stock key stream invariant mismatch")
    summary_expected = {
        "stock_state_nodes": 4, "residual_merges": 8,
        "projection_shortcuts": 3, "identity_shortcuts": 5,
        "stock_key_items": 62, "stock_stochastic_truncations": 21,
    }
    if any(summary.get(field) != value
           for field, value in summary_expected.items()):
        fail("source manifest stock topology summary mismatch")
    return layers


def parse_plan_csv(payload: str, layer_name: str) -> dict[str, Any]:
    try:
        rows = list(csv.reader(payload.splitlines()))
    except csv.Error as error:
        fail(f"cannot parse plan CSV for {layer_name}: {error}")
    if len(rows) != 1 or len(rows[0]) != len(PLAN_HEADER) or any(value == "" for value in rows[0]):
        fail(f"plan CSV for {layer_name} must contain exactly one complete operator row")
    parsed: dict[str, Any] = {PLAN_HEADER[0]: rows[0][0]}
    for field, value in zip(PLAN_HEADER[1:], rows[0][1:]):
        try:
            parsed[field] = int(value)
        except ValueError:
            fail(f"non-integer plan field {layer_name}.{field}")
    return parsed


def run_plan(binary: pathlib.Path, layer: dict[str, Any],
             profile: dict[str, Any], probe_timeout: int) -> dict[str, Any]:
    command = [
        str(binary), "--plan", "--qbits", str(profile["qbits"]), "--bw", str(profile["bw"]),
        "--ole-n", str(profile["ole_n"]), "--ole-c", str(profile["ole_c"]),
        "--ole-t", str(profile["ole_t"]), "--noise", str(profile["noise"]), *layer["cli"],
    ]
    completed = run_bounded_capture(
        command, f"plan probe for {layer['layer']}", probe_timeout
    )
    if completed.returncode != 0:
        fail(f"plan rejected for {layer['layer']}: {completed.stderr.strip()}")
    actual = parse_plan_csv(completed.stdout, layer["layer"])
    expected = expected_plan_row(layer, profile)
    for field in PLAN_HEADER:
        if actual[field] != expected[field]:
            fail(f"binary plan mismatch for {layer['layer']}.{field}")
    return actual


def load_single_csv_bytes(
        payload: bytes, label: str,
        expected_headers: tuple[str, ...]) -> dict[str, str]:
    try:
        with io.StringIO(payload.decode("utf-8"), newline="") as source:
            reader = csv.DictReader(source)
            headers = reader.fieldnames
            rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        fail(f"cannot read {label} CSV: {error}")
    if headers != list(expected_headers):
        fail(f"{label} CSV does not have the exact ordered schema")
    if len(rows) != 1 or None in rows[0] or set(rows[0]) != set(headers):
        fail(f"{label} CSV must contain exactly one well-formed row")
    if any(value is None or value == "" for value in rows[0].values()):
        fail(f"{label} CSV row has missing values")
    if rows[0].get("status") != "pass":
        fail(f"{label} CSV does not report pass")
    text_fields = (
        CHECKER_TEXT_FIELDS
        if "matched_dealer_keygen_contract" in rows[0]
        else PARTY_TEXT_FIELDS
    )
    for field, value in rows[0].items():
        if field in text_fields:
            continue
        if value == "NA":
            if field not in PARTY_NA_FIELDS:
                fail(f"{label} CSV has unexpected NA for {field}")
            continue
        try:
            number = float(value)
        except ValueError:
            fail(f"{label} CSV has non-numeric value for {field}")
        if not math.isfinite(number) or number < 0:
            fail(f"{label} CSV has invalid numeric value for {field}")
    return rows[0]


def load_single_csv(path: pathlib.Path, label: str,
                    expected_headers: tuple[str, ...]) -> dict[str, str]:
    return load_single_csv_bytes(
        read_bytes_once(path, f"{label} CSV"), label, expected_headers
    )


def compare_fields(row: dict[str, str], expected: dict[str, str], label: str) -> None:
    for field, value in expected.items():
        if field not in row:
            fail(f"{label} CSV is missing required header {field}")
        if row[field] != value:
            fail(f"{label} CSV mismatch for {field}")


def require_metric_uint(row: dict[str, str], field: str, label: str) -> int:
    value = row.get(field)
    if value is None or not value.isdigit():
        fail(f"{label} CSV requires an unsigned integer for {field}")
    return int(value)

def validate_metrics(layer: dict[str, Any], profile: dict[str, Any],
                     invocation: str, party0: dict[str, str],
                     party1: dict[str, str],
                     checker: dict[str, str]) -> str:
    name = layer["layer"]
    party_expected = {
        "qbits": str(profile["qbits"]), "bw": str(profile["bw"]),
        "ole_n": str(profile["ole_n"]), "ole_c": str(profile["ole_c"]),
        "ole_t": str(profile["ole_t"]), "noise": str(profile["noise"]),
        "ring_batches": str(layer["ring_batches"]),
        "ring_application_slots": str(layer["ring_application_slots"]),
        "ring_bootstrap_slots": str(layer["ring_bootstrap_slots"]),
        "invocation_id": invocation, "status": "pass",
        "transport_bytes_include_base_ot": "yes",
        "base_ot_setup_bytes_received": "NA",
        "base_ot_setup_dependency_rounds": "NA",
        "ot_backend": "sci-iknp",
        "ot_backend_revision": "SCI-IKNP-IN-TREE",
        "ot_correlation_straight_bytes_received": "NA",
        "ot_correlation_reversed_bytes_received": "NA",
        "ot_adjustment_bytes_received": "NA",
        "ot_ciphertext_bytes_received": "NA",
        "ot_backend_review_status": "existing-default",
        "ot_backend_bridge_sha256": "NA",
    }
    party_expected.update({field: "NA" for field in PARTY_NA_FIELDS})
    checker_expected = {
        "qbits": str(profile["qbits"]), "bw": str(profile["bw"]),
        "ring_batches": str(layer["ring_batches"]),
        "invocation_id": invocation,
        "final_payload_bytes_per_party":
            str(layer["linear_key_payload_bytes_per_party"]),
        "matched_dealer_keygen_contract": "pass", "key_order": "pass",
        "status": "pass",
    }
    if layer["operator"] == "conv2d":
        shape = layer["conv"]
        shape_expected = {field: str(shape[field]) for field in (
            "n", "h", "w", "ci", "fh", "fw", "co", "padding", "stride",
            "oh", "ow",
        )}
        checker_expected["unchanged_online_contract"] = "pass"
    else:
        shape = layer["matmul"]
        shape_expected = {
            field: str(shape[field]) for field in ("rows", "inner", "cols")
        }
        checker_expected["online_contract"] = "pass"
    party_expected.update(shape_expected)
    checker_expected.update(shape_expected)



    qbits = require_int(profile.get("qbits"), "profile.qbits", 64)
    if qbits % 64 != 0:
        fail(f"{name} qbits is not a whole number of 64-bit CRT limbs")
    limbs = qbits // 64
    batches = require_int(layer.get("ring_batches"), f"{name}.ring_batches", 1)
    cross_terms = require_int(layer.get("cross_terms"), f"{name}.cross_terms", 1)
    application_slots = require_int(
        profile.get("ring_application_slots"), "profile.ring_application_slots", 1
    )
    bootstrap_slots = require_int(
        profile.get("ring_bootstrap_slots"), "profile.ring_bootstrap_slots", 1
    )
    ole_n = require_int(profile.get("ole_n"), "profile.ole_n", 1)
    ole_c = require_int(profile.get("ole_c"), "profile.ole_c", 1)
    ole_t = require_int(profile.get("ole_t"), "profile.ole_t", 1)
    ring_instances = 2 * limbs * batches
    slots_used = 2 * limbs * cross_terms
    slot_capacity = ring_instances * application_slots
    if slots_used > slot_capacity:
        fail(f"{name} uses more Ring-OLE application slots than available")
    dpf_scalar_oles = ring_instances * bootstrap_slots
    epoch_zero_oles = limbs * bootstrap_slots
    dpf_pcg_oles = dpf_scalar_oles - epoch_zero_oles
    invariant_counters = {
        "ring_ole_instances": ring_instances,
        "slots_used": slots_used,
        "dpf_trees": ring_instances * 2 * ole_c * ole_t * ole_t,
        "dpf_scalar_oles": dpf_scalar_oles,
        "dpf_epoch_zero_scalar_oles": epoch_zero_oles,
        "dpf_pcg_scalar_oles": dpf_pcg_oles,
        "dpf_pcg_oles_reserved": dpf_scalar_oles,
        "dpf_pcg_oles_discarded": epoch_zero_oles,
        "dpf_pcg_opening_words_sent": dpf_pcg_oles,
        "public_a_seed_words_sent": 4,
        "ring_application_slots_discarded": slot_capacity - slots_used,
    }

    for party, party_number in ((party0, "0"), (party1, "1")):
        label = f"{name} party {party_number}"
        compare_fields(
            party, {"party": party_number, **party_expected}, label
        )
        for field, expected in invariant_counters.items():
            if require_metric_uint(party, field, label) != expected:
                fail(f"{label} CSV violates the {field} invariant")
        breadth_calls = require_metric_uint(
            party, "dpf_breadth_evaluator_calls", label
        )
        root_to_leaf_calls = require_metric_uint(
            party, "dpf_root_to_leaf_evaluator_calls", label
        )
        if breadth_calls == 0 and root_to_leaf_calls == 0:
            fail(f"{label} CSV reports no DPF evaluator path")
    compare_fields(checker, checker_expected, f"{name} checker")
    ledger_digest = require_hex_digest(
        party0.get("ledger_digest"), f"{name}.ledger_digest"
    )
    if (party1.get("ledger_digest") != ledger_digest or
            checker.get("ledger_digest") != ledger_digest):
        fail(f"{name} ledger binding differs across party/checker CSVs")
    return ledger_digest
class SchedulerCancelled(RuntimeError):
    """A worker stopped because another resource lane failed."""


def wait_for_p0_claim(process: subprocess.Popen[bytes],
                      claim_path: pathlib.Path, deadline: float,
                      cancelled: threading.Event) -> None:
    while True:
        if cancelled.is_set():
            terminate_and_reap(process)
            raise SchedulerCancelled("resource-lane execution cancelled")
        return_code = process.poll()
        if return_code is not None:
            fail(f"party 0 exited before publishing its claim (rc={return_code})")
        try:
            metadata = claim_path.lstat()
        except FileNotFoundError:
            metadata = None
        except OSError as error:
            fail(f"cannot inspect party 0 claim {claim_path}: {error}")
        if metadata is not None:
            if (not stat.S_ISREG(metadata.st_mode) or
                    metadata.st_uid != os.geteuid() or
                    stat.S_IMODE(metadata.st_mode) & 0o077 or
                    metadata.st_nlink != 1 or metadata.st_size != 132):
                fail("party 0 published a malformed freshness claim")
            return
        if time.monotonic() >= deadline:
            fail("party 0 did not publish its freshness claim before timeout")
        time.sleep(0.01)


def terminate_and_reap(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group = None
    shared_group = process_group == os.getpgrp()
    try:
        if shared_group:
            process.terminate()
        elif process_group is not None:
            os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if shared_group:
                process.kill()
            elif process_group is not None:
                os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return
    if not shared_group and process_group is not None:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_bounded_capture(
        command: list[str], label: str,
        timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    process: subprocess.Popen[str] | None = None
    try:
        isolate_process = os.environ.get(
            "RINGLPN_COOPERATIVE_PROCESS_GROUP", "0"
        ) != "1"
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=isolate_process,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_and_reap(process)
        if process is not None:
            process.communicate()
        fail(f"{label} timed out after {timeout_seconds} seconds")
    except OSError as error:
        fail(f"{label} failed to execute: {error}")
    assert process is not None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def wait_pair(p0: subprocess.Popen[bytes], p1: subprocess.Popen[bytes],
              deadline: float,
              cancelled: threading.Event) -> tuple[int, int]:
    processes = (p0, p1)
    while True:
        if cancelled.is_set():
            for process in processes:
                terminate_and_reap(process)
            raise SchedulerCancelled("resource-lane execution cancelled")
        rc0 = p0.poll()
        rc1 = p1.poll()
        if rc0 is not None and rc0 != 0 and rc1 is None:
            terminate_and_reap(p1)
            return rc0, p1.returncode if p1.returncode is not None else 1
        if rc1 is not None and rc1 != 0 and rc0 is None:
            terminate_and_reap(p0)
            return p0.returncode if p0.returncode is not None else 1, rc1
        if rc0 is not None and rc1 is not None:
            return rc0, rc1
        if time.monotonic() >= deadline:
            for process in processes:
                terminate_and_reap(process)
            return 124, 124
        time.sleep(0.1)


def wait_process(process: subprocess.Popen[bytes], deadline: float,
                 cancelled: threading.Event) -> int:
    while True:
        if cancelled.is_set():
            terminate_and_reap(process)
            raise SchedulerCancelled("resource-lane execution cancelled")
        return_code = process.poll()
        if return_code is not None:
            return return_code
        if time.monotonic() >= deadline:
            terminate_and_reap(process)
            return 124
        time.sleep(0.1)


def reject_symlink_components(path: pathlib.Path, label: str) -> None:
    absolute = path.absolute()
    current = pathlib.Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            fail(f"cannot inspect {label} component {current}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} must not contain symlink components: {current}")


def paths_overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


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

def parse_lane_descriptor(raw: str, index: int) -> dict[str, int]:
    fields = raw.split(":")
    if len(fields) != 4:
        fail(
            f"malformed resource lane {index}; expected "
            "P0_GPU:P1_GPU:CHECK_GPU:FIRST_PORT-LAST_PORT"
        )
    port_fields = fields[3].split("-")
    if len(port_fields) != 2:
        fail(
            f"malformed resource lane {index}; expected an inclusive "
            "FIRST_PORT-LAST_PORT range"
        )
    try:
        p0_gpu, p1_gpu, check_gpu = (int(field, 10) for field in fields[:3])
        port_first, port_last = (int(field, 10) for field in port_fields)
    except ValueError:
        fail(f"malformed resource lane {index}; ordinals and ports must be integers")
    if min(p0_gpu, p1_gpu, check_gpu) < 0:
        fail(f"resource lane {index} has an invalid GPU ordinal")
    if len({p0_gpu, p1_gpu, check_gpu}) != 3:
        fail(f"resource lane {index} party/checker GPUs must be pairwise distinct")
    if port_first <= 0 or port_last > 65535 or port_last < port_first + 85:
        fail(f"resource lane {index} port range is invalid or too small for 21 jobs")
    reject_linux_ephemeral_overlap(
        port_first, port_last, f"resource lane {index} port range"
    )
    return {
        "lane": index,
        "p0_gpu": p0_gpu,
        "p1_gpu": p1_gpu,
        "check_gpu": check_gpu,
        "port_first": port_first,
        "port_last": port_last,
    }


def validate_resource_lanes(lanes: list[dict[str, int]], job_count: int) -> None:
    if not lanes:
        fail("at least one resource lane is required")
    if len(lanes) > job_count:
        fail("resource-lane count exceeds the number of linear jobs")
    for left_index, left in enumerate(lanes):
        if min(left["p0_gpu"], left["p1_gpu"], left["check_gpu"]) < 0:
            fail(f"resource lane {left['lane']} has an invalid GPU ordinal")
        if len({left["p0_gpu"], left["p1_gpu"], left["check_gpu"]}) != 3:
            fail(
                f"resource lane {left['lane']} party/checker GPUs "
                "must be pairwise distinct"
            )
        if (
            left["port_first"] <= 0
            or left["port_last"] > 65535
            or left["port_last"] < left["port_first"] + 85
        ):
            fail(
                f"resource lane {left['lane']} port range is invalid "
                "or too small for 21 jobs"
            )
        left_gpus = {left["p0_gpu"], left["p1_gpu"], left["check_gpu"]}
        for right in lanes[left_index + 1:]:
            right_gpus = {right["p0_gpu"], right["p1_gpu"], right["check_gpu"]}
            if left_gpus & right_gpus:
                fail(
                    f"resource lanes {left['lane']} and {right['lane']} "
                    "share a GPU ordinal"
                )
            if (left["port_first"] <= right["port_last"] and
                    right["port_first"] <= left["port_last"]):
                fail(
                    f"resource lanes {left['lane']} and {right['lane']} "
                    "have overlapping port ranges"
                )


def build_lpt_schedule(
        layers: list[dict[str, Any]], lanes: list[dict[str, int]]
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in lanes]
    loads = [0 for _ in lanes]
    if len(lanes) == 1:
        jobs = sorted(
            layers,
            key=lambda layer: require_int(
                layer.get("linear_order"), "linear_order", 1
            ),
        )
    else:
        jobs = sorted(
            layers,
            key=lambda layer: (
                -require_int(
                    layer.get("ring_batches"),
                    f"{layer.get('layer', 'layer')}.ring_batches",
                    1,
                ),
                require_int(layer.get("linear_order"), "linear_order", 1),
            ),
        )
    for layer in jobs:
        lane_index = min(range(len(lanes)), key=lambda index: (loads[index], index))
        assignments[lane_index].append(layer)
        loads[lane_index] += require_int(
            layer.get("ring_batches"), f"{layer['layer']}.ring_batches", 1
        )
    schedule = [
        {
            **lane,
            "estimated_ring_batches": loads[index],
            "linear_orders": [
                require_int(layer.get("linear_order"), "linear_order", 1)
                for layer in assignments[index]
            ],
        }
        for index, lane in enumerate(lanes)
    ]
    return assignments, schedule

def execute_schedule(
        lanes: list[dict[str, int]],
        assignments: list[list[dict[str, Any]]],
        run_job: Callable[
            [dict[str, Any], dict[str, int], threading.Event], dict[str, Any]
        ],
) -> tuple[list[dict[str, Any]], list[int]]:
    cancelled = TERMINATION_REQUESTED
    if cancelled.is_set():
        fail("resource-lane execution cancelled before launch")
    lock = threading.Lock()
    first_error: list[BaseException] = []
    completed: list[dict[str, Any]] = []
    completion_order: list[int] = []
    lane_leases = {lane["lane"]: threading.Lock() for lane in lanes}

    def lane_worker(
        lane: dict[str, int], jobs: list[dict[str, Any]]
    ) -> None:
        for layer in jobs:
            if cancelled.is_set():
                return
            try:
                with lane_leases[lane["lane"]]:
                    result = run_job(layer, lane, cancelled)
            except SchedulerCancelled:
                return
            except BaseException as error:
                with lock:
                    if not first_error:
                        first_error.append(error)
                cancelled.set()
                return
            with lock:
                completed.append(result)
                completion_order.append(
                    require_int(result.get("linear_order"), "result.linear_order", 1)
                )

    with ThreadPoolExecutor(
        max_workers=len(lanes), thread_name_prefix="linear-resource-lane"
    ) as executor:
        futures = [
            executor.submit(lane_worker, lane, assignments[index])
            for index, lane in enumerate(lanes)
        ]
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                with lock:
                    if not first_error:
                        first_error.append(error)
                cancelled.set()
    if first_error:
        raise first_error[0]
    if len(completed) != sum(len(jobs) for jobs in assignments):
        fail("resource-lane scheduler stopped without a reported worker failure")
    completed.sort(key=lambda result: result["linear_order"])
    return completed, completion_order


def prepare_ledger_root(raw: pathlib.Path, output_root: pathlib.Path) -> pathlib.Path:
    reject_symlink_components(raw, "ledger root")
    try:
        raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as error:
        fail(f"cannot create ledger root {raw}: {error}")
    reject_symlink_components(raw, "ledger root")
    ledger = raw.resolve(strict=True)
    try:
        metadata = ledger.lstat()
    except OSError as error:
        fail(f"cannot inspect ledger root {ledger}: {error}")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        fail("ledger root must be an owner-controlled directory")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        fail("ledger root must not grant group or other permissions")
    if paths_overlap(ledger, output_root.resolve(strict=False)):
        fail("ledger root and output root must be outside and non-nested")
    return ledger


def common_cli(profile: dict[str, Any], sid: int, invocation: str,
               ledger: pathlib.Path) -> list[str]:
    return [
        "--sid", str(sid), "--invocation-id", invocation, "--ledger", str(ledger),
        "--qbits", str(profile["qbits"]), "--bw", str(profile["bw"]),
        "--ole-n", str(profile["ole_n"]), "--ole-c", str(profile["ole_c"]),
        "--ole-t", str(profile["ole_t"]), "--noise", str(profile["noise"]),
    ]


def artifact_binding(path: pathlib.Path, root: pathlib.Path,
                     include_bytes: bool = False) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
    }
    if include_bytes:
        binding["bytes"] = path.stat().st_size
    return binding


def safe_artifact(root: pathlib.Path, binding: Any, label: str) -> pathlib.Path:
    if (not isinstance(binding, dict) or
            set(binding) not in ({"path", "sha256"},
                                 {"path", "bytes", "sha256"})):
        fail(f"invalid {label} artifact binding")
    relative_text = require_string(binding.get("path"), f"{label}.path")
    require_hex_digest(binding.get("sha256"), f"{label}.sha256")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None:
        expected_bytes = require_int(expected_bytes, f"{label}.bytes", 1)
    relative = pathlib.PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        fail(f"unsafe {label} artifact path")
    candidate = root.joinpath(*relative.parts)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        fail(f"missing {label} artifact {candidate}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} artifact is not a regular non-symlink file")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        fail(f"changed {label} artifact size")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        fail(f"unsafe resolved {label} artifact path")
    if sha256(candidate) != binding["sha256"]:
        fail(f"changed {label} artifact")
    return candidate


def read_verified_artifact(
        path: pathlib.Path, binding: dict[str, Any], label: str) -> bytes:
    payload = read_bytes_once(path, label)
    if digest_bytes(payload) != binding.get("sha256"):
        fail(f"changed {label} artifact")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and len(payload) != expected_bytes:
        fail(f"changed {label} artifact size")
    return payload


def verify_self_digest(document: dict[str, Any], field: str, label: str) -> str:
    claimed = require_hex_digest(document.get(field), f"{label}.{field}")
    unsigned = dict(document)
    del unsigned[field]
    if digest_bytes(canonical(unsigned)) != claimed:
        fail(f"invalid {label} self-digest")
    return claimed

def validate_portable_provenance_text(value: Any, label: str) -> str:
    text = require_string(value, label)
    placeholders = set(re.findall(r"\$\{[^}]+\}", text))
    if not placeholders.issubset(PROVENANCE_PATH_TOKENS):
        fail(f"{label} contains an unsupported path token")
    if ("/tmp/ringlpn-linear-provenance." in text or
            "/home/" in text or "/work/EzPC/" in text):
        fail(f"{label} contains a workstation-specific path")
    return text



def validate_binary_approval(
        payload: bytes, provenance_payload: bytes, label: str) -> dict[str, str]:
    document = parse_json_bytes(payload, label)
    if (document.get("schema") != BINARY_APPROVAL_SCHEMA or
            document.get("scope") != BINARY_APPROVAL_SCOPE or
            document.get("classification") != BINARY_APPROVAL_CLASSIFICATION):
        fail("unsupported binary approval schema/scope")
    if set(document) != {
            "approval_digest", "binaries", "build_provenance",
            "classification", "schema", "scope", "validation",
    }:
        fail("binary approval has unexpected fields")
    verify_self_digest(document, "approval_digest", "binary approval")
    validation = document.get("validation")
    unavailable = {"receipt": None, "status": "unavailable"}
    if (not isinstance(validation, dict) or
            set(validation) != {
                "advisor_review", "deterministic_build",
                "internal_source_review",
            } or
            validation.get("advisor_review") != unavailable or
            validation.get("internal_source_review") != unavailable):
        fail("binary approval asserts unavailable human review")
    provenance = parse_json_bytes(
        provenance_payload, f"{label} build provenance"
    )
    if (provenance.get("schema") != BINARY_BUILD_PROVENANCE_SCHEMA or
            provenance.get("classification") !=
            BINARY_APPROVAL_CLASSIFICATION or
            set(provenance) != {
                "builds", "byte_comparison", "classification",
                "provenance_digest", "schema",
            }):
        fail("unsupported binary build provenance schema/scope")
    provenance_digest = verify_self_digest(
        provenance, "provenance_digest", "binary build provenance"
    )
    builds = provenance.get("builds")
    if not isinstance(builds, list) or len(builds) != 2:
        fail("binary approval requires two independent build receipts")
    receipt_digests: list[str] = []
    receipt_outputs: list[dict[str, Any]] = []
    for index, receipt in enumerate(builds, 1):
        if (not isinstance(receipt, dict) or
                set(receipt) != {
                    "adapters", "build_id", "environment",
                    "external_inputs", "receipt_digest", "tools",
                    "tracked_inputs",
                } or
                receipt.get("build_id") != f"independent-build-{index}"):
            fail("invalid independent adapter build receipt")
        receipt_digests.append(
            verify_self_digest(
                receipt, "receipt_digest",
                f"independent adapter build receipt {index}",
            )
        )
        tracked_inputs = receipt.get("tracked_inputs")
        external_inputs = receipt.get("external_inputs")
        tools = receipt.get("tools")
        environments = receipt.get("environment")
        if (not isinstance(tracked_inputs, dict) or not tracked_inputs or
                not isinstance(external_inputs, dict) or
                not external_inputs or
                not isinstance(tools, dict) or
                not {
                    "cuda_cicc", "cuda_fatbinary", "cuda_nvlink",
                    "cuda_ptxas", "host_assembler", "host_cc1plus",
                    "host_collect2", "host_cxx", "host_linker", "nvcc",
                    "objcopy",
                }.issubset(tools) or
                not isinstance(environments, dict) or
                set(environments) != {"fc", "conv"}):
            fail("build receipt omits source, system, tool, or environment inputs")
        for kind, environment in environments.items():
            if (not isinstance(environment, dict) or
                    set(environment) != set(PROVENANCE_ENVIRONMENT_KEYS)):
                fail("build receipt environment is not normalized")
            expected_environment = {
                "HOME": "${BUILD_ROOT}/home",
                "LANG": "C",
                "LC_ALL": "C",
                "PWD": "${REPO}/GPU-MPC/ringlpn",
                "RINGLPN_CANONICAL_BUILD_ACTIVE": "1",
                "RINGLPN_COMPONENT_DISPATCH_ACTIVE": "1",
                "RINGLPN_LINEAR_COMMAND_FILE":
                    f"${{BUILD_ROOT}}/build-{index}/{kind}.commands",
                "RINGLPN_LINEAR_DEPFILE":
                    f"${{BUILD_ROOT}}/build-{index}/{kind}.d",
                "RINGLPN_LINEAR_ENVIRONMENT_FILE":
                    f"${{BUILD_ROOT}}/build-{index}/{kind}.environment",
                "RINGLPN_LINEAR_KIND": kind,
                "RINGLPN_LINEAR_LINK_MAP":
                    f"${{BUILD_ROOT}}/build-{index}/{kind}.map",
                "SOURCE_DATE_EPOCH": "0",
                "TMPDIR": f"${{BUILD_ROOT}}/tmp-{index}",
                "TZ": "UTC",
                "ZERO_AR_DATE": "1",
            }
            if any(environment.get(name) != value
                   for name, value in expected_environment.items()):
                fail("build receipt normalized environment differs")
            for name, value in environment.items():
                validate_portable_provenance_text(
                    value, f"{kind} build environment {name}"
                )
        for relative, binding in tracked_inputs.items():
            pure = pathlib.PurePosixPath(relative)
            if (not isinstance(relative, str) or pure.is_absolute() or
                    "." in pure.parts or ".." in pure.parts or
                    not isinstance(binding, dict) or
                    set(binding) != {"sha256", "size"}):
                fail("build receipt has a noncanonical tracked source binding")
            require_hex_digest(binding.get("sha256"), "tracked source sha256")
            require_int(binding.get("size"), "tracked source size", 1)
        for external_path, binding in external_inputs.items():
            if (not isinstance(external_path, str) or
                    not pathlib.PurePosixPath(external_path).is_absolute() or
                    not isinstance(binding, dict) or
                    set(binding) != {"kinds", "sha256", "size"} or
                    not isinstance(binding.get("kinds"), list) or
                    not binding["kinds"]):
                fail("invalid external compiler/system dependency binding")
            require_hex_digest(
                binding.get("sha256"), "external dependency sha256"
            )
            require_int(binding.get("size"), "external dependency size", 1)
        for tool, binding in tools.items():
            if (not isinstance(binding, dict) or
                    set(binding) != {"path", "sha256", "size", "version"} or
                    not isinstance(binding.get("path"), str) or
                    not pathlib.PurePosixPath(binding["path"]).is_absolute() or
                    (binding.get("version") is not None and
                     not isinstance(binding.get("version"), str))):
                fail(f"invalid build tool binding: {tool}")
            require_hex_digest(binding.get("sha256"), f"{tool} sha256")
            require_int(binding.get("size"), f"{tool} size", 1)
        adapters = receipt.get("adapters")
        if not isinstance(adapters, dict) or set(adapters) != BINARY_NAMES:
            fail("build receipt must contain exactly the FC/Conv adapters")
        outputs: dict[str, Any] = {}
        for name, adapter in adapters.items():
            if (not isinstance(adapter, dict) or
                    set(adapter) != {"commands", "output"} or
                    not isinstance(adapter.get("commands"), list) or
                    len(adapter["commands"]) != 2):
                fail(f"build receipt lacks exact commands for {name}")
            expected_cwds = [
                "${CANONICAL_SOURCE}/GPU-MPC/ringlpn",
                "${REPO}/GPU-MPC/ringlpn",
            ]
            for command_index, command in enumerate(adapter["commands"]):
                if (not isinstance(command, dict) or
                        set(command) != {"argv", "cwd"} or
                        command.get("cwd") != expected_cwds[command_index] or
                        not isinstance(command.get("argv"), list) or
                        not command["argv"]):
                    fail(f"build receipt has an invalid portable command for {name}")
                validate_portable_provenance_text(
                    command["cwd"], f"{name} command cwd"
                )
                for argument in command["argv"]:
                    validate_portable_provenance_text(
                        argument, f"{name} command argument"
                    )
            output = adapter.get("output")
            if not isinstance(output, dict) or set(output) != {"sha256", "size"}:
                fail(f"invalid build receipt output for {name}")
            require_hex_digest(output.get("sha256"), f"{name} output sha256")
            require_int(output.get("size"), f"{name} output size", 1)
            outputs[name] = output
        receipt_outputs.append(outputs)
    comparison = provenance.get("byte_comparison")
    if (not isinstance(comparison, dict) or
            comparison.get("status") != "identical" or
            set(comparison) != {"binaries", "status"} or
            comparison.get("binaries") != receipt_outputs[0] or
            receipt_outputs[0] != receipt_outputs[1]):
        fail("independent adapter build receipts are not byte-identical")
    binding = document.get("build_provenance")
    if (not isinstance(binding, dict) or
            set(binding) != {
                "path", "provenance_digest", "receipt_digests",
                "sha256", "size",
            }):
        fail("binary approval lacks exact build provenance binding")
    provenance_path = pathlib.PurePosixPath(
        require_string(binding.get("path"), "build_provenance.path")
    )
    if (provenance_path.is_absolute() or "." in provenance_path.parts or
            ".." in provenance_path.parts or
            digest_bytes(provenance_payload) !=
            require_hex_digest(binding.get("sha256"), "build_provenance.sha256") or
            len(provenance_payload) !=
            require_int(binding.get("size"), "build_provenance.size", 1) or
            binding.get("provenance_digest") != provenance_digest or
            binding.get("receipt_digests") != receipt_digests):
        fail("binary approval build provenance binding differs")
    deterministic = validation.get("deterministic_build")
    if deterministic != {
            "byte_comparison": "identical",
            "receipt_digests": receipt_digests,
            "status": "pass",
    }:
        fail("automated approval status is not hash-bound to both receipts")
    binaries = document.get("binaries")
    if binaries != comparison["binaries"]:
        fail("binary approval outputs differ from build receipts")
    return {
        name: require_hex_digest(
            binding.get("sha256"), f"binary approval.{name}.sha256"
        )
        for name, binding in binaries.items()
    }


def verify_record_set(
        manifest_path: pathlib.Path,
        artifact_root: pathlib.Path | None = None) -> str:
    if manifest_path.name != "LINEAR_RECORD_SET.manifest":
        fail("record-set verifier requires LINEAR_RECORD_SET.manifest")
    reject_symlink_components(manifest_path, "record-set manifest")
    if artifact_root is None:
        root = manifest_path.parent.resolve(strict=True)
    else:
        reject_symlink_components(artifact_root, "record-set artifact root")
        root = artifact_root.resolve(strict=True)
        if manifest_path.parent.resolve(strict=True) == root:
            fail("separate artifact root is redundant")
    if stat.S_IMODE(root.stat().st_mode) != 0o700:
        fail("record-set run root must have mode 0700")
    manifest_bytes = read_bytes_once(manifest_path, "record-set manifest")
    document = parse_json_bytes(manifest_bytes, str(manifest_path))
    schema = document.get("schema")
    if schema == RECORD_SET_SCHEMA:
        stateful = False
        composition_status = COMPOSITION_STATUS
        artifact_scope = ARTIFACT_SCOPE
        execution_exclusions = EXECUTION_EXCLUSIONS
        artifact_labels = ARTIFACT_LABELS
    elif schema == STATEFUL_RECORD_SET_SCHEMA:
        stateful = True
        composition_status = STATEFUL_COMPOSITION_STATUS
        artifact_scope = STATEFUL_ARTIFACT_SCOPE
        execution_exclusions = STATEFUL_EXECUTION_EXCLUSIONS
        artifact_labels = STATEFUL_ARTIFACT_LABELS
    else:
        fail("unsupported record-set schema")
    if (document.get("composition_status") != composition_status or
            document.get("artifact_scope") != artifact_scope or
            document.get("security_scope") != SECURITY_SCOPE or
            document.get("execution_exclusions") != execution_exclusions):
        fail("unsupported record-set claim scope")
    record_set_digest = verify_self_digest(document, "record_set_digest", "record set")
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o400:
        fail("record-set manifest must be an immutable private snapshot")
    source_binding = document.get("source_execution_manifest")
    if not isinstance(source_binding, dict) or source_binding.get("path") != "private_inputs/source_execution_manifest.json":
        fail("source execution-manifest snapshot has unexpected path")
    source_path = safe_artifact(root, source_binding, "source execution manifest")
    source_document = parse_json_bytes(
        read_verified_artifact(
            source_path, source_binding, "source execution-manifest snapshot"
        ),
        str(source_path),
    )
    if stat.S_IMODE(source_path.stat().st_mode) != 0o400:
        fail("source execution-manifest snapshot must have mode 0400")
    if stat.S_IMODE(source_path.parent.stat().st_mode) != 0o700:
        fail("private snapshot directory must have mode 0700")
    if (source_document.get("schema") != SOURCE_SCHEMA or
            source_document.get("plan_digest") != document.get("source_plan_digest") or
            source_document.get("profile") != document.get("profile")):
        fail("source execution-manifest binding mismatch")
    source_layers = source_document.get("layers")
    profile = source_document.get("profile")
    if not isinstance(source_layers, list) or len(source_layers) != 21 or not isinstance(profile, dict):
        fail("invalid bound source execution manifest")
    planned_binding = document.get("planned")
    if not isinstance(planned_binding, dict) or set(planned_binding) != {"path", "sha256", "digest"}:
        fail("invalid planned artifact binding")
    if planned_binding["path"] != "PLANNED.json":
        fail("planned snapshot has unexpected path")
    planned_path = safe_artifact(root, {"path": planned_binding["path"], "sha256": planned_binding["sha256"]}, "planned")
    planned_document = parse_json_bytes(
        read_verified_artifact(planned_path, planned_binding, "planned artifact"),
        str(planned_path),
    )
    if (planned_document.get("schema") != PLAN_SCHEMA or
            planned_document.get("composition_status") != composition_status or
            planned_document.get("artifact_scope") != artifact_scope or
            planned_document.get("security_scope") != SECURITY_SCOPE or
            planned_document.get("execution_exclusions") !=
            execution_exclusions or
            verify_self_digest(planned_document, "digest", "planned") !=
            planned_binding.get("digest")):
        fail("planned schema/status/self-digest binding mismatch")
    if (planned_document.get("source_execution_manifest") != source_binding or
            planned_document.get("source_plan_digest") != document.get("source_plan_digest") or
            planned_document.get("profile") != profile or
            planned_document.get("plan_csv_header") != list(PLAN_HEADER)):
        fail("planned source/profile/header binding mismatch")
    if stat.S_IMODE(planned_path.stat().st_mode) != 0o400:
        fail("planned snapshot must have mode 0400")
    binary_bindings = document.get("executable_provenance")
    if not isinstance(binary_bindings, dict) or set(binary_bindings) != BINARY_NAMES:
        fail("record set lacks exact FC/Conv binary bindings")
    if planned_document.get("executable_provenance") != binary_bindings:
        fail("planned/record-set binary bindings differ")
    approval_binding = document.get("binary_approval")
    if (not isinstance(approval_binding, dict) or
            approval_binding.get("path") !=
            "private_inputs/binary_approval.json"):
        fail("record set lacks the binary approval snapshot")
    approval_path = safe_artifact(
        root, approval_binding, "binary approval"
    )
    provenance_binding = document.get("linear_adapter_build_provenance")
    if (not isinstance(provenance_binding, dict) or
            provenance_binding.get("path") !=
            "private_inputs/linear_adapter_build_provenance.json"):
        fail("record set lacks the linear adapter build provenance snapshot")
    provenance_path = safe_artifact(
        root, provenance_binding, "linear adapter build provenance"
    )
    provenance_payload = read_verified_artifact(
        provenance_path, provenance_binding,
        "linear adapter build provenance",
    )
    approval_payload = read_verified_artifact(
        approval_path, approval_binding, "binary approval"
    )
    approved = validate_binary_approval(
        approval_payload, provenance_payload, str(approval_path),
    )
    if (planned_document.get("binary_approval") != approval_binding or
            planned_document.get("linear_adapter_build_provenance") !=
            provenance_binding or
            stat.S_IMODE(approval_path.stat().st_mode) != 0o400 or
            stat.S_IMODE(provenance_path.stat().st_mode) != 0o400):
        fail("planned/record-set approval or build provenance binding differs")
    expected_files = {
        (root / "LINEAR_RECORD_SET.manifest").resolve(strict=True),
        source_path, planned_path, approval_path, provenance_path,
    }
    for name, binding in binary_bindings.items():
        if (not isinstance(binding, dict) or
                binding.get("provenance") != VALIDATED_PROVENANCE or
                binding.get("sha256") != approved[name]):
            fail(f"invalid validated executable provenance for {name}")
        if binding.get("path") != f"private_inputs/{name}":
            fail(f"binary snapshot has unexpected path: {name}")
        path = safe_artifact(
            root, {"path": binding.get("path"), "sha256": binding.get("sha256")},
            f"binary {name}",
        )
        if stat.S_IMODE(path.stat().st_mode) != 0o500:
            fail(f"binary snapshot has mutable or non-private mode: {name}")
        expected_files.add(path)
    plan_layers = planned_document.get("layers")
    layers = document.get("layers")
    if not isinstance(plan_layers, list) or not isinstance(layers, list) or len(plan_layers) != 21 or len(layers) != 21:
        fail("record set and plan must each contain exactly 21 layers")
    resource_schedule = document.get("resource_schedule")
    scheduled_ports: dict[int, int] | None = None
    if resource_schedule is None:
        if planned_document.get("resource_schedule") is not None:
            fail("planned/record-set resource schedules differ")
    else:
        if (
            not isinstance(resource_schedule, dict)
            or resource_schedule != planned_document.get("resource_schedule")
            or set(resource_schedule)
            != {"algorithm", "stable_tie_break", "lanes"}
            or resource_schedule.get("algorithm")
            != "deterministic_lpt_by_ring_batches"
            or resource_schedule.get("stable_tie_break")
            != "canonical_linear_order_then_lane_order"
        ):
            fail("invalid or unbound record-set resource schedule")
        raw_lanes = resource_schedule.get("lanes")
        if not isinstance(raw_lanes, list):
            fail("invalid record-set resource lanes")
        lane_keys = {
            "lane", "p0_gpu", "p1_gpu", "check_gpu",
            "port_first", "port_last", "estimated_ring_batches",
            "linear_orders",
        }
        scheduled_lanes: list[dict[str, int]] = []
        for lane_index, raw_lane in enumerate(raw_lanes):
            if not isinstance(raw_lane, dict) or set(raw_lane) != lane_keys:
                fail(f"invalid record-set resource lane {lane_index}")
            lane = {
                field: require_int(
                    raw_lane.get(field), f"resource lane {lane_index}.{field}"
                )
                for field in (
                    "lane", "p0_gpu", "p1_gpu", "check_gpu",
                    "port_first", "port_last",
                )
            }
            if lane["lane"] != lane_index:
                fail("record-set resource lane order is noncanonical")
            scheduled_lanes.append(lane)
        validate_resource_lanes(scheduled_lanes, len(source_layers))
        _assignments, expected_schedule = build_lpt_schedule(
            source_layers, scheduled_lanes
        )
        expected_resource_schedule = {
            "algorithm": "deterministic_lpt_by_ring_batches",
            "stable_tie_break": "canonical_linear_order_then_lane_order",
            "lanes": expected_schedule,
        }
        if resource_schedule != expected_resource_schedule:
            fail("record-set resource schedule differs from deterministic LPT")
        if (
            document.get("p0_gpu") != scheduled_lanes[0]["p0_gpu"]
            or document.get("p1_gpu") != scheduled_lanes[0]["p1_gpu"]
            or document.get("check_gpu") != scheduled_lanes[0]["check_gpu"]
        ):
            fail("record-set primary resource-lane metadata differs")
        scheduled_ports = {}
        for lane in expected_schedule:
            for order in lane["linear_orders"]:
                scheduled_ports[order] = lane["port_first"] + 4 * (order - 1)
    expected_directories = {root, source_path.parent, planned_path.parent}
    seen_names: set[str] = set()
    for index, (source_layer, plan_layer, result) in enumerate(zip(source_layers, plan_layers, layers), start=1):
        if not isinstance(source_layer, dict) or not isinstance(plan_layer, dict) or not isinstance(result, dict):
            fail(f"invalid layer object at order {index}")
        name = source_layer.get("layer")
        compatibility = source_layer.get("compatibility_id")
        if (name in seen_names or result.get("linear_order") != index or
                plan_layer.get("linear_order") != index or result.get("layer") != name or
                plan_layer.get("layer") != name or result.get("compatibility_id") != compatibility or
                plan_layer.get("compatibility_id") != compatibility):
            fail(f"record-set/source/plan order mismatch at layer {index}")
        source_binary = source_layer.get("binary")
        if source_binary not in binary_bindings:
            fail(f"unknown source binary at layer {index}")
        if (plan_layer.get("operator") != source_layer.get("operator") or
                plan_layer.get("binary") != source_binary or
                plan_layer.get("binary_sha256") != binary_bindings[source_binary]["sha256"] or
                plan_layer.get("plan") != expected_plan_row(source_layer, profile)):
            fail(f"planned structured row/binary mismatch at layer {index}")
        if result.get("operator") != source_layer.get("operator") or result.get("binary") != source_layer.get("binary"):
            fail(f"record-set operator binding mismatch at layer {index}")
        if result.get("binary_sha256") != binary_bindings[result["binary"]]["sha256"]:
            fail(f"record-set binary digest mismatch at layer {index}")
        if scheduled_ports is not None and result.get("port") != scheduled_ports[index]:
            fail(f"record-set scheduled port mismatch at layer {index}")
        artifacts = result.get("artifacts")
        metrics = result.get("metrics")
        if (not isinstance(artifacts, dict) or
                set(artifacts) != artifact_labels or
                not isinstance(metrics, dict) or
                set(metrics) != {"party0", "party1", "checker"}):
            fail(f"invalid artifact/metrics bindings at layer {index}")
        suffix = ".conv" if source_layer.get("operator") == "conv2d" else ".fc"
        layer_prefix = f"{index:02d}_{name}"
        expected_artifact_paths = {
            "p0_record": f"{layer_prefix}/party0/key_p0{suffix}",
            "p1_record": f"{layer_prefix}/party1/key_p1{suffix}",
            "p0_metrics_csv": f"{layer_prefix}/party0/metrics.csv",
            "p1_metrics_csv": f"{layer_prefix}/party1/metrics.csv",
            "p0_log": f"{layer_prefix}/party0/party.log",
            "p1_log": f"{layer_prefix}/party1/party.log",
            "checker_csv": f"{layer_prefix}/checker.csv",
            "checker_log": f"{layer_prefix}/checker.log",
        }
        if stateful:
            expected_artifact_paths.update({
                "p0_state": f"{layer_prefix}/party0/mask.state",
                "p1_state": f"{layer_prefix}/party1/mask.state",
            })
        paths: dict[str, pathlib.Path] = {}
        for label, binding in artifacts.items():
            expected_fields = {"path", "sha256"}
            if stateful and label in MANIFEST_BOUND_PRIVATE_ARTIFACTS:
                expected_fields.add("bytes")
            if (not isinstance(binding, dict) or
                    set(binding) != expected_fields or
                    binding.get("path") != expected_artifact_paths[label]):
                fail(f"unexpected layer {index} {label} artifact binding")
            paths[label] = safe_artifact(
                root, binding, f"layer {index} {label}"
            )
            expected_files.add(paths[label])
            expected_directories.add(paths[label].parent)
        expected_record_bytes = require_int(
            source_layer.get("linear_record_bytes_per_party"),
            f"layer {index}.linear_record_bytes_per_party", 1)
        if (paths["p0_record"].stat().st_size != expected_record_bytes or
                paths["p1_record"].stat().st_size != expected_record_bytes):
            fail(f"record size differs from source manifest at layer {index}")
        party_header = (
            CONV_PARTY_HEADER
            if source_layer.get("operator") == "conv2d"
            else FC_PARTY_HEADER
        )
        checker_header = (
            CONV_CHECKER_HEADER
            if source_layer.get("operator") == "conv2d"
            else FC_CHECKER_HEADER
        )
        party0 = load_single_csv_bytes(
            read_verified_artifact(
                paths["p0_metrics_csv"], artifacts["p0_metrics_csv"],
                f"{name} party 0 metrics CSV",
            ),
            f"{name} party 0", party_header,
        )
        party1 = load_single_csv_bytes(
            read_verified_artifact(
                paths["p1_metrics_csv"], artifacts["p1_metrics_csv"],
                f"{name} party 1 metrics CSV",
            ),
            f"{name} party 1", party_header,
        )
        checker = load_single_csv_bytes(
            read_verified_artifact(
                paths["checker_csv"], artifacts["checker_csv"],
                f"{name} checker CSV",
            ),
            f"{name} checker", checker_header,
        )
        if metrics != {"party0": party0, "party1": party1, "checker": checker}:
            fail(f"stored metrics rows differ from CSV artifacts at layer {index}")
        invocation = require_string(result.get("invocation_id"), f"layer {index}.invocation_id")
        ledger_digest = validate_metrics(source_layer, profile, invocation, party0, party1, checker)
        if result.get("ledger_digest") != ledger_digest:
            fail(f"record-set ledger digest mismatch at layer {index}")
        seen_names.add(name)
    actual_files: set[pathlib.Path] = set()
    actual_directories = {root}
    for entry in root.rglob("*"):
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"unexpected symlink under run root: {entry}")
        if stat.S_ISREG(metadata.st_mode):
            actual_files.add(entry.resolve(strict=True))
        elif stat.S_ISDIR(metadata.st_mode):
            actual_directories.add(entry.resolve(strict=True))
        else:
            fail(f"unexpected non-file data under run root: {entry}")
    expected_directories.update(path.parent.resolve(strict=True) for path in expected_files)
    if actual_files != expected_files or actual_directories != expected_directories:
        fail("missing or extra data under record-set run root")
    return record_set_digest


def run_layer_job(
    layer: dict[str, Any],
    lane: dict[str, int],
    cancelled: threading.Event,
    *,
    output_root: pathlib.Path,
    ledger_root: pathlib.Path,
    profile: dict[str, Any],
    snapshot_binaries: dict[str, pathlib.Path],
    binary_bindings: dict[str, dict[str, str]],
    stateful: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    if cancelled.is_set():
        raise SchedulerCancelled("resource-lane execution cancelled")
    order = require_int(layer.get("linear_order"), "linear_order", 1)
    layer_dir = output_root / f"{order:02d}_{layer['layer']}"
    layer_dir.mkdir(mode=0o700)
    p0_dir = layer_dir / "party0"
    p1_dir = layer_dir / "party1"
    p0_dir.mkdir(mode=0o700)
    p1_dir.mkdir(mode=0o700)
    sid = secrets.randbelow((1 << 63) - 1) + 1
    invocation = secrets.token_hex(16)
    common = common_cli(profile, sid, invocation, ledger_root) + layer["cli"]
    port = lane["port_first"] + 4 * (order - 1)
    if port + 1 > lane["port_last"]:
        fail(f"resource lane {lane['lane']} has no port slot for layer {order}")
    binary = snapshot_binaries[layer["binary"]]
    p0_prefix = p0_dir / "key"
    p1_prefix = p1_dir / "key"
    channel_auth = bytearray(secrets.token_bytes(32))
    p0_auth = p0_dir / "channel-auth.key"
    p1_auth = p1_dir / "channel-auth.key"
    try:
        write_private_bytes(p0_auth, channel_auth, 0o600)
        write_private_bytes(p1_auth, channel_auth, 0o600)
    except BaseException:
        p0_auth.unlink(missing_ok=True)
        p1_auth.unlink(missing_ok=True)
        fsync_directory(p0_dir)
        fsync_directory(p1_dir)
        raise
    finally:
        for index in range(len(channel_auth)):
            channel_auth[index] = 0
        del channel_auth
    p0_command = [
        str(binary), "--party", "0", "--port", str(port),
        "--channel-auth-file", str(p0_auth),
        "--out-prefix", str(p0_prefix), "--csv-header", *common,
    ]
    p1_command = [
        str(binary), "--party", "1", "--host", "127.0.0.1",
        "--port", str(port), "--channel-auth-file", str(p1_auth),
        "--out-prefix", str(p1_prefix), "--csv-header", *common,
    ]
    p0_state = p0_dir / "mask.state"
    p1_state = p1_dir / "mask.state"
    if stateful:
        p0_command.extend(["--state-record", str(p0_state)])
        p1_command.extend(["--state-record", str(p1_state)])
    p0_metrics_path = p0_dir / "metrics.csv"
    p1_metrics_path = p1_dir / "metrics.csv"
    p0_log_path = p0_dir / "party.log"
    p1_log_path = p1_dir / "party.log"
    p0_process: subprocess.Popen[bytes] | None = None
    p1_process: subprocess.Popen[bytes] | None = None
    deadline = time.monotonic() + timeout_seconds
    isolate_workers = os.environ.get(
        "RINGLPN_COOPERATIVE_PROCESS_GROUP", "0"
    ) != "1"
    try:
        env0 = os.environ.copy()
        env1 = os.environ.copy()
        env0["CUDA_VISIBLE_DEVICES"] = str(lane["p0_gpu"])
        env1["CUDA_VISIBLE_DEVICES"] = str(lane["p1_gpu"])
        with (
            p0_metrics_path.open("wb") as p0_metrics,
            p1_metrics_path.open("wb") as p1_metrics,
            p0_log_path.open("wb") as p0_log,
            p1_log_path.open("wb") as p1_log,
        ):
            p0_process = subprocess.Popen(
                p0_command,
                stdout=p0_metrics,
                stderr=p0_log,
                env=env0,
                start_new_session=isolate_workers,
            )
            wait_for_p0_claim(
                p0_process,
                ledger_root / f"{invocation}.p0.claim",
                deadline,
                cancelled,
            )
            if cancelled.is_set():
                raise SchedulerCancelled("resource-lane execution cancelled")
            p1_process = subprocess.Popen(
                p1_command,
                stdout=p1_metrics,
                stderr=p1_log,
                env=env1,
                start_new_session=isolate_workers,
            )
            if time.monotonic() >= deadline:
                fail("run deadline expired before party 1 launch")
            rc0, rc1 = wait_pair(
                p0_process, p1_process, deadline, cancelled
            )
    except BaseException:
        terminate_and_reap(p1_process)
        terminate_and_reap(p0_process)
        raise
    finally:
        p0_auth.unlink(missing_ok=True)
        p1_auth.unlink(missing_ok=True)
        fsync_directory(p0_dir)
        fsync_directory(p1_dir)
    if rc0 != 0 or rc1 != 0:
        fail(
            f"layer {order} {layer['layer']} failed "
            f"(p0={rc0}, p1={rc1}); no record set published"
        )
    party_header = (
        CONV_PARTY_HEADER
        if layer["operator"] == "conv2d"
        else FC_PARTY_HEADER
    )
    checker_header = (
        CONV_CHECKER_HEADER
        if layer["operator"] == "conv2d"
        else FC_CHECKER_HEADER
    )
    metrics0 = load_single_csv(
        p0_metrics_path, f"{layer['layer']} party 0", party_header
    )
    metrics1 = load_single_csv(
        p1_metrics_path, f"{layer['layer']} party 1", party_header
    )
    suffix = ".conv" if layer["operator"] == "conv2d" else ".fc"
    record0 = pathlib.Path(str(p0_prefix) + "_p0" + suffix)
    record1 = pathlib.Path(str(p1_prefix) + "_p1" + suffix)
    if not record0.is_file() or not record1.is_file():
        fail(f"layer {order} omitted a party record")
    if stateful and (not p0_state.is_file() or not p1_state.is_file()):
        fail(f"layer {order} omitted a party mask-state record")
    expected_record_bytes = require_int(
        layer.get("linear_record_bytes_per_party"),
        f"{layer['layer']}.linear_record_bytes_per_party",
        1,
    )
    if (
        record0.stat().st_size != expected_record_bytes
        or record1.stat().st_size != expected_record_bytes
    ):
        fail(f"layer {order} record size differs from source manifest")
    checker_path = layer_dir / "checker.csv"
    checker_log = layer_dir / "checker.log"
    checker_command = [
        str(binary), "--check", "--csv-header",
        "--p0-record", str(record0), "--p1-record", str(record1),
    ]
    if stateful:
        checker_command.extend([
            "--p0-state", str(p0_state), "--p1-state", str(p1_state),
        ])
    checker_env = os.environ.copy()
    checker_env["CUDA_VISIBLE_DEVICES"] = str(lane["check_gpu"])
    if time.monotonic() >= deadline:
        fail(f"run deadline expired before checker for layer {order}")
    checker_process: subprocess.Popen[bytes] | None = None
    try:
        with (
            checker_path.open("wb") as stdout,
            checker_log.open("wb") as stderr,
        ):
            checker_process = subprocess.Popen(
                checker_command,
                stdout=stdout,
                stderr=stderr,
                env=checker_env,
                start_new_session=isolate_workers,
            )
            checker_rc = wait_process(
                checker_process, deadline, cancelled
            )
    except BaseException:
        terminate_and_reap(checker_process)
        raise
    if checker_rc == 124:
        fail(f"checker timed out for layer {order}; no record set published")
    if checker_rc != 0:
        fail(
            f"unchanged per-record consumer rejected layer {order}; "
            "no record set published"
        )
    checker_metrics = load_single_csv(
        checker_path, f"{layer['layer']} checker", checker_header
    )
    ledger_digest = validate_metrics(
        layer, profile, invocation, metrics0, metrics1, checker_metrics
    )
    artifact_paths = {
        "p0_record": record0,
        "p1_record": record1,
        "p0_metrics_csv": p0_metrics_path,
        "p1_metrics_csv": p1_metrics_path,
        "p0_log": p0_log_path,
        "p1_log": p1_log_path,
        "checker_csv": checker_path,
        "checker_log": checker_log,
    }
    if stateful:
        artifact_paths.update({"p0_state": p0_state, "p1_state": p1_state})
    return {
        "linear_order": order,
        "layer": layer["layer"],
        "operator": layer["operator"],
        "compatibility_id": layer["compatibility_id"],
        "binary": layer["binary"],
        "binary_sha256": binary_bindings[layer["binary"]]["sha256"],
        "sid": sid,
        "invocation_id": invocation,
        "ledger_digest": ledger_digest,
        "port": port,
        "artifacts": {
            label: artifact_binding(
                path,
                output_root,
                stateful and label in MANIFEST_BOUND_PRIVATE_ARTIFACTS,
            )
            for label, path in artifact_paths.items()
        },
        "metrics": {
            "party0": metrics0,
            "party1": metrics1,
            "checker": checker_metrics,
        },
    }


def run_scheduler_control(
    control: str,
    lanes: list[dict[str, int]],
    assignments: list[list[dict[str, Any]]],
) -> None:
    barrier = threading.Barrier(len(lanes))
    first_by_lane = {lane["lane"]: True for lane in lanes}
    state_lock = threading.Lock()

    if control == "canonical-aggregation":
        def complete_out_of_order(
            layer: dict[str, Any],
            lane: dict[str, int],
            cancelled: threading.Event,
        ) -> dict[str, Any]:
            with state_lock:
                first = first_by_lane[lane["lane"]]
                first_by_lane[lane["lane"]] = False
            if first:
                barrier.wait(timeout=5)
            if lane["lane"] == 0 and first:
                time.sleep(0.05)
            elif cancelled.wait(0.001):
                raise SchedulerCancelled("resource-lane execution cancelled")
            return {"linear_order": layer["linear_order"]}

        results, completion_order = execute_schedule(
            lanes, assignments, complete_out_of_order
        )
        canonical_order = sorted(layer["linear_order"] for layer in sum(assignments, []))
        if [result["linear_order"] for result in results] != canonical_order:
            fail("canonical-aggregation control produced noncanonical results")
        if completion_order == canonical_order:
            fail("canonical-aggregation control did not shuffle worker completion")
        print(
            "linear-record-set-runner: SCHEDULER CONTROL PASS "
            "(shuffled completion aggregated canonically)"
        )
        return

    class InjectedWorkerFailure(RuntimeError):
        pass

    cancelled_peers: set[int] = set()

    def inject_failure(
        layer: dict[str, Any],
        lane: dict[str, int],
        cancelled: threading.Event,
    ) -> dict[str, Any]:
        with state_lock:
            first = first_by_lane[lane["lane"]]
            first_by_lane[lane["lane"]] = False
        if first:
            barrier.wait(timeout=5)
        if lane["lane"] == 0:
            raise InjectedWorkerFailure("injected worker failure")
        while not cancelled.wait(0.001):
            pass
        with state_lock:
            cancelled_peers.add(lane["lane"])
        raise SchedulerCancelled("resource-lane execution cancelled")

    try:
        execute_schedule(lanes, assignments, inject_failure)
    except InjectedWorkerFailure:
        expected_peers = {lane["lane"] for lane in lanes[1:]}
        if cancelled_peers != expected_peers:
            fail("worker-failure control did not cancel every peer lane")
        # execute_schedule shares the process-wide signal cancellation event.
        # The injected failure sets it deliberately; an actual signal raises
        # SystemExit in the main thread and cannot reach this reset.
        TERMINATION_REQUESTED.clear()
        print(
            "linear-record-set-runner: SCHEDULER CONTROL PASS "
            "(injected failure cancelled all peer lanes; no manifest)"
        )
        return
    fail("worker-failure control did not propagate the injected failure")


def parse_args() -> argparse.Namespace:
    script = pathlib.Path(__file__).resolve()
    root = script.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-record-set", type=pathlib.Path)
    parser.add_argument(
        "--verify-record-set-root", type=pathlib.Path,
        help=(
            "artifact root for an immutable manifest snapshot supplied to "
            "--verify-record-set"
        ),
    )
    parser.add_argument("--repo-root", type=pathlib.Path, default=root)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--bin-dir", type=pathlib.Path)
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--binary-approval", type=pathlib.Path)
    parser.add_argument("--ledger-root", type=pathlib.Path)
    parser.add_argument("--p0-gpu", type=int)
    parser.add_argument("--p1-gpu", type=int)
    parser.add_argument("--check-gpu", type=int)
    parser.add_argument("--base-port", type=int)
    parser.add_argument(
        "--lane",
        action="append",
        default=[],
        metavar="P0:P1:CHECK:FIRST-LAST",
        help=(
            "exclusive GPU-pair lane with an inclusive reserved port range; "
            "repeat for deterministic concurrent LPT scheduling"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--scheduler-control",
        choices=("canonical-aggregation", "worker-failure"),
    )
    parser.add_argument("--emit-mask-states", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_record_set is not None:
        if any(value is not None for value in (
                args.manifest, args.bin_dir, args.output_root, args.ledger_root,
                args.binary_approval, args.p0_gpu, args.p1_gpu, args.check_gpu,
                args.base_port, args.scheduler_control,
        )) or args.plan_only or args.emit_mask_states or args.lane:
            fail("--verify-record-set cannot be combined with run options")
        artifact_root = (
            args.verify_record_set_root.absolute()
            if args.verify_record_set_root is not None else None
        )
        digest = verify_record_set(
            args.verify_record_set.absolute(), artifact_root
        )
        print(f"linear-record-set-runner: VERIFY PASS {digest} (21 isolated records)")
        return
    if args.verify_record_set_root is not None:
        fail("--verify-record-set-root requires --verify-record-set")
    signal.signal(signal.SIGINT, request_termination)
    signal.signal(signal.SIGTERM, request_termination)
    if args.manifest is None or args.bin_dir is None or args.output_root is None:
        fail("--manifest, --bin-dir, and --output-root are required for plan/run mode")
    stateful = args.emit_mask_states
    record_set_schema = (
        STATEFUL_RECORD_SET_SCHEMA if stateful else RECORD_SET_SCHEMA
    )
    composition_status = (
        STATEFUL_COMPOSITION_STATUS if stateful else COMPOSITION_STATUS
    )
    artifact_scope = STATEFUL_ARTIFACT_SCOPE if stateful else ARTIFACT_SCOPE
    execution_exclusions = (
        STATEFUL_EXECUTION_EXCLUSIONS if stateful else EXECUTION_EXCLUSIONS
    )
    repo_root = args.repo_root.resolve(strict=True)
    manifest_path = args.manifest.absolute()
    bin_dir = args.bin_dir.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    reject_symlink_components(output_root, "output root")
    if output_root.exists():
        fail(f"output root already exists: {output_root}")
    explicit_lanes = bool(args.lane)
    legacy_resources = (
        args.p0_gpu, args.p1_gpu, args.check_gpu, args.base_port,
    )
    if explicit_lanes:
        if any(value is not None for value in legacy_resources):
            fail("--lane cannot be combined with legacy GPU/base-port options")
        lanes = [
            parse_lane_descriptor(descriptor, index)
            for index, descriptor in enumerate(args.lane)
        ]
    else:
        args.p0_gpu = 0 if args.p0_gpu is None else args.p0_gpu
        args.p1_gpu = 1 if args.p1_gpu is None else args.p1_gpu
        args.check_gpu = 2 if args.check_gpu is None else args.check_gpu
        args.base_port = 24800 if args.base_port is None else args.base_port
        last_port = args.base_port + 4 * 21 + 1
        if args.base_port <= 0 or last_port > 65535:
            fail("base port leaves insufficient room")
        reject_linux_ephemeral_overlap(
            args.base_port, last_port, "linear port range"
        )
        if min(args.p0_gpu, args.p1_gpu, args.check_gpu) < 0 or len({
            args.p0_gpu, args.p1_gpu, args.check_gpu
        }) != 3:
            fail("party and checker GPUs must be pairwise distinct nonnegative indices")
        lanes = [{
            "lane": 0,
            "p0_gpu": args.p0_gpu,
            "p1_gpu": args.p1_gpu,
            "check_gpu": args.check_gpu,
            "port_first": args.base_port,
            "port_last": last_port,
        }]
    validate_resource_lanes(lanes, 21)
    if args.scheduler_control is not None:
        if not args.plan_only or not explicit_lanes:
            fail("--scheduler-control requires --plan-only and explicit --lane options")
        if len(lanes) < 2:
            fail("--scheduler-control requires at least two resource lanes")
    if args.timeout_seconds <= 0:
        fail("timeout must be positive")
    approval_bytes: bytes | None = None
    provenance_bytes: bytes | None = None
    approved_binaries: dict[str, str] | None = None
    if args.binary_approval is not None:
        approval_path = args.binary_approval.absolute()
        reject_symlink_components(approval_path, "binary approval")
        approval_bytes = read_bytes_once(approval_path, "binary approval")
        approval_document = parse_json_bytes(
            approval_bytes, str(approval_path)
        )
        raw_provenance_binding = approval_document.get("build_provenance")
        if not isinstance(raw_provenance_binding, dict):
            fail("binary approval lacks build provenance")
        relative_provenance = pathlib.PurePosixPath(
            require_string(
                raw_provenance_binding.get("path"),
                "binary approval.build_provenance.path",
            )
        )
        if (relative_provenance.is_absolute() or
                "." in relative_provenance.parts or
                ".." in relative_provenance.parts):
            fail("binary approval build provenance path is noncanonical")
        ringlpn_root = (repo_root / "GPU-MPC/ringlpn").resolve(strict=True)
        provenance_source = ringlpn_root.joinpath(
            *relative_provenance.parts
        )
        reject_symlink_components(
            provenance_source, "linear adapter build provenance"
        )
        provenance_bytes = read_bytes_once(
            provenance_source, "linear adapter build provenance"
        )
        approved_binaries = validate_binary_approval(
            approval_bytes, provenance_bytes, str(approval_path)
        )
    elif not args.plan_only:
        fail("execution requires --binary-approval")
    ledger_root: pathlib.Path | None = None
    if not args.plan_only:
        if args.ledger_root is None:
            fail("execution requires an explicit persistent --ledger-root outside output root")
        ledger_root = prepare_ledger_root(args.ledger_root.absolute(), output_root)

    source_bytes = read_bytes_once(manifest_path, "source execution manifest")
    source_sha256 = digest_bytes(source_bytes)
    document = parse_json_bytes(source_bytes, str(manifest_path))
    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)
    private_root = output_root / "private_inputs"
    private_root.mkdir(mode=0o700)
    source_snapshot = private_root / "source_execution_manifest.json"
    write_private_bytes(source_snapshot, source_bytes, 0o400)
    fsync_directory(private_root)
    binary_approval_binding: dict[str, str] | None = None
    build_provenance_binding: dict[str, Any] | None = None
    if approval_bytes is not None:
        approval_snapshot = private_root / "binary_approval.json"
        write_private_bytes(approval_snapshot, approval_bytes, 0o400)
        binary_approval_binding = artifact_binding(
            approval_snapshot, output_root
        )
        assert provenance_bytes is not None
        provenance_snapshot = (
            private_root / "linear_adapter_build_provenance.json"
        )
        write_private_bytes(provenance_snapshot, provenance_bytes, 0o400)
        build_provenance_binding = artifact_binding(
            provenance_snapshot, output_root, include_bytes=True
        )
    probe_timeout = min(args.timeout_seconds, PROBE_TIMEOUT_SECONDS)
    layers = verify_source_manifest(
        repo_root, source_snapshot, document, probe_timeout
    )
    profile = document["profile"]

    snapshot_binaries: dict[str, pathlib.Path] = {}
    binary_bindings: dict[str, dict[str, str]] = {}
    for name in sorted(BINARY_NAMES):
        snapshot = private_root / name
        digest = snapshot_executable(bin_dir / name, snapshot)
        if (approved_binaries is not None and
                digest != approved_binaries[name]):
            fail(f"binary snapshot differs from approval: {name}")
        snapshot_binaries[name] = snapshot
        binary_bindings[name] = {
            "path": snapshot.relative_to(output_root).as_posix(),
            "sha256": digest,
            "provenance": (
                VALIDATED_PROVENANCE if approved_binaries is not None
                else OBSERVED_PROVENANCE
            ),
        }
    fsync_directory(private_root)
    assignments, resource_schedule = build_lpt_schedule(layers, lanes)

    plan_rows: list[dict[str, Any]] = []
    for layer in layers:
        plan_rows.append({
            "layer": layer["layer"], "linear_order": layer["linear_order"],
            "operator": layer["operator"], "compatibility_id": layer["compatibility_id"],
            "binary": layer["binary"], "binary_sha256": binary_bindings[layer["binary"]]["sha256"],
            "plan": run_plan(
                snapshot_binaries[layer["binary"]], layer, profile,
                probe_timeout,
            ),
        })
    source_binding = {
        "path": source_snapshot.relative_to(output_root).as_posix(),
        "sha256": source_sha256,
    }
    planned: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "composition_status": composition_status,
        "artifact_scope": artifact_scope,
        "security_scope": SECURITY_SCOPE,
        "execution_exclusions": execution_exclusions,
        "profile": profile,
        "plan_csv_header": list(PLAN_HEADER),
        "source_plan_digest": document["plan_digest"],
        "source_execution_manifest": source_binding,
        "executable_provenance": binary_bindings,
        "binary_approval": binary_approval_binding,
        "linear_adapter_build_provenance": build_provenance_binding,
        "layers": plan_rows,
    }
    if explicit_lanes:
        planned["resource_schedule"] = {
            "algorithm": "deterministic_lpt_by_ring_batches",
            "stable_tie_break": "canonical_linear_order_then_lane_order",
            "lanes": resource_schedule,
        }
    planned["digest"] = digest_bytes(canonical(planned))
    planned_path = output_root / "PLANNED.json"
    write_atomic(planned_path, planned, 0o400)
    if args.scheduler_control is not None:
        run_scheduler_control(args.scheduler_control, lanes, assignments)
        if (output_root / "LINEAR_RECORD_SET.manifest").exists():
            fail("scheduler control unexpectedly published a record-set manifest")
    if TERMINATION_REQUESTED.is_set():
        fail("execution cancelled; no record-set manifest published")
    if args.plan_only:
        mode = "stateful graph-input" if stateful else "isolated adapter"
        print(f"linear-record-set-runner: PLAN PASS {planned['digest']} (21 {mode} rows)")
        return

    assert ledger_root is not None
    assert binary_approval_binding is not None
    assert build_provenance_binding is not None
    record_set_invocation = secrets.token_hex(16)
    def scheduled_job(
        layer: dict[str, Any],
        lane: dict[str, int],
        cancelled: threading.Event,
    ) -> dict[str, Any]:
        return run_layer_job(
            layer,
            lane,
            cancelled,
            output_root=output_root,
            ledger_root=ledger_root,
            profile=profile,
            snapshot_binaries=snapshot_binaries,
            binary_bindings=binary_bindings,
            stateful=stateful,
            timeout_seconds=args.timeout_seconds,
        )

    results, _completion_order = execute_schedule(
        lanes, assignments, scheduled_job
    )

    record_set: dict[str, Any] = {
        "schema": record_set_schema,
        "composition_status": composition_status,
        "artifact_scope": artifact_scope,
        "security_scope": SECURITY_SCOPE,
        "execution_exclusions": execution_exclusions,
        "source_model_label": document["model"],
        "record_set_invocation_id": record_set_invocation,
        "source_plan_digest": document["plan_digest"],
        "profile": profile,
        "source_execution_manifest": source_binding,
        "planned": {
            "path": planned_path.relative_to(output_root).as_posix(),
            "sha256": sha256(planned_path), "digest": planned["digest"],
        },
        "executable_provenance": binary_bindings,
        "binary_approval": binary_approval_binding,
        "linear_adapter_build_provenance": build_provenance_binding,
        "ledger_root": str(ledger_root),
        "p0_gpu": lanes[0]["p0_gpu"],
        "p1_gpu": lanes[0]["p1_gpu"],
        "check_gpu": lanes[0]["check_gpu"],
        "layers": results,
    }
    if explicit_lanes:
        record_set["resource_schedule"] = planned["resource_schedule"]
    record_set["record_set_digest"] = digest_bytes(canonical(record_set))
    publication = output_root / "LINEAR_RECORD_SET.manifest"
    if TERMINATION_REQUESTED.is_set():
        fail("execution cancelled; no record-set manifest published")
    write_atomic(publication, record_set, 0o400)
    try:
        verified_digest = verify_record_set(publication)
    except BaseException:
        os.chmod(publication, 0o600)
        publication.unlink(missing_ok=True)
        fsync_directory(output_root)
        raise
    print(f"linear-record-set-runner: VERIFY PASS {verified_digest} (21 isolated records)")
    print(f"linear-record-set-runner: RECORD SET PUBLISHED {verified_digest} (21 isolated records)")


if __name__ == "__main__":
    main()
