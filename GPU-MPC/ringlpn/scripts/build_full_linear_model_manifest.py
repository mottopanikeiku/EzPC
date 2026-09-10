#!/usr/bin/env python3
"""Build or verify a fail-closed forward-linear Ring-LPN execution manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import subprocess
import sys
from typing import Any, NoReturn

SCHEMA = "ringlpn-full-linear-execution-v1"
SUPPORTED_MODEL = "ResNet18"
WORD_BYTES = 8
CONV_HEADER_BYTES = 224
FC_HEADER_BYTES = 176
RECORD_DIGEST_BYTES = 32
CONV_PATTERN = re.compile(
    r"new\s+Conv2D<T>\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(true|false)\s*\)"
)
FC_PATTERN = re.compile(
    r"new\s+FC<T>\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(true|false)\s*\)"
)


def fail(message: str) -> NoReturn:
    print(f"full-linear-manifest: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot load {path}: {error}")
    if not isinstance(value, dict):
        fail(f"top-level JSON must be an object: {path}")
    return value


def relative_repo_path(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(repo_root.resolve(strict=True)).as_posix()
    except (OSError, ValueError):
        fail(f"path is not a regular repository input: {path}")


@dataclass(frozen=True)
class ValidatedSourceRegistry:
    entries: tuple[dict[str, str], ...]
    files: dict[str, bytes]


def canonical_source_parts(relative: str) -> tuple[str, ...]:
    if not relative or "\\" in relative or "\0" in relative:
        fail(f"source registry path must be canonical relative POSIX: {relative}")
    parts = tuple(relative.split("/"))
    path = pathlib.PurePosixPath(relative)
    windows_path = pathlib.PureWindowsPath(relative)
    if (path.is_absolute() or windows_path.drive
            or any(part in ("", ".", "..") for part in parts)
            or path.as_posix() != relative):
        fail(f"source registry path must be canonical relative POSIX: {relative}")
    return parts


def held_regular_source(
    repo_root: pathlib.Path, relative: str, parts: tuple[str, ...]
) -> bytes:
    root = repo_root.resolve(strict=True)
    candidate = root.joinpath(*parts)
    current = root
    try:
        for part in parts:
            current = current / part
            if stat.S_ISLNK(current.lstat().st_mode):
                fail(
                    "source registry path contains a symlink component: "
                    f"{relative}"
                )
        candidate.resolve(strict=True).relative_to(root)
    except ValueError:
        fail(f"source registry path resolves outside repository: {relative}")
    except OSError as error:
        fail(f"cannot inspect source registry input {relative}: {error}")
    require_git_tracked(root, relative)

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    source_fd = -1
    try:
        descriptors.append(os.open(root, directory_flags | nofollow))
        for part in parts[:-1]:
            descriptors.append(os.open(
                part, directory_flags | nofollow, dir_fd=descriptors[-1]
            ))
        source_fd = os.open(
            parts[-1], os.O_RDONLY | os.O_CLOEXEC | nofollow,
            dir_fd=descriptors[-1],
        )
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            fail(f"source registry input is not a regular file: {relative}")
        with os.fdopen(source_fd, "rb", closefd=True) as source_stream:
            source_fd = -1
            data = source_stream.read()
            after = os.fstat(source_stream.fileno())
        before_identity = (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if len(data) != before.st_size or before_identity != after_identity:
            fail(f"source registry input changed while being read: {relative}")
        return data
    except OSError as error:
        fail(f"cannot securely read source registry input {relative}: {error}")
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def require_git_tracked(repo_root: pathlib.Path, relative: str) -> None:
    try:
        completed = subprocess.run(
            [
                "git", "--literal-pathspecs", "-C", str(repo_root),
                "ls-files", "-z", "--stage", "--error-unmatch", "--", relative,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as error:
        fail(f"cannot verify Git tracking for source registry: {error}")
    records = [record for record in completed.stdout.split(b"\0") if record]
    if (completed.returncode != 0 or len(records) != 1
            or records[0].split(maxsplit=1)[0] not in (b"100644", b"100755")):
        fail(f"source registry path is not a Git-tracked regular file: {relative}")


def validate_source_registry(
    document: dict[str, Any], repo_root: pathlib.Path
) -> ValidatedSourceRegistry:
    registry = document.get("source_registry")
    if not isinstance(registry, list) or not registry:
        fail("source registry is missing or empty")
    validated: list[dict[str, str]] = []
    held_files: dict[str, bytes] = {}
    seen: set[str] = set()
    for entry in registry:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            fail("source registry entry has an unexpected schema")
        relative = entry["path"]
        expected = entry["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str):
            fail("source registry path/digest must be strings")
        parts = canonical_source_parts(relative)
        if relative in seen or not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail(f"duplicate or malformed source registry entry: {relative}")
        data = held_regular_source(repo_root, relative, parts)
        actual = sha256_bytes(data)
        if actual != expected:
            fail(f"source registry digest mismatch: {relative}")
        seen.add(relative)
        held_files[relative] = data
        validated.append({"path": relative, "sha256": actual})
    return ValidatedSourceRegistry(tuple(validated), held_files)


def parse_shape(text: Any, dimensions: int, label: str) -> tuple[int, ...]:
    if not isinstance(text, str):
        fail(f"{label} is not a shape string")
    parts = text.split("x")
    if len(parts) != dimensions:
        fail(f"{label} must contain {dimensions} dimensions: {text}")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError:
        fail(f"{label} contains a non-integer dimension: {text}")
    if any(value <= 0 for value in shape):
        fail(f"{label} contains a nonpositive dimension: {text}")
    return shape


def source_lines(
    relative: str, registry: ValidatedSourceRegistry, label: str
) -> list[str]:
    data = registry.files.get(relative)
    if data is None:
        fail(f"{label} source is absent from registry: {relative}")
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeError as error:
        fail(f"{label} source is not UTF-8: {relative}: {error}")


def source_line(
    row: dict[str, Any], registry: ValidatedSourceRegistry
) -> tuple[str, str, int]:
    anchor = row.get("source_anchor")
    expected = row.get("source_text_sha256")
    if not isinstance(anchor, str) or not isinstance(expected, str):
        fail("layer source anchor/digest is missing")
    match = re.fullmatch(r"([^:]+):(\d+)", anchor)
    if not match:
        fail(f"layer source anchor must name exactly one line: {anchor}")
    relative, line_text = match.groups()
    line_number = int(line_text)
    lines = source_lines(relative, registry, "layer")
    if line_number <= 0 or line_number > len(lines):
        fail(f"source anchor is out of range: {anchor}")
    text = lines[line_number - 1].strip()
    if sha256_bytes(text.encode("utf-8")) != expected:
        fail(f"source line digest mismatch: {anchor}")
    return text, relative, line_number


def bound_source_line(
    relative: str, line_number: int, registry: ValidatedSourceRegistry
) -> dict[str, Any]:
    lines = source_lines(relative, registry, "state")
    if line_number <= 0 or line_number > len(lines):
        fail(f"state source line is out of range: {relative}:{line_number}")
    text = lines[line_number - 1].strip()
    return {
        "source_anchor": f"{relative}:{line_number}",
        "source_text_sha256": sha256_bytes(text.encode("utf-8")),
    }


def selected_scale(configured: Any, bw: int) -> int:
    if not isinstance(configured, str):
        fail("configured_bw_scale must be a string")
    choices: dict[int, int] = {}
    for choice in configured.split("|"):
        match = re.fullmatch(r"(\d+)/(\d+)", choice)
        if not match:
            fail(f"malformed configured_bw_scale: {configured}")
        choice_bw, scale = (int(value) for value in match.groups())
        if choice_bw in choices:
            fail(f"duplicate bit width in configured_bw_scale: {configured}")
        choices[choice_bw] = scale
    if bw not in choices or choices[bw] <= 0 or choices[bw] >= bw:
        fail(f"no valid scale for bw={bw}: {configured}")
    return choices[bw]


def conv_cross_terms(
    batch: int,
    height: int,
    width: int,
    ci: int,
    filter_size: int,
    co: int,
    padding: int,
    stride: int,
) -> tuple[int, int, int]:
    output_h = (height - filter_size + 2 * padding) // stride + 1
    output_w = (width - filter_size + 2 * padding) // stride + 1
    if output_h <= 0 or output_w <= 0:
        fail("convolution has a nonpositive output shape")
    per_image = 0
    for out_h in range(output_h):
        input_h0 = out_h * stride - padding
        valid_h = max(0, min(filter_size, height - input_h0) - max(0, -input_h0))
        for out_w in range(output_w):
            input_w0 = out_w * stride - padding
            valid_w = max(0, min(filter_size, width - input_w0) - max(0, -input_w0))
            per_image += valid_h * valid_w * ci * co
    return batch * per_image, output_h, output_w


def common_layer_contract(
    row: dict[str, Any], order: int, shift: int, application_slots: int,
    bootstrap_slots: int, cross_terms: int, size_input: int, size_weight: int,
    size_output: int, header_bytes: int,
) -> dict[str, Any]:
    ring_batches = (cross_terms + application_slots - 1) // application_slots
    transition = row.get("truncation_state")
    is_classifier = row.get("is_classifier")
    if is_classifier:
        if transition != "FinalOutputLocalARS(scale);G->ClearOutput":
            fail(f"unsupported terminal transition for {row.get('layer')}: {transition}")
        state_input_kind = "G"
        state_output_kind = "ClearOutput"
        truncation = {
            "kind": "LocalARSAfterFinalUnmask",
            "bin": row["bw"],
            "shift": shift,
            "bout": row["bw"] - shift,
            "ringlpn_transition": False,
        }
    else:
        match = re.fullmatch(
            r"StochasticTR\(scale\);(S|G|G\(branch\))->T", str(transition)
        )
        if not match:
            fail(f"unsupported state transition for {row.get('layer')}: {transition}")
        state_input_kind = match.group(1)
        state_output_kind = "T"
        truncation = {
            "kind": "StochasticTR",
            "bin": row["bw"],
            "shift": shift,
            "bout": row["bw"] - shift,
            "ringlpn_transition": False,
        }
    contract: dict[str, Any] = {
        "linear_order": order,
        "forward_order": row.get("forward_order"),
        "layer": row.get("layer"),
        "operator": row.get("operator"),
        "source_anchor": row.get("source_anchor"),
        "source_text_sha256": row.get("source_text_sha256"),
        "batch_source_anchor": row.get("batch_source_anchor"),
        "batch": row.get("batch"),
        "matmul_batch": row.get("matmul_batch"),
        "collapsed_rows": row.get("collapsed_rows"),
        "input_shape": row.get("input_shape"),
        "output_shape": row.get("output_shape"),
        "layout": row.get("layout"),
        "stock_key_abi": row.get("stock_key_abi"),
        "keygen_api": row.get("keygen_api"),
        "online_consumer": row.get("online_consumer"),
        "bias": row.get("bias"),
        "is_classifier": is_classifier,
        "state_transition": transition,
        "state_input_kind": state_input_kind,
        "state_output_kind": state_output_kind,
        "state_transition_execution": "not_executed_by_linear_record_artifact",
        "truncation": truncation,
        "cross_terms": cross_terms,
        "ring_batches": ring_batches,
        "ring_application_slots": application_slots,
        "ring_bootstrap_slots": bootstrap_slots,
        "size_input_words": size_input,
        "size_weight_words": size_weight,
        "size_output_words": size_output,
        "linear_key_payload_bytes_per_party":
            (size_input + size_weight + size_output) * WORD_BYTES,
        "linear_record_bytes_per_party":
            header_bytes + (size_input + size_weight + size_output) * WORD_BYTES
            + RECORD_DIGEST_BYTES,
    }
    if not isinstance(contract["forward_order"], int) or contract["forward_order"] < 0:
        fail(f"invalid forward order for {contract['layer']}")
    if not all(isinstance(contract[field], str) and contract[field]
               for field in ("layer", "operator", "layout", "stock_key_abi",
                             "keygen_api", "online_consumer")):
        fail(f"missing public contract field for layer {order}")
    if not isinstance(contract["bias"], bool) or not isinstance(contract["is_classifier"], bool):
        fail(f"invalid Boolean layer field for {contract['layer']}")
    if (not isinstance(contract["batch"], int) or contract["batch"] <= 0 or
            not isinstance(contract["matmul_batch"], int) or
            contract["matmul_batch"] <= 0 or
            not isinstance(contract["collapsed_rows"], int) or
            contract["collapsed_rows"] <= 0):
        fail(f"invalid batch contract for {contract['layer']}")
    return contract


def build_document(
    source: dict[str, Any], source_path: pathlib.Path, repo_root: pathlib.Path,
    model: str, ole_n_override: int | None = None,
) -> dict[str, Any]:
    if source.get("schema_version") != "ringlpn.orca-forward-linear-layers.v1":
        fail("unsupported source layer-manifest schema")
    if model != SUPPORTED_MODEL:
        fail(f"only {SUPPORTED_MODEL} is currently supported")
    registry = validate_source_registry(source, repo_root)
    profile = source.get("matrix_profile")
    if not isinstance(profile, dict):
        fail("matrix_profile is missing")
    required_profile = ("bw", "qbits", "noise", "ole_n", "ole_c", "ole_t")
    if any(key not in profile for key in required_profile):
        fail("matrix_profile is incomplete")
    bw = profile["bw"]
    qbits = profile["qbits"]
    source_ole_n = profile["ole_n"]
    ole_n = source_ole_n if ole_n_override is None else ole_n_override
    ole_c = profile["ole_c"]
    ole_t = profile["ole_t"]
    regular_trees = 2 * ole_c * ole_t * ole_t if (
        isinstance(ole_c, int) and isinstance(ole_t, int)
    ) else 0
    regular_domain = 2 * (ole_n // ole_t) if (
        isinstance(ole_n, int) and isinstance(ole_t, int) and ole_t > 0
    ) else 0
    if (not isinstance(bw, int) or not isinstance(qbits, int) or
            not isinstance(source_ole_n, int) or not isinstance(ole_n, int) or
            not isinstance(ole_c, int) or not isinstance(ole_t, int) or
            profile["noise"] != "regular" or qbits not in (64, 128) or
            bw <= 2 or bw > 32 or min(ole_n, ole_c, ole_t) <= 0 or
            ole_n < 8192 or ole_n > 262144 or ole_n & (ole_n - 1) or
            ole_n % ole_t != 0 or regular_domain < 4 or
            regular_domain & (regular_domain - 1) or
            regular_trees * regular_domain > (1 << 24)):
        fail("unsupported matrix profile or Ring-OLE degree override")
    bootstrap_slots = 3 * (ole_c * ole_t) ** 2
    application_slots = ole_n - bootstrap_slots
    if application_slots <= 0:
        fail("Ring-OLE profile has no application capacity")

    raw_layers = source.get("layers")
    if not isinstance(raw_layers, list):
        fail("source layers are missing")
    rows = [row for row in raw_layers if isinstance(row, dict) and row.get("model") == model]
    if not rows:
        fail(f"model is absent from source manifest: {model}")
    rows.sort(key=lambda row: row.get("linear_order", -1))
    if [row.get("linear_order") for row in rows] != list(range(1, len(rows) + 1)):
        fail("linear orders are duplicated, missing, or non-contiguous")
    if len({row.get("layer") for row in rows}) != len(rows):
        fail("layer names are duplicated")
    if any(row.get("route") != "orca" or row.get("bw") != bw or
           row.get("qbits") != qbits or row.get("noise") != profile["noise"] or
           row.get("ole_n") != source_ole_n or row.get("ole_c") != ole_c or
           row.get("ole_t") != ole_t for row in rows):
        fail("model layer does not match the source execution profile")

    layers: list[dict[str, Any]] = []
    branch_specs = {
        "conv16": ("projection_stage2", 12, 17, 1, "conv15", 503, 504),
        "conv27": ("projection_stage3", 23, 28, 1, "conv26", 514, 515),
        "conv38": ("projection_stage4", 34, 39, 1, "conv37", 525, 526),
    }
    for order, row in enumerate(rows, 1):
        line, relative, line_number = source_line(row, registry)
        batch_anchor = row.get("batch_source_anchor")
        batch_match = re.fullmatch(r"([^:]+):(\d+)-(\d+)", str(batch_anchor))
        if (not batch_match or batch_match.group(1) not in registry.files or
                int(batch_match.group(2)) > int(batch_match.group(3))):
            fail(f"invalid batch source anchor for {row.get('layer')}: {batch_anchor}")
        shift = selected_scale(row.get("configured_bw_scale"), bw)
        operator = row.get("operator")
        if operator == "conv2d":
            constructor = CONV_PATTERN.search(line)
            if not constructor:
                fail(f"cannot parse Conv2D constructor: {row.get('source_anchor')}")
            ci, co, filter_size, padding, stride = (
                int(value) for value in constructor.groups()[:5]
            )
            constructor_bias = constructor.group(6) == "true"
            batch, height, width, input_ci = parse_shape(
                row.get("input_shape"), 4, f"{row.get('layer')} input"
            )
            output_batch, output_h_expected, output_w_expected, output_co = parse_shape(
                row.get("output_shape"), 4, f"{row.get('layer')} output"
            )
            cross_terms, output_h, output_w = conv_cross_terms(
                batch, height, width, ci, filter_size, co, padding, stride
            )
            if (input_ci != ci or output_batch != batch or output_co != co or
                    output_h_expected != output_h or output_w_expected != output_w or
                    row.get("rows") != output_h * output_w or
                    row.get("inner") != filter_size * filter_size * ci or
                    row.get("cols") != co or row.get("bias") != constructor_bias or
                    row.get("batch") != batch or row.get("matmul_batch") != batch or
                    row.get("collapsed_rows") != batch * output_h * output_w):
                fail(f"convolution shape/constructor mismatch: {row.get('layer')}")
            size_input = batch * height * width * ci
            size_weight = co * filter_size * filter_size * ci
            size_output = batch * output_h * output_w * co
            contract = common_layer_contract(
                row, order, shift, application_slots, bootstrap_slots,
                cross_terms, size_input, size_weight, size_output,
                CONV_HEADER_BYTES,
            )
            contract["conv"] = {
                "n": batch, "h": height, "w": width, "ci": ci,
                "fh": filter_size, "fw": filter_size, "co": co,
                "padding": padding, "stride": stride,
                "oh": output_h, "ow": output_w,
            }
            contract["binary"] = "test_two_party_conv_preprocess"
            contract["cli"] = [
                "--layer-ordinal", str(order),
                "--n", str(batch), "--h", str(height), "--w", str(width),
                "--ci", str(ci), "--fh", str(filter_size), "--fw", str(filter_size),
                "--co", str(co), "--padding", str(padding), "--stride", str(stride),
            ]
        elif operator == "fc":
            constructor = FC_PATTERN.search(line)
            if not constructor:
                fail(f"cannot parse FC constructor: {row.get('source_anchor')}")
            inner, cols = (int(value) for value in constructor.groups()[:2])
            constructor_bias = constructor.group(3) == "true"
            rows_count, input_inner = parse_shape(
                row.get("input_shape"), 2, f"{row.get('layer')} input"
            )
            output_rows, output_cols = parse_shape(
                row.get("output_shape"), 2, f"{row.get('layer')} output"
            )
            if (input_inner != inner or output_rows != rows_count or
                    output_cols != cols or row.get("rows") != rows_count or
                    row.get("inner") != inner or row.get("cols") != cols or
                    row.get("bias") != constructor_bias or
                    row.get("batch") != rows_count or
                    row.get("matmul_batch") != rows_count or
                    row.get("collapsed_rows") != rows_count):
                fail(f"FC shape/constructor mismatch: {row.get('layer')}")
            cross_terms = rows_count * inner * cols
            size_input = rows_count * inner
            size_weight = inner * cols
            size_output = rows_count * cols
            contract = common_layer_contract(
                row, order, shift, application_slots, bootstrap_slots,
                cross_terms, size_input, size_weight, size_output, FC_HEADER_BYTES,
            )
            contract["matmul"] = {"rows": rows_count, "inner": inner, "cols": cols}
            contract["binary"] = "test_two_party_fc_preprocess"
            contract["cli"] = [
                "--layer-ordinal", str(order),
                "--rows", str(rows_count), "--inner", str(inner), "--cols", str(cols),
            ]
        else:
            fail(f"unsupported operator in selected model: {operator}")
        contract["source_file"] = relative
        contract["source_line"] = line_number
        if contract["layer"] in branch_specs:
            if contract["state_input_kind"] != "G(branch)":
                fail(f"projection layer lacks branch state: {contract['layer']}")
            (branch_id, source_order, merge_order, merge_operand,
             main_terminal, branch_line, merge_line) = branch_specs[contract["layer"]]
            contract["residual_branch"] = {
                "branch_id": branch_id,
                "branch_source_forward_order": source_order,
                "merge_forward_order": merge_order,
                "merge_operand_index": merge_operand,
                "parallel_main_terminal_layer": main_terminal,
                "branch_call": bound_source_line(
                    "GPU-MPC/experiments/orca/cnn.h", branch_line, registry
                ),
                "merge_call": bound_source_line(
                    "GPU-MPC/experiments/orca/cnn.h", merge_line, registry
                ),
                "execution": "not_executed_by_linear_record_artifact",
            }
        elif contract["state_input_kind"] == "G(branch)":
            fail(f"unbound residual branch: {contract['layer']}")
        contract["compatibility_id"] = sha256_bytes(canonical_bytes(contract))
        layers.append(contract)

    if len(layers) != 21 or sum(layer["operator"] == "conv2d" for layer in layers) != 20:
        fail(f"{SUPPORTED_MODEL} contract must contain exactly 20 Conv2D and one FC layer")
    total_cross_terms = sum(layer["cross_terms"] for layer in layers)
    total_ring_batches = sum(layer["ring_batches"] for layer in layers)
    limbs = qbits // 64
    total_ring_ole_instances = 2 * limbs * total_ring_batches
    log_domain = regular_domain.bit_length() - 1
    total_dpf_trees = total_ring_ole_instances * regular_trees
    total_private_dpf_prg_node_expansions = (
        total_dpf_trees * (regular_domain - 1)
    )
    total_dpf_string_ots = 2 * total_dpf_trees * log_domain
    total_dpf_bit_triples = total_dpf_trees * (log_domain - 1)
    total_dpf_scalar_oles = total_ring_ole_instances * bootstrap_slots
    total_public_a_seed_words = len(layers) * 4
    total_ring_application_slots = 2 * limbs * total_cross_terms
    total_ring_application_capacity = (
        total_ring_ole_instances * application_slots
    )
    stock_state_nodes = [
        {
            "node": "globalaveragepool46",
            "forward_order": 46,
            "operator": "globalaveragepool2d",
            "state_input_kind": "G",
            "state_output_kind": "T",
            "operation": "local_gpuAddPool_then_StochasticTR",
            "truncation": {"kind": "StochasticTR", "bin": bw, "shift": shift,
                           "bout": bw - shift},
            "ringlpn_scope": "excluded_non_ringlpn_linear_sum",
            "execution": "not_executed_by_linear_record_artifact",
            "constructor": bound_source_line(
                "GPU-MPC/experiments/orca/cnn.h", 480, registry
            ),
            "forward_call": bound_source_line(
                "GPU-MPC/experiments/orca/cnn.h", 533, registry
            ),
            "generic_truncation_call": bound_source_line(
                "GPU-MPC/ext/sytorch/include/sytorch/layers/layers.h",
                169, registry,
            ),
        },
        {
            "node": "flatten47",
            "forward_order": 47,
            "operator": "flatten_view",
            "state_input_kind": "T",
            "state_output_kind": "T",
            "ringlpn_scope": "excluded_no_key_material",
            "execution": "not_executed_by_linear_record_artifact",
            "constructor": bound_source_line(
                "GPU-MPC/experiments/orca/cnn.h", 481, registry
            ),
            "forward_call": bound_source_line(
                "GPU-MPC/experiments/orca/cnn.h", 534, registry
            ),
        },
        {
            "node": "gemm48_pre_sign_extend",
            "forward_order": 48,
            "operator": "SignExtend",
            "state_input_kind": "T",
            "state_output_kind": "G",
            "ringlpn_scope": "excluded_dcf_transition",
            "execution": "not_executed_by_linear_record_artifact",
            "generic_sign_extension_call": bound_source_line(
                "GPU-MPC/ext/sytorch/include/sytorch/layers/layers.h",
                162, registry,
            ),
            "optimizer_terminal_rule": bound_source_line(
                "GPU-MPC/nn/orca_opt.h", 67, registry
            ),
        },
        {
            "node": "gemm48_terminal_output",
            "forward_order": 49,
            "operator": "FinalUnmaskLocalARS",
            "state_input_kind": "MaskedOutput",
            "state_output_kind": "ClearOutput",
            "truncation": {"kind": "LocalARSAfterFinalUnmask", "bin": bw,
                           "shift": shift, "bout": bw - shift},
            "ringlpn_scope": "terminal_mask_share_required_but_not_recorded",
            "execution": "not_executed_by_linear_record_artifact",
            "stock_output_call": bound_source_line(
                "GPU-MPC/backend/orca_base.h", 231, registry
            ),
        },
    ]
    residual_merge_specs = [
        ("residual_stage1_block1", 6, "conv5", "relu2", "identity", 493),
        ("residual_stage1_block2", 11, "conv10", "relu7", "identity", 498),
        ("residual_stage2_block1", 17, "conv15", "conv16", "projection", 504),
        ("residual_stage2_block2", 22, "conv21", "relu18", "identity", 509),
        ("residual_stage3_block1", 28, "conv26", "conv27", "projection", 515),
        ("residual_stage3_block2", 33, "conv32", "relu29", "identity", 520),
        ("residual_stage4_block1", 39, "conv37", "conv38", "projection", 526),
        ("residual_stage4_block2", 44, "conv43", "relu40", "identity", 531),
    ]
    residual_merges = [
        {
            "branch_id": branch_id,
            "merge_forward_order": merge_order,
            "main_operand": main_operand,
            "shortcut_operand": shortcut_operand,
            "shortcut_kind": shortcut_kind,
            "main_operand_index": 0,
            "shortcut_operand_index": 1,
            "merge_call": bound_source_line(
                "GPU-MPC/experiments/orca/cnn.h",
                source_line_number, registry,
            ),
            "execution": "not_executed_by_linear_record_artifact",
        }
        for (branch_id, merge_order, main_operand, shortcut_operand,
             shortcut_kind, source_line_number) in residual_merge_specs
    ]
    forward_key_specs = [
        ("conv0", 487, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("maxpool1", 488, (("GPUMaxpoolKey", False),)),
        ("relu2", 489, (("GPUReluExtendKey", False),)),
        ("conv3", 490, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu4", 491, (("GPUReluExtendKey", False),)),
        ("conv5", 492, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu7", 494, (("GPUReluExtendKey", False),)),
        ("conv8", 495, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu9", 496, (("GPUReluExtendKey", False),)),
        ("conv10", 497, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu12", 499, (("GPUReluExtendKey", False),)),
        ("conv13", 500, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu14", 501, (("GPUReluExtendKey", False),)),
        ("conv15", 502, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("conv16", 503, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu18", 505, (("GPUReluExtendKey", False),)),
        ("conv19", 506, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu20", 507, (("GPUReluExtendKey", False),)),
        ("conv21", 508, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu23", 510, (("GPUReluExtendKey", False),)),
        ("conv24", 511, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu25", 512, (("GPUReluExtendKey", False),)),
        ("conv26", 513, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("conv27", 514, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu29", 516, (("GPUReluExtendKey", False),)),
        ("conv30", 517, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu31", 518, (("GPUReluExtendKey", False),)),
        ("conv32", 519, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu34", 521, (("GPUReluExtendKey", False),)),
        ("conv35", 522, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu36", 523, (("GPUReluExtendKey", False),)),
        ("conv37", 524, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("conv38", 525, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu40", 527, (("GPUReluExtendKey", False),)),
        ("conv41", 528, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu42", 529, (("GPUReluExtendKey", False),)),
        ("conv43", 530, (("GPUConv2DKey[I,F,O]", True), ("GPUStTRKey", False))),
        ("relu45", 532, (("GPUReluExtendKey", False),)),
        ("globalaveragepool46", 533, (("GPUStTRKey", False),)),
        ("gemm48", 535, (("GPUSignExtendKey", False),
                         ("GPUMatmulKey[A,B,C]", True))),
    ]
    stock_key_stream: list[dict[str, Any]] = []
    for node, source_line_number, keys in forward_key_specs:
        for key_kind, ringlpn in keys:
            stock_key_stream.append({
                "stream_position": len(stock_key_stream) + 1,
                "node": node,
                "key_kind": key_kind,
                "ringlpn": ringlpn,
                "forward_call": bound_source_line(
                    "GPU-MPC/experiments/orca/cnn.h",
                    source_line_number, registry,
                ),
                "execution": "not_executed_by_linear_record_artifact",
            })
    stock_key_stream.append({
        "stream_position": len(stock_key_stream) + 1,
        "node": "output",
        "key_kind": "raw_mask[1000_words]",
        "ringlpn": False,
        "forward_call": bound_source_line(
            "GPU-MPC/backend/orca_base.h", 231, registry
        ),
        "execution": "not_executed_by_linear_record_artifact",
    })
    if len(stock_key_stream) != 62:
        fail("stock ResNet18 key stream must contain exactly 62 keyed items")
    source_relative = relative_repo_path(source_path, repo_root)
    generator_path = pathlib.Path(__file__).resolve(strict=True)
    generator_relative = relative_repo_path(generator_path, repo_root)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "model": model,
        "claim_scope": "internal/advisor source-bound forward-linear record plan; graph state nodes are unexecuted; qbits is not a security level",
        "generator": {
            "path": generator_relative,
            "sha256": sha256_bytes(generator_path.read_bytes()),
        },
        "source_layer_manifest": {
            "path": source_relative,
            "sha256": sha256_bytes(source_path.read_bytes()),
            "schema": source["schema_version"],
        },
        "source_registry": registry.entries,
        "profile": {
            "qbits": qbits, "bw": bw, "noise": profile["noise"],
            "ole_n": ole_n, "ole_c": ole_c, "ole_t": ole_t,
            "ring_application_slots": application_slots,
            "ring_bootstrap_slots": bootstrap_slots,
            "public_ring_vector_scope":
                "one_jointly_seeded_shake256_random_oracle_per_linear_layer_independent_domain_separated_vector_per_ring_ole",
            "ring_lpn_assumption_scope":
                "independent_public_vector_ring_lpn_in_shake256_random_oracle_model_no_concrete_security_claim",
        },
        "summary": {
            "linear_layers": len(layers),
            "conv2d_layers": sum(layer["operator"] == "conv2d" for layer in layers),
            "fc_layers": sum(layer["operator"] == "fc" for layer in layers),
            "branch_input_layers": sum(layer["state_input_kind"] == "G(branch)" for layer in layers),
            "stock_state_nodes": len(stock_state_nodes),
            "residual_merges": len(residual_merges),
            "projection_shortcuts": sum(
                merge["shortcut_kind"] == "projection"
                for merge in residual_merges
            ),
            "identity_shortcuts": sum(
                merge["shortcut_kind"] == "identity"
                for merge in residual_merges
            ),
            "stock_key_items": len(stock_key_stream),
            "stock_stochastic_truncations": sum(
                item["key_kind"] == "GPUStTRKey"
                for item in stock_key_stream
            ),
            "total_cross_terms": total_cross_terms,
            "total_ring_batches": total_ring_batches,
            "total_ring_ole_instances": total_ring_ole_instances,
            "total_dpf_trees": total_dpf_trees,
            "total_private_dpf_prg_node_expansions":
                total_private_dpf_prg_node_expansions,
            "total_dpf_string_ots": total_dpf_string_ots,
            "total_dpf_bit_triples": total_dpf_bit_triples,
            "total_dpf_scalar_oles": total_dpf_scalar_oles,
            "total_public_a_seed_words": total_public_a_seed_words,
            "total_ring_application_slots": total_ring_application_slots,
            "total_ring_application_capacity":
                total_ring_application_capacity,
            "total_ring_application_slots_discarded":
                total_ring_application_capacity - total_ring_application_slots,
            "total_linear_key_payload_bytes_per_party":
                sum(layer["linear_key_payload_bytes_per_party"] for layer in layers),
        },
        "stock_state_nodes": stock_state_nodes,
        "residual_merges": residual_merges,
        "stock_key_stream": stock_key_stream,
        "layers": layers,
    }
    if ole_n != source_ole_n:
        document["profile"]["source_ole_n"] = source_ole_n
        document["profile"]["ole_n_selection"] = (
            "explicit_execution_override_validated_by_adapter_plan"
        )
    document["plan_digest"] = sha256_bytes(canonical_bytes(document))
    return document


def rendered(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_args() -> argparse.Namespace:
    script = pathlib.Path(__file__).resolve()
    default_root = script.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=default_root)
    parser.add_argument("--layer-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--model", default=SUPPORTED_MODEL)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--ole-n", type=int,
        help="validated power-of-two Ring-OLE degree override for execution",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify --out byte-for-byte instead of writing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve(strict=True)
    source_path = args.layer_manifest.resolve(strict=True)
    source = load_json(source_path)
    document = build_document(
        source, source_path, repo_root, args.model, args.ole_n
    )
    output = rendered(document)
    if args.check:
        try:
            current = args.out.read_bytes()
        except OSError as error:
            fail(f"cannot read output manifest {args.out}: {error}")
        if current != output:
            fail(f"output manifest is stale or modified: {args.out}")
        print(f"full-linear-manifest: PASS {document['plan_digest']}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    try:
        temporary.write_bytes(output)
        temporary.replace(args.out)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        fail(f"cannot write output manifest {args.out}: {error}")
    print(f"full-linear-manifest: wrote {args.out} {document['plan_digest']}")


if __name__ == "__main__":
    main()
