#!/usr/bin/env python3
"""Cross-check every execution-manifest layer against the compiled adapters."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from typing import Any, NoReturn

SCHEMA = "ringlpn-full-linear-execution-v1"
FIELDS = (
    "kind", "qbits", "bw", "size_a", "size_b", "size_c", "cross_terms",
    "ring_batches", "application_slots", "bootstrap_slots", "oh", "ow",
)


def fail(message: str) -> NoReturn:
    print(f"full-linear-shapes: {message}", file=sys.stderr)
    raise SystemExit(2)


def load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot load {path}: {error}")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        fail(f"unsupported execution manifest: {path}")
    return value


def parse_row(text: str, layer: str) -> dict[str, int | str]:
    rows = list(csv.reader(text.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        fail(f"adapter returned malformed plan for {layer}: {text!r}")
    result: dict[str, int | str] = {"kind": rows[0][0]}
    try:
        result.update({name: int(value) for name, value in zip(FIELDS[1:], rows[0][1:])})
    except ValueError:
        fail(f"adapter returned non-integer plan fields for {layer}: {text!r}")
    return result


def expected(layer: dict[str, Any], profile: dict[str, Any]) -> dict[str, int | str]:
    operator = layer.get("operator")
    if operator == "conv2d":
        shape = layer.get("conv")
        kind = "conv"
        oh = shape.get("oh") if isinstance(shape, dict) else None
        ow = shape.get("ow") if isinstance(shape, dict) else None
    elif operator == "fc":
        kind = "fc"
        oh = 0
        ow = 0
    else:
        fail(f"unsupported operator in execution manifest: {operator}")
    values = {
        "kind": kind,
        "qbits": profile.get("qbits"),
        "bw": profile.get("bw"),
        "size_a": layer.get("size_input_words"),
        "size_b": layer.get("size_weight_words"),
        "size_c": layer.get("size_output_words"),
        "cross_terms": layer.get("cross_terms"),
        "ring_batches": layer.get("ring_batches"),
        "application_slots": layer.get("ring_application_slots"),
        "bootstrap_slots": layer.get("ring_bootstrap_slots"),
        "oh": oh,
        "ow": ow,
    }
    if any(not isinstance(value, int) or value < 0
           for key, value in values.items() if key != "kind"):
        fail(f"manifest has malformed public work for {layer.get('layer')}")
    return values


def run_layer(
    layer: dict[str, Any], profile: dict[str, Any], bin_dir: pathlib.Path
) -> dict[str, int | str]:
    name = layer.get("layer")
    binary_name = layer.get("binary")
    cli = layer.get("cli")
    if (not isinstance(name, str) or not isinstance(binary_name, str) or
            not isinstance(cli, list) or not all(isinstance(value, str) for value in cli)):
        fail("manifest layer is missing its binary/CLI binding")
    binary = bin_dir / binary_name
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        fail(f"compiled adapter is missing or not executable: {binary}")
    command = [
        str(binary), "--plan", "--qbits", str(profile["qbits"]),
        "--bw", str(profile["bw"]), "--ole-n", str(profile["ole_n"]),
        "--ole-c", str(profile["ole_c"]), "--ole-t", str(profile["ole_t"]),
        "--noise", str(profile["noise"]), *cli,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        fail(f"adapter rejected {name}: rc={completed.returncode}: {completed.stderr.strip()}")
    return parse_row(completed.stdout, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--bin-dir", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load(args.manifest.resolve(strict=True))
    profile = document.get("profile")
    layers = document.get("layers")
    if not isinstance(profile, dict) or not isinstance(layers, list):
        fail("execution manifest is missing profile or layers")
    if len(layers) != 21 or [layer.get("linear_order") for layer in layers] != list(range(1, 22)):
        fail("execution manifest does not contain 21 contiguous ordered layers")
    stock_nodes = document.get("stock_state_nodes")
    merges = document.get("residual_merges")
    key_stream = document.get("stock_key_stream")
    if (not isinstance(stock_nodes, list) or
            not all(isinstance(node, dict) for node in stock_nodes) or
            [node.get("node") for node in stock_nodes] != [
                "globalaveragepool46", "flatten47",
                "gemm48_pre_sign_extend", "gemm48_terminal_output",
            ] or any(node.get("execution") !=
                     "not_executed_by_linear_record_artifact"
                     for node in stock_nodes)):
        fail("execution manifest stock-state contract mismatch")
    expected_merges = [
        ("conv5", "relu2", "identity"), ("conv10", "relu7", "identity"),
        ("conv15", "conv16", "projection"),
        ("conv21", "relu18", "identity"),
        ("conv26", "conv27", "projection"),
        ("conv32", "relu29", "identity"),
        ("conv37", "conv38", "projection"),
        ("conv43", "relu40", "identity"),
    ]
    if (not isinstance(merges, list) or
            not all(isinstance(merge, dict) for merge in merges) or
            [(merge.get("main_operand"), merge.get("shortcut_operand"),
              merge.get("shortcut_kind")) for merge in merges] !=
            expected_merges):
        fail("execution manifest residual topology mismatch")
    if (not isinstance(key_stream, list) or len(key_stream) != 62 or
            not all(isinstance(item, dict) for item in key_stream) or
            [item.get("stream_position") for item in key_stream] !=
            list(range(1, 63)) or
            sum(item.get("key_kind") == "GPUStTRKey"
                for item in key_stream) != 21 or
            sum(item.get("ringlpn") is True for item in key_stream) != 21 or
            key_stream[-1].get("key_kind") != "raw_mask[1000_words]"):
        fail("execution manifest stock key-stream contract mismatch")
    seen: set[str] = set()
    conv_count = 0
    branch_count = 0
    total_cross_terms = 0
    total_batches = 0
    for layer in layers:
        if not isinstance(layer, dict):
            fail("execution manifest contains a non-object layer")
        name = layer.get("layer")
        if not isinstance(name, str) or name in seen:
            fail(f"missing or duplicate layer name: {name}")
        truncation = layer.get("truncation")
        if not isinstance(truncation, dict):
            fail(f"missing truncation contract for {name}")
        expected_truncation = (
            "LocalARSAfterFinalUnmask" if layer.get("operator") == "fc"
            else "StochasticTR"
        )
        bout = truncation.get("bout")
        shift = truncation.get("shift")
        if (truncation.get("kind") != expected_truncation or
                truncation.get("bin") != profile.get("bw") or
                not isinstance(bout, int) or not isinstance(shift, int) or
                bout + shift != profile.get("bw") or
                layer.get("state_transition_execution") !=
                "not_executed_by_linear_record_artifact"):
            fail(f"unsafe state-transition contract for {name}")
        seen.add(name)
        observed = run_layer(layer, profile, args.bin_dir.resolve(strict=True))
        wanted = expected(layer, profile)
        if observed != wanted:
            fail(f"compiled public work disagrees for {name}: expected={wanted} observed={observed}")
        conv_count += layer.get("operator") == "conv2d"
        branch_count += layer.get("state_input_kind") == "G(branch)"
        total_cross_terms += int(observed["cross_terms"])
        total_batches += int(observed["ring_batches"])
    summary = document.get("summary")
    qbits = profile.get("qbits")
    ole_n = profile.get("ole_n")
    ole_c = profile.get("ole_c")
    ole_t = profile.get("ole_t")
    application_slots = profile.get("ring_application_slots")
    bootstrap_slots = profile.get("ring_bootstrap_slots")
    if not all(isinstance(value, int) and value > 0 for value in (
            qbits, ole_n, ole_c, ole_t, application_slots, bootstrap_slots)):
        fail("execution manifest has malformed aggregate profile values")
    limbs = qbits // 64
    domain = 2 * (ole_n // ole_t)
    log_domain = domain.bit_length() - 1
    ring_instances = 2 * limbs * total_batches
    dpf_trees = ring_instances * 2 * ole_c * ole_t * ole_t
    application_words = 2 * limbs * total_cross_terms
    application_capacity = ring_instances * application_slots
    expected_summary = {
        "total_ring_ole_instances": ring_instances,
        "total_dpf_trees": dpf_trees,
        "total_private_dpf_prg_node_expansions":
            dpf_trees * (domain - 1),
        "total_dpf_string_ots": 2 * dpf_trees * log_domain,
        "total_dpf_bit_triples": dpf_trees * (log_domain - 1),
        "total_dpf_scalar_oles": ring_instances * bootstrap_slots,
        "total_public_a_seed_words": len(layers) * 4,
        "total_ring_application_slots": application_words,
        "total_ring_application_capacity": application_capacity,
        "total_ring_application_slots_discarded":
            application_capacity - application_words,
    }
    if (not isinstance(summary, dict) or conv_count != 20 or
            branch_count != 3 or
            total_cross_terms != summary.get("total_cross_terms") or
            total_batches != summary.get("total_ring_batches") or
            any(summary.get(field) != value
                for field, value in expected_summary.items())):
        fail("aggregate shape/cost coverage disagrees with the execution manifest")
    print(
        "full-linear-shapes: PASS "
        f"layers={len(layers)} conv={conv_count} branch={branch_count} "
        f"cross_terms={total_cross_terms} ring_batches={total_batches} "
        f"plan_digest={document.get('plan_digest')}"
    )


if __name__ == "__main__":
    main()
