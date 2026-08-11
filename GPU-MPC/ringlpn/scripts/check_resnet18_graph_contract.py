#!/usr/bin/env python3
"""Compare the compiled full-graph contract with the pinned Orca source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import signal
import sys
from typing import Any, NoReturn


EXPECTED_MANIFEST_SHA256 = (
    "f5f17ee9be08a7d94f89927b0314a95dec59104f73d0c645f9acdd421555a986"
)
PROBE_SCHEMA = "ringlpn-resnet18-compiled-graph-contract-v1"
CNN_ANCHOR = re.compile(r"GPU-MPC/experiments/orca/cnn\.h:(\d+)")
STREAM_KIND = {
    "GPUConv2DKey[I,F,O]": "linear",
    "GPUMatmulKey[A,B,C]": "linear",
    "GPUMaxpoolKey": "maxpool",
    "GPUReluExtendKey": "relu_extend",
    "GPUSignExtendKey": "sign_extend",
    "raw_mask[1000_words]": "output",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"resnet18-graph-contract: {message}")


def load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"invalid {label}: {error}")
    if not isinstance(data, dict):
        fail(f"{label} must be a JSON object")
    return data


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_regular_file(path: pathlib.Path, label: str,
                         executable: bool = False) -> pathlib.Path:
    candidate = path.absolute()
    current = pathlib.Path(candidate.anchor)
    try:
        for part in candidate.parts[1:]:
            current /= part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                fail(f"{label} contains a symlink component: {current}")
    except OSError as error:
        fail(f"{label} is unavailable: {error}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or (executable and not os.access(resolved, os.X_OK)):
        requirement = "executable regular file" if executable else "regular file"
        fail(f"{label} must be a {requirement}")
    return resolved


def cnn_line(binding: dict[str, Any], label: str) -> int:
    anchor = binding.get("source_anchor")
    if not isinstance(anchor, str):
        fail(f"{label} lacks a source anchor")
    match = CNN_ANCHOR.fullmatch(anchor)
    if match is None:
        fail(f"{label} is not anchored in the ResNet18 forward graph")
    return int(match.group(1))


def indexed(rows: list[dict[str, Any]], section: str,
            count: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("section") == section]
    if len(selected) != count:
        fail(f"compiled {section} row count is {len(selected)}, expected {count}")
    try:
        selected.sort(key=lambda row: row["index"])
    except (KeyError, TypeError):
        fail(f"compiled {section} rows lack integer indexes")
    if [row.get("index") for row in selected] != list(range(count)):
        fail(f"compiled {section} indexes are not canonical")
    return selected


def expect_rows(actual: list[dict[str, Any]],
                expected: list[dict[str, Any]], label: str) -> None:
    if len(actual) != len(expected):
        fail(f"{label} row count mismatch")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            fail(
                f"{label} row {index} differs from the source manifest: "
                f"compiled={json.dumps(got, sort_keys=True)} "
                f"expected={json.dumps(want, sort_keys=True)}"
            )


def parse_probe(binary: pathlib.Path) -> list[dict[str, Any]]:
    try:
        metadata = binary.stat()
    except OSError as error:
        fail(f"contract probe unavailable: {error}")
    if (not stat.S_ISREG(metadata.st_mode) or binary.is_symlink() or
            not os.access(binary, os.X_OK)):
        fail("contract probe must be a non-symlink executable regular file")
    process: subprocess.Popen[bytes] | None = None
    isolated = os.environ.get(
        "RINGLPN_COOPERATIVE_PROCESS_GROUP", "0"
    ) != "1"
    try:
        process = subprocess.Popen(
            [str(binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=isolated,
        )
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                if isolated:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if isolated:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            except ProcessLookupError:
                pass
        fail("contract probe timed out after 30 seconds")
    except OSError as error:
        fail(f"contract probe failed to execute: {error}")
    assert process is not None
    if process.returncode != 0 or stderr:
        fail(
            f"contract probe rejected its compiled contract "
            f"(exit={process.returncode}, stderr_bytes={len(stderr)})"
        )
    try:
        text = stdout.decode("ascii")
    except UnicodeDecodeError as error:
        fail(f"contract probe output is not ASCII: {error}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid probe JSONL line {line_number}: {error}")
        if not isinstance(row, dict):
            fail(f"probe JSONL line {line_number} is not an object")
        rows.append(row)
    if not rows:
        fail("contract probe produced no rows")
    return rows


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=pathlib.Path,
        default=root / "results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json",
    )
    parser.add_argument(
        "--contract-bin", type=pathlib.Path,
        default=root / "bin/test_resnet18_graph_contract",
    )
    args = parser.parse_args()

    manifest_path = resolve_regular_file(
        args.manifest, "source execution manifest",
    )
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        fail("manifest digest differs from the approved source-bound checkpoint")
    manifest = load_object(manifest_path, "source execution manifest")
    if manifest.get("schema") != "ringlpn-full-linear-execution-v1" or \
            manifest.get("model") != "ResNet18":
        fail("unexpected source execution manifest schema or model")

    layers = manifest.get("layers")
    stream = manifest.get("stock_key_stream")
    merges = manifest.get("residual_merges")
    state_nodes = manifest.get("stock_state_nodes")
    profile = manifest.get("profile")
    if not (isinstance(layers, list) and len(layers) == 21 and
            isinstance(stream, list) and len(stream) == 62 and
            isinstance(merges, list) and len(merges) == 8 and
            isinstance(state_nodes, list) and len(state_nodes) == 4 and
            isinstance(profile, dict)):
        fail("source manifest graph cardinalities are invalid")

    layer_by_name: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("linear_order") != index + 1:
            fail("source linear layers are not canonically ordered")
        name = layer.get("layer")
        if not isinstance(name, str) or name in layer_by_name:
            fail("source linear layer names are invalid or duplicated")
        layer_by_name[name] = (index, layer)

    stream_by_position: list[dict[str, Any]] = []
    for index, item in enumerate(stream):
        if not isinstance(item, dict) or item.get("stream_position") != index + 1:
            fail("source stock-key stream is not canonically ordered")
        stream_by_position.append(item)

    linear_stream: dict[str, dict[str, Any]] = {}
    stock_stream: list[dict[str, Any]] = []
    truncation_stream: list[dict[str, Any]] = []
    for item in stream_by_position:
        key_kind = item.get("key_kind")
        node = item.get("node")
        if not isinstance(key_kind, str) or not isinstance(node, str):
            fail("source stream row lacks key kind or node")
        if key_kind in ("GPUConv2DKey[I,F,O]", "GPUMatmulKey[A,B,C]"):
            if node in linear_stream:
                fail("source linear stream node is duplicated")
            linear_stream[node] = item
        elif key_kind == "GPUStTRKey":
            truncation_stream.append(item)
        elif key_kind in ("GPUMaxpoolKey", "GPUReluExtendKey", "GPUSignExtendKey"):
            stock_stream.append(item)
        elif key_kind != "raw_mask[1000_words]":
            fail(f"unsupported source stream key kind: {key_kind}")
    if set(linear_stream) != set(layer_by_name) or len(truncation_stream) != 21 or \
            len(stock_stream) != 19:
        fail("source stream does not cover the graph contract")

    calls_by_line: dict[int, tuple[str, int, str]] = {}

    def add_call(line: int, descriptor: tuple[str, int, str]) -> None:
        previous = calls_by_line.get(line)
        if previous is not None and previous != descriptor:
            fail(f"conflicting source graph calls on cnn.h:{line}")
        calls_by_line[line] = descriptor

    for index, layer in enumerate(layers):
        item = linear_stream[layer["layer"]]
        add_call(cnn_line(item["forward_call"], f"linear {layer['layer']}"),
                 ("layer", index, layer["layer"]))
    stock_index_by_node: dict[str, int] = {}
    for index, item in enumerate(stock_stream):
        node = item["node"]
        if node in stock_index_by_node:
            fail("source stock nonlinear node is duplicated")
        stock_index_by_node[node] = index
        if item["key_kind"] != "GPUSignExtendKey":
            add_call(cnn_line(item["forward_call"], f"stock node {node}"),
                     ("stock", index, node))
    residual_index_by_id: dict[str, int] = {}
    for index, merge in enumerate(merges):
        if not isinstance(merge, dict):
            fail("residual merge row must be an object")
        branch_id = merge.get("branch_id")
        if not isinstance(branch_id, str) or branch_id in residual_index_by_id:
            fail("residual merge identifiers are invalid")
        residual_index_by_id[branch_id] = index
        add_call(cnn_line(merge["merge_call"], f"residual {branch_id}"),
                 ("residual", index, branch_id))
    for node in state_nodes:
        if not isinstance(node, dict):
            fail("stock state node must be an object")
        forward = node.get("forward_call")
        if isinstance(forward, dict):
            name = node.get("node")
            if not isinstance(name, str):
                fail("stock state node lacks a name")
            add_call(cnn_line(forward, f"state node {name}"),
                     ("state", int(node["forward_order"]), name))

    ordered_calls = sorted(calls_by_line.items())
    if len(ordered_calls) != 49:
        fail(f"source forward graph has {len(ordered_calls)} calls, expected 49")
    call_by_order: list[tuple[str, int, str]] = []
    order_by_line: dict[int, int] = {}
    for order, (line, descriptor) in enumerate(ordered_calls):
        order_by_line[line] = order
        call_by_order.append(descriptor)
    for index, layer in enumerate(layers):
        line = cnn_line(linear_stream[layer["layer"]]["forward_call"],
                        f"linear {layer['layer']}")
        if layer.get("forward_order") != order_by_line[line]:
            fail(f"linear {layer['layer']} forward order is not source-derived")
    for index, merge in enumerate(merges):
        line = cnn_line(merge["merge_call"], f"residual {index}")
        if merge.get("merge_forward_order") != order_by_line[line]:
            fail(f"residual merge {index} forward order is not source-derived")
    for node in state_nodes:
        forward = node.get("forward_call")
        if isinstance(forward, dict):
            line = cnn_line(forward, f"state node {node.get('node')}")
            if node.get("forward_order") != order_by_line[line]:
                fail(f"state node {node.get('node')} forward order is not source-derived")

    shifts = {
        layer["truncation"]["shift"] for layer in layers[:20]
        if isinstance(layer.get("truncation"), dict)
    }
    if len(shifts) != 1 or not isinstance(profile.get("bw"), int):
        fail("source arithmetic profile is not uniform")
    shift = shifts.pop()
    full_bw = profile["bw"]
    truncated_bw = full_bw - shift
    if full_bw != 32 or shift <= 0 or truncated_bw <= 0:
        fail("unsupported source arithmetic profile")

    expected_linear: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        kind = layer["operator"]
        conv = layer.get("conv") if kind == "conv2d" else None
        matmul = layer.get("matmul") if kind == "fc" else None
        if kind == "conv2d" and not isinstance(conv, dict):
            fail(f"linear {layer['layer']} lacks convolution parameters")
        if kind == "fc" and not isinstance(matmul, dict):
            fail(f"linear {layer['layer']} lacks matrix parameters")
        expected_linear.append({
            "section": "linear", "index": index, "name": layer["layer"],
            "kind": kind,
            "stream_position": linear_stream[layer["layer"]]["stream_position"],
            "source_line": layer["source_line"],
            "n": conv["n"] if conv else 0,
            "h": conv["h"] if conv else 0,
            "w": conv["w"] if conv else 0,
            "ci": conv["ci"] if conv else 0,
            "fh": conv["fh"] if conv else 0,
            "fw": conv["fw"] if conv else 0,
            "co": conv["co"] if conv else 0,
            "padding": conv["padding"] if conv else 0,
            "stride": conv["stride"] if conv else 0,
            "rows": matmul["rows"] if matmul else 0,
            "inner": matmul["inner"] if matmul else 0,
            "cols": matmul["cols"] if matmul else 0,
            "input_words": layer["size_input_words"],
            "weight_words": layer["size_weight_words"],
            "output_words": layer["size_output_words"],
        })

    expected_truncation: list[dict[str, Any]] = []
    truncation_index_by_node: dict[str, int] = {}
    for index, item in enumerate(truncation_stream):
        node = item["node"]
        if node == "globalaveragepool46":
            linear_index = -1
            words = layers[-1]["size_input_words"]
        else:
            if node not in layer_by_name:
                fail(f"truncation node {node} is not a linear layer")
            linear_index, layer = layer_by_name[node]
            words = layer["size_output_words"]
        truncation_index_by_node[node] = index
        expected_truncation.append({
            "section": "truncation", "index": index, "node": node,
            "stream_position": item["stream_position"],
            "linear_index": linear_index, "words": words,
        })

    expected_residual: list[dict[str, Any]] = []
    residual_words: list[int] = []
    for index, merge in enumerate(merges):
        main_name = merge.get("main_operand")
        shortcut_name = merge.get("shortcut_operand")
        if main_name not in layer_by_name:
            fail(f"residual {index} main operand is not a linear layer")
        main_index, main_layer = layer_by_name[main_name]
        if shortcut_name in layer_by_name:
            shortcut_kind = "truncation"
            shortcut_index = layer_by_name[shortcut_name][0]
        elif shortcut_name in stock_index_by_node:
            shortcut_kind = "stock"
            shortcut_index = stock_index_by_node[shortcut_name]
        else:
            fail(f"residual {index} shortcut operand is not source-bound")
        merge_order = merge["merge_forward_order"]
        if merge_order + 1 >= len(call_by_order):
            fail(f"residual {index} has no following activation")
        result_call = call_by_order[merge_order + 1]
        if result_call[0] != "stock":
            fail(f"residual {index} is not followed by a stock activation")
        words = main_layer["size_output_words"]
        residual_words.append(words)
        expected_residual.append({
            "section": "residual", "index": index,
            "main_kind": "truncation", "main_index": main_index,
            "shortcut_kind": shortcut_kind,
            "shortcut_index": shortcut_index,
            "result_stock_index": result_call[1], "words": words,
        })

    expected_stock_input: list[dict[str, Any]] = []
    stock_input_sources: list[tuple[str, int]] = []
    for index, item in enumerate(stock_stream):
        if item["key_kind"] == "GPUSignExtendKey":
            source = ("truncation", truncation_index_by_node["globalaveragepool46"])
        else:
            line = cnn_line(item["forward_call"], f"stock {item['node']}")
            order = order_by_line[line]
            if order == 0:
                fail(f"stock {item['node']} has no source predecessor")
            predecessor = call_by_order[order - 1]
            if predecessor[0] == "layer":
                source = ("truncation", predecessor[1])
            elif predecessor[0] in ("stock", "residual"):
                source = (predecessor[0], predecessor[1])
            else:
                fail(f"stock {item['node']} predecessor is not executable")
        stock_input_sources.append(source)
        expected_stock_input.append({
            "section": "stock_input", "index": index,
            "source_kind": source[0], "source_index": source[1],
        })

    stock_output_words: list[int] = []
    expected_stock: list[dict[str, Any]] = []

    def source_words(source: tuple[str, int]) -> int:
        kind, index = source
        if kind == "truncation":
            return expected_truncation[index]["words"]
        if kind == "stock":
            return stock_output_words[index]
        if kind == "residual":
            return residual_words[index]
        fail(f"invalid derived value source kind: {kind}")

    for index, item in enumerate(stock_stream):
        kind = STREAM_KIND[item["key_kind"]]
        input_words = source_words(stock_input_sources[index])
        if kind == "maxpool":
            line = cnn_line(item["forward_call"], f"stock {item['node']}")
            order = order_by_line[line]
            later_layers = [
                layer for layer in layers if layer["forward_order"] > order
            ]
            if not later_layers:
                fail("maxpool has no following linear layer")
            output_words = later_layers[0]["size_input_words"]
        else:
            output_words = input_words
        stock_output_words.append(output_words)
        node = (item["node"] + "-signextend"
                if kind == "sign_extend" else item["node"])
        expected_stock.append({
            "section": "stock", "index": index, "node": node,
            "kind": kind, "stream_position": item["stream_position"],
            "input_bw": truncated_bw,
            "output_bw": truncated_bw if kind == "maxpool" else full_bw,
            "input_words": input_words, "output_words": output_words,
        })

    expected_remask: list[dict[str, Any]] = []
    for target in range(1, len(layers)):
        layer = layers[target]
        branch = layer.get("residual_branch")
        if isinstance(branch, dict):
            source_order = branch.get("branch_source_forward_order")
            if not isinstance(source_order, int) or not (0 <= source_order < len(call_by_order)):
                fail(f"linear {layer['layer']} has invalid branch source order")
            source_call = call_by_order[source_order]
        elif layer["operator"] == "fc":
            source_call = ("sign_extend", -1, "sign_extend")
        else:
            source_order = layer["forward_order"] - 1
            source_call = call_by_order[source_order]
        if source_call[0] == "stock":
            source_name = source_call[2]
        elif source_call[0] == "sign_extend":
            source_name = "sign_extend"
        else:
            fail(f"linear {layer['layer']} input is not a remaskable state")
        expected_remask.append({
            "section": "remask", "index": target - 1,
            "target_linear_index": target, "source": source_name,
            "words": layer["size_input_words"],
        })

    expected_stream: list[dict[str, Any]] = []
    for index, item in enumerate(stream_by_position):
        key_kind = item["key_kind"]
        if key_kind == "GPUStTRKey":
            kind = ("global_average" if item["node"] == "globalaveragepool46"
                    else "truncation")
        else:
            kind = STREAM_KIND[key_kind]
        expected_stream.append({
            "section": "stream", "index": index,
            "position": item["stream_position"], "node": item["node"],
            "kind": kind,
        })

    rows = parse_probe(resolve_regular_file(
        args.contract_bin, "compiled contract probe", executable=True,
    ))
    meta_rows = [row for row in rows if row.get("section") == "meta"]
    expected_meta = {
        "section": "meta", "schema": PROBE_SCHEMA,
        "full_bw": full_bw, "scale": shift,
        "truncated_bw": truncated_bw,
        "linear_count": len(expected_linear),
        "truncation_count": len(expected_truncation),
        "stock_count": len(expected_stock),
        "remask_count": len(expected_remask),
        "residual_count": len(expected_residual),
        "stream_count": len(expected_stream), "status": "pass",
    }
    if meta_rows != [expected_meta]:
        fail("compiled contract metadata differs from the source manifest")

    expected_sections = {
        "meta", "linear", "truncation", "stock", "remask", "residual",
        "stock_input", "stream",
    }
    if {row.get("section") for row in rows} != expected_sections:
        fail("compiled contract contains missing or extra sections")
    expect_rows(indexed(rows, "linear", len(expected_linear)),
                expected_linear, "linear contract")
    expect_rows(indexed(rows, "truncation", len(expected_truncation)),
                expected_truncation, "truncation contract")
    expect_rows(indexed(rows, "stock", len(expected_stock)),
                expected_stock, "stock nonlinear contract")
    expect_rows(indexed(rows, "remask", len(expected_remask)),
                expected_remask, "remask dependency contract")
    expect_rows(indexed(rows, "residual", len(expected_residual)),
                expected_residual, "residual dependency contract")
    expect_rows(indexed(rows, "stock_input", len(expected_stock_input)),
                expected_stock_input, "stock input dependency contract")
    expect_rows(indexed(rows, "stream", len(expected_stream)),
                expected_stream, "stock key stream contract")

    print(
        "[resnet18-graph-contract] PASS "
        "(21 linear, 21 truncation, 19 stock, 20 remask, "
        "8 residual, 62 stream items; projection shortcuts source-bound)"
    )


if __name__ == "__main__":
    main()
