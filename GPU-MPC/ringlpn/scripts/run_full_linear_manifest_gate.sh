#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
BUILDER="$ROOT/scripts/build_full_linear_model_manifest.py"
LAYER_MANIFEST="$ROOT/results/fc/orca_forward_linear_layer_manifest_2026_08_04.json"
EXECUTION_MANIFEST="$ROOT/results/fc/resnet18_full_linear_execution_manifest_2026_08_06.json"
ADAPTIVE_MANIFEST="$ROOT/results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json"
RUNNER="$ROOT/scripts/run_full_linear_record_set.py"
SHAPE_CHECKER="$ROOT/scripts/check_full_linear_shape_coverage.py"
RUNNER_PLAN_CHECK="${RUNNER_PLAN_CHECK:-0}"
BINARY_APPROVAL="${BINARY_APPROVAL:-$ROOT/results/fc/linear_adapter_binary_approval_2026_08_07.json}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

"$BUILDER" --repo-root "$REPO_ROOT" \
  --layer-manifest "$LAYER_MANIFEST" --model ResNet18 \
  --out "$EXECUTION_MANIFEST" --check >/dev/null
"$BUILDER" --repo-root "$REPO_ROOT" \
  --layer-manifest "$LAYER_MANIFEST" --model ResNet18 --ole-n 262144 \
  --out "$ADAPTIVE_MANIFEST" --check >/dev/null
"$SHAPE_CHECKER" --manifest "$ADAPTIVE_MANIFEST" \
  --bin-dir "$ROOT/bin" >/dev/null
if "$BUILDER" --repo-root "$REPO_ROOT" \
    --layer-manifest "$LAYER_MANIFEST" --model ResNet18 --ole-n 1048576 \
    --out "$WORKDIR/invalid-degree.json" \
    >"$WORKDIR/invalid-degree.log" 2>&1; then
  echo "[linear-record-set-manifest] oversized frontier control unexpectedly passed" >&2
  exit 1
fi
grep -q 'unsupported matrix profile or Ring-OLE degree override' \
  "$WORKDIR/invalid-degree.log"

python3 - "$EXECUTION_MANIFEST" "$WORKDIR/stale.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
digest = document["plan_digest"]
document["plan_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if "$BUILDER" --repo-root "$REPO_ROOT" \
    --layer-manifest "$LAYER_MANIFEST" --model ResNet18 \
    --out "$WORKDIR/stale.json" --check >"$WORKDIR/stale.log" 2>&1; then
  echo "[linear-record-set-manifest] stale-output control unexpectedly passed" >&2
  exit 1
fi
grep -q 'output manifest is stale or modified' "$WORKDIR/stale.log"

python3 - "$LAYER_MANIFEST" "$WORKDIR/bad-registry.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
document["source_registry"][0]["sha256"] = "0" * 64
out.write_text(json.dumps(document), encoding="utf-8")
PY
if "$BUILDER" --repo-root "$REPO_ROOT" \
    --layer-manifest "$WORKDIR/bad-registry.json" --model ResNet18 \
    --out "$WORKDIR/unused.json" >"$WORKDIR/bad-registry.log" 2>&1; then
  echo "[linear-record-set-manifest] source-registry control unexpectedly passed" >&2
  exit 1
fi
grep -q 'source registry digest mismatch' "$WORKDIR/bad-registry.log"

python3 - "$BUILDER" "$WORKDIR/source-registry-controls" <<'PY'
import contextlib
import copy
import hashlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys

builder_path = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
repo.mkdir()
spec = importlib.util.spec_from_file_location(
    "full_linear_manifest_builder_control", builder_path
)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import full-linear manifest builder")
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

tracked = repo / "source" / "tracked.txt"
tracked.parent.mkdir()
trusted = b"trusted source line\n"
tracked.write_bytes(trusted)
subprocess.run(["git", "init", "-q", str(repo)], check=True)
subprocess.run(
    ["git", "-C", str(repo), "add", "--", "source/tracked.txt"],
    check=True,
)
digest = hashlib.sha256(trusted).hexdigest()
base = {
    "source_registry": [
        {"path": "source/tracked.txt", "sha256": digest}
    ]
}

def require_rejection(name, document, expected):
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
        try:
            builder.validate_source_registry(document, repo)
        except SystemExit:
            pass
        else:
            raise SystemExit(f"{name} source-registry control unexpectedly passed")
    if expected not in errors.getvalue():
        raise SystemExit(
            f"{name} source-registry control produced the wrong rejection: "
            f"{errors.getvalue().strip()}"
        )

absolute = copy.deepcopy(base)
absolute["source_registry"][0]["path"] = "/etc/passwd"
require_rejection(name="absolute-escape", document=absolute,
                  expected="canonical relative POSIX")

parent = copy.deepcopy(base)
parent["source_registry"][0]["path"] = "../escape.txt"
require_rejection(name="parent-escape", document=parent,
                  expected="canonical relative POSIX")

(repo / "alias").symlink_to("source", target_is_directory=True)
symlink = copy.deepcopy(base)
symlink["source_registry"][0]["path"] = "alias/tracked.txt"
require_rejection(name="symlink-component", document=symlink,
                  expected="contains a symlink component")

untracked_path = repo / "source" / "untracked.txt"
untracked_path.write_bytes(trusted)
untracked = copy.deepcopy(base)
untracked["source_registry"][0]["path"] = "source/untracked.txt"
require_rejection(name="untracked-file", document=untracked,
                  expected="not a Git-tracked regular file")

held = builder.validate_source_registry(base, repo)
replacement = repo / "replacement.txt"
replacement.write_bytes(b"replacement source line\n")
os.replace(replacement, tracked)
row = {
    "source_anchor": "source/tracked.txt:1",
    "source_text_sha256": hashlib.sha256(b"trusted source line").hexdigest(),
}
line, relative, number = builder.source_line(row, held)
if (line, relative, number) != ("trusted source line", "source/tracked.txt", 1):
    raise SystemExit("post-validation replacement entered held source bytes")
bound = builder.bound_source_line("source/tracked.txt", 1, held)
if bound["source_text_sha256"] != row["source_text_sha256"]:
    raise SystemExit("post-validation replacement entered bound source line")
PY

python3 - "$LAYER_MANIFEST" "$WORKDIR/bad-layer.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
layer = next(row for row in document["layers"] if row.get("model") == "ResNet18")
layer["source_text_sha256"] = "0" * 64
out.write_text(json.dumps(document), encoding="utf-8")
PY
if "$BUILDER" --repo-root "$REPO_ROOT" \
    --layer-manifest "$WORKDIR/bad-layer.json" --model ResNet18 \
    --out "$WORKDIR/unused.json" >"$WORKDIR/bad-layer.log" 2>&1; then
  echo "[linear-record-set-manifest] source-line control unexpectedly passed" >&2
  exit 1
fi
grep -q 'source line digest mismatch' "$WORKDIR/bad-layer.log"

python3 - "$EXECUTION_MANIFEST" "$WORKDIR" <<'PY'
import copy
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
controls = {}
controls["bad-state"] = copy.deepcopy(document)
controls["bad-state"]["stock_state_nodes"][0]["node"] = "missing_gap_transition"
controls["bad-merge"] = copy.deepcopy(document)
controls["bad-merge"]["residual_merges"][0]["shortcut_operand"] = "relu7"
controls["bad-key"] = copy.deepcopy(document)
controls["bad-key"]["stock_key_stream"][1]["stream_position"] = 1
controls["bad-terminal"] = copy.deepcopy(document)
controls["bad-terminal"]["layers"][-1]["truncation"]["kind"] = "StochasticTR"
controls["bad-cost"] = copy.deepcopy(document)
controls["bad-cost"]["summary"]["total_dpf_string_ots"] += 1
for name, value in controls.items():
    (root / f"{name}.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
PY
for control in bad-state bad-merge bad-key bad-terminal bad-cost; do
  if "$SHAPE_CHECKER" --manifest "$WORKDIR/$control.json" --bin-dir "$ROOT/bin" \
      >"$WORKDIR/$control.log" 2>&1; then
    echo "[linear-record-set-manifest] $control control unexpectedly passed" >&2
    exit 1
  fi
done

if [[ "$RUNNER_PLAN_CHECK" != "1" ]]; then
  echo "[linear-record-set-manifest] PASS (fresh source-bound manifest; stale-output, canonical-path, symlink, tracked-file, held-byte, source-registry, source-line controls rejected; executable plan deferred)"
  exit 0
fi
[[ -f "$BINARY_APPROVAL" ]] || {
  echo "[linear-record-set-manifest] missing binary approval: $BINARY_APPROVAL" >&2
  exit 1
}

"$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
  --bin-dir "$ROOT/bin" --output-root "$WORKDIR/planned-run" --plan-only \
  --binary-approval "$BINARY_APPROVAL" >"$WORKDIR/planned-run.log"
grep -q 'PLAN PASS.*(21 isolated adapter rows)' "$WORKDIR/planned-run.log"
"$RUNNER" --repo-root "$REPO_ROOT" --manifest "$ADAPTIVE_MANIFEST" \
  --bin-dir "$ROOT/bin" --output-root "$WORKDIR/adaptive-plan" --plan-only \
  --binary-approval "$BINARY_APPROVAL" >"$WORKDIR/adaptive-plan.log"
grep -q 'PLAN PASS.*(21 isolated adapter rows)' "$WORKDIR/adaptive-plan.log"
python3 - "$WORKDIR/planned-run/PLANNED.json" "$WORKDIR/planned-run" <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
document = json.loads(path.read_text(encoding="utf-8"))
unsigned = dict(document)
claimed_digest = unsigned.pop("digest")
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
assert hashlib.sha256(canonical).hexdigest() == claimed_digest
assert document["schema"] == "ringlpn-forward-linear-plan-validation-v2"
assert document["composition_status"] == "isolated_linear_records_no_graph_state"
assert document["artifact_scope"] == "ordered_isolated_forward_linear_adapter_records"
assert document["security_scope"] == "static_semi_honest_component_composition_only_no_concrete_security_level"
assert document["execution_exclusions"] == [
    "graph_state", "residual_merges", "stochastic_truncation",
    "nonlinear_layers", "full_model_execution",
]
approval = document["binary_approval"]
assert approval["path"] == "private_inputs/binary_approval.json"
assert hashlib.sha256((root / approval["path"]).read_bytes()).hexdigest() == approval["sha256"]
provenance = document["linear_adapter_build_provenance"]
assert provenance["path"] == "private_inputs/linear_adapter_build_provenance.json"
provenance_path = root / provenance["path"]
assert provenance_path.stat().st_size == provenance["bytes"]
assert hashlib.sha256(provenance_path.read_bytes()).hexdigest() == provenance["sha256"]
assert document["plan_csv_header"] == [
    "operator", "qbits", "bw", "size_input_words", "size_weight_words",
    "size_output_words", "cross_terms", "ring_batches",
    "ring_application_slots", "ring_bootstrap_slots", "output_height",
    "output_width",
]
assert len(document["layers"]) == 21
assert [row["linear_order"] for row in document["layers"]] == list(range(1, 22))
assert len({row["compatibility_id"] for row in document["layers"]}) == 21
expected_header = {
    "operator", "qbits", "bw", "size_input_words", "size_weight_words",
    "size_output_words", "cross_terms", "ring_batches",
    "ring_application_slots", "ring_bootstrap_slots", "output_height",
    "output_width",
}
assert all(set(row["plan"]) == expected_header and "row" not in row for row in document["layers"])
binaries = document["executable_provenance"]
assert set(binaries) == {"test_two_party_fc_preprocess", "test_two_party_conv_preprocess"}
assert all(binding["provenance"] == "validated_reproducible_build_receipts_snapshot" for binding in binaries.values())
assert all(hashlib.sha256((root / binding["path"]).read_bytes()).hexdigest() == binding["sha256"] for binding in binaries.values())
source_binding = document["source_execution_manifest"]
assert hashlib.sha256((root / source_binding["path"]).read_bytes()).hexdigest() == source_binding["sha256"]
assert all(stat.S_IMODE((root / binding["path"]).stat().st_mode) == 0o500 for binding in binaries.values())
assert not (root / "COMMITTED.manifest").exists()
assert not (root / "LINEAR_RECORD_SET.manifest").exists()
PY
if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/planned-run" --plan-only \
    --binary-approval "$BINARY_APPROVAL" >"$WORKDIR/stale-root.log" 2>&1; then
  echo "[linear-record-set-manifest] stale run-root control unexpectedly passed" >&2
  exit 1
fi
grep -q 'output root already exists' "$WORKDIR/stale-root.log"
if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/same-gpu" --plan-only \
    --p0-gpu 1 --p1-gpu 1 >"$WORKDIR/same-gpu.log" 2>&1; then
  echo "[linear-record-set-manifest] same-GPU isolation control unexpectedly passed" >&2
  exit 1
fi
grep -q 'party GPUs must be distinct' "$WORKDIR/same-gpu.log"

"$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
  --bin-dir "$ROOT/bin" --output-root "$WORKDIR/lpt-plan" --plan-only \
  --binary-approval "$BINARY_APPROVAL" \
  --lane 0:1:1:22000-22085 --lane 2:3:3:22200-22285 \
  --scheduler-control canonical-aggregation >"$WORKDIR/lpt-plan.log"
grep -q 'SCHEDULER CONTROL PASS.*shuffled completion aggregated canonically' \
  "$WORKDIR/lpt-plan.log"
python3 - "$WORKDIR/lpt-plan/PLANNED.json" <<'PY'
import json
import pathlib
import sys

document = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
schedule = document["resource_schedule"]
assert schedule["algorithm"] == "deterministic_lpt_by_ring_batches"
assert schedule["stable_tie_break"] == "canonical_linear_order_then_lane_order"
lanes = schedule["lanes"]
assert [lane["lane"] for lane in lanes] == [0, 1]
assert sorted(order for lane in lanes for order in lane["linear_orders"]) == list(range(1, 22))
costs = {
    row["linear_order"]: row["plan"]["ring_batches"]
    for row in document["layers"]
}
jobs = sorted(costs, key=lambda order: (-costs[order], order))
expected = [[], []]
loads = [0, 0]
for order in jobs:
    lane = min(range(2), key=lambda index: (loads[index], index))
    expected[lane].append(order)
    loads[lane] += costs[order]
assert [lane["linear_orders"] for lane in lanes] == expected
assert [lane["estimated_ring_batches"] for lane in lanes] == loads
PY

"$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
  --bin-dir "$ROOT/bin" --output-root "$WORKDIR/worker-failure" --plan-only \
  --binary-approval "$BINARY_APPROVAL" \
  --lane 0:1:1:22400-22485 --lane 2:3:3:22600-22685 \
  --scheduler-control worker-failure >"$WORKDIR/worker-failure.log"
grep -q 'SCHEDULER CONTROL PASS.*injected failure cancelled all peer lanes' \
  "$WORKDIR/worker-failure.log"
[[ ! -e "$WORKDIR/worker-failure/LINEAR_RECORD_SET.manifest" ]]

if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/malformed-lane" --plan-only \
    --lane 0:1:1:not-a-range >"$WORKDIR/malformed-lane.log" 2>&1; then
  echo "[linear-record-set-manifest] malformed-lane control unexpectedly passed" >&2
  exit 1
fi
grep -q 'malformed resource lane' "$WORKDIR/malformed-lane.log"

if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/negative-lane-gpu" --plan-only \
    --lane=-1:0:0:22000-22085 >"$WORKDIR/negative-lane-gpu.log" 2>&1; then
  echo "[linear-record-set-manifest] negative-lane-GPU control unexpectedly passed" >&2
  exit 1
fi
grep -q 'invalid GPU ordinal' "$WORKDIR/negative-lane-gpu.log"

if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/shared-lane-gpu" --plan-only \
    --lane 0:1:1:22000-22085 --lane 1:2:2:22200-22285 \
    >"$WORKDIR/shared-lane-gpu.log" 2>&1; then
  echo "[linear-record-set-manifest] shared-lane-GPU control unexpectedly passed" >&2
  exit 1
fi
grep -q 'share a GPU ordinal' "$WORKDIR/shared-lane-gpu.log"

if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/overlap-lane-port" --plan-only \
    --lane 0:1:1:22000-22085 --lane 2:3:3:22080-22165 \
    >"$WORKDIR/overlap-lane-port.log" 2>&1; then
  echo "[linear-record-set-manifest] overlapping-lane-port control unexpectedly passed" >&2
  exit 1
fi
grep -q 'overlapping port ranges' "$WORKDIR/overlap-lane-port.log"

many_lane_args=()
for ((index = 0; index < 22; ++index)); do
  first_port=$((2000 + 100 * index))
  many_lane_args+=(
    --lane
    "$((3 * index)):$((3 * index + 1)):$((3 * index + 1)):${first_port}-$((first_port + 85))"
  )
done
if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/too-many-lanes" --plan-only \
    "${many_lane_args[@]}" >"$WORKDIR/too-many-lanes.log" 2>&1; then
  echo "[linear-record-set-manifest] too-many-lanes control unexpectedly passed" >&2
  exit 1
fi
grep -q 'resource-lane count exceeds the number of linear jobs' \
  "$WORKDIR/too-many-lanes.log"

python3 - "$BINARY_APPROVAL" "$WORKDIR/stale-approval.json" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
binding = document["binaries"]["test_two_party_conv_preprocess"]
digest = binding["sha256"]
binding["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
unsigned = dict(document)
del unsigned["approval_digest"]
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
document["approval_digest"] = hashlib.sha256(canonical).hexdigest()
out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/stale-approval-plan" --plan-only \
    --binary-approval "$WORKDIR/stale-approval.json" \
    >"$WORKDIR/stale-approval.log" 2>&1; then
  echo "[linear-record-set-manifest] stale binary-approval control unexpectedly passed" >&2
  exit 1
fi
grep -q 'binary approval outputs differ from build receipts' \
  "$WORKDIR/stale-approval.log"

python3 - "$BINARY_APPROVAL" "$WORKDIR/direct-approval.json" <<'PY'
import hashlib
import json
import pathlib
import sys

document = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
document["build_provenance"] = None
unsigned = dict(document)
del unsigned["approval_digest"]
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
document["approval_digest"] = hashlib.sha256(canonical).hexdigest()
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
if "$RUNNER" --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$ROOT/bin" --output-root "$WORKDIR/direct-approval-plan" --plan-only \
    --binary-approval "$WORKDIR/direct-approval.json" \
    >"$WORKDIR/direct-approval.log" 2>&1; then
  echo "[linear-record-set-manifest] receipt-free direct approval unexpectedly passed" >&2
  exit 1
fi
grep -q 'binary approval lacks build provenance' \
  "$WORKDIR/direct-approval.log"

mkdir -m 700 "$WORKDIR/control-bin"
cat >"$WORKDIR/control-bin/plan-control" <<'PY'
#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys

target = pathlib.Path(os.environ["REAL_BIN_DIR"]) / pathlib.Path(sys.argv[0]).name
completed = subprocess.run([str(target), *sys.argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stderr.buffer.write(completed.stderr)
if completed.returncode != 0 or "--plan" not in sys.argv[1:] or "conv" not in target.name:
    sys.stdout.buffer.write(completed.stdout)
    raise SystemExit(completed.returncode)
text = completed.stdout.decode("utf-8")
if os.environ["PLAN_CONTROL"] == "duplicate":
    sys.stdout.write(text)
    sys.stdout.write(text)
elif os.environ["PLAN_CONTROL"] == "mismatch":
    values = text.strip().split(",")
    values[6] = str(int(values[6]) + 1)
    sys.stdout.write(",".join(values) + "\n")
else:
    raise SystemExit("unknown PLAN_CONTROL")
PY
chmod 500 "$WORKDIR/control-bin/plan-control"
cp "$WORKDIR/control-bin/plan-control" "$WORKDIR/control-bin/test_two_party_conv_preprocess"
cp "$WORKDIR/control-bin/plan-control" "$WORKDIR/control-bin/test_two_party_fc_preprocess"
chmod 500 "$WORKDIR/control-bin/test_two_party_conv_preprocess" "$WORKDIR/control-bin/test_two_party_fc_preprocess"

if PLAN_CONTROL=duplicate REAL_BIN_DIR="$ROOT/bin" "$RUNNER" \
    --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$WORKDIR/control-bin" --output-root "$WORKDIR/tampered-plan" --plan-only \
    >"$WORKDIR/tampered-plan.log" 2>&1; then
  echo "[linear-record-set-manifest] plan-row tamper control unexpectedly passed" >&2
  exit 1
fi
grep -q 'must contain exactly one complete operator row' "$WORKDIR/tampered-plan.log"

if PLAN_CONTROL=mismatch REAL_BIN_DIR="$ROOT/bin" "$RUNNER" \
    --repo-root "$REPO_ROOT" --manifest "$EXECUTION_MANIFEST" \
    --bin-dir "$WORKDIR/control-bin" --output-root "$WORKDIR/binary-plan-mismatch" --plan-only \
    >"$WORKDIR/binary-plan-mismatch.log" 2>&1; then
  echo "[linear-record-set-manifest] binary-plan mismatch control unexpectedly passed" >&2
  exit 1
fi
grep -q 'binary plan mismatch for conv0.cross_terms' "$WORKDIR/binary-plan-mismatch.log"

echo "[linear-record-set-manifest] PASS (fresh baseline/adaptive 21-layer approved executable plans; deterministic LPT/canonical aggregation/fail-fast controls pass; malformed/shared-GPU/overlapping-port/excess-lane, degree-frontier, cost-summary, stale-output, source-registry, source-line, stale-root, same-GPU, stale-approval, plan-tamper, binary-plan controls rejected)"
