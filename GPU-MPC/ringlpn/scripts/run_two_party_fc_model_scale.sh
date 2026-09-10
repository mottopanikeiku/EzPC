#!/usr/bin/env bash
# Source-anchored Orca forward-linear workload matrix. The live path generates
# untruncated FC Beaver material only; model convolution and truncation remain
# explicit fail-closed gaps. qbits labels CRT limbs, not security.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(realpath "$ROOT/../..")"
BIN="$ROOT/bin/test_two_party_fc_preprocess"
OUTDIR="$ROOT/results/fc"
WORKDIR="${WORKDIR:-}"
PRIVATE_WORKDIR=0
if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ringlpn-fc-model-scale.XXXXXX")"
  PRIVATE_WORKDIR=1
fi
CHANNEL_AUTH_FILES=()
cleanup_private_workdir() {
  local rc=$?
  trap - EXIT
  trap '' INT TERM HUP
  local pid
  # GNU timeout owns each background party's process group.
  for pid in $(jobs -pr); do
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  local auth_file
  for auth_file in "${CHANNEL_AUTH_FILES[@]}"; do
    rm -f -- "$auth_file"
  done
  if (( PRIVATE_WORKDIR )); then rm -rf -- "$WORKDIR"; fi
  exit "$rc"
}
trap cleanup_private_workdir EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
LAYER_MANIFEST="${LAYER_MANIFEST:-$OUTDIR/orca_forward_linear_layer_manifest_2026_08_04.json}"
WORKLOAD_MANIFEST="${WORKLOAD_MANIFEST:-$OUTDIR/orca_model_scale_workload_manifest_2026_08_04.json}"
RESULT_SCHEMAS="$OUTDIR/two_party_fc_model_scale_result_schemas_2026_08_04.json"
CSV="${CSV:-$OUTDIR/two_party_fc_model_scale_2026_08_04.csv}"
AGGREGATE="${AGGREGATE:-$OUTDIR/two_party_fc_model_scale_aggregate_2026_08_04.csv}"
CONTROLS="${CONTROLS:-$OUTDIR/two_party_fc_model_scale_controls_2026_08_04.csv}"
SUMMARY="${SUMMARY:-$OUTDIR/two_party_fc_model_scale_summary_2026_08_04.csv}"
ENVIRONMENT="${ENVIRONMENT:-$OUTDIR/two_party_fc_model_scale_environment_2026_08_04.txt}"
LOG="${LOG:-$OUTDIR/two_party_fc_model_scale_2026_08_04.log}"
AB_AUDIT="${AB_AUDIT:-$OUTDIR/two_party_fc_model_scale_ab_audit_2026_08_14.csv}"
PLAN="$WORKDIR/execution_plan.tsv"
PLAN_META="$WORKDIR/execution_plan.json"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
CONTROLLED_SAME_GPU="${CONTROLLED_SAME_GPU:-0}"
BASE_PORT="${BASE_PORT:-24280}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
OT_BACKEND="${OT_BACKEND:-sci-iknp}"
RINGLPN_EMP_SILENT_BRIDGE="${RINGLPN_EMP_SILENT_BRIDGE:-}"
OT_ARGS=(--ot-backend "$OT_BACKEND")
case "$OT_BACKEND" in
  sci-iknp)
    if [[ -n "$RINGLPN_EMP_SILENT_BRIDGE" ]]; then
      echo "RINGLPN_EMP_SILENT_BRIDGE must be unset for OT_BACKEND=sci-iknp" >&2
      exit 2
    fi
    ;;
  emp-silent)
    if [[ "$RINGLPN_EMP_SILENT_BRIDGE" != /* ||
          ! -f "$RINGLPN_EMP_SILENT_BRIDGE" ]]; then
      echo "OT_BACKEND=emp-silent requires an existing absolute RINGLPN_EMP_SILENT_BRIDGE" >&2
      exit 2
    fi
    OT_ARGS+=(--emp-silent-bridge "$RINGLPN_EMP_SILENT_BRIDGE")
    ;;
  *)
    echo "OT_BACKEND must be sci-iknp or emp-silent" >&2
    exit 2
    ;;
esac
TRIALS="${TRIALS:-10}"
MODELS="${MODELS:-ResNet18}"
WORKLOAD="${WORKLOAD:-classifier}"
FAIL_LAYER="${FAIL_LAYER:-}"
SWAP_LAYER="${SWAP_LAYER:-}"
SCHEMA_VERSION="ringlpn.two-party-fc-model-scale.v6"
PUBLICATION_DATE="2026-08-10"
RESULT_COLUMNS=176
AB_AUDIT_SCHEMA="ringlpn.controlled-fc-ab.v1"

if [[ "$P0_GPU" == "$P1_GPU" ]]; then
  echo "P0_GPU and P1_GPU must be distinct" >&2
  exit 2
fi
if [[ "$CONTROLLED_SAME_GPU" != 0 && "$CONTROLLED_SAME_GPU" != 1 ]]; then
  echo "CONTROLLED_SAME_GPU must be 0 or 1" >&2
  exit 2
fi
if (( CONTROLLED_SAME_GPU )) &&
   [[ -n "$FAIL_LAYER" || -n "$SWAP_LAYER" ]]; then
  echo "controlled A/B measurements reject injected failure controls" >&2
  exit 2
fi
if ! [[ "$BASE_PORT" =~ ^[0-9]+$ && "$TIMEOUT_SECONDS" =~ ^[0-9]+$ &&
        "$TRIALS" =~ ^[0-9]+$ ]] ||
   (( BASE_PORT < 1 || BASE_PORT > 65534 || TIMEOUT_SECONDS < 1 ||
      TRIALS < 1 )); then
  echo "invalid BASE_PORT, TIMEOUT_SECONDS, or TRIALS" >&2
  exit 2
fi
mkdir -p "$OUTDIR"
outdir_real="$(realpath "$OUTDIR")"
declare -A seen_output_paths=()
for output_path in "$CSV" "$AGGREGATE" "$CONTROLS" "$SUMMARY" \
                   "$ENVIRONMENT" "$LOG" "$AB_AUDIT"; do
  output_real="$(realpath -m "$output_path")"
  if [[ "$output_real" != "$outdir_real/"* ]]; then
    echo "result outputs must remain children of $OUTDIR: $output_path" >&2
    exit 2
  fi
  if [[ -n "${seen_output_paths[$output_real]+present}" ]]; then
    echo "result output paths must be distinct: $output_path" >&2
    exit 2
  fi
  seen_output_paths["$output_real"]=1
done
if (( PRIVATE_WORKDIR )); then
  workdir_real="$(realpath -e "$WORKDIR")"
else
  if [[ -e "$WORKDIR" || -L "$WORKDIR" ]]; then
    echo "caller-supplied WORKDIR must be fresh; refusing to erase records or ledger" >&2
    exit 2
  fi
  workdir_real="$(realpath -m "$WORKDIR")"
  if [[ "$workdir_real" != "$outdir_real/"* ]]; then
    echo "caller-supplied WORKDIR must be a child of $OUTDIR" >&2
    exit 2
  fi
  mkdir -m 700 -- "$workdir_real"
fi
WORKDIR="$workdir_real"
LEDGER_ROOT="${LEDGER_ROOT:-$WORKDIR/ledger}"
ledger_real="$(realpath -m "$LEDGER_ROOT")"
if [[ "$ledger_real" != "$(realpath -ms "$LEDGER_ROOT")" ]]; then
  echo "LEDGER_ROOT must not contain symlink components" >&2
  exit 2
fi
LEDGER_ROOT="$ledger_real"
mkdir -p -m 700 -- "$LEDGER_ROOT"
ledger_mode="$(stat -c %a "$LEDGER_ROOT")"
if [[ ! -d "$LEDGER_ROOT" || -L "$LEDGER_ROOT" || ! -O "$LEDGER_ROOT" ]] ||
   (( (8#$ledger_mode & 077) != 0 )); then
  echo "LEDGER_ROOT must be an existing owner-only directory or a fresh path" >&2
  exit 2
fi
: > "$LOG"
require_quiescent_gpu() {
  local gpu="$1" boundary="$2" processes
  if ! processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>&1)"; then
    echo "cannot query GPU $gpu occupancy at $boundary: $processes" >&2
    return 1
  fi
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    echo "GPU $gpu is not quiescent at $boundary: $processes" >&2
    return 1
  fi
}

if (( CONTROLLED_SAME_GPU )); then
  LOCK_ROOT="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/ringlpn-gpu-locks-$UID}"
  mkdir -p -m 700 "$LOCK_ROOT"
  chmod 700 "$LOCK_ROOT"
  exec {P0_GPU_LOCK_FD}>"$LOCK_ROOT/gpu-$P0_GPU.lock"
  exec {P1_GPU_LOCK_FD}>"$LOCK_ROOT/gpu-$P1_GPU.lock"
  if ! flock -n "$P0_GPU_LOCK_FD" || ! flock -n "$P1_GPU_LOCK_FD"; then
    echo "another controlled Ring-LPN run holds a selected GPU lock" >&2
    exit 2
  fi
  require_quiescent_gpu "$P0_GPU" initial-preflight
  require_quiescent_gpu "$P1_GPU" initial-preflight
fi

# Validate every executable dimension, bit width, layout, constructor anchor,
# batch anchor, and pinned source digest before building or invoking a layer.
# Coverage rows expose every selected but unexecuted Conv2D/FC declaration.
python3 - "$REPO_ROOT" "$LAYER_MANIFEST" "$WORKLOAD_MANIFEST" "$RESULT_SCHEMAS" \
  "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$MODELS" "$WORKLOAD" "$PLAN" \
  "$PLAN_META" "$CSV" "$CONTROLS" <<'PY'
import csv
import hashlib
import json
import pathlib
import re
import sys

(repo_arg, layer_arg, workload_arg, schema_arg, schema_version, publication_date,
 models_arg, profile, plan_arg, meta_arg, csv_arg, controls_arg) = sys.argv[1:]
repo = pathlib.Path(repo_arg).resolve()
layer_path = pathlib.Path(layer_arg).resolve()
workload_path = pathlib.Path(workload_arg).resolve()
if profile not in {"classifier", "all-fc", "full-model"}:
    raise SystemExit("WORKLOAD must be classifier, all-fc, or full-model")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def anchored_text(anchor):
    match = re.fullmatch(r"([^:]+):(\d+)(?:-(\d+))?", anchor)
    if not match:
        raise ValueError(f"invalid source anchor {anchor!r}")
    path = (repo / match.group(1)).resolve()
    path.relative_to(repo)
    lines = path.read_text(encoding="utf-8").splitlines()
    first, last = int(match.group(2)), int(match.group(3) or match.group(2))
    if first < 1 or last < first or last > len(lines):
        raise ValueError(f"out-of-range source anchor {anchor!r}")
    return lines[first - 1:last]

layer_doc = json.loads(layer_path.read_text(encoding="utf-8"))
workload_doc = json.loads(workload_path.read_text(encoding="utf-8"))
layer_sha, workload_sha = digest(layer_path), digest(workload_path)
if workload_doc["layer_manifest"]["sha256"] != layer_sha:
    raise SystemExit("workload manifest does not pin the supplied layer manifest")
for source in layer_doc["source_registry"]:
    source_path = (repo / source["path"]).resolve()
    if not source_path.is_file() or digest(source_path) != source["sha256"]:
        raise SystemExit(f"source digest mismatch for {source['path']}")

layers = layer_doc["layers"]
available, seen = [], set()
last_order = (0, 0)
for row in layers:
    key = (row["model"], row["layer"])
    order = (int(row["model_order"]), int(row["linear_order"]))
    if key in seen or order <= last_order:
        raise SystemExit("manifest has a duplicate or out-of-order layer")
    seen.add(key)
    last_order = order
    if row["model"] not in available:
        available.append(row["model"])
    for field in ("bw", "layout", "source_anchor", "batch_source_anchor"):
        if row.get(field) in (None, ""):
            raise SystemExit(f"{key} lacks explicit {field}")
    if int(row["bw"]) < 3 or int(row["bw"]) > 32:
        raise SystemExit(f"{key} has an unsupported runner bw")
    source_lines = anchored_text(row["source_anchor"])
    if len(source_lines) != 1:
        raise SystemExit(f"layer anchor must identify one line: {key}")
    source_line = source_lines[0].strip()
    if hashlib.sha256(source_line.encode()).hexdigest() != row["source_text_sha256"]:
        raise SystemExit(f"source text digest mismatch for {key}")
    if str(row["batch"]) not in "\n".join(anchored_text(row["batch_source_anchor"])):
        raise SystemExit(f"batch {row['batch']} is not source-anchored for {key}")
    for value in row.values():
        if isinstance(value, str) and any(char in value for char in ",\t\r\n"):
            raise SystemExit(f"manifest CSV/TSV invariant violated for {key}")
    if row["operator"] == "fc":
        for field in ("rows", "inner", "cols", "qbits", "ole_n", "ole_c", "ole_t"):
            if not isinstance(row.get(field), int) or row[field] <= 0:
                raise SystemExit(f"{key} lacks positive {field}")
        constructor = re.search(r"new\s+FC<T>\((\d+)\s*,\s*(\d+)\s*,", source_line)
        if not constructor or (int(constructor.group(1)), int(constructor.group(2))) != (row["inner"], row["cols"]):
            raise SystemExit(f"FC dimensions do not match constructor for {key}")
        if row["rows"] != row["batch"] or row["matmul_batch"] != 1:
            raise SystemExit(f"FC batch/rows do not match stock MatmulParams for {key}")
        if row["ringlpn_status"] != "supported_untruncated":
            raise SystemExit(f"FC support label is not executable for {key}")
    elif row["operator"] != "conv2d":
        raise SystemExit(f"unknown linear operator for {key}")

if models_arg == "all":
    selected_models = available
else:
    selected_models = [item.strip() for item in models_arg.split(",") if item.strip()]
    if not selected_models or len(selected_models) != len(set(selected_models)):
        raise SystemExit("MODELS must be a nonempty unique list or all")
    unknown = [item for item in selected_models if item not in available]
    if unknown:
        raise SystemExit(f"unknown MODELS entries: {','.join(unknown)}")
selected = [row for row in layers if row["model"] in set(selected_models)]
def executable(row):
    return row["operator"] == "fc" and (profile != "classifier" or row["is_classifier"])

fields = [
    "model", "layer", "trial", "sample_role", "rows", "inner", "cols", "bw",
    "qbits", "noise", "ole_n", "ring_batches", "p0_ring_oles", "p1_ring_oles",
    "p0_dpf_trees", "p1_dpf_trees", "p0_public_a_seed_words", "p1_public_a_seed_words",
    "p0_protocol_bytes", "p1_protocol_bytes", "p0_total_us", "p1_total_us",
    "p0_record_bytes", "p1_record_bytes", "final_payload_bytes_per_party",
    "matched_dealer_keygen_us", "checker_two_share_online_us",
    "matched_dealer_keygen_contract", "key_order", "unchanged_online", "status",
    "schema_version", "publication_date", "manifest_sha256", "workload_manifest_sha256",
    "model_order", "source_layer", "linear_order", "forward_order", "operator",
    "source_anchor", "source_text_sha256", "batch_source_anchor", "batch", "layout",
    "ole_c", "ole_t", "workload", "retained", "support_status", "truncation_status",
    "gap", "stock_gpuKeygenMatmul_two_party_sequential_us",
    "unchanged_gpuMatmulBeaver_two_share_sequential_us", "p0_record_sha256",
    "p1_record_sha256", "p0_stdout_sha256", "p1_stdout_sha256", "checker_stdout_sha256",
    "ring_application_slots", "ring_bootstrap_slots",
    "p0_dpf_epoch_zero_scalar_oles", "p1_dpf_epoch_zero_scalar_oles",
    "p0_dpf_pcg_scalar_oles", "p1_dpf_pcg_scalar_oles",
    "p0_dpf_pcg_oles_reserved", "p1_dpf_pcg_oles_reserved",
    "p0_dpf_pcg_oles_discarded", "p1_dpf_pcg_oles_discarded",
    "p0_dpf_pcg_opening_words_sent", "p1_dpf_pcg_opening_words_sent",
    "p0_ring_application_slots_discarded",
    "p1_ring_application_slots_discarded",
]
party_metric_names = [
    "protocol_dependency_rounds", "preflight_us", "ot_setup_us", "dpf_phase_a_us",
    "dpf_phase_b_us", "dpf_phase_c_us", "spfss_grouping_us",
    "public_polynomial_exchange_us", "gpu_ringlpn_expansion_us",
    "derandomization_openings_us", "conversion_us", "serialization_us", "commit_us",
    "peak_host_rss_bytes", "peak_gpu_bytes", "min_gpu_free_bytes",
    "transport_straight_bytes_sent", "transport_straight_bytes_received",
    "transport_reversed_bytes_sent", "transport_reversed_bytes_received", "base_ots",
    "base_ot_setup_bytes_sent", "base_ot_setup_bytes_received",
    "transport_bytes_include_base_ot", "base_ot_setup_dependency_rounds",
]
for name in party_metric_names:
    fields.extend((f"p0_{name}", f"p1_{name}"))
fields.extend((
    "checker_us", "checker_peak_host_rss_bytes", "checker_peak_gpu_bytes",
    "checker_min_gpu_free_bytes", "invocation_id", "ledger_digest",
))
silent_ot_metric_names = [
    "ot_backend", "ot_backend_revision",
    "ot_correlation_straight_bytes_sent",
    "ot_correlation_straight_bytes_received",
    "ot_correlation_reversed_bytes_sent",
    "ot_correlation_reversed_bytes_received",
    "ot_adjustment_bytes_sent", "ot_adjustment_bytes_received",
    "ot_ciphertext_bytes_sent", "ot_ciphertext_bytes_received",
    "ot_inventory_straight_declared", "ot_inventory_straight_consumed",
    "ot_inventory_reversed_declared", "ot_inventory_reversed_consumed",
    "ot_backend_review_status",
]
for name in silent_ot_metric_names:
    fields.extend((f"p0_{name}", f"p1_{name}"))
extra_party_metric_names = [
    "channel_auth_straight_bytes_sent",
    "channel_auth_straight_bytes_received",
    "channel_auth_reversed_bytes_sent",
    "channel_auth_reversed_bytes_received",
    "ot_backend_bridge_sha256",
    "dpf_breadth_evaluator_calls",
    "dpf_root_to_leaf_evaluator_calls",
]
for name in extra_party_metric_names:
    fields.extend((f"p0_{name}", f"p1_{name}"))
fields.extend(("binary_sha256", "environment_sha256", "result_schema_sha256"))
schema_doc = json.loads(pathlib.Path(schema_arg).read_text(encoding="utf-8"))
schema_columns = schema_doc.get("per_layer", {}).get("columns")
if not isinstance(schema_columns, list) or any(
        not isinstance(column, str) or not column for column in schema_columns):
    raise SystemExit("result schema has no valid per_layer.columns list")
if len(schema_columns) != len(set(schema_columns)):
    raise SystemExit("result schema has duplicate per-layer columns")
if fields != schema_columns:
    raise SystemExit("runner raw header does not exactly match result schema")
if len(fields) != 176:
    raise SystemExit(f"runner raw header has {len(fields)} columns; expected 176")
with open(csv_arg, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in selected:
        if executable(row):
            continue
        result = {field: "" for field in fields}
        result.update({
            "model": row["model"], "layer": "classifier" if row["is_classifier"] else row["layer"],
            "trial": -1, "sample_role": "coverage", "rows": row["rows"],
            "inner": row["inner"], "cols": row["cols"], "bw": row["bw"],
            "qbits": row["qbits"], "noise": row["noise"], "ole_n": row["ole_n"],
            "status": "unsupported" if row["operator"] == "conv2d" else "not_selected",
            "schema_version": schema_version,
            "publication_date": publication_date, "manifest_sha256": layer_sha,
            "workload_manifest_sha256": workload_sha, "model_order": row["model_order"],
            "source_layer": row["layer"], "linear_order": row["linear_order"],
            "forward_order": row["forward_order"], "operator": row["operator"],
            "source_anchor": row["source_anchor"], "source_text_sha256": row["source_text_sha256"],
            "batch_source_anchor": row["batch_source_anchor"], "batch": row["batch"],
            "layout": row["layout"], "ole_c": row["ole_c"], "ole_t": row["ole_t"],
            "workload": profile, "retained": "no", "support_status": row["ringlpn_status"],
            "truncation_status": row["truncation_status"], "gap": row["gap"],
        })
        writer.writerow(result)

plan_rows = [row for row in selected if executable(row)]
with open(plan_arg, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    for row in plan_rows:
        writer.writerow([
            row["model"], row["model_order"], row["layer"], row["linear_order"],
            row["forward_order"], row["source_anchor"], row["source_text_sha256"],
            row["batch_source_anchor"], row["batch"], row["rows"], row["inner"],
            row["cols"], row["bw"], row["layout"], row["qbits"], row["noise"],
            row["ole_n"], row["ole_c"], row["ole_t"], row["truncation_status"],
            row["gap"], "yes" if row["is_classifier"] else "no",
        ])
metadata = {
    "schema_version": schema_version, "publication_date": publication_date,
    "manifest_sha256": layer_sha, "workload_manifest_sha256": workload_sha,
    "workload": profile, "models": [],
}
for model in selected_models:
    model_rows = [row for row in selected if row["model"] == model]
    metadata["models"].append({
        "model": model, "model_order": model_rows[0]["model_order"],
        "expected_executable_layers": sum(executable(row) for row in model_rows),
        "unsupported_convolution_layers": sum(row["operator"] == "conv2d" for row in model_rows),
        "unsupported_truncation_layers": sum(row["truncation_status"] != "supported" for row in model_rows),
    })
pathlib.Path(meta_arg).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
with open(controls_arg, "w", newline="", encoding="utf-8") as handle:
    csv.writer(handle, lineterminator="\n").writerow([
        "schema_version", "publication_date", "model", "source_layer", "trial",
        "sample_role", "control", "expected", "observed", "artifact_sha256", "status",
    ])
PY
printf '%s\n' \
  'schema_version,publication_date,model,source_layer,trial,sample_role,invocation_id,party0_gpu,party1_gpu,critical_party,party0_setup_included_us,party1_setup_included_us,preprocess_gpu,checker_gpu,same_physical_gpu,pre_protocol_quiescent,pre_checker_quiescent,post_checker_quiescent,comparison_order,status' \
  > "$AB_AUDIT"

manifest_sha256="$(sha256sum "$LAYER_MANIFEST" | cut -d' ' -f1)"
workload_manifest_sha256="$(sha256sum "$WORKLOAD_MANIFEST" | cut -d' ' -f1)"
validate_control_selector() {
  local selector="$1" name="$2"
  [[ -z "$selector" || "$selector" == all ]] && return 0
  if ! awk -F '\t' -v target="$selector" '$1 ":" $3 == target { found=1 } END { exit !found }' "$PLAN"; then
    echo "$name must be empty, all, or exact Model:source_layer" >&2
    exit 2
  fi
}
validate_control_selector "$FAIL_LAYER" FAIL_LAYER
validate_control_selector "$SWAP_LAYER" SWAP_LAYER

{
  echo "[two-party-fc-model] build"
  "$ROOT/scripts/build_two_party_fc_preprocess.sh"
} >> "$LOG" 2>&1

append_result_row() {
  local -a row=("$@")
  if (( ${#row[@]} != RESULT_COLUMNS )); then
    echo "internal result schema mismatch: got ${#row[@]} expected $RESULT_COLUMNS" >&2
    return 1
  fi
  local IFS=,
  printf '%s\n' "${row[*]}" >> "$CSV"
}
append_control_row() {
  local -a row=("$@")
  local IFS=,
  printf '%s\n' "${row[*]}" >> "$CONTROLS"
}
append_ab_audit_row() {
  local -a row=("$@")
  if (( ${#row[@]} != 20 )); then
    echo "internal controlled A/B audit schema mismatch" >&2
    return 1
  fi
  local IFS=,
  printf '%s\n' "${row[*]}" >> "$AB_AUDIT"
}
matches_control() {
  [[ "$1" == all || "$1" == "$2:$3" ]]
}
file_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}
public_sha256() {
  local path="$1"
  if [[ "$path" != "$REPO_ROOT/"* ]]; then
    echo "Refusing host-identifying non-repository provenance path: $path" >&2
    return 2
  fi
  printf '%s  %s\n' "$(file_sha256 "$path")" "${path#"$REPO_ROOT/"}"
}
BINARY_SHA256="$(file_sha256 "$BIN")"
{
  echo "measurement_timestamp=$(date --iso-8601=seconds)"
  echo "publication_date=$PUBLICATION_DATE"
  echo "schema_version=$SCHEMA_VERSION"
  echo "claim_scope=internal/advisor feasibility matrix; qbits is a CRT construction label, not a security level"
  echo "provenance_sanitization=public derivative; hostname, GPU UUIDs, and absolute workstation paths are intentionally omitted"
  echo "host=withheld_for_publication"
  echo "kernel=$(uname -srvmo)"
  echo "cpu_count=$(nproc)"
  if (( CONTROLLED_SAME_GPU )); then
    echo "process_gpu_map=party0:$P0_GPU,party1:$P1_GPU,checker:per-sample-critical-party"
    echo "controlled_same_gpu=1"
    echo "critical_party_rule=max(total_us+preflight_us+ot_setup_us); ties select party0"
    echo "comparison_order=dealer baseline after both preprocessing parties exit"
    echo "gpu_quiescence_contract=selected GPUs empty before protocol and after both parties exit; selected critical-party GPU empty after checker; advisory per-GPU locks held for full run"
  else
    echo "process_gpu_map=party0:$P0_GPU,party1:$P1_GPU,checker:$CHECK_GPU"
    echo "controlled_same_gpu=0"
  fi
  echo "network=single-host IPv4 loopback"
  echo "counters=legacy protocol bytes exclude preflight/OT setup; transport stream bytes include selected-backend setup, exclude TCP framing, and add no metrics message"
  echo "ot_backend=$OT_BACKEND"
  if [[ "$OT_BACKEND" == emp-silent ]]; then
    echo "emp_silent_bridge=external-input-basename:$(basename "$RINGLPN_EMP_SILENT_BRIDGE")"
  else
    echo "emp_silent_bridge=NA"
  fi
  echo "warmups=1"
  echo "measured_trials=$TRIALS"
  echo "models=$MODELS"
  echo "workload=$WORKLOAD"
  echo "fail_layer_control=${FAIL_LAYER:-none}"
  echo "swap_layer_control=${SWAP_LAYER:-none}"
  echo "aggregate_rule=complete retained layer groups only; statistics are per model over measured aggregate rows"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version
  for artifact in \
    "$BIN" "$LAYER_MANIFEST" "$WORKLOAD_MANIFEST" "$RESULT_SCHEMAS" \
    "$ROOT/scripts/aggregate_two_party_fc_model_scale.py" \
    "$ROOT/scripts/verify_controlled_fc_ab.py" \
    "$ROOT/scripts/two_party_fc_metrics_schema_2026_08_04.csv" \
    "$ROOT/scripts/run_two_party_fc_model_scale.sh" \
    "$ROOT/src/test_two_party_fc_preprocess.cu" "$ROOT/src/two_party_spfss.h" \
    "$ROOT/src/linear_preprocess.h" \
    "$ROOT/src/linear_preprocess_backend.cuh" \
    "$ROOT/src/linear_preprocess_fc.cu" \
    "$ROOT/src/two_party_linear_preprocess.cuh" \
    "$ROOT/src/two_party_spfss_gpu.cuh" "$ROOT/src/two_party_dpf_protocol.h" \
    "$ROOT/src/two_party_dpf_gpu.cuh" "$ROOT/src/two_party_ot.h" \
    "$ROOT/src/emp_silent_adapter.h" "$ROOT/src/emp_silent_bridge.h" \
    "$ROOT/src/emp_silent_bridge_authorization.h" \
    "$ROOT/src/emp_silent_bridge.cpp" "$ROOT/src/ringlpn_ole_party.cuh" \
    "$ROOT/src/secure_convert.h" \
    "$REPO_ROOT/GPU-MPC/experiments/orca/cnn.h" \
    "$REPO_ROOT/GPU-MPC/experiments/orca/orca_inference.cu" \
    "$REPO_ROOT/GPU-MPC/experiments/orca/piranha.cu" \
    "$REPO_ROOT/GPU-MPC/nn/orca/fc_layer.cu"; do
    public_sha256 "$artifact"
  done
  if [[ "$OT_BACKEND" == emp-silent ]]; then
    printf '%s  %s\n' "$(file_sha256 "$RINGLPN_EMP_SILENT_BRIDGE")" \
      "external-input-basename:$(basename "$RINGLPN_EMP_SILENT_BRIDGE")"
  fi
} > "$ENVIRONMENT"
RESULT_SCHEMA_SHA256="$(file_sha256 "$RESULT_SCHEMAS")"
ENVIRONMENT_SHA256="$(file_sha256 "$ENVIRONMENT")"
python3 - "$CSV" "$BINARY_SHA256" "$ENVIRONMENT_SHA256" "$RESULT_SCHEMA_SHA256" <<'PY'
import csv
import os
import pathlib
import sys
import tempfile

csv_path = pathlib.Path(sys.argv[1])
identity = dict(zip(
    ("binary_sha256", "environment_sha256", "result_schema_sha256"),
    sys.argv[2:],
))
with csv_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)
if fields is None or any(field not in fields for field in identity):
    raise SystemExit("raw header lacks capture identity columns")
for row in rows:
    if row["sample_role"] != "coverage" or any(row[field] for field in identity):
        raise SystemExit("unexpected pre-measurement raw row")
    row.update(identity)
temporary = tempfile.NamedTemporaryFile(
    mode="w", newline="", encoding="utf-8", dir=csv_path.parent,
    prefix=f".{csv_path.name}.", delete=False)
try:
    with temporary:
        writer = csv.DictWriter(temporary, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.chmod(temporary.name, 0o600)
    os.replace(temporary.name, csv_path)
    directory_fd = os.open(csv_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    try:
        os.unlink(temporary.name)
    except FileNotFoundError:
        pass
    raise
PY
EMPTY_METRICS=()
for ((metric_index = 0; metric_index < 100; ++metric_index)); do
  EMPTY_METRICS+=("")
done

had_failure=0
run_sample() {
  local model="$1" model_order="$2" source_layer="$3" linear_order="$4"
  local forward_order="$5" source_anchor="$6" source_text_sha256="$7"
  local batch_source_anchor="$8" batch="$9" rows="${10}" inner="${11}"
  local cols="${12}" bw="${13}" layout="${14}" qbits="${15}" noise="${16}"
  local ole_n="${17}" ole_c="${18}" ole_t="${19}" truncation_status="${20}"
  local gap="${21}" is_classifier="${22}" trial="${23}" role="${24}"
  if [[ "$(file_sha256 "$BIN")" != "$BINARY_SHA256" ]]; then
    echo "[two-party-fc-model] binary changed after the measured build" >&2
    return 2
  fi
  local layer_label="$source_layer"
  [[ "$is_classifier" == yes ]] && layer_label=classifier
  local bootstrap_slots=$((3 * (ole_c * ole_t) * (ole_c * ole_t)))
  local application_slots=$((ole_n - bootstrap_slots))
  if (( application_slots <= 0 )); then
    echo "[two-party-fc-model] Ring-LPN instance cannot self-bootstrap" >&2
    return 2
  fi
  local ring_batches=$(( (rows * inner * cols + application_slots - 1) / application_slots ))
  local safe_model="${model//[^A-Za-z0-9_-]/_}"
  local safe_layer="${source_layer//[^A-Za-z0-9_-]/_}"
  local dir="$WORKDIR/${safe_model}_${safe_layer}/${role}_${trial}"
  local sid invocation_id
  sid="$(python3 -c 'import secrets; print(secrets.randbelow((1 << 63) - 1) + 1)')"
  invocation_id="$(openssl rand -hex 16)"
  local p0_record_sha="" p1_record_sha="" p0_stdout_sha="" p1_stdout_sha="" checker_stdout_sha=""
  mkdir -p "$dir/party0" "$dir/party1"

  append_failed() {
    local failure_status="$1" support_status="$2"
    [[ -f "$dir/p0.out" ]] && p0_stdout_sha="$(file_sha256 "$dir/p0.out")"
    [[ -f "$dir/p1.out" ]] && p1_stdout_sha="$(file_sha256 "$dir/p1.out")"
    [[ -f "$dir/check.out" ]] && checker_stdout_sha="$(file_sha256 "$dir/check.out")"
    append_result_row \
      "$model" "$layer_label" "$trial" "$role" "$rows" "$inner" "$cols" "$bw" \
      "$qbits" "$noise" "$ole_n" "$ring_batches" "" "" "" "" "" "" "" "" \
      "" "" "" "" "" "" "" "" "" "" "$failure_status" \
      "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$manifest_sha256" "$workload_manifest_sha256" \
      "$model_order" "$source_layer" "$linear_order" "$forward_order" fc "$source_anchor" \
      "$source_text_sha256" "$batch_source_anchor" "$batch" "$layout" "$ole_c" "$ole_t" \
      "$WORKLOAD" no "$support_status" "$truncation_status" "$gap" "" "" \
      "$p0_record_sha" "$p1_record_sha" "$p0_stdout_sha" "$p1_stdout_sha" "$checker_stdout_sha" \
      "" "" "" "" "" "" "" "" "" "" "" "" "" "" "${EMPTY_METRICS[@]}" \
      "$BINARY_SHA256" "$ENVIRONMENT_SHA256" "$RESULT_SCHEMA_SHA256"
  }

  if matches_control "$FAIL_LAYER" "$model" "$source_layer"; then
    append_control_row "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$model" "$source_layer" \
      "$trial" "$role" injected_layer_failure fail_closed rejected "" pass
    append_failed FAIL injected_failure
    had_failure=1
    return 0
  fi

  local p0_prefix="$dir/party0/key" p1_prefix="$dir/party1/key"
  local p0_record="${p0_prefix}_p0.fc" p1_record="${p1_prefix}_p1.fc"
  local p0_auth="$dir/party0/channel-auth.key"
  local p1_auth="$dir/party1/channel-auth.key"
  CHANNEL_AUTH_FILES+=("$p0_auth" "$p1_auth")
  openssl rand 32 > "$p0_auth"
  cp -- "$p0_auth" "$p1_auth"
  chmod 600 "$p0_auth" "$p1_auth"
  local -a common=(--host 127.0.0.1 --port "$BASE_PORT" --sid "$sid"
    --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
    --qbits "$qbits" --bw "$bw" --rows "$rows" --inner "$inner" --cols "$cols"
    --ole-n "$ole_n" --ole-c "$ole_c" --ole-t "$ole_t" --noise "$noise"
    "${OT_ARGS[@]}")
  local rc0 rc1 check_rc swap_rc pid0 pid1
  if (( CONTROLLED_SAME_GPU )); then
    require_quiescent_gpu "$P0_GPU" "$model:$source_layer:$role:$trial:pre-protocol"
    require_quiescent_gpu "$P1_GPU" "$model:$source_layer:$role:$trial:pre-protocol"
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --party 0 \
    --channel-auth-file "$p0_auth" \
    "${common[@]}" --out-prefix "$p0_prefix" > "$dir/p0.out" 2>&1 &
  pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --party 1 \
    --channel-auth-file "$p1_auth" \
    "${common[@]}" --out-prefix "$p1_prefix" > "$dir/p1.out" 2>&1 &
  pid1=$!
  wait "$pid0"; rc0=$?
  wait "$pid1"; rc1=$?
  set -e
  {
    echo "===== $model $source_layer / $role $trial / party 0 ====="
    cat "$dir/p0.out"
    echo "===== $model $source_layer / $role $trial / party 1 ====="
    cat "$dir/p1.out"
  } >> "$LOG"
  if (( CONTROLLED_SAME_GPU )); then
    require_quiescent_gpu "$P0_GPU" "$model:$source_layer:$role:$trial:pre-checker"
    require_quiescent_gpu "$P1_GPU" "$model:$source_layer:$role:$trial:pre-checker"
  fi
  if (( rc0 != 0 || rc1 != 0 )) || [[ ! -f "$p0_record" || ! -f "$p1_record" ]]; then
    rm -f "$p0_record" "$p1_record" "${p0_record}.tmp" "${p1_record}.tmp"
    append_failed FAIL supported_untruncated
    echo "[two-party-fc-model] $model:$source_layer preprocessing failed: p0=$rc0 p1=$rc1" >&2
    had_failure=1
    return 0
  fi
  if [[ -e "${p0_prefix}_p1.fc" || -e "${p1_prefix}_p0.fc" ]]; then
    rm -f "$p0_record" "$p1_record"
    append_failed FAIL supported_untruncated
    echo "[two-party-fc-model] party-local output ownership violation" >&2
    had_failure=1
    return 0
  fi
  local p0_row p1_row check_row
  local -a f0 f1 fc
  p0_row="$(sed -n '/^0,/p' "$dir/p0.out" | tail -n 1)"
  p1_row="$(sed -n '/^1,/p' "$dir/p1.out" | tail -n 1)"
  IFS=',' read -r -a f0 <<< "$p0_row"
  IFS=',' read -r -a f1 <<< "$p1_row"
  local sample_check_gpu="$CHECK_GPU" critical_party=NA
  local p0_setup_included_us=NA p1_setup_included_us=NA
  if (( CONTROLLED_SAME_GPU )); then
    if [[ "${#f0[@]}" -ne 86 || "${#f1[@]}" -ne 86 ||
          "${f0[35]}" != pass || "${f1[35]}" != pass ]]; then
      echo "cannot select the critical-party GPU from malformed party output" >&2
      return 2
    fi
    read -r critical_party p0_setup_included_us p1_setup_included_us < <(
      python3 - "${f0[34]}" "${f0[37]}" "${f0[38]}" \
                  "${f1[34]}" "${f1[37]}" "${f1[38]}" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

try:
    values = [Decimal(value) for value in sys.argv[1:]]
except InvalidOperation as error:
    raise SystemExit(f"invalid critical-path timing: {error}")
if any(not value.is_finite() or value < 0 for value in values):
    raise SystemExit("critical-path timings must be finite and nonnegative")
party0 = sum(values[:3])
party1 = sum(values[3:])
print(0 if party0 >= party1 else 1, party0, party1)
PY
    )
    if [[ "$critical_party" == 0 ]]; then
      sample_check_gpu="$P0_GPU"
    elif [[ "$critical_party" == 1 ]]; then
      sample_check_gpu="$P1_GPU"
    else
      echo "critical-party selector returned an invalid party" >&2
      return 2
    fi
  fi

  local p0_record_bytes p1_record_bytes
  p0_record_bytes="$(stat -c %s "$p0_record")"
  p1_record_bytes="$(stat -c %s "$p1_record")"
  p0_record_sha="$(file_sha256 "$p0_record")"
  p1_record_sha="$(file_sha256 "$p1_record")"
  if matches_control "$SWAP_LAYER" "$model" "$source_layer"; then
    set +e
    CUDA_VISIBLE_DEVICES="$sample_check_gpu" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --check \
      --p0-record "$p1_record" --p1-record "$p0_record" > "$dir/check_swapped.out" 2>&1
    swap_rc=$?
    set -e
    local swap_sha
    swap_sha="$(file_sha256 "$dir/check_swapped.out")"
    if (( swap_rc == 0 )); then
      append_control_row "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$model" "$source_layer" \
        "$trial" "$role" swapped_party_records reject accepted "$swap_sha" FAIL
      rm -f "$p0_record" "$p1_record"
      append_failed FAIL supported_untruncated
      had_failure=1
      return 0
    fi
    append_control_row "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$model" "$source_layer" \
      "$trial" "$role" swapped_party_records reject rejected "$swap_sha" pass
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$sample_check_gpu" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --check \
    --p0-record "$p0_record" --p1-record "$p1_record" > "$dir/check.out" 2>&1
  check_rc=$?
  set -e
  if (( CONTROLLED_SAME_GPU )); then
    require_quiescent_gpu "$sample_check_gpu" \
      "$model:$source_layer:$role:$trial:post-checker"
  fi
  {
    echo "===== $model $source_layer / $role $trial / post-exit checker ====="
    cat "$dir/check.out"
  } >> "$LOG"
  if (( check_rc != 0 )); then
    rm -f "$p0_record" "$p1_record"
    append_failed FAIL supported_untruncated
    had_failure=1
    return 0
  fi

  check_row="$(sed -n "/^${qbits},/p" "$dir/check.out" | tail -n 1)"
  IFS=',' read -r -a f0 <<< "$p0_row"
  IFS=',' read -r -a f1 <<< "$p1_row"
  IFS=',' read -r -a fc <<< "$check_row"
  local limbs=1
  (( qbits == 128 )) && limbs=2
  local expected_application_discarded=$((
    ring_batches * 2 * limbs * application_slots -
    2 * limbs * rows * inner * cols
  ))
  if [[ "${#f0[@]}" -ne 86 || "${#f1[@]}" -ne 86 || "${#fc[@]}" -ne 19 ||
        "${f0[0]}" -ne 0 || "${f1[0]}" -ne 1 ||
        "${f0[1]}" -ne "$qbits" || "${f1[1]}" -ne "$qbits" ||
        "${f0[2]}" -ne "$bw" || "${f1[2]}" -ne "$bw" ||
        "${f0[3]}" -ne "$rows" || "${f1[3]}" -ne "$rows" ||
        "${f0[4]}" -ne "$inner" || "${f1[4]}" -ne "$inner" ||
        "${f0[5]}" -ne "$cols" || "${f1[5]}" -ne "$cols" ||
        "${f0[6]}" != "$ole_n" || "${f1[6]}" != "$ole_n" ||
        "${f0[7]}" != "$ole_c" || "${f1[7]}" != "$ole_c" ||
        "${f0[8]}" != "$ole_t" || "${f1[8]}" != "$ole_t" ||
        "${f0[9]}" != "$noise" || "${f1[9]}" != "$noise" ||
        "${f0[10]}" -ne "$ring_batches" || "${f1[10]}" -ne "$ring_batches" ||
        "${f0[11]}" -ne "$application_slots" ||
        "${f1[11]}" -ne "$application_slots" ||
        "${f0[12]}" -ne "$bootstrap_slots" ||
        "${f1[12]}" -ne "$bootstrap_slots" ||
        "${f0[13]}" -ne "${f1[13]}" || "${f0[15]}" -ne "${f1[15]}" ||
        "${f0[27]}" -ne "${f1[27]}" || "${f0[35]}" != pass || "${f1[35]}" != pass ||
        "${fc[0]}" -ne "$qbits" || "${fc[1]}" -ne "$bw" ||
        "${fc[2]}" -ne "$rows" || "${fc[3]}" -ne "$inner" ||
        "${fc[4]}" -ne "$cols" || "${fc[5]}" -ne "$ring_batches" ||
        "${fc[9]}" != pass || "${fc[10]}" != pass ||
        "${fc[11]}" != pass || "${fc[12]}" != pass ||
        "${f0[64]}" != NA || "${f1[64]}" != NA ||
        "${f0[65]}" != "$invocation_id" || "${f1[65]}" != "$invocation_id" ||
        "${fc[17]}" != "$invocation_id" ||
        "${f0[66]}" != "${f1[66]}" || "${f0[66]}" != "${fc[18]}" ||
        ("${f0[67]}" != sci-iknp && "${f0[67]}" != emp-silent) ||
        "${f0[67]}" != "${f1[67]}" ||
        -z "${f0[68]}" || "${f0[68]}" != "${f1[68]}" ||
        ("${f0[67]}" == sci-iknp &&
         ("${f0[63]}" != yes || "${f1[63]}" != yes ||
          "${f0[83]}" != NA || "${f1[83]}" != NA)) ||
        ("${f0[67]}" == emp-silent &&
         ("${f0[56]}" != NA || "${f1[56]}" != NA ||
          "${f0[57]}" != NA || "${f1[57]}" != NA ||
          "${f0[58]}" != NA || "${f1[58]}" != NA ||
          "${f0[63]}" != NA || "${f1[63]}" != NA ||
          -z "${f0[83]}" || "${f0[83]}" != "${f1[83]}")) ||
        "${f0[70]}" != NA || "${f1[70]}" != NA ||
        "${f0[72]}" != NA || "${f1[72]}" != NA ||
        "${f0[74]}" != NA || "${f1[74]}" != NA ||
        "${f0[76]}" != NA || "${f1[76]}" != NA ||
        -z "${f0[81]}" || "${f0[81]}" != "${f1[81]}" ||
        "${f0[82]}" -ne "$expected_application_discarded" ||
        "${f1[82]}" -ne "$expected_application_discarded" ||
        ("${f0[84]}" -eq 0 && "${f0[85]}" -eq 0) ||
        ("${f1[84]}" -eq 0 && "${f1[85]}" -eq 0) ]]; then
    rm -f "$p0_record" "$p1_record"
    append_failed FAIL supported_untruncated
    had_failure=1
    return 0
  fi

  p0_stdout_sha="$(file_sha256 "$dir/p0.out")"
  p1_stdout_sha="$(file_sha256 "$dir/p1.out")"
  checker_stdout_sha="$(file_sha256 "$dir/check.out")"
  local -a raw_metrics=()
  for ((metric_index = 36; metric_index <= 58; ++metric_index)); do
    case "$metric_index" in
      53) raw_metrics+=("${f1[52]}" "${f0[52]}") ;;
      55) raw_metrics+=("${f1[54]}" "${f0[54]}") ;;
      58) raw_metrics+=("${f1[57]}" "${f0[57]}") ;;
      *) raw_metrics+=("${f0[metric_index]}" "${f1[metric_index]}") ;;
    esac
  done
  raw_metrics+=("${f0[63]}" "${f1[63]}" "${f0[64]}" "${f1[64]}")
  for ((metric_index = 13; metric_index <= 16; ++metric_index)); do
    raw_metrics+=("${fc[metric_index]}")
  done
  raw_metrics+=("$invocation_id" "${f0[66]}")
  for ((metric_index = 67; metric_index <= 81; ++metric_index)); do
    case "$metric_index" in
      70) raw_metrics+=("${f1[69]}" "${f0[69]}") ;;
      72) raw_metrics+=("${f1[71]}" "${f0[71]}") ;;
      74) raw_metrics+=("${f1[73]}" "${f0[73]}") ;;
      76) raw_metrics+=("${f1[75]}" "${f0[75]}") ;;
      *) raw_metrics+=("${f0[metric_index]}" "${f1[metric_index]}") ;;
    esac
  done
  for metric_index in 59 60 61 62 83 84 85; do
    raw_metrics+=("${f0[metric_index]}" "${f1[metric_index]}")
  done
  append_result_row \
    "$model" "$layer_label" "$trial" "$role" "$rows" "$inner" "$cols" "$bw" \
    "$qbits" "$noise" "$ole_n" "$ring_batches" "${f0[13]}" "${f1[13]}" \
    "${f0[15]}" "${f1[15]}" "${f0[27]}" "${f1[27]}" "${f0[32]}" "${f1[32]}" \
    "${f0[34]}" "${f1[34]}" "$p0_record_bytes" "$p1_record_bytes" "${fc[6]}" \
    "${fc[7]}" "${fc[8]}" "${fc[9]}" "${fc[10]}" "${fc[11]}" pass \
    "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$manifest_sha256" "$workload_manifest_sha256" \
    "$model_order" "$source_layer" "$linear_order" "$forward_order" fc "$source_anchor" \
    "$source_text_sha256" "$batch_source_anchor" "$batch" "$layout" "$ole_c" "$ole_t" \
    "$WORKLOAD" yes supported_untruncated "$truncation_status" "$gap" "${fc[7]}" "${fc[8]}" \
    "$p0_record_sha" "$p1_record_sha" "$p0_stdout_sha" "$p1_stdout_sha" "$checker_stdout_sha" \
    "$application_slots" "$bootstrap_slots" "${f0[19]}" "${f1[19]}" \
    "${f0[20]}" "${f1[20]}" "${f0[21]}" "${f1[21]}" "${f0[22]}" "${f1[22]}" \
    "${f0[23]}" "${f1[23]}" "${f0[82]}" "${f1[82]}" "${raw_metrics[@]}" \
    "$BINARY_SHA256" "$ENVIRONMENT_SHA256" "$RESULT_SCHEMA_SHA256"
  if (( CONTROLLED_SAME_GPU )); then
    append_ab_audit_row \
      "$AB_AUDIT_SCHEMA" "$PUBLICATION_DATE" "$model" "$source_layer" "$trial" \
      "$role" "$invocation_id" "$P0_GPU" "$P1_GPU" "$critical_party" \
      "$p0_setup_included_us" "$p1_setup_included_us" "$sample_check_gpu" \
      "$sample_check_gpu" yes yes yes yes protocol_then_dealer pass
  fi
  printf 'validated_after_both_party_exits sid=%s invocation_id=%s ledger_digest=%s\n' \
    "$sid" "$invocation_id" "${f0[66]}" > "$dir/COMMITTED"
  rm -f "$p0_record" "$p1_record"
  echo "[two-party-fc-model] $model:$source_layer $role $trial pass"
}

while IFS=$'\t' read -r model model_order source_layer linear_order forward_order \
  source_anchor source_text_sha256 batch_source_anchor batch rows inner cols bw \
  layout qbits noise ole_n ole_c ole_t truncation_status gap is_classifier; do
  run_sample "$model" "$model_order" "$source_layer" "$linear_order" "$forward_order" \
    "$source_anchor" "$source_text_sha256" "$batch_source_anchor" "$batch" "$rows" \
    "$inner" "$cols" "$bw" "$layout" "$qbits" "$noise" "$ole_n" "$ole_c" \
    "$ole_t" "$truncation_status" "$gap" "$is_classifier" 0 warmup
  for ((trial = 1; trial <= TRIALS; ++trial)); do
    run_sample "$model" "$model_order" "$source_layer" "$linear_order" "$forward_order" \
      "$source_anchor" "$source_text_sha256" "$batch_source_anchor" "$batch" "$rows" \
      "$inner" "$cols" "$bw" "$layout" "$qbits" "$noise" "$ole_n" "$ole_c" \
      "$ole_t" "$truncation_status" "$gap" "$is_classifier" "$trial" measured
  done
done < "$PLAN"
if [[ "$(file_sha256 "$BIN")" != "$BINARY_SHA256" ]]; then
  echo "[two-party-fc-model] binary changed during the repeated-run workflow" >&2
  exit 2
fi


python3 "$ROOT/scripts/aggregate_two_party_fc_model_scale.py" \
  --source-csv "$CSV" \
  --plan-metadata "$PLAN_META" \
  --layer-manifest "$LAYER_MANIFEST" \
  --workload-manifest "$WORKLOAD_MANIFEST" \
  --aggregate-csv "$AGGREGATE" \
  --statistics-csv "$SUMMARY" \
  --trials "$TRIALS" \
  --binary "$BIN" \
  --require-current-binary \
  --environment "$ENVIRONMENT" \
  --result-schema "$RESULT_SCHEMAS"
if (( CONTROLLED_SAME_GPU )); then
  python3 "$ROOT/scripts/verify_controlled_fc_ab.py" \
    --raw "$CSV" --audit "$AB_AUDIT" --environment "$ENVIRONMENT"
fi

aggregate_failed="$(python3 -c 'import csv, sys; rows = csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")); print(int(any(row.get("status") != "pass" for row in rows)))' "$AGGREGATE")"
if (( had_failure != 0 || aggregate_failed != 0 )); then
  echo "[two-party-fc-model] workload failed closed; inspect $CSV and $AGGREGATE" >&2
  exit 1
fi

echo "[two-party-fc-model] selected workload pass; full_model_status remains independently fail-closed"
echo "[two-party-fc-model] per-layer results: $CSV"
echo "[two-party-fc-model] aggregate results: $AGGREGATE"
echo "[two-party-fc-model] controls: $CONTROLS"
echo "[two-party-fc-model] summary: $SUMMARY"
echo "[two-party-fc-model] environment: $ENVIRONMENT"
if (( CONTROLLED_SAME_GPU )); then
  echo "[two-party-fc-model] controlled A/B audit: $AB_AUDIT"
fi
echo "[two-party-fc-model] log: $LOG"
