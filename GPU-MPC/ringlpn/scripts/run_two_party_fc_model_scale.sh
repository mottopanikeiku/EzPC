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
WORKDIR="${WORKDIR:-$OUTDIR/two_party_fc_model_scale_work_2026_08_04}"
LAYER_MANIFEST="${LAYER_MANIFEST:-$OUTDIR/orca_forward_linear_layer_manifest_2026_08_04.json}"
WORKLOAD_MANIFEST="${WORKLOAD_MANIFEST:-$OUTDIR/orca_model_scale_workload_manifest_2026_08_04.json}"
RESULT_SCHEMAS="$OUTDIR/two_party_fc_model_scale_result_schemas_2026_08_04.json"
CSV="$OUTDIR/two_party_fc_model_scale_2026_08_04.csv"
AGGREGATE="$OUTDIR/two_party_fc_model_scale_aggregate_2026_08_04.csv"
CONTROLS="$OUTDIR/two_party_fc_model_scale_controls_2026_08_04.csv"
SUMMARY="$OUTDIR/two_party_fc_model_scale_summary_2026_08_04.csv"
ENVIRONMENT="$OUTDIR/two_party_fc_model_scale_environment_2026_08_04.txt"
LOG="$OUTDIR/two_party_fc_model_scale_2026_08_04.log"
PLAN="$WORKDIR/execution_plan.tsv"
PLAN_META="$WORKDIR/execution_plan.json"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
BASE_PORT="${BASE_PORT:-48280}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
LEDGER_ROOT="${LEDGER_ROOT:-$ROOT/results/deployment/correlation-ledger/party-claims}"
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
SCHEMA_VERSION="ringlpn.two-party-fc-model-scale.v3"
PUBLICATION_DATE="2026-08-04"
RESULT_COLUMNS=145

if [[ "$P0_GPU" == "$P1_GPU" ]]; then
  echo "P0_GPU and P1_GPU must be distinct" >&2
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
workdir_real="$(realpath -m "$WORKDIR")"
if [[ "$workdir_real" != "$outdir_real/"* ]]; then
  echo "WORKDIR must be a child of $OUTDIR" >&2
  exit 2
fi
WORKDIR="$workdir_real"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
LEDGER_ROOT="$(realpath -m "$LEDGER_ROOT")"
mkdir -p "$LEDGER_ROOT"
chmod 700 "$LEDGER_ROOT"
: > "$LOG"

# Validate every executable dimension, bit width, layout, constructor anchor,
# batch anchor, and pinned source digest before building or invoking a layer.
# Coverage rows expose every selected but unexecuted Conv2D/FC declaration.
python3 - "$REPO_ROOT" "$LAYER_MANIFEST" "$WORKLOAD_MANIFEST" "$MODELS" \
  "$WORKLOAD" "$PLAN" "$PLAN_META" "$CSV" "$CONTROLS" <<'PY'
import csv
import hashlib
import json
import pathlib
import re
import sys

(repo_arg, layer_arg, workload_arg, models_arg, profile,
 plan_arg, meta_arg, csv_arg, controls_arg) = sys.argv[1:]
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
    "p0_dpf_trees", "p1_dpf_trees", "p0_public_a_words", "p1_public_a_words",
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
            "schema_version": "ringlpn.two-party-fc-model-scale.v2",
            "publication_date": "2026-08-04", "manifest_sha256": layer_sha,
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
    "schema_version": "ringlpn.two-party-fc-model-scale.v2", "publication_date": "2026-08-04",
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
matches_control() {
  [[ "$1" == all || "$1" == "$2:$3" ]]
}
file_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}
EMPTY_METRICS=()
for ((metric_index = 0; metric_index < 86; ++metric_index)); do
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
  local layer_label="$source_layer"
  [[ "$is_classifier" == yes ]] && layer_label=classifier
  local ring_batches=$(( (rows * inner * cols + ole_n - 1) / ole_n ))
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
      "${EMPTY_METRICS[@]}"
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
  local -a common=(--host 127.0.0.1 --port "$BASE_PORT" --sid "$sid"
    --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
    --qbits "$qbits" --bw "$bw" --rows "$rows" --inner "$inner" --cols "$cols"
    --ole-n "$ole_n" --ole-c "$ole_c" --ole-t "$ole_t" --noise "$noise"
    "${OT_ARGS[@]}")
  local rc0 rc1 check_rc swap_rc pid0 pid1
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --party 0 \
    "${common[@]}" --out-prefix "$p0_prefix" > "$dir/p0.out" 2>&1 &
  pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --party 1 \
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

  local p0_record_bytes p1_record_bytes
  p0_record_bytes="$(stat -c %s "$p0_record")"
  p1_record_bytes="$(stat -c %s "$p1_record")"
  p0_record_sha="$(file_sha256 "$p0_record")"
  p1_record_sha="$(file_sha256 "$p1_record")"
  if matches_control "$SWAP_LAYER" "$model" "$source_layer"; then
    set +e
    CUDA_VISIBLE_DEVICES="$CHECK_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --check \
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
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --check \
    --p0-record "$p0_record" --p1-record "$p1_record" > "$dir/check.out" 2>&1
  check_rc=$?
  set -e
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

  local p0_row p1_row check_row
  local -a f0 f1 fc
  p0_row="$(sed -n '/^0,/p' "$dir/p0.out" | tail -n 1)"
  p1_row="$(sed -n '/^1,/p' "$dir/p1.out" | tail -n 1)"
  check_row="$(sed -n "/^${qbits},/p" "$dir/check.out" | tail -n 1)"
  IFS=',' read -r -a f0 <<< "$p0_row"
  IFS=',' read -r -a f1 <<< "$p1_row"
  IFS=',' read -r -a fc <<< "$check_row"
  if [[ "${#f0[@]}" -ne 71 || "${#f1[@]}" -ne 71 || "${#fc[@]}" -ne 19 ||
        "${f0[0]}" -ne 0 || "${f1[0]}" -ne 1 ||
        "${f0[1]}" -ne "$qbits" || "${f1[1]}" -ne "$qbits" ||
        "${f0[2]}" -ne "$bw" || "${f1[2]}" -ne "$bw" ||
        "${f0[3]}" -ne "$rows" || "${f1[3]}" -ne "$rows" ||
        "${f0[4]}" -ne "$inner" || "${f1[4]}" -ne "$inner" ||
        "${f0[5]}" -ne "$cols" || "${f1[5]}" -ne "$cols" ||
        "${f0[10]}" -ne "$ring_batches" || "${f1[10]}" -ne "$ring_batches" ||
        "${f0[11]}" -ne "${f1[11]}" || "${f0[13]}" -ne "${f1[13]}" ||
        "${f0[20]}" -ne "${f1[20]}" || "${f0[28]}" != pass || "${f1[28]}" != pass ||
        "${fc[0]}" -ne "$qbits" || "${fc[1]}" -ne "$bw" ||
        "${fc[2]}" -ne "$rows" || "${fc[3]}" -ne "$inner" ||
        "${fc[4]}" -ne "$cols" || "${fc[5]}" -ne "$ring_batches" ||
        "${fc[9]}" != pass || "${fc[10]}" != pass ||
        "${fc[11]}" != pass || "${fc[12]}" != pass ||
        "${f0[53]}" != NA || "${f1[53]}" != NA ||
        "${f0[54]}" != "$invocation_id" || "${f1[54]}" != "$invocation_id" ||
        "${fc[17]}" != "$invocation_id" ||
        "${f0[55]}" != "${f1[55]}" || "${f0[55]}" != "${fc[18]}" ||
        ("${f0[56]}" != sci-iknp && "${f0[56]}" != emp-silent) ||
        "${f0[56]}" != "${f1[56]}" ||
        -z "${f0[57]}" || "${f0[57]}" != "${f1[57]}" ||
        ("${f0[56]}" == sci-iknp &&
         ("${f0[52]}" != yes || "${f1[52]}" != yes)) ||
        ("${f0[56]}" == emp-silent &&
         ("${f0[49]}" != NA || "${f1[49]}" != NA ||
          "${f0[50]}" != NA || "${f1[50]}" != NA ||
          "${f0[51]}" != NA || "${f1[51]}" != NA ||
          "${f0[52]}" != NA || "${f1[52]}" != NA)) ||
        "${f0[59]}" != NA || "${f1[59]}" != NA ||
        "${f0[61]}" != NA || "${f1[61]}" != NA ||
        "${f0[63]}" != NA || "${f1[63]}" != NA ||
        "${f0[65]}" != NA || "${f1[65]}" != NA ||
        -z "${f0[70]}" || "${f0[70]}" != "${f1[70]}" ]]; then
    rm -f "$p0_record" "$p1_record"
    append_failed FAIL supported_untruncated
    had_failure=1
    return 0
  fi

  p0_stdout_sha="$(file_sha256 "$dir/p0.out")"
  p1_stdout_sha="$(file_sha256 "$dir/p1.out")"
  checker_stdout_sha="$(file_sha256 "$dir/check.out")"
  local -a raw_metrics=()
  for ((metric_index = 29; metric_index <= 53; ++metric_index)); do
    case "$metric_index" in
      46) raw_metrics+=("${f1[45]}" "${f0[45]}") ;;
      48) raw_metrics+=("${f1[47]}" "${f0[47]}") ;;
      51) raw_metrics+=("${f1[50]}" "${f0[50]}") ;;
      *) raw_metrics+=("${f0[metric_index]}" "${f1[metric_index]}") ;;
    esac
  done
  for ((metric_index = 13; metric_index <= 16; ++metric_index)); do
    raw_metrics+=("${fc[metric_index]}")
  done
  raw_metrics+=("$invocation_id" "${f0[55]}")
  for ((metric_index = 56; metric_index <= 70; ++metric_index)); do
    case "$metric_index" in
      59) raw_metrics+=("${f1[58]}" "${f0[58]}") ;;
      61) raw_metrics+=("${f1[60]}" "${f0[60]}") ;;
      63) raw_metrics+=("${f1[62]}" "${f0[62]}") ;;
      65) raw_metrics+=("${f1[64]}" "${f0[64]}") ;;
      *) raw_metrics+=("${f0[metric_index]}" "${f1[metric_index]}") ;;
    esac
  done
  append_result_row \
    "$model" "$layer_label" "$trial" "$role" "$rows" "$inner" "$cols" "$bw" \
    "$qbits" "$noise" "$ole_n" "$ring_batches" "${f0[11]}" "${f1[11]}" \
    "${f0[13]}" "${f1[13]}" "${f0[20]}" "${f1[20]}" "${f0[25]}" "${f1[25]}" \
    "${f0[27]}" "${f1[27]}" "$p0_record_bytes" "$p1_record_bytes" "${fc[6]}" \
    "${fc[7]}" "${fc[8]}" "${fc[9]}" "${fc[10]}" "${fc[11]}" pass \
    "$SCHEMA_VERSION" "$PUBLICATION_DATE" "$manifest_sha256" "$workload_manifest_sha256" \
    "$model_order" "$source_layer" "$linear_order" "$forward_order" fc "$source_anchor" \
    "$source_text_sha256" "$batch_source_anchor" "$batch" "$layout" "$ole_c" "$ole_t" \
    "$WORKLOAD" yes supported_untruncated "$truncation_status" "$gap" "${fc[7]}" "${fc[8]}" \
    "$p0_record_sha" "$p1_record_sha" "$p0_stdout_sha" "$p1_stdout_sha" "$checker_stdout_sha" \
    "${raw_metrics[@]}"
  printf 'validated_after_both_party_exits sid=%s invocation_id=%s ledger_digest=%s\n' \
    "$sid" "$invocation_id" "${f0[55]}" > "$dir/COMMITTED"
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

python3 - "$CSV" "$PLAN_META" "$AGGREGATE" "$SUMMARY" "$TRIALS" <<'PY'
import csv
import json
import statistics
import sys
from decimal import Decimal

source, meta_path, aggregate_path, summary_path, trials_arg = sys.argv[1:]
trials = int(trials_arg)
metadata = json.load(open(meta_path, encoding="utf-8"))
with open(source, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
integers = [
    "p0_ring_oles", "p1_ring_oles", "p0_dpf_trees", "p1_dpf_trees",
    "p0_public_a_words", "p1_public_a_words", "p0_protocol_bytes", "p1_protocol_bytes",
    "p0_record_bytes", "p1_record_bytes", "final_payload_bytes_per_party",
]
decimals = [
    "p0_total_us", "p1_total_us", "matched_dealer_keygen_us",
    "checker_two_share_online_us", "stock_gpuKeygenMatmul_two_party_sequential_us",
    "unchanged_gpuMatmulBeaver_two_share_sequential_us",
]
party_sum_metrics = [
    "protocol_dependency_rounds", "preflight_us", "ot_setup_us", "dpf_phase_a_us",
    "dpf_phase_b_us", "dpf_phase_c_us", "spfss_grouping_us",
    "public_polynomial_exchange_us", "gpu_ringlpn_expansion_us",
    "derandomization_openings_us", "conversion_us", "serialization_us", "commit_us",
    "transport_straight_bytes_sent", "transport_straight_bytes_received",
    "transport_reversed_bytes_sent", "transport_reversed_bytes_received", "base_ots",
    "base_ot_setup_bytes_sent", "base_ot_setup_bytes_received",
]
party_max_metrics = ["peak_host_rss_bytes", "peak_gpu_bytes"]
party_min_metrics = ["min_gpu_free_bytes"]
party_sum_fields = [
    f"p{party}_{name}_total" for name in party_sum_metrics for party in (0, 1)
]
party_max_fields = [
    f"p{party}_{name}_max" for name in party_max_metrics for party in (0, 1)
]
party_min_fields = [
    f"p{party}_{name}_min" for name in party_min_metrics for party in (0, 1)
]
availability_fields = [
    "p0_transport_bytes_include_base_ot", "p1_transport_bytes_include_base_ot",
    "p0_base_ot_setup_dependency_rounds", "p1_base_ot_setup_dependency_rounds",
]
checker_fields = [
    "checker_us_total", "checker_peak_host_rss_bytes_max",
    "checker_peak_gpu_bytes_max", "checker_min_gpu_free_bytes_min",
]
critical_metrics = [
    "protocol_dependency_rounds", "preflight_us", "ot_setup_us", "dpf_phase_a_us",
    "dpf_phase_b_us", "dpf_phase_c_us", "spfss_grouping_us",
    "public_polynomial_exchange_us", "gpu_ringlpn_expansion_us",
    "derandomization_openings_us", "conversion_us", "serialization_us", "commit_us",
]
critical_fields = [f"critical_path_{name}_total" for name in critical_metrics]
fields = [
    "schema_version", "publication_date", "manifest_sha256", "workload_manifest_sha256",
    "model", "model_order", "trial", "sample_role", "workload",
    "expected_executable_layers", "retained_layer_rows", "failed_layer_rows",
    "unsupported_convolution_layers", "unsupported_truncation_layers",
] + [name + "_total" for name in integers + decimals] + party_sum_fields + \
    party_max_fields + party_min_fields + availability_fields + checker_fields + critical_fields + [
    "critical_path_preprocess_us_total", "execution_status", "full_model_status",
    "workload_status", "status",
]
aggregates = []
for model_meta in metadata["models"]:
    model, expected = model_meta["model"], int(model_meta["expected_executable_layers"])
    for trial in range(trials + 1):
        role = "warmup" if trial == 0 else "measured"
        candidates = [row for row in rows if row["model"] == model and row["trial"] == str(trial)
                      and row["sample_role"] == role and row["operator"] == "fc"]
        retained = [row for row in candidates if row["retained"] == "yes" and row["status"] == "pass"]
        if len({row["source_layer"] for row in retained}) != len(retained):
            raise SystemExit(f"duplicate retained layer for {model} trial {trial}")
        result = {
            "schema_version": metadata["schema_version"], "publication_date": metadata["publication_date"],
            "manifest_sha256": metadata["manifest_sha256"],
            "workload_manifest_sha256": metadata["workload_manifest_sha256"],
            "model": model, "model_order": model_meta["model_order"], "trial": trial,
            "sample_role": role, "workload": metadata["workload"],
            "expected_executable_layers": expected, "retained_layer_rows": len(retained),
            "failed_layer_rows": len(candidates) - len(retained),
            "unsupported_convolution_layers": model_meta["unsupported_convolution_layers"],
            "unsupported_truncation_layers": model_meta["unsupported_truncation_layers"],
        }
        for name in integers:
            result[name + "_total"] = sum(int(row[name]) for row in retained)
        for name in decimals:
            result[name + "_total"] = sum((Decimal(row[name]) for row in retained), Decimal(0))
        def complete_numeric(column, operation):
            values = [row[column] for row in retained]
            if not values:
                return Decimal(0)
            if any(value in ("", "NA") for value in values):
                return "NA"
            parsed = [Decimal(value) for value in values]
            return operation(parsed)
        for name in party_sum_metrics:
            for party in (0, 1):
                column = f"p{party}_{name}"
                result[column + "_total"] = complete_numeric(column, sum)
        for name in party_max_metrics:
            for party in (0, 1):
                column = f"p{party}_{name}"
                result[column + "_max"] = complete_numeric(column, max)
        for name in party_min_metrics:
            for party in (0, 1):
                column = f"p{party}_{name}"
                result[column + "_min"] = complete_numeric(column, min)
        for party in (0, 1):
            include_column = f"p{party}_transport_bytes_include_base_ot"
            include_values = [row[include_column] for row in retained]
            result[include_column] = "yes" if include_values and all(value == "yes" for value in include_values) else "NA"
            result[f"p{party}_base_ot_setup_dependency_rounds"] = "NA"
        result["checker_us_total"] = complete_numeric("checker_us", sum)
        result["checker_peak_host_rss_bytes_max"] = complete_numeric("checker_peak_host_rss_bytes", max)
        result["checker_peak_gpu_bytes_max"] = complete_numeric("checker_peak_gpu_bytes", max)
        result["checker_min_gpu_free_bytes_min"] = complete_numeric("checker_min_gpu_free_bytes", min)
        for name in critical_metrics:
            pairs = [(row[f"p0_{name}"], row[f"p1_{name}"]) for row in retained]
            if any(left in ("", "NA") or right in ("", "NA") for left, right in pairs):
                result[f"critical_path_{name}_total"] = "NA"
            else:
                result[f"critical_path_{name}_total"] = sum(
                    (max(Decimal(left), Decimal(right)) for left, right in pairs), Decimal(0))
        result["critical_path_preprocess_us_total"] = sum(
            (max(Decimal(row["p0_total_us"]), Decimal(row["p1_total_us"])) for row in retained), Decimal(0))
        execution_ok = len(retained) == expected and len(candidates) == expected
        full_ok = (execution_ok and int(model_meta["unsupported_convolution_layers"]) == 0
                   and int(model_meta["unsupported_truncation_layers"]) == 0)
        workload_ok = execution_ok and (metadata["workload"] != "full-model" or full_ok)
        result["execution_status"] = "pass" if execution_ok else "FAIL"
        result["full_model_status"] = "pass" if full_ok else "FAIL_UNSUPPORTED"
        result["workload_status"] = "pass" if workload_ok else "FAIL"
        result["status"] = result["workload_status"]
        aggregates.append(result)
with open(aggregate_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(aggregates)

measured = [row for row in aggregates if row["sample_role"] == "measured" and row["status"] == "pass"]
metrics = {
    "party0_preprocess_us": [float(row["p0_total_us_total"]) for row in measured],
    "party1_preprocess_us": [float(row["p1_total_us_total"]) for row in measured],
    "critical_path_preprocess_us": [float(row["critical_path_preprocess_us_total"]) for row in measured],
    "public_a_words_total": [float(row["p0_public_a_words_total"]) + float(row["p1_public_a_words_total"]) for row in measured],
    "application_bytes_total": [float(row["p0_protocol_bytes_total"]) + float(row["p1_protocol_bytes_total"]) for row in measured],
    "matched_dealer_keygen_us": [float(row["matched_dealer_keygen_us_total"]) for row in measured],
    "checker_two_share_online_us": [float(row["checker_two_share_online_us_total"]) for row in measured],
    "preprocess_over_matched_dealer_ratio": [float(row["critical_path_preprocess_us_total"]) / float(row["matched_dealer_keygen_us_total"])
                                                for row in measured if Decimal(row["matched_dealer_keygen_us_total"]) != 0],
}
def available_values(field):
    return [float(row[field]) for row in measured if row[field] not in ("", "NA")]
metrics["protocol_dependency_rounds"] = available_values(
    "critical_path_protocol_dependency_rounds_total")
for stage in (
    "preflight_us", "ot_setup_us", "dpf_phase_a_us", "dpf_phase_b_us",
    "dpf_phase_c_us", "spfss_grouping_us", "public_polynomial_exchange_us",
    "gpu_ringlpn_expansion_us", "derandomization_openings_us", "conversion_us",
    "serialization_us", "commit_us",
):
    metrics[f"critical_path_{stage}"] = available_values(f"critical_path_{stage}_total")
for output_name, left, right, operation in (
    ("peak_host_rss_bytes", "p0_peak_host_rss_bytes_max", "p1_peak_host_rss_bytes_max", max),
    ("peak_gpu_bytes", "p0_peak_gpu_bytes_max", "p1_peak_gpu_bytes_max", max),
    ("minimum_observed_gpu_free_bytes", "p0_min_gpu_free_bytes_min", "p1_min_gpu_free_bytes_min", min),
):
    metrics[output_name] = [
        operation(float(row[left]), float(row[right])) for row in measured
        if row[left] not in ("", "NA") and row[right] not in ("", "NA")
    ]
metrics["transport_bytes_total_including_base_ot"] = [
    sum(float(row[field]) for field in (
        "p0_transport_straight_bytes_sent_total", "p0_transport_reversed_bytes_sent_total",
        "p1_transport_straight_bytes_sent_total", "p1_transport_reversed_bytes_sent_total",
    ))
    for row in measured
    if all(row[field] not in ("", "NA") for field in (
        "p0_transport_straight_bytes_sent_total", "p0_transport_reversed_bytes_sent_total",
        "p1_transport_straight_bytes_sent_total", "p1_transport_reversed_bytes_sent_total",
    ))
]
metrics["base_ot_setup_bytes_total"] = [
    float(row["p0_base_ot_setup_bytes_sent_total"]) +
    float(row["p1_base_ot_setup_bytes_sent_total"])
    for row in measured
    if row["p0_base_ot_setup_bytes_sent_total"] not in ("", "NA")
    and row["p1_base_ot_setup_bytes_sent_total"] not in ("", "NA")
]
metrics["checker_us"] = available_values("checker_us_total")
metrics["checker_peak_host_rss_bytes"] = available_values("checker_peak_host_rss_bytes_max")
metrics["checker_peak_gpu_bytes"] = available_values("checker_peak_gpu_bytes_max")
metrics["checker_min_gpu_free_bytes"] = available_values("checker_min_gpu_free_bytes_min")
units = {
    "application_bytes_total": "bytes", "public_a_words_total": "field_words",
    "preprocess_over_matched_dealer_ratio": "ratio",
    "protocol_dependency_rounds": "rounds",
    "peak_host_rss_bytes": "bytes", "peak_gpu_bytes": "bytes",
    "minimum_observed_gpu_free_bytes": "bytes",
    "transport_bytes_total_including_base_ot": "bytes",
    "base_ot_setup_bytes_total": "bytes",
    "checker_peak_host_rss_bytes": "bytes",
    "checker_peak_gpu_bytes": "bytes", "checker_min_gpu_free_bytes": "bytes",
}
with open(summary_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["metric", "n", "mean", "sample_stdev", "median", "min", "max", "unit"])
    for name, values in metrics.items():
        if not values:
            writer.writerow([name, 0, "", "", "", "", "", units.get(name, "us")])
        else:
            writer.writerow([name, len(values), statistics.fmean(values),
                             statistics.stdev(values) if len(values) > 1 else 0.0,
                             statistics.median(values), min(values), max(values), units.get(name, "us")])
PY

{
  echo "measurement_timestamp=$(date --iso-8601=seconds)"
  echo "publication_date=$PUBLICATION_DATE"
  echo "schema_version=$SCHEMA_VERSION"
  echo "claim_scope=internal/advisor feasibility matrix; qbits is a CRT construction label, not a security level"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srvmo)"
  echo "cpu_count=$(nproc)"
  echo "process_gpu_map=party0:$P0_GPU,party1:$P1_GPU,checker:$CHECK_GPU"
  echo "network=single-host IPv4 loopback"
  echo "counters=legacy protocol bytes exclude preflight/OT setup; transport stream bytes include selected-backend setup, exclude TCP framing, and add no metrics message"
  echo "ot_backend=$OT_BACKEND"
  echo "emp_silent_bridge=${RINGLPN_EMP_SILENT_BRIDGE:-NA}"
  echo "warmups=1"
  echo "measured_trials=$TRIALS"
  echo "models=$MODELS"
  echo "workload=$WORKLOAD"
  echo "fail_layer_control=${FAIL_LAYER:-none}"
  echo "swap_layer_control=${SWAP_LAYER:-none}"
  echo "aggregate_rule=all totals are arithmetic sums over retained per-layer rows"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version
  sha256sum "$BIN" "$LAYER_MANIFEST" "$WORKLOAD_MANIFEST" "$RESULT_SCHEMAS" \
    "$ROOT/scripts/two_party_fc_metrics_schema_2026_08_04.csv" \
    "$ROOT/src/test_two_party_fc_preprocess.cu" "$ROOT/src/two_party_spfss.h" \
    "$ROOT/src/two_party_spfss_gpu.cuh" "$ROOT/src/two_party_dpf_protocol.h" \
    "$ROOT/src/two_party_dpf_gpu.cuh" "$ROOT/src/two_party_ot.h" \
    "$ROOT/src/emp_silent_adapter.h" "$ROOT/src/emp_silent_bridge.h" \
    "$ROOT/src/emp_silent_bridge.cpp" "$ROOT/src/ringlpn_ole_party.cuh" \
    "$ROOT/src/secure_convert.h" \
    "$REPO_ROOT/GPU-MPC/experiments/orca/cnn.h" \
    "$REPO_ROOT/GPU-MPC/experiments/orca/orca_inference.cu" \
    "$REPO_ROOT/GPU-MPC/experiments/orca/piranha.cu" \
    "$REPO_ROOT/GPU-MPC/nn/orca/fc_layer.cu"
  if [[ "$OT_BACKEND" == emp-silent ]]; then
    sha256sum "$RINGLPN_EMP_SILENT_BRIDGE"
  fi
} > "$ENVIRONMENT"

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
echo "[two-party-fc-model] log: $LOG"
