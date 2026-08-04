#!/usr/bin/env bash
# Exact ResNet18 inference-classifier FC shape from experiments/orca/cnn.h:
# M=1, K=512, N=1000 at the configured 32-bit Table-9 inference width.
# qbits=128 means two ~62-bit CRT arithmetic limbs, not 128-bit security.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/test_two_party_fc_preprocess"
OUTDIR="$ROOT/results/fc"
WORKDIR="${WORKDIR:-$OUTDIR/two_party_fc_model_scale_work_2026_08_04}"
CSV="$OUTDIR/two_party_fc_model_scale_2026_08_04.csv"
SUMMARY="$OUTDIR/two_party_fc_model_scale_summary_2026_08_04.csv"
ENVIRONMENT="$OUTDIR/two_party_fc_model_scale_environment_2026_08_04.txt"
LOG="$OUTDIR/two_party_fc_model_scale_2026_08_04.log"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
BASE_PORT="${BASE_PORT:-48280}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
SID_BASE="${SID_BASE:-202608045120}"
WARMUPS="${WARMUPS:-1}"
TRIALS="${TRIALS:-10}"
if [[ "$P0_GPU" == "$P1_GPU" ]]; then
  echo "P0_GPU and P1_GPU must be distinct: each process reserves 25 GiB." >&2
  exit 2
fi
if (( BASE_PORT < 1 || BASE_PORT > 65534 || SID_BASE == 0 ||
      WARMUPS < 0 || TRIALS < 1 )); then
  echo "invalid BASE_PORT, SID_BASE, WARMUPS, or TRIALS" >&2
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
mkdir -p "$WORKDIR/party0" "$WORKDIR/party1"
: > "$LOG"

{
  echo "[two-party-fc-model] build"
  "$ROOT/scripts/build_two_party_fc_preprocess.sh"
} >> "$LOG" 2>&1
printf '%s\n' \
  'model,layer,trial,sample_role,rows,inner,cols,bw,qbits,noise,ole_n,ring_batches,p0_ring_oles,p1_ring_oles,p0_dpf_trees,p1_dpf_trees,p0_public_a_words,p1_public_a_words,p0_protocol_bytes,p1_protocol_bytes,p0_total_us,p1_total_us,p0_record_bytes,p1_record_bytes,final_payload_bytes_per_party,matched_dealer_keygen_us,checker_two_share_online_us,matched_dealer_keygen_contract,key_order,unchanged_online,status' \
  > "$CSV"

run_sample() {
  local sample="$1" role="$2"
  local dir="$WORKDIR/sample_$sample"
  local sid=$((SID_BASE + sample))
  local p0_prefix="$dir/party0/resnet18_classifier"
  local p1_prefix="$dir/party1/resnet18_classifier"
  local p0_record="${p0_prefix}_p0.fc"
  local p1_record="${p1_prefix}_p1.fc"
  local p0_record_bytes p1_record_bytes p0_row p1_row check_row
  local rc0 rc1 check_rc pid0 pid1
  local -a f0 f1 fc
  mkdir -p "$dir/party0" "$dir/party1"
  local -a common=(--host 127.0.0.1 --port "$BASE_PORT" --sid "$sid"
    --qbits 128 --bw 32 --rows 1 --inner 512 --cols 1000 --ole-n 8192
    --ole-c 2 --ole-t 8 --noise regular)

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
    echo "===== ResNet18 classifier / $role $sample / party 0 ====="
    cat "$dir/p0.out"
    echo "===== ResNet18 classifier / $role $sample / party 1 ====="
    cat "$dir/p1.out"
  } >> "$LOG"
  if (( rc0 != 0 || rc1 != 0 )) ||
     [[ ! -f "$p0_record" || ! -f "$p1_record" ]]; then
    rm -f "$p0_record" "$p1_record" "${p0_record}.tmp" "${p1_record}.tmp"
    echo "[two-party-fc-model] $role $sample preprocessing failed: p0=$rc0 p1=$rc1" >&2
    return 1
  fi
  if [[ -e "$dir/party0/resnet18_classifier_p1.fc" ||
        -e "$dir/party1/resnet18_classifier_p0.fc" ]]; then
    rm -f "$p0_record" "$p1_record"
    echo "[two-party-fc-model] $role $sample violated party-local output ownership" >&2
    return 1
  fi

  p0_record_bytes="$(stat -c %s "$p0_record")"
  p1_record_bytes="$(stat -c %s "$p1_record")"
  set +e
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --check \
    --p0-record "$p0_record" --p1-record "$p1_record" \
    > "$dir/check.out" 2>&1
  check_rc=$?
  set -e
  {
    echo "===== ResNet18 classifier / $role $sample / post-exit checker ====="
    cat "$dir/check.out"
  } >> "$LOG"
  if (( check_rc != 0 )); then
    rm -f "$p0_record" "$p1_record"
    echo "[two-party-fc-model] $role $sample checker failed: rc=$check_rc" >&2
    return 1
  fi

  p0_row="$(sed -n '/^0,/p' "$dir/p0.out" | tail -n 1)"
  p1_row="$(sed -n '/^1,/p' "$dir/p1.out" | tail -n 1)"
  check_row="$(sed -n '/^128,/p' "$dir/check.out" | tail -n 1)"
  IFS=',' read -r -a f0 <<< "$p0_row"
  IFS=',' read -r -a f1 <<< "$p1_row"
  IFS=',' read -r -a fc <<< "$check_row"
  if [[ "${#f0[@]}" -ne 29 || "${#f1[@]}" -ne 29 ||
        "${#fc[@]}" -ne 13 || "${f0[10]}" -ne 63 ||
        "${f0[11]}" -ne 252 || "${f1[11]}" -ne 252 ||
        "${f0[13]}" -ne 64512 || "${f1[13]}" -ne 64512 ||
        "${f0[20]}" -ne 4128768 || "${f1[20]}" -ne 4128768 ||
        "${f0[28]}" != pass || "${f1[28]}" != pass ||
        "${fc[9]}" != pass || "${fc[10]}" != pass ||
        "${fc[11]}" != pass || "${fc[12]}" != pass ]]; then
    rm -f "$p0_record" "$p1_record"
    echo "[two-party-fc-model] $role $sample malformed or failing result rows" >&2
    return 1
  fi
  printf '%s\n' \
    "ResNet18,classifier,$sample,$role,1,512,1000,32,128,regular,8192,${f0[10]},${f0[11]},${f1[11]},${f0[13]},${f1[13]},${f0[20]},${f1[20]},${f0[25]},${f1[25]},${f0[27]},${f1[27]},$p0_record_bytes,$p1_record_bytes,${fc[6]},${fc[7]},${fc[8]},${fc[9]},${fc[10]},${fc[11]},pass" \
    >> "$CSV"
  printf 'validated_after_both_party_exits sid=%s\n' "$sid" > "$dir/COMMITTED"
  rm -f "$p0_record" "$p1_record"
  echo "[two-party-fc-model] $role $sample pass"
}

total=$((WARMUPS + TRIALS))
for ((sample = 0; sample < total; ++sample)); do
  if (( sample < WARMUPS )); then
    role=warmup
  else
    role=measured
  fi
  run_sample "$sample" "$role"
done

python3 - "$CSV" "$SUMMARY" <<'PY'
import csv
import statistics
import sys

source, destination = sys.argv[1:]
with open(source, newline="", encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle)
            if row["sample_role"] == "measured" and row["status"] == "pass"]
if not rows:
    raise SystemExit("no passing measured rows")

metrics = {
    "party0_preprocess_us": [float(row["p0_total_us"]) for row in rows],
    "party1_preprocess_us": [float(row["p1_total_us"]) for row in rows],
    "critical_path_preprocess_us": [
        max(float(row["p0_total_us"]), float(row["p1_total_us"])) for row in rows
    ],
    "public_a_words_total": [
        float(row["p0_public_a_words"]) + float(row["p1_public_a_words"])
        for row in rows
    ],
    "application_bytes_total": [
        float(row["p0_protocol_bytes"]) + float(row["p1_protocol_bytes"])
        for row in rows
    ],
    "matched_dealer_keygen_us": [
        float(row["matched_dealer_keygen_us"]) for row in rows
    ],
    "checker_two_share_online_us": [
        float(row["checker_two_share_online_us"]) for row in rows
    ],
    "preprocess_over_matched_dealer_ratio": [
        max(float(row["p0_total_us"]), float(row["p1_total_us"])) /
        float(row["matched_dealer_keygen_us"]) for row in rows
    ],
}
units = {
    "application_bytes_total": "bytes",
    "public_a_words_total": "field_words",
    "preprocess_over_matched_dealer_ratio": "ratio",
}
with open(destination, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "n", "mean", "sample_stdev", "median", "min", "max", "unit"])
    for name, values in metrics.items():
        writer.writerow([
            name, len(values), statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
            statistics.median(values), min(values), max(values),
            units.get(name, "us"),
        ])
PY

{
  echo "measurement_timestamp=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srvmo)"
  echo "cpu_count=$(nproc)"
  echo "process_gpu_map=party0:$P0_GPU,party1:$P1_GPU,checker:$CHECK_GPU"
  echo "network=single-host IPv4 loopback"
  echo "counter=PartyChannel application payload bytes; excludes TCP/IP and base-OT setup"
  echo "warmups=$WARMUPS"
  echo "measured_trials=$TRIALS"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version
  sha256sum "$BIN" "$ROOT/src/test_two_party_fc_preprocess.cu" \
    "$ROOT/src/two_party_spfss.h" "$ROOT/src/two_party_dpf_protocol.h" \
    "$ROOT/src/two_party_ot.h" "$ROOT/src/ringlpn_ole_party.cuh" \
    "$ROOT/src/secure_convert.h"
} > "$ENVIRONMENT"

echo "[two-party-fc-model] ResNet18 classifier evaluation pass"
echo "[two-party-fc-model] raw results: $CSV"
echo "[two-party-fc-model] summary: $SUMMARY"
echo "[two-party-fc-model] environment: $ENVIRONMENT"
echo "[two-party-fc-model] log: $LOG"
