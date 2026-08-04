#!/usr/bin/env bash
# Build and run the live two-process Ring-LPN -> Orca forward-FC artifact.
#
# The two live processes use distinct GPUs because Orca's unchanged allocator
# reserves 25 GiB per process. They write into disjoint party directories. Only
# the post-exit checker receives both record paths and reconstructs validation
# values before invoking readGPUMatmulKey/gpuMatmulBeaver.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/test_two_party_fc_preprocess"
OUTDIR="$ROOT/results/fc"
WORKDIR="${WORKDIR:-$OUTDIR/two_party_fc_work_2026_08_04}"
CSV="$OUTDIR/two_party_fc_preprocess_2026_08_04.csv"
CONTROLS="$OUTDIR/two_party_fc_preprocess_controls_2026_08_04.csv"
LOG="$OUTDIR/two_party_fc_preprocess_2026_08_04.log"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
BASE_PORT="${BASE_PORT:-48080}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

if [[ "$P0_GPU" == "$P1_GPU" ]]; then
  echo "P0_GPU and P1_GPU must name distinct GPUs: each process reserves 25 GiB." >&2
  exit 2
fi
if (( BASE_PORT < 1 || BASE_PORT > 65490 )); then
  echo "BASE_PORT must leave room for every two-socket case/control." >&2
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
: > "$LOG"
printf '%s\n' \
  'case,qbits,bw,rows,inner,cols,noise,ring_batches,p0_ring_oles,p1_ring_oles,p0_dpf_trees,p1_dpf_trees,p0_public_a_words,p1_public_a_words,p0_protocol_bytes,p1_protocol_bytes,p0_total_us,p1_total_us,final_payload_bytes_per_party,matched_dealer_keygen_us,checker_two_share_online_us,matched_dealer_keygen_contract,key_order,unchanged_online,status' \
  > "$CSV"
printf '%s\n' 'control,expected,p0_rc,p1_rc,checker_rc,status' > "$CONTROLS"

{
  echo "[two-party-fc] build"
  "$ROOT/scripts/build_two_party_fc_preprocess.sh"
} >> "$LOG" 2>&1

case_index=0
first_p0_record=""
first_p1_record=""

append_logs() {
  local label="$1"
  local dir="$2"
  {
    echo
    echo "===== $label / party 0 ====="
    cat "$dir/p0.out"
    echo "===== $label / party 1 ====="
    cat "$dir/p1.out"
    if [[ -f "$dir/check.out" ]]; then
      echo "===== $label / offline checker ====="
      cat "$dir/check.out"
    fi
  } >> "$LOG"
}

run_case() {
  local name="$1" qbits="$2" bw="$3" rows="$4" inner="$5" cols="$6" noise="$7"
  local sid=$((202608040000 + case_index + 1))
  local port=$((BASE_PORT + 4 * case_index))
  local dir="$WORKDIR/$name"
  local p0_prefix="$dir/party0/key"
  local p1_prefix="$dir/party1/key"
  local p0_record="${p0_prefix}_p0.fc"
  local p1_record="${p1_prefix}_p1.fc"
  mkdir -p "$dir/party0" "$dir/party1"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid" --qbits "$qbits"
                --bw "$bw" --rows "$rows" --inner "$inner" --cols "$cols"
                --ole-n 8192 --ole-c 2 --ole-t 8 --noise "$noise")

  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" \
    --party 0 "${common[@]}" --out-prefix "$p0_prefix" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" \
    --party 1 "${common[@]}" --out-prefix "$p1_prefix" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e

  local records=("$p0_record" "$p1_record"
                 "${p0_record}.tmp" "${p1_record}.tmp")
  if (( rc0 != 0 || rc1 != 0 )); then
    append_logs "$name" "$dir"
    rm -rf "${records[@]}"
    echo "[two-party-fc] $name live parties failed: p0=$rc0 p1=$rc1" >&2
    return 1
  fi
  if [[ ! -f "$p0_record" || ! -f "$p1_record" ||
        -e "$dir/party0/key_p1.fc" || -e "$dir/party1/key_p0.fc" ]]; then
    append_logs "$name" "$dir"
    rm -rf "${records[@]}"
    echo "[two-party-fc] $name violated party-local output ownership" >&2
    return 1
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --check \
    --p0-record "$p0_record" --p1-record "$p1_record" > "$dir/check.out" 2>&1
  local check_rc=$?
  set -e
  append_logs "$name" "$dir"
  if (( check_rc != 0 )); then
    rm -rf "${records[@]}"
    echo "[two-party-fc] $name offline Orca checker failed: rc=$check_rc" >&2
    return 1
  fi

  local p0_row p1_row check_row
  p0_row="$(sed -n '/^0,/p' "$dir/p0.out")"
  p1_row="$(sed -n '/^1,/p' "$dir/p1.out")"
  check_row="$(sed -n '/^[0-9][0-9]*,/p' "$dir/check.out")"
  local -a f0 f1 fc
  IFS=',' read -r -a f0 <<< "$p0_row"
  IFS=',' read -r -a f1 <<< "$p1_row"
  IFS=',' read -r -a fc <<< "$check_row"
  if [[ "${#f0[@]}" -ne 29 || "${#f1[@]}" -ne 29 || "${#fc[@]}" -ne 13 ||
        "${f0[28]}" != pass || "${f1[28]}" != pass || "${fc[9]}" != pass ||
        "${fc[10]}" != pass || "${fc[11]}" != pass || "${fc[12]}" != pass ]]; then
    rm -rf "${records[@]}"
    echo "[two-party-fc] $name malformed or failing result rows" >&2
    return 1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$name" "$qbits" "$bw" "$rows" "$inner" "$cols" "$noise" \
    "${f0[10]}" "${f0[11]}" "${f1[11]}" "${f0[13]}" "${f1[13]}" \
    "${f0[20]}" "${f1[20]}" "${f0[25]}" "${f1[25]}" \
    "${f0[27]}" "${f1[27]}" "${fc[6]}" "${fc[7]}" "${fc[8]}" \
    "${fc[9]}" "${fc[10]}" "${fc[11]}" pass >> "$CSV"
  printf 'validated_after_both_party_exits sid=%s\n' "$sid" > "$dir/COMMITTED"

  if (( case_index == 0 )); then
    first_p0_record="$p0_record"
    first_p1_record="$p1_record"
  fi
  case_index=$((case_index + 1))
  echo "[two-party-fc] $name pass"
}

run_preflight_mismatch_control() {
  local dir="$WORKDIR/control_preflight_mismatch"
  local port=$((BASE_PORT + 4 * case_index))
  mkdir -p "$dir/party0" "$dir/party1"
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout 30 "$BIN" --party 0 --host 127.0.0.1 \
    --port "$port" --sid 202608049001 --qbits 64 --bw 16 --rows 2 --inner 2 \
    --cols 2 --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout 30 "$BIN" --party 1 --host 127.0.0.1 \
    --port "$port" --sid 202608049001 --qbits 64 --bw 16 --rows 3 --inner 2 \
    --cols 2 --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular \
    --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  append_logs control_preflight_mismatch "$dir"
  local status=FAIL
  if (( rc0 == 2 && rc1 == 2 )) &&
     [[ ! -e "$dir/party0/key_p0.fc" && ! -e "$dir/party1/key_p1.fc" ]]; then
    status=pass
  fi
  printf 'preflight_mismatch,bilateral_reject_before_output,%s,%s,NA,%s\n' \
    "$rc0" "$rc1" "$status" >> "$CONTROLS"
  [[ "$status" == pass ]]
  case_index=$((case_index + 1))
}

run_stale_output_control() {
  local dir="$WORKDIR/control_stale_output"
  local port=$((BASE_PORT + 4 * case_index))
  mkdir -p "$dir/party0" "$dir/party1"
  printf 'DO_NOT_OVERWRITE' > "$dir/party0/key_p0.fc"
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout 30 "$BIN" --party 0 --host 127.0.0.1 \
    --port "$port" --sid 202608049002 --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout 30 "$BIN" --party 1 --host 127.0.0.1 \
    --port "$port" --sid 202608049002 --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  append_logs control_stale_output "$dir"
  local status=FAIL
  if (( rc0 == 2 && rc1 == 2 )) &&
     [[ "$(cat "$dir/party0/key_p0.fc")" == DO_NOT_OVERWRITE &&
        ! -e "$dir/party1/key_p1.fc" ]]; then
    status=pass
  fi
  printf 'stale_output,bilateral_reject_without_overwrite,%s,%s,NA,%s\n' \
    "$rc0" "$rc1" "$status" >> "$CONTROLS"
  [[ "$status" == pass ]]
  case_index=$((case_index + 1))
}

run_rename_failure_control() {
  local dir="$WORKDIR/control_rename_failure"
  local port=$((BASE_PORT + 4 * case_index))
  mkdir -p "$dir/party0" "$dir/party1"
  local common=(--host 127.0.0.1 --port "$port" --sid 202608049003 --qbits 64
                --bw 16 --rows 2 --inner 2 --cols 2 --ole-n 8192 --ole-c 2
                --ole-t 8 --noise regular --force-rename-failure)
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --party 0 \
    "${common[@]}" --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout "$TIMEOUT_SECONDS" "$BIN" --party 1 \
    "${common[@]}" --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  append_logs control_rename_failure "$dir"
  local status=FAIL
  if (( rc0 == 1 && rc1 == 1 )) &&
     [[ ! -e "$dir/party0/key_p0.fc" && ! -e "$dir/party1/key_p1.fc" &&
        ! -e "$dir/party0/key_p0.fc.tmp" && ! -e "$dir/party1/key_p1.fc.tmp" ]]; then
    status=pass
  fi
  printf 'rename_failure,bilateral_cleanup_after_staging,%s,%s,NA,%s\n' \
    "$rc0" "$rc1" "$status" >> "$CONTROLS"
  [[ "$status" == pass ]]
  case_index=$((case_index + 1))
}

run_checker_controls() {
  local dir="$WORKDIR/control_checker"
  mkdir -p "$dir"
  python3 -c 'import pathlib,sys; p=bytearray(pathlib.Path(sys.argv[1]).read_bytes()); p[80]^=1; pathlib.Path(sys.argv[2]).write_bytes(p)' \
    "$first_p0_record" "$dir/corrupt_p0.fc"
  set +e
  "$BIN" --check --p0-record "$dir/corrupt_p0.fc" --p1-record "$first_p1_record" \
    > "$dir/corrupt.out" 2>&1
  local corrupt_rc=$?
  "$BIN" --check --p0-record "$first_p1_record" --p1-record "$first_p0_record" \
    > "$dir/swapped.out" 2>&1
  local swapped_rc=$?
  set -e
  {
    echo
    echo '===== control_checker / corrupt digest ====='
    cat "$dir/corrupt.out"
    echo '===== control_checker / swapped party records ====='
    cat "$dir/swapped.out"
  } >> "$LOG"
  local corrupt_status=FAIL swapped_status=FAIL
  [[ "$corrupt_rc" -eq 1 ]] && corrupt_status=pass
  [[ "$swapped_rc" -eq 1 ]] && swapped_status=pass
  printf 'corrupt_record,offline_digest_reject,NA,NA,%s,%s\n' \
    "$corrupt_rc" "$corrupt_status" >> "$CONTROLS"
  printf 'swapped_records,offline_party_header_reject,NA,NA,%s,%s\n' \
    "$swapped_rc" "$swapped_status" >> "$CONTROLS"
  [[ "$corrupt_status" == pass && "$swapped_status" == pass ]]
}

run_case q64_regular_small 64 16 2 2 2 regular
run_case q64_uniform_small 64 16 2 2 2 uniform
run_case q128_regular_small 128 32 2 2 2 regular
run_case q128_uniform_small 128 32 2 2 2 uniform
run_case q64_regular_multibatch 64 16 8 65 16 regular
run_preflight_mismatch_control
run_stale_output_control

run_rename_failure_control
run_checker_controls
# Raw key records are validation inputs, not public evidence. Remove them after
# every positive and negative checker has completed; retain only metrics/logs
# and the per-case post-validation COMMITTED markers.
rm -f "$WORKDIR"/*/party0/key_p0.fc "$WORKDIR"/*/party1/key_p1.fc \
      "$WORKDIR/control_checker/corrupt_p0.fc"

echo "[two-party-fc] all live cases and controls pass"
echo "[two-party-fc] results: $CSV"
echo "[two-party-fc] controls: $CONTROLS"
echo "[two-party-fc] log: $LOG"
