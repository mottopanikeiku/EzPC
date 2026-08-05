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
METRICS_SCHEMA="$ROOT/scripts/two_party_fc_metrics_schema_2026_08_04.csv"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
BASE_PORT="${BASE_PORT:-48080}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
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
LEDGER_ROOT="$(realpath -m "$LEDGER_ROOT")"
mkdir -p "$LEDGER_ROOT"
chmod 700 "$LEDGER_ROOT"
: > "$LOG"
printf '%s' \
  'case,qbits,bw,rows,inner,cols,noise,ring_batches,p0_ring_oles,p1_ring_oles,p0_dpf_trees,p1_dpf_trees,p0_public_a_words,p1_public_a_words,p0_protocol_bytes,p1_protocol_bytes,p0_total_us,p1_total_us,final_payload_bytes_per_party,matched_dealer_keygen_us,checker_two_share_online_us,matched_dealer_keygen_contract,key_order,unchanged_online,status,p0_protocol_dependency_rounds,p1_protocol_dependency_rounds,p0_preflight_us,p1_preflight_us,p0_ot_setup_us,p1_ot_setup_us,p0_dpf_phase_a_us,p1_dpf_phase_a_us,p0_dpf_phase_b_us,p1_dpf_phase_b_us,p0_dpf_phase_c_us,p1_dpf_phase_c_us,p0_spfss_grouping_us,p1_spfss_grouping_us,p0_public_polynomial_exchange_us,p1_public_polynomial_exchange_us,p0_gpu_ringlpn_expansion_us,p1_gpu_ringlpn_expansion_us,p0_derandomization_openings_us,p1_derandomization_openings_us,p0_conversion_us,p1_conversion_us,p0_serialization_us,p1_serialization_us,p0_commit_us,p1_commit_us,p0_peak_host_rss_bytes,p1_peak_host_rss_bytes,p0_peak_gpu_bytes,p1_peak_gpu_bytes,p0_min_gpu_free_bytes,p1_min_gpu_free_bytes,p0_transport_straight_bytes_sent,p1_transport_straight_bytes_sent,p0_transport_straight_bytes_received,p1_transport_straight_bytes_received,p0_transport_reversed_bytes_sent,p1_transport_reversed_bytes_sent,p0_transport_reversed_bytes_received,p1_transport_reversed_bytes_received,p0_base_ots,p1_base_ots,p0_base_ot_setup_bytes_sent,p1_base_ot_setup_bytes_sent,p0_base_ot_setup_bytes_received,p1_base_ot_setup_bytes_received,p0_transport_bytes_include_base_ot,p1_transport_bytes_include_base_ot,p0_base_ot_setup_dependency_rounds,p1_base_ot_setup_dependency_rounds,checker_us,checker_peak_host_rss_bytes,checker_peak_gpu_bytes,checker_min_gpu_free_bytes,invocation_id,ledger_digest' \
  > "$CSV"
printf '%s\n' \
  ',p0_ot_backend,p1_ot_backend,p0_ot_backend_revision,p1_ot_backend_revision,p0_ot_correlation_straight_bytes_sent,p1_ot_correlation_straight_bytes_sent,p0_ot_correlation_straight_bytes_received,p1_ot_correlation_straight_bytes_received,p0_ot_correlation_reversed_bytes_sent,p1_ot_correlation_reversed_bytes_sent,p0_ot_correlation_reversed_bytes_received,p1_ot_correlation_reversed_bytes_received,p0_ot_adjustment_bytes_sent,p1_ot_adjustment_bytes_sent,p0_ot_adjustment_bytes_received,p1_ot_adjustment_bytes_received,p0_ot_ciphertext_bytes_sent,p1_ot_ciphertext_bytes_sent,p0_ot_ciphertext_bytes_received,p1_ot_ciphertext_bytes_received,p0_ot_inventory_straight_declared,p1_ot_inventory_straight_declared,p0_ot_inventory_straight_consumed,p1_ot_inventory_straight_consumed,p0_ot_inventory_reversed_declared,p1_ot_inventory_reversed_declared,p0_ot_inventory_reversed_consumed,p1_ot_inventory_reversed_consumed,p0_ot_backend_review_status,p1_ot_backend_review_status' \
  >> "$CSV"
printf '%s\n' 'control,expected,p0_rc,p1_rc,checker_rc,status' > "$CONTROLS"

{
  echo "[two-party-fc] build"
  "$ROOT/scripts/build_two_party_fc_preprocess.sh"
} >> "$LOG" 2>&1

case_index=0
first_p0_record=""
first_p1_record=""
first_sid=""
first_invocation_id=""
FRESH_SID=""
FRESH_INVOCATION=""

fresh_identity() {
  FRESH_SID="$(python3 -c 'import secrets; print(secrets.randbelow((1 << 63) - 1) + 1)')"
  FRESH_INVOCATION="$(openssl rand -hex 16)"
  [[ "$FRESH_SID" =~ ^[1-9][0-9]*$ &&
     "$FRESH_INVOCATION" =~ ^[0-9a-f]{32}$ ]] ||
    { echo "[two-party-fc] failed to generate high-entropy invocation identity" >&2; return 1; }
}

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
  fresh_identity
  local sid="$FRESH_SID"
  local invocation_id="$FRESH_INVOCATION"
  local port=$((BASE_PORT + 4 * case_index))
  local dir="$WORKDIR/$name"
  local p0_prefix="$dir/party0/key"
  local p1_prefix="$dir/party1/key"
  local p0_record="${p0_prefix}_p0.fc"
  local p1_record="${p1_prefix}_p1.fc"
  mkdir -p "$dir/party0" "$dir/party1"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
                --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
                --qbits "$qbits" --bw "$bw" --rows "$rows" --inner "$inner"
                --cols "$cols" --ole-n 8192 --ole-c 2 --ole-t 8
                --noise "$noise" "${OT_ARGS[@]}")

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
  if [[ "${#f0[@]}" -ne 71 || "${#f1[@]}" -ne 71 || "${#fc[@]}" -ne 19 ||
        "${f0[28]}" != pass || "${f1[28]}" != pass || "${fc[9]}" != pass ||
        "${fc[10]}" != pass || "${fc[11]}" != pass || "${fc[12]}" != pass ||
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
    rm -rf "${records[@]}"
    echo "[two-party-fc] $name malformed or failing result rows" >&2
    return 1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
    "$name" "$qbits" "$bw" "$rows" "$inner" "$cols" "$noise" \
    "${f0[10]}" "${f0[11]}" "${f1[11]}" "${f0[13]}" "${f1[13]}" \
    "${f0[20]}" "${f1[20]}" "${f0[25]}" "${f1[25]}" \
    "${f0[27]}" "${f1[27]}" "${fc[6]}" "${fc[7]}" "${fc[8]}" \
    "${fc[9]}" "${fc[10]}" "${fc[11]}" pass >> "$CSV"
  for ((metric_index = 29; metric_index <= 53; ++metric_index)); do
    case "$metric_index" in
      46) printf ',%s,%s' "${f1[45]}" "${f0[45]}" >> "$CSV" ;;
      48) printf ',%s,%s' "${f1[47]}" "${f0[47]}" >> "$CSV" ;;
      51) printf ',%s,%s' "${f1[50]}" "${f0[50]}" >> "$CSV" ;;
      *) printf ',%s,%s' "${f0[$metric_index]}" "${f1[$metric_index]}" >> "$CSV" ;;
    esac
  done
  printf ',%s,%s,%s,%s,%s,%s' \
    "${fc[13]}" "${fc[14]}" "${fc[15]}" "${fc[16]}" \
    "$invocation_id" "${f0[55]}" >> "$CSV"
  for ((metric_index = 56; metric_index <= 70; ++metric_index)); do
    case "$metric_index" in
      59) printf ',%s,%s' "${f1[58]}" "${f0[58]}" >> "$CSV" ;;
      61) printf ',%s,%s' "${f1[60]}" "${f0[60]}" >> "$CSV" ;;
      63) printf ',%s,%s' "${f1[62]}" "${f0[62]}" >> "$CSV" ;;
      65) printf ',%s,%s' "${f1[64]}" "${f0[64]}" >> "$CSV" ;;
      *) printf ',%s,%s' "${f0[$metric_index]}" "${f1[$metric_index]}" >> "$CSV" ;;
    esac
  done
  printf '\n' >> "$CSV"
  printf 'validated_after_both_party_exits sid=%s invocation_id=%s ledger_digest=%s\n' \
    "$sid" "$invocation_id" "${f0[55]}" > "$dir/COMMITTED"

  if (( case_index == 0 )); then
    first_p0_record="$p0_record"
    first_p1_record="$p1_record"
    first_sid="$sid"
    first_invocation_id="$invocation_id"
  fi
  case_index=$((case_index + 1))
  echo "[two-party-fc] $name pass"
}

run_preflight_mismatch_control() {
  local dir="$WORKDIR/control_preflight_mismatch"
  local port=$((BASE_PORT + 4 * case_index))
  fresh_identity
  local sid="$FRESH_SID" invocation_id="$FRESH_INVOCATION"
  mkdir -p "$dir/party0" "$dir/party1"
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout 30 "$BIN" --party 0 --host 127.0.0.1 \
    --port "$port" --sid "$sid" --invocation-id "$invocation_id" \
    --ledger "$LEDGER_ROOT" --qbits 64 --bw 16 --rows 2 --inner 2 \
    --cols 2 --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular \
    "${OT_ARGS[@]}" --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout 30 "$BIN" --party 1 --host 127.0.0.1 \
    --port "$port" --sid "$sid" --invocation-id "$invocation_id" \
    --ledger "$LEDGER_ROOT" --qbits 64 --bw 16 --rows 3 --inner 2 \
    --cols 2 --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular \
    "${OT_ARGS[@]}" --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
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
  fresh_identity
  local sid="$FRESH_SID" invocation_id="$FRESH_INVOCATION"
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout 30 "$BIN" --party 0 --host 127.0.0.1 \
    --port "$port" --sid "$sid" --invocation-id "$invocation_id" \
    --ledger "$LEDGER_ROOT" "${OT_ARGS[@]}" \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout 30 "$BIN" --party 1 --host 127.0.0.1 \
    --port "$port" --sid "$sid" --invocation-id "$invocation_id" \
    --ledger "$LEDGER_ROOT" "${OT_ARGS[@]}" \
    --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
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
  fresh_identity
  local sid="$FRESH_SID" invocation_id="$FRESH_INVOCATION"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
                --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
                --qbits 64 --bw 16 --rows 2 --inner 2 --cols 2 --ole-n 8192
                --ole-c 2 --ole-t 8 --noise regular --force-rename-failure
                "${OT_ARGS[@]}")
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

run_freshness_reject_control() {
  local name="$1" expected="$2" sid="$3" invocation_id="$4" rows="$5"
  local ledger="${6:-$LEDGER_ROOT}"
  local dir="$WORKDIR/control_$name"
  local port=$((BASE_PORT + 4 * case_index))
  mkdir -p "$dir/party0" "$dir/party1"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
                --invocation-id "$invocation_id" --ledger "$ledger"
                --qbits 64 --bw 16 --rows "$rows" --inner 2 --cols 2
                --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular
                "${OT_ARGS[@]}")
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout 30 "$BIN" --party 0 "${common[@]}" \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout 30 "$BIN" --party 1 "${common[@]}" \
    --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  append_logs "control_$name" "$dir"
  local status=FAIL
  if (( rc0 == 2 && rc1 == 2 )) &&
     [[ ! -e "$dir/party0/key_p0.fc" && ! -e "$dir/party1/key_p1.fc" &&
        ! -e "$dir/party0/key_p0.fc.tmp" && ! -e "$dir/party1/key_p1.fc.tmp" ]]; then
    status=pass
  fi
  printf '%s,%s,%s,%s,NA,%s\n' "$name" "$expected" "$rc0" "$rc1" "$status" \
    >> "$CONTROLS"
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
run_freshness_reject_control duplicate_id duplicate_consume_once_reject \
  "$first_sid" "$first_invocation_id" 2
run_freshness_reject_control restart_retry restart_cannot_rollback_ledger \
  "$first_sid" "$first_invocation_id" 2
run_freshness_reject_control tail_slot_reuse unused_tail_is_discarded \
  "$first_sid" "$first_invocation_id" 1
fresh_identity
run_freshness_reject_control invocation_collision \
  same_invocation_different_compatibility_sid_reject \
  "$FRESH_SID" "$first_invocation_id" 2
truncated_ledger="$WORKDIR/truncated-ledger"
mkdir -m 700 "$truncated_ledger"
printf 'TRUNCATED' > "$truncated_ledger/broken.claim"
fresh_identity
run_freshness_reject_control ledger_truncation malformed_append_only_entry_reject \
  "$FRESH_SID" "$FRESH_INVOCATION" 2 "$truncated_ledger"
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
echo "[two-party-fc] metrics schema: $METRICS_SCHEMA"
