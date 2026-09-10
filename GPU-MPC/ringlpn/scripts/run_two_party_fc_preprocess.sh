#!/usr/bin/env bash
# Build and run the live two-process Ring-LPN -> Orca forward-FC artifact.
#
# The two live processes use distinct GPUs for process isolation and to avoid
# cross-party allocator contention. They write into disjoint party directories. Only
# the post-exit checker receives both record paths and reconstructs validation
# values before invoking readGPUMatmulKey/gpuMatmulBeaver.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/test_two_party_fc_preprocess"
OUTDIR="$ROOT/results/fc"
WORKDIR="${WORKDIR:-}"
PRIVATE_WORKDIR=0
if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ringlpn-two-party-fc.XXXXXX")"
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
CSV="${CSV:-$OUTDIR/two_party_fc_preprocess_2026_08_04.csv}"
CONTROLS="${CONTROLS:-$OUTDIR/two_party_fc_preprocess_controls_2026_08_04.csv}"
LOG="${LOG:-$OUTDIR/two_party_fc_preprocess_2026_08_04.log}"
METRICS_SCHEMA="$ROOT/scripts/two_party_fc_metrics_schema_2026_08_04.csv"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
BASE_PORT="${BASE_PORT:-24080}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
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
  echo "P0_GPU and P1_GPU must name distinct GPUs." >&2
  exit 2
fi
if ! [[ "$BASE_PORT" =~ ^[0-9]+$ && "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] ||
   (( BASE_PORT < 1 || BASE_PORT > 65490 || TIMEOUT_SECONDS < 1 )); then
  echo "invalid BASE_PORT or TIMEOUT_SECONDS; timeout must be positive." >&2
  exit 2
fi

mkdir -p "$OUTDIR"
outdir_real="$(realpath "$OUTDIR")"
declare -A seen_output_paths=()
for output_path in "$CSV" "$CONTROLS" "$LOG"; do
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
printf '%s' \
  'case,qbits,bw,rows,inner,cols,noise,ring_batches,ring_application_slots,ring_bootstrap_slots,p0_ring_oles,p1_ring_oles,p0_dpf_trees,p1_dpf_trees,p0_dpf_scalar_oles,p1_dpf_scalar_oles,p0_dpf_epoch_zero_scalar_oles,p1_dpf_epoch_zero_scalar_oles,p0_dpf_pcg_scalar_oles,p1_dpf_pcg_scalar_oles,p0_dpf_pcg_oles_reserved,p1_dpf_pcg_oles_reserved,p0_dpf_pcg_oles_discarded,p1_dpf_pcg_oles_discarded,p0_dpf_pcg_opening_words_sent,p1_dpf_pcg_opening_words_sent,p0_public_a_seed_words,p1_public_a_seed_words,p0_protocol_bytes,p1_protocol_bytes,p0_total_us,p1_total_us,final_payload_bytes_per_party,matched_dealer_keygen_us,checker_two_share_online_us,matched_dealer_keygen_contract,key_order,unchanged_online,status,p0_protocol_dependency_rounds,p1_protocol_dependency_rounds,p0_preflight_us,p1_preflight_us,p0_ot_setup_us,p1_ot_setup_us,p0_dpf_phase_a_us,p1_dpf_phase_a_us,p0_dpf_phase_b_us,p1_dpf_phase_b_us,p0_dpf_phase_c_us,p1_dpf_phase_c_us,p0_spfss_grouping_us,p1_spfss_grouping_us,p0_public_polynomial_exchange_us,p1_public_polynomial_exchange_us,p0_gpu_ringlpn_expansion_us,p1_gpu_ringlpn_expansion_us,p0_derandomization_openings_us,p1_derandomization_openings_us,p0_conversion_us,p1_conversion_us,p0_serialization_us,p1_serialization_us,p0_commit_us,p1_commit_us,p0_peak_host_rss_bytes,p1_peak_host_rss_bytes,p0_peak_gpu_bytes,p1_peak_gpu_bytes,p0_min_gpu_free_bytes,p1_min_gpu_free_bytes,p0_transport_straight_bytes_sent,p1_transport_straight_bytes_sent,p0_transport_straight_bytes_received,p1_transport_straight_bytes_received,p0_transport_reversed_bytes_sent,p1_transport_reversed_bytes_sent,p0_transport_reversed_bytes_received,p1_transport_reversed_bytes_received,p0_base_ots,p1_base_ots,p0_base_ot_setup_bytes_sent,p1_base_ot_setup_bytes_sent,p0_base_ot_setup_bytes_received,p1_base_ot_setup_bytes_received,p0_transport_bytes_include_base_ot,p1_transport_bytes_include_base_ot,p0_base_ot_setup_dependency_rounds,p1_base_ot_setup_dependency_rounds,checker_us,checker_peak_host_rss_bytes,checker_peak_gpu_bytes,checker_min_gpu_free_bytes,invocation_id,ledger_digest' \
  > "$CSV"
printf '%s\n' \
  ',p0_ot_backend,p1_ot_backend,p0_ot_backend_revision,p1_ot_backend_revision,p0_ot_correlation_straight_bytes_sent,p1_ot_correlation_straight_bytes_sent,p0_ot_correlation_straight_bytes_received,p1_ot_correlation_straight_bytes_received,p0_ot_correlation_reversed_bytes_sent,p1_ot_correlation_reversed_bytes_sent,p0_ot_correlation_reversed_bytes_received,p1_ot_correlation_reversed_bytes_received,p0_ot_adjustment_bytes_sent,p1_ot_adjustment_bytes_sent,p0_ot_adjustment_bytes_received,p1_ot_adjustment_bytes_received,p0_ot_ciphertext_bytes_sent,p1_ot_ciphertext_bytes_sent,p0_ot_ciphertext_bytes_received,p1_ot_ciphertext_bytes_received,p0_ot_inventory_straight_declared,p1_ot_inventory_straight_declared,p0_ot_inventory_straight_consumed,p1_ot_inventory_straight_consumed,p0_ot_inventory_reversed_declared,p1_ot_inventory_reversed_declared,p0_ot_inventory_reversed_consumed,p1_ot_inventory_reversed_consumed,p0_ot_backend_review_status,p1_ot_backend_review_status,p0_ring_application_slots_discarded,p1_ring_application_slots_discarded,p0_channel_auth_straight_bytes_sent,p1_channel_auth_straight_bytes_sent,p0_channel_auth_straight_bytes_received,p1_channel_auth_straight_bytes_received,p0_channel_auth_reversed_bytes_sent,p1_channel_auth_reversed_bytes_sent,p0_channel_auth_reversed_bytes_received,p1_channel_auth_reversed_bytes_received,p0_ot_backend_bridge_sha256,p1_ot_backend_bridge_sha256,p0_dpf_breadth_evaluator_calls,p1_dpf_breadth_evaluator_calls,p0_dpf_root_to_leaf_evaluator_calls,p1_dpf_root_to_leaf_evaluator_calls' \
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
AUTH_P0=
AUTH_P1=
make_channel_auth_pair() {
  local dir="$1"
  AUTH_P0="$dir/party0/channel-auth.key"
  AUTH_P1="$dir/party1/channel-auth.key"
  CHANNEL_AUTH_FILES+=("$AUTH_P0" "$AUTH_P1")
  openssl rand 32 > "$AUTH_P0"
  cp -- "$AUTH_P0" "$AUTH_P1"
  chmod 600 "$AUTH_P0" "$AUTH_P1"
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
  make_channel_auth_pair "$dir"
  local replay_auth=
  if (( case_index == 0 )); then
    mkdir -m 700 "$dir/replay-control-private"
    replay_auth="$dir/replay-control-private/channel-auth.key"
    cp -- "$AUTH_P0" "$replay_auth"
    chmod 600 "$replay_auth"
    CHANNEL_AUTH_FILES+=("$replay_auth")
  fi
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
                --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
                --qbits "$qbits" --bw "$bw" --rows "$rows" --inner "$inner"
                --cols "$cols" --ole-n 8192 --ole-c 2 --ole-t 8
                --noise "$noise" "${OT_ARGS[@]}")

  set +e
  local rogue_rc=NA replay_rc=NA reflection_rc=NA
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" \
    --party 0 --channel-auth-file "$AUTH_P0" "${common[@]}" \
    --out-prefix "$p0_prefix" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  if (( case_index == 0 )); then
    timeout --kill-after=5 10 "$ROOT/scripts/channel_auth_decoy.py" --port "$port" \
      --invocation-id "$invocation_id" \
      --claim-file "$LEDGER_ROOT/${invocation_id}.p0.claim" \
      --frame-direction 0
    rogue_rc=$?
    timeout --kill-after=5 10 "$ROOT/scripts/channel_auth_decoy.py" --port "$port" \
      --invocation-id "$invocation_id" \
      --claim-file "$LEDGER_ROOT/${invocation_id}.p0.claim" \
      --frame-direction 0 \
      --valid-replay-secret-file "$replay_auth"
    replay_rc=$?
    timeout --kill-after=5 10 "$ROOT/scripts/channel_auth_decoy.py" --port "$port" \
      --invocation-id "$invocation_id" \
      --claim-file "$LEDGER_ROOT/${invocation_id}.p0.claim" \
      --frame-direction 1
    reflection_rc=$?
  else
    sleep 1
  fi
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" \
    --party 1 --channel-auth-file "$AUTH_P1" "${common[@]}" \
    --out-prefix "$p1_prefix" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  if (( case_index == 0 )) &&
     [[ "$rogue_rc" == 0 && "$replay_rc" == 0 && "$reflection_rc" == 0 ]]; then
    printf 'rogue_first_connector,reject_before_preflight_then_accept_genuine,0,0,NA,pass\n' >> "$CONTROLS"
    printf 'replayed_authenticator,reject_replayed_nonce_tag_then_accept_genuine,0,0,NA,pass\n' >> "$CONTROLS"
    printf 'opposite_direction_reflection,reject_direction_swap_then_accept_genuine,0,0,NA,pass\n' >> "$CONTROLS"
  elif (( case_index == 0 )); then
    echo "[two-party-fc] channel authentication decoy control failed" >&2
    return 1
  fi

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
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --check \
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
  local limbs=1
  (( qbits == 128 )) && limbs=2
  local expected_application_discarded=$((f0[10] * 2 * limbs * f0[11] -
                                           2 * limbs * rows * inner * cols))
  if [[ "${#f0[@]}" -ne 86 || "${#f1[@]}" -ne 86 || "${#fc[@]}" -ne 19 ||
        "${f0[0]}" != 0 || "${f1[0]}" != 1 ||
        "${f0[1]}" != "$qbits" || "${f1[1]}" != "$qbits" ||
        "${f0[2]}" != "$bw" || "${f1[2]}" != "$bw" ||
        "${f0[3]}" != "$rows" || "${f1[3]}" != "$rows" ||
        "${f0[4]}" != "$inner" || "${f1[4]}" != "$inner" ||
        "${f0[5]}" != "$cols" || "${f1[5]}" != "$cols" ||
        "${f0[6]}" != 8192 || "${f1[6]}" != 8192 ||
        "${f0[7]}" != 2 || "${f1[7]}" != 2 ||
        "${f0[8]}" != 8 || "${f1[8]}" != 8 ||
        "${f0[9]}" != "$noise" || "${f1[9]}" != "$noise" ||
        "${fc[0]}" != "$qbits" || "${fc[1]}" != "$bw" ||
        "${fc[2]}" != "$rows" || "${fc[3]}" != "$inner" ||
        "${fc[4]}" != "$cols" ||
        "${f0[35]}" != pass || "${f1[35]}" != pass || "${fc[9]}" != pass ||
        "${fc[10]}" != pass || "${fc[11]}" != pass || "${fc[12]}" != pass ||
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
    rm -rf "${records[@]}"
    echo "[two-party-fc] $name malformed or failing result rows" >&2
    return 1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s' \
    "$name" "$qbits" "$bw" "$rows" "$inner" "$cols" "$noise" >> "$CSV"
  printf ',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
    "${f0[10]}" "${f0[11]}" "${f0[12]}" \
    "${f0[13]}" "${f1[13]}" "${f0[15]}" "${f1[15]}" \
    "${f0[18]}" "${f1[18]}" "${f0[19]}" "${f1[19]}" \
    "${f0[20]}" "${f1[20]}" "${f0[21]}" "${f1[21]}" \
    "${f0[22]}" "${f1[22]}" "${f0[23]}" "${f1[23]}" >> "$CSV"
  printf ',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
    "${f0[27]}" "${f1[27]}" "${f0[32]}" "${f1[32]}" \
    "${f0[34]}" "${f1[34]}" "${fc[6]}" "${fc[7]}" "${fc[8]}" \
    "${fc[9]}" "${fc[10]}" "${fc[11]}" pass >> "$CSV"
  for ((metric_index = 36; metric_index <= 58; ++metric_index)); do
    case "$metric_index" in
      53) printf ',%s,%s' "${f1[52]}" "${f0[52]}" >> "$CSV" ;;
      55) printf ',%s,%s' "${f1[54]}" "${f0[54]}" >> "$CSV" ;;
      58) printf ',%s,%s' "${f1[57]}" "${f0[57]}" >> "$CSV" ;;
      *) printf ',%s,%s' "${f0[$metric_index]}" "${f1[$metric_index]}" >> "$CSV" ;;
    esac
  done
  printf ',%s,%s,%s,%s' \
    "${f0[63]}" "${f1[63]}" "${f0[64]}" "${f1[64]}" >> "$CSV"
  printf ',%s,%s,%s,%s,%s,%s' \
    "${fc[13]}" "${fc[14]}" "${fc[15]}" "${fc[16]}" \
    "$invocation_id" "${f0[66]}" >> "$CSV"
  for ((metric_index = 67; metric_index <= 81; ++metric_index)); do
    case "$metric_index" in
      70) printf ',%s,%s' "${f1[69]}" "${f0[69]}" >> "$CSV" ;;
      72) printf ',%s,%s' "${f1[71]}" "${f0[71]}" >> "$CSV" ;;
      74) printf ',%s,%s' "${f1[73]}" "${f0[73]}" >> "$CSV" ;;
      76) printf ',%s,%s' "${f1[75]}" "${f0[75]}" >> "$CSV" ;;
      *) printf ',%s,%s' "${f0[$metric_index]}" "${f1[$metric_index]}" >> "$CSV" ;;
    esac
  done
  printf ',%s,%s' "${f0[82]}" "${f1[82]}" >> "$CSV"
  for metric_index in 59 60 61 62 83 84 85; do
    printf ',%s,%s' "${f0[$metric_index]}" "${f1[$metric_index]}" >> "$CSV"
  done
  printf '\n' >> "$CSV"
  printf 'validated_after_both_party_exits sid=%s invocation_id=%s ledger_digest=%s\n' \
    "$sid" "$invocation_id" "${f0[66]}" > "$dir/COMMITTED"

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
  make_channel_auth_pair "$dir"
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 30 "$BIN" --party 0 --host 127.0.0.1 \
    --channel-auth-file "$AUTH_P0" \
    --port "$port" --sid "$sid" --invocation-id "$invocation_id" \
    --ledger "$LEDGER_ROOT" --qbits 64 --bw 16 --rows 2 --inner 2 \
    --cols 2 --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular \
    "${OT_ARGS[@]}" --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 30 "$BIN" --party 1 --host 127.0.0.1 \
    --channel-auth-file "$AUTH_P1" \
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

run_bootstrap_capacity_control() {
  local dir="$WORKDIR/control_bootstrap_capacity"
  local port=$((BASE_PORT + 4 * case_index))
  fresh_identity
  local sid="$FRESH_SID" invocation_id="$FRESH_INVOCATION"
  mkdir -p "$dir/party0" "$dir/party1"
  make_channel_auth_pair "$dir"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
    --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
    --qbits 64 --bw 16 --rows 2 --inner 2 --cols 2
    --ole-n 8192 --ole-c 2 --ole-t 64 --noise regular "${OT_ARGS[@]}")
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 30 "$BIN" --party 0 "${common[@]}" \
    --channel-auth-file "$AUTH_P0" \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 30 "$BIN" --party 1 "${common[@]}" \
    --channel-auth-file "$AUTH_P1" \
    --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  append_logs control_bootstrap_capacity "$dir"
  local status=FAIL
  if (( rc0 == 2 && rc1 == 2 )) &&
     [[ ! -e "$dir/party0/key_p0.fc" && ! -e "$dir/party1/key_p1.fc" ]]; then
    status=pass
  fi
  printf 'bootstrap_capacity,nonpositive_epoch_budget_reject,%s,%s,NA,%s\n' \
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
  make_channel_auth_pair "$dir"
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 30 "$BIN" --party 0 --host 127.0.0.1 \
    --channel-auth-file "$AUTH_P0" \
    --port "$port" --sid "$sid" --invocation-id "$invocation_id" \
    --ledger "$LEDGER_ROOT" "${OT_ARGS[@]}" \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 30 "$BIN" --party 1 --host 127.0.0.1 \
    --channel-auth-file "$AUTH_P1" \
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
  make_channel_auth_pair "$dir"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
                --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
                --qbits 64 --bw 16 --rows 2 --inner 2 --cols 2 --ole-n 8192
                --ole-c 2 --ole-t 8 --noise regular --force-rename-failure
                "${OT_ARGS[@]}")
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --party 0 \
    --channel-auth-file "$AUTH_P0" \
    "${common[@]}" --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" --party 1 \
    --channel-auth-file "$AUTH_P1" \
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
  make_channel_auth_pair "$dir"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
                --invocation-id "$invocation_id" --ledger "$ledger"
                --qbits 64 --bw 16 --rows "$rows" --inner 2 --cols 2
                --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular
                "${OT_ARGS[@]}")
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 30 "$BIN" --party 0 "${common[@]}" \
    --channel-auth-file "$AUTH_P0" \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 1
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 30 "$BIN" --party 1 "${common[@]}" \
    --channel-auth-file "$AUTH_P1" \
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

run_wrong_secret_control() {
  local dir="$WORKDIR/control_wrong_channel_secret"
  local port=$((BASE_PORT + 4 * case_index))
  fresh_identity
  local sid="$FRESH_SID" invocation_id="$FRESH_INVOCATION"
  mkdir -p "$dir/party0" "$dir/party1"
  local p0_auth="$dir/party0/channel-auth.key"
  local p1_auth="$dir/party1/channel-auth.key"
  CHANNEL_AUTH_FILES+=("$p0_auth" "$p1_auth")
  openssl rand 32 > "$p0_auth"
  openssl rand 32 > "$p1_auth"
  chmod 600 "$p0_auth" "$p1_auth"
  local common=(--host 127.0.0.1 --port "$port" --sid "$sid"
    --invocation-id "$invocation_id" --ledger "$LEDGER_ROOT"
    --qbits 64 --bw 16 --rows 2 --inner 2 --cols 2
    --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular "${OT_ARGS[@]}")
  set +e
  CUDA_VISIBLE_DEVICES="$P0_GPU" timeout --kill-after=5 15 "$BIN" --party 0 \
    --channel-auth-file "$p0_auth" "${common[@]}" \
    --out-prefix "$dir/party0/key" > "$dir/p0.out" 2>&1 &
  local pid0=$!
  sleep 0.2
  CUDA_VISIBLE_DEVICES="$P1_GPU" timeout --kill-after=5 15 "$BIN" --party 1 \
    --channel-auth-file "$p1_auth" "${common[@]}" \
    --out-prefix "$dir/party1/key" > "$dir/p1.out" 2>&1 &
  local pid1=$!
  wait "$pid1"; local rc1=$?
  sleep 6
  timeout --kill-after=5 10 "$ROOT/scripts/channel_auth_decoy.py" --port "$port" \
    --invocation-id "$invocation_id" \
    --claim-file "$LEDGER_ROOT/${invocation_id}.p0.claim" \
    --frame-direction 0 >/dev/null 2>&1
  local wake_rc=$?
  wait "$pid0"; local rc0=$?
  set -e
  local status=FAIL
  if (( rc0 == 2 && rc1 == 2 && wake_rc == 0 )) &&
     [[ ! -e "$p0_auth" && ! -e "$p1_auth" &&
        ! -e "$dir/party0/key_p0.fc" && ! -e "$dir/party1/key_p1.fc" ]]; then
    status=pass
  fi
  printf 'wrong_channel_secret,reject_before_preflight_ot_drbg_output,%s,%s,NA,%s\n' \
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
ledger_payload_corruption="$WORKDIR/corrupt-ledger"
mkdir -m 700 "$ledger_payload_corruption"
valid_claim="$LEDGER_ROOT/${first_invocation_id}.p0.claim"
corrupt_claim="$ledger_payload_corruption/corrupt.claim"
python3 -c 'import os,pathlib,sys
source = pathlib.Path(sys.argv[1]).read_bytes()
if len(source) != 132 or source[:16] != b"RLPNFRESHLEDGER1":
    raise SystemExit("unexpected freshness claim format")
mutated = bytearray(source)
mutated[20] ^= 1
if len(mutated) != len(source) or mutated[-32:] != source[-32:] or \
        sum(left != right for left, right in zip(source, mutated)) != 1:
    raise SystemExit("ledger payload corruption is not length/digest preserving")
descriptor = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as output:
    output.write(mutated)
    output.flush()
    os.fsync(output.fileno())' "$valid_claim" "$corrupt_claim"
fresh_identity
run_freshness_reject_control ledger_payload_corruption \
  full_length_claim_digest_reject_before_network \
  "$FRESH_SID" "$FRESH_INVOCATION" 2 "$ledger_payload_corruption"
for party in 0 1; do
  if ! grep -Fq \
      'local preflight failed: work=1 output-absent=1 state-absent=1 paths-disjoint=1 plan=1 claim=0' \
      "$WORKDIR/control_ledger_payload_corruption/p${party}.out"; then
    echo "[two-party-fc] ledger payload mutant missed claim-digest validation" >&2
    exit 1
  fi
done
run_preflight_mismatch_control
run_wrong_secret_control
run_stale_output_control
run_bootstrap_capacity_control

run_rename_failure_control
run_checker_controls
# Raw key records are validation inputs, not public evidence. Remove them after
# every positive and negative checker; default private scratch and its test
# ledger are also removed by the EXIT trap.
rm -f "$WORKDIR"/*/party0/key_p0.fc "$WORKDIR"/*/party1/key_p1.fc \
      "$WORKDIR/control_checker/corrupt_p0.fc"
if [[ "$OT_BACKEND" == emp-silent ]]; then
  python3 "$ROOT/scripts/verify_emp_silent_fc_evidence.py" \
    --csv "$CSV" --controls "$CONTROLS" \
    --bridge "$RINGLPN_EMP_SILENT_BRIDGE" | tee -a "$LOG"
fi


echo "[two-party-fc] all live cases and controls pass"
echo "[two-party-fc] results: $CSV"
echo "[two-party-fc] controls: $CONTROLS"
echo "[two-party-fc] log: $LOG"
echo "[two-party-fc] metrics schema: $METRICS_SCHEMA"
