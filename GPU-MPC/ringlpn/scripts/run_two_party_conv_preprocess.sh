#!/usr/bin/env bash
# Exercise the one supported live Ring-LPN -> Orca forward-Conv2D contract.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin/test_two_party_conv_preprocess"
OUTDIR="$ROOT/results/conv"
WORKDIR="${WORKDIR:-$OUTDIR/two_party_conv_work_2026_08_04}"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-$P0_GPU}"
PORT="${BASE_PORT:-48680}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
LEDGER_ROOT="${LEDGER_ROOT:-$ROOT/results/deployment/correlation-ledger/party-claims}"

[[ "$P0_GPU" != "$P1_GPU" ]] || { echo "P0_GPU and P1_GPU must differ" >&2; exit 2; }
(( PORT > 0 && PORT < 65530 )) || { echo "BASE_PORT out of range" >&2; exit 2; }
[[ -x "$BIN" ]] || "$ROOT/scripts/build_two_party_conv_preprocess.sh"
mkdir -p "$OUTDIR"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR/party0" "$WORKDIR/party1" "$WORKDIR/controls"
LEDGER_ROOT="$(realpath -m "$LEDGER_ROOT")"
mkdir -p "$LEDGER_ROOT"
chmod 700 "$LEDGER_ROOT"

SID=""
INVOCATION_ID=""
COMMON=()
fresh_common() {
  SID="$(python3 -c 'import secrets; print(secrets.randbelow((1 << 63) - 1) + 1)')"
  INVOCATION_ID="$(openssl rand -hex 16)"
  [[ "$SID" =~ ^[1-9][0-9]*$ && "$INVOCATION_ID" =~ ^[0-9a-f]{32}$ ]] ||
    { echo "failed to generate high-entropy invocation identity" >&2; exit 2; }
  COMMON=(--sid "$SID" --invocation-id "$INVOCATION_ID" --ledger "$LEDGER_ROOT"
    --qbits 64 --bw 16 --n 1 --h 4 --w 4 --ci 1 --fh 3 --fw 3 --co 2
    --padding 1 --stride 1 --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular)
}
fresh_common
PUBLIC_SID="$SID"
PUBLIC_INVOCATION_ID="$INVOCATION_ID"
P0_PREFIX="$WORKDIR/party0/key"
P1_PREFIX="$WORKDIR/party1/key"

set +e
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" \
  --party 0 --port "$PORT" --out-prefix "$P0_PREFIX" "${COMMON[@]}" \
  >"$WORKDIR/party0.log" 2>&1 &
p0_pid=$!
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P1_GPU" "$BIN" \
  --party 1 --host 127.0.0.1 --port "$PORT" --out-prefix "$P1_PREFIX" \
  "${COMMON[@]}" >"$WORKDIR/party1.log" 2>&1 &
p1_pid=$!
wait "$p0_pid"; p0_rc=$?
wait "$p1_pid"; p1_rc=$?
set -e
(( p0_rc == 0 && p1_rc == 0 )) || {
  echo "live Conv2D parties failed: p0=$p0_rc p1=$p1_rc" >&2
  exit 1
}

P0_RECORD="${P0_PREFIX}_p0.conv"
P1_RECORD="${P1_PREFIX}_p1.conv"
CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$BIN" --check --csv-header \
  --p0-record "$P0_RECORD" --p1-record "$P1_RECORD" \
  >"$OUTDIR/two_party_conv_preprocess_2026_08_04.csv"

# Swapped party order must fail before the unchanged GPU consumer is entered.
if CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$BIN" --check \
    --p0-record "$P1_RECORD" --p1-record "$P0_RECORD" \
    >"$WORKDIR/controls/swapped.log" 2>&1; then
  echo "swapped-record control unexpectedly passed" >&2
  exit 1
fi

# Digest-protected corruption and malformed length must fail closed.
cp "$P0_RECORD" "$WORKDIR/controls/corrupt_p0.conv"
python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); b=bytearray(p.read_bytes()); b[len(b)//2]^=1; p.write_bytes(b)' \
  "$WORKDIR/controls/corrupt_p0.conv"
if CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$BIN" --check \
    --p0-record "$WORKDIR/controls/corrupt_p0.conv" --p1-record "$P1_RECORD" \
    >"$WORKDIR/controls/corrupt.log" 2>&1; then
  echo "corrupt-record control unexpectedly passed" >&2
  exit 1
fi
cp "$P0_RECORD" "$WORKDIR/controls/malformed_p0.conv"
truncate -s 17 "$WORKDIR/controls/malformed_p0.conv"
if CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$BIN" --check \
    --p0-record "$WORKDIR/controls/malformed_p0.conv" --p1-record "$P1_RECORD" \
    >"$WORKDIR/controls/malformed.log" 2>&1; then
  echo "malformed-record control unexpectedly passed" >&2
  exit 1
fi

reuse_control() {
  local name="$1" sid="$2" invocation_id="$3" port="$4" co="$5"
  local ledger="${6:-$LEDGER_ROOT}"
  local dir="$WORKDIR/$name"
  mkdir -p "$dir/p0" "$dir/p1"
  local reuse=(--sid "$sid" --invocation-id "$invocation_id"
    --ledger "$ledger" --qbits 64 --bw 16 --n 1 --h 4 --w 4 --ci 1
    --fh 3 --fw 3 --co "$co" --padding 1 --stride 1 --ole-n 8192 --ole-c 2
    --ole-t 8 --noise regular)
  set +e
  timeout 30 env CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" --party 0 --port "$port" \
    --out-prefix "$dir/p0/key" "${reuse[@]}" >"$dir/p0.log" 2>&1 &
  local pid0=$!
  timeout 30 env CUDA_VISIBLE_DEVICES="$P1_GPU" "$BIN" --party 1 \
    --host 127.0.0.1 --port "$port" --out-prefix "$dir/p1/key" \
    "${reuse[@]}" >"$dir/p1.log" 2>&1 &
  local pid1=$!
  wait "$pid0"; local rc0=$?
  wait "$pid1"; local rc1=$?
  set -e
  (( rc0 == 2 && rc1 == 2 )) ||
    { echo "$name consume-once control failed: p0=$rc0 p1=$rc1" >&2; exit 1; }
  [[ ! -e "$dir/p0/key_p0.conv" && ! -e "$dir/p1/key_p1.conv" ]]
}

reuse_control duplicate_id "$PUBLIC_SID" "$PUBLIC_INVOCATION_ID" "$((PORT + 4))" 2
reuse_control restart_retry "$PUBLIC_SID" "$PUBLIC_INVOCATION_ID" "$((PORT + 5))" 2
reuse_control tail_slot_reuse "$PUBLIC_SID" "$PUBLIC_INVOCATION_ID" "$((PORT + 6))" 3
collision_sid="$(python3 -c 'import secrets; print(secrets.randbelow((1 << 63) - 1) + 1)')"
reuse_control invocation_collision "$collision_sid" "$PUBLIC_INVOCATION_ID" \
  "$((PORT + 7))" 2
truncated_ledger="$WORKDIR/truncated-ledger"
mkdir -m 700 "$truncated_ledger"
printf TRUNCATED > "$truncated_ledger/broken.claim"
trunc_sid="$(python3 -c 'import secrets; print(secrets.randbelow((1 << 63) - 1) + 1)')"
trunc_invocation="$(openssl rand -hex 16)"
reuse_control ledger_truncation "$trunc_sid" "$trunc_invocation" \
  "$((PORT + 8))" 2 "$truncated_ledger"

# A stale destination is rejected in preflight; neither party may overwrite it.
fresh_common
mkdir -p "$WORKDIR/stale0" "$WORKDIR/stale1"
printf stale >"$WORKDIR/stale0/key_p0.conv"
set +e
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" \
  --party 0 --port "$((PORT + 1))" --out-prefix "$WORKDIR/stale0/key" \
  "${COMMON[@]}" >"$WORKDIR/controls/stale0.log" 2>&1 &
s0=$!
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P1_GPU" "$BIN" \
  --party 1 --host 127.0.0.1 --port "$((PORT + 1))" \
  --out-prefix "$WORKDIR/stale1/key" "${COMMON[@]}" \
  >"$WORKDIR/controls/stale1.log" 2>&1 &
s1=$!
wait "$s0"; s0_rc=$?
wait "$s1"; s1_rc=$?
set -e
(( s0_rc != 0 && s1_rc != 0 )) || { echo "stale-output control failed" >&2; exit 1; }
[[ "$(cat "$WORKDIR/stale0/key_p0.conv")" == stale ]]
[[ ! -e "$WORKDIR/stale1/key_p1.conv" ]]

# A unilateral final-rename failure rolls back the peer's already-renamed file.
fresh_common
mkdir -p "$WORKDIR/rename0" "$WORKDIR/rename1"
set +e
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" \
  --party 0 --port "$((PORT + 2))" --out-prefix "$WORKDIR/rename0/key" \
  --force-rename-failure "${COMMON[@]}" \
  >"$WORKDIR/controls/rename0.log" 2>&1 &
r0=$!
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P1_GPU" "$BIN" \
  --party 1 --host 127.0.0.1 --port "$((PORT + 2))" \
  --out-prefix "$WORKDIR/rename1/key" "${COMMON[@]}" \
  >"$WORKDIR/controls/rename1.log" 2>&1 &
r1=$!
wait "$r0"; r0_rc=$?
wait "$r1"; r1_rc=$?
set -e
(( r0_rc != 0 && r1_rc != 0 )) || {
  echo "forced-rename bilateral rollback control failed" >&2
  exit 1
}
[[ ! -e "$WORKDIR/rename0/key_p0.conv" ]]
[[ ! -e "$WORKDIR/rename1/key_p1.conv" ]]

# A mismatched public Conv2D shape is rejected before OT setup/publication.
fresh_common
mkdir -p "$WORKDIR/mismatch0" "$WORKDIR/mismatch1"
set +e
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P0_GPU" "$BIN" \
  --party 0 --port "$((PORT + 3))" --out-prefix "$WORKDIR/mismatch0/key" \
  "${COMMON[@]}" >"$WORKDIR/controls/mismatch0.log" 2>&1 &
m0=$!
timeout "$TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$P1_GPU" "$BIN" \
  --party 1 --host 127.0.0.1 --port "$((PORT + 3))" \
  --out-prefix "$WORKDIR/mismatch1/key" "${COMMON[@]}" --co 3 \
  >"$WORKDIR/controls/mismatch1.log" 2>&1 &
m1=$!
wait "$m0"; m0_rc=$?
wait "$m1"; m1_rc=$?
set -e
(( m0_rc != 0 && m1_rc != 0 )) || {
  echo "mismatched-shape preflight control failed" >&2
  exit 1
}
[[ ! -e "$WORKDIR/mismatch0/key_p0.conv" ]]
[[ ! -e "$WORKDIR/mismatch1/key_p1.conv" ]]

printf '%s\n' \
  'control,expected,status' \
  'swapped_records,reject,pass' \
  'corrupt_record,reject,pass' \
  'malformed_length,reject,pass' \
  'duplicate_id,reject,pass' \
  'restart_retry,reject,pass' \
  'invocation_collision,reject,pass' \
  'ledger_truncation,reject,pass' \
  'tail_slot_reuse,reject,pass' \
  'stale_output_bilateral_abort,reject,pass' \
  'forced_rename_bilateral_rollback,reject,pass' \
  'mismatched_shape_preflight,reject,pass' \
  >"$OUTDIR/two_party_conv_preprocess_controls_2026_08_04.csv"
rm -f "$P0_RECORD" "$P1_RECORD" "$WORKDIR/controls/"*.conv
printf '[two-party-conv] canonical live path and controls pass\n'
