#!/usr/bin/env bash
# Runs secure stochastic truncation as two OS processes over real SCI/IKNP
# loopback transport, checks reconstructed outputs offline, and exercises common
# preflight disagreement controls. TEST-ONLY records contain private shares.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/host_bin/test_secure_truncate"
OUTDIR="$ROOT/results/secure_truncate"
WORKDIR="${WORKDIR:-}"
PRIVATE_WORKDIR=0
if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ringlpn-secure-truncate.XXXXXX")"
  PRIVATE_WORKDIR=1
fi
cleanup_private_workdir() {
  local rc=$?
  if (( PRIVATE_WORKDIR )); then rm -rf -- "$WORKDIR"; fi
  exit "$rc"
}
trap cleanup_private_workdir EXIT
CHECKCSV="$OUTDIR/secure_truncate_check_2026_08_06.csv"
LOG="$OUTDIR/secure_truncate_2026_08_06.log"
BASE_PORT="${BASE_PORT:-20700}"
TRIALS="${TRIALS:-128}"

if [[ ! -x "$BIN" ]]; then
  echo "secure-truncate test not built. Run scripts/build_secure_truncate_test.sh first." >&2
  exit 1
fi
if [[ ! $BASE_PORT =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT > 65503 )); then
  echo "BASE_PORT must be in 1..65503" >&2
  exit 2
fi
if [[ ! $TRIALS =~ ^[1-9][0-9]*$ ]] || (( TRIALS > 65520 )); then
  echo "TRIALS must be in 1..65520" >&2
  exit 2
fi

mkdir -p "$OUTDIR" "$WORKDIR"
: >"$LOG"
rm -f "$CHECKCSV"

run_pair() {
  local name=$1
  local port=$2
  shift 2
  local prefix="$WORKDIR/$name"
  rm -f "${prefix}_p0.truncate" "${prefix}_p1.truncate"
  timeout 120s "$BIN" --party 0 --port "$port" --out-prefix "$prefix" "$@" \
    >"$WORKDIR/${name}_p0.stdout" 2>>"$LOG" &
  local p0=$!
  sleep 0.2
  timeout 120s "$BIN" --party 1 --host 127.0.0.1 --port "$port" \
    --out-prefix "$prefix" "$@" \
    >"$WORKDIR/${name}_p1.stdout" 2>>"$LOG" &
  local p1=$!
  local rc0=0 rc1=0
  wait "$p0" || rc0=$?
  wait "$p1" || rc1=$?
  if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    echo "[secure-truncate] $name FAILED (p0=$rc0 p1=$rc1)" | tee -a "$LOG"
    return 1
  fi
}

# Each disagreement reaches the common 57-byte preflight, then both peers must
# exit successfully only because --expect-reject observed zero OT consumption.
run_pair reject_shift "$BASE_PORT" --bw 16 --shift 8 --trials 1 --seed 91 \
  --expect-reject --mismatch-shift
run_pair reject_correlation "$((BASE_PORT + 4))" --bw 16 --shift 8 \
  --trials 1 --seed 92 --expect-reject --mismatch-correlation
run_pair reject_share "$((BASE_PORT + 8))" --bw 16 --shift 8 --trials 1 \
  --seed 93 --expect-reject --invalid-share-party 1

# bw shift seed
CASES=(
  "3 1 1"
  "8 1 2"
  "8 7 3"
  "16 8 4"
  "32 10 5"
  "32 24 6"
  "32 31 7"
)

port=$((BASE_PORT + 12))
header=0
for cfg in "${CASES[@]}"; do
  read -r bw shift seed <<<"$cfg"
  name="bw${bw}_shift${shift}"
  run_pair "$name" "$port" --bw "$bw" --shift "$shift" \
    --trials "$TRIALS" --seed "$seed"
  prefix="$WORKDIR/$name"
  h=()
  [[ $header -eq 0 ]] && h=(--csv-header)
  "$BIN" --check --out-prefix "$prefix" "${h[@]}" \
    >"$WORKDIR/${name}_check.csv" 2>>"$LOG"
  cat "$WORKDIR/${name}_check.csv" >>"$CHECKCSV"
  cat "$WORKDIR/${name}_check.csv" >>"$LOG"
  header=1
  port=$((port + 4))
done

# Parser control: a hostile serialized count must be rejected without trying to
# allocate it. The peer file is valid, so failure is attributable to p0 parsing.
malformed="$WORKDIR/malformed"
printf 'RLPTRUNC3 0 16 8 8 16 18446744073709551615\n' \
  >"${malformed}_p0.truncate"
cp "$WORKDIR/bw16_shift8_p1.truncate" "${malformed}_p1.truncate"
malformed_rc=0
timeout 10s "$BIN" --check --out-prefix "$malformed" \
  >"$WORKDIR/malformed_check.csv" 2>>"$LOG" || malformed_rc=$?
if [[ $malformed_rc -ne 1 ]]; then
  echo "[secure-truncate] malformed-count control FAILED (rc=$malformed_rc)" \
    | tee -a "$LOG"
  exit 1
fi
echo "[secure-truncate] malformed-count control rejected as expected" >>"$LOG"

echo "[secure-truncate] all cases pass"
