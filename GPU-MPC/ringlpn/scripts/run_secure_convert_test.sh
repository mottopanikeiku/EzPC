#!/usr/bin/env bash
# Runs the OT-backed conversion as two independent OS processes over loopback,
# then invokes the TEST-ONLY offline checker on their separate output files.
# Outputs per-party transport rows, per-configuration checks, and a raw log.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/host_bin/test_secure_convert"
OUTDIR="$ROOT/results/secure_convert"
WORKDIR="${WORKDIR:-$OUTDIR/two_party_outputs}"
CSV="$OUTDIR/secure_convert_two_party_2026_08_03.csv"
CHECKCSV="$OUTDIR/secure_convert_two_party_check_2026_08_03.csv"
LOG="$OUTDIR/secure_convert_two_party_2026_08_03.log"
BASE_PORT="${BASE_PORT:-42600}"
SELFTEST="${SELFTEST:-4}"

if [[ ! -x "$BIN" ]]; then
  echo "secure-convert test not built. Run scripts/build_secure_convert_test.sh first."
  exit 1
fi
mkdir -p "$OUTDIR" "$WORKDIR"
: > "$LOG"
rm -f "$CSV" "$CHECKCSV"

# Public-parameter validation must reject before either process opens a socket.
invalid_rc=0
"$BIN" --qbits 64 --bw 16 --value-bound 65536 \
  >/dev/null 2>>"$LOG" || invalid_rc=$?
if [[ $invalid_rc -ne 2 ]]; then
  echo "[two-party-convert] invalid-input control returned $invalid_rc" | tee -a "$LOG"
  exit 1
fi
echo "[two-party-convert] invalid-input control rejected as expected" >>"$LOG"

# qbits bw trials forced_wraps inner value_bound input_seed
CASES=(
  "64 16 32 8 8 255 1"
  "64 24 32 8 8 4095 2"
  "128 16 32 8 8 255 3"
  "128 32 32 8 16 65535 4"
)

port=$BASE_PORT
header=0
check_header=0
status=0
for cfg in "${CASES[@]}"; do
  read -r qbits bw trials forced inner bound seed <<<"$cfg"
  prefix="$WORKDIR/q${qbits}_bw${bw}"
  rm -f "${prefix}_p0.convert" "${prefix}_p1.convert"
  h=(); [[ $header -eq 0 ]] && h=(--csv-header)
  echo "=== qbits=$qbits bw=$bw port=$port ===" >> "$LOG"

  "$BIN" --party 0 --port "$port" --qbits "$qbits" --bw "$bw" \
    --trials "$trials" --forced-wraps "$forced" --inner "$inner" \
    --value-bound "$bound" --input-seed "$seed" --selftest "$SELFTEST" \
    --out-prefix "$prefix" "${h[@]}" > "$WORKDIR/p0.csv" 2>> "$LOG" &
  p0=$!
  sleep 0.2
  "$BIN" --party 1 --host 127.0.0.1 --port "$port" --qbits "$qbits" --bw "$bw" \
    --trials "$trials" --forced-wraps "$forced" --inner "$inner" \
    --value-bound "$bound" --input-seed "$seed" --selftest "$SELFTEST" \
    --out-prefix "$prefix" > "$WORKDIR/p1.csv" 2>> "$LOG" &
  p1=$!

  rc0=0; rc1=0
  wait "$p0" || rc0=$?
  wait "$p1" || rc1=$?
  cat "$WORKDIR/p0.csv" "$WORKDIR/p1.csv" >> "$CSV"
  cat "$WORKDIR/p0.csv" "$WORKDIR/p1.csv" >> "$LOG"
  header=1
  if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    echo "[two-party-convert] protocol FAILED (p0=$rc0 p1=$rc1) q$qbits bw=$bw" | tee -a "$LOG"
    status=1
  fi

  vh=(); [[ $check_header -eq 0 ]] && vh=(--csv-header)
  if ! "$BIN" --check --out-prefix "$prefix" "${vh[@]}" > "$WORKDIR/check.csv" 2>> "$LOG"; then
    echo "[two-party-convert] offline check FAILED q$qbits bw=$bw" | tee -a "$LOG"
    status=1
  fi
  cat "$WORKDIR/check.csv" >> "$CHECKCSV"
  cat "$WORKDIR/check.csv" >> "$LOG"
  check_header=1
  port=$((port + 4))
done

if grep -q "FAIL" "$CSV" || grep -q "FAIL" "$CHECKCSV"; then status=1; fi
echo "wrote $CSV"
echo "wrote $CHECKCSV"
echo "wrote $LOG"
if [[ $status -ne 0 ]]; then exit 1; fi
echo "[two-party-convert] all configurations pass"
