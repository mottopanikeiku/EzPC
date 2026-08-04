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
MISMATCH_PORT="${MISMATCH_PORT:-42590}"

if [[ ! -x "$BIN" ]]; then
  echo "secure-convert test not built. Run scripts/build_secure_convert_test.sh first."
  exit 1
fi
mkdir -p "$OUTDIR" "$WORKDIR"
: > "$LOG"
rm -f "$CSV" "$CHECKCSV"

# All local validation controls must reject with code 2 before opening sockets.
expect_rejection() {
  local name=$1
  shift
  local rc=0
  timeout 10s "$BIN" "$@" >/dev/null 2>>"$LOG" || rc=$?
  if [[ $rc -ne 2 ]]; then
    echo "[two-party-convert] $name control returned $rc (expected 2)" | tee -a "$LOG"
    exit 1
  fi
  echo "[two-party-convert] $name control rejected with code 2 as expected" >>"$LOG"
}

if [[ ! $BASE_PORT =~ ^(0|[1-9][0-9]*)$ ]] ||
   (( BASE_PORT < 1 || BASE_PORT > 65514 )); then
  echo "[two-party-convert] BASE_PORT must leave room for both sockets in all cases (1..65514)" >&2
  exit 2
fi
if [[ ! $MISMATCH_PORT =~ ^(0|[1-9][0-9]*)$ ]] ||
   (( MISMATCH_PORT < 1 || MISMATCH_PORT > 65534 )); then
  echo "[two-party-convert] MISMATCH_PORT must be in 1..65534" >&2
  exit 2
fi
if [[ ! $SELFTEST =~ ^(0|[1-9][0-9]*)$ ]] ||
   (( SELFTEST < 0 || SELFTEST > 1024 )); then
  echo "[two-party-convert] SELFTEST must be in 0..1024" >&2
  exit 2
fi
if (( MISMATCH_PORT == BASE_PORT || MISMATCH_PORT == BASE_PORT + 4 ||
      MISMATCH_PORT == BASE_PORT + 8 || MISMATCH_PORT == BASE_PORT + 12 )); then
  echo "[two-party-convert] MISMATCH_PORT must be dedicated to the mismatch control" >&2
  exit 2
fi

expect_rejection "invalid value-bound" \
  --qbits 64 --bw 16 --value-bound 65536
expect_rejection "base port 65535" \
  --port 65535 --qbits 64 --bw 16 --inner 8 --value-bound 255
expect_rejection "invalid Layer/FC no-wrap bound" \
  --port "$MISMATCH_PORT" --qbits 64 --bw 32 --inner 1 --value-bound 255

# Both peers exchange canonical public parameters on a dedicated port before
# OT setup. A disagreement must terminate promptly and produce neither CSV nor
# party output files.
mismatch_prefix="$WORKDIR/public_parameter_mismatch"
rm -f "${mismatch_prefix}_p0.convert" "${mismatch_prefix}_p1.convert"
: >"$WORKDIR/mismatch_p0.csv"
: >"$WORKDIR/mismatch_p1.csv"
timeout 15s "$BIN" --party 0 --port "$MISMATCH_PORT" \
  --qbits 64 --bw 16 --trials 1 --forced-wraps 1 --inner 1 \
  --value-bound 255 --input-seed 99 --selftest "$SELFTEST" \
  --out-prefix "$mismatch_prefix" >"$WORKDIR/mismatch_p0.csv" 2>>"$LOG" &
mismatch_p0=$!
sleep 0.2
timeout 15s "$BIN" --party 1 --host 127.0.0.1 --port "$MISMATCH_PORT" \
  --qbits 64 --bw 16 --trials 1 --forced-wraps 1 --inner 1 \
  --value-bound 254 --input-seed 99 --selftest "$SELFTEST" \
  --out-prefix "$mismatch_prefix" >"$WORKDIR/mismatch_p1.csv" 2>>"$LOG" &
mismatch_p1=$!
mismatch_rc0=0
mismatch_rc1=0
wait "$mismatch_p0" || mismatch_rc0=$?
wait "$mismatch_p1" || mismatch_rc1=$?
if [[ $mismatch_rc0 -ne 2 || $mismatch_rc1 -ne 2 ||
      -s "$WORKDIR/mismatch_p0.csv" || -s "$WORKDIR/mismatch_p1.csv" ||
      -e "${mismatch_prefix}_p0.convert" || -e "${mismatch_prefix}_p1.convert" ]]; then
  echo "[two-party-convert] public-parameter mismatch control FAILED (p0=$mismatch_rc0 p1=$mismatch_rc1)" | tee -a "$LOG"
  exit 1
fi
echo "[two-party-convert] public-parameter mismatch rejected before OT setup/output as expected" >>"$LOG"

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

# Exercise the real version-1 read_file path with a hostile serialized count.
# The layout is asserted before deriving the count offset; the checker must
# reject promptly (code 1) rather than allocating from the forged value.
malformed_prefix="$WORKDIR/malformed_record_count"
if ! python3 - \
    "$WORKDIR/q64_bw16_p0.convert" "$WORKDIR/q64_bw16_p1.convert" \
    "${malformed_prefix}_p0.convert" "${malformed_prefix}_p1.convert" <<'PY'
import struct
import sys
from pathlib import Path

src0, src1, dst0, dst1 = map(Path, sys.argv[1:])
header = struct.Struct("<8s8I3Q")
data = bytearray(src0.read_bytes())
peer = src1.read_bytes()
assert header.size == 64 and len(data) >= header.size
fields = header.unpack_from(data)
magic, version = fields[0], fields[1]
trials, forced, serialized_n = fields[6], fields[7], fields[11]
assert magic == b"RLPNCVT1" and version == 1
assert serialized_n == 2 * trials + forced + 4
assert len(data) == header.size + serialized_n * 25
struct.pack_into("<Q", data, header.size - 8, 1 << 63)
dst0.write_bytes(data)
dst1.write_bytes(peer)
PY
then
  echo "[two-party-convert] malformed record-count fixture creation FAILED" | tee -a "$LOG"
  status=1
else
  malformed_rc=0
  timeout 10s "$BIN" --check --out-prefix "$malformed_prefix" \
    >"$WORKDIR/malformed_check.csv" 2>>"$LOG" || malformed_rc=$?
  if [[ $malformed_rc -ne 1 ]]; then
    echo "[two-party-convert] malformed record-count control returned $malformed_rc (expected 1)" | tee -a "$LOG"
    status=1
  else
    echo "[two-party-convert] oversized serialized record count rejected with code 1 before allocation" >>"$LOG"
  fi
fi

# A zero session identifier must be rejected bilaterally before OT setup or
# publication; --input-seed is the executable's public session identifier.
zero_prefix="$WORKDIR/zero_sid"
rm -f "${zero_prefix}_p0.convert" "${zero_prefix}_p1.convert" \
      "${zero_prefix}_p0.convert.tmp" "${zero_prefix}_p1.convert.tmp"
"$BIN" --party 0 --port "$port" --qbits 64 --bw 16 --trials 1 \
  --forced-wraps 1 --inner 1 --value-bound 10 --input-seed 0 \
  --selftest 0 --out-prefix "$zero_prefix" \
  >"$WORKDIR/zero_sid_p0.csv" 2>>"$LOG" &
zero_p0=$!
sleep 0.2
"$BIN" --party 1 --host 127.0.0.1 --port "$port" --qbits 64 --bw 16 \
  --trials 1 --forced-wraps 1 --inner 1 --value-bound 10 --input-seed 0 \
  --selftest 0 --out-prefix "$zero_prefix" \
  >"$WORKDIR/zero_sid_p1.csv" 2>>"$LOG" &
zero_p1=$!
set +e
wait "$zero_p0"; zero_rc0=$?
wait "$zero_p1"; zero_rc1=$?
set -e
if [[ $zero_rc0 -ne 2 || $zero_rc1 -ne 2 ||
      -e "${zero_prefix}_p0.convert" || -e "${zero_prefix}_p1.convert" ||
      -e "${zero_prefix}_p0.convert.tmp" || -e "${zero_prefix}_p1.convert.tmp" ]]; then
  echo "[two-party-convert] zero-session control FAILED (p0=$zero_rc0 p1=$zero_rc1)" | tee -a "$LOG"
  status=1
else
  echo "[two-party-convert] zero session rejected bilaterally before OT/output" >>"$LOG"
fi
port=$((port + 4))

# Transactional publication control: a non-empty directory at party 0's final
# path forces only that local rename to fail. The second bilateral result
# exchange must make party 1 delete its already-renamed final as well.
rename_prefix="$WORKDIR/rename_failure"
rm -rf "${rename_prefix}_p0.convert"
rm -f "${rename_prefix}_p1.convert" \
      "${rename_prefix}_p0.convert.tmp" "${rename_prefix}_p1.convert.tmp"
mkdir -p "${rename_prefix}_p0.convert"
printf 'force rename failure\n' >"${rename_prefix}_p0.convert/blocker"
"$BIN" --party 0 --port "$port" --qbits 64 --bw 16 --trials 1 \
  --forced-wraps 1 --inner 1 --value-bound 10 --input-seed 909 \
  --selftest 0 --out-prefix "$rename_prefix" \
  >"$WORKDIR/rename_failure_p0.csv" 2>>"$LOG" &
rename_p0=$!
sleep 0.2
"$BIN" --party 1 --host 127.0.0.1 --port "$port" --qbits 64 --bw 16 \
  --trials 1 --forced-wraps 1 --inner 1 --value-bound 10 \
  --input-seed 909 --selftest 0 --out-prefix "$rename_prefix" \
  >"$WORKDIR/rename_failure_p1.csv" 2>>"$LOG" &
rename_p1=$!
set +e
wait "$rename_p0"; rename_rc0=$?
wait "$rename_p1"; rename_rc1=$?
set -e
if [[ $rename_rc0 -ne 1 || $rename_rc1 -ne 1 ||
      -e "${rename_prefix}_p1.convert" ||
      -e "${rename_prefix}_p0.convert.tmp" ||
      -e "${rename_prefix}_p1.convert.tmp" ]]; then
  echo "[two-party-convert] bilateral rename-failure control FAILED (p0=$rename_rc0 p1=$rename_rc1)" | tee -a "$LOG"
  status=1
else
  echo "[two-party-convert] bilateral rename failure removed both parties' staged/final records" >>"$LOG"
fi
rm -rf "${rename_prefix}_p0.convert"
if grep -q "FAIL" "$CSV" || grep -q "FAIL" "$CHECKCSV"; then status=1; fi
echo "wrote $CSV"
echo "wrote $CHECKCSV"
echo "wrote $LOG"
if [[ $status -ne 0 ]]; then exit 1; fi
echo "[two-party-convert] all configurations pass"
