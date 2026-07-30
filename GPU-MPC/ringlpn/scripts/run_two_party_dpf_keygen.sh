#!/usr/bin/env bash
# Runs the two-PROCESS distributed DPF key generation over loopback TCP and
# validates the resulting key pairs offline with the unchanged evaluator.
#
# Each configuration starts two independent OS processes (no shared memory, no
# shared seed), each writing only its own key file. The TEST-ONLY checker then
# reads both key files plus both private-input records and verifies
# beta * [x == alpha] over the full domain, with a corrupted-key negative
# control.
#
# Outputs:
#   results/dpf/two_party_dpf_keygen_2026_07_29.csv   (per-party rows)
#   results/dpf/two_party_dpf_validate_2026_07_29.csv (per-config validation)
#   results/dpf/two_party_dpf_keygen_2026_07_29.log   (raw stdout + stderr)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/host_bin/test_two_party_dpf_keygen"
VALIDATE="$ROOT/host_bin/test_two_party_dpf_validate"
OUTDIR="$ROOT/results/dpf"
WORKDIR="${WORKDIR:-$ROOT/results/dpf/two_party_keys}"
CSV="$OUTDIR/two_party_dpf_keygen_2026_07_29.csv"
VCSV="$OUTDIR/two_party_dpf_validate_2026_07_29.csv"
LOG="$OUTDIR/two_party_dpf_keygen_2026_07_29.log"
BASE_PORT="${BASE_PORT:-42400}"
SELFTEST="${SELFTEST:-16}"

if [[ ! -x "$BIN" || ! -x "$VALIDATE" ]]; then
  "$ROOT/scripts/build_two_party_dpf_keygen.sh"
fi

mkdir -p "$OUTDIR" "$WORKDIR"
: > "$LOG"
rm -f "$CSV" "$VCSV"

# config: log_domain trees modulus_idx
CONFIGS=(
  "4 8 0"
  "8 8 0"
  "11 4 0"
  "14 2 0"
  "8 8 1"
  "14 2 1"
)

port=$BASE_PORT
header_written=0
vheader_written=0
status=0

for cfg in "${CONFIGS[@]}"; do
  read -r L TREES MIDX <<<"$cfg"
  prefix="$WORKDIR/L${L}_m${MIDX}"
  rm -f "${prefix}_p0.key" "${prefix}_p1.key" \
        "${prefix}_p0.testmeta" "${prefix}_p1.testmeta"

  hdr=()
  if [[ $header_written -eq 0 ]]; then hdr=(--csv-header); fi

  echo "=== L=$L trees=$TREES modulus_idx=$MIDX port=$port ===" >> "$LOG"

  "$BIN" --party 0 --port "$port" --log-domain "$L" --trees "$TREES" \
         --modulus-idx "$MIDX" --selftest "$SELFTEST" \
         --input-seed "$((L * 100 + MIDX))" --out-prefix "$prefix" \
         "${hdr[@]}" > "$WORKDIR/p0.csv" 2>> "$LOG" &
  p0=$!
  # Party 1 connects; party 0 is the listener, so a short grace period avoids a
  # connect storm in the log.
  sleep 0.2
  "$BIN" --party 1 --host 127.0.0.1 --port "$port" --log-domain "$L" \
         --trees "$TREES" --modulus-idx "$MIDX" --selftest "$SELFTEST" \
         --input-seed "$((L * 100 + MIDX))" --out-prefix "$prefix" \
         > "$WORKDIR/p1.csv" 2>> "$LOG" &
  p1=$!

  rc0=0; rc1=0
  wait "$p0" || rc0=$?
  wait "$p1" || rc1=$?
  cat "$WORKDIR/p0.csv" "$WORKDIR/p1.csv" | tee -a "$LOG" >> "$CSV"
  header_written=1
  if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    echo "[run] keygen FAILED (party0=$rc0 party1=$rc1) for L=$L m=$MIDX" | tee -a "$LOG"
    status=1
  fi

  vhdr=()
  if [[ $vheader_written -eq 0 ]]; then vhdr=(--csv-header); fi
  if ! "$VALIDATE" --prefix "$prefix" "${vhdr[@]}" \
        >> "$VCSV" 2>> "$LOG"; then
    echo "[run] validation FAILED for L=$L m=$MIDX" | tee -a "$LOG"
    status=1
  fi
  vheader_written=1
  tail -1 "$VCSV" >> "$LOG"

  port=$((port + 4))
done

echo "wrote $CSV"
echo "wrote $VCSV"
echo "wrote $LOG"

if [[ $status -ne 0 ]]; then
  echo "[two-party-dpf] FAILURES present"
  exit 1
fi

if grep -q "FAIL" "$CSV" || grep -q "FAIL" "$VCSV"; then
  echo "[two-party-dpf] FAIL marker in CSV"
  exit 1
fi

echo "[two-party-dpf] all configurations pass"
