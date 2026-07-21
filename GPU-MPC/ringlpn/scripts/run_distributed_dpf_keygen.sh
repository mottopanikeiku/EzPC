#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/host_bin/test_distributed_dpf_keygen"
OUT_DIR="$ROOT/results/dpf"
CSV="$OUT_DIR/distributed_dpf_keygen_prototype.csv"
LOG="$OUT_DIR/distributed_dpf_keygen_prototype.log"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "distributed-dpf-keygen test not built. Run scripts/build_distributed_dpf_keygen.sh first."
  exit 1
fi

# log_domain trees modulus_idx seed
# L=14 is the production point (domain 2n = 16384 for n = 8192); both CRT primes.
CASES=(
  "4 512 0 11"
  "8 512 0 12"
  "11 384 0 13"
  "14 256 0 14"
  "14 256 1 15"
  "8 512 1 16"
)

rm -f "$CSV" "$LOG"
header_written=0
status=0
for case in "${CASES[@]}"; do
  read -r ld trees midx seed <<<"$case"
  extra=()
  if [[ "$header_written" -eq 0 ]]; then
    extra+=(--csv-header)
    header_written=1
  fi
  if ! "$BIN" --log-domain "$ld" --trees "$trees" --modulus-idx "$midx" \
       --seed "$seed" "${extra[@]}" >> "$CSV" 2>> "$LOG"; then
    status=1
  fi
done

echo "Wrote $CSV"
cat "$CSV"
if [[ "$status" -ne 0 ]]; then
  echo "[distributed-dpf] AT LEAST ONE CASE FAILED"
  exit 1
fi
echo "[distributed-dpf] all cases pass"
