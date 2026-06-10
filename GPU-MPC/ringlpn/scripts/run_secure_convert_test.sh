#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/host_bin/test_secure_convert"
OUT_DIR="$ROOT/results/secure_convert"
CSV="$OUT_DIR/secure_convert_prototype.csv"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "secure-convert test not built. Run scripts/build_secure_convert_test.sh first."
  exit 1
fi

# qbits bw trials forced_wraps inner value_bound seed
CASES=(
  "64 16 4000 512 8 255 1"
  "64 24 4000 512 8 4095 2"
  "128 16 4000 512 8 255 3"
  "128 32 4000 512 16 65535 4"
)

rm -f "$CSV"
header_written=0
for case in "${CASES[@]}"; do
  read -r qbits bw trials fw inner vb seed <<<"$case"
  if [[ "$header_written" -eq 0 ]]; then
    "$BIN" --qbits "$qbits" --bw "$bw" --trials "$trials" --forced-wraps "$fw" \
      --inner "$inner" --value-bound "$vb" --seed "$seed" --csv-header >> "$CSV"
    header_written=1
  else
    "$BIN" --qbits "$qbits" --bw "$bw" --trials "$trials" --forced-wraps "$fw" \
      --inner "$inner" --value-bound "$vb" --seed "$seed" >> "$CSV"
  fi
done

echo "Wrote $CSV"
cat "$CSV"
