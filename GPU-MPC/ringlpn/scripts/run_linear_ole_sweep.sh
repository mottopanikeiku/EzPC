#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_linear_ole_ringlpn_cuda"
OUT_DIR="$BASE_DIR/results/linear_ole"

QBITS="${QBITS:-64}"
N="${N:-8192}"
ROWS="${ROWS:-2}"
INNER="${INNER:-2}"
COLS="${COLS:-2}"
C="${C:-2}"
T="${T:-8}"
NOISE="${NOISE:-uniform}"
CHUNK_SIZE="${CHUNK_SIZE:-8192}"
ITERS="${ITERS:-1}"
WARMUP="${WARMUP:-0}"
SEED="${SEED:-1}"
OUT_TAG="${OUT_TAG:-q${QBITS}_${NOISE}_r${ROWS}_k${INNER}_c${COLS}_n${N}_t${T}}"

mkdir -p "$OUT_DIR"

if [[ "$QBITS" != "64" && "$QBITS" != "128" ]]; then
  echo "Unsupported QBITS=$QBITS. Supported: 64 (single q62 limb) or 128 (two q62 CRT limbs)."
  exit 1
fi

if [[ "$NOISE" != "uniform" && "$NOISE" != "regular" ]]; then
  echo "Unsupported NOISE=$NOISE. Expected uniform or regular."
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Linear OLE binary not built. Run scripts/build_linear_ole_bench.sh first."
  exit 1
fi

CSV="$OUT_DIR/linear_ole_gpu_${OUT_TAG}.csv"
MD="$OUT_DIR/linear_ole_gpu_${OUT_TAG}.md"
LOG="$OUT_DIR/linear_ole_gpu_${OUT_TAG}.log"

rm -f "$CSV" "$MD" "$LOG"

output=$("$BIN" \
  --n "$N" \
  --qbits "$QBITS" \
  --rows "$ROWS" \
  --inner "$INNER" \
  --cols "$COLS" \
  --c "$C" \
  --t "$T" \
  --noise "$NOISE" \
  --chunk-size "$CHUNK_SIZE" \
  --iters "$ITERS" \
  --warmup "$WARMUP" \
  --seed "$SEED" \
  --csv-header 2>>"$LOG")

printf '%s\n' "$output" >> "$LOG"

header_cols=0
header_written=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  [[ "$line" == reserved\ memory:* ]] && continue
  cols=$(awk -F',' '{print NF}' <<<"$line")
  if [[ "$header_written" -eq 0 ]]; then
    header_cols="$cols"
    printf '%s\n' "$line" >> "$CSV"
    header_written=1
    continue
  fi
  if [[ "$cols" -ne "$header_cols" ]]; then
    printf '%s\n' "$line" >> "$LOG"
    continue
  fi
  printf '%s\n' "$line" >> "$CSV"
done <<<"$output"

python3 "$BASE_DIR/scripts/summarize_linear_ole_results.py" --csv "$CSV" --out-md "$MD"

printf "\nWrote %s and %s (stderr log in %s)\n" "$CSV" "$MD" "$LOG"
