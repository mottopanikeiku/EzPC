#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_orca_fc_ideal_ole_transcript"
OUT_DIR="$BASE_DIR/results"

OUT_TAG="${OUT_TAG:-transcript_suite}"

# qbits rows inner cols bw value_bound seed
# Single q62 limb (qbits=64). The conservative no-wrap bound is
# K * 2^(2*bw+2) < p62, so bw stays small for these shapes.
CASES=(
  "64 2 2 2 16 255 1"
  "64 2 3 2 16 255 3"
  "64 3 2 2 16 255 5"
  "64 4 4 4 16 255 7"
  "64 2 2 2 20 255 9"
)

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Ideal-OLE transcript binary not built. Run scripts/build_orca_fc_ideal_ole_transcript.sh first."
  exit 1
fi

CSV="$OUT_DIR/orca_fc_ideal_ole_transcript_${OUT_TAG}.csv"
LOG="$OUT_DIR/orca_fc_ideal_ole_transcript_${OUT_TAG}.log"

rm -f "$CSV" "$LOG"

header_cols=0
header_written=0

for case in "${CASES[@]}"; do
  read -r qbits rows inner cols bw value_bound seed <<<"$case"
  output=$("$BIN" \
    --qbits "$qbits" \
    --rows "$rows" \
    --inner "$inner" \
    --cols "$cols" \
    --bw "$bw" \
    --value-bound "$value_bound" \
    --seed "$seed" \
    --csv-header 2>>"$LOG")

  printf '%s\n' "$output" >> "$LOG"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" == reserved\ memory:* ]] && continue
    if [[ "$header_written" -eq 1 && "$line" == device,* ]]; then
      continue
    fi
    cols_n=$(awk -F',' '{print NF}' <<<"$line")
    if [[ "$header_written" -eq 0 ]]; then
      header_cols="$cols_n"
      printf '%s\n' "$line" >> "$CSV"
      header_written=1
      continue
    fi
    if [[ "$cols_n" -ne "$header_cols" ]]; then
      printf '%s\n' "$line" >> "$LOG"
      continue
    fi
    printf '%s\n' "$line" >> "$CSV"
  done <<<"$output"
done

printf "\nWrote %s (stderr log in %s)\n" "$CSV" "$LOG"
cat "$CSV"
