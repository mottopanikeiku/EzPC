#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_orca_fc_real_ole_transcript"
OUT_DIR="$BASE_DIR/results/orca_fc"

OUT_TAG="${OUT_TAG:-transcript_suite}"

# qbits rows inner cols bw noise seed
# Slot packing requires rows*inner*cols <= ole_n (8192). The no-wrap bound is
# K * 2^(2*bw+2) < M, so bw=16 works at q64 and bw=32 at q128 (two CRT limbs).
# The 16x32x16 case uses every slot of the ring OLE pair: 2 ring OLE instances
# back 8192 scalar cross terms per direction.
CASES=(
  "64 2 2 2 16 uniform 1"
  "64 4 4 4 16 uniform 3"
  "64 8 8 8 16 uniform 5"
  "64 16 32 16 16 uniform 7"
  "64 4 4 4 16 regular 15"
  "128 2 2 2 32 uniform 9"
  "128 4 4 4 32 uniform 11"
  "128 16 16 16 32 uniform 13"
  "128 4 4 4 32 regular 17"
)

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Real-OLE transcript binary not built. Run scripts/build_orca_fc_real_ole_transcript.sh first."
  exit 1
fi

CSV="$OUT_DIR/orca_fc_real_ole_transcript_${OUT_TAG}.csv"
LOG="$OUT_DIR/orca_fc_real_ole_transcript_${OUT_TAG}.log"

rm -f "$CSV" "$LOG"

header_cols=0
header_written=0

for case in "${CASES[@]}"; do
  read -r qbits rows inner cols bw noise seed <<<"$case"
  output=$("$BIN" \
    --qbits "$qbits" \
    --rows "$rows" \
    --inner "$inner" \
    --cols "$cols" \
    --bw "$bw" \
    --noise "$noise" \
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
