#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_orca_fc_ringlpn_demo"
OUT_DIR="$BASE_DIR/results"

OUT_TAG="${OUT_TAG:-bounded_suite}"

CASES=(
  "64 2 2 2 16 255 1 2"
  "64 2 3 2 16 255 3 4"
  "64 3 2 2 16 255 5 6"
  "64 2 2 3 32 255 7 8"
  "128 2 2 2 32 4294967295 9 10"
)

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Orca FC Ring-LPN demo binary not built. Run scripts/build_orca_fc_ringlpn_demo.sh first."
  exit 1
fi

CSV="$OUT_DIR/orca_fc_ringlpn_demo_${OUT_TAG}.csv"
MD="$OUT_DIR/orca_fc_ringlpn_demo_${OUT_TAG}.md"
LOG="$OUT_DIR/orca_fc_ringlpn_demo_${OUT_TAG}.log"

rm -f "$CSV" "$MD" "$LOG"

header_cols=0
header_written=0

for case in "${CASES[@]}"; do
  read -r qbits rows inner cols bw value_bound seed second_seed <<<"$case"
  output=$("$BIN" \
    --qbits "$qbits" \
    --rows "$rows" \
    --inner "$inner" \
    --cols "$cols" \
    --bw "$bw" \
    --value-bound "$value_bound" \
    --seed "$seed" \
    --second-seed "$second_seed" \
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

python3 "$BASE_DIR/scripts/summarize_orca_fc_demo.py" --csv "$CSV" --out-md "$MD"

printf "\nWrote %s and %s (stderr log in %s)\n" "$CSV" "$MD" "$LOG"
