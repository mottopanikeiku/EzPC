#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_orca_fc_ringlpn_demo"
OUT_DIR="$BASE_DIR/results"

SEED="${SEED:-1}"
SECOND_SEED="${SECOND_SEED:-2}"
OUT_TAG="${OUT_TAG:-seed${SEED}_seed${SECOND_SEED}}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Orca FC Ring-LPN demo binary not built. Run scripts/build_orca_fc_ringlpn_demo.sh first."
  exit 1
fi

CSV="$OUT_DIR/orca_fc_ringlpn_demo_${OUT_TAG}.csv"
MD="$OUT_DIR/orca_fc_ringlpn_demo_${OUT_TAG}.md"
LOG="$OUT_DIR/orca_fc_ringlpn_demo_${OUT_TAG}.log"

rm -f "$CSV" "$MD" "$LOG"

output=$("$BIN" --seed "$SEED" --second-seed "$SECOND_SEED" --csv-header 2>>"$LOG")
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

python3 "$BASE_DIR/scripts/summarize_orca_fc_demo.py" --csv "$CSV" --out-md "$MD"

printf "\nWrote %s and %s (stderr log in %s)\n" "$CSV" "$MD" "$LOG"
