#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_ntt_gpu_ntt_baseline"
OUT_DIR="$BASE_DIR/results/ntt"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Baseline-compare binary not built. Run scripts/build_ntt_gpu_ntt_baseline.sh"
  echo "(requires the external GPU-NTT checkout; see that script's header)."
  exit 1
fi

CSV="$OUT_DIR/ntt_gpu_ntt_baseline_compare.csv"
LOG="$OUT_DIR/ntt_gpu_ntt_baseline_compare.log"

rm -f "$CSV" "$LOG"

# prime n batch iters warmup
CASES=(
  "pool60 8192 4 400 50"
  "pool60 8192 64 400 50"
  "pool60 65536 4 200 20"
  "pool60 65536 64 100 10"
  "pool60 1048576 2 50 5"
  "p62 8192 4 400 50"
  "p62 8192 64 400 50"
)

header_written=0
for case in "${CASES[@]}"; do
  read -r prime n batch iters warmup <<<"$case"
  extra=()
  if [[ "$header_written" -eq 0 ]]; then
    extra+=(--csv-header)
  fi
  output=$("$BIN" --prime "$prime" --n "$n" --batch "$batch" --iters "$iters" \
           --warmup "$warmup" "${extra[@]}" 2>>"$LOG")
  printf '%s\n' "$output" >> "$LOG"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$header_written" -eq 1 && "$line" == device,* ]]; then
      continue
    fi
    printf '%s\n' "$line" >> "$CSV"
    [[ "$line" == device,* ]] && header_written=1
  done <<<"$output"
done

printf "\nWrote %s (log in %s)\n" "$CSV" "$LOG"
cat "$CSV"
