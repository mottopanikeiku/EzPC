#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_vole_ringlpn"
OUT_DIR="$BASE_DIR/results"
QBITS="${QBITS:-32}"
M="${M:-32}"
C="${C:-2}"
NOISE_WEIGHT="${NOISE_WEIGHT:-64}"
OUT_TAG="${OUT_TAG:-q${QBITS}_m${M}_c${C}_w${NOISE_WEIGHT}}"
CSV="$OUT_DIR/vole_gpu_${OUT_TAG}.csv"
MD="$OUT_DIR/vole_gpu_${OUT_TAG}.md"

mkdir -p "$OUT_DIR"

if [[ "$QBITS" != "32" && "$QBITS" != "64" ]]; then
  echo "Unsupported QBITS=$QBITS. Expected 32 or 64."
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "bench_vole_ringlpn not built. Run scripts/build_vole_bench.sh first."
  exit 1
fi

N_LIST=(8192 16384 32768 65536 131072 262144 524288 1048576)

choose_schedule() {
  local n="$1"
  if (( n <= 32768 )); then
    echo "200 20"
    return
  fi

  if (( n <= 131072 )); then
    echo "100 10"
    return
  fi

  if (( n <= 262144 )); then
    echo "40 5"
    return
  fi

  if (( n <= 524288 )); then
    echo "20 3"
    return
  fi

  echo "10 2"
}

rm -f "$CSV" "$MD"

header_written=0

for n in "${N_LIST[@]}"; do
  read -r iters warmup < <(choose_schedule "$n")

  extra_args=()
  if [[ "$header_written" -eq 0 ]]; then
    extra_args+=(--csv-header)
  fi

  output=$("$BIN" \
    --n "$n" \
    --qbits "$QBITS" \
    --m "$M" \
    --c "$C" \
    --noise-weight "$NOISE_WEIGHT" \
    --iters "$iters" \
    --warmup "$warmup" \
    "${extra_args[@]}" 2>&1)

  printf '%s\n' "$output" >> "$CSV"
  header_written=1
done

python3 "$BASE_DIR/scripts/summarize_vole_results.py" --csv "$CSV" --out-md "$MD"

printf "\nWrote %s and %s\n" "$CSV" "$MD"