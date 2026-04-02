#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_ntt_cuda"
OUT_DIR="$BASE_DIR/results"
QBITS="${QBITS:-32}"
OUT_TAG="${OUT_TAG:-q${QBITS}}"
CSV="$OUT_DIR/ntt_gpu_${OUT_TAG}.csv"
UNSUPPORTED_CSV="$OUT_DIR/ntt_gpu_${OUT_TAG}_unsupported.csv"
MD="$OUT_DIR/ntt_gpu_${OUT_TAG}.md"

mkdir -p "$OUT_DIR"

if [[ "$QBITS" != "32" && "$QBITS" != "64" ]]; then
  echo "Unsupported QBITS=$QBITS. Expected 32 or 64."
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "bench_ntt_cuda not built. Run scripts/build_cuda_bench.sh first."
  exit 1
fi

N_LIST=(8192 16384 32768 65536 131072 262144 524288 1048576)

choose_schedule() {
  local n="$1"
  if (( n <= 32768 )); then
    echo "400 40 64"
    return
  fi

  if (( n <= 131072 )); then
    echo "200 20 16"
    return
  fi

  if (( n <= 262144 )); then
    echo "80 10 8"
    return
  fi

  if (( n <= 524288 )); then
    echo "30 5 4"
    return
  fi

  echo "10 2 2"
}

compute_logn() {
  local n="$1"
  local logn=0
  while (( (1 << logn) < n )); do
    ((logn += 1))
  done
  echo "$logn"
}

printf 'n,logn,requested_qbits,status,reason\n' > "$UNSUPPORTED_CSV"
rm -f "$CSV" "$MD"

header_written=0

for n in "${N_LIST[@]}"; do
  read -r iters warmup batch < <(choose_schedule "$n")

  extra_args=()
  if [[ "$header_written" -eq 0 ]]; then
    extra_args+=(--csv-header)
  fi

  if output=$("$BIN" --n "$n" --qbits "$QBITS" --batch "$batch" --iters "$iters" --warmup "$warmup" "${extra_args[@]}" 2>&1); then
    printf '%s\n' "$output" >> "$CSV"
    header_written=1
    continue
  else
    status=$?
  fi

  if [[ "$status" -eq 2 ]]; then
    reason="${output//$'\n'/ }"
    reason="${reason//,/;}"
    printf '%s,%s,%s,%s,%s\n' "$n" "$(compute_logn "$n")" "$QBITS" "unsupported" "$reason" >> "$UNSUPPORTED_CSV"
    continue
  fi

  printf '%s\n' "$output" >&2
  exit "$status"
done

python3 "$BASE_DIR/scripts/summarize_cuda_results.py" --csv "$CSV" --unsupported-csv "$UNSUPPORTED_CSV" --out-md "$MD"

printf "\nWrote %s, %s, and %s\n" "$CSV" "$UNSUPPORTED_CSV" "$MD"