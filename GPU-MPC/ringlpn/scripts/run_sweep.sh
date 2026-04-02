#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_ntt"
OUT_DIR="$BASE_DIR/results"
CSV="$OUT_DIR/ntt_cpu.csv"
UNSUPPORTED_CSV="$OUT_DIR/ntt_cpu_unsupported.csv"
MD="$OUT_DIR/ntt_cpu.md"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "bench_ntt not built. Run scripts/build_bench.sh first."
  exit 1
fi

N_LIST=(8192 16384 32768 65536 131072 262144 524288 1048576)
Q_BITS=(32 64 128)

choose_schedule() {
  local n="$1"
  if (( n <= 32768 )); then
    echo "800 80"
    return
  fi

  if (( n <= 131072 )); then
    echo "200 20"
    return
  fi

  if (( n <= 524288 )); then
    echo "50 5"
    return
  fi

  echo "12 2"
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
  for q in "${Q_BITS[@]}"; do
    read -r iters warmup < <(choose_schedule "$n")

    extra_args=()
    if [[ "$header_written" -eq 0 ]]; then
      extra_args+=(--csv-header)
    fi

    if output=$("$BIN" --n "$n" --qbits "$q" --iters "$iters" --warmup "$warmup" "${extra_args[@]}" 2>&1); then
      printf '%s\n' "$output" >> "$CSV"
      header_written=1
      continue
    else
      status=$?
    fi

    if [[ "$status" -eq 2 ]]; then
      reason="${output//$'\n'/ }"
      reason="${reason//,/;}"
      printf '%s,%s,%s,%s,%s\n' "$n" "$(compute_logn "$n")" "$q" "unsupported" "$reason" >> "$UNSUPPORTED_CSV"
      continue
    fi

    printf '%s\n' "$output" >&2
    exit "$status"
  done
done

python3 "$BASE_DIR/scripts/summarize_results.py" --csv "$CSV" --unsupported-csv "$UNSUPPORTED_CSV" --out-md "$MD"

printf "\nWrote %s, %s, and %s\n" "$CSV" "$UNSUPPORTED_CSV" "$MD"
