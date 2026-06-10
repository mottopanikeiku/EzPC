#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_ole_ringlpn_cuda"
TEST_BIN="$BASE_DIR/bin/test_spfss_zp_cuda"
OUT_DIR="$BASE_DIR/results/ole"

QBITS="${QBITS:-64}"
C="${C:-2}"
T="${T:-64}"
NOISE="${NOISE:-uniform}"
CHUNK_SIZE="${CHUNK_SIZE:-8192}"
SMOKE="${SMOKE:-0}"

mkdir -p "$OUT_DIR"

if [[ "$QBITS" != "64" && "$QBITS" != "128" ]]; then
  echo "Unsupported QBITS=$QBITS. Supported: 64 (single q62 limb) or 128 (two q62 CRT limbs)."
  exit 1
fi

if [[ "$NOISE" != "uniform" && "$NOISE" != "regular" ]]; then
  echo "Unsupported NOISE=$NOISE. Expected uniform or regular."
  exit 1
fi

if [[ ! -x "$BIN" || ! -x "$TEST_BIN" ]]; then
  echo "OLE CUDA binaries not built. Run scripts/build_ole_cuda_bench.sh first."
  exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
  N_LIST=(8192)
  T="${SMOKE_T:-8}"
else
  N_LIST=(8192 16384)
fi

if [[ "$SMOKE" == "1" ]]; then
  OUT_TAG="${OUT_TAG:-q${QBITS}_${NOISE}_c${C}_t${T}_smoke}"
else
  OUT_TAG="${OUT_TAG:-q${QBITS}_${NOISE}_c${C}_t${T}}"
fi
CSV="$OUT_DIR/ole_gpu_${OUT_TAG}.csv"
MD="$OUT_DIR/ole_gpu_${OUT_TAG}.md"
LOG="$OUT_DIR/ole_gpu_${OUT_TAG}.log"

choose_schedule() {
  local n="$1"
  if [[ "$SMOKE" == "1" ]]; then
    echo "1 0"
    return
  fi
  if (( n <= 8192 )); then
    echo "2 1"
    return
  fi
  echo "1 0"
}

rm -f "$CSV" "$MD" "$LOG"

"$TEST_BIN" >>"$LOG" 2>&1

header_cols=0
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
    --c "$C" \
    --t "$T" \
    --noise "$NOISE" \
    --chunk-size "$CHUNK_SIZE" \
    --iters "$iters" \
    --warmup "$warmup" \
    "${extra_args[@]}" 2>>"$LOG")

  printf '%s\n' "$output" >> "$LOG"

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
done

python3 "$BASE_DIR/scripts/summarize_ole_results.py" --csv "$CSV" --out-md "$MD"

printf "\nWrote %s and %s (stderr/test log in %s)\n" "$CSV" "$MD" "$LOG"
