#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_CPU="$BASE_DIR/bin/bench_ntt"
BIN_GPU="$BASE_DIR/bin/bench_ntt_cuda"
OUT_DIR="$BASE_DIR/results"
N="${1:-8192}"
BATCH="${2:-1}"

if [[ "$N" != "8192" && "$N" != "16384" && "$N" != "32768" ]]; then
  echo "Usage: $0 [8192|16384|32768] [batch]"
  exit 1
fi

if ! [[ "$BATCH" =~ ^[0-9]+$ ]] || (( BATCH <= 0 )); then
  echo "Batch must be a positive integer."
  exit 1
fi

CPU_CSV="$OUT_DIR/cpu_${N}_32.csv"
GPU_CSV="$OUT_DIR/gpu_${N}_32_batch${BATCH}.csv"
OUT_MD="$OUT_DIR/cpu_gpu_${N}_32_batch${BATCH}.md"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN_CPU" ]]; then
  echo "CPU benchmark missing. Run ./scripts/build_bench.sh first."
  exit 1
fi

if [[ ! -x "$BIN_GPU" ]]; then
  echo "GPU benchmark missing. Run ./scripts/build_cuda_bench.sh first."
  exit 1
fi

"$BIN_CPU" --csv-header --n "$N" --qbits 32 --iters 10000 --warmup 1000 > "$CPU_CSV"
"$BIN_GPU" --csv-header --n "$N" --qbits 32 --batch "$BATCH" --iters 10000 --warmup 1000 > "$GPU_CSV"

python3 "$BASE_DIR/scripts/summarize_cpu_gpu_4096.py" \
  --cpu-csv "$CPU_CSV" \
  --gpu-csv "$GPU_CSV" \
  --out-md "$OUT_MD"

printf "\nWrote %s\n" "$OUT_MD"
