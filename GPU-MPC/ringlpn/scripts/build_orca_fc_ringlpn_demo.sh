#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"
OUT_DIR="$BASE_DIR/bin"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"

mkdir -p "$OUT_DIR"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run this inside the CUDA container/toolkit environment."
  exit 1
fi

COMMON_FLAGS=(
  -O3
  -std=c++17
  -arch="sm_${CUDA_ARCH}"
  -Xcompiler="-fpermissive"
  -I"$PROJECT_ROOT"
  -I"$PROJECT_ROOT/ext/cutlass/include"
  -I"$PROJECT_ROOT/ext/cutlass/tools/util/include"
  -I"$PROJECT_ROOT/ext/sytorch/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools"
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack"
)

"$NVCC" "${COMMON_FLAGS[@]}" \
  "$BASE_DIR/src/bench_orca_fc_ringlpn_demo.cu" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  -lcurand \
  -o "$OUT_DIR/bench_orca_fc_ringlpn_demo"

echo "Built:"
ls -la "$OUT_DIR/bench_orca_fc_ringlpn_demo"
