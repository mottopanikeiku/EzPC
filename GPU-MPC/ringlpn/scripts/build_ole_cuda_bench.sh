#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"
OUT_DIR="$BASE_DIR/bin"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
DEVICE_LABEL="${DEVICE_LABEL:-cuda_ringlpn_ole}"

mkdir -p "$OUT_DIR"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run this inside the CUDA container/toolkit environment."
  exit 1
fi

COMMON_FLAGS=(
  -O3
  -std=c++17
  -arch="sm_${CUDA_ARCH}"
  -I"$PROJECT_ROOT"
  -I"$PROJECT_ROOT/ext/sytorch/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools"
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack"
)

"$NVCC" "${COMMON_FLAGS[@]}" \
  -DRINGLPN_DEVICE_LABEL="\"${DEVICE_LABEL}\"" \
  "$BASE_DIR/src/bench_ole_ringlpn_cuda.cu" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  -o "$OUT_DIR/bench_ole_ringlpn_cuda"

"$NVCC" "${COMMON_FLAGS[@]}" \
  -DRINGLPN_DEVICE_LABEL="\"${DEVICE_LABEL}_party\"" \
  "$BASE_DIR/src/bench_ole_ringlpn_party.cu" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  -o "$OUT_DIR/bench_ole_ringlpn_party"

"$NVCC" "${COMMON_FLAGS[@]}" \
  "$BASE_DIR/src/test_spfss_zp_cuda.cu" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  -o "$OUT_DIR/test_spfss_zp_cuda"

echo "Built:"
ls -la "$OUT_DIR/bench_ole_ringlpn_cuda" \
  "$OUT_DIR/bench_ole_ringlpn_party" \
  "$OUT_DIR/test_spfss_zp_cuda"
