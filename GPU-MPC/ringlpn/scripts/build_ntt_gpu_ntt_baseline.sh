#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"
OUT_DIR="$BASE_DIR/bin"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"

# Benchmark-only external dependency (not vendored): the GPU-NTT library of
# Ozcan-Savas (https://github.com/Alisah-Ozcan/GPU-NTT, eprint 2023/1410),
# built with: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#             -DCMAKE_CUDA_ARCHITECTURES=<arch> && cmake --build build
GPU_NTT_HOME="${GPU_NTT_HOME:-/home/fatih/GPU-NTT}"
GPU_NTT_LIB="$GPU_NTT_HOME/build/src/libntt-1.0.a"

mkdir -p "$OUT_DIR"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run this inside the CUDA container/toolkit environment."
  exit 1
fi

if [[ ! -f "$GPU_NTT_LIB" ]]; then
  echo "GPU-NTT not built at $GPU_NTT_HOME (expected $GPU_NTT_LIB)."
  echo "Clone/build it or set GPU_NTT_HOME. This baseline is optional and is"
  echo "not part of the paper-checkpoint gate."
  exit 1
fi

"$NVCC" -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
  -I"$PROJECT_ROOT" \
  -I"$GPU_NTT_HOME/src/include" \
  "$BASE_DIR/src/bench_ntt_gpu_ntt_baseline.cu" \
  "$GPU_NTT_LIB" \
  -o "$OUT_DIR/bench_ntt_gpu_ntt_baseline"

echo "Built:"
ls -la "$OUT_DIR/bench_ntt_gpu_ntt_baseline"
