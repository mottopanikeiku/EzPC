#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$BASE_DIR/bin"
SRC="$BASE_DIR/src/bench_ntt_cuda.cu"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
DEVICE_LABEL="${DEVICE_LABEL:-cuda_legacy}"

mkdir -p "$OUT_DIR"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run this inside the CUDA container/toolkit environment."
  exit 1
fi

if [[ "${ALLOW_LEGACY_CUDA_NTT:-0}" != "1" ]]; then
  echo "The legacy CUDA NTT benchmark is archived and is no longer part of the active Ring-LPN GPU pipeline."
  echo "Use scripts/build_cuda_bench.sh for the Cheddar backend, or set ALLOW_LEGACY_CUDA_NTT=1 for historical comparison."
  exit 1
fi

"$NVCC" -O3 -std=c++14 -arch="sm_${CUDA_ARCH}" -DRINGLPN_DEVICE_LABEL="\"${DEVICE_LABEL}\"" "$SRC" -o "$OUT_DIR/bench_ntt_cuda_legacy"
