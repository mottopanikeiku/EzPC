#!/usr/bin/env bash
# Build the live two-process Ring-LPN -> Orca forward-FC preprocessing artifact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
OUT_DIR="$ROOT/bin"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"

mkdir -p "$OUT_DIR"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run inside the CUDA toolkit environment." >&2
  exit 1
fi

COMMON_FLAGS=(
  -O2
  -std=c++17
  -arch="sm_${CUDA_ARCH}"
  -diag-suppress=20012
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread
  -I"$PROJECT_ROOT"
  -I"$PROJECT_ROOT/ext/cutlass/include"
  -I"$PROJECT_ROOT/ext/cutlass/tools/util/include"
  -I"$PROJECT_ROOT/ext/sytorch/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools"
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack"
  -I"$SCI_SRC"
  -I"$ROOT/src"
)

"$NVCC" "${COMMON_FLAGS[@]}" \
  "$ROOT/src/test_two_party_fc_preprocess.cu" \
  "$ROOT/src/secure_convert.cpp" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  "$ROOT/src/orca_globals_stub.cpp" \
  -lcurand -lcrypto -lssl -lpthread \
  -o "$OUT_DIR/test_two_party_fc_preprocess"

echo "Built $OUT_DIR/test_two_party_fc_preprocess"
