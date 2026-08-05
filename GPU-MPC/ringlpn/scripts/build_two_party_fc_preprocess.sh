#!/usr/bin/env bash
# Build the live two-process Ring-LPN -> Orca forward-FC preprocessing artifact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
OUT_DIR="$ROOT/bin"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
LINEAR_KIND="${RINGLPN_LINEAR_KIND:-fc}"
if [[ "$LINEAR_KIND" != "fc" && "$LINEAR_KIND" != "conv" ]]; then
  echo "RINGLPN_LINEAR_KIND must be fc or conv" >&2
  exit 2
fi
SOURCE="$ROOT/src/test_two_party_${LINEAR_KIND}_preprocess.cu"
OUTPUT="$OUT_DIR/test_two_party_${LINEAR_KIND}_preprocess"

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
  "$SOURCE" \
  "$ROOT/src/secure_convert.cpp" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  "$ROOT/src/orca_globals_stub.cpp" \
  -lcurand -lcrypto -lssl -ldl -lpthread \
  -o "$OUTPUT"

echo "Built $OUTPUT"
