#!/usr/bin/env bash
# Build the exact known-zero ResNet18 Conv0->TR->MaxPool->ReLU->Conv3 control.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
OUT_DIR="$ROOT/bin"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
OUTPUT="$OUT_DIR/test_resnet18_graph_prefix"
KEYGEN_OUTPUT="$OUT_DIR/test_stock_nonlinear_prefix_keygen"
SYTORCH_BUILD="$PROJECT_ROOT/ext/sytorch/build"
mkdir -p "$OUT_DIR"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run inside the CUDA toolkit environment." >&2
  exit 1
fi
"$NVCC" \
  -O2 \
  -std=c++17 \
  -arch="sm_${CUDA_ARCH}" \
  -diag-suppress=20012 \
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread \
  -I"$PROJECT_ROOT" \
  -I"$PROJECT_ROOT/ext/cutlass/include" \
  -I"$PROJECT_ROOT/ext/cutlass/tools/util/include" \
  -I"$PROJECT_ROOT/ext/sytorch/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack" \
  -I"$SCI_SRC" \
  -I"$ROOT/src" \
  "$ROOT/src/test_stock_nonlinear_prefix_keygen.cu" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  "$ROOT/src/orca_globals_stub.cpp" \
  -lcurand -lcrypto -lssl -ldl -lpthread \
  -o "$KEYGEN_OUTPUT"

"$NVCC" \
  -O2 \
  -std=c++17 \
  -arch="sm_${CUDA_ARCH}" \
  -diag-suppress=20012 \
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread \
  -I"$PROJECT_ROOT" \
  -I"$PROJECT_ROOT/ext/cutlass/include" \
  -I"$PROJECT_ROOT/ext/cutlass/tools/util/include" \
  -I"$PROJECT_ROOT/ext/sytorch/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack" \
  -I"$SCI_SRC" \
  -I"$ROOT/src" \
  "$ROOT/src/test_resnet18_graph_prefix.cu" \
  "$ROOT/src/linear_preprocess_conv.cu" \
  "$ROOT/src/secure_convert.cpp" \
  "$ROOT/src/secure_truncate.cpp" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  "$PROJECT_ROOT/utils/gpu_file_utils.cpp" \
  "$PROJECT_ROOT/utils/sigma_comms.cpp" \
  -L"$SYTORCH_BUILD" \
  -L"$SYTORCH_BUILD/ext/cryptoTools" \
  -L"$SYTORCH_BUILD/ext/llama" \
  -L"$SYTORCH_BUILD/ext/bitpack" \
  -L"$SYTORCH_BUILD/lib" \
  -lsytorch -lcryptoTools -lLLAMA -lbitpack \
  -lcurand -lcrypto -lssl -ldl -lpthread \
  -o "$OUTPUT"

echo "Built $KEYGEN_OUTPUT"
echo "Built $OUTPUT"
