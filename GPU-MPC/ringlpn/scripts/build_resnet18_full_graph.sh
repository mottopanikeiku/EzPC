#!/usr/bin/env bash
# Build the exact known-zero full ResNet18 graph and trusted stock-key adapter.
set -euo pipefail

if [[ "${RINGLPN_COMPONENT_DISPATCH_ACTIVE:-0}" != "1" ]]; then
  exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_component.sh" resnet18-full-graph
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"
ringlpn_build_init
ROOT="$RINGLPN_ROOT"
OUT_DIR="$RINGLPN_OUT_DIR"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
CXX="${CXX:-g++}"
OBJCOPY="${OBJCOPY:-objcopy}"
PYTHON="${PYTHON:-python3}"
OUTPUT="$OUT_DIR/test_resnet18_full_graph"
KEYGEN_OUTPUT="$OUT_DIR/test_stock_nonlinear_full_keygen"
CONTRACT_OUTPUT="$OUT_DIR/test_resnet18_graph_contract"
PROVENANCE_OUTPUT="$OUT_DIR/resnet18_full_graph_build_provenance.json"
GRAPH_LIBRARY_BUILD="$RINGLPN_BUILD_DIR/graph-libraries"
GRAPH_LIBRARY_DIR="$GRAPH_LIBRARY_BUILD/lib"
LINEAR_LIBRARY_BUILD="$RINGLPN_BUILD_DIR/linear-library"
LINEAR_ARCHIVE="$LINEAR_LIBRARY_BUILD/lib/libringlpn_linear.a"
mkdir -p "$OUT_DIR"
ringlpn_require_command NVCC nvcc 'Run inside the CUDA toolkit environment.'
ringlpn_require_command CXX g++ 'Install a C++17 compiler.'
ringlpn_require_command OBJCOPY objcopy 'Install binutils.'
ringlpn_require_command PYTHON python3 'Install Python 3.'
CUDA_TOOLKIT_ROOT="$(dirname "$NVCC")/.."
CUDA_TARGET_TRIPLE="${CUDA_TARGET_TRIPLE:-x86_64-linux}"
CUDA_SYSTEM_INCLUDE="$CUDA_TOOLKIT_ROOT/targets/$CUDA_TARGET_TRIPLE/include"
GCC_SYSTEM_INCLUDE="$("$CXX" -print-file-name=include)"
ringlpn_require_linear_sources
for library in libcryptoTools.a libLLAMA.a libbitpack.a; do
  if [[ ! -f "$GRAPH_LIBRARY_DIR/$library" ]]; then
    echo "missing source-built graph library: $GRAPH_LIBRARY_DIR/$library" >&2
    echo "run scripts/build_component.sh graph-libraries first" >&2
    exit 1
  fi
done
if [[ ! -f "$GRAPH_LIBRARY_BUILD/compile_commands.json" ]]; then
  echo "missing graph-library compile database; rebuild graph-libraries" >&2
  exit 1
fi
if [[ ! -f "$LINEAR_ARCHIVE" ]]; then
  echo "missing deterministic linear facade archive: $LINEAR_ARCHIVE" >&2
  echo "run scripts/build_component.sh linear-library first" >&2
  exit 1
fi
KEEP_DIR="$OUT_DIR/.resnet18-full-graph-nvcc"
if ! mkdir "$KEEP_DIR"; then
  echo "stale or concurrent graph build directory: $KEEP_DIR" >&2
  exit 1
fi
trap 'rm -rf -- "$KEEP_DIR"' EXIT
KEEP_DIR_ARG="bin/.resnet18-full-graph-nvcc"

# NVCC salts anonymous-namespace IDs with the absolute source path. Fix that
# nonsemantic input, suppress path-sensitive build IDs, and remove only the
# host package-version comment after linking.
ringlpn_set_cuda_include_flags CUDA_INCLUDE_FLAGS
COMPILE_FLAGS=(
  -O2
  -std=c++17
  -arch="sm_${CUDA_ARCH}"
  -ccbin="$CXX"
  -Xcudafe=--orig_src_path_name=/ringlpn/reproducible-build-source.cu
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread,-frandom-seed=ringlpn-resnet18-full-graph-20260810
  "${CUDA_INCLUDE_FLAGS[@]}"
)
COMMON=(
  "${COMPILE_FLAGS[@]}"
  -Xlinker=--build-id=none
  --keep
  --keep-dir="$KEEP_DIR_ARG"
)
CONTRACT_FLAGS=(
  -O2 -std=c++17 -Wall -Wextra -Werror
  "-ffile-prefix-map=$RINGLPN_REPO_ROOT=/ringlpn/source"
  "-fdebug-prefix-map=$RINGLPN_REPO_ROOT=/ringlpn/source"
  -frandom-seed=ringlpn-resnet18-graph-contract-20260810
  -Wl,--build-id=none
  -I "$ROOT/src"
)
KEYGEN_SOURCES=(
  src/test_stock_nonlinear_full_keygen.cu
  ../utils/gpu_mem.cu
  src/orca_globals_stub.cpp
)
RUNTIME_SOURCES=(
  src/test_resnet18_full_graph.cu
  ../utils/gpu_file_utils.cpp
  ../utils/sigma_comms.cpp
)
KEYGEN_COMMAND=(
  "$NVCC" "${COMMON[@]}" "${KEYGEN_SOURCES[@]}"
  -lcurand -lcrypto -lssl -ldl -lpthread
  -o bin/test_stock_nonlinear_full_keygen
)
RUNTIME_COMMAND=(
  "$NVCC" "${COMMON[@]}" "${RUNTIME_SOURCES[@]}"
  "$LINEAR_ARCHIVE"
  "$GRAPH_LIBRARY_DIR/libLLAMA.a"
  "$GRAPH_LIBRARY_DIR/libcryptoTools.a"
  "$GRAPH_LIBRARY_DIR/libbitpack.a"
  -lcurand -lcrypto -lssl -ldl -lpthread
  -o bin/test_resnet18_full_graph
)
CONTRACT_COMMAND=(
  "$CXX" "${CONTRACT_FLAGS[@]}"
  "$ROOT/src/test_resnet18_graph_contract.cpp"
  -o bin/test_resnet18_graph_contract
)

record_command() {
  local label="$1"
  shift
  {
    printf '%s\0' "$ROOT"
    printf '%s\0' "$@"
  } >"$KEEP_DIR/$label.command"
}
generate_depfile() {
  local label="$1"
  local source="$2"
  (
    cd "$ROOT"
    "$NVCC" "${COMPILE_FLAGS[@]}" -MM -MT provenance \
      -MF "$KEEP_DIR/$label.d" "$source"
  )
}

for index in "${!KEYGEN_SOURCES[@]}"; do
  generate_depfile "adapter-$index" "${KEYGEN_SOURCES[$index]}"
done
for index in "${!RUNTIME_SOURCES[@]}"; do
  generate_depfile "runtime-$index" "${RUNTIME_SOURCES[$index]}"
done
(
  cd "$ROOT"
  "$CXX" "${CONTRACT_FLAGS[@]}" -MM -MT provenance \
    -MF "$KEEP_DIR/contract-0.d" \
    "$ROOT/src/test_resnet18_graph_contract.cpp"
  "${KEYGEN_COMMAND[@]}"
  "${RUNTIME_COMMAND[@]}"
  "${CONTRACT_COMMAND[@]}"
)
record_command adapter-link "${KEYGEN_COMMAND[@]}"
record_command runtime-link "${RUNTIME_COMMAND[@]}"
record_command contract-link "${CONTRACT_COMMAND[@]}"

for binary in "$KEYGEN_OUTPUT" "$OUTPUT"; do
  OBJCOPY_COMMAND=("$OBJCOPY" --remove-section .comment "$binary")
  "${OBJCOPY_COMMAND[@]}"
  record_command "objcopy-$(basename "$binary")" "${OBJCOPY_COMMAND[@]}"
done

PROVENANCE_COMMAND=(
  "$PYTHON" "$ROOT/scripts/graph_build_provenance.py" generate
  --repo-root "$RINGLPN_REPO_ROOT"
  --ringlpn-root "$ROOT"
  --cmake-build "$GRAPH_LIBRARY_BUILD"
  --output "$PROVENANCE_OUTPUT"
  --system-root "cuda-toolkit-include=$CUDA_SYSTEM_INCLUDE"
  --system-root "gcc-internal-include=$GCC_SYSTEM_INCLUDE"
  --system-root "system-usr-include=/usr/include"
  --linked-archive "ringlpn_linear=$LINEAR_ARCHIVE"
  --archive "cryptoTools=$GRAPH_LIBRARY_DIR/libcryptoTools.a"
  --archive "bitpack=$GRAPH_LIBRARY_DIR/libbitpack.a"
  --archive "LLAMA=$GRAPH_LIBRARY_DIR/libLLAMA.a"
  --artifact "test_stock_nonlinear_full_keygen=$KEYGEN_OUTPUT"
  --artifact "test_resnet18_full_graph=$OUTPUT"
  --artifact "test_resnet18_graph_contract=$CONTRACT_OUTPUT"
  --artifact-command "test_stock_nonlinear_full_keygen=adapter-link"
  --artifact-command "test_stock_nonlinear_full_keygen=objcopy-test_stock_nonlinear_full_keygen"
  --artifact-command "test_resnet18_full_graph=runtime-link"
  --artifact-command "test_resnet18_full_graph=objcopy-test_resnet18_full_graph"
  --artifact-command "test_resnet18_graph_contract=contract-link"
  --artifact-dependency-group "test_stock_nonlinear_full_keygen=adapter-0"
  --artifact-dependency-group "test_stock_nonlinear_full_keygen=adapter-1"
  --artifact-dependency-group "test_stock_nonlinear_full_keygen=adapter-2"
  --artifact-dependency-group "test_resnet18_full_graph=runtime-0"
  --artifact-dependency-group "test_resnet18_full_graph=runtime-1"
  --artifact-dependency-group "test_resnet18_full_graph=runtime-2"
  --artifact-dependency-group "test_resnet18_graph_contract=contract-0"
  --artifact-archive "test_resnet18_full_graph=ringlpn_linear"
  --artifact-archive "test_resnet18_full_graph=cryptoTools"
  --artifact-archive "test_resnet18_full_graph=bitpack"
  --artifact-archive "test_resnet18_full_graph=LLAMA"
  --recipe "$ROOT/cmake/graph_libraries/CMakeLists.txt"
  --recipe "$ROOT/scripts/build_common.sh"
  --recipe "$ROOT/scripts/build_component.sh"
  --recipe "$ROOT/scripts/build_resnet18_full_graph.sh"
  --recipe "$ROOT/scripts/build_linear_library.sh"
  --recipe "$ROOT/src/linear_preprocess_backend.cuh"
  --recipe "$ROOT/src/public_ring_vector_xof.h"
  --recipe "$ROOT/src/linear_preprocess_fc.cu"
  --recipe "$ROOT/src/linear_preprocess_conv.cu"
  --recipe "$ROOT/src/secure_convert.cpp"
  --recipe "$ROOT/src/secure_truncate.cpp"
  --recipe "$ROOT/../utils/gpu_mem.cu"
  --recipe "$ROOT/scripts/graph_build_provenance.py"
  --recipe "$ROOT/scripts/run_resnet18_graph_contract_gate.sh"
)
for index in "${!KEYGEN_SOURCES[@]}"; do
  PROVENANCE_COMMAND+=(--depfile "adapter-$index=$KEEP_DIR/adapter-$index.d")
done
for index in "${!RUNTIME_SOURCES[@]}"; do
  PROVENANCE_COMMAND+=(--depfile "runtime-$index=$KEEP_DIR/runtime-$index.d")
done
PROVENANCE_COMMAND+=(--depfile "contract-0=$KEEP_DIR/contract-0.d")
for label in adapter-link runtime-link contract-link \
    objcopy-test_stock_nonlinear_full_keygen objcopy-test_resnet18_full_graph; do
  PROVENANCE_COMMAND+=(--command "$label=$KEEP_DIR/$label.command")
done
"${PROVENANCE_COMMAND[@]}"

printf 'Built %s\nBuilt %s\nBuilt %s\nProvenance %s\n' \
  "$KEYGEN_OUTPUT" "$OUTPUT" "$CONTRACT_OUTPUT" "$PROVENANCE_OUTPUT"
