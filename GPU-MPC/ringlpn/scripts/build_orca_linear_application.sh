#!/usr/bin/env bash
# Build the source-native terminal Ring-LPN Orca application and helper gate.
set -euo pipefail

if [[ "${RINGLPN_COMPONENT_DISPATCH_ACTIVE:-0}" != "1" ]]; then
  exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_component.sh" \
    orca-linear-application
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"
ringlpn_build_init
ROOT="$RINGLPN_ROOT"
PROJECT_ROOT="$RINGLPN_PROJECT_ROOT"
OUT_DIR="$RINGLPN_OUT_DIR"
BUILD_DIR="$RINGLPN_BUILD_DIR/orca-linear-application"
GRAPH_BUILD="$RINGLPN_BUILD_DIR/graph-libraries"
GRAPH_LIB_DIR="$GRAPH_BUILD/lib"
LINEAR_ARCHIVE="$RINGLPN_BUILD_DIR/linear-library/lib/libringlpn_linear.a"
APPLICATION_OUTPUT="$OUT_DIR/orca_inference_ringlpn"
HELPER_OUTPUT="$OUT_DIR/test_orca_linear_helpers"
PROVENANCE_OUTPUT="$OUT_DIR/orca_linear_application_build_provenance.json"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
CXX="${CXX:-g++}"
OBJCOPY="${OBJCOPY:-objcopy}"
NM="${NM:-nm}"
PYTHON="${PYTHON:-python3}"

ringlpn_require_command NVCC nvcc 'Run with the CUDA toolkit in PATH.'
export SOURCE_DATE_EPOCH=0
export TZ=UTC
export ZERO_AR_DATE=1
ringlpn_require_command CXX g++ 'Install a CUDA-compatible C++ compiler.'
ringlpn_require_command OBJCOPY objcopy 'Install binutils.'
ringlpn_require_command NM nm 'Install binutils.'
ringlpn_require_command PYTHON python3 'Install Python 3.'
ringlpn_require_linear_sources
ringlpn_require_source "$PROJECT_ROOT/experiments/orca/orca_inference.cu" \
  'Orca inference application source'
ringlpn_require_source "$ROOT/src/orca_terminal_linear_backend.cuh" \
  'terminal Ring-LPN Orca backend'
ringlpn_require_source "$ROOT/src/test_orca_linear_helpers.cu" \
  'Orca linear helper regression'
ringlpn_require_source "$LINEAR_ARCHIVE" 'deterministic linear facade archive'
ringlpn_require_source "$GRAPH_BUILD/compile_commands.json" \
  'graph-library compile database'
for library in libLLAMA.a libcryptoTools.a libbitpack.a; do
  ringlpn_require_source "$GRAPH_LIB_DIR/$library" "source-built $library"
done

mkdir -p -- "$OUT_DIR" "$BUILD_DIR"
KEEP_DIR="$BUILD_DIR/.nvcc"
if ! mkdir -- "$KEEP_DIR"; then
  printf 'stale or concurrent application build directory: %s\n' "$KEEP_DIR" >&2
  exit 1
fi
trap 'rm -rf -- "$KEEP_DIR"' EXIT

if [[ -e "$PROVENANCE_OUTPUT" || -L "$PROVENANCE_OUTPUT" ]]; then
  if [[ ! -f "$PROVENANCE_OUTPUT" || -L "$PROVENANCE_OUTPUT" ||
        ! -O "$PROVENANCE_OUTPUT" ]]; then
    printf 'refusing unsafe prior provenance output: %s\n' \
      "$PROVENANCE_OUTPUT" >&2
    exit 1
  fi
  rm -- "$PROVENANCE_OUTPUT"
fi

CUDA_TOOLKIT_ROOT="$(dirname "$NVCC")/.."
CUDA_TARGET_TRIPLE="${CUDA_TARGET_TRIPLE:-x86_64-linux}"
CUDA_SYSTEM_INCLUDE="$CUDA_TOOLKIT_ROOT/targets/$CUDA_TARGET_TRIPLE/include"
GCC_SYSTEM_INCLUDE="$("$CXX" -print-file-name=include)"
ringlpn_set_cuda_include_flags CUDA_INCLUDE_FLAGS
COMMON_FLAGS=(
  -O2
  -std=c++17
  -arch="sm_${CUDA_ARCH}"
  -ccbin="$CXX"
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread,-fopenmp
  -Xcompiler="-ffile-prefix-map=$RINGLPN_REPO_ROOT=/ringlpn/source,-fdebug-prefix-map=$RINGLPN_REPO_ROOT=/ringlpn/source"
  "${CUDA_INCLUDE_FLAGS[@]}"
)
APP_FLAGS=(
  "${COMMON_FLAGS[@]}"
  -DORCA_RINGLPN_LINEAR_INTEGRATION=1
  -Xcudafe=--orig_src_path_name=/ringlpn/orca-linear-application.cu
  -Xcompiler=-frandom-seed=ringlpn-orca-linear-application-20260824
)
HELPER_FLAGS=(
  "${COMMON_FLAGS[@]}"
  -Xcudafe=--orig_src_path_name=/ringlpn/orca-linear-helpers.cu
  -Xcompiler=-frandom-seed=ringlpn-orca-linear-helpers-20260824
)
APP_SOURCES=(
  ../experiments/orca/orca_inference.cu
  ../ext/sytorch/src/sytorch/random.cpp
  ../utils/gpu_mem.cu
  ../utils/gpu_file_utils.cpp
  ../utils/sigma_comms.cpp
)
HELPER_SOURCES=(
  src/test_orca_linear_helpers.cu
  ../ext/sytorch/src/sytorch/random.cpp
  ../utils/gpu_mem.cu
  ../utils/gpu_file_utils.cpp
  ../utils/sigma_comms.cpp
)
ARCHIVES=(
  "$GRAPH_LIB_DIR/libLLAMA.a"
  "$GRAPH_LIB_DIR/libcryptoTools.a"
  "$GRAPH_LIB_DIR/libbitpack.a"
)
# Orca's inherited CUDA headers define several non-inline support symbols in
# every consuming translation unit. The facade archive was built from the same
# headers. Keep the application's definitions first and permit only that
# unavoidable identical-definition overlap while still linking the exact
# deterministic facade archive.
APPLICATION_COMMAND=(
  "$NVCC" "${APP_FLAGS[@]}" -Xlinker=--build-id=none
  -Xlinker=--allow-multiple-definition
  "${APP_SOURCES[@]}" "$LINEAR_ARCHIVE" "${ARCHIVES[@]}"
  -lcurand -lcrypto -lssl -ldl -lpthread -lgomp
  -o bin/orca_inference_ringlpn
)
HELPER_COMMAND=(
  "$NVCC" "${HELPER_FLAGS[@]}" -Xlinker=--build-id=none
  "${HELPER_SOURCES[@]}" "${ARCHIVES[@]}"
  -lcurand -lcrypto -lssl -ldl -lpthread -lgomp
  -o bin/test_orca_linear_helpers
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
  local label="$1" source="$2" kind="$3"
  local -a flags
  if [[ "$kind" == app ]]; then
    flags=("${APP_FLAGS[@]}")
  else
    flags=("${HELPER_FLAGS[@]}")
  fi
  (
    cd "$ROOT"
    "$NVCC" "${flags[@]}" -MM -MT provenance \
      -MF "$KEEP_DIR/$label.d" "$source"
  )
}

for index in "${!APP_SOURCES[@]}"; do
  generate_depfile "application-$index" "${APP_SOURCES[$index]}" app
done
for index in "${!HELPER_SOURCES[@]}"; do
  generate_depfile "helper-$index" "${HELPER_SOURCES[$index]}" helper
done
(
  cd "$ROOT"
  "${APPLICATION_COMMAND[@]}"
  "${HELPER_COMMAND[@]}"
)
record_command application-link "${APPLICATION_COMMAND[@]}"
record_command helper-link "${HELPER_COMMAND[@]}"

for binary in "$APPLICATION_OUTPUT" "$HELPER_OUTPUT"; do
  symbol_file="$KEEP_DIR/$(basename "$binary")-nvcc-file.symbols"
  "$NM" -a --format=posix "$binary" |
    awk '$2 == "a" && $1 ~ /^tmpxft_[0-9a-f]+_.*[.]cudafe1[.]cpp$/ { print $1 }' \
      >"$symbol_file"
  OBJCOPY_COMMAND=("$OBJCOPY" --remove-section .comment)
  if [[ -s "$symbol_file" ]]; then
    OBJCOPY_COMMAND+=(--strip-symbols="$symbol_file")
  fi
  OBJCOPY_COMMAND+=("$binary")
  "${OBJCOPY_COMMAND[@]}"
  record_command "objcopy-$(basename "$binary")" "${OBJCOPY_COMMAND[@]}"
  chmod 0755 "$binary"
done

PROVENANCE_COMMAND=(
  "$PYTHON" "$ROOT/scripts/orca_linear_application_build_provenance.py"
  --repo-root "$RINGLPN_REPO_ROOT"
  --ringlpn-root "$ROOT"
  --cmake-build "$GRAPH_BUILD"
  --output "$PROVENANCE_OUTPUT"
  --system-root "cuda-toolkit-include=$CUDA_SYSTEM_INCLUDE"
  --system-root "gcc-internal-include=$GCC_SYSTEM_INCLUDE"
  --system-root system-usr-include=/usr/include
  --linked-archive "ringlpn_linear=$LINEAR_ARCHIVE"
  --archive "cryptoTools=$GRAPH_LIB_DIR/libcryptoTools.a"
  --archive "bitpack=$GRAPH_LIB_DIR/libbitpack.a"
  --archive "LLAMA=$GRAPH_LIB_DIR/libLLAMA.a"
  --artifact "orca_inference_ringlpn=$APPLICATION_OUTPUT"
  --artifact "test_orca_linear_helpers=$HELPER_OUTPUT"
  --artifact-command orca_inference_ringlpn=application-link
  --artifact-command orca_inference_ringlpn=objcopy-orca_inference_ringlpn
  --artifact-command test_orca_linear_helpers=helper-link
  --artifact-command test_orca_linear_helpers=objcopy-test_orca_linear_helpers
  --artifact-archive orca_inference_ringlpn=ringlpn_linear
  --artifact-archive orca_inference_ringlpn=cryptoTools
  --artifact-archive orca_inference_ringlpn=bitpack
  --artifact-archive orca_inference_ringlpn=LLAMA
  --artifact-archive test_orca_linear_helpers=cryptoTools
  --artifact-archive test_orca_linear_helpers=bitpack
  --artifact-archive test_orca_linear_helpers=LLAMA
  --recipe "$ROOT/cmake/graph_libraries/CMakeLists.txt"
  --recipe "$ROOT/scripts/build_common.sh"
  --recipe "$ROOT/scripts/build_component.sh"
  --recipe "$ROOT/scripts/build_linear_library.sh"
  --recipe "$ROOT/scripts/build_orca_linear_application.sh"
  --recipe "$ROOT/scripts/graph_build_provenance.py"
  --recipe "$ROOT/scripts/orca_linear_application_build_provenance.py"
)
for index in "${!APP_SOURCES[@]}"; do
  PROVENANCE_COMMAND+=(
    --depfile "application-$index=$KEEP_DIR/application-$index.d"
    --artifact-dependency-group "orca_inference_ringlpn=application-$index"
  )
done
for index in "${!HELPER_SOURCES[@]}"; do
  PROVENANCE_COMMAND+=(
    --depfile "helper-$index=$KEEP_DIR/helper-$index.d"
    --artifact-dependency-group "test_orca_linear_helpers=helper-$index"
  )
done
for label in application-link helper-link objcopy-orca_inference_ringlpn \
    objcopy-test_orca_linear_helpers; do
  PROVENANCE_COMMAND+=(--command "$label=$KEEP_DIR/$label.command")
done
"${PROVENANCE_COMMAND[@]}"

printf 'Built %s\nBuilt %s\nProvenance %s\n' \
  "$APPLICATION_OUTPUT" "$HELPER_OUTPUT" "$PROVENANCE_OUTPUT"
