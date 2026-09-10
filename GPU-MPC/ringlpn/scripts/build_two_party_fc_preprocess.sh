#!/usr/bin/env bash
# Build the live two-process Ring-LPN -> Orca forward-FC preprocessing artifact.
set -euo pipefail

if [[ "${RINGLPN_COMPONENT_DISPATCH_ACTIVE:-0}" != "1" ]]; then
  component="linear-${RINGLPN_LINEAR_KIND:-fc}"
  exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_component.sh" "$component"
fi

# nvcc records its working tree in the CUDA fatbin even when host compiler
# prefix maps are enabled. Re-enter through one private, fixed symlink so the
# approved ELF is byte-identical across clone and mount locations.
if [[ "${RINGLPN_CANONICAL_BUILD_ACTIVE:-0}" != "1" ]]; then
  ACTUAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  ACTUAL_REPO_ROOT="$(cd "$ACTUAL_ROOT/../.." && pwd -P)"
  CANONICAL_BUILD_PARENT="/tmp/ringlpn-linear-adapter-reproducible-build"
  if ! mkdir -m 700 -- "$CANONICAL_BUILD_PARENT"; then
    echo "stale or concurrent canonical adapter build root: $CANONICAL_BUILD_PARENT" >&2
    exit 1
  fi
  cleanup_canonical_build_root() {
    rm -f -- "$CANONICAL_BUILD_PARENT/source"
    rmdir -- "$CANONICAL_BUILD_PARENT"
  }
  trap cleanup_canonical_build_root EXIT
  ln -s -- "$ACTUAL_REPO_ROOT" "$CANONICAL_BUILD_PARENT/source"
  RINGLPN_CANONICAL_BUILD_ACTIVE=1 \
    "$CANONICAL_BUILD_PARENT/source/GPU-MPC/ringlpn/scripts/build_two_party_fc_preprocess.sh"
  exit 0
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"
ringlpn_build_init
ROOT="$RINGLPN_ROOT"
PROJECT_ROOT="$RINGLPN_PROJECT_ROOT"
REPO_ROOT="$RINGLPN_REPO_ROOT"
SCI_SRC="$RINGLPN_SCI_SRC"
OUT_DIR="$RINGLPN_OUT_DIR"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"
OBJCOPY="${OBJCOPY:-objcopy}"
HOST_CXX="${CXX:-g++}"
LINEAR_KIND="${RINGLPN_LINEAR_KIND:-fc}"
if [[ "$LINEAR_KIND" != "fc" && "$LINEAR_KIND" != "conv" ]]; then
  echo "RINGLPN_LINEAR_KIND must be fc or conv" >&2
  exit 2
fi
OUTPUT="$OUT_DIR/test_two_party_${LINEAR_KIND}_preprocess"

mkdir -p "$OUT_DIR"
ringlpn_require_command NVCC nvcc 'Run inside the CUDA toolkit environment.'
ringlpn_require_command OBJCOPY objcopy 'Install binutils.'
ringlpn_require_command HOST_CXX g++ 'Install a CUDA-compatible host C++ compiler.'
ringlpn_require_linear_sources
KEEP_DIR="$OUT_DIR/.two-party-${LINEAR_KIND}-nvcc"
if ! mkdir "$KEEP_DIR"; then
  echo "stale or concurrent ${LINEAR_KIND} build directory: $KEEP_DIR" >&2
  exit 1
fi
trap 'rm -rf -- "$KEEP_DIR"' EXIT
KEEP_DIR_ARG="bin/.two-party-${LINEAR_KIND}-nvcc"

ringlpn_set_cuda_include_flags CUDA_INCLUDE_FLAGS
COMMON_FLAGS=(
  -O2
  -std=c++17
  -arch="sm_${CUDA_ARCH}"
  -diag-suppress=20012
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread
  "${CUDA_INCLUDE_FLAGS[@]}"
)

PROVENANCE_FLAGS=()
if [[ -n "${RINGLPN_LINEAR_DEPFILE:-}${RINGLPN_LINEAR_LINK_MAP:-}${RINGLPN_LINEAR_COMMAND_FILE:-}${RINGLPN_LINEAR_ENVIRONMENT_FILE:-}" ]]; then
  if [[ -z "${RINGLPN_LINEAR_DEPFILE:-}" || -z "${RINGLPN_LINEAR_LINK_MAP:-}" ||
        -z "${RINGLPN_LINEAR_COMMAND_FILE:-}" ||
        -z "${RINGLPN_LINEAR_ENVIRONMENT_FILE:-}" ]]; then
    echo "linear provenance requires depfile, link-map, command, and environment paths together" >&2
    exit 2
  fi
  for provenance_path in \
      "$RINGLPN_LINEAR_DEPFILE" "$RINGLPN_LINEAR_LINK_MAP" \
      "$RINGLPN_LINEAR_COMMAND_FILE" "$RINGLPN_LINEAR_ENVIRONMENT_FILE"; do
    if [[ "$provenance_path" != /* || ! -d "$(dirname "$provenance_path")" ]]; then
      echo "linear provenance outputs must have existing absolute parent directories" >&2
      exit 2
    fi
  done
  PROVENANCE_FLAGS=(
    -Xlinker="-Map=$RINGLPN_LINEAR_LINK_MAP"
  )
fi

SOURCE_FILES=(
  "src/test_two_party_${LINEAR_KIND}_preprocess.cu"
  "src/linear_preprocess_${LINEAR_KIND}.cu"
  src/secure_convert.cpp
  src/secure_truncate.cpp
  ../utils/gpu_mem.cu
  src/orca_globals_stub.cpp
)

BUILD_COMMAND=(
  "$NVCC" "${COMMON_FLAGS[@]}"
  -ccbin="$HOST_CXX"
  -Xcompiler=-ffile-prefix-map="$ROOT"=.,-fdebug-prefix-map="$ROOT"=.
  -Xcompiler=-ffile-prefix-map="$REPO_ROOT"=/ringlpn/source,-fdebug-prefix-map="$REPO_ROOT"=/ringlpn/source
  -Xcudafe="--orig_src_path_name=/ringlpn/reproducible-two-party-${LINEAR_KIND}.cu"
  -Xcompiler="-frandom-seed=ringlpn-two-party-${LINEAR_KIND}-20260810"
  -Xlinker=--build-id=none
  "${PROVENANCE_FLAGS[@]}"
  --keep
  --keep-dir="$KEEP_DIR_ARG"
  "${SOURCE_FILES[@]}"
  -lcurand -lcrypto -lssl -ldl -lpthread
  -o "$KEEP_DIR_ARG/output"
)
OBJCOPY_COMMAND=(
  "$OBJCOPY" --remove-section .comment "$KEEP_DIR/output"
)

if [[ -n "${RINGLPN_LINEAR_COMMAND_FILE:-}" ]]; then
  {
    printf '%s\0' "$ROOT" "${BUILD_COMMAND[@]}"
    printf '\0'
    printf '%s\0' "$PWD" "${OBJCOPY_COMMAND[@]}"
  } >"$RINGLPN_LINEAR_COMMAND_FILE"
  /usr/bin/env -0 >"$RINGLPN_LINEAR_ENVIRONMENT_FILE"
fi

(
  cd "$ROOT"
  "${BUILD_COMMAND[@]}"
  if [[ -n "${RINGLPN_LINEAR_DEPFILE:-}" ]]; then
    for source_index in "${!SOURCE_FILES[@]}"; do
      "$NVCC" "${COMMON_FLAGS[@]}" -ccbin="$HOST_CXX" -M \
        "${SOURCE_FILES[$source_index]}" \
        -MF "${RINGLPN_LINEAR_DEPFILE}.${source_index}"
    done
  fi
)
"${OBJCOPY_COMMAND[@]}"
chmod 755 "$KEEP_DIR/output"
mv -f "$KEEP_DIR/output" "$OUTPUT"

echo "Built $OUTPUT"
