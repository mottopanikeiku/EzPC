#!/usr/bin/env bash
# Deterministic FC+Conv2D static library and public API contract executable.
set -euo pipefail

if [[ "${RINGLPN_CANONICAL_BUILD_ACTIVE:-0}" != "1" ]]; then
  ACTUAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  ACTUAL_REPO_ROOT="$(cd "$ACTUAL_ROOT/../.." && pwd -P)"
  CANONICAL_PARENT=/tmp/ringlpn-linear-library-reproducible-build
  mkdir -m 700 -- "$CANONICAL_PARENT"
  cleanup() { rm -f -- "$CANONICAL_PARENT/source"; rmdir -- "$CANONICAL_PARENT"; }
  trap cleanup EXIT
  ln -s -- "$ACTUAL_REPO_ROOT" "$CANONICAL_PARENT/source"
  RINGLPN_CANONICAL_BUILD_ACTIVE=1 \
    "$CANONICAL_PARENT/source/GPU-MPC/ringlpn/scripts/build_linear_library.sh"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
NVCC="$(command -v "${NVCC:-nvcc}")"
AR="$(command -v "${AR:-ar}")"
OBJCOPY="$(command -v "${OBJCOPY:-objcopy}")"
NM="$(command -v "${NM:-nm}")"
OUTPUT_ROOT="$ROOT/build/linear-library"
LIB_DIR="$OUTPUT_ROOT/lib"
INCLUDE_DIR="$OUTPUT_ROOT/include/ringlpn"
BIN_DIR="$OUTPUT_ROOT/bin"
BUILD_DIR="$OUTPUT_ROOT/.objects"
ARCHIVE="$LIB_DIR/libringlpn_linear.a"
API_BIN="$BIN_DIR/test_linear_preprocess_api"
PUBLIC_HEADER="$INCLUDE_DIR/linear_preprocess.h"

mkdir -p -- "$LIB_DIR" "$INCLUDE_DIR" "$BIN_DIR"
if ! mkdir -m 700 -- "$BUILD_DIR"; then
  echo "linear-library build directory already exists: $BUILD_DIR" >&2
  exit 1
fi
trap 'rm -rf -- "$BUILD_DIR"' EXIT
mkdir -p -- "$BUILD_DIR/include/ringlpn"
cp -- "$ROOT/src/linear_preprocess.h" \
  "$BUILD_DIR/include/ringlpn/linear_preprocess.h"
chmod 0644 "$BUILD_DIR/include/ringlpn/linear_preprocess.h"


COMMON_FLAGS=(
  -O2
  -std=c++17
  -arch=sm_89
  -diag-suppress=20012
  -Xcompiler=-fvisibility=hidden
  -Xcompiler=-fpermissive,-maes,-msse4.1,-mpclmul,-mavx2,-mrdseed,-pthread
  -Xcompiler=-ffile-prefix-map="$ROOT"=.,-fdebug-prefix-map="$ROOT"=.
  -Xcompiler=-ffile-prefix-map="$REPO_ROOT"=/ringlpn/source,-fdebug-prefix-map="$REPO_ROOT"=/ringlpn/source
  -Xcompiler=-frandom-seed=ringlpn-linear-library-20260810
  -I"$PROJECT_ROOT"
  -I"$PROJECT_ROOT/ext/cutlass/include"
  -I"$PROJECT_ROOT/ext/cutlass/tools/util/include"
  -I"$PROJECT_ROOT/ext/sytorch/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools"
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include"
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack"
  -I"$SCI_SRC"
  -I"$BUILD_DIR/include"
  -I"$ROOT/src"
)
API_FLAGS=(
  -O2
  -std=c++17
  -arch=sm_89
  -Xcompiler=-pthread
  -Xcompiler=-ffile-prefix-map="$ROOT"=.,-fdebug-prefix-map="$ROOT"=.
  -Xcompiler=-ffile-prefix-map="$REPO_ROOT"=/ringlpn/source,-fdebug-prefix-map="$REPO_ROOT"=/ringlpn/source
  -Xcompiler=-frandom-seed=ringlpn-linear-library-api-20260810
  -I"$BUILD_DIR/include"
)


sources=(
  src/linear_preprocess_fc.cu
  src/linear_preprocess_conv.cu
  src/secure_convert.cpp
  src/secure_truncate.cpp
  ../utils/gpu_mem.cu
  src/orca_globals_stub.cpp
)
objects=()
(
  cd "$ROOT"
  for index in "${!sources[@]}"; do
    object="$BUILD_DIR/object_${index}.o"
    "$NVCC" "${COMMON_FLAGS[@]}" \
      -Xcudafe="--orig_src_path_name=/ringlpn/linear-library-unit-${index}.cu" \
      -c "${sources[$index]}" -o "$object"
    "$OBJCOPY" --remove-section .comment "$object"
    "$NM" -a --format=posix "$object" |
      awk '$2 == "a" && $1 ~ /^tmpxft_[0-9a-f]+_.*[.]cudafe1[.]cpp$/ { print $1 }' \
        >"$BUILD_DIR/object_${index}-nvcc-file.symbols"
    if [[ -s "$BUILD_DIR/object_${index}-nvcc-file.symbols" ]]; then
      "$OBJCOPY" --strip-symbols="$BUILD_DIR/object_${index}-nvcc-file.symbols" \
        "$object"
    fi
    objects+=("$object")
  done
  "$NM" -g --defined-only --format=posix "$BUILD_DIR/object_0.o" |
    awk '$2 ~ /^[ABCDGRST]$/ { print $1 }' |
    LC_ALL=C sort -u >"$BUILD_DIR/fc-strong.symbols"
  "$NM" -g --defined-only --format=posix "$BUILD_DIR/object_1.o" |
    awk '$2 ~ /^[ABCDGRST]$/ { print $1 }' |
    LC_ALL=C sort -u >"$BUILD_DIR/conv-strong.symbols"
  LC_ALL=C comm -12 "$BUILD_DIR/fc-strong.symbols" \
    "$BUILD_DIR/conv-strong.symbols" >"$BUILD_DIR/duplicate-strong.symbols"
  if [[ -s "$BUILD_DIR/duplicate-strong.symbols" ]]; then
    "$OBJCOPY" --localize-symbols="$BUILD_DIR/duplicate-strong.symbols" \
      "$BUILD_DIR/object_1.o"
  fi
  ZERO_AR_DATE=1 "$AR" rcsD "$BUILD_DIR/libringlpn_linear.a" "${objects[@]}"

  "$NVCC" "${API_FLAGS[@]}" \
    -Xcudafe="--orig_src_path_name=/ringlpn/linear-library-api.cpp" \
    -Xlinker=--build-id=none \
    src/test_linear_preprocess_api.cpp \
    "$BUILD_DIR/libringlpn_linear.a" \
    -lcurand -lcrypto -lssl -ldl -lpthread -o "$BUILD_DIR/api"
  "$OBJCOPY" --remove-section .comment "$BUILD_DIR/api"
)
chmod 755 "$BUILD_DIR/api"
mv -f -- "$BUILD_DIR/libringlpn_linear.a" "$ARCHIVE"
mv -f -- "$BUILD_DIR/include/ringlpn/linear_preprocess.h" "$PUBLIC_HEADER"
mv -f -- "$BUILD_DIR/api" "$API_BIN"


sha256sum "$ARCHIVE" "$PUBLIC_HEADER" "$API_BIN"
