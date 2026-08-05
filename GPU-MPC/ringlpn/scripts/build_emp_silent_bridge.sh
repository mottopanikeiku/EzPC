#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${RINGLPN_EMP_DEPS_ROOT:-$ROOT/.deps/emp-silent}"
PREFIX="${RINGLPN_EMP_PREFIX:-$DEPS_ROOT/install}"
BUILD_ROOT="${RINGLPN_EMP_BUILD_ROOT:-$DEPS_ROOT/build}"
JOBS="${JOBS:-$(nproc)}"

"$ROOT/scripts/fetch_emp_silent.sh"

cmake -S "$DEPS_ROOT/src/emp-tool" -B "$BUILD_ROOT/emp-tool" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_TESTING=OFF
cmake --build "$BUILD_ROOT/emp-tool" --parallel "$JOBS"
cmake --install "$BUILD_ROOT/emp-tool"

cmake -S "$DEPS_ROOT/src/emp-ot" -B "$BUILD_ROOT/emp-ot" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DEMP_OT_BUILD_TESTS=OFF \
  -DEMP_OT_AUTO_TUNE=OFF
cmake --build "$BUILD_ROOT/emp-ot" --parallel "$JOBS"
cmake --install "$BUILD_ROOT/emp-ot"

cmake -S "$ROOT/src/emp_silent_bridge_build" -B "$BUILD_ROOT/bridge" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build "$BUILD_ROOT/bridge" --parallel "$JOBS"
cmake --install "$BUILD_ROOT/bridge"

BRIDGE="$PREFIX/lib/libringlpn_emp_silent_bridge.so"
if [[ ! -f "$BRIDGE" ]]; then
  echo "bridge build did not produce $BRIDGE" >&2
  exit 1
fi
printf 'Built pinned unreviewed EMP SilentFerret bridge: %s\n' "$BRIDGE"
printf 'Set RINGLPN_EMP_SILENT_BRIDGE=%s only with --ot-backend emp-silent.\n' "$BRIDGE"
