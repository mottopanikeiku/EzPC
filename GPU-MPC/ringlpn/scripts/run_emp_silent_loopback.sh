#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
DEPS_ROOT="${RINGLPN_EMP_DEPS_ROOT:-$ROOT/.deps/emp-silent}"
PREFIX="${RINGLPN_EMP_PREFIX:-$DEPS_ROOT/install}"
BRIDGE="${RINGLPN_EMP_SILENT_BRIDGE:-$PREFIX/lib/libringlpn_emp_silent_bridge.so}"
MODE="${1:---controls-only}"
BASE_PORT="${RINGLPN_EMP_TEST_PORT:-39761}"

if [[ "$MODE" != "--controls-only" && "$MODE" != "--full" ]]; then
  echo "usage: $0 [--controls-only|--full]" >&2
  exit 2
fi
if [[ ! -f "$BRIDGE" ]]; then
  "$ROOT/scripts/build_emp_silent_bridge.sh"
fi
mkdir -p "$ROOT/host_bin"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra \
  -maes -msse4.1 -mpclmul -mavx2 -mrdseed \
  -I"$SCI_SRC" -I"$ROOT/src" \
  "$ROOT/src/test_emp_silent_loopback.cpp" \
  -o "$ROOT/host_bin/test_emp_silent_loopback" \
  -lcrypto -lssl -ldl -pthread

"$ROOT/host_bin/test_emp_silent_loopback" "$BRIDGE" "$MODE" "$BASE_PORT"
