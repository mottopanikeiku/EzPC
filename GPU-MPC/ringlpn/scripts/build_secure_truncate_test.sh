#!/usr/bin/env bash
# Builds the two-process OT-backed stochastic-truncation correctness gate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
mkdir -p "$ROOT/host_bin"

g++ -std=c++17 -O2 -Wall -Wextra -maes -msse4.1 -mpclmul -mavx2 -mrdseed \
  -I "$SCI_SRC" -I "$ROOT/src" \
  "$ROOT/src/test_secure_truncate.cpp" \
  "$ROOT/src/secure_truncate.cpp" \
  "$ROOT/src/secure_convert.cpp" \
  -o "$ROOT/host_bin/test_secure_truncate" \
  -lcrypto -lssl -pthread

echo "Built $ROOT/host_bin/test_secure_truncate"
