#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/host_bin"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$ROOT/src" \
  "$ROOT/src/test_private_file.cpp" \
  -o "$ROOT/host_bin/test_private_file"

echo "Built $ROOT/host_bin/test_private_file"
