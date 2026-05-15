#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/host_bin"

g++ -std=c++17 -O2 -Wall -Wextra \
  "$ROOT/src/test_orca_zp_bridge.cpp" \
  -o "$ROOT/host_bin/test_orca_zp_bridge"

echo "Built $ROOT/host_bin/test_orca_zp_bridge"
