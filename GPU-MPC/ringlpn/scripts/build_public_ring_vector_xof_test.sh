#!/usr/bin/env bash
# Builds the host-only SHAKE256 public Ring-vector known-answer control.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/host_bin"

g++ -std=c++17 -O2 -Wall -Wextra \
  -I "$ROOT/src" \
  "$ROOT/src/test_public_ring_vector_xof.cpp" \
  -o "$ROOT/host_bin/test_public_ring_vector_xof" \
  -lcrypto

echo "Built $ROOT/host_bin/test_public_ring_vector_xof"
