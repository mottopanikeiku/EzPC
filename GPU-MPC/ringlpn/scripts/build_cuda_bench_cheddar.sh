#!/usr/bin/env bash
# Build the same source as build_cuda_bench.sh under the side-by-side binary
# name bin/bench_ntt_cuda_cheddar with device label cuda_cheddar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_LABEL="${DEVICE_LABEL:-cuda_cheddar}" NTT_CUDA_BIN=bench_ntt_cuda_cheddar \
  exec "$SCRIPT_DIR/build_cuda_bench.sh"
