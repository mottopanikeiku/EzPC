#!/usr/bin/env bash
# Initialize the pinned extern/NFLlib submodule and build the static library
# that build_bench.sh links (extern/NFLlib/build/libnfllib_static.a).
# Prerequisites (not installed here): git, cmake, make, a C++ compiler, and the
# GMP/MPFR development headers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NFL_DIR="$BASE_DIR/extern/NFLlib"

git -C "$BASE_DIR" submodule update --init -- extern/NFLlib

cmake -S "$NFL_DIR" -B "$NFL_DIR/build" -DCMAKE_BUILD_TYPE=Release -DNFL_OPTIMIZED=ON
cmake --build "$NFL_DIR/build" --target nfllib_static -j"$(nproc)"
