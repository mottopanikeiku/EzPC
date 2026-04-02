#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NFL_DIR="$BASE_DIR/extern/NFLlib"
OUT_DIR="$BASE_DIR/bin"

mkdir -p "$OUT_DIR"

if [[ ! -d "$NFL_DIR" ]]; then
  echo "NFLlib not found. Run scripts/setup_nfl.sh first."
  exit 1
fi

CXX=${CXX:-g++}
CXXFLAGS="-O3 -DNFL_OPTIMIZED=ON -std=c++11"
INCLUDES="-I$NFL_DIR/include"
LIB_DIR="$NFL_DIR/build"
LIBS="-L$LIB_DIR -lnfllib_static -lgmp -lgmpxx -lpthread"

$CXX $CXXFLAGS $INCLUDES "$BASE_DIR/src/bench_ntt.cpp" -o "$OUT_DIR/bench_ntt" $LIBS
