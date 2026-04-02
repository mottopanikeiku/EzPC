#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NFL_DIR="$BASE_DIR/extern/NFLlib"

if [[ -d "$NFL_DIR" ]]; then
  echo "NFLlib already exists at $NFL_DIR"
  exit 0
fi

mkdir -p "$BASE_DIR/extern"
cd "$BASE_DIR/extern"

git clone https://github.com/quarkslab/NFLlib.git
cd NFLlib

apt-get update
apt-get install -y cmake libgmp-dev libmpfr-dev

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNFL_OPTIMIZED=ON
make -j"$(nproc)"
