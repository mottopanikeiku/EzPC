#!/usr/bin/env bash
# Initialize the pinned extern/NFLlib submodule and build the static library
# that build_bench.sh links (extern/NFLlib/build/libnfllib_static.a).
#
# Idempotent: if the submodule is checked out at its gitlink commit and the
# static library exists, report it and exit 0. Set FORCE=1 to configure and
# rebuild anyway. An existing extern/NFLlib/build cache configured for a
# different source path is never reused or modified; the script stops instead.
#
# Prerequisites (checked, not installed here): git, cmake, make, a C++
# compiler (CXX or c++), and the GMP/MPFR development headers. The
# ringlpn-repro:2026-08-10 image provides all of them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NFL_DIR="$BASE_DIR/extern/NFLlib"
BUILD_DIR="$NFL_DIR/build"
LIB="$BUILD_DIR/libnfllib_static.a"
FORCE="${FORCE:-0}"

die() {
  echo "setup_nfl.sh: $*" >&2
  exit 1
}

command -v git >/dev/null || die "git not found"

# First column of `git submodule status`: ' ' = at the gitlink commit,
# '-' = not initialized, '+' = different commit checked out, 'U' = conflict.
status="$(git -C "$BASE_DIR" submodule status -- extern/NFLlib)"
if [[ "${status:0:1}" == " " && -f "$LIB" && "$FORCE" != 1 ]]; then
  echo "NFLlib already built at pinned commit ${status:1:40}: $LIB"
  echo "Set FORCE=1 to reconfigure and rebuild."
  exit 0
fi

CXX_BIN="${CXX:-c++}"
missing=()
command -v cmake >/dev/null || missing+=("cmake")
command -v make >/dev/null || missing+=("make")
if command -v "$CXX_BIN" >/dev/null; then
  for header in gmp.h mpfr.h; do
    printf '#include <%s>\n' "$header" | "$CXX_BIN" -x c++ -E - >/dev/null 2>&1 \
      || missing+=("$header (GMP/MPFR development headers)")
  done
else
  missing+=("C++ compiler '$CXX_BIN'")
fi
if ((${#missing[@]})); then
  printf 'setup_nfl.sh: missing build prerequisites:\n' >&2
  printf '  - %s\n' "${missing[@]}" >&2
  die "install them (e.g. Debian/Ubuntu: cmake make g++ libgmp-dev libmpfr-dev) or run inside ringlpn-repro:2026-08-10"
fi

git -C "$BASE_DIR" submodule update --init -- extern/NFLlib

CACHE="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$CACHE" ]]; then
  cached_src="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE")"
  if [[ "$cached_src" != "$NFL_DIR" && "$cached_src" != "$(cd "$NFL_DIR" && pwd -P)" ]]; then
    die "$CACHE was configured for source '$cached_src', not '$NFL_DIR'.
Fix: remove (or move aside) $BUILD_DIR and rerun this script, e.g. inside ringlpn-repro:2026-08-10."
  fi
fi

cmake -S "$NFL_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DNFL_OPTIMIZED=ON
cmake --build "$BUILD_DIR" --target nfllib_static -j"$(nproc)"
echo "Built $LIB"
