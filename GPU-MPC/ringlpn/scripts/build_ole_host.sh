#!/usr/bin/env bash
# Builds the host-only Figure 2 OLE Expand correctness artifacts:
#   - host_bin/verify_figure2_expand   (plaintext oracle)
#   - host_bin/test_spfss              (DPF+SPFSS unit test over Z_p)
#   - host_bin/bench_ole_ringlpn_host  (full Figure 2 Expand, z_0+z_1 == x_0*x_1)
#
# No CUDA toolchain required — these are host-only for correctness. The GPU
# acceleration of the x_sigma / z_sigma polymul steps reuses the existing
# run_polymul_prepared_lhs path validated by bench_vole_ringlpn.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$BASE_DIR/src"
OUT_DIR="$BASE_DIR/host_bin"

mkdir -p "$OUT_DIR"
CXX="${CXX:-g++}"
CXXFLAGS="-O2 -std=c++17 -Wall -Wextra"

"$CXX" $CXXFLAGS "$SRC_DIR/verify_figure2_expand.cpp" -o "$OUT_DIR/verify_figure2_expand"
"$CXX" $CXXFLAGS "$SRC_DIR/spfss_host.cpp" "$SRC_DIR/test_spfss.cpp" -o "$OUT_DIR/test_spfss"
"$CXX" $CXXFLAGS "$SRC_DIR/spfss_host.cpp" "$SRC_DIR/bench_ole_ringlpn_host.cpp" -o "$OUT_DIR/bench_ole_ringlpn_host"

echo "Built:"
ls -la "$OUT_DIR"
