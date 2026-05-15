#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-0}"
REQUIRE_GPU_SMOKE="${REQUIRE_GPU_SMOKE:-0}"
RUN_REGULAR_SMOKE="${RUN_REGULAR_SMOKE:-1}"

echo "[paper-smoke] root: $ROOT"
echo "[paper-smoke] checking shell script syntax"
bash -n "$ROOT"/scripts/*.sh

echo "[paper-smoke] running host Orca Zp-to-Z2k bridge smoke"
"$ROOT/scripts/build_orca_zp_bridge_test.sh"
"$ROOT/scripts/run_orca_zp_bridge_test.sh"

if [[ "$RUN_GPU_SMOKE" != "1" ]]; then
  echo "[paper-smoke] GPU smoke skipped; set RUN_GPU_SMOKE=1 inside /home/ringlpn in the orca-dev container to run it"
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[paper-smoke] nvcc not found; cannot run GPU smoke"
  if [[ "$REQUIRE_GPU_SMOKE" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

echo "[paper-smoke] building and running Figure 2 OLE GPU smoke"
"$ROOT/scripts/build_ole_cuda_bench.sh"
"$ROOT/bin/test_spfss_zp_cuda"
SMOKE=1 "$ROOT/scripts/run_ole_sweep.sh"
if [[ "$RUN_REGULAR_SMOKE" == "1" ]]; then
  SMOKE=1 NOISE=regular "$ROOT/scripts/run_ole_sweep.sh"
fi

echo "[paper-smoke] building and running linear OLE-to-Beaver GPU smoke"
"$ROOT/scripts/build_linear_ole_bench.sh"
"$ROOT/scripts/run_linear_ole_sweep.sh"
if [[ "$RUN_REGULAR_SMOKE" == "1" ]]; then
  NOISE=regular "$ROOT/scripts/run_linear_ole_sweep.sh"
fi

echo "[paper-smoke] complete"
