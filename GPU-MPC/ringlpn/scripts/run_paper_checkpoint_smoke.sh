#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-0}"
REQUIRE_GPU_SMOKE="${REQUIRE_GPU_SMOKE:-0}"
RUN_REGULAR_SMOKE="${RUN_REGULAR_SMOKE:-1}"

echo "[paper-smoke] root: $ROOT"
echo "[paper-smoke] checking shell script syntax"
bash -n "$ROOT"/scripts/*.sh

echo "[paper-smoke] building and running host Figure 2 OLE trio"
"$ROOT/scripts/build_ole_host.sh"
expand_out="$("$ROOT/host_bin/verify_figure2_expand" --n 128 --c 2 --t 8 --seed 1)"
grep -q 'expand_pass=1' <<<"$expand_out"
spfss_out="$("$ROOT/host_bin/test_spfss" --log-domain 10 --m 16 --seed 1)"
grep -q 'spfss_pass=1' <<<"$spfss_out"
ole_out="$("$ROOT/host_bin/bench_ole_ringlpn_host" --n 128 --c 2 --t 8 --seed 1)"
grep -q 'ole_pass=1' <<<"$ole_out"

echo "[paper-smoke] running host Orca Zp-to-Z2k bridge smoke"
"$ROOT/scripts/build_orca_zp_bridge_test.sh"
"$ROOT/scripts/run_orca_zp_bridge_test.sh"

echo "[paper-smoke] running host secure Zm-to-Z2k conversion prototype"
"$ROOT/scripts/build_secure_convert_test.sh"
"$ROOT/scripts/run_secure_convert_test.sh"

echo "[paper-smoke] running host distributed DPF keygen prototype (M1 host slice)"
"$ROOT/scripts/build_distributed_dpf_keygen.sh"
"$ROOT/scripts/run_distributed_dpf_keygen.sh"

echo "[paper-smoke] running two-process distributed DPF keygen over real OT/TCP"
"$ROOT/scripts/build_two_party_dpf_keygen.sh"
BASE_PORT="${TWO_PARTY_BASE_PORT:-43600}" "$ROOT/scripts/run_two_party_dpf_keygen.sh"

if [[ "$RUN_GPU_SMOKE" != "1" ]]; then
  echo "[paper-smoke] GPU smoke skipped; set RUN_GPU_SMOKE=1 inside /home/ringlpn in the orca-dev container to run it"
  echo "[paper-smoke] HOST GATES PASS (GPU smoke skipped)"
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

echo "[paper-smoke] running two-process keygen with the GPU PRG + unmodified GPU evaluator"
BASE_PORT="${TWO_PARTY_GPU_BASE_PORT:-45200}" "$ROOT/scripts/run_two_party_gpu_dpf.sh"
SMOKE=1 "$ROOT/scripts/run_ole_sweep.sh"
SMOKE=1 QBITS=128 "$ROOT/scripts/run_ole_sweep.sh"
if [[ "$RUN_REGULAR_SMOKE" == "1" ]]; then
  SMOKE=1 NOISE=regular "$ROOT/scripts/run_ole_sweep.sh"
  SMOKE=1 QBITS=128 NOISE=regular "$ROOT/scripts/run_ole_sweep.sh"
fi

echo "[paper-smoke] building and running linear OLE-to-Beaver GPU smoke"
"$ROOT/scripts/build_linear_ole_bench.sh"
"$ROOT/scripts/run_linear_ole_sweep.sh"
QBITS=128 "$ROOT/scripts/run_linear_ole_sweep.sh"
if [[ "$RUN_REGULAR_SMOKE" == "1" ]]; then
  NOISE=regular "$ROOT/scripts/run_linear_ole_sweep.sh"
  QBITS=128 NOISE=regular "$ROOT/scripts/run_linear_ole_sweep.sh"
fi

echo "[paper-smoke] building and running Orca FC Ring-LPN key-writer demo"
"$ROOT/scripts/build_orca_fc_ringlpn_demo.sh"
"$ROOT/scripts/run_orca_fc_ringlpn_demo.sh"

echo "[paper-smoke] building and running ideal-OLE dealerless FC transcript"
"$ROOT/scripts/build_orca_fc_ideal_ole_transcript.sh"
"$ROOT/scripts/run_orca_fc_ideal_ole_transcript.sh"

echo "[paper-smoke] building and running real-OLE slot-packed FC transcript"
"$ROOT/scripts/build_orca_fc_real_ole_transcript.sh"
"$ROOT/scripts/run_orca_fc_real_ole_transcript.sh"

echo "[paper-smoke] complete"
echo "[paper-smoke] ALL GATES PASS"
