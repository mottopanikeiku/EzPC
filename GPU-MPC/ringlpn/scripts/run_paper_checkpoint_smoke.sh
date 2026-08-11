#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-0}"
REQUIRE_GPU_SMOKE="${REQUIRE_GPU_SMOKE:-0}"
RUN_REGULAR_SMOKE="${RUN_REGULAR_SMOKE:-1}"
RUN_FULL_GRAPH_SMOKE="${RUN_FULL_GRAPH_SMOKE:-1}"
FULL_GRAPH_TEMP=""
cleanup() {
  status=$?
  if [[ -n "$FULL_GRAPH_TEMP" && -d "$FULL_GRAPH_TEMP" ]]; then
    if [[ "$status" -eq 0 ]]; then
      rm -rf -- "$FULL_GRAPH_TEMP"
    else
      echo "[paper-smoke] full-graph failure diagnostics retained at $FULL_GRAPH_TEMP" >&2
    fi
  fi
}
trap cleanup EXIT

echo "[paper-smoke] root: $ROOT"
echo "[paper-smoke] checking shell script syntax"
bash -n "$ROOT"/scripts/*.sh
if [[ "$RUN_GPU_SMOKE" == "1" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[paper-smoke] nvcc not found; cannot run required GPU smoke" >&2
    exit 1
  fi
  echo "[paper-smoke] rebuilding reproducible shared linear adapters"
  "$ROOT/scripts/build_two_party_fc_preprocess.sh"
  "$ROOT/scripts/build_two_party_conv_preprocess.sh"
  echo "[paper-smoke] building source-bound full ResNet18 graph artifacts"
  "$ROOT/scripts/build_resnet18_full_graph.sh"
fi

echo "[paper-smoke] checking the pinned ResNet18 forward-linear execution manifest"
"$ROOT/scripts/run_full_linear_manifest_gate.sh"

if [[ "$RUN_GPU_SMOKE" == "1" ]]; then
  echo "[paper-smoke] checking the compiled full ResNet18 graph contract"
  "$ROOT/scripts/run_resnet18_graph_contract_gate.sh"
else
  echo "[paper-smoke] compiled full ResNet18 graph contract skipped with GPU smoke"
fi

echo "[paper-smoke] running owner-only atomic private-file control"
"$ROOT/scripts/build_private_file_test.sh"
"$ROOT/host_bin/test_private_file"

echo "[paper-smoke] running host public Ring-vector SHAKE/rejection control"
"$ROOT/scripts/build_public_ring_vector_xof_test.sh"
"$ROOT/host_bin/test_public_ring_vector_xof"

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

echo "[paper-smoke] running host OT-backed Zm-to-Z2k share-conversion component"
"$ROOT/scripts/build_secure_convert_test.sh"
"$ROOT/scripts/run_secure_convert_test.sh"

echo "[paper-smoke] running host OT-backed stochastic-truncation state handoff"
"$ROOT/scripts/build_secure_truncate_test.sh"
"$ROOT/scripts/run_secure_truncate_test.sh"

echo "[paper-smoke] running host distributed DPF keygen prototype (M1 host slice)"
"$ROOT/scripts/build_distributed_dpf_keygen.sh"
"$ROOT/scripts/run_distributed_dpf_keygen.sh"

echo "[paper-smoke] running two-process distributed DPF keygen over real OT/TCP"
"$ROOT/scripts/build_two_party_dpf_keygen.sh"
BASE_PORT="${TWO_PARTY_BASE_PORT:-21600}" "$ROOT/scripts/run_two_party_dpf_keygen.sh"

if [[ "$RUN_GPU_SMOKE" != "1" ]]; then
  echo "[paper-smoke] GPU smoke skipped; set RUN_GPU_SMOKE=1 inside /home/ringlpn in the orca-dev container to run it"
  echo "[paper-smoke] HOST GATES PASS (GPU smoke skipped)"
  exit 0
fi


echo "[paper-smoke] validating approved shared linear adapters and all ResNet18 shapes"
BINARY_APPROVAL="$ROOT/results/fc/linear_adapter_binary_approval_2026_08_07.json" \
  RUNNER_PLAN_CHECK=1 "$ROOT/scripts/run_full_linear_manifest_gate.sh"
"$ROOT/scripts/check_full_linear_shape_coverage.py" \
  --manifest "$ROOT/results/fc/resnet18_full_linear_execution_manifest_2026_08_06.json" \
  --bin-dir "$ROOT/bin"

echo "[paper-smoke] building and running Figure 2 OLE GPU smoke"
"$ROOT/scripts/build_ole_cuda_bench.sh"
"$ROOT/bin/test_spfss_zp_cuda"

echo "[paper-smoke] running two-process keygen with the GPU PRG + unmodified GPU evaluator"
BASE_PORT="${TWO_PARTY_GPU_BASE_PORT:-22200}" "$ROOT/scripts/run_two_party_gpu_dpf.sh"
SMOKE=1 "$ROOT/scripts/run_ole_sweep.sh"
SMOKE=1 QBITS=128 "$ROOT/scripts/run_ole_sweep.sh"
if [[ "$RUN_REGULAR_SMOKE" == "1" ]]; then
  SMOKE=1 NOISE=regular "$ROOT/scripts/run_ole_sweep.sh"
  SMOKE=1 QBITS=128 NOISE=regular "$ROOT/scripts/run_ole_sweep.sh"
fi

echo "[paper-smoke] running the real OLE engine on independently sampled per-party noise and two-party SPFSS keys"
QBITS=64 NOISE=regular BASE_PORT="${OLE_TWO_PARTY_BASE_PORT:-22800}" \
  "$ROOT/scripts/run_ole_two_party_keys.sh"
QBITS=64 NOISE=uniform BASE_PORT="${OLE_TWO_PARTY_BASE_PORT:-22800}" \
  "$ROOT/scripts/run_ole_two_party_keys.sh"
QBITS=128 NOISE=regular BASE_PORT="${OLE_TWO_PARTY_BASE_PORT:-22800}" \
  "$ROOT/scripts/run_ole_two_party_keys.sh"
QBITS=128 NOISE=uniform BASE_PORT="${OLE_TWO_PARTY_BASE_PORT:-22800}" \
  "$ROOT/scripts/run_ole_two_party_keys.sh"

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

echo "[paper-smoke] building and running ideal-OLE FC transcript reference (oracle-backed; not dealerless)"
"$ROOT/scripts/build_orca_fc_ideal_ole_transcript.sh"
"$ROOT/scripts/run_orca_fc_ideal_ole_transcript.sh"

echo "[paper-smoke] building and running real-OLE slot-packed FC transcript"
"$ROOT/scripts/build_orca_fc_real_ole_transcript.sh"
"$ROOT/scripts/run_orca_fc_real_ole_transcript.sh"

if [[ "$RUN_FULL_GRAPH_SMOKE" != "1" ]]; then
  echo "[paper-smoke] full ResNet18 graph skipped; GPU component gates pass"
  exit 0
fi

echo "[paper-smoke] building and running the source-bound full ResNet18 graph"
FULL_GRAPH_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/ringlpn-full-graph-smoke.XXXXXX")"
full_graph_args=("$FULL_GRAPH_TEMP/output" "$FULL_GRAPH_TEMP/state")
if [[ -n "${FULL_GRAPH_SUMMARY_ROOT:-}" ]]; then
  full_graph_args+=("$FULL_GRAPH_SUMMARY_ROOT")
fi
P0_GPU="${FULL_GRAPH_P0_GPU:-0}" \
P1_GPU="${FULL_GRAPH_P1_GPU:-1}" \
CHECK_GPU="${FULL_GRAPH_CHECK_GPU:-2}" \
TRUSTED_GPU="${FULL_GRAPH_TRUSTED_GPU:-1}" \
LINEAR_BASE_PORT="${FULL_GRAPH_LINEAR_BASE_PORT:-28800}" \
GRAPH_BASE_PORT="${FULL_GRAPH_BASE_PORT:-29000}" \
TIMEOUT_SECONDS="${FULL_GRAPH_TIMEOUT_SECONDS:-86400}" \
  "$ROOT/scripts/run_resnet18_full_graph.sh" "${full_graph_args[@]}"
rm -rf -- "$FULL_GRAPH_TEMP"
FULL_GRAPH_TEMP=""

echo "[paper-smoke] complete"
echo "[paper-smoke] ALL GATES PASS"
