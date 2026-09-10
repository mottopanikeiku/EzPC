#!/usr/bin/env bash
# Canonical, discoverable entry point for Ring-LPN libraries and executables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=build_common.sh
source "$SCRIPT_DIR/build_common.sh"
ringlpn_build_init

usage() {
  cat <<'USAGE'
Usage: build_component.sh <component>
       build_component.sh list

Libraries:
  graph-libraries             Source-build cryptoTools, bitpack, and LLAMA
  linear-library             Stable FC+Conv archive, public header, API probe

Approved linear adapters (fixed canonical source path):
  linear-fc                   test_two_party_fc_preprocess
  linear-conv                 test_two_party_conv_preprocess

Graph executable pair:
  resnet18-full-graph         graph-libraries + linear-library + full graph/key adapter
  orca-linear-application     Graph/linear libraries, adapters, source-native Orca gate

Focused executable components:
  ole-host                    host Figure-2/OLE trio
  orca-zp-bridge              host Orca Zp bridge
  secure-convert              host secure conversion
  secure-truncate             host secure truncation
  distributed-dpf             host distributed DPF prototype
  two-party-dpf               host two-process DPF keygen and validator
  ole-cuda                    CUDA Figure-2/OLE components
  linear-ole                  CUDA linear OLE benchmark
  orca-fc-ringlpn             CUDA Ring-LPN FC demo
  orca-fc-ideal-ole           CUDA ideal-OLE reference
  orca-fc-real-ole            CUDA real-OLE FC transcript
  resnet18-graph-prefix       historical focused graph prefix
  emp-silent-bridge           optional EMP-Silent bridge
  ntt-gpu-baseline           external GPU-NTT baseline (requires explicit source)

Microbenchmarks:
  ntt-cpu                     NFLlib CPU NTT benchmark
  ntt-cuda                    primary CUDA NTT benchmark
  ntt-cuda-cheddar            explicitly named Cheddar-derived CUDA benchmark
  vole-cuda                   standalone CUDA Ring-LPN VOLE benchmark
  ntt-cuda-legacy             opt-in historical CUDA NTT benchmark
USAGE
}

build_graph_libraries() {
  local cmake_bin cxx_bin
  CMAKE="${CMAKE:-cmake}"
  CXX="${CXX:-g++}"
  ringlpn_require_command CMAKE cmake 'Install CMake 3.17 or newer.'
  ringlpn_require_command CXX g++ 'Install a C++17 compiler.'
  cmake_bin="$CMAKE"
  cxx_bin="$CXX"

  ringlpn_require_source "$RINGLPN_ROOT/cmake/graph_libraries/CMakeLists.txt" 'Ring-LPN library build definition'
  ringlpn_require_source \
    "$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/cryptoTools/cryptoTools/Crypto/PRNG.h" \
    'cryptoTools source tree'
  ringlpn_require_source \
    "$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/bitpack/include/bitpack/bitpack.h" \
    'bitpack source tree'
  ringlpn_require_source \
    "$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/llama/include/llama/api.h" \
    'LLAMA source tree'

  local build_root="$RINGLPN_BUILD_DIR/graph-libraries"
  local lock_root="$RINGLPN_BUILD_DIR/.graph-libraries.lock"
  ringlpn_prepare_build_root
  ringlpn_acquire_build_lock "$lock_root"
  trap ringlpn_release_build_lock EXIT

  "$cmake_bin" \
    -S "$RINGLPN_ROOT/cmake/graph_libraries" \
    -B "$build_root" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$cxx_bin" \
    -DRINGLPN_REPO_ROOT="$RINGLPN_REPO_ROOT"
  "$cmake_bin" --build "$build_root" --target LLAMA --parallel "${RINGLPN_BUILD_JOBS:-1}"

  local library
  for library in libcryptoTools.a libbitpack.a libLLAMA.a; do
    if [[ ! -f "$build_root/lib/$library" ]]; then
      printf 'library build completed without expected output: %s\n' \
        "$build_root/lib/$library" >&2
      exit 1
    fi
  done
  ringlpn_release_build_lock
  trap - EXIT
  printf 'Built graph libraries in %s/lib\n' "$build_root"
}

run_wrapper() {
  local wrapper="$1"
  shift
  ringlpn_prepare_build_root
  ringlpn_acquire_build_lock "$RINGLPN_BUILD_DIR/.component-${component}.lock"
  trap ringlpn_release_build_lock EXIT
  RINGLPN_COMPONENT_DISPATCH_ACTIVE=1 "$SCRIPT_DIR/$wrapper" "$@"
  ringlpn_release_build_lock
  trap - EXIT
}

component="${1:-}"
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$component" in
  list|--list|-l)
    usage
    ;;
  graph-libraries)
    build_graph_libraries
    ;;
  linear-library)
    run_wrapper build_linear_library.sh
    ;;
  linear-fc)
    run_wrapper build_two_party_fc_preprocess.sh
    ;;
  linear-conv)
    run_wrapper build_two_party_conv_preprocess.sh
    ;;
  resnet18-full-graph)
    build_graph_libraries
    run_wrapper build_linear_library.sh
    run_wrapper build_resnet18_full_graph.sh
    ;;
  orca-linear-application)
    build_graph_libraries
    run_wrapper build_linear_library.sh
    run_wrapper build_two_party_fc_preprocess.sh
    run_wrapper build_two_party_conv_preprocess.sh
    run_wrapper build_orca_linear_application.sh
    ;;
  ole-host) run_wrapper build_ole_host.sh ;;
  orca-zp-bridge) run_wrapper build_orca_zp_bridge_test.sh ;;
  secure-convert) run_wrapper build_secure_convert_test.sh ;;
  secure-truncate) run_wrapper build_secure_truncate_test.sh ;;
  distributed-dpf) run_wrapper build_distributed_dpf_keygen.sh ;;
  two-party-dpf) run_wrapper build_two_party_dpf_keygen.sh ;;
  ole-cuda) run_wrapper build_ole_cuda_bench.sh ;;
  linear-ole) run_wrapper build_linear_ole_bench.sh ;;
  orca-fc-ringlpn) run_wrapper build_orca_fc_ringlpn_demo.sh ;;
  orca-fc-ideal-ole) run_wrapper build_orca_fc_ideal_ole_transcript.sh ;;
  orca-fc-real-ole) run_wrapper build_orca_fc_real_ole_transcript.sh ;;
  resnet18-graph-prefix) run_wrapper build_resnet18_graph_prefix.sh ;;
  emp-silent-bridge) run_wrapper build_emp_silent_bridge.sh ;;
  ntt-gpu-baseline) run_wrapper build_ntt_gpu_ntt_baseline.sh ;;
  ntt-cpu) run_wrapper build_bench.sh ;;
  ntt-cuda) run_wrapper build_cuda_bench.sh ;;
  ntt-cuda-cheddar) run_wrapper build_cuda_bench_cheddar.sh ;;
  vole-cuda) run_wrapper build_vole_bench.sh ;;
  ntt-cuda-legacy) run_wrapper build_cuda_bench_legacy.sh ;;
  *)
    printf 'unknown Ring-LPN component: %s\n\n' "$component" >&2
    usage >&2
    exit 2
    ;;
esac
