#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REV="ed044b903fdf6fd213b171eaa125e4eb52363903"
WORK_DIR="${DMPF_WORK_DIR:-/tmp/ringlpn-dmpf-sp25}"
OUT="${OUT:-${ROOT}/results/dpf/dmpf_baseline_comparison_2026_07_29.csv}"
LOG="${LOG:-${ROOT}/results/dpf/dmpf_baseline_comparison_2026_07_29.log}"
RUNS="${RUNS:-3}"
CPU="${CPU:-0}"
RUST_IMAGE="${RUST_IMAGE:-rust:1.88-bookworm}"

if [[ ! -d "${WORK_DIR}/.git" ]]; then
  rm -rf "${WORK_DIR}"
  git clone --filter=blob:none https://github.com/MatanHamilis/dmpf.git "${WORK_DIR}"
fi
git -C "${WORK_DIR}" fetch --quiet origin "${REV}"
git -C "${WORK_DIR}" checkout --quiet --detach "${REV}"
test "$(git -C "${WORK_DIR}" rev-parse HEAD)" = "${REV}"
cp "${SCRIPT_DIR}/dmpf_baseline_bench.rs" "${WORK_DIR}/src/bin/ringlpn_compare.rs"

docker run --rm -v "${WORK_DIR}:/work" -w /work "${RUST_IMAGE}" \
  cargo +nightly-2024-09-29 build --release --bin ringlpn_compare
BIN="${WORK_DIR}/target/release/ringlpn_compare"

mkdir -p "$(dirname "${OUT}")"
: >"${OUT}"
: >"${LOG}"
{
  echo "source=https://github.com/MatanHamilis/dmpf"
  echo "revision=${REV}"
  echo "license=CC0-1.0"
  echo "rust_image=${RUST_IMAGE}"
  echo "toolchain=nightly-2024-09-29"
  echo "rustflags=-C target-cpu=native"
  echo "cpu_pin=${CPU}"
  echo "host=$(uname -srmo)"
  echo "cpu_model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | sed -n '1p')"
  echo "note=centralized generation; one host process; two party evaluations run sequentially"
  echo "note=Goldilocks field 0xFFFFFFFF00000001 in two coordinates, not the deployed q62 primes"
  echo "note=all inputs distinct; private collision coalescing is outside this public implementation"
} >>"${LOG}"

wrote_header=0
run_one() {
  local scheme="$1" log_domain="$2" points="$3" seed="$4" tier="$5"
  local result header row
  result="$(taskset -c "${CPU}" "${BIN}" "${scheme}" "${log_domain}" "${points}" 128 "${seed}")"
  header="$(printf '%s\n' "${result}" | sed -n '1p')"
  row="$(printf '%s\n' "${result}" | sed -n '2p')"
  if [[ "${wrote_header}" -eq 0 ]]; then
    printf 'tier,%s\n' "${header}" >"${OUT}"
    wrote_header=1
  fi
  printf '%s,%s\n' "${tier}" "${row}" >>"${OUT}"
  printf '[%s] %s\n' "${tier}" "${row}" >>"${LOG}"
}

for seed in $(seq 1 "${RUNS}"); do
  run_one dpf 14 256 "${seed}" current_feasibility
  run_one big_state 14 256 "${seed}" current_feasibility
  run_one okvs 14 256 "${seed}" current_feasibility
done

# The preliminary architecture candidate exposed by the S2 audit. It is not a
# security claim, but omitting it would hide the 64x domain-size gap between the
# candidate and the BCG-scale reference below.
for seed in $(seq 1 "${RUNS}"); do
  run_one dpf 15 4096 "${seed}" preliminary_candidate
  run_one big_state 15 4096 "${seed}" preliminary_candidate
  run_one okvs 15 4096 "${seed}" preliminary_candidate
done

# BCG+20 scale reference. These are functionality/scale measurements only: the
# public code uses centralized generation and Goldilocks_x2, not q62 CRT limbs.
for seed in $(seq 1 "${RUNS}"); do
  run_one dpf 21 4096 "${seed}" literature_reference
  run_one big_state 21 4096 "${seed}" literature_reference
  run_one okvs 21 4096 "${seed}" literature_reference
done

printf 'Wrote %s and %s\n' "${OUT}" "${LOG}"
