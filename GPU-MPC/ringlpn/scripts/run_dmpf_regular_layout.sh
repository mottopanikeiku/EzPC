#!/usr/bin/env bash
# Regular-noise counterpart of run_dmpf_baseline_comparison.sh.
#
# The deployed GPU artifact uses REGULAR noise, so its sparse product is not one
# t^2-point function over a 2n domain. For bucket sum g in [0,2t-2] it is one
# function over a 2n/t domain holding m_g points, with m_g = g+1 for g < t and
# m_g = 2t-1-g otherwise. Comparing encoders only on the uniform layout would
# overstate the current baseline's cost, because regular noise already buys the
# depth and point-count reduction. This script measures every distinct per-group
# point count at the regular domain so per-pair totals can be derived exactly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REV="ed044b903fdf6fd213b171eaa125e4eb52363903"
WORK_DIR="${DMPF_WORK_DIR:-/tmp/ringlpn-dmpf-sp25}"
OUT="${OUT:-${ROOT}/results/dpf/dmpf_regular_layout_2026_07_29.csv}"
LOG="${LOG:-${ROOT}/results/dpf/dmpf_regular_layout_2026_07_29.log}"
RUNS="${RUNS:-3}"
CPU="${CPU:-0}"
RUST_IMAGE="${RUST_IMAGE:-rust:1.88-bookworm}"
# (n,c,t)=(2^14,4,16): regular domain 2n/t = 2048 (log 11); groups hold 1..16 pts.
LOG_DOMAIN="${LOG_DOMAIN:-11}"
MAX_POINTS="${MAX_POINTS:-16}"

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
  echo "layout=regular noise, (n,c,t)=(2^14,4,16), per-group domain 2^${LOG_DOMAIN}"
  echo "note=one row per distinct per-group point count; per-pair totals are derived"
  echo "note=Goldilocks_x2 field, centralized generation, sequential party evaluations"
} >"${LOG}"

wrote_header=0
for seed in $(seq 1 "${RUNS}"); do
  for points in $(seq 1 "${MAX_POINTS}"); do
    for scheme in dpf big_state okvs; do
      result="$(taskset -c "${CPU}" "${BIN}" "${scheme}" "${LOG_DOMAIN}" "${points}" 128 "${seed}")"
      if [[ "${wrote_header}" -eq 0 ]]; then
        printf 'tier,%s\n' "$(printf '%s\n' "${result}" | sed -n '1p')" >"${OUT}"
        wrote_header=1
      fi
      printf 'preliminary_candidate_regular,%s\n' "$(printf '%s\n' "${result}" | sed -n '2p')" >>"${OUT}"
    done
  done
done

printf 'Wrote %s and %s\n' "${OUT}" "${LOG}"
