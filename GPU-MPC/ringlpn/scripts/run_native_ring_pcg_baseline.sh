#!/usr/bin/env bash
# Adapted measurement of the MIT native-Z_(2^k)/Galois-ring PCG artifact
# (ePrint 2025/1223). This is NOT a reproduction of the released benchmark:
#   1. the released 64-bit parameter initializer computes `1<<(k+s)` with an
#      `int` literal, so its 121-bit modulus is undefined C (measured as 0 by
#      gcc -O3 on this host) and every following `% modulus128` is invalid;
#   2. the shipped harness runs c=3,t=27, the parameter set the artifact's own
#      README declares insecure after ePrint 2025/892.
# This script fixes (1) with a typed shift and moves (2) to the artifact's
# post-ePrint c=5,t=27 comparison grid. This is not a local security claim or
# reproduction. It records a patch digest and peak RSS, and labels
# every row `adapted`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REV="43959ef19cee4b25d0580ea0c12499c564e2328d"
WORK_DIR="${NATIVE_RING_WORK_DIR:-/tmp/ringlpn-native-ring-pcg}"
OUT="${OUT:-${ROOT}/results/pcg/native_ring_pcg_adapted_2026_07_29.csv}"
LOG="${LOG:-${ROOT}/results/pcg/native_ring_pcg_adapted_2026_07_29.log}"
TRIALS="${TRIALS:-3}"
CPU="${CPU:-4}"
NS="${NS:-13 15}"
C="${C:-5}"
T="${T:-27}"

if [[ ! -d "${WORK_DIR}/.git" ]]; then
  rm -rf "${WORK_DIR}"
  git clone --filter=blob:none https://github.com/zhli271828/Trace-F2-OLE-PCG.git "${WORK_DIR}"
fi
git -C "${WORK_DIR}" fetch --quiet origin "${REV}"
git -C "${WORK_DIR}" checkout --quiet --detach "${REV}"
git -C "${WORK_DIR}" checkout --quiet -- src
git -C "${WORK_DIR}" submodule update --init --recursive --quiet
test "$(git -C "${WORK_DIR}" rev-parse HEAD)" = "${REV}"

mkdir -p "$(dirname "${OUT}")"
{
  echo "source=https://github.com/zhli271828/Trace-F2-OLE-PCG"
  echo "revision=${REV}"
  echo "license=MIT"
  echo "evidence_class=adapted"
  echo "params=c=${C},t=${T},n_log3 in {${NS}},trials=${TRIALS}"
  echo "cflags=$(sed -n 's/^CFLAGS += //p' "${WORK_DIR}/Makefile")"
  echo "ldflags=$(sed -n 's/^LDFLAGS = //p' "${WORK_DIR}/Makefile")"
  echo "cpu_pin=${CPU}"
  echo "cpu_model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | sed -n '1p')"
  echo "host=$(uname -srmo)"
  echo "note=single process, both parties' expansion measured together, no network"
  echo "note=setup is centralized: the benchmark samples both DPF key halves locally"
  echo "note=correctness gate is the artifact's own F4 OLE PCG test, not a Z_2^k triple check"
} >"${LOG}"

printf 'bench,correlation,ring,n_log3,c,t,trials,setup_ms,expand_ms,total_ms,peak_rss_kb,status\n' >"${OUT}"

apply_patch() {
  local n="$1"
  git -C "${WORK_DIR}" checkout --quiet -- src

  # (1) typed shifts so the 58-bit and 121-bit SPDZ2k moduli are representable.
  test "$(grep -c 'modulus64 = 1<<(k+s);' "${WORK_DIR}/src/modular_bench.c")" -eq 2
  test "$(grep -c 'modulus128 = 1<<(k+s);' "${WORK_DIR}/src/modular_bench.c")" -eq 2
  sed -i 's/modulus64 = 1<<(k+s);/modulus64 = ((uint64_t)1)<<(k+s);/g; s/modulus128 = 1<<(k+s);/modulus128 = ((uint128_t)1)<<(k+s);/g' \
    "${WORK_DIR}/src/modular_bench.c"
  test "$(grep -c '((uint64_t)1)<<(k+s)' "${WORK_DIR}/src/modular_bench.c")" -eq 2
  test "$(grep -c '((uint128_t)1)<<(k+s)' "${WORK_DIR}/src/modular_bench.c")" -eq 2

  # (2) post-ePrint comparison parameter set and bounded trial count.
  python3 - "${WORK_DIR}/src/main.c" "${TRIALS}" "${n}" "${C}" "${T}" <<'PY'
import re, sys
path, trials, n, c, t = sys.argv[1], *sys.argv[2:]
src = open(path).read()
src, k = re.subn(r'int num_trials = 10;', f'int num_trials = {trials};', src)
assert k == 1, k
old = """void pcg_bm_with_param(int num_trials, bench_func bf) {

    printf("******************************************\\n");
    size_t c = 3;
    size_t t = 27;
    size_t n = 15;
"""
new = f"""void pcg_bm_with_param(int num_trials, bench_func bf) {{

    printf("******************************************\\n");
    size_t c = {c};
    size_t t = {t};
    size_t n = {n};
"""
assert old in src, "shipped pcg_bm_with_param body changed"
open(path, 'w').write(src.replace(old, new))
PY

  git -C "${WORK_DIR}" diff -- src >"${WORK_DIR}/ringlpn_adapted_n${n}.patch"
  echo "patch_n${n}=${WORK_DIR}/ringlpn_adapted_n${n}.patch" >>"${LOG}"
  echo "patch_n${n}_sha256=$(sha256sum "${WORK_DIR}/ringlpn_adapted_n${n}.patch" | cut -d' ' -f1)" >>"${LOG}"

  make -C "${WORK_DIR}" clean >/dev/null
  make -C "${WORK_DIR}" -j"$(nproc)" >"${WORK_DIR}/build_n${n}.log" 2>&1
}

run_bench() {
  local flag="$1" correlation="$2" ring="$3" n="$4" rc=0 rss out
  rss="${WORK_DIR}/rss_${flag#--}_n${n}.txt"
  set +e
  taskset -c "${CPU}" /usr/bin/time -f '%M' -o "${rss}" \
    "${WORK_DIR}/bin/pcg" "${flag}" >"${WORK_DIR}/run.txt" 2>&1
  rc=$?
  set -e
  cat "${WORK_DIR}/run.txt" >>"${LOG}"
  out="$(sed -n "s/^N=3\\^\\([0-9]*\\), c=\\([0-9]*\\), t=\\([0-9]*\\): Avg PP time \\([0-9.]*\\) ms, expand time \\([0-9.]*\\) ms, total time \\([0-9.]*\\) ms\$/\\1,\\2,\\3,${TRIALS},\\4,\\5,\\6/p" "${WORK_DIR}/run.txt")"
  if [[ ${rc} -eq 0 && -n "${out}" ]]; then
    printf '%s,%s,%s,%s,%s,ok\n' "${flag#--}" "${correlation}" "${ring}" "${out}" "$(tail -1 "${rss}")" >>"${OUT}"
  else
    # 137 is SIGKILL, which on this host means the OOM killer stopped the run.
    printf '%s,%s,%s,%s,%s,%s,%s,,,,%s,exit_%s\n' \
      "${flag#--}" "${correlation}" "${ring}" "${n}" "${C}" "${T}" "${TRIALS}" \
      "$(tail -1 "${rss}" 2>/dev/null || echo)" "${rc}" >>"${OUT}"
    echo "[warn] ${flag} at n=3^${n} exited ${rc}; recorded as a failed row" >&2
  fi
}

for n in ${NS}; do
  apply_patch "${n}"
  if [[ "${n}" = "${NS%% *}" ]]; then
    printf 'correctness_gate,result\n' >"${OUT}.gate"
    if taskset -c "${CPU}" "${WORK_DIR}/bin/pcg" --modular_test >>"${LOG}" 2>&1; then
      printf 'artifact_f4_ole_pcg_test,pass\n' >>"${OUT}.gate"
    else
      printf 'artifact_f4_ole_pcg_test,FAIL\n' >>"${OUT}.gate"
      echo "correctness gate failed" >&2
      exit 1
    fi
  fi
  run_bench --gr128_trace_bench semi_honest_mult_triple Z_2^64 "${n}"
  run_bench --SPDZ2k_64_bench spdz2k_authenticated_triple Z_2^64_with_mac "${n}"
done

grep -q ',ok$' "${OUT}"
printf 'Wrote %s, %s.gate and %s\n' "${OUT}" "${OUT}" "${LOG}"
