#!/usr/bin/env bash
# Two-PROCESS distributed DPF keygen with the full-width GPU AES expansion,
# validated by the same GPU evaluator used by the Ring-LPN OLE engine.
#
# Stage 1 (CPU, two OS processes): generate keys with `--prg gpu-aes`, so the
# host expansion is bit-identical to the device `aes_prg_expand`.
# Stage 2 (GPU, offline): `test_two_party_gpu_dpf_eval` builds GPUDPFZpKey for
# each party and runs `gpuDpfZpFullEvalSum`, checking batched SPFSS semantics,
# per-tree semantics, and a corrupted-correction-word negative control.
#
# Requires nvcc and a GPU. Outputs:
#   results/dpf/two_party_gpu_dpf_2026_07_29.csv
#   results/dpf/two_party_gpu_dpf_2026_07_29.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
KEYGEN="$ROOT/host_bin/test_two_party_dpf_keygen"
EVAL="$ROOT/bin/test_two_party_gpu_dpf_eval"
PARITY="$ROOT/host_bin/test_gpu_aes_prg_parity"
DUMP="$ROOT/bin/dump_gpu_aes_prg_vectors"
OUTDIR="$ROOT/results/dpf"
WORKDIR="${WORKDIR:-$OUTDIR/two_party_gpu_keys}"
CSV="$OUTDIR/two_party_gpu_dpf_2026_07_29.csv"
LOG="$OUTDIR/two_party_gpu_dpf_2026_07_29.log"
BASE_PORT="${BASE_PORT:-45200}"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="${NVCC:-nvcc}"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "[two-party-gpu] nvcc not found; run inside the CUDA environment"
  exit 1
fi

"$ROOT/scripts/build_two_party_dpf_keygen.sh" >/dev/null
g++ -std=c++17 -O2 -Wall -Wextra -I "$ROOT/src" \
  "$ROOT/src/test_gpu_aes_prg_parity.cpp" \
  -o "$PARITY" -lcrypto

mkdir -p "$ROOT/bin" "$OUTDIR" "$WORKDIR"
"$NVCC" -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
  -I"$PROJECT_ROOT" \
  -I"$PROJECT_ROOT/ext/sytorch/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack" \
  -I"$ROOT/src" \
  "$ROOT/src/test_two_party_gpu_dpf_eval.cu" \
  "$ROOT/src/spfss_host.cpp" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  -o "$EVAL" 2> >(grep -v "warning #20012-D" >&2 || true)
"$NVCC" -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
  -I"$PROJECT_ROOT" \
  -I"$PROJECT_ROOT/ext/sytorch/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/cryptoTools" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/llama/include" \
  -I"$PROJECT_ROOT/ext/sytorch/ext/bitpack" \
  -I"$ROOT/src" \
  "$ROOT/src/dump_gpu_aes_prg_vectors.cu" \
  "$PROJECT_ROOT/utils/gpu_mem.cu" \
  -o "$DUMP" 2> >(grep -v "warning #20012-D" >&2 || true)

: > "$LOG"
rm -f "$CSV"

"$DUMP" 16 > "$OUTDIR/gpu_aes_prg_vectors_2026_07_29.csv"
# The host/device PRG parity gate must pass before any key is generated.
echo "[two-party-gpu] host/device PRG parity" | tee -a "$LOG"
"$PARITY" --vectors "$OUTDIR/gpu_aes_prg_vectors_2026_07_29.csv" \
  >> "$LOG" 2>> "$LOG"

# config: log_domain batch_trees modulus_idx
CONFIGS=(
  "4 8 0"
  "8 16 0"
  "11 32 0"
  "11 32 1"
)

port=$BASE_PORT
header=1
status=0
for cfg in "${CONFIGS[@]}"; do
  read -r L TREES MIDX <<<"$cfg"
  prefix="$WORKDIR/gpuL${L}_m${MIDX}_b${TREES}"
  rm -f "${prefix}_p0.key" "${prefix}_p1.key" \
        "${prefix}_p0.testmeta" "${prefix}_p1.testmeta"
  echo "=== gpu-aes L=$L trees=$TREES modulus_idx=$MIDX port=$port ===" >> "$LOG"

  "$KEYGEN" --party 0 --port "$port" --log-domain "$L" --trees "$TREES" \
            --modulus-idx "$MIDX" --prg gpu-aes --selftest 8 \
            --input-seed "$((L * 100 + MIDX))" --out-prefix "$prefix" \
            >> "$LOG" 2>> "$LOG" &
  p0=$!
  sleep 0.2
  "$KEYGEN" --party 1 --host 127.0.0.1 --port "$port" --log-domain "$L" \
            --trees "$TREES" --modulus-idx "$MIDX" --prg gpu-aes --selftest 8 \
            --input-seed "$((L * 100 + MIDX))" --out-prefix "$prefix" \
            >> "$LOG" 2>> "$LOG" &
  p1=$!
  rc0=0; rc1=0
  wait "$p0" || rc0=$?
  wait "$p1" || rc1=$?
  if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    echo "[two-party-gpu] keygen FAILED (p0=$rc0 p1=$rc1) L=$L m=$MIDX" | tee -a "$LOG"
    status=1
  fi

  hdr=()
  if [[ $header -eq 1 ]]; then hdr=(--csv-header); header=0; fi
  if ! "$EVAL" --prefix "$prefix" "${hdr[@]}" >> "$CSV" 2>> "$LOG"; then
    echo "[two-party-gpu] GPU validation FAILED L=$L m=$MIDX" | tee -a "$LOG"
    status=1
  fi
  tail -1 "$CSV" >> "$LOG"
  port=$((port + 4))
done

echo "wrote $CSV"
echo "wrote $LOG"
if [[ $status -ne 0 ]] || grep -q "FAIL" "$CSV"; then
  echo "[two-party-gpu] FAILURES present"
  exit 1
fi
echo "[two-party-gpu] all configurations pass"
