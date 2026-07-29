# External NTT baseline: GPU-NTT (merge) vs cheddar backend — 2026-06-10

Answers "should we adopt the radix-2 CT / 4-step kernels from GPU-NTT
(Ozcan–Savas, eprint 2023/1410) instead of the cheddar derivation?" with
measurements instead of opinion. Harness: `src/bench_ntt_gpu_ntt_baseline.cu`
— both backends in one process, same prime, same psi, same negacyclic polymul
(2 forward NTT + Hadamard + inverse NTT), both GPU outputs validated
elementwise against `host_polymul_reference`. RTX 5000 Ada, CUDA 12.6.

Provenance update (2026-07-29): the active backend is substantially derived
from MIT-licensed Cheddar; its reconstructed source/blob pin, upstream notice,
and local delta are recorded in `extern/Cheddar_PROVENANCE.txt` and
`extern/Cheddar_MIT_LICENSE.txt`. GPU-NTT remains an external Apache-2.0
baseline at local clean checkout `95c739c48d11827277e132f5eec4d4e454d60835`.

## Result table (`ntt/ntt_gpu_ntt_baseline_compare.csv`)

PolyMul mean µs (validation `pass` unless noted):

| prime | n | batch | GPU-NTT merge | cheddar | ratio |
|---|---|---|---|---|---|
| pool60 | 8192 | 4 | 24.8 | 42.9 | 1.7x |
| pool60 | 8192 | 64 | 66.5 | 258.9 | **3.9x** |
| pool60 | 65536 | 4 | 49.1 | 63.4 | 1.3x |
| pool60 | 65536 | 64 | 560.4 | 693.8 | 1.2x |
| pool60 | 2^20 | 2 | 312.8 | 477.0 | 1.5x |
| p62 (ours) | 8192 | 4 | **unsupported** | 40.1 | — |
| p62 (ours) | 8192 | 64 | **unsupported** | 253.7 | — |

pool60 = 576460756061519873 (60-bit GPU-NTT default-pool prime, v2(q−1)=29).
p62 = 4611686018326724609 (the project's deployment prime).

## Findings

1. **At a common 60-bit prime, GPU-NTT's merge kernel is faster than the
   cheddar backend: 1.2–3.9x**, largest at n=8192/batch=64. The earlier claim
   that the cheddar backend is competitive with external kernels is hereby
   revised with data.
2. **GPU-NTT cannot run the project's 62-bit primes.** Measured, not assumed:
   the identical call sequence round-trips correctly at the 60-bit pool prime
   and produces out-of-range values at p62 — its 64-bit Barrett reduction has
   no headroom above ~60-bit moduli. The cheddar backend's signed Montgomery
   supports the 62-bit class; part of GPU-NTT's speed is purchased by the
   smaller modulus class. Additionally, **the 4-step variant cannot accept an
   externally chosen prime at all** (upstream's custom-prime
   `NTTParameters4Step` constructor is commented out), so no apples-to-apples
   4-step row exists.
3. **Pipeline impact of switching today: nil.** NTT/PolyMul is <1% of OLE
   expand (SPFSS evaluation dominates; see
   `reports/orca_fc_real_ole_transcript_memo.md`), so a 1.2–3.9x NTT win
   moves end-to-end cost by approximately nothing.

## Decision

Keep the cheddar backend for now: it runs the deployment primes, the whole
validation surface is bound to it, and the NTT is not the bottleneck. Two
revisit triggers, recorded for the S2/M5 parameter audit in the current v2.3
proposal and `s2_parameter_novelty_provenance_audit_2026_07_29.md`:

- If M5 re-pins parameters anyway, two ~60-bit CRT primes give M ≈ 2^120,
  which still satisfies the bw=32 no-wrap bound (K·2^66 < M) with enormous
  margin — that migration would unlock the faster Barrett modulus class and
  make GPU-NTT (or a port of its butterfly structure) a drop-in win.
- If profiling at M5-scale parameters ever shows NTT exceeding ~10% of
  expand, port the merge-kernel butterfly improvements rather than the
  library (the 62-bit signed-Montgomery reduction must be kept).

## Reproduction

External dependency (benchmark-only, not in the gate): GPU-NTT checkout at
`$GPU_NTT_HOME` (default `/home/fatih/GPU-NTT`), built with
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
&& cmake --build build`.

```bash
bash scripts/build_ntt_gpu_ntt_baseline.sh
bash scripts/run_ntt_baseline_compare.sh
```
