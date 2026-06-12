> **HISTORICAL DOCUMENT (pre-2026-06-10).** Superseded by `GPU-MPC/ringlpn/CLAUDE.md` and the newer reports indexed in `results/README.md`. Statements below describe an **older state of the code** (e.g., "real OLE pending", "q128 summaries missing", "dense packing not implemented" are all RESOLVED since). Quote for history; do not treat as current.

# Ring-LPN Status Report

Generated: 2026-04-09
Updated: 2026-05-16 for the promoted q128 CRT NTT/PolyMul path and VOLE wiring

## Executive Summary

This report summarizes the current implementation status of the Ring-LPN benchmarking track under `GPU-MPC/ringlpn`, with emphasis on the promoted CUDA NTT work derived from cheddar-fhe and the newer standalone online-phase benchmarks, Figure 2 OLE artifact, and ring-polynomial linear-layer artifact built around it.

The project now has nine distinct benchmark/demo layers:

1. a CPU baseline built on NFLLib,
2. an archived legacy CUDA implementation built around a `phi` preprocessing plus fused-first-8-stage design,
3. a promoted primary CUDA implementation extracted from cheddar-fhe and adapted into a standalone Ring-LPN benchmark harness,
4. a standalone Ring-LPN VOLE prototype benchmark built on the promoted Cheddar CUDA PolyMul path,
5. a standalone GPU Figure 2 SPFSS/OLE artifact that validates `z_0 + z_1 == x_0 * x_1` over the promoted single 62-bit prime path,
6. a standalone ring-polynomial linear-layer OLE-to-Beaver artifact that validates matrix multiplication over `Z_p[X]/(X^N+1)`,
7. a host-only Orca scalar bridge smoke for constant-polynomial packing and exact `Z_p -> Z_{2^bw}` dealer/oracle share conversion,
8. a tiny forward-only Orca FC key-writer demo that emits raw `A`, `B`, `C_masked` buffers and validates unchanged `gpuMatmulBeaver`,
9. a standalone DPF online key generation benchmark that measures one-shot versus chunked partial generation.

The main engineering result of this phase is that the cheddar-derived implementation is no longer only a side experiment. It has now been integrated into the main Ring-LPN CUDA pipeline as the default implementation behind `bench_ntt_cuda`, while the older CUDA path is archived and requires `ALLOW_LEGACY_CUDA_NTT=1` for historical comparison runs.

At the same time, the project remains intentionally staged. The current main GPU path now supports requested `q=32`, `q=64`, and `q=128`: q=32 and q=64 use one 30-bit or 62-bit prime, while q=128 uses two q62 CRT prime limbs in the flattened Cheddar launch schedule and reports actual `qbits=124`. The standalone VOLE prototype now uses that same Cheddar residue-limb path for q128. The Figure 2 OLE, linear-layer, and Orca-facing artifacts are still single-prime q62 unless stated otherwise. High-density packing, q128 wiring into those SPFSS/OLE and Orca artifacts, secure distributed conversion, and full Orca training/backward integration remain open.

## Project Objective

The Ring-LPN subproject is a standalone benchmarking harness for NTT, inverse NTT, and full polynomial multiplication over the parameter ranges relevant to Ring-LPN work. It is separate from the Orca training and inference pipeline, even though both live under `GPU-MPC`.

The immediate objective of the CUDA work has been:

1. establish a valid CPU baseline,
2. build a generalized GPU q=32 path over the full degree range from `8192` through `1048576`,
3. extract the stronger NTT/INTT kernel structure from cheddar-fhe into a self-contained local benchmark,
4. promote that extracted implementation to the main Ring-LPN GPU path without importing cheddar-fhe's full runtime stack,
5. extend that promoted path from requested `q=32` to requested `q=64`,
6. extend the promoted Cheddar kernel schedule to requested `q=128` with two q62 CRT prime limbs,
7. prototype a standalone Ring-LPN VOLE-style expansion layer on top of the promoted GPU PolyMul path,
8. prototype a standalone GPU Figure 2 SPFSS/OLE artifact over `Z_p[X]/(X^N+1)`,
9. prototype a standalone ring-polynomial linear-layer OLE-to-Beaver artifact,
10. prototype and validate the first Orca-facing scalar bridge boundary from `Z_p` OLE/Beaver shares to `Z_{2^bw}`,
11. prototype a tiny forward-only Orca FC key-writer demo using raw `A`, `B`, `C_masked` buffers and unchanged `gpuMatmulBeaver`,
12. prototype a standalone DPF online key generation benchmark that quantifies peak staged key-footprint reduction from chunked generation.

## Current Code and Filesystem State

The current high-signal files are:

| Path | Role |
| --- | --- |
| `src/bench_ntt.cpp` | NFLLib-backed CPU reference benchmark |
| `src/bench_ntt_cuda.cu` | Archived legacy CUDA benchmark retained for opt-in historical comparison |
| `src/bench_ntt_cuda_cheddar.cu` | Primary CUDA benchmark source, extracted from cheddar-fhe and adapted locally |
| `src/bench_vole_ringlpn.cu` | Standalone Ring-LPN VOLE prototype benchmark |
| `src/gpu_spfss_zp.cuh` | Standalone GPU DPF/SPFSS path with additive `Z_p` payload shares |
| `src/test_spfss_zp_cuda.cu` | GPU SPFSS payload correctness test |
| `src/bench_ole_ringlpn_cuda.cu` | Standalone GPU Figure 2 SPFSS/OLE benchmark |
| `src/bench_linear_ole_ringlpn_cuda.cu` | Standalone ring-polynomial linear-layer OLE-to-Beaver benchmark |
| `src/test_orca_zp_bridge.cpp` | Host-only Orca scalar bridge test for carry-corrected `Z_p -> Z_{2^bw}` conversion |
| `src/bench_orca_fc_ringlpn_demo.cu` | Tiny Orca FC key-writer demo over bounded q62 constant-polynomial masks |
| `../tests/fss/dpf_online_keygen_bench.cu` | Standalone DPF online key generation benchmark |
| `scripts/build_bench.sh` | CPU build entry point |
| `scripts/build_cuda_bench.sh` | Main CUDA build entry point, now targeting the cheddar-derived implementation |
| `scripts/build_vole_bench.sh` | VOLE benchmark build entry point |
| `scripts/build_ole_cuda_bench.sh` | Figure 2 OLE benchmark and GPU SPFSS test build entry point |
| `scripts/build_linear_ole_bench.sh` | Ring-polynomial linear OLE-to-Beaver build entry point |
| `scripts/build_cuda_bench_cheddar.sh` | Explicit standalone cheddar-derived alias build |
| `scripts/build_cuda_bench_legacy.sh` | Archived opt-in legacy CUDA build |
| `scripts/run_sweep.sh` | CPU sweep driver |
| `scripts/run_cuda_sweep.sh` | Main CUDA sweep driver |
| `scripts/run_vole_sweep.sh` | VOLE sweep driver |
| `scripts/run_ole_sweep.sh` | Figure 2 OLE smoke and bounded sweep driver |
| `scripts/summarize_ole_results.py` | Figure 2 OLE CSV-to-Markdown summarizer |
| `scripts/run_linear_ole_sweep.sh` | Ring-polynomial linear OLE-to-Beaver smoke driver |
| `scripts/summarize_linear_ole_results.py` | Ring-polynomial linear OLE CSV-to-Markdown summarizer |
| `scripts/build_orca_zp_bridge_test.sh` | Host-only scalar bridge build entry point |
| `scripts/run_orca_zp_bridge_test.sh` | Scalar bridge smoke and q62/full-32-bit counterexample driver |
| `scripts/build_orca_fc_ringlpn_demo.sh` | Orca FC demo build entry point |
| `scripts/run_orca_fc_ringlpn_demo.sh` | Orca FC demo run driver |
| `scripts/summarize_orca_fc_demo.py` | Orca FC demo CSV-to-Markdown summarizer |
| `scripts/run_paper_checkpoint_smoke.sh` | Consolidated host smoke with optional CUDA OLE/linear/FC smoke inside the container |
| `scripts/run_cuda_sweep_legacy.sh` | Archived opt-in legacy CUDA sweep driver |
| `scripts/run_cuda_single.sh` | CPU-vs-GPU spot check on CPU-overlap points |
| `../scripts/run_dpf_online_keygen_sweep.py` | DPF online key generation sweep driver |
| `results/ntt_cpu.md` | CPU baseline summary |
| `results/ntt_gpu_q32.md` | Current main CUDA summary |
| `results/ntt_gpu_q64.md` | Current main CUDA q=64 summary |
| `results/ntt_gpu_q128.md` | Current main CUDA q=128 CRT summary |
| `results/ntt_gpu_q32_legacy.md` | Legacy CUDA summary |
| `results/vole_gpu_q32_m32_c2_w64.md` | Current standalone VOLE q=32 summary |
| `results/vole_gpu_q64_m32_c2_w64.md` | Current standalone VOLE q=64 summary |
| `results/vole_gpu_q128_smoke.md` | Current standalone VOLE q=128 CRT smoke summary |
| `results/ole_gpu_handoff.md` | Current Figure 2 OLE handoff, claims, caveats, and commands |
| `results/linear_ole_handoff.md` | Current linear-layer OLE-to-Beaver handoff, claims, caveats, and commands |
| `results/orca_zp_bridge_handoff.md` | Current Orca-facing scalar bridge handoff and counterexample |
| `results/orca_fc_ringlpn_demo_memo.md` | Professor-facing v1 Orca FC demo memo |
| `results/orca_fc_ringlpn_demo_bounded_suite.md` | Current tiny Orca FC bounded-suite result summary |
| `results/paper_execution_next_steps.md` | One-command smoke, hygiene notes, and paper-oriented next checkpoints |
| `results/ole_gpu_q64_uniform_c2_t8_smoke.md` | Figure 2 OLE smoke result summary |
| `results/ole_gpu_q64_uniform_c2_t64.md` | Figure 2 OLE bounded result summary |
| `results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md` | Current ring-polynomial linear-layer smoke summary |
| `results/dpf_online_keygen_bin16_chunk8192.md` | Current standalone DPF online key generation summary |
| `results/ringlpn_vole_abstract_support.md` | Current abstract-safe support note for VOLE plus DPF online key generation |
| `results/cheddar_extract_note.md` | Detailed extraction rationale and earlier benchmark comparison |

## What We Have Implemented So Far

### 1. CPU baseline

The CPU benchmark in `src/bench_ntt.cpp` is complete as a reference harness. It:

1. resolves requested modulus sizes into actual NFLLib-supported configurations,
2. validates roundtrip NTT/INTT and negacyclic multiplication behavior,
3. times forward NTT, inverse NTT, and full polynomial multiplication,
4. reports consistent CSV and Markdown outputs.

The current requested-to-actual modulus contract is:

| Requested qbits | Actual qbits | Backend mode |
| --- | --- | --- |
| 32 | 30 | NFLLib uint32 |
| 64 | 62 | NFLLib uint64 |
| 128 | 124 | NFLLib uint64 with two 62-bit limbs |

This CPU baseline is important because it defines the correctness and reporting contract that the GPU side must match for larger bitwidths. That contract is now met by the promoted GPU q=64 path and the promoted q=128 CRT residue-limb path; CPU q=128 remains the comparison anchor for deeper cross-checking.

### 2. Legacy CUDA q=32 benchmark

The original CUDA path in `src/bench_ntt_cuda.cu` remains present and working. It already completed the original step-1 roadmap item:

1. accepted any power-of-two degree from `8192` through `1048576`,
2. introduced batching,
3. validated roundtrip and polynomial multiplication correctness,
4. produced the full q=32 sweep used earlier in the project.

Architecturally, this path is based on:

1. `phi` and `invphi` preprocessing,
2. fused shared-memory execution for the first eight stages,
3. tail-stage kernels for the remaining stages,
4. a separate postprocessing step for inverse scaling.

This implementation is now explicitly treated as the legacy baseline, not the default direction for future work.

### 3. Cheddar-fhe CUDA kernel extraction

The key implementation result is the standalone extracted benchmark in `src/bench_ntt_cuda_cheddar.cu`.

What was brought over conceptually from cheddar-fhe:

1. two-phase NTT decomposition,
2. two-phase inverse NTT decomposition,
3. launch-configuration-driven stage structure,
4. OF-twiddle handling via an MSB twiddle table,
5. Montgomery-butterfly execution ordering.

What was adapted locally for Ring-LPN:

1. replacement of cheddar-fhe container abstractions with raw CUDA allocations and local host tables,
2. a local flattened `(batch, prime)` benchmark layout,
3. use of one prime limb for q=32/q=64 and two q62 CRT limbs for q=128,
4. a local pointwise multiplication kernel,
5. local validation against host reference code,
6. support widened to `log2(n)` in `[13, 20]`, corresponding to `n` in `[8192, 1048576]`.

In other words, the extracted code is not a thin wrapper around cheddar-fhe. It is a local benchmark implementation that preserves the kernel architecture but removes the original framework dependency.

### 4. Promotion of cheddar-derived kernels to the main path

This implementation pass completes an important integration step: the cheddar-derived source is now the default implementation behind `scripts/build_cuda_bench.sh` and the main binary `bin/bench_ntt_cuda`.

Concretely, the project now provides:

| Binary | Source | Purpose |
| --- | --- | --- |
| `bin/bench_ntt_cuda` | `src/bench_ntt_cuda_cheddar.cu` | Primary Ring-LPN CUDA benchmark |
| `bin/bench_ntt_cuda_cheddar` | `src/bench_ntt_cuda_cheddar.cu` | Explicit standalone cheddar-derived binary |
| `bin/bench_ntt_cuda_legacy` | `src/bench_ntt_cuda.cu` | Archived baseline for opt-in historical comparison |

This matters because the extraction is now operationally complete for the promoted GPU path, including q128 CRT residue-limb scheduling. The code is no longer living only as a side file; it is the main GPU path used by the standard sweep script.

### 5. q=64 and q=128 extensions on the promoted main path

The promoted cheddar-derived path now also supports requested `q=64`, realized with one 62-bit prime over the full `n = 8192 ... 1048576` range.

The implementation work in this phase added:

1. a 64-bit Montgomery specialization using `__umul64hi()` for the 128-bit intermediate product,
2. runtime selection between a 30-bit and 62-bit single-prime configuration,
3. 64-bit twiddle, inverse-twiddle, inverse-degree, and Montgomery-conversion table generation,
4. 64-bit host reference validation for roundtrip NTT/INTT and negacyclic polynomial multiplication,
5. q=64 sweep tooling and result generation under the same promoted `bench_ntt_cuda` binary.

The same promoted source now supports requested `q=128`, realized as actual `qbits=124` with two q62 CRT prime limbs. The implementation work for q128 added:

1. a second 62-bit NTT prime with a primitive `2^21` root for degrees through `2^20`,
2. prime-indexed Cheddar table construction for twiddles, inverse twiddles, inverse-degree constants, Montgomery conversion constants, and inverse-prime constants,
3. flattened `(batch, prime)` scheduling through `grid.y` for all four Cheddar phase kernels,
4. pointwise multiplication that selects the correct modulus per residue limb,
5. validation over zero, one, impulse, max, and random CRT residue patterns, plus q128 sweep tooling and result generation.

### 6. Standalone Ring-LPN VOLE prototype

The project now also includes a standalone Ring-LPN VOLE prototype in `src/bench_vole_ringlpn.cu`.

This implementation is intentionally scoped as a correctness-first online-phase prototype rather than a full end-to-end SPFSS-backed system.

What it does today:

1. reuses the promoted CUDA polynomial multiplication backend from `src/bench_ntt_cuda_cheddar.cu`,
2. synthesizes MPVOLE-consistent inputs locally under the `synthetic_mpvole` mode,
3. validates the coefficient-wise relation `z = y + x * Delta`,
4. supports requested `q=32`, `q=64`, and `q=128`,
5. supports `n = 8192 ... 1048576`.

Current result summaries:

- `results/vole_gpu_q32_m32_c2_w64.md`
- `results/vole_gpu_q64_m32_c2_w64.md`
- `results/vole_gpu_q128_smoke.md`

The q128 path uses the same flattened two-limb q62 Cheddar CRT layout as the core NTT/PolyMul benchmark. The saved smoke sweep uses `m=2`, `c=2`, noise weight `8`, covers the full degree set through `n=1048576`, and passes every validation row. Run `QBITS=128 ./scripts/run_vole_sweep.sh` to generate the full default q128 VOLE sweep artifact.

Key current numbers:

- q=32 full expansion latency ranges from `269.484 us` at `n=8192` to `43.392 ms` at `n=1048576`,
- q=64 full expansion latency ranges from `772.324 us` at `n=8192` to `67.532 ms` at `n=1048576`,
- all sweep points passed validation.

### 7. Standalone GPU Figure 2 SPFSS/OLE artifact

The project now includes a standalone GPU artifact for the Figure 2 SPFSS-based Ring-LPN OLE path in `src/bench_ole_ringlpn_cuda.cu`.

This implementation is intentionally scoped as a correctness-first primitive artifact, not an Orca Beaver-triple integration.

What it does today:

1. uses the promoted single 62-bit prime path, reported as requested `qbits=64` and actual `qbits=62`,
2. samples either uniform-position `t`-sparse noise or regular sparse noise with one point per bucket,
3. evaluates SPFSS over `[0, 2N)` for uniform noise, or grouped domains of size `2N/t` for regular noise, and folds to degree `< N` using `X^N = -1`,
4. uses a new GPU DPF/SPFSS path with additive `uint64_t` payload shares modulo `p` in `src/gpu_spfss_zp.cuh`,
5. validates `z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N+1)`,
6. keeps existing packed one-bit DPF callers unchanged.

Current result summaries:

- `results/ole_gpu_q64_uniform_c2_t8_smoke.md`
- `results/ole_gpu_q64_regular_c2_t8_smoke.md`
- `results/ole_gpu_q64_uniform_c2_t64.md`
- `results/ole_gpu_q64_regular_c2_t64.md`
- `results/ole_gpu_handoff.md`

Current bounded results:

| Run | n | c | t | Validation | Host validation | Pair key bytes | Keygen us | OLE expand mean us |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| uniform smoke | 8192 | 2 | 8 | pass | pass | 141,504 | 443 | 13,278 |
| regular smoke | 8192 | 2 | 8 | pass | pass | 116,544 | 4,977 | 6,823 |
| uniform bounded | 8192 | 2 | 64 | pass | pass | 9,044,160 | 4,797 | 865,253 |
| uniform bounded | 16384 | 2 | 64 | pass | skipped | 9,633,984 | 5,296 | 1,830,210 |
| regular bounded | 8192 | 2 | 64 | pass | pass | 5,529,408 | 40,828 | 58,462.5 |
| regular bounded | 16384 | 2 | 64 | pass | skipped | 6,119,232 | 42,331 | 67,733 |

The scientific claim for this direct OLE artifact is still deliberately narrow: it validates the Figure 2 OLE relation on GPU for single-prime q62 under uniform and regular sparse noise. CRT q=128 and Orca FC integration are not validated here. The separate linear artifact below now validates OLE-to-Beaver conversion for ring-polynomial matrix entries, and the scalar bridge smoke records the first dealer/oracle `Z_p -> Z_{2^bw}` conversion boundary.

### 8. Standalone ring-polynomial linear-layer OLE-to-Beaver artifact

The project now includes a standalone linear-layer artifact in `src/bench_linear_ole_ringlpn_cuda.cu`.

This artifact applies the standard two-OLE-to-Beaver conversion to matrix multiplication over ring-polynomial entries in `Z_p[X]/(X^N+1)`.

For each ring-product term:

1. one Figure 2 OLE gives shares of `A_0 * B_1`,
2. a second Figure 2 OLE gives shares of `A_1 * B_0`,
3. party 0 locally adds `A_0 * B_0`,
4. party 1 locally adds `A_1 * B_1`,
5. the result satisfies `C_0 + C_1 = (A_0 + A_1) * (B_0 + B_1)`.

For a matrix product, the benchmark accumulates those Beaver ring products across the inner dimension. The current implementation samples each shared `A[row,k]` and `B[k,col]` operand once and reuses it across all products that reference it; `shared_operands=1` is reported in CSV as a regression check.

Current result summary:

- `results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md`
- `results/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.md`
- `results/linear_ole_handoff.md`

Current smoke result:

| noise | rows | inner | cols | n | c | t | Validation | Shared operands | Ring products | OLE instances | Pair key bytes | Keygen us | Linear expand mean us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2 | 2 | 2 | 8192 | 2 | 8 | pass | 1 | 8 | 16 | 2,264,064 | 6,587 | 223,667 |
| regular | 2 | 2 | 2 | 8192 | 2 | 8 | pass | 1 | 8 | 16 | 1,864,704 | 81,582 | 114,825 |

This is the first implemented linear-layer OLE-to-Beaver bridge, but it is still a ring-polynomial layer. The Orca FC demo below validates a tiny bounded scalar key-writer path, while high-density scalar packing, secure distributed `Z_p -> Z_{2^bw}` conversion, and q128/CRT remain open.

### 9. Host-only Orca Zp-to-Z2k scalar bridge smoke

The project now includes a host-only bridge smoke in `src/test_orca_zp_bridge.cpp`.

This checkpoint validates:

1. exact carry-corrected dealer/oracle conversion from `Z_p` shares to `Z_{2^bw}` shares,
2. conservative constant-polynomial scalar packing under the explicit no-prime-wrap bound `inner * value_bound^2 < p`,
3. a negative q62/full-32-bit counterexample that prevents an invalid unrestricted Orca scalar-product claim.

Current result summary:

- `results/orca_zp_bridge_constant_scalar.md`
- `results/orca_zp_bridge_handoff.md`

Current smoke result:

| bw | rows | inner | cols | value bound | Naive share failures | Corrected share failures | No-prime-wrap bound | Scalar validation | Counterexample |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 16 | 2 | 2 | 2 | 255 | 633 | 0 | yes | pass | no |
| 32 | 1 | 1 | 1 | 4294967295 | 633 | 0 | no | not claimed | yes |

The implemented correction is `r0 = z0 mod 2^bw`, `r1 = z1 - m*p mod 2^bw`, where `m = floor((z0 + z1) / p)`. The q62/full-32-bit row is intentionally a non-claiming counterexample: unrestricted 32-bit scalar products still need q128/CRT, tighter layer bounds, or another conversion argument.

### 10. Tiny Orca FC Ring-LPN key-writer demo

The project now includes a tiny forward-only Orca FC demo in `src/bench_orca_fc_ringlpn_demo.cu`.

This checkpoint validates:

1. raw key-buffer serialization in `A`, `B`, `C_masked` order with no truncation bytes for `tf=None`,
2. bounded q62 constant-polynomial masks exported to `Z_{2^16}` with the carry-corrected bridge,
3. the unchanged `gpuMatmulBeaver` online contract for small FC layers,
4. deterministic seed replay and a distinct second seed,
5. agreement with Orca's `gpuKeygenMatmul` baseline under the same masks and deterministic P0/P1 random-share stream.

Current result summary:

- `results/orca_fc_ringlpn_demo_bounded_suite.md`
- `results/orca_fc_ringlpn_demo_memo.md`

Current smoke result:

| seed | second seed | shape | bw | value bound | Key bytes per party | Baseline bytes per party | Carry conversion | Replay | Second seed | Online contract | Baseline | Validation |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 2 | 2x2x2 | 16 | 255 | 96 | 96 | 1 | 1 | 1 | pass | pass | pass |
| 3 | 4 | 2x3x2 | 16 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass |
| 5 | 6 | 3x2x2 | 16 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass |
| 7 | 8 | 2x2x3 | 32 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass |

This is a correctness demo for forward FC keys only. It is not q128/CRT, dense packing, secure distributed conversion, backward/training key integration, or trusted-dealer removal.

### 11. Standalone DPF online key generation benchmark

The project now also includes a standalone DPF online key generation benchmark in `../tests/fss/dpf_online_keygen_bench.cu`.

This benchmark measures eval-all DPF key generation in two modes:

1. one-shot generation of the full pair key material,
2. chunked generation of only the current partial key material.

The current sweep is driven by `../scripts/run_dpf_online_keygen_sweep.py` and summarized in `results/dpf_online_keygen_bin16_chunk8192.md`.

Current benchmark contract:

1. eval-all keys,
2. `bin=16`,
3. `chunk_size=8192`,
4. `n = 8192 ... 1048576`,
5. validation of serialized key layout and parsed key metadata for both full and chunked modes.

Key current numbers:

- at `n=8192`, full pair-key footprint is `2.81 MiB`, partial peak pair-key footprint is `2.81 MiB`, and time overhead is about `1.011x`,
- at `n=16384`, full pair-key footprint is `5.63 MiB`, partial peak pair-key footprint is `2.81 MiB`, peak reduction is `2.00x`, and time overhead is about `1.380x`,
- at `n=1048576`, full pair-key footprint is `360.00 MiB`, partial peak pair-key footprint stays `2.81 MiB`, peak reduction reaches about `128.00x`, and time overhead is about `1.885x`,
- all sweep points passed validation.

This is a systems benchmark for online partial key generation, not yet an end-to-end application integration result.

## What We Have Until Now

### 1. CPU baseline status

The CPU sweep in `results/ntt_cpu.md` confirms:

1. requested `q=32` is only feasible up to `n=32768`,
2. requested `q=64` and `q=128` continue through `n=1048576`,
3. the CPU baseline remains the correctness and comparison anchor for the promoted GPU q=64 and q=128 paths.

### 2. Promoted main CUDA sweep status

The promoted main CUDA sweep now lives in `results/ntt_gpu_q32.md` and reflects the cheddar-derived implementation as the default q=32 GPU path.

Current promoted q=32 results:

| n | Batch | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s |
| --- | --- | --- | --- | --- |
| 8192 | 64 | 79.628 | 1.244 | 803741.62 |
| 16384 | 64 | 155.992 | 2.437 | 410276.17 |
| 32768 | 64 | 306.232 | 4.785 | 208996.91 |
| 65536 | 16 | 42.260 | 2.641 | 378612.86 |
| 131072 | 16 | 298.463 | 18.654 | 53598.24 |
| 262144 | 8 | 337.991 | 42.249 | 23669.65 |
| 524288 | 4 | 320.102 | 80.026 | 12496.02 |
| 1048576 | 2 | 332.762 | 166.381 | 6010.90 |

All points in the promoted sweep passed validation.

### 3. Promoted main CUDA q=64 sweep status

The promoted main CUDA q=64 sweep now lives in `results/ntt_gpu_q64.md`.

Current promoted q=64 results:

| n | Batch | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s |
| --- | --- | --- | --- | --- |
| 8192 | 64 | 253.654 | 3.963 | 252312.20 |
| 16384 | 64 | 311.081 | 4.861 | 205734.20 |
| 32768 | 64 | 465.788 | 7.278 | 137401.56 |
| 65536 | 16 | 140.357 | 8.772 | 113995.03 |
| 131072 | 16 | 460.117 | 28.757 | 34773.76 |
| 262144 | 8 | 456.098 | 57.012 | 17540.09 |
| 524288 | 4 | 462.271 | 115.568 | 8652.93 |
| 1048576 | 2 | 488.858 | 244.429 | 4091.17 |

All points in the promoted q=64 sweep also passed validation.

### 4. Promoted main CUDA q=128 CRT sweep status

The promoted main CUDA q=128 sweep now lives in `results/ntt_gpu_q128.md`.

Current promoted q=128 results:

| n | Batch | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s |
| --- | --- | --- | --- | --- |
| 8192 | 64 | 491.715 | 7.683 | 130156.70 |
| 16384 | 64 | 664.390 | 10.381 | 96328.96 |
| 32768 | 64 | 1076.000 | 16.812 | 59479.55 |
| 65536 | 16 | 278.679 | 17.417 | 57413.73 |
| 131072 | 16 | 1052.880 | 65.805 | 15196.41 |
| 262144 | 8 | 1058.820 | 132.352 | 7555.58 |
| 524288 | 4 | 1016.860 | 254.215 | 3933.68 |
| 1048576 | 2 | 1047.250 | 523.625 | 1909.76 |

All points in the promoted q=128 sweep passed validation over two q62 CRT residue limbs.

### 5. Archived legacy CUDA sweep status

The legacy CUDA baseline now lives in `results/ntt_gpu_q32_legacy.md`.

Current legacy q=32 per-polynomial results:

| n | Batch | Full PolyMul mean (us) | Per-poly PolyMul (us) |
| --- | --- | --- | --- |
| 8192 | 64 | 137.286 | 2.145 |
| 16384 | 64 | 223.449 | 3.491 |
| 32768 | 64 | 435.097 | 6.798 |
| 65536 | 16 | 250.858 | 15.679 |
| 131072 | 16 | 477.836 | 29.865 |
| 262144 | 8 | 495.323 | 61.915 |
| 524288 | 4 | 537.118 | 134.280 |
| 1048576 | 2 | 564.374 | 282.187 |

### 6. Main versus legacy comparison

The current promoted main path is faster than the legacy baseline across the entire validated sweep.

| n | Main per-poly PolyMul (us) | Legacy per-poly PolyMul (us) | Legacy/Main speedup |
| --- | --- | --- | --- |
| 8192 | 1.244 | 2.145 | 1.72x |
| 16384 | 2.437 | 3.491 | 1.43x |
| 32768 | 4.785 | 6.798 | 1.42x |
| 65536 | 2.641 | 15.679 | 5.94x |
| 131072 | 18.654 | 29.865 | 1.60x |
| 262144 | 42.249 | 61.915 | 1.47x |
| 524288 | 80.026 | 134.280 | 1.68x |
| 1048576 | 166.381 | 282.187 | 1.70x |

The strongest gain in the current adaptive sweep appears at `n=65536`, where the promoted main path is nearly `6x` faster per polynomial than the archived legacy implementation.

### 7. Earlier batch-1 evidence from the extraction study

The earlier study in `results/cheddar_extract_note.md` remains important because it showed that the extracted cheddar-derived path was not only winning because of aggressive batching. In apples-to-apples batch-1 comparisons, the extracted path was already consistently faster than the old implementation.

Selected batch-1 full polynomial multiplication speedups from that earlier study:

| n | Old GPU PolyMul (us) | Cheddar-derived PolyMul (us) | Old/Cheddar ratio |
| --- | --- | --- | --- |
| 8192 | 45.9478 | 16.6404 | 2.76x |
| 16384 | 52.4809 | 17.4753 | 3.00x |
| 32768 | 60.3129 | 19.8307 | 3.04x |
| 65536 | 72.6720 | 19.6413 | 3.70x |
| 131072 | 89.4712 | 34.6741 | 2.58x |
| 262144 | 124.9970 | 56.8568 | 2.20x |

This earlier result is the strongest evidence that the architectural advantage of the cheddar-derived path is real and not just an artifact of the later sweep schedule.

## What Has Been Implemented So Far, Precisely

For clarity, the implemented scope at the end of this phase is:

1. full CPU benchmark harness with validation,
2. archived legacy CUDA q=32 benchmark harness with validation,
3. generalized q=32 support over `n = 8192 ... 1048576`,
4. generalized q=64 single-prime support over `n = 8192 ... 1048576`,
5. generalized q=128 CRT residue-limb support over `n = 8192 ... 1048576`,
6. batch-aware benchmarking and reporting,
7. standalone cheddar-derived CUDA benchmark with two-phase NTT and inverse NTT,
8. local twiddle-table reconstruction and local host reference validation for the extracted path,
9. promotion of the cheddar-derived implementation to the main `bench_ntt_cuda` workflow,
10. archival of the older CUDA implementation as a named opt-in legacy baseline,
11. separate sweep artifacts for the promoted q=32 path, the promoted q=64 path, the promoted q=128 path, and the legacy baseline,
12. written extraction documentation and this status report,
13. standalone Ring-LPN VOLE prototype implementation with validated q=32 and q=64 sweep artifacts plus a passing q=128 CRT smoke path,
14. standalone GPU Figure 2 SPFSS/OLE artifact with validated q=62 uniform-noise smoke and bounded sweep artifacts,
15. standalone GPU SPFSS payload tests for single point, multiple points, alpha collisions, and edge alphas,
16. standalone ring-polynomial linear-layer OLE-to-Beaver artifact with a validated 2x2 by 2x2 smoke case,
17. standalone DPF online key generation benchmark with validated chunked-versus-one-shot sweep artifacts,
18. abstract support notes that connect Orca profiling, DPF online key generation, Figure 2 OLE, linear-layer OLE-to-Beaver, and Ring-LPN online-phase acceleration.

This is sufficient to say that the cheddar-derived CUDA path has been completed into the project as an operational benchmark path for requested `q=32`, requested `q=64`, and requested `q=128` at the NTT/PolyMul layer, and that the project now also has concrete standalone online-phase evidence for Ring-LPN VOLE expansion, Figure 2 SPFSS/OLE assembly, ring-polynomial OLE-to-Beaver linear layers, and chunked DPF online key generation.

## What Is Not Yet Implemented

The major missing pieces are no longer in the promoted q=32/q=64/q=128 NTT/PolyMul benchmark. They are in explicit CRT recomposition for downstream consumers and in integration with the online-phase artifacts.

### 1. CRT recomposition and downstream q128 consumers

The promoted benchmark validates q128 as two q62 CRT residue limbs. It does not yet expose a separate recomposed coefficient output path because the benchmark and Cheddar-style RNS arithmetic naturally operate limb-wise.

Remaining q128 follow-up work is:

1. add explicit CRT recomposition only if a downstream consumer needs canonical host coefficients,
2. compare promoted q128 timings against the CPU q128 sweep in a dedicated report,
3. wire q128 into Figure 2/OLE, linear-layer OLE-to-Beaver, and Orca key-writer artifacts before making paper-comparable or Orca-wide q128 claims.

### 2. CPU extraction symmetry

The CPU baseline still depends on NFLLib calls rather than a local extracted CPU implementation. That is acceptable for benchmarking, but it means the project currently compares a locally owned GPU implementation against an externally backed CPU implementation.

This is not an immediate blocker for the GPU q128 path, but it is a conceptual asymmetry worth noting.

### 3. Figure 2 paper-parameter and Orca conversion gaps

The GPU Figure 2 OLE artifact is intentionally staged:

1. regular sparse noise is implemented and benchmarked for the bounded `t=64` sweep, but only over the current single-prime modulus,
2. it has not yet been rerun over the promoted q128 CRT NTT/PolyMul backend,
3. its direct OLE benchmark stops at OLE,
4. the new linear-layer benchmark converts OLEs into Beaver products only for ring-polynomial matrix entries,
5. it only has a conservative one-scalar-per-polynomial packing smoke, not a high-density Orca tensor packing scheme,
6. it has a dealer/oracle `Z_p -> Z_{2^bw}` conversion smoke, not a secure distributed conversion protocol.

These gaps must be closed before claiming paper-comparable Figure 2 numbers or Orca trusted-dealer removal.

### 4. End-to-end online-phase integration

The newer VOLE, OLE, and DPF results are currently standalone benchmark artifacts, not end-to-end application integrations.

The missing integration work is:

1. wiring chunked DPF generation into a real Orca or SPFSS-backed online execution path,
2. replacing the `synthetic_mpvole` boundary in the VOLE prototype with a real external input boundary,
3. converting the ring-polynomial linear-layer artifact into Orca-compatible scalar Beaver triples,
4. measuring full application memory-footprint reduction rather than only standalone peak staged key-footprint reduction.

## Recommended Next Steps

### Immediate next step depends on the task

If the task is benchmark-core continuation, use the promoted q128 CRT path as the baseline.

Recommended sequence for that track:

1. add a CPU-vs-GPU q128 comparison report using the existing CPU q128 sweep,
2. profile and tune the flattened `(batch, prime)` q128 schedule,
3. add explicit CRT recomposition code only if a consumer needs canonical coefficients outside RNS form,
4. keep q32/q64 regression sweeps in the loop when tuning q128.

If the task is Figure 2/OLE continuation, first decide whether the goal is paper comparability or Orca integration.

Recommended sequence for paper-comparable Figure 2 numbers:

1. compare the grouped `2N/t` regular-noise output against the existing uniform baseline in the report narrative,
2. port the OLE artifact onto the promoted q128 CRT NTT/PolyMul backend,
3. rerun uniform and regular OLE sweeps under the CRT path,
4. update `results/ole_gpu_handoff.md` with which claims are now valid.

Recommended sequence for Orca integration:

1. keep the conservative constant-polynomial scalar bridge and tiny Orca-compatible triple writer as the regression baseline,
2. wire in the promoted q128 CRT backend or prove concrete layer-wise value bounds for q62,
3. replace or justify the dealer/oracle `Z_p -> Z_{2^bw}` conversion with a secure conversion protocol if trusted-dealer removal remains the claim,
4. connect the triple source to broader Orca linear-layer keygen without changing online `gpuMatmulBeaver`,
5. continue comparing against baseline Orca Beaver triples.

If the task is online-phase or abstract continuation outside Figure 2, the next step is end-to-end integration of the standalone DPF and VOLE prototypes.

Recommended sequence for that track:

1. wire chunked DPF generation into a real online execution boundary,
2. connect the current Ring-LPN VOLE prototype to a real external input boundary instead of `synthetic_mpvole`,
3. measure application-level peak memory, staging footprint, and runtime impact,
4. keep the abstract claims limited to whichever parts are actually integrated and measured.

### Supporting engineering work

To make the research workflow smoother, the following additions would also be useful:

1. an automated comparison script that renders promoted q=32, promoted q=64, promoted q=128, and legacy q=32 results side by side,
2. explicit summary titles that distinguish promoted q=32, promoted q=64, promoted q=128, and legacy outputs,
3. a CPU-vs-GPU q128 comparison page,
4. an integrated benchmark path that combines chunked DPF generation with the current online-phase acceleration prototype,
5. an OLE report appendix that shows smoke, bounded uniform-noise, bounded regular-noise, and future OLE-over-q128 results in one place.

## Final Assessment

The current phase should be described as follows:

The Ring-LPN project has completed the extraction of the cheddar-fhe NTT/INTT kernel architecture into a standalone local benchmark implementation, and that extracted implementation has now been promoted to the main Ring-LPN CUDA path for requested `q=32`, requested `q=64`, and requested `q=128`. The older CUDA benchmark remains archived as an opt-in legacy baseline. On top of that promoted core, the project now also has a standalone Ring-LPN VOLE prototype wired through the same Cheddar q128 CRT limb path, a standalone GPU Figure 2 SPFSS/OLE artifact, a standalone ring-polynomial OLE-to-Beaver linear-layer artifact, a host-only Orca scalar bridge smoke, a tiny bounded Orca FC key-writer demo with baseline comparison, and a standalone DPF online key generation benchmark with saved artifacts.

That distinction is important:

1. extraction into the project is complete for the current q32/q64/q128 benchmark-core target scope,
2. standalone online-phase benchmark evidence now exists for VOLE expansion, Figure 2 OLE assembly, ring-polynomial linear OLE-to-Beaver conversion, Orca scalar bridge arithmetic, tiny FC key writing, and chunked DPF key generation,
3. the next benchmark-core phase is q128 comparison, profiling, and optional explicit CRT recomposition for downstream consumers,
4. the next Figure 2 phase is porting the OLE artifacts onto q128 CRT if the goal is paper comparability,
5. the next Orca systems phase is q128 integration or concrete value-bound evidence, dense packing, and secure conversion before full integration into broader Orca linear-layer keygen.
