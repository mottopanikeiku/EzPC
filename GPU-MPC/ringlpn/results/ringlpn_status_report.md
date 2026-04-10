# Ring-LPN Status Report

Generated: 2026-04-09

## Executive Summary

This report summarizes the current implementation status of the Ring-LPN benchmarking track under `GPU-MPC/ringlpn`, with emphasis on the promoted CUDA NTT work derived from cheddar-fhe and the newer standalone online-phase benchmarks built around it.

The project now has five distinct benchmark layers:

1. a CPU baseline built on NFLLib,
2. a preserved legacy CUDA implementation built around a `phi` preprocessing plus fused-first-8-stage design,
3. a promoted primary CUDA implementation extracted from cheddar-fhe and adapted into a standalone Ring-LPN benchmark harness,
4. a standalone Ring-LPN VOLE prototype benchmark built on the promoted CUDA PolyMul path,
5. a standalone DPF online key generation benchmark that measures one-shot versus chunked partial generation.

The main engineering result of this phase is that the cheddar-derived implementation is no longer only a side experiment. It has now been integrated into the main Ring-LPN CUDA pipeline as the default implementation behind `bench_ntt_cuda`, while the older CUDA path is preserved as a legacy baseline for comparison and regression tracking.

At the same time, the project remains intentionally staged. The current main GPU path is now a single-prime implementation for both requested `q=32` and requested `q=64`, realized with one 30-bit prime or one 62-bit prime depending on the requested configuration. The remaining major research and engineering step on the benchmark-core side is dual-prime CRT composition for requested `q=128`. On the online-phase side, the remaining step is end-to-end integration of the standalone DPF and VOLE prototypes into a real Orca or SPFSS-backed path.

## Project Objective

The Ring-LPN subproject is a standalone benchmarking harness for NTT, inverse NTT, and full polynomial multiplication over the parameter ranges relevant to Ring-LPN work. It is separate from the Orca training and inference pipeline, even though both live under `GPU-MPC`.

The immediate objective of the CUDA work has been:

1. establish a valid CPU baseline,
2. build a generalized GPU q=32 path over the full degree range from `8192` through `1048576`,
3. extract the stronger NTT/INTT kernel structure from cheddar-fhe into a self-contained local benchmark,
4. promote that extracted implementation to the main Ring-LPN GPU path without importing cheddar-fhe's full runtime stack,
5. extend that promoted single-prime path from requested `q=32` to requested `q=64`,
6. prototype a standalone Ring-LPN VOLE-style expansion layer on top of the promoted GPU PolyMul path,
7. prototype a standalone DPF online key generation benchmark that quantifies peak staged key-footprint reduction from chunked generation.

## Current Code and Filesystem State

The current high-signal files are:

| Path | Role |
| --- | --- |
| `src/bench_ntt.cpp` | NFLLib-backed CPU reference benchmark |
| `src/bench_ntt_cuda.cu` | Legacy CUDA benchmark retained for baseline comparison |
| `src/bench_ntt_cuda_cheddar.cu` | Primary CUDA benchmark source, extracted from cheddar-fhe and adapted locally |
| `src/bench_vole_ringlpn.cu` | Standalone Ring-LPN VOLE prototype benchmark |
| `../tests/fss/dpf_online_keygen_bench.cu` | Standalone DPF online key generation benchmark |
| `scripts/build_bench.sh` | CPU build entry point |
| `scripts/build_cuda_bench.sh` | Main CUDA build entry point, now targeting the cheddar-derived implementation |
| `scripts/build_vole_bench.sh` | VOLE benchmark build entry point |
| `scripts/build_cuda_bench_cheddar.sh` | Explicit standalone cheddar-derived build |
| `scripts/build_cuda_bench_legacy.sh` | Legacy CUDA build |
| `scripts/run_sweep.sh` | CPU sweep driver |
| `scripts/run_cuda_sweep.sh` | Main CUDA sweep driver |
| `scripts/run_vole_sweep.sh` | VOLE sweep driver |
| `scripts/run_cuda_sweep_legacy.sh` | Legacy CUDA sweep driver |
| `scripts/run_cuda_single.sh` | CPU-vs-GPU spot check on CPU-overlap points |
| `../scripts/run_dpf_online_keygen_sweep.py` | DPF online key generation sweep driver |
| `results/ntt_cpu.md` | CPU baseline summary |
| `results/ntt_gpu_q32.md` | Current main CUDA summary |
| `results/ntt_gpu_q64.md` | Current main CUDA q=64 summary |
| `results/ntt_gpu_q32_legacy.md` | Legacy CUDA summary |
| `results/vole_gpu_q32_m32_c2_w64.md` | Current standalone VOLE q=32 summary |
| `results/vole_gpu_q64_m32_c2_w64.md` | Current standalone VOLE q=64 summary |
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

This CPU baseline is important because it defines the correctness and reporting contract that the GPU side must match for larger bitwidths. That contract is now met for the current single-prime GPU q=64 path, and it remains the comparison anchor for future q=128 work.

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
2. specialization to a single-prime benchmark layout,
3. use of the batch dimension for independent polynomials rather than cheddar-fhe's multi-prime scheduling dimension,
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
| `bin/bench_ntt_cuda_legacy` | `src/bench_ntt_cuda.cu` | Preserved baseline for comparison |

This matters because the extraction is now operationally complete for the promoted single-prime GPU path. The code is no longer living only as a side file; it is the main GPU path used by the standard sweep script.

### 5. q=64 extension on the promoted main path

The promoted cheddar-derived path now also supports requested `q=64`, realized with one 62-bit prime over the full `n = 8192 ... 1048576` range.

The implementation work in this phase added:

1. a 64-bit Montgomery specialization using `__umul64hi()` for the 128-bit intermediate product,
2. runtime selection between a 30-bit and 62-bit single-prime configuration,
3. 64-bit twiddle, inverse-twiddle, inverse-degree, and Montgomery-conversion table generation,
4. 64-bit host reference validation for roundtrip NTT/INTT and negacyclic polynomial multiplication,
5. q=64 sweep tooling and result generation under the same promoted `bench_ntt_cuda` binary.

### 6. Standalone Ring-LPN VOLE prototype

The project now also includes a standalone Ring-LPN VOLE prototype in `src/bench_vole_ringlpn.cu`.

This implementation is intentionally scoped as a correctness-first online-phase prototype rather than a full end-to-end SPFSS-backed system.

What it does today:

1. reuses the promoted CUDA polynomial multiplication backend from `src/bench_ntt_cuda_cheddar.cu`,
2. synthesizes MPVOLE-consistent inputs locally under the `synthetic_mpvole` mode,
3. validates the coefficient-wise relation `z = y + x * Delta`,
4. supports requested `q=32` and `q=64`,
5. supports `n = 8192 ... 1048576`.

Current result summaries:

- `results/vole_gpu_q32_m32_c2_w64.md`
- `results/vole_gpu_q64_m32_c2_w64.md`

Key current numbers:

- q=32 full expansion latency ranges from `269.484 us` at `n=8192` to `43.392 ms` at `n=1048576`,
- q=64 full expansion latency ranges from `772.324 us` at `n=8192` to `67.532 ms` at `n=1048576`,
- all sweep points passed validation.

### 7. Standalone DPF online key generation benchmark

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
3. the CPU baseline remains the correctness and comparison anchor for the new GPU q=64 path and the future GPU q=128 path.

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

### 4. Preserved legacy CUDA sweep status

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

### 5. Main versus legacy comparison

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

The strongest gain in the current adaptive sweep appears at `n=65536`, where the promoted main path is nearly `6x` faster per polynomial than the preserved legacy implementation.

### 6. Earlier batch-1 evidence from the extraction study

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
2. full legacy CUDA q=32 benchmark harness with validation,
3. generalized q=32 support over `n = 8192 ... 1048576`,
4. generalized q=64 single-prime support over `n = 8192 ... 1048576`,
5. batch-aware benchmarking and reporting,
6. standalone cheddar-derived CUDA benchmark with two-phase NTT and inverse NTT,
7. local twiddle-table reconstruction and local host reference validation for the extracted path,
8. promotion of the cheddar-derived implementation to the main `bench_ntt_cuda` workflow,
9. preservation of the older CUDA implementation as a named legacy baseline,
10. separate sweep artifacts for the promoted q=32 path, the promoted q=64 path, and the legacy baseline,
11. written extraction documentation and this status report,
12. standalone Ring-LPN VOLE prototype implementation with validated q=32 and q=64 sweep artifacts,
13. standalone DPF online key generation benchmark with validated chunked-versus-one-shot sweep artifacts,
14. abstract support notes that connect Orca profiling, DPF online key generation, and Ring-LPN online-phase acceleration.

This is sufficient to say that the single-prime cheddar-derived CUDA path has been completed into the project as an operational benchmark path for both requested `q=32` and requested `q=64`, and that the project now also has concrete standalone online-phase evidence for both Ring-LPN VOLE expansion and chunked DPF online key generation.

## What Is Not Yet Implemented

The major missing pieces are no longer in the promoted single-prime q=32/q=64 path. They are in the next generalization phase.

### 1. q=128 via CRT

The project does not yet provide a GPU path for requested `q=128`.

The intended path is:

1. run two independent 64-bit NTT tracks over separate primes,
2. perform pointwise multiplication in each prime domain,
3. apply inverse NTT for each prime domain,
4. recombine results through CRT.

This requires moving from a single-prime batch model to a multi-prime runtime layout.

### 2. Multi-prime scheduling generalization

The promoted cheddar-derived path is currently specialized to the single-prime case. That is correct for the current q=32/q=64 work, but it is not yet the full multi-prime scheduling model needed for q=128.

### 3. CPU extraction symmetry

The CPU baseline still depends on NFLLib calls rather than a local extracted CPU implementation. That is acceptable for benchmarking, but it means the project currently compares a locally owned GPU implementation against an externally backed CPU implementation.

This is not an immediate blocker for the q=128 GPU roadmap, but it is a conceptual asymmetry worth noting.

### 4. End-to-end online-phase integration

The newer VOLE and DPF results are currently standalone benchmark artifacts, not end-to-end application integrations.

The missing integration work is:

1. wiring chunked DPF generation into a real Orca or SPFSS-backed online execution path,
2. replacing the `synthetic_mpvole` boundary in the VOLE prototype with a real external input boundary,
3. measuring full application memory-footprint reduction rather than only standalone peak staged key-footprint reduction.

## Recommended Next Steps

### Immediate next step depends on the task

If the task is benchmark-core continuation, extend the promoted main path to two-prime CRT for requested `q=128`.

Recommended sequence for that track:

1. introduce prime-indexed scheduling in addition to batch scheduling,
2. instantiate two 64-bit NTT paths,
3. add CRT recomposition code,
4. extend the report scripts to display requested `q=128` GPU results cleanly,
5. validate against the CPU `q=128` benchmark.

If the task is online-phase or abstract continuation, the next step is end-to-end integration of the standalone DPF and VOLE prototypes.

Recommended sequence for that track:

1. wire chunked DPF generation into a real online execution boundary,
2. connect the current Ring-LPN VOLE prototype to a real external input boundary instead of `synthetic_mpvole`,
3. measure application-level peak memory, staging footprint, and runtime impact,
4. keep the abstract claims limited to whichever parts are actually integrated and measured.

### Supporting engineering work

To make the research workflow smoother, the following additions would also be useful:

1. an automated comparison script that renders promoted q=32, promoted q=64, and legacy q=32 results side by side,
2. explicit summary titles that distinguish promoted q=32, promoted q=64, and legacy outputs,
3. a dedicated q=128 sweep pipeline once that implementation lands,
4. an integrated benchmark path that combines chunked DPF generation with the current online-phase acceleration prototype.

## Final Assessment

The current phase should be described as follows:

The Ring-LPN project has completed the extraction of the cheddar-fhe single-prime NTT/INTT kernel architecture into a standalone local benchmark implementation, and that extracted implementation has now been promoted to the main Ring-LPN CUDA path for both requested `q=32` and requested `q=64`. The older CUDA benchmark remains preserved as a legacy baseline. On top of that promoted core, the project now also has a standalone Ring-LPN VOLE prototype and a standalone DPF online key generation benchmark with saved artifacts.

That distinction is important:

1. extraction into the project is complete for the current single-prime target scope,
2. standalone online-phase benchmark evidence now exists for both VOLE expansion and chunked DPF key generation,
3. the next benchmark-core research and engineering phase is dual-prime CRT support for requested `q=128`,
4. the next online-phase systems phase is full integration of those standalone prototypes into a real FSS application path.