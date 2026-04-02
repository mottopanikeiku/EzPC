# Ring-LPN Status Report

Generated: 2026-04-02

## Executive Summary

This report summarizes the current implementation status of the Ring-LPN benchmarking track under `GPU-MPC/ringlpn`, with emphasis on the CUDA NTT work derived from cheddar-fhe.

The project now has three distinct benchmark layers:

1. a CPU baseline built on NFLLib,
2. a preserved legacy CUDA implementation built around a `phi` preprocessing plus fused-first-8-stage design,
3. a promoted primary CUDA implementation extracted from cheddar-fhe and adapted into a standalone Ring-LPN benchmark harness.

The main engineering result of this phase is that the cheddar-derived implementation is no longer only a side experiment. It has now been integrated into the main Ring-LPN CUDA pipeline as the default implementation behind `bench_ntt_cuda`, while the older CUDA path is preserved as a legacy baseline for comparison and regression tracking.

At the same time, the project remains intentionally staged. The current main GPU path is still a single-prime implementation for requested `q=32`, realized with one 30-bit prime. The remaining major research and engineering steps are:

1. 64-bit Montgomery kernels using `__umul64hi()` for requested `q=64`,
2. dual-prime CRT composition for requested `q=128`.

## Project Objective

The Ring-LPN subproject is a standalone benchmarking harness for NTT, inverse NTT, and full polynomial multiplication over the parameter ranges relevant to Ring-LPN work. It is separate from the Orca training and inference pipeline, even though both live under `GPU-MPC`.

The immediate objective of the CUDA work has been:

1. establish a valid CPU baseline,
2. build a generalized GPU q=32 path over the full degree range from `8192` through `1048576`,
3. extract the stronger NTT/INTT kernel structure from cheddar-fhe into a self-contained local benchmark,
4. promote that extracted implementation to the main Ring-LPN GPU path without importing cheddar-fhe's full runtime stack.

## Current Code and Filesystem State

The current high-signal files are:

| Path | Role |
| --- | --- |
| `src/bench_ntt.cpp` | NFLLib-backed CPU reference benchmark |
| `src/bench_ntt_cuda.cu` | Legacy CUDA benchmark retained for baseline comparison |
| `src/bench_ntt_cuda_cheddar.cu` | Primary CUDA benchmark source, extracted from cheddar-fhe and adapted locally |
| `scripts/build_bench.sh` | CPU build entry point |
| `scripts/build_cuda_bench.sh` | Main CUDA build entry point, now targeting the cheddar-derived implementation |
| `scripts/build_cuda_bench_cheddar.sh` | Explicit standalone cheddar-derived build |
| `scripts/build_cuda_bench_legacy.sh` | Legacy CUDA build |
| `scripts/run_sweep.sh` | CPU sweep driver |
| `scripts/run_cuda_sweep.sh` | Main CUDA sweep driver |
| `scripts/run_cuda_sweep_legacy.sh` | Legacy CUDA sweep driver |
| `scripts/run_cuda_single.sh` | CPU-vs-GPU spot check on CPU-overlap points |
| `results/ntt_cpu.md` | CPU baseline summary |
| `results/ntt_gpu_q32.md` | Current main CUDA summary |
| `results/ntt_gpu_q32_legacy.md` | Legacy CUDA summary |
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

This CPU baseline is important because it defines the correctness and reporting contract that the GPU side must eventually match for larger bitwidths.

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

This matters because the extraction is now operationally complete for the q=32 single-prime path. The code is no longer living only as a side file; it is the main GPU path used by the standard sweep script.

## What We Have Until Now

### 1. CPU baseline status

The CPU sweep in `results/ntt_cpu.md` confirms:

1. requested `q=32` is only feasible up to `n=32768`,
2. requested `q=64` and `q=128` continue through `n=1048576`,
3. the CPU baseline remains the correctness and comparison anchor for future GPU q=64 and q=128 work.

### 2. Promoted main CUDA sweep status

The promoted main CUDA sweep now lives in `results/ntt_gpu_q32.md` and reflects the cheddar-derived implementation as the default q=32 GPU path.

Current promoted q=32 results:

| n | Batch | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s |
| --- | --- | --- | --- | --- |
| 8192 | 64 | 79.351 | 1.240 | 806543.08 |
| 16384 | 64 | 155.850 | 2.435 | 410651.27 |
| 32768 | 64 | 292.693 | 4.573 | 218659.14 |
| 65536 | 16 | 42.196 | 2.637 | 379182.86 |
| 131072 | 16 | 298.689 | 18.668 | 53567.42 |
| 262144 | 8 | 320.880 | 40.110 | 24931.44 |
| 524288 | 4 | 320.509 | 80.127 | 12480.15 |
| 1048576 | 2 | 332.160 | 166.080 | 6021.19 |

All points in the promoted sweep passed validation.

### 3. Preserved legacy CUDA sweep status

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

### 4. Main versus legacy comparison

The current promoted main path is faster than the legacy baseline across the entire validated sweep.

| n | Main per-poly PolyMul (us) | Legacy per-poly PolyMul (us) | Legacy/Main speedup |
| --- | --- | --- | --- |
| 8192 | 1.240 | 2.145 | 1.73x |
| 16384 | 2.435 | 3.491 | 1.43x |
| 32768 | 4.573 | 6.798 | 1.49x |
| 65536 | 2.637 | 15.679 | 5.95x |
| 131072 | 18.668 | 29.865 | 1.60x |
| 262144 | 40.110 | 61.915 | 1.54x |
| 524288 | 80.127 | 134.280 | 1.68x |
| 1048576 | 166.080 | 282.187 | 1.70x |

The strongest gain in the current adaptive sweep appears at `n=65536`, where the promoted main path is nearly `6x` faster per polynomial than the preserved legacy implementation.

### 5. Earlier batch-1 evidence from the extraction study

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
4. batch-aware benchmarking and reporting,
5. standalone cheddar-derived CUDA benchmark with two-phase NTT and inverse NTT,
6. local twiddle-table reconstruction and local host reference validation for the extracted path,
7. promotion of the cheddar-derived implementation to the main `bench_ntt_cuda` workflow,
8. preservation of the older CUDA implementation as a named legacy baseline,
9. separate sweep artifacts for both the promoted main path and the legacy baseline,
10. written extraction documentation and this status report.

This is sufficient to say that the q=32 single-prime cheddar extraction has been completed into the project as an operational benchmark path.

## What Is Not Yet Implemented

The major missing pieces are not in the q=32 single-prime extraction anymore. They are in the next generalization phases.

### 1. 64-bit GPU path

The project does not yet provide a promoted GPU implementation for requested `q=64` corresponding to actual `q=62`. The immediate technical requirement is a templated 64-bit Montgomery implementation using `__umul64hi()` for the 128-bit intermediate product.

This phase will require:

1. 64-bit prime selection for the supported degree range,
2. 64-bit twiddle and inverse-twiddle table construction,
3. templated butterfly and pointwise multiply code for 64-bit words,
4. validation against the CPU `q=64` baseline.

### 2. q=128 via CRT

The project does not yet provide a GPU path for requested `q=128`.

The intended path is:

1. run two independent 64-bit NTT tracks over separate primes,
2. perform pointwise multiplication in each prime domain,
3. apply inverse NTT for each prime domain,
4. recombine results through CRT.

This requires moving from a single-prime batch model to a multi-prime runtime layout.

### 3. Multi-prime scheduling generalization

The promoted cheddar-derived path is currently specialized to the single-prime case. That is correct for the current q=32 work, but it is not yet the full multi-prime scheduling model needed for q=128.

### 4. CPU extraction symmetry

The CPU baseline still depends on NFLLib calls rather than a local extracted CPU implementation. That is acceptable for benchmarking, but it means the project currently compares a locally owned GPU implementation against an externally backed CPU implementation.

This is not an immediate blocker for the q=64 or q=128 GPU roadmap, but it is a conceptual asymmetry worth noting.

## Recommended Next Steps

### Immediate next step: Step 2

Implement templated 64-bit Montgomery kernels on top of the promoted cheddar-derived path.

Recommended sequence:

1. generalize the current single-prime kernel templates to a 64-bit word type,
2. implement 64-bit Montgomery multiplication using `__umul64hi()`,
3. add 64-bit table generation,
4. preserve the current CLI and CSV contract,
5. validate against the existing CPU `q=64` benchmark.

### After Step 2: Step 3

Extend the promoted main path to two-prime CRT for requested `q=128`.

Recommended sequence:

1. introduce prime-indexed scheduling in addition to batch scheduling,
2. instantiate two 64-bit NTT paths,
3. add CRT recomposition code,
4. extend the report scripts to display requested `q=128` GPU results cleanly,
5. validate against the CPU `q=128` benchmark.

### Supporting engineering work

To make the research workflow smoother, the following additions would also be useful:

1. an automated comparison script that renders main-versus-legacy speedups directly from the two sweep CSVs,
2. explicit summary titles that distinguish promoted main and legacy outputs,
3. a dedicated q=64 and q=128 sweep pipeline once those implementations land.

## Final Assessment

The current phase should be described as follows:

The Ring-LPN project has completed the extraction of the cheddar-fhe q=32 single-prime NTT/INTT kernel architecture into a standalone local benchmark implementation, and that extracted implementation has now been promoted to the main Ring-LPN CUDA path. The older CUDA benchmark remains preserved as a legacy baseline. The work is therefore complete for the q=32 single-prime extraction objective, but not complete for the broader roadmap of q=64 and q=128 GPU support.

That distinction is important:

1. extraction into the project is complete for the current target scope,
2. generalization to larger effective modulus sizes is the next research and engineering phase.