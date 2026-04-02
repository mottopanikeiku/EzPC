# Cheddar Extraction Note

Generated: 2026-03-31

## Goal

Extract the CUDA NTT and INTT kernel path from cheddar-fhe into the standalone Ring-LPN benchmark harness in `ringlpn`, without importing cheddar-fhe's full runtime stack.

Result:

- Extracted benchmark: `src/bench_ntt_cuda_cheddar.cu`
- Existing CUDA benchmark preserved: `src/bench_ntt_cuda.cu`
- Existing CPU benchmark preserved: `src/bench_ntt.cpp`

## Extraction Process

1. Identify the actual cheddar-fhe public boundary.

The public CUDA NTT entry point in cheddar-fhe is `NTTHandler`, not a single free kernel file. `NTTHandler` owns the twiddle vectors, inverse-degree constants, and Montgomery conversion constants, and it only advertises `log_degree` in `[12, 16]`.

References:

- `cheddar-fhe/include/core/NTT.h:20-47`
- `cheddar-fhe/include/core/NTT.h:55-67`

2. Identify the original CUDA kernels and the wrapper calls that feed them.

The original device-side kernels are `NTTPhase1`, `NTTPhase2`, `INTTPhase1`, and `INTTPhase2`. The host wrappers `NTTHandler::NTT` and `NTTHandler::INTT` show exactly which arrays each phase expects: `primes`, `inv_primes`, `twiddle_factors`, `twiddle_factors_msb`, `montgomery_converter_`, `inv_degree_`, and `inv_degree_mont_`.

References:

- `cheddar-fhe/src/core/NTT.cu:23`
- `cheddar-fhe/src/core/NTT.cu:128`
- `cheddar-fhe/src/core/NTT.cu:231`
- `cheddar-fhe/src/core/NTT.cu:344`
- `cheddar-fhe/src/core/NTT.cu:607-740`

3. Trace the launch configuration and table layout requirements.

The wrapper code delegates launch geometry to `GetLsbSize`, `GetMsbSize`, `GetLogWarpBatching`, `GetStageMerging`, and `GetBlockDim`. The twiddle population path constructs bit-reversed `psi` and `psi^{-1}` tables, converts them into Montgomery form, slices out the MSB twiddle table used for OF-twiddle, and stores `N^{-1}`, `N^{-1}` in Montgomery form, and the Montgomery conversion constant.

References:

- `cheddar-fhe/src/core/NTT.cu:1131-1222`
- `cheddar-fhe/src/core/NTT.cu:1234-1298`

4. Decide what not to import.

The extracted path deliberately does not pull over cheddar-fhe's `DeviceVector`, `DvView`, `InputPtrList`, `Parameter`, `NPInfo`, `PopulateConstantMemory`, or the multi-prime modulus scheduling. Those pieces are framework/runtime glue, not the core kernel logic needed for Ring-LPN's single-prime benchmark.

The adapter boundary is visible in the difference between cheddar's wrappers and the standalone Ring-LPN wrappers.

References:

- `cheddar-fhe/src/core/NTT.cu:607-740`
- `src/bench_ntt_cuda_cheddar.cu:1409-1567`

5. Rebuild the minimum table set in Ring-LPN.

The standalone function `compute_cheddar_tables` mirrors cheddar-fhe's twiddle construction for a single 30-bit prime. It builds bit-reversed `psi` and `psi^{-1}`, converts them to Montgomery form, derives the MSB twiddle slices, and stores the same auxiliary constants expected by the two-phase kernels.

References:

- Original layout source: `cheddar-fhe/src/core/NTT.cu:1234-1298`
- Standalone adapter: `src/bench_ntt_cuda_cheddar.cu:295-332`

6. Keep the two-phase kernel structure, but specialize the runtime to one prime and many batches.

The extracted file keeps the two-phase NTT and INTT structure and the same launch-config template logic, but converts cheddar-fhe's multi-prime `grid.y` indexing into a simpler `batch_count` dimension. That is why the standalone kernels are named `NTTPhase1SinglePrime`, `NTTPhase2SinglePrime`, `INTTPhase1SinglePrime`, and `INTTPhase2SinglePrime`.

References:

- Original wrappers: `cheddar-fhe/src/core/NTT.cu:635-670`
- Original wrappers: `cheddar-fhe/src/core/NTT.cu:701-740`
- Extracted kernels: `src/bench_ntt_cuda_cheddar.cu:968`
- Extracted kernels: `src/bench_ntt_cuda_cheddar.cu:1063`
- Extracted kernels: `src/bench_ntt_cuda_cheddar.cu:1186`
- Extracted kernels: `src/bench_ntt_cuda_cheddar.cu:1298`
- Extracted wrappers: `src/bench_ntt_cuda_cheddar.cu:1409-1567`

7. Add standalone validation so the extracted path is not trusted blindly.

The extracted harness validates roundtrip and polynomial multiplication over zero, one, impulse, max, and random patterns against a host reference before reporting benchmark output.

References:

- `src/bench_ntt_cuda_cheddar.cu:1644-1781`

8. Measure the extracted path using the same timing style as the existing GPU harness.

Both CUDA harnesses use `cudaEventRecord` around forward NTT, inverse NTT, and full polynomial multiplication. That makes their raw CSV outputs directly comparable.

References:

- Existing CUDA timing: `src/bench_ntt_cuda.cu:969-1023`
- Extracted CUDA timing: `src/bench_ntt_cuda_cheddar.cu:1939-2009`

## What Was Extracted vs Adapted

Directly preserved from cheddar-fhe:

- Two-phase NTT / INTT decomposition.
- Launch-config driven block sizing and stage merging.
- OF-twiddle use through the MSB twiddle table.
- Montgomery butterfly structure and kernel ordering.

Adapted for Ring-LPN:

- Single-prime runtime instead of cheddar-fhe's multi-prime `NPInfo` path.
- Plain CUDA allocations and raw pointers instead of cheddar-fhe device containers.
- A standalone pointwise multiply kernel.
- A host reference path for validation.
- Degree support widened in the standalone harness to `log_degree` in `[13, 20]`, while the public cheddar handler only documents `[12, 16]`.

References:

- Cheddar public degree bounds: `cheddar-fhe/include/core/NTT.h:45-47`
- Standalone degree bounds: `src/bench_ntt_cuda_cheddar.cu:20-21`
- Standalone degree check: `src/bench_ntt_cuda_cheddar.cu:122-124`

## Benchmark Methodology Comparison

### CPU benchmark: `bench_ntt.cpp`

This is a local harness built on NFLLib APIs, not an extracted copy of NFLLib's implementation. It resolves the requested modulus size, validates sparse roundtrip and negacyclic multiplication, and then times single-poly `ntt_pow_phi`, `invntt_pow_invphi`, and full `NTT(a) + NTT(b) + pointwise + INTT`.

References:

- NFLLib-backed configuration resolution: `src/bench_ntt.cpp:120-152`
- Sparse validation and roundtrip/product checks: `src/bench_ntt.cpp:295-331`
- Timed benchmark loops: `src/bench_ntt.cpp:335-412`
- NFLLib dispatch type: `src/bench_ntt.cpp:417`
- Requested `qbits=32` maps to actual `qbits=30`: `src/bench_ntt.cpp:131-135`

### Existing GPU benchmark: `bench_ntt_cuda.cu`

This path is not cheddar-derived. It builds `phi` preprocessing vectors, stage-offset twiddle tables, and then runs a different kernel structure: preprocess by `phi`, fuse the first eight stages in shared memory, run stage-tail kernels, bit-reverse before inverse, and postprocess by `inv_n * invphi^i`.

References:

- Table construction with `phi`, `post_scale`, and `stage_offsets`: `src/bench_ntt_cuda.cu:215-273`
- Kernel symbols: `src/bench_ntt_cuda.cu:434`
- Kernel symbols: `src/bench_ntt_cuda.cu:455`
- Kernel symbols: `src/bench_ntt_cuda.cu:471`
- Kernel symbols: `src/bench_ntt_cuda.cu:498`
- Kernel symbols: `src/bench_ntt_cuda.cu:557`
- Execution path: `src/bench_ntt_cuda.cu:578-706`
- Validation: `src/bench_ntt_cuda.cu:775-857`
- Timing and CSV output: `src/bench_ntt_cuda.cu:887-1023`

### Extracted GPU benchmark: `bench_ntt_cuda_cheddar.cu`

This path is structurally aligned with cheddar-fhe. It builds bit-reversed `psi` and `psi^{-1}` tables plus MSB OF-twiddle slices, launches two NTT phases, performs pointwise multiply, and launches two INTT phases.

References:

- Table construction: `src/bench_ntt_cuda_cheddar.cu:295-332`
- Execution path: `src/bench_ntt_cuda_cheddar.cu:1409-1567`
- Validation: `src/bench_ntt_cuda_cheddar.cu:1644-1781`
- Timing and CSV output: `src/bench_ntt_cuda_cheddar.cu:1869-2009`

### How CPU vs GPU speedup is currently computed

The existing comparison script `run_cuda_single.sh` only compares the CPU benchmark against the old CUDA benchmark. The summarizer divides GPU batch latency by `batch_size` before reporting speedup, so its reported speedup is per polynomial, not per batch.

References:

- Comparison driver: `scripts/run_cuda_single.sh:6-10`
- Comparison driver: `scripts/run_cuda_single.sh:38-44`
- Per-poly conversion: `scripts/summarize_cpu_gpu_4096.py:37-44`
- Speedup formula: `scripts/summarize_cpu_gpu_4096.py:63-64`

## Measurement Environment

These measurements were run inside the EzPC container flow, using `./start` and then executing the existing binaries inside `/home/ringlpn` in the `orca-dev` container.

## Measured Results

### CPU vs GPU latency, batch = 1, requested qbits = 32

This is the cleanest apples-to-apples comparison because the CPU benchmark is single-poly and the GPU batch size is also 1.

| n | CPU NTT us | Old GPU NTT us | Cheddar GPU NTT us | Old GPU NTT speedup | Cheddar GPU NTT speedup | CPU PolyMul us | Old GPU PolyMul us | Cheddar GPU PolyMul us | Old GPU PolyMul speedup | Cheddar GPU PolyMul speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 51.8990 | 15.1278 | 6.0464 | 3.43x | 8.58x | 163.6860 | 45.9478 | 16.6404 | 3.56x | 9.84x |
| 16384 | 110.6340 | 17.1827 | 6.1783 | 6.44x | 17.91x | 353.6880 | 52.4809 | 17.4753 | 6.74x | 20.24x |
| 32768 | 229.9780 | 19.7996 | 7.0282 | 11.62x | 32.72x | 763.8530 | 60.3129 | 19.8307 | 12.66x | 38.52x |

### Old GPU vs cheddar-extracted GPU, batch = 1, requested qbits = 32

Values larger than 1 in the ratio columns mean the cheddar-extracted benchmark is faster.

| n | Old GPU NTT us | Cheddar GPU NTT us | Old/Cheddar NTT ratio | Old GPU INTT us | Cheddar GPU INTT us | Old/Cheddar INTT ratio | Old GPU PolyMul us | Cheddar GPU PolyMul us | Old/Cheddar PolyMul ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 15.1278 | 6.0464 | 2.50x | 16.9256 | 5.8709 | 2.88x | 45.9478 | 16.6404 | 2.76x |
| 16384 | 17.1827 | 6.1783 | 2.78x | 18.9916 | 6.1906 | 3.07x | 52.4809 | 17.4753 | 3.00x |
| 32768 | 19.7996 | 7.0282 | 2.82x | 21.5551 | 6.9835 | 3.09x | 60.3129 | 19.8307 | 3.04x |
| 65536 | 24.1496 | 7.0133 | 3.44x | 25.7803 | 6.5501 | 3.94x | 72.6720 | 19.6413 | 3.70x |
| 131072 | 29.7664 | 11.9675 | 2.49x | 31.2869 | 11.7771 | 2.66x | 89.4712 | 34.6741 | 2.58x |
| 262144 | 40.9084 | 19.5056 | 2.10x | 43.4432 | 18.6392 | 2.33x | 124.9970 | 56.8568 | 2.20x |

## Interpretation

1. The existing GPU benchmark and the cheddar-extracted GPU benchmark do not share the same internal design.

The old CUDA path is a `phi` preprocess plus fused-first-8-stage design with stage-offset twiddle tables. The cheddar path is a two-phase OF-twiddle design with bit-reversed `psi` tables and MSB twiddle slices.

2. The cheddar-extracted path is consistently faster in the measured batch-1 runs.

Across the tested points, the extracted path is about 2.1x to 3.7x faster than the old CUDA benchmark for full polynomial multiplication.

3. CPU vs GPU speedup depends on whether you mean latency or throughput.

If you want latency, compare CPU against `batch=1` GPU runs. If you want throughput, use batched GPU runs and divide the GPU batch latency by `batch_size`, which is what the current summarizer already does.

4. `bench_ntt.cpp` is NFLLib-backed, not an extracted CPU implementation.

That matters if you want symmetry with the cheddar extraction. Right now the CPU baseline is a benchmark harness around NFLLib calls, while the cheddar GPU path is now a standalone extracted implementation.

## Practical Next Step

If you want the CPU side to mirror the GPU extraction strategy, the next step is not to change the benchmark formulas. The next step is to replace the current NFLLib-linked CPU harness with a minimal local extraction of the NFLLib CPU NTT path, so both CPU and GPU benchmarks are measured from locally owned implementations rather than one local extraction and one external library call path.