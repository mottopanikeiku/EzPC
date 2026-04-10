# Ring-LPN VOLE Abstract Support Note

Generated: 2026-04-08

## Scope

This note summarizes the current state of the standalone Ring-LPN VOLE prototype under `GPU-MPC/ringlpn` and records the benchmark claims that are safe to use in an abstract.

The current prototype is not a full SPFSS-backed OLE-R-LPN implementation yet. It isolates the algebraic expansion layer using synthetic MPVOLE-consistent inputs and validates the coefficient-wise relation

`z = y + x * Delta`

for all benchmarked points.

## What Was Implemented

The new prototype benchmark is:

- `src/bench_vole_ringlpn.cu`

It reuses the promoted cheddar-derived single-prime GPU polynomial multiplication path from:

- `src/bench_ntt_cuda_cheddar.cu`

instead of introducing a separate CUDA implementation.

The prototype currently supports:

- requested `q=32`, realized as actual `q=30`,
- requested `q=64`, realized as actual `q=62`,
- `n` in `{8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}`,
- fixed sweep configuration `m=32`, `c=2`, and noise weight `64` in the current abstract-ready results.

## Measurement Setup

All VOLE results below were generated inside the `orca-dev` container under `/home/ringlpn` using:

- `bash scripts/build_vole_bench.sh`
- `bash scripts/run_vole_sweep.sh`
- `QBITS=64 bash scripts/run_vole_sweep.sh`

Output artifacts:

- `results/vole_gpu_q32_m32_c2_w64.csv`
- `results/vole_gpu_q32_m32_c2_w64.md`
- `results/vole_gpu_q64_m32_c2_w64.csv`
- `results/vole_gpu_q64_m32_c2_w64.md`

## Current VOLE Prototype Results

### q=32

For requested `q=32` with `m=32`, `c=2`, and noise weight `64`:

- at `n=8192`, full expansion latency is `269.484 us`, per-output expansion is `8.421 us`, and throughput is `118745.45` outputs/s,
- at `n=32768`, full expansion latency is `963.231 us`, per-output expansion is `30.101 us`, and throughput is `33221.52` outputs/s,
- at `n=1048576`, full expansion latency is `43392.000 us`, per-output expansion is `1356.000 us`, and throughput is `737.46` outputs/s.

Every sweep point passed validation.

## Standalone DPF Online Key Generation Results

To support the broader memory-efficiency storyline, the codebase now also has a standalone DPF online key generation benchmark:

- `tests/fss/dpf_online_keygen_bench.cu`

It is built with:

- `make dpf_online_keygen`

and the current abstract-ready sweep is driven by:

- `scripts/run_dpf_online_keygen_sweep.py`

Current output artifacts:

- `results/dpf_online_keygen_bin16_chunk8192.csv`
- `results/dpf_online_keygen_bin16_chunk8192.md`

The current sweep measures eval-all DPF key generation at `bin=16` with chunk size `8192` over:

- `n` in `{8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}`.

Every sweep point passed validation.

The most important current numbers are:

- at `n=8192`, the full pair key is `2.81 MiB`, the partial online peak pair key is also `2.81 MiB`, and time overhead is `1.011x`,
- at `n=16384`, the full pair key is `5.63 MiB`, the partial online peak pair key is `2.81 MiB`, peak reduction is `2.00x`, and time overhead is `1.380x`,
- at `n=1048576`, the full pair key is `360.00 MiB`, the partial online peak pair key remains `2.81 MiB`, peak reduction reaches `128.00x`, and time overhead is `1.885x`.

The total generated key material remains effectively unchanged across the sweep, so this result supports a peak-memory and staging-footprint reduction claim rather than a total-key-volume reduction claim.

### q=64

For requested `q=64` with the same `m=32`, `c=2`, and noise weight `64`:

- at `n=8192`, full expansion latency is `772.324 us`, per-output expansion is `24.135 us`, and throughput is `41433.39` outputs/s,
- at `n=32768`, full expansion latency is `1537.150 us`, per-output expansion is `48.036 us`, and throughput is `20817.75` outputs/s,
- at `n=1048576`, full expansion latency is `67531.700 us`, per-output expansion is `2110.366 us`, and throughput is `473.85` outputs/s.

Every sweep point passed validation.

## How This Connects To Existing Ring-LPN Evidence

The VOLE prototype sits on top of the already-validated promoted CUDA polynomial engine. The strongest existing comparison claims for that underlying engine are:

- requested `q=32` overlap points show about `146x` to `171x` per-polynomial PolyMul speedups over the CPU baseline,
- requested `q=64` overlap points show about `48x` to `220x` per-polynomial PolyMul speedups over the CPU baseline,
- the promoted cheddar-derived path is consistently faster than the preserved legacy CUDA baseline.

Those claims are supported by:

- `results/ringlpn_status_report.md`
- `results/ntt_cpu.md`
- `results/ntt_gpu_q32.md`
- `results/ntt_gpu_q64.md`

## Safe Abstract Claims

The following points are supported by the current code and generated artifacts:

1. We implemented a standalone GPU prototype of the Ring-LPN VOLE-style expansion layer under the existing Ring-LPN benchmark harness.
2. The prototype reuses a validated promoted single-prime GPU polynomial backend rather than a separate ad hoc kernel path.
3. The current prototype validates the intended algebraic VOLE relation coefficient-wise across the full tested degree range from `8192` to `1048576` for both requested `q=32` and requested `q=64`.
4. In the current abstract-ready sweep configuration (`m=32`, `c=2`, noise weight `64`), q=32 full expansion latency ranges from `269.484 us` to `43.392 ms`, and q=64 full expansion latency ranges from `772.324 us` to `67.532 ms`.
5. The underlying GPU polynomial engine already has strong CPU-vs-GPU evidence, so the VOLE prototype is built on top of a well-supported accelerated core.
6. We implemented a standalone DPF online key generation benchmark for eval-all keys and measured that chunked generation can cap peak pair-key footprint at `2.81 MiB` across the current sweep while scaling one-shot full pair-key footprint from `2.81 MiB` to `360.00 MiB`.
7. In that same DPF sweep, peak-footprint reduction grows from `1.00x` at `n=8192` to `128.00x` at `n=1048576`, with current key-generation time overhead rising to about `1.885x` at the largest point.

## Claims To Avoid

The following statements are not supported yet and should not appear in an abstract without additional implementation work:

1. Claiming that the full SPFSS-backed OLE-R-LPN or degree-1 correlation pipeline from the figure is complete end-to-end.
2. Claiming a CPU-vs-GPU speedup number for the current VOLE prototype itself.
3. Claiming end-to-end dealer/evaluator integration or a fully integrated online key generation pipeline for the Ring-LPN path.
4. Claiming q=`128` support for the VOLE prototype.

## Recommended Abstract Positioning

The safest framing is:

- present the current work as a validated GPU prototype of the Ring-LPN VOLE expansion layer,
- use the new VOLE sweep results for concrete protocol-layer latency numbers,
- use the existing Ring-LPN NTT and PolyMul benchmark results for the CPU-vs-GPU acceleration story,
- explicitly describe full SPFSS-backed integration as ongoing or future work.

## Professor-Aligned Storyline

The professor's recommended abstract direction is broader than the current Ring-LPN-only note. The suggested storyline is:

1. introduction of the GPU FSS library,
2. profiling,
3. online key generation based on DPF,
4. accelerating the online phase of FSS with Ring-LPN,
5. possibly direct I/O as an optimization,
6. with emphasis on reducing memory and storage footprint by generating partial keys for partial computation.

That broader storyline is compatible with the current evidence, but only if the abstract separates implemented results from proposed techniques.

### What Is Already Supported Today

- The codebase already provides a concrete GPU FSS and secure-computation library context through `GPU-MPC`, including Orca and related GPU backends.
- Profiling and instrumentation evidence already exist for memory and key-I/O pressure in the Orca path.
- Direct I/O support is already present in `utils/gpu_file_utils.cpp` via `O_DIRECT | O_LARGEFILE`, with 4096-byte-aligned buffers.
- A standalone DPF online key generation benchmark now exists with saved artifacts showing large peak-footprint reduction from chunked partial generation.
- The new Ring-LPN VOLE prototype provides concrete GPU benchmark data for the algebraic expansion layer.

### What Must Still Be Framed As Proposed Work

- an end-to-end partial-key pipeline wired into the existing Orca or SPFSS-backed online execution path,
- end-to-end memory-footprint reduction numbers for the full application pipeline,
- full SPFSS-backed Ring-LPN integration into the online FSS path.

### Profiling Evidence That Fits The Story

The current profiling evidence supports the claim that memory movement and key I/O are central bottlenecks:

- `P-SecureML` average key read time is `9.91 ms` versus average compute time `32.27 ms`,
- `P-LeNet` average key read time is `109.73 ms` versus average compute time `107.73 ms`,
- `P-AlexNet` average key read time is `104.82 ms` versus average compute time `121.73 ms`.

These measurements come from `orca_runner/logs/master.log` and show that key-read cost is already comparable to compute for larger models.

At the same time, the evaluator already overlaps key reading and computation at block granularity, so the residual bottleneck is not just raw disk throughput. Repeated runtime `moveToGPU()` calls in the Orca layers remain part of the online overhead.

### Safe Use In The Abstract

The strongest abstract-safe version of the professor's storyline is:

- present the paper as improving memory efficiency of GPU-accelerated FSS,
- use existing profiling to motivate that storage, key I/O, and runtime transfers are the pressure points,
- present the standalone DPF online key generation benchmark as evidence that chunked key generation can sharply reduce peak staged key footprint,
- present Ring-LPN acceleration as the concrete implemented online-phase prototype result,
- mention direct I/O as already-present optimization infrastructure rather than a new completed contribution,
- keep full Orca or SPFSS-backed integration explicitly framed as ongoing work.

## Next Gap To Close

If stronger abstract claims are needed after this, the next engineering step is to replace `synthetic_mpvole` inputs with an external SPFSS or MPVOLE-backed input boundary while keeping the current GPU expansion path unchanged.