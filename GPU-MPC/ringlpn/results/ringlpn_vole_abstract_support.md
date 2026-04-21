# Ring-LPN VOLE Abstract Support Note

Generated: 2026-04-20

## Scope

This note summarizes the current state of the standalone Ring-LPN VOLE and DPF online-key-generation benchmarks under `GPU-MPC/ringlpn` and records the benchmark claims that are safe to use in an abstract.

The current prototype is not a full SPFSS-backed OLE-R-LPN implementation yet. It isolates the algebraic expansion layer using synthetic MPVOLE-consistent inputs and validates the coefficient-wise relation

`z = y + x * Delta`

for all benchmarked points.

## What Was Implemented

The main benchmark sources are:

- `ringlpn/src/bench_vole_ringlpn.cu`
- `ringlpn/src/bench_ntt_cuda_cheddar.cu`
- `tests/fss/dpf_online_keygen_bench.cu`

The VOLE benchmark reuses the promoted cheddar-derived single-prime GPU polynomial path rather than introducing a second CUDA polynomial engine.

The current prototype supports:

- requested `q=32`, realized as actual `q=30`,
- requested `q=64`, realized as actual `q=62`,
- `n` in `{8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}`,
- baseline VOLE sweep configuration `m=32`, `c=2`, and noise weight `64`,
- one additional VOLE sensitivity sweep at requested `q=32`, `m=64`, `c=2`, noise weight `64`,
- DPF online key-generation sweeps at chunk sizes `8192`, `4096`, and `2048`.

## Measurement Setup

All benchmark runs below were executed inside the `orca-dev` container.

Important path mapping:

- host path: `/home/fatih/EzPC/GPU-MPC`
- container root for GPU-MPC: `/home`
- Ring-LPN workdir inside container: `/home/ringlpn`

### Exact Experiment Commands

VOLE baseline sweeps:

- `docker exec -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh`
- `docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh`

Additional VOLE sensitivity sweep:

- `docker exec -e QBITS=32 -e M=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh`

DPF online key-generation sweeps:

- `docker exec -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py`
- `docker exec -e CHUNK_SIZE=4096 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py`
- `docker exec -e CHUNK_SIZE=2048 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py`

NTT/PolyMul core artifacts referenced below come from the existing Ring-LPN sweep drivers under `/home/ringlpn`:

- `bash scripts/run_cuda_sweep.sh`
- `QBITS=64 bash scripts/run_cuda_sweep.sh`

### Sweep Scheduling Logic

The reported iteration counts are not ad hoc; they come from the sweep drivers:

- `ringlpn/scripts/run_vole_sweep.sh` uses `200/20` warmup-iters up to `n=32768`, then `100/10`, `40/5`, `20/3`, and `10/2` as `n` grows.
- `scripts/run_dpf_online_keygen_sweep.py` uses `100/10` up to `n=32768`, then `50/5`, `20/3`, `10/2`, and `3/1`.

This means the newer `chunk=2048` and `m=64` datasets were collected under the same scheduling logic as the earlier saved artifacts.

## Artifact Inventory

The primary saved raw artifacts are:

- `results/vole_gpu_q32_m32_c2_w64.csv`
- `results/vole_gpu_q32_m32_c2_w64.md`
- `results/vole_gpu_q64_m32_c2_w64.csv`
- `results/vole_gpu_q64_m32_c2_w64.md`
- `results/vole_gpu_q32_m64_c2_w64.csv`
- `results/vole_gpu_q32_m64_c2_w64.md`
- `results/dpf_online_keygen_bin16_chunk8192.csv`
- `results/dpf_online_keygen_bin16_chunk8192.md`
- `results/dpf_online_keygen_bin16_chunk4096.csv`
- `results/dpf_online_keygen_bin16_chunk4096.md`
- `results/dpf_online_keygen_bin16_chunk2048.csv`
- `results/dpf_online_keygen_bin16_chunk2048.md`
- `results/ntt_gpu_q32.md`
- `results/ntt_gpu_q64.md`
- `results/ntt_cpu.md`
- `results/cpu_gpu_8192_32_batch64.md`
- `results/cheddar_extract_note.md`

Detailed raw tables, exact commands, and code-path provenance are collected in:

- `results/abstract_benchmark_appendix.md`

## Current VOLE Prototype Results

### q=32, baseline sweep (`m=32`, `c=2`, noise weight `64`)

For requested `q=32` with the baseline sweep configuration:

- at `n=8192`, full expansion latency is `191.485 us`, per-output expansion is `5.984 us`, and throughput is `167114.92` outputs/s,
- at `n=32768`, full expansion latency is `681.873 us`, per-output expansion is `21.309 us`, and throughput is `46929.56` outputs/s,
- at `n=1048576`, full expansion latency is `32144.700 us`, per-output expansion is `1004.522 us`, and throughput is `995.50` outputs/s.

Every saved sweep point passed validation.

### q=64, baseline sweep (`m=32`, `c=2`, noise weight `64`)

For requested `q=64` with the same baseline configuration:

- at `n=8192`, full expansion latency is `549.802 us`, per-output expansion is `17.181 us`, and throughput is `58202.77` outputs/s,
- at `n=32768`, full expansion latency is `1169.510 us`, per-output expansion is `36.547 us`, and throughput is `27361.89` outputs/s,
- at `n=1048576`, full expansion latency is `50952.700 us`, per-output expansion is `1592.272 us`, and throughput is `628.03` outputs/s.

Every saved sweep point passed validation.

### Additional VOLE sensitivity sweep: q=32 with `m=64`

To add one more experimental axis beyond the baseline abstract sweep, an additional q=`32` run was collected at `m=64`, `c=2`, noise weight `64`.

The most useful observations are:

- at `n=8192`, full expansion latency is `319.993 us`, per-output expansion is `5.000 us`, and throughput is `200004.38` outputs/s,
- at `n=32768`, full expansion latency is `1421.340 us`, per-output expansion is `22.208 us`, and throughput is `45027.93` outputs/s,
- at `n=1048576`, full expansion latency is `64076.700 us`, per-output expansion is `1001.198 us`, and throughput is `998.80` outputs/s.

Compared with the baseline `m=32` sweep, doubling `m` improves per-output cost at small `n` but largely converges by the largest degrees. That pattern is consistent with the benchmark design: larger `m` increases `pair_batch = m * c`, which amortizes fixed launch overheads at small sizes, while large-degree points are dominated by the polynomial core itself.

## Standalone DPF Online Key Generation Results

The codebase now has a standalone DPF online key-generation benchmark in:

- `tests/fss/dpf_online_keygen_bench.cu`

It is built with:

- `make dpf_online_keygen`

and swept by:

- `scripts/run_dpf_online_keygen_sweep.py`

### Baseline chunk-size sweep points

Chunk size `8192`:

- at `n=8192`, the full pair key is `2.81 MiB`, the partial online peak pair key is also `2.81 MiB`, and time overhead is `0.996x`,
- at `n=16384`, the full pair key is `5.63 MiB`, the partial online peak pair key is `2.81 MiB`, peak reduction is `2.00x`, and time overhead is `1.381x`,
- at `n=1048576`, the full pair key is `360.00 MiB`, the partial online peak pair key remains `2.81 MiB`, peak reduction reaches `128.00x`, and time overhead is `1.834x`.

Chunk size `4096`:

- at `n=8192`, the partial online peak pair key is `1.41 MiB` and time overhead is `1.583x`,
- at `n=1048576`, the partial online peak pair key is `1.41 MiB`, peak reduction reaches about `255.99x`, and time overhead is `2.942x`.

### Additional DPF experiment: chunk size `2048`

To extend the tradeoff curve with one more measured point, an additional full sweep was collected at `chunk_size=2048`.

The key numbers are:

- at `n=8192`, the partial online peak pair key is `0.70 MiB`, peak reduction is `4.00x`, and time overhead is `2.738x`,
- at `n=65536`, the partial online peak pair key is `0.70 MiB`, peak reduction is `32.00x`, and time overhead is `4.587x`,
- at `n=1048576`, the partial online peak pair key is `0.70 MiB`, peak reduction reaches about `511.97x`, and time overhead is `4.975x`.

Across all three chunk sizes, the total generated key bytes remain effectively unchanged. The measured `total_bytes_multiplier` stays at about `1.00005` to `1.00006`, which is exactly the behavior expected from the code path: chunking changes peak staged footprint and loop structure, not the underlying logical payload, and the small multiplier above `1.0` is header/layout overhead from repeating the per-chunk key structure.

## How This Connects To Existing Ring-LPN Evidence

The VOLE prototype sits on top of the already-validated promoted CUDA polynomial engine. The strongest current claims for that underlying engine are:

- a saved direct `n=8192` CPU-vs-GPU comparison records `87.59x` forward-NTT and `89.24x` full-PolyMul per-polynomial speedup over the NFLLib baseline,
- sweep-derived requested `q=32` overlap points show about `145.68x`, `160.25x`, and `170.92x` per-polynomial PolyMul speedup over the CPU baseline at `n=8192`, `16384`, and `32768`,
- sweep-derived requested `q=64` overlap points show about `48.00x` to `219.64x` per-polynomial PolyMul speedup over the CPU baseline across `n=8192` to `1048576`,
- the validated promoted q=`64` sweep reports a `33.07 GB/s` estimated coefficient-throughput proxy at `n=8192`,
- the promoted cheddar-derived path is consistently faster than the preserved legacy CUDA baseline.

Important comparison note:

The direct `87.59x` and `89.24x` numbers from `results/cpu_gpu_8192_32_batch64.md` and the larger q=`32` sweep-derived ratios above come from different saved benchmark campaigns. They are complementary evidence, not one single merged ratio range, and should be cited that way.

## Code Design Behind The Measured Behavior

### DPF online key generation

The DPF benchmark behavior is driven by three explicit code paths in `tests/fss/dpf_online_keygen_bench.cu`:

- `generate_pair_full(...)` generates both parties' eval-all keys in one shot,
- `generate_pair_partial(...)` loops over `offset += chunk_size` and generates only the current chunk's keys,
- `validate_key_layout(...)` checks serialized layout and parsed metadata for both full and chunked modes.

That design directly explains the measured trend:

- peak memory drops approximately in proportion to chunk size because only one chunk is staged at a time,
- total bytes stay effectively flat because the logical key payload is unchanged,
- time overhead grows as chunk size shrinks because the loop repeats setup and serialization work more times.

### VOLE prototype

The VOLE benchmark behavior is driven by the structure of `ringlpn/src/bench_vole_ringlpn.cu`:

- `sample_sparse_noise_polys(...)`, `sample_uniform_polys(...)`, and `scalar_mul_add_batches(...)` synthesize MPVOLE-consistent inputs,
- `repeat_rhs_for_outputs(...)` materializes the batched right-hand-side layout,
- `run_inner_product_phase(...)` performs the batched polynomial-multiplication-and-reduction step,
- the benchmark computes `x`, `y`, and `z` by calling `run_inner_product_phase(...)` three times,
- `host_expand_phase(...)` and `validate_vole_relation(...)` provide reference checking and coefficient-wise validation.

This means the VOLE benchmark is not measuring a hidden SPFSS pipeline. It is intentionally measuring the cost of three batched inner-product phases on top of the promoted polynomial core. The extra `m=64` sweep is therefore informative: when `m` grows, the benchmark increases the pair batch `m * c`, which helps small-`n` overhead amortization but offers much less benefit once large-`n` polynomial work dominates.

### Promoted NTT/PolyMul core

The promoted polynomial engine behavior comes from `ringlpn/src/bench_ntt_cuda_cheddar.cu` and is documented in `results/cheddar_extract_note.md`.

The most important design points are:

- `compute_cheddar_tables(...)` builds bit-reversed `psi` and `psi^{-1}` tables and the OF-twiddle MSB slices expected by the extracted kernels,
- the extracted path keeps the cheddar-style two-phase NTT and INTT structure,
- the standalone adaptation uses a single-prime runtime and converts `grid.y` into a simpler batch dimension for many independent polynomials,
- `run_full_polymul(...)` benchmarks full polynomial multiplication directly on batched inputs.

This is why the core results are strong: the promoted path is both architecturally better than the legacy CUDA path and explicitly optimized for batch throughput rather than single-poly orchestration overhead.

## Safe Abstract Claims

The following points are supported by the current code and saved artifacts:

1. We implemented a standalone GPU prototype of the Ring-LPN VOLE-style expansion layer under the existing Ring-LPN benchmark harness.
2. The prototype reuses a validated promoted single-prime GPU polynomial backend rather than a separate ad hoc kernel path.
3. The current prototype validates the intended algebraic VOLE relation coefficient-wise across the full tested degree range from `8192` to `1048576` for both requested `q=32` and requested `q=64`.
4. In the baseline abstract-ready sweep configuration (`m=32`, `c=2`, noise weight `64`), q=32 full expansion latency ranges from `191.485 us` to `32.145 ms`, and q=64 full expansion latency ranges from `549.802 us` to `50.953 ms`.
5. The underlying GPU polynomial engine already has strong CPU-vs-GPU evidence, including a saved `n=8192` comparison with `87.59x` forward-NTT and `89.24x` full-PolyMul per-polynomial speedup over NFLLib.
6. We implemented a standalone DPF online key-generation benchmark for eval-all keys and measured that chunked generation can cap peak pair-key footprint at `2.81 MiB`, `1.41 MiB`, or `0.70 MiB` across the current sweeps while scaling one-shot full pair-key footprint up to `360.00 MiB`.
7. In the saved DPF sweeps, peak-footprint reduction reaches `128.00x` at chunk `8192`, about `255.99x` at chunk `4096`, and about `511.97x` at chunk `2048`, all at `n=1048576`.
8. The cost of that stronger footprint reduction is a tunable time-overhead curve: about `1.834x`, `2.942x`, and `4.975x` at `n=1048576` for chunk sizes `8192`, `4096`, and `2048` respectively.
9. An additional q=`32`, `m=64` VOLE sweep shows that per-output latency improves at small `n` and largely converges at larger `n`, which is consistent with the benchmark's batched inner-product design.

## Claims To Avoid

The following statements are not supported yet and should not appear in an abstract without more implementation work:

1. Claiming that the full SPFSS-backed OLE-R-LPN or degree-1 correlation pipeline from the figure is complete end to end.
2. Claiming a CPU-vs-GPU speedup number for the current VOLE prototype itself.
3. Claiming end-to-end dealer/evaluator integration or a fully integrated online key-generation pipeline for the Ring-LPN path.
4. Claiming q=`128` support for the VOLE prototype.

## Recommended Abstract Positioning

The safest framing is:

- open by explaining what FSS is, why GPU acceleration is attractive for machine-learning workloads, and that Orca is the relevant GPU-accelerated FSS library in this codebase,
- use the first paragraph to motivate both key I/O and memory pressure, not just I/O, with the existing Orca profiling data,
- start the second paragraph from Ring-LPN-based PCG as the proposed way to reduce staged key material, then introduce the Orca-oriented three-part path of chunked DPF online key generation, GPU Ring-LPN VOLE expansion, and the promoted GPU NTT/PolyMul backend,
- present the current work as standalone measured prototypes for chunked DPF online key generation and GPU Ring-LPN VOLE expansion,
- use the saved NTT and PolyMul core artifacts for the acceleration story,
- use the DPF chunk-size curve for the memory-efficiency story,
- use the additional `m=64` VOLE sweep as extra support for the batching design rather than as a headline claim,
- explicitly describe full Orca or SPFSS-backed integration as ongoing work.

## Raw Data And Provenance Appendix

For the full raw tables, exact commands, artifact index, derived comparison tables, and code-path provenance, use:

- `results/abstract_benchmark_appendix.md`

That appendix is the best place to pull benchmark tables directly into a submission, advisor email, or slide deck.

## Next Gap To Close

If stronger abstract claims are needed after this, the next engineering step is still to replace `synthetic_mpvole` inputs with an external SPFSS or MPVOLE-backed input boundary while keeping the current GPU expansion path unchanged.

## Building-block Changes Since 2026-04-10

All VOLE and DPF numbers above were re-collected on 2026-04-20 after the following fixes to the underlying building blocks. The refresh is load-bearing for the scientific accuracy of the abstract numbers:

- The VOLE benchmark now computes `NTT(a)` once per `(m, c, n)` configuration and reuses it across the `x`, `y`, and `z` inner-product phases through a new `run_polymul_prepared_lhs(...)` entry point in the promoted polynomial core. This removes two redundant forward NTTs on the shared left operand per iteration, and shrinks end-to-end VOLE expand latency by roughly 24-31% across the full sweep range for both requested `q=32` and requested `q=64`.
- The DPF benchmark now calls `cudaDeviceSynchronize()` at the end of each timed lambda, so the measured full-pair and partial-pipeline latencies include every asynchronously queued kernel and memcpy.
- The DPF benchmark now folds per-chunk layout validation into `generate_pair_partial(...)` in the same pass that measures peak pair bytes, eliminating a redundant second pass over all chunks that did not match the measured pipeline timing.
- `run_vole_sweep.sh` now routes stderr to a sibling `.log` file and only appends CSV-shape lines to the CSV, so stderr content (usage, validation mismatch, CUDA error text) can no longer corrupt the saved CSV.
- `summarize_vole_results.py` now mirrors the DPF summarizer and filters out non-CSV-shape lines before reading, so a stray log line cannot crash the markdown generator or quietly emit wrong rows.

Every row in every refreshed sweep still reports `correct=1` on the coefficient-wise validator.