# GPU FSS Benchmark Appendix

Generated: 2026-04-20

## Purpose

This appendix collects the raw benchmark tables, exact experiment commands, and implementation provenance behind the current abstract-ready GPU FSS results.

## What Changed In The 2026-04-20 Refresh

All VOLE and DPF tables below were re-collected on 2026-04-20 after a set of building-block fixes:

- `run_vole_sweep.sh` now routes stderr to a sibling `.log` file and filters non-CSV-shape lines out of the output, so CSV files cannot be contaminated by usage, validation, or CUDA messages.
- `summarize_vole_results.py` now applies the same filter, matching the DPF summarizer.
- The DPF benchmark now calls `cudaDeviceSynchronize()` at the end of each timed lambda, so measured latencies include all asynchronously queued work.
- The DPF benchmark now validates per-chunk key layout inside `generate_pair_partial(...)` in the same pass that measures peak pair bytes, eliminating a redundant second pass over all chunks.
- The VOLE benchmark now computes `NTT(a)` once outside the per-iteration loop via a new `run_polymul_prepared_lhs(...)` entry point in the promoted polynomial core, avoiding two redundant forward NTTs on `a` per iteration.
- VOLE timing variable `ms` was renamed to `elapsed_ms`, which is what `cudaEventElapsedTime(...)` actually returns.

Every row in every refreshed sweep still reports `correct=1` on the coefficient-wise validator.

## Artifact Index

Primary raw artifacts:

- `results/ntt_cpu.md`
- `results/ntt_gpu_q32.md`
- `results/ntt_gpu_q64.md`
- `results/cpu_gpu_8192_32_batch64.md`
- `results/vole_gpu_q32_m32_c2_w64.md`
- `results/vole_gpu_q64_m32_c2_w64.md`
- `results/vole_gpu_q32_m64_c2_w64.md`
- `results/dpf_online_keygen_bin16_chunk8192.md`
- `results/dpf_online_keygen_bin16_chunk4096.md`
- `results/dpf_online_keygen_bin16_chunk2048.md`
- `results/cheddar_extract_note.md`

The matching `.csv` files in the same directory preserve the raw comma-separated output emitted by the benchmark binaries.

## Runtime And Command Provenance

All GPU experiments were run inside the `orca-dev` container.

Path mapping:

- host: `/home/fatih/EzPC/GPU-MPC`
- container root for GPU-MPC: `/home`
- Ring-LPN workdir: `/home/ringlpn`

Commands used for the saved artifact families:

### NTT / PolyMul core

- `docker exec -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh`
- `docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh`

### VOLE baseline sweeps

- `docker exec -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh`
- `docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh`

### Additional VOLE sensitivity sweep

- `docker exec -e M=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh`

### DPF online key-generation sweeps

- `docker exec -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py`
- `docker exec -e CHUNK_SIZE=4096 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py`
- `docker exec -e CHUNK_SIZE=2048 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py`

## Source Files Behind The Experiments

### DPF online key generation

- `tests/fss/dpf_online_keygen_bench.cu`
- `scripts/run_dpf_online_keygen_sweep.py`

The key benchmark functions are:

- `estimate_dpf_key_bytes_single_party(...)`
- `generate_pair_full(...)`
- `generate_pair_partial(...)` (now carries the layout check inline)
- `validate_key_layout(...)`
- `run_benchmark(...)`

### VOLE prototype

- `ringlpn/src/bench_vole_ringlpn.cu`
- `ringlpn/scripts/run_vole_sweep.sh`
- `ringlpn/scripts/summarize_vole_results.py`

The key benchmark functions are:

- `sample_uniform_polys(...)`
- `sample_sparse_noise_polys(...)`
- `repeat_rhs_for_outputs(...)`
- `run_inner_product_phase(...)` (now takes the pre-computed `NTT(a)` as input)
- `host_expand_phase(...)`
- `validate_vole_relation(...)`

### Promoted NTT/PolyMul core

- `ringlpn/src/bench_ntt_cuda_cheddar.cu`
- `ringlpn/results/cheddar_extract_note.md`

The most important design hooks are:

- `compute_cheddar_tables(...)`
- the extracted two-phase NTT / INTT kernels
- the single-prime batched wrapper path
- `run_full_polymul(...)`
- `run_polymul_prepared_lhs(...)` (new entry point that skips the forward NTT on an already-transformed left operand)

## Implementation Decisions That Explain The Results

### DPF chunking

The DPF benchmark compares one-shot full-key generation with a loop over chunks. That directly causes the tradeoff curve:

- smaller chunks reduce peak staged key footprint,
- the total logical payload stays effectively unchanged,
- time overhead rises because the generation loop repeats chunk setup and serialization work.

The measured `total_bytes_multiplier` staying at `1.000x` is expected from repeated per-chunk headers, not from any change in logical key content.

### VOLE batching

The VOLE benchmark computes `x`, `y`, and `z` through three calls to `run_inner_product_phase(...)`. Each call runs the promoted polynomial core across `pair_batch = m * c` batched pairs, then reduces over lanes and adds offsets. `NTT(a)` is now computed once before the timing loop and shared across the three phases.

That design explains why the `m=64` sensitivity sweep helps most at small `n`: larger batches amortize fixed launch overhead, while high-degree points converge because the polynomial core dominates the runtime.

### Promoted NTT / PolyMul core

The promoted core inherits the cheddar-style two-phase NTT design, bit-reversed `psi` / `psi^{-1}` tables, and OF-twiddle MSB slices, but adapts the runtime into a single-prime batched Ring-LPN harness.

That design is what makes the core fast enough to support both the standalone NTT benchmark story and the VOLE expansion prototype.

## Important Comparison Note

The direct `n=8192` CPU-vs-GPU artifact in `results/cpu_gpu_8192_32_batch64.md` and the broader sweep-derived speedups from `results/ntt_cpu.md`, `results/ntt_gpu_q32.md`, and `results/ntt_gpu_q64.md` are separate saved benchmark campaigns. They should be cited as complementary evidence, not merged into one single speedup range.

## Derived Comparison Summaries

### q=32 CPU-overlap speedups derived from saved sweep tables

| n | CPU per-poly PolyMul (us) | GPU per-poly PolyMul (us) | Speedup |
| --- | --- | --- | --- |
| 8192 | 181.228 | 1.244 | 145.68x |
| 16384 | 390.535 | 2.437 | 160.25x |
| 32768 | 817.860 | 4.785 | 170.92x |

### q=64 CPU-overlap speedups derived from saved sweep tables

| n | CPU per-poly PolyMul (us) | GPU per-poly PolyMul (us) | Speedup |
| --- | --- | --- | --- |
| 8192 | 190.245 | 3.963 | 48.00x |
| 16384 | 410.371 | 4.861 | 84.42x |
| 32768 | 874.429 | 7.278 | 120.15x |
| 65536 | 1838.590 | 8.772 | 209.60x |
| 131072 | 3994.480 | 28.757 | 138.90x |
| 262144 | 10011.200 | 57.012 | 175.60x |
| 524288 | 23712.900 | 115.568 | 205.18x |
| 1048576 | 53685.800 | 244.429 | 219.64x |

### DPF chunk-size tradeoff at `n=1048576`

| Chunk size | Peak pair key (MiB) | Peak reduction | Time overhead |
| --- | --- | --- | --- |
| 8192 | 2.81 | 128.00x | 1.834x |
| 4096 | 1.41 | 255.99x | 2.942x |
| 2048 | 0.70 | 511.97x | 4.975x |

### VOLE q=32 batching sensitivity: `m=32` vs `m=64`

| n | Per-output us at m=32 | Per-output us at m=64 | Outputs/s at m=32 | Outputs/s at m=64 |
| --- | --- | --- | --- | --- |
| 8192 | 5.984 | 5.000 | 167114.92 | 200004.38 |
| 16384 | 10.967 | 10.472 | 91182.64 | 95493.45 |
| 32768 | 21.309 | 22.208 | 46929.56 | 45027.93 |
| 65536 | 19.568 | 21.888 | 51104.17 | 45686.55 |
| 131072 | 92.713 | 120.454 | 10786.00 | 8301.94 |
| 262144 | 258.957 | 261.125 | 3861.65 | 3829.58 |
| 524288 | 500.503 | 497.825 | 1997.99 | 2008.74 |
| 1048576 | 1004.522 | 1001.198 | 995.50 | 998.80 |

### VOLE expand-latency change from the NTT-caching refactor

All numbers are `Full expand mean` in microseconds at the same `m=32`, `c=2`, noise weight `64` configuration.

| n | q=32 before | q=32 after | q=32 reduction | q=64 before | q=64 after | q=64 reduction |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 269.484 | 191.485 | -28.9% | 772.324 | 549.802 | -28.8% |
| 16384 | 504.242 | 350.944 | -30.4% | 1024.820 | 702.911 | -31.4% |
| 32768 | 963.231 | 681.873 | -29.2% | 1537.150 | 1169.510 | -23.9% |
| 65536 | 757.046 | 626.172 | -17.3% | 2359.950 | 1735.660 | -26.5% |
| 131072 | 4214.560 | 2966.810 | -29.6% | 8064.960 | 6047.390 | -25.0% |
| 262144 | 11058.700 | 8286.620 | -25.1% | 16634.400 | 12604.900 | -24.2% |
| 524288 | 21721.100 | 16016.100 | -26.3% | 33358.700 | 25295.400 | -24.2% |
| 1048576 | 43392.000 | 32144.700 | -25.9% | 67531.700 | 50952.700 | -24.5% |

The "before" numbers are the 2026-04-10 saved sweep values, preserved in git history and copied into this table for reference.

## Raw Benchmark Tables

### CPU vs GPU direct comparison at `n=8192`

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU (NFLLib) | 30 | 1 | pass | 57.2021 | 61.8469 | 180.594 | 180.594 | n/a |
| GPU (CUDA) | 30 | 64 | pass | 41.7984 | 45.8986 | 129.509 | 2.024 | 1 |

Saved direct speedups:

- Forward NTT speedup per polynomial: `87.59x`
- Full PolyMul speedup per polynomial: `89.24x`

### Raw NTT q=32 GPU sweep

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 26.777 | 25.925 | 79.628 | 1.244 | 803741.42 | 52.67 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 51.862 | 50.483 | 155.992 | 2.437 | 410277.45 | 53.78 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 100.302 | 99.889 | 306.232 | 4.785 | 208991.88 | 54.79 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 14.006 | 13.266 | 42.260 | 2.641 | 378613.09 | 198.50 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 96.612 | 99.303 | 298.463 | 18.654 | 53607.98 | 56.21 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 110.216 | 112.384 | 337.991 | 42.249 | 23669.27 | 49.64 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 105.022 | 104.251 | 320.102 | 80.025 | 12496.02 | 52.41 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 109.050 | 109.437 | 332.762 | 166.381 | 6010.30 | 50.42 |

### Raw NTT q=64 GPU sweep

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 64 | pass | 400 | 86.117 | 82.512 | 253.654 | 3.963 | 252312.20 | 33.07 |
| 16384 | 14 | 64 | 62 | 64 | pass | 400 | 103.020 | 100.905 | 311.081 | 4.861 | 205734.20 | 53.93 |
| 32768 | 15 | 64 | 62 | 64 | pass | 400 | 157.157 | 146.257 | 465.788 | 7.278 | 137401.56 | 72.04 |
| 65536 | 16 | 64 | 62 | 16 | pass | 200 | 47.803 | 40.438 | 140.357 | 8.772 | 113995.03 | 119.53 |
| 131072 | 17 | 64 | 62 | 16 | pass | 200 | 155.839 | 144.325 | 460.117 | 28.757 | 34773.76 | 72.93 |
| 262144 | 18 | 64 | 62 | 8 | pass | 80 | 153.630 | 143.456 | 456.098 | 57.012 | 17540.09 | 73.57 |
| 524288 | 19 | 64 | 62 | 4 | pass | 30 | 156.621 | 143.510 | 462.271 | 115.568 | 8652.93 | 72.59 |
| 1048576 | 20 | 64 | 62 | 2 | pass | 10 | 167.523 | 151.149 | 488.858 | 244.429 | 4091.17 | 68.64 |

### Raw DPF sweep, chunk size `8192`

| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 16 | 8192 | pass | 100 | 2.81 | 2.81 | 1.00x | 1.000x | 269.920 | 268.750 | 0.996x |
| 16384 | 16 | 8192 | pass | 100 | 5.63 | 2.81 | 2.00x | 1.000x | 378.900 | 523.330 | 1.381x |
| 32768 | 16 | 8192 | pass | 100 | 11.25 | 2.81 | 4.00x | 1.000x | 689.500 | 1057.270 | 1.533x |
| 65536 | 16 | 8192 | pass | 50 | 22.50 | 2.81 | 8.00x | 1.000x | 1235.280 | 2107.640 | 1.706x |
| 131072 | 16 | 8192 | pass | 50 | 45.00 | 2.81 | 16.00x | 1.000x | 2402.640 | 4180.980 | 1.740x |
| 262144 | 16 | 8192 | pass | 20 | 90.00 | 2.81 | 32.00x | 1.000x | 4666.900 | 8364.700 | 1.792x |
| 524288 | 16 | 8192 | pass | 10 | 180.00 | 2.81 | 64.00x | 1.000x | 9177.500 | 16723.200 | 1.822x |
| 1048576 | 16 | 8192 | pass | 3 | 360.00 | 2.81 | 128.00x | 1.000x | 18242.700 | 33458.700 | 1.834x |

### Raw DPF sweep, chunk size `4096`

| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 16 | 4096 | pass | 100 | 2.81 | 1.41 | 2.00x | 1.000x | 271.480 | 429.790 | 1.583x |
| 16384 | 16 | 4096 | pass | 100 | 5.63 | 1.41 | 4.00x | 1.000x | 378.120 | 822.400 | 2.175x |
| 32768 | 16 | 4096 | pass | 100 | 11.25 | 1.41 | 8.00x | 1.000x | 691.320 | 1664.270 | 2.407x |
| 65536 | 16 | 4096 | pass | 50 | 22.50 | 1.41 | 16.00x | 1.000x | 1234.980 | 3297.320 | 2.670x |
| 131072 | 16 | 4096 | pass | 50 | 45.00 | 1.41 | 32.00x | 1.000x | 2412.360 | 6590.140 | 2.732x |
| 262144 | 16 | 4096 | pass | 20 | 90.00 | 1.41 | 64.00x | 1.000x | 4677.850 | 13201.200 | 2.822x |
| 524288 | 16 | 4096 | pass | 10 | 180.00 | 1.41 | 128.00x | 1.000x | 9176.100 | 26310.000 | 2.867x |
| 1048576 | 16 | 4096 | pass | 3 | 360.00 | 1.41 | 255.99x | 1.000x | 18242.700 | 53662.000 | 2.942x |

### Raw DPF sweep, chunk size `2048`

| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 16 | 2048 | pass | 100 | 2.81 | 0.70 | 4.00x | 1.000x | 269.840 | 738.720 | 2.738x |
| 16384 | 16 | 2048 | pass | 100 | 5.63 | 0.70 | 8.00x | 1.000x | 377.880 | 1421.810 | 3.763x |
| 32768 | 16 | 2048 | pass | 100 | 11.25 | 0.70 | 16.00x | 1.000x | 690.840 | 2844.310 | 4.117x |
| 65536 | 16 | 2048 | pass | 50 | 22.50 | 0.70 | 32.00x | 1.000x | 1236.620 | 5672.160 | 4.587x |
| 131072 | 16 | 2048 | pass | 50 | 45.00 | 0.70 | 64.00x | 1.000x | 2403.340 | 11386.100 | 4.738x |
| 262144 | 16 | 2048 | pass | 20 | 90.00 | 0.70 | 127.99x | 1.000x | 4666.100 | 22696.300 | 4.864x |
| 524288 | 16 | 2048 | pass | 10 | 180.00 | 0.70 | 255.98x | 1.000x | 9181.000 | 45335.200 | 4.938x |
| 1048576 | 16 | 2048 | pass | 3 | 360.00 | 0.70 | 511.97x | 1.000x | 18239.300 | 90744.000 | 4.975x |

### Raw VOLE sweep, requested q=`32`, baseline `m=32`

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 59.087 | 59.076 | 59.079 | 191.485 | 5.984 | 167114.92 | 1002689.51 |
| 16384 | 14 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 112.068 | 112.147 | 112.082 | 350.944 | 10.967 | 91182.64 | 547095.83 |
| 32768 | 15 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 222.655 | 222.491 | 221.462 | 681.873 | 21.309 | 46929.56 | 281577.36 |
| 65536 | 16 | 32 | 30 | 32 | 2 | 64 | pass | 100 | 203.034 | 202.637 | 203.440 | 626.172 | 19.568 | 51104.17 | 306625.02 |
| 131072 | 17 | 32 | 30 | 32 | 2 | 64 | pass | 100 | 985.359 | 983.248 | 982.725 | 2966.810 | 92.713 | 10786.00 | 64715.97 |
| 262144 | 18 | 32 | 30 | 32 | 2 | 64 | pass | 40 | 2768.370 | 2749.860 | 2750.620 | 8286.620 | 258.957 | 3861.65 | 23169.88 |
| 524288 | 19 | 32 | 30 | 32 | 2 | 64 | pass | 20 | 5335.770 | 5330.620 | 5333.980 | 16016.100 | 500.503 | 1997.99 | 11987.94 |
| 1048576 | 20 | 32 | 30 | 32 | 2 | 64 | pass | 10 | 10717.700 | 10705.500 | 10705.200 | 32144.700 | 1004.522 | 995.50 | 5972.99 |

### Raw VOLE sweep, requested q=`64`, baseline `m=32`

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 178.396 | 178.413 | 178.432 | 549.802 | 17.181 | 58202.77 | 349216.63 |
| 16384 | 14 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 228.715 | 229.491 | 229.230 | 702.911 | 21.966 | 45524.97 | 273149.80 |
| 32768 | 15 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 384.284 | 383.792 | 384.338 | 1169.510 | 36.547 | 27361.89 | 164171.32 |
| 65536 | 16 | 64 | 62 | 32 | 2 | 64 | pass | 100 | 574.454 | 572.316 | 572.054 | 1735.660 | 54.239 | 18436.79 | 110620.74 |
| 131072 | 17 | 64 | 62 | 32 | 2 | 64 | pass | 100 | 2020.030 | 2010.170 | 2000.340 | 6047.390 | 188.981 | 5291.54 | 31749.23 |
| 262144 | 18 | 64 | 62 | 32 | 2 | 64 | pass | 40 | 4187.240 | 4196.770 | 4204.290 | 12604.900 | 393.903 | 2538.70 | 15232.17 |
| 524288 | 19 | 64 | 62 | 32 | 2 | 64 | pass | 20 | 8413.860 | 8433.120 | 8431.390 | 25295.400 | 790.481 | 1265.05 | 7590.31 |
| 1048576 | 20 | 64 | 62 | 32 | 2 | 64 | pass | 10 | 16934.000 | 17001.500 | 16999.300 | 50952.700 | 1592.272 | 628.03 | 3768.20 |

### Raw VOLE sweep, requested q=`32`, sensitivity run at `m=64`

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | 2 | 64 | pass | 200 | 101.808 | 101.740 | 101.720 | 319.993 | 5.000 | 200004.38 | 1200026.25 |
| 16384 | 14 | 32 | 30 | 64 | 2 | 64 | pass | 200 | 218.534 | 218.379 | 217.760 | 670.203 | 10.472 | 95493.45 | 572960.73 |
| 32768 | 15 | 32 | 30 | 64 | 2 | 64 | pass | 200 | 468.334 | 467.335 | 468.665 | 1421.340 | 22.208 | 45027.93 | 270167.59 |
| 65536 | 16 | 32 | 30 | 64 | 2 | 64 | pass | 100 | 461.861 | 462.433 | 461.221 | 1400.850 | 21.888 | 45686.55 | 274119.28 |
| 131072 | 17 | 32 | 30 | 64 | 2 | 64 | pass | 100 | 2574.300 | 2558.190 | 2558.720 | 7709.040 | 120.454 | 8301.94 | 49811.65 |
| 262144 | 18 | 32 | 30 | 64 | 2 | 64 | pass | 40 | 5568.210 | 5562.250 | 5564.710 | 16712.000 | 261.125 | 3829.58 | 22977.50 |
| 524288 | 19 | 32 | 30 | 64 | 2 | 64 | pass | 20 | 10610.100 | 10620.200 | 10614.300 | 31860.800 | 497.825 | 2008.74 | 12052.43 |
| 1048576 | 20 | 32 | 30 | 64 | 2 | 64 | pass | 10 | 21371.100 | 21345.900 | 21343.500 | 64076.700 | 1001.198 | 998.80 | 5992.82 |
