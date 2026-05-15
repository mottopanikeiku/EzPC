# Poster Creation Prompt: Methodology And Benchmark Data

Use this prompt to create the workshop poster for the GPU-MPC project. Do not focus on design philosophy. Focus on the submitted abstract, methodology, experiments, compute setup, benchmark tables, supported claims, and current integration status.

## Project

Title:

**Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing**

System context:

- Project: `GPU-MPC`
- Main system: Orca, a GPU-accelerated FSS-based PPML system.
- Current research track: memory-efficient GPU-FSS preprocessing via chunked DPF online key generation and GPU Ring-LPN PCG building blocks.
- Main benchmark area: `GPU-MPC/ringlpn`
- Orca profiling evidence: `GPU-MPC/orca_runner/logs/master.log`
- Benchmark appendix: `GPU-MPC/ringlpn/results/abstract_benchmark_appendix.md`

## Submitted Abstract Block

Use this lightly shortened version of the submitted abstract if the poster needs an abstract text box. Keep the meaning and claims close to the submitted version.

```text
Privacy-preserving machine learning (PPML) protocols partition computation across two paradigms: function secret sharing (FSS) handles non-linear operations such as ReLU and comparison, while additive secret sharing handles linear operations. Both use offline/online decompositions to minimize online latency, but this shifts the burden to the offline phase, where generating and storing correlated randomness becomes the dominant bottleneck. Our profiling of Orca, a GPU-accelerated FSS-based PPML system, reveals that precomputed FSS keys grow from gigabytes to tens of gigabytes as model complexity increases, with key storage, I/O, and memory movement consuming as much time as GPU computation even for moderate-sized models.

We propose a unified acceleration framework addressing this offline bottleneck across both paradigms. For FSS-based non-linear evaluation, we replace large precomputed keys with compact seeds expanded on-the-fly on GPU. Chunked online key generation reduces peak staged key footprint by up to 128x over one-shot generation with under 2x time overhead, providing a tunable memory-efficiency knob. For secret-sharing-based linear evaluation, we GPU-accelerate Ring-LPN pseudorandom correlation generators (PCGs). Our GPU NTT engine, adapted from Cheddar's two-phase kernel structure, achieves roughly 89x polynomial multiplication speedup over the NFLLib CPU baseline, providing the fast polynomial core that online PCG expansion requires. Built on this, our GPU Ring-LPN VOLE expansion prototype demonstrates efficient GPU PCG-based correlation generation, with validated correctness across polynomial degrees from 8,192 to over one million for both 32-bit and 64-bit modulus widths. We are currently integrating these components into Orca.
```

## Poster Content Goal

The poster should make this technical case:

1. PPML systems split work between FSS non-linear operations and additive-secret-sharing linear operations.
2. Offline preprocessing makes online inference/training fast, but it creates large key/correlation material.
3. Orca profiling shows that key storage, key I/O, and memory movement are already major costs.
4. Chunked DPF online key generation is the memory-footprint experiment.
5. GPU Ring-LPN NTT/PolyMul is the polynomial acceleration experiment.
6. GPU Ring-LPN VOLE is the PCG expansion experiment.
7. Figure 2 OLE and ring-polynomial OLE-to-Beaver artifacts are bridge experiments toward linear-layer integration.
8. Full Orca integration is ongoing work.

## Methodology

### Orca Profiling

Purpose:

- Measure whether precomputed FSS keys cause storage/I/O/memory pressure in GPU-MPC's Orca workflow.

What was measured:

- Key file sizes per party.
- Average key-read time.
- Average compute time.
- Communication per iteration.

Source:

- `GPU-MPC/orca_runner/logs/master.log`

Important implementation context:

- Orca uses GPU acceleration for FSS-based PPML.
- Existing key I/O uses direct I/O infrastructure:
  - `GPU-MPC/utils/gpu_file_utils.cpp` uses `O_DIRECT | O_LARGEFILE`.
  - key buffers are 4096-byte aligned.
  - `GPU-MPC/experiments/orca/orca_evaluator.cu` overlaps key reading and compute.
- Because overlap/direct I/O already exist, key movement remains a structural bottleneck rather than only a missing optimization.

### DPF Online Key Generation

Purpose:

- Evaluate whether chunked online DPF key generation can reduce peak staged key footprint.

Benchmark:

- `GPU-MPC/tests/fss/dpf_online_keygen_bench.cu`
- sweep driver: `GPU-MPC/scripts/run_dpf_online_keygen_sweep.py`

Method:

- Compare one-shot full-pair eval-all DPF key generation with partial chunked generation.
- Full mode materializes both parties' full eval-all keys at once.
- Chunked mode loops over fixed-size chunks and materializes only the current chunk.
- Validation checks serialized key layout and parsed metadata for full and chunked modes.

Configurations:

- `bin=16`
- chunk sizes: `8192`, `4096`, `2048`
- `N` from `8192` to `1048576`

Metrics:

- full pair key footprint
- partial peak pair-key footprint
- peak-footprint reduction
- total bytes multiplier
- full keygen mean
- partial pipeline mean
- time overhead

Important interpretation:

- Chunking reduces peak staged footprint, not total logical key material.
- Smaller chunks reduce peak memory more, but increase generation time overhead.
- This is a standalone key-generation systems benchmark, not an end-to-end FSS evaluation benchmark yet.

### GPU NTT / PolyMul Core

Purpose:

- Build the fast polynomial core needed for Ring-LPN PCG/VOLE expansion.

Benchmark:

- primary GPU source: `GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu`
- CPU baseline: `GPU-MPC/ringlpn/src/bench_ntt.cpp` using NFLLib
- legacy GPU baseline: `GPU-MPC/ringlpn/src/bench_ntt_cuda.cu`

Method:

- Adapt Cheddar's two-phase NTT/INTT kernel structure into the Ring-LPN benchmark harness.
- Benchmark batched full polynomial multiplication as:
  - `NTT(a)`
  - `NTT(b)`
  - pointwise multiply
  - `INTT`
- Report batch latency and per-polynomial latency by dividing by batch size.

Parameter mapping:

- requested `q=32` is implemented as actual `q=30` using one prime.
- requested `q=64` is implemented as actual `q=62` using one prime.
- requested `q=128` / CRT is not implemented yet.

Metrics:

- NTT mean
- INTT mean
- full PolyMul mean
- per-poly PolyMul
- PolyMul polys/s
- estimated coefficient GB/s
- validation status

### Ring-LPN VOLE Prototype

Purpose:

- Evaluate GPU-side Ring-LPN PCG-style VOLE expansion on top of the promoted NTT/PolyMul backend.

Benchmark:

- `GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu`

Method:

- Reuses promoted GPU PolyMul backend.
- Synthesizes MPVOLE-consistent inputs locally under `synthetic_mpvole`.
- Computes `x`, `y`, and `z` through three batched inner-product phases.
- Validates coefficient-wise VOLE relation:
  - `z = y + x * Delta`
- Uses `m=32`, `c=2`, noise weight `64` for baseline sweeps.
- Includes one q=32 sensitivity sweep with `m=64`.

Important interpretation:

- This measures the algebraic expansion layer.
- It is not a full SPFSS-backed OLE-R-LPN pipeline yet.
- Do not claim CPU-vs-GPU speedup for VOLE itself because there is no CPU VOLE baseline.

### Figure 2 OLE And Linear OLE-To-Beaver Bridge

Purpose:

- Show bridge artifacts toward replacing linear-layer Beaver preprocessing.

Benchmarks:

- OLE artifact: `GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu`
- GPU SPFSS payload path: `GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh`
- SPFSS test: `GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu`
- linear OLE-to-Beaver artifact: `GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu`

Method:

- Figure 2 OLE validates `z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N+1)`.
- Uniform sparse noise evaluates SPFSS over `[0, 2N)`.
- Regular sparse noise uses grouped SPFSS domains of size `2N/t`.
- Linear OLE-to-Beaver uses two Figure 2 OLE instances per ring product:
  - one OLE for `A_0 * B_1`
  - one OLE for `A_1 * B_0`
  - combine with local products to form Beaver shares over ring-polynomial matrix entries.

Important interpretation:

- This validates OLE-to-Beaver over ring-polynomial matrix entries.
- It is not yet Orca FC integration.
- Missing pieces before Orca `gpuMatmulBeaver` integration:
  - scalar packing from Orca tensor values into polynomial slots
  - `Z_p -> Z_{2^bw}` share conversion
  - Orca-compatible triple writer

## Compute / Runtime Setup

Known:

- All GPU experiments were run inside the `orca-dev` Docker container.
- Host GPU-MPC root: `/home/fatih/EzPC/GPU-MPC`
- Container GPU-MPC root: `/home`
- Container Ring-LPN workdir: `/home/ringlpn`
- GPU-MPC README says the environment has been tested on Ubuntu 20.04 with CUDA 11.7, CMake 3.27.2, and g++-9.

Unknown / fill as TODO if not available:

- GPU model
- CPU model
- RAM
- exact NVIDIA driver
- exact CUDA runtime reported during benchmark run
- compiler version used for each saved run
- container image hash/tag

Do not invent compute environment details. If the poster includes an experiment setup box and these details are not provided, use `TODO`.

## Reproduction Commands

Use these as provenance, not as commands to run during poster creation.

NTT / PolyMul core:

```bash
docker exec -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh
docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh
```

VOLE baseline sweeps:

```bash
docker exec -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh
docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh
```

VOLE sensitivity sweep:

```bash
docker exec -e M=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh
```

DPF online key-generation sweeps:

```bash
docker exec -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
docker exec -e CHUNK_SIZE=4096 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
docker exec -e CHUNK_SIZE=2048 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
```

## Exact Benchmark Data

Use exact numbers below. Do not invent, extrapolate, or round more aggressively unless needed for figure labels.

### Orca Profiling Table

| Model | Key files observed | Avg key read (ms) | Avg compute (ms) | Comm per iteration (B) |
| --- | --- | ---: | ---: | ---: |
| P-SecureML | P0 338M, P1 338M | 9.909 | 32.273 | 5,692,170.18 |
| P-LeNet | P0 4.0G, P1 4.0G | 109.727 | 107.727 | 65,572,810.18 |
| P-AlexNet | P0 3.8G, P1 3.8G | 104.818 | 121.727 | 113,913,098.18 |

Caption: Orca key-read time approaches or matches compute time for larger local training runs.

### CPU vs GPU Direct Comparison At n=8192

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| CPU (NFLLib) | 30 | 1 | pass | 57.2021 | 61.8469 | 180.594 | 180.594 | n/a |
| GPU (CUDA) | 30 | 64 | pass | 41.7984 | 45.8986 | 129.509 | 2.024 | 1 |

Saved direct speedups:

- Forward NTT speedup per polynomial: `87.59x`
- Full PolyMul speedup per polynomial: `89.24x`

### GPU NTT / PolyMul q=32 Sweep

| n | q req | q actual | batch | validate | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 32 | 30 | 64 | pass | 26.777 | 25.925 | 79.628 | 1.244 | 803741.42 | 52.67 |
| 16384 | 32 | 30 | 64 | pass | 51.862 | 50.483 | 155.992 | 2.437 | 410277.45 | 53.78 |
| 32768 | 32 | 30 | 64 | pass | 100.302 | 99.889 | 306.232 | 4.785 | 208991.88 | 54.79 |
| 65536 | 32 | 30 | 16 | pass | 14.006 | 13.266 | 42.260 | 2.641 | 378613.09 | 198.50 |
| 131072 | 32 | 30 | 16 | pass | 96.612 | 99.303 | 298.463 | 18.654 | 53607.98 | 56.21 |
| 262144 | 32 | 30 | 8 | pass | 110.216 | 112.384 | 337.991 | 42.249 | 23669.27 | 49.64 |
| 524288 | 32 | 30 | 4 | pass | 105.022 | 104.251 | 320.102 | 80.025 | 12496.02 | 52.41 |
| 1048576 | 32 | 30 | 2 | pass | 109.050 | 109.437 | 332.762 | 166.381 | 6010.30 | 50.42 |

### GPU NTT / PolyMul q=64 Sweep

| n | q req | q actual | batch | validate | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 64 | 62 | 64 | pass | 86.117 | 82.512 | 253.654 | 3.963 | 252312.20 | 33.07 |
| 16384 | 64 | 62 | 64 | pass | 103.020 | 100.905 | 311.081 | 4.861 | 205734.20 | 53.93 |
| 32768 | 64 | 62 | 64 | pass | 157.157 | 146.257 | 465.788 | 7.278 | 137401.56 | 72.04 |
| 65536 | 64 | 62 | 16 | pass | 47.803 | 40.438 | 140.357 | 8.772 | 113995.03 | 119.53 |
| 131072 | 64 | 62 | 16 | pass | 155.839 | 144.325 | 460.117 | 28.757 | 34773.76 | 72.93 |
| 262144 | 64 | 62 | 8 | pass | 153.630 | 143.456 | 456.098 | 57.012 | 17540.09 | 73.57 |
| 524288 | 64 | 62 | 4 | pass | 156.621 | 143.510 | 462.271 | 115.568 | 8652.93 | 72.59 |
| 1048576 | 64 | 62 | 2 | pass | 167.523 | 151.149 | 488.858 | 244.429 | 4091.17 | 68.64 |

### Derived CPU/GPU PolyMul Speedups

q=32 CPU-overlap points:

| n | CPU per-poly PolyMul (us) | GPU per-poly PolyMul (us) | Speedup |
| ---: | ---: | ---: | ---: |
| 8192 | 181.228 | 1.244 | 145.68x |
| 16384 | 390.535 | 2.437 | 160.25x |
| 32768 | 817.860 | 4.785 | 170.92x |

q=64 CPU-overlap points:

| n | CPU per-poly PolyMul (us) | GPU per-poly PolyMul (us) | Speedup |
| ---: | ---: | ---: | ---: |
| 8192 | 190.245 | 3.963 | 48.00x |
| 16384 | 410.371 | 4.861 | 84.42x |
| 32768 | 874.429 | 7.278 | 120.15x |
| 65536 | 1838.590 | 8.772 | 209.60x |
| 131072 | 3994.480 | 28.757 | 138.90x |
| 262144 | 10011.200 | 57.012 | 175.60x |
| 524288 | 23712.900 | 115.568 | 205.18x |
| 1048576 | 53685.800 | 244.429 | 219.64x |

Comparison note:

- The direct `87.59x` / `89.24x` n=8192 artifact and the derived q=32/q=64 sweep speedups are separate benchmark campaigns. They are complementary evidence, not one merged speedup range.

### DPF Sweep: Chunk Size 8192

| N | bin | chunk | validate | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 16 | 8192 | pass | 2.81 | 2.81 | 1.00x | 1.000x | 269.920 | 268.750 | 0.996x |
| 16384 | 16 | 8192 | pass | 5.63 | 2.81 | 2.00x | 1.000x | 378.900 | 523.330 | 1.381x |
| 32768 | 16 | 8192 | pass | 11.25 | 2.81 | 4.00x | 1.000x | 689.500 | 1057.270 | 1.533x |
| 65536 | 16 | 8192 | pass | 22.50 | 2.81 | 8.00x | 1.000x | 1235.280 | 2107.640 | 1.706x |
| 131072 | 16 | 8192 | pass | 45.00 | 2.81 | 16.00x | 1.000x | 2402.640 | 4180.980 | 1.740x |
| 262144 | 16 | 8192 | pass | 90.00 | 2.81 | 32.00x | 1.000x | 4666.900 | 8364.700 | 1.792x |
| 524288 | 16 | 8192 | pass | 180.00 | 2.81 | 64.00x | 1.000x | 9177.500 | 16723.200 | 1.822x |
| 1048576 | 16 | 8192 | pass | 360.00 | 2.81 | 128.00x | 1.000x | 18242.700 | 33458.700 | 1.834x |

### DPF Chunk-Size Tradeoff At N=1048576

| Chunk size | Peak pair key (MiB) | Peak reduction | Time overhead |
| ---: | ---: | ---: | ---: |
| 8192 | 2.81 | 128.00x | 1.834x |
| 4096 | 1.41 | 255.99x | 2.942x |
| 2048 | 0.70 | 511.97x | 4.975x |

### Ring-LPN VOLE q=32 Baseline Sweep

Configuration: requested q=32, actual q=30, `m=32`, `c=2`, noise weight `64`.

| n | validate | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8192 | pass | 191.485 | 5.984 | 167114.92 | 1002689.51 |
| 16384 | pass | 350.944 | 10.967 | 91182.64 | 547095.83 |
| 32768 | pass | 681.873 | 21.309 | 46929.56 | 281577.36 |
| 65536 | pass | 626.172 | 19.568 | 51104.17 | 306625.02 |
| 131072 | pass | 2966.810 | 92.713 | 10786.00 | 64715.97 |
| 262144 | pass | 8286.620 | 258.957 | 3861.65 | 23169.88 |
| 524288 | pass | 16016.100 | 500.503 | 1997.99 | 11987.94 |
| 1048576 | pass | 32144.700 | 1004.522 | 995.50 | 5972.99 |

### Ring-LPN VOLE q=64 Baseline Sweep

Configuration: requested q=64, actual q=62, `m=32`, `c=2`, noise weight `64`.

| n | validate | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8192 | pass | 549.802 | 17.181 | 58202.77 | 349216.63 |
| 16384 | pass | 702.911 | 21.966 | 45524.97 | 273149.80 |
| 32768 | pass | 1169.510 | 36.547 | 27361.89 | 164171.32 |
| 65536 | pass | 1735.660 | 54.239 | 18436.79 | 110620.74 |
| 131072 | pass | 6047.390 | 188.981 | 5291.54 | 31749.23 |
| 262144 | pass | 12604.900 | 393.903 | 2538.70 | 15232.17 |
| 524288 | pass | 25295.400 | 790.481 | 1265.05 | 7590.31 |
| 1048576 | pass | 50952.700 | 1592.272 | 628.03 | 3768.20 |

### Ring-LPN VOLE q=32 Sensitivity Sweep

Configuration: requested q=32, actual q=30, `m=64`, `c=2`, noise weight `64`.

| n | validate | Full expand mean (us) | Per-output expand (us) | Outputs/s |
| ---: | --- | ---: | ---: | ---: |
| 8192 | pass | 319.993 | 5.000 | 200004.38 |
| 16384 | pass | 670.203 | 10.472 | 95493.45 |
| 32768 | pass | 1421.340 | 22.208 | 45027.93 |
| 65536 | pass | 1400.850 | 21.888 | 45686.55 |
| 131072 | pass | 7709.040 | 120.454 | 8301.94 |
| 262144 | pass | 16712.000 | 261.125 | 3829.58 |
| 524288 | pass | 31860.800 | 497.825 | 2008.74 |
| 1048576 | pass | 64076.700 | 1001.198 | 998.80 |

### Figure 2 OLE GPU Artifact

Configuration: requested q=64, actual single 62-bit prime, `c=2`, `t=64`.

| Noise | n | SPFSS domain | validation | host validation | Key bytes MiB | Keygen us | OLE expand mean us |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| uniform | 8192 | 16384 | pass | pass | 8.63 | 4797.000 | 865253.000 |
| uniform | 16384 | 32768 | pass | skipped | 9.19 | 5296.000 | 1830210.000 |
| regular | 8192 | 256 | pass | pass | 5.27 | 40828.000 | 58462.500 |
| regular | 16384 | 512 | pass | skipped | 5.84 | 42331.000 | 67733.000 |

Interpretation:

- Direct OLE artifact stops at OLE.
- It validates `z_0 + z_1 == x_0 * x_1`.
- It is not an Orca Beaver-triple integration.

### Ring-Polynomial Linear OLE-To-Beaver Artifact

Configuration: rows=2, inner=2, cols=2, n=8192, c=2, t=8. Each ring product uses two Figure 2 OLE instances.

| Noise | SPFSS domain | validation | OLE instances | Key bytes MiB | Keygen us | Linear expand mean us |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| uniform | n/a | pass | 16 | 2.16 | 6594.000 | 222355.000 |
| regular | 2048 | pass | 16 | 1.78 | 82726.000 | 115447.000 |

Interpretation:

- Validates two-OLE-to-Beaver conversion over ring-polynomial matrix multiplication.
- Not yet Orca FC integration.

## Claims The Poster May Make

Use these claims:

- Orca profiling shows key-read time can approach GPU compute time in moderate local training runs.
- Chunked DPF online key generation reduces peak staged pair-key footprint by up to `128x` at chunk size `8192` with `1.834x` overhead.
- More aggressive chunking reaches `255.99x` and `511.97x` peak-footprint reduction at higher time overhead.
- The promoted GPU NTT/PolyMul backend validates requested q=32 and q=64 sweeps over n=8192 to 1048576.
- The direct n=8192 comparison reports `87.59x` forward-NTT and `89.24x` full-PolyMul per-polynomial speedup over NFLLib.
- The Ring-LPN VOLE prototype validates the coefficient-wise relation over n=8192 to 1048576 for requested q=32 and q=64.
- Figure 2 OLE and linear OLE-to-Beaver artifacts are validated standalone bridge experiments.
- These components are currently being integrated into Orca.

## Claims To Avoid Or Qualify

Do not claim:

- finished end-to-end Orca integration,
- trusted-dealer removal in Orca,
- q=128 / CRT support,
- paper-parameter Figure 2 results,
- CPU-vs-GPU speedup for the VOLE prototype itself,
- that chunking reduces total logical key bytes,
- that the linear OLE-to-Beaver artifact already emits Orca-compatible scalar Beaver triples.

Qualify:

- The submitted abstract says keys grow from gigabytes to tens of gigabytes. The saved local profile included here shows hundreds of MiB to about 4.0G per party. If challenged, cite the local measured table and treat "tens of gigabytes" as scaling/large-model motivation rather than this local profile's maximum observed point.
- Requested q=32/q=64 are implemented as actual q=30/q=62 single-prime paths.

## Required Poster Outputs

Ask the poster creation agent to produce:

1. poster title and author/affiliation placeholders,
2. submitted abstract text block,
3. methodology sections for Orca profiling, DPF chunking, GPU NTT/PolyMul, Ring-LPN VOLE, and bridge artifacts,
4. figures/tables from the benchmark data above,
5. experiment setup box with TODO placeholders for missing hardware details,
6. current integration status / limitations box,
7. provenance list of source files and result artifacts.

## Source Artifacts

Use these paths as the source-of-truth list:

- `GPU-MPC/orca_runner/logs/master.log`
- `GPU-MPC/ringlpn/results/abstract_benchmark_appendix.md`
- `GPU-MPC/ringlpn/results/submitted_abstract_support_audit.md`
- `GPU-MPC/ringlpn/results/ntt_gpu_q32.md`
- `GPU-MPC/ringlpn/results/ntt_gpu_q64.md`
- `GPU-MPC/ringlpn/results/cpu_gpu_8192_32_batch64.md`
- `GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md`
- `GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk4096.md`
- `GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk2048.md`
- `GPU-MPC/ringlpn/results/vole_gpu_q32_m32_c2_w64.md`
- `GPU-MPC/ringlpn/results/vole_gpu_q64_m32_c2_w64.md`
- `GPU-MPC/ringlpn/results/vole_gpu_q32_m64_c2_w64.md`
- `GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md`
- `GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t64.md`
- `GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md`
- `GPU-MPC/ringlpn/results/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.md`
- `GPU-MPC/ringlpn/results/ringlpn_status_report.md`
- `GPU-MPC/ringlpn/results/ole_gpu_handoff.md`
- `GPU-MPC/ringlpn/results/linear_ole_handoff.md`

