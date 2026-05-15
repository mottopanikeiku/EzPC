# Paste-Ready Workshop Poster Content

Poster title:

**Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing**

Purpose: paste this content into the IIT-style three-column poster template or pass it to a plotting/poster-generation agent. This is a content artifact, not a new benchmark result. It preserves the current evidence boundary: the DPF, Ring-LPN VOLE, Figure 2 OLE, and OLE-to-Beaver results are validated standalone building blocks; full Orca integration is ongoing. For NTT/PolyMul, use the promoted GPU backend as the base GPU implementation and compare it only against the NFLLib CPU baseline; do not include a third GPU implementation or a three-way GPU comparison.

## Header Block

Title:

**Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing**

Authors:

`Author A.*, Author B., ..., Author Z.`

Affiliation:

`Department of TODO, Illinois Institute of Technology`

Presenter marker:

`*presenting author`

Contact / QR:

`TODO email / TODO artifact QR`

Workshop / venue:

`TODO workshop name, date, and poster dimensions`

## Abstract

Privacy-preserving machine learning protocols split computation between function secret sharing for non-linear operations and additive secret sharing for linear operations. Both rely on offline/online decompositions to reduce online latency, but this shifts cost to generating, storing, and moving correlated randomness. Our profiling of Orca, a GPU-accelerated FSS-based PPML system in GPU-MPC, shows that precomputed keys reach several gigabytes per party and that key-read time can match GPU computation for moderate models.

We develop GPU building blocks toward a unified acceleration framework for this bottleneck. For FSS-based non-linear evaluation, standalone chunked DPF online key generation reduces peak staged pair-key footprint by up to 128x with under 2x time overhead. For secret-sharing-based linear evaluation, we accelerate Ring-LPN PCG components. Our GPU NTT/PolyMul backend achieves 89.24x per-polynomial full-PolyMul speedup over NFLLib at n=8192. Built on this backend, a standalone GPU Ring-LPN VOLE prototype validates correctness across degrees from 8192 to 1048576 for requested q=32 and q=64. Full Orca integration is ongoing.

## Statement Of The Problem

- PPML preprocessing creates large FSS keys and correlation material.
- Orca already uses GPU acceleration, direct I/O, 4096-byte-aligned key buffers, and read/compute overlap, yet key movement remains comparable to compute for larger local runs.
- The research question: can we reduce peak staged key footprint and accelerate GPU-side PCG expansion without claiming completed end-to-end Orca replacement?

Suggested one-sentence takeaway:

**GPU-FSS performance is increasingly constrained by key material movement and staged key footprint; chunked DPF generation and GPU Ring-LPN building blocks provide validated standalone evidence for a path toward more memory-efficient online execution.**

## Background And Method Of Approach

Pipeline diagram text:

```text
Orca/FSS workload
  -> profiling bottleneck
  -> chunked DPF online key generation
  -> GPU NTT/PolyMul
  -> Ring-LPN VOLE
  -> OLE-to-Beaver bridge
  -> Orca integration
```

Method bullets:

- Orca profiling measures key file sizes, average key-read time, average compute time, and communication per iteration from `GPU-MPC/orca_runner/logs/master.log`.
- DPF online key generation compares one-shot full-pair eval-all key generation against fixed-size chunked generation. Chunked mode materializes only the current chunk.
- GPU NTT/PolyMul benchmarks batched polynomial multiplication as `NTT(a)`, `NTT(b)`, pointwise multiply, and `INTT`, then reports batch and per-polynomial latency.
- Ring-LPN VOLE reuses the promoted GPU PolyMul backend, synthesizes MPVOLE-consistent inputs under `synthetic_mpvole`, and validates `z = y + x * Delta`.
- Figure 2 OLE validates `z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N+1)`.
- The linear bridge uses two Figure 2 OLE instances per ring product to validate OLE-to-Beaver conversion for ring-polynomial matrix multiplication.

## Experimental Setup And Reproduction Context

Use this as a visible setup box or small methods sidebar. Hardware and container details below were queried from the current host and running `orca-dev` container on 2026-05-11. The saved benchmark artifacts did not independently log every environment field per run, so treat these as observed current runtime provenance unless the runs are re-collected with embedded environment capture.

Known runtime setup:

| Item | Value |
| --- | --- |
| Project | `GPU-MPC` |
| Main system | Orca, GPU-accelerated FSS-based PPML |
| Benchmark area | `GPU-MPC/ringlpn` |
| Container | `orca-dev` |
| Host GPU-MPC root | `/home/fatih/EzPC/GPU-MPC` |
| Container GPU-MPC root | `/home` |
| Container Ring-LPN workdir | `/home/ringlpn` |
| Current container OS | Ubuntu 22.04.4 LTS |
| Current CUDA toolkit in container | CUDA compilation tools 12.3, V12.3.107 |
| Driver-reported CUDA version | 12.6 |
| Current compiler in container | gcc/g++ 9.5.0 |
| Python in container | Python 3.10.12 |
| README-tested environment | Ubuntu 20.04, CUDA 11.7, CMake 3.27.2, g++-9 |

Hardware/software details:

| Item | Value |
| --- | --- |
| GPU model | 4x NVIDIA RTX 5000 Ada Generation |
| GPU memory | 32760 MiB per GPU |
| GPU compute capability | 8.9 |
| CPU model | Intel Xeon w5-3435X, 16 cores / 32 threads, 1 socket |
| RAM | 109 GiB system memory, 9 GiB swap |
| NVIDIA driver | 560.35.03 |
| CUDA runtime/toolkit observed | driver-reported CUDA 12.6; container nvcc CUDA 12.3.107 |
| Compiler observed | gcc/g++ 9.5.0; nvcc 12.3.107 |
| Container image tag | `fatih` |
| Container image ID | `sha256:8734209bcc3b2f07fd99f236ba499a4fa7d0e8cda2ee109ddf2ebc9ea6d17b0c` |
| Container ID | `7706f2441465100149beb1c8455bffae73ce00f48efcb34efa1fc645ea9886f8` |

Benchmark command provenance:

```bash
# NTT / PolyMul core
docker exec -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh
docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh

# Ring-LPN VOLE baseline sweeps
docker exec -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh
docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh

# Ring-LPN VOLE sensitivity sweep
docker exec -e M=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh

# DPF online key-generation sweeps
docker exec -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
docker exec -e CHUNK_SIZE=4096 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
docker exec -e CHUNK_SIZE=2048 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
```

Measurement notes:

- Orca profile rows report local loopback training measurements from `GPU-MPC/orca_runner/logs/master.log`.
- GPU experiments were run inside `orca-dev`; host paths and container paths differ because only `GPU-MPC` is mounted into the container as `/home`.
- NTT/PolyMul timings report CUDA benchmark means; per-polynomial latency divides the batched full-polynomial multiply by batch size.
- DPF timings compare full pair-key materialization against chunked partial generation for eval-all keys.
- VOLE timings measure standalone algebraic expansion over synthetic MPVOLE-consistent inputs, not a full end-to-end SPFSS-backed pipeline.

## Results Panel 1: Orca Profiling Motivation

Use as a grouped bar chart or compact table.

| Model | Key files observed | Avg key read (ms) | Avg compute (ms) | Comm per iteration (B) |
| --- | --- | ---: | ---: | ---: |
| P-SecureML | P0 338M, P1 338M | 9.909 | 32.273 | 5,692,170.18 |
| P-LeNet | P0 4.0G, P1 4.0G | 109.727 | 107.727 | 65,572,810.18 |
| P-AlexNet | P0 3.8G, P1 3.8G | 104.818 | 121.727 | 113,913,098.18 |

Caption:

**Orca key-read time approaches or matches compute time for larger local training runs.** This is measured despite existing direct I/O and overlap infrastructure, so key movement is a structural bottleneck rather than only a missing optimization.

## Results Panel 2: DPF Online Key Generation

Headline:

**At N=1048576 and chunk size 8192, chunked DPF online generation keeps the peak staged pair-key footprint at 2.81 MiB while one-shot full-pair generation reaches 360.00 MiB.**

Baseline chunk-size table:

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

Chunk-size tradeoff at N=1048576:

| Chunk size | Peak pair key (MiB) | Peak reduction | Time overhead |
| ---: | ---: | ---: | ---: |
| 8192 | 2.81 | 128.00x | 1.834x |
| 4096 | 1.41 | 255.99x | 2.942x |
| 2048 | 0.70 | 511.97x | 4.975x |

Caption:

**Chunked online generation trades extra generation time for a tunable reduction in peak staged key footprint.** It reduces peak staged footprint, not total logical key material.

## Results Panel 3: GPU NTT / PolyMul Core

Headline:

**Direct n=8192 comparison: 87.59x forward-NTT speedup and 89.24x full-PolyMul per-polynomial speedup over NFLLib.**

Direct CPU/GPU comparison at n=8192:

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| CPU (NFLLib) | 30 | 1 | pass | 57.2021 | 61.8469 | 180.594 | 180.594 | n/a |
| GPU (CUDA) | 30 | 64 | pass | 41.7984 | 45.8986 | 129.509 | 2.024 | 1 |

Selected promoted GPU q=32 sweep rows:

| n | q req | q actual | batch | validate | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 32 | 30 | 64 | pass | 26.777 | 25.925 | 79.628 | 1.244 | 803741.42 | 52.67 |
| 32768 | 32 | 30 | 64 | pass | 100.302 | 99.889 | 306.232 | 4.785 | 208991.88 | 54.79 |
| 1048576 | 32 | 30 | 2 | pass | 109.050 | 109.437 | 332.762 | 166.381 | 6010.30 | 50.42 |

Selected promoted GPU q=64 sweep rows:

| n | q req | q actual | batch | validate | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 64 | 62 | 64 | pass | 86.117 | 82.512 | 253.654 | 3.963 | 252312.20 | 33.07 |
| 32768 | 64 | 62 | 64 | pass | 157.157 | 146.257 | 465.788 | 7.278 | 137401.56 | 72.04 |
| 1048576 | 64 | 62 | 2 | pass | 167.523 | 151.149 | 488.858 | 244.429 | 4091.17 | 68.64 |

Caption:

**The promoted GPU backend validates requested q=32 and q=64 sweeps over n=8192 to 1048576.** Requested q=32 and q=64 are implemented as actual q=30 and q=62 single-prime paths.

Plot instruction:

Use a two-way NTT/PolyMul story: NFLLib CPU baseline versus promoted GPU backend. Do not include a third GPU implementation in the poster figures, captions, or provenance. The main headline is the direct n=8192 artifact: 87.59x forward NTT and 89.24x full PolyMul per-polynomial speedup.

## Results Panel 4: Ring-LPN VOLE And Bridge Artifacts

Headline:

**The standalone GPU Ring-LPN VOLE prototype validates coefficient-wise correctness from n=8192 through n=1048576 for requested q=32 and q=64.**

Selected Ring-LPN VOLE rows, m=32, c=2, noise weight 64:

| q req | q actual | n | validate | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32 | 30 | 8192 | pass | 191.485 | 5.984 | 167114.92 | 1002689.51 |
| 32 | 30 | 1048576 | pass | 32144.700 | 1004.522 | 995.50 | 5972.99 |
| 64 | 62 | 8192 | pass | 549.802 | 17.181 | 58202.77 | 349216.63 |
| 64 | 62 | 1048576 | pass | 50952.700 | 1592.272 | 628.03 | 3768.20 |

Bridge mini-table:

| Artifact | Configuration | validation | Key bytes MiB | Keygen us | Expand mean us |
| --- | --- | --- | ---: | ---: | ---: |
| Figure 2 OLE, regular noise | q req=64, q actual=62, n=8192, c=2, t=64, SPFSS domain=256 | pass | 5.27 | 40828.000 | 58462.500 |
| Linear OLE-to-Beaver, regular smoke | rows=2, inner=2, cols=2, n=8192, c=2, t=8, 16 OLE instances | pass | 1.78 | 82726.000 | 115447.000 |

Caption:

**VOLE measures the algebraic expansion layer, not a full SPFSS-backed OLE-R-LPN pipeline.** The bridge artifacts validate Figure 2 OLE and two-OLE-to-Beaver conversion over ring-polynomial matrix entries, but they are not yet Orca fully connected layer integration.

## Plot-Ready Benchmark Appendix For External Graphing Agent

Use this section as the data source for richer plots. It can be omitted from the final visible poster if space is tight, but it is meant to give the outside agent enough material to make real graphs rather than dummy visuals.

Recommended plot set:

- Orca bottleneck plot: grouped bars for key-read time and compute time by model; annotate key-file footprint per party.
- DPF memory-time tradeoff: line plot with N on a log2 x-axis, full pair key footprint and partial peak pair key footprint on the y-axis; overlay or separate panel for time overhead.
- DPF chunk-size tradeoff: at N=1048576, scatter or bar plot of peak reduction versus time overhead for chunk sizes 8192, 4096, and 2048.
- NTT/PolyMul performance: plot per-poly PolyMul latency versus n for q req=32 and q req=64. A companion throughput plot can show PolyMul polys/s. Use promoted GPU backend only.
- CPU/GPU speedup plot: use derived CPU/GPU speedups as a small bar chart, separated by q=32 and q=64. Label this as sweep-derived, and keep it separate from the direct n=8192 87.59x/89.24x artifact.
- VOLE expansion plot: plot full expand mean or per-output expand versus n for q req=32 and q req=64. A companion plot can show outputs/s.
- Bridge artifact plot: use compact tables or two small bars comparing uniform versus regular Figure 2 OLE and linear OLE-to-Beaver smoke results. Keep this visually smaller than DPF and NTT.

### Full GPU NTT / PolyMul q=32 Sweep

Requested q=32 is implemented as actual q=30 using one prime. All rows pass validation.

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

### Full GPU NTT / PolyMul q=64 Sweep

Requested q=64 is implemented as actual q=62 using one prime. All rows pass validation.

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

These are sweep-derived CPU/GPU speedups, not the same artifact as the direct n=8192 comparison. Use them as complementary evidence.

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

### Full Ring-LPN VOLE q=32 Sweep

Configuration: requested q=32, actual q=30, m=32, c=2, noise weight 64. All rows pass validation.

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

### Full Ring-LPN VOLE q=64 Sweep

Configuration: requested q=64, actual q=62, m=32, c=2, noise weight 64. All rows pass validation.

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

Configuration: requested q=32, actual q=30, m=64, c=2, noise weight 64. All rows pass validation.

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

Configuration: requested q=64, actual single 62-bit prime, c=2, t=64.

| Noise | n | SPFSS domain | validation | host validation | Key bytes MiB | Keygen us | OLE expand mean us |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| uniform | 8192 | 16384 | pass | pass | 8.63 | 4797.000 | 865253.000 |
| uniform | 16384 | 32768 | pass | skipped | 9.19 | 5296.000 | 1830210.000 |
| regular | 8192 | 256 | pass | pass | 5.27 | 40828.000 | 58462.500 |
| regular | 16384 | 512 | pass | skipped | 5.84 | 42331.000 | 67733.000 |

Interpretation: direct OLE artifact stops at OLE; it validates `z_0 + z_1 == x_0 * x_1`; it is not an Orca Beaver-triple integration.

### Ring-Polynomial Linear OLE-To-Beaver Artifact

Configuration: rows=2, inner=2, cols=2, n=8192, c=2, t=8. Each ring product uses two Figure 2 OLE instances.

| Noise | SPFSS domain | validation | OLE instances | Key bytes MiB | Keygen us | Linear expand mean us |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| uniform | n/a | pass | 16 | 2.16 | 6594.000 | 222355.000 |
| regular | 2048 | pass | 16 | 1.78 | 82726.000 | 115447.000 |

Interpretation: validates two-OLE-to-Beaver conversion over ring-polynomial matrix multiplication; not yet Orca FC integration.

## Conclusions And Impact

- Orca profiling shows key movement is a structural bottleneck, not merely missing I/O optimization.
- Chunked DPF gives a tunable memory-time tradeoff and reduces peak staged footprint, not total logical key bytes.
- The promoted GPU polynomial backend validates q=32/q=64 single-prime sweeps over n=8192 to 1048576.
- Standalone VOLE, Figure 2 OLE, and OLE-to-Beaver artifacts validate the next integration building blocks.
- Current status: building blocks are validated; end-to-end Orca/SPFSS-backed integration remains ongoing.

Suggested final callout:

**Validated standalone building blocks now cover the memory-footprint experiment, the polynomial acceleration core, the VOLE expansion layer, and early linear-layer bridge artifacts. The next step is Orca-compatible integration.**

## Experiment Setup Box

Known:

- Container: `orca-dev`
- Host GPU-MPC root: `/home/fatih/EzPC/GPU-MPC`
- Container GPU-MPC root: `/home`
- Container Ring-LPN workdir: `/home/ringlpn`
- Current host GPUs: 4x NVIDIA RTX 5000 Ada Generation, 32760 MiB each, compute capability 8.9
- CPU/RAM: Intel Xeon w5-3435X, 16 cores / 32 threads, 109 GiB RAM
- NVIDIA driver: 560.35.03; driver-reported CUDA version 12.6
- Current `orca-dev` container OS/toolchain: Ubuntu 22.04.4 LTS, CUDA toolkit 12.3.107, gcc/g++ 9.5.0, Python 3.10.12
- Container image/tag: `fatih`, image ID `sha256:8734209bcc3b2f07fd99f236ba499a4fa7d0e8cda2ee109ddf2ebc9ea6d17b0c`
- Container ID: `7706f2441465100149beb1c8455bffae73ce00f48efcb34efa1fc645ea9886f8`
- Note: these setup fields were queried from the current running host/container on 2026-05-11; saved benchmark result files did not embed a separate environment snapshot for every run.

## Current Boundary / Limitations

- Requested q=32/q=64 map to actual q=30/q=62 single-prime paths.
- q=128 / CRT is not implemented.
- DPF chunking is a standalone key-generation systems benchmark, not an end-to-end FSS evaluation benchmark.
- VOLE uses synthetic MPVOLE-consistent inputs; do not claim CPU-vs-GPU speedup for VOLE itself.
- Figure 2 OLE and linear OLE-to-Beaver artifacts are standalone bridge experiments.
- OLE-to-Beaver does not yet emit Orca-compatible scalar Beaver triples.
- Missing before Orca `gpuMatmulBeaver` integration: scalar packing, `Z_p -> Z_{2^bw}` share conversion, and an Orca-compatible triple writer.
- Do not claim trusted-dealer removal in Orca or paper-parameter Figure 2 results.

## Footer Provenance

Source artifacts:

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

Code provenance:

- `GPU-MPC/tests/fss/dpf_online_keygen_bench.cu`
- `GPU-MPC/scripts/run_dpf_online_keygen_sweep.py`
- Primary GPU NTT/PolyMul source under `GPU-MPC/ringlpn/src/`
- `GPU-MPC/ringlpn/src/bench_ntt.cpp`
- `GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu`
- `GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh`
- `GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu`
- `GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu`
- `GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu`

## Template Placement Map

Use this mapping for the provided IIT-style template:

- Top red banner: title, authors, affiliation, presenter marker, IIT logo area.
- Left column, first section: Abstract.
- Left column, second section: Statement Of The Problem.
- Middle column: Background And Method Of Approach, plus pipeline diagram.
- Middle/lower sidebar: Experimental Setup And Reproduction Context.
- Right column, top: Results Panel 1 and Results Panel 2.
- Right column, middle: Results Panel 3 and Results Panel 4.
- Right column, lower red box: Conclusions And Impact.
- Bottom strip or lower-middle small boxes: Current Boundary / Limitations, Footer Provenance, TODO contact/QR.

## Pre-Submission Checklist

- Every table value is copied from the saved prompt/source artifact set.
- The abstract says "several gigabytes per party" rather than claiming the local profile itself shows tens of gigabytes.
- The poster does not claim finished end-to-end Orca integration.
- The poster does not claim trusted-dealer removal in Orca.
- The poster does not claim q=128 / CRT support.
- The poster uses only the promoted GPU NTT/PolyMul backend for GPU results; it does not include a third GPU implementation or three-way NTT comparison.
- The poster does not claim CPU-vs-GPU speedup for VOLE itself.
- The poster does not imply chunking reduces total logical key bytes.
- The poster does not claim the linear bridge already emits Orca-compatible scalar Beaver triples.
- Missing venue, author, contact, QR, and citation details remain visible as `TODO`; hardware and current container provenance have been filled from the current host/container query.
