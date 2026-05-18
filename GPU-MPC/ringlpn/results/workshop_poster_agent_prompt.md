# Workshop Poster Agent Prompt

Historical snapshot: this prompt predates the current q128 NTT/VOLE and Orca FC v1 transition plan. For current claims, use `ringlpn_status_report.md` and `orca_ringlpn_linear_integration_plan.md`.

Use this prompt to create a research-workshop poster for the GPU-MPC / Ring-LPN project.

## Main Prompt

Create a polished, publication-ready research workshop poster for the project:

**Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing**

Audience: systems, GPU, cryptography, and privacy-preserving machine-learning researchers. The poster should be technically precise, visually clean, and suitable for a workshop hallway discussion. Default to a 48 in x 36 in landscape poster unless the venue gives another size.

The poster should tell this story:

1. GPU-accelerated Function Secret Sharing (FSS), as used in the GPU-MPC / Orca secure computation stack, enables fast privacy-preserving ML but creates large offline key material and online key movement pressure.
2. Profiling shows that key-read time is already comparable to compute time for larger Orca models, even though the system already uses direct I/O and overlaps key reading with computation.
3. We explore two complementary directions:
   - chunked online DPF key generation to reduce peak staged key footprint,
   - GPU Ring-LPN polynomial and VOLE building blocks to accelerate the online phase.
4. The current work is a set of validated standalone systems prototypes and benchmarks, not a full end-to-end Orca integration yet.

## Poster Layout

Use a 3-column or 4-column research-poster layout with these sections:

1. **Problem**
   - Large FSS key material stresses storage, memory, and online data movement.
   - Orca already uses GPU acceleration, direct I/O, and overlapping, so the remaining bottleneck is structural, not just a naive implementation issue.

2. **Approach**
   - Show a pipeline diagram:
     `Orca/FSS workload -> profiling bottleneck -> chunked DPF key generation -> Ring-LPN VOLE / GPU PolyMul backend -> future Orca integration`
   - Emphasize that DPF chunking targets peak staged key footprint.
   - Emphasize that Ring-LPN VOLE builds on the promoted GPU polynomial backend.

3. **Benchmarks**
   Include 3 to 5 compact figures or tables:
   - Orca profiling table or bar chart: key-read time vs compute time.
   - DPF chunking tradeoff: peak key footprint vs N, plus time overhead.
   - GPU NTT/PolyMul performance table or plot for requested q=32 and q=64.
   - Ring-LPN VOLE expansion latency/throughput table or plot.
   - Optional small "bridge artifact" table for Figure 2 OLE and ring-polynomial OLE-to-Beaver, clearly marked as standalone and not Orca-integrated.

4. **Contributions**
   Use concise bullet points:
   - Identified key movement and memory pressure in a GPU-FSS / Orca workflow.
   - Implemented and measured standalone chunked DPF online key generation.
   - Promoted a validated cheddar-derived GPU NTT/PolyMul backend for requested q=32 and q=64.
   - Implemented and validated a standalone GPU Ring-LPN VOLE expansion prototype.
   - Implemented early standalone Figure 2 SPFSS/OLE and ring-polynomial OLE-to-Beaver artifacts, with Orca integration left as ongoing work.

5. **Current Boundary / Future Work**
   Make this visible but not apologetic:
   - Current q=32 and q=64 GPU paths are single-prime paths: requested q=32 maps to actual q=30, requested q=64 maps to actual q=62.
   - q=128 / CRT is not implemented yet.
   - DPF chunked key generation is a standalone systems benchmark, not integrated end-to-end into Orca yet.
   - VOLE uses synthetic MPVOLE-consistent inputs; it is not a full SPFSS-backed OLE-R-LPN pipeline yet.
   - Ring-polynomial OLE-to-Beaver is validated, but scalar packing and `Z_p -> Z_{2^bw}` share conversion are still needed before Orca `gpuMatmulBeaver` integration.

## Exact Data To Use

Do not invent or extrapolate numbers. Use the numbers below exactly, with units.

### Orca Profiling Motivation

Use this as a small table or grouped bar chart:

| Model | Key read | Compute |
| --- | ---: | ---: |
| P-SecureML | 9.91 ms | 32.27 ms |
| P-LeNet | 109.73 ms | 107.73 ms |
| P-AlexNet | 104.82 ms | 121.73 ms |

Caption idea: "For larger Orca models, key movement approaches compute time even with direct I/O and overlap infrastructure."

Implementation context:

- `utils/gpu_file_utils.cpp` uses `O_DIRECT | O_LARGEFILE`.
- Key buffers are 4096-byte aligned.
- `experiments/orca/orca_evaluator.cu` overlaps key reading and computation.
- `nn/orca/fc_layer.cu` still performs repeated runtime `moveToGPU()` calls.

### GPU NTT / PolyMul Core

All listed sweep points passed validation.

Requested q=32 is realized with actual q=30 using one prime:

| n | batch | full PolyMul mean | per-poly PolyMul | PolyMul polys/s |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 64 | 79.628 us | 1.244 us | 803741.42 |
| 32768 | 64 | 306.232 us | 4.785 us | 208991.88 |
| 1048576 | 2 | 332.762 us | 166.381 us | 6010.30 |

Requested q=64 is realized with actual q=62 using one prime:

| n | batch | full PolyMul mean | per-poly PolyMul | PolyMul polys/s |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 64 | 253.654 us | 3.963 us | 252312.20 |
| 32768 | 64 | 465.788 us | 7.278 us | 137401.56 |
| 1048576 | 2 | 488.858 us | 244.429 us | 4091.17 |

Safe acceleration context:

- Saved direct n=8192 comparison: 87.59x forward-NTT and 89.24x full-PolyMul per-polynomial speedup over NFLLib.
- Sweep-derived requested q=32 CPU-overlap points show about 145.68x, 160.25x, and 170.92x per-polynomial PolyMul speedup over CPU at n=8192, 16384, and 32768.
- Sweep-derived requested q=64 overlap points show about 48.00x to 219.64x per-polynomial PolyMul speedup over CPU across n=8192 to 1048576.
- The promoted main q=32 path is faster than the preserved legacy CUDA baseline across the validated sweep, with the largest observed per-polynomial gain at n=65536: 2.641 us vs 15.679 us, a 5.94x legacy/main speedup.

### Ring-LPN VOLE Prototype

All saved VOLE sweep points passed coefficient-wise validation. The benchmark uses synthetic MPVOLE-consistent inputs and measures the algebraic expansion layer. It does not measure a full SPFSS-backed pipeline.

Baseline configuration: m=32, c=2, noise weight=64.

Requested q=32, actual q=30:

| n | full expand mean | per-output expand | outputs/s |
| ---: | ---: | ---: | ---: |
| 8192 | 191.485 us | 5.984 us | 167114.92 |
| 32768 | 681.873 us | 21.309 us | 46929.56 |
| 1048576 | 32144.700 us | 1004.522 us | 995.50 |

Requested q=64, actual q=62:

| n | full expand mean | per-output expand | outputs/s |
| ---: | ---: | ---: | ---: |
| 8192 | 549.802 us | 17.181 us | 58202.77 |
| 32768 | 1169.510 us | 36.547 us | 27361.89 |
| 1048576 | 50952.700 us | 1592.272 us | 628.03 |

Optional sensitivity point:

- q=32, m=64, c=2, noise weight=64:
  - n=8192: full expansion 319.993 us, per-output 5.000 us, throughput 200004.38 outputs/s.
  - n=1048576: full expansion 64076.700 us, per-output 1001.198 us, throughput 998.80 outputs/s.

### DPF Online Key Generation

All saved DPF sweep points passed validation. This is a standalone eval-all DPF key generation benchmark, not an end-to-end FSS evaluation benchmark.

Baseline chunk size: 8192, bin=16.

| N | full pair key | partial peak pair key | peak reduction | full keygen mean | partial pipeline mean | time overhead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 2.81 MiB | 2.81 MiB | 1.00x | 269.920 us | 268.750 us | 0.996x |
| 16384 | 5.63 MiB | 2.81 MiB | 2.00x | 378.900 us | 523.330 us | 1.381x |
| 1048576 | 360.00 MiB | 2.81 MiB | 128.00x | 18242.700 us | 33458.700 us | 1.834x |

Optional tradeoff curve at N=1048576:

| chunk size | partial peak pair key | peak reduction | time overhead |
| ---: | ---: | ---: | ---: |
| 8192 | 2.81 MiB | 128.00x | 1.834x |
| 4096 | 1.41 MiB | about 255.99x | 2.942x |
| 2048 | 0.70 MiB | about 511.97x | 4.975x |

Main caption idea: "Chunked online generation trades extra generation time for a tunable reduction in peak staged key footprint; total logical key bytes remain essentially unchanged."

### Optional Figure 2 OLE / Linear-Layer Bridge Data

Use this only as a small "building-block progress" box. Do not make it the headline unless the poster is specifically about OLE integration.

GPU Figure 2 OLE, requested q=64 / actual q=62, c=2, t=64:

| noise | n | SPFSS domain | key bytes | keygen | OLE expand mean | validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| uniform | 8192 | 16384 | 8.63 MiB | 4797.000 us | 865253.000 us | pass |
| regular | 8192 | 256 | 5.27 MiB | 40828.000 us | 58462.500 us | pass |
| regular | 16384 | 512 | 5.84 MiB | 42331.000 us | 67733.000 us | pass |

Ring-polynomial linear-layer OLE-to-Beaver smoke, rows=2, inner=2, cols=2, n=8192, c=2, t=8:

| noise | OLE instances | key bytes | keygen | linear expand mean | validation |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform | 16 | 2.16 MiB | 6594.000 us | 222355.000 us | pass |
| regular | 16 | 1.78 MiB | 82726.000 us | 115447.000 us | pass |

Caption boundary: "This validates OLE-to-Beaver over ring-polynomial matrix entries. It is not yet Orca FC integration."

## Claims To Use

Use these claims verbatim or nearly verbatim:

- "We implemented validated standalone GPU building blocks for memory-efficient FSS key generation and Ring-LPN online-phase acceleration."
- "Chunked DPF online key generation caps peak staged pair-key footprint at 2.81 MiB for chunk size 8192 while the one-shot pair key grows to 360.00 MiB at N=1048576."
- "The promoted GPU polynomial backend validates requested q=32 and q=64 sweeps over n=8192 to 1048576."
- "The standalone Ring-LPN VOLE prototype validates the coefficient-wise relation across the full tested degree range for requested q=32 and q=64."
- "Full Orca/SPFSS-backed integration remains ongoing work."

## Claims To Avoid

Do not claim:

- end-to-end Orca integration,
- full SPFSS-backed OLE-R-LPN or degree-1 correlation pipeline completion,
- q=128 or CRT support,
- paper-parameter Figure 2 results,
- CPU-vs-GPU speedup for the VOLE prototype itself,
- trusted-dealer removal for Orca,
- that chunked DPF reduces total generated bytes,
- that the current OLE-to-Beaver artifact produces Orca-compatible scalar Beaver triples.

## Visual Style

Make the poster feel like a serious systems/security artifact:

- restrained palette, high contrast, no decorative clutter;
- large readable title;
- one central pipeline diagram;
- compact tables with highlighted best or headline values;
- clear captions under every plot/table;
- a small QR-code placeholder for code/artifacts;
- author and affiliation placeholders if not provided;
- include artifact paths in tiny footer text if space allows.

Suggested color semantics:

- blue or graphite for existing Orca/GPU-MPC system,
- green for measured memory reduction,
- amber for ongoing/future integration,
- purple or teal for Ring-LPN/GPU polynomial acceleration.

## Output Requirements

Produce:

1. A complete poster draft with title, sections, captions, and final text.
2. A short 150-250 word poster abstract.
3. A list of figures/tables used and the exact source data for each.
4. A visible "Current Boundary" or "Limitations / Next Steps" box.
5. No invented benchmarks, citations, or implementation claims.

## Required Placeholders To Fill Before Submission

If any of the following are not provided to the poster agent, leave a visible `TODO` placeholder rather than inventing them:

- author names and affiliations,
- workshop name, date, and required poster dimensions,
- GPU model, CPU model, RAM, CUDA version, compiler version, and container/image details for the benchmark environment,
- exact code/artifact URL or QR-code target,
- citation list for Orca, FSS/DPF, Ring-LPN/PCG, and cheddar-fhe/NFLlib if the poster includes a references box,
- contact email.

## Source Artifact Paths

Use these paths as provenance when checking data:

- `GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md`
- `GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md`
- `GPU-MPC/ringlpn/results/ntt_gpu_q32.md`
- `GPU-MPC/ringlpn/results/ntt_gpu_q64.md`
- `GPU-MPC/ringlpn/results/vole_gpu_q32_m32_c2_w64.md`
- `GPU-MPC/ringlpn/results/vole_gpu_q64_m32_c2_w64.md`
- `GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md`
- `GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md`
- `GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t64.md`
- `GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md`
- `GPU-MPC/ringlpn/results/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.md`
- `GPU-MPC/ringlpn/results/abstract_benchmark_appendix.md`

## Final Poster Takeaway

The viewer should leave with one clear message:

**GPU-FSS performance is increasingly constrained by key material movement and staged key footprint; chunked DPF generation and GPU Ring-LPN building blocks provide validated standalone evidence for a path toward more memory-efficient online execution, with end-to-end Orca integration as the next step.**
