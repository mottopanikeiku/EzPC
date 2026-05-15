# GPU-FSS / Ring-LPN Implementation Status Report

Date: 2026-05-11

Project: `GPU-MPC`

Poster title: **Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing**

## Executive Summary

This work has implemented and validated a set of standalone building blocks toward memory-efficient GPU-FSS preprocessing in `GPU-MPC`. The current evidence supports four main claims:

1. Orca profiling shows key movement is already a substantial cost: for P-LeNet, average key-read time is 109.727 ms versus 107.727 ms compute time.
2. Chunked DPF online key generation reduces peak staged pair-key footprint by up to 128.00x at chunk size 8192 with 1.834x time overhead for N=1048576.
3. The promoted GPU NTT/PolyMul backend validates requested q=32 and q=64 single-prime sweeps over n=8192 to 1048576 and gives 89.24x per-polynomial full-PolyMul speedup over NFLLib in the direct n=8192 comparison.
4. Standalone Ring-LPN VOLE, Figure 2 OLE, and ring-polynomial OLE-to-Beaver bridge artifacts validate the algebraic relations needed for future PCG-based linear-layer preprocessing.

The current work is not yet an end-to-end Orca integration. It does not remove the trusted dealer in Orca, does not implement q=128/CRT, and does not yet emit Orca-compatible scalar Beaver triples for `gpuMatmulBeaver`.

## Observed Experimental Environment

These fields were queried from the current host and running `orca-dev` container on 2026-05-11. The saved benchmark artifacts did not embed a separate environment snapshot for every run, so this should be treated as observed current runtime provenance unless the experiments are re-run with explicit environment capture.

| Item | Value |
| --- | --- |
| Host GPU-MPC root | `/home/fatih/EzPC/GPU-MPC` |
| Container GPU-MPC root | `/home` |
| Container Ring-LPN workdir | `/home/ringlpn` |
| Container | `orca-dev` |
| Container image tag | `fatih` |
| Container image ID | `sha256:8734209bcc3b2f07fd99f236ba499a4fa7d0e8cda2ee109ddf2ebc9ea6d17b0c` |
| Container ID | `7706f2441465100149beb1c8455bffae73ce00f48efcb34efa1fc645ea9886f8` |
| Current container OS | Ubuntu 22.04.4 LTS |
| GPU | 4x NVIDIA RTX 5000 Ada Generation |
| GPU memory | 32760 MiB per GPU |
| GPU compute capability | 8.9 |
| CPU | Intel Xeon w5-3435X, 16 cores / 32 threads, 1 socket |
| RAM | 109 GiB system memory, 9 GiB swap |
| NVIDIA driver | 560.35.03 |
| Driver-reported CUDA version | 12.6 |
| Container CUDA toolkit | CUDA compilation tools 12.3, V12.3.107 |
| Container compiler | gcc/g++ 9.5.0 |
| Python | 3.10.12 |

## Implemented And Validated

### 1. Orca Profiling Evidence

Status: implemented measurement and saved local evidence.

What exists:

- Local Orca loopback profiling evidence in `GPU-MPC/orca_runner/logs/master.log`.
- Extracted poster-ready profile rows for P-SecureML, P-LeNet, and P-AlexNet.
- Supporting implementation context showing Orca already uses direct I/O and overlap infrastructure.

Measured rows:

| Model | Key files observed | Avg key read (ms) | Avg compute (ms) | Comm per iteration (B) |
| --- | --- | ---: | ---: | ---: |
| P-SecureML | P0 338M, P1 338M | 9.909 | 32.273 | 5,692,170.18 |
| P-LeNet | P0 4.0G, P1 4.0G | 109.727 | 107.727 | 65,572,810.18 |
| P-AlexNet | P0 3.8G, P1 3.8G | 104.818 | 121.727 | 113,913,098.18 |

Interpretation:

- Key-read time approaches or matches compute time for larger local Orca training runs.
- The local profile supports “several GiB per party,” not a measured “tens of GiB” maximum in this artifact set.
- Because Orca already has direct I/O and read/compute overlap, the bottleneck should be framed as structural key movement pressure rather than a purely naive I/O issue.

### 2. Standalone DPF Online Key Generation Benchmark

Status: implemented and validated as a standalone systems benchmark.

What exists:

- Benchmark source: `GPU-MPC/tests/fss/dpf_online_keygen_bench.cu`
- Sweep driver: `GPU-MPC/scripts/run_dpf_online_keygen_sweep.py`
- Result files for chunk sizes 8192, 4096, and 2048.

What it does:

- Compares one-shot full-pair eval-all DPF key generation against fixed-size chunked generation.
- Full mode materializes both parties' full eval-all key material at once.
- Chunked mode materializes only the current chunk, validates layout/metadata, and moves on.

Validated configurations:

- `bin=16`
- N from 8192 to 1048576
- chunk sizes 8192, 4096, 2048

Main result:

| N | chunk | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Time overhead |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1048576 | 8192 | 360.00 | 2.81 | 128.00x | 1.834x |
| 1048576 | 4096 | 360.00 | 1.41 | 255.99x | 2.942x |
| 1048576 | 2048 | 360.00 | 0.70 | 511.97x | 4.975x |

Important boundary:

- This reduces peak staged footprint, not total logical key material.
- This is not yet integrated into end-to-end Orca FSS evaluation.

### 3. Promoted GPU NTT / PolyMul Backend

Status: implemented and validated for requested q=32 and q=64 single-prime paths.

What exists:

- Primary GPU NTT/PolyMul benchmark path under `GPU-MPC/ringlpn`.
- CPU baseline using NFLLib.
- q=32 and q=64 GPU sweep result files.
- Direct CPU/GPU comparison at n=8192.

Parameter mapping:

| Requested qbits | Actual qbits | Current mode |
| ---: | ---: | --- |
| 32 | 30 | single-prime GPU path |
| 64 | 62 | single-prime GPU path |

Direct n=8192 comparison:

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| CPU NFLLib | 30 | 1 | pass | 57.2021 | 61.8469 | 180.594 | 180.594 |
| GPU CUDA | 30 | 64 | pass | 41.7984 | 45.8986 | 129.509 | 2.024 |

Speedups from the direct artifact:

- Forward NTT speedup per polynomial: 87.59x
- Full PolyMul speedup per polynomial: 89.24x

Validated sweep ranges:

- requested q=32 / actual q=30: n=8192 to 1048576, validation pass.
- requested q=64 / actual q=62: n=8192 to 1048576, validation pass.

Important boundary:

- q=128 / CRT is not implemented.
- The poster/report should use a two-way NTT story: NFLLib CPU baseline versus the promoted GPU backend.

### 4. Standalone Ring-LPN VOLE Prototype

Status: implemented and validated as a standalone algebraic expansion benchmark.

What exists:

- Source: `GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu`
- Result files for q=32, q=64, and a q=32 m=64 sensitivity sweep.

What it does:

- Reuses the promoted GPU PolyMul backend.
- Synthesizes MPVOLE-consistent inputs locally under `synthetic_mpvole`.
- Computes x, y, and z through batched inner-product phases.
- Validates the coefficient-wise relation `z = y + x * Delta`.

Baseline configuration:

- m=32
- c=2
- noise weight 64
- n=8192 to 1048576
- requested q=32 / actual q=30
- requested q=64 / actual q=62

Selected results:

| q req | q actual | n | Full expand mean (us) | Per-output expand (us) | Outputs/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 30 | 8192 | 191.485 | 5.984 | 167114.92 |
| 32 | 30 | 1048576 | 32144.700 | 1004.522 | 995.50 |
| 64 | 62 | 8192 | 549.802 | 17.181 | 58202.77 |
| 64 | 62 | 1048576 | 50952.700 | 1592.272 | 628.03 |

Important boundary:

- This measures the algebraic expansion layer.
- It is not a full SPFSS-backed OLE-R-LPN pipeline.
- Do not claim CPU-vs-GPU VOLE speedup; no CPU VOLE baseline is saved.

### 5. Standalone Figure 2 OLE GPU Artifact

Status: implemented and validated as a standalone OLE artifact.

What exists:

- OLE source: `GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu`
- GPU SPFSS payload path: `GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh`
- SPFSS correctness test: `GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu`
- Uniform and regular sparse-noise result files.

What it validates:

```text
z_0 + z_1 == x_0 * x_1 in Z_p[X]/(X^N + 1)
```

Configuration:

- requested q=64
- actual q=62
- c=2
- t=64
- n in 8192 and 16384
- uniform and regular sparse noise

Selected results:

| Noise | n | SPFSS domain | validation | host validation | Key bytes MiB | Keygen us | OLE expand mean us |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| uniform | 8192 | 16384 | pass | pass | 8.63 | 4797.000 | 865253.000 |
| uniform | 16384 | 32768 | pass | skipped | 9.19 | 5296.000 | 1830210.000 |
| regular | 8192 | 256 | pass | pass | 5.27 | 40828.000 | 58462.500 |
| regular | 16384 | 512 | pass | skipped | 5.84 | 42331.000 | 67733.000 |

Important boundary:

- This stops at OLE.
- It is not Orca Beaver-triple integration.
- It is not paper-parameter q=128 / CRT.
- It is correctness-first, not final optimized scheduling.

### 6. Standalone Ring-Polynomial OLE-To-Beaver Linear-Layer Artifact

Status: implemented and validated as a small standalone ring-polynomial matrix multiplication smoke.

What exists:

- Source: `GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu`
- Uniform and regular smoke result files.

What it validates:

- Each ring product uses two Figure 2 OLE instances:
  - one for `A_0 * B_1`
  - one for `A_1 * B_0`
- Local products add `A_0 * B_0` and `A_1 * B_1`.
- Validation checks that `C_0 + C_1` equals the clear matrix product over `Z_p[X]/(X^N+1)`.

Configuration:

- rows=2
- inner=2
- cols=2
- n=8192
- c=2
- t=8
- 16 OLE instances

Results:

| Noise | SPFSS domain | validation | OLE instances | Key bytes MiB | Keygen us | Linear expand mean us |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| uniform | n/a | pass | 16 | 2.16 | 6594.000 | 222355.000 |
| regular | 2048 | pass | 16 | 1.78 | 82726.000 | 115447.000 |

Important boundary:

- This is ring-polynomial matrix multiplication, not Orca FC integration.
- It does not yet emit Orca-compatible scalar Beaver triples.

### 7. Documentation And Poster Artifacts

Status: implemented.

What exists:

- Paste-ready poster content:
  - `GPU-MPC/ringlpn/results/gpu_fss_workshop_poster_paste_ready.md`
- Speaker/backstory/Q&A brief:
  - `GPU-MPC/ringlpn/results/gpu_fss_workshop_poster_speaker_brief.md`
- Submitted abstract support audit:
  - `GPU-MPC/ringlpn/results/submitted_abstract_support_audit.md`
- Benchmark appendix:
  - `GPU-MPC/ringlpn/results/abstract_benchmark_appendix.md`
- This implementation status report:
  - `GPU-MPC/ringlpn/results/gpu_fss_professor_implementation_status_2026_05_11.md`

## Partially Implemented / Prototype-Only

| Area | Current status | Why it is only partial |
| --- | --- | --- |
| Chunked DPF online generation | Standalone benchmark implemented and validated | Not wired into real Orca online FSS evaluation |
| Ring-LPN VOLE | Standalone algebraic expansion implemented and validated | Uses synthetic MPVOLE-consistent inputs, not full SPFSS-backed pipeline |
| Figure 2 OLE | Standalone OLE relation validated | Stops at OLE; does not generate Orca Beaver triples |
| OLE-to-Beaver bridge | Ring-polynomial matrix smoke validated | Missing Orca scalar packing and share conversion |
| Hardware/runtime provenance | Current environment queried and documented | Original benchmark files did not embed full per-run environment snapshots |

## Not Implemented Yet

The following are not implemented in the current artifact set:

- End-to-end Orca integration of chunked DPF online key generation.
- End-to-end Orca integration of Ring-LPN PCG/VOLE preprocessing.
- Trusted-dealer removal in Orca.
- requested q=128 / CRT support in the promoted GPU polynomial backend.
- Multi-prime scheduling and CRT recomposition for q=128.
- Full SPFSS-backed OLE-R-LPN pipeline beyond the standalone bridge artifacts.
- Paper-parameter Figure 2 OLE results.
- Scalar packing from Orca tensor values into Ring-LPN polynomial slots.
- `Z_p -> Z_{2^bw}` share conversion for Orca's linear-layer arithmetic.
- Orca-compatible `(A, B, C)` triple writer.
- Integration behind `gpuMatmulBeaver`.
- FC-layer-only Orca validation against baseline Beaver triples.
- End-to-end P-LeNet/P-AlexNet measurements with generated Ring-LPN triples.
- Application-level peak memory reduction measurements for the combined online path.
- CPU baseline for the VOLE prototype itself.
- Final optimized SPFSS/OLE scheduling path.

## Recommended Next Steps

Priority order:

1. Add explicit environment-capture output to all benchmark sweep scripts so future result files record GPU, CPU, driver, CUDA, compiler, image ID, and git status.
2. Implement q=128 / CRT support in the promoted GPU polynomial backend.
3. Specify scalar packing from Orca tensor values into polynomial coefficients/slots.
4. Specify and implement `Z_p -> Z_{2^bw}` share conversion.
5. Add an Orca-compatible triple writer that emits the `(A, B, C)` shape consumed by `gpuMatmulBeaver`.
6. Run a tiny FC-layer-only Orca validation against baseline Beaver triples.
7. Integrate chunked DPF generation into an online FSS evaluation path and measure real peak memory.
8. Run end-to-end Orca experiments for P-LeNet/P-AlexNet once the bridge is integrated.

## Safe Language For Professor / Poster

Use:

- “validated standalone building blocks”
- “toward Orca integration”
- “peak staged pair-key footprint”
- “requested q=32/q=64 implemented as actual q=30/q=62 single-prime paths”
- “q=128/CRT remains future work”
- “Ring-polynomial OLE-to-Beaver bridge, not Orca FC integration”

Avoid:

- “finished Orca integration”
- “trusted dealer removed”
- “total key bytes reduced by 128x”
- “q=128 implemented”
- “VOLE speedup over CPU”
- “Orca-compatible scalar Beaver triples are produced”
- “local profile proves tens of GiB”

## Source Artifacts

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

