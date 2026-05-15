# Ring-LPN to Orca Linear Layer Integration: Presentation Deliverable

Date: 2026-05-15

## One-Sentence Claim

We implemented and validated a correctness-first Ring-LPN-to-Orca forward-FC integration demo: Ring-LPN-style bounded q62 Beaver masks are exported into Orca-compatible `A`, `B`, `C_masked` key buffers, run through the unchanged Orca `gpuMatmulBeaver` online path, and checked against Orca's baseline `gpuKeygenMatmul` behavior on a bounded small-shape suite.

This is a credible v1 integration demo. It is not yet paper-parameter q128/CRT, dense packing, secure distributed conversion, or full Orca training integration.

## What Changed

| Area | Change | Why it matters |
| --- | --- | --- |
| Ring-polynomial linear artifact | Fixed matrix semantics so each `A[row,k]` and `B[k,col]` operand is generated once and reused across products. | Validation now proves a true matrix product, not independent per-output product sums. |
| Orca FC demo | Added `src/bench_orca_fc_ringlpn_demo.cu`. | Produces raw party key buffers in Orca read order: `A`, `B`, `C_masked`. |
| Online path | Kept `gpuMatmulBeaver` unchanged. | Demonstrates compatibility with Orca's existing Beaver online contract. |
| Baseline comparison | Added Orca `gpuKeygenMatmul` comparison using the same masks. | Shows the new raw writer and Orca baseline reconstruct the same online output. |
| Bridge correctness | Reused carry-corrected `Z_p -> Z_{2^bw}` conversion. | Prevents the prime-carry bug that breaks naive per-share reduction. |
| Documentation | Added professor memo, bounded-suite results, status updates, and one-command smoke coverage. | Makes the result reproducible and honest about boundaries. |

Primary implementation files:

- `ringlpn/src/bench_linear_ole_ringlpn_cuda.cu`
- `ringlpn/src/bench_orca_fc_ringlpn_demo.cu`
- `ringlpn/src/test_orca_zp_bridge.cpp`
- `ringlpn/scripts/run_paper_checkpoint_smoke.sh`
- `ringlpn/scripts/run_orca_fc_ringlpn_demo.sh`

## Correctness Argument

For Orca FC Beaver evaluation, the dealer supplies additive shares of:

- `A`: input mask,
- `B`: weight mask,
- `C_masked = A * B + output_mask`.

The online parties receive masked operands:

- `X + A`,
- `W + B`.

The unchanged Orca online expression reconstructs:

`X * W + output_mask`.

For the Ring-LPN scalar bridge, q62 field shares are exported to Orca's `Z_{2^bw}` ring using the carry-corrected rule:

- `r0 = z0 mod 2^bw`,
- `r1 = z1 - m*p mod 2^bw`,
- `m = floor((z0 + z1) / p)`.

For the bounded demo, `inner * value_bound^2 < p`, so the q62 field dot product equals the integer dot product before prime wrap. This is why the q62-to-ring export is mathematically valid for the current demo.

## Current Benchmark Results

### 1. Orca FC Ring-LPN Bounded Suite

All cases use `poly_n=8192`, `c=2`, `t=8`, `noise=regular`, `tf=None`, zero bias, and `value_bound=255`.

| Shape | bw | Seeds | Key bytes / party | Baseline bytes / party | Online | Baseline | Baseline matches Ring-LPN writer |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: |
| `2x2x2` | 16 | `1,2` | 96 | 96 | pass | pass | 1 |
| `2x3x2` | 16 | `3,4` | 128 | 128 | pass | pass | 1 |
| `3x2x2` | 16 | `5,6` | 128 | 128 | pass | pass | 1 |
| `2x2x3` | 32 | `7,8` | 128 | 128 | pass | pass | 1 |

Takeaway: the raw Ring-LPN-style writer is compatible with Orca's existing FC Beaver online contract for the current bounded suite.

Result artifact:

- `ringlpn/results/orca_fc_ringlpn_demo_bounded_suite.md`

### 2. Zp-to-Z2k Bridge

| Case | bw | Shape | Value bound | Naive failures | Corrected failures | Scalar validation | Counterexample |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| bounded demo | 16 | `2x2x2` | 255 | 633 | 0 | pass | 0 |
| negative control | 32 | `1x1x1` | 4,294,967,295 | 633 | 0 | not claimed | 1 |

Takeaway: independent per-share reduction is wrong; the carry-corrected conversion is required. The q62/full-32-bit counterexample remains present, so we are not overclaiming unrestricted 32-bit support.

### 3. Shared Linear OLE-to-Beaver Artifact

| Noise | n | Shape | SPFSS domain | Ring products | OLE instances | Key bytes | Keygen us | Expand mean us | Shared operands | Validation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| uniform | 8192 | `2x2x2` | 16384 | 8 | 16 | 2,264,064 | 6,587 | 223,667 | 1 | pass |
| regular | 8192 | `2x2x2` | 2048 | 8 | 16 | 1,864,704 | 81,582 | 114,825 | 1 | pass |

Performance note:

- Regular noise reduces expand time from `223,667 us` to `114,825 us`, about `1.95x` faster for this smoke.
- Regular noise also reduces pair key bytes from `2.26 MB` to `1.86 MB`.
- Regular-noise keygen is slower in this small smoke because grouped SPFSS key generation does more structured work.

### 4. Figure 2 GPU OLE Smoke

| Noise | n | c | t | SPFSS domain | Key bytes | Keygen us | Expand mean us | Validation | Host validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| uniform | 8192 | 2 | 8 | 16384 | 141,504 | 455 | 13,330 | pass | pass |
| regular | 8192 | 2 | 8 | 2048 | 116,544 | 5,061 | 6,960 | pass | pass |

Performance note:

- Regular noise roughly halves OLE expand latency in this smoke (`13,330 us` to `6,960 us`) and reduces key bytes.
- This supports the direction of regular-noise/grouped-domain SPFSS for the online path.

### 5. Promoted CUDA NTT/PolyMul Core

The promoted cheddar-derived CUDA path is the default GPU polynomial engine.

| Requested q | Actual q | n range | Representative per-poly PolyMul | Validation |
| ---: | ---: | --- | ---: | --- |
| 32 | 30 | `8192` to `1048576` | `1.244 us` at `n=8192`; `166.381 us` at `n=1048576` | pass |
| 64 | 62 | `8192` to `1048576` | `3.963 us` at `n=8192`; `244.429 us` at `n=1048576` | pass |

Prior CPU/GPU comparison summaries show:

- requested q32 overlap points: about `146x` to `171x` per-polynomial PolyMul speedup over CPU,
- requested q64 points: about `48x` to `220x` per-polynomial PolyMul speedup over CPU.

### 6. DPF Online Key Generation Memory Result

For `bin=16`, `chunk_size=8192`:

| n | Full pair key | Partial peak pair key | Peak reduction | Time overhead |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 2.81 MiB | 2.81 MiB | 1.00x | 0.996x |
| 1048576 | 360.00 MiB | 2.81 MiB | 128.00x | 1.834x |

Takeaway: chunked online generation can hold peak pair-key footprint nearly constant as `n` grows, at a measured runtime overhead.

## Exact Reproduction Command

Run inside the existing `orca-dev` container:

```bash
cd /home/ringlpn
RUN_GPU_SMOKE=1 scripts/run_paper_checkpoint_smoke.sh
```

This currently rebuilds/runs:

- host Orca `Z_p -> Z_{2^bw}` bridge smoke,
- GPU SPFSS payload tests,
- uniform and regular Figure 2 OLE smokes,
- uniform and regular shared-linear OLE-to-Beaver smokes,
- bounded Orca FC Ring-LPN key-writer suite with Orca baseline comparison.

Latest local verification passed on 2026-05-15.

## What Is Safe To Present

Safe claims:

1. The Ring-LPN linear artifact now validates true shared matrix semantics.
2. The q62-to-Orca-ring bridge correctly handles hidden prime carries.
3. A bounded forward-FC Orca demo now writes raw compatible key buffers and runs through unchanged `gpuMatmulBeaver`.
4. The new writer matches Orca `gpuKeygenMatmul` baseline output on the current bounded suite.
5. Regular-noise SPFSS reduces online expansion time and key bytes in the current OLE/linear smokes.
6. The promoted GPU polynomial engine is validated across `n=8192` through `1048576` for requested q32/q64.

Do not claim yet:

1. q128/CRT support.
2. Unrestricted 32-bit Orca scalar products under single-prime q62.
3. Dense packing of many Orca tensor entries into one Ring-LPN polynomial.
4. Secure distributed `Z_p -> Z_{2^bw}` conversion without dealer/oracle knowledge.
5. Backward/training/optimizer key integration.
6. End-to-end P-LeNet/P-AlexNet replacement.
7. Trusted-dealer removal.

## Suggested Slide Outline

1. **Problem**
   - Orca FC uses Beaver triples over `Z_{2^bw}`.
   - Ring-LPN OLE naturally produces shares over `Z_p[X]/(X^N+1)`.
   - Need an honest bridge from Ring-LPN arithmetic to Orca key buffers.

2. **What Was Built**
   - Shared matrix OLE-to-Beaver artifact.
   - Carry-corrected q62-to-ring scalar bridge.
   - Tiny Orca FC key-writer demo using raw `A`, `B`, `C_masked`.

3. **Correctness**
   - Show `C_masked = A*B + output_mask`.
   - Show carry correction.
   - State bound: `inner * value_bound^2 < p`.

4. **Orca Compatibility**
   - Online `gpuMatmulBeaver` unchanged.
   - Baseline `gpuKeygenMatmul` comparison passes.
   - Four small bounded cases pass.

5. **Performance Evidence**
   - OLE regular vs uniform.
   - Linear regular vs uniform.
   - NTT/PolyMul q32/q64 speedups.
   - DPF peak key-footprint reduction.

6. **Boundary**
   - q62 bounded demo, not q128 paper parameters.
   - One scalar per polynomial, not dense packing.
   - Dealer/oracle conversion, not secure distributed conversion.

7. **Next Steps**
   - q128/CRT.
   - Dense packing with host oracle.
   - Secure conversion.
   - Extend to backward/training.
   - End-to-end Orca layer/model measurement.

## Next Implementation Steps

| Priority | Step | Deliverable |
| ---: | --- | --- |
| 1 | Add q128/CRT to promoted GPU path. | q128 NTT/OLE/linear smoke with requested-vs-actual modulus reporting. |
| 2 | Build dense scalar packing model. | Host oracle proving no coefficient/sign-fold mistakes before GPU benchmarking. |
| 3 | Replace dealer/oracle bridge if trusted-dealer removal is the target. | Secure or cited `Z_p` share to `Z_{2^bw}` share conversion. |
| 4 | Extend FC demo to backward/training keys. | Forward/backward/optimizer key writer checks against Orca baseline. |
| 5 | Measure model-level integration. | P-LeNet/P-AlexNet or selected FC-heavy layer report with memory and runtime. |

## Repository State

Recent local commits:

- `71c6d3e Add bounded Ring-LPN Orca FC demo`
- `5fbacbe Broaden Orca FC Ring-LPN demo checks`

Current known environment note: GitHub HTTPS push from this container previously failed because credentials were unavailable. The local repository contains the deliverable and benchmark artifacts.
