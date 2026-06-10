# Ring-LPN for Orca Linear Layer: Integration Plan

Drafted: 2026-04-20
Updated: 2026-04-27 after the standalone GPU Figure 2 SPFSS/OLE artifact
Updated: 2026-05-06 after regular-noise GPU OLE and linear smokes
Targets: FC and Conv2D (both forward and backward matmuls). Nonlinear layers remain on the existing DPF/FSS path.

## Context

- Orca's linear layers use a Beaver-triple protocol: the dealer samples `mask_X`, `mask_W`, computes `masked_Z = mask_X * mask_W + mask_Z`, and writes three key shares per layer per iteration. All this is in [gpu_matmul.cu:249](../../fss/gpu_matmul.cu#L249) (`gpuKeygenMatmul`) consumed at runtime by [gpu_matmul.cu:311](../../fss/gpu_matmul.cu#L311) (`gpuMatmulBeaver`).
- Our building blocks (as of 2026-04-20, after the NTT-caching refresh) are:
  - Ring-LPN VOLE expansion on GPU: 191.485 us at n=8192, q=32; passes z = y + x*Delta on all tested n. Entry: [bench_vole_ringlpn.cu](../src/bench_vole_ringlpn.cu).
  - Chunked DPF online key generation: 1.00x to 1.83x time overhead in exchange for up to 128x peak key-memory reduction. Entry: [dpf_online_keygen_bench.cu](../../tests/fss/dpf_online_keygen_bench.cu).
- New as of 2026-04-27: the standalone GPU Figure 2 SPFSS/OLE artifact validates `z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N+1)` for requested q=64 / actual single-prime q=62, uniform sparse noise, `c=2`, `t=64`, and bounded `n in {8192, 16384}`. Entry points: [gpu_spfss_zp.cuh](../src/gpu_spfss_zp.cuh), [bench_ole_ringlpn_cuda.cu](../src/bench_ole_ringlpn_cuda.cu), and [ole_gpu_handoff.md](ole_gpu_handoff.md).
- New as of 2026-05-04: the standalone ring-polynomial linear-layer artifact validates the two-OLE-to-Beaver conversion for matrix multiplication over `Z_p[X]/(X^N+1)`. The smoke case is `rows=2`, `inner=2`, `cols=2`, `n=8192`, `c=2`, `t=8`; it uses 8 ring products and 16 OLE instances, and validation passes. Entry points: [bench_linear_ole_ringlpn_cuda.cu](../src/bench_linear_ole_ringlpn_cuda.cu), [linear_ole_handoff.md](linear_ole_handoff.md), and [linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md](linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md).
- New as of 2026-05-06: regular sparse noise is implemented for the GPU OLE and ring-polynomial linear artifacts. The OLE regular smoke uses SPFSS domain `2N/t = 2048` at `N=8192`, `t=8` and passes validation. The linear regular smoke uses the same regular-noise OLE mode and passes validation.
- The older abstract claims on the VOLE / DPF building blocks are recorded in [ringlpn_vole_abstract_support.md](ringlpn_vole_abstract_support.md) and [abstract_benchmark_appendix.md](abstract_benchmark_appendix.md). The newer OLE-specific claim is intentionally narrower and lives in [ole_gpu_handoff.md](ole_gpu_handoff.md).

What we want next: an *Orca-linear-layer-level* artifact that shows Orca running with Ring-LPN-derived correlations instead of a monolithic trusted-dealer precomputation, measured head-to-head against baseline Orca on P-LeNet / P-AlexNet. The new ring-polynomial linear artifact has implemented the OLE-to-Beaver step over `Z_p[X]/(X^N+1)`, but it still needs scalar packing and `Z_p -> Z_{2^bw}` share conversion before it can feed Orca's `gpuMatmulBeaver` path.

## Orca's Linear-Layer Hook Points

The integration surface is narrow and well-factored:

| Hook | File:line | Role |
| --- | --- | --- |
| `gpuKeygenMatmul` | [fss/gpu_matmul.cu:249](../../fss/gpu_matmul.cu#L249) | Samples `mask_X`, accepts `mask_W`, writes three shares per matmul. The single dealer-side entry point. |
| `FCLayer::genForwardKey` | [nn/orca/fc_layer.cu:120](../../nn/orca/fc_layer.cu#L120) | Per-layer wrapper: samples `mask_Z`, computes `masked_Z = mask_X*mask_W + mask_Z` via `gpuMatmulPlaintext`, calls the matmul keygen. |
| `FCLayer::genBackwardKey` | [nn/orca/fc_layer.cu](../../nn/orca/fc_layer.cu) | Parallel structure for pdW/pdX backward matmuls. |
| `Conv2DLayer::genForwardKey` / `genBackwardKey` | [nn/orca/conv2d_layer.cu](../../nn/orca/conv2d_layer.cu) | Same structure; Conv is a reshape + matmul under the hood. |
| `genModelKey` | [experiments/orca/orca_dealer.cu:63](../../experiments/orca/orca_dealer.cu#L63) | Top-level model walk calling every layer's keygen. |
| `dealerE2E` | [experiments/orca/orca_dealer.cu:90](../../experiments/orca/orca_dealer.cu#L90) | Epoch loop over `genModelKey`; writes keys to disk. |

Online-side (`gpuMatmulBeaver`) does not need to change — as long as the dealer writes the same three shares, the online protocol is unchanged. That is the invariant we lean on.

## Three-Phase Rollout

Note after the 2026-05-04 linear artifact: Phase A below is still useful as a low-risk Orca plumbing exercise, but it does not remove the trusted dealer. For the actual dealer-removal path, Phase B should start from the new ring-polynomial OLE-to-Beaver artifact and add scalar packing plus share-conversion machinery.

### Phase A — Mechanical: swap the dealer's PRG for Ring-LPN VOLE expansion

**Goal.** Replace `randomGEOnGpu<u64>(...)` calls inside the matmul dealer with a PRG driven by Ring-LPN VOLE expansion output. Dealer stays centralized; no new crypto. This is an *engineering-only* step: it verifies the plumbing end-to-end and gives us a correctness baseline for later phases.

**What changes.**
1. Add `ringlpn::LinearMaskSource` (new TU, e.g. `ringlpn/src/linear_mask_source.{cu,h}`): owns persistent `d_a_pairs`, `d_a_ntt`, one-time NTT tables, and a call `fill(T *d_out, size_t n_elems, int bw)` that drains one or more VOLE batches and packs into `d_out`.
2. Edit `gpuKeygenMatmul`: replace `d_mask_X = randomGEOnGpu<T>(p.size_A, p.bw)` and `d_mask_Z = randomGEOnGpu<T>(p.size_C, p.bw)` with calls to `LinearMaskSource::fill`.
3. Edit `FCLayer::genForwardKey` (and the Conv2D / backward counterparts) analogously for their local `d_mask_O` sampling.
4. Thread a `LinearMaskSource*` through `genModelKey` / `dealerE2E` so there is one per party, seeded deterministically.

**What does NOT change.** The Beaver product `masked_Z = mask_X * mask_W + mask_Z` is still computed by `gpuMatmulPlaintext` locally on the dealer's GPU. No distributed protocol yet.

**Correctness witness.** Loss curves on P-LeNet / P-AlexNet must match the baseline within float tolerance (the masks are uniform regardless of generator, so the protocol is statistically identical to baseline).

**Measurable artifact.**
- Peak key-file size on disk (should be unchanged — same three shares written).
- Dealer wall-clock (expected: slightly higher, from VOLE expansion overhead replacing fast `randomGEOnGpu`; will be quantified).
- GPU peak resident memory during `genModelKey` (baseline for Phase B).

**Estimated effort.** 2-3 days. Contained in ≤4 files.

### Phase B — Protocol: distributed dealer via Ring-LPN-backed OLE → matrix Beaver triples

**Goal.** Remove the trusted dealer. Each server generates its own share of `(mask_X, mask_W, masked_Z)` using a Ring-LPN PCG-derived OLE; the sum of the two servers' shares is a correct Beaver triple.

**Open crypto questions that must be answered before this phase starts.**
1. **Scalar packing.** The Figure 2 GPU artifact gives OLE over Ring-LPN polynomials, and the 2026-05-04 artifact converts those OLEs into Beaver matrix products over ring-polynomial entries. Orca's `gpuMatmulBeaver` expects scalar tensor entries over `Z_{2^bw}`. The next protocol step is to define how Orca scalar entries are packed into polynomial entries and how those packed triples are unpacked or consumed.
2. **Share encoding.** Current `gpuMatmulBeaver` expects *additive* shares of `mask_X, mask_W, masked_Z` over Z_{2^bw}. The new linear artifact produces shares modulo a 62-bit NTT-friendly prime. We need an explicit modular reduction / share-conversion step and a written argument that it preserves security. Modulus switching from a prime field/ring to `Z_{2^bw}` is non-trivial for active security and should be treated as a reviewed protocol step, not a mechanical cast.
3. **Noise / parameter set.** The current OLE artifact supports uniform sparse noise and bounded regular-noise sweeps over a single 62-bit prime. Paper-comparable parameters still require a CRT lift toward `log p ~= 128`. For Orca we also need `bw in {32, 64}` rings and matrix sizes up to roughly 1M entries per layer.

**This phase blocks on user sign-off on (1)–(3) before I write crypto-bearing code.** I will draft a protocol spec document for review.

**Measurable artifact after this phase.**
- Dealer bytes-on-wire (should drop dramatically: only seeds transmitted, not keys).
- Peak resident GPU memory (should drop — no full `masked_Z` materialized at once).
- End-to-end Orca epoch wall-clock vs baseline.

### Phase C — Integration harness + evaluation

Flag `use_ringlpn_pcg={0|1}` in `orca_dealer.cu` / `orca_evaluator.cu` to switch between baseline and Ring-LPN paths. Add a measurement wrapper that logs:

- Peak dealer-side RSS (`/proc/self/status: VmHWM`) and peak GPU memory (`cudaMemGetInfo`) per epoch.
- Wall-clock `genModelKey` and per-layer breakdown.
- Online protocol wall-clock (should be identical to baseline — invariant check).
- Loss curves, to confirm numerical equivalence.

Run matrix: {P-LeNet, P-AlexNet} x {baseline Orca, Phase-A Orca, Phase-B Orca} x {1 epoch, 1 block, batchSz=128}. Three runs each, report median.

Deliverable: `ringlpn_linear_results.md` with table + plot. This is the headline abstract artifact for the linear-layer claim.

## Files That Will Change (Phase A only)

- **New**: `GPU-MPC/ringlpn/src/linear_mask_source.cu` + `.h` (VOLE-backed mask source; owns NTT tables and a pair of persistent device buffers per party).
- `GPU-MPC/fss/gpu_matmul.cu` — swap `randomGEOnGpu` for `LinearMaskSource::fill` at two sites in `gpuKeygenMatmul`.
- `GPU-MPC/nn/orca/fc_layer.cu` — swap `randomGEOnGpu` call in `genForwardKey` / `genBackwardKey`.
- `GPU-MPC/nn/orca/conv2d_layer.cu` — same.
- `GPU-MPC/experiments/orca/orca_dealer.cu` — construct `LinearMaskSource` in `dealerE2E`, thread through `genModelKey`.
- `GPU-MPC/experiments/orca/Makefile` — link in new TU.

## Verification

- Phase A: `./dealer lenet 1 1 128 1 1 0` (baseline) and with `USE_RINGLPN=1` env var. Compare loss curves; they must match. Compare dealer wall-clock and key-file size; record delta.
- Phase B: cryptographic spec review + re-run above with Phase-B path; key bytes on wire should drop to a small constant (seeds only).
- Phase C: full benchmark sweep across the 3x2x3 matrix above.

## Open Questions For User

1. **Next priority.** For paper-comparable primitive numbers, do CRT q128 and rerun the Figure 2 OLE sweeps. For Orca trusted-dealer removal, do scalar packing plus `Z_p -> Z_{2^bw}` share conversion next.
2. **Security model.** Semi-honest only, or does the Orca integration need to be malicious-secure? The answer changes the share-conversion argument and may require extra checks.
3. **Triple packing.** Should the first Orca integration pack scalar tensor entries into ring-polynomial triples directly, or use a more structured batching layout tuned to FC/Conv dimensions?
4. **P-AlexNet availability.** P-LeNet is readily runnable from `experiments/orca/`. Is P-AlexNet currently wired up and runnable in this tree, or does it need restoration first?

Phase B is now concrete enough to spec from the ring-polynomial linear artifact, but it should not proceed into Orca key files until the share-conversion and scalar-packing choices are written down.
