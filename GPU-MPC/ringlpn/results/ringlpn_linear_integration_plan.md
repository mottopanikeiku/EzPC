# Ring-LPN for Orca Linear Layer: Integration Plan

Drafted: 2026-04-20
Targets: FC and Conv2D (both forward and backward matmuls). Nonlinear layers remain on the existing DPF/FSS path.

## Context

- Orca's linear layers use a Beaver-triple protocol: the dealer samples `mask_X`, `mask_W`, computes `masked_Z = mask_X * mask_W + mask_Z`, and writes three key shares per layer per iteration. All this is in [gpu_matmul.cu:249](../../fss/gpu_matmul.cu#L249) (`gpuKeygenMatmul`) consumed at runtime by [gpu_matmul.cu:311](../../fss/gpu_matmul.cu#L311) (`gpuMatmulBeaver`).
- Our building blocks (as of 2026-04-20, after the NTT-caching refresh) are:
  - Ring-LPN VOLE expansion on GPU: 191.485 us at n=8192, q=32; passes z = y + x*Delta on all tested n. Entry: [bench_vole_ringlpn.cu](../src/bench_vole_ringlpn.cu).
  - Chunked DPF online key generation: 1.00x to 1.83x time overhead in exchange for up to 128x peak key-memory reduction. Entry: [dpf_online_keygen_bench.cu](../../tests/fss/dpf_online_keygen_bench.cu).
- The abstract claims on the VOLE / DPF building blocks are now locked in ([ringlpn_vole_abstract_support.md](ringlpn_vole_abstract_support.md), [abstract_benchmark_appendix.md](abstract_benchmark_appendix.md)).

What we want next: a *linear-layer-level* artifact that shows Orca running with Ring-LPN-derived correlations instead of a monolithic trusted-dealer precomputation, measured head-to-head against baseline Orca on P-LeNet / P-AlexNet.

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
1. **Which Ring-LPN PCG construction feeds which OLE?** The VOLE expansion we have produces scalar-by-polynomial correlations `z = y + x·Delta`. For matmul, we need *matrix*-structured Beaver triples. Two candidate paths: (a) batched scalar OLE → pack into rows of `mask_X, mask_W`; (b) Gilboa-style reduction from several VOLE instances into OLE into multiplicative triples. Need user to confirm which Ring-LPN paper / construction to anchor on.
2. **Share encoding.** Current `gpuMatmulBeaver` expects *additive* shares of `mask_X, mask_W, masked_Z` over Z_{2^bw}. Our VOLE produces shares modulo a 30- or 62-bit NTT-friendly prime. We need an explicit modular reduction / share-conversion step and a written argument that it preserves security (modulus switch from prime to 2^bw ring is non-trivial for active security; fine for semi-honest).
3. **Noise / parameter set.** Ring-LPN paper parameter sets (noise weight, code rate) translate into concrete (m, c, w) choices at the bench level. For Orca we need `bw in {32, 64}` rings, matrix sizes up to ~1M entries per layer. Need to pick a parameter set consistent with security.

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

1. **Phase ordering.** Do Phase A first as a pure engineering step, or go directly to Phase B's protocol spec? Recommended: A first — it de-risks plumbing and gives a working artifact this week.
2. **Which Ring-LPN PCG construction** should Phase B anchor to? (BCGIKRS-style quasi-linear VOLE-to-OLE? Something else? I need the paper reference before I can write the spec.)
3. **Security model.** Semi-honest only, or does Phase B need to be malicious-secure? The answer changes the share-conversion step.
4. **P-AlexNet availability.** P-LeNet is readily runnable from `experiments/orca/`. Is P-AlexNet currently wired up and runnable in this tree, or does it need restoration first?

Phase A can start as soon as (1) is answered. Phase B needs (2) and (3).
