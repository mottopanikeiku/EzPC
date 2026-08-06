# Full forward-linear-layer systems plan and session checkpoint

**Date:** 2026-08-06
**Status:** approved route; planning complete; implementation not started
**Scope:** one real forward inference model, every convolution/FC layer, exact truncation/state handoff, matched stock-dealer and closest compatible dealerless-PCG comparisons

## Decision and acceptance boundary

Proceed with the full linear-layer systems route after the specialized regular-DMPF audit returned NO-GO. Preserve the current plain Ring-LPN/static-semi-honest boundary: no support-dependent public transcript, no weakening to leakage-robust Ring-LPN, and no claim that q64/q128 are security levels.

The target model is ResNet18, because the current exact workload manifest already contains all 21 ordered forward linear layers: 20 convolutions and the `1x512x1000` classifier. Its configured profile is `64/24|32/10`; 17 convolutions use graph-to-truncation transitions, three shortcut convolutions use graph-branch-to-truncation transitions, and the classifier uses graph-to-truncation. The current classifier-only checkpoint remains valid but is not a full-model result.

A completed artifact must satisfy, for every ordered layer:

1. live two-process dealerless preprocessing with each party reading only its own state;
2. byte-compatible stock Orca forward key records;
3. the unchanged stock online convolution or matmul consumer;
4. exact equality with matched stock-dealer output for the exercised masks/weights;
5. exact post-linear truncation and next-state masks at the configured precision transition;
6. fail-closed layer-order, shape, compatibility-ID, invocation, ledger and stale-output controls;
7. one warmup plus at least ten measured full-model trials on a fixed GPU pair;
8. raw per-layer and aggregate timings, bytes, GPU/host peak memory, dependency layers and confidence intervals;
9. matched stock-dealer and closest functionality-compatible dealerless-PCG baselines under the same shape/precision accounting.

This route does not include nonlinear DCF key generation, training/backward state transitions, malicious security, parameter pinning, or a full dealerless Orca claim.

## Implementation plan

### L1 — Freeze the exact model contract

- Use `results/fc/orca_forward_linear_layer_manifest_2026_08_04.json` as the source-checked ordered inventory.
- Retain exact model/order/operator/input/output/truncation/profile fields and source digests.
- Add an execution manifest binding every layer to qbits, bw, scale, `(n,c,t)`, ring-batch count, expected key sizes and compatibility IDs.
- Fail closed on missing, reordered, duplicated, unsupported or shape-mismatched layers. Do not silently skip padding, branch, or terminal layers.

### L2 — Generalize the live producer without a second protocol

- Refactor the shared party-local cross-term, Ring-OLE, bootstrap and conversion logic already used by `src/test_two_party_fc_preprocess.cu` into one internal linear-preprocessing engine.
- Keep FC and Conv2D as thin public-shape adapters over that engine. Reuse `two_party_spfss.h`, `two_party_dpf_protocol.h`, `two_party_ot.h`, `ringlpn_ole_party.cuh`, `secure_convert.{h,cpp}` and the existing freshness ledger.
- Preserve the unsent identity polynomial, exact `(c-1)*n` exchange, consume-once correlations, sealed bilateral commit and owner-only state boundary.
- Do not create a centralized fallback, clear conversion path, oracle path, or second key format.

### L3 — Integrate every convolution layer

- Extend the live Conv2D path from the current one-case smoke to all 20 ResNet18 convolution shapes, including the three shortcut branches.
- Bind native Orca `Conv2DParams`, padding, stride, channel dimensions, use-bias semantics, and the stock `GPUConv2DKey` ABI.
- Validate through unchanged `readGPUConvKey`/`gpuConv2DBeaver`; compare against matched stock `gpuKeygenConv2D` records and output masks.
- Use safe chunking/ring batching for the largest early layers. Record padding-aware term counts rather than treating im2col upper bounds as executed work.

### L4 — Implement exact truncation and state handoff

- Implement the configured stochastic truncation transitions rather than treating them as metadata.
- For `b` input bits and shift `f`, use the exact shared-mask identity over `Z_(2^(b-f))`: with masked linear output `y=x+r`, shared random low part `u`, opened masked low comparison target `t=(u+r_low) mod 2^f`, carry `c=[t<r_low]`, and comparison `q=[y_low<=t]`, the next share is `y_high + 1 - c - r_high - q`. This realizes signed stochastic truncation with the correct wrap behavior.
- Source daBits/edaBits/Boolean comparisons through the existing secure-convert/OT boundary; never open `r`, `u`, wrap, or comparison bits.
- Define separate graph-to-truncation, graph-branch-to-truncation and terminal-state contracts. The three shortcut outputs must preserve both branch identities and merge order.
- Emit the exact stock next-layer mask representation and validate the next unchanged linear consumer, not just a standalone arithmetic formula.

### L5 — Compose the model runner and controls

- Build one two-process runner that traverses all 21 ordered layers, consumes a distinct correlation namespace per layer/limb/batch/tree/phase, and writes bilateral records plus one sealed model manifest.
- Add deterministic controls for reordered layer, duplicated layer, shape mismatch, qbits/bw/scale mismatch, branch swap, stale prior-layer output, truncation rejection, tail-slot reuse, ledger rollback/collision, partial publication and peer-record corruption.
- A failure at layer `k` must publish no model-level success and must not roll back consumed state.

### L6 — Measure the matched experiment matrix

- Run one warmup plus at least ten measured full-model passes on a fixed pair of distinct free GPUs.
- Report per-layer and aggregate median, spread/confidence interval, host/GPU peaks, network bytes, setup/application split, dependency rounds, bootstrap consumption/discard and unchanged-online time.
- Run matched stock dealer keygen and unchanged online consumers at the same layer shapes and precision profile.
- Re-evaluate the closest dealerless-PCG baseline only if it matches the same functionality, field, factors, setup inclusion, layer shape and output ABI. Keep Reverse Cuckoo native-folded rows explicitly non-comparable otherwise.

### L7 — Publication gate

- Run the canonical host/GPU gate, focused model controls, clean-clone reproduction and authenticated two-host LAN/WAN trials.
- Update the security contract for the composed per-layer state functionality and truncation simulator.
- Require independent human cryptographic review before advancing any security theorem or concrete parameter claim.
- Report a negative systems result if the full-model route is slower than the dealer and/or closest compatible baseline. Do not select best-of-run timings.

## Resource sizing already established

The exact manifest contains 20 convolution layers and one FC layer. A padding-inclusive convolution upper-bound diagnostic gives 181,407,334,400 scalar cross terms and about 31,681,536 q128 key words across the 20 convolutions; this is a sizing upper bound, not executed-work evidence. The first layer alone has a 118,013,952-term upper bound and requires 15,897 Ring-LPN batches under the current 7,424-application-slot budget. These numbers require streaming/chunking and evidence for actual padding-aware work.

## Session-end verification checkpoint

The canonical existing convolution smoke was retried with:

```bash
cd GPU-MPC/ringlpn
./scripts/run_two_party_conv_preprocess.sh
```

It exited nonzero after 600.21 s. Party 0 failed in `initGPUMemPool()` with `cudaErrorMemoryAllocation`; party 1 was terminated by the runner after the peer failure. The contemporaneous GPU snapshot was:

```text
GPU 0: 30,275 MiB used / 1,985 MiB free
GPU 1: 20,446 MiB used / 11,815 MiB free
GPU 2: 25,686 MiB used / 6,575 MiB free
GPU 3: 18 MiB used / 32,242 MiB free
```

Only one GPU had enough free memory for the current pool, while the runner requires two distinct GPUs. Therefore this attempt is **resource-blocked, not a protocol failure**, and it is not counted as completed verification. The dated passing convolution CSV/control evidence was not replaced. Retry only when two distinct GPUs have sufficient free memory, with explicit `P0_GPU`/`P1_GPU` assignments.

## Repository checkpoint

This session intentionally leaves code unchanged after the DMPF NO-GO and systems-plan decision. The only pre-existing dirty paths are external scratch submodules `GPU-MPC/ext/cutlass` and `GPU-MPC/ringlpn/extern/NFLlib`; they are excluded from this checkpoint. Continue from this document, the canonical `CLAUDE.md`, the exact model manifest, and the current live FC/Conv sources.
