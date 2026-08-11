> **HISTORICAL (superseded 2026-08-10):** superseded by `CLAUDE.md`; statements below may describe the older prefix-only checkpoint.

# Full forward-linear-layer systems plan and session checkpoint

**Date:** 2026-08-06; updated 2026-08-09
**Status:** L1--L3, SHAKE256 public-vector generation, bounded breadth-first
prepared DPF evaluation, isolated truncation, the fail-closed 21-layer
record-set runner, and one source-bound known-zero
Conv0→TR→MaxPool→ReLU→Conv3 checkpoint are implemented. The only attempted
full record set is incomplete after Conv0. The prefix checkpoint closes its
digest-bound per-party mask-state/order plumbing and invokes live truncation
plus unchanged Conv2D, MaxPool, and ReLU consumers. Its exact stock-format
nonlinear keys come from a TEST-ONLY trusted adapter that reads both parties'
mask-state records; it is not dealerless nonlinear preprocessing. All residual
branches, a complete fresh record/graph run, dealerless nonlinear key
generation, private/trained inference, repeated model evaluation, authenticated
two-host runs, and human cryptographic review remain open.
**Scope:** the target remains every ResNet18 convolution/FC preprocessing record
and exact stock layer shape/order. The dated prefix checkpoint is narrow
graph-composition evidence, not completed stock graph execution.

## Decision and acceptance boundary

Proceed with the full linear-layer systems route after the specialized regular-DMPF audit returned NO-GO. Preserve the current plain Ring-LPN/static-semi-honest boundary: no support-dependent public transcript, no weakening to leakage-robust Ring-LPN, and no claim that q64/q128 are security levels.

The target model is ResNet18. The source-bound plan contains all 21 ordered Ring-LPN record rows: 20 convolutions and the `1x512x1000` classifier. The adaptive execution profile uses q128/bw32, regular `(c,t)=(2,8)`, and the largest power-of-two `n` supported by the 32-GiB GPU budget (`n=262144` here). Stock execution applies stochastic truncation after the 20 convolutions and `GlobalAvgPool2D`, performs a pre-classifier sign extension, and disables stochastic truncation on the terminal classifier. Those stock transitions are bound as source metadata but are not executed by the current isolated-record artifact.

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
- Preserve the unsent identity polynomial, one four-word joint public seed per layer, independently scoped SHAKE256 public vectors in the explicit random-oracle model, consume-once correlations, sealed bilateral commit and owner-only state boundary.
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

- `scripts/run_full_linear_record_set.py` traverses all 21 ordered layer contracts, consumes a distinct correlation namespace per layer/limb/batch/tree/phase, and writes bilateral records plus one sealed record-set manifest.
- Deterministic controls reject reordered/duplicated/shape-mismatched plans, stale output, source tamper, binary-plan mismatch, malformed metrics, ledger inconsistency, partial publication, and peer-record corruption.
- A failure at layer `k` publishes no record-set success and does not roll back consumed state. Residual state and truncation controls remain model-composition work, not claims of this runner.

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

## Resource sizing and degree adaptation

The exact source manifest contains 20 convolution layers and one FC layer with 1,680,390,912 padding-aware cross terms. The executable degree frontier selects `n=262144`, reducing the record-set plan from 226,361 baseline batches at `n=8192` to 6,439 adaptive batches while preserving the same cross-term count and key ABI. The first convolution has 116,214,528 executed cross terms and 445 adaptive batches. Degree probes reject unsupported or memory-infeasible values before correlation use.

## Session-end verification checkpoint

The earlier 600.21-s Conv2D retry exposed a local integration bug rather than a
protocol-capacity limit: upstream Orca's `initGPUMemPool()` pre-reserves 25 GiB
per process, so two otherwise-small party processes could not coexist on the
available GPUs. The shared Ring-LPN engine now selects CUDA's default
asynchronous pool with maximum retention (`UINT64_MAX`) but no eager 25-GiB
allocation. Upstream Orca remains unchanged.

After rebuilding both shared-engine adapters with `GPU_ARCH=89`, the same
two-distinct-GPU controls were rerun:

```bash
cd GPU-MPC/ringlpn
P0_GPU=1 P1_GPU=3 CHECK_GPU=3 ./scripts/run_two_party_conv_preprocess.sh
# [two-party-conv] canonical live path and controls pass

P0_GPU=1 P1_GPU=3 CHECK_GPU=3 ./scripts/run_two_party_fc_preprocess.sh
# five q64/q128 regular/uniform/multi-batch rows and eleven controls pass
```

The 2026-08-07 SHAKE/scoped-vector focused Conv2D runner completes in about
1.5 s and the five-case FC runner in about 58 s on distinct local GPUs. A
q128/bw32 `conv0` run at `n=262144` is measured separately because it exercises
1,780 Ring-OLE instances and 455,680 DPF trees. Its breadth/pre-breadth party
metrics, record digests, and archive-time checker revalidations are retained
under `results/conv/conv0_breadth_comparison_2026_08_09/`; raw private records
are excluded. These are per-layer records, not full-model inference or
authenticated two-host evidence.

## Repository checkpoint

Implementation has advanced past the original manifest-only checkpoint:

- `src/two_party_linear_preprocess.cuh` owns the shared FC/Conv party protocol,
  checker, record, cost, freshness, and publication machinery.
- `src/test_two_party_{fc,conv}_preprocess.cu` are thin public-shape adapters.
- `scripts/check_full_linear_shape_coverage.py` validates all 20 ResNet18
  convolution contracts plus the classifier FC against the compiled adapters.
- `src/secure_truncate.{h,cpp}` and its two-process gate implement and validate
  the exact stochastic-truncation remasking functionality in isolation.
- `scripts/run_full_linear_record_set.py` validates all 21 ordered layer plans,
  launches two distinct local GPU processes per layer, checks both records with
  the unchanged per-layer Orca consumer, captures stable party/checker metrics,
  and publishes no record-set manifest unless every layer passes.

The adaptive execution manifest has embedded plan digest
`c637362d0496a7490837e5594bf34c8d7f0c9099310af42062bd84443883ce43`,
with 21 linear layers, 1,680,390,912 exact padding-aware cross terms, 6,439
Ring-LPN application batches, and 6,593,536 DPF trees. Its profile explicitly
records the independently scoped SHAKE256 public-vector random-oracle boundary.
The executable plan gate checks all 21 compiled adapter invocations only after
the binaries exist; the host-only source gate remains clean-clone safe.

Verified focused commands:

```bash
cd GPU-MPC/ringlpn
./scripts/run_full_linear_manifest_gate.sh
RUNNER_PLAN_CHECK=1 ./scripts/run_full_linear_manifest_gate.sh
./scripts/run_secure_truncate_test.sh
./scripts/run_two_party_conv_preprocess.sh
./scripts/run_two_party_fc_preprocess.sh
./scripts/run_resnet18_graph_prefix.sh
```

The isolated ordered runner still is not stock Orca ResNet18 execution: it
generates and independently checks forward-linear preprocessing records and
does not carry masked activations through the graph.
`results/fc/forward_linear_record_set_2026_08_07/` is a consumed incomplete
attempt containing only Conv0 party outputs; it has no checker row or final
manifest and must never be resumed or cited as a completed run.

The owner-approved state-mask seam is implemented in
`src/graph_mask_state.h` and exercised by
`src/test_resnet18_graph_prefix.cu`. At exact source-bound shapes, the known-zero
checkpoint binds each linear record to its party-local input/output mask,
invokes unchanged Conv2D consumers, and consumes the Conv0 output mask in live
secure truncation. A labelled TEST-ONLY trusted adapter reads both source mask
states and emits one exact stock MaxPool/ReLU key record per party; each live
party reads only its own records and invokes unchanged `gpuMaxPool` and
`gpuReluExtend`. The 2026-08-09 artifact reports 318.150-s and 297.313-s linear
preprocessing critical paths, 2.458-s trusted nonlinear key generation, and a
4.498-s maximum live checkpoint. Replay, corrupt-peer input, swapped order,
nonlinear-key corruption, stale output, forced partial publication, and
corrupt-output controls pass; no branch exists in this prefix, so the branch
control is explicitly not applicable. This closes the stock-format
branch-free composition seam, not dealerless nonlinear preprocessing,
private/trained inference, residual composition, deployment, or full inference.
The next seam is the three projection and five identity residual branches,
followed by `GlobalAvgPool2D`, classifier sign extension, terminal
reconstruction, and a fresh complete source-bound record/graph run. The
trusted nonlinear adapter must ultimately be replaced by a dealerless protocol.
Authenticated two-host repetitions, repeated graph-level measurements,
clean-clone reproduction, and independent human cryptographic review remain
open.
The pre-existing dirty external scratch submodules `GPU-MPC/ext/cutlass` and
`GPU-MPC/ringlpn/extern/NFLlib` remain outside scope.
