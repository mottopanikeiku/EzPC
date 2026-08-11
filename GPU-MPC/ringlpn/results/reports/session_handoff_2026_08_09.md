> **HISTORICAL (superseded 2026-08-10):** superseded by `CLAUDE.md`; statements below may describe the older prefix-only checkpoint.

# Ring-LPN successor handoff — 2026-08-09

**Status:** current main-agent handoff for advisor-level systems work. Read this file, `GPU-MPC/ringlpn/CLAUDE.md`, and `GPU-MPC/ringlpn/results/README.md` before acting.

**Owner direction:** pursue work worthy of a strong professor/advisor checkpoint, but this is not an autonomous project. Consult the owner before changing functionality, claim scope, publication route, author/credit decisions, or launching a long full-model experiment. Do not trade correctness or provenance for a superficially stronger result.

## 1. Bottom line

The repository now has a substantial dealer/oracle-free forward-linear
preprocessing artifact plus one narrow stock-nonlinear graph checkpoint:

- two party processes;
- SCI/IKNP or opt-in EMP-Silent OT;
- epoch-zero Gilboa OLE and later consume-once Ring-OLE-output Phase-C bootstrap;
- GPU-batched distributed DPF/SPFSS;
- jointly seeded, domain-separated SHAKE256 public Ring-LPN vectors in an explicit random-oracle model;
- exact `Z_M -> Z_2^bw` conversion;
- FC and Conv2D stock-format records consumed by unchanged online linear kernels;
- a source-bound 21-linear-layer ResNet18 plan;
- a two-party stochastic-truncation protocol;
- fail-closed invocation/correlation ledgers and record-set plan controls;
- bounded breadth-first prepared GPU DPF evaluation;
- digest-bound per-party graph mask-state records; and
- an exact known-zero Conv0→TR→MaxPool→ReLU→Conv3 checkpoint that invokes
  unchanged `gpuConv2DBeaver`, `gpuMaxPool`, and `gpuReluExtend`.

This is **not full ResNet18 execution**. The 21-record plan still independently
samples operand/output masks. The separate prefix checkpoint invokes live
secure truncation for all 802,816 Conv0 outputs and stock MaxPool/ReLU
consumers, but its exact nonlinear keys come from a TEST-ONLY trusted adapter
that reads both parties' source-bound mask states. Each live party reads only
its own digest-bound records. This closes the branch-free stock-format
composition seam; it is not dealerless nonlinear preprocessing, private or
trained inference, an accuracy result, residual-branch execution, deployment,
or a full 62-item stream.

The existing `results/fc/forward_linear_record_set_2026_08_07/` is a consumed,
incomplete attempt. It contains Conv0 party outputs but no Conv0 checker row, no
layers 2--21, and no `LINEAR_RECORD_SET.manifest`. Never resume it, reuse its
ledger, or cite it as a model run.

## 2. Exact current evidence

### 2.1 Source-bound plans

| Artifact | Observed state |
|---|---|
| `results/fc/resnet18_full_linear_execution_manifest_2026_08_06.json` | Baseline 21-layer source-bound plan; file SHA-256 `f00bc7ed8f14e06181e660f9a1c351cdb8cc186c639acbcfccada9d2bcf46771` |
| `results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json` | q128/bw32, `(n,c,t)=(262144,2,8)` for every linear layer; file SHA-256 `685501ddba417d597607403dbd9fe6c2954081d6edca161f6cef40c9e4be9427`; embedded plan digest `c637362d0496a7490837e5594bf34c8d7f0c9099310af42062bd84443883ce43` |
| Adaptive totals | 20 convolutions, one FC, 1,680,390,912 cross terms, 6,439 Ring-LPN batches, 25,756 Ring-OLE instances, 6,593,536 DPF trees, 130,774,336 final linear-key bytes per party |
| Explicit non-execution fields | The isolated manifest has 62 stock key items, 21 stochastic truncations, eight residual merges, and four graph-state nodes marked unexecuted. The later prefix checkpoint separately exercises two linear records, one live truncation, and stock MaxPool/ReLU consumers with a trusted test-only key source; it does not update or complete this manifest. |

### 2.2 Maximum Conv0 isolated run

The last accepted q128/bw32 Conv0 run used one adaptive point,
`n=262144,c=2,t=8`, and distinct party/check GPUs. Both producers passed, and
their digest-bound records were revalidated by the archive-time
`gpuConv2DBeaver` checker (binary SHA-256
`1db001a9e86df261cbba30900cd70c75140468f25377a8d15de2635273ebdab7`)
on 2026-08-09 before the privacy-sensitive records were deleted.

| Metric | Observed value |
|---|---:|
| Party critical path | 358.085 s |
| Ring-LPN batches | 445 |
| Ring-OLE instances | 1,780 |
| Ring application slots | 464,585,112 |
| DPF trees | 455,680 |
| DPF scalar OLEs | 1,367,040 |
| Final key payload per party | 7,702,016 bytes |
| Reported protocol bytes, P0 / P1 | 11,018,798,352 / 9,400,401,680 |
| Peak host RSS, P0 / P1 | 1,144,520,704 / 854,548,480 bytes |
| Archive-time matched dealer checker | 16,159.8 us |
| Archive-time unchanged online checker | 4,311.24 us |

The otherwise-matched pre-breadth run was 384.816 s. The prepared
breadth-first evaluator reduced this one-trial critical path by 6.946%. The
retained artifact is
`results/conv/conv0_breadth_comparison_2026_08_09/` (metadata SHA-256
`ed5d43ab48ae014eb7366ae7ceb1e9985e72176c2e3fd1495c0bfae3d2117d09`).
It contains both party metric rows, record hashes, and checker rows, but no raw
private key records. This is an isolated-layer engineering result, not a
full-model or publication performance distribution. The separate focused Conv
CSV remains `results/conv/two_party_conv_preprocess_2026_08_04.csv`, SHA-256
`6669ee20c227ddf143c2f1bf229ddf28d057400b453aa4c1b5c3138536af62ad`.

### 2.3 Classifier and component evidence

- `results/fc/two_party_fc_model_scale_2026_08_04.*`: regenerated v5 q128/bw32
  `1x512x1000`, `(8192,2,8)`, one warmup plus ten accepted SCI/IKNP trials.
  Median preprocessing 4.064 s; application traffic 182,372,344 bytes; matched
  stock trusted-dealer keygen 14.863 ms; unchanged online checker 1.152 ms;
  272x slower median of trial ratios. Per-layer CSV SHA-256
  `97941cadeb2a8571d873e6f04dcc16ac61c5bea4a11185cacb1a873e76db1f07`;
  summary SHA-256
  `430bd572dc34652b1437d48150d1861ca498027686405851712da8da45adc93d`.
  The artifact pins layer-manifest SHA-256
  `6bd5e24f5e18eb744458d4aaf86d42a823857bc134ac1d4be25c95d49cf3c984`
  and workload-manifest SHA-256
  `718570af4bf7afa523d029dfddcf0f3dae9e554c9c6e26b744999df193956f72`.
- `results/secure_truncate/secure_truncate_two_party_2026_08_03.csv`: isolated stochastic-truncation protocol passes. The dedicated graph-prefix control now calls the same live protocol with predecessor output-mask state; stock Orca's nonlinear path still does not.
- `results/dpf/`: ideal host protocol, two-process real OT/OLE key generation, four-call full-width GPU-AES parity, GPU evaluator, invalid-input and corruption controls.
- After the graph-state header changed the shared adapters, the focused FC and
  Conv suites were rebuilt and rerun on party GPUs 1/3 with checker GPU 2: all cases and controls passed
  against binaries
  `73b365066e3c0b12ed7d711505a29fe673a42f7ffbcd2f17c43c654c98fcdd0d`
  and
  `8834189756a0dc972e3877a94745c62311fa498410fb115b124dedff9ccd6d48`.
  Their refreshed binary-approval snapshot has self-digest
  `6cbf745682ba09aa4b29fad3fcf0d12d9aed42cd0f4df3dbc773bfb986be505f`
  and file SHA-256
  `ee39736e120edc2ec7e34c9c567d368f74ce035a522dc221962e4e2c97babaa5`;
  the full runner-plan control gate passes.
- `results/reports/dealerless_orca_fc_security_contract_2026_07_29.md`: conditional static-semi-honest source/transcript contract; SHA-256 `d436375ffce008829ccbc1fc152d9451308e28d4f5cb492b03fa97d130023ed1`.
- `results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.{tex,pdf}`:
  internal v2.12 report rebuilt under the pinned TeX Live 2023 image on
  2026-08-09. Final pass: 26 pages, no
  warnings/overfull/underfull/unresolved references, every rendered page
  inspected. TeX SHA-256
  `85f965bf8abff7991767b5ad62e86b1ce39130d2f3c7f8247ae38bef11bf6911`;
  PDF SHA-256
  `f9814c2721dace39939f5fc0a8af1f55a2bc7b083074e8e667f319f5ff872f3a`.
- The complete required-GPU checkpoint gate was re-run collision-free on GPU 2
  after the stock-nonlinear prefix checkpoint: exit 0, literal
  `[paper-smoke] ALL GATES PASS`, 553.58 s. Its compiler warnings are
  pre-existing upstream/third-party warnings or two unused host-reference
  helpers; every canonical stage executed. The long graph-prefix gate is
  separate and is represented by the artifact above.

### 2.4 First stock-nonlinear prefix checkpoint

`results/graph/resnet18_graph_prefix_*_2026_08_09.*` records one source-bound,
q128/bw32, `(n,c,t)=(262144,2,8)` known-zero checkpoint at the
Conv0→TR(10)→MaxPool→ReLU→Conv3 shapes. Independently generated Conv0 and Conv3
records pass unchanged `gpuConv2DBeaver`.
`secure_stochastic_truncate_batch` processes all 802,816 Conv0 outputs and
emits post-truncation shares plus fresh output-mask shares. A TEST-ONLY trusted
adapter reads both parties' source-bound mask states, invokes Orca's stock
trusted-dealer MaxPool/ReLU key generation, and emits one digest-bound private
record per party. Each live party reads only its own records and invokes
unchanged `gpuMaxPool` and `gpuReluExtend`.

| Measurement | Observed value |
|---|---:|
| Conv0 preprocessing critical path | 318.150 s |
| Conv3 preprocessing critical path | 297.313 s |
| Trusted stock nonlinear key generation | 2.458129 s |
| Live record-consuming checkpoint, max party | 4.497533 s |
| Live secure truncation, max party | 2.362976 s |
| Live MaxPool / ReLU, max party | 76.491 ms / 9.560 ms |
| Conv0 / Conv3 DPF trees | 455,680 / 442,368 |
| Conv0 / Conv3 final payload per party | 7,702,016 / 3,506,176 bytes |
| Truncated values | 802,816 |
| Truncation logical opened bits per party | 57,802,752 |
| Stock MaxPool / ReLU key bytes per party | 503,968,064 / 68,289,576 |
| Stock nonlinear traffic sent per party | 5,519,361 bytes |
| Total checkpoint protocol bytes sent, P0 / P1 | 600,999,564 / 446,858,892 |

This is one shared-machine engineering run, not a timing distribution. Replay,
corrupt-peer input, swapped layer order, corrupted nonlinear-key record, stale
nonlinear output, forced second-rename failure, and corrupted run-record
controls all reject. The forced-rename control leaves both outputs absent. The
residual-branch control is explicitly not applicable because this prefix has no
branch.

The nonlinear source is trusted and sees both mask states. Therefore this is
stock-format systems-composition evidence, not dealerless nonlinear
preprocessing, private/trained inference, accuracy, deployment, full-model
performance, or security evidence.

Key public hashes: graph binary
`b0480b7ae6555ba090d4fab9b9cb830a4cabea9af0e7eb42ebcc0cf70cad77c5`;
runtime source
`e9d626f248a209030c292dba07348aaa184cdf2d6581a4e7d81aa1a09e982a62`;
stock-keygen binary
`28577217b69bc13dc070147fce21c904fcd3a55bee5be3dbcf23cea8171b82f0`;
stock-keygen source
`64aa25bfdf1cfc7b1563fd1a7103ba1a19d3b16ae02dc40dbb503d469ed0766e`;
graph mask-state header
`da4d67b1c2c3f2ab86464e1554276172d5a0a81f713b76c204a89415b17b4e23`;
stock nonlinear-record header
`6249b3aadb21bb45bece5021281f96e2f224fb952f18781a3ada8caf0b573f0c`;
runner
`4f0c5bc7cf0a5fecea4c3092951444b539fc4106c20851cdbc132319f9236c0d`;
metadata
`f32303fe950f08d01006463850f1d090a95b493824299c0b563252e3f17c1f1c`;
artifact-hash list
`8ffb59a15af2d734c2ed5b272627c100f482f63f0d209590e5b371000055fa88`.
No private key, mask-state, or run records are retained.

## 3. Material implementation changes in this workstream

### Distributed DPF and transcript accounting

- `src/test_distributed_dpf_keygen.cpp`: corrected Phase-C payload generation; only standard public `finalCW` opens. Old-sign proper-subset regression proves the removed sign opening would reveal a point-dependent class.
- `src/two_party_dpf_protocol.h`, `src/two_party_ot.h`: shared live DPF protocol, consume-once OT/OLE correlations, separated logical opened bits and encoded meaningful-share bits.
- `src/test_two_party_dpf_keygen.cpp`, `src/test_two_party_spfss_keygen.cpp`: exact accounting and precondition gates.
- `src/dpf_key_io.h`: input validation and overflow-safe record sizing.

### Shared FC/Conv producer

- `src/two_party_linear_preprocess.cuh`: shared live producer, exact correlation planning, invocation/ledger binding, public-vector setup, Ring-OLE reserve/consume/discard accounting, conversion, record publication, peak memory and stage metrics.
- `src/test_two_party_fc_preprocess.cu`, `src/test_two_party_conv_preprocess.cu`: thin public-shape adapters and unchanged online checkers.
- `src/ringlpn_ole_party.cuh`: party-local Ring-OLE API and public-vector validation.

### Public Ring-LPN vector

The prior implementation generated and exchanged four raw seed words but then used `mt19937_64`; a deterministic expander is not automatically a cryptographic PRG. The current path:

1. exchanges four 64-bit seed-share words per party once per layer;
2. XOR-combines them into a 32-byte joint seed;
3. uses SHAKE256 with exact per-layer/per-direction/per-limb/per-ring-batch domain separation;
4. rejection-samples uniform field coefficients;
5. keeps `a0=1` implicit and unsent;
6. counts only four public seed words per party per layer.

Claim scope is explicitly the independent-public-vector Ring-LPN assumption in the SHAKE256 random-oracle model, with no concrete security claim.

### Prepared DPF evaluation

- `src/gpu_spfss_zp.cuh` prepares a contiguous regular-group descriptor batch.
- For at most `32 * 1024 * 1024` breadth states, it expands every tree level-wise into two buffers, then reduces leaves without atomics.
- Oversized batches retain the original exact per-point root-to-leaf evaluator.
- The Conv0 result above is the only current large-shape measurement of this optimization.
- Focused Ring-LPN GPU executables allocate only their explicit working
  buffers; the upstream 25-GiB eager `initGPUMemPool()` reservation was removed
  because it is not a Ring-LPN correctness prerequisite and made valid
  shared-GPU smoke runs fail before their first real allocation.

### Full-linear orchestration and truncation

- `scripts/build_full_linear_model_manifest.py`
- `scripts/check_full_linear_shape_coverage.py`
- `scripts/run_full_linear_manifest_gate.sh`
- `scripts/run_full_linear_record_set.py`
- `scripts/run_secure_truncate_test.sh`
- `src/secure_truncate.h`, `src/secure_truncate.cpp`, `src/test_secure_truncate.cpp`

The record-set runner requires an owner-approved, hash-bound binary approval file and rejects stale source/binary plans. An approval binds only the current FC/Conv adapter binaries; rebuilds require regeneration of the approval.

### Stock-nonlinear graph prefix

- `src/graph_mask_state.h`: fixed-size, versioned party-local mask records with
  SHA-256 self-digest and exact invocation/layer/shape/record binding.
- `src/stock_nonlinear_prefix_record.h`: bounded, versioned, digest-bound
  per-party MaxPool/ReLU key record tied to both source states and the exact
  known-zero scope.
- `src/test_stock_nonlinear_prefix_keygen.cu`: TEST-ONLY trusted adapter that
  reads both mask states, invokes stock nonlinear key generation, and
  bilaterally publishes one record per party.
- `src/test_resnet18_graph_prefix.cu`: party-separated prefix runtime and
  post-exit checker. It consumes Conv0/Conv3, mask-state, and nonlinear records
  once; invokes live truncation and unchanged Conv2D/MaxPool/ReLU consumers;
  fail-closes paired run/state publication, and validates the exact known-zero
  composition.
- `scripts/build_resnet18_graph_prefix.sh`,
  `scripts/run_resnet18_graph_prefix.sh`: fail-closed build/run gate and
  publication of public summary, party/linear/keygen metrics, controls, hashes,
  and metadata.

## 4. Stock-nonlinear prefix checkpoint and next blocker

The owner-approved per-party graph mask-state seam now reaches the first
branch-free source prefix. Each linear preprocessing record is paired with
digest-bound private input/output mask shares; the runtime verifies the pair
and uses those masks to form/check the public masked Conv2D inputs and outputs.
Secure truncation consumes the Conv0 output mask and produces private
post-truncation shares plus fresh output-mask shares.

The TEST-ONLY trusted adapter then generates exact stock MaxPool/ReLU key
streams and a remask delta to the independently generated Conv3 input mask.
Each live party reads only its own nonlinear record and executes Orca's
unchanged stock consumers. Preflight and publication controls reject
missing/stale state, wrong order, peer metadata mismatch, replay, corrupt input
or nonlinear key material, stale output, forced partial publication, and
corrupt final records.

The remaining graph blocker is residual and full-stream composition. Extend the
same invariant through the three projection and five identity residual
branches, `GlobalAvgPool2D`, classifier sign extension, and terminal
reconstruction with exact order/branch/freshness controls. The trusted
nonlinear adapter must be replaced by a dealerless protocol before making a
dealerless-model claim.

Exact stock order remains 62 items:

- Conv0: `I,F,O`;
- 20 remaining linear layers: `A/B` or `F`, then `C`;
- 21 stochastic truncation keys;
- `GlobalAvgPool2D`;
- FC: `A,B,C`.

Do not permit filename order, layer-name order, the known-zero reference, or a
concatenation of isolated records to stand in for this stream.

## 5. Stock and external baseline boundary

Canonical stock ResNet18 graph command, source-grounded from `GPU-MPC/nn/orca`:

```bash
cd /home/fatih/EzPC/GPU-MPC/experiments/orca
mkdir -p /tmp/P0_keys /tmp/P1_keys output/P0/inference output/P1/inference
# Run simultaneously in two terminals or a supervised launcher:
CUDA_VISIBLE_DEVICES=0 ./orca_inference_u32 ResNet18 32 10 0 0 /tmp/P0_keys
CUDA_VISIBLE_DEVICES=2 ./orca_inference_u32 ResNet18 32 10 1 0 /tmp/P1_keys
```

This exercises stock graph choreography and all 62 stock keys, but current `orca_inference.cu` zero-initializes input and weights. It is not trained ImageNet inference or an accuracy baseline.

Reverse Cuckoo/libOTe and the native-ring artifacts are primitive baselines with different fields, layouts, factor ownership, setup, GPU, and graph semantics. Their rows are useful diagnostics; do not compute a model speedup against them.

## 6. Security and publication hard stops

Never claim:

- 64- or 128-bit security from q64/q128;
- a pinned `(n,c,t,p0,p1)` parameter set;
- a reviewed structured-code reduction;
- authenticated deployment from loopback TCP;
- a complete 21-record set from the incomplete directory;
- full ResNet18, trained inference, or accuracy;
- full stock graph execution from isolated linear or prefix-checkpoint passes;
- dealerless nonlinear preprocessing from the trusted test-only adapter;
- publication or conference readiness; or
- a performance win over stock dealer key generation.

Still required for security/publication:

1. independently reviewed structured projected-code reduction and exact attack model;
2. two-CRT-limb and all-hybrid advantage composition;
3. qualified human proof/source review;
4. residual/full key-stream composition and a dealerless nonlinear protocol;
5. fresh complete model evidence and honest stock/compatible baselines;
6. authenticated two-host LAN/WAN evidence;
7. clean-clone reproduction;
8. a new warning-free PDF build and rendered-page inspection after any later
   evidence or manuscript change;
9. further algorithmic Phase-B, round-count, and end-to-end performance work.

## 7. Immediate next-agent procedure

1. Read `CLAUDE.md`, `results/README.md`, this handoff, the full-linear plan,
   security contract, and v2.12 TeX source.
2. Inventory current workspace changes. Treat unexpected edits as owner work;
   never reset or delete them.
3. Treat `results/fc/forward_linear_record_set_2026_08_07/` and its ledger as
   consumed evidence of an interrupted attempt only.
4. Verify the public graph-prefix artifact hashes, run the focused prefix gate,
   and preserve its exact known-zero/trusted-nonlinear-source boundary.
5. Verify current adapter hashes against
   `results/fc/linear_adapter_binary_approval_2026_08_07.json`; regenerate the
   owner-approved snapshot after any rebuild.
6. Run focused manifest, truncation, FC, and Conv gates with fresh invocation
   IDs and unused ports, then the canonical required-GPU gate on a GPU with
   enough free memory. Another process occupying a GPU or TCP port is an
   environment failure, not a PASS.
7. Consult the owner before extending graph functionality or launching a full
   record/model run. The next recommended implementation is exact residual-
   branch state/key composition, not 21 independent records.
8. Add branch, stale-output, partial-publication, order, replay, corrupt-key,
   and corrupt-peer controls before any full-stream claim.
9. Rebuild the PDF warning-free and inspect every rendered page after evidence
   stabilizes.

## 8. Owner decisions after the accepted stock-nonlinear prefix

The graph-level per-party mask-state record and first known-zero
stock-nonlinear prefix checkpoint are now accepted repository facts. Before the
next functionality expansion, ask together:

1. whether to close all residual branches first with the current labelled
   trusted compatibility source or prioritize a dealerless nonlinear-key
   protocol before further graph work;
2. whether trained weights/data are required immediately after full graph
   choreography, or whether the zero harness should first close all 62 items;
   and
3. whether to remain an internal advisor artifact until independent security
   review or prepare a systems-paper evaluation while retaining the explicit
   no-concrete-security boundary.

## 9. Verification commands

From `GPU-MPC/ringlpn`, verify the current approved snapshot without rebuilding
the adapters:

```bash
RUNNER_PLAN_CHECK=1 PATH=/usr/local/cuda/bin:$PATH \
  ./scripts/run_full_linear_manifest_gate.sh
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=<free-gpu> \
  PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh
```

The canonical gate deliberately validates, but does not rebuild, the
hash-approved FC/Conv adapters. The focused FC runner rebuilds FC; the graph
runner rebuilds Conv plus the stock-nonlinear compatibility adapter. Therefore
an intentional adapter/graph refresh must use this order:

```bash
PATH=/usr/local/cuda/bin:$PATH ./scripts/run_two_party_fc_preprocess.sh
PATH=/usr/local/cuda/bin:$PATH P0_GPU=<gpu0> P1_GPU=<gpu1> CHECK_GPU=<gpu2> \
  ./scripts/run_resnet18_graph_prefix.sh
# Validate the final Conv binary produced by the graph runner.
PATH=/usr/local/cuda/bin:$PATH ./scripts/run_two_party_conv_preprocess.sh
# STOP: refresh linear_adapter_binary_approval_2026_08_07.json only with owner approval.
RUNNER_PLAN_CHECK=1 PATH=/usr/local/cuda/bin:$PATH \
  ./scripts/run_full_linear_manifest_gate.sh
```

If the graph runner is not part of a refresh, run
`scripts/build_two_party_conv_preprocess.sh` immediately before the focused
Conv runner instead. Any adapter rebuild invalidates the previous approval;
the hardened runner-plan gate rejects that state before a long record-set
launch.

Canonical success is only exit zero plus literal `[paper-smoke] ALL GATES PASS`.

Manifest/adaptive validation:

```bash
./scripts/build_full_linear_model_manifest.py \
  --layer-manifest results/fc/orca_forward_linear_layer_manifest_2026_08_04.json \
  --model ResNet18 --ole-n 262144 \
  --out results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json
./scripts/check_full_linear_shape_coverage.py \
  --manifest results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json \
  --bin-dir bin
```

Do not run `run_full_linear_record_set.py` without a fresh output root, a fresh private ledger root, current binary approval, free GPUs, and owner approval for the cost.

## 10. SSH-independent successor launch

Run from the host, not inside the Orca container. `tmux` survives SSH loss and does not attach the successor to the current OMP session:

```bash
tmux new-session -d -s ringlpn-successor \
  "cd /home/fatih/EzPC && exec env PATH=/usr/local/cuda/bin:\$PATH \
   omp --model openai-codex/gpt-5.6-sol \
   'Read GPU-MPC/ringlpn/results/reports/session_handoff_2026_08_09.md, GPU-MPC/ringlpn/CLAUDE.md, and GPU-MPC/ringlpn/results/README.md in full. You are the successor main agent, independent of the prior session. Verify the documented checkpoint and inventory only. Preserve the accepted known-zero stock-nonlinear prefix and its trusted-adapter boundary; do not extend residual/full-graph semantics or launch a full record/model run without owner consultation. Keep unexpected workspace changes.'"
```

Observe later with:

```bash
tmux attach -t ringlpn-successor
```

At the start of the 2026-08-09 handoff validation, unrelated `vllm` processes
occupied substantial memory on every GPU and an early GPU-3 canonical attempt
failed with `cudaErrorMemoryAllocation`; always re-check current occupancy
rather than assuming that snapshot still holds.
A later default-port canonical attempt reached secure truncation but hit
`bind: Address already in use`; it was an environment failure and is not
counted as protocol evidence. The successful 553.58-s rerun used disjoint
`BASE_PORT=56200`, `TWO_PARTY_BASE_PORT=56300`,
`TWO_PARTY_GPU_BASE_PORT=56400`, and `OLE_TWO_PARTY_BASE_PORT=56500`.
