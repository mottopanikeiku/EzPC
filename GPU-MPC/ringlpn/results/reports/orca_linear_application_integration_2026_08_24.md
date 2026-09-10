# Source-native Orca terminal linear integration — 2026-08-24

## Decision

Gap 1 is closed for terminal FC and Conv2D layers. The stable linear facade now
opens one party-local version-3 linear record together with its bound mask-state
record as a move-only `OwnedLayerMaterial`. A macro-gated role-2 branch in the
real `experiments/orca/orca_inference.cu` consumes that material through a
one-layer Sytorch module and Orca's stock `gpuMatmulBeaver` or
`gpuConv2DBeaver` execution path.

This is deliberately terminal-only. Sytorch disables post-linear truncation for
a last linear node; the backend rejects truncation, nonlinear, residual,
second-linear, and premature/duplicate-output callbacks. Arbitrary multi-layer
mask-state chaining remains open.

## Engineering review supplement — 2026-09-10

The actual source-native FC/Conv gate passes again, including signed and
wrapping arithmetic checked by independent clear oracles. Before/after
application probes demonstrate that a same-sized substituted GPU output
allocation and an inherited standalone add-bias callback were previously
accepted; both now reject bilaterally. A private FIFO input also rejects
without blocking or publishing output. The loader applies the same nonblocking
regular-file policy to record/state files and checks file identity/timestamps
around reads; private input/weight words are scrubbed on exceptional exits.

Stock `OrcaBase` online timing again encloses key parsing, weight transfer,
kernel execution, and weight release. The terminal backend owns its enclosing
timers, including masked-share reconstruction, rather than changing the stock
baseline's measurement scope. Conv key sizes come from the validated plan,
which rejects coordinates outside the stock signed-integer ABI.

The root-distinct build check exposed checkout-dependent CUDA fatbins despite
identical normalized source/library inputs. The application recipe now uses
the same private fixed-source build convention as the FC/Conv adapters.
Dependency-only preprocessing uses the equivalent physical view, so the
unchanged provenance collector still rejects symlinked source dependencies.
The main and source-only worktrees now produce byte-identical application and
helper executables and complete provenance. Both API probes and both
post-cleanup provenance verifiers pass. Current application/helper SHA-256:

- `f00f7b89db5156426eca1df28629a0572addf552339ae83c0648701e538f7885`;
- `5f702062a3b999451abdff39d3e90a3db6a0769a20aecb924e68f9b4ef5e52b3`;
- build-provenance digest:
  `f2d16c7578f1ca6fd697c69b7702ffa49ab7327c3502b0d6d7dfe9b2c089ff0b`.

All ten compiler dependency groups, environment inputs, and linked archives
match the pre-fix build; only the build recipe changed. The August bindings
below remain historical and are not silently rebound to these executables.

The producer's consume-once ledger is **not** persistent application-consumer
replay prevention. The terminal API still requires fresh, single-use material
from its caller; reloading a record in a later process is not prevented by
the move-only C++ object. Public bias remains party-0-authoritative, and
terminal-output publication is not a bilateral durable transaction. These
boundaries remain explicit; this review does not claim a production
multi-invocation service or private/trained-model inference.

## Public material contract

`linear_preprocess.h` exposes `LayerExpectation` and `OwnedLayerMaterial` plus
kind-specific `open_and_validate_layer_material` entrypoints. The loader:

- accepts only owner-owned, single-link, regular, non-symlink `0600` record and
  state files;
- distinguishes I/O-policy failures, malformed/corrupt encodings, and valid but
  mismatched material;
- checks kind, derived plan, party, SID, invocation, layer ordinal, expected
  record/state self-digests, state-to-record digest binding, layer identity,
  widths, counts, canonical ring words, and byte-identical input-mask shares;
- adopts neither object unless every check passes; and
- scrubs record payload and both state vectors on failure, reset, move
  assignment, and destruction.

The public API regression loads FC and Conv2D party material sequentially and
covers wrong kind, shape, bit width, party, SID, invocation, ordinal, expected
digests, swapped state, state/record digest mismatch, state/input mismatch,
truncation, corruption, noncanonical words, symlink substitution, and file-mode
rejection. Every rejection leaves the destination empty.

## Orca execution path

`OrcaBase` now factors its stock flat-key consumers into protected
`runMatmulWithKey` and `runConv2DWithKey` helpers. Stock methods retain their
flat-buffer parsing and transfer the online weight to GPU before delegation.
The helper regression generates trusted stock material with
`gpuKeygenMatmul`/`gpuKeygenConv2D` and checks the extracted helpers against
independent clear modulo-$2^{32}$ FC and Conv2D oracles.

`TerminalLinearBackend<uint64_t>` owns exactly one bound material pair. It:

1. exchanges a fixed-width bilateral preflight containing local validity, kind,
   plan, SID, invocation, ordinal, layer identity, and ledger digest;
2. adds $A_i$ to the local input share and reconstructs the public masked input;
3. adds $B_i$ to the local weight share on GPU and reconstructs the public
   masked weight without a GPU-to-CPU round trip;
4. constructs stock key views directly over $A_i || B_i || C_i$ and calls the
   shared Orca helper; and
5. reveals the clear terminal result from shares `masked_output - Y_0` and
   `-Y_1`, never reconstructing the mask separately.

Public nonzero bias follows Orca's existing party-0 convention. The historical
`ORCA_RINGLPN_FC_KEYS` keywriter path is not built or called.

## Source-native application gate

`run_orca_linear_application.py` creates fresh private q128/bw32 regular
fixtures for:

- FC `(rows,inner,cols)=(2,3,2)`; and
- Conv2D `(N,H,W,CI)=(1,4,4,1)`, filter `3x3x1x2`, padding 1, stride 1.

Clear inputs, weights, and public biases are all nonzero. Input and weight
shares are independently sampled additive shares. Each role reads only its own
record, state, share, bias, ledger-derived metadata, and output path. Temporary
records, states, ledgers, authentication keys, shares, outputs, and logs are
under an owner-only root and are deleted after public rows are finalized. An
explicit caller-supplied new private work root is the only retention mode.

The retained public gate log reports:

```text
linear-library-api,ok
ringlpn-orca-linear-application,fc,pass
ringlpn-orca-linear-application,conv2d,pass
ringlpn-orca-linear-control,wrong-kind,pass
ringlpn-orca-linear-control,wrong-shape,pass
ringlpn-orca-linear-control,wrong-ordinal,pass
ringlpn-orca-linear-control,swapped-state,pass
ringlpn-orca-linear-control,corrupt-state,pass
ringlpn-orca-linear-control,permissive-share-file,pass
ringlpn-orca-linear-control,duplicate-output,pass
ringlpn-orca-linear-control,one-sided-metadata-mismatch,pass
ORCA LINEAR APPLICATION PASS
```

Every control made both application processes exit nonzero with the bilateral
preflight rejection marker and no newly published output. The duplicate-output
control preserved its deliberate preexisting sentinel and published nothing on
the peer.

## Build and provenance

The discoverable component is:

```bash
PATH=/usr/local/cuda/bin:$PATH GPU_ARCH=89 \
  ./scripts/build_component.sh orca-linear-application
```

It source-builds LLAMA, cryptoTools, and bitpack; builds the deterministic
linear facade and FC/Conv adapters; compiles the real Orca inference source with
`ORCA_RINGLPN_LINEAR_INTEGRATION=1`; and builds the helper regression. The
application-specific provenance profile reuses the full-graph provenance
machinery without changing the pinned full-graph provenance implementation.

Final bindings from the validated tree:

- `orca_inference_ringlpn` SHA-256:
  `bfd124ecf2417126ef9d6fe497a6624fa2c506eb9a3929aea531170be8b387ab`;
- `test_orca_linear_helpers` SHA-256:
  `cb103065d9b73f9f4e3aef0a9b296d6052bc5c1713c22a5354d6238fad0ed436`;
- application build-provenance digest:
  `ed891d6618e6edd508c210a6a708f4b9228a68450e30528096b7da9da2f47f72`;
- refreshed deterministic linear-adapter approval digest:
  `2764ac2ae7f37576f0ed107584f2109a0811f4652a5d1be39f71ea067610a1d1`;
- refreshed full-graph build-provenance digest:
  `c1ffc4335538e409b155b4e0038b0110d379bf4c8752ee6a50e1d8b1af80bc25`;
- refreshed full-graph approval digest:
  `7c21b8891343b33a1633bec1cb7e6803c5ef693fc011bb5ba7742aa7bbb982f3`;
- adaptive source-manifest SHA-256:
  `cecb699ab73e2348b4c733bc627ded76a6cf1e81dccd406fa62dd4180902e4d3`.

The macro-off stock application was independently rebuilt by `make GPU_ARCH=89
orca_inference`; its link command contains no Ring-LPN archive.

## Complete checkpoint

The complete canonical checkpoint was run with all four GPUs visible. Existing
unrelated five-day workloads occupied GPUs 0 and 2, so focused/default work and
both live preprocessing parties used GPUs 1 and 3; the post-party full-graph
checker used GPU 0. This occupancy makes the run correctness evidence only.
The gate retained every existing component and full-graph check, included the
new application markers, reported fresh ephemeral full-graph digest
`ec026fa850dfca7b3f51fd7eaef1e729b17a82d03464976a63c662a662a7b410`,
and ended:

```text
[paper-smoke] ALL GATES PASS
```

The private full-graph output was temporary and was not retained.

## Reproduction

```bash
cd GPU-MPC/ringlpn
PATH=/usr/local/cuda/bin:$PATH GPU_ARCH=89 \
  ./scripts/build_component.sh orca-linear-application
P0_GPU=0 P1_GPU=1 PATH=/usr/local/cuda/bin:$PATH \
  ./scripts/run_orca_linear_application.sh
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PATH=/usr/local/cuda/bin:$PATH \
  ./scripts/run_paper_checkpoint_smoke.sh
```

GPU assignments must be changed, without overlap, when those devices have
resident workloads.

## Claim boundary

This is nonzero same-host functional integration for one terminal FC or Conv2D
layer with an explicitly public bias. It is not trained-model or accuracy
evidence, arbitrary multi-layer dispatch, dealerless nonlinear preprocessing,
an authenticated deployment, a performance distribution, a concrete Ring-LPN
security level, or a new cryptographic claim.
