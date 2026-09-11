# Chief-of-staff handoff for the next model — 2026-09-10

## Mission

Act as the repository's engineering chief of staff. Review all work already in
the worktree, audit repository-wide integration and claim boundaries, correct
every evidence-backed defect, execute the applicable gates, and leave a clear,
frequent, atomic Git history.

“Review the whole repository” does not mean restyling vendored or upstream code.
It means accounting for every dirty path, every changed public contract, every
relevant caller, every build and evidence dependency, and every repository area
that can invalidate the active Ring-LPN work. The active implementation is
`GPU-MPC/ringlpn/`; upstream EzPC/Orca/Sytorch code is read-only except for the
explicit integration points already modified or a demonstrated cross-repository
correctness defect.

Correctness and evidence come before commit count. Commit often at coherent
boundaries, not after arbitrary file edits. A commit is valid only when its
scope is understood, its applicable check has passed, and its documentation and
provenance are synchronized.

## Bootstrap prompt

Use the following as the first instruction to the new model:

> You are the engineering chief of staff for this EzPC worktree. Read
> `/home/fatih/EzPC/AGENTS.md`, `GPU-MPC/ringlpn/CLAUDE.md`,
> `GPU-MPC/ringlpn/results/README.md`, this handoff, and
> `GPU-MPC/ringlpn/results/reports/orca_linear_application_integration_2026_08_24.md`
> before acting. Audit every dirty path and all relevant callsites across the
> repository. Preserve retained evidence, consumed state, private data, user
> work, and deliberate submodule state. Fix root causes, not symptoms. Use LSP
> references before changing exported symbols when a server is available.
> Validate each coherent slice, then commit it atomically with path- or
> hunk-level staging. Never sweep unrelated work into a commit. Never reset,
> clean, rebase, force-push, or publish unapproved internal history/evidence.
> Maintain a commit ledger and finish with exact findings, commands, outputs,
> commit hashes, and any remaining blockers. The authorized source-publication
> boundary is recorded below; never substitute an internal `master` push.

## Authority and reading order

Read in this order:

1. `/home/fatih/EzPC/AGENTS.md` — repository ownership and container rules.
2. `GPU-MPC/ringlpn/CLAUDE.md` — canonical live status, source map, claims,
   gotchas, and documentation contract.
3. `GPU-MPC/ringlpn/results/README.md` — evidence and report index.
4. `GPU-MPC/ringlpn/results/reports/orca_linear_application_integration_2026_08_24.md`
   — terminal record/state integration design, verification, and hashes.
5. `GPU-MPC/ringlpn/results/reports/publication_readiness_plan_2026_07_21.md`
   — binding stage and claim gates.
6. `GPU-MPC/ringlpn/results/reports/dealerless_orca_fc_security_contract_2026_07_29.md`
   — functionality, transcript, leakage, and proof boundary.
7. `GPU-MPC/ringlpn/scripts/publication_environment_manifest_2026_08_10.json`
   — environment and retained-evidence bindings.

If these sources disagree, do not choose the convenient statement. Establish
which statement is newer and gate-backed, correct all live documents in the
same slice, and mark superseded material explicitly. Historical checkpoint
bytes are evidence and must not be edited in place.

## Completed review and current publication boundary

The engineering review has replaced the pre-review work inventory below.
Those inventories and reusable process instructions are historical context,
not a request to repeat completed work without a relevant change.

- Local source commit: `051e7f0e48f4901b6c12911a583eaf015b2da20b`.
- Local paper/evidence commit: `325d54b5986f186ba125f5f3c4d0e5187fd03583`.
- Exact regression observations and final runtime checks are indexed by
  `engineering_review_verification_2026_09_10.json`; deterministic paper
  builds and all-page visual inspection are recorded in
  `paper_review_verification_2026_09_10.json`.
- The final required-GPU checkpoint completes with `ALL GATES PASS` in
  5,228.406 s, including the source-bound known-zero full graph and seven
  graph controls. The subsequent public-worktree component rebuild,
  terminal FC/Conv gate with eight controls, and actual q64/bw29/inner-3
  FC/Conv online checks all pass. The application build-only relocation
  correction is committed locally as `1f42430d2b8b1aa89d0bf3983ae6abf69fae109b`
  and publicly as `a1f006af7357d9a0bc937d26da6c83be0a6793c6`; complete
  provenance remains identical across checkout roots.
- The fresh `../graph/resnet18_full_graph_review_2026_09_10/` bundle is
  separate internal evidence, not a replacement publication pin. Its
  nonlinear material is still TEST-ONLY trusted, and its linear manifest
  retains non-secret private-ledger path metadata. Original August evidence
  remains byte-immutable. Completed private fixtures and throwaway probes
  were removed after verification.
- `../fc/sci_duplex_worker_review_2026_09_10.json` retains the counterbalanced
  one-shape SCI experiment: twenty measured invocations pass, all 68
  per-party contract/accounting fields agree, and process-latency median
  falls from 1.0413 s to 0.9652 s. This is a scoped scheduling improvement,
  not a matched-security dealerless or full-model result.
- New commits use sole Git author `mottopanikeiku`. The public-safe branch
  `ringlpn/clinical-review-2026-09-10` starts at public `origin/master`
  `1433e0a81cc73a23909726092804c3eb33148e02`. It excludes the unpublished
  internal checkpoint ancestry, manuscript, historical measurements, binary
  approvals, and private records. Its own README gives the source-only
  execution contract; do not copy internal release inputs into it.
- The user authorized this source publication, but no push succeeded:
  HTTPS askpass stalled, noninteractive Git reported a missing username,
  and no token, SSH agent, or SSH private key was available. Authenticate
  the host's Git credential helper, then push only this branch:

  ```bash
  git push origin \
    refs/heads/ringlpn/clinical-review-2026-09-10:refs/heads/ringlpn/clinical-review-2026-09-10
  ```

- The paper remains **not submission-ready**. The updated publication plan
  names the unclosed independent cryptographic/ownership review, exact
  parameter pin, matched-security dealerless baseline, held-out cost-model
  and hardware evidence, authenticated distinct-host runs, and authorized
  release/disclosure gates. Do not turn engineering passes into any of
  these research claims.

## Pre-review repository snapshot (historical)

Snapshot before creating this handoff:

- repository root: `/home/fatih/EzPC`;
- active branch: `master`;
- HEAD: `6466e64` (`ringlpn: finalize internal advisor checkpoint`);
- relationship: `master...upstream/master [ahead 57]`;
- index: no staged content;
- worktree: 47 unstaged or intent-to-add paths and 5 untracked paths, plus
  deliberate dirty/uninitialized nested repositories.

Re-run `git status --short --branch` before relying on this snapshot. The
entries displayed as added application files were registered with
intent-to-add; the prior status reported zero staged content. Confirm with
`git diff --cached --stat` before the first commit.

Never use `git reset`, `git checkout`, `git clean`, or an indiscriminate
`git restore` to recreate a clean baseline. The dirty tree contains multiple
valid workstreams and shared-document edits.

## Pre-review dirty-work classification (historical)

This handoff itself and its index hunks in `CLAUDE.md` and
`results/README.md` are documentation-only work for the next-model transition.
Keep them in a dedicated handoff/documentation commit unless a later review
updates their factual snapshot.

### Terminal Orca linear integration — review as one dependency graph

Core/API:

- `GPU-MPC/ringlpn/src/linear_preprocess.h`
- `GPU-MPC/ringlpn/src/linear_preprocess_backend.cuh`
- `GPU-MPC/ringlpn/src/test_linear_preprocess_api.cpp`

Orca execution and application:

- `GPU-MPC/backend/orca_base.h`
- `GPU-MPC/experiments/orca/orca_inference.cu`
- `GPU-MPC/ringlpn/src/orca_terminal_linear_backend.cuh`
- `GPU-MPC/ringlpn/src/orca_linear_application_entry.cuh`
- `GPU-MPC/ringlpn/src/test_orca_linear_helpers.cu`

Build, runner, provenance, and gate:

- `GPU-MPC/ringlpn/scripts/build_component.sh`
- `GPU-MPC/ringlpn/scripts/build_orca_linear_application.sh`
- `GPU-MPC/ringlpn/scripts/orca_linear_application_build_provenance.py`
- `GPU-MPC/ringlpn/scripts/run_orca_linear_application.py`
- `GPU-MPC/ringlpn/scripts/run_orca_linear_application.sh`
- `GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh`

Source bindings and refreshed approvals:

- `GPU-MPC/ringlpn/results/fc/orca_forward_linear_layer_manifest_2026_08_04.json`
- `GPU-MPC/ringlpn/results/fc/resnet18_full_linear_execution_manifest_2026_08_06.json`
- `GPU-MPC/ringlpn/results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json`
- `GPU-MPC/ringlpn/scripts/check_resnet18_graph_contract.py`
- `GPU-MPC/ringlpn/scripts/run_resnet18_full_graph.py`
- `GPU-MPC/ringlpn/results/fc/linear_adapter_build_provenance_2026_08_10.json`
- `GPU-MPC/ringlpn/results/fc/linear_adapter_binary_approval_2026_08_07.json`
- `GPU-MPC/ringlpn/results/fc/resnet18_full_graph_binary_approval_2026_08_10.json`
- the three `resnet18_full_graph_*_receipt_2026_08_10.json` files.

Retained public evidence and documentation:

- `GPU-MPC/ringlpn/results/application/orca_linear_application_2026_08_24.csv`
- `GPU-MPC/ringlpn/results/application/orca_linear_application_2026_08_24.log`
- `GPU-MPC/ringlpn/results/application/orca_linear_application_build_provenance_2026_08_24.json`
- `GPU-MPC/ringlpn/results/reports/orca_linear_application_integration_2026_08_24.md`
- relevant hunks in `GPU-MPC/ringlpn/CLAUDE.md`,
  `GPU-MPC/ringlpn/results/README.md`,
  `dealerless_orca_fc_security_contract_2026_07_29.md`,
  `publication_readiness_plan_2026_07_21.md`, and
  `scripts/publication_environment_manifest_2026_08_10.json`.

### Earlier EMP-Silent, model-scale, publication, and manuscript work

These paths were already dirty before the terminal integration and must be
audited as separate workstreams, not silently absorbed into an Orca-linear
commit:

- `ringlpn/src/two_party_ot.h`
- `ringlpn/src/two_party_linear_preprocess.cuh`
- `ringlpn/src/test_emp_silent_loopback.cpp`
- `ringlpn/scripts/run_two_party_fc_preprocess.sh`
- `ringlpn/scripts/run_two_party_fc_model_scale.sh`
- `ringlpn/scripts/verify_emp_silent_fc_evidence.py` (untracked)
- `ringlpn/scripts/verify_controlled_fc_ab.py` (untracked)
- `ringlpn/results/fc/two_party_fc_emp_silent_correctness_2026_08_14.log`
  (untracked)
- `ringlpn/results/fc/two_party_fc_model_scale_cnn2_cnn3_2026_08_14.log`
  (untracked)
- `ringlpn/results/reports/dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex`
  (untracked)
- `ringlpn/scripts/reproduce_publication.sh`
- `ringlpn/scripts/run_two_host_authenticated.sh`
- `ringlpn/results/reports/authenticated_two_host_deployment_2026_08_04.md`
- `ringlpn/results/reports/publication_portfolio_2026_08_04.md`
- the modified v2 predecessor `.tex` and `.pdf`.

The authoritative docs and some manifests contain hunks from both workstreams.
Use hunk staging and inspect every staged line.

### Deliberate nested-repository state — do not normalize

- `GPU-MPC/ext/cutlass`
- `GPU-MPC/ringlpn/extern/NFLlib`
- `GPU-MPC/experiments/orca/datasets/mnist`
- `GPU-MPC/experiments/orca/weights`

`CLAUDE.md` explicitly says the dataset/weight submodules carry deliberate
internal renames. Determine whether each `?`/`m` is expected, but do not update,
initialize, clean, or commit a gitlink merely to make status look clean.

## What the terminal integration is supposed to guarantee

These remain the integration invariants. Use the completed review evidence
above and re-establish affected invariants after subsequent changes:

1. `OwnedLayerMaterial` is move-only and atomically adopts one party's linear
   record and mask state only after all checks pass.
2. Record/state files are owner-private regular non-symlinks; malformed
   encodings and noncanonical words are corruption, while valid-but-wrong
   bindings are mismatches.
3. Record payload and both state vectors are scrubbed on failure, reset, move
   assignment, and destruction, including parser-local failure paths.
4. State widths/counts match the derived plan; state record digest, layer
   identity, invocation, ordinal, party, SID, and input-mask bytes bind exactly.
5. `OrcaBase` flat-key parsing and stock semantics remain unchanged after helper
   extraction; callers retain online weight/filter ownership.
6. The Ring-LPN backend performs exactly one matching terminal matmul or Conv2D
   callback followed by exactly one output. It rejects truncation, nonlinear,
   residual, second-linear, premature-output, and duplicate-output paths.
7. Input and weight shares are masked and reconstructed on GPU; key views point
   directly at owned material; there is no GPU-to-CPU-to-GPU payload path.
8. Output reveals additive clear-output shares without separately revealing
   `Y_0` or `Y_1`.
9. Both parties exchange the same fixed-width validity/binding preflight and
   reject together instead of hanging when only one side is invalid.
10. Macro-off `orca_inference` does not include or link Ring-LPN and roles 0/1
    retain their source-bound behavior.
11. The historical `ORCA_RINGLPN_FC_KEYS` keywriter is not used by this path.
12. Temporary records, state, shares, auth material, ledgers, outputs, and logs
    are owner-only and deleted unless an explicit private debug root is supplied.
13. Application and helper builds are repeatable after path, build-id,
    random-seed, and NVCC temporary-symbol normalization; retained provenance
    verifies against the built artifacts.
14. Claim scope remains terminal FC/Conv2D, same-host, public-bias,
    feasibility-only functional evidence.

## Whole-repository review map

Trace integration outward from the changed contracts:

- public ABI and archive: `ringlpn/src/linear_preprocess*`,
  `scripts/build_linear_library.sh`;
- producer and state format: `two_party_linear_preprocess.cuh`,
  `graph_mask_state.h`, `correlation_freshness.h`, `private_file.h`;
- stock linear kernels and ownership: `GPU-MPC/fss/gpu_matmul*`,
  `gpu_conv2d*`, `backend/orca_base.h`, `backend/orca.h`;
- Sytorch callback lifecycle: `ext/sytorch/include/sytorch/module.h`,
  `layers/layers.h`, `backend/backend.h`, and `nn/orca_opt.h`;
- application entry: `experiments/orca/orca_inference.cu` and `cnn.h`;
- communication: `utils/gpu_comms.h`, `utils/sigma_comms.*`;
- build/provenance: component/build scripts, graph-library CMake, linear and
  graph provenance generators;
- source/approval binding: layer manifests, graph contract checker, approvals,
  receipts, publication environment manifest;
- evidence/claims: canonical docs, current reports, retained application rows,
  and historical checkpoint boundaries.

For the rest of EzPC, review dependency edges and shared APIs rather than
performing cosmetic line-by-line churn. Search for every changed exported symbol
and every macro/callback entry. A second convention beside an existing one is a
defect.

## Chief-of-staff audit sequence

### Phase A — forensic baseline

1. Record `git status --short --branch`, recent log, remotes, worktrees, and
   submodule state without mutating anything.
2. Capture unstaged, cached, and untracked inventories. Confirm intent-to-add
   entries and file modes.
3. Build a table mapping every dirty path to workstream, owner, dependencies,
   validation evidence, and proposed commit.
4. Read the full diff of each shared file before editing or staging it.

### Phase B — static and security review

Review, at minimum:

- integer overflow and signed/unsigned conversions in sizes and offsets;
- exact file mode, owner, link-count, symlink-component, stable-inode, and
  read-to-EOF checks;
- allocation/failure scrubbing and move semantics;
- digest taxonomy and comparison order;
- preflight deadlocks, asymmetric errors, fixed-port collisions, and process
  cleanup;
- host/device ownership, aliasing, double-free, leak, and reconstruction order;
- Sytorch graph initialization, terminal truncation disabling, and callback
  overload coverage;
- atomic output publication and duplicate-output behavior;
- macro-on/off compilation isolation;
- deterministic build inputs and provenance closure;
- runner timeout/process-group handling and private-root cleanup;
- manifest source anchors, refreshed binary approvals, and historical evidence
  immutability;
- claims that say “current” when they actually describe a retained older binary.

Do not suppress warnings or weaken a gate to make it pass.

### Phase C — targeted execution

Check GPU ownership before any heavy run. Do not kill or interfere with another
user's process. The August 24 run used GPUs 1 and 3 because GPUs 0 and 2 had
unrelated long-lived workloads; that assignment is historical, not a current
reservation.

Run the narrowest check after each correction:

```bash
cd /home/fatih/EzPC/GPU-MPC/ringlpn
PATH=/usr/local/cuda/bin:$PATH GPU_ARCH=89 \
  ./scripts/build_component.sh orca-linear-application
P0_GPU=<free-gpu> P1_GPU=<other-free-gpu> \
  PATH=/usr/local/cuda/bin:$PATH \
  ./scripts/run_orca_linear_application.sh
```

Macro-off stock build:

```bash
cd /home/fatih/EzPC/GPU-MPC
PATH=/usr/local/cuda/bin:$PATH make GPU_ARCH=89 orca_inference
```

Approval/provenance checks:

```bash
cd /home/fatih/EzPC/GPU-MPC/ringlpn
python3 scripts/orca_linear_application_build_provenance.py verify \
  --repo-root ../.. --cmake-build build/graph-libraries \
  --manifest results/application/orca_linear_application_build_provenance_2026_08_24.json
python3 scripts/linear_adapter_build_provenance.py verify-approval \
  --repo-root ../.. --ringlpn-root . \
  --approval results/fc/linear_adapter_binary_approval_2026_08_07.json
python3 scripts/graph_build_provenance.py verify \
  --repo-root ../.. --cmake-build build/graph-libraries \
  --manifest bin/resnet18_full_graph_build_provenance.json
```

Complete canonical gate only after focused checks and approvals pass:

```bash
cd /home/fatih/EzPC/GPU-MPC/ringlpn
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
  CUDA_VISIBLE_DEVICES=<exclusive-visible-gpu-list> \
  PATH=/usr/local/cuda/bin:$PATH \
  ./scripts/run_paper_checkpoint_smoke.sh
```

The full gate is long, consumes fresh correlation namespaces, and regenerates
several tracked result files. Snapshot status first. Afterward, keep only
intentional refreshed evidence; revert incidental generated timing drift only
when the file was clean in the pre-run snapshot. Never “clean up” a file that
was dirty before the run.

Role 2 inherits `GpuPeer` port 42003, which lies in the host ephemeral range.
Application/helper pairs must remain serialized.

### Phase D — frequent atomic version control

Use this protocol for every commit:

1. Finish one coherent slice and its applicable check.
2. Synchronize source, tests, evidence, approval/provenance, `CLAUDE.md`, the
   results index, and the dated memo as required by that slice.
3. Stage exact paths or hunks. Shared documents require hunk staging.
4. Inspect:

   ```bash
   git diff --cached --stat
   git diff --cached --check
   git diff --cached
   ```

5. Confirm no private file, raw key/state record, ledger, auth secret, scratch
   path, host identifier, unexpected generated timing row, or unrelated user
   hunk is staged.
6. Commit with an imperative, scoped message. Record the hash and validation in
   the running ledger.
7. Never amend an older checkpoint merely for tidiness. Corrections get a new
   commit and rerun the affected gate.

Recommended commit boundaries, subject to the actual dependency graph:

1. `ringlpn: expose bound terminal layer material`
   - public API, loader, and public API controls.
2. `orca: factor linear execution helpers`
   - `orca_base.h` plus focused helper regression.
3. `ringlpn: add source-native terminal Orca backend`
   - backend, application entry, and macro-gated inference change.
4. `ringlpn: add terminal Orca application gate`
   - component build, runner, wrapper, provenance profile, and smoke hook.
5. `ringlpn: refresh source-bound approvals`
   - manifests, pins, deterministic approvals, receipts, and retained sanitized
     application evidence.
6. `ringlpn: document terminal integration checkpoint`
   - authoritative docs and current reports, with only the relevant hunks.
7. Separate commits for reviewed EMP-Silent, model-scale, two-host publication,
   and manuscript work. Do not fold these into the terminal integration merely
   to reduce the dirty count.

If a newly discovered fix has not yet passed its gate, leave it uncommitted and
state the exact proposed commit boundary. Frequent commits do not justify
committing an unverified state.

## Evidence already recorded — verify, do not blindly restate

The August 24 checkpoint reports:

- terminal FC and Conv2D application PASS with nonzero inputs, weights, and
  public biases;
- eight bilateral fail-closed controls;
- macro-off stock Orca build without the Ring-LPN archive;
- deterministic linear and full-graph approval refreshes;
- complete canonical `[paper-smoke] ALL GATES PASS`;
- ephemeral full-graph digest
  `ec026fa850dfca7b3f51fd7eaef1e729b17a82d03464976a63c662a662a7b410`.

Exact application hashes and provenance digests live in the dated integration
memo and retained provenance JSON. Treat them as evidence for that source
snapshot. Recompute them after any relevant edit; never edit a digest merely to
match a changed binary.

## Non-negotiable boundaries

- No concrete Ring-LPN security level is pinned.
- q64/q128 describe arithmetic limbs, not security bits.
- No matched-security dealerless or full-model performance win is supported;
  the September 10 scheduling result is explicitly one-host and one-shape.
- The terminal integration does not implement arbitrary multi-layer mask-state
  chaining.
- Full-graph nonlinear material remains TEST-ONLY trusted.
- No private/trained-model, accuracy, WAN, malicious-security, authenticated
  deployment, side-channel, or conference-readiness claim follows.
- Existing local channel authentication covers endpoint/context establishment,
  not every later application byte.
- Raw records, state, auth material, shares, ledgers, and outputs are private and
  not publication evidence.
- The retained 2026-08-10 checkpoint is internal-only and byte-immutable.
- Do not contact external authors or decide ownership/credit without the owner.
- Push only the explicitly authorized public-safe source branch after host
  authentication. Never publish internal `master`, retained private checkpoint
  ancestry, or manuscript/release evidence without its separate authorization.

## Definition of done

The new model is done only when:

1. every dirty and untracked path is classified;
2. every changed public symbol and relevant callsite is reviewed;
3. all concrete findings are fixed or blocked by an exact unavailable
   prerequisite;
4. focused checks pass after each affected slice;
5. approval and provenance manifests verify against actual artifacts;
6. the final canonical gate passes after the last runtime-affecting change;
7. current documentation, claims, hashes, and evidence agree;
8. private and incidental generated files are absent;
9. coherent work is represented by small atomic commits;
10. remaining dirt is deliberate, named, and assigned;
11. the final report lists findings by severity, every commit hash/message,
    every command and observed result, remaining risks, and the next three
    actions.

## Required final report format

```text
Decision
- overall readiness and exact boundary

Findings
- Critical: ...
- High: ...
- Medium: ...
- Low: ...

Fixes
- file:symbol — change and invariant

Verification
- exact command — exact observed result

Commit ledger
- <hash> <message> — paths, validation

Remaining worktree
- path group — owner/workstream/reason

Open blockers
- missing prerequisite, evidence, and attempted resolution

Next three actions
1. ...
2. ...
3. ...
```
