# ringlpn — canonical project guide

Single live document for project state. History lives in `results/reports/`; `results/README.md`
indexes every artifact.

## 1. What and why

Orca (the GPU FSS secure-ML system in this repository) needs a trusted dealer to produce Beaver
keys for its linear layers. This subproject replaces that dealer for forward FC/Conv2D layers
with a two-party protocol built on a Ring-LPN pseudorandom correlation generator:

GPU NTT → `Z_p` SPFSS (sum of DPFs) → Figure-2 (BCG+20) Ring-LPN OLE → slot-packed Beaver cross
terms → `Z_M → Z_2^bw` conversion → byte-compatible stock keys consumed by the **unchanged**
`gpuMatmulBeaver` / `gpuConv2DBeaver`.

- Candidate contribution: only the exact Ring-LPN → Orca GPU FC/Conv integration. The per-point
  DPF, the Ring-LPN generator, the conversion primitives, Orca, and the GPU polynomial backends
  are prior or inherited work.
- No "first" claim of any kind. Prior art: Silentium (ePrint 2025/1013); libOTe `RingLpnTriple`
  and Reverse Cuckoo (MIT); Agarwal–Raghuraman–Rindal (ePrint 2025/2294); Rivinius et al.
  (PoPETs 2023, ePrint 2023/359, source `618301c…`).
- Cheddar is an attributed MIT dependency (`extern/Cheddar_PROVENANCE.txt`,
  `extern/Cheddar_MIT_LICENSE.txt`). GPU-NTT is a cited external baseline.
- The separate private GPU-PCG/PIM stream must not be imported or claimed.

## 2. Status snapshot (2026-10-06; last engineering commit `bf239ff`, then documentation, evidence, and build-script hygiene commits)

- The live source is trusted-dealer-free in the stated random-oracle model for forward
  FC/Conv2D at feasibility parameters: two OS processes on distinct GPUs, terminal FC/Conv2D
  through real Orca, full ResNet18 graph composition whose nonlinear keys and
  truncation/remask/terminal masks still come from a TEST-ONLY trusted adapter.
- Performance is strongly negative (882×, 60.6×, 268.7× slower than the stock dealer). The
  frozen prospective capacity-prediction gate **failed**.
- No security level, parameter set, two-host run, clean-clone run, or independent human
  cryptographic review exists.
- Last full canonical gate: `ALL GATES PASS` on 2026-09-10 in 5,228.406 s
  (`results/reports/engineering_review_verification_2026_09_10.json`). Latest engineering
  record: `results/reports/autonomous_technical_closure_2026_09_22.json`.
- `origin` is public: internal history through `9685463` (2026-09-11), including the
  internal-only full-graph checkpoint, is on `origin/master` (§14).
- Public source branch `ringlpn/clinical-review-2026-09-10`: the remote is at `afc93c6`. Local
  `9bf1ab0` plus 12 source-cleanup mirror commits (tip `ac711e2`) are bundled but **not
  pushed**: pushes stall in the VS Code askpass username prompt and no GitHub SSH key exists.
- The fresh Ring-LPN vs direct-OT comparison is **blocked**: `--mode compare` needs idle GPUs 1,
  2 and 3, and GPU 2 is occupied by another user's process (2026-09-22 record; vLLM on GPUs 0
  and 2 as of 2026-10-06).
- Production cryptographic sources, the failed prediction gate, and the 36-page v2.17 PDF are
  unchanged by the 2026-09-22 work and this cleanup.

## 3. Validated claims

"Feasibility" means `(n,c,t)` values that are not security-pinned.

| Claim | Evidence | Boundary |
|---|---|---|
| Trusted-dealer-free two-process forward FC: OpenSSL private DRBG roots, jointly seeded SHAKE256 vector `a=(1,a1..a_{c-1})`, 128-bit invocation / 256-bit correlation IDs, consume-once ledger, epoch-zero Gilboa OLE then reserved Ring-OLE slots (`3c²t²` per instance, `n−3c²t²>0`) | `results/fc/two_party_fc_preprocess_2026_08_04.csv` (5 q64/q128 cases pass); `results/fc/two_party_fc_preprocess_controls_2026_08_04.csv` (16 controls reject) | random-oracle model; feasibility parameters; plain TCP after HMAC handshake |
| Terminal FC `(2,3,2)` and Conv2D `(1,4,4,1; 3x3x1x2, pad 1, stride 1)` through real Orca role 2, nonzero values and biases, 8 bilateral controls | `results/application/orca_linear_application_2026_08_24.csv`, `.log`, `orca_linear_application_build_provenance_2026_08_24.json`; memo `results/reports/orca_linear_application_integration_2026_08_24.md` | terminal only, same host, public bias; producer freshness gives no persistent application-consumer replay prevention (callers must supply fresh material) |
| ResNet18 classifier layer `1x512x1000`, q128/bw32, `(8192,2,8)`, 10/10 trials | `results/fc/two_party_fc_model_scale_2026_08_04.csv` (v6, regenerated 2026-08-10), binary `02eaaac9…` | uncontrolled GPU occupancy; excludes channel/socket/auth time |
| Controlled CNN2 FC4/FC5 and CNN3 FC5 matrix, 30/30 layer trials, same-GPU stock-dealer comparison | `results/fc/two_party_fc_model_scale_cnn2_cnn3_2026_08_14.csv` plus `*_2026_09_11` derived files and `results/fc/model_scale_inputs_2026_08_14/` | strongly negative; binary `bf4e4f90…`, GPUs 1/3 |
| Opt-in EMP-Silent rerun: same 5 cases and 16 controls, exact OT inventory; FC binary `bf4e4f90…`, bridge library `435f3be6…`, EMP-OT revision `2fca139f…` | `results/fc/two_party_fc_emp_silent_correctness_2026_08_14.csv`, `_environment_2026_08_14.txt` (ignored paths; force-added) | correctness/accounting only; unreviewed backend; one trial per case |
| Source-bound 21-layer ResNet18 plan `(262144,2,8)`: 1,680,390,912 cross terms, 6,439 batches, 25,756 Ring-OLE, 6,593,536 DPF trees, 130,774,336 payload B/party | `results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json` | plan and isolated records only |
| Full-graph known-zero composition, one run: 62-item stream (21 linear, 20 truncation, 17 ReLU-extend, MaxPool, GlobalAvgPool, sign-extend, output); the run executes 21 truncations over 2,484,224 values, 19 stock nonlinear keys, 20 remasks, 3 projection + 5 identity residuals; 7 graph controls. Linear critical-path sum 4,720.417 s; graph 11.852 s (party 0); adapter 9.356 s and 1,084,582,352 stock-key B/party; 17 payload files + `INDEX.json` | `results/graph/resnet18_full_graph_checkpoint_2026_08_10/INDEX.json`, `FULL_GRAPH.manifest` (manifest digest `fdf51f25902afd94a1e67b8bdffa33f762d89c104836538d913c2d7e392c5395`) | internal-only, byte-immutable; the TEST-ONLY trusted adapter reads both parties' mask states and supplies the nonlinear keys, both truncation successor-mask shares, and remask/terminal material; each linear record was first consumed by its omniscient checker (not consume-once deployment evidence); host strings and GPU-1 reuse |
| Isolated secure truncation | `results/secure_truncate/secure_truncate_check_2026_08_06.csv` | component |
| Two-process DPF transport: 369/369 pairs, depths 4–14, `6L+4` protocol direction switches per batch plus 6 setup switches, 256 base OTs, 21,829 B setup; L=11 batch 1→256 (party 0) gives 52,618→3,788.5 B/tree and 11,996.4→168.4 µs/tree | `results/dpf/two_party_dpf_keygen_2026_07_29.csv` (canonical-gate rerun, committed 2026-08-11 in `09996aa`); memo `results/reports/two_party_dpf_transport_memo_2026_07_29.md` (original run: `6L+6`, 52,626→3,789 B, 11.6 ms→146 µs) | IKNP, not silent OT; direction switches are not network rounds |
| GPU-AES four-call PRG: 16 device vectors, 88 two-process keys | `results/dpf/two_party_gpu_dpf_2026_07_29.csv` | key compatibility, not GPU keygen |
| Ideal host DPF reference: 2,432 pairs; `2·depth` string OTs, `depth−1` triples, 3 OLEs, `2(depth−1)+130·depth+⌈log2 p⌉` opened bits | `results/dpf/` | ideal-functionality hybrid |
| Exact conversion: 76/76 × 4 configs; `5ℓ−3` logical, `10ℓ−6` meaningful, `2ℓ−1` post-mask | `results/secure_convert/` | hybrid simulator in the security contract |
| Host Figure-2 OLE suites 135/57/36 | `results/reports/ole_figure2_host_results.md` | host reference |
| Conditional static-semi-honest theorem for one forward FC matmul | `results/reports/dealerless_orca_fc_security_contract_2026_07_29.md` | proof boundary, not a parameter claim; human review open |
| SCI duplex worker: 22 invocations; launch→exit median 0.976021855→0.906485317 s (−7.12%); Phase B 280.205→223.2515 ms (−20.33%); 68 fields equal | `results/fc/sci_duplex_worker_reproducible_2026_09_11.json`, `.plan.json`; controller `scripts/run_fc_review_experiments.py` | one host, one shape |
| Conversion audit: 72 identities over 294,912 worlds, 1,120 sampled cases, 96 retained-mask cases; fresh bw32 ideal share passes with probability 2^-31 | `scripts/audit_conversion_simulator.py`; `results/reports/technical_followthrough_2026_09_22.json` (72 / 294,912 / 1,120); `results/reports/autonomous_technical_closure_2026_09_22.json` (96, 2^-31) | no DRBG or concrete-OT simulation |
| Leaf audit: 8 pinned sources, 676 reduced checks; ε=(2^64 mod p)²/2^128 ≈ 2^-70.83 per comparison (first prime) | `scripts/audit_dpf_leaf_loss.py` (8 `SOURCE_PINS`); `results/reports/autonomous_technical_closure_2026_09_22.json` (676 role/tag checks) | not a lifetime bound |
| Per-limb OT inventory (regular B=256, depth 11): 5,632 chosen-message 128-bit OTs, not 2,816 bit-COTs | `results/reports/independent_review_verification_2026_09_11.json` | accounting |
| Direct-Gilboa-OT FC baseline: 8 cases; median 1.079694043 s over 3 runs of `100x64x10` | `results/reports/direct_ot_fc_complete_2026_09_22.json`, `.plan.json` | CPU producers with GPU3 checker; no ratio |
| Source-only runtime candidate: image `sha256:632f3d1e…`, archive 7,754,951,680 B, binary `a5630582…` | `~/.local/share/ringlpn/artifacts/runtime-candidate-afc93c6/` (outside the repo) | not a registry image or deployment |
| Upstream Orca byte-identical with `ORCA_RINGLPN_FC_KEYS` off | test source `GPU-MPC/tests/nn/orca/fc_test.cu` (`make orca_fc` → ignored binary `tests/nn/orca/fc`); no retained run artifact | macro-off build; rerun before citing |
| Manuscript v2.17, 36 pages, PDF sha `f0544ee14d076289320492364a269a076ed61f13e63ba3d752684c5a3e716d4a`, two byte-identical TeX Live 2023 builds (`SOURCE_DATE_EPOCH=1786320000`) | `results/reports/dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex`, `.pdf` | internal/advisor only; not submission-ready |

Classifier detail (`results/fc/two_party_fc_model_scale_summary_2026_08_04.csv`): setup-included
critical path mean 4.011203588 s, SD 0.036601967373448, median 4.0193924415 s, R-7 IQR
0.03481908225, t95 [3.985020117867291, 4.037387058132709]. Application bytes 182,372,344; total
182,416,324 (includes 43,658 base-OT bytes). Payload 4,108,096 B/party; 11,023 dependency layers;
median host peak 142,542,848 B; device-wide GPU peak 31,929,597,952 B (includes other users).
Medians: Phase B 1.955515 s, Phase C 0.041723 s, Phase A 0.2462935 s, expansion 1.226650 s. 276
instances, 70,656 trees; 1,536 epoch-zero and 210,432 PCG Phase-C products; 210,432 consumed +
1,536 discarded reserved slots; 1,024 unused application slots.

Gate digests: same-worktree 2026-08-10 `2588eac6…`, 2026-08-24 `ec026fa8…`; 2026-09-10 fresh
graph `ccd598f8…` (18-file summary, owner-private, `~/.local/share/ringlpn/reviews/2026-09-10/full-graph/`).

## 4. Not claimable

- Any security level. q64/q128 mean one or two ~62-bit arithmetic limbs; no `(n,c,t,p0,p1)` is pinned.
- A secure network deployment. After the HMAC handshake, traffic is plain TCP.
- A performance win of any kind.
- Full-model dealer removal, multi-layer record dispatch, or dealerless nonlinear preprocessing.
- Private, trained, or accuracy results; training; malicious security; WAN; side-channel resistance.
- Conference readiness. The ASPLOS 2027 deadline (2026-09-09 AoE) has passed.

## 5. Negative results (keep prominent)

| Result | Evidence |
|---|---|
| CNN2: sum of layer means 26.8818423365 s, CI [26.84972841498, 26.9117673864825]; dealer 30.46697 ms; ratio 882.3273970631×, CI [876.3872084103, 890.1395707186] | `results/fc/two_party_fc_model_scale_cnn2_cnn3_summary_2026_09_11.csv` |
| CNN3: mean 0.903894393 s, CI [0.8929673378725, 0.9152799675375], SD 0.01897115914, median 0.904909782, IQR 0.02851206825; dealer 14.92146 ms; ratio 60.5768063581×, CI [59.3418846688, 61.8764437656]; paired-ratio median 60.3181599795375× | same; bootstrap 10,000 draws, seed 20260911, R-7 |
| CNN bytes (application/total): CNN2 1,302,752,736 / 1,302,840,696; CNN3 39,432,504 / 39,476,484; dependency layers 72,278 / 1,663 | same |
| Classifier medians: stock dealer 14.73535 ms, online 1.14969 ms; median per-trial setup-included ratio 268.6769431352700× (descriptive, uncontrolled GPUs) | `results/fc/two_party_fc_model_scale_summary_2026_08_04.csv` |
| Prospective capacity gate FAILS: 88 training + 22 held-out runs; FC2 predicted 14.577957 vs 14.358284 s (inside interval); AlexNet gemm13 42.931651 vs 43.026200 s, outside [42.886032, 42.978961]; no refit or widening | `results/fc/capacity_model_prospective_2026_09_11.json`, `.plan.json`, `.seal.json` |
| Regular-DMPF design NO-GO | `results/reports/regular_dmpf_design_no_go_2026_08_06.md` |
| Native-ring PCG NO-GO | `results/reports/native_ring_technology_audit_2026_08_04.md` |
| `n=2^17,c=4,t=34` implementation NO-GO: `t∤n`; uniform needs 7,272,923,136 slots, ≥123.6 GB; the 257.02-bit label is withdrawn | `results/security/README.md` (slot count, memory); `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` (NO-GO, withdrawn label) |
| Conservative-pin transcripts invalid: 57.293/111.244/218.641/257.023 are out-of-domain estimator calls; 135.12/145.85/190.53/470.77 are unproved heuristics; BCG+20 §8.2, §9.1 and Table 1 disagree | `results/security/README.md` |
| Architecture microbenchmark: OKVS 275× faster with uniform noise but 0.79× with the regular layout; big-state 2.29× at 37× key bytes | `results/reports/s2_architecture_comparison_2026_07_29.md` |
| Reverse Cuckoo stock libOTe: 12.43 s, 22,939,444 KiB RSS, `setBase` 446.448 ms; exact-p0 folded run: setup 18,523,424 µs, online 2,116,894 µs, end-to-end 20,688,314 µs; no ratio allowed | `results/reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md`, `results/reports/reverse_cuckoo_p0_baseline_2026_08_04.json` |
| GPU-NTT 1.2–3.9× faster but cannot run 62-bit primes; keep Cheddar (revisit if primes drop to ≤60 bits) | `results/reports/ntt_baseline_comparison_2026_06_10.md` |
| Older-binary Conv0 rows 358.085 s and 384.816 s (binary `1db001…`, not the currently approved Conv binary `8130d135…`); no breadth-first speedup claim | `results/conv/conv0_breadth_comparison_2026_08_09/` |
| Withdrawn: old CNN2 model-trial median/SD/IQR/t-CI and paired median. The consumed, incomplete 2026-08-07 forward-linear record set (records and ledger deleted): never cite or resume | — |

## 6. Open gates and exact blockers (gate IDs per §8 of the security contract)

| Gate | Blocker |
|---|---|
| P-POS / P-PCG / S2 parameters | Hard theorem blockers. P-POS: the exact projection law is pinned to the live sampler, but the hardness bridge is open. P-PCG: no reviewed structured projected-code reduction, two-CRT-limb advantage composition, modern direct RSD or 2025/2026 QA-SD dispositions; BCG+20 discrepancy unresolved. Needs qualified human review. |
| Regular-DMPF route | Design NO-GO; reopen only with a source-reviewed fixed-transcript construction below the audited cost ceiling (`results/reports/regular_dmpf_design_no_go_2026_08_06.md`). |
| P-KEY | Actual AES/leaf-map reduction and lifetime loss composition. |
| P-CONV | Independent human review of the hybrid proof; no DRBG or concrete-OT simulation. |
| P-RNG | SHAKE/random-oracle instantiation review. |
| P-FRESH | Closed only under SHA-256 collision resistance and trusted durable no-rollback storage; human review open. |
| P-MAP / P-PROC | Independent human audit. |
| P-TOPO | Training-state extension. |
| Nonlinear dealer | TEST-ONLY trusted adapter needs a reviewed dealerless DCF/DMPF replacement. |
| Two-host deployment | Not executed. Missing: authorized distinct host and identities, native rootless Podman with subuid/subgid, block-backed ext4/xfs ledger mounts (tmpfs rejected), annotated tag and source authorization, immutable runtime digest. Manifest fields are null by design. |
| Clean-clone reproduction | Not executed. `reproduce_publication.sh check`/`local-smoke` fail closed until annotated tag `ringlpn-publication-candidate-v1` (manifest `source_release.required_annotated_tag`) points at HEAD with a matching `GPU-MPC/ringlpn` tree; no such tag exists (owner-approved release commit needed). No two-build image receipt. |
| Build-input drift | `build_component.sh` changed in `bf239ff` (direct-OT dispatch) after the 2026-09-10 gate, so `verify-approval` of `results/fc/linear_adapter_binary_approval_2026_08_07.json` and the local `bin/` provenances exit 1 ("… differs: …/build_component.sh"); the retained 08-24 application provenance predates the `051e7f0` `orca_base.h` change. The approved binary hashes still pass the gate's plan check (`RUNNER_PLAN_CHECK=1 run_full_linear_manifest_gate.sh`). Approval refresh needs owner approval (§11). |
| Matched baseline | No dealerless baseline at reviewed parameters. Fresh Ring-LPN vs direct-OT `--mode compare` locks GPUs 1 and 2 (producers) and 3 (checker), all of which must be idle; GPU 2 is occupied. The harness never uses GPU 0. |
| Full-graph checkpoint | Replace with a fresh normalized 3-GPU run before any further circulation; the 2026-08-10 bytes are already public via `origin/master` (`9685463`). |
| Source push of `afc93c6..ac711e2` | VS Code askpass route stalls; no SSH key. Bundle: `~/.local/share/ringlpn/artifacts/source-bundles/ringlpn-source-ac711e2.bundle` (36,664 B, sha256 `0c72fca5a0c6576645d6888cbd823ebcd4340407431c32142daef12d34562622`, requires `afc93c6`). |
| Owner rulings | Authorship, credit, reuse, circulation, outreach (`results/reports/s2_professor_decision_request_2026_07_29.md`), and whether to restrict or rewrite the public `origin/master` history that holds internal-only artifacts (§14). |

## 7. Source map (`src/`)

| File(s) | Role |
|---|---|
| `test_two_party_fc_preprocess.cu`, `test_two_party_conv_preprocess.cu`, `two_party_linear_preprocess.cuh` | Live forward-linear engine and thin FC/Conv entry points |
| `linear_preprocess.h`, `linear_preprocess_backend.cuh`, `linear_preprocess_fc.cu`, `linear_preprocess_conv.cu` | Public move-only `OwnedLayerMaterial` record/state boundary (`libringlpn_linear.a`) |
| `correlation_freshness.h` | Fixed-width correlation IDs and owner-only consume-once ledger |
| `private_file.h`, `public_ring_vector_xof.h` | Owner-private file I/O; SHAKE256 public-vector XOF |
| `two_party_spfss.h`, `two_party_spfss_gpu.cuh` | Party-local noise binding and distributed SPFSS keygen |
| `two_party_dpf_protocol.h`, `two_party_dpf_gpu.cuh`, `dpf_key_io.h` | Batched two-party DPF keygen (Phase A/B OT, Phase C OLE); key serialization |
| `two_party_ot.h` | SCI/IKNP transport, HMAC endpoint handshake, Gilboa OLE |
| `emp_silent_adapter.h`, `emp_silent_bridge.{h,cpp}`, `emp_silent_bridge_authorization.h`, `emp_silent_bridge_build/` | Opt-in sealed EMP-Silent backend |
| `gpu_spfss_zp.cuh`, `gpu_aes_prg_host.h` | GPU DPF/SPFSS with `Z_p` payloads; four-call AES PRG and host twin |
| `ringlpn_ole_party.cuh` | Party-local Figure-2 Ring-LPN expansion |
| `bench_ntt_cuda_cheddar.cu` | Cheddar-derived GPU NTT/polymul (primes 2^62−6·2^24+1, 2^62−7·2^24+1) |
| `secure_convert.{h,cpp}`, `secure_truncate.{h,cpp}` | Exact two-process conversion; secure truncation |
| `orca_terminal_linear_backend.cuh`, `orca_linear_application_entry.cuh` | Terminal FC/Conv Orca backend; role-2 entry included by `experiments/orca/orca_inference.cu` |
| `graph_mask_state.h`, `resnet18_graph_contract.h` | Digest-bound mask-state records; compiled 21-linear/62-item ResNet18 contract |
| `stock_nonlinear_full_record.h`, `test_stock_nonlinear_full_keygen.cu` | TEST-ONLY trusted stock nonlinear key adapter |
| `test_resnet18_full_graph.cu`, `test_resnet18_graph_contract.cpp` | Full-graph runtime/checker; contract test |
| `test_direct_ot_fc_preprocess.cu` | Direct-Gilboa-OT FC baseline producer |
| `orca_fc_ringlpn_keywriter.cuh` | Historical `ORCA_RINGLPN_FC_KEYS` keywriter included by `nn/orca/fc_layer.cu` |
| `test_*` host/GPU tests (`private_file`, `correlation_freshness`, `public_ring_vector_xof`, `secure_convert`, `secure_truncate`, `distributed_dpf_keygen`, `two_party_dpf_keygen`, `two_party_dpf_validate`, `two_party_spfss_keygen`, `two_party_gpu_dpf_eval`, `gpu_aes_prg_parity`, `spfss`, `spfss_zp_cuda`, `linear_preprocess_api`, `orca_linear_helpers`, `orca_zp_bridge`, `emp_silent_loopback`) | Gate components |
| `spfss_host.{h,cpp}`, `bench_ole_ringlpn_host.cpp`, `verify_figure2_expand.cpp` | Host references (135/57/36) |
| `bench_ole_ringlpn_cuda.cu`, `bench_ole_ringlpn_party.cu`, `bench_linear_ole_ringlpn_cuda.cu`, `bench_orca_fc_{ringlpn_demo,ideal_ole_transcript,real_ole_transcript}.cu`, `orca_fc_ideal_ole_transcript.cuh`, `dump_gpu_aes_prg_vectors.cu` | Component diagnostics still exercised by the gate |
| `bench_ntt.cpp`, `bench_ntt_cuda.cu` (legacy), `bench_vole_ringlpn.cu`, `bench_ntt_gpu_ntt_baseline.cu` | Microbenchmarks behind retained `results/ntt/` and `results/vole/` rows |
| `test_resnet18_graph_prefix.cu`, `test_stock_nonlinear_prefix_keygen.cu`, `stock_nonlinear_prefix_record.h` | Superseded prefix regression; generators of retained `results/graph/resnet18_graph_prefix_*` rows; never cite as current |
| `orca_globals_stub.cpp` | Defines `OneGB` for standalone binaries |

Upstream integration points (all other upstream edits: `/home/fatih/EzPC/AGENTS.md`):
`GPU-MPC/backend/orca_base.h` (behaviour-preserving extraction of the stock matmul/Conv2D
helpers); `GPU-MPC/experiments/orca/orca_inference.cu` (`ORCA_RINGLPN_LINEAR_INTEGRATION` role 2;
macro-off roles 0/1 unchanged and do not link Ring-LPN); `GPU-MPC/nn/orca/fc_layer.cu`
(`ORCA_RINGLPN_FC_KEYS`; flag off is the byte-identical baseline).

`scripts/`: every artifact has a `build_*.sh`/`run_*.sh` pair writing `.csv` + `.md` + `.log`
under `results/<area>/`; `scripts/build_component.sh list` names the maintained targets. Do not
edit hash-pinned scripts (`build_full_linear_model_manifest.py`, the audit scripts' source pins,
runtime-candidate recipes, `publication_environment_manifest_2026_08_10.json`) without rebinding.

## 8. Commands

Run from `GPU-MPC/ringlpn` unless noted. Check `nvidia-smi` first.

```bash
# Canonical gate (~1.5 h; 2026-09-10 took 5,228 s with 1/2/3). Consumes fresh namespaces and
# regenerates tracked results. <a>,<b>,<c> = three idle, distinct physical GPUs.
# CUDA_VISIBLE_DEVICES pins only stages that inherit it; the application and full-graph runners
# overwrite it per child from ORCA_LINEAR_* (default 0/1) and FULL_GRAPH_* (default 0/1/2, trusted 1).
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=<a> \
  ORCA_LINEAR_P0_GPU=<a> ORCA_LINEAR_P1_GPU=<b> \
  FULL_GRAPH_P0_GPU=<a> FULL_GRAPH_P1_GPU=<b> FULL_GRAPH_CHECK_GPU=<c> \
  FULL_GRAPH_TRUSTED_GPU=<b> PATH=/usr/local/cuda/bin:$PATH GPU_ARCH=89 \
  ./scripts/run_paper_checkpoint_smoke.sh
# success: exit 0 and "[paper-smoke] ALL GATES PASS". With only two idle GPUs (now 1 and 3; vLLM
# holds 0 and 2) add RUN_FULL_GRAPH_SMOKE=0: all but the full graph run, ending
# "full ResNet18 graph skipped; GPU component gates pass". Focused runners, the terminal
# application, and the macro-off stock build: README.md "Gate and focused runners".

# Provenance verification (2026-10-06: all three exit 1, see §6 "Build-input drift")
python3 scripts/linear_adapter_build_provenance.py verify-approval --repo-root ../.. --ringlpn-root . --approval results/fc/linear_adapter_binary_approval_2026_08_07.json
python3 scripts/orca_linear_application_build_provenance.py verify --repo-root ../.. --cmake-build \
  build/graph-libraries --manifest results/application/orca_linear_application_build_provenance_2026_08_24.json
python3 scripts/graph_build_provenance.py verify --repo-root ../.. --cmake-build \
  build/graph-libraries --manifest bin/resnet18_full_graph_build_provenance.json  # ignored output of the resnet18-full-graph build

# CPU-only audits (from GPU-MPC/)
python3 ringlpn/scripts/audit_dpf_leaf_loss.py --workload 8192,2,8,2,regular,1,1 \
  --budget-bits 64 --check-reduced
python3 ringlpn/scripts/audit_conversion_simulator.py
python3 ringlpn/scripts/test_fc_model_scale_statistics.py --replay-dir <new-dir>
# also audit_ringlpn_regular_projection.py, audit_regular_isd_crypto2024.py,
# audit_hybrid_rsd_asiacrypt2025.py; audit_ringlpn_finite_field_models.py exits nonzero by design.
```

Evidence checks, from `/home/fatih/EzPC`, after any change under `results/` or to the manifest.
The private-artifact guard (silent exit 0 = pass) checks the manifest self-digest, the full-graph
INDEX, and leftover private files; it does not re-hash bound files. The bound-digest check
re-hashes every `required_tracked_evidence` entry and the v2.17 TeX/PDF `build` pins:

```bash
python3 GPU-MPC/ringlpn/scripts/retained_public_evidence.py --repo /home/fatih/EzPC \
  --manifest /home/fatih/EzPC/GPU-MPC/ringlpn/scripts/publication_environment_manifest_2026_08_10.json
python3 - <<'PY'   # prints "bound evidence OK"
import hashlib, json
m = json.load(open("GPU-MPC/ringlpn/scripts/publication_environment_manifest_2026_08_10.json")); b = m["build"]
pins = [(e["path"], e["sha256"]) for e in m["required_tracked_evidence"]] + [(b["publication_source"], b["publication_source_sha256"]), (b["publication_pdf"], b["publication_pdf_sha256"])]
bad = [p for p, h in pins if hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
print("\n".join(bad) or "bound evidence OK"); raise SystemExit(bool(bad))
PY
```

Direct-OT baseline, runtime candidate, and component builds: `README.md`. Clean-clone and two-host
reproduction: `results/README.md`. Authorized source push (public-safe branch only, never `master`):
`git push origin refs/heads/ringlpn/clinical-review-2026-09-10:refs/heads/ringlpn/clinical-review-2026-09-10`.
TeX rebuild (overrides the image's dispatcher entrypoint and uid 65532; the image sets
`SOURCE_DATE_EPOCH`). On 2026-10-06 two passes on a scratch copy reproduced the pinned PDF exactly:

```bash
T="$(mktemp -d)"; cp GPU-MPC/ringlpn/results/reports/dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex "$T/"  # repo root
for pass in 1 2; do docker run --rm --network=none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$T:/work/t" -w /work/t --entrypoint pdflatex ringlpn-repro:2026-08-10 -interaction=nonstopmode \
  -halt-on-error dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex >/dev/null || break; done
sha256sum "$T"/*.pdf; rm -rf "$T"
```

For a deliberate manuscript change, edit the `.tex`, rebuild, inspect every page, copy the PDF back,
and rebind `build.publication_*_sha256` and the PDF's `required_tracked_evidence` entry (§11).

## 9. Perf anchors (RTX 5000 Ada; diagnostic unless stated)

| Metric | Value |
|---|---|
| OLE expand n=8192 c=2 t=8 regular | 9.086 ms (q64) / 18.230 ms (q128), one iteration |
| OLE expand t=64 (q64) | 881 ms uniform / 61 ms regular; different domains, no ratio |
| Linear OLE→Beaver 2×2×2 regular | 143.686 ms (q64) / 289.511 ms (q128) |
| Cheddar polymul n=8192 batch 64 (q64) | ~255–265 µs |

Setup-included time = per-layer preflight + OT setup + legacy total; it excludes `PartyChannel`
construction, sockets, and authentication. Never call it end-to-end.

## 10. Environment gotchas

- Shared server, no sudo. Docker group membership is root-equivalent: use it only for ephemeral
  containers and to chown your own files (`docker run --rm -v <dir>:/x ubuntu:22.04 chown -R 1013:1014 /x/...`);
  never touch other users' files or processes (e.g. vLLM on GPUs 0/2).
- `nvcc` is in `/usr/local/cuda/bin` (not on `PATH`). `GPU_ARCH=89`, 4× RTX 5000 Ada.
- Ephemeral ports are 32768–60999; runners use 20400–29761. A full-graph run at base 57400 lost
  a bind; a consume-once run must restart fresh, never resume. Role 2 inherits `GpuPeer` port
  42003 (inside the ephemeral range): serialize application runs.
- Never call Orca's eager 25 GiB `initGPUMemPool()` from focused Ring-LPN executables.
- `extern/NFLlib` is a submodule pinned at `quarkslab/NFLlib@5cf40ed`. `scripts/setup_nfl.sh`
  exits 0 when it is built at that commit (`FORCE=1` rebuilds); a fresh build needs git, cmake,
  make, a C++ compiler, and GMP/MPFR headers (absent on this host: use `ringlpn-repro:2026-08-10`).
  It refuses a `build/` cache configured for another source path (e.g. the container's
  `/home/ringlpn`): move `extern/NFLlib/build` aside and rerun. The in-tree build makes
  `git status` show untracked content (`?`), as for `GPU-MPC/ext/cutlass`; both gitlinks are
  unchanged: do not commit, clean, or re-pin them. The mnist/weights submodules carry
  pre-existing internal staged renames (status `m`): leave them.
- Root `.gitignore` hides `*.csv`/`*.txt`; `ringlpn/.gitignore` hides `*.pdf` and re-allows
  `src/emp_silent_bridge_build/CMakeLists.txt` (tracked since `a458db2`, so
  `build_component.sh emp-silent-bridge` works from a clean checkout). Add evidence with `git add -f`.
- The local `ringlpn-repro:2026-08-10` tag is not publication provenance. In Debian containers
  `/bin/sh` is dash (no brace expansion). Inside `orca-dev`, `/home/ringlpn` = `GPU-MPC/ringlpn`.
- Flags: `ORCA_RINGLPN_LINEAR_INTEGRATION`, `ORCA_RINGLPN_FC_KEYS` (+`_QBITS`, `_SEED`),
  `RINGLPN_NTT_NO_FUSE` / `RINGLPN_NTT_FORCE_FUSE`, `SMOKE=1 QBITS NOISE`.
- Never reuse ledgers or work roots. Raw records, states, ledgers, and auth files are never evidence.
- `/tmp` is wiped at boot and after 30 days. Durable artifacts live in `~/.local/share/ringlpn/`
  (`artifacts/runtime-candidate-afc93c6/`, `artifacts/source-bundles/`,
  `reviews/2026-09-10/full-graph/`). The public source worktree
  `/tmp/ringlpn-review-20260910-_m6jgfab/public-source` is registered with
  `ringlpn/clinical-review-2026-09-10` checked out; reuse it while it exists. After a wipe, run
  `git worktree prune && git worktree add <dir> ringlpn/clinical-review-2026-09-10` (git refuses
  a second checkout of the branch while the stale registration remains).

## 11. Working rules

Hash-bound files stay in place: everything in the manifest's `required_tracked_evidence`, the
v2.17 TeX/PDF pinned by `build.publication_*_sha256`, files pinned by other JSON (approvals,
checkpoint manifests, seals, plans), and all of
`results/graph/resnet18_full_graph_checkpoint_2026_08_10/` (byte-immutable). Never edit a digest
to match a changed artifact. A deliberate edit to a bound report (as in `c097af9`) must, in the
same commit, rebind its `sha256` and the `manifest_digest` (SHA-256 of the manifest minus that
key, `json.dumps` with `sort_keys=True`, separators `(",", ":")`, ASCII) and pass both §8
checks. Refresh `results/fc/linear_adapter_binary_approval_2026_08_07.json` or any other approval
only with explicit owner approval; until then adapter source/build drift makes the canonical gate
fail by design.

**House rules.** (1) New artifact = source + build script + run script + CSV/MD/log in its
`results/` area + dated memo in `results/reports/` + gate hook if it guards a claim; suites exit
non-zero on any failure. (2) Validate against an independent oracle (host reference or unchanged
Orca online path), never the code under test. (3) No performance claim without an A/B at the
consumer's real shape. (4) A component PASS never skips a claim gate.

**Terminal-integration invariants** (re-establish after any related change):
1. `OwnedLayerMaterial` is move-only and adopts record + state atomically.
2. Record/state files are owner-private regular non-symlinks.
3. Private vectors are scrubbed on every failure, reset, move, and destruction.
4. Widths, counts, digests, identity, invocation, ordinal, party, and SID bind exactly.
5. `OrcaBase` flat-key parsing and stock semantics are unchanged.
6. Exactly one terminal linear callback, then one output; all else rejects.
7. Masking and reconstruction stay on the GPU.
8. Output reveals only additive clear-output shares.
9. Both parties run the same fixed-width preflight and reject together.
10. Macro-off `orca_inference` neither includes nor links Ring-LPN.
11. The historical `ORCA_RINGLPN_FC_KEYS` keywriter is unused by this path.
12. Temporary records/shares/auth/ledgers/outputs are owner-only and deleted.
13. Builds are repeatable; retained provenance verifies against the artifacts.
14. Claim scope: terminal FC/Conv2D, same host, public bias, feasibility only.

**Commit protocol.** One coherent slice per commit, after its applicable check: stage exact paths
or hunks → `git diff --cached --stat`, `--check`, full `git diff --cached` → confirm no private
file, raw record, ledger, secret, scratch path, host identifier, generated timing drift, or
unrelated hunk → scoped message (`ringlpn:`, `docs:`, `evidence:`) → never amend; corrections get
new commits and rerun the affected gate. Never push internal `master`.

## 12. Chief-of-staff checklist (reusable; replaces `results/reports/chief_of_staff_handoff_2026_09_10.md`)

1. Baseline without mutating: `git status --short --branch`, recent log, remotes, worktrees,
   submodules, `nvidia-smi`.
2. Classify every dirty and untracked path by workstream, owner, dependency, evidence, and
   proposed commit; read each shared file's full diff first.
3. Review integration edges, not style: public ABI, producer/state formats, stock kernels
   (`fss/gpu_matmul*`, `gpu_conv2d*`, `backend/orca_base.h`), Sytorch callbacks,
   `orca_inference.cu`, comms, build/provenance, approvals.
4. Static review targets: integer widths, file mode/owner/symlink checks, scrubbing, digest
   order, preflight deadlocks, fixed ports, process cleanup, macro isolation, deterministic
   builds, "current" claims about old binaries.
5. Run the narrowest check after each fix; the canonical gate only after focused checks and
   approvals pass. Never weaken a gate.
6. Snapshot status before a gate; afterwards keep only intended evidence refreshes and never
   "clean" a file that was dirty before the run.
7. Commit per the protocol above; keep a ledger of hash, message, check.
8. Final report: decision; findings by severity; fixes; exact commands and results; commit
   ledger; remaining worktree; blockers; next three actions.

## 13. Roadmap

1. Preserve consumed and retained state; keep the full-graph boundary exact.
2. Obtain a reviewed structured-code reduction and parameters; then reduce Phase B/rounds and remeasure.
3. Replace the trusted nonlinear adapter with a reviewed dealerless protocol.
4. Run repeated authenticated LAN/WAN two-host trials and the clean-clone publication mode.
5. Owner decisions on authorship, credit, reuse, outreach, and the public `origin/master`
   history before any further circulation.

## 14. Version control facts

- Remotes: `origin` = `github.com/mottopanikeiku/EzPC` (public fork); `upstream` =
  `mpc-msri/EzPC`; `gpu-mpc` = `mottopanikeiku/GPU-MPC`.
- `origin/master` = `9685463` (pushed 2026-09-11) is an ancestor of internal `master`, so internal
  history through it is public, including the internal-only 2026-08-10 full-graph checkpoint (18
  files with `private_inputs/` provenance) and the 34-page v2.17 PDF build (`dd1de27a…`). Later
  commits are unpublished. Never push `master` again; restricting or rewriting the public history
  is an owner ruling (§6).
- Public branch history: `1433e0a` base → `a1f006a` → `4a2e29a` → `afc93c6` (published,
  remote-verified 2026-10-06) → `9bf1ab0` → 12 source-cleanup mirror commits ending at `ac711e2`
  (all local only).
- Bundles in `~/.local/share/ringlpn/artifacts/source-bundles/`: `ringlpn-source-audit-afc93c6.bundle`
  (sha256 `eb6d7d98…`, requires `4a2e29a`), `ringlpn-source-9bf1ab0.bundle`, and
  `ringlpn-source-ac711e2.bundle` (§6); the last two require `afc93c6`.
- Recent commits use Git author `mottopanikeiku`; the paper names Alp `<fcetin@hawk.iit.edu>` by
  user direction. Authorship and credit remain owner rulings.

## 15. Documentation contract (binding)

Before ending any session that changed code, results, or plans:

1. Update this file: status, claims, gates, source map, perf anchors, gotchas.
2. Add or update the row in `results/README.md`.
3. Write or refresh a dated memo in `results/reports/` with reproduction commands.
4. Banner superseded documents (`> **HISTORICAL …** superseded by CLAUDE.md`) or move unbound
   ones to `results/archive/reports/`; never edit hash-bound bytes outside the §11 rebind rule.
5. Never let two live documents disagree: the newer gate-verified statement wins; fix or banner
   the other in the same commit.
6. `results/outreach/`, `results/archive/`, and bannered reports are read-only history: quote
   them, do not trust them.

Key reports (`results/reports/`): security contract `dealerless_orca_fc_security_contract_2026_07_29.md`,
roadmap `publication_readiness_plan_2026_07_21.md`, attack audit `structured_attack_audit_2026_08_04.md`,
two-host contract `authenticated_two_host_deployment_2026_08_04.md`, portfolio
`publication_portfolio_2026_08_04.md`; security artifacts: `results/security/README.md`.
