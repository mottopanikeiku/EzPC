# ringlpn results — index

Each row gives what an artifact proves and its status. Claim boundaries,
numbers, and open gates are in [`../CLAUDE.md`](../CLAUDE.md); build and run
commands are in [`../README.md`](../README.md).

Status values:
- **live**: current evidence.
- **bound**: live and hash-bound; never edit the bytes.
- **historical**: correct for its date, superseded since.
- **negative**: a measured or audited NO-GO.

Files marked bound are listed in
`../scripts/publication_environment_manifest_2026_08_10.json`
`required_tracked_evidence`, pinned by its `build.publication_*_sha256` (v2.17
TeX/PDF), or pinned by other JSON. After any change under `results/`, run both
evidence checks in `../CLAUDE.md` §8: the private-artifact guard does not
re-hash bound files; the bound-digest check does.

## Directories

| Directory | Status | Contents and what it proves | Produced by |
|---|---|---|---|
| `fc/` | live, bound, negative | Live two-process FC: 5 SCI/IKNP q64/q128 cases with 16 controls. Opt-in EMP-Silent rerun. Controlled CNN2/CNN3 matrix (strongly negative). ResNet18 classifier trials. 21-layer manifests and approvals. SCI duplex experiment. Failed prospective capacity gate. | `run_two_party_fc_preprocess.sh`, `run_two_party_fc_model_scale.sh`, `run_fc_review_experiments.py` |
| `conv/` | live | Focused Conv2D rows and controls; `conv0_breadth_comparison_2026_08_09/` holds older-binary Conv0 metrics (historical, no speedup claim) | `run_two_party_conv_preprocess.sh` |
| `application/` | bound | Terminal Orca FC/Conv2D pass/control rows, gate log, and build provenance | `run_orca_linear_application.sh` |
| `graph/` | bound, historical | `resnet18_full_graph_checkpoint_2026_08_10/`: immutable, internal-only known-zero full-graph checkpoint (a TEST-ONLY trusted adapter supplies nonlinear keys and truncation/remask/terminal masks; linear records were first consumed by an omniscient checker). `resnet18_graph_prefix_*_2026_08_09.*`: superseded prefix regression. | `run_resnet18_full_graph.sh`, `run_resnet18_graph_prefix.sh` |
| `secure_convert/` | live | Two-process exact `Z_M → Z_2^bw` conversion (76/76 × 4 configs), split transcript counters | `run_secure_convert_test.sh` |
| `secure_truncate/` | live | Isolated secure truncation check | `run_secure_truncate_test.sh` |
| `dpf/` | live, historical | Two-process DPF keygen transport (369/369), GPU-AES parity and 88 GPU-evaluated keys, ideal host prototype, DMPF microbenchmarks (07-29), standalone DPF online-keygen sweeps | `run_two_party_dpf_keygen.sh`, `run_two_party_gpu_dpf.sh`, `run_distributed_dpf_keygen.sh`, `run_dmpf_*.sh` |
| `ole/` | historical | Figure-2 OLE q64/q128 × uniform/regular component rows; two-process SPFSS key rows; t=8/t=64 feasibility only | `run_ole_sweep.sh`, `run_ole_two_party_keys.sh` |
| `linear_ole/` | historical | Ring-matrix OLE→Beaver 2×2×2, n=8192, q64/q128 | `run_linear_ole_sweep.sh` |
| `orca_fc/` | historical | Keywriter demo, ideal- and real-OLE transcripts, `Z_p` bridge (with q62/32-bit counterexample) | `run_orca_fc_*.sh`, `run_orca_zp_bridge_test.sh` |
| `ntt/` | historical | CPU NFLlib and GPU Cheddar q32/q64/q128 sweeps, legacy CUDA, GPU-NTT baseline | `run_sweep.sh`, `run_cuda_sweep*.sh`, `run_ntt_baseline_compare.sh` |
| `vole/` | historical | Standalone VOLE expansion prototype | `run_vole_sweep.sh` |
| `pcg/` | negative | Adapted native-`Z_(2^bw)` PCG rows with patch digest and gate; not a reproduction | `run_native_ring_pcg_baseline.sh` |
| `security/` | live, negative | Start at [`security/README.md`](security/README.md). Regular-projection law, 2024 regular-ISD and 2025 hybrid-RSD formula artifacts; invalid conservative-pin transcripts. Diagnostics only, no parameter pin. | `audit_*.py` |
| `profiling/` | historical | VTune hotspot text summaries; raw result directories are local-only and ignored | `run_vtune_*.sh` |
| `outreach/` | historical | Abstracts, posters, professor memos; an unsent Reverse Cuckoo request draft awaits owner approval | hand-written |
| `archive/` | historical | Superseded one-off runs and plan drafts; `archive/reports/` holds superseded reports | frozen |
| `reports/` | see below | Memos, contracts, audits, manuscripts, JSON verification records | hand-written |

## Reports (`reports/`)

| File | Date | Status | What it is |
|---|---|---|---|
| `autonomous_technical_closure_2026_09_22.json` | 09-22 | bound | Latest engineering record: conversion and leaf audits, direct-OT baseline, runtime candidate, `9bf1ab0` push attempts (not pushed) |
| `technical_followthrough_2026_09_22.json` | 09-22 | bound | Conversion-simulator and leaf-loss audits; `afc93c6` publication; prerequisite matrix |
| `direct_ot_fc_complete_2026_09_22.json` + `.plan.json` | 09-22 | bound | Direct-Gilboa-OT FC baseline, 8 cases, median 1.079694043 s; no Ring-LPN ratio |
| `direct_ot_fc_{baseline,functional,verified,comparison,final}_2026_09_22.json` + `.plan.json` | 09-22 | bound, historical | Earlier direct-OT plan iterations, superseded by `complete` |
| `independent_review_verification_2026_09_11.json` | 09-11 | bound | 15 automated-review findings; SCI duplex experiment; OT inventory; public-source commits |
| `paper_review_verification_2026_09_11.json` | 09-11 | bound | v2.17 36-page PDF build verification (sha `f0544ee1…`) |
| `engineering_review_verification_2026_09_10.json` | 09-10 | bound | Canonical gate `ALL GATES PASS` (5,228.406 s), fresh graph digest `ccd598f8…` |
| `chief_of_staff_handoff_2026_09_10.md` | 09-10/22 | historical | Operating brief; its checklist now lives in `../CLAUDE.md` §11–12 |
| `paper_review_verification_2026_09_10.json` | 09-10 | bound, historical | 34-page build (sha `dd1de2…`), superseded by the 09-11 build |
| `orca_linear_application_integration_2026_08_24.md` | 08-24 | bound | Terminal Orca FC/Conv2D integration design, verification, hashes |
| `dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex` + `.pdf` | 08-17 | bound | Live internal/advisor manuscript, 36 pages; not a submission |
| `dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` + `.pdf` | 07-10 | historical | v2.16 predecessor; named by the 2026-08-04 manifest in git history |
| `regular_dmpf_design_no_go_2026_08_06.md` | 08-06 | bound, negative | Specialized regular-DMPF design NO-GO; selects the full-linear route |
| `full_linear_layer_systems_plan_2026_08_06.md` | 08-06 | historical | Plan that led to the 21-record/full-graph checkpoint (kept in place: a bound report links it) |
| `dealerless_orca_fc_security_contract_2026_07_29.md` | 09-22 | bound | Forward security contract, simulators, P-* gate table, source map |
| `structured_attack_audit_2026_08_04.md` | 08-10 | bound | Attack inventory; projection law, regular-ISD, hybrid-RSD rebound (`fbdb56f8…`, `cbcedaf6…`/`1f671d94…`); no pin |
| `authenticated_two_host_deployment_2026_08_04.md` | 09-11 | bound | Two-host deployment contract; not executed |
| `publication_portfolio_2026_08_04.md` | 08-10 | live | Systems vs cryptography tracks and their blocking gates |
| `closest_dmpf_baseline_audit_2026_08_04.md` | 08-04 | live | Reverse Cuckoo/libOTe as closest distributed baseline; noncomparability rules |
| `libote_reverse_cuckoo_stock_baseline_2026_08_04.md` | 08-04 | negative | Stock libOTe run: 12.43 s, 22,939,444 KiB RSS; no ratio |
| `reverse_cuckoo_p0_baseline_2026_08_04.json` | 08-04 | live | Exact-p0 native-folded row: setup 18,523,424 µs, online 2,116,894 µs; schema `reverse_cuckoo_p0_baseline_schema_2026_08_04.json` |
| `native_ring_technology_audit_2026_08_04.md` | 08-04 | negative | Native-ring QA-SD PCG NO-GO |
| `s2_regular_projection_law_2026_08_04.md` | 08-10 | live | Exact projection/cancellation law bound to the live sampler |
| `s2_parameter_novelty_provenance_audit_2026_07_29.md` | 08-04 | live, negative | S2 hard stop: invalid estimator calls, `n=2^17,c=4,t=34` NO-GO, no pin |
| `s2_architecture_comparison_2026_07_29.md` | 07-29 | historical | Encoder microbenchmarks (275×, 0.79×, 2.29×); a bound report cites it |
| `s2_professor_decision_request_2026_07_29.md` | 07-29 | historical | Advisor questions; the provenance/credit questions remain open |
| `publication_readiness_plan_2026_07_21.md` | 09-11 | live | S1–S10 and M1–M6 roadmap and gates |
| `two_party_dpf_transport_memo_2026_07_29.md` | 07-29 | historical | SCI/IKNP/Gilboa transport component evidence |
| `dealerless_ole_two_party_keys_memo_2026_07_29.md` | 07-29 | historical | Paired-record two-process SPFSS key evidence |
| `distributed_dpf_keygen_memo_2026_07_21.md` | 07-21 | historical | Ideal-OT host prototype, 2,432-tree controls |
| `ntt_baseline_comparison_2026_06_10.md` | 06-10 | negative | GPU-NTT 1.2–3.9× faster but no 62-bit primes; keep Cheddar |
| `baseline_2026_06_10.md` | 06-10 | historical | June environment, PASS counts, anchors |
| `orca_fc_real_ole_transcript_memo.md` | 06-10 | historical | Single-process real-OLE transcript diagnostic |
| `ole_figure2_host_results.md` | 04-21 | historical | Host OLE validation (135/57/36) |

`archive/reports/` holds superseded reports that no bound file links to.
These are four session handoffs (07-10, 07-21, 07-29, 08-09), the May linear
integration plan, the pre-June status report, Cheddar note, next-steps memo,
FC demo memo, per-artifact handoffs, and three pre-v2 TeX drafts. Cheddar
provenance is in `../extern/Cheddar_PROVENANCE.txt`.

## Conventions

- Every run writes `.csv` (data), usually `.md` (summary), and `.log` (raw
  output). `validation` and `*_contract` columns must read `pass`; suites exit
  non-zero on any failure.
- The root `.gitignore` ignores `*.csv` and `*.txt`, and `ringlpn/.gitignore` ignores `*.pdf`, so add evidence
  with `git add -f`.
- Raw private records, states, ledgers, shares, and auth files are never
  evidence and are ignored.
- Staleness: superseded documents carry a `> **HISTORICAL …**` banner, or
  they move to `archive/reports/` when nothing bound links them.
  `outreach/` and `archive/` are wholly historical. Banner or archive in the
  same commit that supersedes a document.

## Clean-clone and two-host reproduction

The clean-clone configuration is
`../scripts/publication_environment_manifest_2026_08_10.json` (schema
`ringlpn-publication-environment/v4`) plus `../scripts/Dockerfile.reproduction`.
The dispatcher is `../scripts/reproduce_publication.sh`, which fails closed.
The schema-v1 2026-08-04 manifest was removed and survives in git history.

The Docker image serves only clean-clone `check`, `local-smoke`, and build
gates. It holds no source or credentials and never gets a Podman socket. The
manifest pins `platform.reproduction_gate_image_id=sha256:63b6d387733d145bd26e6d1ada75c3dea568c5144f0e2831dc3590079ab9bd35`.
No two-build reproducibility receipt exists. Every visible CPU must have
`aes`, `avx2`, `pclmulqdq`, `rdseed`, and `sse4_1`.

`check` and `local-smoke` are currently blocked. On the host both first run
`verify_authorized_worktree` (`../scripts/reproduce_publication.sh:22-50,165`):
HEAD must be the annotated tag `ringlpn-publication-candidate-v1` (manifest
`source_release.required_annotated_tag`) with a matching `GPU-MPC/ringlpn` tree.
No such tag exists; `check` exits 1 with "cannot resolve authorized tagged
commit". `local-smoke` also requires the local image ID to equal the pin. The
local `ringlpn-repro:2026-08-10` already matches
(`docker image inspect --format '{{.Id}}' ringlpn-repro:2026-08-10`), so run the
`--no-cache` build below only on a host without it; a rebuild can retag a
different ID.

```bash
SOURCE_DATE_EPOCH=1786320000 docker build --no-cache \
  --build-arg SOURCE_DATE_EPOCH=1786320000 \
  -f GPU-MPC/ringlpn/scripts/Dockerfile.reproduction \
  -t ringlpn-repro:2026-08-10 GPU-MPC/ringlpn/scripts
install -d -m 700 /absolute/mount/local-smoke-evidence
RINGLPN_REPRODUCTION_IMAGE=ringlpn-repro:2026-08-10 \
RINGLPN_EVIDENCE_DIR=/absolute/mount/local-smoke-evidence \
  ./GPU-MPC/ringlpn/scripts/reproduce_publication.sh local-smoke
# [ringlpn-reproduce] LOCAL SMOKE PASS — NOT TWO-HOST PUBLICATION EVIDENCE
```

Local smoke mounts the clone read-only and runs in an owner-private copy. It
retains only sanitized external evidence. GPU roles default to 0/1/2: export
idle, pairwise-distinct `P0_GPU`/`P1_GPU`/`CHECK_GPU` (forwarded as the
`FULL_GRAPH_*` roles). The dispatcher forwards neither `ORCA_LINEAR_*` nor
`CUDA_VISIBLE_DEVICES`, so in the container the application gate always uses
GPUs 0/1 and inheriting stages use GPU 0; do not run local smoke while GPU 0 or
1 is occupied.

Two-host publication runs on the coordinator host and uses each host's native
rootless Podman, never `docker run`. It is blocked: annotated-tag/operator
source authorization and the immutable runtime digest are unset. Evidence,
local-smoke evidence, and the coordinator and party ledgers must be
owner-only, pairwise non-nested, block-backed mounts. Ledgers are retained and
never published or rolled back. Once the release prerequisites exist:

```bash
SESSION="$(date -u +%Y%m%d%H%M%S)"
export RINGLPN_LOCAL_PARTY_LEDGER=/absolute/mount/persistent-party0-ledger
INVOCATION="$(openssl rand -hex 16)"
export RINGLPN_EVIDENCE_DIR=/absolute/mount/publication-evidence
export RINGLPN_LEDGER_DIR=/absolute/mount/persistent-ledger
export RINGLPN_LOCAL_SMOKE_EVIDENCE=/absolute/mount/local-smoke-evidence
export RINGLPN_SOURCE_AUTHORIZATION=/absolute/readonly/source-authorization.json
export RINGLPN_RUNTIME_MANIFEST="$RINGLPN_EVIDENCE_DIR/runtime-$INVOCATION.json"

./GPU-MPC/ringlpn/scripts/reproduce_publication.sh two-host-publication \
  --checker-container-uid 10003 --checker-gpu 2 -- \
  --peer USER@REMOTE_HOST --identity /absolute/private-ssh/id_ed25519 \
  --known-hosts /absolute/private-ssh/known_hosts \
  --remote-executor /absolute/remote/EzPC/GPU-MPC/ringlpn/scripts/peer_private_execution.py \
  --local-private-root "/absolute/private/$INVOCATION-p0" \
  --remote-private-root "/absolute/remote/private/$INVOCATION-p1" \
  --local-party-ledger-root "$RINGLPN_LOCAL_PARTY_LEDGER" \
  --remote-party-ledger-root "/absolute/remote/persistent-party1-ledger" \
  --local-party-manifest "$RINGLPN_EVIDENCE_DIR/$INVOCATION/party0-sealed.json" \
  --remote-party-manifest "/absolute/remote/evidence/$INVOCATION/party1-sealed.json" \
  --remote-peer-manifest "/absolute/remote/evidence/$INVOCATION/party0-peer.json" \
  --local-export-root "/absolute/private/$INVOCATION-p0-export" \
  --remote-export-root "/absolute/remote/private/$INVOCATION-p1-export" \
  --checker-stage "$RINGLPN_EVIDENCE_DIR/$INVOCATION/checker-stage" \
  --output-dir "$RINGLPN_EVIDENCE_DIR/$INVOCATION" \
  --local-container-uid 10001 --remote-container-uid 10002 \
  --local-gpu 0 --remote-gpu 1 \
  --session-id "$SESSION" --invocation-id "$INVOCATION" \
  --ledger-root "$RINGLPN_LEDGER_DIR" --base-port 30000 \
  --qbits 128 --bw 32 --rows 1 --inner 512 --cols 1000 \
  --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular
```

A run succeeds only after private and export roots, checker records, channel
keys, and containers are removed on both hosts. `deletion-receipt.json` and
`checker-stage/COMMITTED.manifest` bind that cleanup. The evidence manifest is
committed last and inventories every retained public file by relative path,
size, and SHA-256. The full deployment contract is
[`reports/authenticated_two_host_deployment_2026_08_04.md`](reports/authenticated_two_host_deployment_2026_08_04.md).
