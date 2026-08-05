# ringlpn — agent & human catch-up guide

**What this is:** a research subproject building *dealerless* preprocessing for
Orca (the GPU FSS-based secure ML system in this repo) from Ring-LPN
pseudorandom correlation generators (PCGs). Orca's linear layers consume
Beaver-triple keys that a trusted dealer normally produces; this project
replaces the dealer with a two-party protocol: GPU NTT/polynomial arithmetic →
Z_p SPFSS (sum of DPFs) → Figure 2 Ring-LPN OLE → slot-packed Beaver cross
terms → Z_M→Z_2^bw conversion → byte-compatible Orca keys, validated through
Orca's **unchanged** online path (`gpuMatmulBeaver`).

**Status (2026-08-04): the current live source is a dealer/oracle-free,
GPU-batched two-process forward-FC path at feasibility parameters. It uses the
required `a=(1,a1,...,a_{c-1})` distribution, a canonical 128-bit
invocation/256-bit correlation namespace, and a persistent consume-once ledger.
Current q64/q128 suites and a ten-trial exact ResNet18 classifier-layer rerun
pass. A conference/security-level claim remains a NO-GO.**

The current live composition is
`src/test_two_party_fc_preprocess.cu` plus:

- `src/correlation_freshness.h`: canonical fixed-width correlation IDs and the
  owner-only append-only consume-before-release ledger;
- `src/two_party_spfss.h`: party-local sparse-noise binding and distributed
  SPFSS key generation;
- `src/two_party_dpf_protocol.h` / `src/two_party_ot.h`: full-width GPU-AES
  DPF semantics over real SCI/IKNP and Gilboa OLE;
- `src/ringlpn_ole_party.cuh`: party-local Figure-2 Ring-LPN expansion;
- `src/secure_convert.{h,cpp}`: exact two-process `Z_M -> Z_2^bw`
  conversion; and
- the unchanged Orca `readGPUMatmulKey` / `gpuMatmulBeaver` consumer, called
  only by a post-exit checker.

Each live party is a separate OS process on a distinct GPU, reads only its own
noise record, and samples private roots/masks/noise with OpenSSL's DRBG. Before
OT setup or DRBG construction it claims the complete high-entropy public
invocation namespace in a private persistent ledger; duplicate/restart,
truncated/colliding state, and tail reuse fail before publication. The claim
digest and invocation ID are bound into preflight and both version-2 records.
The public Ring-LPN vector is exactly `a=(1,a1,...,a_{c-1})`: the identity
polynomial is unsent, and each party exchanges one full uniform field-element
share for each of the `(c-1)*n` remaining coefficients. The runner has no
selectable centralized DPF keygen, clear conversion, dealer, or oracle path. Its
temp-write/rename/peer-ack publication is bilateral best-effort, not
crash-transactional. Raw records are never public evidence; only the
authenticated coordinator's sealed two-record, fsynced digest-bound
`COMMITTED.manifest` admits a consumer.

**Executable evidence.**

- `results/fc/two_party_fc_preprocess_2026_08_04.csv`: six current
  q64/q128 regular/uniform/small/multi-batch rows. Every key-order,
  current-transcript, record, and unchanged-online contract passes.
- `results/fc/two_party_fc_preprocess_controls_2026_08_04.csv`: ten current
  duplicate/restart/tail-reuse/invocation-collision/ledger-truncation/preflight/
  stale/rename/corrupt/swapped controls; every expected rejection passes.
- `results/fc/two_party_fc_model_scale_2026_08_04.csv` plus aggregate,
  summary, schemas, controls, and environment: exact ResNet18 classifier-layer
  `1x512x1000`, q128/bw32 feasibility `(n,c,t)=(8192,2,8)`, one warmup plus ten
  current measured trials, 10/10 pass. Median critical-path preprocessing is
  25.715 s, application traffic is 575,846,872 bytes, matched stock
  `gpuKeygenMatmul` is 10.642 ms median, unchanged two-share online execution
  is 1.106 ms median, and final payload is 4,108,096 bytes per party.
  The aggregate records 10,338 protocol dependency stages, 142,493,696 peak
  host bytes, and 27,241,086,976 peak GPU bytes. This is not a full ResNet18
  inference or scale-10 truncation run.
- The model result's median Phase-C payload correction is 20.432 s, 79.4% of
  critical-path preprocessing. GPU Ring-LPN expansion is 1.704 s median.
  GPU-batched DPF generation is live; the tree-per-point Phase-C protocol, not
  the former host-only implementation, is now the dominant bottleneck.

The matched comparison is deliberately negative: median preprocessing is
2,414 times stock dealer keygen. These measurements use single-host IPv4
loopback. The application-byte counter excludes TCP/IP framing and base OT;
the separate transport counter includes the measured 43,658 base-OT bytes.
They prove feasibility/correctness and locate the bottleneck, not a speedup or
WAN behavior.

**Proof boundary.** The corrected security contract and report contain:

1. exact level-by-level and final-CW coupling to standard DPF generation
   conditioned on party roots;
2. complete role-specific simulators for correlated/repeated DPF batches,
   including the three-OLE Phase C without the removed sign leak;
3. exact ideal-OT wrapper lemmas and a `Z_Q -> Z_2^bw` conversion simulator in
   the edaBit/daBit/Boolean-triple hybrid;
4. a role-indexed Figure-2 simulator that retains the corrupt party's local
   noise/key state and recomputes both
   `X_b=e_(b,0)+sum_(i>=1)a_i e_(b,i)` and its local `Z_b`;
5. a canonical `P-FRESH` functionality matching the fixed-width source tuple,
   consume-once ledger, record binding, and exact SHA-256/durable-filesystem
   assumptions;
6. an updated live source-to-transcript map; and
7. a conditional static-semi-honest theorem for one forward FC matmul under
   authenticated channels, standard AES/DPF/IKNP/Gilboa, SHA-256 and durable
   no-rollback ledger assumptions, and exact decisional module-Ring-LPN for
   `a=(1,a1,...,a_{c-1})`.

Renewed model-assisted source, composition, and proof audits were run after the
identity, freshness, batching, and transport changes; they do not substitute
for independent human cryptographic review, which remains open. The live
evidence remains unauthenticated local-only loopback. The pinned-SSH
two-stream, peer-private deployment boundary is implemented in
`scripts/run_two_host_authenticated.sh` and documented in
`results/reports/authenticated_two_host_deployment_2026_08_04.md`; no
authenticated two-host result is claimed until that launcher is run and its
durable digest-bound `COMMITTED.manifest` passes the checker gate.

**Hard blockers.** The exact regular-projection distribution and cancellation
law is now machine-checkable, but no reviewed reduction maps the structured
projected code to a concrete attack advantage, and no two-CRT-limb composition
supports a security level. q64/q128 mean one/two approximately 62-bit
arithmetic limbs, not 64/128-bit security. No `(n,c,t,p0,p1)` set is pinned.
Other publication gates are Phase-C/DMPF redesign and performance, reviewed and
measured silent OT/VOLE, authenticated two-host LAN/WAN runs, every forward
linear layer plus truncation/state handoff for one real model, a
functionality-compatible dealerless-PCG baseline, clean-clone reproduction,
and independent human proof/source review. Training state transitions,
nonlinear DCF keys, malicious security, and full dealerless Orca remain out of
scope.

The exact S2 audit is
`results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md`.
It invalidates the former “conservative pins”: BCG+20's projection
formulas/prose/Table 1 conflict, several saved estimator calls are
out-of-domain, and surviving finite-field rows lack the necessary reduction.
The `n=2^17,c=4,t=34` implementation NO-GO remains an engineering result, but
its former 257.02-bit label is withdrawn.
The current attack inventory is
`results/reports/structured_attack_audit_2026_08_04.md`. It separates exact
direct RSD from projected occupancy/cancellation and proves the cyclic orbit,
but neither generic-estimator rows nor a square-root orbit sensitivity are
reviewed concrete Ring-LPN security. Modern direct RSD and 2025/2026 QA-SD
dispositions, structured-code reductions, resource/success accounting, and
independent human review remain parameter-pin blockers.

**Paper and publication verdict.** The current TeX source and rebuilt PDF are
v2.6 at
`results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.{tex,pdf}`.
The 24-page PDF is warning-free and every rendered page was inspected after
the current evidence/proof/limitations update. It is a strong internal
advisor checkpoint, not a submission-ready paper. A crypto paper needs a new
protocol/reduction/parameter result; a systems paper needs a much faster
end-to-end implementation and broader model/network evaluation.
The binding route is
`results/reports/publication_readiness_plan_2026_07_21.md`; the current proof
boundary is
`results/reports/dealerless_orca_fc_security_contract_2026_07_29.md`.

The complete canonical component gate must be rerun after any source change:
`RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH
./scripts/run_paper_checkpoint_smoke.sh`.
The separate live/model runners are
`scripts/run_two_party_fc_preprocess.sh` and
`scripts/run_two_party_fc_model_scale.sh`.

**Contribution/provenance boundary.** The paper thesis is the integrated
forward-FC systems path. The per-point distributed DPF, Ring-LPN generator,
conversion primitives, Orca, and GPU polynomial backends are prior/inherited
work, not protocol contributions. The separate private GPU-PCG/PIM stream has
multiple contributors, no repository license, and unresolved ownership,
credit, chronology, reuse, and overlap decisions; do not import or claim it.
Cheddar remains an attributed MIT-licensed dependency under
`extern/Cheddar_{PROVENANCE,MIT_LICENSE}.txt`; GPU-NTT remains an external
cited baseline. Alp `<fcetin@hawk.iit.edu>` is the sole paper/repository author
by user direction, but that does not erase attribution or resolve reuse rights.
Before external circulation, obtain the professor's provenance/credit/reuse
decisions recorded as open in
`results/reports/s2_professor_decision_request_2026_07_29.md`.

The first architecture comparison is measured in
`results/reports/s2_architecture_comparison_2026_07_29.md`. Its headline result
is negative and must not be misquoted. With uniform noise an OKVS-style DMPF
expands the sparse product 275x faster than the then-current sum of point DPFs
at `(n,c,t)=(2^14,4,16)`. With the deployed regular layout
(`spfss_domain=2048`, `log_domain=11`, 31 diagonal groups per pair), the same
encoder is 0.79x (slower), while the big-state candidate wins 2.29x at 37x the
key bytes. This is a microarchitecture result, not an end-to-end result.

Reverse Cuckoo became public on 2026-08-03. The pinned stock libOTe run in
`results/reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md` completes
in 12.43 s with 22,939,444 KiB peak RSS, but uses Goldilocks, internally
sampled factors, a native 16-folded layout, synthetic preloaded base
correlations, CPU local sockets, and no GPU output. It is the closest
reproducible distributed baseline, not a compatible comparator, and no timing
ratio is claimable. The separate native-ring artifact remains an adapted
diagnostic, never a reproduced dealerless result.

**2026-08-04 closest-baseline correction (supersedes only the no-code/public-source
sentences above; preserves the 2026-07-29 measurement as history).** Reverse
Cuckoo became public on 2026-08-03. The current source-pinned ranking and claim
boundary are in
`results/reports/closest_dmpf_baseline_audit_2026_08_04.md`: MIT-licensed
`osu-crypto/libOTe:dmpf@edb5d32822eabf2dda9f6844d85d0ce2e402cdd5` is the
rank-1 distributed candidate, with paper source
`ladnir/dmpf@b55bcc4696d10e57bdea8c282a851fdd4fad0c2b`. The stock runner is
not zero-change exact, GPU, or setup-inclusive evidence: it uses a mismatched
field, internally sampled factors, native 16-folded layout, CPU expansion, and
synthetic base correlations. The companion
`results/reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md` records a
pinned clean run using the required `-bench` dispatcher: 12.43-s process wall,
22,939,444-KiB peak RSS, 11-s printed internal total, and 446.448-ms synthetic
`setBase`. Live `genBaseCors` was excluded. The literal command without
`-bench` only printed help and exited 0. This separately labelled stock row
does not permit a speedup claim. The separate completed
`results/reports/reverse_cuckoo_p0_baseline_2026_08_04.json` exercises exact
caller-factor `p0`, the canonical 62-bit context, live `genBaseCors`, duplicate
accumulation, full-domain differential validation, and corruption rejection for
the explicitly labelled native 16-folded layout. It records 18,832,990-us
setup, 2,070,844-us online full-domain evaluation, and 20,948,042-us end-to-end
including validation. This is not raw 31-diagonal timing or GPU evidence;
speedup/security claims remain null, and no ratio may cross layout, field,
trust, correlation, execution, or setup boundaries.

**2026-08-04 primary-source parameter correction (supersedes the 2026-07-29
“conservative pin” interpretation).** The 2026-07-29 owner decision still lifts
the S2->S3 ordering gate **for implementation only**: GPU distributed-keygen
and real OT/OLE transport work may proceed without a security or parameter
claim. Its requested “conservative minimum” selection method is not usable.
BCG+20's corrected full version is internally inconsistent: Section 8.2 derives
`c*d*(1-(1-1/d)^t)`, while Section 9.1 uses
`w-c*d+(c*(d-1)+w)*(1-1/d)^(t-1)`; its literal smallest-factor criterion
selects degree 16 for `(c,w)=(4,64)`, while Table 1 reports degree 128. No
published erratum or proof resolves these differences.

The accepted EUROCRYPT 2024 artifact also cannot evaluate every locally
“admitted” row. Its aggregate finite-field function unconditionally calls
formulas containing `C(N-k,t)` and `C(N-k-1,t)`, so a projected row must at
least satisfy `t' <= N-k-1 = d-1`. The artifact's combination helper silently
returns 1 for out-of-range inputs. Consequently the saved 57.293-bit
`(c,t,d)=(4,16,16)`, 218.641-bit `(4,64,64)`, and 257.023-bit
`(4,34,64)` values are invalid estimator calls, not attack costs. For
`c=4,t=16`, the first mechanically defined saved row is degree 64 at 135.12
regular-model bits; BCG's degree-128 row gives 145.85. For `c=4,t=64`, the
first mechanically defined saved row is degree 256 at 470.77. The
`c=2,t=128,d=256` row is mechanically defined and reports 190.53, but all of
these finite-field numbers remain heuristics because no reviewed reduction
maps the dependent projected noise/code to the estimator's exact or regular
model, bounds the lower tail and rounding loss, or composes the two limbs and
PCG advantage.

All `results/security/*conservative_pin*` artifacts and
`s2_conservative_parameter_pin_2026_07_29.csv` are historical failed-rule
transcripts, invalid for parameter selection or security claims.
`s2_projection_estimator_preliminary_2026_07_29.csv` is a raw function
transcript only; rows with `floor(expected) > degree-1` are invalid calls and
all other rows are unproved model outputs. No `(n,c,t,p0,p1)` set is pinned,
no 128-bit classical or quantum claim is made, and S2 remains blocked pending
a reviewed sparse-factor projection/distribution/tail/structured-code and
advantage-composition analysis for both limbs.

**2026-07-29 owner route decisions (recorded after the measured comparison).**
Presented `results/reports/s2_architecture_comparison_2026_07_29.md` §6-§7 to the
owner; the four answers are binding:
1. **Encoder:** keep the per-point DPF. Implementation effort goes to *real*
   silent-OT/OLE transports and a two-process deployment. Rationale is measured,
   not aesthetic: at the deployed regular layout the best DMPF wins only
   2.29x expansion at 37x the key bytes, and OKVS is 0.79x (slower), so the
   encoder is not the pipeline's lever. A dealerless DMPF stays future work.
2. **Parameters:** the requested immediate conservative pin is superseded by
   the 2026-08-04 primary-source correction above. Obtain a reviewed
   projection/distribution/tail/structured-code and two-limb advantage
   reduction before another estimator sweep. Current rows are feasibility-only.
3. **Claim scope:** the paper's headline claim waits for a real two-process
   dealerless FC run. Protocol-logic slices are supporting evidence only, never
   the contribution.
4. **Prior art:** draft (do not send) an artifact/clarification request to the
   Reverse Cuckoo authors for owner approval:
   `results/outreach/reverse_cuckoo_artifact_request_2026_07_29.md`.

**Invalidated parameter-pin transcripts (measured 2026-07-29; corrected
2026-08-04).** The values in
`results/security/ringlpn_conservative_pin_2026_07_29.{csv,log}`,
`ringlpn_conservative_pin_refine_2026_07_29.{csv,log}`,
`ringlpn_conservative_pin_n16_n17_2026_07_29.{csv,log}`, and
`s2_conservative_parameter_pin_2026_07_29.csv` document a failed rule and are
not current evidence. Their `meets_target=yes`, “conservative,” “pin,”
“surviving,” and `t=32 -> 34` projection-eviction interpretations are invalid.
The model values 57.293, 111.244, 218.641, and 257.023 were selected from
out-of-domain aggregate calls. Mechanically defined values such as 135.12,
145.85, 190.53, and 470.77 remain unproved finite-field-model heuristics, not
Ring-LPN security estimates. `scripts/audit_ringlpn_finite_field_models.py`
now rejects undefined tuples, requires both primes, labels outputs as model
diagnostics, and exits nonzero so automation cannot treat them as a pin.

**Real two-party transport (2026-07-29; rerun 2026-08-03).** The frozen keygen
protocol runs as **two OS processes over TCP with real OT**:
`src/test_two_party_dpf_keygen.cpp` + `src/two_party_ot.h`, using this repo's
unmodified SCI IKNP OT extension (header-only, links only OpenSSL), Gilboa
`Z_p` OLE, OT-based Boolean triples, and OpenSSL-private-DRBG party roots.
Keygen is **batched level-synchronously**: the measured direction-switch count
is `6L+6` for depth `L`, independent of batch size. It is not a network-round
count. 369/369 key pairs across ten configurations (depths 4–14, both primes,
batches 1–256) validate through unchanged `dpfEvalAll` in a separate offline
checker with a corrupted-key control. Logical/meaningful-share columns match
the contract's closed forms at every batch size. Setup costs 256 base OTs and
21,829 bytes per party; at `L=11`, batch 1 → 256 lowers per-tree bytes
52,626 → 3,789 (13.9x) and loopback time 11.2 ms → 148 us (75x), with 72
direction switches. IKNP is OT *extension*, not silent OT; splitmix mode is a
host-reference correctness path, not a security claim. See
`results/reports/two_party_dpf_transport_memo_2026_07_29.md`.

**Candidate feasibility result (2026-08-03): NO-GO on the current
implementation.** The measured `n=2^17,c=4,t=34` tuple is not runnable in the
current layouts and has no accepted security estimate. Regular-noise equal
buckets require `t | n`; `34` does not divide a power-of-two ring. Uniform
noise would materialize `7,272,923,136` host slots at 17 bytes each, at least
123.6 GB for one process, before validation and two-party duplication. These
implementation facts remain valid, but the former 257.02-bit and
projection-eviction narrative is withdrawn as an out-of-domain estimator
calculation. No replacement parameter is pinned.

**GPU key compatibility from the two-party protocol (2026-07-29; full-width
PRG correction 2026-08-03).** The deployed Ring-LPN device PRG and its host
twin now use four domain-separated AES calls per node
(`src/gpu_aes_prg_host.h` and `aes_prg_expand` in `src/gpu_spfss_zp.cuh`):
plaintexts 0 and 2 produce full 128-bit child seeds; plaintexts 1 and 3
produce separate control bits. Every GPU run regenerates 16 device vectors,
including low-bit-one seeds; host parity reports zero left/right/tag
mismatches plus a seed-sensitivity control. With `--prg gpu-aes`, 88
two-process key pairs over four configurations (`L=4/8/11`, both primes) pass
batched-SPFSS and per-tree full-domain GPU reconstruction with corrupted-CW
controls firing (`scripts/run_two_party_gpu_dpf.sh`,
`results/dpf/two_party_gpu_dpf_2026_07_29.csv`). This is GPU key compatibility
and GPU-validated correctness, not GPU-side key generation. The 127-bit
encoding defect is removed. The exact coupling now reduces joint-key
distribution/privacy to the standard DPF/PRG theorem, but no concrete
single-key or 128-bit security claim is attached.

**M2 core gate reached (2026-07-29, new): the real Ring-LPN OLE engine runs on
two-party dealerless SPFSS keys.** `build_spfss_keys()` - the pipeline's
centralized-keygen oracle - is now replaceable by two OS processes over real
IKNP OT: `src/test_two_party_spfss_keygen.cpp` (shared protocol in
`src/two_party_dpf_protocol.h`, GPU expansion PRG) plus two env-gated hooks in
`src/bench_ole_ringlpn_cuda.cu` (`RINGLPN_OLE_EXPORT_NOISE`,
`RINGLPN_OLE_SPFSS_KEYS`; with neither set the bench is byte-for-byte its old
self). The engine's own `validation`/`host_validation`/`correct` all pass on
dealerless keys in **all four deployed configurations** (q64/q128 x
uniform/regular), expansion and validation code unmodified:
256 trees/limb, one level-synchronous batch, 89 stages at `L=14` (uniform) and 71
at `L=11` (regular), ~1.0 MB per party per limb, keygen 0.90 s uniform / 0.13 s
regular after a 6.8x host-PRG speedup. Regular noise is ~7x cheaper to key for
the same tree count (domain `2^11` vs `2^14`) - the same effect that collapsed the
DMPF encoder advantage. Wired into the required-GPU gate (`ALL GATES PASS`).
This component checkpoint has since been superseded by the live
`test_two_party_fc_preprocess` composition, which performs party-local
expansion and consumes the exact conversion API. IKNP remains the principal
host/transport bottleneck and is not silent OT.

**Two-process conversion transport validated (2026-08-03).**
`src/test_secure_convert.cpp` and `scripts/run_secure_convert_test.sh` replace
the standalone prototype's labelled edaBit/daBit/AND-triple dealer with the
live two-socket SCI/IKNP path. One exact daBit over `Z_(2^k)` uses one 128-bit
OT; `ell` daBits over `Z_(2^ell)` compose an exact edaBit because the
coefficient-one arithmetic share is uniform independently of the Boolean
shares. Boolean triples use two one-bit OTs each. Independent processes write
only party-local versioned records; the offline checker covers exact
`0,Q-1,Q,2Q-2` boundaries, random inputs, forced wraps, and layer-shaped
inner products, and requires a corruption control to fire. All 76 conversions
in each of four q64/q128/bit-width configurations pass. Per-party rows
separate base-OT setup, correlation, and online bytes/direction switches and
gate `5ell-3` logical / `10ell-6` meaningful-share / `2ell-1` post-mask
accounting. An invalid public-bound control rejects before opening a socket.
Plain unauthenticated TCP, IKNP rather than silent OT, linear-depth ripple, and
CPU execution remain explicit limits. The live forward-FC path now consumes
this API, and the security contract contains its exact hybrid simulator.

## Catch up in 10 minutes (read in this order)

1. This file.
2. `results/reports/publication_readiness_plan_2026_07_21.md` — binding
   S1--S10 execution order, gates, proof/evaluation requirements, and
   per-stage commit discipline.
3. `results/reports/dealerless_orca_fc_security_contract_2026_07_29.md` —
   S1 functionality, exact DPF/FC transcript, leakage, simulators, and proof
   obligations.
4. `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` and
   `s2_professor_decision_request_2026_07_29.md` — S2 hard stops, current prior
   art, attribution, parameter transcript, and eight advisor decisions.
5. `results/reports/distributed_dpf_keygen_memo_2026_07_21.md` — corrected
   Phase C protocol, executable controls, and regenerated D1 counts.
6. `results/README.md` — where every result/report lives and what produces it.
7. `results/reports/s2_architecture_comparison_2026_07_29.md` — measured
   encoder comparison, dealerless-setup status, artifact defects, and decision
   table.
8. `results/reports/orca_fc_real_ole_transcript_memo.md` — real-OLE
   slot-packed transcript and NTT backend changes.
9. `results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` —
   current v2.6 source; its canonical freshness functionality postdates the
   existing v2.5 PDF, whose rebuild/page inspection is pending.
10. `results/reports/baseline_2026_06_10.md` — historical full-GPU
    environment, PASS counts, and performance anchors.

Then re-validate everything with one command (~15 min, needs GPU):

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  scripts/run_paper_checkpoint_smoke.sh
# must exit 0 and print "[paper-smoke] ALL GATES PASS"
```

## Source map (`src/`)

| File | What it is |
|---|---|
| `bench_ntt_cuda_cheddar.cu` | The GPU NTT backend (substantially Cheddar-derived merged-stage kernels, signed Montgomery, q32/q64/q128-CRT, negacyclic). Upstream MIT notice and reconstructed source/blob pin plus local delta are retained in `extern/Cheddar_MIT_LICENSE.txt` and `extern/Cheddar_PROVENANCE.txt`; cite Cheddar. Included by every GPU bench via `RINGLPN_DISABLE_MAIN`. Contains `run_full_polymul`, `run_polymul_prepared_lhs`, adaptive fused-INTT (`RINGLPN_NTT_NO_FUSE`/`FORCE_FUSE`), `host_polymul_reference` (the host oracle), `kConfig62`/`kConfig62Crt2` (the primes: 2^62−6·2^24+1, 2^62−7·2^24+1). |
| `correlation_freshness.h` | **Live consume-once boundary (2026-08-04).** Fixed-width version/invocation/layer/kind/direction/limb/ring-batch/tree/phase/ordinal/conversion-chunk/output-slot encoding, SHA-256 IDs, compatibility-handle collision rejection, and owner-only immutable claim files written with no-replace creation, fsync, atomic rename, and directory fsync. Duplicate, pending, truncated, corrupt, retry, and restart state fails closed. |
| `gpu_spfss_zp.cuh` | GPU DPF/SPFSS with additive `Z_p` payloads (`gpuKeyGenDPFZpPair`, `gpuDpfZpFullEvalSum`). Expansion uses four domain-separated AES calls: full 128-bit child seeds from plaintexts 0/2 and separate tags from 1/3, with device/host parity gated. Centralized diagnostics still derive roots from one 64-bit `seed_base` and are not security evidence; the live path uses OpenSSL-private roots. Exact joint-key coupling is in the security contract; concrete DPF/PRG and Ring-LPN parameter review remain open. |
| `ringlpn_ole_party.cuh` | **Party-local Figure-2 API (2026-08-04).** Own-party public parameters, noise/key validation and packing, GPU `x`/SPFSS/`z` expansion, and own `X`/`Z` slot shares. The live forward-FC runner calls it separately in each process; `bench_ole_ringlpn_cuda.cu` remains a both-party diagnostic. Ring-LPN pseudorandomness/parameters remain outside this functional API. |
| `bench_ole_ringlpn_party.cu` | Standalone one-party executable over one noise/key record, retained for focused diagnostics. The canonical live FC path calls the same party API in-process rather than exchanging intermediate slot files. |
| `bench_ole_ringlpn_cuda.cu` | Existing two-party-in-one-process Figure 2 Ring-LPN OLE engine/checker (random ring OLE: z0+z1 = x0·x1 in Z_p[X]/(X^n+1)); now consumes `ringlpn_ole_party.cuh`. It still owns both party states and retains `build_spfss_keys()` as its centralized-keygen fallback, while `RINGLPN_OLE_{NOISE,SPFSS_KEYS}` load the separately generated party records. |
| `bench_linear_ole_ringlpn_cuda.cu` | Ring-polynomial matrix Beaver from two OLEs per ring product. |
| `bench_vole_ringlpn.cu` | Older standalone VOLE expansion prototype. |
| `orca_fc_ringlpn_keywriter.cuh` | Host helpers + dealer/oracle keywriter used by `nn/orca/fc_layer.cu` behind `ORCA_RINGLPN_FC_KEYS` (bw≤32; baseline Orca byte-identical with flag off). Has `exactZmToRingShares` (conversion oracle), CRT/q128 helpers, and a clear value-dependent `dot >= Q` abort; the target replaces that predicate with the public admissibility check `K*2^(2*bw+2)<Q`. |
| `orca_fc_ideal_ole_transcript.cuh` + `bench_orca_fc_ideal_ole_transcript.cu` | Step-1 artifact: dealerless FC transcript with an *ideal* OLE oracle. Kept as reference; superseded by the real-OLE transcript. |
| `bench_orca_fc_real_ole_transcript.cu` | Historical single-process real-generator diagnostic: Ring-LPN expansion, slot packing, per-slot derandomization, Garner lift, clear exact conversion, key write, and unchanged consumer. It retains centralized DPF keygen and O1/O2 boundaries; do not compare its narrow stage timers to live end-to-end results. |
| `bench_orca_fc_ringlpn_demo.cu` | Byte-compatibility demo: forward + dW + dX key contracts at q64/q128. |
| `secure_convert.{h,cpp}` | **Party-local exact conversion API (2026-08-04).** `secure_convert_batch` validates canonical shares and common preflight, generates OT-backed daBits/edaBits and Boolean triples, and reports split transcript counters. The live forward-FC path calls it; the security contract supplies the exact hybrid simulator. |
| `test_secure_convert.cpp` | Standalone two-process conversion harness/checker for exact boundaries, random/forced-wrap and invalid/corrupted cases, bounded bilateral best-effort records, and split counters. Plain unauthenticated TCP, SCI/IKNP, linear-depth ripple, and CPU execution remain limits. |
| `test_distributed_dpf_keygen.cpp` | **Corrected M1 host protocol-logic prototype (2026-07-29).** Two-party DPF keygen: secure adder for α's bits (L−1 bit triples), cancellation-lemma level walk (2 string OTs/level), and Phase C arithmetic-share multiplication (3 scalar OLEs) that opens only standard `finalCW`. Six invalid-input controls, five independent key corruptions (root seed, `sCW`, `tLCW`, `tRCW`, `finalCW`), omniscient old-sign regression, per-phase transcript accounting, ideal-functionality mask-draw accounting, and consume-once correlation-ID reuse control. Emits standard `spfss_host::DPFKey`s validated by unchanged `dpfEvalAll`; ideal OT/triple/OLE interfaces and non-cryptographic splitmix64 correctness PRG mean functional compatibility, not computational privacy. Party private-random-tape freshness is not executable evidence. `spfss_host.cpp` remains untouched. |
| `two_party_ot.h` | **Real two-party transport.** SCI IKNP/Gilboa plus opt-in reviewed-backend hooks; protocol randomness uses buffered `RAND_priv_bytes`, while `mt19937_64` is confined to explicitly seeded standalone correctness inputs. `PartyChannel` and `PartyRandom` are noncopyable/nonmovable and expose no reset, preventing in-process state rollback. |
| `dpf_key_io.h` | Versioned little-endian `spfss_host::DPFKey` batch serialization (magic `RLPNDPF1`) plus the explicitly TEST-ONLY private-input record the offline checker needs. |
| `test_two_party_dpf_keygen.cpp` | **The two-PROCESS keygen artifact.** Same frozen protocol, but two OS processes over two TCP sockets with real OT/triples/OLE, each party writing only its own key file; gates the contract's closed forms in-process and reports measured wire bytes, direction switches, and setup cost. Primitive self-tests (`--selftest`) open triple/OLE shares in a labelled test-only mode. |
| `two_party_spfss.h` | **Party-local SPFSS API.** Validates/samples local noise, derives public work/group order, binds the full live Ring-OLE correlation-scope ID plus compatibility SID into the v3 public manifest, generates grouped DPF keys, and computes provenance digests. Standalone component baselines may retain an explicitly test-only zero scope; live FC/Conv may not. |
| `test_two_party_spfss_keygen.cpp` | Two-process CLI harness for `two_party_spfss.h`; each process samples or explicitly labels TEST-ONLY external noise, exchanges the manifest/validity state, emits per-party provenance/cost rows, and uses temp writes, a bilateral publishability exchange, and rename to publish only its own versioned noise/key records. This is bilateral best-effort, not crash-transactional. The focused q64 regular evidence is indexed above; the later OLE checker still reads both outputs. |
| `test_two_party_fc_preprocess.cu` | **Canonical live forward-FC/Conv executable.** Claims a high-entropy invocation and exact correlation plan before OT/CSPRNG state, binds full scope IDs into SPFSS and conversion, samples the exact identity-`a0` public vector, expands/derandomizes/converts, and publishes version-2 records naming invocation and ledger digest. FC/Conv runners generate fresh IDs and include duplicate/restart/collision/truncation/tail controls. Publication remains bilateral best-effort; the authenticated coordinator's durable manifest is the consumer gate. |
| `test_two_party_dpf_validate.cpp` | TEST-ONLY offline checker: runs after both parties exit, reads both key files, validates `beta*[x=alpha]` through unchanged `dpfEvalAll`, requires identical public material and differing seeds, and includes a corrupted-`finalCW` negative control. |
| `test_orca_zp_bridge.cpp` | Carry-corrected Z_p→Z_2^bw share export + the bw=32/q62 counterexample (negative control). |
| `bench_ntt_gpu_ntt_baseline.cu` | External baseline: GPU-NTT (Ozcan–Savas) vs cheddar, same prime/psi/operation. Needs external checkout (`GPU_NTT_HOME`, default `/home/fatih/GPU-NTT`); benchmark-only, not in the gate. |
| `spfss_host.{h,cpp}`, `test_spfss.cpp`, `bench_ole_ringlpn_host.cpp`, `verify_figure2_expand.cpp` | Host reference implementations + the 135/57/36 host validation suites. |
| `test_spfss_zp_cuda.cu` | GPU SPFSS payload correctness tests. |
| `orca_globals_stub.cpp` | Defines `OneGB` for standalone benches (instead of linking the comms stack). |

One upstream touch outside this dir: `GPU-MPC/nn/orca/fc_layer.cu` — the
feature-flagged keygen integration (flag off = byte-identical baseline,
verified through the two-party `tests/nn/orca/fc` test).

## Scripts (`scripts/`)

Every artifact has a `build_*.sh` / `run_*.sh` pair; runners write
`.csv` (data) + usually `.md` (summary) + `.log` (raw stdout+stderr) into
their directory under `results/` (see `results/README.md` for the mapping).
`run_paper_checkpoint_smoke.sh` is the canonical gate: host trio + bridge +
secure convert + GPU smokes (OLE q64/q128 × uniform/regular, linear, demo,
both transcripts), ends with `ALL GATES PASS`.

## Validated claims vs. open boundaries

Safe to state (scoped to observed evidence):

- The corrected source is dealer/oracle-free, runs as two party processes on
  distinct GPUs, and produces party-local stock-format keys. Six current
  q64/q128 regular/uniform/multi-batch executions pass the unchanged
  `gpuMatmulBeaver` contract; ten focused freshness/record controls reject the
  intended duplicate, restart, reuse, collision, truncation, mismatch, stale,
  rename, corrupt, and swapped cases.
- The current exact ResNet18-classifier-layer run at `1x512x1000`, q128/bw32,
  feasibility `(n,c,t)=(8192,2,8)` passes 10/10 measured trials after one
  warmup. Median preprocessing is 25.715 s, application traffic is
  575,846,872 bytes, matched stock dealer keygen is 10.642 ms, and the
  unchanged online checker is 1.106 ms.
- Slot packing gives `2*limbs*ceil(MKN/n)` Ring-OLE instances per party and up
  to `n` scalar slots per instance. The live classifier layer uses 252
  instances and
  64,512 DPF trees per party.
- The standalone real DPF transport validates 369/369 host-reference pairs;
  the four-call full-width AES path matches 16 device vectors and 88
  two-process keys pass GPU evaluation. SCI/IKNP/Gilboa, private OpenSSL roots,
  bytes, direction switches, invalid-input and corruption controls are
  executable evidence.
- The ideal-functionality host reference validates 2,432 DPF pairs and exact
  transcript counts: `2*depth` string OTs, `depth-1` bit triples, three scalar
  OLEs, `2*(depth-1)+130*depth+ceil(log2(p))` logical opened bits, and twice
  that encoded share width.
- The exact DPF correction-word coupling, both correlated-batch simulators,
  conversion simulator, and source map yield a conditional static-semi-honest
  theorem for forward FC under the assumptions stated in the security
  contract. This is a proof boundary, not a concrete parameter/security claim.
- The canonical correlation tuple and append-only claim implementation close
  the executable source/proof `P-FRESH` boundary at the explicit SHA-256
  collision-resistance, trusted owner-only persistent filesystem, one
  deployment-wide ledger root, no-replace/fsync/atomic-rename semantics, and
  no storage cloning/rollback assumptions. All ten focused controls pass.
- Baseline Orca is byte-identical with the feature flag off.

NOT claimable (never blur these):

- Any concrete security level. The exact projected Ring-LPN
  distribution/structured-code/two-limb reduction is unreviewed and no
  parameter set is pinned.
- A secure network deployment. The theorem assumes authenticated channels;
  the live experiment uses unauthenticated local loopback.
- A performance win. The matched median is about 2,414 times slower than stock
  dealer keygen and sends about 576 MB of application traffic for one
  classifier layer.
- Full-model dealer removal, stateful training, nonlinear keys, malicious
  security, WAN behavior, or side-channel resistance.
- Conference readiness. Parameter security, Phase-C/silent-transport
  performance, all-linear-layer coverage, a compatible dealerless baseline,
  two-host evidence, clean-clone reproduction, and independent human
  cryptographic review remain open.

## Prioritized next-agent runbook

Do these in order; a component PASS never permits skipping a claim gate.

1. **Close the exact-parameter proof gap first.** The exact regular-sampler
   projection/cancellation law is now proved and machine checked. Next obtain
   an independently reviewed structured projected-code reduction, justify the
   direct/structured attack model and orbit treatment, and compose advantage
   across both CRT limbs and every PCG hybrid. Only then run
   `scripts/audit_ringlpn_finite_field_models.py --estimator
   <pinned-lpn-estimator.py> --reference-bits 128 ...`, review every tuple, and
   measure a resulting candidate. The script deliberately exits nonzero and
   cannot itself pin a parameter.
2. **Execute the theorem-aligned deployment boundary.** Run and audit the
   authenticated two-host launcher under genuinely peer-private identities.
   Preserve the durable coordinator commit gate and bilateral crash cleanup;
   do not relabel the executable's best-effort rename/ack as transactional.
   Repeat the source/transcript/proof audit after protocol changes.
3. **Optimize the measured bottleneck.** Replace or restructure the
   tree-per-point Phase-C payload correction using a source-reviewed
   DMPF/P-DPF design; measure the existing EMP-Silent backend and preserve
   full-width seed/tag semantics and stock-key compatibility. Continue
   reporting stage time, setup/steady-state bytes, dependency stages, GPU/CPU
   overlap, and peak host/GPU memory.
4. **Re-gate at the secure pinned set.** Once the parameter and optimized
   protocol paths land, rerun every host/GPU/live negative gate with fresh
   invocation IDs and unused ports. Regenerate all CSVs, manifests, digests,
   theorem assumptions, cost formulas, and paper tables; current feasibility
   rows remain separately labelled.
5. **Broaden evaluation.** Execute every forward linear layer and its
   truncation/state handoff for at least one real inference model at the pinned
   set, with the exact matched stock dealer and closest compatible
   dealerless-PCG baseline. Keep one warmup plus at least ten raw model trials,
   add authenticated two-host LAN and controlled WAN environments, and report
   confidence intervals, memory, dependency rounds, throughput, and unchanged
   online time.
6. **Handoff discipline.** Keep Alp `<fcetin@hawk.iit.edu>` as the sole paper
   and final-history author; re-audit every intended commit for author identity
   and co-author/generated-by trailers. Do not send the Reverse Cuckoo outreach
   draft, claim push credentials, or imply ownership/credit permission is
   resolved; those decisions remain with the owner/professor.

## Binding execution order (S1–S10 plan; proposal components D1–D5)

S1 froze the functionality/proof contract. S2 remains the hard
security/publication gate for exact Ring-LPN parameters, novelty, and
provenance. The owner explicitly lifted only the implementation-order
dependency, which allowed the feasibility D1–D4 forward path, conditional
proof, and exact classifier-layer evaluation to proceed without a
parameter/security
claim. Those branches are now executable and measured; they do not close S2.

The next order is: reviewed structured-code/advantage result -> any required
distribution/prime/backend changes -> Phase-C/DMPF and silent-transport
optimization -> renewed source/proof audit -> all-forward-linear-layer matched
evaluation -> authenticated two-host evaluation -> clean-clone submission
candidate. GPU-batched DPF generation, dependency-stage/memory instrumentation,
focused EMP-Silent correctness, exact regular-projection analysis, and one
source-pinned mismatched closest-baseline run are already complete; none closes
the remaining strict gates. The full gates and atomic checkpoint discipline
remain in `results/reports/publication_readiness_plan_2026_07_21.md`.

Current D1 functionality uses real two-process SCI/IKNP/Gilboa transport,
OpenSSL-private roots, GPU-batched full-width GPU-AES-compatible keys, and
measured bytes/dependency stages/memory. Remaining performance work is Phase C
and measured/reviewed silent transport. The exact coupling and hybrid
simulators close the conditional algebraic obligations; structured-code
security, concrete parameters, and independent human review remain open.
Components are D1–D5 in the v2.6 report.

## Perf anchors (RTX 5000 Ada, this repo's gate configs)

| Metric | Value |
|---|---|
| OLE expand, n=8192 c=2 t=8 smoke | 13.3 ms (q64) / 26.8 ms (q128) |
| OLE expand, t=64 | 881 ms uniform / 61 ms regular (q64) |
| Linear OLE-to-Beaver 2×2×2 | 224 ms (q64) / 448 ms (q128) |
| Cheddar polymul n=8192 batch=64 | ~255–265 µs (q64) |
| Current ResNet18 classifier-layer forward-FC preprocess | 25.715 s median (10 trials; q128/bw32 feasibility point) |
| Current classifier-layer application / total transport bytes | 575,846,872 / 575,890,852 |
| Matched stock dealer / unchanged online | 10.642 ms / 1.106 ms median |
| Current live/dealer preprocessing ratio | 2,414× median of trial ratios |
| Current dominant stage | Phase C: 20.432 s median (79.4%); GPU Ring-LPN expansion: 1.704 s |
| Current protocol/memory counters | 10,338 dependency stages; 142,493,696 host / 27,241,086,976 GPU peak bytes |

NTT decision (measured, `reports/ntt_baseline_comparison_2026_06_10.md`):
keep cheddar — external GPU-NTT merge is 1.2–3.9× faster but cannot run
62-bit primes (Barrett headroom). Revisit at M5 if primes are re-pinned to
the ≤60-bit class.

## Environment & gotchas (will bite you)

- **This is a shared school server; the user is NOT a sudoer.** Never attempt
  privileged operations. The user is in the docker group (root-equivalent in
  principle) — use it only for ephemeral containers and for chown-ing the
  user's OWN files under `/home/fatih`; never touch system paths or other
  users' files. Check `nvidia-smi` before heavy GPU runs and prefer pinning
  with `CUDA_VISIBLE_DEVICES` — others may be working.
- `nvcc` is at `/usr/local/cuda/bin` — NOT in default PATH. `GPU_ARCH=89`
  (4× RTX 5000 Ada). Two-party tests: run party 0 and 1 with
  `CUDA_VISIBLE_DEVICES=0/1`, args `<party> 127.0.0.1`.
- Historical builds ran as root in docker (`pcg-accel:*` images; user is in
  the docker group, sudo needs a password). Root-owned files can reappear;
  fix: `docker run --rm -v <dir>:/x ubuntu:22.04 chown -R 1013:1014 /x/...`.
- `bench_ntt` (CPU NFLlib) needs libmpfr-dev — absent on host and in the
  images; build in a container with an ephemeral `apt-get install libmpfr-dev`.
- **No LaTeX anywhere** (host, orca-dev, all local images). Compile reports
  with an ephemeral container: `docker run -d --name texbuild -v <reports
  dir>:/work debian:bookworm sleep infinity`, then `apt-get install -y
  --no-install-recommends texlive-latex-base texlive-latex-recommended
  texlive-pictures` (~300 MB; covers tikz/booktabs/hyperref; avoid enumitem —
  it needs texlive-latex-extra), `pdflatex` twice in `/work`, then delete the
  root-owned .aux/.log/.out via a root container and chown the PDF to
  1013:1014. `/bin/sh` in these containers is dash: no `{a,b}` brace expansion.
- Root `.gitignore` ignores `*.csv` globally — committing new result CSVs
  requires `git add -f`.
- `extern/NFLlib` is a registered submodule (quarkslab/NFLlib @ 5cf40ed);
  the mnist/weights data submodules carry deliberate internal renames — leave.
- Env flags: `ORCA_RINGLPN_FC_KEYS` (+`_QBITS`,`_SEED`) for the fc_layer path;
  `RINGLPN_NTT_NO_FUSE` / `RINGLPN_NTT_FORCE_FUSE` for polymul A/B;
  `SMOKE=1 QBITS=... NOISE=...` for the sweep runners.

## House rules for new work

1. Every new artifact = source + build script + run script + CSV/MD/log in
   its `results/` subdir + a memo in `results/reports/` + a gate hook if it
   guards a claim. Suites exit non-zero on any failure.
2. Validate against an independent oracle (host reference or unchanged Orca
   online path), not against the code under test.
3. State oracle boundaries in the source header and the memo. Keep the
   "safe to claim / not yet claimable" split current.
4. Don't claim perf wins without an A/B at the consumer's actual shape
   (cf. the fused-INTT adaptive threshold story).
5. **Commit every stage:** a stage is complete only after its mechanical gate
   passes, evidence and current docs are synchronized, and an atomic checkpoint
   commit is created. Preserve completed gate commits; corrections get new
   commits and rerun affected gates. See the publication-readiness plan.

## Documentation contract — BINDING for every agent working here

This file is the single source of truth for project state. Stale documentation
is worse than no documentation: it corrupts the context of whoever reads it
next. Therefore, **before ending any session that changed code, results, or
plans, you must**:

1. **Update this file** — the Status line, the source map (if files were
   added/moved), the claims split (if a boundary moved), the perf anchors
   (if regenerated), and the gotchas (if you hit a new one).
2. **Update `results/README.md`** with a row for any new artifact or report.
3. **Write or refresh the memo** in `results/reports/` for the artifact you
   touched, dated, with reproduction commands.
4. **Mark superseded documents**, never delete-without-trace and never leave
   them unmarked: prepend the standard banner
   (`> **HISTORICAL ...** superseded by CLAUDE.md; statements below may
   describe an older state`) the moment a document's claims stop being
   current. All pre-2026-06-10 reports already carry it; keep the convention.
5. **Never let two live documents disagree.** If you find a contradiction,
   the newer gate-verified statement wins; banner or fix the other on the
   spot, in the same commit.
6. Historical/outreach/archive documents are read-only context: quote them,
   don't trust them. Anything without a banner and dated 2026-06-10 or later,
   plus the gate output, is current; everything else is history.
