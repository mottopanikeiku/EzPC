# ringlpn — agent & human catch-up guide

**What this is:** a research subproject building *dealerless* preprocessing for
Orca (the GPU FSS-based secure ML system in this repo) from Ring-LPN
pseudorandom correlation generators (PCGs). Orca's linear layers consume
Beaver-triple keys that a trusted dealer normally produces; this project
replaces the dealer with a two-party protocol: GPU NTT/polynomial arithmetic →
Z_p SPFSS (sum of DPFs) → Figure 2 Ring-LPN OLE → slot-packed Beaver cross
terms → Z_M→Z_2^bw conversion → byte-compatible Orca keys, validated through
Orca's **unchanged** online path (`gpuMatmulBeaver`).

**Status (2026-08-03):** the GPU chain works end-to-end with real Figure 2
OLEs at q64/q128 and dense slot packing. The corrected M1 protocol logic
(`src/test_distributed_dpf_keygen.cpp`) is party-separated and functionally
validated by unchanged `spfss_host::dpfEvalAll` on 2,432 trees across both
primes and depths 4–14. Phase C uses three scalar OLEs and opens only standard
`finalCW`; five independent key corruptions, six invalid encodings, mask-draw
accounting, correlation-ID reuse, and the removed-sign leakage regression are
gated.

The same keygen protocol runs as two OS processes over TCP with SCI/IKNP,
Gilboa OLE, OT-backed Boolean triples, and OpenSSL-private-DRBG roots:
369/369 host-reference key pairs pass across ten configurations. A corrected
four-call AES expansion retains full 128-bit child seeds and derives control
bits separately; 16 fresh device vectors match the host twin and 88
two-process keys pass batched/per-tree GPU evaluation across both primes.
These are correctness, compatibility, and transport results—not a DPF privacy
proof.

The standalone share-conversion artifact is also two-process and OT-backed:
304/304 outputs pass exact boundary, random, forced-wrap, and layer-shaped
checks across four q64/q128/bit-width cases, with a corrupted-output control
and split setup/correlation/online traffic. It remains CPU-side, ripple-depth,
IKNP rather than silent OT, plain TCP, and not integrated into the flagship FC
transcript. The flagship still contains centralized in-process setup and exact
conversion oracle boundaries.

The M1–M6 proposal is **v2.4** (stable filename:
`results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` + PDF).
No parameter set or end-to-end security theorem is pinned; S2's reduction and
estimator audit remains blocking. See the current reports indexed in
`results/README.md`.
The binding staged route to publication readiness is
`results/reports/publication_readiness_plan_2026_07_21.md`; every completed
stage must end in a gate-verified checkpoint commit.
Direction was locked with the user on 2026-07-29: the paper's thesis is the
integrated dealerless Orca FC-preprocessing system. The S2 audit subsequently
found that the corrected per-point distributed DPF is **not presently a
defensible protocol contribution**: BCG+20 already uses distributed DPF setup,
Programmable DPF gives constant-round generation, and 2026 fully distributed
DMPF/SLAMP-FSS work advances the multi-point PCG/FSS bottleneck. Treat the host
DPF as a compatibility artifact/baseline unless advisor review identifies a
concrete delta. The paper's sole author is Alp by user direction; commits use
only the configured user without co-author/generated-by trailers. The separate
private GPU PCG/PIM work has multiple contributors, no repository license, and
unresolved ownership, credit, chronology, reuse, and overlap decisions. In
particular, Chenkai's private commit `e821141` (leaky output replaced by a
hash/Beaver-corrected CW) predates this fork's `28f8451` three-OLE correction;
the implementations differ, but independence/credit cannot be inferred. Do not
import the private work or claim its design/performance/Phase-C idea.
The first deliverable is an advisor-ready
technical report; work remains at publication-grade proof/transport/evaluation
standards. Stay ringlpn-first, present any minimal upstream integration or
external crypto dependency before adoption, and consult the user before every
S1--S10 stage.
The S1 contract is frozen **for advisor review** at
`results/reports/dealerless_orca_fc_security_contract_2026_07_29.md` after
the user-requested Opus 5 model-assisted audit reported no remaining freeze
blocker. This is not an independent human cryptographic review, security
proof, computational-security result, or publication-readiness claim.
S2 is **blocked pending professor approval**, but the project-owner
consultation is recorded. The chosen direction is an integrated dealerless
Orca FC systems contribution, with matched public-source comparisons of
fully distributed DMPF versus SLAMP-FSS and of regular Ring-LPN/NTT versus
Stationary-SD, direct-`Z_(2^k)`, and QA-SD/WHT before architecture freeze.
Use `n=2^14,c=4,t=16` only as a feasibility tier and `n=2^20,c=4,t=16`
only as a literature-reference tier. The preliminary audit at
`results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md`
records the projection-rule/Table-1 contradiction, unproved projected-noise
mapping, incomplete epoch budget, and current prior art. The private PCG/PIM
repository is the same research stream but remains quarantined: its
multi-contributor/no-license status and output-layer chronology still require
the professor's written authorship/credit/overlap disposition. Cheddar remains
an attributed MIT-licensed dependency under
`extern/Cheddar_{PROVENANCE,MIT_LICENSE}.txt`. No parameter set is pinned and
S3 must not start. The consultation record and remaining professor questions
are in `results/reports/s2_professor_decision_request_2026_07_29.md`.
The comparison the owner asked for is now **measured**, not just specified:
`results/reports/s2_architecture_comparison_2026_07_29.md`. Its headline result
is a negative one, and it must not be misquoted. With *uniform* noise an
OKVS-style DMPF expands the sparse product 275x faster than the current sum of
point DPFs at `(n,c,t)=(2^14,4,16)`. With the **regular** noise the artifact
actually deploys (`spfss_domain=2048`, `log_domain=11`, 31 diagonal groups per
pair) the same encoder is **0.79x - slower** - and the best DMPF (big-state)
wins only **2.29x** at 37x the key bytes. Regular noise already buys the
domain/point reduction a DMPF is designed to buy, so the encoder is **not** the
dominant lever here, and the 275x figure must never be presented as this
project's result. Every dealerless candidate remains unavailable, unlicensed, or
unimplemented. Reverse Cuckoo's printed Figure 7 is internally inconsistent
(dummy padding versus full row rank) and has no public source. Two defects in
the published native-ring PCG artifact (undefined `1<<(k+s)` for the 121-bit
modulus; shipped insecure `c=3,t=27` grid) mean any local row from it is
`adapted`, never `reproduced`.

**2026-07-29 owner decisions (gate lift).** The self-imposed S2->S3 ordering
gate is lifted **for implementation only**: the GPU distributed-keygen core and
real OT/OLE transports may proceed in parallel with the unresolved parameter and
security questions, provided no security claim is attached. Where BCG+20's
literal projection rule and its Table 1 disagree, adopt the **conservative
minimum**: `scripts/pin_ringlpn_parameters.py` reports the cheapest attack over
every projection the accepted EUROCRYPT 2024 estimator admits and over both
noise models, for both deployed primes. Under that reading `c=4,t=16` is
**57.29 bits, not 128**; the conservative cost depends on `c*t`, so the pinned
set must raise the noise weight and the artifact's current parameters must be
described as feasibility-only.

**2026-07-29 owner route decisions (recorded after the measured comparison).**
Presented `results/reports/s2_architecture_comparison_2026_07_29.md` §6-§7 to the
owner; the four answers are binding:
1. **Encoder:** keep the per-point DPF. Implementation effort goes to *real*
   silent-OT/OLE transports and a two-process deployment. Rationale is measured,
   not aesthetic: at the deployed regular layout the best DMPF wins only
   2.29x expansion at 37x the key bytes, and OKVS is 0.79x (slower), so the
   encoder is not the pipeline's lever. A dealerless DMPF stays future work.
2. **Parameters:** pin the conservative minimum *now* - raise the noise weight
   until the cheapest admissible projection clears 128 bits for both primes -
   then re-run every feasibility/GPU gate at the pinned set. The current
   `(n,c,t)` rows are feasibility-only until that lands.
3. **Claim scope:** the paper's headline claim waits for a real two-process
   dealerless FC run. Protocol-logic slices are supporting evidence only, never
   the contribution.
4. **Prior art:** draft (do not send) an artifact/clarification request to the
   Reverse Cuckoo authors for owner approval:
   `results/outreach/reverse_cuckoo_artifact_request_2026_07_29.md`.

**Conservative pin transcript (measured 2026-07-29).**
`results/security/ringlpn_conservative_pin_2026_07_29.{csv,log}` holds 72 rows
from the pinned EUROCRYPT 2024 estimator. Two facts drive the pinned set:
the conservative cost was a function of the total weight `w = c*t` across every
ring size the coarse sweep completed (`n=2^13..2^16`, 72 rows), and the
point-DPF bootstrap needs
`3*c^2*t^2 < n`. Measured: `w=128` gives 108.66-111.24 bits (fails), `w=256`
gives 190.53 bits at `c=2,t=128` and 218.64 at `c=4,t=64` (passes). The finer
sweep (`results/security/ringlpn_conservative_pin_refine_2026_07_29.csv`) shows
the curve is **not monotone in `w`**: the rule admits a projection only while
the expected reduced weight still fits the reduced dimension
(`w_i <= (c-1)*2^i`), so raising `t` from 32 to 34 evicts the cheap degree-32
projection and the conservative minimum jumps from 111.24 to 257.02 bits
(`c=4,t=34`, degree 64, exact model, at `n=2^18`). Any pin must therefore quote
the *evicted* projection alongside the surviving minimum; `c=4,t=34` needs
`3c^2t^2 = 55,488` setup slots, so it bootstraps from `n>=2^16` (15.3% net) and
has 78.8% net capacity at `n=2^18`. Bits at `n=2^16/2^17` for that candidate are
being measured into
`results/security/ringlpn_conservative_pin_n16_n17_2026_07_29.csv`.

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

**2026-07-29 owner pin + slice decision.** Conservative candidate pinned for
re-measurement: **`n=2^17`, `c=4`, `t=34`** (257.02 conservative bits, 55,488
point-DPF setup slots, 57.7% net epoch capacity). Every feasibility/GPU gate
must be re-measured at that set before any table quotes it, and the pin write-up
MUST state the eviction that produces it: at `c=4,t=32` the cheapest admissible
projection is degree 32 with **111.24** bits, and raising `t` to 34 evicts that
projection (expected reduced weight no longer fits `(c-1)*2^i`), leaving degree
64 at 257.02 bits. Next implementation slice, also owner-selected: GPU-side
batched keygen with byte-identical GPU keys (M1 remainder), then drive the
real-OLE GPU transcript from two-party keys (M2).

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
and GPU-validated correctness, not a GPU implementation of keygen. The
127-bit encoding defect is removed; D-DIST/P-RNG/P-KEY and the concrete
single-key privacy reduction remain open, so no end-to-end 128-bit
DPF-security claim is attached.

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
DMPF encoder advantage. Wired into the required-GPU gate (`ALL GATES PASS`). Still
oracles: benchmark-side noise sampling, dealer-labelled conversion correlations,
single-process expansion measurement, IKNP instead of silent OT. See
`results/reports/dealerless_ole_two_party_keys_memo_2026_07_29.md`.

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
Plain unauthenticated TCP, IKNP rather than silent OT, linear-depth ripple,
CPU-only execution, and missing flagship integration remain explicit limits.

## Catch up in 10 minutes (read in this order)

1. This file.
2. `results/reports/session_handoff_2026_07_29_dmpf_comparison.md` — current
   verification, consultation decisions, corrected point-count notation, and
   next actions.
3. `results/reports/publication_readiness_plan_2026_07_21.md` — binding
   S1--S10 execution order, gates, proof/evaluation requirements, and
   per-stage commit discipline.
4. `results/reports/dealerless_orca_fc_security_contract_2026_07_29.md` —
   S1 functionality, exact DPF/FC transcript, leakage, simulators, and proof
   obligations.
5. `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` and
   `s2_professor_decision_request_2026_07_29.md` — S2 hard stops, current prior
   art, attribution, parameter transcript, and eight advisor decisions.
6. `results/reports/distributed_dpf_keygen_memo_2026_07_21.md` — corrected
   Phase C protocol, executable controls, and regenerated D1 counts.
7. `results/README.md` — where every result/report lives and what produces it.
8. `results/reports/s2_architecture_comparison_2026_07_29.md` — measured
   encoder comparison, dealerless-setup status, artifact defects, decision
   table, and the open owner questions.
9. `results/reports/orca_fc_real_ole_transcript_memo.md` — real-OLE
   slot-packed transcript and NTT backend changes.
10. `results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` —
   v2.3 S1/S2 state, forward plan, cost models, and claims discipline.
11. `results/reports/baseline_2026_06_10.md` — historical full-GPU
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
| `gpu_spfss_zp.cuh` | GPU DPF/SPFSS with additive Z_p payloads (`gpuKeyGenDPFZpPair`, `gpuDpfZpFullEvalSum`). The expansion uses four domain-separated AES calls: full 128-bit child seeds from plaintexts 0/2 and separate tags from 1/3, device/host parity gated. **Security blockers:** centralized benchmark roots still come from one 64-bit `seed_base`; the two-party path uses OpenSSL-private-DRBG roots, but D-DIST/P-RNG/P-KEY and the concrete single-key reduction remain open. |
| `bench_ole_ringlpn_cuda.cu` | The Figure 2 Ring-LPN OLE engine (random ring OLE: z0+z1 = x0·x1 in Z_p[X]/(X^n+1)). Reusable via `#define RINGLPN_OLE_DISABLE_MAIN` + include; caches NTT(a)/NTT(a·a) across expand iterations. **`build_spfss_keys()` is the centralized-keygen oracle boundary.** |
| `bench_linear_ole_ringlpn_cuda.cu` | Ring-polynomial matrix Beaver from two OLEs per ring product. |
| `bench_vole_ringlpn.cu` | Older standalone VOLE expansion prototype. |
| `orca_fc_ringlpn_keywriter.cuh` | Host helpers + dealer/oracle keywriter used by `nn/orca/fc_layer.cu` behind `ORCA_RINGLPN_FC_KEYS` (bw≤32; baseline Orca byte-identical with flag off). Has `exactZmToRingShares` (conversion oracle), CRT/q128 helpers, and a clear value-dependent `dot >= Q` abort; the target replaces that predicate with the public admissibility check `K*2^(2*bw+2)<Q`. |
| `orca_fc_ideal_ole_transcript.cuh` + `bench_orca_fc_ideal_ole_transcript.cu` | Step-1 artifact: dealerless FC transcript with an *ideal* OLE oracle. Kept as reference; superseded by the real-OLE transcript. |
| `bench_orca_fc_real_ole_transcript.cu` | **The flagship artifact.** Real Figure 2 OLE + slot packing (forward negacyclic NTT is a ring isomorphism on the fully-split ring → one ring OLE = up to n scalar OLEs) + per-slot derandomization + per-party Garner CRT lift (q128) + conversion + Orca key write, validated via unchanged `gpuMatmulBeaver`. Ring-OLE count: `2·limbs·ceil(MKN/n)`. |
| `bench_orca_fc_ringlpn_demo.cu` | Byte-compatibility demo: forward + dW + dX key contracts at q64/q128. |
| `test_secure_convert.cpp` | **Validated two-process OT-backed partial S6 artifact (2026-08-03).** Party-separated Z_M→Z_2^bw conversion over SCI sockets: exact OT daBits/edaBits, OT-backed Boolean triples, batched ripple openings, versioned party-local records, TEST-ONLY correlation/output/corruption checks, invalid-bound rejection, and separate setup/correlation/online byte and direction-switch accounting. All four q64/q128/bit-width cases pass. Still plain unauthenticated TCP, IKNP rather than silent OT, linear-depth ripple, CPU-side, and not wired into the flagship transcript. |
| `test_distributed_dpf_keygen.cpp` | **Corrected M1 host protocol-logic prototype (2026-07-29).** Two-party DPF keygen: secure adder for α's bits (L−1 bit triples), cancellation-lemma level walk (2 string OTs/level), and Phase C arithmetic-share multiplication (3 scalar OLEs) that opens only standard `finalCW`. Six invalid-input controls, five independent key corruptions (root seed, `sCW`, `tLCW`, `tRCW`, `finalCW`), omniscient old-sign regression, per-phase transcript accounting, ideal-functionality mask-draw accounting, and consume-once correlation-ID reuse control. Emits standard `spfss_host::DPFKey`s validated by unchanged `dpfEvalAll`; ideal OT/triple/OLE interfaces and non-cryptographic splitmix64 correctness PRG mean functional compatibility, not computational privacy. Party private-random-tape freshness is not executable evidence. `spfss_host.cpp` remains untouched. |
| `two_party_ot.h` | **Real two-party transport (2026-07-29; CSPRNG correction 2026-08-03).** Wraps the repo's unmodified SCI IKNP stack into 128-bit string OT, Boolean triples, and Gilboa `Z_p` OLE; protocol randomness comes from buffered `RAND_priv_bytes`, while `mt19937_64` is confined to explicitly seeded public test inputs. Reports semantic logical/meaningful share widths separately from encoded NetIO bytes and direction switches. IKNP is OT **extension**, not silent OT; plain TCP is unauthenticated. |
| `dpf_key_io.h` | Versioned little-endian `spfss_host::DPFKey` batch serialization (magic `RLPNDPF1`) plus the explicitly TEST-ONLY private-input record the offline checker needs. |
| `test_two_party_dpf_keygen.cpp` | **The two-PROCESS keygen artifact.** Same frozen protocol, but two OS processes over two TCP sockets with real OT/triples/OLE, each party writing only its own key file; gates the contract's closed forms in-process and reports measured wire bytes, direction switches, and setup cost. Primitive self-tests (`--selftest`) open triple/OLE shares in a labelled test-only mode. |
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

Safe to state (every item gate-verified):
- FC Beaver keys from **real** Figure 2 Ring-LPN OLEs validate through
  unchanged `gpuMatmulBeaver` (q62 bw≤16, q124 bw=32; uniform+regular noise).
- Slot packing resolves PCG amortization: 2 ring OLEs back 8192 scalar cross
  terms (vs 16,384 ideal-OLE calls); openings are the standard `d=a−X0`,
  `e=b−X1` derandomization messages.
- The live two-process secure-conversion artifact passes 76 conversions in
  each of four q64/q128/bit-width configurations: 124/248 AND triples,
  125/249 post-mask dependency rounds, 312/622 logical opened bits, and
  624/1,244 meaningful share bits per q64/q128 output. It separately reports
  encoded setup/correlation/online bytes and direction switches; neither
  semantic counter is wire traffic.
- The distributed key-generation protocol logic is implemented
  party-separated and functionally validated by the unchanged evaluator,
  using ideal OT/triple/OLE functionalities and a non-cryptographic
  correctness PRG (2,432 trees; both primes; depth ≤14; deterministic edges;
  5/5 corruption controls; 6/6 invalid-input controls; old-sign leak
  regression; ideal-mask-draw and duplicate-correlation-ID controls). Per tree,
  executable transcript accounting reports `2*depth` string OTs,
  `depth-1` bit triples, 3 scalar OLEs,
  `2*(depth-1) + 130*depth + ceil(log2(p))` logical opened bits, and
  `4*(depth-1) + 260*depth + 2*ceil(log2(p))` meaningful share bits. At
  depth 14 and the 62-bit primes these are 1,908 and 3,816 bits. The real
  transport separately measures bytes and direction switches. This is not
  "M1 done".
- The real DPF transport validates 369/369 host-reference key pairs across ten
  depth/prime/batch configurations; SCI/IKNP, Gilboa OLE, OpenSSL-private-DRBG
  roots, versioned party-local files, measured bytes/direction switches, and
  invalid/corruption controls are executable. It is not silent OT and the
  direction-switch counter is not a network-round count.
- The full-width four-call AES path matches 16 freshly device-dumped vectors
  and 88 two-process keys pass both batched and per-tree GPU evaluation across
  both primes. This is correctness/compatibility evidence, not a DPF privacy
  proof.
- Baseline Orca is untouched with the flag off.

NOT claimable yet (the honest boundaries — never blur these):
- \"Dealer removed\" or \"M3 complete\": the standalone conversion source now has
  an OT-backed two-process path, but the flagship transcript still reads both
  conversion shares through its exact oracle, and benchmark setup is not a
  complete two-process composition.
- Any concrete security level: c=2/t=8 are correctness parameters; the primes
  fully split the ring → **splittable** Ring-LPN assumption (stronger than
  irreducible). S2's preliminary audit is blocked on the sparse-projection
  selection and projected-noise reduction, so no parameter set is secure-pinned.
- Anything about nonlinear layers (ReLU/truncation FSS keys are a separate
  dealer axis).
- The ideal-functionality D1 artifact alone does not establish computational
  privacy. The real transport and GPU-consumable path add process isolation,
  measured bytes/direction switches, OpenSSL-private-DRBG roots, and full
  128-bit seed/tag separation, but they still do not prove DPF key
  distribution or single-key privacy. Centralized benchmark roots remain
  derived from one 64-bit `seed_base` and are not a security realization.

## Binding execution order (S1–S10 plan; proposal components M1–M6)

S1 freezes the functionality/proof contract. **S2/M5 then audits and pins the
exact splittable-Ring-LPN parameters and resolves the novelty/provenance
boundary before performance implementation** because either can force changes
to `n,c,t`, primes, NTT/backend code, memory, bootstrap capacity, or the thesis.
Only after S1/S2 may M1 distributed GPU DPF keygen and M3 protocol-backed
conversion proceed in parallel. M2 wires M1 into the real-OLE transcript; M4
performs the complete forward/bias/truncation/`dW`/`dX`/bias-gradient/
dual-optimizer mask-and-velocity transition under coin-tossed two-process
composition; M6 runs model-scale evaluation including the closest dealerless
baseline and the publication report.
The full staged route and required atomic commits
are in `results/reports/publication_readiness_plan_2026_07_21.md`.

The ideal M1 protocol-logic slice, real two-process SCI/IKNP transport,
OpenSSL-private-DRBG roots, GPU-consumable full-width AES keys, and measured
bytes/direction switches are functionally validated. Remaining M1 work is a
silent-OT backend, GPU-side batched keygen, independently generated GPU roots
for any centralized security path, and actual network-round measurement.
D-DIST/P-RNG/P-KEY and single-key privacy reductions remain open. Design
components are D1–D5 in the v2.4 proposal (D1=keygen, D2=conversion,
D3=coin-tossed seeds, D4=two-process, D5=parameter audit); their numbering is
not execution order.

## Perf anchors (RTX 5000 Ada, this repo's gate configs)

| Metric | Value |
|---|---|
| OLE expand, n=8192 c=2 t=8 smoke | 13.3 ms (q64) / 26.8 ms (q128) |
| OLE expand, t=64 | 881 ms uniform / 61 ms regular (q64) |
| Linear OLE-to-Beaver 2×2×2 | 224 ms (q64) / 448 ms (q128) |
| Cheddar polymul n=8192 batch=64 | ~255–265 µs (q64) |
| **Bottleneck** | SPFSS full-domain eval; NTT is <1% of expand |

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
