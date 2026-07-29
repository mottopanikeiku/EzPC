# Decision request: dealerless Orca Ring-LPN S2

**Date:** 2026-07-29
**From:** Alp
**Subject:** S2 hard stops: 2026 DMPF prior art, parameter proof gap, and project overlap

Professor,

I completed the pre-implementation parameter/novelty/provenance audit for the
dealerless Orca FC project. I have not started the next implementation stage
because the audit found three decisions that change the research direction.

## Findings needing your decision

1. **Closest prior art changed the DPF contribution boundary.** BCG+20 already
   specifies semi-honest distributed Ring-LPN seed setup using
   Doerner--shelat DPF generation on shared positions and payloads.
   Programmable DPFs (CRYPTO 2022) give constant-round distributed generation.
   More directly, Agarwal, Raghuraman, and Rindal, *Fully Distributed
   Multi-Point Functions for PCGs and Beyond* (ePrint 2025/2294, posted
   2026-01-23), gives a fully distributed DMPF with a semi-honest proof and
   prototype specifically for Ring-LPN/Stationary-LPN PCGs, replacing the sum
   of point DPFs and reporting order-of-magnitude gains. SLAMP-FSS (IACR CiC,
   May 2026) independently advances two-party multi-point FSS using tree PRGs
   and linear systems. Our current per-point host DPF is therefore a
   compatibility prototype/baseline, not presently a defensible protocol
   contribution.

2. **The 128-bit parameter step still has a proof gap.** The literature mapping
   is `c=4,total weight w=64`, hence this code's per-polynomial `t=16`. Running
   the accepted EUROCRYPT 2024 estimator over each deployed 62-bit prime gives
   145.85 bits for the BCG-published degree-128 projection. But the paper's
   literal `w_i <= (c-1)2^i` rule selects degree 16 here (`47.0967 <= 48`),
   whose estimate is only 57.29 bits, while Table 1 reports degree 128. The
   separate PCG script reproduces the table only by substituting a fitted
   constant (`3.3`), not a theorem. We also need to justify mapping
   the dependent projected noise to the estimator's exact/regular model. I am
   not calling any point 128-bit secure until a cryptographic review closes
   those steps.

3. **Newer architectures compete with Ring-LPN/NTT plus conversion.** Li,
   Xing, Yao, and Yuan, CRYPTO 2025/ePrint 2025/1223, construct silent PCGs and
   triples directly over `Z_(2^k)`, potentially removing conversion. Li et al.,
   ePrint 2026/196, give a fully implemented QA-SD/Walsh--Hadamard prime-field
   OLE/VOLE PCG that avoids FFT multiplications and FFT-friendly-prime
   restrictions. CRYPTO 2025 Stationary Syndrome Decoding also amortizes the
   high Ring-LPN noise-generation cost under a different structured-noise
   assumption. All three routes should be considered before architecture freeze.

A preliminary `n=2^14,c=4,t=16` point satisfies only the per-point bootstrap
inequality (`12,288 < 16,384`) and passes the existing GPU correctness
validator: 68.35 ms q64 and 133.26 ms q128 mean expansion over three measured
iterations. After reserving next-epoch keygen correlations, only 4,096 slots
(25%) remain, so the `16x32x16` case needs at least four q64/eight q128 ring
OLEs—twice the isolated artifact count—before conversion and safety reserves.
This is feasibility only, not a complete epoch, security, or end-to-end result;
adopting DMPF changes the setup count.

## Provenance issue

The private `yanxue820/PCG-acceleration` repository has no LICENSE file and has
multiple contributors. Its history attributes the initial DPF/PCG to Jiayan
Xue and Chenkai Weng; Alp later added GPU NTT and GPU DPF work; LYCesh added CPU
benchmark work; T. K. Gong added PIM/simulator/campaign work. No code or result
from that repository has been imported into the Orca project during this audit.
Private commit `e821141` by Chenkai (2026-07-12) replaced a leaky DPF output
with a hash plus Beaver-corrected correction word before this fork's corrected
three-OLE Phase C was committed as `28f8451` (2026-07-21). The implementations
differ, but both securely multiply private aggregates before opening a
correction word. Git cannot establish independent derivation; this specific
idea/provenance/credit question must be answered, not inferred.
The active Orca NTT backend is separately Cheddar-derived. This audit
added the citation and complete MIT notice and recorded the reconstructed
source pin and local delta.

## Recommended default pending your answers

- Frame the contribution as the integrated dealerless Orca FC preprocessing
  system. Keep the current point DPF only as an executable compatibility
  baseline; do not claim it as a new protocol.
- Before more optimization, make a small, assumption-matched comparison of the
  2026 fully distributed DMPF, SLAMP-FSS, Stationary-SD, direct-`Z_(2^k)`, and
  QA-SD/WHT routes. Prefer a published implementation with a clear license over
  another local protocol unless a precise new delta is identified.
- Keep `n=2^14,c=4,t=16` labelled only as a feasibility point and
  `n=2^20,c=4,t=16` only as a literature reference until an independent
  cryptographic reviewer resolves the projection-selection contradiction,
  projected-noise distribution, advantage loss, and quantum cost.
- Quarantine the private PCG/PIM repository from this paper until contributors
  approve a written component/credit/reuse matrix and specifically resolve the
  `e821141`/`28f8451` output-layer overlap.
- Retain the Cheddar-derived NTT for internal experiments under its MIT notice
  and citation, but describe it as a measured dependency, not a contribution.
  Use a clean external backend boundary for release if provenance cannot remain
  complete and auditable.

## Project-owner consultation record (2026-07-29)

The project owner selected the following direction:

1. lead with the **integrated dealerless Orca FC system**;
2. compare the 2026 fully distributed DMPF and SLAMP-FSS against the current
   point-DPF compatibility baseline before choosing the key-generation route;
3. compare regular Ring-LPN/NTT, Stationary-SD, direct-`Z_(2^k)`, and
   QA-SD/WHT before choosing the PCG architecture;
4. require professor approval for the projection argument and 128-bit
   interpretation;
5. use `n=2^14,c=4,t=16` only as a feasibility tier and
   `n=2^20,c=4,t=16` only as a literature-reference tier until approval;
6. treat the private PCG/PIM repository as part of the same research stream;
7. retain the Cheddar-derived backend with its source pin, MIT notice,
   citation, and local-delta record.

The owner offered unrestricted reuse of the private repository. That offer
does not establish permission from its other contributors or cure the missing
repository license. No private code, measurements, figures, or prose are
needed for the selected public-prior-art comparison, so the safe execution
boundary is stricter: import none. The only remaining private-project question
is the professor's written authorship/credit/overlap disposition for the
same-research-stream relationship and the `e821141`/`28f8451` chronology.

These owner decisions authorize the public-source comparison work below. They
do not approve a Ring-LPN parameter set, a 128-bit claim, private-project
reuse, external circulation, or the binding S2 checkpoint.

## Requested answers

Please answer each item explicitly:

1. Is the intended contribution the **integrated dealerless Orca FC system**, or
   must the paper include a new distributed-DPF protocol?
2. Should we adopt/benchmark the 2026 fully distributed DMPF and SLAMP-FSS,
   keep the current per-point DPF only as a baseline, or pursue a specific
   protocol delta?
3. Should we stay with regular Ring-LPN/NTT plus conversion, adopt Stationary
   Syndrome Decoding, or compare/pivot to the 2025 direct-`Z_(2^k)` and 2026
   QA-SD/WHT PCGs?
4. Who should review and approve the sparse-factor projection criterion,
   projected-noise reduction, and classical/quantum 128-bit interpretation?
5. Should the parameter target start at literature scale `n=2^20,c=4,t=16`,
   preliminary bootstrap scale `n=2^14,c=4,t=16`, or another reviewed point?
6. For the private PCG/PIM project, which contributors own each component and
   which code, algorithms, measurements, figures, or prose may Alp reuse?
   What citation, acknowledgement, or overlap disclosure is required?
7. What is that project's submission/public-release status relative to this
   sole-author Orca work?
8. May the now-attributed Cheddar-derived backend remain, or should the paper
   use a clean external backend boundary?

The detailed evidence and source links are in
`results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md`. Until
these decisions are recorded, I will not start S3, import overlapping code,
claim protocol novelty, claim 128-bit security, or circulate the paper.
