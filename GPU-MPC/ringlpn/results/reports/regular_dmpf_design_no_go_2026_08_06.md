# Specialized regular-DMPF design checkpoint — NO-GO

**Date:** 2026-08-06
**Scope:** internal/advisor design audit; no implementation or benchmark claim

## Decision

Do not implement a DMPF replacement for the deployed regular-noise encoder at this checkpoint. This is a design NO-GO, not an impossibility theorem. No reviewed construction found in the audited sources simultaneously preserves the current fixed transcript and plain Ring-LPN/semi-honest boundary, leaves application OLE capacity, and improves the exact deployed shape.

The live regular layout has 60 public groups, 256 point functions, depth 11, 524,032 private PRG-node expansions, 56,832 DPF-key bytes per party, 768 bootstrap OLE slots, and 7,424 application slots per Ring-OLE instance. The measured classifier checkpoint attributes 19.776 ms to distributed DPF key generation and 7.034 ms to Ring-LPN expansion per 276-instance-normalized plan, giving the deliberately generous replacement ceiling of 26.810 ms and 577,787 wire bytes per instance. These are feasibility parameters, not security-pinned parameters.

## Required replacement functionality

For every CRT limb and every public regular group, a replacement must output additive shares of

\[
F_{i,j,g}(x)=\sum_{r,s:\,r+s=g}u_{i,r}v_{j,s}[x=a_{i,r}+b_{j,s}]\pmod p,
\]

while preserving duplicate accumulation, coefficient cancellation, both deployed primes, fixed public descriptors and failure schedule, and consume-once correlation identities. Neither party may learn support, multiplicities, peer factors, the combined function, or the peer key/output share.

## Candidates rejected

| Candidate | Privacy/correctness disposition | Exact deployed-shape blocker |
|---|---|---|
| Reverse Cuckoo | Published simulator reveals support-dependent hash descriptors; using it requires an explicit leakage-robust assumption rather than the current plain Ring-LPN boundary. | Pinned stock and exact-`p0` native-folded runs are CPU/layout/setup mismatched and have no raw 31-diagonal GPU evidence. The exact diagnostic is slower/larger than the current generous ceilings. |
| Dense NTT-slot multiplication | Fixed transcript and straightforward simulation. | Requires all 8,192 output OLE slots instead of 768 bootstrap slots, leaving no application expansion. |
| Input-independent cuckoo with secret placement | Can hide placement, occupancy, stash and retry state. | Oblivious insertion, padding and support-independent failure exceed the cost of the existing 1–8 point groups and reproduce the known theoretical-cuckoo route. |
| Programmable DPF/P-PPRF | Fixed-round distributed setup without support descriptors. | The cited estimate `M ≈ 0.318 N / epsilon^2` already gives about 83,362 leaves at `N=1024, epsilon=2^-4` and 21.34 million at `epsilon=2^-8`, before 256 points, payloads and amplification. |
| Secure active-path generation | Avoids full-frontier expansion. | The next AES seed and branch remain secret-shared; evaluating AES in MPC replaces 524,032 local expansions with tens of millions of nonlinear gates under optimistic accounting. |
| Reused Ring-OLE generation mask | Algebraically produces many products. | Unsafe on later consumption: opening `e_j=B_j-X` during generation and `epsilon_j=y_j-B_j` during use reveals `y_j-y_k` across reused slots. |

The specialized regular decomposition is useful engineering context but is not by itself a new primitive. Tree-based distributed generation, programmable DPF, ordinary cuckoo insertion, Reverse Cuckoo, Improved DMPF and SLAMP-FSS already cover the relevant abstraction choices. A future GO requires a genuinely new fixed-transcript construction with roughly `O(D)` local expansion, setup below the 256-tree baseline, no support-dependent descriptor, and a complete simulator.

## Source boundary

Primary sources audited for this decision:

- Boyle et al., programmable distributed point functions: <https://crypto.iacr.org/2022/papers/538806_1_En_5_Chapter_OnlinePDF.pdf>
- Boyle et al., Improved DMPF, IEEE S&P 2025: <https://doi.org/10.1109/SP61157.2025.00044>
- Agarwal, Raghuraman, Rindal, Reverse Cuckoo / fully distributed DMPF: <https://eprint.iacr.org/2025/2294>
- Pisetskaia et al., SLAMP-FSS: <https://cic.iacr.org/p/3/1/16>
- 2026 DPF/FSS survey: <https://arxiv.org/abs/2607.27696>

Repository evidence remains in `closest_dmpf_baseline_audit_2026_08_04.md`, `s2_architecture_comparison_2026_07_29.md`, the pinned Reverse-Cuckoo reports, and the exact-`p0` adapter artifact. The design exercise changed no protocol source or benchmark; this checkpoint updates the live roadmap and manuscript so they no longer prescribe an unsupported DMPF implementation.

## Route selected after the NO-GO

The owner selected the forward-linear-layer systems route. The binding continuation plan is `full_linear_layer_systems_plan_2026_08_06.md`. Reverse Cuckoo remains a labelled non-comparable diagnostic; it is not the implementation route.
