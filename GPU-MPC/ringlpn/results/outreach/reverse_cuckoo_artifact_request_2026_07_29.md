# Outreach draft: Reverse Cuckoo / fully distributed DMPF artifact request

**Not sent yet - requires owner approval before sending**

**Date drafted:** 2026-07-29
**Purpose:** request the Fully Distributed Multi-Point Function (Reverse Cuckoo)
prototype artifact from its authors and ask them to clarify the four printed
inconsistencies our audit recorded, so the only fully distributed prior design
can be benchmarked or safely built on.

**Context for the owner (two lines).** This closes open question 2 of
`results/reports/s2_architecture_comparison_2026_07_29.md` (lines 273-276): no
public source exists, so neither the artifact nor a matched private-factor row
is currently obtainable. Nothing here claims anything about our own security
level, novelty, or performance, and no parameter is pinned by sending it.

**Source anchors (every technical assertion below traces to these):**

- `results/reports/s2_architecture_comparison_2026_07_29.md:156` - candidate
  row: private shared positions/payloads (Fig. 6), explicit dedup, additive
  over a group, **no public source found**, printed Fig. 7 internally
  inconsistent.
- `results/reports/s2_architecture_comparison_2026_07_29.md:160-171` - the four
  specific blockers and the quoted Table 1 row.
- `results/reports/s2_architecture_comparison_2026_07_29.md:260,273-276` - it is
  the only construction whose ideal functionality matches our private-factor
  setting; the recorded action is to contact the authors.
- `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md:399-419`
  - Figure 7 padding/rank/solve detail, missing bin-solver type conversion,
  Figure 9's omitted payload reduction; "specification blockers, not an
  experimentally reproduced attack".
- `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md:421-423`
  - rank bound `2^-(q-d)` conditional on the Section 7.3 *conjectured* hash
  property; no concrete finite-parameter cuckoo abort bound.
- `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md:430-441`
  - Table 1, p. 43, i7-13700H, single-threaded per party: 128 points at `2^20`
  = 988.50 ms setup, 38.50 ms average expansion.
- `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md:386` -
  ePrint record links no repository; repository search found no matching source;
  paper is CC BY 4.0, no code license found.

---

Subject: Artifact request and four questions on ePrint 2025/2294 (fully distributed multi-point functions)

Dear Agarwal, Raghuraman, and Rindal,

We are a GPU MPC preprocessing project evaluating fully distributed multi-point
FSS as the setup component of a Ring-LPN PCG, and your *Fully Distributed
Multi-Point Functions for PCGs and Beyond* (ePrint 2025/2294, revision dated
2026-01-23) is the closest match we have found to that setting.

We could not locate a public repository for the prototype. Would you be willing
to share the source, or a binary/artifact? Our reference row for benchmarking is
Table 1's 128 points at domain 2^20 (988.50 ms setup, 38.50 ms average
expansion, i7-13700H).

Four questions from reading the printed protocols:

1. Figure 7 pads the `m-t` dummy rows with `H_i(N)=0`, then requires each `d x q`
   block to have rank `d` and to solve against `(0,1,...,d-1)`. With `t <= d` we
   could not see how a block containing a zero row satisfies both. Is there an
   additional convention we are missing?
2. In the characteristic-two bin-solver, how is the integer bin-label
   right-hand side intended to be converted to the solver's type?
3. Figure 9 states the collision accumulation; is the final payload reduction
   described in the prose meant to be applied there?
4. Is a concrete finite-parameter cuckoo abort probability available, and is the
   rank bound intended to rest on the conjectured hash property of Section 7.3?

We would gladly share our reproduction harness, and we will cite and credit any
clarification you provide.

Thank you for your time.

Alp
<affiliation/email to fill in>
