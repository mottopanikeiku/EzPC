> **HISTORICAL 2026-07-29 pre-sweep/pre-transport snapshot.** Superseded by
> `CLAUDE.md`; statements below may describe an older state.

# Session handoff: consultation-driven DMPF/PCG comparison (2026-07-29)

## Stop point

The project owner selected the **integrated dealerless Orca FC system** as the
paper thesis, instructed us to compare the current per-point DPF path with
published DMPF/PCG alternatives before freezing an architecture, and authorized
use of public artifacts. No architecture or parameter set is frozen. S3 must
not start yet.

The last pushed evidence checkpoint is commit `aec8d7d` on `origin/master`
(`ringlpn(audit): record S2 hard stops and evidence`). It is a preliminary
hard-stop checkpoint, **not** an advisor-reviewed S2 pass. Work described below
was performed after that checkpoint and has not yet been committed or pushed.

## Decisions recorded

The project-owner consultation is recorded in:

- `results/reports/s2_professor_decision_request_2026_07_29.md`
- `CLAUDE.md`

The selected direction is:

1. lead with the integrated dealerless Orca FC systems contribution;
2. compare the current point-DPF route with fully distributed DMPF and
   SLAMP-FSS before selecting an architecture;
3. compare Ring-LPN plus conversion with Stationary-SD, direct
   `Z_(2^bw)` PCGs, and QA-SD/WHT prime-field PCGs;
4. keep Alp as sole author;
5. do not use or attribute material from the private `yanxue820/PCG-acceleration`
   repository without an explicit professor decision. No code from it was
   imported in this work.

## Corrected comparison scale

A critical notation error was found and corrected in the active audit report:

- DMPF sparse-point count is `m = c^2*t^2`.
- `(c,t)=(2,8)` therefore has `m=256`, not approximately 98.
- `(c,t)=(4,16)` has `m=4,096`, not 12,288.
- `768` and `12,288` are the current protocol's **scalar-OLE setup-slot
  counts**, because it consumes three scalar OLEs per DPF point.

The matched sparse functionality and tier table are now in Section 7 of:

- `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md`

Current requested tiers are:

- smoke/feasibility: `n=2^13,c=2,t=8`, DMPF domain log 14, 256 total raw
  Cartesian-product points;
- BCG-scale reference: `n=2^20,c=4,t=16`, domain log 21, 4,096 total raw
  points;
- a reviewer has additionally flagged the preliminary architecture candidate
  `n=2^14,c=4,t=16` as necessary for a decision-grade comparison because it
  differs from the literature reference by 64x in full-domain size.

Every final row must also report the post-collision unique nonzero support.
The CC0 S&P DMPF implementation requires distinct inputs, so secure private
collision coalescing is an additional protocol cost, not a free preprocessing
step.

## Public artifact and license findings

Recorded in the same Section 7 audit:

| route | exact pin | current disposition |
|---|---|---|
| IEEE S&P 2025 improved DMPF | `MatanHamilis/dmpf@ed044b903fdf6fd213b171eaa125e4eb52363903` | CC0-1.0; safe to rerun as a centralized expansion/key-size baseline; not fully distributed setup |
| SLAMP-FSS | `jrmngndr/slamp-fss@893650f6a2ce902172ffeb016d82683db295c4df` | paper CC BY 4.0, but repository has no license; published/analytic rows only unless written code permission is obtained |
| Fully Distributed DMPF / Reverse Cuckoo | ePrint revision dated 2026-01-23 | paper CC BY 4.0; paper reports a prototype but no public source repository was found; request artifact from authors |
| current point-DPF prototype | EzPC checkpoint `28f8451`, corrected contract `63a0c05` | reproducible protocol-logic baseline with ideal transports; not a cryptographic dealerless implementation |

SLAMP's published `Gen` is centralized and distributed setup is explicitly
future work. Its outputs are XOR shares over characteristic-two fields, not
additive shares over the deployed odd q62 primes. Published Table 4 uses
`n=20,t=22,k=v=64,v_bar=128` on an Intel Xeon E5-2640 v3 and reports:

- BGH+25 big-state: Gen 396.0 us, EvalRand 2.82 us, FullEval 102 ms;
- SLAMP-FSS: Gen 1840 us, EvalRand 9.55 us, FullEval 1480 ms.

The paper says those full-evaluation implementations are not optimization
matched. No speedup ratio is valid for the Orca decision.

## New reproducible benchmark adapter

Created but not yet run as the final evidence sweep:

- `scripts/dmpf_baseline_bench.rs`
- `scripts/run_dmpf_baseline_comparison.sh`

The runner pins the CC0 source revision, compiles it with
`nightly-2024-09-29`, pins a CPU, and records three repeated rows for:

1. current feasibility: log-domain 14, 256 points;
2. preliminary candidate: log-domain 15, 4,096 points;
3. literature reference: log-domain 21, 4,096 points.

It measures centralized key generation, each party's full evaluation
separately, serial two-party evaluation, allocator-derived retained key bytes,
peak additional allocation, output bytes, retry count, and full-domain
correctness. It records that the public implementation uses two coordinates of
the Goldilocks prime `0xFFFFFFFF00000001`, not the deployed q62 CRT primes.

Final command (not yet executed):

```bash
cd GPU-MPC/ringlpn
RUNS=3 CPU=0 scripts/run_dmpf_baseline_comparison.sh
```

Expected outputs:

- `results/dpf/dmpf_baseline_comparison_2026_07_29.csv`
- `results/dpf/dmpf_baseline_comparison_2026_07_29.log`

Exploratory one-run observations, **not final evidence**, were:

- log 14 / 256 points: point-DPF full evaluation about 146 ms; big-state about
  24.6 ms; OKVS about 6.6--6.8 ms;
- log 21 / 4,096 points: big-state Gen about 4.74 s, serial two-party full
  evaluation about 548 s, retained two-key allocation about 2.82 GB; OKVS Gen
  about 102 ms, serial full evaluation about 981 ms, retained two-key
  allocation about 79.7 MB.

All exploratory correctness checks passed. These runs used one host process,
centralized generation, distinct random points, and sequential evaluation, so
they do not establish dealerless latency or q62 compatibility.

## Parallel research state

Four read-only parallel tasks were launched:

- `SlampPaperExtract`: completed; detailed result is available at
  `agent://SlampPaperExtract` / `history://SlampPaperExtract`.
- `DmpfPaperExtract`: completed; detailed result is available at
  `agent://DmpfPaperExtract` / `history://DmpfPaperExtract`; the local revised
  paper is `/tmp/librarian-dmpf-2025-2294.pdf`. Direct inspection confirmed
  that Figure 7's dummy rows, `H_i(N)=0`, full-row-rank requirement, and
  nonzero bin-label right-hand side are mutually inconsistent as printed.
- `PcgRouteMatrix`: still running. Preliminary source-verified findings:
  direct-`Z_(2^bw)` artifact `zhli271828/Trace-F2-OLE-PCG@43959ef` is MIT and
  reproduces expansion only; its distributed setup is not reproduced. The
  QA-WHT May-27 paper snapshot reports distributed-setup rows, but its anonymous
  artifact was unavailable and no code license was verified.
- `ComparisonReview`: completed at `agent://ComparisonReview`; it requires raw
  versus unique support counts, the `n=2^14,c=4,t=16` candidate tier,
  factor-to-product MPC costs, net epoch capacity, and a separate whole-PCG
  route comparison before the contract is decision-grade.

Do not treat preliminary IRC messages as final citations; read and verify each
agent's completed output first.

## Remaining active work before the next consultation

1. Public-implementation/license inventory is complete.
2. The pinned DMPF comparison sweep is running; inspect every CSV row after it
   exits.
3. Reverse Cuckoo's printed protocol ambiguities were independently confirmed;
   obtain the authors' artifact/clarification or keep it
   published-only/unreproduced and architecture-blocking.
4. Integrate exact SLAMP, Reverse Cuckoo, and four-route PCG source citations
   into the S2 audit.
5. Reproduce the MIT direct-`Z_(2^bw)` expansion artifact at the two matched
   tiers if its fixed parameters permit a truthful match; otherwise record the
   exact mismatch.
6. Produce one decision table separating functionality, setup model, security
   evidence, field/ring, key bytes, setup communication/rounds, expansion time,
   memory, and evidence status (`published`, `reproduced`, or `derived`).
7. Present the table and unresolved choices to the project owner **before**
   freezing a DPF/DMPF or PCG architecture.

## Binding blockers unchanged

- No Ring-LPN parameter set or 128-bit security claim is pinned.
- The accepted EUROCRYPT 2024 estimator gives 145.85 bits for the published
  BCG degree-128 regular finite-field projection, but BCG's literal reduction
  rule selects degree 16 with only about 57.293 bits. The mapping/proof gap is
  unresolved.
- The preliminary `n=2^14,c=4,t=16` candidate closes only the three-OLE-per-DPF
  bootstrap lower bound; it does not close the complete epoch budget.
- No real distributed OT/OLE transport, private deduplication protocol,
  conversion budget, two-process deployment, or independent cryptographic
  review exists yet.
- S3 remains blocked; no paper may claim 128-bit security, a new DPF protocol,
  dealerless end-to-end security, or publication readiness.
