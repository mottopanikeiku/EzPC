# Hybrid regular-SD formula audit (internal/advisor, 2026-08-04)

## Decision

The classical large-field formulas in Wang et al., *A Hybrid Algorithm for the Regular Syndrome Decoding Problem*, are sufficiently explicit to instantiate the **direct** regular-SD tuples

\[
(N,K,h,\beta,p)=(cn,(c-1)n,ct,n/t,p).
\]

`scripts/audit_hybrid_rsd_asiacrypt2025.py` therefore performs an exhaustive integer optimization of the paper's Theorem 1 and writes `hybrid_regular_sd_asiacrypt2025_2026_08_04.csv`. These are diagnostic field-operation counts, not concrete time, not an executable attack, not a structured-code reduction, not a Ring-LPN claim, and not a parameter pin.

## Primary-source pin and artifact search

- IACR ePrint [2025/1284](https://eprint.iacr.org/2025/1284), whose landing page reports “2025-09-09: last of 2 revisions.”
- Formula source: the 39-page PDF created 2025-09-04 and captured by the Internet Archive on 2025-09-07 at [this immutable URL](https://web.archive.org/web/20250907044313id_/https://eprint.iacr.org/2025/1284.pdf), SHA-256 `a8d050905021bc737537d054ed33de643512d017a4d3d5d893167d844b6d494a`. The ePrint landing page is the authority for revision history; the capture timestamp is recorded rather than asserted to be a separately authenticated ePrint revision identifier.
- ASIACRYPT 2025 proceedings version: DOI [10.1007/978-981-95-5113-2_15](https://doi.org/10.1007/978-981-95-5113-2_15), pp. 466–497, first online 2025-12-08.
- No author implementation or estimator was located as of 2026-08-04. The ePrint page exposes only the PDF, the paper contains no code or artifact URL, the proceedings page contains no supplementary-code link, and an exact-phrase GitHub repository search returned zero repositories. This is an absence report, not proof that private or unindexed code does not exist.

The calculator pins equations (9)–(13), not table values. Its self-test independently reproduces two published rows including their reported optimizers: Table 6 `RSD(2^12,1589,172)` after the paper's truncation convention gives `132.60` with `(f_bar,u_bar,g)=(21,7,2)`; Table 5 `RSD(2^14,3482,338)` gives `133.15` with `(148,7,2)`.

## Implemented formula and optimization

Let `r=beta-u_bar`, `omega=2.8`, and use the paper's notation:

- `m1 = f_bar*C(beta-u_bar-1,2) + (h-g-f_bar)*C(beta-u_bar,2)`;
- `n0 = K-f_bar-(h-g)*u_bar`, `w=g*beta`, `v=n0-w`;
- `z = v+2g+gv+C(g,2)`;
- equation (9): `m1-v(v+1)/2 >= 5z/2`;
- `P' = (1-(u_bar+1)/beta)^f_bar * (1-u_bar/beta)^(h-g-f_bar)`;
- `Ttotal = P'^(-1) * (T1+T2+beta^g*T3)`, with `T1`, `T2`, and `T3` exactly as in Theorem 1;
- space is reported only as the base-2 logarithm of the expression inside the paper's `O((N-K)N + m1(n0^2+3n0)/2)` field-element bound.

Every admissible integer `(f_bar,u_bar,g)` satisfying the stated bounds and equation (9) is covered. For fixed `(g,f_bar)`, equation (9) is a concave integer quadratic in `r`; exact integer binary searches identify its complete feasible interval and every point in that interval is evaluated. Search over increasing `g` stops only when the proved lower bound `beta^g >= incumbent Ttotal` excludes that `g` and all larger values.

## Direct candidate results

The two deployed primes give identical **counts of field operations**; no claim is made that their operation latency is identical. The CSV contains one row per prime and orbit treatment. The baseline rows use no orbit reduction.

| candidate | direct `(N,K,h,beta)` | prime limbs | optimum `(f_bar,u_bar,g)` | log2 expected iterations | log2 classical F_p operations | log2 space expression (F_p elements) |
|---|---|---|---|---:|---:|---:|
| `n20_c4_t16` | `(4194304,3145728,64,65536)` | p0, p1 | `(64,49136,0)` | 127.915506 | 187.102387 | 51.822429 |
| `n20_c4_t32` | `(4194304,3145728,128,32768)` | p0, p1 | `(49,24567,0)` | 255.805853 | 315.011577 | 51.223626 |
| `n20_c4_t64` | `(4194304,3145728,256,16384)` | p0, p1 | `(242,12282,0)` | 511.544508 | 570.799009 | 50.685300 |
| `n20_c8_t8` | `(8388608,7340032,64,131072)` | p0, p1 | `(14,114672,0)` | 191.911107 | 251.542715 | 51.970128 |
| `n20_c8_t16` | `(8388608,7340032,128,65536)` | p0, p1 | `(99,57334,0)` | 383.792131 | 443.445578 | 51.422530 |

Here `p0=4611686018326724609` and `p1=4611686018309947393`. The input distribution recorded in every row is exactly `h=ct` equal blocks of width `beta=n/t`, one nonzero position in each block, with independent uniform `F_p^*` payloads. The paper's cost does not depend on the distribution of nonzero values, but the live distribution is stated to prevent a silent model substitution.

Success means the probability `P'` that one independently selected puncturing pattern is error-free. `Ttotal` uses the expected `1/P'` iterations. It is not a fixed-work success quantile. Data means one `H in F_p^((N-K) x N)` and syndrome; the formula does not price acquiring or materializing that matrix.

## Classical/quantum and orbit scope

The source supplies a classical field-operation analysis. It gives no quantum algorithm, quantum memory model, or quantum success-amplification cost. The CSV records this explicitly rather than taking a square root of the attack time.

Optional orbit sensitivity is emitted as separate rows. For the full, unprojected negacyclic-to-cyclic instance, those rows use orbit size `n=2^20` and mechanically subtract `log2(sqrt(n))=10` from the baseline. This is **not in Wang et al.**, and treating the orbit as an end-to-end decoder speedup is heuristic. It does not alter the baseline rows or their optimizer, and it must not be read as a concrete structured attack.

## Assumptions and blockers to an executable attack

1. Theorem 1 counts arithmetic operations over `F_p`; it does not define wall-clock costs, bit-operation costs, parallel depth, or a concrete memory allocator.
2. The space statement is big-O. Reporting its displayed expression in field elements does not turn it into a byte bound.
3. The paper heuristically treats the relevant Macaulay submatrix as full rank when it has many more rows than columns. It reports experiments for paper instances, but no experiment or failure bound is available here for these five direct tuples.
4. The factor `5/2` in equation (9) is a conservative author choice inherited from BDT; the paper notes that other constants greater than one are possible. This audit uses exactly `5/2` and does not tune it.
5. There is no author attack artifact to execute. Reproducing an actual attack requires the authors' implementation (or an independently reviewed implementation), precise finite-field linear-algebra kernels and rank/failure behavior, concrete memory accounting, and test instances.
6. Mapping the direct regular-SD formula count to the deployed structured Ring-LPN public matrix requires a reviewed structured-code reduction. None is supplied by the paper or this audit. The direct tuple is standard regular-SD before projection; projected occupancy/cancellation noise is not modeled here.

## Reproduction

From `GPU-MPC/ringlpn`:

```sh
python3 scripts/audit_hybrid_rsd_asiacrypt2025.py --self-test
python3 scripts/audit_hybrid_rsd_asiacrypt2025.py
```

The first command reproduces the two source table rows named above. The second deterministically regenerates the 20 CSV rows (five candidates, two primes, baseline plus separately labelled orbit sensitivity).
