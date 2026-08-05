# Structured Ring-LPN attack audit

**Date:** 2026-08-04
**Status:** internal/advisor; attack inventory and proof obligation ledger; **not a parameter pin or concrete-security review**
**Scope:** the live regular sampler, its direct expanded instance, and every one-sparse fully split projection
**Review state:** source-grounded model/attack triage plus a new elementary orbit lemma; independent human cryptographic review is still required

## 1. Decision and non-claim

No deployed or candidate tuple has a source-supported concrete-security level. In particular:

- q64 and q128 mean one and two approximately 62-bit arithmetic limbs. They do not mean 64- or 128-bit security.
- A number printed by the accepted EUROCRYPT-2024 finite-field estimator is a random-code/model attack estimate. The estimator has no ring, factor, orbit, projection-law, memory-limit, multi-instance, or CRT-advantage input.
- The existence of a cyclic orbit is proved in §5. It proves that one public syndrome supplies related same-code decoding instances. It does **not** prove a square-root running-time gain for an arbitrary large-field ISD, RSD, algebraic, or statistical decoder.
- The direct expanded noise is standard regular syndrome-decoding noise before projection. A projected noise vector follows the exact occupancy-and-cancellation law, not regular syndrome decoding.
- The 2025/2026 quasi-Abelian attacks have no published cost transfer to the deployed large prime, univariate dimensions. That transfer is unresolved, not disproved.
- Candidate rows and orbit-adjusted rows are diagnostics only until a structured-code reduction or attack theorem and an independent human review both close.

A reader must not cite this report, any generic-estimator CSV, or the orbit lemma as “reviewed concrete Ring-LPN security.”

## 2. Exact live instance

The source pin `SRC-LIVE` below samples, independently for every one of the `c` error polynomials and each of its `t` public contiguous buckets,

```text
position[j] = j*(n/t) + U_j,  U_j uniform in {0,...,n/t-1},
payload[j]  uniform in F_p^*.
```

The payload is not fixed to one. Positions and nonzero payloads are independently drawn. The live code has parity check

```text
H = [M_(a_1) | ... | M_(a_(c-1)) | I_n],
```

where each `M_(a_i)` is negacyclic multiplication by an independently uniform `a_i` in `R^- = F_p[X]/(X^n+1)`. Thus the direct expanded decoding instance is

```text
(N,k,w,q) = (c*n, (c-1)*n, c*t, p),
regular block count = c*t,
regular block width = n/t.
```

The two exact deployed fields are

```text
p0 = 4611686018326724609,
p1 = 4611686018309947393.
```

This is the standard RSD distribution: one nonzero, uniform position and uniform `F_p^*` value in every equal consecutive block. Applicability of a published RSD cost still assumes the attack's random-code/rank/list model for the structured multiplication-matrix ensemble.

For a one-sparse factor of degree `d`, the projected code has

```text
(N_d,k_d,q) = (c*d, (c-1)*d, p).
```

Its realized nonzero weight `h` is random. Collisions and prime-specific cancellations give the exact law in [the companion projection report](s2_regular_projection_law_2026_08_04.md) and `ART-LOCAL`. It is invalid to replace this law by regular RSD or by only `floor(E[h])`. A finite-field estimator call at `(c*d,(c-1)*d,h,p)` is mechanically accepted by the current guarded adapter only for `0 <= h <= d-1`; this domain check is necessary, not sufficient for applicability.

## 3. Reproducible source and tool pins

Pins identify exactly what was reviewed. “Current ePrint revision” means the archive record at the stated immutable retrieval date; no unrecorded PDF checksum is claimed.

| ID | Source pin used by this audit | Role |
|---|---|---|
| `SRC-LIVE` | `src/two_party_spfss.h`, SHA-256 `05d2fb62530f445e42f20ad1db8c484979846339c2cfdc2f276327e50b6f1017`, especially `validate_party_noise` and `sample_party_noise` | Exact deployed bucket and iid-uniform-`F_p^*` payload distribution |
| `BCG` | Boyle--Couteau--Gilboa--Ishai--Kohl--Scholl, corrected full version dated 2022-08-10, HAL `hal-03374154v1` / ePrint 2022/1035, §§8.2--8.4 and 9.1 | Ring-LPN projection, algebraic-code and quasi-cyclic/DOOM discussion; its informal full-square statement is not a theorem |
| `FF-2024` | Liu--Wang--Yang--Yu, EUROCRYPT 2024, DOI `10.1007/978-3-031-58751-1_6`; accepted artifact `eurocrypt-2024-a1`, immutable downloaded script SHA-256 `c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc` | Random-code finite-field exact/regular LPN estimator; includes pooled Gauss, statistical decoding, generic finite-field ISD, and AGB |
| `RISD-2024` | Esser--Santini, CRYPTO 2024, DOI `10.1007/978-3-031-68391-6_6`; accepted artifact `crypto-2024-a1`, published 2024-08-15, ZIP SHA-256 `04ae2586fccb10481efb861104176e4aaabb380c3cb9704b97ce3c4768a282cb`; upstream snapshot commit `afe1e408f8a46aebc15293462480f478ff969923` | Permutation, enumeration, representation, depth-2 representation, CCJ and generic BJMM regular-ISD costs |
| `ART-RISD` | `scripts/audit_regular_isd_crypto2024.py`, SHA-256 `b0864d27f03d76dd3f0bd660d33c71063ad0380498d1e07dcddc1e2f4907eff9`; `results/security/regular_isd_crypto2024_2026_08_04.csv`, 50 rows, SHA-256 `68b8329dc77d992a90257b2b6b808fc1076534305e0ec0c434831ddafb17d255`, embedded `analysis_sha256` `39159736d43e954c565645c76e0cbe1ac433e92ba1f2dcc8f2ab847af8f89dfc`; notebook member SHA-256 `cebb0861f1faa53be59eb4c11a2e38219612e1fc8d39e6cb3ce597e28717c9ec` | Executable stdlib transcription of pinned Perm/Enum/Rep/RepD2/CCJ formulas on all five direct candidates and both primes; explicit fail-closed incompatibility rows for delegated generic BJMM and projected non-RSD |
| `AGB-2023` | Briaud--Øygarden, EUROCRYPT 2023 / ePrint 2023/176; implementation used here is the `AGBforq` function frozen inside `FF-2024` | Algebraic RSD estimate under polynomial-system assumptions |
| `QA-BASE` | Bombar--Couteau--Couvreur--Ducros, ePrint 2023/845 full version, especially §§6.4 and 6.6 | Quasi-Abelian structural-code boundary and generalized orbit sensitivity statement |
| `SENDRIER` | Sendrier, PQCrypto 2011, DOI `10.1007/978-3-642-25405-5_4` | Executable-source scope: almost-square-root gain for a Stern collision-decoding variant in the stated McEliece range |
| `HYBRID-2025` | Wang--Wang--Yang--Liu--Yu--Zhang--Wang, ePrint 2025/1284 / ASIACRYPT 2025; archive revision published 2025-07-14, modified 2025-09-09, retrieved 2026-08-04 | New hybrid RSD algorithm replacing ISD meet-in-the-middle enumeration by quadratic-equation solving |
| `ART-HYBRID` | `scripts/audit_hybrid_rsd_asiacrypt2025.py`, SHA-256 `001c7c68fe53ec5f266631500f72e835940f09586aea75f134f4e0e2b87dc8aa`; pinned archived ePrint PDF SHA-256 `a8d050905021bc737537d054ed33de643512d017a4d3d5d893167d844b6d494a`; `results/security/hybrid_regular_sd_asiacrypt2025_2026_08_04.csv`, 20 data rows, SHA-256 `9a442eec7c41fc01afcd2df84494a5703330d2a041693320fc2c0b0248d978d0`; companion report SHA-256 `d05b00618a25866d7b166f4ae3e3c08ce11742a66e816e5757d302560c5a6e72` | Executable Theorem-1 formula calculator, exhaustively optimizing admissible integer `(f_bar,u_bar,g)` for all five direct candidates and both primes; not an attack implementation |
| `SSD-2025` | Kolesnikov--Peceny--Raghuraman--Rindal, CRYPTO 2025 / ePrint 2025/295; archive revision published 2025-02-20, modified 2025-08-19, retrieved 2026-08-04 | Stationary-SD with several noise vectors sharing one hidden support |
| `QA-CS-2025` | Bouillaguet--Delaplace--Hamdad--Vergnaud, ePrint 2025/892, archive revision 4 modified 2025-11-14, retrieved 2026-08-04 | Practical QA-SD interpolation/compressed-sensing attacks over small fields |
| `QA-CORR-2026` | Joux, ePrint 2026/1126 v1, published and retrieved revision dated 2026-06-01 | QA-SD correlation attack; about `1000x` time and memory improvement over the 2025 attack over `F_3` is an author-reported comparison |
| `SPARSE-SPEC-2026` | Agrawal--Bagchi--Kumar, ePrint 2026/614, published 2026-03-28, modified 2026-07-02, retrieved 2026-08-04 | Spectral/Kikuchi attacks when public equations are `k`-sparse |
| `SPARSE-SECRET-2026` | Agrawal--Bagchi--Kumar, ePrint 2026/1550 v1, published 2026-07-29, retrieved 2026-08-04 | Sparse LWE/LPN with a sparse coefficient matrix and bounded small secret; distinct from `SPARSE-SPEC-2026` |
| `MO-2025` | Bouillaguet--Delaplace--Hamdad, *The May--Ozerov Algorithm for Syndrome Decoding is “Galactic”*, CiC 2(1), 2025, DOI `10.62056/akjbksuc2` | Concrete warning against assuming asymptotically faster MO is the best practical generic-ISD row |
| `ART-LOCAL` | `scripts/audit_ringlpn_regular_projection.py`, current SHA-256 `993a37f72a59aed7803068225b5d3108a948f2fe248ae189a1b5e84ac62acc52`; exact-law CSV SHA-256 `3531fa7637e717ba563e469f72e1f798c4740e49470450eaa64cd1157373b0cb` (the former `6ddd1bf5...` transcript is superseded history); corrected estimator-sensitivity CSV SHA-256 `ffd335a7d9f7670073b611f390380aa44974f9501b33b2e12504f669e757a5db` | Exact projection distribution plus guarded random-code model diagnostics; never a security pin |

## 4. Attack inventory and execution semantics

Labels used below:

- **PUBLISHED REDUCTION/THEOREM** means only the theorem and scope named in the row, not a Ring-LPN reduction unless the row says so.
- **EXACT LOCAL CALCULATION** means an exact distribution/data fact, not hardness.
- **MODEL ESTIMATE** means a cost under unproved applicability assumptions.
- **REVIEW ITEM** means the source exposes an attack family but no defensible deployed-input cost has been produced.
- `M` is the number of distinct orbit syndromes. `M=n` at full degree and `M=d` after degree-`d` projection except on the explicitly bounded stabilizer event in §5.

| Attack and source pin | Label; exact distribution/model and input | Orbit treatment | Cost tool and executable status | Assumptions still charged | Memory, data and success semantics | Current disposition / blocker |
|---|---|---|---|---|---|---|
| Sparse one-factor projection (`BCG`, `FF-2024`, `ART-LOCAL`) | **EXACT LOCAL CALCULATION** for the projection law; **MODEL ESTIMATE** for each attack call. Input is each factor degree `d|n`, exact realized/tail weight `h`, and `(N,k,w,q)=(c*d,(c-1)*d,h,p)` separately for `p0,p1`. Projected noise is occupancy/cancellation, never RSD. | Orbit exists with `M=d/|Stab(y_d)|`; `sqrt(d)` is a separately labelled sensitivity only. Shifts preserve realized weight, not independence. | Exact law/lower-tail calculator is executable in `ART-LOCAL`. Guarded `FF-2024` calls are executable only for `h<=d-1`. | No reduction from dependent projected noise and structured code to the estimator's random-code model; no theorem making `h>=d` harmless or uniform. | Exact-law CSV records integer probability denominators/lower tails, not attack RAM. Estimator outputs log-work only and do not normalize RAM, preprocessing, sample count, or success probability. Data are one public syndrome plus implicit orbit. | Required attack composition, not a pin. Dense regime and structured-code bridge remain blockers. |
| Direct regular ISD (`RISD-2024`, `ART-RISD`) | **MODEL ESTIMATE.** Exact full input `(c*n,(c-1)*n,c*t,p)` with `c*t` blocks of width `n/t` and iid uniform `F_p^*` values; all five `n=2^20` candidates and both deployed primes are recorded. | Raw results are retained. `0.5*log2(n)` is a separate `heuristic_sensitivity_only` column, never silently subtracted. Shifted bucket partitions still require decoder support. | `ART-RISD` reproducibly executes the accepted Perm/Enum/Rep/RepD2/CCJ and CCJ-linear formulas. Some direct CCJ rows retain the artifact's floating-point overflow as incompatibility rows rather than reformulating it. Generic BJMM is fail-closed because the immutable notebook delegates to an unpinned binary `CryptographicEstimators`/Sage dependency and accepts no `q`; no BJMM cost is emitted. Projected `d=64,h=63` rows explicitly reject regular-ISD and the delegated generic path as incompatible. | Artifact rank/list/independence and source field model; transfer to the structured negacyclic matrix and q-ary iid payloads remains unproved. Formula cost units are not a reviewed bit-operation model. | CSV reports the artifact expected-work/success term where present, its log2 storage proxy with the notebook's unspecified unit, one public sample, no materialized orbit, exact dependency status, and warnings. These normalized semantics prevent treating a raw cost as concrete security. | Executable direct diagnostic now exists, but its model assumptions, unspecified memory unit, CCJ overflow rows and generic-BJMM dependency incompatibility remain blockers. No row is a pin. |
| Hybrid regular-SD (`HYBRID-2025`, `ART-HYBRID`) | **MODEL ESTIMATE.** Exact direct RSD input `(N,K,h,beta,p)=(c*n,(c-1)*n,c*t,n/t,p)` for all five candidates and both live primes, one uniform nonzero per block with iid uniform `F_p^*` payloads. | Baseline rows apply no orbit adjustment. Separate rows subtract `0.5*log2(n)` only as `sqrt_full_orbit_heuristic_sensitivity`; no published hybrid-decoder composition was found. | `ART-HYBRID` is an executable calculator for Theorem 1, equations (9)--(13), from the pinned 2025-09-07 archived PDF, with exhaustive admissible integer optimization. It is not an executable attack implementation and no author artifact was located. | Theorem's full-rank Macaulay heuristic, random RSD matrix model, classical field-operation model with `omega=2.8`, and transfer to the structured negacyclic multiplication-matrix ensemble. The archive landing page reports a later 2025-09-09 revision, whose formula delta must be checked rather than assumed absent. | Time is classical expected `F_p` operations and includes expected `1/P` independent puncturing iterations. Memory is log2 of the theorem's big-O expression in field elements, not bytes or a concrete bound. Data are one `H` and syndrome; acquisition/materialization are unpriced. Success and both resource semantics are explicit in each CSV row. | Reproducible pinned-revision formula diagnostic now exists. Later-revision reconciliation, attack implementation, concrete bytes/bit operations, structured-code reduction, orbit composition and independent review remain blockers; no row is a pin. |
| Generic finite-field ISD: BJMM/MMT and simpler variants (`FF-2024`, `MO-2025`) | **MODEL ESTIMATE.** Full-degree arbitrary fixed weight `(c*n,(c-1)*n,c*t,p)` and every valid projected `(c*d,(c-1)*d,h,p)`. It ignores regular block structure unless a separately reviewed RSD routine is used. | No automatic orbit subtraction. `0.5*log2(M)` is only a sensitivity for a compatible one-out-of-many decoder. | `SD_ISD_q` in `FF-2024` is executable. `MO-2025` shows why asymptotic MO superiority is not a concrete-cost default; it is not a deployed calculator. | Random linear code, ranks, list independence and cost model; projected distribution bridge; decoder-specific DOOM support. | Accepted estimator emits log-work, not a reproducible memory/data/success ledger. Each exported row must add list RAM, table representation, preprocessing, samples and repetition semantics. | Generic upper-bound candidate only. Not concrete Ring-LPN security evidence. |
| Pooled Gauss / linear-system guessing (`FF-2024`) | **MODEL ESTIMATE.** Generic fixed-weight random-code input at the direct tuple `(c*n,(c-1)*n,c*t,p)` and mechanically valid projected tuples `(c*d,(c-1)*d,h,p)`. The routine does not use regular bucket metadata, projected dependencies or ring structure. | No cyclic-orbit composition is implemented or justified; retain the raw result and treat any one-out-of-many adjustment as a separate decoder-specific sensitivity. | `Gauss(N,k,t)` is executable inside the pinned artifact and is included by `SD_ISD_q`/`analysisforq`; it has no `q` argument and must not be mistaken for a prime-specific bit-cost implementation. | Random-code rank/guessing model, field-operation interpretation, projected-law bridge and structured-matrix transfer. | The routine folds a per-iteration success guess into an expected-work expression but the aggregate does not export normalized peak RAM, field-element/byte units, input acquisition, target success or repetition confidence. Data are one public syndrome unless a separately reviewed decoder says otherwise. | Already present inside the 2024 aggregate; required to name in breakdowns so the minimum is not mislabelled “ISD.” It remains model-only evidence. |
| QC/negacyclic one-out-of-many decoding (`SENDRIER`, `QA-BASE`, §5) | Orbit/data multiplicity is **PROVED HERE**. Runtime transfer is a **MODEL SENSITIVITY** outside Sendrier's concrete Stern scope. Inputs are one full or projected syndrome and a decoder accepting one of `M` same-code, same-weight targets. | Full: `M=n/|Stab(y)|`; projection: `M=d/|Stab(y_d)|`. Section 5 gives the exact upper bound on a non-full orbit event; do not replace it by an unquantified “overwhelming” claim and do not use `(c-1)d`. | Sendrier's Stern variant is the only reviewed executable-algorithm scope identified. No local general large-field one-out-of-many decoder is present. | A blanket `sqrt(M)` transfer to arbitrary ISD/RSD/statistical/algebraic decoders is unproved. Regular decoders must support the translated public partitions. | No extra oracle samples and no need to materialize all syndromes: shifts are computed on demand. “Success” means recovering any shifted error and undoing the public shift. A concrete row must include orbit-enumeration overhead and peak memory, not only subtract bits. | Orbit existence may be cited; generic square-root speedup may not be cited as a reviewed cost. |
| Statistical decoding / low-weight parity checks (`FF-2024`, `BCG`) | **MODEL ESTIMATE.** Direct exact weight and valid projected `h`; generic fixed-weight/random-code dual model. It is primarily decisional rather than recovery. | No proved extra cyclic gain found. Any structured dual-word precomputation or one-out-of-many composition must be costed, not assumed. | `SDforq` in `FF-2024` is executable. BCG's adapted projection lower-bound expression is analytical, not a live calculator. | Existence and search cost of sufficiently low-weight dual words for the structured ensemble; projected-law bridge; independence of checks. | Must separate dual-word precomputation, stored check count/RAM, sample/syndrome count, distinguishing advantage, false-positive/false-negative target and repetition. The accepted aggregate does not supply this complete ledger. | Required generic candidate; dense guarded calls remain undefined. |
| Briaud--Øygarden algebraic RSD (`AGB-2023`, `FF-2024`) | **MODEL ESTIMATE.** Direct RSD input `(c*n,(c-1)*n,c*t,p)`. `analysisforqregular` already takes the minimum of `AGBforq` and generic finite-field analysis. Projected noise is not RSD and must not call AGB by analogy. | No published composition with the negacyclic orbit was found; no subtraction. | `AGBforq` in the pinned accepted artifact is executable. | Semi-regular/random polynomial-system behavior and algebraic-complexity model; structured multiplication matrices may change regularity. | Output is estimated algebraic work. A valid row must add matrix/polynomial memory, field-operation/bit-cost convention, success probability and retries. | Already included in 2024 aggregate; do not double count it as a Schur-square attack. |
| Schur/componentwise-product structural decoding (`BCG`, `QA-BASE`) | **REVIEW ITEM.** Exact target is the public negacyclic multiplication-matrix code at full and projected degrees. A square-dimension experiment is only a distinguisher diagnostic. | Orbit does not resolve whether the square code has exploitable rank. No speedup is assigned. | No validated attack-cost tool exists for this ensemble. | BCG's statement that pairwise products span the whole ambient space with overwhelming probability is informal and unproved; efficient algebraic decoding of random quasi-Abelian/cyclic codes remains an open problem in `QA-BASE`. | Any future experiment must report matrix field, sampled public matrices, rank convention, trials/confidence, RAM and whether it merely distinguishes or actually recovers noise. | A reviewed Schur-rank theorem or executable decoder remains a blocker; random-code behavior cannot be assumed. |
| QA-SD compressed sensing (`QA-CS-2025`) | **REVIEW ITEM.** The negacyclic code is monomially equivalent to a cyclic one-variable QA form, but the published attacks target small fields and sparse multivariate interpolation/random evaluations. Exact live inputs would be one variable, `p≈2^62`, `(c,t,n)` or `(c,t,d)`, and prime-specific projected cancellation. | Uses evaluation/interpolation structure, not a generic `sqrt(M)` adjustment. No orbit subtraction. | The source reports practical small-field implementations; no reviewed large-prime/univariate deployed calculator is present. | Transfer of complex/convex methods, numerical stability and sample scaling to the large prime/univariate regime. | Published headline includes distinguishing advantage (about 60% for F4OLEage) and hours-scale examples, but those are not live inputs. A live row must record evaluations/data, precision, RAM, recovery vs distinguishing success and confidence. | Must be explicitly dispositioned; neither “breaks live tuple” nor “inapplicable” is established. |
| QA-SD correlation (`QA-CORR-2026`) | **REVIEW ITEM.** Same structural problem-form transfer and exact live inputs as the preceding row. Published benchmarks/analysis are over `F_3/F_4`. | Correlation/evaluation attack; no generic orbit subtraction. | No large-prime/univariate deployed executable cost was found. The author's `~1000x` time/RAM comparison over `F_3` is not a live-row multiplier. | Large-prime correlation magnitude, sample complexity, implementation precision, and projected-noise transfer are unresolved. | Must report data/evaluations, peak RAM, distinguishing or recovery probability, confidence and repetitions. Current live semantics are absent. | New 2026 mandatory review item; parameter pin is blocked until dispositioned. |
| Stationary syndrome decoding attacks (`SSD-2025`) | **PUBLISHED MODEL/THEOREMS for SSD, not this one-sample instance.** SSD needs several correlated noise vectors on the same unknown support. With one vector it collapses to ordinary RSD. | No SSD-specific orbit gain assigned. Ordinary orbit analysis applies only after returning to the one-sample RSD instance. | Paper analysis exists; no live SSD calculator is needed while support reuse is absent. | Must verify across direction, limb, layer, batch and epoch that no hidden support is reused with fresh payloads. | SSD data semantics require multiple correlated syndromes. Current disposition assumes one sample per support; if that invariant changes, record correlated-sample count, amortized memory/work and joint success. | Not an SSD-specific attack today. Support-reuse audit remains a continuous proof/source obligation. |
| Sparse-public-equation spectral/Kikuchi (`SPARSE-SPEC-2026`) | **PUBLISHED ATTACK, NOT APPLICABLE AS STATED.** It requires `k`-sparse public coefficient rows/equations. Live negacyclic multiplication matrices are dense; the error is sparse. | No orbit effect established. | Paper formulas are not a live calculator. | A dense-negacyclic-syndrome to sparse-row reduction was not found. | Published semantics trade samples against spectral/Kikuchi time. The live system does not supply the required sparse-row sample population. | Record as reviewed/nonmatching, not silently omit and not claim a break. Reopen if public equations become sparse. |
| Sparse LWE/LPN with small secrets (`SPARSE-SECRET-2026`) | **PUBLISHED ATTACK, NOT APPLICABLE AS STATED.** Distinct from the spectral paper: it assumes a sparse coefficient matrix and bounded small secret. The live matrix is dense and the sparse error has uniform nonzero field values, not a small bounded secret. | No orbit effect established. | No live calculator. | No model reduction to the deployed syndrome instance. | Published sample/runtime tradeoff and walk semantics do not map to one dense structured syndrome; no live success/RAM row exists. | Keep as a separate 2026 negative disposition. Do not conflate it with `SPARSE-SPEC-2026`. |

## 5. Formal negacyclic-to-cyclic orbit lemma

### 5.1 Statement

Let `p` be an odd prime, `n` a power of two, `2n | (p-1)`, and

```text
R^- = F_p[X]/(X^n+1),     R^+ = F_p[Y]/(Y^n-1).
```

Let

```text
H = [M_(a_1) | ... | M_(a_(c-1)) | I]
```

with independent uniform `a_i in R^-`. Each error polynomial contains one uniform position in every public bucket and independent uniform `F_p^*` payloads. For `y=He`:

1. there is a weight- and support-preserving diagonal algebra isomorphism from the negacyclic instance to a cyclic instance;
2. a single syndrome yields an explicitly computable orbit of `M=n/|Stab(y)|` distinct, same-code, same-weight syndromes;
3. except with probability at most

```text
(n/(p-1))^(c-1) + (n-1)*p^(-n/2),
```

that orbit is full (`M=n`);
4. for every fully split one-sparse degree-`d` projection, the same statements hold with `n` replaced by `d`.

The probability is over the public multipliers and noise. The statement proves orbit/data multiplicity, not a decoder running-time gain.

### 5.2 Diagonal twist and support preservation

Choose `alpha in F_p` of order `2n`. Then `alpha^n=-1`. Define

```text
phi: R^- -> R^+,       phi(f)(Y)=f(alpha*Y).
```

Because `(alpha Y)^n+1 = 1-Y^n`, the map is a well-defined `F_p`-algebra isomorphism. In coefficient coordinates it is

```text
diag(1, alpha, alpha^2, ..., alpha^(n-1)).
```

Every diagonal entry is nonzero. Therefore `phi` preserves zero support and Hamming weight exactly. It also preserves the public bucket membership metadata and the independence/uniformity of `F_p^*` payloads: multiplication by a fixed nonzero scalar permutes `F_p^*`.

Applying `phi` blockwise converts the parity check into a cyclic/quasi-Abelian parity check without replacing the live distribution by a random code.

### 5.3 Shift correspondence and orbit

The maps satisfy

```text
phi(X*f)=alpha*Y*phi(f).
```

Thus multiplication by `Y^s` in cyclic coordinates corresponds to `alpha^(-s) X^s` in negacyclic coordinates: a signed negacyclic shift and one global nonzero scalar. Since all ring multipliers commute with `Y^s`,

```text
Y^s y = H (Y^s e).
```

Consequently one public syndrome supplies

```text
Orb(y) = {Y^s y : 0 <= s < n}
```

with `n/|Stab(y)|` distinct same-code targets. No extra LPN query or public sample is needed and an implementation can generate shifts on demand rather than materializing `n` vectors. At full degree the regular bucket partition is translated publicly by the same shift; a regular decoder must accept or explicitly permute this translated partition. At a projection, the shift preserves the realized Hamming weight but does not turn the occupancy/cancellation law into RSD.

Recovering any shifted error solves the original instance after applying the inverse public shift and twist.

### 5.4 Stabilizer bound

For any fixed split root `rho` and `t>=2`, condition on all but one independent nonzero payload in an error polynomial. The remaining term is uniform over `F_p^*` times a fixed nonzero scalar. It equals the unique value cancelling the conditioned sum with probability at most `1/(p-1)`; if the conditioned sum is zero, cancellation is impossible. For `t=1`, the evaluation cannot be zero. A union bound over the `n` split roots gives

```text
Pr[e_i is not a unit in R^-] <= n/(p-1).
```

The first `c-1` error polynomials are independent, so the probability that all are nonunits is at most `(n/(p-1))^(c-1)`. If some `e_i` is a unit, then multiplication by it is a bijection and uniform `a_i` makes `a_i e_i`, hence `y` after conditioning on the other summands, uniform in `R^-`.

For uniform `phi(y) in R^+`, a nonidentity shift `Y^s` fixes a subspace of dimension `gcd(n,s)`. Since `n` is a power of two and `0<s<n`, `gcd(n,s)<=n/2`. A union bound yields

```text
Pr[Stab(y) is nontrivial]
 <= (n/(p-1))^(c-1)
    + sum_(s=1)^(n-1) p^(-(n-gcd(n,s)))
 <= (n/(p-1))^(c-1) + (n-1)*p^(-n/2).
```

This is an explicit parameter-dependent full-orbit failure bound. It is not a hardness reduction, and it must be evaluated rather than replaced by an unquantified “overwhelming probability” claim.

### 5.5 Every one-sparse projection

For a fully split factor `f_d(X)=X^d+c_d`, choose `beta in F_p` with `beta^d=-c_d`. Substitution `f(X) -> f(beta Y)` gives the same diagonal twist into `F_p[Y]/(Y^d-1)`. Repeat §§5.2--5.4 with `n` replaced by `d`. The projected error may have collisions and cancellations, but its root evaluation is still a sum of independent uniform-nonzero payloads times fixed nonzero scalars, so the unit bound applies. The orbit is `d/|Stab(y_d)|`; a full degree-`d` orbit has `d` elements, not `(c-1)d`.

### 5.6 What the lemma does not prove

Sendrier proves an almost-`sqrt(M)` gain for a Stern collision-decoding variant and parameter range. `QA-BASE` and BCG use broader conservative estimation language. This lemma does not extend Sendrier's running-time analysis to:

- arbitrary large-field ISD or regular-ISD implementations;
- the hybrid RSD algorithm;
- statistical decoding or dual-word precomputation;
- algebraic/Gröbner attacks;
- QA-SD interpolation/correlation attacks; or
- a decoder with memory, setup, success-probability or data costs that do not scale as assumed.

Therefore `0.5*log2(M)` is a sensitivity until the exact decoder, resource model and success experiment are reviewed.

## 6. Current-tool correction and candidate impact

An earlier local implementation used

```text
doom_loss(c,d) = 0.5*log2((c-1)*d).
```

That counted code-tail dimension rather than the orbit. It over-subtracted `0.5*log2(c-1)` and even assigned a nonzero orbit gain at `d=1` when `c>2`.

The current source-pinned `ART-LOCAL` implementation is corrected to

```text
doom_loss(d) = 0.5*log2(d).
```

It labels the orbit formal and the square-root decoder transfer `heuristic_diagnostic_only`. The corrected CSV is the `ffd335a7...` artifact in §3. The rejected pre-correction `c1b9cb53...` CSV is history and must not be cited.

All five default diagnostic candidates have `n=2^20`:

| candidate | correct full-orbit sensitivity | rejected old loss | rejected over-subtraction |
|---|---:|---:|---:|
| `n20_c4_t16` | 10.000000 bits | 10.792481 bits | 0.792481 bit |
| `n20_c4_t32` | 10.000000 bits | 10.792481 bits | 0.792481 bit |
| `n20_c4_t64` | 10.000000 bits | 10.792481 bits | 0.792481 bit |
| `n20_c8_t8` | 10.000000 bits | 11.403677 bits | 1.403677 bits |
| `n20_c8_t16` | 10.000000 bits | 11.403677 bits | 1.403677 bits |

At `d=2^j`, the full projected-orbit sensitivity is `j/2` bits. At `d=1` it is zero. These are sensitivity values, not security levels or validated attack costs.

## 7. Reduction versus estimate ledger

| Claim | Classification | Exact boundary |
|---|---|---|
| Liu--Wang--Yang--Yu Theorem 2 | **PUBLISHED REDUCTION/THEOREM** | Random-matrix exact finite-field LPN to random-matrix regular LPN with its stated dimension and advantage loss; does not cover ring multiplication matrices or projected dependent noise |
| Sendrier one-out-of-many result | **PUBLISHED ATTACK THEOREM** | Almost-square-root improvement for a Stern collision-decoding variant in its stated range; not a blanket decoder theorem |
| Diagonal twist, orbit and stabilizer bound in §5 | **PROVED IN THIS INTERNAL NOTE** | Exact algebra/data fact under the stated split-prime and sampler hypotheses; requires independent human review before changing a gate |
| Exact projected occupancy/cancellation/lower tails | **EXACT LOCAL CALCULATION** | Exact distribution fact checked by `ART-LOCAL`; not hardness or a code-model reduction |
| `FF-2024`, `RISD-2024`, AGB, generic ISD/statistical outputs on the live structured code | **MODEL ESTIMATE** | Random-code, rank/list, semi-regularity and distribution-bridge assumptions remain charged |
| Schur full-square behavior | **UNRESOLVED** | BCG's informal statement is not a rank theorem for the deployed ensemble |
| `0.5*log2(n)` or `0.5*log2(d)` against an arbitrary live decoder | **HEURISTIC SENSITIVITY** | Orbit is formal; runtime, memory, data and success scaling are not |
| QA-SD 2025/2026 transfer to `p≈2^62` univariate live parameters | **UNRESOLVED REVIEW ITEM** | Small-field published attacks cannot be numerically transferred without a source-supported analysis |
| 2026 sparse-equation/small-secret attacks on the live matrix | **NONMATCHING AS STATED** | Both require sparse public coefficients; the live multiplication matrices are dense. No dense-to-sparse reduction was found |

## 8. Explicit blockers before any parameter pin

1. **Direct modern RSD attacks:** independently review `ART-RISD` and `ART-HYBRID`. For `ART-RISD`, resolve the CCJ overflow/cost-unit semantics and pin or replace the delegated generic-BJMM dependency without pretending a current binary reproduces the archive. For `ART-HYBRID`, reconcile the pinned 2025-09-07 PDF formulas with the archive's 2025-09-09 revision, validate the theorem transcription/optimizer, and supply an attack implementation or independently justified concrete execution model; convert field-operation and big-O field-element expressions to reviewed bit/byte costs. For both, retain raw baseline costs, decoder-specific orbit sensitivities, data and success/repetition semantics separately.
2. **Projected bridge:** prove or attack the dependent prime-specific occupancy/cancellation distribution for every useful factor. Expected weight and a random-code analogy are insufficient.
3. **Dense projection:** resolve `h>=d`, where the accepted aggregate is outside its combinatorial domain. Do not interpret an undefined call, estimator guard, or dense support as security.
4. **Structured matrix:** supply a reduction or ensemble-specific attack analysis for ranks/lists, dual checks, AGB semi-regularity, and Schur-square behavior.
5. **Decoder-specific orbit composition:** review a concrete implementation before applying any `sqrt(M)` adjustment; account for shifted regular partitions, preprocessing, RAM, data and success.
6. **QA-SD 2025/2026:** obtain an explicit large-prime/univariate disposition. “No published live cost” is not evidence of inapplicability.
7. **Support reuse:** audit direction, limb, batch, layer and epoch freshness. Any reuse changes the problem to SSD and reopens its correlated attacks.
8. **Multi-instance advantage:** compose both CRT limbs, both directions, every factor, ring batch, layer, epoch, DPF/PRG/OT/OLE hybrid, conversion and sampler bad event. Separate classical and quantum scopes.
9. **Source-pinned resources:** every attack row must retain exact source revision/checksum, exact inputs, distribution/model, orbit treatment, assumptions, peak/total memory, data, preprocessing, success and executable command/output provenance.
10. **Independent human review:** the orbit proof, attack-model bridges, cost transcriptions and final advantage budget require independent cryptographic review. Model-assisted review does not close this gate.

Until all blockers close, diagnostics may rank engineering candidates but cannot select, advertise or benchmark a “secure” tuple. No concrete-security claim is unlocked by this audit.
