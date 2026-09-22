# Structured Ring-LPN attack audit

**Date:** 2026-08-04; live source and hybrid-formula bindings refreshed 2026-08-10; ideal-leaf arithmetic and conditional lifetime supplement 2026-09-22
**Status:** internal/advisor; attack inventory and proof obligation ledger; **not a parameter pin or concrete-security review**
**Scope:** the current audited regular sampler revision, its direct expanded instance, every one-sparse fully split projection, and the implemented DPF map's ideal-leaf statistical loss
**Review state:** source-grounded model/attack triage, an elementary orbit lemma, and exact ideal-leaf arithmetic; independent human cryptographic review is required

## 1. Decision and non-claim

No deployed or candidate tuple has a source-supported concrete-security level. In particular:

- q64 and q128 mean one and two approximately 62-bit arithmetic limbs. They do not mean 64- or 128-bit security.
- A number printed by the accepted EUROCRYPT-2024 finite-field estimator is a random-code/model attack estimate. The estimator has no ring, factor, orbit, projection-law, memory-limit, multi-instance, or CRT-advantage input.
- The existence of a cyclic orbit is proved in §5. It proves that one public syndrome supplies related same-code decoding instances. It does **not** prove a square-root running-time gain for an arbitrary large-field ISD, RSD, algebraic, or statistical decoder.
- The direct expanded noise is standard regular syndrome-decoding noise before projection. A projected noise vector follows the exact occupancy-and-cancellation law, not regular syndrome decoding.
- The 2025/2026 quasi-Abelian attacks have no published cost transfer to the deployed large prime, univariate dimensions. That transfer is unresolved, not disproved.
- Candidate rows and orbit-adjusted rows are diagnostics only until a structured-code reduction or attack theorem and an independent human review both close.

A reader must not cite this report, any generic-estimator CSV, or the orbit lemma as “reviewed concrete Ring-LPN security.”

## 2. Exact dated audited instance

The source pin `SRC-AUDITED-2026-08-10` below samples, independently for every one of the `c` error polynomials and each of its `t` public contiguous buckets,

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
| `SRC-AUDITED-2026-08-10` | `src/two_party_spfss.h`, SHA-256 `fbdb56f84b1da9db63dad8b3464217fb05d21729344ae9ef88054b0677428461`, especially `validate_party_noise` and `sample_party_noise` | Exact current bucket/payload distribution. A semantic diff from audited revision `5ab544996925ad57b0cb422b67ca08c7cf89accf` to current source revision `86b1323ce25792ca1158abc1c57c3051dec1a9ef` changes only the optional Phase-C OLE-source argument and forwarding in the CPU baseline; both sampling functions are unchanged. |
| `BCG` | Boyle--Couteau--Gilboa--Ishai--Kohl--Scholl, corrected full version dated 2022-08-10, HAL `hal-03374154v1` / ePrint 2022/1035, §§8.2--8.4 and 9.1 | Ring-LPN projection, algebraic-code and quasi-cyclic/DOOM discussion; its informal full-square statement is not a theorem |
| `FF-2024` | Liu--Wang--Yang--Yu, EUROCRYPT 2024, DOI `10.1007/978-3-031-58751-1_6`; accepted artifact `eurocrypt-2024-a1`, immutable downloaded script SHA-256 `c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc` | Random-code finite-field exact/regular LPN estimator; includes pooled Gauss, statistical decoding, generic finite-field ISD, and AGB |
| `RISD-2024` | Esser--Santini, CRYPTO 2024, DOI `10.1007/978-3-031-68391-6_6`; accepted artifact `crypto-2024-a1`, published 2024-08-15, ZIP SHA-256 `04ae2586fccb10481efb861104176e4aaabb380c3cb9704b97ce3c4768a282cb`; upstream snapshot commit `afe1e408f8a46aebc15293462480f478ff969923` | Permutation, enumeration, representation, depth-2 representation, CCJ and generic BJMM regular-ISD costs |
| `ART-RISD` | `scripts/audit_regular_isd_crypto2024.py`, SHA-256 `b0864d27f03d76dd3f0bd660d33c71063ad0380498d1e07dcddc1e2f4907eff9`; `results/security/regular_isd_crypto2024_2026_08_04.csv`, 50 rows, SHA-256 `68b8329dc77d992a90257b2b6b808fc1076534305e0ec0c434831ddafb17d255`, embedded `analysis_sha256` `39159736d43e954c565645c76e0cbe1ac433e92ba1f2dcc8f2ab847af8f89dfc`; notebook member SHA-256 `cebb0861f1faa53be59eb4c11a2e38219612e1fc8d39e6cb3ce597e28717c9ec` | Executable stdlib transcription of pinned Perm/Enum/Rep/RepD2/CCJ formulas on all five direct candidates and both primes; explicit fail-closed incompatibility rows for delegated generic BJMM and projected non-RSD |
| `AGB-2023` | Briaud--Øygarden, EUROCRYPT 2023 / ePrint 2023/176; implementation used here is the `AGBforq` function frozen inside `FF-2024` | Algebraic RSD estimate under polynomial-system assumptions |
| `QA-BASE` | Bombar--Couteau--Couvreur--Ducros, ePrint 2023/845 full version, especially §§6.4 and 6.6 | Quasi-Abelian structural-code boundary and generalized orbit sensitivity statement |
| `SENDRIER` | Sendrier, PQCrypto 2011, DOI `10.1007/978-3-642-25405-5_4` | Executable-source scope: almost-square-root gain for a Stern collision-decoding variant in the stated McEliece range |
| `HYBRID-2025` | Wang--Wang--Yang--Liu--Yu--Zhang--Wang, ePrint 2025/1284 / ASIACRYPT 2025; archive revision published 2025-07-14, modified 2025-09-09, retrieved 2026-08-04 | New hybrid RSD algorithm replacing ISD meet-in-the-middle enumeration by quadratic-equation solving |
| `ART-HYBRID` | `scripts/audit_hybrid_rsd_asiacrypt2025.py` SHA-256 `cbcedaf6cdb1e6818aa968ab43c386f1b046b8640dd3d257039d462e7e949764`; regenerated 20-row CSV SHA-256 `1f671d941189180397479f2212ba6445fa218f7b2f60f55301bcbca4066a5f25`, with the same embedded script digest | Self-test reproduces the published 132.60 and 133.15 table rows. These are formula diagnostics, not an attack implementation, quantum cost, structured-code reduction, or security pin. |
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

## 9. P-KEY: exact ideal-leaf arithmetic, not a DPF security certificate

**Classification: EXACT LOCAL CALCULATION, conditional on the ideal-leaf
experiment below.** This supplement sharpens the September 11 `P-01` bound;
it changes neither the leaf map nor the independent-human-review gate.
`P-KEY` remains open. The previous `epsilon` is a valid single-law upper
bound and an exact disjoint-payload-shift gap, not the exact single-law TV.

### 9.1 Source binding and probability experiment

The executable companion is `scripts/audit_dpf_leaf_loss.py`. It fails closed
unless SHA-256 pins match all eight reviewed source files, extracts both primes
from `two_party_dpf_protocol.h`, and emits source/script hashes with its JSON.
Its source bindings cover:

- `src/two_party_dpf_protocol.h`: `kPrime62`, `kPrime62Crt2`, and
  `convert_zp`, which reduces each 64-bit half separately and adds modulo `p`;
- `src/gpu_spfss_zp.cuh`: the same `convert_zp`, `block_lo`/`block_hi`,
  and separate full-width child-seed/control-bit AES outputs;
- `src/spfss_host.cpp`: the host reference conversion, final correction
  `beta-c0+c1` when `t0=1,t1=0`, and signed evaluation;
- `src/two_party_dpf_gpu.cuh`: live `sum_party_leaves_kernel` calls that GPU
  conversion on every leaf of its local frontier;
- `src/ringlpn_ole_party.cuh`: `(c*t)^2` trees per Ring-OLE instance;
- `src/two_party_linear_preprocess.cuh`: aggregation and the expected
  `2*limbs*ring_batches` Ring-OLE instances per linear-layer invocation.
- `src/two_party_spfss.h`: regular/uniform domains, Cartesian-product tree
  layout, local noise-factor reuse, and frontier limits;
- `src/gpu_aes_prg_host.h`: four distinct AES input blocks under each node seed.

Hashes bind the reviewed source, not a compiled binary or a proof of its
execution. The arithmetic below treats `lo,hi` as independent uniform integers
in `[0,H-1]`, `H=2^64`. This is an idealized input law, **not an established
conditional law for live AES-derived seeds, related leaves, or a key holder's
view**. SplitMix host reference semantics are not a security assumption.

### 9.2 Exact distance of one mapped leaf from uniform

Write `H=k*p+r`, `0<=r<p`. Each half-residue has `k+[v<r]` preimages.
For `Q=Law((lo mod p + hi mod p) mod p)`,

```text
H^2 Q(v) = p*k^2 + 2*k*r + T(v),
T(v) = #{(a,b) in [0,r-1]^2 : a+b = v mod p},
sum_v T(v) = r^2,
Q = (1-epsilon) U_p + epsilon D_r,   epsilon = r^2/H^2.
```

Here `D_r=T/r^2` for `r>0`; for `r=0`, `Q=U_p` exactly. At both deployed
primes `2r-2<p`, so `T` is the nonwrapping triangle
`T(v)=max(0,min(v+1,2r-1-v,r))`. No enumeration of the field is needed.
Define `j=floor(r^2/p)`. Since `Q(v)>1/p` exactly when `T(v)>r^2/p`,
the positive set contains `2(r-j)-1` residues and its triangular mass is
`r^2-j(j+1)`. Therefore the **exact** total-variation distance is

```text
delta = TV(Q,U_p)
      = [p*(r^2-j*(j+1)) - (2*(r-j)-1)*r^2] / (p*H^2).
```

For both deployed primes `k=4` and `r^2<p`, hence `j=0`, giving:

| limb | deployed `p` | `r=2^64 mod p` | exact `TV(Q,U_p)` | exact disjoint-shift gap / single-law upper bound |
|---|---:|---:|---|---|
| p0 | 4611686018326724609 | 402653180 | `402653180^2*(4611686018326724609-805306359)/(4611686018326724609*2^128)` | `402653180^2/2^128` |
| p1 | 4611686018309947393 | 469762044 | `469762044^2*(4611686018309947393-939524087)/(4611686018309947393*2^128)` | `469762044^2/2^128` |

Equivalently, `delta=epsilon*(1-(2r-1)/p)<epsilon`. These are exact rational
expressions, not floating-point estimates. The calculator emits reduced
numerator/denominator pairs; any displayed logarithms are approximations,
not security-bit certifications.

### 9.3 Two payloads, a conditioned tag, and a joint event are different

Fix a known common `alpha`, a payload `beta`, and `s=floor(p/2)`. Put
`S=[0,2r-2]`. At both deployed primes, `S` and `S+s` are disjoint modulo `p`.
Uniform mass cancels in a translation comparison, so

```text
TV(Q, Q+s) = epsilon*TV(D_r, D_r+s) = epsilon.
Pr[beta+Q in beta+S] - Pr[beta+s+Q in beta+S] = epsilon.
```

For an arbitrary shift, `TV(Q,Q+s)<=min(epsilon,2*delta)` by the mixture
representation and triangle inequality; equality to `epsilon` here is
justified by the disjoint supports, not by treating the single-law distance
as a distinguishing gap. A TV gap is the maximum difference of event
probabilities; in equal-prior binary guessing, success advantage over `1/2`
is half the TV.

For a party-0 DPF key, when its computable on-path tag is `t0=1`, the
correction/evaluation identity gives `Eval0(alpha)=beta+Convert(s1)`.
**If**, in the relevant hybrid, the hidden leaf has the above ideal law
conditional on that tag, the interval predicate has conditional probability
gap `epsilon` between the two payloads. **If additionally** the tag is
independent and fair in both experiments, the predicate
`{t0=1 AND Eval0(alpha) in beta+S}` has unconditioned joint-event gap
`epsilon/2`. A tag of probability `q` instead scales this predicate's gap by
`q`, provided the requisite conditional law holds.

The joint-event gap is a witness, **not** an upper bound on the entire key
view: it does not analyze the tag-0 branch or other key observables.
No empirical AES attack, live-key distinguisher advantage, or full DPF
privacy theorem is claimed.

### 9.4 Finite-comparison budgets and missing reduction

Let `N0,N1` be explicit counts of conditional comparisons in a specified
hybrid over a declared lifetime. Two different arithmetic diagnostics are:

```text
mapped-leaf to uniform replacement:
    B_uniform = min(1, N0*delta0 + N1*delta1);
two-payload translated-leaf comparison:
    B_shift   = min(1, N0*epsilon0 + N1*epsilon1).
```

They describe alternative experiments, not two losses to add automatically.
Their conditional-law hypotheses must hold at every hybrid step, including
conditioning on the adversary's prior view. Under those hypotheses,
telescoping TV and data processing justify the bounds; no independence
between steps is needed. **Uniform-looking marginals alone do not establish
those hypotheses.** In particular, this is not an independent-leaf theorem
for leaves sharing seeds, correction words, keys, or PRG calls. We do not
multiply independent-sample success probabilities, nor insert the joint-tag
witness's factor `1/2` into an alleged full-key upper bound.

Tree/evaluator counters alone do not determine `N0,N1`. Runtime accounting gives
`2*ring_batches*(c*t)^2` tree pairs per active limb per linear invocation,
already including the two Ring-OLE directions. The two parties hold shares
of those same tree pairs; summing their identical counters double-counts
them. GPU frontier summation may convert every leaf, and evaluation can
repeat a conversion without producing a fresh secret or a new hybrid step.
Conversely, a DPF reduction may need multiple primitive/conditional
replacements per tree. None of `trees`, `trees*domain`, number of key files,
or evaluator calls is silently promoted to a proved comparison count.

The legacy command mode requires both counts and a leaf-only diagnostic target `2^-b`.
The reported Boolean says only whether the stated conditional upper bound
fits that target by exact rational comparison. A false result means this
upper-bound certificate misses the target, not a lower bound on the
composed advantage. Exit success means arithmetic/source processing
succeeded, not that any security budget passed.

The subsequent source-specific supplement (§§9.6–9.10) derives a conditional
one-hidden-map-per-tree charge and an explicit workload inventory. It does
not establish the live conditional laws or distributed-view reduction.
Still missing are the AES/DPF hidden-path reduction, distributed transcript
and adaptive Ring-LPN bootstrap lifting, numerical primitive/bad-event
advantages, and independent human review. Exact coupling of distributed and
centralized key generation with the same biased map cannot substitute for
this privacy argument.

### 9.5 Focused reproduction (CPU only)

From `GPU-MPC/`, first run the exact tiny-domain check together
with the two-prime one-comparison diagnostic:

```sh
python3 ringlpn/scripts/audit_dpf_leaf_loss.py \
  --comparisons-p0 1 --comparisons-p1 1 --budget-bits 128 --check-reduced
```

The reduced checker enumerates every pair for `(half_bits,p)` equal to
`(8,61)`, `(8,127)`, `(5,13)`, `(4,13)`, and `(4,16)`. The last is an
arithmetic zero-remainder boundary, not a prime-field instance. It checks
the entire distribution, exact TV, every payload shift's upper bound, and
the disjoint-support interval equality where applicable. This includes
nonzero-threshold and overlapping-support cases. For `(8,61)`, the
half-modulus shift has interval count gap `144/256^2`; its independent
fair-tag joint event has half that probability gap.

For an explicitly hypothetical finite lifetime, not a runtime-derived one:

```sh
python3 ringlpn/scripts/audit_dpf_leaf_loss.py \
  --comparisons-p0 1000000 --comparisons-p1 1000000 --budget-bits 64
```

Use `--comparisons-p1 0` for a one-limb hypothetical scope, not to omit p1
from a two-limb deployment. Counts here are illustrative; no deployed
lifetime or acceptable security target has been approved. Both commands
were executed successfully on September 22, and the five reduced-domain
enumerations agree exactly with the formulas. Neither illustrative bound
meets its requested leaf-only target. Source-bound outputs and the precise
non-claims are retained in `technical_followthrough_2026_09_22.json`.

### 9.6 One key, one role: the frontier cancels; the final word does not

**New result: a conditional whole-key map-replacement lemma, not a live
AES or distributed-protocol privacy theorem.** Fix one corrupted role
`b in {0,1}`, a tree's point `alpha` and payload `beta`, and its modulus.
Write `c_b(x)=Convert(s_b(x))`. The correction recurrence in
`two_party_dpf_protocol.h::apply_level_correction` (and its GPU twin)
preserves the usual DPF invariant:

```text
x != alpha: s0(x)=s1(x), t0(x)=t1(x);
x  = alpha: t0(alpha) XOR t1(alpha)=1.
```

Indeed, at a level all already-off-path pairs have equal seeds/tags and
identical expansions. Their XORs cancel in the two parties' aggregate
left/right values. The remaining pair gives exactly the centralized losing
child correction. The selected child keeps opposite tags; the losing child
acquires equal seeds/tags. This induction also explains why breadth/frontier
keygen and on-path centralized keygen have the same final correction
semantics; it does **not** simulate the messages used to compute it.

Let `S_b` and `T_b` denote the protocol's **signed** seed/control sums.
Every off-path field term cancels between the parties. Consequently:

```text
S0+S1 = c0(alpha)-c1(alpha);
T0+T1 = t0(alpha)-t1(alpha) in {+1,-1};
d0+d1 = beta-(S0+S1);
finalCW = (d0+d1)*(T0+T1)
        = (t0-t1)*(beta-c0+c1).
```

This is exactly the three-product Phase-C code: one multiplication shares
`beta=beta_factor0*beta_factor1`; two more provide the cross terms of
`(d0+d1)*(s0+s1)`. There are `2*D` frontier map calls per tree pair
(`D=2^L`) in key generation, but **one hidden terminal coordinate for a
fixed corrupted role** after this cancellation. There is no need to
uniformize the corrupted party's computable leaves or either party's
off-path leaves to obtain this identity.

Define `V` to include the corrupted party's root, all `L` seed/tag
correction words, its entire computable seed/tag tree, its input, and the
auxiliary history allowed in the experiment, but **not finalCW**. Fix
`alpha,beta` when conditioning, so the corrupted party's `a=c_b(alpha)`
and `t=t_b(alpha)` are functions of `V`. Hypothesis H is that the opposite
party's terminal raw seed, conditional on every such `V` of positive
probability, is uniform in `{0,1}^128`. Equivalently for this lemma it
suffices that its conversion `Z` has the exact law `Q` of §9.2.
The rest of the view must be obtained from `(V,finalCW)` by the same
possibly randomized channel in the two experiments, not by additionally
revealing the replaced raw seed.

For the two roles the entire public final word has the following law:

```text
b=0: finalCW = (2*t-1)*(beta-a+Z);
b=1: finalCW = (1-2*t)*(beta-Z+a).
```

For **both values of t**, these are bijections of `Z` onto `F_p`.
Replacing `Z~Q` by `U_p` therefore changes the joint key `(V,finalCW)`
by exactly `delta=TV(Q,U_p)` under H, not `D*delta`, and not `delta/2`.
The uniform final word is independent of `V`. Any evaluation transcript
has distance at most `delta` by data processing. No fairness assumption
on the corrupted party's tag is needed.

For a fixed `alpha`, two payloads, and an identical prefix law in both
experiments, the final-word distance for any payload shift is at most
`epsilon`; for shift `floor(p/2)` at these primes it is exactly `epsilon`.
This holds in the tag-zero branch as well: although the party's
`Eval_b(alpha)` then omits finalCW, the **key contains finalCW** and the
party can inspect it. The `epsilon/2` event from §9.3 remains a valid
conditioned witness, never a whole-key upper bound.

### 9.7 What establishes H in an ideal path experiment, and what does not

An explicit ideal **programmed hidden-path** experiment supplies H as
follows. At each level replace the *opposite party's on-path parent*
expansion by a fresh independent tuple `(sL,tL,sR,tR)` with uniform
128-bit seeds and fair independent tags. The losing seed one-time-pads
the seed correction word against the known party's losing seed. The two
hidden tags one-time-pad the two tag correction words. The hidden
**keeping** seed is independent of all those outputs; XOR with the
published correction word preserves uniformity. Inductively, the
corrupted key prefix can be generated by its uniform root and independent
uniform correction seeds/tags, without `alpha` or `beta`. At the final
level the keeping seed gives H. This uses `L` hidden expansion sites and
one final map replacement, per tree and per comparison world.

This experiment is not silently identified with a globally consistent
AES/random-function execution. A hidden seed might coincide with a known
seed, a seed from another tree, or an adversarially queried seed. Repeated
seed queries must receive the same expansion, not a new tuple. Corrections
also deliberately make off-path seeds equal; a birthday bound over all
frontier nodes would misclassify these forced equalities as rare.
The missing primitive reduction must show that the programmed path can
be coupled to a consistent expansion oracle except for explicitly
bounded exposure/collision events, including all related-key dependencies.
An `L`-site list is the proposed hybrid's size, **not a theorem charging
`L` ordinary independent-key AES advantages to the live execution**.

The source uses four AES evaluations under the seed-as-key, at plaintext
blocks `0,1,2,3`; it does not use a single fixed-key PRF with independent
domain-tagged tree inputs. In a fresh independent ideal-permutation node
experiment, replacing its four distinct-block outputs by independent
blocks costs at most `binom(4,2)/2^128=6/2^128`. Extracting the tag bits
cannot increase that distance. The calculator reports this switching
term separately for the candidate `L` sites. It does not prove AES is a
PRP under the required auxiliary/related-key distribution, or establish
freshness of the hidden sites.

**Exact counterexample to a marginal-law shortcut.** Let a raw seed `S`
be uniform but include `V=S` in the auxiliary view. Then `Convert(S)` has
the exact marginal `Q`, yet

```text
TV((S,Convert(S)), (S,U_p)) = 1-1/p,
```

not `delta`. The equality event verifies this directly. The same failure
arises in the final-word experiment when both terminal seeds are known.
This is why the conditional H hypothesis cannot be replaced by DRBG
freshness, histogram checks, or an assertion that AES output “looks random.”
The source's distributed transcript is **more** than a centralized key:
it includes local randomness, selected OTs, openings, noise bindings and
prior Ring-OLE correlations. A simulator must condition on the corrupt
party's prescribed input/output and handle all of those dependencies.
The field-cancellation identity alone gives no such simulator.

Repeated or adaptive local evaluations of one fixed key cost **zero new
map replacements**. Give the adversary the key once, fix its private tape,
and generate each query from the preceding answers: the entire transcript
is a function of the same key/tape, so data processing covers every finite
number of adaptive queries. In particular, evaluating the same input
twice must return the same answer. A supposed hybrid that resamples an
independent uniform answer each time breaks this equality with probability
`1-1/p`; it is not a valid same-key hybrid. Adaptive *new key generation*
requires H conditional on prior views at each new key, and is charged
once per newly generated tree. Computation time/query bounds still enter
the AES reduction even though they do not multiply the map term.

### 9.8 Source-bound workload inventory, including the reuse chain

For each declared fresh linear invocation, let `X>0` be its cross-term
count (`M*K*N` for FC; the count of valid, nonpadding scalar products for
Conv2D). The existing source computes:

```text
B=(c*t)^2; R=3*B; A=n-R>0; J=ceil(X/A);
D=2*n/t (regular) or 2*n (uniform); L=log2(D).
```

Every `(batch,direction,limb)` runs one Ring-OLE with `B` new tree pairs,
fresh noise calls and fresh roots. For `I` invocations and `ell` active
limbs, the inventory is:

| quantity | exact count |
|---|---:|
| Ring-OLE instances, all limbs | `2*I*J*ell` |
| distinct tree pairs, **each active limb** | `K_l=2*I*J*B` |
| party key halves, all limbs | `2*sum_l K_l` |
| keygen frontier map calls, both parties | `2*sum_l K_l*D` |
| one complete evaluation's map calls, both parties | `2*sum_l K_l*D` |
| keygen expanded nodes, both parties | `2*sum_l K_l*(D-1)` |
| keygen AES block evaluations, both parties | `8*sum_l K_l*(D-1)` |
| conditional hidden map sites, one role/one world | `sum_l K_l` |
| candidate hidden expansion sites, one role/one world | `sum_l K_l*L` |
| Phase-B 128-bit string OTs, both OT directions | `2*sum_l K_l*L` |
| Phase-C scalar products | `3*sum_l K_l` |
| external epoch-zero scalar products | `I*ell*R` |
| prior-Ring-OLE scalar products consumed | `I*ell*(2*J-1)*R` |
| Ring-OLE tail slots reserved / finally discarded | `2*I*J*ell*R` / `I*ell*R` |
| application slots used / discarded | `2*I*ell*X` / `2*I*ell*(J*A-X)` |

Sum these formulas across workload rows when shapes/parameters differ.
Parties do not create two different tree-pair populations; summing their
identical tree counters double-counts. DPF evaluator launch counts group
many trees and have no direct security meaning. The evaluation row counts
one full materialization, not every possible fallback's AES operations
(root-to-leaf evaluation may repeat ancestor expansions).

There is one bootstrap pool **per limb**, not one per direction.
Within that limb the epochs are ordered `(batch0,direction0)`,
`(batch0,direction1)`, `(batch1,direction0)`, and so on.
Only the first uses external Gilboa products. Each epoch reserves its
last `R` output slots for the next epoch's three products per tree; the
final tail is discarded. The first `A` slots serve application products,
including unused-capacity discards. Thus using one expanded vector for
both bootstrap and application does not generate another DPF key or
another hidden-leaf map draw. It **does** require a joint pseudorandomness
and simulation theorem for the two projections and the adaptive chain.
Charging independent copies of the same vector is invalid.

Within one Ring-OLE, `make_party_spfss_batch` repeats each local noise
position/payload across Cartesian-product pairs `(i,j,k,l)`. The resulting
`B` points/payloads are not independent challenges, even though root
sampling occurs for every tree. The conditional telescoping argument
allows correlated inputs, but only if H and the continuation simulation
hold with this auxiliary information. New scopes per limb/direction/batch
do not prove conditional seed independence. The default `(8192,2,8)` and
any workload printed here remain feasibility examples, not security tuples.

### 9.9 A complete conditional expression, with the unknowns left visible

Fix a static semi-honest corrupted role, finite declared workload and
adversarial resource bound. Consider two admissible secret worlds with
the same allowed public/corrupt-input/output leakage and the same workload.
The following are **hypotheses**, not properties inferred from counters:

1. The DRBG is replaced by the specified independent logical random coins
   with advantage `rng_w` in world `w`.
2. Distributed keygen/transcript simulation reduces the view to the
   single-key experiment with advantage `transcript_w`, conditional on
   the party's prescribed input/output and previous history.
3. A consistent hidden-path AES replacement with the auxiliary information
   of §9.7 costs `aes_w`, ideal PRP/PRF switching costs `switch_w`, and all
   seed exposure/collision/domain failures cost `bad_w`. Outside those
   events H holds at each of the `K_l` replacements. Do not condition
   away a bad event and then assume unchanged uniform laws without proving it.
4. The remaining joint Figure-2/Ring-LPN, OT/OLE, conversion and sampler
   transitions cost respectively `ring_w`, `ot_ole_w`, `conversion_w`,
   `sampler_w`. These transitions must be defined to avoid charging the
   same primitive simulation twice. Bootstrap/application projections and
   other outputs involving the raw replaced state must be simulated, not
   declared data processing of a single key without proof.
5. After these transitions and uniform final-word replacement the two
   ideal views have distance at most `sigma`. For the standalone programmed
   one-key experiment `sigma=0` follows from the uniform prefix/final-word
   simulator above; it is **not set to zero for the full Ring-LPN protocol**.

Then telescoping, the conditional lemma and triangle inequality yield:

```text
Adv_event <= min(1,
  sigma
  + sum_(w=0,1) [
      rng_w + transcript_w + aes_w + switch_w + bad_w
      + ring_w + ot_ole_w + conversion_w + sampler_w
    ]
  + 2*sum_l K_l*delta_l).
```

Here event advantage is a probability gap, not excess guessing success.
For a one-world simulation only one copy of its terms and
`sum_l K_l*delta_l` occurs. The more economical
`sum_l K_l*epsilon_l` applies to the **separate fixed-alpha, payload-only,
common-prefix experiment**, not arbitrary point changes or the complete
protocol. It must not be added to the uniform-replacement bound.
Neither argument assumes independence of the trees; both require the
conditional hypotheses for every step, with a common continuation kernel.

The calculator computes the map sums exactly and, separately, the
candidate ideal switching sum `6*sum_l K_l*L/2^128` **per world**.
Under the explicitly independent-uniform root experiment alone, with
`Qroot=2*sum_l K_l` sampled roots, it also computes the valid union bound
`binom(Qroot,2)/2^128` for repeated roots across roles/limbs/scopes.
This is only a root-repetition event, not a bound for all corrected
internal seeds or adversarial seed queries. Forced equal off-path seeds
are excluded from this root sample population.
No numerical domain/namespace failure rate is supplied: invocation
freshness, hash/RO collisions and ledger rollback require their own model;
an identifier's bit width does not prove uniform independent sampling.
Neither term repairs a missing transcript or Ring-LPN reduction.

Accordingly `live_complete_bound` is emitted as JSON `null`; unknown
advantages are never silently zeroed. The unconditional numerical
whole-protocol upper bound available without those hypotheses is merely
the trivial `1`. P-KEY and independent qualified review remain open.

### 9.10 Executable finite cases and verification status

The extended calculator is CPU/standard-library-only. It retains legacy
`--comparisons-p*` arithmetic, mutually exclusive with repeated `--workload`
arguments of form `n,c,t,limbs,noise,cross_terms,invocations`.
Workloads describe explicit **successful fresh invocations**, not a model
name from which layers, aborted sessions, retries or an indefinite service
lifetime are guessed. The calculator checks the source degree, depth,
frontier, tree, bootstrap and integer-domain constraints, but is not a
replacement for native shape/record admission.

Commands for Main to execute after all edits settle:

```sh
# Both roles, both tags, all own residues and payloads in a tiny field:
python3 ringlpn/scripts/audit_dpf_leaf_loss.py \
  --workload 8192,2,8,2,regular,1,1 --budget-bits 64 --check-reduced

# Exact capacity edge and one term past it, aggregated as two invocations:
python3 ringlpn/scripts/audit_dpf_leaf_loss.py \
  --workload 8192,2,8,1,regular,7424,1 \
  --workload 8192,2,8,1,regular,7425,1 --budget-bits 64

# Explicit 22-invocation FC5-shaped population, not inferred model lifetime:
python3 ringlpn/scripts/audit_dpf_leaf_loss.py \
  --workload 8192,2,8,2,regular,64000,22 --budget-bits 64

# Uniform sampler changes the domain/depth, not trees per Ring-OLE:
python3 ringlpn/scripts/audit_dpf_leaf_loss.py \
  --workload 8192,2,8,1,uniform,1,1 --budget-bits 64
```

Deterministic expectations from the formulas: the first case has `J=1`,
`D=2048`, `L=11`, `K0=K1=512`; two parties' keygen converts `4,194,304`
leaves but the conditional one-role/one-world map count is `1,024`.
The capacity-edge rows have `J=1,2` and aggregate `K0=1536,K1=0`.
The 22-invocation population has `J=9` and `K0=K1=101376`.
The uniform case has `D=16384,L=14,K0=512,K1=0`.
All are arithmetic expectations, not claimed runtime measurements.

`--check-reduced` retains the five historical distribution checks and adds
`4*13*13=676` exact role/tag/own-residue/payload comparisons at
`half_bits=4,p=13`. It checks whole-final-word distance `delta` and the
disjoint payload-shift distance `epsilon` in **every** tag branch. It
also enumerates the revealed-seed counterexample and reports the exact
repeated-resampling equality gap `12/13`. These cases distinguish the
proved conditional lemma from the invalid marginal-law/independent-query
shortcuts. No AES or DPF implementation is executed by these checks.

**Executed by Main on September 22.** All four commands above exited zero
and matched the stated exact counts; the first also passed the five
distribution checks and all 676 role/tag cases. The exported source-only
checkout additionally passed `--workload 8192,2,8,2,regular,64000,1
--budget-bits 64 --check-reduced`, with `K0=K1=4608`.
Mixing `--workload` with hypothetical `--comparisons-p*` rejected before
JSON output. Exact outputs are retained in
`autonomous_technical_closure_2026_09_22.json`. A second independent automated
source-only challenge found no counterexample to the expressly conditional
lemmas; it did not discharge the live primitive or composition hypotheses.
No capacity predictor, historical failed gate, runtime protocol source,
or security approval was changed.
