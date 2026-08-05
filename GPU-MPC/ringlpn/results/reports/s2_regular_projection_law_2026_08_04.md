# Exact regular-sampler projection law — S2 proof note

**Date:** 2026-08-04
**Status:** internal/advisor; exact distribution result, **not** a parameter pin
**Scope:** the implemented regular sampler only; no protocol source is changed

## 1. Result and claim boundary

This note derives the exact distribution obtained by reducing the implemented
regular Ring-LPN noise modulo every two-power one-sparse factor degree `d | n`.
It distinguishes two random variables that must not be conflated:

- `K_d`: the number of projected coordinates that receive at least one source
  term (the **occupied-bin count**); and
- `W_{p,d}`: the number of projected coefficients that are nonzero in
  `F_p` after all terms in a bin are added (the actual **Hamming support**).

The complete law is a product of independent occupancy groups, but its group
shape changes at `d=B=n/t`:

| factor degree | independent groups | balls per group | bins per group |
|---|---:|---:|---:|
| `d <= B` | `g=c` | `r=t` | `m=d` |
| `d=B*k >= B` | `g=c*k` | `r=t/k=n/d` | `m=B` |

At `d=B` the descriptions agree. In both cases `g*r=c*t`.

This is an exact distribution theorem. It is **not** a reduction from the
implemented structured Ring-LPN problem to finite-field LPN with a random
code, not a proof that an estimator row is an attack bound, and not a concrete
security claim.

## 2. Primary sources and implementation being modeled

1. E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, and P. Scholl,
   *Efficient Pseudorandom Correlation Generators from Ring-LPN*, corrected
   full version, 10 August 2022, [IACR ePrint 2022/1035](https://eprint.iacr.org/2022/1035)
   and [HAL hal-03374154](https://hal.science/hal-03374154/document)
   (called **BCG+20** below):
   - Section 3.1, Definition 3.1 defines the regular variant as one random
     position in each block of size `N/t`;
   - Section 8.2, “Taking Advantage of Reducible F,” pp. 60–62, gives the
     balls-into-bins expectation, discusses low projected-weight events, and
     proposes rejection sampling;
   - Sections 8.3 and 8.4 discuss algebraic and quasi-cyclic-code attacks;
   - Section 9.1, pp. 65–67, prints a different projected-weight formula and
     Table 1's factor choices.
2. H. Liu, X. Wang, K. Yang, and Y. Yu, *The Hardness of LPN over Any Integer
   Ring and Field for PCG Applications*, EUROCRYPT 2024, LNCS 14656,
   pp. 149–179, [DOI 10.1007/978-3-031-58751-1_6](https://doi.org/10.1007/978-3-031-58751-1_6):
   - Section 2.2 defines global exact and regular finite-field noise and says
     explicitly that its reductions use random linear codes, do not analyze
     quasi-cyclic codes, and leave extension to other codes for future work;
   - Section 2.2 also notes the `sqrt(N)` DOOM speedup for quasi-cyclic codes;
   - Section 3 proves a parameterized exact-noise-to-regular-noise reduction.
   The [accepted EUROCRYPT 2024 artifact](https://artifacts.iacr.org/eurocrypt/2024/a1/)
   is an attack-cost calculator for those models; it is not a Ring-LPN
   projection theorem.
3. The live implementation is `src/two_party_spfss.h`, especially
   `derive_spfss_work`, `validate_party_noise`, and `sample_party_noise`:
   regular mode requires power-of-two `t`, sets the ring size to `t*B`, samples
   position `j*B + U_j` with independent `U_j` uniform in `[0,B)`, and samples
   each payload uniformly in `{1,...,p-1}`. The deployed fields are

   ```text
   p0 = 4611686018326724609 = 2^62 - 6*2^24 + 1
   p1 = 4611686018309947393 = 2^62 - 7*2^24 + 1.
   ```

The notation differs across sources: BCG+20 uses total weight `w=c*t`, whereas
this implementation's `t` is the weight of **each** of the `c` polynomials.

## 3. Algebra of one-sparse projection

Assume `n`, `t`, and `d` are positive powers of two, `t | n`, and `d | n`.
Put

```text
B = n/t,
A_j = j*B + U_j,       j=0,...,t-1,
U_j <-$ {0,...,B-1},
V_j <-$ F_p^*.
```

For any one-sparse factor

```text
f_{d,gamma}(X) = X^d - gamma,       gamma^(n/d) = -1 in F_p,
```

reduction sends

```text
V_j X^A_j  ->  V_j gamma^floor(A_j/d) X^(A_j mod d).
```

(The equivalent `X^d+gamma'` sign convention makes no difference.) The NTT
condition `2n | p-1` supplies such factors for the configured two-power `n` in
each deployed field. Every multiplier `gamma^floor(A_j/d)` is nonzero.
Conditional on any position information, multiplying independent uniform
`V_j in F_p^*` by these fixed nonzero scalars preserves independent uniform
`F_p^*` payloads. Consequently the support law is independent of which
one-sparse degree-`d` factor is chosen; `p` enters only through cancellation.

### 3.1 Case `d <= B`

Because `d | B`,

```text
A_j mod d = U_j mod d.
```

Every residue has exactly `B/d` preimages in a bucket, so the `t` residues in
one polynomial are independent uniform balls in `d` bins. Different
polynomials are independent. Thus `(m,r,g)=(d,t,c)`.

### 3.2 Case `d >= B`

Write `d=B*k`. Since all quantities are powers of two and `d | n=B*t`, `k | t`.
For `j=a*k+ell`, where `ell in {0,...,k-1}`,

```text
A_j mod d = ell*B + U_j.
```

The degree-`d` coordinates split into `k` disjoint consecutive intervals of
width `B`. Each interval receives exactly `t/k=n/d` independent uniform balls
in its `B` bins. There are `k` independent groups per polynomial and hence
`(m,r,g)=(B,t/k,c*k)`.

This second branch is important: for `d>B`, the projected positions are not `t`
i.i.d. balls over all `d` coordinates. They are stratified across `k` disjoint
intervals with an exact number of balls per interval.

## 4. Exact occupied-support distribution

For one group with `r` labeled balls and `m` bins, let `A_j(s)` be the number
of length-`j` bin sequences occupying exactly `s` bins. The integer recurrence
is

```text
A_0(0) = 1,
A_0(s) = 0                         for s != 0,
A_{j+1}(s) = s*A_j(s) + (m-s+1)*A_j(s-1),
A_j(s) = 0                         outside 0 <= s <= min(j,m).
```

The two terms append the next ball to an occupied bin or to one of the
`m-(s-1)` previously empty bins. Therefore

```text
Pr[K_group=s] = A_r(s) / m^r,
F_{m,r}(Y) = sum_s A_r(s) Y^s,
Pr[K_d=s] = [Y^s] F_{m,r}(Y)^g / m^(r*g).
```

Equivalently, `A_r(s)=(m)_s S(r,s)` with a falling factorial and a Stirling
number of the second kind. The recurrence is preferable for exact integer
computation.

The exact occupied lower tail is

```text
Pr[K_d <= L] = (sum_{s=0}^L [Y^s] F_{m,r}(Y)^g) / m^(r*g).
```

No normal approximation, expected-weight substitution, or floating-point
rounding is needed.

## 5. Exact coefficient cancellation and nonzero-support distribution

Occupied bins need not remain nonzero. Let `h` independent terms be uniform in
`F_p^*`. Character orthogonality gives the exact counts

```text
Z_p(h) = # {(x1,...,xh) in (F_p^*)^h : sum xi = 0}
       = ((p-1)^h + (p-1)*(-1)^h) / p,
N_p(h) = (p-1)^h - Z_p(h).
```

This formula includes `Z_p(0)=1`, `N_p(0)=0`, `Z_p(1)=0`, and
`Z_p(2)=p-1`. For `h>=1`, the conditional cancellation probability is

```text
kappa_p(h) = Z_p(h)/(p-1)^h
           = (1 + (-1)^h*(p-1)^(1-h))/p.
```

For the deployed limbs this is, exactly,

```text
Z_p0(h) = (4611686018326724608^h
           + 4611686018326724608*(-1)^h) / 4611686018326724609,
Z_p1(h) = (4611686018309947392^h
           + 4611686018309947392*(-1)^h) / 4611686018309947393.
```

Thus cancellation is prime-specific and has an inverse-field-size scale once
`h>1`; it cannot be discarded in a proof whose claimed advantage is much
smaller than `1/p`. The alternating correction also shows why simply assigning
an independent `1/p` cancellation coin to each occupied bin is not exact.

### 5.1 Exact integer recurrence for actual Hamming support

Let `D_j(u,s)` count all assignments of `u` labeled balls, each carrying a
nonzero field value, to the first `j` bins such that exactly `s` resulting bin
sums are nonzero. Set

```text
D_0(0,0) = 1,
D_0(u,s) = 0                         otherwise,
D_j(u,s) = sum_{h=0}^u binom(u,h) *
           ( Z_p(h)*D_{j-1}(u-h,s)
             + N_p(h)*D_{j-1}(u-h,s-1) ).
```

The recurrence chooses the `h` labels assigned to the new bin, then counts
whether their sum is zero or nonzero. It obeys the check

```text
sum_s D_m(r,s) = (m*(p-1))^r.
```

Define

```text
H_{p;m,r}(Y) = sum_s D_m(r,s) Y^s.
```

The exact implemented nonzero-support law is

```text
Pr[W_{p,d}=s]
  = [Y^s] H_{p;m,r}(Y)^g / (m*(p-1))^(r*g),
```

with `(m,r,g)` from Section 1. This recurrence automatically preserves all
within-group occupancy dependence and all coefficient-sum dependence.

For an exact lower tail, convolve with integers only:

```text
Q_0(0)=1,
Q_{ell+1}(s)=sum_v Q_ell(s-v)*D_m(r,v),
Pr[W_{p,d} <= L]
  = (sum_{s=0}^L Q_g(s)) / (m*(p-1))^(r*g).
```

Comparing the integer numerator with the denominator times `2^-lambda` gives a
machine-exact tail decision; decimal logarithms are presentation only.

### 5.2 Conditional coefficient-value law

There is a stronger exact fact. Fix a projected support set `S` and, if
required, a complete allocation of source terms to bins. For a bin containing
`h>=1` terms, the number of coefficient tuples summing to any specified
`a in F_p^*` is

```text
((p-1)^h-(-1)^h)/p,
```

independent of `a`. Thus its sum, conditioned on being nonzero, is uniform in
`F_p^*`. Different bins use disjoint source coefficients, so their sums are
independent conditional on the allocation and support. Mixing over allocations
preserves the same product-uniform law because the product law does not depend
on the allocation. Therefore:

> Conditional on any realized projected support `S`, the values on `S` are
> independent uniform elements of `F_p^*`.

This may support a future exact entropy or rank argument. It does **not** make
the public projected parity-check map random, universal, or full rank; those
are separate structured-code lemmas.

### 5.3 Exact means

For one coordinate in one group, a single ball contributes zero with
probability `1-1/m` and each nonzero field value with probability
`1/(m*(p-1))`. Hence

```text
rho_p(m,r)
  = Pr[the projected coefficient is nonzero]
  = (p-1)/p * (1 - (1 - p/(m*(p-1)))^r).
```

Therefore

```text
E[K_d]     = g*m*(1-(1-1/m)^r),
E[W_{p,d}] = g*m*rho_p(m,r).
```

The difference is exactly the expected number of occupied bins whose values
cancel; it is not a rounding artifact.

## 6. Boundary cases

- **`d=1`.** Here `(m,r,g)=(1,t,c)`. Every polynomial occupies its sole
  projected coordinate, so `K_1=c`. Its actual coordinate survives with

  ```text
  theta_p(t)=1-Z_p(t)/(p-1)^t
            =(p-1)/p*(1-(-1/(p-1))^t),
  W_{p,1} ~ Binomial(c,theta_p(t)).
  ```

  In particular, a degree-one projection is not guaranteed to have weight `c`
  after coefficient addition.
- **`d=B`.** Both branches give `(m,r,g)=(B,t,c)`. This is the unique crossover;
  there is no discontinuity or extra independence assumption.
- **`d=n`.** Here `k=t`, `(m,r,g)=(B,1,c*t)`. Each group contains one nonzero
  term, so `K_n=W_{p,n}=c*t` deterministically. Projection modulo the original
  degree cannot create collisions.
- **`t=1`.** Then `B=n`, each polynomial has one nonzero term, and every
  `d | n` projection has support exactly `c`; the formulas reduce to that law.
- **`B=1`.** The case split still agrees at `d=1`; for larger `d`, each of the
  `d` one-bin groups receives `n/d` terms per polynomial. At `d=n` every group
  again has one term.

## 7. Reconciliation of BCG+20 Sections 8.2 and 9.1 with Table 1

Let `w=c*t` and call the factor degree `d` (BCG+20 Section 8.2 uses `n` for this
reduced degree). Section 8.2 derives the occupied-bin expectation

```text
E_8.2(d) = c*d*(1-(1-1/d)^t).                    (1)
```

Equation (1) is exact for the implemented **occupied** support only when
`d<=B`. It does not include value cancellation. For `d=B*k>B`, the exact
implemented occupied expectation is instead

```text
E_impl_occ(d)
  = c*k*B*(1-(1-1/B)^(t/k))
  = c*d*(1-(1-1/B)^(n/d)).                       (2)
```

Section 9.1 prints

```text
E_9.1(d)
  = w-c*d + (c*(d-1)+w)*(1-1/d)^(t-1).           (3)
```

Equations (1) and (3) are algebraically different, and (3) is not the mean of
either branch of the implemented regular sampler. The actual nonzero mean is
the prime-dependent `E[W_{p,d}]` in Section 5.3, so neither BCG formula includes
coefficient cancellation.

The inconsistency is concrete in BCG+20 Table 1's quoted `lambda=128`,
`N=2^20`, `c=4`, `w=64` row. In repository notation `t=w/c=16`. Section 9.1
says to take the smallest power-of-two degree satisfying

```text
E_9.1(d) <= (c-1)*d.
```

At `d=16`,

```text
E_9.1(16) = 47.09673832109046... <= 48,
```

so the literal rule selects degree 16. Table 1 instead prints `(i,w_i)=(7,60)`,
namely degree `2^7=128`, where (3) is `60.513279...`; the table prints 60
without specifying the integer-rounding rule. Section 8.2 gives yet another
`E_8.2(16)=41.211255651...`. No published equation in the corrected full
version reconciles these three choices.

There is an additional mechanical mismatch with the accepted EUROCRYPT 2024
artifact. Mapping a projected module-LPN row in the usual way gives

```text
(N',k',t') = (c*d,(c-1)*d,floor(projected weight)).
```

The aggregate functions call binomial terms with top parameters `N'-k'=d` and
`N'-k'-1=d-1`; thus at minimum `t'<=d-1` is required for those formulas to be
defined. The degree-16 substitution has `t'=47>d-1=15`. It is an invalid
function call, not a low attack cost. This domain check still does **not** prove
that any in-domain row models the projected Ring-LPN instance.

## 8. Exact lower-tail diagnostic: why the mean is not a pin

For the BCG+20 reference shape `n=2^20,c=4,t=16` and `d=64`, `d<=B=2^16`, so
`(m,r,g)=(64,16,4)`. The exact occupied recurrence gives

```text
E[K_64] = 57.02011623848631...,
Pr[K_64 <= 51] ~= 2^-6.589.
```

The companion exact-integer audit defines

```text
W_lambda = max { W : Pr[support < W] <= 2^-lambda }
```

using the integer comparison `numerator*2^lambda <= denominator`. It gives
`W_128=21` for occupied support at this shape: except with the explicitly
budgeted `2^-128` tail event, the support is at least 21 under that convention.
This is a probability diagnostic, not “128-bit security.” It shows that
substituting the mean into an attack estimator does not budget even moderately
likely low-weight projections. Any attack-cost sensitivity sweep is also only
a heuristic diagnostic until the distribution-to-code reduction is proved.

BCG+20 Section 8.2 explicitly recognizes this issue and suggests rejecting
noise whose projection is below a threshold, claiming an acceptance probability
about one half in its model. The implementation audited here performs no such
post-projection rejection. Moreover, conditioning jointly on every useful
factor degree changes the noise assumption and requires the **joint** tail law;
marginal tail probabilities cannot be multiplied because all projections come
from the same original noise. A future rejection sampler would therefore need
its own definition, acceptance/entropy bound, constant-time implementation
review, and a hardness assumption or reduction for the conditioned law.

## 9. What remains heuristic

The exact theorem closes only the sampler-to-projected-support calculation.
All of the following mappings remain unproved:

1. **Projected law to an estimator noise model.** The exact law is a product of
   `c` groups for `d<=B` or `c*d/B` groups for `d>=B`; within each group the
   count is coupled. It is neither global `HW_{t',c*d}` nor the EUROCRYPT 2024
   regular distribution with exactly one nonzero in each of `t'` equal blocks.
   Replacing it by either model, replacing it by its mean or floor, or charging
   only a variance term is heuristic.
2. **Tail to attack success.** A valid calculation must average an attack's
   success/cost over the exact `W_{p,d}` law, or choose a proved monotone cutoff
   and add the exact bad-tail probability to the advantage. Reporting the
   attack cost at `E[W]` does neither.
3. **Random code to structured code.** Reduction modulo a factor preserves a
   highly structured module/quasi-cyclic parity-check matrix. Liu et al.
   Section 2.2 limits its reductions to random linear codes and explicitly
   leaves other codes open; its `sqrt(N)` DOOM note is an attack warning, not a
   theorem that one generic multiplicative correction captures this code.
   BCG+20 Sections 8.3–8.4 likewise discuss, but do not reduce away, algebraic
   and quasi-cyclic structure.
4. **Useful-factor selection.** Neither “first degree with expected weight below
   dimension,” “first estimator-valid degree,” nor Table 1's degree is proved
   to dominate all attacks. Every available one-sparse factor degree and any
   useful denser factor/structured-code attack must be considered.
5. **Classical/quantum interpretation.** The accepted script returns modeled
   classical operation costs for selected attacks. It does not by itself prove
   a classical distinguishing-advantage bound, a quantum cost, or a Ring-LPN
   reduction.

## 10. Two primes, CRT, and PCG hybrid accounting

`q64` is one instance over `p0`. `q128` is two prime-field instances followed
by deterministic CRT reconstruction over `Q=p0*p1`; it is not one LPN instance
over a 124-bit field. CRT/Garner is exact and deterministic and contributes no
statistical or computational loss. It also adds no hardness: an adversary can
always reduce a CRT output to one limb.

The source draws limb samples sequentially from one OpenSSL `PartyRandom`
stream, not from two independently instantiated RNG objects. Independence of
ideal draws therefore requires a whole-stream CSPRNG-to-independent-draw
hybrid. Without that hybrid, only union bounds that do not assume independence
are justified.

For live forward invocation `mu`, define

```text
B_mu = ceil(M_mu*K_mu*N_mu/n_mu),
L_mu in {1,2} = number of CRT limbs,
R_ell = 2 * sum_{mu: ell < L_mu} B_mu.
```

`R_ell` counts fresh Ring-OLE instances for limb `ell`: one for each packed
ring batch in each of the two multiplication directions. It does **not** union
bound over the `n_mu` slots produced by one Ring-OLE.

For one exact limb sample, let

```text
Good_ell = intersection over all analyzed factor degrees d of Good_{ell,d},
delta_ell = Pr[not Good_ell].
```

Factor-degree events for a single sample are correlated. Define the joint event
first; only then may one upper-bound `delta_ell` by a union over degrees. Under
ideal independent sample draws, the all-instance sampler loss is exactly

```text
1 - product_ell (1-delta_ell)^R_ell
```

and at most `sum_ell R_ell*delta_ell`. Without the whole-stream independence
hybrid, retain the latter union bound. Cancellation and lower-tail failure
belong in `delta_ell`; they are not correctness failures.

Let `Adv_RLPN,p_ell^(R_ell)` denote the multi-instance distinguishing advantage
for the exact deployed regular distribution on limb `ell`. The replacement
hybrid pays

```text
Adv_PCG,q128 <= sum_ell Adv_RLPN,p_ell^(R_ell)
              + sampler/conditioning losses
              + Adv_other_primitives.
```

If only single-instance bounds `epsilon_ell` are available, the Ring-LPN term
is at most `sum_ell R_ell*epsilon_ell`. For `q64`, only the `p0` terms remain.
No `max`, `min`, or addition of “security bits” replaces this advantage sum. A
reviewed pin must include every Ring-LPN replacement made by the PCG proof and
reserve advantage for the CSPRNG, DPF/PRG, OT/OLE, conversion, and other
computational hybrids. The exact `Z_p(h)` law must be instantiated separately
for both primes.

There is no single sparse-noise law over `Z_Q` to which the finite-field
estimator can be applied: the live q128 route is a composition of two field
instances. Even if one asks for CRT-coordinate support, the union support is a
joint two-limb statistic and is not determined by the two marginal weight
histograms alone.

## 11. Machine-checkable theorem statement and audit schema

The following statement uses only finite integer counts and is suitable for a
proof assistant or an exact-integer audit.

```text
THEOREM RegularProjectionSupportLaw
INPUT n,c,t,d,p : positive integers
ASSUME
  Pow2(n) and Pow2(t) and Pow2(d),
  t divides n, d divides n,
  p is an odd prime, 2*n divides p-1,
  B = n/t,
  gamma in F_p^*, gamma^(n/d) = -1.
DEFINE
  if d <= B:
      (m,r,g) = (d,t,c)
  else:
      k = d/B
      (m,r,g) = (B,t/k,c*k)
  A_0(0)=1,
  A_{j+1}(s)=s*A_j(s)+(m-s+1)*A_j(s-1),
  Z_p(h)=((p-1)^h+(p-1)*(-1)^h)/p,
  N_p(h)=(p-1)^h-Z_p(h),
  D_0(0,0)=1,
  D_j(u,s)=SUM_{h=0}^u binom(u,h)*
            (Z_p(h)*D_{j-1}(u-h,s)
             +N_p(h)*D_{j-1}(u-h,s-1)),
  F(Y)=SUM_s A_r(s)Y^s,
  H(Y)=SUM_s D_m(r,s)Y^s.
CLAIM
  g*r = c*t,
  Law(ProjectOccupiedCount) = Law(SUM_{a=1}^g Occupancy(m,r)),
  Law(ProjectNonzeroCount_p) = Law(SUM_{a=1}^g NonzeroSum(m,r,p)),
  FOR ALL s:
    Pr[ProjectOccupiedCount=s] = coeff(F(Y)^g,Y^s)/m^(r*g),
    Pr[ProjectNonzeroCount_p=s]
      = coeff(H(Y)^g,Y^s)/(m*(p-1))^(r*g),
  AND
    SUM_s coeff(F(Y)^g,Y^s)=m^(r*g),
    SUM_s coeff(H(Y)^g,Y^s)=(m*(p-1))^(r*g),
  AND conditional on ProjectSupport=S,
    ProjectValues|_S is uniform over (F_p^*)^S.
```

The companion executable is
`scripts/audit_ringlpn_regular_projection.py`. Its schema rejects inputs unless
all divisibility and power-of-two preconditions hold. Its exact-law/tail
`record_type` values include `occupied_distribution_exact`,
`nonzero_distribution_exact`, `occupied_lower_tail_exact`, and
`nonzero_lower_tail_exact`; separate rows cover `any_cancellation_bound`,
`candidate_diagnostic`, `estimator_model_diagnostic`, and
`required_nonzero_weight_sensitivity`. It records `tail_budget_bits`,
`support_threshold_w`, `event`, exact `bound_numerator`/`bound_denominator`,
rigorous `bound_log2_floor`/`bound_log2_ceiling`,
`guaranteed_rejection_entropy_bits`, and `tail_budget_pass`. Prime rows include
`coefficient_limb`, `prime`, and exact `nonzero_distribution_*` fields. Every
estimator/structured-DOOM value is labeled `heuristic_diagnostic_only`, and the
estimator guard rejects `t'>d-1` before calling the accepted artifact.

## 12. Executed evidence and artifact identity

The companion tool's self-test passed. The current exact-law artifact is
`results/security/s2_regular_projection_exact_2026_08_04.csv` (1,160 records):

```text
CSV SHA-256      3531fa7637e717ba563e469f72e1f798c4740e49470450eaa64cd1157373b0cb
analysis_sha256  f05100a56e0b8c064fbffa1393a0b23a349e75aa2c7fcfb8d1714c561ef5eb00
```

The former `6ddd1bf5...` exact transcript was superseded when the corrected
script/schema was rerun for source reproducibility; it is not current evidence.

The corrected guarded optional-model artifact is
`results/security/s2_regular_projection_estimator_sensitivity_2026_08_04.csv`
(575 records):

```text
CSV SHA-256       ffd335a7d9f7670073b611f390380aa44974f9501b33b2e12504f669e757a5db
analysis_sha256   ed9a229f57df0b7301f43b6e17d80f108852af72078e50052e04b849c6421cd3
estimator SHA-256 c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc
```

This rerun passed after correcting the structured-DOOM orbit size from
`(c-1)*d` to `d`; the former `c1b9cb53...` CSV is rejected history. Accepted
estimator runtime warnings are retained. Every model/structured-DOOM value
remains diagnostic only. The exact-law CSV does not call the estimator, and
neither artifact pins a parameter or supports a concrete-security label.

## 13. Minimum obligations before a reviewed concrete pin

A concrete tuple `(n,c,t,p0,p1)` can be reviewed only after all of these are
closed:

1. **Distribution proof:** independently verify the two-branch theorem, the
   conditional value theorem, prime-specific cancellation recurrence, all
   boundary cases, and exact tool arithmetic against a second implementation
   or proof assistant.
2. **Factor coverage:** enumerate every one-sparse factor degree for each
   deployed prime and justify treatment of denser factors; prove any claimed
   dominance relation rather than checking only a favorite degree.
3. **Tail rule:** give an advantage-level lower-tail calculation for actual
   `W_{p,d}`, including rounding. If conditioning is used, specify the exact
   joint event over factors, acceptance probability, entropy/statistical loss,
   sampler implementation, and hardness of the conditioned distribution.
4. **Estimator reduction:** either reduce the exact projected group-product
   law to a precisely named finite-field LPN distribution with explicit loss,
   or analyze attacks directly under the exact law. Every call must satisfy its
   mathematical and source-code domain, including `t'<=d-1` where required.
5. **Structured-code analysis:** justify the parity-check/code distribution
   after factor projection and cover quasi-cyclic DOOM, algebraic, ISD,
   statistical-decoding, and any more effective structured attack. Conditional
   uniform coefficient values and a `sqrt(N)` adjustment do not prove the
   structured map universal or full-rank.
6. **Per-prime result:** perform the complete analysis independently over `p0`
   and `p1`, including their factorization/root conditions and cancellation
   laws. No result over an approximately 128-bit field may substitute for
   either approximately 62-bit limb.
7. **Composition budget:** state `R_ell`, prove a multi-instance bound or pay
   the single-instance hybrid factor, establish the CSPRNG stream hybrid, and
   sum both limb advantages plus rejection/statistical and all other primitive
   losses.
8. **Attack model:** state classical and quantum models, memory/data limits,
   success probability, operation-to-bit-cost convention, and uncertainty;
   reproduce every estimator row from a pinned accepted source revision.
9. **Independent review:** obtain independent human cryptographic review of
   the distribution, reduction, structured-code analysis, estimator use, and
   composition before attaching a concrete-security label or changing runtime
   parameters.

## 14. Explicit non-claims

This report does **not** claim:

- that any `(n,c,t,p0,p1)` tuple is pinned;
- 128-bit, 80-bit, classical, quantum, or any other concrete security level;
- that `q64` or `q128` denotes security bits;
- that BCG+20 Table 1 validates this implementation's parameters;
- that a finite-field estimator output is a Ring-LPN attack cost or lower bound;
- that occupied weight equals nonzero Hamming weight;
- that expected projected weight controls the lower tail;
- that checking one factor degree controls all factor degrees;
- that conditional uniform support values make the structured code random or
  full rank;
- that the accepted estimator covers the projected dependent noise or the
  structured/quasi-cyclic code;
- that two CRT limbs add their modeled bit costs;
- that BCG+20's suggested rejection sampling is implemented; or
- that this exact sampler theorem closes the PCG, DPF/PRG, transport,
  conversion, malicious-security, or side-channel proof obligations.

The only affirmative claim is the finite combinatorial law in Sections 3–6,
subject to its stated sampler and factor preconditions. It is a starting point
for cryptographic review, not its conclusion.
