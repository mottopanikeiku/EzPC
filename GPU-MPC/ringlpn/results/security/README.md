# Security artifacts — current interpretation

**Status correction (2026-08-04): no Ring-LPN parameter is pinned. No 128-bit
classical or quantum security claim is supported.** Read this file before using
any dated artifact in this directory.

## Invalid for parameter selection or security claims

The following files are immutable historical transcripts of a failed local
selection rule:

- `ringlpn_conservative_pin_2026_07_29.csv`
- `ringlpn_conservative_pin_2026_07_29.log`
- `ringlpn_conservative_pin_refine_2026_07_29.csv`
- `ringlpn_conservative_pin_refine_2026_07_29.log`
- `ringlpn_conservative_pin_n16_n17_2026_07_29.csv`
- `ringlpn_conservative_pin_n16_n17_2026_07_29.log`
- `s2_conservative_parameter_pin_2026_07_29.csv`

Their `meets_target=yes`, “conservative,” “pin,” “surviving,” and `t=32 -> 34`
projection-eviction interpretations are withdrawn. The files remain unchanged
only to preserve the failed experiment and its checksums.

`s2_projection_estimator_preliminary_2026_07_29.csv` is likewise not a
security-results CSV. It is a raw transcript of accepted-artifact function
outputs. It mixes mechanically undefined calls with finite-field model outputs
whose applicability to this Ring-LPN construction is unproved.

## Why the rule failed

For a projected tuple
`(N',k',t')=(c*d,(c-1)*d,floor(expected_weight))`, the accepted EUROCRYPT 2024
artifact's aggregate finite-field function unconditionally evaluates formulas
containing `C(N'-k',t')` and `C(N'-k'-1,t')`. Every call must therefore satisfy
`t' <= N'-k'-1 = d-1`. Its `com(n,m)` helper has no range check and silently
returns 1 when `m>n`, so a finite aggregate output does not prove the call was
defined.

| local tuple | degree `d` | `(N',k',t')` | domain | recorded output | disposition |
|---|---:|---:|---|---:|---|
| `c=4,t=16` | 16 | `(64,48,47)` | invalid (`47>15`) | 57.293 | withdraw |
| `c=4,t=16` | 32 | `(128,96,52)` | invalid (`52>31`) | 128.932 | withdraw |
| `c=4,t=16` | 64 | `(256,192,57)` | defined | 135.120 regular | model output only |
| `c=4,t=16` | 128 | `(512,384,60)` | defined | 145.850 regular | model output only |
| `c=2,t=128` | 256 | `(512,256,209)` | defined | 190.530 regular | model output only |
| `c=4,t=64` | 64 | `(256,192,188)` | invalid (`188>63`) | 218.641 | withdraw |
| `c=4,t=64` | 128 | `(512,384,210)` | invalid (`210>127`) | 505.207 | withdraw |
| `c=4,t=64` | 256 | `(1024,768,229)` | defined | 470.770 regular | model output only |
| `c=4,t=34` | 64 | `(256,192,110)` | invalid (`110>63`) | 257.023 | withdraw |

The domain check is necessary, not sufficient. BCG+20's corrected full version
is internally inconsistent: Section 8.2 derives
`c*d*(1-(1-1/d)^t)`, while Section 9.1 uses
`w-c*d+(c*(d-1)+w)*(1-1/d)^(t-1)`; the literal Section 9.1 criterion selects
degree 16 for `(c,w)=(4,64)`, while Table 1 reports degree 128. No published
erratum or proof resolves this.

The deployed samplers also do not directly match the estimator models. Uniform
mode fixes exactly `t` positions separately inside each polynomial block,
rather than globally sampling `ct` positions. Projection merges coordinates
and creates occupancy/cancellation dependencies. Regular mode has one position
per equal bucket before projection when `t | n`, but its projected distribution
is no longer the estimator's regular distribution. No reviewed reduction or
lower-tail bound justifies replacing either distribution by
`floor(expected_weight)`, and the estimator analyzes random linear codes rather
than this fully split quasi-cyclic structure. The two 62-bit CRT limbs and all
PCG hybrids also require an explicit distinguishing-advantage composition.

## Evidence that remains usable

- `s2_candidate_gpu_feasibility_2026_07_29.csv` is engineering-only correctness,
  memory, and timing evidence for its exercised tuple. It gives no security
  level.
- `s2_epoch_budget_preliminary_2026_07_29.csv` is setup-slot arithmetic with an
  explicitly incomplete epoch budget. It gives no security level.
- The measured `n=2^17,c=4,t=34` implementation result remains a NO-GO:
  regular noise requires `t | n`, and uniform noise would materialize
  7,272,923,136 host slots at 17 bytes each, at least 123.6 GB in one process.
  Its former 257.023-bit label is invalid and must not accompany that result.

Current diagnostic scripts are:

- `scripts/audit_ringlpn_projection_security.py`, which omits mechanically
  undefined aggregate calls;
- `scripts/audit_ringlpn_finite_field_models.py`, which requires a defined
  result for both deployed primes, labels outputs as finite-field model values,
  and deliberately exits nonzero so automation cannot interpret them as a pin.

## Required next step

Do not choose or benchmark another tuple by convention. Obtain an author
clarification/erratum or independently reviewed lemma that resolves the BCG
formula/Table discrepancy; derives the projected distribution and coefficient
cancellation for the actual samplers over both primes; proves and budgets a
lower-tail bound; justifies the useful/estimator-valid factor criterion;
handles structured-code/DOOM applicability; and composes both CRT limbs and all
PCG uses. Only then rerun the fail-closed diagnostics and measure a resulting
candidate.

Primary sources:

- BCG+20 corrected full version (2022-08-10), Sections 8.2 and 9.1 and Table 1:
  <https://hal.science/hal-03374154/document>
- Liu–Wang–Yang–Yu, EUROCRYPT 2024:
  <https://doi.org/10.1007/978-3-031-58751-1_6>
- Accepted estimator artifact `a1`, SHA-256
  `c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc`:
  <https://artifacts.iacr.org/eurocrypt/2024/a1/>
