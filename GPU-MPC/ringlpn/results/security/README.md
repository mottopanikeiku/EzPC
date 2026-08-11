# Security artifacts — current interpretation

**Status correction (2026-08-04): no Ring-LPN parameter is pinned. No 128-bit
classical or quantum security claim is supported.** Read this file before using
any dated artifact in this directory.

## Live public-vector random-oracle binding

The implementation, functionality, theorem, and simulator use this exact byte
query and no abbreviated tuple:

```text
ASCII("RINGLPN_PUBLIC_A_V2\0")
|| seed[32] || scope_id[32]
|| LE64(n) || LE64(c) || LE64(t) || LE64(log_domain)
|| LE64(direction) || LE64(limb) || LE64(slot_batch)
|| LE64(modulus) || LE64(regular) || LE64(chunk)
```

The ASCII domain is 20 bytes including the terminal NUL. `seed` is the raw
256-bit XOR of the two parties' four-word seed shares, `scope_id` is the full
raw 256-bit scope identifier, and every encoded field is one unsigned 64-bit
little-endian word in the displayed order, with `regular` canonically encoded
as zero or one. SHAKE256 output is rejection-sampled to exact-uniform field
elements. The q128 path uses this one jointly combined
seed for both limbs; `limb` and `modulus` independently domain-separate the two
queries. Any `PUBLIC_A_V1`, shortened-scope, omitted-field, or independently
seeded-limb description is stale.

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


## Current live-source bindings

The security index is fail-closed on source/evidence drift. The audited live
sampler `src/two_party_spfss.h` has SHA-256
`fbdb56f84b1da9db63dad8b3464217fb05d21729344ae9ef88054b0677428461`.
A semantic diff from the prior audited `05d2fb62...` revision changes only the
optional Phase-C OLE-source argument and forwarding in the CPU baseline;
`validate_party_noise` and `sample_party_noise` are unchanged. The structured
attack report is therefore freshly rebound to the live sampling distribution.

The hybrid-RSD formula script has SHA-256
`cbcedaf6cdb1e6818aa968ab43c386f1b046b8640dd3d257039d462e7e949764`;
after its self-test reproduced the two published table rows, its 20-row CSV was
regenerated with the same embedded script digest and has SHA-256
`1f671d941189180397479f2212ba6445fa218f7b2f60f55301bcbca4066a5f25`.
These refreshed bindings close only evidence staleness. They remain internal
formula/distribution diagnostics and supply no concrete security pin.

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

Current diagnostic entry points (not current live-source security bindings) are:

- `scripts/audit_ringlpn_regular_projection.py`, which implements the exact
  integer occupied-support law for every two-power `d|n`, the exact
  prime-specific projected nonzero-support Markov law for both deployed CRT
  limbs, rigorous integer lower tails, and guarded optional model diagnostics.
  Its default rows cover the five `n=2^20` study candidates. Its estimator and
  separately labelled structured-DOOM outputs are internal/advisor
  sensitivities, not security pins.
- `scripts/audit_regular_isd_crypto2024.py`, which checksum-verifies the
  immutable Esser--Santini CRYPTO 2024 `crypto-2024-a1` archive and source
  members, reproduces its permutation, enumeration, representation, depth-2,
  CCJ, and linearization formulas on the five direct live RSD candidates, and
  fails closed for its unpinned external generic-BJMM dependency. Raw costs are
  preserved; `0.5*log2(n)` or `0.5*log2(d)` is only a separately labelled
  heuristic orbit sensitivity. Prime-specific projected fixed-weight rows are
  explicit incompatibility diagnostics because projection is not RSD.
- `scripts/audit_hybrid_rsd_asiacrypt2025.py`, which pins ePrint 2025/1284
  Theorem 1, exhaustively optimizes its integer parameters on the five direct
  live RSD candidates, and self-tests against two published table rows.
  Classical time remains a count of `F_p` operations, memory remains the
  expression inside a big-O field-element bound, and success means expected
  puncturing iterations. The calculator is executable; no author attack
  artifact, quantum cost, or reviewed structured-code reduction is available.
  Its full-orbit square-root subtraction is a separate heuristic sensitivity.
- `../reports/structured_attack_audit_2026_08_04.md`, which retains an exact
  regular-projection law, current sampler binding, source-pinned 2024
  regular-ISD and self-tested 2025 hybrid-RSD diagnostics, omitted 2025/2026
  attack notes, and a formal negacyclic/cyclic orbit and stabilizer bound.
  Outside the explicitly bounded stabilizer event, the orbit has `d` elements
  at a degree-`d` projection; a `sqrt(d)` decoder speedup remains heuristic
  outside Sendrier's concrete Stern scope. The report is an internal/advisor
  attack ledger, not reviewed concrete-security evidence.

Executed 2026-08-04 evidence:

- `s2_regular_projection_exact_2026_08_04.csv`: 1,160 exact-law records after
  the companion self-test passed; SHA-256
  `3531fa7637e717ba563e469f72e1f798c4740e49470450eaa64cd1157373b0cb`,
  embedded `analysis_sha256`
  `f05100a56e0b8c064fbffa1393a0b23a349e75aa2c7fcfb8d1714c561ef5eb00`.
  The former `6ddd1bf5...` exact transcript is superseded history.
- Corrected `s2_regular_projection_estimator_sensitivity_2026_08_04.csv`: 575
  guarded model-sensitivity records after fixing structured-DOOM orbit size
  from `(c-1)*d` to `d`; SHA-256
  `ffd335a7d9f7670073b611f390380aa44974f9501b33b2e12504f669e757a5db`,
  embedded `analysis_sha256`
  `ed9a229f57df0b7301f43b6e17d80f108852af72078e50052e04b849c6421cd3`.
  Accepted-estimator warnings are retained; all rows remain diagnostics only.
  The former `c1b9cb53...` artifact is rejected history and must not be cited.
- `regular_isd_crypto2024_2026_08_04.csv`: 50 source-pinned direct/formula and
  projected-incompatibility records for both live prime limbs where the field
  model matters; SHA-256
  `68b8329dc77d992a90257b2b6b808fc1076534305e0ec0c434831ddafb17d255`,
  embedded `analysis_sha256`
  `39159736d43e954c565645c76e0cbe1ac433e92ba1f2dcc8f2ab847af8f89dfc`.
  The accepted archive is pinned at
  `04ae2586fccb10481efb861104176e4aaabb380c3cb9704b97ce3c4768a282cb`;
  retained CCJ numeric failures and the unversioned binary-BJMM/Sage dependency
  are incompatibility evidence, not missing values to replace with a current
  estimator. No row pins security.
- `hybrid_regular_sd_asiacrypt2025_2026_08_04.csv`: 20 current formula rows,
  SHA-256
  `1f671d941189180397479f2212ba6445fa218f7b2f60f55301bcbca4066a5f25`,
  with embedded calculator SHA-256
  `cbcedaf6cdb1e6818aa968ab43c386f1b046b8640dd3d257039d462e7e949764`.
  The calculator self-test reproduced the published 132.60 and 133.15 table
  rows before regeneration. This is not an executable attack, concrete
  Ring-LPN evidence, a quantum cost, or a parameter pin.

- `scripts/audit_ringlpn_projection_security.py`, which omits mechanically
  undefined aggregate calls;
- `scripts/audit_ringlpn_finite_field_models.py`, which requires a defined
  result for both deployed primes, labels outputs as finite-field model values,
  and deliberately exits nonzero so automation cannot interpret them as a pin.

## Required next step

Do not choose or benchmark another tuple by convention. The executable exact
regular-projection support laws and lower tails now address the sampler-law and
coefficient-cancellation calculations, but do not supply a reduction to either
accepted-estimator model. Obtain independent human cryptographic review; an
author clarification/erratum or independently reviewed lemma resolving the
BCG formula/Table discrepancy; a justification of the useful/estimator-valid
factor criterion and structured-code applicability; executable direct checks
of both the CRYPTO-2024 regular-ISD and ASIACRYPT-2025 hybrid-RSD algorithms;
a decoder-specific, resource-accounted treatment of the cyclic orbit rather
than a blanket square-root subtraction; explicit dispositions for the
2025/2026 QA-SD attacks; and an explicit composition of both CRT limbs and all
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
