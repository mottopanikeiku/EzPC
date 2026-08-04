# S2/M5 parameter, novelty, and provenance audit — corrected hard-stop report

**Date:** 2026-07-29
**Status:** **not passed; no parameter set or contribution boundary is pinned**
**Circulation:** internal/advisor only

**Status correction (2026-08-04):** the primary-source parameter reaudit is
complete and strengthens the hard stop: no parameter is pinned and no 128-bit
claim is supported. The saved 57.293-, 111.244-, 218.641-, and 257.023-bit
selected rows feed out-of-range binomial parameters to the accepted estimator
and are invalid attack-cost outputs. The mechanically defined 135.12-,
145.85-, 190.53-, and 470.77-bit rows remain only hypothetical finite-field
exact/regular-model outputs; the mapping from dependent projected Ring-LPN
noise, its lower tail and coefficient cancellation, the structured code, and
the two-limb/PCG advantage loss is unproved. All dated `conservative_pin`
artifacts are invalid for parameter selection. The measured
`n=2^17,c=4,t=34` implementation NO-GO remains valid independently: regular
noise cannot satisfy `t | n`, and the uniform layout exceeds host memory.

This report records the S2 work completed before implementation stage S3. It
also records two hard stops that were not visible in the v2.3 proposal:

1. the local reduction-selection rule used by the separate PCG project is an
   empirical fit, not a cited theorem, so it cannot support a 128-bit claim;
2. January-2026 prior art directly provides fully distributed multi-point
   generation for Ring-LPN PCGs, with a proof and prototype, so this project's
   per-point distributed DPF cannot currently be claimed as a protocol
   contribution.

No code or measurement from the separate PCG/PIM repository has been imported.

## 1. Exact implemented algebra and parameter names

The GPU generator runs independently over these NTT primes:

- `p0 = 4611686018326724609 = 2^62 - 6*2^24 + 1`;
- `p1 = 4611686018309947393 = 2^62 - 7*2^24 + 1`.

`q64` means one instance over `p0` (62 actual bits). `q128` means two
independently seeded prime-field instances followed by CRT reconstruction over
`Q=p0*p1` (124 actual bits). It is **not** one Ring-LPN instance over a
124-bit field. Any q128 security argument needs a two-limb hybrid/composition
argument and a Ring-LPN claim for each 62-bit limb.

For each party, the code samples `c` independent sparse polynomials and chooses
every nonzero coefficient uniformly from `Z_p^*`. In `uniform` mode, each
polynomial separately has exact weight `t`, but concatenating `c` such
polynomials conditions the support to exactly `t` positions in every block; it
is not the estimator's global `HW_{ct,cn}` distribution. `regular` mode matches
consecutive equal buckets before projection when `t | n`; after projection
neither sampler has the estimator's exact or regular distribution. The BCG+20
paper's total weight is `w=c*t`; this repository's command-line `--t` is the
per-polynomial weight.
The existing `c=2,t=8` rows are correctness smokes only.
The BCG+20 Table-1 128-bit row `c=4,w=64` maps to this code's `c=4,t=16`, not
`t=64`.

The active ring is `Z_p[X]/(X^n+1)`, with `n` a power of two and `2n | p-1`,
so it is deliberately fully split for NTT slot packing. This is the reducible
Ring-LPN setting analyzed in BCG+20. It is **not** directly an instance of the
quasi-abelian group-algebra setting of Bombar et al.; `X^n+1` over the odd
prime fields here is not the group algebra `F_p[C_{2n}]` or `F_p[C_n]`.

## 2. Historical finite-field function transcript and correction

Inputs:

- accepted artifact: Hanlin Liu, Xiao Wang, Kang Yang, Yu Yu, *The Hardness of
  LPN over Any Integer Ring and Field for PCG Applications*, EUROCRYPT 2024;
- artifact page/license: <https://artifacts.iacr.org/eurocrypt/2024/a1/>
  (MIT, as declared by the authors);
- exact downloaded `lpn-estimator.py` SHA-256:
  `c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc`;
- runtime used for the dated raw transcript: CPython 3.12.3 and NumPy 2.5.1
  in a fresh temporary virtual environment;
- historical wrapper: the pre-domain-check revision of
  `scripts/audit_ringlpn_projection_security.py`;
- immutable raw output:
  `results/security/s2_projection_estimator_preliminary_2026_07_29.csv`,
  SHA-256
  `ae6ec67336b0a4d6da13a08212d77a415adbf0921e4d6ea314627aaab4a2646e`.

The current wrapper adds the missing aggregate-domain check and deliberately
does not reproduce invalid rows. Do not overwrite the dated transcript or use
its raw function outputs as security evidence; its disposition is recorded in
`results/security/README.md`.

For a projected tuple `(N',k',t')=(c*d,(c-1)*d,floor(expected))`, the accepted
artifact's aggregate finite-field function unconditionally evaluates
combinations `C(N'-k',t')` and `C(N'-k'-1,t')`. It is therefore mechanically
defined only when `t' <= d-1`; this is a source-code domain check, not a proof
that a projection is useful.

| local tuple | degree | `(N',k',t')` | domain | recorded result | status |
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

BCG+20 is itself unresolved. Section 8.2 derives
`c*d*(1-(1-1/d)^t)`, whereas Section 9.1 uses the different formula implemented
locally. Its literal criterion selects degree 16 for `(c,w)=(4,64)`, while
Table 1 reports degree 128. Moreover, BCG explicitly warns about lower-than-
expected projected weights and proposes rejection sampling; the local sampler
does not perform that projection check. The accepted estimator models global
exact or regular finite-field noise and random linear codes, not this dependent
projected distribution or structured Ring-LPN code.

**Parameter disposition:** no `(n,c,t,p0,p1)` set is pinned. The raw projection
CSV is retained only as a function transcript with the current erratum in
`results/security/README.md`; every dated `conservative_pin` result is invalid
for parameter selection. A reviewed projection-distribution/tail/structured-
code lemma, advantage budget for both limbs, and BCG rule clarification are
required before another estimator sweep.

## 3. Candidate feasibility, not security

The current per-point DPF prototype consumes three scalar OLEs per DPF. With
`c^2*t^2` product points, its bootstrap condition is

`3*c^2*t^2 < n`.

For `c=4,t=16`, this is `12,288 < n`; the smallest supported power-of-two ring
is `n=16,384`, with only a 1.333x output/cost surplus. The centralized-keygen
GPU artifact was run at this point on 2026-07-29:

| path | validation | mean expand (3 measured iterations) | keygen | pair-key bytes |
|---|---|---:|---:|---:|
| q64 / p0 | pass | 68.3517 ms | 43.043 ms | 1,842,432 |
| q128 / p0+p1 | pass | 133.255 ms | 84.682 ms | 3,684,864 |

Raw rows:
`results/security/s2_candidate_gpu_feasibility_2026_07_29.csv`.
Host validation is skipped by this benchmark at `n=16,384`; the device
validator passed. These are local central-keygen timings, not real distributed
transport, model-scale throughput, or a security result.

The inequality is only a bootstrap lower bound, not the required epoch budget.
Let `C_key=3*c^2*t^2` be next-epoch keygen consumption per expanded ring OLE.
The steady-state capacity left for application work is at most `n-C_key`:

| point | `C_key` OLEs | raw surplus `n-C_key` | raw application fraction |
|---|---:|---:|---:|
| smoke `n=8192,c=2,t=8` | 768 | 7,424 | 90.625% |
| preliminary `n=16384,c=4,t=16` | 12,288 | 4,096 | 25.000% |
| BCG scale `n=2^20,c=4,t=16` | 12,288 | 1,036,288 | 98.828% |
| rejected `n=8192,c=2,t=64` | 49,152 | -40,960 | impossible |

Machine-readable rows:
`results/security/s2_epoch_budget_preliminary_2026_07_29.csv`.

The present FC artifact counts `2*L*ceil(M*K*N/n)` ring OLEs for two cross
directions and `L` CRT limbs. A self-sustaining implementation needs at least
`2*L*ceil(M*K*N/(n-C_key))` before conversion demand, discarded tail slots,
safety margin, abort/retry handling, and epoch-zero Gilboa setup. Thus the
`16x32x16` candidate would require at least four q64 or eight q128 ring OLEs,
twice the current artifact count. No source currently assigns conversion
daBits/AND triples to a concrete PCG output pool, identifies every consume-once
slot, or specifies the safety margin. The full epoch budget therefore remains
open and cannot be closed until the DPF/DMPF and conversion routes are selected.

The candidate is intentionally not frozen because adopting the 2026 DMPF prior
art would change the number and shape of setup correlations and invalidate the
`3*c^2*t^2` design constraint.

## 4. Formal novelty and overlap audit

| Work | Established before this project | Consequence here |
|---|---|---|
| Boyle et al., CRYPTO 2020; corrected full version HAL `hal-03374154` (2022-08-10) | Ring-LPN PCGs for OLE/triples/bilinear correlations; fully split slot packing; semi-honest distributed setup using generic 2PC plus Doerner--shelat DPF generation on shared positions/payloads | Generator algebra, slot decomposition, and dealerless setup blueprint are inherited, not contributions |
| Doerner--shelat, CCS 2017 | OT-based distributed DPF key generation | A level-walk distributed DPF is prior art |
| Boyle et al., *Programmable DPFs*, CRYPTO 2022, ePrint 2022/1060 | `O(1)`-round distributed DPF generation for feasible/full-domain settings | The current logarithmic-round tree walk is not the round-complexity frontier |
| Boyle, Gilboa, Hamilis, Ishai, Tu, IEEE S&P 2025 | Improved DMPFs with optimized implementations and PCG applications | Naively summing point DPFs already has a stronger public implementation baseline |
| Kolesnikov, Peceny, Raghuraman, Rindal, CRYPTO 2025 / ePrint 2025/295 | Stationary Syndrome Decoding and PCGs amortize one support set across correlated noise vectors; for Ring-LPN Beaver triples the paper specifically identifies the high noise-generation overhead as an amortization target | Standard regular Ring-LPN is not the only structured-noise route; SSD changes both the assumption and setup/evaluation budget and needs explicit consideration |
| Agarwal, Raghuraman, Rindal, *Fully Distributed Multi-Point Functions for PCGs and Beyond*, ePrint 2025/2294, posted 2026-01-23 | Fully distributed efficient DMPF setup, semi-honest straight-line proof, prototype; directly replaces the `t`-DPF bottleneck in Ring-LPN/Stationary-LPN PCGs and reports order-of-magnitude gains | **Direct closest prior art.** The present per-point DPF cannot be the paper's enabling protocol contribution without a concrete, reviewed delta; adopting this work is the default technical comparison |
| Külaots, Krips, Eerikson, Pisetskaya, Pullonen-Raudvere, *SLAMP-FSS*, IACR CiC 2026, DOI 10.62056/avommpxqi | Two-party multi-point FSS from tree PRGs and linear systems; introduces distributed random multi-point functions and improves state-of-the-art PRG-call cost | A second 2026 multi-point design must be compared on key generation, expansion, correlations, and GPU suitability; a sum of point DPFs is not the current FSS frontier |
| Li, Xing, Yao, Yuan, EUROCRYPT 2025 / ePrint 2025/169; Li, Liu, Xing, Yao, Yuan, ePrint 2026/196 | Programmable PCGs over any finite field, followed by a fully implemented QA-SD/Walsh--Hadamard OLE/VOLE design that avoids FFT multiplications and FFT-friendly-prime restrictions | QA-SD/WHT is a current prime-field PCG architecture and performance baseline; Ring-LPN/NTT cannot be selected solely because the existing GPU kernel is convenient |
| Li, Xing, Yao, Yuan, CRYPTO 2025 / ePrint 2025/1223 | PCGs and multiplication triples directly over `Z_(2^k)` with silent sublinear preprocessing | A direct-Orca-ring alternative may remove the prime-to-`Z_(2^bw)` conversion component; it must be compared before architecture freeze |
| Jawalkar et al., Orca, IEEE S&P 2024 | GPU FSS-based secure training/inference with dealer preprocessing | Orca and its online path are the inherited host system |
| Cheddar; GPU-NTT | Published GPU NTT implementations/algorithms | Polynomial kernels are dependencies or derived code, not novelty |
| private `yanxue820/PCG-acceleration` project | Real two-process Ring-LPN PCG, Ferret/Gilboa setup, half-tree DPF, GPU DPF, GPU-NTT, and PIM work by multiple contributors | GPU-PCG protocol and performance claims overlap internal prior work; reuse and disclosure require explicit permission |

**Current defensible contribution candidate:** the end-to-end systems
integration that maps protocol-backed Ring-LPN preprocessing into Orca's exact
FC training state machine and validates byte-compatible keys through the
unchanged GPU online consumer, together with a model-scale evaluation against
appropriate CPU/GPU/dealer baselines. That contribution does not exist yet at
the publication gate: the current transcript still has centralized keygen,
conversion oracles, common seeds, and one-process components.

The corrected three-OLE Phase C is currently classified as a local protocol
bug fix and compatibility artifact. No reviewed delta from Doerner--shelat,
BCG+20 distributed setup, Programmable DPF, or the 2026 DMPF work has been
identified. The removed sign opening was a flaw in this project's prior
prototype, not evidence of a flaw in those papers.

## 5. Source, license, and contributor inventory

| Component | Pin / origin | License and status | Required action before circulation |
|---|---|---|---|
| EzPC/Orca base | `mpc-msri/EzPC`; repository root | MIT, Microsoft Research copyright | Retain root notice and cite Orca |
| NFLlib submodule | `5cf40ed6a4929bfc304f3283aafd62c4149c55e2` | MIT; notice present in submodule | Retain submodule notice and cite NFLlib when used |
| active `bench_ntt_cuda_cheddar.cu` | substantial adaptation of `scale-snu/cheddar-fhe` `src/core/NTT.cu`; reconstructed attribution pin `307b49cbe03e7f8f14bf31485f716c1090c9ec9d`, NTT blob `b681f4c68f56b6fafd1db13c5dd58822ed6d2d51`; the immediate parent has the identical NTT blob | MIT, copyright 2026 Scalable Computer Architecture Laboratory, Seoul National University. This audit added the source header, complete `extern/Cheddar_MIT_LICENSE.txt`, paper citation, and `extern/Cheddar_PROVENANCE.txt` | Keep the provenance/notice with every source or artifact release; professor still decides whether this dependency remains in the paper |
| external GPU-NTT baseline | clean local checkout `95c739c48d11827277e132f5eec4d4e454d60835` | Apache-2.0; external, not linked into canonical path | Cite ePrint 2023/1410 and IEEE Access 2025; preserve license if packaged; document any patch |
| EUROCRYPT 2024 estimator | accepted artifact `eurocrypt/2024/a1`; exact file digest above | MIT declared on artifact page; not vendored | Keep digest, artifact citation, dependency version, and raw transcript |
| LEDAtools comparison | `e88810086be5bd9fba75937954af68fbd54ead01` | Unlicense/public domain dedication | Record pin and tool assumptions if its figures remain |
| local Ring-LPN/Orca integration | commits in this fork under configured Alp identity | Covered by repository license subject to inherited-code notices; protocol ideas remain cited prior art | Professor must settle overlap with the private project before a sole-author paper circulates |
| private `PCG-acceleration` project | private remote `yanxue820/PCG-acceleration`; inspected branch HEAD `fe6bbe05cff3fce17b35f57febb06af4b714fbfd`; no repository LICENSE found | no public license grant; multiple authors | Do not copy code, figures, measurements, or prose without explicit contributor/professor permission and disclosure |

Recorded private-project chronology from its Git history:

| Git author | Dated commit evidence | Changes attributable from commit history |
|---|---|---|
| `yanxue820` (Jiayan Xue) | `0d239c7` (2026-04-09), followed by `4277918`, `a48f9e7`, `dc733ef`, `ab8b51c` | initial half-tree DPF and CRT/RNS PCG-OLE prototype; initial DPF/Ring-LPN documentation; configurable parameters and early profiling |
| Chenkai Weng | `f6c7cf2`, `2f5d666`, `52aaca7` (2026-04-15/16); `be4bbe9`, `31cdece`, `95c0b3e`, `4b80ed8` (2026-04-18); later `e821141`, `2ea01a1`, `1d24cf7`, `e6362a8`, `c791a51`, `cc86004` | DPF repository restructure, two-party/full generation and evaluation; polynomial ring, reduced rounds, NTT/OLE, and triples; later hash/Beaver output correction, fitted parameter script, tree-PRG/output alignment, payload/Gilboa merge, and communication instrumentation/reduction |
| Alp | `23d0598` (2026-05-26); `ff316e7` (2026-06-03); `a95404f`, `d5a57d7`, `e240b68` (2026-06-10); `fe6bbe0` (2026-07-13) | CUDA polynomial baseline; GPU-NTT four-step integration; AES-128 GPU half-tree DPF, end-to-end GPU baseline, DPF-kernel optimization, reports, and later GPU DPF conversion |
| `LYCesh` | `bc89f11` (2026-06-25); `82a94f4`, `b1a4f9e`, `e5c0f5e` | CPU baseline, benchmark single-source refactor, measured campaign/data completion, and CPU report updates |
| `tkgong` (T. K. Gong) | `5826489` (2026-06-17); `e33bfcb`, `5f38839`, `27e4bd9`, `fab4148`; `bd3a70b`, `6d977cf`; `c0c52fa`, `88160a2`, `5316815`, `f477eaa`; later security-size and roofline campaigns through `a96b30d` | HBM2/HBM3 PIM models, PIM/GPU co-execution and NTT offload, simulator-grounded GPU baselines; ChaCha device PRG and device-resident leaves; Beaver-corrected CPU/PIM branches, fused PIM algorithms, security-size campaigns, and roofline evidence |

**Specific DPF-output overlap requiring a professor/contributor ruling.**
Private-project commit `e821141` by Chenkai Weng (2026-07-12), titled
“Replace leaky output layer with hash + Beaver-corrected CW,” predates this
fork's `28f8451` corrected host DPF artifact (2026-07-21). The private code
hashes half-tree leaves, forms additive `A,B` shares, uses a field Beaver triple
to compute/open `CW=A*B`, and later commit `e6362a8` merges cross products into
the payload Gilboa batch. The EzPC artifact is a different standard-key-format
construction consumed by `spfss_host::dpfEvalAll`; its corrected Phase C uses
three scalar OLEs and opens only `finalCW`. The code and exact protocols differ,
but both histories correct a leaky output layer by securely multiplying
private aggregates before opening a correction word. Git chronology cannot
establish whether this was independent derivation or adaptation. Do not claim
the EzPC Phase-C idea as Alp's novel contribution unless Chenkai and the
professor explicitly settle provenance and credit.

Commit authorship establishes chronology, not ownership allocation or reuse
permission. The private repository has no LICENSE file and is not publicly
readable at its configured GitHub URL, so no permission can be inferred.

## 6. Decisions required from the professor

S2 remains blocked until the professor answers all of these in writing:

1. Is the paper's research contribution the integrated dealerless Orca FC
   system, or is a new distributed-DPF protocol contribution required?
2. Given Agarwal--Raghuraman--Rindal and SLAMP-FSS (2026), should the
   implementation adopt and benchmark one of the multi-point designs, retain
   per-point DPF only as a baseline, or pursue a specifically identified delta?
3. Given Stationary-SD, the 2025 direct-`Z_(2^bw)` PCG, and the 2026
   QA-SD/WHT prime-field PCG, should the project stay with regular
   Ring-LPN/NTT plus conversion, compare all routes, or pivot?
4. Who will review and approve the sparse-factor reduction-usefulness criterion,
   projected-noise mapping, advantage loss, and classical/quantum security
   interpretation before parameters are called 128-bit secure?
5. Which parameter should be measured after a reviewed projection/distribution,
   structured-code, and two-limb advantage analysis establishes a valid set?
6. For the private PCG/PIM project, which contributors own the DPF, CPU PCG,
   GPU DPF/NTT, PIM, measurements, figures, and integration work; what may Alp
   reuse; and what citation/acknowledgement/overlap disclosure is required?
7. What is that private project's submission/release status and chronology
   relative to this sole-author Orca paper?
8. May the now-attributed Cheddar-derived backend remain, or should
   publication use a clean external backend boundary?

The owner later lifted only the implementation ordering: M1 GPU
distributed-keygen core work and real-transport/component S3--S6 work may
proceed without security, parameter, end-to-end, or publication claims. Do not
import private-project code, call any parameter set 128-bit secure, present the
per-point DPF as novel, or circulate the paper externally until the remaining
advisor, proof, and provenance decisions are closed.

## 7. Consultation-driven matched architecture comparison

The 2026-07-29 project-owner consultation selected an integrated systems
contribution and required a comparison before architecture freeze. The
comparison must use the exact sparse-product functionality consumed by this
repository, not each paper's preferred demonstration workload.

For one Ring-OLE direction, CRT limb, and polynomial pair `(i,j)`, define

```text
F_i,j(x) = sum_(r=0)^(t-1) sum_(s=0)^(t-1)
           u_i,r * v_j,s * [x = a_i,r + b_j,s] mod p.
```

The parties privately hold the respective positions and nonzero coefficients.
The domain is `[0,2n)`, duplicate sums must accumulate rather than abort, and
the two keys must full-evaluate to additive `Z_p` shares accepted by the
existing polynomial-product path. There are `c^2` such functions per
direction, two directions per Beaver cross term, and one or two independent
prime limbs for q64 or q128.

The end-to-end setup boundary begins at those private factor lists, not at a
DMPF API that already holds shared point/value vectors. It includes secure
Cartesian point addition, coefficient multiplication, duplicate coalescing,
zero removal, and conversion into the candidate's input shares. A standalone
DMPF setup time and this factor-to-DMPF MPC stage must be reported separately
and summed; omitting the latter is not a Ring-LPN setup comparison.

Two noise layouts are separate workloads:

- **uniform:** one `t^2`-point function over a `2n` domain per polynomial pair;
- **regular:** for bucket sum `g in [0,2t-2]`, one function over a `2n/t`
  domain with triangular point count
  `m_g=g+1` for `g<t` and `m_g=2t-1-g` otherwise. The `2t-1` groups contain
  exactly `t^2` points in total and are scattered back to the same polynomial.

The three comparison tiers are:

| tier | `n,c,t` | uniform `(log domain, points/function, raw total)` | regular `(log domain, groups/function, raw total)` | claim |
|---|---|---:|---:|---|
| current feasibility | `2^13,2,8` | `(14,64,256)` | `(11,15,256)` | smoke configuration; not a security claim |
| preliminary candidate | `2^14,4,16` | `(15,256,4,096)` | `(11,31,4,096)` | architecture-cost candidate; not a security claim |
| literature reference | `2^20,4,16` | `(21,256,4,096)` | `(17,31,4,096)` | BCG+20 parameter-scale reference only |

Here `m_raw=c^2*t^2` counts raw Cartesian product terms: 256 or 4,096.
The current protocol's three OLEs per raw term consume 768 or 12,288
scalar-OLE slots. The suggested values `m≈98` and `m≈12,288` are therefore
not matched point counts for these `(c,t)` pairs; the latter is a
setup-correlation count. Every workload must additionally report
`m_unique`, the number of distinct positions after accumulation, and
`m_nonzero`, the support after coefficient cancellation. A distinct-random
support of size `m_raw` is an expansion-mechanics stress test, not the exact
private Ring-LPN workload. Combining the `c^2` functions into one DMPF is a
separate candidate optimization and requires a proof that pair identity is
irrelevant to the unchanged consumer.

This section is a **DMPF subcomparison** inside the current Ring-LPN route. It
cannot select the overall PCG architecture; Stationary-SD, native
`Z_(2^bw)`, and QA-SD/WHT remain a separate required comparison. The DMPF
candidates also have different functionality and must not be collapsed into
one runtime bar:

1. the current sum of point DPFs is the compatibility baseline;
2. Agarwal--Raghuraman--Rindal Reverse Cuckoo is the only named candidate that
   specifies fully distributed setup from private shared sparse inputs, but
   the factor-to-product MPC stage and secure deduplication remain part of its
   Ring-LPN total;
3. SLAMP-FSS is a functionality-incompatible published reference: its `Gen`
   is centralized, distributed setup is future work, it requires distinct
   points, and it outputs XOR shares over characteristic-two fields rather
   than additive shares over the deployed odd q62 primes;
4. the CC0 IEEE S&P 2025 `MatanHamilis/dmpf` implementation is an executable
   centralized expansion baseline. Its fixed Goldilocks-x2 output and
   distinct-input generator make the new harness an `adapted` scale
   measurement, not a reproduction over the deployed primes.

Every executable row must report:

- exact source revision, patch/harness digest, license, compiler flags,
  CPU/GPU, process and thread count, field, PRG, security/statistical
  parameters, domain, `m_raw`, `m_unique`, `m_nonzero`, and collision policy;
- factor-to-product MPC, secure deduplication, and DMPF generation separately,
  then total key-generation wall time, rounds, sent/received bytes, ideal or
  real OT/OLE/triple counts, abort probability, peak memory, and key bytes per
  party;
- single-point and full-domain evaluation time, PRG calls, peak memory, and
  output bytes;
- correctness against the same accumulated sparse vector, including duplicate
  positions and both deployed primes;
- transport protocol: process placement, link bandwidth/RTT, serialization,
  host/device-transfer boundaries, warmups, repetitions, and dispersion;
- per expanded ring OLE, gross setup consumption `C_setup`, net usable slots
  `n_net=n-C_setup-conversion-reserve-abort-reserve`, epoch-zero cost, and
  route-specific abort/retry policy;
- per FC layer, `R=ceil(M*K*N/n_net)` expanded outputs and every cost multiplied
  by `R`, two cross directions, and the CRT limb count.

Published timings remain `published` rows. An unchanged local rerun is
`reproduced`; a changed field, parameterization, optimization, or harness is
`adapted`; formula-only values are `derived`. Hardware, functionality, field,
output width, or setup-model differences prohibit a speedup ratio.

A route can pass the dealerless architecture gate only if it supports private
shared positions and payloads from the factor-list boundary, has a semi-honest
proof for the observed transcript, uses licensed/auditable code, and passes
the unchanged Ring-OLE consumer. No currently runnable alternative satisfies
that gate: Reverse Cuckoo lacks an available auditable artifact, while all
local DMPF rows use centralized generation. The architecture freeze therefore
remains blocked after the expansion comparison.

### Public artifact and license status (checked 2026-07-29)

| artifact | exact pin | license | executable disposition |
|---|---|---|---|
| Reverse Cuckoo / fully distributed DMPF | ePrint revision 2026-01-23 | paper CC BY 4.0; no code license found | The paper reports a prototype, but the ePrint record links no repository; GitHub repository search and Peter Rindal's public repositories exposed no matching source. Request the artifact from the authors; do not reconstruct benchmark claims from an unavailable implementation. |
| SLAMP-FSS | `jrmngndr/slamp-fss@893650f6a2ce902172ffeb016d82683db295c4df` | paper CC BY 4.0; repository has no `LICENSE` and GitHub reports no license | Use the published formulas and Table 4. Do not vendor, modify, redistribute, or call its timings locally reproduced without written code permission. Its current key generation is centralized in any case. |
| IEEE S&P 2025 improved DMPF | `MatanHamilis/dmpf@ed044b903fdf6fd213b171eaa125e4eb52363903` | CC0-1.0 | Safe to pin and rerun as an executable expansion/key-size baseline; not fully distributed key generation. |
| current point-DPF prototype | EzPC checkpoint `28f8451`, corrected security contract `63a0c05` | repository license | Reproducible compatibility and protocol-logic baseline; ideal transports and per-point setup prevent a cryptographic dealerless claim. |

This availability result narrows, but cannot complete, the experiment. A
truthful local expansion-mechanics comparison can include the current baseline
and adapted CC0 S&P 2025 implementations. SLAMP contributes only
functionality-incompatible published/analytic rows. Reverse Cuckoo needs an
author-provided licensed artifact, resolution of the paper-level ambiguities
identified below, and a matched private-factor setup before a reproduced
dealerless row is possible.

### Reverse Cuckoo: published evidence and specification blockers

The revised 49-page ePrint is dated 2026-01-23. Its Figure 6 ideal
functionality is the closest semantic match found: it accepts secret-shared
positions and payloads, adds payloads at duplicate positions, and returns a
dense additive share vector over a group. Section 10.1 nevertheless places a
generic-MPC Cartesian addition/product stage before the DMPF; Figure 14 does
not expose whether that stage is included in its claimed end-to-end setup.

The exact printed Figure 7 cannot currently justify implementation. It sets
`w=2`, `d=2^ceil(log2(t))`, `m=w*d`, pads with `m-t` copies of dummy key `N`,
requires `H_i(N)=0`, shuffles all rows into `d x q` matrices, and requires each
matrix to have rank `d` and solve against `(0,1,...,d-1)`. Since `t<=d`, at
least `d` zero dummy rows exist. Any block containing one has rank below `d`
and cannot map that row to a nonzero right-hand side. Thus at least one of the
two required solves is impossible as printed. The characteristic-two
`bin-solver`, integer bin-label right-hand side, and claimed `q`-bit descriptor
also lack a printed type conversion. Figure 9 states collision accumulation
but omits the final payload reduction. These are specification blockers, not
an experimentally reproduced attack; author source or clarification is
required.

The paper's rank failure bound `2^-(q-d)` assumes independently uniform rows,
but Section 7.3 only conjectures that its concrete Goldreich-style hash meets
that condition. It gives no concrete finite-parameter cuckoo abort bound. The
printed generic solver costs
`O(d*log(q))` rounds and `O(d*q^2+d*q*log|G|)` bits per block; printed dedup
performs `t*(t+1)/2` equality checks. These polynomial terms are material at
4,096 points and cannot be replaced by the paper's earlier
`O(t*log(N))` communication goal.

Closest standalone published rows (Table 1, p. 43; u64 field, linear security
40, cuckoo security 2, three expansions, one set; i7-13700H, single-threaded
per party, local >10-Gbps socket) are:

| domain | points | setup | average expand | total |
|---:|---:|---:|---:|---:|
| `2^16` | 16 | 72.90 ms | 1.37 ms | 77 ms |
| `2^18` | 16 | 172.90 ms | 6.37 ms | 194 ms |
| `2^20` | 16 | 516.20 ms | 27.27 ms | 598 ms |
| `2^16` | 128 | 201.80 ms | 2.40 ms | 209 ms |
| `2^18` | 128 | 363.80 ms | 7.07 ms | 385 ms |
| `2^20` | 128 | 988.50 ms | 38.50 ms | 1,104 ms |

No row uses 256 or 4,096 points, domains `2^15` or `2^21`, private-factor
inputs, or dual q62 outputs. Figure 14's Ring-LPN rows report Goldilocks
throughput up to 1,811,012 OLE/s at `n=2^20` (12.490-s setup, 170.44-MB setup
communication, 12.99-MB expansion communication) and Fp31 throughput up to
2,709,498 OLE/s (11.583 s, 163.95 MB, 6.51 MB), but omit sparsity and DMPF
point counts. The surrounding prose also contradicts Figure 14 by quoting
gigabytes instead of megabytes. These remain `published`, incompatible rows;
no project speedup is inferred.

### Whole-PCG route comparison

The DMPF study chooses only a sparse encoder inside Ring-LPN. The overall
architecture comparison starts at two private PCG seeds and ends at the exact
correlation consumed by Orca: plain additive Beaver shares over
`Z_(2^bw)`, with setup, expansion, conversion, packing, and key serialization
all charged. Authenticated SPDZ2k triples, scalar OLEs, degree-1 OT/VOLE, and
prime-field OLEs are separate functionalities, not interchangeable throughput
units.

| route | output / assumption | distributed setup status | executable evidence | Orca disposition |
|---|---|---|---|---|
| current splittable Ring-LPN | one or two q62 prime limbs under reducible Ring-LPN, then conversion | BCG analyses programmable setup; local path has separate ideal-transport keygen and conversion prototypes, not an integrated two-process setup | exact local q64/q128 algebra and unchanged Orca consumer pass | compatibility baseline only; parameter proof, real setup, conversion budget, and epoch budget remain open |
| Stationary-SD | measured degree-1 OT/VOLE under Stationary-SD; degree-2 Ring-LPN is only sketched | libOTe implements stationary PPRF/OT/VOLE, not the Ring-LPN/Beaver construction | degree-1 primitive can be rerun; Section 7.2 explicitly leaves the needed degree-2 evaluation to future work | no direct `Z_(2^bw)` output and no executable Orca route |
| native `Z_(2^bw)` / Galois ring | programmable OLE and SPDZ2k correlations under Ring-LPN or QA-SD | paper says malicious setup can be adapted; released benchmark centrally samples both DPF keys | MIT source at `zhli271828/Trace-F2-OLE-PCG@43959ef19cee4b25d0580ea0c12499c564e2328d`; source-native benchmark is not dealerless | strongest algebraic candidate for eliminating conversion, but plain Orca Beaver semantics, real setup, and correctness validation are missing |
| QA-SD/WHT prime field | prime-field OLE/VOLE under QA-SD; WHT avoids NTT roots | paper gives a semi-honest hybrid setup; benchmark fills base correlations from one PRNG | anonymous artifact has no top-level license, hard-codes `p=2^61-1`, and does not implement either q62 prime | would still require two prime routes plus CRT/conversion for q128; target adaptation and parameter review are new work |

The evidence points are intentionally not ratioed:

- **BCG+20 estimated**, `N=2^20,c=4,w=64`, one i7-7600U core,
  approximately 124-bit modulus: 1.26-MB seed per party, 2.86-MB passive setup,
  and 10.0-s expansion (1.4-s ring arithmetic plus 8.6-s DPF);
- **Stationary-SD published degree 1**, fixed `q*k=2^24` on an i7-12650H:
  approximately 45 ns/output and 5 bits/output versus 68 ns/output for ordinary
  SD; this says nothing measured about degree-2 Beaver generation;
- **native-ring published**, AMD EPYC 9754, secure `c=5,t=27`: SPDZ2k
  authenticated-triple throughput for batches `2^13..2^16` is
  115k/113k/109k/102k triples/s at `(k,s)=(32,26)` and
  65k/64k/62k/52k triples/s at `(64,57)`. The faster `c=3,t=27` rows are
  explicitly broken after the 2025 QA-SD attack and must not be used;
- **WHT published**, one EPYC 9754 core: at `N=2^20`, benchmark
  `(c,t)=(2,64)` takes 37.68 s and 14.86 MB, while `(3,32)` takes 43.37 s and
  8.93 MB. These do not equal Table 3's conservative 128-bit points
  `(2,67)` and `(3,42)`.

Source inspection adds a reproducibility blocker for the native-ring artifact.
Both `init_SPDZ2k_64_bench_params` and its higher-degree variant compute
`modulus128 = 1 << (k+s)` with a 32-bit integer literal; the published
64-bit row uses `k+s=121`, so this shift is undefined C rather than
`2^121`. A separate GCC `-O3` probe on this workstation evaluated the same
expression to zero. The following `% modulus128` is consequently invalid.
The shipped main also runs `c=3,t=27`, the paper's explicitly broken row, and
contains no SPDZ2k correlation validation. Any repaired `c=5` run is therefore
an `adapted` artifact measurement with a patch digest and new correctness gate,
not a reproduction of the released benchmark.

This comparison does not select a route. Only the current baseline has been
exercised against Orca's exact consumer; none has a reproduced, security-pinned
dealerless setup. The native-ring route deserves the next bounded prototype
only if the project owner prioritizes removing conversion, but that is an
architecture decision requiring consultation, not an inference from
incompatible published throughput.

## 8. Primary sources

- BCG+20 corrected full version (2022-08-10): <https://hal.science/hal-03374154/document>
- Programmable DPF: <https://eprint.iacr.org/2022/1060>
- Stationary Syndrome Decoding: <https://eprint.iacr.org/2025/295>
- Fully Distributed Multi-Point Functions: <https://eprint.iacr.org/2025/2294>
- SLAMP-FSS: <https://cic.iacr.org/p/3/1/16>
- Any-finite-field PCG: <https://eprint.iacr.org/2025/169>
- Walsh--Hadamard PCG: <https://eprint.iacr.org/2026/196>
- Native `Z_(p^k)` PCGs: <https://eprint.iacr.org/2025/1223>
- EUROCRYPT 2024 hardness artifact: <https://artifacts.iacr.org/eurocrypt/2024/a1/>
- Cheddar source: <https://github.com/scale-snu/cheddar-fhe>
- GPU-NTT source: <https://github.com/Alisah-Ozcan/GPU-NTT>
