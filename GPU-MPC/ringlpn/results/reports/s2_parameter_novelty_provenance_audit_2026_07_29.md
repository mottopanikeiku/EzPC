# S2/M5 parameter, novelty, and provenance audit — preliminary hard-stop report

**Date:** 2026-07-29
**Status:** **not passed; no parameter set or contribution boundary is pinned**
**Circulation:** internal/advisor only

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
polynomial has exactly `t` distinct positions sampled uniformly without
replacement from `[0,n)`. This is the accepted estimator's **exact
fixed-weight** input model before any sparse-factor projection. In `regular`
mode, `[0,n)` is split into `t` equal public contiguous bins and one position
is sampled uniformly in each bin. Therefore the BCG+20 paper's total weight is
`w=c*t`; this repository's command-line `--t` is the per-polynomial weight.
The existing `c=2,t=8` rows are correctness smokes only.
The BCG+20 Table-1 128-bit row `c=4,w=64` maps to this code's `c=4,t=16`, not
`t=64`.

The active ring is `Z_p[X]/(X^n+1)`, with `n` a power of two and `2n | p-1`,
so it is deliberately fully split for NTT slot packing. This is the reducible
Ring-LPN setting analyzed in BCG+20. It is **not** directly an instance of the
quasi-abelian group-algebra setting of Bombar et al.; `X^n+1` over the odd
prime fields here is not the group algebra `F_p[C_{2n}]` or `F_p[C_n]`.

## 2. Reproducible finite-field attack-cost transcript

Inputs:

- accepted artifact: Hanlin Liu, Xiao Wang, Kang Yang, Yu Yu, *The Hardness of
  LPN over Any Integer Ring and Field for PCG Applications*, EUROCRYPT 2024;
- artifact page/license: <https://artifacts.iacr.org/eurocrypt/2024/a1/>
  (MIT, as declared by the authors);
- exact downloaded `lpn-estimator.py` SHA-256:
  `c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc`;
- runtime used for the recorded transcript: CPython 3.12.3 and NumPy 2.5.1
  in a fresh temporary virtual environment;
- wrapper: `scripts/audit_ringlpn_projection_security.py`;
- raw output:
  `results/security/s2_projection_estimator_preliminary_2026_07_29.csv`.

Reproduction after installing the estimator's NumPy dependency:

```bash
python scripts/audit_ringlpn_projection_security.py \
  --estimator /path/to/pinned/lpn-estimator.py \
  > /tmp/s2.csv
sha256sum /tmp/s2.csv
# ae6ec67336b0a4d6da13a08212d77a415adbf0921e4d6ea314627aaab4a2646e
```

For the literature-mapped `c=4,t=16` candidate, the BCG sparse-factor formula
gives the following representative projections for **both** `p0` and `p1`
(the two primes round to the same displayed costs):

| factor degree `2^i` | field-LPN `(N,k)` | expected projected weight | floor used | exact estimator | regular estimator |
|---:|---:|---:|---:|---:|---:|
| 16 (`i=4`) | `(64,48)` | 47.0967 | 47 | 57.293 | 57.293 |
| 32 (`i=5`) | `(128,96)` | 52.7706 | 52 | 128.932 | 128.932 |
| 64 (`i=6`) | `(256,192)` | 57.5145 | 57 | 179.364 | 135.120 |
| 128 (`i=7`) | `(512,384)` | 60.5133 | 60 | 158.461 | 145.850 |
| 256 (`i=8`) | `(1024,768)` | 62.1921 | 62 | 155.311 | 154.570 |

BCG+20 Table 1 reports `i=7,w_i=60` for its `c=4,w=64`, 128-bit row over a
field of about 128 bits. The accepted 2024 estimator gives 145.85 bits for the
corresponding regular finite-field instance over either deployed 62-bit prime.
That number is **not yet a Ring-LPN security claim** for two reasons:

1. The CRYPTO 2020 paper says the adversary chooses the smallest `i` with
   `w_i <= (c-1)2^i`, so the reduced instance is uniquely decodable and not
   statistically close to random. Applied literally to `c=4,w=64`, this selects
   `i=4` because `47.0967 <= 48`, whereas Table 1 reports `i=7,w_i=60`.
   Running the separate PCG repository's `bench/derive_params.py --test
   --criterion paper` reproduces only one of the six Table-1 rows. Its
   `--criterion calibrated` instead uses `w_i <= 3.3*2^i/c`; the comments state
   that `3.3` was fitted to reproduce all six rows, which the self-test does.
   Thus the published prose/table do not yield one reproducible executable
   rule, and a fitted replacement is not a security reduction. Applying the
   accepted estimator to every projection exposes the 57.293-bit `i=4` row;
   selecting the table's `i=7` yields 145.85 bits. S2 cannot choose by
   convention.
2. Reduction modulo a sparse factor produces a dependent projected-noise
   distribution summarized by an expected weight. The accepted estimator takes
   exact or regular finite-field LPN. A proof or cryptographic review must
   justify the mapping, tail bound, rounding, and advantage loss instead of
   substituting the expected weight directly.

The 2024 analysis also explains why the older BCG attack formulas are
insufficient by themselves: it corrects Pooled-Gauss, field-size-sensitive ISD,
and statistical-decoding estimates and includes attacks on regular noise.

**Parameter disposition:** no `(n,c,t,p0,p1)` set is pinned. The preliminary
performance candidate is `(n=2^14,c=4,t=16,p0,p1)`, conditional on resolving
both proof gaps above and on advisor direction about the newer DMPF route.

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
| Boyle et al., CRYPTO 2020; corrected full version ePrint 2022/1035 | Ring-LPN PCGs for OLE/triples/bilinear correlations; fully split slot packing; semi-honest distributed setup using generic 2PC plus Doerner--shelat DPF generation on shared positions/payloads | Generator algebra, slot decomposition, and dealerless setup blueprint are inherited, not contributions |
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
5. Should the performance pin use the literature-scale `n=2^20,c=4,t=16`, the
   preliminary bootstrap-sized `n=2^14,c=4,t=16`, or another reviewed point?
6. For the private PCG/PIM project, which contributors own the DPF, CPU PCG,
   GPU DPF/NTT, PIM, measurements, figures, and integration work; what may Alp
   reuse; and what citation/acknowledgement/overlap disclosure is required?
7. What is that private project's submission/release status and chronology
   relative to this sole-author Orca paper?
8. May the now-attributed Cheddar-derived backend remain, or should
   publication use a clean external backend boundary?

Until these are answered: do not start S3, do not import private-project code,
do not call any parameter set 128-bit secure, do not present the per-point DPF
as novel, and do not circulate the paper externally.

## 7. Primary sources

- BCG+20 corrected full-version record: <https://eprint.iacr.org/2022/1035>
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
