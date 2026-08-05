# Closest DMPF baseline audit

**Date:** 2026-08-04
**Status:** internal/advisor; source audit, not benchmark evidence
**Target diagnostic:** `(n,c,t)=(2^20,4,16)`
**Exact project field:** `p0 = 4611686018326724609`

## Verdict and supersession

The 2026-07-29 statement that Reverse Cuckoo had no public code is **superseded, not deleted**. Peter Rindal's paper repository `ladnir/dmpf` became public on 2026-08-03, and it identifies the executable implementation in `osu-crypto/libOTe`. The implementation is on libOTe's non-default `dmpf` branch at commit `edb5d32822eabf2dda9f6844d85d0ce2e402cdd5`, under libOTe's MIT license. The paper-source pin audited here is `ladnir/dmpf@b55bcc4696d10e57bdea8c282a851fdd4fad0c2b`; GitHub reports no license for that separate TeX repository.

Reverse Cuckoo is therefore ranked first: it is the only located artifact that brings together distributed point setup, position-share conversion, coefficient-product sharing, duplicate accumulation, and Ring-LPN expansion. It is **not** zero-change evidence for the project's exact function, field, layout, GPU path, or setup-inclusive timing. The stock executable uses Goldilocks/Fp31 runner modes, samples factors internally, emits a folded 16-group Ring-LPN OLE rather than 31 raw diagonal arrays, runs on CPU, and injects synthetic base correlations with `ringSetBase`. A subsequent pinned stock run is recorded below and in the companion report; no performance ratio follows from it.

No public artifact simultaneously provides all of the following without adaptation: caller-supplied private factors, additive output over exact `p0`, live distributed setup included in measurement, duplicate accumulation, the project's raw 31-diagonal regular layout, and GPU full-domain evaluation.

## Source-verified ranked matrix

| rank | artifact and exact pin | code license | key generation / setup | exact-function gaps | duplicates | output group | full-domain evaluation | disposition |
|---:|---|---|---|---|---|---|---|---|
| **1** | **Reverse Cuckoo / ARR**, `osu-crypto/libOTe:dmpf@edb5d32822eabf2dda9f6844d85d0ce2e402cdd5`; paper source `ladnir/dmpf@b55bcc4696d10e57bdea8c282a851fdd4fad0c2b` | MIT for executable libOTe code; separate TeX repository reports `license: null` | Fully distributed `setPoints` and `expand`; Ring-LPN includes arithmetic-to-binary conversion, GMW correlations, coefficient tensoring, and `genBaseCors`. The stock benchmark instead calls `ringSetBase`, which synthesizes base correlations. | Stock Ring-LPN runner samples factors internally and emits final Ring-LPN OLE rather than accepting caller factors and returning 31 raw diagonal arrays. Built-in runner fields are Goldilocks/Fp31, not `p0`. | Native secure deduplication retains the first key, sums all values at that key, and replaces later occurrences with alternate keys carrying zero. | Generic prime-field-capable CPU templates; exact `p0` needs the 62-bit coefficient-context adapter below. | CPU callback expansion; no CUDA path. | **Primary distributed baseline and only setup-capable candidate.** Publication use requires author-confirmed release/tag and parameter clarification. |
| **2** | **IEEE S&P 2025 improved DMPF**, `MatanHamilis/dmpf@ed044b903fdf6fd213b171eaa125e4eb52363903` | CC0-1.0 | Central `try_gen` takes clear `(index,value)` pairs and returns both keys; no distributed setup. | Actual optimized CPU DMPF and `eval_all`, but public field is `PrimeField64x2`: two coordinates over `0xFFFFFFFF00000001`, not project `p0`. | `BigStateDmpf` and `OkvsDmpf` require strictly increasing indices. Pre-aggregate collisions modulo the output group, pad with fresh unused zero-payload points to original arity, then sort. Naive `DpfDmpf` naturally sums separate DPFs. | Fixed Goldilocks-pair output in the published implementation. | CPU `eval_all`; no GPU. | **Best executable centralized optimized-DMPF baseline.** Match the regular layout, label the field mismatch, and do not derive cross-workload speedups. |
| **3** | **`myl7/fss` 2026 artifact**, `566fc5a614612ac78b80bc86f917c5074693f79a` | Apache-2.0 | Central DPF/VDPF/VDMPF generation; no distributed setup. VDMPF is under **Unreleased**, not v1.1.0. | Public VDMPF requires `max_points>=30` and runtime `t>=30`; every regular group here has at most 16 points. Padding to 30 changes the executed workload. | VDMPF has no specified collision-accumulation contract; pre-aggregate and pad. A sum of separate DPFs accumulates exactly under group addition. | `fss::group::Uint<T,mod>` directly supports compile-time `p0` and reduces inputs modulo it. | Public CPU/OpenMP `EvalAll`. GPU kernels batch independent point evaluations, not raw full-domain expansion; Python `EvalAll` rejects GPU tensors. | **Exact central CPU control is possible with public DPF API and `Uint<uint64_t,p0>`.** It is not optimized DMPF, distributed generation, or GPU full-domain evaluation. |
| **4** | **SLAMP-FSS**, `jrmngndr/slamp-fss@893650f6a2ce902172ffeb016d82683db295c4df` | No repository `LICENSE`; GitHub reports `license: null`. The CiC paper's CC BY 4.0 does not license code. | Central `generate_keys(points)` returns both shares; no distributed setup. | Compile-time `N=20`; a regular diagonal has domain `2^17`, so matching it requires a source rebuild. | Repeated indices create repeated identical rows in `de_rand`; unequal payloads raise rank mismatch and cause retry. Equal duplicates do not provide additive accumulation. Pre-aggregate first. | Binary extension-field arithmetic with `V_BAR=128`, not odd-prime addition. | CPU `evaluate_full`; no GPU. | **Functionality- and license-mismatched reference only.** Author contact required for code license and duplicate semantics. |
| **5** | **Facebook GPU-DPF**, `facebookresearch/GPU-DPF@ce23a06af884ee54300b5bc5fd5350e445f10b0b` | Apache-2.0 | Central single-point `gen`; no distributed setup. Source retains a TODO to replace key-generation RNG. | GPU path evaluates private one-hot selection against a public table. It neither accepts arbitrary `Z_p` point payloads nor returns the raw DPF domain. Multi-point behavior would be an external sum of DPFs. | Only through external summation/pre-aggregation. | One-hot/table interface, not additive `p0` payload shares. | Raw one-hot output is CPU-only (`one_hot_only=True`); GPU returns table products. | **GPU point-retrieval control only, not a GPU DMPF/full-domain baseline.** |
| **6** | **Programmable DPF**, ePrint 2022/1060 | Paper only; no located code license | Paper states an `O(1)`-round distributed DPF-generation result, but no author implementation was found in the official record, exact-title GitHub search, author repositories, or post-audit release search. | No executable source, license, setup accounting, duplicate behavior, odd-prime instantiation, or GPU path can be audited. | Not executable. | Paper-level construction only. | Not executable. | **Author-contact-only baseline.** Request code and license; never reconstruct timings. |

The archived `myl7/vdmpf@72d6202eeaded92d2e81d08ae8ee5bc5d4918737` repository redirects to `myl7/fss` and adds no separate candidate.

## Exact regular-group adaptation

For `(n,c,t)=(2^20,4,16)`:

- bucket size `B=n/t=2^16=65,536`;
- each diagonal DMPF domain `D=2B=2^17=131,072`;
- `c^2=16` polynomial pairs;
- diagonal indices `g=0,...,30`;
- point arity `m_g=g+1` for `g<16`, and `m_g=31-g` for `g>=16`.

The exact explicit layout has **31 groups per polynomial pair, 496 group-functions, and 4,096 point terms total**:

```text
16 * sum(g=0..30, m_g) = 16 * 256 = 4096.
```

A fixed-arity DMPF API must be invoked in **16 strata**, not as one padded 16-point workload:

- for each `k=1,...,15`, one batch with `numPointsPerSet=k` and `numSets=32` (two diagonals of arity `k` for each of 16 polynomial pairs);
- for `k=16`, one batch with `numPointsPerSet=16` and `numSets=16`.

For pair `(i,j)` and diagonal `g`, the required function is

```text
F_{i,j,g}(x) = sum_{r+s=g} u_{i,r} v_{j,s} [x=a_{i,r}+b_{j,s}] mod p0,
a,b in [0,B), x in [0,D).
```

Equal `x` values are accumulated modulo `p0`. To stream a raw group back into the negacyclic polynomial without materializing 496 full arrays, set `e=gB+x`: add the share at `e` if `e<n`, otherwise subtract it at `e-n`. Since `e<=2n-2`, at most one wrap occurs.

libOTe executes a mathematically equivalent but operationally different layout. Its Ring-LPN path folds `g` modulo 16, constructs `c^2*t=256` sets of 16 points over `D`, pre-negates coefficients whose block sum wrapped, writes a possible one-block leaf overflow, and finally subtracts modulo `X^n+1`. It still contains 4,096 terms, but it is not the same per-group workload as the raw 31-diagonal evaluator. Results must label **raw 31-diagonal** and **native 16-folded** layouts separately.

## Exact Reverse Cuckoo adapter boundary

The library exposes the required cryptographic components, but an exact project harness/API adapter must do all of the following:

1. Instantiate `Fp<4611686018326724609ULL,u64,__uint128_t>`.
2. Supply a coefficient context whose `bitSize`, `binaryDecomposition`, `fromBlock`, and `powerOfTwo` operate on **62 canonical bits**. `CoeffCtxInteger` is incorrect for this modulus: it chooses `sizeof(F)*8`, byte-copies a random block into `F`, and constructs powers of two by mutating raw object bits.
3. Accept caller-provided per-party positions and coefficients rather than `genDpf`'s internally sampled factors.
4. Use live `genBaseCors(prng,sock)` rather than preloading synthetic correlations through the benchmark's `ringSetBase`.
5. Expose the 31-stratum raw-group callback, or explicitly label the different native 16-folded layout.
6. Measure live setup separately and include it in end-to-end numbers. Report key/point setup, base correlations, online expansion/evaluation, and total rather than hiding setup in a preloaded state.
7. Validate each party's expanded shares over all `D` points against the collision-accumulating `p0` reference.

Relevant public APIs at the pinned libOTe revision are:

```cpp
void init(u64 partyIdx, u64 numPointsPerSet, u64 numSets, u64 domain,
          u64 numPartitions = 2, u64 cuckooSecParam = 2,
          u64 linearSecParam = 10, bool characteristicTwo = false);
macoro::task<> setPoints(MatrixView<const u64> points, PRNG& prng,
                         coproto::Socket& sock);
macoro::task<> expand(auto&& values, PRNG& prng, coproto::Socket& sock,
                      Output output, CoeffCtx ctx = {});
macoro::task<> genBaseCors(PRNG& prng, Socket& sock);
```

The initial source-audit boundary is now superseded by the separate dated result `reverse_cuckoo_p0_baseline_2026_08_04.json` with `status: complete`. The exercised adapter uses exact `Fp<p0,u64,__uint128_t>`, a canonical 62-bit context, caller factors, live `genBaseCors`, collision accumulation, and full-domain differential validation. It deliberately exposes and labels libOTe's **native 16-folded raw** layout: 256 sets, 16 points per set, and 4,096 terms. It is therefore an exact distributed runnable row for that native layout, not a raw 31-diagonal timing row and not GPU evidence.

The dated result records setup **18,832,990 us / 52,791,184 wire bytes**, online full-domain evaluation **2,070,844 us / 1,425,584 wire bytes**, and end-to-end including validation **20,948,042 us** with **54,216,768 protocol wire bytes**. It checked 16,777,216 full-domain positions, accumulated 3,840 duplicate terms, and passed the corruption-rejection control. Its `performance_speedup` and `security_level` claims are both null. These values must not be ratioed against the raw 31-diagonal GPU path.

## Duplicate-preserving normalization

For every `(i,j,g)`, centralized and fixed-distinct-input baselines must:

1. compute all `m_g` coefficient products in `Z_p0`;
2. aggregate equal positions modulo `p0`;
3. append fresh unused positions carrying zero until the original `m_g` is restored when the implementation requires distinct fixed-length input;
4. sort by position when the API requires it;
5. expand both party keys over all `D` positions and require their modular sum to equal the collision-accumulating reference.

This preserves both the function and nominal input arity. Dropping collisions without zero padding silently changes the workload. Reverse Cuckoo performs the corresponding alternate-key/zero-value transformation internally.

## Runnable commands and exact controls

### Closest stock distributed runner

The companion [pinned stock-baseline report](libote_reverse_cuckoo_stock_baseline_2026_08_04.md) records the following no-source-edit minimal build and corrected benchmark dispatch:

```bash
git clone --branch dmpf --single-branch https://github.com/osu-crypto/libOTe.git
git -C libOTe checkout --detach edb5d32822eabf2dda9f6844d85d0ce2e402cdd5
cd libOTe
cmake -S . -B out/build/ring-minimal-clean \
  -DENABLE_RINGLPN=ON \
  -DENABLE_SPARSE_DPF=ON \
  -DENABLE_SIMPLESTOT_ASM=ON \
  -DENABLE_SILENTOT=ON \
  -DFETCH_AUTO=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/ring-minimal-clean \
  --target frontend_libOTe --parallel 16
./out/build/ring-minimal-clean/frontend/frontend_libOTe \
  -bench -ring -nn 20 -c 4 -t 16 -exp 1 -trials 1 -gold
```

`-bench` is required to enter the benchmark dispatcher. The otherwise identical literal command without `-bench` only printed generic help and exited 0; a zero status alone is not proof that this frontend ran the benchmark.

The pinned stock command completed with process wall **12.43 s**, peak RSS **22,939,444 KiB**, and libOTe's printed internal total **11 s**. The process wall includes a locally synthesized `setBase` interval of **446.448 ms** but excludes live `genBaseCors`; the runner preloads correlations using `ringSetBase`, so the live cryptographic setup fallback is bypassed. These are measured **CPU, local two-party process, Goldilocks, internally sampled factors, native 16-folded layout, synthetic-preloaded-base-correlation** numbers. They are neither exact-`p0` nor raw-31-diagonal nor GPU nor live-setup-inclusive evidence, and no speedup against the project artifact may be computed from them.

### Exact `p0` centralized CPU control

At pinned `myl7/fss@566fc5a614612ac78b80bc86f917c5074693f79a`, an exact sum-of-DPF control can use:

```cpp
using G = fss::group::Uint<uint64_t, 4611686018326724609ULL>;
using D = fss::Dpf<17, G, fss::prg::ChaCha<2>, uint>;
```

Generate one DPF with `D::Gen` for every point term, evaluate each party using `D::EvalAll`, and accumulate with `G::operator+` into the 31 raw diagonal groups. This exactly realizes the additive collision-accumulating function over `p0` with public APIs and no library modification. It remains a **central sum-of-DPF CPU control**, not optimized DMPF, distributed generation, live setup, or GPU full-domain evaluation.

## Explicit noncomparability rules

The following are mandatory for every table, plot, abstract, and paper claim:

- Do not state a speedup ratio between libOTe's native 16-folded layout and the project's raw 31-diagonal layout.
- Do not ratio Goldilocks/Fp31, a Goldilocks pair, binary extension fields, one-hot table products, or `p0` timings as though they were the same output group.
- Do not compare centralized generation with distributed generation without separately labeling the trust/setup model.
- Do not compare a preloaded `ringSetBase` run with live `genBaseCors` as though both include setup.
- Do not call per-key GPU point evaluation or GPU table lookup GPU full-domain DMPF expansion.
- Do not call the `myl7/fss` sum of independent DPFs an optimized DMPF.
- Do not call an internally sampled-factor runner a caller-factor implementation.
- Do not use padded-to-30 VDMPF timings for the arity-1-to-16 regular strata as a matched result.
- Do not turn source-derived commands into measured evidence; record revision, environment, command, raw output, failures, stage accounting, and repetition policy for any future run.
- Do not form any speedup claim across mismatched layouts, fields, correlation types, setup boundaries, CPU/GPU semantics, or central/distributed trust models.
- Include setup in end-to-end numbers and also report it as a separate stage.

No estimator row or artifact benchmark in this audit supports a concrete-security claim for the project parameters.

## Mandatory author-contact claim gates

- **Agarwal--Raghuraman--Rindal / Reverse Cuckoo:** request an archival code tag, intended code/paper linkage and license, external-factor API guidance, and intended cuckoo/linear parameters. The newly public source says current exact two-choice experiments with `d=t` have non-decaying placement-failure observations (`0.0620`, `0.1005`, `0.1158` for `t=16,64,128`), currently supports only two or three choices, and does not use `cuckooSecParam`. This is a correctness/availability warning, not a concrete-security conclusion. The paper source separately says support leakage requires a leakage-robust assumption. Keep results internal/advisor until clarified.
- **Boyle--Gilboa--Ishai--Kolobov / Programmable DPF:** request implementation, exact revision, build recipe, and code license.
- **SLAMP-FSS authors:** request an explicit repository code license and a contract for duplicate indices and odd-prime outputs.
- **myl7:** ask whether Unreleased VDMPF will receive a tag, whether `t<30` will be supported, and whether a broadcast-key GPU full-domain kernel is planned.

## Source map

All source claims below refer to the exact pins in the ranked matrix.

1. `ladnir/dmpf`, GitHub repository metadata: created `2026-08-03T04:40:32Z`, language TeX, `license: null`.
2. `ladnir/dmpf`, `RevCuckoo.tex:181-200`: exact placement experiments; current two/three-choice support; `cuckooSecParam` not used.
3. `osu-crypto/libOTe`, `libOTe/Dpf/RevCuckoo/Dedup.h:10-15`: first duplicate retained, values summed, later entries assigned alternate keys with zero value.
4. `osu-crypto/libOTe`, `libOTe/Triple/RingLpn/RingLpnTriple.h:366-400`: block size/tree-depth derivation and `mNumPolys*mNumPolys*mPolyWeight` folded-set initialization.
5. `osu-crypto/libOTe`, `frontend/benchmark.h:1710-1838`: coefficient tensor construction, `setBaseCors`, and `ringSetBase` in the stock runner.
6. `osu-crypto/libOTe`, `libOTe/Triple/RingLpn/RingLpnTriple.h:1087-1195`: live `genBaseCors`, wrap pre-negation, and final negacyclic subtraction.
7. `osu-crypto/libOTe`, `libOTe/Tools/CoeffCtx.h:69-132`: `sizeof(F)*8`, byte-view decomposition, `memcpy` from block, and raw-bit power construction.
8. `MatanHamilis/dmpf`, `src/lib.rs:61-103`: central `try_gen` and CPU `eval_all` traits.
9. `MatanHamilis/dmpf`, `src/field.rs:87-90`: public 64-bit field modulus `0xFFFFFFFF00000001`.
10. `MatanHamilis/dmpf`, `src/big_state.rs:25-36`: strictly increasing input assertion.
11. `myl7/fss`, `CHANGELOG.md:8-21`: VDMPF under Unreleased; v1.1.0 contains VDPF.
12. `myl7/fss`, `include/fss/group/uint.cuh:22-68`: compile-time modulus and modular reduction.
13. `myl7/fss`, `src/bench_gpu.cu:70-125`: GPU kernel evaluates one supplied `x` per independently indexed key.
14. `myl7/fss`, `fss_crypto/_csrc/dpf_binding_impl.cuh:107-130`: `EvalAll` requires CPU tensors.
15. `myl7/fss`, `include/fss/vdmpf.cuh:66-148`: `max_points>=30` static assertion and runtime `t>=30` assertion.
16. `jrmngndr/slamp-fss`, GitHub repository metadata: `license: null`.
17. `jrmngndr/slamp-fss`, `src/lib.rs:23-60`: central `generate_keys` and CPU `evaluate_full`.
18. `jrmngndr/slamp-fss`, `src/key_gen.rs:124-158`: repeated-row handling and rank-mismatch retry path.
19. `jrmngndr/slamp-fss`, `src/config.rs:1-7`: `N=20`, `V_BAR=128`, and binary-field configuration.
20. `facebookresearch/GPU-DPF`, `dpf.py:67-127`: central `gen`, RNG TODO, CPU one-hot path, and GPU table-product path.
21. ePrint 2022/1060 abstract: paper-level polylog-key DPF with `O(1)`-round distributed generation; no executable artifact located.
22. `results/reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md`: pinned clean build, corrected `-bench` dispatch, exact process/timer output, dependency and environment digests, and setup/mismatch accounting for the stock run.
23. `results/reports/reverse_cuckoo_p0_baseline_2026_08_04.json`: completed exact-`p0` native-folded adapter result; 62-bit coefficient context, live setup, full-domain collision-accumulating differential control, duplicate control, corruption rejection, stage timings, and wire bytes.

## Audit boundary

Investigated revisions: `osu-crypto/libOTe:dmpf@edb5d32822eabf2dda9f6844d85d0ce2e402cdd5`; `ladnir/dmpf@b55bcc4696d10e57bdea8c282a851fdd4fad0c2b`; `MatanHamilis/dmpf@ed044b903fdf6fd213b171eaa125e4eb52363903`; `myl7/fss@566fc5a614612ac78b80bc86f917c5074693f79a`; `myl7/vdmpf@72d6202eeaded92d2e81d08ae8ee5bc5d4918737`; `jrmngndr/slamp-fss@893650f6a2ce902172ffeb016d82683db295c4df`; `facebookresearch/GPU-DPF@ce23a06af884ee54300b5bc5fd5350e445f10b0b`; ePrint 2022/1060; and ePrint 2025/2294 record checked 2026-08-04.

The source-ranking audit itself ran no benchmarks, builds, tests, GPU jobs, or formatters. Linked companion experiments separately record the isolated stock build/run and the exact-`p0` native-folded adapter run; no project validation was run for this documentation update. The audit provides a source-pinned baseline selection and preserves the raw-31-diagonal/native-folded noncomparability boundary.