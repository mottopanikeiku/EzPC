# S2 architecture comparison: sparse encoder and PCG route (2026-07-29)

**One sentence:** the sparse-encoder advantage reported in the DMPF literature
is **layout-dependent, and it largely disappears for the noise layout this
artifact actually deploys** - with uniform noise an OKVS-style DMPF expands the
sparse product 275x faster than the current sum of point DPFs at the candidate
parameters, but with the deployed *regular* noise the same encoder is 0.79x
(slower) and the best DMPF wins only 2.29x, because regular noise already buys
the domain and point-count reduction the DMPF is designed to buy; no route can
be frozen yet because every dealerless candidate is unavailable, unlicensed, or
unimplemented.

Evidence classes used below: `published` (printed in the cited paper),
`reproduced` (unchanged local rerun), `adapted` (local rerun of licensed source
with a recorded patch), `derived` (formula only). No cross-class speedup ratio
is claimed.

## 1. Why the encoder was expected to matter

The proposal's own profiling states that SPFSS full-domain evaluation dominates
the pipeline (about 99% at realistic noise weights; the NTT is under 1%). The
current artifact expands the sparse product polynomial as a **sum of `m_raw`
point DPFs**, so under *uniform* noise its expansion work scales as
`O(N*m_raw)`. Every DMPF construction in the 2025-2026 literature is designed
to make that work essentially `O(N)`, independent of the point count. Section
2.2 shows why that argument does not transfer to the regular layout.

For one Ring-OLE direction, CRT limb, and polynomial pair, the required
functionality is

```text
F_i,j(x) = sum_(r,s) u_i,r * v_j,s * [x = a_i,r + b_j,s] mod p,
```

with `m_raw = c^2*t^2` raw Cartesian terms across the `c^2` pairs: 256 at
`(c,t)=(2,8)` and 4,096 at `(4,16)`. The current protocol consumes three
scalar OLEs per raw term, i.e. 768 and 12,288 setup slots. Section 7 of
`s2_parameter_novelty_provenance_audit_2026_07_29.md` holds the full matched
contract, including the private-factor setup boundary, the raw/unique/nonzero
support requirement, and the net-epoch accounting that any final row must add.

## 2. Measured sparse-encoder comparison (`adapted`)

### 2.1 Uniform layout

Source: `MatanHamilis/dmpf@ed044b903fdf6fd213b171eaa125e4eb52363903` (CC0-1.0),
the IEEE S&P 2025 improved-DMPF implementation used as a baseline by both newer
DMPF papers. Harness: `scripts/dmpf_baseline_bench.rs`, driven by
`scripts/run_dmpf_baseline_comparison.sh`.

```bash
cd GPU-MPC/ringlpn
RUNS=3 CPU=0 scripts/run_dmpf_baseline_comparison.sh
```

Artifacts: `results/dpf/dmpf_baseline_comparison_2026_07_29.csv` and `.log`.
Each row is one pinned CPU core, `nightly-2024-09-29`,
`-C target-cpu=native`, three seeds, medians below, full-domain correctness
checked against the accumulated sparse vector on every row (27/27 `pass`).
`eval` is the sum of both parties' full-domain expansions.

| tier | encoder | keygen (ms) | full-domain expand, both parties (ms) | keys, both parties (MB) | peak extra keygen (MB) | expand vs sum-of-DPF |
|---|---|---:|---:|---:|---:|---:|
| `2^13,c=2,t=8` (log domain 14, 256 pts) | sum of point DPFs | 0.341 | 147.598 | 0.156 | 0.156 | 1.00x |
| | big-state DMPF | 5.701 | 22.514 | 9.750 | 10.005 | 6.56x |
| | OKVS DMPF | 3.597 | 6.979 | 3.326 | 3.540 | **21.2x** |
| `2^14,c=4,t=16` (log domain 15, 4,096 pts) | sum of point DPFs | 5.346 | 4,583.885 | 2.621 | 2.621 | 1.00x |
| | big-state DMPF | 1,804.810 | 8,808.031 | 1,129.381 | 1,144.523 | 0.52x |
| | OKVS DMPF | 40.925 | 16.669 | 26.692 | 33.319 | **275.0x** |
| `2^20,c=4,t=16` (log domain 21, 4,096 pts) | sum of point DPFs | 7.054 | 323,069.839 | 3.408 | 3.408 | 1.00x |
| | big-state DMPF | 4,907.633 | 578,722.300 | 2,816.574 | 2,831.716 | 0.56x |
| | OKVS DMPF | 104.784 | 980.729 | 79.679 | 86.306 | **329.4x** |

Seed-to-seed spread is small relative to the gaps (worst case 28.1 s on the
578.7-s big-state literature row; 1.7 ms on the 16.7-ms OKVS candidate row), so
the ordering is not measurement noise.

Readings that are safe **for the uniform layout only**:

1. **The uniform-layout encoder gap is real and grows with the point count.**
   OKVS expansion is essentially point-count-independent, so its advantage over
   the sum of DPFs tracks `m_raw`.
2. **Big-state is not a candidate at these sizes.** At 4,096 points it is slower
   than the current sum of DPFs and needs gigabyte-scale keys.
3. **The cost moves, it does not vanish.** OKVS trades a 10x larger key (26.7 MB
   versus 2.6 MB at the candidate tier) and an 8x more expensive generation for
   its expansion win. In a dealerless design that generation must run *inside
   MPC*, so key size and solve structure become communication, not just memory.

Readings that are **not** supported:

- This is `adapted`, not `reproduced`: the harness is new, and the field is the
  implementation's fixed `Goldilocks_x2` (`0xFFFFFFFF00000001` in two
  coordinates), not the deployed 62-bit primes `4611686018326724609` /
  `4611686018309947393`.
- Generation is **centralized** in one process. These rows rank expansion
  mechanics; they do not measure any dealerless protocol.
- Inputs are distinct random points, not the accumulated Cartesian support, so
  they do not exercise private duplicate coalescing or coefficient cancellation.
- The current production expansion is on GPU; these are single-core CPU numbers.
  What transfers is the **work-count** argument (`O(N*m_raw)` versus `O(N)`
  PRG calls), not the wall-clock ratio.

### 2.2 Regular layout - what the artifact actually deploys

The deployed GPU path uses **regular** noise. Its measured feasibility rows
(`results/security/s2_candidate_gpu_feasibility_2026_07_29.csv`) record
`noise_mode=regular`, `spfss_domain=2048`, `log_domain=11` at
`(n,c,t)=(2^14,4,16)`, not one 4,096-point function over `2^15`. Regular noise
puts the `t^2` product points of each polynomial pair on `2t-1=31` predictable
diagonals, so each pair becomes 31 functions over a `2n/t=2^11` domain holding
`m_g = g+1` (`g<t`) or `2t-1-g` points - one 16-point function and two each of
1..15 points, 256 points per pair in total.

That is precisely the reduction a DMPF is supposed to provide, obtained for free
from the noise distribution. Measuring every distinct per-group point count and
summing with its exact multiplicity:

```bash
cd GPU-MPC/ringlpn
RUNS=3 CPU=6 scripts/run_dmpf_regular_layout.sh
```

Artifacts: `results/dpf/dmpf_regular_layout_2026_07_29.csv` and `.log`
(48 measured point-count/encoder rows per seed, three seeds, all `pass`).
Per-pair figures are `derived` by multiplying each measured row by its group
multiplicity; the all-pairs column multiplies by `c^2=16`.

| encoder | per pair: keygen (ms) | per pair: expand (ms) | per pair: keys (MB) | all 16 pairs: expand (ms) | all 16 pairs: keys (MB) | expand vs sum-of-DPF |
|---|---:|---:|---:|---:|---:|---:|
| sum of point DPFs | 0.278 | 20.488 | 0.131 | 327.808 | 2.097 | 1.00x |
| big-state DMPF | 2.835 | 8.955 | 4.856 | 143.284 | 77.694 | **2.29x** |
| OKVS DMPF | 9.479 | 25.955 | 14.051 | 415.277 | 224.821 | **0.79x** |

This inverts the uniform-layout conclusion:

1. **OKVS loses under regular noise.** Its band/solve fixed cost dominates at
   1-16 points per function, so it is 21% *slower* than the current baseline and
   needs 107x the key bytes.
2. **Big-state becomes the best encoder, but only by 2.29x**, and it pays 37x
   the key bytes and 10x the generation cost - before any of that generation is
   moved inside MPC.
3. **The literature's DMPF advantage is measured on the layout we do not use.**
   Any claim that a DMPF is the decisive lever for this pipeline must be stated
   for regular noise, where the honest figure is 2.29x, not 275x.

Consequence for the paper: the encoder is **not** the dominant lever at the
deployed configuration. Reporting the 275x uniform-layout number as this
project's headline would be misleading, and is explicitly rejected here.

## 3. Dealerless-setup status of each encoder

| candidate | private shared positions and payloads? | duplicate accumulation? | output algebra | executable dealerless setup |
|---|---|---|---|---|
| current sum of point DPFs | yes, per point | inherent (sum of unit vectors) | additive `Z_p` | protocol-logic host prototype with ideal OT/OLE only |
| Reverse Cuckoo DMPF (ePrint 2025/2294, rev. 2026-01-23) | yes (Fig. 6) | yes, charged as an explicit dedup protocol | additive over a group; prime-order `convert_G` case | **no public source found**; printed Fig. 7 is internally inconsistent |
| S&P 2025 improved DMPF (OKVS / big-state) | no | no; generator requires distinct inputs | additive over its field | centralized `try_gen` only |
| SLAMP-FSS | no | no; requires distinct points | XOR shares over `F_(2^m)` | distributed setup is explicitly future work |

The Reverse Cuckoo blockers are specific and independently confirmed against
the revised PDF: Figure 7 pads `m-t` dummy rows with `H_i(N)=0`, then requires
every `d x q` block to have rank `d` and to solve against
`(0,1,...,d-1)`; with `t<=d` at least one block is all-dummy-contaminated, so
at least one required solve is impossible as printed. Its
characteristic-two `bin-solver` also has no printed type conversion for the
integer right-hand side, and Figure 9 omits the payload reduction it describes
in prose. Its rank bound is conditional on a *conjectured* hash property, and
it publishes no concrete cuckoo abort probability. Section 7 of the audit
records the exact citations and its closest published rows (Table 1: 128 points
at `2^20` costs 988.50-ms setup and 38.50-ms average expansion on an
i7-13700H).

## 4. Whole-PCG route status

The encoder study cannot decide whether to keep Ring-LPN plus conversion at
all. The four-route matrix, with exact published numbers, licenses, and
per-route blockers, is in the "Whole-PCG route comparison" subsection of
`s2_parameter_novelty_provenance_audit_2026_07_29.md`. Summary:

- **current splittable Ring-LPN**: the only route exercised against Orca's
  unchanged consumer with the real q64/q128 representation; parameters,
  integrated setup, conversion, and epoch budget remain open.
- **Stationary-SD**: measured only at degree 1 (OT/VOLE). Its own Section 7.2
  leaves the Ring-LPN degree-2 evaluation this project needs to future work.
- **native `Z_(2^bw)` / Galois ring** (ePrint 2025/1223, MIT source): the only
  route whose output ring removes conversion entirely. Its released benchmark
  is centralized, and two shipped defects block a straight reproduction (see
  section 5).
- **QA-SD/WHT prime field** (ePrint 2026/196): anonymous artifact with no
  top-level license, hard-coded `p=2^61-1`, and benchmark base correlations
  synthesized in one process.

## 5. Defects found in the published native-ring artifact

While preparing an executable row for the direct-`Z_(2^bw)` route at
`zhli271828/Trace-F2-OLE-PCG@43959ef19cee4b25d0580ea0c12499c564e2328d` (MIT):

1. `init_SPDZ2k_64_bench_params` and `init_SPDZ2k_64_HD_bench_params` compute
   `param->modulus128 = 1<<(k+s)` from an `int` literal. The published 64-bit
   row uses `k+s=121`, so the shift is undefined behaviour rather than
   `2^121`. An independent `gcc -O3` probe on this host evaluates the same
   expression to zero, after which every `% modulus128` is invalid and the MAC
   key reduction divides by zero. The 32-bit initializers have the same bug at
   `k+s=58`.
2. The shipped `main.c` harness benchmarks `c=3,t=27`, which the artifact's own
   README declares insecure after ePrint 2025/892; the paper's Table 5 secure
   rows use `c=5,t=27`.
3. There is no correctness gate for the `Z_(2^k)` or SPDZ2k correlations; the
   only end-to-end validation in the tree is the inherited `F_4` OLE test.

`scripts/run_native_ring_pcg_baseline.sh` pins the revision, applies typed
shifts, moves to `c=5,t=27`, records a patch digest, runs the artifact's own
correctness gate, and emits `adapted` rows to
`results/pcg/native_ring_pcg_adapted_2026_07_29.csv`. It is explicitly not a
reproduction of the released benchmark.

### Measured `adapted` rows

```bash
cd GPU-MPC/ringlpn
TRIALS=3 CPU=4 NS="13 15" scripts/run_native_ring_pcg_baseline.sh
```

One pinned core of an Intel Xeon w5-3435X, `gcc -O3 -march=native`, `c=5,t=27`,
three trials, artifact `F_4` OLE correctness gate `pass`, patch digests recorded
in the log (`n=13`: `81d26db8...`, `n=15`: `aac0c534...`).

| correlation | `N` | setup (ms) | expand (ms) | total (ms) | peak RSS | status |
|---|---:|---:|---:|---:|---:|---|
| semi-honest `Z_(2^64)` triple | `3^13` | 1,529.1 | 5,357.4 | 7,295.2 | 8.2 GB | ok |
| SPDZ2k authenticated triple | `3^13` | 2,198.0 | 16,640.2 | 19,494.5 | 14.7 GB | ok |
| semi-honest `Z_(2^64)` triple | `3^15` | 11,248.2 | 52,100.4 | 65,591.3 | 73.2 GB | ok |
| SPDZ2k authenticated triple | `3^15` | - | - | - | 99.5 GB | **OOM-killed** |

Three readings matter for the architecture question:

1. **The route runs and scales, but its memory is the binding constraint here.**
   The semi-honest `Z_(2^64)` triple at `N=3^15` already needs 73.2 GB of
   resident memory on one core, and the authenticated variant at the same size
   was killed by the OOM killer at 99.5 GB on a 109-GB host. Any GPU port of
   this route would face the same `c^2`-tensor working-set growth, which is
   information the published tables do not expose.
2. **Authentication is not free**: at `N=3^13` the SPDZ2k MAC path costs 3.1x
   the expansion time and 1.8x the memory of the plain triple. Orca consumes
   plain Beaver shares, so the plain row is the relevant one - and it is the
   cheaper one.
3. **The published throughput claim is not reproducible from this artifact.**
   The released 64-bit benchmark cannot compute its own modulus (defect 1), the
   shipped grid is the insecure `c=3,t=27` (defect 2), and the tree documents no
   mapping from `N=3^n` to the paper's `2^13..2^16` triple batch sizes, so
   Table 5's 52k-65k triples/s cannot be checked against these rows. This is
   recorded as a reproduction failure of the published claim, not as a
   contradicting measurement.

## 6. Decision table

| question | current evidence | what it does *not* settle |
|---|---|---|
| Which sparse encoder should the dealerless pipeline target? | layout-dependent: with uniform noise OKVS wins expansion 275x (candidate) / 329x (BCG scale), but at the **deployed regular layout** OKVS is 0.79x and big-state wins only 2.29x at 37x the key bytes | whether a 2.29x expansion win survives moving generation inside MPC, and whether the ranking changes at the conservatively pinned noise weight |
| Is Reverse Cuckoo the dealerless answer? | it is the only construction whose ideal functionality matches our private-factor setting | its printed protocol does not currently type-check or rank-check, and no source exists to resolve it |
| Can SLAMP replace the point DPF? | no: centralized `Gen`, XOR output algebra, distinct-point requirement | nothing; this is settled against |
| Should Ring-LPN plus conversion be replaced? | native `Z_(2^bw)` is the only route that removes conversion by construction, and its plain-triple expansion runs here at `N=3^13`/`3^15` | its distributed setup, Orca Beaver semantics, and correctness are unimplemented; its published throughput is not reproducible from the released artifact; its `N=3^15` working set is 73-99 GB on one core |
| Can any route be frozen now? | **no** | every dealerless candidate is unavailable, unlicensed, or unimplemented |

## 7. Questions for the project owner

These are the decisions this comparison cannot make:

1. **Encoder scope.** Should the next implementation stage build a *dealerless
   OKVS-style DMPF* for the Ring-LPN product support - accepting that its
   in-MPC generation is the new research risk - or keep the per-point DPF and
   spend the effort on real OT/OLE transports instead?
2. **Reverse Cuckoo dependency.** Should we contact Agarwal, Raghuraman, and
   Rindal for their artifact and for clarification of the Figure 7 padding/rank
   and type gaps? Without that, the only fully distributed prior design cannot
   be benchmarked or safely built on.
3. **Conversion versus ring.** Is removing the `Z_M -> Z_(2^bw)` conversion
   worth changing the algebra to a native `Z_(2^bw)` PCG, given that this
   restarts parameter selection, NTT/GPU work, and the Orca key bridge?
4. **Scope of the paper's measured claim.** Should the contribution be measured
   on the current GPU Ring-LPN pipeline with a *documented* dealer-free
   protocol-logic slice, or held until a real two-process dealerless run exists?
   The first is achievable now; the second is the honest end state.

## 8. Reproduction

```bash
cd GPU-MPC/ringlpn
RUNS=3 CPU=0 scripts/run_dmpf_baseline_comparison.sh      # section 2
TRIALS=3 CPU=4 NS="13 15" scripts/run_native_ring_pcg_baseline.sh   # section 5
```

Both scripts pin an upstream revision, record host/compiler metadata, and fail
non-zero on a correctness-gate failure.
