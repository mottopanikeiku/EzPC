# Pinned libOTe Reverse-Cuckoo stock baseline — 2026-08-04

## Disposition

A stock libOTe Ring-LPN/Reverse-Cuckoo run completed successfully in an isolated `/tmp` clone. This is the closest public **distributed** baseline, but it is not a function-, field-, setup-, or layout-matched measurement of the project's explicit 31-diagonal GPU artifact. Accordingly, this report makes **no speedup or timing-ratio claim**.

The measured stock row is labeled:

> **CPU, local two-party process, Goldilocks, internally sampled factors, native 16-folded Ring-LPN layout, synthetic preloaded base correlations; live setup excluded.**

No EzPC build, test, formatter, benchmark, or project validation was run. No upstream libOTe source was edited. The only project artifact created by this experiment is this report.

## Source, pin, submodule, and license verification

- Clone URL: `https://github.com/osu-crypto/libOTe.git`
- Requested branch at clone time: `dmpf`, `--single-branch`
- Detached `HEAD`: `edb5d32822eabf2dda9f6844d85d0ce2e402cdd5`
- `git status --short --branch`: `## HEAD (no branch)` before the build
- Declared and resolved submodule: `cryptoTools` at `0cf6986873e2b83966d5110398dca99172d63c20` (`v1.6.0-387-g0cf6986`)
- License: MIT text in the pinned root `LICENSE`, attributed there to Peter Rindal (2016) and Visa (2022). SHA-256: `39a218ef068824bd03e653b675f4cc8880a155632370aa0cab0419b7010fadcd`.
- Isolated experiment root: `/tmp/libote-dmpf-stock-20260804`

Exact source acquisition commands:

```sh
git clone --branch dmpf --single-branch https://github.com/osu-crypto/libOTe.git libOTe
git -C libOTe checkout --detach edb5d32822eabf2dda9f6844d85d0ce2e402cdd5
git -C libOTe rev-parse HEAD
git -C libOTe status --short --branch
git -C libOTe submodule status --recursive
sha256sum libOTe/LICENSE
```

All commands above exited 0. Before initialization, submodule status was `-0cf698... cryptoTools`; after the build initialized it, status was ` 0cf698... cryptoTools`.

## Minimal frontend build

The stock shorthand `python3 build.py -D ENABLE_RINGLPN=ON` is not a sufficient dependency closure at this pin and, importantly, `build.py` does not propagate its failed nested build status: it returned 0 while compilation failed first on disabled Regular-DPF declarations. Direct target builds then exposed the remaining compile-time dependencies (`SparseDpf`, a `DefaultBaseOT`, and `NoisyVoleSender`). The minimal successful feature closure found without editing source was:

- `ENABLE_RINGLPN=ON`
- `ENABLE_SPARSE_DPF=ON` (which turns Regular DPF on)
- `ENABLE_SIMPLESTOT_ASM=ON` (provides `DefaultBaseOT`)
- `ENABLE_SILENTOT=ON` (provides the guarded noisy-VOLE types and turns PPRF on)
- all other OT families left off

A final, separate build directory was configured and compiled from no object files:

```sh
cmake -S . -B out/build/ring-minimal-clean \
  -DENABLE_RINGLPN=ON \
  -DENABLE_SPARSE_DPF=ON \
  -DENABLE_SIMPLESTOT_ASM=ON \
  -DENABLE_SILENTOT=ON \
  -DFETCH_AUTO=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/ring-minimal-clean \
  --target frontend_libOTe --parallel 16
```

| step | exact exit | outer wall | peak RSS (`time -v`) | stdout bytes / SHA-256 | stderr bytes / SHA-256 |
|---|---:|---:|---:|---|---|
| clean configure | 0 | 2.126463 s | 125,740 KiB | 7,040 / `a7c4e84164f1bbc3848d682e9410a63858d0093c8054dcb6534e8aa8ff519b44` | 2,295 / `5b75a510129da68c70811852e1b91c1588d89f363f57565a380bdb8d62d4f32c` |
| `frontend_libOTe` target | 0 | 145.656246 s | 1,517,072 KiB | 18,233 / `0ae919b3e4b28eceb38b3a00d8cafd48893393031416c498420c82bad2136871` | 8,346 / `1c4bd9bcafe1394df136c0b8cc10f748c1de94228df358248dab855ba0cf4c56` |

The final lines were `[100%] Linking CXX executable frontend_libOTe` and `[100%] Built target frontend_libOTe`. Although only that target was requested, this frontend directly links the repositories' test utility archives, so CMake built those target prerequisites; no tests were executed.

The clean executable is byte-identical to the executable used for the run below:

```text
386553545b1814cf7c55cce3d92ac9450490b29c6b8f197fc24fc2fe07a10dc7  frontend_libOTe
3302c0f36296ccdfad588359d1805d7cb44af10c26adf3f9d72da260321fdbe6  ring-minimal-clean/CMakeCache.txt
```

## Command-dispatch finding

The requested source-derived command was executed literally first:

```sh
./out/build/ring-minimal/frontend/frontend_libOTe \
  -ring -nn 20 -c 4 -t 16 -exp 1 -trials 1 -gold
```

It **did not dispatch the benchmark**. It printed the generic libOTe help, produced no Ring-LPN result, and exited 0 in 0.005241 s with peak RSS 4,864 KiB. At this pin, `frontend/main.cpp:104-109` only enters `benchmark(cmd)` when `-bench` or `-benchmark` is present; `frontend/benchmark.h:2598-2599` handles `-ring` only inside that dispatcher.

Literal-command capture:

- stdout: 3,838 bytes, SHA-256 `7141fccec7988cbcd14282f2de3264b232014e32f9eb3569c88f645819445c8f`
- stderr including `/usr/bin/time -v`: 846 bytes, SHA-256 `665f35047268a10980c35b0ffd364f0f572312cac59fdf9385021aa10fce5c96`
- exact process exit: 0

The minimally corrected stock command was therefore:

```sh
./out/build/ring-minimal/frontend/frontend_libOTe \
  -bench -ring -nn 20 -c 4 -t 16 -exp 1 -trials 1 -gold
```

No semantic or parameter change was made; `-bench` only reaches the source's benchmark dispatcher.

## Measured stock result

### Process-level measurements

| metric | observed value |
|---|---:|
| exact exit status | 0 |
| `/usr/bin/time -v` elapsed wall | 12.43 s |
| independently recorded outer wall | 12.439095 s |
| user CPU | 14.25 s |
| system CPU | 8.75 s |
| CPU utilization | 184% |
| maximum resident set size | 22,939,444 KiB (21.8768 GiB) |
| major page faults | 0 |
| minor page faults | 6,041,258 |
| voluntary / involuntary context switches | 5,080 / 209 |
| stdout bytes / SHA-256 | 2,125 / `56bc90f8da390cd0c00ef54be8dc173318e229f6040ac2a93d0a630544eb38da` |
| stderr bytes / SHA-256 | 868 / `39292aa8df0084dd9babba0d3ece2eeedc490ca205f9b6ea5fa513fa05f0c73a` |

The process-level wall and RSS include process startup, allocations, stock synthetic `ringSetBase`, one protocol expansion, reporting, and teardown. They do **not** include cloning, dependency fetching, configuring, compiling, or live cryptographic base-correlation generation.

### Exact stdout

```text
goldilocks Time taken:
Label                   Time (ms)  diff (ms)
__________________________________
setBase                     446.4    446.448  *******
begin                       446.4      0.000
expand start                446.5      0.049
expand start                446.6      0.108
dpfParams                   447.3      0.692
setPoints                   503.0     55.718  ****
dedup Begin                 504.4      1.400
dedup done                  535.2     30.789  ****
perm Begin                  535.3      0.070
perm done                   537.2      1.959  *
hash Begin                  537.4      0.137
done Begin                  547.0      9.651  ***
solver Begin                547.1      0.040
solver done                 587.0     39.957  ****
reveal s                    588.9      1.841  *
sparseSets Begin            598.6      9.697  ***
sparseSets alloc            701.7    103.152  *****
sparseSets done            1430.6    728.901  *******
sparseDpf begin            1796.9    366.301  *******
sparseDpf done            10197.4   8400.502  **********
negate done               10330.2    132.793  *****
tesnor                    10336.7      6.516  **
dpfParams                 10336.9      0.119
expandValue               10403.3     66.417  *****
dedup begin               10403.5      0.209
dedup done                10411.7      8.245  **
perm begin                10411.8      0.087
perm done                 10415.2      3.358  *
expanded alloc done       10423.8      8.619  **
expandLeaves done         10742.6    318.816  ******
gamma done                10796.4     53.764  ****
update done               11065.9    269.573  ******
mainDpf                   11067.7      1.800  *
output fft                11465.3    397.536  *******
input fft                 11559.2     93.904  *****
done______                11581.5     22.264  ***
finish                    11636.9     55.487  ****

RingLpnTriple<goldilocks> n=1,048,576, log2=20 exp=1 trials=1 total/sec = 90,114 median time = 94,169 op/s  total time = 11 sec
setup  10,067,864 bytes, 10,448,024 bytes

```

### Exact stderr (`/usr/bin/time -v`)

```text
	Command being timed: "/tmp/libote-dmpf-stock-20260804/libOTe/out/build/ring-minimal/frontend/frontend_libOTe -bench -ring -nn 20 -c 4 -t 16 -exp 1 -trials 1 -gold"
	User time (seconds): 14.25
	System time (seconds): 8.75
	Percent of CPU this job got: 184%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.43
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 22939444
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 6041258
	Voluntary context switches: 5080
	Involuntary context switches: 209
	Swaps: 0
	File system inputs: 0
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

## Setup accounting and exclusions

The stock timer and the full process measurement must not be described as live-setup-inclusive protocol timing:

1. `frontend/benchmark.h:1825-1829` initializes both parties, invokes `ringSetBase(oles)`, then records `setBase`. Its observed local synthetic-preload interval was **446.448 ms**.
2. `ringSetBase` (`frontend/benchmark.h:1711-1776`) uses a fixed local PRNG seed, directly constructs OT receive strings, coefficient shares, coefficient-product shares, and OLE correlations, and calls `setBaseCors` on both parties. It does not execute those correlations cryptographically.
3. The production fallback in `RingLpnTriple.h:938-941` would call live `genBaseCors` only if base correlations were absent. Stock `ringSetBase` makes them present, so that live path was bypassed.
4. The printed `setup 10,067,864 bytes, 10,448,024 bytes` is source-defined as the bytes received during the **first expansion** (`benchmark.h:1872-1891,1933`). It is not a measurement of live base-correlation setup traffic.
5. The internally reported `median time = 94,169 op/s` uses only the `begin` to `done______` interval (about 11.135 s). The reported `total/sec = 90,114` uses the timer from before initialization/synthetic preload through `finish` (about 11.637 s). The independently measured 12.439095 s is the process-level end-to-end wall under the same synthetic-preload limitation.
6. Both parties ran inside one process over `LocalAsyncSocket` (`benchmark.h:1798`) with one worker thread per party (`benchmark.h:1807-1813`), not over a two-host transport.

No number in this report includes the omitted live setup. The synthetic-preload interval is reported separately rather than silently merged with online expansion.

## Mandatory mismatch labels

### Goldilocks, not project `p0`

`-gold` selects `RingLpnBenchImpl<Goldilocks>` (`frontend/benchmark.h:1960-1961`). The output group is Goldilocks, not the project's exact `p0 = 4611686018326724609`. Field arithmetic and representation are therefore mismatched.

### Internal factors, not caller-supplied factors

The stock benchmark locally synthesizes coefficient and tensor shares in `ringSetBase` (`benchmark.h:1737-1753`). During `genDpf`, each party samples sparse positions from its PRNG (`RingLpnTriple.h:947-952`). It does not accept the project's caller-provided private factors and does not return the requested raw diagonal arrays.

### Native folded 16-group layout, not explicit 31 diagonals

For `c=4,t=16`, libOTe groups block-pair products using `(ABlkIdx + BBlkIdx) % 16`, pre-negates wrapped terms, and expands 16 folded groups per polynomial pair (`RingLpnTriple.h:1126-1181`), followed by the `X^n+1` overflow subtraction (`1184-1189`). This is 256 fixed-arity sets of 16 points (4,096 terms) over a `2^17` DMPF domain.

The project artifact instead exposes 31 raw diagonal groups per polynomial pair, 496 variable-arity group-functions total, also containing 4,096 terms. These layouts are mathematically related but execute different group workloads and materialize different interfaces. Their timings must remain separate.

### Synthetic base correlations, not live `genBaseCors`

As detailed above, the runner's `ringSetBase` constructs correlations locally and preloads them. It bypasses live OT/OLE/tensor setup. This is a setup-excluded stock benchmark even though the process-level timer includes the 446.448 ms local synthetic construction.

## Malformed-parameter control

The frontend exposes typed values without robust failure signaling. The following control was run:

```sh
./out/build/ring-minimal/frontend/frontend_libOTe \
  -bench -ring -nn not-an-integer -c 4 -t 16 -exp 1 -trials 1 -gold
```

Observed stdout:

```text
RingLpnBench exception: /tmp/libote-dmpf-stock-20260804/libOTe/libOTe/Dpf/SparseDpf.h:57
```

Observed exact exit status: **0**, wall 0.006083 s, peak RSS 6,656 KiB. Stdout was 89 bytes (SHA-256 `bb4c716e9a9e432346df88edc0d572aad78f765c89ba109300c73b7b4d3da93f`); stderr/time output was 866 bytes (SHA-256 `59794345bc91136b13b0e1d4019499c7a1f4bac8cea78a727070df0324a980a7`). Thus a zero exit status alone does not prove this frontend accepted or completed a benchmark; a valid Ring-LPN result line is also required.

## Compiler, dependency, and environment digest

### Toolchain and build configuration

- OS/kernel: Ubuntu 24.04.3 LTS (Noble Numbat); `Linux 6.8.0-78-generic #78-Ubuntu SMP PREEMPT_DYNAMIC Tue Aug 12 11:34:18 UTC 2025 x86_64`
- CPU: Intel Xeon w5-3435X, 1 socket, 16 cores, 32 hardware threads, one NUMA node
- Memory observed after the run: 117,638,909,952 bytes total; swap 10,737,414,144 bytes total
- Compiler: `/usr/bin/c++` -> `/usr/bin/x86_64-linux-gnu-g++-13`, GCC 13.3.0 (`Ubuntu 13.3.0-6ubuntu2~24.04`)
- Compiler executable SHA-256: `52f1ddb33fe78b9441e0f42e9cd22c571f1101938e046c8a26582494e041cc73`
- CMake 3.31.4; Git 2.43.0; Python 3.12.3
- CMake mode: Release; `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`; SSE enabled
- Locale: `LANG=en_US.UTF-8`; `LC_ALL`, `OMP_NUM_THREADS`, `CC`, `CXX`, and `CMAKE_PREFIX_PATH` were unset

### Source dependency pins resolved by the pinned build

| dependency | commit |
|---|---|
| cryptoTools submodule | `0cf6986873e2b83966d5110398dca99172d63c20` |
| coproto fetched source | `4ac8bd7b900b75d37dca273828103a28bfb1ae91` |
| macoro fetched source | `6869ffa31f9e94211815f9fc3921d2a924ab8646` |
| function2 fetched source | `02ca99831de59c7c3a4b834789260253cace0ced` |
| libdivide fetched source | `66190e9daa603cabe95d99f09fb79b5c186d0417` |

`thirdparty/SimplestOT` is part of the pinned libOTe tree rather than a separate resolved Git worktree.

### Dynamic runtime dependencies

`ldd` resolved only the standard runtime libraries below; SHA-256 values hash the resolved files on this machine:

| library | SHA-256 |
|---|---|
| `/lib/x86_64-linux-gnu/libstdc++.so.6` | `a68762c86d371e6041f03f03a33a78fa235809ae7d81c90185940c93c3535aed` |
| `/lib/x86_64-linux-gnu/libm.so.6` | `1b87a1a50b496cfead2b0ad134c2ff536705c82608db240c7e8aa48d6c0e4217` |
| `/lib/x86_64-linux-gnu/libgcc_s.so.1` | `02f3f192bf5f79b811f1e34a650fea443d407408703448906585232383957f60` |
| `/lib/x86_64-linux-gnu/libc.so.6` | `d8db8739a1633c972cec6a4fe0566bdcec6fd088f98723492ab0361f66238f75` |
| `/lib64/ld-linux-x86-64.so.2` | `1cd555ac46b7887edeaf3c42aac5408c8135e52f6b37870da2cf82d5fe14e829` |

## Interpretation boundary

This stock result establishes that the newly public pinned Reverse-Cuckoo/libOTe implementation builds and runs at the requested nominal `(n,c,t)=(2^20,4,16)` shape. It does **not** measure project `p0`, caller-provided factors, live base-correlation generation, two-host networking, raw 31-diagonal output, or GPU evaluation. It must remain a separately labeled closest-baseline row and must never be used to derive a speedup against the explicit 31-diagonal GPU artifact.
