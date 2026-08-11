# Ring-LPN Benchmarks (CPU + GPU)

> **Start at [`CLAUDE.md`](CLAUDE.md)** — the canonical catch-up document
> (current status, source map, validated claims vs. open boundaries, roadmap,
> environment gotchas). The full-graph checkpoint is driven by
> [`scripts/run_resnet18_full_graph.sh`](scripts/run_resnet18_full_graph.sh);
> results and reports are indexed in
> [`results/README.md`](results/README.md), including the current
> [publication portfolio](results/reports/publication_portfolio_2026_08_04.md)
> and [authenticated two-host deployment contract](results/reports/authenticated_two_host_deployment_2026_08_04.md).
> Older per-artifact pointers below are historical and may lag.

The wrapper's default linear preprocessing is serial.  Set `LINEAR_LANES` to
comma-separated `P0_GPU:P1_GPU:CHECK_GPU:FIRST_PORT-LAST_PORT` descriptors to
opt into explicitly isolated concurrent GPU-pair lanes; see `CLAUDE.md` for
the current command example and exclusivity rules.

This folder is the dealerless forward-linear preprocessing artifact and its
source-bound Orca integration harness. Stock Orca's online consumers remain
unchanged.

## Canonical component builds

Use `./scripts/build_component.sh list` to discover the maintained library and
executable targets. In particular:

```bash
./scripts/build_component.sh linear-library
./scripts/build_component.sh linear-fc
./scripts/build_component.sh linear-conv
./scripts/build_component.sh graph-libraries
./scripts/build_component.sh resnet18-full-graph
```
`linear-library` produces the deterministic
`build/linear-library/lib/libringlpn_linear.a`, installs the standalone
`<ringlpn/linear_preprocess.h>`, and links the focused public-API probe under
`build/linear-library/bin`. The production facade opens and validates exactly
one party-local private record at a time; it deliberately exposes no record-pair
loader. The API probe's optional synthetic-fixture mode also closes each
party's record before opening the other and includes symlink,
path-replacement, and non-0600-mode negative controls.


The FC and Conv targets retain the fixed private canonical source symlink used
by the binary-approval gate. The full-graph target first builds and links the
deterministic `libringlpn_linear.a` facade, then source-builds cryptoTools,
bitpack, and LLAMA—but not the unused `libsytorch.a`—under
`build/graph-libraries`; it neither compiles a second raw Conv backend,
downloads dependencies, nor relies on an untracked
`GPU-MPC/ext/sytorch/build`. A missing recursive submodule, compiler,
CUDA toolkit, or concurrently held build root fails before compilation. Legacy
`build_*.sh` entry points remain callable and route the approved adapters and
full graph through this entry point.

The reproduction image is built with the fixed non-root UID/GID `65532:65532`
and deterministic epoch. The host dispatcher overrides the runtime user with
the invoking non-root UID/GID so bind-mounted outputs remain caller-owned;
runtime UID or GID zero is rejected.

## Layout
- src/bench_ntt.cpp: NFLLib CPU microbenchmark (NTT, INTT, PolyMul)
- src/bench_ntt_cuda_cheddar.cu: primary CUDA benchmark, extracted from cheddar-fhe and adapted to the Ring-LPN harness
- src/bench_ntt_cuda.cu: archived legacy CUDA NTT benchmark retained only for opt-in historical comparison
- src/bench_vole_ringlpn.cu: standalone Ring-LPN VOLE prototype benchmark built on the promoted Cheddar CUDA PolyMul path
- src/gpu_spfss_zp.cuh: standalone GPU DPF/SPFSS path with additive Z_p payload shares for the Figure 2 OLE artifact
- src/test_spfss_zp_cuda.cu: GPU SPFSS payload correctness test
- src/bench_ole_ringlpn_cuda.cu: standalone GPU Figure 2 SPFSS/OLE benchmark
- src/bench_linear_ole_ringlpn_cuda.cu: standalone ring-polynomial linear-layer Beaver artifact built from two Figure 2 OLEs per product
- src/test_orca_zp_bridge.cpp: host-only Orca bridge test for carry-corrected `Z_p -> Z_{2^bw}` share conversion and constant-polynomial scalar packing
- src/bench_orca_fc_ringlpn_demo.cu: tiny Orca FC key-writer demo that emits raw `A`, `B`, `C_masked` buffers and exercises unchanged `gpuMatmulBeaver`
- ../tests/fss/dpf_online_keygen_bench.cu: standalone DPF online key generation benchmark for one-shot versus chunked partial generation
- scripts/setup_nfl.sh: clone + build NFLLib
- scripts/build_bench.sh: build the benchmark
- scripts/run_sweep.sh: run 10 configs and generate CSV + Markdown
- scripts/build_cuda_bench.sh: build the primary CUDA benchmark (cheddar-derived implementation)
- scripts/build_vole_bench.sh: build the standalone Ring-LPN VOLE prototype benchmark
- scripts/run_vole_sweep.sh: run the standalone Ring-LPN VOLE prototype sweep and generate CSV + Markdown
- scripts/build_ole_cuda_bench.sh: build the standalone GPU Figure 2 OLE benchmark and GPU SPFSS test
- scripts/run_ole_sweep.sh: run the OLE smoke or bounded sweep and generate CSV + Markdown
- scripts/summarize_ole_results.py: summarize OLE CSV output
- scripts/build_linear_ole_bench.sh: build the ring-polynomial linear OLE-to-Beaver benchmark
- scripts/run_linear_ole_sweep.sh: run the linear OLE smoke benchmark and generate CSV + Markdown
- scripts/summarize_linear_ole_results.py: summarize linear OLE CSV output
- scripts/build_orca_zp_bridge_test.sh: build the host-only Orca `Z_p -> Z_{2^bw}` bridge test
- scripts/run_orca_zp_bridge_test.sh: run the bridge smoke and write CSV + Markdown
- scripts/build_orca_fc_ringlpn_demo.sh: build the tiny Orca FC Ring-LPN key-writer demo
- scripts/run_orca_fc_ringlpn_demo.sh: run the tiny Orca FC demo and write CSV + Markdown
- scripts/summarize_orca_fc_demo.py: summarize the FC demo CSV output
- scripts/run_paper_checkpoint_smoke.sh: one-command host smoke, with optional CUDA/OLE/linear/FC smoke inside the container
- scripts/build_cuda_bench_cheddar.sh: build an explicit standalone cheddar-derived alias binary
- scripts/build_cuda_bench_legacy.sh: archived opt-in build for the old CUDA benchmark path
- scripts/run_cuda_single.sh: run a CPU vs GPU spot check for requested q=32 at n in {8192, 16384, 32768}
- scripts/run_cuda_sweep.sh: run the requested q=32, q=64, or q=128 CUDA sweep with batching and generate CSV + Markdown
- scripts/run_cuda_sweep_legacy.sh: archived opt-in legacy CUDA q=32 sweep for historical comparison
- scripts/summarize_cuda_results.py: summarize CUDA sweep outputs
- ../scripts/run_dpf_online_keygen_sweep.py: run the standalone DPF online key generation sweep and generate CSV + Markdown
- scripts/run_vtune_hotspots.sh: VTune hotspots wrapper for CPU benchmark
- scripts/run_vtune_memory.sh: VTune memory-access wrapper for CPU benchmark
- results/: output files, organized per artifact (see results/README.md for the index)

The following per-artifact handoffs are historical implementation context:
`results/reports/ole_gpu_handoff.md`,
`results/reports/linear_ole_handoff.md`,
`results/reports/orca_zp_bridge_handoff.md`,
`results/reports/orca_fc_ringlpn_demo_memo.md`, and
`results/reports/orca_ringlpn_linear_integration_plan.md`. Their claims and
next steps are superseded by `CLAUDE.md`; do not treat
`results/reports/paper_execution_next_steps.md` as a current roadmap.

## Quick start (inside container)
```bash
cd /home/ringlpn

# 1) Clone and build NFLLib
./scripts/setup_nfl.sh

# 2) Build the benchmark
./scripts/build_bench.sh

# 3) Run sweep (10 configs)
./scripts/run_sweep.sh
```

## Outputs
- results/ntt/ntt_cpu.csv
- results/ntt/ntt_cpu.md

## CUDA q=32 / q=64 / q=128 sweeps
The current primary GPU deliverable is a batched CUDA NTT path for requested `q=32`, `q=64`, and `q=128`. The q=32 and q=64 modes use one 30-bit or 62-bit prime, while q=128 uses two 62-bit CRT prime limbs, all supporting `n` through `2^20`.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_bench.sh
./scripts/build_cuda_bench.sh
./scripts/run_cuda_sweep.sh

# Optional q=64 sweep
QBITS=64 ./scripts/run_cuda_sweep.sh

# Optional q=128 CRT sweep
QBITS=128 ./scripts/run_cuda_sweep.sh
```

Outputs:
- results/ntt/ntt_gpu_q32.csv
- results/ntt/ntt_gpu_q32_unsupported.csv
- results/ntt/ntt_gpu_q32.md
- results/ntt/ntt_gpu_q64.csv
- results/ntt/ntt_gpu_q64_unsupported.csv
- results/ntt/ntt_gpu_q64.md
- results/ntt/ntt_gpu_q128.csv
- results/ntt/ntt_gpu_q128_unsupported.csv
- results/ntt/ntt_gpu_q128.md

Notes:
- The primary CUDA path is now the cheddar-derived implementation in `src/bench_ntt_cuda_cheddar.cu`, built into `bin/bench_ntt_cuda` by `scripts/build_cuda_bench.sh`.
- The promoted path keeps the existing CLI and CSV contract while replacing the internal kernel implementation with the cheddar-derived two-phase NTT / INTT structure.
- `bench_ntt_cuda` accepts `--n`, `--qbits 30|32|64|128`, `--batch`, `--iters`, and `--warmup`.
- Requested `qbits=32` maps to actual `qbits=30`, requested `qbits=64` maps to actual `qbits=62`, and requested `qbits=128` maps to actual `qbits=124` via two q62 CRT limbs.
- The selected NTT prime sets support the full `n in {8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}` sweep.
- `run_cuda_single.sh` remains available for spot CPU-vs-GPU comparisons at the CPU-supported points up to `n=32768`.

## Archived Legacy CUDA Baseline
The original hand-written CUDA path is retained for historical comparison only. It is not part of the active GPU NTT pipeline; the active `bench_ntt_cuda` binary is the Cheddar-derived implementation.

Build and sweep the legacy path inside the CUDA-enabled container by opting in explicitly:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

ALLOW_LEGACY_CUDA_NTT=1 ./scripts/build_cuda_bench_legacy.sh
ALLOW_LEGACY_CUDA_NTT=1 ./scripts/run_cuda_sweep_legacy.sh
```

Legacy outputs:
- results/ntt/ntt_gpu_q32_legacy.csv
- results/ntt/ntt_gpu_q32_legacy_unsupported.csv
- results/ntt/ntt_gpu_q32_legacy.md

## Cheddar Backend Alias
The canonical GPU NTT binary is `bench_ntt_cuda`, built from `src/bench_ntt_cuda_cheddar.cu`. The repository also includes an explicit standalone alias, `bench_ntt_cuda_cheddar`, which builds the same source under a separate name for manual checks.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_cuda_bench_cheddar.sh
./bin/bench_ntt_cuda_cheddar --n 8192 --qbits 32 --batch 1 --iters 100 --warmup 10
```

Notes:
- The extracted kernels use a flattened `(batch, prime)` Ring-LPN layout. q32/q64 run one prime limb, and q128 runs two q62 limbs in the same Cheddar phase kernels.
- The build uses `-std=c++17`, which matches the cheddar-fhe kernel templates.
- The same source also backs the default `bench_ntt_cuda` binary used by `run_cuda_sweep.sh`.

## VTune
If VTune is installed on the machine:
```bash
cd /home/ringlpn
./scripts/run_vtune_hotspots.sh
./scripts/run_vtune_memory.sh
```

## Notes
- Uses NFLLib native primes for each bitwidth via poly_from_modulus.
- Default sweep: n in {1024, 2048, 4096, 8192, 16384}, qbits in {30, 60}
- Iterations: 10,000; Warmup: 1,000
- Coefficients use uint32_t with 30-bit primes; qbits=60 aggregates two moduli.
- The GPU q=32 path deliberately diverges from the CPU NFLLib uint32_t cutoff so larger `n` can be measured.
- The GPU q=64 path now uses a single 62-bit prime and a 64-bit Montgomery specialization built on `__umul64hi()`.
- The GPU q=128 path now uses two 62-bit primes and validates the CRT residue lanes; CPU-side NFLLib remains the comparison anchor for cross-checking q128 timings.

## Ring-LPN VOLE prototype
The repository now also includes an initial standalone Ring-LPN VOLE prototype benchmark that reuses the promoted cheddar-derived GPU PolyMul path.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_vole_bench.sh
./bin/bench_vole_ringlpn --n 8192 --qbits 32 --m 4 --c 2 --iters 10 --warmup 2

# Optional full sweep for abstract/supporting tables
./scripts/run_vole_sweep.sh
QBITS=64 ./scripts/run_vole_sweep.sh
QBITS=128 ./scripts/run_vole_sweep.sh
```

Notes:
- This is a correctness-first prototype for the Section 5.3 Ring-LPN VOLE-style expansion layer, not a full SPFSS-backed degree-1 correlation implementation yet.
- The current input mode is `synthetic_mpvole`, meaning the benchmark synthesizes MPVOLE-consistent inputs locally and validates the relation `z = y + x * Delta` coefficient-wise.
- The prototype reuses the promoted GPU polynomial multiplication path from `src/bench_ntt_cuda_cheddar.cu` instead of introducing a separate CUDA implementation.
- Requested `qbits=32` maps to actual `qbits=30`, requested `qbits=64` maps to actual `qbits=62`, and requested `qbits=128` maps to actual `qbits=124` with two q62 CRT limbs in the same flattened Cheddar launch schedule.
- The prototype is intentionally scoped for bring-up and benchmarking of the algebraic expansion step; SPFSS key generation and evaluation are still external to this harness.
- The default sweep emits `results/vole/vole_gpu_q32_m32_c2_w64.csv`, `results/vole/vole_gpu_q32_m32_c2_w64.md`, and the q64/q128 counterparts when run with `QBITS=64` or `QBITS=128`. A smaller q128 CRT smoke sweep is saved as `results/vole/vole_gpu_q128_smoke.md`.

## Figure 2 GPU OLE artifact
The repository also includes a standalone GPU artifact for the Figure 2 SPFSS-based Ring-LPN OLE path.

Build and run inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_ole_cuda_bench.sh

# Fast correctness smoke: N=8192, c=2, t=8, one timed iteration.
SMOKE=1 ./scripts/run_ole_sweep.sh

# Paper-aligned regular-noise smoke: SPFSS domain is 2N/t.
SMOKE=1 NOISE=regular ./scripts/run_ole_sweep.sh

# Bounded first-pass uniform-noise sweep: N in {8192, 16384}, c=2, t=64.
./scripts/run_ole_sweep.sh

# Bounded regular-noise sweep with grouped SPFSS domains of size 2N/t.
NOISE=regular ./scripts/run_ole_sweep.sh
```

Outputs:
- `results/ole/ole_gpu_q64_uniform_c2_t8_smoke.csv` and `.md` for the smoke run
- `results/ole/ole_gpu_q64_regular_c2_t8_smoke.csv` and `.md` for the regular-noise smoke run
- `results/ole/ole_gpu_q64_regular_c2_t64.csv` and `.md` for the bounded regular-noise run
- `results/ole/ole_gpu_q64_uniform_c2_t64.csv` and `.md` for the bounded sweep

Notes:
- This artifact uses the promoted single 62-bit prime and reports requested `qbits=64`, actual `qbits=62`.
- Noise can be `uniform` or `regular`. Regular noise picks one position per bucket and uses grouped SPFSS domains of size `2N/t`.
- The SPFSS path uses a new `Z_p` DPF payload path in `src/gpu_spfss_zp.cuh`; the existing packed one-bit `gpu_dpf.cu` callers are unchanged.
- The benchmark validates `z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N+1)` and intentionally stops before OLE-to-Beaver conversion or Orca FC integration.

## Ring-polynomial linear OLE artifact
The repository now includes a standalone linear-layer artifact that converts two Figure 2 OLE instances into a Beaver product and sums those products into a matrix multiplication over `Z_p[X]/(X^N+1)`.

Build and run inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_linear_ole_bench.sh
./scripts/run_linear_ole_sweep.sh
NOISE=regular ./scripts/run_linear_ole_sweep.sh
```

Default smoke output:
- `results/linear_ole/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.csv`
- `results/linear_ole/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md`
- `results/linear_ole/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.csv`
- `results/linear_ole/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.md`

Notes:
- The default smoke validates a `2 x 2` by `2 x 2` ring-polynomial matrix product using 8 ring products and 16 OLE instances.
- Each `A[row,k]` and `B[k,col]` operand share is generated once and reused across all products, and the CSV reports `shared_operands=1` when that regression check passes.
- The regular-noise smoke uses SPFSS domain `2N/t` and validates the same Beaver relation.
- This is a dated OLE-to-Beaver component artifact, not a first-implementation claim. The tiny FC key-writer below is a historical Orca online-integration smoke; use `CLAUDE.md` for the current two-process FC/Conv boundary.

## Orca Zp-to-Z2k bridge smoke
The repository includes a host-only bridge test for a dated Orca-facing scalar conversion boundary after the ring-polynomial OLE-to-Beaver artifact.

Build and run from the host repository root:
```bash
GPU-MPC/ringlpn/scripts/build_orca_zp_bridge_test.sh
GPU-MPC/ringlpn/scripts/run_orca_zp_bridge_test.sh
```

Outputs:
- `results/orca_fc/orca_zp_bridge_constant_scalar.csv`
- `results/orca_fc/orca_zp_bridge_constant_scalar.md`
- `results/reports/orca_zp_bridge_handoff.md`

Notes:
- Reducing each `Z_p` share independently modulo `2^bw` is wrong when the hidden prime carry is one.
- The exact dealer/oracle correction is `r0 = z0 mod 2^bw`, `r1 = z1 - m*p mod 2^bw`, where `m = floor((z0 + z1) / p)`.
- Constant-polynomial scalar packing is validated only under the explicit no-prime-wrap bound `inner * value_bound^2 < p`.
- The smoke intentionally records a q62/full-32-bit counterexample, so unrestricted 32-bit Orca products are not claimed under the current single-prime path.

## Orca FC Ring-LPN v1 demo (historical component)
The repository includes a forward-only Orca FC demo for the bounded q62 bridge and one q128 full-`uint32` control. It is not the live two-process FC/Conv path.

Build and run inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_orca_fc_ringlpn_demo.sh
./scripts/run_orca_fc_ringlpn_demo.sh
```

Outputs:
- `results/orca_fc/orca_fc_ringlpn_demo_bounded_suite.csv`
- `results/orca_fc/orca_fc_ringlpn_demo_bounded_suite.md`
- `results/reports/orca_fc_ringlpn_demo_memo.md`
- `results/outreach/professor_ringlpn_orca_fc_deliverable_2026_05_15.md`

Notes:
- The five-case default suite covers q64 `2x2x2`, `2x3x2`, and `3x2x2` at
  `bw=16`; q64 bounded `2x2x3` at `bw=32`; and q128 `2x2x2` at `bw=32`
  with full `uint32` bounds. All use `poly_n=8192,c=2,t=8`.
- This dated demo generates both party buffers in one dealer call and writes
  `A`, `B`, `C_masked` raw additive shares with no truncation bytes.
- It calls `gpuMatmulBeaver` unchanged and checks both its Ring-LPN-style writer
  and stock `gpuKeygenMatmul` reconstruction under the same logical inputs.
- It is forward FC component evidence only. The current two-process
  conversion/full-linear/full-graph disposition is documented in `CLAUDE.md`.

## Paper checkpoint smoke
Run the host-only gate from the repository root:
```bash
GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh
```

The complete default GPU/full-graph gate needs three visible GPU roles:
```bash
cd GPU-MPC/ringlpn
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>,<gpu-c> \
PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh
```

Host mode checks manifests/contracts, private-file and SHAKE controls, host OLE
and bridge/conversion/truncation components, and host/two-process DPF keygen.
GPU mode additionally rebuilds and approval-checks the FC/Conv adapters, checks
all 21 shapes, exercises DPF GPU validation, q64/q128 uniform/regular OLE,
linear OLE, the five-case demo, ideal/real transcript references, and—by
default—the fresh 21-record full ResNet18 graph. Only a zero exit ending
`[paper-smoke] ALL GATES PASS` is the complete-gate marker.

## Standalone DPF online key generation benchmark
The repository also includes a standalone DPF online key generation benchmark for the memory-efficiency track. This benchmark lives outside `ringlpn/src`, but its sweep artifacts are written into `ringlpn/results` so they can be used alongside the Ring-LPN VOLE and NTT results.

Build and run inside the CUDA-enabled container from the project root:
```bash
cd /home

# Set GPU_ARCH explicitly if it is not already exported.
make GPU_ARCH=89 dpf_online_keygen

# Quick smoke test
./tests/fss/dpf_online_keygen --bin 16 --n 8192 --chunk-size 4096 --iters 3 --warmup 1 --csv-header

# Full abstract-ready sweep
python3 scripts/run_dpf_online_keygen_sweep.py
```

Outputs:
- ringlpn/results/dpf/dpf_online_keygen_bin16_chunk8192.csv
- ringlpn/results/dpf/dpf_online_keygen_bin16_chunk8192.md

Notes:
- This is a standalone systems benchmark, not yet an end-to-end Orca or SPFSS-backed integration.
- The current abstract-ready sweep uses eval-all keys with `bin=16` and `chunk_size=8192`.
- The current sweep shows one-shot full pair-key footprint growing from `2.81 MiB` to `360.00 MiB`, while chunked generation holds peak pair-key footprint to `2.81 MiB`, reaching about `128x` peak-footprint reduction at `n=1048576` with about `1.885x` key-generation time overhead.
- `initGPUMemPool()` prints `reserved memory:` to stdout; the sweep script filters those lines before writing CSV.
