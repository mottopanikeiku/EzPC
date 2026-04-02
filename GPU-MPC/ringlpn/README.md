# Ring-LPN Benchmarks (CPU + GPU)

This folder is a standalone Ring-LPN benchmarking harness. It is separate from ORCA.

## Layout
- src/bench_ntt.cpp: NFLLib CPU microbenchmark (NTT, INTT, PolyMul)
- src/bench_ntt_cuda.cu: legacy CUDA NTT benchmark for requested q=32 using a 30-bit prime, runtime root derivation, and batching
- src/bench_ntt_cuda_cheddar.cu: primary CUDA benchmark, extracted from cheddar-fhe and adapted to the Ring-LPN harness
- scripts/setup_nfl.sh: clone + build NFLLib
- scripts/build_bench.sh: build the benchmark
- scripts/run_sweep.sh: run 10 configs and generate CSV + Markdown
- scripts/build_cuda_bench.sh: build the primary CUDA benchmark (cheddar-derived implementation)
- scripts/build_cuda_bench_cheddar.sh: build an explicit standalone cheddar-derived binary for side-by-side checks
- scripts/build_cuda_bench_legacy.sh: build the legacy CUDA benchmark path
- scripts/run_cuda_single.sh: run a CPU vs GPU spot check for requested q=32 at n in {8192, 16384, 32768}
- scripts/run_cuda_sweep.sh: run the requested q=32 or q=64 CUDA sweep with batching and generate CSV + Markdown
- scripts/run_cuda_sweep_legacy.sh: run the legacy CUDA q=32 sweep for baseline comparison
- scripts/summarize_cuda_results.py: summarize CUDA sweep outputs
- scripts/run_vtune_hotspots.sh: VTune hotspots wrapper for CPU benchmark
- scripts/run_vtune_memory.sh: VTune memory-access wrapper for CPU benchmark
- results/: output files

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
- results/ntt_cpu.csv
- results/ntt_cpu.md

## CUDA q=32 / q=64 sweeps
The current primary GPU deliverable is a batched CUDA NTT path for requested `q=32` and `q=64`, realized with one 30-bit or one 62-bit prime that supports `n` through `2^20`.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_bench.sh
./scripts/build_cuda_bench.sh
./scripts/run_cuda_sweep.sh

# Optional q=64 sweep
QBITS=64 ./scripts/run_cuda_sweep.sh
```

Outputs:
- results/ntt_gpu_q32.csv
- results/ntt_gpu_q32_unsupported.csv
- results/ntt_gpu_q32.md
- results/ntt_gpu_q64.csv
- results/ntt_gpu_q64_unsupported.csv
- results/ntt_gpu_q64.md

Notes:
- The primary CUDA path is now the cheddar-derived implementation in `src/bench_ntt_cuda_cheddar.cu`, built into `bin/bench_ntt_cuda` by `scripts/build_cuda_bench.sh`.
- The promoted path keeps the existing CLI and CSV contract while replacing the internal kernel implementation with the cheddar-derived two-phase NTT / INTT structure.
- `bench_ntt_cuda` accepts `--n`, `--qbits 30|32|64`, `--batch`, `--iters`, and `--warmup`.
- Requested `qbits=32` maps to actual `qbits=30` on GPU, and requested `qbits=64` maps to actual `qbits=62`.
- The selected single-prime parameter sets support the full `n in {8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}` sweep.
- `run_cuda_single.sh` remains available for spot CPU-vs-GPU comparisons at the CPU-supported points up to `n=32768`.

## Legacy CUDA baseline
The original hand-written CUDA path is retained for comparison and regression tracking.

Build and sweep the legacy path inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_cuda_bench_legacy.sh
./scripts/run_cuda_sweep_legacy.sh
```

Legacy outputs:
- results/ntt_gpu_q32_legacy.csv
- results/ntt_gpu_q32_legacy_unsupported.csv
- results/ntt_gpu_q32_legacy.md

## CUDA q=32 cheddar extract
The repository also includes an explicit standalone cheddar-derived binary, `bench_ntt_cuda_cheddar`, which uses the same promoted source implementation but builds it under a separate name for side-by-side testing.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_cuda_bench_cheddar.sh
./bin/bench_ntt_cuda_cheddar --n 8192 --qbits 32 --batch 1 --iters 100 --warmup 10
```

Notes:
- The extracted kernels are adapted for the single-prime Ring-LPN benchmark layout, so they reuse one modulus across many batches instead of cheddar-fhe's original prime-limb dimension.
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
- The GPU q=32 path deliberately diverges from the CPU NFLLib uint32_t cutoff so larger `n` can be measured before the 64-bit and CRT phases land.
- The GPU q=64 path now uses a single 62-bit prime and a 64-bit Montgomery specialization built on `__umul64hi()`.
- The current roadmap is: primary cheddar-derived single-prime q=32 and q=64 paths, then dual-prime CRT support for requested `q=128`.
