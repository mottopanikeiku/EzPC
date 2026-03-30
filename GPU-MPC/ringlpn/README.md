# Ring-LPN Benchmarks (CPU + GPU)

This folder is a standalone Ring-LPN benchmarking harness. It is separate from ORCA.

## Layout
- src/bench_ntt.cpp: NFLLib CPU microbenchmark (NTT, INTT, PolyMul)
- src/bench_ntt_cuda.cu: CUDA NTT benchmark for requested q=32 using a 30-bit prime, runtime root derivation, and batching
- scripts/setup_nfl.sh: clone + build NFLLib
- scripts/build_bench.sh: build the benchmark
- scripts/run_sweep.sh: run 10 configs and generate CSV + Markdown
- scripts/build_cuda_bench.sh: build the CUDA benchmark
- scripts/run_cuda_single.sh: run a CPU vs GPU spot check for requested q=32 at n in {8192, 16384, 32768}
- scripts/run_cuda_sweep.sh: run the requested q=32 CUDA sweep with batching and generate CSV + Markdown
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

## CUDA q=32 sweep
The current GPU deliverable is a batched CUDA NTT path for requested `q=32`, realized with one 30-bit prime that supports `n` through `2^20`.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_bench.sh
./scripts/build_cuda_bench.sh
./scripts/run_cuda_sweep.sh
```

Outputs:
- results/ntt_gpu_q32.csv
- results/ntt_gpu_q32_unsupported.csv
- results/ntt_gpu_q32.md

Notes:
- The CUDA path implements negacyclic NTT with `phi` / `invphi` preprocessing and Montgomery modular multiplication.
- `bench_ntt_cuda` accepts `--n`, `--qbits 30|32`, `--batch`, `--iters`, and `--warmup`.
- Requested `qbits=32` currently maps to actual `qbits=30` on GPU.
- The selected 30-bit prime supports the full `n in {8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}` sweep.
- `run_cuda_single.sh` remains available for spot CPU-vs-GPU comparisons at the CPU-supported points up to `n=32768`.

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
