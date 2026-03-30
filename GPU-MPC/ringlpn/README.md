# Ring-LPN CPU Baseline (NFLLib)

This folder is a standalone Ring-LPN benchmarking harness. It is separate from ORCA.

## Layout
- src/bench_ntt.cpp: NFLLib CPU microbenchmark (NTT, INTT, PolyMul)
- src/bench_ntt_cuda.cu: CUDA NTT baseline for n=4096, q=30 using Montgomery arithmetic
- scripts/setup_nfl.sh: clone + build NFLLib
- scripts/build_bench.sh: build the benchmark
- scripts/run_sweep.sh: run 10 configs and generate CSV + Markdown
- scripts/build_cuda_bench.sh: build the CUDA benchmark
- scripts/run_cuda_single.sh: run CPU vs GPU comparison for n=4096, q=30
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

## CUDA baseline (single config first)
This is the first GPU deliverable: one CUDA baseline for `n=4096`, `q=30`.

Build inside the CUDA-enabled container:
```bash
cd /home/ringlpn
chmod +x scripts/*.sh

./scripts/build_bench.sh
./scripts/build_cuda_bench.sh
./scripts/run_cuda_single.sh
```

Outputs:
- results/cpu_4096_30.csv
- results/gpu_4096_30.csv
- results/cpu_gpu_4096_30.md

Notes:
- The CUDA path implements negacyclic NTT with `phi` / `invphi` preprocessing and Montgomery modular multiplication.
- The current GPU target is intentionally narrow: one defensible CPU-vs-GPU number before sweeping all 10 configs.

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
