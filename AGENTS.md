# GPU-MPC Workspace Guide For Agents

This workspace contains many sibling projects, but current work is usually only about [GPU-MPC](GPU-MPC).

If a task does not explicitly mention another top-level project, treat everything outside [GPU-MPC](GPU-MPC) as out of scope.

This file is meant to answer four questions quickly for a new agent:

1. What is the real project boundary here?
2. How does the Docker and filesystem mapping work?
3. What are the major GPU-MPC pipelines?
4. Which files and directories matter first for Orca, Sigma, and Ring-LPN work?

## 1. Scope

The repository root is a larger CrypTFlow-era monorepo, but for day-to-day work the meaningful project is [GPU-MPC](GPU-MPC).

Unless the user explicitly asks about other subsystems, the practical rule is:

- ignore [Athos](Athos), [SCI](SCI), [FSS](FSS), [Porthos](Porthos), [Beacon](Beacon), [OnnxBridge](OnnxBridge), [sytorch](sytorch), and other siblings,
- focus on [GPU-MPC](GPU-MPC),
- and remember that the active workflows are mostly:
  - Orca training and inference,
  - Orca local loopback and profiling,
  - Ring-LPN CPU/GPU NTT benchmarking,
  - and occasionally Sigma.

## 2. Container Model

The root entrypoint for local GPU work is [start](start).

Important behavior:

- it launches or attaches to a Docker container named `orca-dev`,
- it mounts only [GPU-MPC](GPU-MPC) into the container,
- inside the container that mount appears as `/home`.

That means host paths and container paths are different:

- host repo root: `/home/fatih/EzPC`
- host GPU project root: `/home/fatih/EzPC/GPU-MPC`
- container project root: `/home`

Most important path translations:

- [GPU-MPC/experiments/orca](GPU-MPC/experiments/orca) -> `/home/experiments/orca`
- [GPU-MPC/orca_runner](GPU-MPC/orca_runner) -> `/home/orca_runner`
- [GPU-MPC/ringlpn](GPU-MPC/ringlpn) -> `/home/ringlpn`
- [GPU-MPC/scripts](GPU-MPC/scripts) -> `/home/scripts`
- [GPU-MPC/keys](GPU-MPC/keys) -> `/home/keys`

This is the most important operational fact in the workspace. A large fraction of debugging confusion comes from forgetting that container commands must usually run under `/home/...`, not the host path.

## 3. GPU-MPC Identity

[GPU-MPC/README.md](GPU-MPC/README.md) describes GPU-MPC as the implementation of protocols from the Orca and SIGMA papers.

In practice, the project is organized as:

- protocol backends and shared GPU runtime code,
- experiment binaries and paper harnesses,
- local loopback runners and profiling scripts,
- benchmark harnesses,
- and large external dependencies such as CUTLASS and Sytorch.

The most important active subprojects are:

- [GPU-MPC/experiments/orca](GPU-MPC/experiments/orca): formal Orca training and inference binaries plus experiment harness,
- [GPU-MPC/orca_runner](GPU-MPC/orca_runner): local single-machine loopback automation and logs,
- [GPU-MPC/ringlpn](GPU-MPC/ringlpn): standalone CPU/GPU NTT benchmark harness,
- [GPU-MPC/backend](GPU-MPC/backend): backend protocol headers,
- [GPU-MPC/utils](GPU-MPC/utils): shared GPU memory, file I/O, comms, and helper utilities.

## 4. Filesystem Snapshot

Current high-signal directory map for GPU-MPC:

```text
GPU-MPC
GPU-MPC/backend
GPU-MPC/experiments
GPU-MPC/experiments/orca
GPU-MPC/experiments/sigma
GPU-MPC/ext
GPU-MPC/ext/cutlass
GPU-MPC/ext/sytorch
GPU-MPC/fss
GPU-MPC/fss/dcf
GPU-MPC/keys
GPU-MPC/keys/P0
GPU-MPC/keys/P1
GPU-MPC/nn
GPU-MPC/nn/orca
GPU-MPC/orca_runner
GPU-MPC/orca_runner/logs
GPU-MPC/ringlpn
GPU-MPC/ringlpn/bin
GPU-MPC/ringlpn/extern
GPU-MPC/ringlpn/results
GPU-MPC/ringlpn/scripts
GPU-MPC/ringlpn/src
GPU-MPC/scripts
GPU-MPC/tests
GPU-MPC/tests/fss
GPU-MPC/tests/nn
GPU-MPC/utils
```

Quick meaning of the main paths:

- [GPU-MPC/Makefile](GPU-MPC/Makefile): main NVCC build graph for Orca, Sigma, Piranha, and many tests.
- [GPU-MPC/setup.sh](GPU-MPC/setup.sh): dependency/bootstrap helper for CUTLASS, Sytorch, datasets, and output directories.
- [GPU-MPC/experiments/orca](GPU-MPC/experiments/orca): Orca binaries, configs, outputs, datasets, and experiment harness.
- [GPU-MPC/experiments/sigma](GPU-MPC/experiments/sigma): Sigma binaries and paper path.
- [GPU-MPC/orca_runner](GPU-MPC/orca_runner): local loopback scripts and logs.
- [GPU-MPC/ringlpn](GPU-MPC/ringlpn): standalone benchmarking track for Ring-LPN NTT work.
- [GPU-MPC/scripts](GPU-MPC/scripts): profiling and summary helpers for Orca.
- [GPU-MPC/backend](GPU-MPC/backend): backend abstractions such as Orca, Sigma, and Piranha headers.
- [GPU-MPC/utils](GPU-MPC/utils): shared low-level GPU utilities.
- [GPU-MPC/ext/cutlass](GPU-MPC/ext/cutlass): CUTLASS dependency.
- [GPU-MPC/ext/sytorch](GPU-MPC/ext/sytorch): Sytorch plus LLAMA, cryptoTools, bitpack, and related dependencies.

## 5. Top-Level GPU-MPC Files

### 5.1 Build Entry Points

- [GPU-MPC/README.md](GPU-MPC/README.md): top-level build and Docker notes.
- [GPU-MPC/Makefile](GPU-MPC/Makefile): actual build targets.
- [GPU-MPC/setup.sh](GPU-MPC/setup.sh): setup script for dependencies, datasets, and output directories.
- [GPU-MPC/Dockerfile_Gen](GPU-MPC/Dockerfile_Gen): image build path for GPU-MPC environment setup.

### 5.2 Runtime/Experiment Entry Points

- [GPU-MPC/experiments/orca/README.md](GPU-MPC/experiments/orca/README.md): formal Orca usage notes.
- [GPU-MPC/experiments/orca/run_experiment.py](GPU-MPC/experiments/orca/run_experiment.py): figure and table harness.
- [GPU-MPC/orca_runner/run_and_log.sh](GPU-MPC/orca_runner/run_and_log.sh): local loopback end-to-end runner.
- [GPU-MPC/orca_runner/run_remaining.sh](GPU-MPC/orca_runner/run_remaining.sh): follow-on local runs for remaining models.
- [GPU-MPC/scripts/run_orca_profiling.sh](GPU-MPC/scripts/run_orca_profiling.sh): ORCA profiling harness.

### 5.3 Benchmark Entry Points

- [GPU-MPC/ringlpn/README.md](GPU-MPC/ringlpn/README.md): benchmark harness guide.
- [GPU-MPC/ringlpn/src/bench_ntt.cpp](GPU-MPC/ringlpn/src/bench_ntt.cpp): CPU NFLLib benchmark.
- [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu): primary CUDA benchmark source, extracted from cheddar-fhe and adapted locally.
- [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu): preserved legacy CUDA benchmark path.
- [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md): current Ring-LPN status and roadmap handoff.
- [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md): extraction rationale and earlier batch-1 comparison study.

## 6. Build Pipeline

### 6.1 Makefile Model

[GPU-MPC/Makefile](GPU-MPC/Makefile) is the central build graph.

Important facts from the file:

- the compiler is `nvcc`,
- the build uses `-std=c++17`,
- architecture is selected by `GPU_ARCH`,
- include/lib paths point into CUTLASS and Sytorch,
- common runtime utilities come from [GPU-MPC/utils](GPU-MPC/utils).

Core libraries linked by most targets:

- `sytorch`
- `cryptoTools`
- `LLAMA`
- `bitpack`
- CUDA runtime libs
- SCI floating-point libs for some paths

Important build targets:

- `make orca`: builds `orca_dealer`, `orca_evaluator`, `orca_inference`, `orca_inference_u32`, and `piranha`
- `make sigma`: builds Sigma
- individual FSS/NN test binaries under [GPU-MPC/tests](GPU-MPC/tests)

Primary Orca binaries built by the Makefile:

- [GPU-MPC/experiments/orca/orca_dealer.cu](GPU-MPC/experiments/orca/orca_dealer.cu)
- [GPU-MPC/experiments/orca/orca_evaluator.cu](GPU-MPC/experiments/orca/orca_evaluator.cu)
- [GPU-MPC/experiments/orca/orca_inference.cu](GPU-MPC/experiments/orca/orca_inference.cu)
- [GPU-MPC/experiments/orca/piranha.cu](GPU-MPC/experiments/orca/piranha.cu)

### 6.2 setup.sh Behavior

[GPU-MPC/setup.sh](GPU-MPC/setup.sh) does the following:

1. updates submodules,
2. installs gcc-9/g++-9 and core build dependencies,
3. builds CUTLASS under [GPU-MPC/ext/cutlass](GPU-MPC/ext/cutlass),
4. builds Sytorch under [GPU-MPC/ext/sytorch](GPU-MPC/ext/sytorch),
5. downloads CIFAR-10 into [GPU-MPC/experiments/orca/datasets/cifar-10](GPU-MPC/experiments/orca/datasets/cifar-10),
6. builds and runs `share_data`,
7. creates Orca and Sigma output directories,
8. installs `matplotlib`.

Important gotcha: the script currently leaves `make orca` commented out. Running `setup.sh` does not automatically build Orca binaries.

## 7. Orca Pipeline

There are three practically important Orca workflows:

1. the formal paper harness,
2. the local loopback runner,
3. and the profiling runner.

### 7.1 Formal Orca Experiment Harness

Primary files:

- [GPU-MPC/experiments/orca/config.json](GPU-MPC/experiments/orca/config.json)
- [GPU-MPC/experiments/orca/run_experiment.py](GPU-MPC/experiments/orca/run_experiment.py)
- [GPU-MPC/experiments/orca/output](GPU-MPC/experiments/orca/output)

The execution model is two-party and each party has:

- a dealer configuration: GPU id and key directory,
- an evaluator configuration: GPU id and peer IP.

The harness runs experiments with:

- `--figure`
- `--table`
- `--all`
- `--party 0|1`

Output layout:

- figures land under `output/P<party>/Fig<id>`
- tables land under `output/P<party>/Table<n>`
- logs live under the corresponding `logs/` subdirectories

High-level mapping from [GPU-MPC/experiments/orca/run_experiment.py](GPU-MPC/experiments/orca/run_experiment.py):

- Figure 5a: CNN2 loss curve on MNIST
- Figure 5b: CNN3 loss curve on CIFAR-10
- Table 3: training summaries for CNN2 / CNN3-2e / CNN3-5e
- Table 4: P-SecureML, P-LeNet, P-AlexNet, P-VGG16 training and Piranha inference summaries
- Table 6: CNN2, ModelB, AlexNet, CNN3 training summaries
- Table 7: CNN2 and CNN3 training vs inference summaries
- Table 8: training and inference key-size summaries
- Table 9: inference summaries for VGG16 / ResNet18 / ResNet50 across bitwidth/scale settings

Binary-level flow is usually:

1. dealer generates keys into the configured key directory,
2. evaluator consumes those keys while communicating with its peer,
3. logs and metrics are written under the appropriate output subtree,
4. keys are removed after use by the harness.

### 7.2 Local Orca Loopback Pipeline

Primary files:

- [GPU-MPC/orca_runner/run_and_log.sh](GPU-MPC/orca_runner/run_and_log.sh)
- [GPU-MPC/orca_runner/run_remaining.sh](GPU-MPC/orca_runner/run_remaining.sh)
- [GPU-MPC/orca_runner/logs](GPU-MPC/orca_runner/logs)

This is the most important operational path for local single-machine testing.

It assumes the container layout:

- workdir: `/home/experiments/orca`
- logs: `/home/orca_runner/logs`
- keys: `/home/keys/P0` and `/home/keys/P1`

[GPU-MPC/orca_runner/run_and_log.sh](GPU-MPC/orca_runner/run_and_log.sh) currently does roughly this:

- training: `P-SecureML`, `P-LeNet`, `P-AlexNet`
- inference: `CNN2`, `CNN3`, `VGG16`
- training/perf: `CNN2-perf`
- optional larger run: `CNN3-perf` only if disk space is sufficient

[GPU-MPC/orca_runner/run_remaining.sh](GPU-MPC/orca_runner/run_remaining.sh) continues with more inference and training runs, including `ModelB` and `AlexNet`.

Operational pattern:

1. remove stale key files for the model,
2. run dealer for P0 and P1,
3. run evaluator pair on localhost,
4. append key sizes and tail summaries into `master.log`,
5. remove keys again.

### 7.3 Orca Profiling Pipeline

Primary files:

- [GPU-MPC/scripts/run_orca_profiling.sh](GPU-MPC/scripts/run_orca_profiling.sh)
- [GPU-MPC/scripts/summarize_orca_results.py](GPU-MPC/scripts/summarize_orca_results.py)

This path is meant for instrumented ORCA runs rather than paper-table reproduction.

Important configuration in the script:

- `WORKDIR=/home/experiments/orca`
- `LOG_DIR=/home/orca_runner/logs`
- `REPORT_DIR=/home/orca_runner/reports`
- `RUN_DD`, `RUN_DEALER`, `RUN_EVAL`, `RUN_INFERENCE`, `RUN_NSYS_TRAIN`, `RUN_NSYS_INF` switches

It can do all of the following:

- direct `dd` bandwidth checks over existing key files,
- dealer-only runs,
- evaluator runs,
- inference runs,
- Nsight Systems profiling,
- summary generation into markdown and CSV.

## 8. Sigma

Sigma is present under [GPU-MPC/experiments/sigma](GPU-MPC/experiments/sigma) and built with `make sigma`.

For most current work it is secondary to Orca and Ring-LPN, but agents should know:

- Sigma is part of the same build system,
- Sigma output directories are created by [GPU-MPC/setup.sh](GPU-MPC/setup.sh),
- and shared utilities/backends under [GPU-MPC/utils](GPU-MPC/utils) and [GPU-MPC/backend](GPU-MPC/backend) can matter to both Orca and Sigma.

## 9. Ring-LPN Pipeline

Primary path: [GPU-MPC/ringlpn](GPU-MPC/ringlpn)

This harness is separate from Orca. Do not treat it as part of the Orca training/inference workflow.

### 9.1 Core Files

- [GPU-MPC/ringlpn/src/bench_ntt.cpp](GPU-MPC/ringlpn/src/bench_ntt.cpp): CPU NFLLib benchmark
- [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu): promoted primary CUDA benchmark
- [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu): preserved legacy CUDA benchmark baseline

### 9.2 Shell Pipeline

Important scripts under [GPU-MPC/ringlpn/scripts](GPU-MPC/ringlpn/scripts):

- `setup_nfl.sh`: clone and build NFLlib
- `build_bench.sh`: build CPU benchmark
- `run_sweep.sh`: run CPU sweep and generate markdown summary
- `build_cuda_bench.sh`: build the promoted cheddar-derived main CUDA benchmark
- `build_cuda_bench_cheddar.sh`: build the same cheddar-derived source under an explicit side-by-side binary name
- `build_cuda_bench_legacy.sh`: build the preserved legacy CUDA benchmark
- `run_cuda_sweep.sh`: promoted q=32 or q=64 CUDA sweep
- `run_cuda_sweep_legacy.sh`: legacy q=32 CUDA sweep
- `run_cuda_single.sh`: CPU-vs-GPU spot check at CPU-overlap points
- `run_vtune_hotspots.sh` and `run_vtune_memory.sh`: CPU profiling wrappers
- `summarize_results.py`, `summarize_cuda_results.py`, `summarize_cpu_gpu_4096.py`: reporting scripts

### 9.3 Current Validated State

Current active benchmark state:

- CPU benchmark supports requested `qbits` 32, 64, and 128 by mapping to actual NFLLib sizes 30, 62, and 124.
- CPU requested `qbits=32` is only feasible up to `n=32768` because NFLLib uint32 mode stops there.
- The promoted main GPU benchmark is now [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu), built by [GPU-MPC/ringlpn/scripts/build_cuda_bench.sh](GPU-MPC/ringlpn/scripts/build_cuda_bench.sh) into `bin/bench_ntt_cuda`.
- The promoted GPU benchmark supports requested `qbits=32|64`, mapping them to actual `qbits=30|62` with one prime per run.
- The promoted GPU q=32 and q=64 paths both support `n` in `{8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}`.
- The legacy CUDA path in [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu) is still available for q=32 comparison and regression tracking.
- The current promoted CUDA benchmark is batched and reports `requested_qbits`, `actual_qbits`, `batch_size`, `validation`, and `correct` in CSV.
- `run_cuda_single.sh` is intentionally limited to the CPU-overlap points up to `n=32768`.
- Current promoted result summaries live in [GPU-MPC/ringlpn/results/ntt_gpu_q32.md](GPU-MPC/ringlpn/results/ntt_gpu_q32.md) and [GPU-MPC/ringlpn/results/ntt_gpu_q64.md](GPU-MPC/ringlpn/results/ntt_gpu_q64.md).
- Legacy comparison results live in [GPU-MPC/ringlpn/results/ntt_gpu_q32_legacy.md](GPU-MPC/ringlpn/results/ntt_gpu_q32_legacy.md).
- The most complete human handoff for Ring-LPN is [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md).

Inside the container, the benchmark workdir is `/home/ringlpn`.

Important output locations:

- [GPU-MPC/ringlpn/results](GPU-MPC/ringlpn/results)
- [GPU-MPC/ringlpn/bin](GPU-MPC/ringlpn/bin)

### 9.4 Benchmark Roadmap Context

The active generalization roadmap for the main CUDA benchmark is:

1. complete generalized single-prime q=32 support over `8192` to `1048576`,
2. complete generalized single-prime q=64 support with 64-bit Montgomery kernels using `__umul64hi()`,
3. add requested `q=128` via two independent 64-bit NTT tracks and CRT composition.

At the moment, steps 1 and 2 are complete on the promoted main path. Step 3 is the next active research and engineering target.

### 9.5 Current Mission Handoff

If a new agent is asked to continue the current Ring-LPN mission, the practical mission is:

1. treat the promoted cheddar-derived path as the default GPU implementation,
2. preserve the legacy CUDA path as a comparison baseline,
3. avoid reworking the CPU baseline unless a validation issue requires it,
4. carry the GPU path from single-prime q=32/q=64 to dual-prime CRT q=128.

What is already done:

- CPU NFLLib benchmark and full sweep are complete.
- Promoted cheddar-derived GPU q=32 sweep is complete and validated.
- Promoted cheddar-derived GPU q=64 sweep is complete and validated.
- Legacy GPU q=32 sweep is preserved for comparison.
- The current GPU-vs-CPU comparison is strong: q=32 overlap points show roughly `146x` to `171x` per-polynomial PolyMul speedups over CPU, and q=64 points show roughly `48x` to `220x` per-polynomial PolyMul speedups over CPU.
- The promoted main path is consistently faster than the legacy baseline, with the largest observed per-polynomial gain near `6x` at `n=65536` in the adaptive sweep.

What is not done:

- no promoted GPU path yet exists for requested `q=128`,
- no multi-prime scheduling layer exists yet in the promoted path,
- no CRT recomposition path exists yet in the promoted path,
- VTune wrappers exist, but VTune is not guaranteed to be installed in the active container.

Files the next agent should read first for Ring-LPN continuation:

- [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md)
- [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md)
- [GPU-MPC/ringlpn/README.md](GPU-MPC/ringlpn/README.md)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu)
- [GPU-MPC/ringlpn/src/bench_ntt.cpp](GPU-MPC/ringlpn/src/bench_ntt.cpp)
- [GPU-MPC/ringlpn/scripts/build_cuda_bench.sh](GPU-MPC/ringlpn/scripts/build_cuda_bench.sh)
- [GPU-MPC/ringlpn/scripts/build_cuda_bench_legacy.sh](GPU-MPC/ringlpn/scripts/build_cuda_bench_legacy.sh)
- [GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh](GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh)
- [GPU-MPC/ringlpn/scripts/run_cuda_sweep_legacy.sh](GPU-MPC/ringlpn/scripts/run_cuda_sweep_legacy.sh)

Commands worth knowing for continuation work:

- build promoted CPU and GPU benchmarks inside the container by running `./scripts/build_bench.sh` and `./scripts/build_cuda_bench.sh` under `/home/ringlpn`,
- run the promoted q=32 sweep with `./scripts/run_cuda_sweep.sh`,
- run the promoted q=64 sweep with `QBITS=64 ./scripts/run_cuda_sweep.sh`,
- run the legacy comparison sweep with `./scripts/build_cuda_bench_legacy.sh` and `./scripts/run_cuda_sweep_legacy.sh`,
- use `./scripts/run_cuda_single.sh 8192 4` or another CPU-overlap point only for quick spot checks.

Operational cautions for continuation work:

- do Ring-LPN runtime work inside the `orca-dev` container under `/home/ringlpn`, not from the host path,
- remember that the root [.gitignore](.gitignore) ignores `*.sh`, `*.csv`, `*.txt`, and `*.out`, so helper scripts and generated summaries may not be tracked,
- do not assume the current shell scripts have Git history just because they exist in the working tree,
- treat [GPU-MPC/ringlpn/results](GPU-MPC/ringlpn/results) as the authoritative place to confirm what has actually been benchmarked.

## 10. Backend, Utils, and Tests

### 10.1 Backend Headers

Important files under [GPU-MPC/backend](GPU-MPC/backend):

- [GPU-MPC/backend/orca_base.h](GPU-MPC/backend/orca_base.h)
- [GPU-MPC/backend/orca.h](GPU-MPC/backend/orca.h)
- [GPU-MPC/backend/piranha.h](GPU-MPC/backend/piranha.h)
- [GPU-MPC/backend/sigma.h](GPU-MPC/backend/sigma.h)

These are the main backend abstractions and are the right place to start if a task is about protocol mechanics rather than experiment orchestration.

### 10.2 Shared Utilities

Important files under [GPU-MPC/utils](GPU-MPC/utils):

- [GPU-MPC/utils/gpu_mem.cu](GPU-MPC/utils/gpu_mem.cu)
- [GPU-MPC/utils/gpu_file_utils.cpp](GPU-MPC/utils/gpu_file_utils.cpp)
- [GPU-MPC/utils/sigma_comms.cpp](GPU-MPC/utils/sigma_comms.cpp)
- [GPU-MPC/utils/gpu_random.cu](GPU-MPC/utils/gpu_random.cu)

Current verified local facts from recent work:

- `gpu_mem.cu` is patched in this checkout to reserve 25 GB rather than 40 GB for the mempool,
- `gpu_file_utils.cpp` uses `O_DIRECT | O_LARGEFILE` for key reads/writes,
- key buffers are 4096-byte aligned,
- some remaining Orca overhead likely comes from repeated `moveToGPU()` calls on masks, weights, or activations.

### 10.3 Tests

[GPU-MPC/tests](GPU-MPC/tests) contains:

- [GPU-MPC/tests/fss](GPU-MPC/tests/fss)
- [GPU-MPC/tests/nn](GPU-MPC/tests/nn)

These are useful when a task is about validating kernels or protocol primitives outside the large end-to-end runners.

## 11. Important Gotchas

### 11.1 The Root .gitignore Is Aggressive

The root [.gitignore](.gitignore) ignores many file types globally, including:

- `*.sh`
- `*.csv`
- `*.txt`
- `*.out`

Practical consequence:

- shell helpers under [GPU-MPC/ringlpn/scripts](GPU-MPC/ringlpn/scripts) may not be tracked,
- generated summaries and result tables often have no Git history,
- a script that looks "local only" may simply be ignored by Git.

### 11.2 Key Material Is Huge

Orca key files can be very large. Multiple scripts assume hundreds of GB of free space or explicitly skip runs when space is low.

### 11.3 The Container Is The Real Runtime

For GPU work, commands that appear to work on the host may still be the wrong environment. The intended runtime for Orca and Ring-LPN is usually inside `orca-dev`.

### 11.4 VTune Is Not Guaranteed

The Ring-LPN VTune wrappers are valid scripts, but the current container does not necessarily have `vtune` installed.

## 12. Where To Start For Common Tasks

If the task mentions Orca training, inference, figures, or tables:

- start with [GPU-MPC/experiments/orca/README.md](GPU-MPC/experiments/orca/README.md)
- then read [GPU-MPC/experiments/orca/run_experiment.py](GPU-MPC/experiments/orca/run_experiment.py)
- then check [GPU-MPC/experiments/orca/config.json](GPU-MPC/experiments/orca/config.json)

If the task mentions local logs, loopback execution, or quick reproduction:

- start with [GPU-MPC/orca_runner/run_and_log.sh](GPU-MPC/orca_runner/run_and_log.sh)
- check [GPU-MPC/orca_runner/run_remaining.sh](GPU-MPC/orca_runner/run_remaining.sh)
- then inspect [GPU-MPC/orca_runner/logs](GPU-MPC/orca_runner/logs)

If the task mentions profiling Orca or key I/O:

- start with [GPU-MPC/scripts/run_orca_profiling.sh](GPU-MPC/scripts/run_orca_profiling.sh)
- then inspect [GPU-MPC/scripts/summarize_orca_results.py](GPU-MPC/scripts/summarize_orca_results.py)
- and the shared utilities in [GPU-MPC/utils](GPU-MPC/utils)

If the task mentions Ring-LPN, NTT, NFLLib, CUDA sweeps, or benchmark tables:

- start with [GPU-MPC/ringlpn/README.md](GPU-MPC/ringlpn/README.md)
- then read [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md)
- then read [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md)
- then inspect [GPU-MPC/ringlpn/src](GPU-MPC/ringlpn/src)
- then use [GPU-MPC/ringlpn/scripts](GPU-MPC/ringlpn/scripts)

If the task mentions Sigma:

- start with [GPU-MPC/README.md](GPU-MPC/README.md)
- then inspect [GPU-MPC/experiments/sigma](GPU-MPC/experiments/sigma)
- and the shared backend/util files.

## 13. Minimal Orientation Checklist

If dropped into this workspace cold, the fastest reliable orientation path is:

1. Read [start](start) and translate host paths to `/home/...` container paths.
2. Read [GPU-MPC/README.md](GPU-MPC/README.md) and [GPU-MPC/Makefile](GPU-MPC/Makefile).
3. Decide whether the task is Orca, Sigma, or Ring-LPN.
4. For Orca, decide whether the task is the formal harness, local loopback, or profiling runner.
5. For Ring-LPN, decide whether the task is CPU NFLLib, promoted cheddar-derived CUDA, or legacy CUDA baseline.
6. Check whether the files you care about are actually tracked, because the root ignore rules hide many script and result files.

## 14. Highest-Signal Paths For Current Work

If the task is about the currently active GPU work, these are the first paths to check:

- [start](start)
- [GPU-MPC/README.md](GPU-MPC/README.md)
- [GPU-MPC/Makefile](GPU-MPC/Makefile)
- [GPU-MPC/setup.sh](GPU-MPC/setup.sh)
- [GPU-MPC/experiments/orca](GPU-MPC/experiments/orca)
- [GPU-MPC/experiments/orca/run_experiment.py](GPU-MPC/experiments/orca/run_experiment.py)
- [GPU-MPC/orca_runner](GPU-MPC/orca_runner)
- [GPU-MPC/orca_runner/logs](GPU-MPC/orca_runner/logs)
- [GPU-MPC/scripts/run_orca_profiling.sh](GPU-MPC/scripts/run_orca_profiling.sh)
- [GPU-MPC/utils](GPU-MPC/utils)
- [GPU-MPC/ringlpn](GPU-MPC/ringlpn)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu)
- [GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh](GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh)
- [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md)
- [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md)
- [GPU-MPC/ringlpn/results](GPU-MPC/ringlpn/results)

These paths cover container entry, main build, Orca orchestration, profiling, logs, shared utilities, and the active Ring-LPN benchmark work.
