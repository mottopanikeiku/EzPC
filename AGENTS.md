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
  - Ring-LPN VOLE prototype benchmarking and abstract support,
  - Ring-LPN Figure 2 SPFSS/OLE GPU artifact benchmarking,
  - Ring-LPN OLE-to-Beaver ring-polynomial linear-layer benchmarking,
  - Ring-LPN Orca `Z_p -> Z_{2^bw}` scalar bridge validation,
  - Ring-LPN Orca forward-FC key-writer demo validation,
  - standalone DPF online key generation benchmarking,
  - and occasionally Sigma.

For substantive implementation or benchmarking jobs, update the relevant handoff documentation before finishing the turn. At minimum, keep this `AGENTS.md` guide and the closest project-level handoff or status file accurate about what changed, how to reproduce it, what passed validation, and what remains out of scope.

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
- [GPU-MPC/ringlpn](GPU-MPC/ringlpn): standalone benchmarking track for Ring-LPN NTT, VOLE, and related abstract-support artifacts.
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
- [GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu](GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu): standalone Ring-LPN VOLE prototype benchmark built on the promoted CUDA PolyMul path.
- [GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh](GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh): standalone GPU DPF/SPFSS path with `Z_p` payloads for the Figure 2 OLE artifact.
- [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu): standalone GPU Figure 2 SPFSS/OLE benchmark over the promoted q=64 single-prime PolyMul path.
- [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu): standalone ring-polynomial linear-layer Beaver benchmark built from two Figure 2 OLEs per ring product.
- [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp): host-only Orca bridge smoke for carry-corrected `Z_p -> Z_{2^bw}` share conversion and constant-polynomial scalar packing.
- [GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu](GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu): tiny bounded q62 Orca FC key-writer demo that emits raw `A`, `B`, `C_masked` buffers, calls unchanged `gpuMatmulBeaver`, and compares against Orca `gpuKeygenMatmul`.
- [GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu](GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu): GPU SPFSS payload correctness test.
- [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu): standalone DPF online key generation benchmark.
- [GPU-MPC/scripts/run_dpf_online_keygen_sweep.py](GPU-MPC/scripts/run_dpf_online_keygen_sweep.py): DPF online key generation sweep driver.
- [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md): current Ring-LPN status and roadmap handoff.
- [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md): extraction rationale and earlier batch-1 comparison study.
- [GPU-MPC/ringlpn/results/ole_gpu_handoff.md](GPU-MPC/ringlpn/results/ole_gpu_handoff.md): current GPU Figure 2 OLE handoff, claims, caveats, commands, and next steps.
- [GPU-MPC/ringlpn/results/linear_ole_handoff.md](GPU-MPC/ringlpn/results/linear_ole_handoff.md): current OLE-to-Beaver ring-polynomial linear-layer handoff and remaining Orca integration gaps.
- [GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md](GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md): current Orca scalar bridge handoff, carry-correction argument, and q62/full-32-bit counterexample.
- [GPU-MPC/ringlpn/results/orca_fc_ringlpn_demo_memo.md](GPU-MPC/ringlpn/results/orca_fc_ringlpn_demo_memo.md): professor-facing v1 Orca FC demo memo, proof sketch, command log, and paper gaps.
- [GPU-MPC/ringlpn/results/paper_execution_next_steps.md](GPU-MPC/ringlpn/results/paper_execution_next_steps.md): current one-command smoke, hygiene notes, and paper-oriented next checkpoints.
- [GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md](GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md): bounded GPU Figure 2 OLE result summary.
- [GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md](GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md): current ring-polynomial linear-layer OLE-to-Beaver smoke result.
- [GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md](GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md): current abstract-safe support note for Ring-LPN VOLE plus DPF online key generation.
- [GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md](GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md): professor-aligned abstract outline for the current evidence.

## 6. Build Pipeline

### 6.1 Makefile Model

[GPU-MPC/Makefile](GPU-MPC/Makefile) is the central build graph.

Important facts from the file:

- the compiler is `nvcc`,
- the build uses `-std=c++17`,
- architecture is selected by `GPU_ARCH`,
- if `GPU_ARCH` is unset, plain `make` can fail with `nvcc fatal: Unsupported gpu architecture 'compute_'`,
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
- `make dpf_online_keygen`: builds the standalone DPF online key generation benchmark under `tests/fss/dpf_online_keygen`
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
- [GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu](GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu): standalone Ring-LPN VOLE prototype benchmark
- [GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh](GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh): standalone GPU DPF/SPFSS path with `Z_p` payloads for Figure 2 OLE
- [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu): standalone GPU Figure 2 SPFSS/OLE benchmark
- [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu): standalone ring-polynomial linear-layer OLE-to-Beaver benchmark
- [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp): host-only Orca `Z_p -> Z_{2^bw}` scalar bridge smoke
- [GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu](GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu): tiny Orca FC key-writer demo for bounded q62 constant-polynomial masks
- [GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu](GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu): GPU SPFSS payload correctness test
- [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu): standalone DPF online key generation benchmark for one-shot versus chunked partial generation

### 9.2 Shell Pipeline

Important scripts under [GPU-MPC/ringlpn/scripts](GPU-MPC/ringlpn/scripts) and [GPU-MPC/scripts](GPU-MPC/scripts):

- `setup_nfl.sh`: clone and build NFLlib
- `build_bench.sh`: build CPU benchmark
- `run_sweep.sh`: run CPU sweep and generate markdown summary
- `build_cuda_bench.sh`: build the promoted cheddar-derived main CUDA benchmark
- `build_vole_bench.sh`: build the standalone Ring-LPN VOLE prototype benchmark
- `build_cuda_bench_cheddar.sh`: build the same cheddar-derived source under an explicit side-by-side binary name
- `build_cuda_bench_legacy.sh`: build the preserved legacy CUDA benchmark
- `run_cuda_sweep.sh`: promoted q=32 or q=64 CUDA sweep
- `run_cuda_sweep_legacy.sh`: legacy q=32 CUDA sweep
- `run_cuda_single.sh`: CPU-vs-GPU spot check at CPU-overlap points
- `run_vole_sweep.sh`: standalone Ring-LPN VOLE prototype sweep
- `build_ole_cuda_bench.sh`: build the standalone GPU Figure 2 SPFSS/OLE benchmark and GPU SPFSS test
- `run_ole_sweep.sh`: run the smoke or bounded GPU Figure 2 OLE sweep and generate CSV/Markdown summaries
- `summarize_ole_results.py`: summarize GPU Figure 2 OLE CSV output
- `build_linear_ole_bench.sh`: build the ring-polynomial linear-layer OLE-to-Beaver benchmark
- `run_linear_ole_sweep.sh`: run the linear-layer OLE-to-Beaver smoke and generate CSV/Markdown summaries
- `summarize_linear_ole_results.py`: summarize ring-polynomial linear-layer OLE CSV output
- `build_orca_zp_bridge_test.sh`: build the host-only Orca scalar bridge test under `host_bin`
- `run_orca_zp_bridge_test.sh`: run the bridge smoke and q62/full-32-bit counterexample
- `build_orca_fc_ringlpn_demo.sh`: build the tiny Orca FC Ring-LPN key-writer demo
- `run_orca_fc_ringlpn_demo.sh`: run the tiny Orca FC demo and write CSV/Markdown summaries
- `summarize_orca_fc_demo.py`: summarize Orca FC demo CSV output
- `run_paper_checkpoint_smoke.sh`: one-command host smoke plus optional CUDA OLE/linear/FC smoke inside the container
- `run_dpf_online_keygen_sweep.py`: standalone DPF online key generation sweep and Markdown generation
- `run_vtune_hotspots.sh` and `run_vtune_memory.sh`: CPU profiling wrappers
- `summarize_results.py`, `summarize_cuda_results.py`, `summarize_cpu_gpu_4096.py`, `summarize_dpf_online_keygen.py`: reporting scripts

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
- The standalone Ring-LPN VOLE prototype in [GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu](GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu) is validated for requested `q=32|64` over `n` in `{8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}` using `synthetic_mpvole` inputs.
- Current VOLE result summaries live in [GPU-MPC/ringlpn/results/vole_gpu_q32_m32_c2_w64.md](GPU-MPC/ringlpn/results/vole_gpu_q32_m32_c2_w64.md) and [GPU-MPC/ringlpn/results/vole_gpu_q64_m32_c2_w64.md](GPU-MPC/ringlpn/results/vole_gpu_q64_m32_c2_w64.md).
- The standalone GPU Figure 2 OLE artifact in [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu) is validated for requested `q=64` over single-prime actual `q=62`, uniform sparse noise and regular sparse noise, `c=2`, `t=64`, and bounded `n` in `{8192, 16384}`. Regular noise uses grouped SPFSS domains `2N/t`, so the bounded `t=64` domains are `256` and `512`.
- The GPU Figure 2 OLE artifact validates `z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N+1)` and stops at OLE. It is not yet an Orca Beaver-triple integration or trusted-dealer removal.
- Current GPU Figure 2 OLE summaries live in [GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t8_smoke.md](GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t8_smoke.md), [GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t8_smoke.md](GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t8_smoke.md), [GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md](GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md), and [GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t64.md](GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t64.md). The detailed handoff is [GPU-MPC/ringlpn/results/ole_gpu_handoff.md](GPU-MPC/ringlpn/results/ole_gpu_handoff.md).
- The standalone ring-polynomial linear-layer OLE-to-Beaver artifact in [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu) validates the two-OLE-to-Beaver conversion for matrix multiplication over `Z_p[X]/(X^N+1)` with shared `A[row,k]` and `B[k,col]` operands reused across products.
- The current uniform linear-layer smoke uses `rows=2`, `inner=2`, `cols=2`, `n=8192`, `c=2`, `t=8`, 8 ring products, and 16 OLE instances. It passed validation with `shared_operands=1`, `linear_expand_mean_us=229748`, and `spfss_pair_key_bytes=2264064`.
- The current regular-noise linear-layer smoke uses the same shape and passed validation with `shared_operands=1`, `linear_expand_mean_us=137877`, `spfss_pair_key_bytes=1864704`, and SPFSS domain `2048`.
- The host-only Orca scalar bridge in [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp) validates the exact dealer/oracle carry correction for converting `Z_p` shares to `Z_{2^bw}` shares and a conservative constant-polynomial scalar packing model under the bound `inner * value_bound^2 < p`.
- The bridge smoke records `633` naive share-conversion failures, `0` corrected failures, a passing bounded `bw=16` scalar case, and an intentional q62/full-32-bit counterexample. It is not a secure distributed conversion protocol, high-density packing scheme, q128 path, or Orca key writer.
- The tiny Orca FC Ring-LPN demo in [GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu](GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu) writes bounded q62 constant-polynomial raw party buffers in `A`, `B`, `C_masked` order and validates unchanged `gpuMatmulBeaver` for a bounded suite: `2x2x2 bw16`, `2x3x2 bw16`, `3x2x2 bw16`, and `2x2x3 bw32`, all with `value_bound=255`; every case matches Orca `gpuKeygenMatmul` baseline output. It is forward FC only, not q128/CRT, dense packing, secure distributed conversion, training/backward integration, or trusted-dealer removal.
- The standalone DPF online key generation benchmark in [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu) is validated for eval-all keys at `bin=16`, `chunk_size=8192`, and `n` in `{8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}`.
- Current DPF online key generation summaries live in [GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md](GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md) and the corresponding CSV.
- The current DPF sweep shows full pair-key footprint growing from `2.81 MiB` to `360.00 MiB` while chunked online generation holds peak pair-key footprint to `2.81 MiB`, reaching about `128x` peak-footprint reduction at `n=1048576` with about `1.885x` key-generation time overhead.
- Current promoted result summaries live in [GPU-MPC/ringlpn/results/ntt_gpu_q32.md](GPU-MPC/ringlpn/results/ntt_gpu_q32.md) and [GPU-MPC/ringlpn/results/ntt_gpu_q64.md](GPU-MPC/ringlpn/results/ntt_gpu_q64.md).
- Legacy comparison results live in [GPU-MPC/ringlpn/results/ntt_gpu_q32_legacy.md](GPU-MPC/ringlpn/results/ntt_gpu_q32_legacy.md).
- The most complete human handoff for Ring-LPN is [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md).

Inside the container, the benchmark workdir is `/home/ringlpn`.

Important output locations:

- [GPU-MPC/ringlpn/results](GPU-MPC/ringlpn/results)
- [GPU-MPC/ringlpn/bin](GPU-MPC/ringlpn/bin)

### 9.4 Benchmark Roadmap Context

The active Ring-LPN roadmap now has two tracks:

1. Core NTT and PolyMul benchmark track:
  - generalized single-prime q=32 support is complete,
  - generalized single-prime q=64 support is complete,
  - requested `q=128` via dual-prime CRT is the next benchmark-core target.
2. Online-phase systems track:
  - the standalone Ring-LPN VOLE prototype is implemented and benchmarked,
  - the standalone GPU Figure 2 SPFSS/OLE artifact is implemented and benchmarked for single-prime q=62, uniform sparse noise, and regular sparse noise,
  - the standalone ring-polynomial linear-layer OLE-to-Beaver artifact is implemented and smoke-tested with shared matrix operand reuse,
  - the host-only Orca scalar bridge smoke is implemented for carry-corrected dealer/oracle `Z_p -> Z_{2^bw}` conversion and conservative constant-polynomial packing,
  - the tiny forward-only Orca FC key-writer demo is implemented for bounded q62 constant-polynomial masks and unchanged `gpuMatmulBeaver`,
  - the standalone DPF online key generation benchmark is implemented and benchmarked,
  - CRT/q128, high-density Orca scalar packing, secure distributed `Z_p -> Z_{2^bw}` conversion, full Orca training/backward integration, and end-to-end Orca/SPFSS integration are not implemented yet.

### 9.5 Current Mission Handoff

If a new agent is asked to continue the current Ring-LPN mission, the practical mission is:

1. treat the promoted cheddar-derived path as the default GPU implementation,
2. preserve the legacy CUDA path as a comparison baseline,
3. avoid reworking the CPU baseline unless a validation issue requires it,
4. treat [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu), [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu), [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp), [GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu](GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu), [GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu](GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu), and [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu) as the current online-phase evidence,
5. if the task is benchmark-core continuation, carry the GPU path from single-prime q=32/q=64 to dual-prime CRT q=128,
6. if the task is Figure 2/OLE continuation, add CRT before claiming paper-comparable numbers,
7. if the task is linear-layer continuation, treat the current ring-polynomial OLE-to-Beaver artifact, host scalar bridge, and tiny FC key-writer demo as the bridge layer; the next implementation step is q128/CRT or concrete layer bounds plus denser packing before attempting broader Orca replacement,
8. if the task is Orca integration, keep online `gpuMatmulBeaver` unchanged and make the generated key shares match its existing `(A, B, C)` Beaver contract.

What is already done:

- CPU NFLLib benchmark and full sweep are complete.
- Promoted cheddar-derived GPU q=32 sweep is complete and validated.
- Promoted cheddar-derived GPU q=64 sweep is complete and validated.
- Legacy GPU q=32 sweep is preserved for comparison.
- Standalone Ring-LPN VOLE q=32 and q=64 sweeps are complete and validated.
- Standalone GPU Figure 2 SPFSS/OLE artifact is complete and validated for single-prime q=62, uniform sparse noise and regular sparse noise, `c=2`, `t=64`, bounded `n={8192,16384}`.
- GPU SPFSS payload tests cover single point, multiple points, alpha collisions, and edge alphas.
- Standalone ring-polynomial linear-layer OLE-to-Beaver artifact is complete and validated for the smoke case `rows=2`, `inner=2`, `cols=2`, `n=8192`, `c=2`, `t=8`, with `shared_operands=1`.
- Host-only Orca scalar bridge smoke is complete for the dealer/oracle carry correction from `Z_p` shares to `Z_{2^bw}` shares, validates bounded constant-polynomial scalar packing, and records a q62/full-32-bit counterexample.
- Tiny forward-only Orca FC key-writer demo is complete for the current bounded small-shape suite, `value_bound=255`, `poly_n=8192`, `c=2`, `t=8`, `tf=None`, zero bias, deterministic replay, second-seed checks, and Orca `gpuKeygenMatmul` baseline comparison.
- Standalone DPF online key generation sweep at `bin=16`, `chunk_size=8192` is complete and validated.
- The current GPU-vs-CPU comparison is strong: q=32 overlap points show roughly `146x` to `171x` per-polynomial PolyMul speedups over CPU, and q=64 points show roughly `48x` to `220x` per-polynomial PolyMul speedups over CPU.
- The promoted main path is consistently faster than the legacy baseline, with the largest observed per-polynomial gain near `6x` at `n=65536` in the adaptive sweep.
- [GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md](GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md) and [GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md](GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md) record the current abstract-safe claims and open gaps.

What is not done:

- no promoted GPU path yet exists for requested `q=128`,
- no multi-prime scheduling layer exists yet in the promoted path,
- no CRT recomposition path exists yet in the promoted path,
- no full Orca-scalar OLE-to-Beaver replacement exists yet beyond the tiny forward FC demo,
- no high-density scalar packing layer exists yet from Orca tensor elements into Ring-LPN polynomial entries,
- no secure distributed `Z_p -> Z_{2^bw}` share conversion exists yet for Orca parties that do not know both prime-field shares,
- no Orca-compatible training/backward/optimizer key writer exists yet for the conservative constant-polynomial bridge,
- no end-to-end Orca or SPFSS-backed integration exists yet for the chunked DPF online key generation benchmark,
- no end-to-end Orca or SPFSS-backed integration exists yet for the Ring-LPN VOLE prototype,
- no full application-level memory-footprint reduction measurements exist yet for the combined online path,
- VTune wrappers exist, but VTune is not guaranteed to be installed in the active container.

Files the next agent should read first for Ring-LPN continuation:

- [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md)
- [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md)
- [GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md](GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md)
- [GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md](GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md)
- [GPU-MPC/ringlpn/results/ole_gpu_handoff.md](GPU-MPC/ringlpn/results/ole_gpu_handoff.md)
- [GPU-MPC/ringlpn/results/linear_ole_handoff.md](GPU-MPC/ringlpn/results/linear_ole_handoff.md)
- [GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md](GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md)
- [GPU-MPC/ringlpn/results/orca_fc_ringlpn_demo_memo.md](GPU-MPC/ringlpn/results/orca_fc_ringlpn_demo_memo.md)
- [GPU-MPC/ringlpn/results/orca_zp_bridge_constant_scalar.md](GPU-MPC/ringlpn/results/orca_zp_bridge_constant_scalar.md)
- [GPU-MPC/ringlpn/results/paper_execution_next_steps.md](GPU-MPC/ringlpn/results/paper_execution_next_steps.md)
- [GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md](GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md)
- [GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t8_smoke.md](GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t8_smoke.md)
- [GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t64.md](GPU-MPC/ringlpn/results/ole_gpu_q64_regular_c2_t64.md)
- [GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md](GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md)
- [GPU-MPC/ringlpn/README.md](GPU-MPC/ringlpn/README.md)
- [GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh](GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh)
- [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu)
- [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu)
- [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp)
- [GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu](GPU-MPC/ringlpn/src/bench_orca_fc_ringlpn_demo.cu)
- [GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu](GPU-MPC/ringlpn/src/test_spfss_zp_cuda.cu)
- [GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu](GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu)
- [GPU-MPC/ringlpn/src/bench_ntt.cpp](GPU-MPC/ringlpn/src/bench_ntt.cpp)
- [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu)
- [GPU-MPC/ringlpn/scripts/build_cuda_bench.sh](GPU-MPC/ringlpn/scripts/build_cuda_bench.sh)
- [GPU-MPC/ringlpn/scripts/build_ole_cuda_bench.sh](GPU-MPC/ringlpn/scripts/build_ole_cuda_bench.sh)
- [GPU-MPC/ringlpn/scripts/run_ole_sweep.sh](GPU-MPC/ringlpn/scripts/run_ole_sweep.sh)
- [GPU-MPC/ringlpn/scripts/build_linear_ole_bench.sh](GPU-MPC/ringlpn/scripts/build_linear_ole_bench.sh)
- [GPU-MPC/ringlpn/scripts/run_linear_ole_sweep.sh](GPU-MPC/ringlpn/scripts/run_linear_ole_sweep.sh)
- [GPU-MPC/ringlpn/scripts/build_vole_bench.sh](GPU-MPC/ringlpn/scripts/build_vole_bench.sh)
- [GPU-MPC/ringlpn/scripts/build_cuda_bench_legacy.sh](GPU-MPC/ringlpn/scripts/build_cuda_bench_legacy.sh)
- [GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh](GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh)
- [GPU-MPC/ringlpn/scripts/run_vole_sweep.sh](GPU-MPC/ringlpn/scripts/run_vole_sweep.sh)
- [GPU-MPC/ringlpn/scripts/run_cuda_sweep_legacy.sh](GPU-MPC/ringlpn/scripts/run_cuda_sweep_legacy.sh)
- [GPU-MPC/scripts/run_dpf_online_keygen_sweep.py](GPU-MPC/scripts/run_dpf_online_keygen_sweep.py)
- [GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md](GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md)

Commands worth knowing for continuation work:

- build promoted CPU and GPU benchmarks inside the container by running `./scripts/build_bench.sh` and `./scripts/build_cuda_bench.sh` under `/home/ringlpn`,
- run the promoted q=32 sweep with `./scripts/run_cuda_sweep.sh`,
- run the promoted q=64 sweep with `QBITS=64 ./scripts/run_cuda_sweep.sh`,
- build the standalone VOLE prototype with `./scripts/build_vole_bench.sh` under `/home/ringlpn`,
- run the standalone VOLE sweep with `./scripts/run_vole_sweep.sh` or `QBITS=64 ./scripts/run_vole_sweep.sh` under `/home/ringlpn`,
- build the standalone GPU Figure 2 OLE artifact with `./scripts/build_ole_cuda_bench.sh` under `/home/ringlpn`,
- run the quick GPU Figure 2 OLE smoke with `SMOKE=1 ./scripts/run_ole_sweep.sh` under `/home/ringlpn`,
- run the bounded GPU Figure 2 OLE sweep with `./scripts/run_ole_sweep.sh` under `/home/ringlpn`,
- build the ring-polynomial linear-layer OLE-to-Beaver artifact with `./scripts/build_linear_ole_bench.sh` under `/home/ringlpn`,
- run the linear-layer smoke with `./scripts/run_linear_ole_sweep.sh` under `/home/ringlpn`,
- build and run the host-only Orca scalar bridge from the host with `GPU-MPC/ringlpn/scripts/build_orca_zp_bridge_test.sh` and `GPU-MPC/ringlpn/scripts/run_orca_zp_bridge_test.sh`,
- run the consolidated host checkpoint smoke with `GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh`,
- run the consolidated CUDA checkpoint smoke inside `/home/ringlpn` with `RUN_GPU_SMOKE=1 ./scripts/run_paper_checkpoint_smoke.sh`,
- run the legacy comparison sweep with `./scripts/build_cuda_bench_legacy.sh` and `./scripts/run_cuda_sweep_legacy.sh`,
- build the standalone DPF online key generation benchmark under `/home` with `make GPU_ARCH=<cc> dpf_online_keygen`,
- run the standalone DPF sweep under `/home` with `python3 scripts/run_dpf_online_keygen_sweep.py`,
- use `./tests/fss/dpf_online_keygen --bin 16 --n 8192 --chunk-size 4096 --iters 3 --warmup 1 --csv-header` under `/home` for a quick smoke test,
- use `./scripts/run_cuda_single.sh 8192 4` or another CPU-overlap point only for quick spot checks.

Operational cautions for continuation work:

- do Ring-LPN runtime work inside the `orca-dev` container under `/home/ringlpn`, not from the host path,
- do standalone DPF online key generation runtime work inside the same `orca-dev` container under `/home`, not from the host path,
- remember that the root [.gitignore](.gitignore) ignores `*.sh`, `*.csv`, `*.txt`, and `*.out`, so helper scripts and generated summaries may not be tracked,
- do not assume the current shell scripts have Git history just because they exist in the working tree,
- raw stdout from programs that call `initGPUMemPool()` can include `reserved memory:` lines from `gpu_mem.cu`, so use the current sweep scripts or filter those lines before treating stdout as CSV,
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

### 11.5 gpu_mem.cu Prints To Stdout

`initGPUMemPool()` currently prints a `reserved memory:` line to stdout.

Practical consequence:

- raw benchmark stdout is not guaranteed to be clean CSV,
- the current DPF sweep script filters these lines before writing CSV,
- if a future agent adds another CSV-emitting benchmark on top of `gpu_mem.cu`, they should handle this explicitly.

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
- then read [GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md](GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md) if the task is abstract-facing or online-phase related
- then inspect [GPU-MPC/ringlpn/src](GPU-MPC/ringlpn/src)
- then use [GPU-MPC/ringlpn/scripts](GPU-MPC/ringlpn/scripts)

If the task mentions Figure 2, SPFSS, OLE, DPF payloads over `Z_p`, or trusted-dealer removal for linear layers:

- start with [GPU-MPC/ringlpn/results/ole_gpu_handoff.md](GPU-MPC/ringlpn/results/ole_gpu_handoff.md)
- then read [GPU-MPC/ringlpn/results/linear_ole_handoff.md](GPU-MPC/ringlpn/results/linear_ole_handoff.md)
- then read [GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md](GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md)
- then read [GPU-MPC/ringlpn/results/paper_execution_next_steps.md](GPU-MPC/ringlpn/results/paper_execution_next_steps.md)
- then read [GPU-MPC/ringlpn/results/ole_figure2_host_results.md](GPU-MPC/ringlpn/results/ole_figure2_host_results.md)
- then inspect [GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh](GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh)
- then inspect [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu)
- then inspect [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu)
- then inspect [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp)
- then read [GPU-MPC/ringlpn/results/ringlpn_linear_integration_plan.md](GPU-MPC/ringlpn/results/ringlpn_linear_integration_plan.md)

If the task mentions DPF, online key generation, partial keys, or memory-footprint reduction:

- start with [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu)
- then read [GPU-MPC/scripts/run_dpf_online_keygen_sweep.py](GPU-MPC/scripts/run_dpf_online_keygen_sweep.py)
- then read [GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md](GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md)
- then read [GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md](GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md)
- then read [GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md](GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md)

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
5. For Ring-LPN, decide whether the task is CPU NFLLib, promoted cheddar-derived CUDA, standalone VOLE, GPU Figure 2 OLE/SPFSS, ring-polynomial linear OLE-to-Beaver, Orca scalar bridge, tiny Orca FC demo, or standalone DPF online key generation.
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
- [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](GPU-MPC/tests/fss/dpf_online_keygen_bench.cu)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu)
- [GPU-MPC/ringlpn/src/bench_ntt_cuda.cu](GPU-MPC/ringlpn/src/bench_ntt_cuda.cu)
- [GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh](GPU-MPC/ringlpn/src/gpu_spfss_zp.cuh)
- [GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_ole_ringlpn_cuda.cu)
- [GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu](GPU-MPC/ringlpn/src/bench_linear_ole_ringlpn_cuda.cu)
- [GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp](GPU-MPC/ringlpn/src/test_orca_zp_bridge.cpp)
- [GPU-MPC/ringlpn/scripts/run_ole_sweep.sh](GPU-MPC/ringlpn/scripts/run_ole_sweep.sh)
- [GPU-MPC/ringlpn/scripts/run_linear_ole_sweep.sh](GPU-MPC/ringlpn/scripts/run_linear_ole_sweep.sh)
- [GPU-MPC/ringlpn/scripts/run_orca_zp_bridge_test.sh](GPU-MPC/ringlpn/scripts/run_orca_zp_bridge_test.sh)
- [GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh](GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh)
- [GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu](GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu)
- [GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh](GPU-MPC/ringlpn/scripts/run_cuda_sweep.sh)
- [GPU-MPC/ringlpn/scripts/run_vole_sweep.sh](GPU-MPC/ringlpn/scripts/run_vole_sweep.sh)
- [GPU-MPC/scripts/run_dpf_online_keygen_sweep.py](GPU-MPC/scripts/run_dpf_online_keygen_sweep.py)
- [GPU-MPC/ringlpn/results/ringlpn_status_report.md](GPU-MPC/ringlpn/results/ringlpn_status_report.md)
- [GPU-MPC/ringlpn/results/ole_gpu_handoff.md](GPU-MPC/ringlpn/results/ole_gpu_handoff.md)
- [GPU-MPC/ringlpn/results/linear_ole_handoff.md](GPU-MPC/ringlpn/results/linear_ole_handoff.md)
- [GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md](GPU-MPC/ringlpn/results/orca_zp_bridge_handoff.md)
- [GPU-MPC/ringlpn/results/paper_execution_next_steps.md](GPU-MPC/ringlpn/results/paper_execution_next_steps.md)
- [GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md](GPU-MPC/ringlpn/results/ole_gpu_q64_uniform_c2_t64.md)
- [GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md](GPU-MPC/ringlpn/results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md)
- [GPU-MPC/ringlpn/results/cheddar_extract_note.md](GPU-MPC/ringlpn/results/cheddar_extract_note.md)
- [GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md](GPU-MPC/ringlpn/results/ringlpn_vole_abstract_support.md)
- [GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md](GPU-MPC/ringlpn/results/gpu_fss_memory_efficiency_outline.md)
- [GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md](GPU-MPC/ringlpn/results/dpf_online_keygen_bin16_chunk8192.md)
- [GPU-MPC/ringlpn/results](GPU-MPC/ringlpn/results)

These paths cover container entry, main build, Orca orchestration, profiling, logs, shared utilities, and the active Ring-LPN plus DPF online key generation work.
