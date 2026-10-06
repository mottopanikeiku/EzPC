> **OPERATIONAL GUIDE, WRITTEN 2026-05-18 — read with care.** The container model
> (§2), filesystem mapping (§4), build pipeline (§6), Orca runbooks (§7), and
> gotchas (§11) remain useful operational reference. Ring-LPN content was removed
> on 2026-10-06: use [`GPU-MPC/ringlpn/CLAUDE.md`](../ringlpn/CLAUDE.md). Verify any
> pipeline against the current tree before relying on it.

# GPU-MPC Workspace Guide For Agents

This workspace contains many sibling projects, but current work is usually only about [GPU-MPC](..).

If a task does not explicitly mention another top-level project, treat everything outside [GPU-MPC](..) as out of scope.

This file is meant to answer four questions quickly for a new agent:

1. What is the real project boundary here?
2. How does the Docker and filesystem mapping work?
3. What are the major GPU-MPC pipelines?
4. Which files and directories matter first for Orca and Sigma work?

## 1. Scope

The repository root is a larger CrypTFlow-era monorepo, but for day-to-day work the meaningful project is [GPU-MPC](..).

Unless the user explicitly asks about other subsystems, the practical rule is:

- ignore [Athos](../../Athos), [SCI](../../SCI), [FSS](../../FSS), [Porthos](../../Porthos), [Beacon](../../Beacon), [OnnxBridge](../../OnnxBridge), [sytorch](../../sytorch), and other siblings,
- focus on [GPU-MPC](..),
- and remember that the active workflows are mostly:
  - Orca training and inference,
  - Orca local loopback and profiling,
  - dealerless Ring-LPN preprocessing for Orca linear layers (see §9),
  - standalone DPF online key generation benchmarking,
  - and occasionally Sigma.

For substantive implementation or benchmarking jobs, update the relevant documentation before finishing the turn: this guide for Orca workflows, and [`GPU-MPC/ringlpn/CLAUDE.md`](../ringlpn/CLAUDE.md) for Ring-LPN.

## 2. Container Model

The root entrypoint for local GPU work is [start](../../start).

Important behavior:

- it launches or attaches to a Docker container named `orca-dev`,
- it mounts only [GPU-MPC](..) into the container,
- inside the container that mount appears as `/home`.

That means host paths and container paths are different:

- host repo root: `/home/fatih/EzPC`
- host GPU project root: `/home/fatih/EzPC/GPU-MPC`
- container project root: `/home`

Most important path translations:

- [GPU-MPC/experiments/orca](../experiments/orca) -> `/home/experiments/orca`
- [GPU-MPC/orca_runner](../orca_runner) -> `/home/orca_runner`
- [GPU-MPC/ringlpn](../ringlpn) -> `/home/ringlpn`
- [GPU-MPC/scripts](../scripts) -> `/home/scripts`
- [GPU-MPC/keys](../keys) -> `/home/keys`

This is the most important operational fact in the workspace. A large fraction of debugging confusion comes from forgetting that container commands must usually run under `/home/...`, not the host path.

## 3. GPU-MPC Identity

[GPU-MPC/README.md](../README.md) describes GPU-MPC as the implementation of protocols from the Orca and SIGMA papers.

In practice, the project is organized as:

- protocol backends and shared GPU runtime code,
- experiment binaries and paper harnesses,
- local loopback runners and profiling scripts,
- benchmark harnesses,
- and large external dependencies such as CUTLASS and Sytorch.

The most important active subprojects are:

- [GPU-MPC/experiments/orca](../experiments/orca): formal Orca training and inference binaries plus experiment harness,
- [GPU-MPC/orca_runner](../orca_runner): local single-machine loopback automation and logs,
- [GPU-MPC/ringlpn](../ringlpn): dealerless Ring-LPN preprocessing for Orca linear layers (see §9),
- [GPU-MPC/backend](../backend): backend protocol headers,
- [GPU-MPC/utils](../utils): shared GPU memory, file I/O, comms, and helper utilities.

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

- [GPU-MPC/Makefile](../Makefile): main NVCC build graph for Orca, Sigma, Piranha, and many tests.
- [GPU-MPC/setup.sh](../setup.sh): dependency/bootstrap helper for CUTLASS, Sytorch, datasets, and output directories.
- [GPU-MPC/experiments/orca](../experiments/orca): Orca binaries, configs, outputs, datasets, and experiment harness.
- [GPU-MPC/experiments/sigma](../experiments/sigma): Sigma binaries and paper path.
- [GPU-MPC/orca_runner](../orca_runner): local loopback scripts and logs.
- [GPU-MPC/ringlpn](../ringlpn): dealerless Ring-LPN preprocessing subproject; see §9.
- [GPU-MPC/scripts](../scripts): profiling and summary helpers for Orca.
- [GPU-MPC/backend](../backend): backend abstractions such as Orca, Sigma, and Piranha headers.
- [GPU-MPC/utils](../utils): shared low-level GPU utilities.
- [GPU-MPC/ext/cutlass](../ext/cutlass): CUTLASS dependency.
- [GPU-MPC/ext/sytorch](../ext/sytorch): Sytorch plus LLAMA, cryptoTools, bitpack, and related dependencies.

## 5. Top-Level GPU-MPC Files

### 5.1 Build Entry Points

- [GPU-MPC/README.md](../README.md): top-level build and Docker notes.
- [GPU-MPC/Makefile](../Makefile): actual build targets.
- [GPU-MPC/setup.sh](../setup.sh): setup script for dependencies, datasets, and output directories.
- [GPU-MPC/Dockerfile_Gen](../Dockerfile_Gen): image build path for GPU-MPC environment setup.

### 5.2 Runtime/Experiment Entry Points

- [GPU-MPC/experiments/orca/README.md](../experiments/orca/README.md): formal Orca usage notes.
- [GPU-MPC/experiments/orca/run_experiment.py](../experiments/orca/run_experiment.py): figure and table harness.
- [GPU-MPC/orca_runner/run_and_log.sh](../orca_runner/run_and_log.sh): local loopback end-to-end runner.
- [GPU-MPC/orca_runner/run_remaining.sh](../orca_runner/run_remaining.sh): follow-on local runs for remaining models.
- [GPU-MPC/scripts/run_orca_profiling.sh](../scripts/run_orca_profiling.sh): ORCA profiling harness.

### 5.3 Benchmark Entry Points

- Ring-LPN entry points: see §9.
- [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](../tests/fss/dpf_online_keygen_bench.cu): standalone DPF online key generation benchmark.
- [GPU-MPC/scripts/run_dpf_online_keygen_sweep.py](../scripts/run_dpf_online_keygen_sweep.py): DPF online key generation sweep driver.

## 6. Build Pipeline

### 6.1 Makefile Model

[GPU-MPC/Makefile](../Makefile) is the central build graph.

Important facts from the file:

- the compiler is `nvcc`,
- the build uses `-std=c++17`,
- architecture is selected by `GPU_ARCH`,
- if `GPU_ARCH` is unset, plain `make` can fail with `nvcc fatal: Unsupported gpu architecture 'compute_'`,
- include/lib paths point into CUTLASS and Sytorch,
- common runtime utilities come from [GPU-MPC/utils](../utils).

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
- individual FSS/NN test binaries under [GPU-MPC/tests](../tests)

Primary Orca binaries built by the Makefile:

- [GPU-MPC/experiments/orca/orca_dealer.cu](../experiments/orca/orca_dealer.cu)
- [GPU-MPC/experiments/orca/orca_evaluator.cu](../experiments/orca/orca_evaluator.cu)
- [GPU-MPC/experiments/orca/orca_inference.cu](../experiments/orca/orca_inference.cu)
- [GPU-MPC/experiments/orca/piranha.cu](../experiments/orca/piranha.cu)

### 6.2 setup.sh Behavior

[GPU-MPC/setup.sh](../setup.sh) does the following:

1. updates submodules,
2. installs gcc-9/g++-9 and core build dependencies,
3. builds CUTLASS under [GPU-MPC/ext/cutlass](../ext/cutlass),
4. builds Sytorch under [GPU-MPC/ext/sytorch](../ext/sytorch),
5. downloads CIFAR-10 into [GPU-MPC/experiments/orca/datasets/cifar-10](../experiments/orca/datasets/cifar-10),
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

- [GPU-MPC/experiments/orca/config.json](../experiments/orca/config.json)
- [GPU-MPC/experiments/orca/run_experiment.py](../experiments/orca/run_experiment.py)
- [GPU-MPC/experiments/orca/output](../experiments/orca/output)

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

High-level mapping from [GPU-MPC/experiments/orca/run_experiment.py](../experiments/orca/run_experiment.py):

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

- [GPU-MPC/orca_runner/run_and_log.sh](../orca_runner/run_and_log.sh)
- [GPU-MPC/orca_runner/run_remaining.sh](../orca_runner/run_remaining.sh)
- [GPU-MPC/orca_runner/logs](../orca_runner/logs)

This is the most important operational path for local single-machine testing.

It assumes the container layout:

- workdir: `/home/experiments/orca`
- logs: `/home/orca_runner/logs`
- keys: `/home/keys/P0` and `/home/keys/P1`

[GPU-MPC/orca_runner/run_and_log.sh](../orca_runner/run_and_log.sh) currently does roughly this:

- training: `P-SecureML`, `P-LeNet`, `P-AlexNet`
- inference: `CNN2`, `CNN3`, `VGG16`
- training/perf: `CNN2-perf`
- optional larger run: `CNN3-perf` only if disk space is sufficient

[GPU-MPC/orca_runner/run_remaining.sh](../orca_runner/run_remaining.sh) continues with more inference and training runs, including `ModelB` and `AlexNet`.

Operational pattern:

1. remove stale key files for the model,
2. run dealer for P0 and P1,
3. run evaluator pair on localhost,
4. append key sizes and tail summaries into `master.log`,
5. remove keys again.

### 7.3 Orca Profiling Pipeline

Primary files:

- [GPU-MPC/scripts/run_orca_profiling.sh](../scripts/run_orca_profiling.sh)
- [GPU-MPC/scripts/summarize_orca_results.py](../scripts/summarize_orca_results.py)

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

Sigma is present under [GPU-MPC/experiments/sigma](../experiments/sigma) and built with `make sigma`.

For most current work it is secondary to Orca and Ring-LPN, but agents should know:

- Sigma is part of the same build system,
- Sigma output directories are created by [GPU-MPC/setup.sh](../setup.sh),
- and shared utilities/backends under [GPU-MPC/utils](../utils) and [GPU-MPC/backend](../backend) can matter to both Orca and Sigma.

## 9. Ring-LPN

Ring-LPN work lives in [GPU-MPC/ringlpn](../ringlpn). Its canonical status, claims, source map, commands, and gotchas are in [GPU-MPC/ringlpn/CLAUDE.md](../ringlpn/CLAUDE.md); build and run steps are in [GPU-MPC/ringlpn/README.md](../ringlpn/README.md); the evidence index is [GPU-MPC/ringlpn/results/README.md](../ringlpn/results/README.md). The May 2026 pipeline description formerly here is in git history.

## 10. Backend, Utils, and Tests

### 10.1 Backend Headers

Important files under [GPU-MPC/backend](../backend):

- [GPU-MPC/backend/orca_base.h](../backend/orca_base.h)
- [GPU-MPC/backend/orca.h](../backend/orca.h)
- [GPU-MPC/backend/piranha.h](../backend/piranha.h)
- [GPU-MPC/backend/sigma.h](../backend/sigma.h)

These are the main backend abstractions and are the right place to start if a task is about protocol mechanics rather than experiment orchestration.

### 10.2 Shared Utilities

Important files under [GPU-MPC/utils](../utils):

- [GPU-MPC/utils/gpu_mem.cu](../utils/gpu_mem.cu)
- [GPU-MPC/utils/gpu_file_utils.cpp](../utils/gpu_file_utils.cpp)
- [GPU-MPC/utils/sigma_comms.cpp](../utils/sigma_comms.cpp)
- [GPU-MPC/utils/gpu_random.cu](../utils/gpu_random.cu)

Current verified local facts from recent work:

- `gpu_mem.cu` is patched in this checkout to reserve 25 GB rather than 40 GB for the mempool,
- `gpu_file_utils.cpp` uses `O_DIRECT | O_LARGEFILE` for key reads/writes,
- key buffers are 4096-byte aligned,
- some remaining Orca overhead likely comes from repeated `moveToGPU()` calls on masks, weights, or activations.

### 10.3 Tests

[GPU-MPC/tests](../tests) contains:

- [GPU-MPC/tests/fss](../tests/fss)
- [GPU-MPC/tests/nn](../tests/nn)

These are useful when a task is about validating kernels or protocol primitives outside the large end-to-end runners.

## 11. Important Gotchas

### 11.1 The Root .gitignore Is Aggressive

The root [.gitignore](../../.gitignore) ignores many file types globally, including:

- `*.sh`
- `*.csv`
- `*.txt`
- `*.out`

Practical consequence:

- shell helpers outside [GPU-MPC/ringlpn/scripts](../ringlpn/scripts) (which the root ignore file re-allows) may not be tracked,
- generated summaries and result tables often have no Git history,
- a script that looks "local only" may simply be ignored by Git.

### 11.2 Key Material Is Huge

Orca key files can be very large. Multiple scripts assume hundreds of GB of free space or explicitly skip runs when space is low.

### 11.3 The Container Is The Real Runtime

For GPU work, commands that appear to work on the host may still be the wrong environment. The intended runtime for Orca is usually inside `orca-dev`; Ring-LPN builds run on the host (see its CLAUDE.md).

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

- start with [GPU-MPC/experiments/orca/README.md](../experiments/orca/README.md)
- then read [GPU-MPC/experiments/orca/run_experiment.py](../experiments/orca/run_experiment.py)
- then check [GPU-MPC/experiments/orca/config.json](../experiments/orca/config.json)

If the task mentions local logs, loopback execution, or quick reproduction:

- start with [GPU-MPC/orca_runner/run_and_log.sh](../orca_runner/run_and_log.sh)
- check [GPU-MPC/orca_runner/run_remaining.sh](../orca_runner/run_remaining.sh)
- then inspect [GPU-MPC/orca_runner/logs](../orca_runner/logs)

If the task mentions profiling Orca or key I/O:

- start with [GPU-MPC/scripts/run_orca_profiling.sh](../scripts/run_orca_profiling.sh)
- then inspect [GPU-MPC/scripts/summarize_orca_results.py](../scripts/summarize_orca_results.py)
- and the shared utilities in [GPU-MPC/utils](../utils)

If the task mentions Ring-LPN, NTT, SPFSS, OLE, or trusted-dealer removal for linear layers:

- start with [GPU-MPC/ringlpn/CLAUDE.md](../ringlpn/CLAUDE.md)
- then use [GPU-MPC/ringlpn/README.md](../ringlpn/README.md) and [GPU-MPC/ringlpn/results/README.md](../ringlpn/results/README.md)

If the task mentions DPF, online key generation, partial keys, or memory-footprint reduction:

- start with [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](../tests/fss/dpf_online_keygen_bench.cu)
- then read [GPU-MPC/scripts/run_dpf_online_keygen_sweep.py](../scripts/run_dpf_online_keygen_sweep.py) (it runs `make dpf_online_keygen` itself and writes to the top of `GPU-MPC/ringlpn/results/`)
- then read the retained copy [GPU-MPC/ringlpn/results/dpf/dpf_online_keygen_bin16_chunk8192.md](../ringlpn/results/dpf/dpf_online_keygen_bin16_chunk8192.md)
- historical abstract context: [GPU-MPC/ringlpn/results/outreach/gpu_fss_memory_efficiency_outline.md](../ringlpn/results/outreach/gpu_fss_memory_efficiency_outline.md)

If the task mentions Sigma:

- start with [GPU-MPC/README.md](../README.md)
- then inspect [GPU-MPC/experiments/sigma](../experiments/sigma)
- and the shared backend/util files.

## 13. Minimal Orientation Checklist

If dropped into this workspace cold, the fastest reliable orientation path is:

1. Read [start](../../start) and translate host paths to `/home/...` container paths.
2. Read [GPU-MPC/README.md](../README.md) and [GPU-MPC/Makefile](../Makefile).
3. Decide whether the task is Orca, Sigma, or Ring-LPN (Ring-LPN: go to [GPU-MPC/ringlpn/CLAUDE.md](../ringlpn/CLAUDE.md)).
4. For Orca, decide whether the task is the formal harness, local loopback, or profiling runner.
5. Check whether the files you care about are actually tracked, because the root ignore rules hide many script and result files.

## 14. Highest-Signal Paths For Current Work

If the task is about the currently active GPU work, these are the first paths to check:

- [start](../../start)
- [GPU-MPC/README.md](../README.md)
- [GPU-MPC/Makefile](../Makefile)
- [GPU-MPC/setup.sh](../setup.sh)
- [GPU-MPC/experiments/orca](../experiments/orca)
- [GPU-MPC/experiments/orca/run_experiment.py](../experiments/orca/run_experiment.py)
- [GPU-MPC/orca_runner](../orca_runner)
- [GPU-MPC/orca_runner/logs](../orca_runner/logs)
- [GPU-MPC/scripts/run_orca_profiling.sh](../scripts/run_orca_profiling.sh)
- [GPU-MPC/utils](../utils)
- [GPU-MPC/ringlpn](../ringlpn)
- [GPU-MPC/tests/fss/dpf_online_keygen_bench.cu](../tests/fss/dpf_online_keygen_bench.cu)
- [GPU-MPC/ringlpn/CLAUDE.md](../ringlpn/CLAUDE.md)
- [GPU-MPC/ringlpn/results/dpf/dpf_online_keygen_bin16_chunk8192.md](../ringlpn/results/dpf/dpf_online_keygen_bin16_chunk8192.md)
- [GPU-MPC/ringlpn/results](../ringlpn/results)

These paths cover container entry, main build, Orca orchestration, profiling, logs, shared utilities, Ring-LPN entry, and DPF online key generation.
