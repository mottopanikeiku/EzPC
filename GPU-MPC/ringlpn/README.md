# ringlpn — build and run

Dealerless Ring-LPN preprocessing for Orca's forward linear layers; Orca's
online consumers stay unchanged. Project status, claims, open gates, and
gotchas live in [`CLAUDE.md`](CLAUDE.md); the evidence index is
[`results/README.md`](results/README.md). This file covers building and running
only. The public source branch `ringlpn/clinical-review-2026-09-10` has its own,
different README.

## Prerequisites

- CUDA toolkit at `/usr/local/cuda/bin` (not on the default `PATH`);
  `GPU_ARCH=89` (RTX 5000 Ada).
- Submodule `GPU-MPC/ext/cutlass` initialized, plus `GPU-MPC/ringlpn/extern/NFLlib`
  for the CPU NTT benchmark (`scripts/setup_nfl.sh`); `ext/sytorch` with
  cryptoTools/LLAMA/bitpack is tracked source. Linear, graph, and application
  builds fail before compiling if CUTLASS/Sytorch/SCI sources are missing.
- Shared machine: check `nvidia-smi`. Two-party runners overwrite
  `CUDA_VISIBLE_DEVICES` per child, so pin physical GPUs with their
  `P0_GPU`/`P1_GPU`/`CHECK_GPU` variables; several default to GPU 0, and
  two-party runs need distinct GPUs.

## Canonical component builds

`./scripts/build_component.sh list` prints every maintained target.

```bash
cd GPU-MPC/ringlpn
export PATH=/usr/local/cuda/bin:$PATH GPU_ARCH=89
./scripts/build_component.sh linear-library          # build/linear-library/lib/libringlpn_linear.a
./scripts/build_component.sh linear-fc               # bin/test_two_party_fc_preprocess
./scripts/build_component.sh linear-conv             # bin/test_two_party_conv_preprocess
./scripts/build_component.sh graph-libraries         # cryptoTools, bitpack, LLAMA
./scripts/build_component.sh resnet18-full-graph
./scripts/build_component.sh orca-linear-application
```

- `linear-library` installs `<ringlpn/linear_preprocess.h>` and a public-API
  probe; the facade opens one party-local private record at a time and has no
  record-pair loader.
- `linear-fc`/`linear-conv` build through a fixed private canonical source
  symlink so the binaries match `results/fc/linear_adapter_binary_approval_2026_08_07.json`;
  any drift rejects before records are produced.
- `graph-libraries` source-builds its dependencies under `build/graph-libraries`
  (no downloads, no untracked `GPU-MPC/ext/sytorch/build`). Legacy `build_*.sh`
  entry points route the approved adapters and the full graph through it.

## Gate and focused runners

The canonical gate command, with its GPU selectors, is in
[`CLAUDE.md`](CLAUDE.md) §8. It needs three idle, distinct GPUs for the
full-graph stage, takes about 1.5 h (2026-09-10: 5,228 s), consumes fresh
correlation namespaces, and regenerates tracked results.

Without `RUN_GPU_SMOKE=1` the gate runs host checks only (shell syntax, 21-layer
manifest gate, private-file, consume-once ledger and SHAKE controls, host OLE,
Zp bridge, conversion, truncation, host and two-process DPF keygen), rewrites
their tracked results, and ends with
`[paper-smoke] HOST GATES PASS (GPU smoke skipped)`; build `linear-fc` and
`linear-conv` first for its manifest gate.

| Runner | Output |
|---|---|
| `scripts/run_two_party_fc_preprocess.sh` (rebuilds its adapter, so nvcc must be on `PATH`; GPUs default `P0_GPU=1 P1_GPU=3`) | `results/fc/two_party_fc_preprocess_*` |
| `scripts/run_two_party_conv_preprocess.sh` (builds if missing; GPUs default 1/3) | `results/conv/` |
| `scripts/run_two_party_fc_model_scale.sh` | `results/fc/two_party_fc_model_scale_*` |
| `scripts/build_component.sh secure-convert` → `scripts/run_secure_convert_test.sh`; `scripts/build_component.sh secure-truncate` → `scripts/run_secure_truncate_test.sh` | `results/secure_convert/`, `results/secure_truncate/` |
| `scripts/run_two_party_dpf_keygen.sh`, `scripts/run_two_party_gpu_dpf.sh` | `results/dpf/` |
| `scripts/run_full_linear_manifest_gate.sh` (needs `linear-fc`/`linear-conv` built) | checks the 21-layer manifests (no output) |
| `P0_GPU=<a> P1_GPU=<b> CHECK_GPU=<c> TRUSTED_GPU=<b> scripts/run_resnet18_full_graph.sh /abs/new-out /abs/new-state` (defaults 0/1/2, trusted 1; a stale `bin/` graph provenance aborts it: rebuild `resnet18-full-graph` first) | external output root |
| `P0_GPU=<a> P1_GPU=<b> scripts/run_orca_linear_application.sh` (rebuilds unless `ORCA_LINEAR_SKIP_BUILD=1`; defaults 0/1) | stdout only; `results/application/*_2026_08_24.*` are retained captures |
| `(cd .. && make GPU_ARCH=89 orca_inference)` | macro-off stock `../experiments/orca/orca_inference` |

The full-graph wrapper is serial by default. To opt into parallel lanes, set
`LINEAR_LANES` to comma-separated `P0_GPU:P1_GPU:CHECK_GPU:FIRST-LAST` lanes
(e.g. `1:2:3:22000-22085`; also set `P0_GPU`/`P1_GPU`/`CHECK_GPU`/`TRUSTED_GPU`,
which default to 0/1/2/1). Each lane needs three distinct GPUs and at least
86 ports, and lanes must not overlap. Application runs share fixed port 42003:
never run two at once.

## Same-function direct-OT FC baseline

`direct-ot-fc` builds an experimental producer that computes both FC cross products
with Gilboa OLE (both CRT fields for q128), reusing the exact conversion, authenticated
channel, consume-once claims, and record writer; it generates no Ring-LPN noise, DPF
keys, or expansion slots. The unchanged reference executable checks its records.

```bash
cd "$(git rev-parse --show-toplevel)/GPU-MPC"
export PATH=/usr/local/cuda/bin:$PATH GPU_ARCH=89
bash ringlpn/scripts/build_component.sh linear-fc      # the --reference checker binary
bash ringlpn/scripts/build_component.sh direct-ot-fc
python3 ringlpn/scripts/run_direct_ot_fc_baseline.py --mode direct-only \
  --direct ringlpn/bin/test_direct_ot_fc_preprocess \
  --reference ringlpn/bin/test_two_party_fc_preprocess \
  --output "/tmp/direct-ot-fc-$(date -u +%Y%m%dT%H%M%S).json"
```

- Direct producers run on the CPU with CUDA hidden; the checker locks GPU3,
  which must be idle. The output file and its `.plan.json` must not exist yet.
- `--mode compare` interleaves the same `100x64x10` q128/bw32 workload with the
  Ring-LPN producer on GPU1 and GPU2 and locks GPU3 for the checker; all three
  must be idle. It never selects GPU0 and never reuses a failed plan.

## Source-only runtime candidate

`scripts/build_source_runtime_candidate.py` fetches public commit
`afc93c6fb94af01239f51b06d5c29a40c6fe7d84` and its CUTLASS gitlink, admits
only source, licenses, and the hashed recipe, and builds the FC producer
inside `scripts/Dockerfile.runtime`. The output is a local image plus a
Docker-save archive, not a registry publication. Roots must be absolute, new,
disjoint, and outside the repo, with existing parents; keep them outside `/tmp`.

```bash
cd "$(git rev-parse --show-toplevel)/GPU-MPC"
STAMP="$(date -u +%Y%m%dT%H%M%S)"
install -d -m 700 ~/.local/share/ringlpn/work
python3 ringlpn/scripts/build_source_runtime_candidate.py \
  --work-root ~/.local/share/ringlpn/work/runtime-$STAMP \
  --output-root ~/.local/share/ringlpn/artifacts/runtime-$STAMP
```

Restore the retained candidate with `docker image load --input runtime-candidate.docker.tar`
in `~/.local/share/ringlpn/artifacts/runtime-candidate-afc93c6/`.

## Component microbenchmarks (historical rows)

These generate the retained component results; none is the live two-process path.

| Build → run | Results |
|---|---|
| `scripts/setup_nfl.sh` (inits pinned `extern/NFLlib` and builds it; exits 0 if already built, `FORCE=1` rebuilds; needs git, cmake, make, C++, GMP/MPFR headers, e.g. inside `ringlpn-repro:2026-08-10`) → `scripts/build_bench.sh` → `scripts/run_sweep.sh` | `results/ntt/ntt_cpu*` |
| `scripts/build_cuda_bench.sh` → `scripts/run_cuda_sweep.sh` (`QBITS=32\|64\|128`), `scripts/run_cuda_single.sh` | `results/ntt/ntt_gpu_q*` |
| `scripts/build_cuda_bench_cheddar.sh` (same source, binary `bin/bench_ntt_cuda_cheddar`) | manual checks |
| `ALLOW_LEGACY_CUDA_NTT=1 scripts/build_cuda_bench_legacy.sh` → `ALLOW_LEGACY_CUDA_NTT=1 scripts/run_cuda_sweep_legacy.sh` | `results/ntt/ntt_gpu_q32_legacy*` |
| `scripts/build_ntt_gpu_ntt_baseline.sh` → `scripts/run_ntt_baseline_compare.sh` (needs `GPU_NTT_HOME`) | `results/ntt/ntt_gpu_ntt_baseline_compare.*` |
| `scripts/build_vole_bench.sh` → `scripts/run_vole_sweep.sh` | `results/vole/` |
| `scripts/build_ole_cuda_bench.sh` → `SMOKE=1 [NOISE=regular] scripts/run_ole_sweep.sh`; `scripts/run_ole_two_party_keys.sh` | `results/ole/` |
| `scripts/build_linear_ole_bench.sh` → `[NOISE=regular] scripts/run_linear_ole_sweep.sh` | `results/linear_ole/` |
| `scripts/build_orca_zp_bridge_test.sh` → `scripts/run_orca_zp_bridge_test.sh` | `results/orca_fc/orca_zp_bridge_*` |
| `scripts/build_orca_fc_{ringlpn_demo,ideal_ole_transcript,real_ole_transcript}.sh` → `scripts/run_orca_fc_{ringlpn_demo,ideal_ole_transcript,real_ole_transcript}.sh` | `results/orca_fc/` |
| `scripts/run_dmpf_baseline_comparison.sh`, `scripts/run_dmpf_regular_layout.sh` | `results/dpf/dmpf_*` |
| `scripts/run_native_ring_pcg_baseline.sh` | `results/pcg/` |
| `scripts/run_reverse_cuckoo_p0_adapter.sh` | `results/reports/reverse_cuckoo_p0_baseline_2026_08_04.json` |
| `scripts/run_vtune_hotspots.sh`, `scripts/run_vtune_memory.sh` (needs VTune) | `results/profiling/` |
| `python3 ../scripts/run_dpf_online_keygen_sweep.py` (runs `make dpf_online_keygen` in `GPU-MPC/` itself) | `results/dpf_online_keygen_bin16_chunk8192.{csv,md}` at the top of `results/`; retained copies live in `results/dpf/` |

`bench_ntt_cuda` accepts `--n`, `--qbits 30|32|64|128`, `--batch`, `--iters`,
and `--warmup`. Requested q32/q64/q128 map to actual 30/62/124 bits; q128
uses two 62-bit CRT limbs.
