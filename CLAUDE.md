# EzPC (fork) — orientation

`AGENTS.md` is a symlink to this file. Active work is **`GPU-MPC/ringlpn/`**:
dealerless Ring-LPN preprocessing for Orca's linear layers. Read
`GPU-MPC/ringlpn/CLAUDE.md` first (canonical status, claims, gates, commands,
gotchas); evidence is indexed in `GPU-MPC/ringlpn/results/README.md`.

Everything else (SCI, CrypTFlow2, sytorch, stock Orca under `GPU-MPC/`) is
upstream EzPC code: read-only unless a task says otherwise. Fork edits to
upstream files (`git diff f24bf3e master`):
- Ring-LPN integration points: `GPU-MPC/backend/orca_base.h` (helper
  extraction), `GPU-MPC/experiments/orca/orca_inference.cu`
  (`ORCA_RINGLPN_LINEAR_INTEGRATION` role 2), `GPU-MPC/nn/orca/fc_layer.cu`
  (`ORCA_RINGLPN_FC_KEYS`). With both macros off, Orca is the stock baseline.
- Local tooling: `GPU-MPC/Makefile` (nvcc at `/usr/local/cuda/bin`,
  `dpf_online_keygen` target), `GPU-MPC/README.md`,
  `GPU-MPC/experiments/orca/config.json` (per-party key dirs, loopback peers),
  `GPU-MPC/utils/gpu_mem.cu` (pool reserve 40→25 GiB),
  `GPU-MPC/utils/gpu_file_utils.cpp` (key-write timing print), `.gitignore`,
  `.gitmodules` (`GPU-MPC/ringlpn/extern/NFLlib`).

**Container:** `./start` launches the `orca-dev` container, which mounts only
`GPU-MPC/` as `/home` (so `/home/ringlpn` = `GPU-MPC/ringlpn`). Host builds also
work. Container builds run as root; fix ownership with a docker chown (the user
has no sudo rights). Orca runbooks: `GPU-MPC/docs/workspace_guide_2026_05_18.md`.

Gotchas: shared machine (check `nvidia-smi`; never touch other users' jobs);
`GPU_ARCH=89`; `.gitignore` rules hide `*.csv`, `*.txt`, `*.sh` (except
`GPU-MPC/ringlpn/scripts/*.sh`), and ringlpn `*.pdf`, so evidence needs
`git add -f`. Full re-validation: the canonical gate in
`GPU-MPC/ringlpn/CLAUDE.md` §8. It needs three idle, distinct GPUs selected
through its `ORCA_LINEAR_*`/`FULL_GRAPH_*` variables (`CUDA_VISIBLE_DEVICES`
alone still lands on GPUs 0/1/2) and prints `[paper-smoke] ALL GATES PASS`.
