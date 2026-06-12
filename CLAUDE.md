# EzPC (fork) — orientation

This fork's active work is **`GPU-MPC/ringlpn/`**: dealerless Ring-LPN
preprocessing for Orca's linear layers. Read `GPU-MPC/ringlpn/CLAUDE.md`
first — it is the canonical catch-up document (source map, validated claims,
roadmap, gotchas). Results and reports are indexed in
`GPU-MPC/ringlpn/results/README.md`.

Everything else (SCI, CrypTFlow2, sytorch, the stock Orca under `GPU-MPC/`)
is upstream EzPC/Orca code — treat as read-only unless the task says
otherwise. The one deliberate upstream change is the feature-flagged keygen
path in `GPU-MPC/nn/orca/fc_layer.cu` (`ORCA_RINGLPN_FC_KEYS`; flag off =
byte-identical baseline).

**Container model:** `./start` launches/attaches the `orca-dev` docker
container, which mounts **only `GPU-MPC/` as `/home`** — so
`GPU-MPC/ringlpn` is `/home/ringlpn` inside it. Old logs/scripts that mention
`/home/...` paths mean the container. Host-side builds also work
(`nvcc` at `/usr/local/cuda/bin`); container builds run as root and leave
root-owned files (fix via docker chown; sudo needs a password). For Orca
experiment runbooks, build pipeline, and filesystem map, see
`GPU-MPC/docs/workspace_guide_2026_05_18.md` (its Ring-LPN sections are
HISTORICAL — trust `GPU-MPC/ringlpn/CLAUDE.md`).

Repo-wide gotchas: `GPU_ARCH=89`; the root `.gitignore` ignores `*.csv` so
result CSVs need `git add -f`. One-command re-validation of all
ringlpn claims:

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh   # "ALL GATES PASS"
```
