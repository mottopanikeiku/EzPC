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

Repo-wide gotchas: `nvcc` lives at `/usr/local/cuda/bin` (not in PATH);
`GPU_ARCH=89`; the root `.gitignore` ignores `*.csv` so result CSVs need
`git add -f`; some files may be root-owned from docker builds (fix via
docker chown, sudo needs a password). One-command re-validation of all
ringlpn claims:

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh   # "ALL GATES PASS"
```
