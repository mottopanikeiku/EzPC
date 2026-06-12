# ringlpn — agent & human catch-up guide

**What this is:** a research subproject building *dealerless* preprocessing for
Orca (the GPU FSS-based secure ML system in this repo) from Ring-LPN
pseudorandom correlation generators (PCGs). Orca's linear layers consume
Beaver-triple keys that a trusted dealer normally produces; this project
replaces the dealer with a two-party protocol: GPU NTT/polynomial arithmetic →
Z_p SPFSS (sum of DPFs) → Figure 2 Ring-LPN OLE → slot-packed Beaver cross
terms → Z_M→Z_2^bw conversion → byte-compatible Orca keys, validated through
Orca's **unchanged** online path (`gpuMatmulBeaver`).

**Status (2026-06-10):** the full chain works end-to-end with *real* Figure 2
OLEs (not oracles) at q64 and q128, densely slot-packed, all gates passing.
Two oracle boundaries remain before "dealerless" is honest: centralized SPFSS
key generation, and dealer-style correlations inside the share conversion.
The removal plan is the M1–M6 proposal (below).

## Catch up in 10 minutes (read in this order)

1. This file.
2. `results/README.md` — where every result/report lives and what produces it.
3. `results/reports/orca_fc_real_ole_transcript_memo.md` — the newest artifact
   (real-OLE slot-packed transcript) and the NTT backend changes.
4. `results/reports/dealerless_orca_ringlpn_full_proposal_2026_06_10.tex` —
   the forward plan (M1–M6 milestones, cost models, claims discipline).
5. `results/reports/baseline_2026_06_10.md` — environment, all PASS counts,
   perf anchors.

Then re-validate everything with one command (~15 min, needs GPU):

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  scripts/run_paper_checkpoint_smoke.sh
# must exit 0 and print "[paper-smoke] ALL GATES PASS"
```

## Source map (`src/`)

| File | What it is |
|---|---|
| `bench_ntt_cuda_cheddar.cu` | The GPU NTT backend (cheddar-derived merged-stage kernels, signed Montgomery, q32/q64/q128-CRT, negacyclic). Included by every GPU bench via `RINGLPN_DISABLE_MAIN`. Contains `run_full_polymul`, `run_polymul_prepared_lhs`, adaptive fused-INTT (`RINGLPN_NTT_NO_FUSE`/`FORCE_FUSE`), `host_polymul_reference` (the host oracle), `kConfig62`/`kConfig62Crt2` (the primes: 2^62−6·2^24+1, 2^62−7·2^24+1). |
| `gpu_spfss_zp.cuh` | GPU DPF/SPFSS with additive Z_p payloads (`gpuKeyGenDPFZpPair`, `gpuDpfZpFullEvalSum`). The expand-side workhorse — SPFSS full-domain eval dominates OLE cost. |
| `bench_ole_ringlpn_cuda.cu` | The Figure 2 Ring-LPN OLE engine (random ring OLE: z0+z1 = x0·x1 in Z_p[X]/(X^n+1)). Reusable via `#define RINGLPN_OLE_DISABLE_MAIN` + include; caches NTT(a)/NTT(a·a) across expand iterations. **`build_spfss_keys()` is the centralized-keygen oracle boundary.** |
| `bench_linear_ole_ringlpn_cuda.cu` | Ring-polynomial matrix Beaver from two OLEs per ring product. |
| `bench_vole_ringlpn.cu` | Older standalone VOLE expansion prototype. |
| `orca_fc_ringlpn_keywriter.cuh` | Host helpers + dealer/oracle keywriter used by `nn/orca/fc_layer.cu` behind `ORCA_RINGLPN_FC_KEYS` (bw≤32; fails fast otherwise; baseline Orca byte-identical with flag off). Has `exactZmToRingShares` (the conversion oracle), CRT/q128 helpers. |
| `orca_fc_ideal_ole_transcript.cuh` + `bench_orca_fc_ideal_ole_transcript.cu` | Step-1 artifact: dealerless FC transcript with an *ideal* OLE oracle. Kept as reference; superseded by the real-OLE transcript. |
| `bench_orca_fc_real_ole_transcript.cu` | **The flagship artifact.** Real Figure 2 OLE + slot packing (forward negacyclic NTT is a ring isomorphism on the fully-split ring → one ring OLE = up to n scalar OLEs) + per-slot derandomization + per-party Garner CRT lift (q128) + conversion + Orca key write, validated via unchanged `gpuMatmulBeaver`. Ring-OLE count: `2·limbs·ceil(MKN/n)`. |
| `bench_orca_fc_ringlpn_demo.cu` | Byte-compatibility demo: forward + dW + dX key contracts at q64/q128. |
| `test_secure_convert.cpp` | Step-2 artifact (host): secure party-separated Z_M→Z_2^bw conversion (edaBit-masked open, ripple comparator, daBit B2A). Bit-exact vs oracle; costs measured. Not yet wired into the transcript (proposal M3). |
| `test_orca_zp_bridge.cpp` | Carry-corrected Z_p→Z_2^bw share export + the bw=32/q62 counterexample (negative control). |
| `bench_ntt_gpu_ntt_baseline.cu` | External baseline: GPU-NTT (Ozcan–Savas) vs cheddar, same prime/psi/operation. Needs external checkout (`GPU_NTT_HOME`, default `/home/fatih/GPU-NTT`); benchmark-only, not in the gate. |
| `spfss_host.{h,cpp}`, `test_spfss.cpp`, `bench_ole_ringlpn_host.cpp`, `verify_figure2_expand.cpp` | Host reference implementations + the 135/57/36 host validation suites. |
| `test_spfss_zp_cuda.cu` | GPU SPFSS payload correctness tests. |
| `orca_globals_stub.cpp` | Defines `OneGB` for standalone benches (instead of linking the comms stack). |

One upstream touch outside this dir: `GPU-MPC/nn/orca/fc_layer.cu` — the
feature-flagged keygen integration (flag off = byte-identical baseline,
verified through the two-party `tests/nn/orca/fc` test).

## Scripts (`scripts/`)

Every artifact has a `build_*.sh` / `run_*.sh` pair; runners write
`.csv` (data) + usually `.md` (summary) + `.log` (raw stdout+stderr) into
their directory under `results/` (see `results/README.md` for the mapping).
`run_paper_checkpoint_smoke.sh` is the canonical gate: host trio + bridge +
secure convert + GPU smokes (OLE q64/q128 × uniform/regular, linear, demo,
both transcripts), ends with `ALL GATES PASS`.

## Validated claims vs. open boundaries

Safe to state (every item gate-verified):
- FC Beaver keys from **real** Figure 2 Ring-LPN OLEs validate through
  unchanged `gpuMatmulBeaver` (q62 bw≤16, q124 bw=32; uniform+regular noise).
- Slot packing resolves PCG amortization: 2 ring OLEs back 8192 scalar cross
  terms (vs 16,384 ideal-OLE calls); openings are the standard `d=a−X0`,
  `e=b−X1` derandomization messages.
- Secure conversion prototype is bit-exact with measured cost
  (124/248 AND triples, 125/249 ripple rounds per output at q64/q128).
- Baseline Orca is untouched with the flag off.

NOT claimable yet (the honest boundaries — never blur these):
- "Dealer removed": SPFSS keys are generated centrally; the transcript's
  conversion reads both shares (oracle); benchmark RNG is common-seed.
- Any concrete security level: c=2/t=8 are correctness parameters; the primes
  fully split the ring → **splittable** Ring-LPN assumption (stronger than
  irreducible), unaudited.
- Anything about nonlinear layers (ReLU/truncation FSS keys are a separate
  dealer axis).

## Roadmap (proposal M1–M6, in `reports/dealerless_*_full_proposal_*.tex`)

M1 distributed DPF keygen via OT (GPU level-synchronous batching) →
M2 wire M1 into the real-OLE transcript → M3 PCG-sourced conversion
correlations + log-round comparator → M4 two-process TCP deployment →
M5 splittable-Ring-LPN parameter audit (pins n, c, t; may re-pin primes —
see NTT note below) → M6 model-scale report. M1 ∥ M3 can run in parallel.

## Perf anchors (RTX 5000 Ada, this repo's gate configs)

| Metric | Value |
|---|---|
| OLE expand, n=8192 c=2 t=8 smoke | 13.3 ms (q64) / 26.8 ms (q128) |
| OLE expand, t=64 | 881 ms uniform / 61 ms regular (q64) |
| Linear OLE-to-Beaver 2×2×2 | 224 ms (q64) / 448 ms (q128) |
| Cheddar polymul n=8192 batch=64 | ~255–265 µs (q64) |
| **Bottleneck** | SPFSS full-domain eval; NTT is <1% of expand |

NTT decision (measured, `reports/ntt_baseline_comparison_2026_06_10.md`):
keep cheddar — external GPU-NTT merge is 1.2–3.9× faster but cannot run
62-bit primes (Barrett headroom). Revisit at M5 if primes are re-pinned to
the ≤60-bit class.

## Environment & gotchas (will bite you)

- `nvcc` is at `/usr/local/cuda/bin` — NOT in default PATH. `GPU_ARCH=89`
  (4× RTX 5000 Ada). Two-party tests: run party 0 and 1 with
  `CUDA_VISIBLE_DEVICES=0/1`, args `<party> 127.0.0.1`.
- Historical builds ran as root in docker (`pcg-accel:*` images; user is in
  the docker group, sudo needs a password). Root-owned files can reappear;
  fix: `docker run --rm -v <dir>:/x ubuntu:22.04 chown -R 1013:1014 /x/...`.
- `bench_ntt` (CPU NFLlib) needs libmpfr-dev — absent on host and in the
  images; build in a container with an ephemeral `apt-get install libmpfr-dev`.
- Root `.gitignore` ignores `*.csv` globally — committing new result CSVs
  requires `git add -f`.
- `extern/NFLlib` is a registered submodule (quarkslab/NFLlib @ 5cf40ed);
  the mnist/weights data submodules carry deliberate internal renames — leave.
- Env flags: `ORCA_RINGLPN_FC_KEYS` (+`_QBITS`,`_SEED`) for the fc_layer path;
  `RINGLPN_NTT_NO_FUSE` / `RINGLPN_NTT_FORCE_FUSE` for polymul A/B;
  `SMOKE=1 QBITS=... NOISE=...` for the sweep runners.

## House rules for new work

1. Every new artifact = source + build script + run script + CSV/MD/log in
   its `results/` subdir + a memo in `results/reports/` + a gate hook if it
   guards a claim. Suites exit non-zero on any failure.
2. Validate against an independent oracle (host reference or unchanged Orca
   online path), not against the code under test.
3. State oracle boundaries in the source header and the memo. Keep the
   "safe to claim / not yet claimable" split current.
4. Don't claim perf wins without an A/B at the consumer's actual shape
   (cf. the fused-INTT adaptive threshold story).
