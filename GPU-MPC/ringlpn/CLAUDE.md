# ringlpn — agent & human catch-up guide

**What this is:** a research subproject building *dealerless* preprocessing for
Orca (the GPU FSS-based secure ML system in this repo) from Ring-LPN
pseudorandom correlation generators (PCGs). Orca's linear layers consume
Beaver-triple keys that a trusted dealer normally produces; this project
replaces the dealer with a two-party protocol: GPU NTT/polynomial arithmetic →
Z_p SPFSS (sum of DPFs) → Figure 2 Ring-LPN OLE → slot-packed Beaver cross
terms → Z_M→Z_2^bw conversion → byte-compatible Orca keys, validated through
Orca's **unchanged** online path (`gpuMatmulBeaver`).

**Status (2026-07-21):** the GPU chain works end-to-end with real Figure 2
OLEs at q64/q128 and dense slot packing. The complete required-GPU checkpoint
gate was freshly revalidated on idle GPU 3 and ended `ALL GATES PASS`.
**The corrected M1 distributed-keygen protocol logic is implemented as a host
prototype** (`src/test_distributed_dpf_keygen.cpp`): two-party DPF keygen from
additively shared α and multiplicatively shared β, party-separated and
functionally validated by the unchanged `spfss_host::dpfEvalAll` on 2,432
trees (both primes, depths 4–14, deterministic edges, centralized reference,
corrupted-key control). Phase C now uses three scalar OLEs and opens only the
standard `finalCW`; an executable old-sign regression demonstrates why the
removed sign opening leaked one point bit when conditioned on a party's key.
The fresh host and required-GPU gates pass. This is a protocol-logic and compatibility
prototype using ideal OT/triple/OLE functionalities and the evaluator's
non-cryptographic correctness PRG — not evidence of computational privacy,
128-bit security, or M1 completion. M1 still needs AES/CSPRNG evaluation, real
silent OT/OLE, GPU batching and bytes, and round/traffic measurements. Two
pipeline oracle boundaries remain: centralized SPFSS key generation in the
GPU pipeline and dealer-style share-conversion correlations. The M1–M6
proposal is **v2.2** (stable filename:
`results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` + PDF),
with corrected Table 1/Table 5 costs and bootstrap condition
`3*c^2*t^2 < n`. See
`results/reports/session_handoff_2026_07_21.md` and
`results/reports/distributed_dpf_keygen_memo_2026_07_21.md`.
The corrected host checkpoint and v2.2 evidence are committed at `28f8451`.
The binding staged route to publication readiness is
`results/reports/publication_readiness_plan_2026_07_21.md`; every completed
stage must end in a gate-verified checkpoint commit.

## Catch up in 10 minutes (read in this order)

1. This file.
2. `results/reports/session_handoff_2026_07_21.md` — current verification,
   boundaries, paper state, and next actions.
3. `results/reports/publication_readiness_plan_2026_07_21.md` — binding
   S1--S10 execution order, gates, proof/evaluation requirements, and
   per-stage commit discipline.
4. `results/reports/distributed_dpf_keygen_memo_2026_07_21.md` — corrected
   Phase C protocol, executable controls, and regenerated D1 counts.
5. `results/README.md` — where every result/report lives and what produces it.
6. `results/reports/orca_fc_real_ole_transcript_memo.md` — real-OLE
   slot-packed transcript and NTT backend changes.
7. `results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` —
   v2.2 forward plan, cost models, and claims discipline.
8. `results/reports/baseline_2026_06_10.md` — older full-GPU environment,
   PASS counts, and performance anchors.

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
| `test_distributed_dpf_keygen.cpp` | **Corrected M1 host protocol-logic prototype (2026-07-21).** Two-party DPF keygen: secure adder for α's bits (L−1 bit triples), cancellation-lemma level walk (2 string OTs/level), and Phase C arithmetic-share multiplication (3 scalar OLEs) that opens only standard `finalCW`. Omniscient old-sign leakage regression, deterministic edge trees, centralized reference, and corrupted-key control. Emits standard `spfss_host::DPFKey`s validated by unchanged `dpfEvalAll`; ideal OT/triple/OLE interfaces and non-cryptographic splitmix64 correctness PRG mean functional compatibility, not computational privacy. `spfss_host.cpp` remains untouched. |
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
- The distributed key-generation protocol logic is implemented
  party-separated and functionally validated by the unchanged evaluator,
  using ideal OT/triple/OLE functionalities and a non-cryptographic
  correctness PRG (2,432 trees; both primes; depth ≤14; deterministic edges;
  centralized and corruption controls; old-sign leak regression). Measured
  per tree: `2*depth` string OTs, `depth-1` bit triples, 3 scalar OLEs, and
  `2*(depth-1) + 260*depth + 124` opened bits. This is not "M1 done".
- Baseline Orca is untouched with the flag off.

NOT claimable yet (the honest boundaries — never blur these):
- "Dealer removed": SPFSS keys are generated centrally; the transcript's
  conversion reads both shares (oracle); benchmark RNG is common-seed.
- Any concrete security level: c=2/t=8 are correctness parameters; the primes
  fully split the ring → **splittable** Ring-LPN assumption (stronger than
  irreducible), unaudited.
- Anything about nonlinear layers (ReLU/truncation FSS keys are a separate
  dealer axis).
- The host D1 artifact does not establish computational privacy, 128-bit
  security, GPU byte compatibility, real transports, or two-process
  isolation.

## Roadmap (proposal M1–M6, in `reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex`)

M1 distributed DPF keygen via OT (GPU level-synchronous batching) —
**host protocol-logic slice functionally validated 2026-07-21**
(`test_distributed_dpf_keygen`; remaining: AES/CSPRNG evaluator, real silent
OT/OLE transport, GPU batching, GPU byte format, and measured network bytes
and rounds) →
M2 wire M1 into the real-OLE transcript → M3 PCG-sourced conversion
correlations + log-round comparator → M4 two-process TCP deployment →
M5 splittable-Ring-LPN parameter audit (pins n, c, t; may re-pin primes —
see NTT note below; must re-check `3*c^2*t^2 < n`, coupling M5 to M1) →
M6 model-scale report. M1 ∥ M3 can run in parallel. Design components are
D1–D5 in the v2.2 proposal (D1=keygen, D2=conversion,
D3=coin-tossed seeds, D4=two-process, D5=parameter audit); Table 3 maps
G1–G5 → D1–D5 → milestone gates → claims.

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

- **This is a shared school server; the user is NOT a sudoer.** Never attempt
  privileged operations. The user is in the docker group (root-equivalent in
  principle) — use it only for ephemeral containers and for chown-ing the
  user's OWN files under `/home/fatih`; never touch system paths or other
  users' files. Check `nvidia-smi` before heavy GPU runs and prefer pinning
  with `CUDA_VISIBLE_DEVICES` — others may be working.
- `nvcc` is at `/usr/local/cuda/bin` — NOT in default PATH. `GPU_ARCH=89`
  (4× RTX 5000 Ada). Two-party tests: run party 0 and 1 with
  `CUDA_VISIBLE_DEVICES=0/1`, args `<party> 127.0.0.1`.
- Historical builds ran as root in docker (`pcg-accel:*` images; user is in
  the docker group, sudo needs a password). Root-owned files can reappear;
  fix: `docker run --rm -v <dir>:/x ubuntu:22.04 chown -R 1013:1014 /x/...`.
- `bench_ntt` (CPU NFLlib) needs libmpfr-dev — absent on host and in the
  images; build in a container with an ephemeral `apt-get install libmpfr-dev`.
- **No LaTeX anywhere** (host, orca-dev, all local images). Compile reports
  with an ephemeral container: `docker run -d --name texbuild -v <reports
  dir>:/work debian:bookworm sleep infinity`, then `apt-get install -y
  --no-install-recommends texlive-latex-base texlive-latex-recommended
  texlive-pictures` (~300 MB; covers tikz/booktabs/hyperref; avoid enumitem —
  it needs texlive-latex-extra), `pdflatex` twice in `/work`, then delete the
  root-owned .aux/.log/.out via a root container and chown the PDF to
  1013:1014. `/bin/sh` in these containers is dash: no `{a,b}` brace expansion.
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
5. **Commit every stage:** a stage is complete only after its mechanical gate
   passes, evidence and current docs are synchronized, and an atomic checkpoint
   commit is created. Preserve completed gate commits; corrections get new
   commits and rerun affected gates. See the publication-readiness plan.

## Documentation contract — BINDING for every agent working here

This file is the single source of truth for project state. Stale documentation
is worse than no documentation: it corrupts the context of whoever reads it
next. Therefore, **before ending any session that changed code, results, or
plans, you must**:

1. **Update this file** — the Status line, the source map (if files were
   added/moved), the claims split (if a boundary moved), the perf anchors
   (if regenerated), and the gotchas (if you hit a new one).
2. **Update `results/README.md`** with a row for any new artifact or report.
3. **Write or refresh the memo** in `results/reports/` for the artifact you
   touched, dated, with reproduction commands.
4. **Mark superseded documents**, never delete-without-trace and never leave
   them unmarked: prepend the standard banner
   (`> **HISTORICAL ...** superseded by CLAUDE.md; statements below may
   describe an older state`) the moment a document's claims stop being
   current. All pre-2026-06-10 reports already carry it; keep the convention.
5. **Never let two live documents disagree.** If you find a contradiction,
   the newer gate-verified statement wins; banner or fix the other on the
   spot, in the same commit.
6. Historical/outreach/archive documents are read-only context: quote them,
   don't trust them. Anything without a banner and dated 2026-06-10 or later,
   plus the gate output, is current; everything else is history.
