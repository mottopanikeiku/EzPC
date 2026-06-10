# Ring-LPN Linear Layer Integration Plan for Orca

Updated: 2026-05-21

## Goal

Complete a staged Ring-LPN linear-layer integration with Orca by replacing Orca FC Beaver key generation while keeping Orca's existing online `gpuMatmulBeaver` path unchanged at first.

The integration target is:

1. generate byte-compatible Orca FC keys in `A`, `B`, `C_masked` order,
2. compute `C = A * B` from Ring-LPN OLE-to-Beaver instead of Orca's trusted-dealer plaintext matmul,
3. export the resulting shares into Orca's `Z_{2^bw}` ring correctly,
4. preserve existing Orca truncation, bias, optimizer, and online code until the FC replacement is validated,
5. then extend from forward FC to backward FC and model-level runs.

## Current Baseline

Already implemented:

- q32/q64/q128 promoted Cheddar NTT/PolyMul benchmark.
- q128 is represented as two q62 CRT residue limbs and reports actual qbits 124.
- q128 is wired into the standalone Ring-LPN VOLE prototype.
- Figure 2 SPFSS/OLE source now accepts q64/q128 and iterates over one or two q62 limbs, but saved validation summaries are still primarily q62/q64; q128 OLE summaries remain to be regenerated before a paper-parameter OLE claim.
- Ring-polynomial linear OLE-to-Beaver source now accepts q64/q128 and accumulates per CRT limb, but q128 saved summaries remain to be regenerated before a q128 linear-layer claim.
- Host scalar bridge validates dealer/oracle `Z_p -> Z_{2^bw}` and q128 CRT-to-`Z_{2^bw}` carry correction under explicit no-wrap bounds.
- Tiny Orca FC demo writes byte-compatible `A`, `B`, `C_masked` buffers and validates unchanged `gpuMatmulBeaver` for bounded q62 cases plus a bounded q128/full-32-bit `2x2x2` case; it also checks synthetic forward, `dW`, and `dX` contracts.

Main missing pieces:

- q128/CRT saved validation summaries for Figure 2 OLE.
- q128/CRT saved validation summaries for the linear OLE-to-Beaver artifact.
- secure q128/CRT export from residue-limb shares to Orca `Z_{2^bw}` shares.
- Dense scalar packing, or a documented reason to keep constant-polynomial packing for v1 correctness.
- Secure distributed conversion if claiming trusted-dealer removal.
- Real Orca FC keygen integration, then backward/training integration.

## Pipeline

### Phase 0: Freeze Contracts and Regression Gates

Keep these invariants fixed:

- Orca online matmul stays `gpuMatmulBeaver`.
- Orca FC key byte order stays `A`, `B`, `C_masked`.
- `FCLayer::readForwardKey` and `readGPUMatmulKey` remain compatible.
- Existing truncation key generation remains Orca-native in the first integration pass.

Required regression commands:

```bash
cd /home/ringlpn
./bin/bench_ntt_cuda --csv-header --n 8192 --qbits 128 --batch 2 --iters 1 --warmup 0
./bin/bench_vole_ringlpn --csv-header --n 8192 --qbits 128 --m 2 --c 2 --noise-weight 8 --iters 1 --warmup 0
./scripts/run_orca_zp_bridge_test.sh
./scripts/run_orca_fc_ringlpn_demo.sh
```

Exit gate:

- all current q128 NTT/VOLE and q62 FC bridge/demo tests pass before any refactor.

### Phase 1: Extract Reusable Ring-LPN Keygen Components

Move benchmark-only logic behind reusable helper APIs without changing behavior.

Suggested modules:

- `src/ringlpn_ntt_backend.cuh`: Cheddar table setup, qbits-to-prime-set selection, batched PolyMul wrappers.
- `src/ringlpn_ole_core.cuh`: Figure 2 OLE state, SPFSS key generation, expansion, validation helpers.
- `src/ringlpn_linear_beaver.cuh`: two-OLE-to-Beaver matrix product over ring-polynomial entries.
- `src/orca_ringlpn_fc_keywriter.cuh`: Orca-facing `A`, `B`, `C_masked` key-buffer writer.

Exit gate:

- existing `bench_ole_ringlpn_cuda`, `bench_linear_ole_ringlpn_cuda`, and `bench_orca_fc_ringlpn_demo` build against the reusable helpers and produce identical pass/fail results.

### Phase 2: Lift Figure 2 OLE to q128/CRT

Generalize OLE from one q62 modulus to a prime-limb set.

Conservative implementation:

- keep SPFSS payloads scalar over one q62 modulus,
- run the OLE construction independently per CRT limb,
- lay out buffers as `(object, prime_limb, coeff)`,
- use the promoted q128 Cheddar table schedule for NTT/PolyMul,
- validate OLE relation independently per limb:

```text
z0[p] + z1[p] == x0[p] * x1[p] mod p_i
```

Exit gate:

- `bench_ole_ringlpn_cuda --qbits 128` accepts uniform and regular noise.
- q128 OLE smoke passes for `n=8192`, then bounded sweeps pass for `n in {8192,16384}`.
- result summaries clearly report requested qbits 128 and actual qbits 124.

### Phase 3: Lift Linear OLE-to-Beaver to q128/CRT

Port the ring-polynomial matrix artifact onto the q128 OLE core.

Implementation notes:

- keep the two-OLE-to-Beaver algebra unchanged,
- accumulate `A0*B1`, `A1*B0`, `A0*B0`, and `A1*B1` per CRT limb,
- preserve shared operand reuse checks for `A[row,k]` and `B[k,col]`,
- validate matrix outputs per limb before any scalar export.

Exit gate:

- `bench_linear_ole_ringlpn_cuda --qbits 128` passes uniform and regular smokes.
- validation still checks shared operands and matrix-product correctness.
- summaries include key bytes, OLE count, limb count, and per-limb validation.

### Phase 4: q128 CRT Export to Orca Rings

Define and test the q128 analogue of the existing q62 carry-corrected export.

For a CRT modulus `M = p0 * p1`, dealer/oracle export should:

1. CRT-recompose each party's residue-limb share into canonical `s0, s1 in [0, M)`,
2. compute `m = floor((s0 + s1) / M)`,
3. output `r0 = s0 mod 2^bw`,
4. output `r1 = s1 - m*M mod 2^bw`.

The scalar no-wrap condition becomes:

```text
inner * input_bound * weight_bound < M
```

Exit gate:

- host bridge test covers q62 and q128 side by side.
- q62/full-32-bit negative control remains.
- q128/full-32-bit bounded Orca FC shapes pass under explicit layer bounds.
- exported ring shares reconstruct exactly in `Z_{2^bw}`.

### Phase 5: Orca Forward FC Key Writer

Build a real forward-FC key writer that can be called from Orca key generation under a feature flag.

Initial scope:

- FC only.
- Forward only.
- `tf=None` first, then existing truncation path.
- Constant-polynomial one-scalar-per-polynomial packing first.
- Dealer/oracle q128 export first.

Feature flag:

```text
ORCA_RINGLPN_FC_KEYS=1
```

Integration point:

- replace only the `A`, `B`, `C_masked` generation inside `FCLayer::genForwardKey`.
- keep output mask generation, bias handling, and truncate key generation Orca-native.

Exit gate:

- tiny FC demo uses the same library function as Orca.
- Orca `gpuKeygenMatmul` baseline and Ring-LPN key writer match for bounded small shapes.
- a local Orca forward-only FC/inference smoke passes with the feature flag enabled and disabled.

### Phase 6: Packing Strategy

After correctness is locked, replace one-scalar-per-polynomial packing if performance matters.

Required proof/test sequence:

1. define slot layout for input and weight tensors,
2. prove which polynomial coefficients contain valid dot products,
3. account for negacyclic wraparound signs,
4. add host oracle tests for packing/unpacking,
5. add GPU tests before model-level benchmarking.

Exit gate:

- packed and constant-polynomial paths agree on small FC layers.
- packed path reports slot utilization and effective scalar products per Ring-LPN OLE expansion.

### Phase 7: Backward FC and Training Keys

Extend the same key-writer interface to the two backward FC matmuls:

- `dW`: `pdW`, using `X` and incoming gradient masks.
- `dX`: `pdX`, using incoming gradient and `W` masks when `computedX` is true.

Keep optimizer, bias-gradient, and truncation keys Orca-native until the matmul replacement is stable.

Exit gate:

- forward FC, `dW`, and `dX` unit tests pass independently.
- a one-epoch tiny training smoke passes with the feature flag enabled.
- feature-flag-off behavior remains unchanged.

### Phase 8: Model-Level Validation and Benchmarks

Run progressively larger Orca workloads:

1. synthetic FC-only model,
2. P-SecureML forward/inference subset,
3. P-LeNet FC layers,
4. P-AlexNet FC layers,
5. training path after backward FC passes.

Report separately:

- Ring-LPN keygen time,
- SPFSS/OLE expansion time,
- CRT export time,
- Orca online time,
- key bytes per party,
- packing utilization,
- validation mode and modulus mode.

Exit gate:

- benchmark reports distinguish q62 bounded demos from q128/CRT claims.
- no paper table mixes trusted-dealer, dealer/oracle export, and secure-conversion claims.

## Security Boundary

There are two distinct milestones:

- Orca-compatible integration: Ring-LPN generates byte-compatible Beaver keys for Orca and the online path works.
- Trusted-dealer removal: parties can obtain/export the needed shares without a dealer/oracle seeing both shares.

The first milestone can use dealer/oracle q128 CRT export. The second requires a secure `Z_M -> Z_{2^bw}` share conversion protocol or a cited construction with a local implementation plan.

Do not claim trusted-dealer removal until that second milestone is complete.

Professor-facing protocol memo:

- `results/dealerless_orca_ringlpn_protocol_plan.tex` is the current research-checked academic-style writeup for the dealerless direction. It separates the existing Orca-compatible dealer/oracle demo from the intended two-party protocol, states the OLE cross-term algebra, identifies secure `Z_M -> Z_{2^bw}` conversion and Ring-LPN parameter auditing as the main theory gaps, and lists the next deliverables.

## Immediate Next Task

Start with a validation-and-protocol checkpoint rather than a broader Orca rewrite:

1. regenerate saved q128 OLE summaries for uniform and regular noise using the existing q128 limb plumbing,
2. regenerate saved q128 linear OLE-to-Beaver summaries for uniform and regular noise,
3. add an ideal dealerless FC transcript test that computes cross terms through an ideal OLE oracle and writes party-local Orca key buffers,
4. write the parameter/factorization audit for the selected primes and `X^N+1`,
5. design the secure `Z_M -> Z_{2^bw}` share-conversion protocol before claiming trusted-dealer removal.

This is the shortest route from the current codebase to an honest q128 Ring-LPN linear-layer bridge for Orca without overstating the dealerless security claim.

## Update (2026-06-05): Dealerless roadmap Steps 1-2 landed

Two new standalone artifacts move the work from "byte-compatible oracle keywriter"
toward the dealerless protocol. Both are additive (no change to baseline Orca or to
the feature-flagged keywriter), and both keep the existing oracle as the reference.

### Step 1: Ideal-OLE dealerless FC transcript (proves the reduction, not just the format)

- `src/orca_fc_ideal_ole_transcript.cuh`, `src/bench_orca_fc_ideal_ole_transcript.cu`,
  `scripts/build_orca_fc_ideal_ole_transcript.sh`, `scripts/run_orca_fc_ideal_ole_transcript.sh`.
- Unlike `buildCShare` (which multiplies the *clear* masks and shares the product), the
  transcript samples `A_i, B_i, Y_i` per party and forms the Beaver cross terms
  `A0*B1`, `A1*B0` through an **ideal OLE oracle**, accumulating each party's Beaver
  share over `Z_M` and converting once per output entry. It writes party-local
  `A_i || B_i || C_i` buffers that pass the **unchanged** `gpuMatmulBeaver`.
- Scope: single q62 limb (qbits=64). Because the OLE multiplies *full-width* shares
  (not bounded masks), the conservative no-wrap bound is `K * 2^(2*bw+2) < p62`, so the
  demo default `bw=16` works; `bw=32` needs the q128/CRT per-limb extension (folds into
  Step 3 below). Validated 2x2x2, 2x3x2, 3x2x2, 4x4x4 (bw16) and 2x2x2 (bw20): all pass,
  with `#OLE = 2*M*K*N` and `#conversions = M*N` reported per case.
- Still ideal-oracle: the OLE is a trusted functionality (Step 5 replaces it with the
  Figure 2 engine) and the conversion is still the carry-correction oracle (Step 2 below).

### Step 2: Secure Z_M -> Z_{2^bw} conversion prototype (the central protocol gap)

- `src/test_secure_convert.cpp`, `scripts/build_secure_convert_test.sh`,
  `scripts/run_secure_convert_test.sh` (host-only g++).
- Party-separated semi-honest protocol: edaBit-masked open of `S = z0+z1`, a boolean
  ripple comparator (public `A` + boolean-shared `R`) to extract the wrap bit
  `w = [S >= M]` via boolean Beaver triples, a daBit B2A of `w`, and the local
  correction `r_i = (z_i - M*w_i) mod 2^bw`. Matches the oracle `exactZmToRingShares`
  **bit-for-bit** on randomized, forced-wrap, and layer-shaped (bounded-dot) vectors;
  q64 and q128 moduli both pass with zero mismatches.
- Measured cost per converted scalar (the Route-A vs Route-B input):
  - q64 (ell=63): ~124 AND triples, 63 edaBit bits, 1 daBit, 375 opened bits, 125 seq. rounds.
  - q128 (ell=125): ~248 AND triples, 125 edaBit bits, 1 daBit, 747 opened bits, 249 seq. rounds.
- Honest scope: the edaBits/daBits/boolean-triples are produced by a labeled prototype
  offline dealer; in the full dealerless system these come from PCG/OT (silent OT ->
  edaBits). The high sequential-round count is the ripple's; the standard edaBits
  constant-round comparison removes it at the cost of more correlated randomness.

### Implication for Route A vs Route B (kept open per user decision)

Conversion cost is ~2*ell boolean AND triples + ell edaBit bits + an ell-bit opening
per *output element* (one conversion per output, independent of K). For FC layers with
large inner dim K, this amortizes against the 2K OLE cross-term calls; for small K or
constant-polynomial packing it is comparable to the per-output OLE work. This is the
concrete measurement needed before committing to prime-field+convert (Route A) vs the
Z_2^k-native triple route (Route B). Next: Step 3 (regenerate q128 OLE/linear summaries)
and Step 5 (replace the ideal OLE oracle with the Figure 2 engine).

## Update (2026-06-10): Step 5 landed — real-OLE slot-packed transcript; NTT backend improved

### Step 5: Real Figure 2 OLE replaces the ideal oracle (with dense slot packing)

- `src/bench_orca_fc_real_ole_transcript.cu`,
  `scripts/build_orca_fc_real_ole_transcript.sh`,
  `scripts/run_orca_fc_real_ole_transcript.sh`; memo in
  `results/orca_fc_real_ole_transcript_memo.md`.
- The Figure 2 engine (`bench_ole_ringlpn_cuda.cu`, included via its
  `RINGLPN_OLE_DISABLE_MAIN` guard) produces random ring OLEs; the fully-split
  primes make the forward negacyclic NTT a slot isomorphism, so one ring OLE
  backs up to n scalar OLEs. Cross terms are derandomized per slot (open
  `d = a - X0[s]`, `e = b - X1[s]`), accumulated per limb, Garner-lifted to
  Z_M per party (q128), converted, and written in Orca key order.
- Suite: 9/9 pass through unchanged `gpuMatmulBeaver` (q64 bw<=16, q128 bw=32,
  uniform+regular). Ring-OLE count is `2*limbs*ceil(MKN/n)`: the
  q64 16x32x16 case backs 16,384 ideal-OLE-equivalents with 2 ring OLEs.
  **This resolves the dense-packing gap** (evaluation-domain packing; no
  negacyclic sign handling needed in slots).
- Remaining oracle boundaries: centralized SPFSS keygen; conversion oracle in
  the transcript (the secure prototype from Step 2 is not yet wired in);
  c=2/t=8 correctness parameters. Removal plan:
  `results/dealerless_orca_ringlpn_full_proposal_2026_06_10.tex` (six
  milestones M1-M6 with gates; M1 = OT-based distributed DPF keygen).

### NTT backend (cheddar) improvements

- Adaptive fused-INTT polymul: Hadamard product folded into the INTT phase-1
  load when `batch*primes <= 16` (saves a launch + a full coefficient-vector
  round trip; ~2-8% at OLE batch sizes, large-batch path unchanged where the
  separate kernel was faster). Env: `RINGLPN_NTT_NO_FUSE`,
  `RINGLPN_NTT_FORCE_FUSE`.
- OLE engine caches `NTT(a)` and `NTT(a_i*a_j)` and uses
  `run_polymul_prepared_lhs` in the x/z phases: half the forward NTTs per
  expand iteration.
- All polymul/NTT validation passes (2^13-2^20, q32/64/128, both modes); OLE
  expand unchanged (SPFSS dominates) — re-confirming the keep-Cheddar /
  defer-four-step decision and pointing the optimization budget at the
  SPFSS/OT side (see proposal M1).
