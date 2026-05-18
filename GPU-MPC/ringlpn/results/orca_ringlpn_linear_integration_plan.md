# Ring-LPN Linear Layer Integration Plan for Orca

Updated: 2026-05-18

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
- Figure 2 SPFSS/OLE artifact is validated for single-prime q62 only.
- Ring-polynomial linear OLE-to-Beaver artifact is validated for single-prime q62 only.
- Host scalar bridge validates dealer/oracle `Z_p -> Z_{2^bw}` carry correction.
- Tiny forward-only Orca FC demo writes byte-compatible `A`, `B`, `C_masked` buffers and validates unchanged `gpuMatmulBeaver` for bounded q62 cases.

Main missing pieces:

- q128/CRT support in Figure 2 OLE.
- q128/CRT support in the linear OLE-to-Beaver artifact.
- q128/CRT export from residue-limb shares to Orca `Z_{2^bw}` shares.
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

## Immediate Next Task

Start with Phase 2:

1. generalize `OleState` from one modulus to a vector of CRT limb contexts,
2. make q128 run OLE independently over both q62 limbs,
3. validate per-limb OLE correctness,
4. then port the linear artifact in Phase 3.

This is the shortest route from the current codebase to an honest q128 Ring-LPN linear-layer bridge for Orca.
