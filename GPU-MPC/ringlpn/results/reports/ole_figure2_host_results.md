> **Historical record (2026-04-21).** The 135/57/36 host validation counts below were re-verified unchanged on 2026-06-10 (`reports/baseline_2026_06_10.md`); surrounding prose may reference an older roadmap.

# Figure 2 (Ring-LPN OLE from SPFSS) — Host Correctness Artifact

Date: 2026-04-21
Status: **end-to-end correctness validated on host**; GPU acceleration of the
x_σ / z_σ polymul path is a follow-up.

## What was built

| Artifact | File | Purpose |
| --- | --- | --- |
| Plaintext oracle | [src/verify_figure2_expand.cpp](../src/verify_figure2_expand.cpp) | Ground truth: sparse-noise Figure 2 algebraic identity, no sharing |
| DPF + SPFSS (Z_p payload) | [src/spfss_host.h](../src/spfss_host.h), [src/spfss_host.cpp](../src/spfss_host.cpp) | Host DPF with Z_p final-level correction word; SPFSS = sum of m DPF evals |
| SPFSS unit test | [src/test_spfss.cpp](../src/test_spfss.cpp) | `share0[x] + share1[x] == Σ_k β_k·[x==α_k]` mod p |
| OLE Expand bench | [src/bench_ole_ringlpn_host.cpp](../src/bench_ole_ringlpn_host.cpp) | Full Figure 2 Expand, `z_0 + z_1 == x_0·x_1 mod (X^N+1)` |
| Build script | [scripts/build_ole_host.sh](../scripts/build_ole_host.sh) | One-shot g++ build; no CUDA dependency |

## Configuration

- Modulus: single 62-bit NTT-friendly prime `p = 4611686018326724609`
  (kConfig62 in [bench_ntt_cuda_cheddar.cu](../src/bench_ntt_cuda_cheddar.cu)).
- Ring: `R = Z_p[X] / (X^N + 1)` (negacyclic). X^N wraps with a sign flip.
- Noise: t-sparse uniform-position (not regular noise — regular is a planned
  follow-up configuration).
- SPFSS domain: `[0, 2N)`; each (i,j) pair ⇒ one SPFSS with `t²` point
  functions; α_{k,l} = A^i_0[k] + A^j_1[l], β_{k,l} = b^i_0[k]·b^j_1[l] mod p.
- PRG: splitmix64 (deterministic, non-cryptographic). Documented in
  `spfss_host.cpp`. A cryptographic PRG is a drop-in replacement at
  `prg_expand()`.

## Correctness sweeps

### Plaintext oracle (Figure 2 algebraic identity, no DPF)

`verify_figure2_expand` over seeds ∈ {1, 2, 3, 42, 1337} × N ∈ {64, 128, 256} ×
c ∈ {2, 3, 4} × t ∈ {4, 8, 16} ⇒ **135/135 expand_pass=1**.

### DPF + SPFSS unit test

`test_spfss` over log_domain ∈ {6, 8, 10, 12, 14} × m ∈ {1, 4, 16, 64} (with
m < 2^log_domain) × seed ∈ {1, 2, 42} ⇒ **57/57 spfss_pass=1**.

### Figure 2 OLE Expand (end-to-end)

`bench_ole_ringlpn_host` over N ∈ {32, 64, 128} × c ∈ {2, 3} × t ∈ {4, 8} ×
seed ∈ {1, 2, 42} ⇒ **36/36 ole_pass=1**. Each run validates
`(z_0 + z_1)[k] == (x_0·x_1)[k]` mod p for every coefficient k ∈ [0, N).

## What this artifact does *not* yet cover

- **GPU acceleration.** The x_σ and z_σ steps are host schoolbook O(N²) here.
  The GPU polymul path [run_polymul_prepared_lhs](../src/bench_ntt_cuda_cheddar.cu)
  used by `bench_vole_ringlpn` drops these to O(N log N) and is already
  coefficient-validated — plug it in for the benchmark binary.
- **Regular-noise distribution.** §A.2 of the draft cuts per-DPF domain by
  `log(t)` via bucketed noise positions. Reported bench numbers (follow-up)
  should use this.
- **OLE → Beaver triple.** §8's "two OLEs → one triple" is not yet implemented.
  That belongs in the Orca linear-layer plan (Phase B).
- **Security-grade PRG.** splitmix64 is for correctness; swap for AES-NI or
  ChaCha20 before any timing / security claims.
- **2-prime CRT (log p ≈ 128).** Single 62-bit prime here. CRT lift is a
  follow-up, mechanical addition.
- **GPU DPF.** Host DPF here; `gpu_dpf.cu` is payload=1 only (see
  [project_gpu_dpf_payload_limit.md](../../../.claude/memory/) note). Follow-up:
  either template `gpu_dpf.cu` or build a sibling `gpu_dpf_zp.cu` for the
  device-side SPFSS fast path.

## Reproducing

```bash
cd GPU-MPC/ringlpn
./scripts/build_ole_host.sh

# Plaintext oracle
./host_bin/verify_figure2_expand --n 128 --c 2 --t 16 --seed 1

# DPF + SPFSS unit test
./host_bin/test_spfss --log-domain 10 --m 16 --seed 1 --trials 5

# Full Figure 2 OLE Expand
./host_bin/bench_ole_ringlpn_host --n 64 --c 2 --t 8 --seed 1 --verbose
```
