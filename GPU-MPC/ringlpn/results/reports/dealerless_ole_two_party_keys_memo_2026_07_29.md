# The real Ring-LPN OLE engine running on two-party dealerless keys (2026-07-29)

**One sentence:** the Figure 2 Ring-LPN OLE engine's centralized-keygen oracle
(`build_spfss_keys()`) is replaceable by **two OS processes talking real
oblivious transfer**, and the engine's own validation passes on those keys in
all four deployed configurations (q64/q128 × uniform/regular).

This is milestone **M2's core gate**: "the Figure~2 engine validates with
`build_spfss_keys()` replaced by the two-party protocol." It is *not* the full
nine-case flagship FC composition and not a security result. Each keygen
process now samples only its own noise using OpenSSL's private DRBG, writes its
own record/key, and the engine loads those matching records and keys. OT is
IKNP rather than silent OT, expansion/validation still run in one process, and
the DPF distribution/single-key privacy reductions remain open. The deployed
expansion uses four domain-separated AES calls with full 128-bit child seeds
and separate control-bit outputs; host/device parity and GPU evaluation are
gated separately.

## 1. Why the structure fits exactly

For one polynomial pair `(i,j)` and noise group, the sparse product the engine
needs is

```text
F_(i,j)(x) = sum_(k,l) u_(i,k) * v_(j,l) * [x == a_(i,k) + b_(j,l)]  mod p,
```

so every point has an **additively shared position** (`a` from party 0, `b` from
party 1) and a **multiplicatively shared payload** (`u` times `v`). That is
precisely the input shape of the corrected two-party DPF protocol frozen in the
S1 contract, which is why no protocol change was needed - only plumbing.

Position ranges line up too: uniform noise has `a,b < n` and domain `2n`, regular
noise has bucket offsets `< n/t` and domain `2n/t`, and the protocol requires
each summand `< 2^(L-1)`, which is exactly `n` and `n/t` respectively.

## 2. Pipeline

```bash
cd GPU-MPC/ringlpn
for q in 64 128; do
  QBITS=$q NOISE=regular scripts/run_ole_two_party_keys.sh
  QBITS=$q NOISE=uniform scripts/run_ole_two_party_keys.sh
done
# ... independently sampled per-party noise and matching two-party keys
```

Two stages, both in the runner:

1. Two independent OS processes of `test_two_party_spfss_keygen` per limb each
   sample **only their own** `RLPNNOIS` record using OpenSSL's private DRBG,
   then run the shared keygen over TCP with real IKNP/Gilboa OT/OLE and write
   **only their own** `RLPNSPF1` key file.
2. `bench_ole_ringlpn_cuda` with both `RINGLPN_OLE_NOISE=<prefix>` and
   `RINGLPN_OLE_SPFSS_KEYS=<prefix>` loads the persisted records/keys instead
   of centrally sampling noise or calling `gpuKeyGenDPFZpPair`, then runs the
   unchanged expansion and `validation` / `host_validation` checks.

All hooks are environment-gated: with none set the benchmark behaves exactly
as before. `RINGLPN_OLE_EXPORT_NOISE` remains only as the older deterministic
benchmark-export mode; it is not used by the current two-party gate.

## 3. Measured (fresh run 2026-08-04; RTX 5000 Ada, loopback, single-threaded keygen)

Keygen transcript per limb (`results/ole/ole_two_party_keygen_*.csv`), 256 trees
per limb at `(c,t)=(2,8)`:

| config | limbs | trees/limb | `L` | groups | direction switches/batch | P0 bytes | P1 bytes | keygen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q64 uniform | 1 | 256 | 14 | 1 | 89 | 1,033,792 | 1,057,600 | 0.9296--0.9311 s |
| q64 regular | 1 | 256 | 11 | 15 | 71 | 969,856 | 993,664 | 0.1289--0.1292 s |
| q128 uniform | 2 | 256 | 14 | 1 | 89 | 1,033,792 | 1,057,600 | 0.9105--0.9143 s/limb |
| q128 regular | 2 | 256 | 11 | 15 | 71 | 969,856 | 993,664 | 0.1281--0.1287 s/limb |

Engine result on those keys (`results/ole/ole_two_party_keys_*.csv`):

| config | spfss domain | validation | host validation | correct | pair key bytes |
|---|---:|---|---|---:|---:|
| q64 uniform | 16,384 | pass | pass | 1 | 141,504 |
| q64 regular | 2,048 | pass | pass | 1 | 116,544 |
| q128 uniform | 16,384 | pass | pass | 1 | 283,008 |
| q128 regular | 2,048 | pass | pass | 1 | 233,088 |

Per-tree correlation and opening counts are unchanged from the frozen contract
(`1,908` logical / `3,816` meaningful share bits at `L=14`; `1,512` / `3,024`
at `L=11`), and `transcript_accounting=pass` on every row. Direction switches
are an implementation counter, not measured network rounds.

Two readings worth keeping:

1. **Regular noise is about 7.2x cheaper to key than uniform** here
   (0.1289--0.1292 s versus 0.9296--0.9311 s for the same 256 trees) purely
   because its domain is `2^11` instead of `2^14`: distributed keygen cost is
   dominated by full-frontier expansion, so the noise layout that helps the
   expander helps the key generator too. This is the same
   effect that made the DMPF encoder advantage collapse at the deployed layout
   (see `s2_architecture_comparison_2026_07_29.md`).
2. **Stage count depends only on depth** (89 at `L=14`, 71 at `L=11`) for the
   whole 256-tree batch, so latency amortises over the batch rather than
   multiplying with it.

The persistent-AES host PRG optimization remains visible against its dated
6.15 s predecessor: the current q64-uniform keygen is 0.9296--0.9311 s
(about 6.6x faster) with the device-parity gate still exact.

## 4. What is still an oracle

- **Noise sampling is party-local in the setup gate.** Each independent keygen
  process samples and persists only its own record. The single-process engine
  later reads both records solely to run the current composition and validation
  artifact; a live two-process expansion must remove that paired read.
- **Conversion is not integrated.** Standalone `test_secure_convert` uses real
  OT-backed correlations in two processes. Its tested transcript never
  reconstructs or explicitly opens the wrap bit; this is syntactic/correctness
  evidence, not a `P-CONV` privacy or composition proof. The flagship FC
  transcript still calls `exactZmToRingShares()`.
- **Expansion measurement** is one process. The keys are dealerless; the
  benchmark that consumes them is still a single-process expansion benchmark.
- **Silent OT** is not used; IKNP is OT extension, so setup bytes are an upper
  bound for this protocol shape.
- **No security claim.** `D-SEED`, `P-RNG`, `P-DIST`, `P-KEY`, `P-PCG` and
  `P-CONV` in `dealerless_orca_fc_security_contract_2026_07_29.md` remain open,
  and no parameter set is pinned.

## 5. Gate

Wired into the required-GPU checkpoint gate
(`scripts/run_paper_checkpoint_smoke.sh`) at q64/q128 with uniform/regular
noise. The complete command must end `[paper-smoke] ALL GATES PASS`.
