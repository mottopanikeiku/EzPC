# Two-process distributed DPF key generation on a real transport (2026-07-29)

**One sentence:** the frozen S1 keygen protocol now runs as **two OS processes
over TCP with real oblivious transfer**, level-synchronously batched so the
measured direction-switch count is `6L+6` for depth `L` **independent of how
many trees are in the batch**, and 369/369 generated key pairs validate through
the *unchanged* evaluator while reproducing the contract's opening accounting
bit-for-bit.

This closes the "ideal transports" half of the S1 artifact gap. It does **not**
close the PRG/seed-format obligations (`D-SEED`, `P-RNG`, `P-KEY`): the DPF
expansion PRG is still `spfss_host`'s non-cryptographic splitmix64, because the
independent consumer `spfss_host::dpfEvalAll` is unmodified. **No 128-bit
security claim is made here.**

## 1. What became real

| Element | Single-process prototype (`test_distributed_dpf_keygen`) | This artifact (`test_two_party_dpf_keygen`) |
|---|---|---|
| Processes | one, party-tagged state | **two OS processes**, no shared memory |
| Channel | in-process function calls | **two TCP sockets** (`sci::NetIO`) |
| 1-of-2 string OT | ideal functionality with counters | **IKNP OT extension** over Naor-Pinkas base OTs (unmodified `SCI/src/OT/split-iknp.h`, `SCI/src/OT/np.h`) |
| Boolean AND triple | ideal correlation oracle | **two 1-bit OTs per triple** (Gilboa cross terms) |
| `Z_p` scalar OLE | ideal correlation oracle | **Gilboa OLE**, `ceil(log2(p-1)) = 62` field-element OTs per OLE |
| Party root seed | benchmark PRG stream | **OpenSSL private CSPRNG** (`RAND_priv_bytes`), never shared |
| Scheduling | one tree at a time | **level-synchronous batch**: one OT batch and one opening per level for the whole batch |
| Key format | in-memory struct | **versioned little-endian serialization** (`src/dpf_key_io.h`, magic `RLPNDPF1`) |
| Correctness check | in-process, same run | **separate offline binary** reading both key files after both parties exit |
| Wire cost | not measurable | **measured bytes and direction switches per party** |

Party 0 is the TCP listener and SCI's `ALICE`; party 1 connects and is `BOB`.
Two sockets carry the two OT directions exactly as SCI's `OTPack` does, and the
schedule is fixed (party-0-as-sender first at every step) so the pair is
deadlock-free without extra synchronisation.

## 2. Reproduce

```bash
cd GPU-MPC/ringlpn
scripts/build_two_party_dpf_keygen.sh          # needs OpenSSL only
BASE_PORT=44500 scripts/run_two_party_dpf_keygen.sh
# prints "[two-party-dpf] all configurations pass"
```

It is also wired into the canonical checkpoint gate
(`scripts/run_paper_checkpoint_smoke.sh`). Artifacts:

- `results/dpf/two_party_dpf_keygen_2026_07_29.csv` — one row per party per
  configuration;
- `results/dpf/two_party_dpf_validate_2026_07_29.csv` — offline validation rows;
- `results/dpf/two_party_dpf_keygen_2026_07_29.log` — raw stdout and stderr.

The displayed CSV paths were rerun on 2026-08-03 after moving party-private
randomness to OpenSSL's private DRBG; all 369 pairs and controls passed.

Build dependency note: SCI's OT and IO headers are header-only, so the binaries
link only `-lcrypto -lssl -pthread`. SEAL, GMP and libOTe are **not** needed.
Instruction sets required by those headers: AES-NI, SSE4.1, PCLMUL, AVX2,
RDSEED.

## 3. Measured transcript

Both parties independently report identical correlation and opening counts per
tree; the byte columns differ because the IKNP receiver also transmits its
choice corrections. `selftest=pass` means the labelled test-only primitive
checks (opened triples and OLE shares) found 0 failures.

### 3.1 Depth sweep (both CRT primes)

| `L` | prime | batch | string OTs/tree | triple OTs/tree | OLE OTs/tree | logical bits/tree | meaningful share bits/tree | P0 batch bytes | P1 batch bytes | direction switches/batch | us/tree |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | q62 | 8 | 8 | 6 | 186 | 588 | 1,176 | 45,230 | 46,742 | 30 | 662 |
| 8 | q62 | 8 | 16 | 14 | 186 | 1,116 | 2,232 | 63,254 | 64,766 | 54 | 1,080 |
| 11 | q62 | 4 | 22 | 20 | 186 | 1,512 | 3,024 | 62,974 | 63,718 | 72 | 3,102 |
| 14 | q62 | 2 | 28 | 26 | 186 | 1,908 | 3,816 | 68,671 | 75,175 | 90 | 7,214 |
| 8 | q62b | 8 | 16 | 14 | 186 | 1,116 | 2,232 | 63,254 | 64,766 | 54 | 1,038 |
| 14 | q62b | 2 | 28 | 26 | 186 | 1,908 | 3,816 | 68,671 | 75,175 | 90 | 7,133 |

Measured direction-switch count per batch is exactly `6L+6` (30, 54, 72, 90
at `L`=4, 8, 11, 14). Setup, once per connection per party: **256 base OTs,
21,829 bytes, 6 direction switches** (two Naor-Pinkas base-OT batches, one per
direction).

### 3.2 Batch scaling at `L=11` — the point of batching

| batch trees | direction switches/batch | P0 bytes/tree | us/tree | per-tree correlations and openings |
|---:|---:|---:|---:|---|
| 1 | 72 | 52,626 | 11,163 | 22 / 20 / 186, 1,512 / 3,024 |
| 16 | 72 | 6,524 | 734 | identical |
| 64 | 72 | 4,349 | 336 | identical |
| 256 | 72 | 3,789 | 148 | identical |

Three readings:

1. **Direction switches are depth-bound, not batch-bound.** 72 switches at every batch size.
   Round-trip latency therefore amortises across the whole batch, which is what
   the pipeline needs (it consumes thousands of DPFs per epoch).
2. **Bytes per tree fall 13.9x** from batch 1 to 256, because the IKNP extension
   and base-OT setup costs amortise; the residual 3.8 kB/tree is the actual
   per-tree payload floor of this protocol shape.
3. **Wall clock per tree falls 75x** (11.2 ms -> 148 us) on loopback. This is a
   scheduling result, not a cryptographic one, and it is single-threaded CPU.

Timings are loopback, single-threaded; they are not a throughput claim. The
direction-switch count characterizes this implementation's schedule and is
not a measured network-round count.

### 3.3 Closed forms, gated in-process

- string OTs `= 2L`; triple OTs `= 2(L-1)`; OLE OTs `= 3*ceil(log2(p-1)) = 186`;
- bit triples `= L-1`; scalar OLEs `= 3`;
- logical opened bits `= 2(L-1) + 130L + ceil(log2 p)`;
- meaningful share bits `= 4(L-1) + 260L + 2*ceil(log2 p)`;
- measured direction switches per batch `= 6L+6` (measured;
  batching-dependent, not fixed by the contract).

**The logical/meaningful-share columns are bit-identical to the frozen
contract's closed
forms and to the ideal-functionality prototype's CSV** (588/1,176 at `L=4`
through 1,908/3,816 at `L=14`), at every batch size. That agreement is the
cross-check between the protocol-logic artifact and the real transport, and it
also shows batching changed the schedule without changing the transcript.

## 3.4 GPU-consumable keys from the two-party protocol

The same two-process protocol can expand with a host PRG that is **bit-identical
to the deployed Ring-LPN GPU device PRG**, so the keys it emits are consumed by
the GPU evaluator used by the Ring-LPN OLE engine.

- `src/gpu_aes_prg_host.h` reproduces `aes_prg_expand` from
  `src/gpu_spfss_zp.cuh` with four domain-separated AES calls. Plaintexts 0 and
  2 produce full 128-bit child seeds; plaintexts 1 and 3 produce separate
  control bits.
- `src/dump_gpu_aes_prg_vectors.cu` regenerates device ground truth on every
  gated run (`results/dpf/gpu_aes_prg_vectors_2026_07_29.csv`, 16 vectors
  including all-zero, LSB-only, and all-ones full-width seeds).
- `host_bin/test_gpu_aes_prg_parity` recomputes every vector on the host: 0
  left/right/tag mismatches, plus a seed-sensitivity control.
- `test_two_party_dpf_keygen --prg gpu-aes` runs the identical protocol with that
  PRG; the transcript, correlation counts, and accounting are unchanged.
- `bin/test_two_party_gpu_dpf_eval` (TEST-ONLY, offline) builds
  `ringlpn_spfss_zp::GPUDPFZpKey` for each party from the two key files and runs
  `gpuDpfZpFullEvalSum`, the same expansion entry point the Ring-LPN OLE engine
  uses.

Measured on one RTX 5000 Ada (`scripts/run_two_party_gpu_dpf.sh`,
`results/dpf/two_party_gpu_dpf_2026_07_29.csv`):

| `L` | prime | keys | batched SPFSS mismatch | per-tree pass | root seed low-bit ones | public-material mismatch | negative control |
|---:|---|---:|---:|---:|---:|---:|---|
| 4 | q62 | 8 | 0 | 8/8 | 9 | 0 | failed as expected |
| 8 | q62 | 16 | 0 | 16/16 | 16 | 0 | failed as expected |
| 11 | q62 | 32 | 0 | 32/32 | 35 | 0 | failed as expected |
| 11 | q62b | 32 | 0 | 32/32 | 34 | 0 | failed as expected |

88 key pairs, two checks each: the batch must reconstruct
`sum_b beta_b [x = alpha_b]` (SPFSS semantics, as the OLE engine consumes it) and
every single tree must reconstruct `beta_b [x = alpha_b]`. What this does **not**
claim: the keygen itself is still CPU-side (two processes); this is GPU *key
compatibility and GPU-validated correctness*, not a GPU implementation of the
keygen. Full 128-bit seed/tag separation removes the prior encoding defect, but
the joint-key distribution proof, single-key privacy reduction, and complete
CSPRNG-state/composition audit remain open. No end-to-end 128-bit DPF-security
claim is attached.

## 4. Validation and controls

`host_bin/test_two_party_dpf_validate` runs **after both parties exit** and only
reads files:

- **369/369** key pairs across ten configurations reconstruct
  `beta * [x == alpha]` over the full domain through the unchanged
  `spfss_host::dpfEvalAll`, with `alpha = off_0 + off_1` and
  `beta = beta_0 * beta_1 mod p` recomputed from each party's own recorded
  private input;
- public material must match on both sides (`sCW`, `tLCW`, `tRCW`, `finalCW`)
  while seeds and control bits must differ — checked, 0 mismatches;
- corrupted-key negative control (`finalCW + 1`) fails as expected on every
  configuration, so the pass is not vacuous;
- primitive self-tests inside the protocol binary (`--selftest 16`) open triples
  and OLE shares in a labelled test-only mode: 0 triple failures and 0 OLE
  failures on every configuration;
- every batch covers two deterministic edges (`alpha = 0` with `beta = p-1`, and
  `alpha = 2^L-2` with `beta = 1`) plus random legal inputs.

The `.testmeta` files exist only so an offline checker can recompute
`alpha`/`beta`; they are explicitly test-only and are never read by the
protocol.

## 5. Honest boundaries

- **IKNP is OT extension, not silent OT.** Ferret/Silver-class silent OT would
  cut the setup material; the byte counts above are an upper bound for this
  protocol shape, not a silent-OT figure.
- **Semi-honest only**, authenticated point-to-point channels assumed. No
  malicious security, no active-attack handling, no side-channel scope.
- **Non-cryptographic expansion PRG** (splitmix64) inherited from the unchanged
  evaluator; the GPU path additionally still packs a control bit into the seed
  LSB. `D-SEED`, `P-RNG`, `P-DIST` and `P-KEY` stay open.
- **Loopback measurements.** No WAN/LAN profile, no bandwidth cap, no latency
  injection.
- This artifact generates DPF keys for the SPFSS/Ring-LPN pipeline; it is not
  yet wired into the GPU OLE transcript, and the FC preprocessing pipeline still
  runs in one process with dealer-labelled conversion correlations.
- Keygen is CPU-side. GPU-side batched keygen and byte-identical GPU key
  emission remain M1 work.

## 6. Where this sits in the plan

Milestone M1 requires real silent OT/OLE transport, GPU batching and bytes, and
round/traffic measurement. This artifact discharges the **real two-party
transport, level-synchronous batching, measured bytes, and measured
direction-switch** parts for host keygen with real OT rather than silent OT.
Direction switches are not a network-round measurement. Still open for M1: a
silent-OT backend, GPU-side batched keygen, byte-identical GPU key emission,
and driving the real-OLE GPU transcript from these keys (M2).
