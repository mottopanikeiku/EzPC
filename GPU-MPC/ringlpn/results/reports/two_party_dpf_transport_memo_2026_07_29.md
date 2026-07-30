# Two-process distributed DPF key generation on a real transport (2026-07-29)

**One sentence:** the frozen S1 keygen protocol now runs as **two OS processes
over TCP with real oblivious transfer**, produces standard `spfss_host::DPFKey`
halves that the *unchanged* evaluator accepts on 32/32 pairs across six
configurations, and reproduces the contract's opening accounting exactly while
adding the wire-byte and round measurements the single-process prototype could
not produce.

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
| Party root seed | benchmark PRG stream | **OS CSPRNG** (`std::random_device`), never shared |
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
BASE_PORT=43400 scripts/run_two_party_dpf_keygen.sh
# prints "[two-party-dpf] all configurations pass"
```

Artifacts:

- `results/dpf/two_party_dpf_keygen_2026_07_29.csv` — one row per party per
  configuration;
- `results/dpf/two_party_dpf_validate_2026_07_29.csv` — offline validation rows;
- `results/dpf/two_party_dpf_keygen_2026_07_29.log` — raw stdout and stderr.

Build dependency note: SCI's OT and IO headers are header-only, so the binary
links only `-lcrypto -lssl -pthread`. SEAL, GMP and libOTe are **not** needed.
Instruction sets required by those headers: AES-NI, SSE4.1, PCLMUL, AVX2,
RDSEED.

## 3. Measured transcript (per tree, per party)

Both parties independently report identical correlation and opening counts; the
byte columns differ because the IKNP receiver also transmits its choice
corrections.

| `L` | prime | trees | string OTs | triple OTs | OLE OTs | bit triples | scalar OLEs | logical opened bits | revealed-share bits | P0 bytes sent | P1 bytes sent | direction switches | us/tree |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | q62 | 8 | 8 | 6 | 186 | 3 | 3 | 588 | 1,176 | 23,595 | 32,979 | 44 | 5,299 |
| 8 | q62 | 8 | 16 | 14 | 186 | 7 | 3 | 1,116 | 2,232 | 40,184 | 49,568 | 84 | 8,509 |
| 11 | q62 | 4 | 22 | 20 | 186 | 10 | 3 | 1,512 | 3,024 | 52,626 | 62,010 | 114 | 11,754 |
| 14 | q62 | 2 | 28 | 26 | 186 | 13 | 3 | 1,908 | 3,816 | 65,068 | 74,452 | 144 | 15,543 |
| 8 | q62b | 8 | 16 | 14 | 186 | 7 | 3 | 1,116 | 2,232 | 40,184 | 49,568 | 84 | 9,453 |
| 14 | q62b | 2 | 28 | 26 | 186 | 13 | 3 | 1,908 | 3,816 | 65,068 | 74,452 | 144 | 15,888 |

Setup, once per connection per party: **256 base OTs, 21,829 bytes, 6 direction
switches** (two Naor-Pinkas base-OT batches, one per direction).

Closed forms, all gated in-process (`transcript_accounting=pass`):

- string OTs `= 2L`; triple OTs `= 2(L-1)`; OLE OTs `= 3*ceil(log2(p-1)) = 186`;
- bit triples `= L-1`; scalar OLEs `= 3`;
- logical opened bits `= 2(L-1) + 130L + ceil(log2 p)`;
- revealed-share bits `= 4(L-1) + 260L + 2*ceil(log2 p)`;
- observed direction switches `= 10L + 4` (measured, not derived).

**The logical/revealed columns are bit-identical to the frozen contract's
closed forms and to the ideal-functionality prototype's CSV** (588/1,176 at
`L=4` through 1,908/3,816 at `L=14`). That agreement is the cross-check between
the protocol-logic artifact and the real transport.

Timings are loopback, single-threaded, one tree at a time, and include the
primitive self-tests' base-OT amortisation; they are not a throughput claim.
Batching across trees is not implemented, so the per-tree round count is the
worst case.

## 4. Validation and controls

`host_bin/test_two_party_dpf_validate` runs **after both parties exit** and only
reads files:

- 32/32 key pairs reconstruct `beta * [x == alpha]` over the full domain through
  the unchanged `spfss_host::dpfEvalAll`, with `alpha = off_0 + off_1` and
  `beta = beta_0 * beta_1 mod p` recomputed from each party's own recorded
  private input;
- public material must match on both sides (`sCW`, `tLCW`, `tRCW`, `finalCW`)
  while seeds and control bits must differ - checked, 0 mismatches;
- corrupted-key negative control (`finalCW + 1`) fails as expected on every
  configuration, so the pass is not vacuous;
- primitive self-tests inside the protocol binary (`--selftest 16`) open
  triples and OLE shares in a labelled test-only mode: 0 triple failures and
  0 OLE failures on every configuration.

The `.testmeta` files exist only so an offline checker can recompute
`alpha`/`beta`; they are explicitly test-only and are never read by the
protocol.

## 5. Honest boundaries

- **IKNP is OT extension, not silent OT.** Ferret/Silver-class silent OT would
  cut the setup material; the byte counts above are therefore an upper bound for
  this protocol shape, not a silent-OT figure.
- **Semi-honest only**, authenticated point-to-point channels assumed. No
  malicious security, no active-attack handling, no side-channel scope.
- **Non-cryptographic expansion PRG** (splitmix64) inherited from the unchanged
  evaluator; the GPU path additionally still packs a control bit into the seed
  LSB. `D-SEED`, `P-RNG`, `P-DIST` and `P-KEY` stay open.
- **Loopback measurements.** No WAN/LAN profile, no bandwidth cap, no latency
  injection. Round counts are the transport-independent number to quote.
- This artifact generates DPF keys for the SPFSS/Ring-LPN pipeline; it is not
  yet wired into the GPU OLE transcript, and the FC preprocessing pipeline still
  runs in one process with dealer-labelled conversion correlations.

## 6. Where this sits in the plan

Milestone M1 required "real silent OT/OLE transport, GPU batching and bytes, and
round/traffic measurement". This artifact discharges the *real transport and
measured traffic* part for the keygen protocol on the host, with real OT rather
than silent OT. Still open for M1: silent-OT backend, GPU-side batched keygen,
byte-identical GPU key emission, and driving the real-OLE GPU transcript from
these keys (M2).
