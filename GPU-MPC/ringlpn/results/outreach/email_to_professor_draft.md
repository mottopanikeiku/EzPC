# Email draft: Figure 2 verification + host correctness artifact

**To:** [Professor]
**From:** Fatih
**Subject:** Ring-LPN OLE (Figure 2) — verification + host correctness artifact; three follow-ups needing direction

---

Hi [Professor],

Status update on the "Accelerating LPN-Based Cryptographic Schemes" draft.
You asked me to verify correctness of the algorithm on page 5 / Figure 2 (the
SPFSS-based Π_{OLE-Rc-LPN} from §6.4, BCG+20). The short version: Figure 2 was
not previously implemented in the tree — only the VOLE construction from §5.3
(`z = y + x·Δ`) existed, in `bench_vole_ringlpn`. I closed that gap with a
host-side correctness artifact this week.

## What I built and validated

Three new pieces, all host-only (no CUDA dependency), reproducible from
`GPU-MPC/ringlpn/scripts/build_ole_host.sh`:

1. **Plaintext oracle** (`verify_figure2_expand`). Implements the Figure 2
   algebraic identity `⟨a⊗a, u⟩ == x_0·x_1 mod (X^N+1)` directly, no sharing,
   no DPF. Sweep over seeds × N × c × t: **135/135 pass**. This establishes
   unambiguous ground truth independent of any SPFSS implementation.

2. **DPF + SPFSS with Z_p payload** (`spfss_host.{h,cpp}`, unit test
   `test_spfss`). Standard Boyle-Gilboa-Ishai GGM tree with one Z_p final-level
   correction word so that `share_0[x] + share_1[x] == Σ_k β_k·[x==α_k] mod p`.
   Sweep over log_domain ∈ {6,8,10,12,14} × m ∈ {1,4,16,64} × seed:
   **57/57 pass**.

3. **Full Figure 2 OLE Expand** (`bench_ole_ringlpn_host`). Uses the SPFSS to
   produce shares `u^{i,j}_σ` of `e^i_0·e^j_1`, folds 2N→N via X^N=-1,
   computes `x_σ = ⟨a, e_σ⟩` and `z_σ = ⟨a⊗a, u_σ⟩` on the sparse reps, and
   validates `(z_0+z_1)[k] == (x_0·x_1)[k]` for every k. Sweep over N ∈
   {32,64,128} × c ∈ {2,3} × t ∈ {4,8} × seed: **36/36 pass**.

Full results and reproduction: `GPU-MPC/ringlpn/results/ole_figure2_host_results.md`.

## One structural finding worth flagging

The existing `GPU-MPC/fss/gpu_dpf.cu` (Neha Jawalkar / MSR) has a
payload-of-1-bit constraint (noted in the file itself). Figure 2 needs Z_p
payloads, so I couldn't thin-wrap it. I wrote a standalone host DPF instead —
it doesn't touch Jawalkar's file. For the GPU fast path we'll either template
`gpu_dpf.cu` to accept a `PayloadT` or add a sibling `gpu_dpf_zp.cu`.

## Parameter choices I made (and where I'd want your input)

With the correctness artifact done, I picked conservative defaults to move. If
any of these are wrong for the paper we're pointing at, I'd want to course-correct
before running numbers:

1. **Modulus.** Single 62-bit NTT-friendly prime (same one the cheddar NTT path
   uses). Paper §7 uses a ~128-bit modulus via 2-prime CRT. Lift to CRT is a
   mechanical follow-up — I'd planned to do it after the core bench validates,
   not before.
2. **Noise distribution.** Uniform-position t-sparse (the generic Figure 2
   statement). §A.2 regular noise cuts DPF domain by `log(t)` and is what the
   §7 table numbers assume. Regular noise is the config I'd report benchmarks
   on; uniform was the straightforward first cut for correctness.
3. **Scope.** I stopped at "OLE works: `z_0+z_1 == x_0·x_1`". §8's "two OLEs →
   one Beaver triple" and the Z_p → Z_{2^bw} share conversion live in the Orca
   linear-layer plan (Phase B), where they belong.
4. **PRG.** splitmix64 for correctness. Swap for AES-NI or ChaCha20 before any
   timing claim. (Clearly documented in the source.)

## Follow-ups, in priority order

- **GPU polymul plug-in.** Replace the host schoolbook x_σ / z_σ with
  `run_polymul_prepared_lhs` (the kernel the VOLE bench already
  coefficient-validates against a host reference). Mechanical.
- **GPU DPF with Z_p payload.** Either template `gpu_dpf.cu` or add
  `gpu_dpf_zp.cu`. My recommendation is the template approach — localized
  change, doesn't alter existing payload=1 callers.
- **Regular-noise variant** and **2-prime CRT lift** for paper-comparable
  benchmark numbers.
- **OLE → Beaver triple** + Z_p → Z_{2^bw} conversion when we turn to Phase B
  of the Orca linear-layer integration.

Happy to walk through any of this in our next meeting. Please let me know if
you'd like me to change any of the four parameter choices above before I put
time into the GPU path.

Best,
Fatih
