# Dealerless Orca FC preprocessing — security contract (S1 frozen for advisor review)

**Date:** 2026-07-29
**Status:** frozen for advisor review after the user-requested Opus 5
model-assisted audit; not an independent human cryptographic review, security
proof, computational-security result, or publication-readiness claim
**Target:** integrated dealerless Orca linear/FC preprocessing
**Adversary:** one statically corrupted semi-honest party; authenticated point-to-point channels; external network observers, active attacks, denial of service, and side channels are out of scope
**Proof structure selected with the user:** an end-to-end FC functionality with the distributed DPF as a named subfunction and theorem
**Phase C decision:** freeze the corrected three-OLE transcript unless its simulator or composition proof exposes a concrete gap
**Author:** Alp (sole author, by user direction; inherited work remains cited
and its ownership/reuse boundary remains subject to S2)

This document freezes what the implementation must realize and what the paper
must prove. It is deliberately stronger than the current evidence. The
2026-07-21 host artifact proves functional compatibility and primitive counts
only; it uses ideal OT/triple/OLE calls, splitmix64, and one process.

## 1. Contribution and provenance boundary

The candidate paper contribution is the **integrated two-party preprocessing
path for Orca FC layers**. The corrected per-point distributed DPF below is a
named subfunction and compatibility artifact, not presently a protocol
contribution: BCG+20 already invokes Doerner--shelat distributed DPF setup,
Programmable DPFs give constant-round generation, Agarwal--Raghuraman--
Rindal's 2026 fully distributed DMPF directly targets Ring-LPN PCGs with a
proof and prototype, and 2026 SLAMP-FSS is another multi-point construction.
Advisor review must select a multi-point route, retain this artifact only as a
baseline, or identify a concrete delta. The relationship to the separate
private GPU-PCG/PIM work is
also unsettled; it has multiple contributors and no repository license. Do
not assume common ownership or permission, and do not import or claim any of
its GPU-PCG design or performance. The paper's author list is fixed to Alp
alone. This does not reassign ownership or erase attribution for inherited
code, protocols, or measurements.

GPU-NTT is separate upstream work by Ali Şah Özcan, Erkay Savaş, and
collaborators. The local upstream repository identifies ePrint 2023/1410 and the
2025 IEEE Access article (DOI `10.1109/ACCESS.2025.3570024`) as its citations.
This paper must cite that work whenever it discusses or uses the GPU-NTT merge
or four-step algorithms. The current EzPC deployment backend is documented as
cheddar-derived; GPU-NTT is presently an external measured baseline. Any claim
that the active backend incorporates GPU-NTT code or algorithmic ideas requires
a separate source/license audit before the wording changes.
The active Cheddar-derived backend is a substantial adaptation of MIT-licensed
upstream code. This audit added the upstream copyright/license notice and
paper citation and recorded the reconstructed source pin and local delta in
`extern/Cheddar_PROVENANCE.txt`.

**Questions to resolve with the professor in S2 before S3 implementation or
external circulation:**

1. Is the contribution the integrated dealerless Orca FC system, or is a new
   distributed-DPF protocol required?
2. Should S3 adopt/benchmark the 2026 fully distributed DMPF and SLAMP-FSS,
   retain this per-point DPF only as a baseline, or pursue a specifically
   identified delta?
3. Should the architecture remain regular Ring-LPN/NTT plus conversion, adopt
   Stationary Syndrome Decoding, or compare/pivot to the 2025 direct-
   `Z_(2^k)` and 2026 QA-SD/WHT PCGs?
4. Who will approve the sparse-factor projection criterion, projected-noise
   mapping, advantage loss, and classical/quantum 128-bit interpretation?
5. Should the parameter target use `n=2^20,c=4,t=16`, the preliminary
   `n=2^14,c=4,t=16`, or another reviewed point?
6. Which contributors own each DPF, CPU/GPU PCG, GPU-NTT, PIM, integration,
   measurement, figure, and prose component in the private project; what may
   Alp reuse; and what citation, acknowledgement, or disclosure is required?
7. What is that work's chronology and submission/public-release status
   relative to this sole-author project?
8. May the now-attributed Cheddar-derived backend remain, or should
   publication use a clean external backend boundary?

## 2. Notation and fixed conventions

- Parties are `P0` and `P1`; the corrupted party index is `b` and the other
  party is `1-b`.
- `sid` is a unique public session identifier. It is included in every
  randomness/correlation domain separator.
- `p` is an odd prime and `Z_p` its field. `ell_p = ceil(log2 p)`.
- A DPF domain has size `D=2^L`.
- `off_b` is party `b`'s private point summand, with
  `0 <= off_b < 2^(L-1)`. Thus `alpha=off_0+off_1` is an integer in
  `[0,2^L-2]`; there is no modular wrap in the position addition. For uniform
  Ring-LPN noise, these are the two parties' independent positions in
  `[0,n)` with `2^(L-1)=n`, so `alpha` has the intended triangular convolution
  distribution over the unreduced degree-`<2n` product. For regular noise they
  are independent offsets inside a public bucket pair and have the analogous
  triangular distribution over `[0,2n/t-2]`.
- `beta_b in Z_p^*` is party `b`'s private nonzero payload factor and
  `beta=beta_0 beta_1 mod p`. Zero and noncanonical field encodings abort.
- `xor` denotes bit/string XOR. Field addition and subtraction are modulo `p`.
- A standard DPF key is
  `K_b=(L,p,seed_b,t0_b,{sCW_i,tLCW_i,tRCW_i}_{i=0}^{L-1},finalCW)`.
  The root seed is private to one key; correction words and `finalCW` occur in
  both keys. “Common key material” means known to both parties, not necessarily
  visible to an external network observer.
- An Orca matmul key is serialized in the existing `A_b || B_b || C_b` byte
  order consumed by unchanged `gpuMatmulBeaver`.

## 3. Ideal functionalities

### 3.1 Distributed DPF key generation `F_DDPF`

For one tree, on public input `(sid,tree_id,L,p)` and private inputs
`(off_b,beta_b,root_b)` from each `P_b`, where each party samples its 128-bit
`root_b` uniformly from its local random tape:

1. Validate the public parameters, session/tree-ID uniqueness, point-share
   ranges, nonzero payload factors, canonical field encodings, and root length.
   On failure, send the same `abort` to both parties before consuming
   correlation.
2. Set `alpha=off_0+off_1` and `beta=beta_0 beta_1 mod p`.
3. Run the **standard DPF key-generation algorithm conditioned on the supplied
   root seeds** for `f(x)=beta [x=alpha]` over `[0,2^L)`. This is deterministic
   once `(alpha,beta,root_0,root_1)` and the fixed PRG are fixed.
4. Deliver only `K_b` to `P_b`.

For a batch, accept arbitrarily correlated private input vectors, but require
distinct tree IDs and independent party root draws and primitive correlation
per tree. Batching may change only the public schedule/framing.

The functionality leaks to `P_b` only:

- public `(sid,L,p,batch size,tree IDs)` and validation/abort status;
- its own `(off_b,beta_b,root_b)`;
- its output key `K_b`, whose root field equals `root_b`.

The functionality does not reveal `alpha`, `beta`, the other input shares, the
other root seed, the other key, a hidden leaf-control sign, or intermediate
adder/tree/OLE state.

**Distribution obligation D-DIST.** Full-domain correctness is insufficient.
The real protocol's correction words must equal those of standard
`DPF.Gen(alpha,beta;root_0,root_1)` level by level, or its joint output must be
computationally indistinguishable from that distribution. If only a new
distribution can be proved, the functionality and paper must name it and
separately prove single-key privacy.

**Seed-format obligation D-SEED.** The target follows the formal seed/tag
separation of Boyle--Gilboa--Ishai, *Function Secret Sharing: Improvements and
Extensions*, CCS 2016, DOI `10.1145/2976749.2978429`: a full
`lambda`-bit secret seed plus separate child control bits. The current GPU PRG
instead uses a Doerner--shelat-style low-bit control encoding (*Scaling ORAM
for Secure Computation*, CCS 2017): `gpu_spfss_zp.cuh` masks each AES-output
LSB from the child seed, leaving `lambda=127` secret seed bits.
More severely, the centralized GPU keygen deterministically expands every root
from one 64-bit `seed_base`, so its root entropy is at most 64 bits. S3 must
replace that benchmark root generation with independent OS-CSPRNG roots and
either widen the PRG output/state to 128 secret seed bits plus separate tags or
lower the concrete-security target. No 128-bit DPF-security claim is permitted
while either mismatch remains.

**Position-distribution obligation D-POS.** `D-DIST` is conditional on the
actual `(alpha,beta)`: it concerns the DPF key distribution, not whether
`alpha` itself is uniform. The induced point distribution is fixed by the
Ring-LPN noise sampler. Two independent uniform positions in `[0,n)` sum to a
triangular distribution on `[0,2n-2]`; the endpoint `2n-1` is unreachable.
That is the exact unreduced polynomial-product exponent used by
`bench_ole_ringlpn_cuda.cu`, not an accidental point sampler. Bucket-regular
noise similarly produces a public bucket-diagonal plus the sum of two uniform
bucket offsets. S2 must audit the original noise distribution and this derived
mapping explicitly. Replacing the sum by addition modulo `2^L` would change the
polynomial product unless the carry also negates/folds the payload; it is a
different protocol, not a harmless sampler fix.

### 3.2 Stateful FC preprocessing `F_FC`

`F_FC` maintains a private additive-share store indexed by public wire/mask
handles. A handle names one tensor over `Z_(2^bw)` and its two shares; reusing a
handle means reusing that mask intentionally, not sampling a new value. The
public layer-state table names `(h_X,h_W,h_Ybias,h_VW,h_VY,h_G)`, the
truncation/optimizer parameters, and `useMomentum`. `h_Ybias` is Orca's
source-level `mask_Y`; `h_VW,h_VY` are `mask_Vw,mask_Vy`. Current `FCLayer`
asserts `useBias=true`.

For a matrix-product invocation
`mu=(sid,invocation,kind,batchSz,M,K,N,bw,layout,parameter_set,h_A,h_B,h_R)`:

1. Validate `mu`, handle shapes, the unique correlation domain, canonical
   encodings, expected field/byte counts, and the public admissibility
   predicate `2 < bw <= 32`, `K >= 1`,
   `K * 2^(2*bw+2) < Q(parameter_set)`. A malformed, duplicate, or inadmissible
   invocation returns common `(abort,stage)` before reading masks or consuming
   correlation.
   This bound is derived from the actual additive-share representation, not
   the clear mask: canonical shares satisfy
   `a_0+a_1 < 2^(bw+1)` and `b_0+b_1 < 2^(bw+1)`, hence one length-`K`
   integer dot product is strictly below `K*2^(2*bw+2)`. Reducing that integer
   modulo `2^bw` gives the intended mask product, while the strict bound
   prevents an intervening reduction modulo `Q`.
2. Read operand masks `A` and `B` from `h_A,h_B`. Create a fresh uniform
   product-output mask `R_C` and additive shares under new handle `h_R`.
   (`R_C` is source-level `mask_Z`.) Compute
   `C=A*B+R_C mod 2^bw`; sample `C_0` uniformly and set `C_1=C-C_0`.
3. Deliver the fields Orca actually stores:
   - forward: `A_b || B_b || C_b`;
   - weight gradient `dW`: `B_b || C_b`, reusing forward's `A_b`;
   - input gradient `dX`: `C_b`, reusing `dW`'s gradient-mask `B_b` as `A_b`
     and forward's weight-mask `B_b`.
   Store `Z_b` for the enclosing state transition; it is not common leakage.

One complete training-layer transition has the following source-aligned
topology.

1. Forward invokes the product on `(h_X,h_W)` to obtain `h_Rf`, then locally
   broadcasts and adds the persistent bias mask:
   `h_pretrunc = h_Rf + broadcast(h_Ybias)`. `F_TRUNC` consumes that combined
   handle and returns the next activation handle.
2. Backward invokes `dW` on `(h_X,h_G)` and `dX`, when enabled, on `(h_G,h_W)`.
   The `dX` product-output handle passes through backward truncation.
3. The bias-gradient mask is the public-shape row sum
   `h_dY = row_sum(h_G)`, matching `getBiasGrad`; it is a local linear handle
   transform, not fresh correlation.
4. The named optimizer boundary `F_OPT` updates both persistent state pairs:
   `(h_W,h_VW,h_dW) -> (h_W',h_VW')` and
   `(h_Ybias,h_VY,h_dY) -> (h_Ybias',h_VY')`, with public epoch, momentum,
   learning-rate, scale, and truncation parameters. The target must match
   Orca's `genOptimizerKey` transition and serialized fields exactly.

Current Orca initializes `mask_W`, `mask_Y`, `mask_Vw`, and `mask_Vy` to zero.
`h_X` comes from the preceding layer's truncated output, and `h_G` comes from
the backward chain. These operand/state masks are neither resampled nor assumed
independent per invocation. S7 must implement the complete forward/bias/
truncation/`dW`/`dX`/bias-gradient/dual-optimizer handle transition; S8 must
compose `F_TRUNC` and `F_OPT` with this product contract.

There are no private model/data inputs to preprocessing; private mask shares
are functionality state. Every matmul still consumes fresh, independently
domain-separated OT/DPF/OLE/conversion correlation even when operand-mask
handles are intentionally reused.

### 3.3 Hybrid functionalities used by the proof

Every call below includes `(sid,invocation,tree-or-slot,phase,ordinal)` as a
public correlation identifier. A functionality records consumed identifiers
and returns common `abort` on reuse before releasing output.

- `F_BT`: sample uniform bits `a,b`, set `c=a AND b`, sample uniform XOR shares
  `a_0,b_0,c_0`, set the complementary shares, and deliver
  `(a_i,b_i,c_i)` only to `P_i`. Phase A then performs the two concrete masked
  openings; the functionality does not open them.
- `F_OT^128`: receive sender messages `(m_0,m_1)` and the other party's choice
  bit `q`; deliver only `m_q` to the receiver and no output to the sender.
- `F_OLE^p`: receive `x_0` from `P0` and `x_1` from `P1`; sample `g_0`
  uniformly in `Z_p`, set `g_1=x_0 x_1-g_0`, and deliver only `g_i` to `P_i`.

The end-to-end proof additionally names:

- `F_COIN`: on a fresh identifier, sample and reveal one uniform public seed;
  its real realization uses commit/open coin tossing and common abort.
- `F_RINGOLE`: for public `R=Z_p[X]/(X^n+1)`, sample uniform
  `X_0,X_1,Z_0 in R`, set `Z_1=X_0 X_1-Z_0`, and deliver only `(X_b,Z_b)` to
  `P_b`. One call is made per direction, CRT limb, and slot batch.
- `F_EDABIT^ell`: sample uniform `R in Z_(2^ell)`, sample uniform arithmetic
  share `R_0` and set `R_1=R-R_0 mod 2^ell`; for every bit of `R`, sample one
  uniform XOR share and deliver consistent arithmetic and Boolean shares only
  to the corresponding party.
- `F_DABIT^bw`: sample uniform `d in {0,1}` and an independent uniform Boolean
  mask `d_0^B`, then set `d_1^B=d xor d_0^B`. Independently sample uniform
  `d_0^A in Z_(2^bw)`, set `d_1^A=d-d_0^A mod 2^bw`, and deliver only
  `(d_b^B,d_b^A)` to `P_b`.
- `F_CONV`: on canonical private `z_0,z_1 in [0,Q)`, where public `Q` is one
  prime or the product of the two CRT primes, set
  `S=z_0+z_1`, `wrap=[S>=Q]`, and `v=S-wrap*Q`. Sample uniform
  `r_0 in Z_(2^bw)`, set `r_1=v-r_0 mod 2^bw`, and deliver only `r_b` to
  `P_b`. The wrap bit is internal and is never opened.
- authenticated point-to-point delivery with explicit common abort.

S4, S6, and S8 replace these hybrids with selected real protocols and compose
their security theorems. The present ideal interfaces are contract boundaries,
not evidence that those replacements exist.

## 4. Exact distributed DPF protocol transcript

The transcript below matches `src/test_distributed_dpf_keygen.cpp`. Any
production optimization must be shown equivalent or update this contract before
implementation.

### 4.1 Phase A — shared position bits

Write `u_j=bit_j(off_0)` and `v_j=bit_j(off_1)`. Carries are XOR-shared as
`c_j=c_{j,0} xor c_{j,1}`, initially `(c_{0,0},c_{0,1})=(0,0)`.

At bit `j`:

1. Output shares of the sum bit are
   `a_{j,0}=u_j xor c_{j,0}` and `a_{j,1}=v_j xor c_{j,1}`.
2. If `j<L-1`, form XOR shares
   `x=(u_j xor c_{j,0}) xor c_{j,1}` and
   `y=c_{j,0} xor (v_j xor c_{j,1})`.
3. Obtain one fresh triple from `F_BT`. Each party reveals its share of
   `delta=x xor triple.a` and `epsilon=y xor triple.b`, producing two common
   one-bit values.
4. Compute XOR shares of `x AND y` with the Beaver equation and update the
   carry shares exactly as in the source.

At completion, `a_{j,0} xor a_{j,1}=bit_j(alpha)` for all `j`.

**Accounting:** `L-1` bit triples and `2(L-1)` logical opened bits per tree.
Each logical `delta` or `epsilon` opening sends one share from each party, so
the raw revealed-share payload is `4(L-1)` bits. The carry dependency is
sequential; batching trees does not remove it.

### 4.2 Root keys

Each party samples an independent 128-bit root seed `seed_b` from its private
CSPRNG. Set `t0_0=0`, `t0_1=1`. The host prototype uses splitmix64 here only as
a labelled correctness substitute; the production path must use the S3 AES/
CSPRNG semantics.

### 4.3 Phase B — level-synchronous correction words

At level `i`, let `j=L-1-i` be the MSB-first point-bit index. Party `b` locally
expands every node in its current frontier and computes XOR aggregates
`S_b^L,S_b^R` and control-bit aggregates `T_b^L,T_b^R`. Define
`Z_b=S_b^L xor S_b^R` and let `a_b=a_{j,b}`.

The two directional OTs are:

1. `P1` samples uniform `r_1`, sends OT messages `(r_1,r_1 xor Z_1)`, and `P0`
   receives `q_0=r_1 xor a_0 Z_1` using choice `a_0`.
2. `P0` samples uniform `r_0`, sends OT messages `(r_0,r_0 xor Z_0)`, and `P1`
   receives `q_1=r_0 xor a_1 Z_0` using choice `a_1`.

The seed-correction-word shares are

```text
sCW_0 = S_0^R xor a_0 Z_0 xor q_0 xor r_0
sCW_1 = S_1^R xor a_1 Z_1 xor q_1 xor r_1.
```

Each party reveals its 128-bit share; both set
`sCW=sCW_0 xor sCW_1`.

The control-correction-word shares are linear:

```text
tLCW_0 = T_0^L xor a_0 xor 1    tLCW_1 = T_1^L xor a_1
tRCW_0 = T_0^R xor a_0          tRCW_1 = T_1^R xor a_1.
```

Each party reveals both one-bit shares; both obtain `tLCW` and `tRCW`. Each
party advances its own frontier using only its local expanded nodes and the
common correction words.

**Accounting per level:** two 128-bit string OTs; one logical 128-bit `sCW`
opening; and two logical one-bit flag openings. This is `130L` logical opened
bits across the tree. The two parties reveal a share of every value, giving
`260L` raw revealed-share bits. These are dependency stages, not a claim of
network rounds; S4 must measure the real transport.

### 4.4 Phase C — multiplicative payload correction

After level `L`, each party computes signed aggregates

```text
A_0 =  sum_x Convert(seed_{0,x})     F_0 =  sum_x t_{0,x}
A_1 = -sum_x Convert(seed_{1,x})     F_1 = -sum_x t_{1,x}          (mod p).
```

The cancellation invariant gives
`A_0+A_1=Convert(seed_{0,alpha})-Convert(seed_{1,alpha})` and
`F_0+F_1 in {+1,-1}`.

1. Invoke `F_OLE^p(beta_0,beta_1)` to obtain
   `gamma_0+gamma_1=beta_0 beta_1=beta`.
2. Set `d_b=gamma_b-A_b`, `s_b=F_b`.
3. Invoke `F_OLE^p(d_0,s_1)` to obtain shares `x_0,x_1` of `d_0 s_1`.
4. Invoke `F_OLE^p(s_0,d_1)` to obtain shares `y_0,y_1` of `s_0 d_1`.
5. Compute
   `w_b=d_b s_b+x_b+y_b`.
6. Each party reveals `w_b`; both set
   `finalCW=w_0+w_1=(d_0+d_1)(s_0+s_1)`.

Only `w_0,w_1` are revealed in Phase C. `d_0,d_1,s_0,s_1` and the sign
`s_0+s_1` are never opened. `finalCW` is already common material in both
standard output keys.

**Accounting:** three scalar OLEs and one logical opened field element,
`ell_p` bits. The two parties each reveal one field share, so raw
revealed-share payload is `2 ell_p` bits. For the current 62-bit primes these
are respectively 62 and 124 bits. The implementation's fixed-width encoding
must use `ceil(log2 p)` after any parameter re-pin.

### 4.5 Complete per-tree accounting

```text
string OTs                 = 2L
bit triples                = L-1
scalar OLEs                = 3
logical opened bits        = 2(L-1) + 130L + ceil(log2 p)
raw revealed-share bits    = 4(L-1) + 260L + 2 ceil(log2 p).
```

At the current `L=14`, 62-bit primes, these are 1,908 logical opened bits and
3,816 raw revealed-share bits. The previously published 3,790 mixed Phase A's
logical opening count with Phases B/C's transmitted shares and is therefore
not a coherent metric. Neither corrected counter is real network traffic:
OT/OLE setup and payloads, framing, commitments, and retransmission remain
absent from the host prototype. S4 replaces estimates with measured bytes.

## 5. D2–D4 integrated FC transcript to be composed

For each accepted matmul invocation `mu`, the target two-process transcript is:

1. Both parties validate the same public dimensions, layout, parameter set,
   identifiers, mask handles, slot capacity, output lengths, and membership in
   the public admissible set
   `2 < bw <= 32`, `K >= 1`, `K * 2^(2*bw+2) < Q(qbits)`.
   For the q62 limb this permits at most `K=2^28-1` at `bw=16`,
   `K=2^12-1` at `bw=24`, and no `K>=1` at `bw=32`; q128 permits
   `K<2^90`, `K<2^74`, and `K<2^58`, respectively. A failure produces common
   `(abort,stage)` before reading masks or consuming correlation.
2. `F_COIN` establishes domain-separated public seeds. Each party obtains its
   private root randomness from an independent OS CSPRNG stream; public seeds
   never seed private masks or DPF roots.
3. Party `P_b` reads only its local shares of the operand handles fixed by
   `F_FC` and samples its share of the fresh output mask. Mask-handle reuse is
   dictated by the forward/`dW`/`dX` topology; correlation IDs remain fresh.
4. For each `(direction,CRT limb,slot batch)`, D1 generates the distributed DPF
   keys required by the Figure-2 Ring-LPN expansion. The resulting ideal
   boundary is one fresh `F_RINGOLE` output `(X_b,Z_b)` per party, with
   `Z_0+Z_1=X_0 X_1` in the public ring. No key, noise share, or PCG seed
   crosses the party boundary.
5. Each party applies the public forward negacyclic NTT to its own
   `(X_b,Z_b)`. At the fully split primes this is a local ring isomorphism, so
   every used slot satisfies `Z_0[s]+Z_1[s]=X_0[s]X_1[s] mod p`.
6. Each matrix cross term consumes one unique slot in each limb. In direction
   0, `P0`'s operand is an `A_0` entry and `P1`'s is the corresponding `B_1`
   entry. In direction 1, `P0` uses `B_0` and `P1` uses `A_1`. For either
   direction, `P0` sends
   `d=a-X_0[s] mod p` and `P1` sends `e=b-X_1[s] mod p`; both then know
   `(d,e)` and compute

   ```text
   u_0 = d e + e X_0[s] + Z_0[s]
   u_1 = d X_1[s] + Z_1[s].
   ```

   Thus `u_0+u_1=ab mod p`. Slot indices and message order are public and fixed
   by `(batch,row,column,k,direction,limb)`.
7. For every output coordinate and limb, each party sums its local product
   `A_b B_b` and its two cross-term shares. For two limbs it locally Garner
   lifts its residue vector to one canonical share `z_b in [0,Q)`, where
   `Q=p_0 p_1`; for one limb `Q=p_0`.
8. D2 invokes `F_CONV(Q,z_0,z_1)`. Internally
   `S=z_0+z_1`, `wrap=[S>=Q]`, and
   `v=S-wrap*Q=(z_0+z_1) mod Q`; its outputs satisfy
   `r_0+r_1=v mod 2^bw`. Neither party learns `wrap`. The public no-wrap
   bound makes the reconstructed residue the intended integer mask product
   before reduction to `Z_(2^bw)`; without it, reduction modulo `Q` could
   change the Orca ring result.
9. Party `P_b` sets `C_b=r_b+R_{C,b} mod 2^bw`, emits only the fields prescribed
   by the invocation kind in §3.2, and checks its local byte count.
10. Unchanged `gpuMatmulBeaver` consumes the resolved `A_b,B_b,C_b` handles.
    It is an independent correctness consumer, not part of the preprocessing
    proof.
11. For forward, the state layer broadcasts and adds persistent `mask_Y` to
    the fresh `R_C` (source `mask_Z`) before `genGPUTruncateKey`; truncation
    result becomes the next activation-mask handle.
12. For backward, `getBiasGrad` row-sums the incoming gradient-mask handle.
    `genOptimizerKey` then consumes both the `(mask_W,mask_Vw,dW)` and
    `(mask_Y,mask_Vy,dY)` state triples and emits their updated persistent
    handles. These local/adjacent operations use no Ring-LPN slot, but their
    handle identities and serialized key fields are part of `P-TOPO`.

The selected D2 prototype realizes step 8 in the
`(F_EDABIT,F_BT,F_DABIT)` hybrid. For
`ell=ceil(log2(2Q))`, it opens the `ell`-bit masked value
`A=(z_0+z_1+R) mod 2^ell`, evaluates two ripple adders using
`2ell-2` fresh bit triples, opens one masked bit for B2A, and applies the local
`Q*wrap` correction. This exposes `5ell-3` logical opened bits and
`10ell-6` raw revealed-share bits per conversion; the wrap bit, edaBit, daBit,
and input shares remain hidden. These are hybrid-protocol counts, not measured
transport bytes or rounds.

Current code has four target-breaking boundaries: centralized
`build_spfss_keys()`, conversion through `exactZmToRingShares()`, the clear
value-dependent `buildCShare()` bound check, and shared deterministic benchmark
seeding/direct memory in one process. S5, S6, and S7 replace them.

### 5.1 Source-to-transcript map for the current prototypes

| Current source boundary | Cross-party value or read | Contract line | Status |
|---|---|---|---|
| `Functionalities::bit_triple` + `shared_and` | fresh triple shares; transmitted shares of `delta_j,epsilon_j` | Phase A | mapped ideal interface plus concrete openings |
| `Functionalities::ot` | one selected 128-bit string per direction and level | Phase B OT products `a_i Z_(1-b)` | mapped ideal interface |
| `open_phase_b_seed_cw` | two 128-bit shares reconstruct `sCW_i` | Phase B seed-CW opening | mapped |
| `open_phase_b_flag_cw` | two shares reconstruct each `tLCW_i,tRCW_i` | Phase B flag-CW openings | mapped |
| three `Functionalities::ole` calls | shares of `beta_0 beta_1`, `d_0 s_1`, `s_0 d_1` | Phase C steps 1, 3, 4 | mapped ideal interfaces |
| `open_phase_c_final_cw` | `w_0,w_1` reconstruct `finalCW` | sole Phase C opening | mapped |
| `consumed_correlation_ids` | rejects reuse before an ideal primitive releases output | all D1 hybrid calls | mapped executable control |
| D1 validation helpers | read both generated keys and clear `(alpha,beta)` | test gate only | validation oracle; excluded from protocol |
| `make_and_triple()` in `test_secure_convert.cpp` | dealer-generated Boolean triple shares | D2 `F_BT` calls | current ideal/dealer boundary; S6 replaces |
| `make_edabit()` / `make_dabit()` | dealer-generated correlated arithmetic/Boolean shares | `F_EDABIT` / `F_DABIT` | current ideal/dealer boundaries; S6 replaces |
| masked `A=(y0+y1) mod 2^ell` reconstruction | two revealed arithmetic shares | D2 step 8 masked opening | mapped one-process opening |
| `secure_and()` | two shares each of Beaver `d,e` per AND | D2 ripple adders | mapped one-process openings |
| B2A `e=wrap.clear() xor da.b.clear()` | two revealed shares of one masked bit | D2 B2A opening | mapped one-process opening |
| `ConvOut.wrap` | returns clear wrap bit | D2 validation only | test oracle; excluded from protocol |
| `build_spfss_keys()` | reads both parties' Ring-LPN noise vectors | D1 inside `F_RINGOLE` realization | **centralized oracle; forbidden target behavior** |
| `build_slot_ole` container | stores both parties' ring-OLE shares | step 4 | one-process container; target state is party-local |
| slot/cross-identity checks | read both parties' slot or product shares | steps 5–7 | validation oracles; excluded from protocol |
| `cross_share` | `P0` sends `d`; `P1` sends `e` per direction, limb, and used slot | step 6 | mapped one-process transcript model |
| `exactZmToRingShares()` | reads both canonical `Z_Q` shares and the wrap predicate | step 8 | **dealer/oracle boundary; forbidden target behavior** |
| `buildCShare(): dot >= Q` | reads clear mask operands and aborts on their product | public bound in step 1 | **value-dependent abort/oracle; forbidden target behavior** |
| mask/key/online validation | reads both party arrays, clear masks, or outputs | steps 3, 9, 10 | validation oracle; excluded from protocol |
| shared benchmark seed/process | common `mt19937_64` setup and direct memory access | `F_COIN`, private streams, transport | **one-process oracle boundary; forbidden target behavior** |
| `gpuAddBias(mask_Z,mask_Y)` then `genGPUTruncateKey` | adds persistent bias-mask handle to fresh forward product-output handle; truncation returns next activation mask | §3.2 forward state transition | adjacent state boundary; S7 implementation and S8 composition required |
| `getBiasGrad(mask_grad)` | row-sums incoming gradient-mask handle into bias-gradient-mask handle | §3.2 backward state transition | local linear handle transform; no new correlation |
| weight/bias `genOptimizerKey` calls | read and update `(mask_W,mask_Vw,dW)` and `(mask_Y,mask_Vy,dY)` | §3.2 dual optimizer transition | adjacent state boundary; S7 implementation and S8 composition required |

Every current cross-party read in D1 and the integrated FC artifact is either a
mapped protocol message, a named target-breaking oracle, or an explicitly
test-only validation oracle above. New reads/messages require a contract update
before implementation.

## 6. Leakage contract

### 6.1 Permitted common leakage

- protocol/version and session identifiers;
- party roles and selected parameter-set identifier;
- `L,p,n,c,t`, CRT limb count, noise mode, batch/tree/epoch counts;
- public layer shape/layout, invocation kind, public wire/mask-handle topology,
  slot/access order, `bw`, and the fields serialized by each invocation;
- fixed message lengths, dependency schedule, and total bytes/rounds;
- common DPF correction words and `finalCW` contained in both output keys;
- masked Beaver/derandomization openings `delta,epsilon,d,e` and the masked
  conversion openings specified in §5;
- public coin-toss commitments/reveals and derived public seeds;
- accept/abort and the stage at which an abort occurs.

### 6.2 Permitted per-party view

In addition to common leakage, `P_b` sees only:

- its private DPF inputs/root draws and its shares in the persistent mask store,
  including intentionally reused forward/`dW`/`dX` operand shares;
- its OT sender inputs, receiver choices/outputs, bit-triple/OLE shares, local
  frontiers, noise shares, fresh output-mask shares, and correlation IDs;
- its output DPF keys, emitted Orca key fields, and next-state mask shares;
- authenticated messages it sends or receives.

### 6.3 Prohibited leakage

The proof and implementation must not reveal:

- `off_(1-b)`, `beta_(1-b)`, `alpha`, or `beta` beyond what is implied by
  `P_b`'s own input/output under DPF single-key privacy;
- the other root seed, DPF key, noise polynomial, mask-store shares, or Orca
  key fields;
- the hidden leaf-control sign or either `s_b`/`d_b` value;
- the conversion wrap bit, edaBit/daBit contents, or unmasked conversion input;
- OT receiver choices, unused sender messages, base-OT secrets, or OLE masks;
- reused primitive correlation or reused private random-tape draws across tree,
  limb, direction, layer, invocation, epoch, or session. This does not prohibit
  the explicit public mask-handle reuse required by Orca's topology;
- private values in logs, packet traces, failure messages, or filenames.

Traffic metadata is in scope as permitted leakage; external packet observers
and microarchitectural side channels are excluded from the S1 theorem and must
be stated as limitations.

## 7. Hybrid-model simulators

### 7.1 D1 batch simulation lemma

Fix a corruption index `b`, the complete public batch schedule, the corrupt
party's arbitrarily correlated input vector `{(off_b^k,beta_b^k)}_k`, the
independently uniform roots required by `F_DDPF`, their realized values
`{root_b^k}_k`, and the ideal output-key vector `{K_b^k}_k`. Require each
`K_b^k.root=root_b^k`. In the `(F_BT,F_OT^128,F_OLE^p)` hybrid, the algorithms
below generate the corrupt party's complete batch view with the same
distribution as D1, conditioned on those inputs, realized roots, and outputs.
`P-DIST` and `P-KEY` remain separate obligations relating the conditioned output
to standard uniform-root DPF keygen and its privacy theorem.

S1 models each party's random tape as the independent draws consumed by the
protocol; it does not expose or promise consistency with a retained master
derivation key. The root draw is made explicit as an ideal input because it is
also a checkable field of `K_b`. S3 must either preserve this independent-draw
interface or define and prove a state-consistent domain-separated CSPRNG
realization before the theorem is instantiated with concrete AES streams.

### 7.2 Simulator for corrupted `P0`

For each tree in the public batch order:

1. Take `root_0` from `K_0` (never resample it), set root tag 0, read the common
   correction words from `K_0`, and expand the local frontier exactly.
2. At carry `j`, sample `P0`'s three `F_BT` shares uniformly. Compute its sent
   shares of `delta_j=x_j xor a_j` and `epsilon_j=y_j xor b_j` from its actual
   local carry/input shares. Sample the two reconstructed masked bits uniformly
   and set the honest sent shares to the XOR complements of `P0`'s sent shares.
   Apply the real Beaver equation (including the public
   `delta_j AND epsilon_j` term on `P0`) and the source recurrence. The hidden
   honest triple shares one-time-pad both reconstructed openings; induction
   from carry `(0,0)` gives the exact joint carry/point-share view.
3. At each Phase-B level, derive `P0`'s aggregates from that frontier. In the
   first OT, sample its selected receiver output
   `q_0=r_1 xor a_0 Z_1`, which is uniform under fresh hidden `r_1`. In the
   second OT, sample fresh local `r_0` and record the sender pair
   `(r_0,r_0 xor Z_0)`; ideal OT reveals no honest choice. Compute `P0`'s
   seed-CW opening share with the real equation and set the honest share to its
   XOR complement of the `sCW_i` fixed by `K_0`. Compute the corrupt flag-CW
   shares deterministically and complement them to the fixed
   `tLCW_i,tRCW_i`. This preserves the full per-level joint view, not only the
   common words.
4. Sample the three `P0` `F_OLE^p` outputs
   `gamma_0,x_0,y_0` independently and uniformly. Derive
   `d_0=gamma_0-A_0`, `s_0=F_0`, and
   `w_0=d_0 s_0+x_0+y_0`; emit only the honest opening share
   `w_1=finalCW-w_0`. This remains valid for zero intermediate values because
   the three OLE masks are sampled before the deterministic equations.
5. Record the correlation ID for every ideal call and reject any duplicate
   before sampling. Output the local random-tape draws, ideal-primitive views,
   sent/received shares, common values, frontier state, and `K_0`.

### 7.3 Simulator for corrupted `P1`

For each tree in the same public order:

1. Take `root_1` from `K_1`, set root tag 1, and expand from the common
   correction words in `K_1`.
2. Sample `P1`'s `F_BT` shares, compute its actual outgoing masked-opening
   shares, sample uniform reconstructed `delta_j,epsilon_j`, derive the honest
   complements, and apply `P1`'s Beaver equation without the public
   `delta_j AND epsilon_j` term. The same carry induction yields the exact
   point-share/local-state view.
3. In Phase B, sample fresh `r_1` and record `P1`'s first-OT sender pair
   `(r_1,r_1 xor Z_1)`. Sample its selected second-OT receiver output
   `q_1=r_0 xor a_1 Z_0`, uniform under hidden fresh `r_0`. Compute its own
   seed/flag opening shares and set the honest shares to complements of the
   correction words fixed by `K_1`.
4. Sample `gamma_1,x_1,y_1` independently and uniformly from the `P1`
   marginals of the three OLEs. Compute `d_1,s_1,w_1` with the real equations
   and emit only `w_0=finalCW-w_1`. This reveals neither `s_0` nor the sign.
5. Enforce the same correlation-ID ledger and output the complete `P1` view.

The proof is joint across the batch: private inputs may repeat or correlate,
but each replacement conditions on the complete prior simulated state and
changes only fresh masks under a new ID. Sequential composition therefore
preserves arbitrary input correlation and the public batching order. It does
not assume that the hidden sign is marginally random.

### 7.4 End-to-end FC simulator outline

For a corrupt `P_b`, condition on its full persistent mask store, DPF root/input
vector, public topology, and ideal `F_FC` outputs:

1. Simulate the real commit/open transcript that yields each public seed.
2. Execute the corrupt party's random tape and PRG computations consistently;
   apply PRF replacements only to the honest party's hidden streams.
3. Apply the joint D1 simulator above, then the future `P-DIST/P-KEY` result, to
   replace distributed keygen with `F_DDPF`.
4. Replace the Ring-LPN expansion by `F_RINGOLE` under the exact S2 assumption.
5. For each fresh slot, compute the corrupt party's own `d` or `e` from its
   operand and ring-OLE share; sample the honest opening uniformly from the
   fresh hidden opposing share. Preserve repeated operand-mask handles but
   never reuse the OLE slot/correlation.
6. Invoke the future D2 simulator for the exact masked openings in §5 and
   replace the result by `F_CONV`.
7. Derive only the forward/`dW`/`dX` fields emitted in §3.2. Apply the
   source-level bias add, forward/backward truncation, bias-gradient row sum,
   and both weight/velocity and bias/velocity optimizer transitions to the
   named local handles. Match the ideal party view. The `F_TRUNC` and `F_OPT`
   reductions and implementation audit remain S8 obligations after S7
   implements this complete state transition.

The intended hybrid sequence is:

```text
H0 real two-process protocol
H1 honest-party CSPRNG/PRG streams -> ideal hidden random streams
H2 real bit-triple/OT/OLE transports -> F_BT/F_OT/F_OLE
H3 distributed DPF -> F_DDPF
H4 splittable Ring-LPN expansion -> F_RINGOLE
H5 protocol-backed conversion -> F_CONV
H6 stateful FC preprocessing -> F_FC.
```

The role-specific D1 simulators are complete at S1's named ideal boundary. The
end-to-end FC sequence remains an explicit outline: S2/S4/S6/S7 supply the real
primitive protocols and S8 supplies reductions, advantages, adjacent-state
composition, and implementation audit.

## 8. Proof obligations and current evidence

| ID | Obligation | Current evidence | Status |
|---|---|---|---|
| `P-CORR` | Keys reconstruct `beta [x=alpha]` | 2,432/2,432 full-domain passes; both primes; two deterministic point/payload edges; 6/6 invalid-input rejections; root seed, `sCW`, `tLCW`, `tRCW`, and `finalCW` corruption controls (5/5) | executable evidence |
| `P-DIST` | Joint output matches standard DPF distribution conditioned on party roots | No level-by-level coupling proof | open for S3/S8; blocks a security claim, not the S1 contract |
| `P-POS` | DPF points implement the exponent distribution induced by the exact uniform/regular Ring-LPN noise sampler | Non-wrapping position/offset sums specified; S2 reduction audit absent | contract fixed; open for S2 |
| `P-KEY` | One standard key hides point/payload | Relies on the cited construction and exact PRG semantics | open for S3/S8 |
| `P-ADD` | Ripple-adder view is simulatable for either party | Party-specific triple shares, sent/opened values, Beaver equations, and carry induction in §7.2–7.3 | closed in ideal-bit-triple hybrid |
| `P-LEVEL` | OT/CW view is simulatable conditioned on each full key/local state | Both sender/receiver roles and seed/flag share complements in §7.2–7.3 | closed in ideal-OT hybrid |
| `P-PAYLOAD` | Three-OLE Phase C realizes payload correction without sign leakage | Joint three-mask simulation for both parties; zero intermediates covered; old-sign regression | closed in ideal-OLE hybrid |
| `P-BATCH` | D1 simulation preserves correlated/repeated tree inputs and public order | Joint state-conditioned sequential hybrid with unique IDs in §7.1–7.3 | closed in D1 hybrid; production batching open |
| `P-FRESH` | No primitive correlation or private random-tape draw is reused | Ideal-functionality mask-draw accounting plus ideal-call duplicate-ID rejection; party private-tape draws are not counted | ideal-functionality control; private-tape/end-to-end freshness open for S3/S8 |
| `P-RNG` | Concrete PRG/CSPRNG state realizes the S1 random-tape interface | S1 exposes party roots explicitly and replaces only honest hidden streams | open for S3/S8 |
| `P-PCG` | Ring-LPN output is pseudorandom OLE at exact parameters/distribution | Figure-2 correctness; no S2 security audit | open/blocking |
| `P-CONV` | D2 securely realizes exact modulo-`Q` `F_CONV` without revealing wrap | Party-separated correctness prototype; deterministic sums `0,Q-1,Q,2Q-2`; executable `5ell-3` logical / `10ell-6` raw-share / `2ell-1` post-mask dependency-round accounting; offline correlations still dealer-generated | open for S6/S8 |
| `P-TOPO` | Stateful forward/bias/truncation/`dW`/`dX`/bias-gradient/dual-optimizer handle reuse, velocity evolution, and emitted fields match Orca | Source topology and serialization fixed in §3.2/§5; standalone artifact covers only one forward-shaped matmul | contract fixed; complete one-layer implementation required in S7 before S8; S9 scale only |
| `P-PROC` | Two-process implementation matches the transcript | Not implemented | open/blocking |
| `P-MAP` | Every current cross-party read/send maps to the contract | §5.1 maps protocol messages, target oracles, and validation-only reads | closed for current artifacts |

## 9. S1 mechanical gate and review boundary

This checkpoint may be committed as **“protocol/proof contract frozen for
review,” not “security proved,”** only after all internal conditions and the
required user/advisor review below pass:

1. the user-approved functionality/theorem hierarchy and metric choice are
   recorded;
2. every current D1 and D2–D4 cross-party value/read maps to this document or
   is named as a target or validation oracle;
3. the D1 gate separately enforces logical openings
   `2(L-1)`, `130L`, `ceil(log2 p)`, raw share payload
   `4(L-1)`, `260L`, `2 ceil(log2 p)`, ideal-mask-draw accounting, and
   ideal-call duplicate-ID rejection before output;
4. the D2 gate separately enforces wrap/no-wrap boundaries, `2ell-2` triples,
   `5ell-3` logical opened bits, `10ell-6` raw revealed-share bits,
   `2ell-1` post-mask dependency rounds, and exact modulo-`Q` conversion;
5. the paper contains the ideal functionalities, stateful Orca mask topology,
   exact DPF transcript, complete joint D1 hybrid simulators, and open-obligation
   table without upgrading claims;
6. GPU-NTT is cited as upstream comparison/dependency, while backend provenance
   and separate GPU-PCG/PIM reuse remain external-review items;
7. focused and canonical host gates pass; current docs/evidence are synchronized;
8. user/advisor or cryptography review reports no unresolved S1
   correctness/privacy blocker, and its disposition is recorded.

**Review disposition (2026-07-29).** The requested Opus 5 model-assisted
advisor audit found no remaining S1 freeze/commit blocker after the final
functionality, topology, notation, evidence, and 18-page build corrections. It
explicitly approved only the label **“contract frozen for advisor review.”**
Independent human cryptographic review remains an S8 prerequisite before any
computational-security or publication-readiness claim.

Open `P-DIST/P-KEY/P-PCG/P-CONV/P-RNG/P-PROC/P-TOPO` implementation or
reduction work prevents a computational-security/publication claim but does not
prevent freezing the explicit S1 contract that later stages must satisfy.
The checkpoint commit is created only after condition 8; stage S2 does not start
before that commit.

**Intended checkpoint subject:**
`ringlpn(proof): freeze semi-honest protocol and leakage contract`
