# Dealerless Orca forward-FC preprocessing — security contract and proof boundary

**Original freeze:** 2026-07-29  
**Updated:** 2026-08-04  
**Status:** exact forward-FC coupling, role-specific batch simulators,
conversion simulator, source map, and conditional theorem are complete for the
current live artifact. Two independent model-assisted reviews found no
critical defect after fixes. This is not an independent human cryptographic
review, a concrete Ring-LPN security result, an authenticated deployment, or a
publication-readiness claim.  
**Target:** one integrated dealerless Orca forward-FC matmul  
**Adversary:** one statically corrupted semi-honest party; authenticated point-to-point channels; external network observers, active attacks, denial of service, and side channels are out of scope  
**Proof structure:** an end-to-end forward-FC functionality with the distributed DPF as a named subfunction and theorem  
**Phase C:** corrected three-OLE transcript; only `finalCW` is reconstructed  
**Author:** Alp (sole author, by user direction; inherited work remains cited
and its ownership/reuse boundary remains subject to S2)

This document fixes what the current forward implementation realizes and what
its conditional theorem proves. The older splitmix64/ideal-functionality host
artifact remains a correctness reference. The live path separately uses real
SCI/IKNP/Gilboa transport, full-width GPU AES, private OpenSSL DRBG state, exact
public-polynomial exchange, GPU Ring-LPN expansion, exact conversion, and
party-local Orca key records.

The 2026-08-04 exact primary-source parameter audit is complete: it invalidated
several out-of-domain estimator rows and found no reviewed mapping from the
deployed projected distribution/structured code to the accepted finite-field
models. No parameter or 128-bit classical/quantum claim is pinned; this
strengthens, rather than closes, `P-POS` and `P-PCG`.

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

**Questions to resolve with the professor before claim advancement or external
circulation** (the owner separately allowed implementation-only S3--S6 work):

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
5. Which parameter should be measured after a reviewed projection/distribution,
   structured-code, and two-limb advantage analysis establishes a valid set?
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

**Exact coupling theorem D-DIST.** Conditioned on the two supplied roots, the
real protocol emits exactly the correction words and `finalCW` of
`DPF.Gen(alpha,beta;root_0,root_1)`; this is equality, not merely
computational indistinguishability. Section 4.5 proves the coupling by induction
on the standard DPF prefix invariant and then identifies the Phase-C equation
with the standard final correction. Uniform independent root draws therefore
give the same joint key-pair distribution as standard generation. Single-key
privacy remains the separate `D-KEY` reduction to the seed-expansion PRG.

**Seed-format obligation D-SEED.** The target follows the formal seed/tag
separation of Boyle--Gilboa--Ishai, *Function Secret Sharing: Improvements and
Extensions*, CCS 2016, DOI `10.1145/2976749.2978429`: a full
`lambda`-bit secret seed plus separate child control bits. The deployed
Ring-LPN GPU expansion now makes four domain-separated AES calls per node:
plaintexts 0 and 2 produce full 128-bit child seeds, while plaintexts 1 and 3
produce the two control bits. The host twin matches 16 freshly device-dumped
vectors with zero seed/tag mismatches; 88 two-process keys pass batched and
per-tree GPU evaluation across both CRT primes. Party roots come from
OpenSSL's private DRBG.

The centralized benchmark-only GPU keygen still derives roots from one 64-bit
`seed_base`; it is not a security realization and must not be used as one.
The two-party transport closes the 127-bit encoding defect for its key path,
but D-DIST, P-RNG state/composition review, P-KEY, and the concrete reduction
still block a 128-bit DPF-security claim.

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

- `F_COIN^p`: on a fresh identifier for `c` public degree-`<n` polynomials,
  each party samples `c*n` independent uniform field elements, the parties
  exchange canonical coefficient vectors, and both set the public vector to
  their componentwise sum in `Z_p`. Conditioned on either party's contribution,
  the honest contribution makes the result exactly uniform. The semi-honest
  implementation uses one fixed-order exchange; a malicious extension would
  require commit/open.
- `F_RINGOLE`: for public `R=Z_p[X]/(X^n+1)` and the exact uniform public
  polynomial vector from `F_COIN^p`, sample uniform
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
Each logical `delta` or `epsilon` opening sends one meaningful bit from each
party, so the meaningful share-width count is `4(L-1)` bits. This excludes
byte padding, framing, setup, and OT traffic. The carry dependency is
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
`260L` meaningful share bits. These are neither byte-aligned wire bits nor a
claim of network rounds; S4 must measure the real transport.

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
`ell_p` bits. The two parties each reveal one field share, so the meaningful
share-width count is `2 ell_p` bits. For the current 62-bit primes these are
respectively 62 and 124 bits. This is not encoded wire traffic. The
implementation's fixed-width encoding must use `ceil(log2 p)` after any
parameter re-pin.

### 4.5 Exact coupling to standard DPF generation

**Lemma.** Fix canonical inputs, the two root seeds, and the PRG outputs. At
every level the protocol's common seed and flag correction words equal those
computed by standard BGI-style DPF generation for `(alpha,beta)`.

**Proof.** After any fixed common-CW prefix, pair the two parties' frontier
states at every tree prefix. The standard DPF invariant says that every
off-path pair has equal seeds and equal tags, while the pair at the prefix of
`alpha` has opposite tags. Hence all off-path terms cancel from the XOR
aggregates. If the next point bit is `a=a_0 xor a_1`, then

```text
sCW_0 xor sCW_1
 = S_0^R xor S_1^R xor (a_0 xor a_1)(Z_0 xor Z_1)
 = S_0^R xor S_1^R                              when a=0
 = S_0^L xor S_1^L                              when a=1.
```

The first equality substitutes
`q_0=r_1 xor a_0 Z_1` and `q_1=r_0 xor a_1 Z_0`; both fresh OT masks cancel.
The remaining aggregate is exactly the two path seeds on the losing branch,
which is the standard seed correction. Likewise,

```text
tLCW = T_0^L xor T_1^L xor a xor 1
tRCW = T_0^R xor T_1^R xor a,
```

the standard two flag corrections. Applying these words makes the losing child
states equal and preserves opposite tags on the keeping child, establishing
the invariant for the next level. The root state supplies the base case.

At the leaves, the same cancellation gives
`A=A_0+A_1=Convert(seed_0,alpha)-Convert(seed_1,alpha)` and
`F=F_0+F_1=t_0,alpha-t_1,alpha in {+1,-1}`. The three OLEs reconstruct
`gamma=beta`, both cross terms, and therefore

```text
w_0+w_1 = (d_0+d_1)(s_0+s_1) = (beta-A)F.
```

Because `F^(-1)=F`, this is exactly the standard final correction: `beta-A`
when `(t_0,t_1)=(1,0)` and `A-beta` when `(t_0,t_1)=(0,1)`. Thus every common
word and each party root field equals the conditioned standard output. QED.

This is an algebraic distribution proof at the fixed-PRG boundary. It does not
by itself prove that one key hides `alpha,beta`; that statement invokes the
standard DPF single-key theorem and the concrete PRG assumption.

### 4.6 Complete per-tree accounting

```text
string OTs                 = 2L
bit triples                = L-1
scalar OLEs                = 3
logical opened bits        = 2(L-1) + 130L + ceil(log2 p)
meaningful share bits       = 4(L-1) + 260L + 2 ceil(log2 p).
```

At the current `L=14`, 62-bit primes, these are 1,908 logical opened bits and
3,816 meaningful share bits. The previously published 3,790 mixed Phase A's
logical opening count with Phases B/C's share-width count and is therefore
not a coherent metric. Neither corrected counter is real network traffic:
OT/OLE setup and payloads, byte padding, framing, commitments, and
retransmission remain absent from the host prototype. S4 replaces estimates
with measured bytes.

## 5. Live forward-FC transcript and target stateful extension

For each accepted matmul invocation `mu`, the target two-process transcript is:

1. Both parties validate the same public dimensions, layout, parameter set,
   identifiers, mask handles, slot capacity, output lengths, and membership in
   the public admissible set
   `2 < bw <= 32`, `K >= 1`, `K * 2^(2*bw+2) < Q(qbits)`.
   For the q62 limb this permits at most `K=2^28-1` at `bw=16`,
   `K=2^12-1` at `bw=24`, and no `K>=1` at `bw=32`; q128 permits
   `K<2^90`, `K<2^74`, and `K<2^58`, respectively. A failure produces common
   `(abort,stage)` before reading masks or consuming correlation.
2. `F_COIN^p` establishes the exact uniform public Ring-LPN polynomial by
   exchanging and adding two canonical coefficient vectors. Each party obtains
   all private roots, sparse noise, masks, and primitive randomness from its
   independent OS CSPRNG; no public value seeds private state.
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

The live two-process D2 artifact realizes step 8 over the SCI/IKNP transport.
For `ell=ceil(log2(2Q))`, one exact daBit uses one 128-bit OT, `ell` daBits
compose one exact edaBit, and each Boolean triple uses two one-bit OTs. The
online protocol opens the `ell`-bit masked value
`A=(z_0+z_1+R) mod 2^ell`, evaluates two ripple adders using
`2ell-2` fresh bit triples, opens one masked bit for B2A, and applies the local
`Q*wrap` correction. This exposes `5ell-3` logical opened bits and
`10ell-6` meaningful share bits per conversion. The wrap bit is neither
reconstructed nor opened. The following lemma closes privacy in the
`(F_DABIT,F_EDABIT,F_BT)` hybrid; transport security remains conditional on
the semi-honest OT theorem and an authenticated channel. The closed forms are
not measured bytes or rounds. Before base-OT setup, the parties exchange
and acknowledge a fixed canonical encoding of every workload-shaping public
parameter. The artifact separately records setup, agreement, TEST-ONLY
selftest, production correlation, online, and final-sync bytes/direction
switches, and gates their exact sum. Its Layer-mode generator also rejects
public inputs outside `K*2^(2bw+2)<Q`; this bound is an FC intended-integer
precondition, not a restriction on generic canonical share conversion.

**Exact-conversion lemma P-CONV.** Let `R` be the edaBit value. Because
`R` is uniform in `Z_(2^ell)`, the opened
`A=(z_0+z_1+R) mod 2^ell` is uniform and independent of the canonical input
sum. The first ripple adder computes secret XOR shares of
`A + bitwise-not(R) + 1 = z_0+z_1 mod 2^ell`; the second adds
`2^ell-Q`, so its final carry is exactly `[z_0+z_1 >= Q]`, using
`z_0+z_1<2Q<=2^ell`. Every AND opening is one-time-padded by a fresh bit
triple and is simulated by choosing the common masked bits uniformly and
complementing the honest sent shares. The B2A opening is
`wrap xor d` for a fresh uniform daBit `d`, hence is also uniform and leaks no
wrap bit. Finally each party returns
`z_b-Q*d_b^A mod 2^bw`. Since `Q` is odd, multiplication by `Q` is a
permutation of `Z_(2^bw)`; the local arithmetic daBit share is uniform, so
either output share is uniform and the two shares sum to
`(z_0+z_1-Q*wrap) mod 2^bw`, exactly `F_CONV`.

For either corruption, a simulator takes the ideal uniform output share,
solves uniquely for the corresponding local arithmetic daBit share using
`Q^(-1) mod 2^bw`, samples the local Boolean/edaBit/triple shares with their
real marginals, chooses the public masked sum and every masked opening
uniformly, and sets only honest transmitted shares to the required
complements. This reproduces the full conditional view. Sequential batching is
valid because every daBit, edaBit, triple, and conversion SID is fresh.

The live two-process artifact now realizes steps 1--10 for one forward-shaped
matmul: party-local mask sampling, distributed GPU-AES DPF key generation,
party-local Ring-LPN expansion, both derandomization directions, exact
two-party conversion, transactional party-local key records, and post-exit
unchanged-Orca validation. It contains no dealer/oracle in the live execution.
The checker is intentionally omniscient and runs only after both party
processes exit. The artifact does not implement the stateful training
transitions in steps 11--12, authenticated transport, a malicious adversary, or
the open Ring-LPN parameter/reduction obligation. Those remain named
boundaries rather than hidden implementation gaps.

### 5.1 Source-to-transcript map for the live composition

| Current source boundary | Cross-party value or read | Contract line | Status |
|---|---|---|---|
| `agree_fc_preflight()` | canonical dimensions, parameters, session ID, local-validity bit | step 1 | common rejection before OT/private-mask sampling/output |
| `PartyRandom` + `sample_ring_words()` | no cross-party value; private `A_b,B_b,R_C,b` draws | steps 2--3 | OpenSSL private DRBG; party-local |
| full public-`a` exchange in `generate_ring_ole()` | one canonical `c*n`-coefficient share vector from each party | `F_COIN^p`, step 2 | exact uniform public polynomial; separately counted |
| `sample_party_noise()` | no cross-party value; one party's sparse noise only | step 4 | party-local independent draw |
| `agree_spfss_public_manifest()` | public SPFSS dimensions/session plus local-validity bit | step 4 | common agreement before DPF output |
| `two_party_dpf_gen_batch()` Phase A | bit-triple shares and transmitted `delta,epsilon` shares | §4.1 | mapped real SCI/IKNP triples and openings |
| Phase-B directional OTs and CW exchange | selected 128-bit OT outputs; opened seed/flag CW shares | §4.3 | mapped; exact-CW coupling proved in §4.5 |
| three batched Gilboa OLE calls + final-CW exchange | field OLE shares; one final-CW share per party | §4.4 | mapped; no sign/difference opening |
| `pack_gpu_party_keys()` / `expand_ring_ole_party()` | no peer read; own noise/key and common public `a` only | steps 4--5 | party-local GPU expansion |
| post-Ring-OLE status exchange | one generated/not-generated byte per direction/limb/batch from each party | step 4 abort boundary | common abort before any derandomization opening for that instance |
| `exchange_openings()` | canonical operand-minus-slot word from each party per used slot | step 6 | mapped `d,e`; direction/limb/batch have distinct SIDs and fresh state |
| `accumulate_local_products()` | no cross-party value | step 7 | party-local same-party products |
| `secure_convert_batch()` preflight | canonical `Q,bw,count,sid` plus local-validity bit | step 8 | common agreement before conversion correlation |
| conversion daBit/edaBit/triple generation | SCI/IKNP OT messages | `F_DABIT/F_EDABIT/F_BT` | live semi-honest transport |
| conversion masked-sum/addition/B2A openings | fixed-width masked shares | step 8 | mapped; wrap is never opened |
| post-conversion status exchange | one conversion-validity byte from each party | step 8 abort boundary | common abort before output-mask addition/publication |
| `publish_record()` | staged-output-validity and rename-result bytes | step 9 | sibling temporary, owner-only mode, bilateral result exchange |
| `PartyChannel::sync()` | no semantic value | transport accounting | counted separately from logical openings |
| `run_check()` | reads both finalized records, reconstructs masks, runs matched dealer and unchanged online consumer | validation only | post-exit test oracle; absent from both live parties |
| runner controls | mismatched preflight, stale output, rename failure, corrupt record, swapped records | abort/publication contract | all must reject or clean up |

Every live cross-party send is mapped above. In the live source trace, neither
party reads its peer's file or private arrays. Each party writes its own final
key record, and the post-exit checker intentionally reads both. The current
single-UID loopback runner does not enforce OS-level peer file isolation; that
requires distinct UIDs/containers and inaccessible mounts. TCP is loopback and
unauthenticated; raw private key records are deleted after the checker. The
per-record SHA-256 is accidental-corruption detection, not authentication. The
runner's `COMMITTED` marker is supervisor evidence that both processes exited
and the checker passed, not a cryptographic commit protocol.

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
- each party's full public-polynomial contribution and their modular sum;
- accept/abort and the stage at which an abort occurs.

### 6.2 Permitted per-party view

In addition to common leakage, `P_b` sees only:

- its private DPF inputs/root draws and its shares in the persistent mask store,
  including intentionally reused forward/`dW`/`dX` operand shares;
- its OT sender inputs, receiver choices/outputs, bit-triple/OLE shares, local
  frontiers, noise shares, fresh output-mask shares, and correlation IDs;
- its output DPF keys, emitted Orca key fields, and next-state mask shares;
- messages it sends or receives (authenticated in the target functionality;
  plain unauthenticated TCP in the current artifact).

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

### 7.4 Conditional forward-FC theorem

Let `chi_U(p,n,t)` sample exactly `t` distinct positions without replacement,
with independent coefficients uniform in `Z_p^*`. Let `chi_R(p,n,t)` divide
the ring into `t` public equal buckets and sample one uniform position per
bucket, again with independent uniform nonzero coefficients. These are the
two distributions implemented by `sample_party_noise()`.

**Exact decisional Ring-LPN assumption.** For the selected `chi` and
`R=Z_p[X]/(X^n+1)`, if `a_1,...,a_c` are independent uniform ring elements and
`e_1,...,e_c <- chi`, then
`(a_1,...,a_c,sum_i a_i e_i)` is computationally indistinguishable from
`(a_1,...,a_c,u)` for uniform `u in R`. The assumption is distribution- and
parameter-specific. This document does not assign a bit-security level to the
exercised `(n=8192,c=2,t=8)` feasibility point.

**Theorem (forward only, conditional).** In the static semi-honest model with
authenticated point-to-point channels, assuming (i) the four-call AES
seed-expansion is a PRG, (ii) the used IKNP OT and Gilboa OLE realizations have
their standard semi-honest security, (iii) the standard BGI DPF single-key
theorem, and (iv) the exact decisional Ring-LPN assumption above together with
the Figure-2 PCG reduction, the protocol in §5 realizes the forward-matmul
restriction of `F_FC` with the leakage of §6.

For a corrupt `P_b`, condition on its full mask store, DPF inputs/roots, public
shape/order, and ideal forward key. The simulator proceeds as follows.

1. For every public polynomial, sample the corrupt party's real uniform
   coefficient contribution and set the honest contribution to
   `a-public_contribution_b`; this exactly simulates `F_COIN^p`.
2. Execute the corrupt party's private random tape and PRG computations
   consistently. Replace only the honest party's hidden AES stream by ideal
   randomness, paying the PRG advantage.
3. Replace real triples/OTs/OLEs by their ideal functionalities. Apply the
   joint D1 simulators in §§7.1--7.3; §4.5 identifies their conditioned output
   with `F_DDPF`, and the standard single-key theorem hides the honest point,
   payload factor, and root.
4. Apply the Figure-2 reduction under the exact `chi`-Ring-LPN assumption to
   replace each fresh party-local expansion by `F_RINGOLE`.
5. For each unique slot, compute the corrupt party's own `d` or `e` from its
   operand and OLE share. The opposing OLE share is fresh uniform in the
   hybrid, so choose the honest opening uniformly and derive its sent share.
   Reused operand handles do not reuse OLE slots or correlation IDs.
6. Apply the exact-conversion simulator of §5 to replace the SCI/IKNP
   conversion transcript by `F_CONV`.
7. The corrupt output field is
   `C_b=converted_b+R_(C,b) mod 2^bw`. Its fresh uniform `R_(C,b)` makes the
   share uniform subject to the ideal sum, while `A_b,B_b` are the conditioned
   operand-mask shares. Serialize exactly `A_b || B_b || C_b`.

Each hybrid conditions on the complete prior batch state and consumes a fresh
domain-separated identifier, so sequential composition preserves repeated or
correlated private DPF inputs. The algebra in §5 proves that the two serialized
keys reconstruct the ideal `AB+R_C` mask and therefore match the unchanged
Orca consumer.

The theorem does **not** cover the live TCP channel against an external
attacker, malicious parties, training-state transitions, or a concrete
security level. The current implementation is theorem-aligned source evidence
for honest loopback execution; authenticating the channel and obtaining a
reviewed parameter/reduction instantiation remain publication gates.

The hybrid sequence is:

```text
H0 real forward protocol over an authenticated channel
H1 honest AES stream -> ideal hidden random stream
H2 IKNP/Gilboa/triple transports -> F_BT/F_OT/F_OLE
H3 distributed DPF -> F_DDPF
H4 Figure-2 expansion -> F_RINGOLE under exact chi-Ring-LPN
H5 conversion -> F_CONV
H6 forward matmul preprocessing -> F_FC|forward.
```

The full training-layer statement remains an outline. It requires a separate
bias/truncation/gradient/optimizer handle-state implementation, source map, and
composition review extending this forward-only theorem.

## 8. Proof obligations and current evidence

| ID | Obligation | Current evidence | Status |
|---|---|---|---|
| `P-CORR` | Keys reconstruct `beta [x=alpha]` | 2,432/2,432 full-domain passes; both primes; two deterministic point/payload edges; 6/6 invalid-input rejections; root seed, `sCW`, `tLCW`, `tRCW`, and `finalCW` corruption controls (5/5) | executable evidence |
| `P-DIST` | Joint output matches standard DPF distribution conditioned on party roots | §4.5 exact level-by-level CW and final-CW coupling; full-width AES/GPU compatibility gates | closed algebraically at fixed-PRG boundary |
| `P-POS` | DPF points implement the exponent distribution induced by the exact uniform/regular Ring-LPN noise sampler | S2 source audit found block-conditioned inputs, dependent projected occupancy/cancellation, and no reviewed distribution/tail reduction to the estimator models | contract fixed; open/blocking |
| `P-KEY` | One standard key hides point/payload | Exact D-DIST coupling reduces this to the standard BGI single-key theorem and the concrete four-call seed-expansion PRG | conditional on standard DPF/PRG theorem; concrete reduction review open |
| `P-ADD` | Ripple-adder view is simulatable for either party | Party-specific triple shares, sent/opened values, Beaver equations, and carry induction in §7.2--7.3 | closed in ideal-bit-triple hybrid |
| `P-LEVEL` | OT/CW view is simulatable conditioned on each full key/local state | Both sender/receiver roles and seed/flag share complements in §7.2--7.3 | closed in ideal-OT hybrid |
| `P-PAYLOAD` | Three-OLE Phase C realizes payload correction without sign leakage | Joint three-mask simulation for both parties; zero intermediates covered; old-sign regression | closed in ideal-OLE hybrid |
| `P-BATCH` | D1 simulation preserves correlated/repeated tree inputs and public order | Joint state-conditioned sequential hybrid with unique IDs in §7.1--7.3 | closed in D1 hybrid |
| `P-FRESH` | No primitive correlation or private random-tape draw is reused | Distinct derived SIDs for every tree group/direction/limb/batch/conversion chunk; duplicate-ID controls | live trace evidence; formal end-to-end ledger review open |
| `P-RNG` | Concrete PRG/CSPRNG state realizes the S1 random-tape interface | Private roots/noise/masks use OpenSSL's private DRBG; public `a` is the sum of two full uniform field vectors; GPU DPF expansion uses four domain-separated AES calls | implementation evidence; concrete reduction review open |
| `P-PCG` | Ring-LPN output is pseudorandom OLE at exact parameters/distribution | Figure-2 correctness only; S2 audit found BCG rule inconsistency, invalid estimator calls, no structured-code/two-limb advantage reduction, and no parameter pin | open/blocking |
| `P-CONV` | D2 securely realizes exact modulo-`Q` `F_CONV` without revealing wrap | Exact-conversion lemma above plus two-process SCI/IKNP boundary/control runs | closed in daBit/edaBit/triple hybrid; real transport conditional on semi-honest OT and authenticated channels |
| `P-TOPO` | Stateful forward/bias/truncation/`dW`/`dX`/bias-gradient/dual-optimizer handle reuse, velocity evolution, and emitted fields match Orca | Live artifact covers one complete forward matmul only | forward closed; training-state extension open |
| `P-PROC` | Two-process implementation matches the forward transcript | Live q64/q128 regular/uniform/multibatch runs use separate OS processes and GPUs; exact ResNet18 classifier-layer run added (not full inference/truncation) | forward component closed; authenticated deployment open |
| `P-MAP` | Every current cross-party read/send maps to the contract | §5.1 maps the complete live source and separates the post-exit checker | closed for current forward artifact; re-audit on source change |

## 9. Review boundary and current disposition

The original S1 checkpoint was correctly committed as **“protocol/proof
contract frozen for review,” not “security proved.”** Its mechanical
conditions were:

1. the user-approved functionality/theorem hierarchy and metric choice are
   recorded;
2. every current D1 and D2–D4 cross-party value/read maps to this document or
   is named as a target or validation oracle;
3. the D1 gate separately enforces logical openings
   `2(L-1)`, `130L`, `ceil(log2 p)`, meaningful share widths
   `4(L-1)`, `260L`, `2 ceil(log2 p)`, ideal-mask-draw accounting, and
   ideal-call duplicate-ID rejection before output;
4. the D2 gate separately enforces wrap/no-wrap boundaries, Layer-mode
   admissibility, cross-party public-parameter agreement before base OT,
   `2ell-2` triples, `5ell-3` logical opened bits, `10ell-6` meaningful share
   bits, `2ell-1` post-mask dependency stages, exact modulo-`Q` conversion, and
   complete traffic-category accounting;
5. the paper contains the ideal functionalities, stateful Orca mask topology,
   exact DPF transcript, complete joint D1 hybrid simulators, and open-obligation
   table without upgrading claims;
6. GPU-NTT is cited as upstream comparison/dependency, while backend provenance
   and separate GPU-PCG/PIM reuse remain external-review items;
7. focused and canonical host gates pass; current docs/evidence are synchronized;
8. user/advisor or cryptography review reports no unresolved S1
   correctness/privacy blocker, and its disposition is recorded.

**Original disposition (2026-07-29).** The requested Opus 5 model-assisted
audit found no remaining S1 freeze/commit blocker and approved only the label
“contract frozen for advisor review.”

**Current disposition (2026-08-04).** The exact DPF coupling, both joint batch
simulators, conversion simulator, live source-to-transcript map, and
conditional forward theorem now close `P-DIST`, `P-ADD`, `P-LEVEL`,
`P-PAYLOAD`, `P-BATCH`, `P-CONV`, and the bounded forward portions of
`P-TOPO/P-PROC/P-MAP` at their stated hybrid boundaries. Independent
model-assisted proof and implementation reviews found no critical defect after
the recorded fixes. Independent human cryptographic review is still required.

`P-POS` and `P-PCG` remain hard blockers because no reviewed
distribution/structured-code/two-limb reduction or concrete parameter pin
exists. `P-KEY` remains conditional on the standard DPF/PRG theorem;
`P-RNG/P-FRESH` require renewed audit after any source change; the live TCP
transport is unauthenticated. Therefore the honest claim is a
*conditional forward-FC theorem and theorem-aligned executable artifact*,
not “security proved,” “128-bit secure,” or publication-ready.

**Next security checkpoint subject:**
`ringlpn(security): close forward simulation boundary`
