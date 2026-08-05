# Native-ring PCG technology audit

**Date:** 2026-08-04
**Status:** internal/advisor; **NO-GO for both current publication tracks**
**Scope:** source-pinned audit of Li--Xing--Yao--Yuan and its released native-$\mathbb Z_{2^{bw}}$/Galois-ring artifact
**Authorship boundary:** Alp remains the sole current paper/checkpoint author. This report records external prior art and model-assisted audit work; it does not assign paper authorship or settle contributor credit. Substantive future theorem development or private-project reuse requires an explicit credit/coauthorship decision before circulation.

> **Do not treat this artifact as dealerless, concretely secure, matrix-capable, or Orca-integrated.** The release is useful source material for a centralized, correctness-only native-ring experiment. It has no executable distributed seed-generation protocol, no current concrete-security point, no circuit-dependent/matrix implementation, and no Orca serializer or integration. It must not displace either the live two-process systems path (Paper A) or the audit of this repository's actually implemented regular Ring-LPN distribution (Paper B).

## Recommendation

**NO-GO as a fallback for either selected publication track; park it as a quarantined second-paper idea.** The printed and released construction is QA-SD-based, not a direct implementation of the current project's Ring-LPN route. Its source also has full-width arithmetic defects and benchmark semantics that do not establish native-ring triple correctness.

A later conditional **GO** requires all four gates:

1. a new, independently checked parameter analysis against the 2025 and 2026 QA-SD attacks, or a fully specified and analyzed Ring-LPN instantiation;
2. an executable two-party distributed SPFSS/PCG seed-generation protocol;
3. an implemented circuit-dependent or matrix-triple construction; and
4. byte-exact `GPUMatmulKey` output accepted by unchanged `gpuMatmulBeaver`.

Satisfying only the toy oracle at the end of this report would not satisfy any security, dealerlessness, matrix, or integration gate.

## Reproducible external pins and licenses

| Item | Immutable/current pin | License and audit fact |
|---|---|---|
| Paper | Li--Xing--Yao--Yuan, *Efficient Pseudorandom Correlation Generators over $\mathbb Z/p^k\mathbb Z$*, CRYPTO 2025, [DOI 10.1007/978-3-032-01884-7_7](https://doi.org/10.1007/978-3-032-01884-7_7); [ePrint 2025/1223](https://eprint.iacr.org/2025/1223), current PDF modified 2025-07-01; downloaded PDF SHA-256 `130178519ec48cbbe64b201e42bb581b513f77503f671e1a56e4b0909b4aa375` | CC BY-NC 4.0 |
| Canonical artifact | [`zhli271828/Trace-F2-OLE-PCG` commit `43959ef19cee4b25d0580ea0c12499c564e2328d`](https://github.com/zhli271828/Trace-F2-OLE-PCG/commit/43959ef19cee4b25d0580ea0c12499c564e2328d), dated 2025-08-26 | [MIT](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/LICENSE). The canonical repository had no tags or releases and its public issues endpoint was empty at audit time. |
| DPF submodule | [`zhli271828/base-ary-dpf` commit `80378aa4d00935792946b5bb5c83de146bb38188`](https://github.com/zhli271828/base-ary-dpf/tree/80378aa4d00935792946b5bb5c83de146bb38188), pinned at `libs/base-ary-dpf` by the artifact's [`.gitmodules`](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/.gitmodules) | MIT |
| Estimator submodule | [`mbombar/estimator_folding` commit `2dd8bd551b4f4ce3133275e69fb5468f27913191`](https://github.com/mbombar/estimator_folding/tree/2dd8bd551b4f4ce3133275e69fb5468f27913191), pinned at `scripts/parameters_selection` by the same `.gitmodules` | MIT |
| 2025 attack/correction | [ePrint 2025/892](https://eprint.iacr.org/2025/892), revision 4, 2025-11-14 | CC BY 4.0 |
| 2026 Joux correlation attack | [ePrint 2026/1126](https://eprint.iacr.org/2026/1126), initial/current version, 2026-06-01 | CC BY 4.0 |
| 2026 large-prime follow-up | [ePrint 2026/196](https://eprint.iacr.org/2026/196), revision 3, 2026-06-03 | Large-prime-field QA-SD OLE/VOLE follow-up; not a native-ring repair |

To reproduce the artifact checkout, clone the canonical repository at the exact commit and initialize its recursive submodules; verify that the two checked-out submodule commits equal the table. To reproduce the paper pin, download the 2025/1223 PDF current on 2025-07-01 and verify the stated SHA-256. The links above, rather than a moving branch name, are the evidence pins.

No public repair or successor for the direct-ring route was found through 2026-08-04 in the canonical repository's releases, issues, tags, public forks, or the official ePrint follow-ups reviewed in this audit.

## What the paper and artifact actually provide

### QA-SD construction, not a released Ring-LPN instantiation

The concrete OLE theorem in [ePrint 2025/1223, Sections 5 and 5.1](https://eprint.iacr.org/2025/1223) is **QA-SD based**. The paper says the construction can “easily” be extended to Ring-LPN, but the printed construction, theorem, released parameter rows, and artifact instantiate QA-SD over the residue field. Calling the repository a direct Ring-LPN implementation would be false.

This distinction also separates the present audit from Paper B: the QA-SD attacks discussed below do not by themselves establish an attack on the project's deployed one-sample, large-$p$, negacyclic regular Ring-LPN distribution. Conversely, that separation supplies no security evidence for this QA-SD native-ring artifact.

### Centralized key generation, not dealerless setup

The theory's `Gen` receives both parties' sparse supports/payloads and emits both SPFSS key shares. The source mirrors that interface. The pinned [`DPFGen`](https://github.com/zhli271828/base-ary-dpf/blob/80378aa4d00935792946b5bb5c83de146bb38188/include/dpf.h#L24-L31) receives the private index and message and writes both `k0` and `k1` in one address space:

```c
void DPFGen(
    struct PRFKeys *prf_keys,
    size_t domain_size,
    size_t index,
    uint128_t *msg_blocks,
    size_t msg_block_len,
    struct DPFKey *k0,
    struct DPFKey *k1);
```

There is no socket, OT, or VOLE transport and no party-isolated seed-generation executable. Section 6 of the paper only says that malicious distributed seed generation is “straightforward to adapt” from finite-field work. That is an availability claim, not a protocol listing or implementation. Its estimate that distributed communication would be only “slightly higher” than seed size is therefore not a measured dealerless result.

### Benchmark kernels, not a complete correlation oracle

The released ring and SPDZ files are benchmark kernels. For example, the pinned [`SPDZ2k_64_bench.c`](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/src/SPDZ2k_64_bench.c#L449-L488) independently chooses a random position and payload for every DPF and derives a MAC payload using the centralized `K`. The only dispatched correctness test is `modular_test`, whose constants are an $\mathbb F_4$ test (`N=16,C=4,T=27`). The ring/SPDZ dispatches only time and free their arrays; the generic dispatch is visible in [`src/main.c`](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/src/main.c#L155-L176).

**[INFERENCE]** Those independently sampled benchmark payloads do not preserve the shared-error cross-key relations required by Construction 3. A successful benchmark run is consequently not evidence that reconstructed native-ring triples are correct.

## Exact arithmetic defects

### Four undefined shifts

There are **four** instances of `1<<(k+s)` in pinned [`src/modular_bench.c`](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/src/modular_bench.c#L17-L55), covering ordinary and high-degree 32/64-bit authenticated paths. Because the left operand has type `int`, the shipped $(k,s)=(32,26)$ and $(64,57)$ shifts are undefined C behavior. In unguarded paths, the resulting zero reaches `% modulus`; guarded high-degree paths can silently continue with a false zero modulus.

For $\ell=k+s$, the minimally correct representable-width formulas are:

- when $\ell<64$: `uint64_t modulus = UINT64_C(1) << ell`;
- when $\ell<128$: `uint128_t modulus = ((uint128_t)1) << ell`.

At $\ell=64$ or $128$, however, $2^\ell$ is not representable in a same-width type. A typed shift remains undefined. Reduction must instead use native unsigned wrap, or a full-width-aware mask:

```c
mask = ell == W ? UINT_W_MAX : (((T)1 << ell) - 1);
```

and replace `% modulus` with `& mask` where that reduction is algebraically appropriate.

Therefore, the local typed-shift adaptation is sufficient only for the degree-2 table rows ($\ell=58,121$). It does **not** repair the degree-3 $(64,64)$ path, where $\ell=128$.

### Centralized MAC key does not implement Construction 3

Construction 3 samples party-local $\Delta_0,\Delta_1\in\mathbb Z_{2^r}$. The source instead samples one combined `K64`/`K128` and reduces it modulo $2^\ell$; see the sampler above and [Construction 3](https://eprint.iacr.org/2025/1223). This distribution mismatch is another reason to label the code a centralized benchmark, not an SPDZ2k setup.

## 2025 correction and shipped mismatch

The repository README's post-2025/892 condition is

$$
n\le 1+\frac{(c-1)(q-1)\log q}{\log(q-1)}.
$$

The paper appendix prints the corresponding sufficient criterion strictly as $n$ less than the right-hand side. The relevant numerical thresholds are:

| $(c,q)$ | Right-hand side | Largest admitted integer $n$ |
|---|---:|---:|
| $(3,4)$ | $8.571157\ldots$ | $8$ |
| $(5,4)$ | $16.142314\ldots$ | $16$ |
| $(9,4)$ | $31.284628\ldots$ | $31$ |
| $(3,8)$ | $15.960702\ldots$ | $15$ |
| $(6,16)$ | $77.787407\ldots$ | $77$ |

Thus the shipped $n=15/16$ with $(c,q)=(3,4)$ is outside the corrected condition. The pinned [artifact README](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/README.md) recommends `(c=5,t=27,q=4,n<=16)` and explicitly warns that `(3,27,4,16)` is insecure. The generic executable nevertheless hard-codes `c=3,t=27` (and `n=15`) in [`src/main.c`](https://github.com/zhli271828/Trace-F2-OLE-PCG/blob/43959ef19cee4b25d0580ea0c12499c564e2328d/src/main.c#L155-L176).

The paper's reported 64-bit SPDZ row uses

$$
(k,s,n,c,t,m)=(64,57,16,5,27,2)
$$

and calls the `c=3` measurements benchmark-only; see Section 7.2 and Table 5 of [ePrint 2025/1223](https://eprint.iacr.org/2025/1223). Any patched `c=5` row is therefore **adapted**, not a reproduction of the released executable.

## 2026 attack status

The official abstract of [ePrint 2026/1126](https://eprint.iacr.org/2026/1126) reports a Joux correlation attack that recovers secret error polynomials at larger weights, improves time and memory by about $1000\times$ over $\mathbb F_3$, obtains still larger gains over $\mathbb F_4$, and requires affected PCG parameters to be “entirely” revisited.

Because this native-ring construction reduces its QA-SD security to the small residue field and its headline degree-2 rows use $q=4$, the README's post-2025 `c=5,t=27` example is not a current concrete-security pin. **The official abstract alone does not prove that this exact row is broken.** Its honest status is **unrevalidated/no concrete claim**, not “128-bit.”

The same authors' [ePrint 2026/196](https://eprint.iacr.org/2026/196) follow-up is WHT-based QA-SD OLE/VOLE for arbitrary **large prime fields**. It is not for $\mathbb Z_{2^k}$ or Galois rings and does not repair the undefined shifts, provide native-ring distributed setup or matrix triples, or re-estimate this artifact's security.

## Plain Beaver versus SPDZ2k semantics

### Plain path

The artifact describes its `gr64_trace`/`gr128_trace` files as semi-honest generalized-trace multiplication-triple benchmarks over $\mathbb Z_{2^{32}}$ and $\mathbb Z_{2^{64}}$. Algebraically, low-`bw` additive shares of $(A,B,C)$ with $C=AB$ can feed Orca after adding party shares of the output mask to $C$. That transformation retains only **semi-honest plain-Beaver semantics**.

### Authenticated path

Construction 3 outputs, for each party,

$$
(\Delta_i,X_i,Y_i,Z_i,M_{X,i},M_{Y,i},M_{Z,i}),
$$

with value/MAC shares in $\mathbb Z_{2^\ell}$ and MAC-key shares in $\mathbb Z_{2^r}$. [Remark 6.2](https://eprint.iacr.org/2025/1223) guarantees the stronger product relation modulo $2^\ell$ and explicitly requires masking the high $\ell-k$ bits during openings to prevent leakage.

Orca's live [`GPUMatmulKey`](../../../fss/gpu_matmul.h) has only sequential `A`, `B`, `C`, followed by a truncation key. Its [`gpuMatmulBeaver`](../../../fss/gpu_matmul.cu) consumer performs no SPDZ MAC or opening check. Dropping $\Delta$/MAC fields and reducing values to their low $k$ bits can produce a semi-honest Beaver input, but **discards all malicious-SPDZ2k security**. Serializing the MAC payloads into `A/B/C` would instead break Orca's ABI.

Therefore SPDZ2k output is not semantically compatible with Orca authentication. A plain reduction is a security-model cutover, not preservation of authenticated preprocessing.

## Orca compatibility and matrix boundary

Stock [`gpuKeygenMatmul`](../../../fss/gpu_matmul.cu) writes additive shares of mask matrices $A$, $B$, and $AB+Z$ in raw `A/B/C` order. [`readGPUMatmulKey`](../../../fss/gpu_matmul.h) and `gpuMatmulBeaver` consume exactly those fields.

The native-ring paper mentions matrix triples only in one extension paragraph in its introduction and gives an asymptotic seed-size statement,

$$
O(m\lambda^3\log N)
$$

for $N/m$ triples of $m\times m$ matrices. It does not give the construction. The artifact exports scalar benchmark arrays and contains no circuit-dependent/matrix implementation or key serializer.

Scalar triples cannot simply be laid out as an Orca fully connected key: an $M\times K$ mask element must be reused across all $N$ products, and a $K\times N$ mask element must be reused across all $M$ products. Independent scalar triples do not enforce that reuse. **[INFERENCE]** A real non-$1\times1$ adapter requires the paper's unimplemented circuit-dependent/matrix PCG, or another protocol that programs the same reuse, before it can preserve Orca's live ABI.

Accordingly, this audit is not evidence of matrix capability or Orca integration.

## Smallest honest future correctness oracle — toy only

The maximum justified prototype before new cryptanalysis is deliberately **toy, centralized, plain, and $1\times1$**. It answers only: “Can the printed native-ring algebra produce one byte-compatible Beaver key?” It must never be described as dealerless, matrix-capable, performant, concretely secure, or maliciously secure.

1. Use $m=2$, `bw=32`, $N=3^6$, $t=27$, and $c=2$. Since $t^2=N$, this minimizes the full-evaluation domain. This tuple is a correctness fixture only and must never appear in a parameter/security table.
2. Port the actual sparse-error/cross-term generation from the $\mathbb F_4$ `modular_test` into $GR(2^{32},2)`. Do **not** call the independent-payload benchmark sampler. Generate both DPF keys centrally, then expand party 0 and party 1 separately.
3. **Dense reference oracle:** explicitly form $b_0=as_0+e_0$ and $b_1=as_1+e_1$, apply the generalized trace, and compute the clear coordinatewise product modulo $2^{32}$.
4. **DPF oracle:** for every generated key pair, reconstruct both full-domain evaluations and compare them with the expected sparse cross-term vector, including duplicate-position accumulation.
5. **Triple oracle:** for all $2N$ outputs require
   $$
   ((x_0+x_1)(y_0+y_1)-(z_0+z_1))\bmod 2^{32}=0.
   $$
   Flip one payload or index as a corruption control and require failure.
6. **ABI oracle:** take one verified coordinate; sample additive output-mask shares $Z_0,Z_1$; serialize `A_i=x_i`, `B_i=y_i`, and `C_i=z_i+Z_i` in stock order; invoke unchanged `readGPUMatmulKey`/`gpuMatmulBeaver` with `M=K=N=batchSz=1` and `TruncateType::None`; compare the reconstructed output with clear $xy+Z$. Run the same masks through stock `gpuKeygenMatmul` as a differential oracle.
7. An optional, separate authenticated oracle may sample $\Delta_0,\Delta_1$, verify every reconstructed MAC equation modulo $2^\ell$, verify the stronger triple equation modulo $2^\ell$, and exercise the required high-bit opening masks. It must not be described as Orca malicious security.

This oracle is a future experiment definition, not implemented evidence or a publication path.

## Audited interface anchors

The released benchmark entry points relevant to this disposition are:

```c
void gr128_trace_bench_pcg(
    size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void SPDZ2k_64_bench_pcg(
    size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);
```

The exact live Orca boundary that a future toy ABI oracle would exercise is:

```c++
template <typename T>
GPUMatmulKey<T> readGPUMatmulKey(
    MatmulParams p, TruncateType t, uint8_t **key_as_bytes);

template <typename T>
T *gpuKeygenMatmul(
    u8 **key_as_bytes, int party, MatmulParams p,
    T *d_mask_X, T *h_mask_W, T *h_mask_Y, TruncateType t,
    AESGlobalContext *gaes, bool wIsOnGpu = false, T *d_mask_Z = NULL);

template <typename T>
T *gpuMatmulBeaver(
    MatmulParams p, GPUMatmulKey<T> k, int party,
    T *d_A, T *d_B, T *d_r0, T *d_r1, T *d_bias, Stats *s);
```

These signatures identify benchmark and oracle boundaries; they do not imply that the native-ring artifact implements or invokes the Orca interfaces.

## Audit conclusion and caveats

- **Publication disposition:** NO-GO for Paper A and Paper B; quarantine as a possible second-paper idea only after all four gates close.
- **Security:** no concrete QA-SD security claim survives the 2026 status without new evaluation. The exact `c=5,t=27` cost under the Joux attack was unavailable from the official abstract, so this report says unrevalidated rather than claiming the row is broken.
- **Dealerlessness:** absent. Centralized `DPFGen` and centralized MAC sampling are not a distributed setup.
- **Correctness:** native-ring/SPDZ dispatches benchmark time, not the needed cross-key reconstruction oracle; four shifts are undefined and the full-width case needs more than a typed literal.
- **Matrix/Orca:** absent. The paper's asymptotic matrix-extension paragraph is not an implementation, scalar triples do not encode matrix-mask reuse, and Orca has no SPDZ authentication fields/checks.
- **Track separation:** the cited QA-SD attacks are not themselves attacks on the repository's distinct regular Ring-LPN distribution, but that distinction does not validate this artifact.
- **Search boundary:** no public repair/successor was found through the audit date; ePrint 2026/196 is a large-prime-field construction, not a native-ring replacement.
- **Validation boundary:** this was a source audit. No artifact benchmark, build, test, formatter, linter, GPU run, or project-wide validation was performed.
