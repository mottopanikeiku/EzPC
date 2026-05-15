# Submitted Abstract Support Audit

Generated: 2026-05-07 08:34 UTC

## Verdict

The submitted abstract is mostly supported by saved GPU-MPC artifacts, but two phrases should be softened for poster or camera-ready use: `tens of gigabytes` is not shown by the saved local Orca profile, and `replace large precomputed keys` sounds like completed Orca integration even though the current DPF/VOLE results are standalone prototypes.

## Claim Matrix

| Claim | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Orca key-read time can approach GPU compute time | supported | P-LeNet key read 109.727 ms vs compute 107.727 ms; P-AlexNet key read 104.818 ms vs compute 121.727 ms | From GPU-MPC/orca_runner/logs/master.log |
| Local Orca key files grow from gigabytes to tens of gigabytes | partially supported | Saved local log reaches P-LeNet P0 4.0G, about 4.00 GiB per party | The saved local profile supports hundreds of MiB to about 4.0G per party, not tens of GB. Use a weaker phrase unless new larger-model key-size data is collected. |
| Chunked DPF online generation reaches 128x peak-footprint reduction with under 2x overhead | supported | N=1048576, chunk=8192, full pair key=360.00 MiB, partial peak=2.81 MiB, reduction=128.00x, overhead=1.834x | Standalone eval-all DPF key-generation benchmark, not end-to-end FSS evaluation. |
| GPU NTT/PolyMul core has roughly 89x full-PolyMul speedup over NFLLib | supported | Direct n=8192 comparison reports 87.59x forward-NTT and 89.24x full-PolyMul per-polynomial speedup | This is a per-polynomial throughput comparison from the saved n=8192 artifact. |
| GPU Ring-LPN VOLE validates across n=8192 to 1048576 for requested q=32 and q=64 | supported | q=32: n=8192 to 1048576, validation=pass/pass; q=64: n=8192 to 1048576, validation=pass/pass | Requested q=32 maps to actual q=30; requested q=64 maps to actual q=62. |
| The current work replaces Orca precomputed keys end-to-end | not supported | Current DPF and VOLE artifacts are standalone prototypes; Orca integration is documented as ongoing work | Use 'toward replacing' or 'study chunked online generation' rather than a completed replacement claim. |

## Shortened Evidence-Safe Abstract

Privacy-preserving machine learning (PPML) protocols often split computation between function secret sharing (FSS) for non-linear operations such as ReLU and comparison, and additive secret sharing for linear operations. Both use offline/online decompositions to reduce online latency, but this shifts cost to generating, storing, and moving correlated randomness. Our profiling of Orca, a GPU-accelerated FSS-based PPML system in GPU-MPC, shows that precomputed keys reach several gigabytes per party and that key-read time can match GPU computation for moderate models.

We develop GPU building blocks toward a unified acceleration framework for this bottleneck. For FSS-based non-linear evaluation, standalone chunked DPF online key generation reduces peak staged pair-key footprint by up to 128x with under 2x time overhead, providing a tunable memory-efficiency knob. For secret-sharing-based linear evaluation, we accelerate Ring-LPN pseudorandom correlation generator components. Our GPU NTT/PolyMul backend, adapted from Cheddar's two-phase kernel structure, achieves roughly 89x per-polynomial full-PolyMul speedup over the NFLLib CPU baseline at n=8192. Built on this backend, our standalone GPU Ring-LPN VOLE prototype validates correctness across degrees from 8192 to 1048576 for requested q=32 and q=64. We are currently integrating these components into Orca.

## Orca Profile Rows Extracted From `master.log`

| Model | Key files observed | Avg key read (ms) | Avg compute (ms) |
| --- | --- | ---: | ---: |
| P-SecureML | P0 338M, P1 338M | 9.909 | 32.273 |
| P-LeNet | P0 4.0G, P1 4.0G | 109.727 | 107.727 |
| P-AlexNet | P0 3.8G, P1 3.8G | 104.818 | 121.727 |

## Provenance

- Orca profiling: `GPU-MPC/orca_runner/logs/master.log`
- DPF chunking: `ringlpn/results/dpf_online_keygen_bin16_chunk8192.csv`
- VOLE q=32: `ringlpn/results/vole_gpu_q32_m32_c2_w64.csv`
- VOLE q=64: `ringlpn/results/vole_gpu_q64_m32_c2_w64.csv`
- CPU/GPU NTT comparison: `ringlpn/results/cpu_gpu_8192_32_batch64.md`
