#pragma once

// Ideal-OLE dealerless FC transcript (Checkpoint 2 of the dealerless roadmap).
//
// Unlike orca_fc_ringlpn_keywriter.cuh's buildCShare(), which is a centralized
// dealer/oracle that multiplies the *clear* masks A*B and then secret-shares the
// product, this transcript never forms A*B from both operands in one place. Each
// party samples its own additive mask shares A_i, B_i, Y_i locally; the two
// Beaver cross terms A0*B1 and A1*B0 are obtained from an *ideal OLE oracle*
// (the exact functionality the real Figure 2 Ring-LPN OLE will later realize);
// each party accumulates its local Beaver share Z_i over Z_M and converts once
// per output entry to Orca's Z_{2^bw} ring.
//
// This is still an *ideal-oracle* artifact: the OLE is a trusted functionality
// and the Z_M -> Z_{2^bw} conversion still uses the exact carry-correction that
// reads both shares (Step 2 replaces that with a secure protocol). But it proves
// the OLE-to-Beaver *reduction*, not merely the Orca key byte format, and it is a
// drop-in target for replacing the oracle with the real OLE engine (Step 5).
//
// Scope: single q62 limb (requested qbits=64). The conservative no-wrap bound is
// K * 2^(2*bw+2) < p62, satisfied by the demo default bw=16 (and any bw with
// K * 2^(2*bw+2) < ~2^62). bw=32 requires the q128/CRT per-limb extension and is
// tracked as a follow-up (Step 3), so this routine rejects qbits!=64.

#include <cstdint>
#include <random>
#include <vector>

#include "ringlpn/src/orca_fc_ringlpn_keywriter.cuh"

namespace ringlpn_orca {

struct TranscriptCounters {
    uint64_t ole_calls = 0;     // ideal-OLE invocations (== 2 * batch * M * K * N)
    uint64_t conversions = 0;   // Z_M -> Z_{2^bw} conversions (== batch * M * N)
    bool bound_ok = true;       // conservative no-prime-wrap bound held
};

// Ideal OLE functionality: returns additive shares (u0, u1) in [0, modulus) with
// u0 + u1 == x*y (mod modulus). Precondition: x*y < 2^128. Callers pass
// bw-bounded share operands (x, y < 2^bw), so the product does not overflow.
inline void idealOleShare(u128 x,
                          u128 y,
                          u128 modulus,
                          std::mt19937_64 &rng,
                          u128 &u0,
                          u128 &u1) {
    u128 prod = ((x % modulus) * (y % modulus)) % modulus;
    u0 = uniformMod(modulus, rng);
    u1 = modSub(prod, u0, modulus);
}

// Conservative no-wrap bound for full-width additive shares. Each share is in
// [0, 2^bw); the unreduced sum A0+A1 is < 2^(bw+1), so the integer dot of K such
// products is < K * 2^(2*bw+2). The single-limb conversion is exact only if this
// stays below the modulus.
inline bool transcriptNoWrapBound(const MatmulParams &p, u128 modulus) {
    if (p.bw <= 0 || p.bw > 30) {
        return false;
    }
    const u128 per_product = u128(1) << (2 * p.bw + 2);
    return u128(p.K) * per_product < modulus;
}

// Builds party-local A, B, C shares for one matmul from an ideal-OLE transcript.
// Outputs the two parties' key-entry vectors plus the reconstructed masks
// (mask_A = a0+a1, mask_B = b0+b1, mask_Y = y0+y1 over Z_{2^bw}) so the caller can
// drive the unchanged gpuMatmulBeaver online path and check the contract.
template <typename T>
inline bool buildIdealOleTranscript(const MatmulParams &p,
                                    int qbits,
                                    uint64_t seed,
                                    std::vector<T> &a0,
                                    std::vector<T> &a1,
                                    std::vector<T> &b0,
                                    std::vector<T> &b1,
                                    std::vector<T> &c0,
                                    std::vector<T> &c1,
                                    std::vector<T> &mask_a,
                                    std::vector<T> &mask_b,
                                    std::vector<T> &mask_y,
                                    TranscriptCounters &counters) {
    if (qbits != 64) {
        return false;  // q128/CRT per-limb transcript is a separate follow-up
    }
    if (p.bw <= 2 || p.bw > 30) {
        return false;
    }
    const u128 modulus = modulusForQbits(qbits);
    counters.bound_ok = transcriptNoWrapBound(p, modulus);
    if (!counters.bound_ok) {
        return false;
    }

    const int bw = p.bw;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> ring_dist(0, ringMask(bw));

    // Each party samples Z_{2^bw} additive shares of its mask locally.
    auto sample_shares = [&](size_t n,
                             std::vector<T> &s0,
                             std::vector<T> &s1,
                             std::vector<T> &mask) {
        s0.resize(n);
        s1.resize(n);
        mask.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const uint64_t x0 = ring_dist(rng);
            const uint64_t x1 = ring_dist(rng);
            s0[i] = static_cast<T>(x0);
            s1[i] = static_cast<T>(x1);
            mask[i] = static_cast<T>(ringAdd(x0, x1, bw));
        }
    };

    std::vector<T> y0;
    std::vector<T> y1;
    sample_shares(static_cast<size_t>(p.size_A), a0, a1, mask_a);
    sample_shares(static_cast<size_t>(p.size_B), b0, b1, mask_b);
    sample_shares(static_cast<size_t>(p.size_C), y0, y1, mask_y);

    c0.assign(static_cast<size_t>(p.size_C), T(0));
    c1.assign(static_cast<size_t>(p.size_C), T(0));

    counters.ole_calls = 0;
    counters.conversions = 0;

    for (int batch = 0; batch < p.batchSz; ++batch) {
        const size_t a_base = static_cast<size_t>(batch) * p.stride_A;
        const size_t b_base = static_cast<size_t>(batch) * p.stride_B;
        const size_t c_base = static_cast<size_t>(batch) * p.stride_C;
        for (int row = 0; row < p.M; ++row) {
            for (int col = 0; col < p.N; ++col) {
                u128 z0 = 0;
                u128 z1 = 0;
                for (int k = 0; k < p.K; ++k) {
                    const u128 av0 = matrixValue(a0, a_base, p.M, p.K, p.rowMaj_A, row, k, bw);
                    const u128 av1 = matrixValue(a1, a_base, p.M, p.K, p.rowMaj_A, row, k, bw);
                    const u128 bv0 = matrixValue(b0, b_base, p.K, p.N, p.rowMaj_B, k, col, bw);
                    const u128 bv1 = matrixValue(b1, b_base, p.K, p.N, p.rowMaj_B, k, col, bw);

                    // Local terms: each party multiplies its own shares.
                    z0 += av0 * bv0;
                    z1 += av1 * bv1;

                    // Cross terms via the ideal OLE oracle.
                    u128 u0 = 0;
                    u128 u1 = 0;
                    u128 v0 = 0;
                    u128 v1 = 0;
                    idealOleShare(av0, bv1, modulus, rng, u0, u1);
                    idealOleShare(av1, bv0, modulus, rng, v0, v1);
                    counters.ole_calls += 2;

                    z0 += u0 + v0;
                    z1 += u1 + v1;
                }
                z0 %= modulus;
                z1 %= modulus;

                uint64_t r0 = 0;
                uint64_t r1 = 0;
                exactZmToRingShares(z0, z1, modulus, bw, r0, r1);
                ++counters.conversions;

                const size_t c_idx = c_base + static_cast<size_t>(row) * p.N + col;
                c0[c_idx] = static_cast<T>(ringAdd(r0, ringValue(y0[c_idx], bw), bw));
                c1[c_idx] = static_cast<T>(ringAdd(r1, ringValue(y1[c_idx], bw), bw));
            }
        }
    }
    return true;
}

}  // namespace ringlpn_orca
