#include "spfss_host.h"

#include <cassert>
#include <cstring>

namespace spfss_host {

namespace {

inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline U128 make_u128(uint64_t lo, uint64_t hi) {
    return (U128)lo | ((U128)hi << 64);
}

inline uint64_t u128_lo(U128 x) { return (uint64_t)x; }
inline uint64_t u128_hi(U128 x) { return (uint64_t)(x >> 64); }

// Domain-separated PRG: expands a 128-bit seed to (sL, tL, sR, tR).
// State derivation: seed two splitmix64 streams from (lo ^ mix_lo, hi ^ mix_hi).
// Not cryptographic — correctness infrastructure only.
inline void prg_expand(U128 seed, U128 &sL, uint8_t &tL, U128 &sR, uint8_t &tR) {
    uint64_t s0 = u128_lo(seed) ^ 0xA24BAED4963EE407ULL;
    uint64_t s1 = u128_hi(seed) ^ 0x9FB21C651E98DF25ULL;
    uint64_t a = splitmix64(s0);
    uint64_t b = splitmix64(s0);
    uint64_t c = splitmix64(s1);
    uint64_t d = splitmix64(s1);
    uint64_t tag = splitmix64(s0) ^ splitmix64(s1);
    sL = make_u128(a, b);
    sR = make_u128(c, d);
    tL = (uint8_t)(tag & 1);
    tR = (uint8_t)((tag >> 1) & 1);
}

inline Word mod_add(Word a, Word b, Word m) {
    Word s = a + b;
    if (s >= m || s < a) s -= m;
    return s;
}

inline Word mod_sub(Word a, Word b, Word m) {
    return a >= b ? a - b : m - (b - a);
}

// Map a 128-bit seed to a Z_p element by (lo + hi) mod p (commutative,
// domain-separated from the PRG). Standard CONVERT for DPFs with Z_p payload.
inline Word convert_zp(U128 s, Word m) {
    Word lo = (Word)(u128_lo(s) % m);
    Word hi = (Word)(u128_hi(s) % m);
    return mod_add(lo, hi, m);
}

inline U128 fresh_seed(uint64_t &rng_state) {
    uint64_t lo = splitmix64(rng_state);
    uint64_t hi = splitmix64(rng_state);
    return make_u128(lo, hi);
}

} // namespace

void dpfGen(uint64_t alpha, int log_domain, Word beta, Word modulus,
            uint64_t &rng_state, DPFKey &K0, DPFKey &K1) {
    assert(log_domain >= 1 && log_domain <= 63);
    assert(alpha < (1ULL << log_domain));
    assert(beta < modulus);

    K0.log_domain = K1.log_domain = log_domain;
    K0.modulus = K1.modulus = modulus;
    K0.seed = fresh_seed(rng_state);
    K1.seed = fresh_seed(rng_state);
    K0.t0 = 0;
    K1.t0 = 1;
    K0.sCW.resize(log_domain);
    K1.sCW.resize(log_domain);
    K0.tLCW.resize(log_domain);
    K1.tLCW.resize(log_domain);
    K0.tRCW.resize(log_domain);
    K1.tRCW.resize(log_domain);

    U128 s0 = K0.seed, s1 = K1.seed;
    uint8_t t0 = 0, t1 = 1;

    for (int i = 0; i < log_domain; ++i) {
        U128 s0L, s0R, s1L, s1R;
        uint8_t t0L, t0R, t1L, t1R;
        prg_expand(s0, s0L, t0L, s0R, t0R);
        prg_expand(s1, s1L, t1L, s1R, t1R);

        // alpha's bit at this level: 0 = go left (on-path left), 1 = go right
        int bit_idx = log_domain - 1 - i;
        uint8_t abit = (uint8_t)((alpha >> bit_idx) & 1ULL);

        U128 lose0, lose1;
        U128 keep0, keep1;
        uint8_t tKeep0, tKeep1;
        if (abit == 0) {
            lose0 = s0R; lose1 = s1R;
            keep0 = s0L; keep1 = s1L;
            tKeep0 = t0L; tKeep1 = t1L;
        } else {
            lose0 = s0L; lose1 = s1L;
            keep0 = s0R; keep1 = s1R;
            tKeep0 = t0R; tKeep1 = t1R;
        }

        U128 sCW = lose0 ^ lose1;
        uint8_t tLCW = (uint8_t)((t0L ^ t1L ^ abit ^ 1) & 1);
        uint8_t tRCW = (uint8_t)((t0R ^ t1R ^ abit) & 1);

        K0.sCW[i] = sCW; K1.sCW[i] = sCW;
        K0.tLCW[i] = tLCW; K1.tLCW[i] = tLCW;
        K0.tRCW[i] = tRCW; K1.tRCW[i] = tRCW;

        uint8_t tCW_chosen = (abit == 0) ? tLCW : tRCW;
        s0 = keep0 ^ (t0 ? sCW : (U128)0);
        s1 = keep1 ^ (t1 ? sCW : (U128)0);
        t0 = (uint8_t)((tKeep0 ^ (t0 ? tCW_chosen : 0)) & 1);
        t1 = (uint8_t)((tKeep1 ^ (t1 ? tCW_chosen : 0)) & 1);
    }

    // Final Z_p correction. Party output at leaf = (-1)^party * (convert(s) + t*finalCW).
    // Sum = (c0 - c1) + (t0_leaf - t1_leaf) * finalCW. On-path at leaf, t0 XOR t1 = 1.
    //   t1_leaf == 0 (so t0_leaf == 1): sum = (c0 - c1) + finalCW = beta ⇒ finalCW = beta - c0 + c1
    //   t1_leaf == 1 (so t0_leaf == 0): sum = (c0 - c1) - finalCW = beta ⇒ finalCW = c0 - c1 - beta
    // Off-path at leaf: t0_leaf == t1_leaf and seeds equal ⇒ sum = 0 regardless of finalCW.
    Word c0 = convert_zp(s0, modulus);
    Word c1 = convert_zp(s1, modulus);
    Word diff = mod_sub(mod_add(beta, c1, modulus), c0, modulus); // beta - c0 + c1 (mod p)
    Word finalCW = (t1 == 0) ? diff : mod_sub(0, diff, modulus);
    K0.finalCW = finalCW;
    K1.finalCW = finalCW;
}

Word dpfEval(int party, const DPFKey &K, uint64_t x) {
    U128 s = K.seed;
    uint8_t t = (party == 0) ? 0 : 1;
    int log_domain = K.log_domain;
    for (int i = 0; i < log_domain; ++i) {
        U128 sL, sR;
        uint8_t tL, tR;
        prg_expand(s, sL, tL, sR, tR);
        int bit_idx = log_domain - 1 - i;
        uint8_t xbit = (uint8_t)((x >> bit_idx) & 1ULL);
        U128 s_next;
        uint8_t t_next;
        if (xbit == 0) {
            s_next = sL ^ (t ? K.sCW[i] : (U128)0);
            t_next = (uint8_t)((tL ^ (t ? K.tLCW[i] : 0)) & 1);
        } else {
            s_next = sR ^ (t ? K.sCW[i] : (U128)0);
            t_next = (uint8_t)((tR ^ (t ? K.tRCW[i] : 0)) & 1);
        }
        s = s_next;
        t = t_next;
    }
    Word c = convert_zp(s, K.modulus);
    Word v = t ? mod_add(c, K.finalCW, K.modulus) : c;
    // Party 0 outputs +v, party 1 outputs -v. Their sum at x=alpha is beta,
    // elsewhere 0.
    return (party == 0) ? v : mod_sub(0, v, K.modulus);
}

void dpfEvalAll(int party, const DPFKey &K, std::vector<Word> &out) {
    uint64_t domain = (1ULL << K.log_domain);
    out.assign((size_t)domain, 0);
    // Simple recursive tree walk to avoid re-expanding seeds per leaf.
    struct Frame {
        U128 s;
        uint8_t t;
        int depth;
        uint64_t prefix;
    };
    std::vector<Frame> stack;
    stack.reserve((size_t)K.log_domain + 2);
    stack.push_back({K.seed, (uint8_t)(party == 0 ? 0 : 1), 0, 0});
    while (!stack.empty()) {
        Frame f = stack.back();
        stack.pop_back();
        if (f.depth == K.log_domain) {
            Word c = convert_zp(f.s, K.modulus);
            Word v = f.t ? mod_add(c, K.finalCW, K.modulus) : c;
            Word share = (party == 0) ? v : mod_sub(0, v, K.modulus);
            out[(size_t)f.prefix] = share;
            continue;
        }
        U128 sL, sR;
        uint8_t tL, tR;
        prg_expand(f.s, sL, tL, sR, tR);
        U128 sL_next = sL ^ (f.t ? K.sCW[f.depth] : (U128)0);
        uint8_t tL_next = (uint8_t)((tL ^ (f.t ? K.tLCW[f.depth] : 0)) & 1);
        U128 sR_next = sR ^ (f.t ? K.sCW[f.depth] : (U128)0);
        uint8_t tR_next = (uint8_t)((tR ^ (f.t ? K.tRCW[f.depth] : 0)) & 1);
        // Push right then left so left is processed first (preorder).
        stack.push_back({sR_next, tR_next, f.depth + 1, (f.prefix << 1) | 1ULL});
        stack.push_back({sL_next, tL_next, f.depth + 1, (f.prefix << 1)});
    }
}

void spfssGen(const std::vector<uint64_t> &alphas,
              const std::vector<Word> &betas,
              int log_domain, Word modulus, uint64_t &rng_state,
              SPFSSKey &K0, SPFSSKey &K1) {
    assert(alphas.size() == betas.size());
    K0.log_domain = K1.log_domain = log_domain;
    K0.modulus = K1.modulus = modulus;
    K0.dpf_keys.resize(alphas.size());
    K1.dpf_keys.resize(alphas.size());
    for (size_t k = 0; k < alphas.size(); ++k) {
        dpfGen(alphas[k], log_domain, betas[k], modulus, rng_state,
               K0.dpf_keys[k], K1.dpf_keys[k]);
    }
}

void spfssFullEval(int party, const SPFSSKey &K, std::vector<Word> &out) {
    uint64_t domain = (1ULL << K.log_domain);
    out.assign((size_t)domain, 0);
    std::vector<Word> per_dpf((size_t)domain, 0);
    for (size_t k = 0; k < K.dpf_keys.size(); ++k) {
        dpfEvalAll(party, K.dpf_keys[k], per_dpf);
        for (size_t x = 0; x < (size_t)domain; ++x) {
            out[x] = mod_add(out[x], per_dpf[x], K.modulus);
        }
    }
}

} // namespace spfss_host
