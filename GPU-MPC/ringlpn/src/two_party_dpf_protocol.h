// two_party_dpf_protocol.h - the batched two-party DPF key generation protocol,
// shared by the standalone keygen artifact and the Ring-LPN SPFSS keygen.
//
// Extracted verbatim from src/test_two_party_dpf_keygen.cpp so both binaries run
// the SAME protocol code; the standalone artifact's gate (identical per-tree
// correlation and opening counts, 369/369 offline-validated pairs) therefore
// covers this implementation too.
//
// Every cross-party value goes through OT or through an explicitly counted
// opening. Trees are processed level-synchronously: one OT batch and one
// opening dependency stage per level for the whole batch, plus three batched
// Gilboa OLE stages. The measured direction-switch count depends on tree depth,
// not batch size; it is not a network-round measurement.
//
// The expansion PRG is selectable: `kSplitmix` matches the unchanged host
// evaluator `spfss_host::dpfEvalAll`; `kGpuAes` is the bit-identical twin of the
// deployed four-call AES PRG with full 128-bit child seeds and independently
// derived control bits. The latter removes D-SEED's 127-bit encoding defect;
// P-RNG/P-DIST/P-KEY and the full reduction remain open.

#pragma once

#include "gpu_aes_prg_host.h"
#include "spfss_host.h"
#include "two_party_ot.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ringlpn_2pdpf {


using ringlpn_2pc::BitTriple;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using ringlpn_2pc::U128;
using ringlpn_2pc::Word;
using ringlpn_2pc::mod_add;
using ringlpn_2pc::mod_mul;
using ringlpn_2pc::mod_sub;

constexpr Word kPrime62 = 4611686018326724609ULL;      // 2^62 - 6*2^24 + 1
constexpr Word kPrime62Crt2 = 4611686018309947393ULL;  // 2^62 - 7*2^24 + 1

// PRG / convert: identical to spfss_host.cpp's file-local versions. Validation
// through the unchanged evaluator guards against drift.

inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline U128 make_u128(uint64_t lo, uint64_t hi) {
    return (U128)lo | ((U128)hi << 64);
}

inline void prg_expand(U128 seed, U128 &sL, uint8_t &tL, U128 &sR, uint8_t &tR) {
    uint64_t s0 = (uint64_t)seed ^ 0xA24BAED4963EE407ULL;
    uint64_t s1 = (uint64_t)(seed >> 64) ^ 0x9FB21C651E98DF25ULL;
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

inline Word convert_zp(U128 s, Word m) {
    Word lo = (Word)((uint64_t)s % m);
    Word hi = (Word)((uint64_t)(s >> 64) % m);
    return mod_add(lo, hi, m);
}

struct Node {
    U128 s;
    uint8_t t;
};

// Expansion PRG selector. `splitmix` matches the unchanged host evaluator
// `spfss_host::dpfEvalAll`; `gpu_aes` is the bit-identical twin of the deployed
// four-call GPU device PRG (src/gpu_aes_prg_host.h), so its keys are consumable
// by the GPU evaluator. The protocol transcript and accounting are unchanged.
enum class PrgMode { kSplitmix, kGpuAes };

inline void expand_node(PrgMode mode, U128 seed, U128 &sL, uint8_t &tL,
                        U128 &sR, uint8_t &tR) {
    if (mode == PrgMode::kGpuAes) {
        ringlpn_gpu_prg::gpu_aes_prg_expand(seed, sL, tL, sR, tR);
    } else {
        prg_expand(seed, sL, tL, sR, tR);
    }
}

// ----- the batched party-local protocol -------------------------------------
//
// Every cross-party value goes through OT or through an explicitly counted
// opening. Trees are processed level-synchronously: one OT batch and one
// opening per level for the whole batch.

bool two_party_dpf_gen_batch(int party, int log_domain, Word p, PrgMode prg,
                             const std::vector<uint64_t> &offs,
                             const std::vector<Word> &beta_factors,
                             PartyChannel &ch, PartyRandom &rng,
                             std::vector<spfss_host::DPFKey> &keys) {
    const int L = log_domain;
    const size_t B = offs.size();
    const size_t max_int =
        static_cast<size_t>(std::numeric_limits<int>::max());
    if ((party != 0 && party != 1) || L < 2 || L > 20 ||
        (p != kPrime62 && p != kPrime62Crt2) || B == 0 ||
        beta_factors.size() != B || B > max_int / size_t(L - 1)) {
        return false;
    }
    const uint64_t half_domain = 1ULL << (L - 1);
    for (size_t b = 0; b < B; ++b) {
        // Local input validation happens before any correlation is consumed.
        if (offs[b] >= half_domain || beta_factors[b] == 0 ||
            beta_factors[b] >= p) {
            return false;
        }
    }

    // --- Phase A: XOR-shared bits of alpha = off_0 + off_1, all trees at once -
    std::vector<BitTriple> triples;
    ringlpn_2pc::generate_bit_triples(ch, (int)(B * (size_t)(L - 1)), rng,
                                      triples);

    std::vector<uint8_t> abit(B * (size_t)L, 0);
    std::vector<uint8_t> carry(B, 0);
    std::vector<uint8_t> open_mine(2 * B), open_theirs(2 * B);
    for (int j = 0; j < L; ++j) {
        for (size_t b = 0; b < B; ++b) {
            const uint8_t u = (uint8_t)((offs[b] >> j) & 1);
            abit[b * (size_t)L + (size_t)j] = (uint8_t)(u ^ carry[b]);
        }
        if (j + 1 >= L) break;
        // d = x ^ a and e = y ^ b are independent, so one stage opens both for
        // every tree in the batch.
        for (size_t b = 0; b < B; ++b) {
            const uint8_t u = (uint8_t)((offs[b] >> j) & 1);
            const uint8_t x_mine =
                (party == 0) ? (uint8_t)(u ^ carry[b]) : carry[b];
            const uint8_t y_mine =
                (party == 0) ? carry[b] : (uint8_t)(u ^ carry[b]);
            const BitTriple &t = triples[b * (size_t)(L - 1) + (size_t)j];
            open_mine[2 * b] = (uint8_t)((x_mine ^ t.a) & 1);
            open_mine[2 * b + 1] = (uint8_t)((y_mine ^ t.b) & 1);
        }
        ch.exchange_bytes(open_mine.data(), open_theirs.data(), 2 * B);
        ch.costs.phase_a.logical_bits += 2ULL * (uint64_t)B;
        ch.costs.phase_a.revealed_bits_sent += 2ULL * (uint64_t)B;
        ch.costs.phase_a.revealed_bits_recv += 2ULL * (uint64_t)B;
        for (size_t b = 0; b < B; ++b) {
            const uint8_t d = (uint8_t)((open_mine[2 * b] ^ open_theirs[2 * b]) & 1);
            const uint8_t e =
                (uint8_t)((open_mine[2 * b + 1] ^ open_theirs[2 * b + 1]) & 1);
            const BitTriple &t = triples[b * (size_t)(L - 1) + (size_t)j];
            const uint8_t and_share = (uint8_t)(
                (((party == 0) ? (uint8_t)(d & e) : (uint8_t)0) ^
                 (uint8_t)(d & t.b) ^ (uint8_t)(e & t.a) ^ t.c) & 1);
            carry[b] = (uint8_t)(and_share ^ carry[b]);
        }
    }

    // --- root seeds: this party's own private-CSPRNG draws -------------------
    keys.assign(B, spfss_host::DPFKey{});
    std::vector<std::vector<Node>> nodes(B), next(B);
    for (size_t b = 0; b < B; ++b) {
        spfss_host::DPFKey &K = keys[b];
        K.log_domain = L;
        K.modulus = p;
        K.seed = rng.u128();
        K.t0 = (uint8_t)party;
        K.sCW.assign((size_t)L, 0);
        K.tLCW.assign((size_t)L, 0);
        K.tRCW.assign((size_t)L, 0);
        nodes[b].assign(1, Node{K.seed, (uint8_t)party});
    }

    // --- Phase B: level-synchronous walk over the whole batch ----------------
    std::vector<U128> expL, expR;
    std::vector<uint8_t> expTL, expTR;
    std::vector<U128> Z(B), masks(B), ot_m0(B), ot_m1(B), ot_out(B);
    std::vector<U128> aggR(B);
    std::vector<uint8_t> aggTL(B), aggTR(B), choices(B);
    // Per level each tree opens 16 bytes of seed-CW share and 1 byte holding
    // the two flag-CW shares; both are independent, so one stage carries them.
    std::vector<uint8_t> level_mine(B * 17), level_theirs(B * 17);
    for (int i = 0; i < L; ++i) {
        const int bi = L - 1 - i;
        for (size_t b = 0; b < B; ++b) {
            const size_t nn = nodes[b].size();
            expL.resize(nn);
            expR.resize(nn);
            expTL.resize(nn);
            expTR.resize(nn);
            U128 aL = 0, aR = 0;
            uint8_t tL = 0, tR = 0;
            for (size_t k = 0; k < nn; ++k) {
                expand_node(prg, nodes[b][k].s, expL[k], expTL[k], expR[k],
                            expTR[k]);
                aL ^= expL[k];
                aR ^= expR[k];
                tL ^= expTL[k];
                tR ^= expTR[k];
            }
            aggR[b] = aR;
            aggTL[b] = tL;
            aggTR[b] = tR;
            Z[b] = aL ^ aR;
            masks[b] = rng.u128();
            ot_m0[b] = masks[b];
            ot_m1[b] = masks[b] ^ Z[b];
            choices[b] = abit[b * (size_t)L + (size_t)bi];
        }
        // One OT batch per direction per level, fixed order (party 0 sends
        // first) so the pair never blocks.
        if (party == 0) {
            ch.ot_send_128(ot_m0, ot_m1);
            ot_out = ch.ot_recv_128(choices);
        } else {
            ot_out = ch.ot_recv_128(choices);
            ch.ot_send_128(ot_m0, ot_m1);
        }

        for (size_t b = 0; b < B; ++b) {
            const uint8_t a_mine = choices[b];
            const U128 sCW_share =
                aggR[b] ^ (a_mine ? Z[b] : (U128)0) ^ ot_out[b] ^ masks[b];
            std::memcpy(&level_mine[b * 17], &sCW_share, 16);
            const uint8_t tL_share =
                (uint8_t)((aggTL[b] ^ a_mine ^ ((party == 0) ? 1u : 0u)) & 1);
            const uint8_t tR_share = (uint8_t)((aggTR[b] ^ a_mine) & 1);
            level_mine[b * 17 + 16] =
                (uint8_t)(tL_share | (uint8_t)(tR_share << 1));
        }
        ch.exchange_bytes(level_mine.data(), level_theirs.data(), B * 17);
        ch.costs.phase_b.logical_bits += 130ULL * (uint64_t)B;
        ch.costs.phase_b.revealed_bits_sent += 130ULL * (uint64_t)B;
        ch.costs.phase_b.revealed_bits_recv += 130ULL * (uint64_t)B;

        for (size_t b = 0; b < B; ++b) {
            U128 mine_cw = 0, theirs_cw = 0;
            std::memcpy(&mine_cw, &level_mine[b * 17], 16);
            std::memcpy(&theirs_cw, &level_theirs[b * 17], 16);
            const U128 sCW = mine_cw ^ theirs_cw;
            const uint8_t flags =
                (uint8_t)(level_mine[b * 17 + 16] ^ level_theirs[b * 17 + 16]);
            const uint8_t tLCW = (uint8_t)(flags & 1);
            const uint8_t tRCW = (uint8_t)((flags >> 1) & 1);
            keys[b].sCW[(size_t)i] = sCW;
            keys[b].tLCW[(size_t)i] = tLCW;
            keys[b].tRCW[(size_t)i] = tRCW;

            const size_t nn = nodes[b].size();
            next[b].resize(nn * 2);
            for (size_t k = 0; k < nn; ++k) {
                U128 sL, sR;
                uint8_t tl, tr;
                const uint8_t t = nodes[b][k].t;
                expand_node(prg, nodes[b][k].s, sL, tl, sR, tr);
                if (t) {
                    sL ^= sCW;
                    sR ^= sCW;
                    tl = (uint8_t)(tl ^ tLCW);
                    tr = (uint8_t)(tr ^ tRCW);
                }
                next[b][2 * k] = Node{sL, (uint8_t)(tl & 1)};
                next[b][2 * k + 1] = Node{sR, (uint8_t)(tr & 1)};
            }
            nodes[b].swap(next[b]);
        }
    }

    // --- Phase C: payload correction words, three batched OLE stages ---------
    std::vector<Word> A(B, 0), Fsum(B, 0);
    for (size_t b = 0; b < B; ++b) {
        for (const Node &nd : nodes[b]) {
            const Word conv = convert_zp(nd.s, p);
            if (party == 0) {
                A[b] = mod_add(A[b], conv, p);
                Fsum[b] = mod_add(Fsum[b], nd.t, p);
            } else {
                A[b] = mod_sub(A[b], conv, p);
                Fsum[b] = mod_sub(Fsum[b], nd.t, p);
            }
        }
    }

    // OLE stage 1: additive shares of beta = beta_0 * beta_1 for every tree.
    const std::vector<Word> gamma =
        ringlpn_2pc::ole_batch_p0_sender(ch, beta_factors, p, rng);
    std::vector<Word> d(B), s(B);
    for (size_t b = 0; b < B; ++b) {
        d[b] = mod_sub(gamma[b], A[b], p);
        s[b] = Fsum[b];
    }
    // OLE stage 2: shares of d_0 * s_1. Stage 3: shares of s_0 * d_1.
    const std::vector<Word> cross01 = ringlpn_2pc::ole_batch_p0_sender(
        ch, (party == 0) ? d : s, p, rng);
    const std::vector<Word> cross10 = ringlpn_2pc::ole_batch_p0_sender(
        ch, (party == 0) ? s : d, p, rng);

    std::vector<uint64_t> final_mine(B), final_theirs(B);
    for (size_t b = 0; b < B; ++b) {
        final_mine[b] = (uint64_t)mod_add(
            mod_add(mod_mul(d[b], s[b], p), cross01[b], p), cross10[b], p);
    }
    // Only the standard public key material is opened: never d, s, or the
    // hidden leaf-control sign (see the contract's Phase C decision).
    ch.exchange_bytes(reinterpret_cast<const uint8_t *>(final_mine.data()),
                      reinterpret_cast<uint8_t *>(final_theirs.data()),
                      B * sizeof(uint64_t));
    const uint64_t fbits = (uint64_t)ringlpn_2pc::field_bits(p);
    ch.costs.phase_c.logical_bits += fbits * (uint64_t)B;
    ch.costs.phase_c.revealed_bits_sent += fbits * (uint64_t)B;
    ch.costs.phase_c.revealed_bits_recv += fbits * (uint64_t)B;
    for (size_t b = 0; b < B; ++b) {
        keys[b].finalCW =
            mod_add((Word)final_mine[b], (Word)final_theirs[b], p);
    }
    return true;
}

}  // namespace ringlpn_2pdpf
