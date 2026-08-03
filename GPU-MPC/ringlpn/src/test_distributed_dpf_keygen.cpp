// Distributed (two-party) DPF key generation — the M1 host prototype.
//
// WHAT THIS PROVES. The proposal's component D1 (milestone M1) claims that two
// parties holding an *additively shared* point position (alpha = off0 + off1,
// each summand private) and a *multiplicatively shared nonzero payload*
// (beta = beta0 * beta1, each factor in Z_p^* and private) can generate the two halves of a
// standard BGI-style DPF key without either party learning alpha, beta, or the
// other party's key half. This file implements that protocol, party-separated,
// on the host — and validates its output keys with the UNCHANGED existing
// evaluator spfss_host::dpfEvalAll (spfss_host.cpp is not modified by this
// artifact; it is the independent consumer, exactly as unmodified
// gpuMatmulBeaver is the consumer for the transcript artifacts).
//
// PROTOCOL (semi-honest). Per tree of depth L over domain 2^L:
//   Phase A (alpha bit-sharing): the walk consumes alpha one bit per level,
//     but alpha is shared arithmetically. A secure ripple adder over the
//     parties' XOR-shared summand bits produces XOR-shared bits of alpha:
//     L-1 AND gates (Beaver bit triples), carries c' = c ^ (u^c)(v^c).
//     The summands are the two parties' actual positions/regular-bucket
//     offsets in [0,2^(L-1)); their non-wrapping sum is the intended triangular
//     exponent distribution of the unreduced polynomial product, not a
//     uniform point sampler.
//   Phase B (level walk), for each level i:
//     - Each party locally expands ALL its current-level nodes with the PRG
//       and XORs the expansions into per-side aggregates S^L, S^R, T^L, T^R.
//       Off-path nodes are identical between the parties (DPF invariant), so
//       they cancel in S0^side ^ S1^side, leaving exactly the on-path
//       contribution: the raw material of the correction word
//       ("cancellation lemma").
//     - Seed CW = MUX by the secret bit a_i:
//         sCW = S^R_joint ^ a_i * (S^L_joint ^ S^R_joint),   S_joint = S0^S1.
//       The product (XOR-shared bit) x (XOR-shared 128-bit string) costs two
//       1-of-2 string OTs (one per direction, unoptimized; the literature's
//       1-OT/level variant halves this). The parties then open their sCW
//       shares — safe: each share is masked by the other party's fresh OT
//       randomness, and the opened value is public key material.
//     - Flag CWs are linear in local aggregates and the shared bit: opened
//       directly, no OT.
//     - Each party applies the opened CWs to its own nodes (where its control
//       bit is set) and advances one level. Work per party = one full-domain
//       expansion — the known cost shape of Doerner-shelat-style keygen.
//   Phase C (payload): signed leaf aggregates A_b (sum of convert(seed)) and
//     F_b (sum of flags) again cancel off-path, so
//     A0+A1 = convert(s0_alpha) - convert(s1_alpha) and F0+F1 = t0 - t1 = +/-1.
//     The first scalar OLE shares beta = beta0*beta1. Two directional OLEs
//     then multiply additive shares d0+d1 = beta-A0-A1 and s0+s1 = F0+F1:
//     local products plus cross-product shares give w0+w1 = (d0+d1)(s0+s1).
//     The parties open only finalCW = w0+w1, which is already public in each
//     standard output key.
//
//   - OT, bit triples, and the three scalar OLEs are ideal functionalities
//     here: party-separated interfaces with counted invocations, NOT
//     cryptographic transports. M1 proper replaces them with silent OT
//     (Ferret-class) and the pipeline's own OLE output (bootstrap:
//     self-sustaining iff 3*c^2*t^2 < n; see the proposal, eq. 1).
//   - The PRG is spfss_host's splitmix64-based expand (duplicated here
//     verbatim because spfss_host keeps it file-local and this artifact
//     refuses to modify the consumer it validates against; any drift is
//     caught instantly by dpfEvalAll validation).
//   - Host-only, single process (party separation is structural: parties are
//     structs, every cross-party value flows through a counted channel).
//     GPU level-synchronous batching and byte-identical GPU key format are
//     M1-proper work.
//   - Benchmark RNG is per-party seeded from the CLI seed (gap G3 applies to
//     the whole pipeline and is closed by D3, not here).
//
// VALIDATION GATE (this artifact exits non-zero if any check fails):
//   For every tree in every configuration: full-domain evaluation of the two
//   generated key halves via unchanged spfss_host::dpfEvalAll sums to
//   beta * [x == alpha] at every point of the domain. A centralized-keygen
//   reference (spfss_host::dpfGen) is validated by the same evaluator in the
//   same run (same-consumer control); independent corruptions of the root seed,
//   sCW, tLCW, tRCW, and finalCW must all FAIL evaluation; invalid point/payload
//   encodings must abort before consuming a correlation. An omniscient
//   regression model also reconstructs the removed sign opening and confirms
//   that, conditioned on party 0's expanded leaf control bits, it selects a
//   proper class containing alpha. This catches reintroduction of the old
//   leak; it is not a privacy proof. The gate separately verifies logical
//   openings and the raw payload of both parties' revealed shares. At 62-bit p
//   these are
//     2*(L-1) + 130*L + 62      logical opened bits, and
//     4*(L-1) + 260*L + 124     meaningful share bits opened.
//   It also verifies one fresh functionality-randomness draw per bit triple
//   and two per scalar OLE, plus consume-once correlation IDs with a duplicate
//   reuse negative control. None of these counters is measured network traffic.

#include "spfss_host.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_set>

namespace {

using Word = spfss_host::Word;
using U128 = spfss_host::U128;

constexpr Word kPrime62 = 4611686018326724609ULL;     // 2^62 - 6*2^24 + 1
constexpr Word kPrime62Crt2 = 4611686018309947393ULL; // 2^62 - 7*2^24 + 1

// ----- PRG / convert: duplicated verbatim from spfss_host.cpp (file-local
// there). dpfEvalAll validation guards against drift. -----

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

inline Word mod_add(Word a, Word b, Word m) {
    Word s = a + b;
    if (s >= m || s < a) s -= m;
    return s;
}

inline Word mod_sub(Word a, Word b, Word m) {
    return a >= b ? a - b : m - (b - a);
}

inline Word mod_mul(Word a, Word b, Word m) {
    return (Word)(((__uint128_t)a * b) % m);
}

inline Word convert_zp(U128 s, Word m) {
    Word lo = (Word)((uint64_t)s % m);
    Word hi = (Word)((uint64_t)(s >> 64) % m);
    return mod_add(lo, hi, m);
}

// ----- transcript / cost accounting -----

struct OpeningCosts {
    uint64_t logical_bits = 0;       // common values reconstructed
    uint64_t meaningful_share_bits = 0; // sum of both parties' meaningful widths
};

struct Costs {
    uint64_t string_ots = 0;      // 1-of-2 OTs on 128-bit strings
    uint64_t bit_triples = 0;     // Beaver bit triples (alpha-bit adder)
    uint64_t scalar_oles = 0;     // ideal Z_p OLE calls (payload)
    uint64_t functionality_random_words = 0;  // fresh triple/OLE mask draws
    OpeningCosts phase_a;
    OpeningCosts phase_b;
    OpeningCosts phase_c;

    uint64_t logical_opened_bits() const {
        return phase_a.logical_bits + phase_b.logical_bits + phase_c.logical_bits;
    }
    uint64_t meaningful_share_bits() const {
        return phase_a.meaningful_share_bits +
               phase_b.meaningful_share_bits +
               phase_c.meaningful_share_bits;
    }
    void reset() { *this = Costs{}; }
};

inline uint64_t field_encoding_bits(Word p) {
    uint64_t bits = 0;
    for (Word x = p - 1; x != 0; x >>= 1) ++bits;
    return bits;
}

inline uint64_t correlation_id(uint64_t tree_id, uint64_t phase,
                               uint64_t ordinal) {
    return (tree_id << 16) | (phase << 12) | ordinal;
}

// ----- ideal functionalities (party-separated interfaces, counted) -----

struct Functionalities {
    uint64_t rng;  // functionality-internal randomness (triples, OLE masks)
    Costs *costs;
    std::unordered_set<uint64_t> consumed_correlation_ids;

    bool consume_once(uint64_t id) {
        return consumed_correlation_ids.insert(id).second;
    }

    // 1-of-2 string OT: sender supplies (m0, m1), receiver supplies choice.
    // Neither party sees the other's inputs; receiver gets m_choice.
    bool ot(uint64_t id, U128 m0, U128 m1, uint8_t choice, U128 &out) {
        if (!consume_once(id)) return false;
        costs->string_ots++;
        out = choice ? m1 : m0;
        return true;
    }

    // XOR-shared random bit triple a & b = c.
    struct BitTripleShare { uint8_t a, b, c; };
    bool bit_triple(uint64_t id, BitTripleShare &sh0, BitTripleShare &sh1) {
        if (!consume_once(id)) return false;
        costs->bit_triples++;
        costs->functionality_random_words++;
        uint64_t r = splitmix64(rng);
        uint8_t a = r & 1, b = (r >> 1) & 1;
        uint8_t c = (uint8_t)(a & b);
        sh0.a = (r >> 2) & 1; sh1.a = a ^ sh0.a;
        sh0.b = (r >> 3) & 1; sh1.b = b ^ sh0.b;
        sh0.c = (r >> 4) & 1; sh1.c = c ^ sh0.c;
        return true;
    }

    // Ideal scalar OLE: gamma0 + gamma1 = x0 * x1 (mod p), gamma0 uniform.
    bool ole(uint64_t id, Word x0, Word x1, Word p,
             Word &gamma0, Word &gamma1) {
        if (!consume_once(id)) return false;
        costs->scalar_oles++;
        costs->functionality_random_words += 2;
        uint64_t lo = splitmix64(rng), hi = splitmix64(rng);
        gamma0 = (Word)((((__uint128_t)hi << 64) | lo) % p);
        gamma1 = mod_sub(mod_mul(x0, x1, p), gamma0, p);
        return true;
    }

    uint8_t open_phase_a_bit(uint8_t sh0, uint8_t sh1) {
        costs->phase_a.logical_bits++;
        costs->phase_a.meaningful_share_bits += 2;
        return (uint8_t)((sh0 ^ sh1) & 1);
    }

    U128 open_phase_b_seed_cw(U128 sh0, U128 sh1) {
        costs->phase_b.logical_bits += 128;
        costs->phase_b.meaningful_share_bits += 2 * 128;
        return sh0 ^ sh1;
    }

    uint8_t open_phase_b_flag_cw(uint8_t sh0, uint8_t sh1) {
        costs->phase_b.logical_bits++;
        costs->phase_b.meaningful_share_bits += 2;
        return (uint8_t)((sh0 ^ sh1) & 1);
    }

    Word open_phase_c_final_cw(Word sh0, Word sh1, Word p) {
        const uint64_t bits = field_encoding_bits(p);
        costs->phase_c.logical_bits += bits;
        costs->phase_c.meaningful_share_bits += 2 * bits;
        return mod_add(sh0, sh1, p);
    }
};

// Shared-bit AND via a Beaver bit triple. x, y are XOR-shared; returns
// XOR-shares of x & y. Opens d = x^a and e = y^b (2 bits, counted).
inline bool shared_and(uint64_t correlation_id,
                       uint8_t x0, uint8_t x1, uint8_t y0, uint8_t y1,
                       Functionalities &F, uint8_t &z0, uint8_t &z1) {
    Functionalities::BitTripleShare t0, t1;
    if (!F.bit_triple(correlation_id, t0, t1)) return false;
    uint8_t d = F.open_phase_a_bit((uint8_t)(x0 ^ t0.a),
                                   (uint8_t)(x1 ^ t1.a));
    uint8_t e = F.open_phase_a_bit((uint8_t)(y0 ^ t0.b),
                                   (uint8_t)(y1 ^ t1.b));
    z0 = (uint8_t)(((d & e) ^ (d & t0.b) ^ (e & t0.a) ^ t0.c) & 1);
    z1 = (uint8_t)(((d & t1.b) ^ (e & t1.a) ^ t1.c) & 1);
    return true;
}

// ----- per-party state -----

struct Node {
    U128 s;
    uint8_t t;
};

struct PartyState {
    uint64_t rng;                    // party-private randomness
    uint64_t off;                    // private position summand (< 2^(L-1))
    Word beta_factor;                // private payload factor
    std::vector<uint8_t> abit;       // XOR-share of alpha's bits (LSB-first)
    std::vector<Node> nodes, next;   // current / next tree level
    std::vector<U128> expL, expR;    // raw expansions of current level
    std::vector<uint8_t> expTL, expTR;
};

// ----- the distributed keygen protocol -----

// Generates one DPF key pair in spfss_host::DPFKey format. Every value that
// crosses between P0 and P1 goes through F (OT/triples/OLE) or is an explicit
// opening whose logical value and two revealed shares are counted separately.
bool distributed_dpf_gen(int log_domain, Word p, uint64_t tree_id,
                         PartyState &P0, PartyState &P1, Functionalities &F,
                         spfss_host::DPFKey &K0, spfss_host::DPFKey &K1) {
    const int L = log_domain;
    const uint64_t half_domain = 1ULL << (L - 1);
    if (P0.off >= half_domain || P1.off >= half_domain ||
        P0.beta_factor == 0 || P0.beta_factor >= p ||
        P1.beta_factor == 0 || P1.beta_factor >= p) {
        return false;
    }

    // --- Phase A: XOR-shared bits of alpha = off0 + off1 via secure adder ---
    P0.abit.assign(L, 0);
    P1.abit.assign(L, 0);
    uint8_t c0 = 0, c1 = 0;  // carry shares
    for (int j = 0; j < L; ++j) {
        uint8_t u = (uint8_t)((P0.off >> j) & 1);  // P0-private
        uint8_t v = (uint8_t)((P1.off >> j) & 1);  // P1-private
        P0.abit[j] = (uint8_t)(u ^ c0);
        P1.abit[j] = (uint8_t)(v ^ c1);
        if (j + 1 < L) {
            // carry' = c ^ (u^c)(v^c); shares of x=u^c: (u^c0, c1); y=v^c: (c0, v^c1)
            uint8_t z0, z1;
            if (!shared_and(correlation_id(tree_id, 1, j),
                            (uint8_t)(u ^ c0), c1, c0,
                            (uint8_t)(v ^ c1), F, z0, z1)) return false;
            c0 = (uint8_t)(z0 ^ c0);
            c1 = (uint8_t)(z1 ^ c1);
        }
    }

    // --- root seeds: each party samples its own ---
    U128 seed0 = make_u128(splitmix64(P0.rng), splitmix64(P0.rng));
    U128 seed1 = make_u128(splitmix64(P1.rng), splitmix64(P1.rng));
    K0.log_domain = K1.log_domain = L;
    K0.modulus = K1.modulus = p;
    K0.seed = seed0; K0.t0 = 0;
    K1.seed = seed1; K1.t0 = 1;
    K0.sCW.assign(L, 0); K1.sCW.assign(L, 0);
    K0.tLCW.assign(L, 0); K1.tLCW.assign(L, 0);
    K0.tRCW.assign(L, 0); K1.tRCW.assign(L, 0);

    P0.nodes.assign(1, {seed0, 0});
    P1.nodes.assign(1, {seed1, 1});

    // --- Phase B: level-by-level walk ---
    for (int i = 0; i < L; ++i) {
        const int bi = L - 1 - i;  // MSB-first bit of alpha at this level
        PartyState *Ps[2] = {&P0, &P1};
        U128 Sside[2][2];          // [party][0=L,1=R] seed aggregates
        uint8_t Tside[2][2];       // flag aggregates
        for (int b = 0; b < 2; ++b) {
            PartyState &P = *Ps[b];
            size_t nn = P.nodes.size();
            P.expL.resize(nn); P.expR.resize(nn);
            P.expTL.resize(nn); P.expTR.resize(nn);
            U128 aggL = 0, aggR = 0;
            uint8_t taggL = 0, taggR = 0;
            for (size_t k = 0; k < nn; ++k) {
                prg_expand(P.nodes[k].s, P.expL[k], P.expTL[k], P.expR[k], P.expTR[k]);
                aggL ^= P.expL[k]; aggR ^= P.expR[k];
                taggL ^= P.expTL[k]; taggR ^= P.expTR[k];
            }
            Sside[b][0] = aggL; Sside[b][1] = aggR;
            Tside[b][0] = taggL; Tside[b][1] = taggR;
        }

        // Seed CW: sCW = S^R_joint ^ a_i * Z_joint, Z = S^L ^ S^R.
        // a_i * Z with XOR-shared a_i, XOR-shared Z: two string OTs.
        U128 Z0 = Sside[0][0] ^ Sside[0][1];
        U128 Z1 = Sside[1][0] ^ Sside[1][1];
        uint8_t a0 = P0.abit[bi], a1 = P1.abit[bi];
        // OT#1: P1 sender masks Z1 with fresh r1; P0 receiver with choice a0.
        U128 r1 = make_u128(splitmix64(P1.rng), splitmix64(P1.rng));
        U128 w0;
        if (!F.ot(correlation_id(tree_id, 2, 2 * i),
                  r1, r1 ^ Z1, a0, w0)) return false;
        // OT#2: P0 sender masks Z0 with fresh r0; P1 receiver with choice a1.
        U128 r0 = make_u128(splitmix64(P0.rng), splitmix64(P0.rng));
        U128 w1;
        if (!F.ot(correlation_id(tree_id, 2, 2 * i + 1),
                  r0, r0 ^ Z0, a1, w1)) return false;
        // Shares of a_i*Z, then of sCW; both parties open their sCW share.
        U128 sCW_sh0 = Sside[0][1] ^ (a0 ? Z0 : (U128)0) ^ w0 ^ r0;
        U128 sCW_sh1 = Sside[1][1] ^ (a1 ? Z1 : (U128)0) ^ w1 ^ r1;
        U128 sCW = F.open_phase_b_seed_cw(sCW_sh0, sCW_sh1);

        // Flag CWs: linear shares, opened as common key material.
        uint8_t tLCW_sh0 = (uint8_t)((Tside[0][0] ^ a0 ^ 1) & 1);
        uint8_t tLCW_sh1 = (uint8_t)((Tside[1][0] ^ a1) & 1);
        uint8_t tRCW_sh0 = (uint8_t)((Tside[0][1] ^ a0) & 1);
        uint8_t tRCW_sh1 = (uint8_t)((Tside[1][1] ^ a1) & 1);
        uint8_t tLCW = F.open_phase_b_flag_cw(tLCW_sh0, tLCW_sh1);
        uint8_t tRCW = F.open_phase_b_flag_cw(tRCW_sh0, tRCW_sh1);

        K0.sCW[i] = sCW; K1.sCW[i] = sCW;
        K0.tLCW[i] = tLCW; K1.tLCW[i] = tLCW;
        K0.tRCW[i] = tRCW; K1.tRCW[i] = tRCW;

        // Each party advances its own level with the now-public CWs.
        for (int b = 0; b < 2; ++b) {
            PartyState &P = *Ps[b];
            size_t nn = P.nodes.size();
            P.next.resize(nn * 2);
            for (size_t k = 0; k < nn; ++k) {
                uint8_t t = P.nodes[k].t;
                U128 sL = P.expL[k] ^ (t ? sCW : (U128)0);
                U128 sR = P.expR[k] ^ (t ? sCW : (U128)0);
                uint8_t tL = (uint8_t)((P.expTL[k] ^ (t ? tLCW : 0)) & 1);
                uint8_t tR = (uint8_t)((P.expTR[k] ^ (t ? tRCW : 0)) & 1);
                P.next[2 * k] = {sL, tL};
                P.next[2 * k + 1] = {sR, tR};
            }
            P.nodes.swap(P.next);
        }
    }

    // --- Phase C: payload correction word from three scalar OLEs ---
    Word A0 = 0, F0 = 0;
    for (const Node &nd : P0.nodes) {
        A0 = mod_add(A0, convert_zp(nd.s, p), p);
        F0 = mod_add(F0, nd.t, p);
    }
    Word A1 = 0, F1 = 0;
    for (const Node &nd : P1.nodes) {
        A1 = mod_sub(A1, convert_zp(nd.s, p), p);
        F1 = mod_sub(F1, nd.t, p);
    }

    Word gamma0, gamma1;
    if (!F.ole(correlation_id(tree_id, 3, 0),
               P0.beta_factor, P1.beta_factor, p, gamma0, gamma1)) return false;
    const Word d0 = mod_sub(gamma0, A0, p);
    const Word d1 = mod_sub(gamma1, A1, p);
    const Word s0 = F0;
    const Word s1 = F1;

    Word cross01_0, cross01_1;
    Word cross10_0, cross10_1;
    if (!F.ole(correlation_id(tree_id, 3, 1),
               d0, s1, p, cross01_0, cross01_1)) return false;
    if (!F.ole(correlation_id(tree_id, 3, 2),
               s0, d1, p, cross10_0, cross10_1)) return false;
    const Word w0 = mod_add(
        mod_add(mod_mul(d0, s0, p), cross01_0, p), cross10_0, p);
    const Word w1 = mod_add(
        mod_add(mod_mul(d1, s1, p), cross01_1, p), cross10_1, p);

    // The old transcript opened s0 and s1, exposing their +/-1 sum. Marginal
    // independence of that sign from alpha is insufficient once conditioned
    // on a party's leaf control-bit vector: the sign selects that party's
    // alpha-containing control-bit class. Open only the standard public key
    // material finalCW, never d0, d1, s0, s1, or their sign.
    const Word finalCW = F.open_phase_c_final_cw(w0, w1, p);
    K0.finalCW = finalCW;
    K1.finalCW = finalCW;
    return true;
}

// ----- validation -----

bool validate_pair(const spfss_host::DPFKey &K0, const spfss_host::DPFKey &K1,
                   uint64_t alpha, Word beta, Word p,
                   std::vector<Word> &e0, std::vector<Word> &e1) {
    spfss_host::dpfEvalAll(0, K0, e0);
    spfss_host::dpfEvalAll(1, K1, e1);
    uint64_t domain = 1ULL << K0.log_domain;
    for (uint64_t x = 0; x < domain; ++x) {
        Word sum = mod_add(e0[x], e1[x], p);
        Word want = (x == alpha) ? beta : 0;
        if (sum != want) return false;
    }
    return true;
}

struct OldSignLeakObservation {
    bool alpha_in_selected_class;
    bool proper_subset;
};

// Omniscient harness model of the removed transcript. This deliberately
// reconstructs the former sign outside distributed_dpf_gen, where the test
// already knows both parties' state, to demonstrate its distinguishing power.
OldSignLeakObservation observe_old_sign_opening_leak(
    const PartyState &P0, const PartyState &P1, uint64_t alpha, Word p) {
    if (P0.nodes.size() != P1.nodes.size() || alpha >= P0.nodes.size()) {
        return {false, false};
    }

    Word f0 = 0;
    Word f1 = 0;
    for (const Node &nd : P0.nodes) f0 = mod_add(f0, nd.t, p);
    for (const Node &nd : P1.nodes) f1 = mod_sub(f1, nd.t, p);
    const Word sigma = mod_add(f0, f1, p);
    if (sigma != 1 && sigma != p - 1) return {false, false};

    const uint8_t selected_bit = sigma == 1 ? 1 : 0;
    const bool alpha_in_selected_class =
        (P0.nodes[alpha].t & 1) == selected_bit;
    bool leaf_outside_selected_class = false;
    for (const Node &nd : P0.nodes) {
        if ((nd.t & 1) != selected_bit) {
            leaf_outside_selected_class = true;
            break;
        }
    }
    return {
        alpha_in_selected_class,
        alpha_in_selected_class && leaf_outside_selected_class,
    };
}

struct Args {
    int log_domain = 14;
    int trees = 256;
    int modulus_idx = 0;
    uint64_t seed = 1;
    bool csv_header = false;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [--log-domain L] [--trees N] [--modulus-idx 0|1] "
                 "[--seed S] [--csv-header]\n", argv0);
}

} // namespace

int main(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--log-domain") args.log_domain = std::atoi(next("--log-domain"));
        else if (a == "--trees") args.trees = std::atoi(next("--trees"));
        else if (a == "--modulus-idx") args.modulus_idx = std::atoi(next("--modulus-idx"));
        else if (a == "--seed") args.seed = std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--csv-header") args.csv_header = true;
        else { usage(argv[0]); return 2; }
    }
    if (args.log_domain < 2 || args.log_domain > 20 || args.trees < 6 ||
        (args.modulus_idx != 0 && args.modulus_idx != 1)) {
        std::fprintf(stderr, "unsupported parameters (log-domain 2..20, trees >= 6, modulus-idx 0|1)\n");
        return 2;
    }
    const Word p = args.modulus_idx == 0 ? kPrime62 : kPrime62Crt2;
    const int L = args.log_domain;
    const uint64_t half = 1ULL << (L - 1);

    Costs costs;
    Functionalities F{
        args.seed * 0x5851F42D4C957F2DULL + 0x1405ULL, &costs, {}};
    PartyState P0, P1;
    P0.rng = args.seed * 0x9E3779B97F4A7C15ULL + 1;
    P1.rng = args.seed * 0xC2B2AE3D27D4EB4FULL + 2;
    uint64_t harness_rng = args.seed * 0xD6E8FEB86659FD93ULL + 3;

    std::vector<Word> e0, e1;
    int pass = 0, fail = 0;
    int centralized_pass = 0;
    bool old_sign_invariant_ok = true;
    bool old_sign_proper_subset_observed = false;
    uint64_t centralized_rng = args.seed ^ 0xFEEDFACECAFEBEEFULL;

    auto t_start = std::chrono::steady_clock::now();
    double eval_us = 0.0;
    spfss_host::DPFKey K0, K1;
    for (int tr = 0; tr < args.trees; ++tr) {
        // Private inputs. The harness knows both sides (it must, to check the
        // point function); the protocol code above never mixes them. The first
        // six trees deterministically cover point endpoints, asymmetric
        // decompositions, carry propagation, and nonzero payload-factor edges.
        if (tr == 0) {
            P0.off = 0;
            P1.off = 0;
            P0.beta_factor = 1;
            P1.beta_factor = p - 1;
        } else if (tr == 1) {
            P0.off = half - 1;
            P1.off = half - 1;
            P0.beta_factor = p - 1;
            P1.beta_factor = p - 1;
        } else if (tr == 2) {
            P0.off = half - 1;
            P1.off = 0;
            P0.beta_factor = p - 1;
            P1.beta_factor = 1;
        } else if (tr == 3) {
            P0.off = 0;
            P1.off = half - 1;
            P0.beta_factor = 1;
            P1.beta_factor = 1;
        } else if (tr == 4) {
            P0.off = half - 1;
            P1.off = 1;
            P0.beta_factor = (p - 1) / 2;
            P1.beta_factor = 2;
        } else if (tr == 5) {
            P0.off = 1;
            P1.off = half - 1;
            P0.beta_factor = 2;
            P1.beta_factor = (p + 1) / 2;
        } else {
            P0.off = splitmix64(harness_rng) % half;
            P1.off = splitmix64(harness_rng) % half;
            P0.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
            P1.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
        }
        uint64_t alpha = P0.off + P1.off;
        Word beta = mod_mul(P0.beta_factor, P1.beta_factor, p);

        bool ok = distributed_dpf_gen(L, p, (uint64_t)tr, P0, P1, F, K0, K1);
        if (ok) {
            const OldSignLeakObservation old_sign =
                observe_old_sign_opening_leak(P0, P1, alpha, p);
            old_sign_invariant_ok &=
                old_sign.alpha_in_selected_class;
            old_sign_proper_subset_observed |= old_sign.proper_subset;
        } else {
            old_sign_invariant_ok = false;
        }
        auto t_eval = std::chrono::steady_clock::now();
        ok = ok && validate_pair(K0, K1, alpha, beta, p, e0, e1);
        eval_us += std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - t_eval).count();
        if (ok) ++pass; else ++fail;
        // Same-consumer control: centralized keygen through the same evaluator.
        if (tr < 8) {
            spfss_host::DPFKey C0, C1;
            spfss_host::dpfGen(alpha, L, beta, p, centralized_rng, C0, C1);
            if (validate_pair(C0, C1, alpha, beta, p, e0, e1)) ++centralized_pass;
        }
    }
    double total_us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t_start).count();

    // Negative controls: corrupt the root seed and every public
    // correction-word class independently; each mutation must fail
    // full-domain validation.
    int corruption_controls_passed = 0;
    constexpr int kCorruptionControls = 5;
    {
        P0.off = splitmix64(harness_rng) % half;
        P1.off = splitmix64(harness_rng) % half;
        P0.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
        P1.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
        uint64_t alpha = P0.off + P1.off;
        Word beta = mod_mul(P0.beta_factor, P1.beta_factor, p);
        if (distributed_dpf_gen(L, p, (uint64_t)args.trees,
                                P0, P1, F, K0, K1)) {
            auto corruption_fails = [&](spfss_host::DPFKey corrupted) {
                return !validate_pair(K0, corrupted, alpha, beta, p, e0, e1);
            };
            auto corrupted = K1;
            corrupted.seed ^= ((U128)1 << 127);
            corruption_controls_passed += corruption_fails(corrupted);
            corrupted = K1;
            corrupted.sCW[L / 2] ^= ((U128)1 << 33);
            corruption_controls_passed += corruption_fails(corrupted);
            corrupted = K1;
            corrupted.tLCW[L / 2] ^= 1;
            corruption_controls_passed += corruption_fails(corrupted);
            corrupted = K1;
            corrupted.tRCW[L / 2] ^= 1;
            corruption_controls_passed += corruption_fails(corrupted);
            corrupted = K1;
            corrupted.finalCW = mod_add(corrupted.finalCW, 1, p);
            corruption_controls_passed += corruption_fails(corrupted);
        }
    }

    // Invalid private encodings must abort before consuming any ideal
    // correlation or emitting a partial key.
    int invalid_inputs_rejected = 0;
    constexpr int kInvalidInputControls = 6;
    auto rejects_invalid = [&](uint64_t control_id,
                               uint64_t off0, uint64_t off1,
                               Word beta0, Word beta1) {
        const uint64_t random_words_before = costs.functionality_random_words;
        const size_t ids_before = F.consumed_correlation_ids.size();
        PartyState I0, I1;
        I0.off = off0; I0.beta_factor = beta0; I0.rng = 1;
        I1.off = off1; I1.beta_factor = beta1; I1.rng = 2;
        spfss_host::DPFKey IKey0, IKey1;
        const bool rejected = !distributed_dpf_gen(
            L, p, control_id, I0, I1, F, IKey0, IKey1);
        return rejected &&
               costs.functionality_random_words == random_words_before &&
               F.consumed_correlation_ids.size() == ids_before;
    };
    const uint64_t invalid_id_base = (uint64_t)args.trees + 1;
    invalid_inputs_rejected += rejects_invalid(invalid_id_base, half, 0, 1, 1);
    invalid_inputs_rejected += rejects_invalid(invalid_id_base + 1, 0, half, 1, 1);
    invalid_inputs_rejected += rejects_invalid(invalid_id_base + 2, 0, 0, 0, 1);
    invalid_inputs_rejected += rejects_invalid(invalid_id_base + 3, 0, 0, 1, 0);
    invalid_inputs_rejected += rejects_invalid(invalid_id_base + 4, 0, 0, p, 1);
    invalid_inputs_rejected += rejects_invalid(invalid_id_base + 5, 0, 0, 1, p);

    Functionalities::BitTripleShare duplicate0, duplicate1;
    const uint64_t random_words_before_reuse =
        costs.functionality_random_words;
    const bool correlation_reuse_control_ok =
        !F.bit_triple(correlation_id(0, 1, 0), duplicate0, duplicate1) &&
        costs.functionality_random_words == random_words_before_reuse;

    const uint64_t trees = (uint64_t)args.trees;
    const uint64_t invocations = (uint64_t)args.trees + 1;  // trees + one control-key generation
    const uint64_t field_bits = field_encoding_bits(p);
    const int centralized_expected = std::min(args.trees, 8);
    const bool old_sign_control_ok =
        old_sign_invariant_ok && old_sign_proper_subset_observed;
    const bool transcript_accounting_ok =
        costs.string_ots == invocations * 2 * (uint64_t)L &&
        costs.bit_triples == invocations * (uint64_t)(L - 1) &&
        costs.scalar_oles == invocations * 3 &&
        costs.phase_a.logical_bits == invocations * 2 * (uint64_t)(L - 1) &&
        costs.phase_a.meaningful_share_bits ==
            invocations * 4 * (uint64_t)(L - 1) &&
        costs.phase_b.logical_bits == invocations * 130 * (uint64_t)L &&
        costs.phase_b.meaningful_share_bits ==
            invocations * 260 * (uint64_t)L &&
        costs.phase_c.logical_bits == invocations * field_bits &&
        costs.phase_c.meaningful_share_bits == invocations * 2 * field_bits;
    const bool ideal_mask_draw_accounting_ok =
        costs.functionality_random_words ==
        costs.bit_triples + 2 * costs.scalar_oles;
    auto per_tree = [invocations](uint64_t value) {
        return (double)value / (double)invocations;
    };
    if (args.csv_header) {
        std::printf("modulus,log_domain,trees,pass,fail,centralized_ref_pass,negctrl_expected_fail,"
                    "corruption_controls,invalid_inputs_rejected,"
                    "old_sign_opening_leak_control,string_ots_per_tree,bit_triples_per_tree,"
                    "scalar_oles_per_tree,phase_a_logical_opened_bits_per_tree,"
                    "phase_a_meaningful_share_bits_per_tree,"
                    "phase_b_logical_opened_bits_per_tree,"
                    "phase_b_meaningful_share_bits_per_tree,"
                    "phase_c_logical_opened_bits_per_tree,"
                    "phase_c_meaningful_share_bits_per_tree,"
                    "logical_opened_bits_per_tree,meaningful_share_bits_per_tree,"
                    "transcript_accounting,"
                    "ideal_mask_draw_accounting,correlation_reuse_control,"
                    "keygen_plus_eval_us_per_tree,validation\n");
    }
    const bool negctrl_ok =
        corruption_controls_passed == kCorruptionControls;
    const bool invalidctrl_ok =
        invalid_inputs_rejected == kInvalidInputControls;
    bool all_ok = fail == 0 && pass == args.trees &&
                  centralized_pass == centralized_expected &&
                  negctrl_ok && invalidctrl_ok && old_sign_control_ok &&
                  transcript_accounting_ok && ideal_mask_draw_accounting_ok &&
                  correlation_reuse_control_ok;
    std::printf("q62%s,%d,%d,%d,%d,%d,%s,%d/%d,%d/%d,%s,"
                "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s,%s,%s,%.1f,%s\n",
                args.modulus_idx == 0 ? "" : "b", L, args.trees, pass, fail,
                centralized_pass, negctrl_ok ? "yes" : "NO",
                corruption_controls_passed, kCorruptionControls,
                invalid_inputs_rejected, kInvalidInputControls,
                old_sign_control_ok ? "yes" : "NO",
                per_tree(costs.string_ots),
                per_tree(costs.bit_triples),
                per_tree(costs.scalar_oles),
                per_tree(costs.phase_a.logical_bits),
                per_tree(costs.phase_a.meaningful_share_bits),
                per_tree(costs.phase_b.logical_bits),
                per_tree(costs.phase_b.meaningful_share_bits),
                per_tree(costs.phase_c.logical_bits),
                per_tree(costs.phase_c.meaningful_share_bits),
                per_tree(costs.logical_opened_bits()),
                per_tree(costs.meaningful_share_bits()),
                transcript_accounting_ok ? "pass" : "FAIL",
                ideal_mask_draw_accounting_ok ? "pass" : "FAIL",
                correlation_reuse_control_ok ? "pass" : "FAIL",
                total_us / trees,
                all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[distributed-dpf] L=%d trees=%d: %d/%d pass, centralized ref %d/%d, "
                 "corruption controls %d/%d, invalid-input controls %d/%d, "
                 "old-sign leak regression %s, transcript accounting %s, "
                 "ideal-mask-draw accounting %s, correlation-reuse control %s; per tree: "
                 "%.0f string-OTs, %.0f bit-triples, %.0f OLE, "
                 "%.0f logical-open bits, %.0f meaningful-share bits; "
                 "%.0f us/tree (incl. validation eval %.0f us)\n",
                 L, args.trees, pass, args.trees,
                 centralized_pass, centralized_expected,
                 corruption_controls_passed, kCorruptionControls,
                 invalid_inputs_rejected, kInvalidInputControls,
                 old_sign_control_ok ? "observed" : "NOT OBSERVED",
                 transcript_accounting_ok ? "pass" : "FAIL",
                 ideal_mask_draw_accounting_ok ? "pass" : "FAIL",
                 correlation_reuse_control_ok ? "pass" : "FAIL",
                 per_tree(costs.string_ots),
                 per_tree(costs.bit_triples),
                 per_tree(costs.scalar_oles),
                 per_tree(costs.logical_opened_bits()),
                 per_tree(costs.meaningful_share_bits()),
                 total_us / trees, eval_us / trees);
    return all_ok ? 0 : 1;
}
