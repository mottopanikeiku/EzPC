// Distributed (two-party) DPF key generation — the M1 host prototype.
//
// WHAT THIS PROVES. The proposal's component D1 (milestone M1) claims that two
// parties holding an *additively shared* point position (alpha = off0 + off1,
// each summand private) and a *multiplicatively shared* payload
// (beta = beta0 * beta1, each factor private) can generate the two halves of a
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
//   same run (same-consumer control), and a corrupted-key negative control
//   must FAIL evaluation. An omniscient regression model also reconstructs
//   the removed sign opening and confirms that, conditioned on party 0's
//   expanded leaf control bits, it selects a proper class containing alpha.
//   This catches reintroduction of the old leak; it is not a privacy proof.

#include "spfss_host.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

struct Costs {
    uint64_t string_ots = 0;      // 1-of-2 OTs on 128-bit strings
    uint64_t bit_triples = 0;     // Beaver bit triples (alpha-bit adder)
    uint64_t scalar_oles = 0;     // ideal Z_p OLE calls (payload)
    uint64_t opened_bits = 0;     // every bit either party reveals outside OT
    void reset() { *this = Costs{}; }
};

// ----- ideal functionalities (party-separated interfaces, counted) -----

struct Functionalities {
    uint64_t rng;  // functionality-internal randomness (triples, OLE masks)
    Costs *costs;

    // 1-of-2 string OT: sender supplies (m0, m1), receiver supplies choice.
    // Neither party sees the other's inputs; receiver gets m_choice.
    U128 ot(U128 m0, U128 m1, uint8_t choice) {
        costs->string_ots++;
        return choice ? m1 : m0;
    }

    // XOR-shared random bit triple a & b = c.
    struct BitTripleShare { uint8_t a, b, c; };
    void bit_triple(BitTripleShare &sh0, BitTripleShare &sh1) {
        costs->bit_triples++;
        uint64_t r = splitmix64(rng);
        uint8_t a = r & 1, b = (r >> 1) & 1;
        uint8_t c = (uint8_t)(a & b);
        sh0.a = (r >> 2) & 1; sh1.a = a ^ sh0.a;
        sh0.b = (r >> 3) & 1; sh1.b = b ^ sh0.b;
        sh0.c = (r >> 4) & 1; sh1.c = c ^ sh0.c;
    }

    // Ideal scalar OLE: gamma0 + gamma1 = x0 * x1 (mod p), gamma0 uniform.
    void ole(Word x0, Word x1, Word p, Word &gamma0, Word &gamma1) {
        costs->scalar_oles++;
        uint64_t lo = splitmix64(rng), hi = splitmix64(rng);
        gamma0 = (Word)((((__uint128_t)hi << 64) | lo) % p);
        gamma1 = mod_sub(mod_mul(x0, x1, p), gamma0, p);
    }
};

// Shared-bit AND via a Beaver bit triple. x, y are XOR-shared; returns
// XOR-shares of x & y. Opens d = x^a and e = y^b (2 bits, counted).
inline void shared_and(uint8_t x0, uint8_t x1, uint8_t y0, uint8_t y1,
                       Functionalities &F, uint8_t &z0, uint8_t &z1) {
    Functionalities::BitTripleShare t0, t1;
    F.bit_triple(t0, t1);
    uint8_t d = (uint8_t)((x0 ^ t0.a) ^ (x1 ^ t1.a));  // opened
    uint8_t e = (uint8_t)((y0 ^ t0.b) ^ (y1 ^ t1.b));  // opened
    F.costs->opened_bits += 2;
    z0 = (uint8_t)(((d & e) ^ (d & t0.b) ^ (e & t0.a) ^ t0.c) & 1);
    z1 = (uint8_t)(((d & t1.b) ^ (e & t1.a) ^ t1.c) & 1);
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
// "open" (both shares revealed; counted in costs.opened_bits).
bool distributed_dpf_gen(int log_domain, Word p,
                         PartyState &P0, PartyState &P1, Functionalities &F,
                         spfss_host::DPFKey &K0, spfss_host::DPFKey &K1) {
    const int L = log_domain;

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
            shared_and((uint8_t)(u ^ c0), c1, c0, (uint8_t)(v ^ c1), F, z0, z1);
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
        U128 w0 = F.ot(r1, r1 ^ Z1, a0);       // = r1 ^ a0*Z1
        // OT#2: P0 sender masks Z0 with fresh r0; P1 receiver with choice a1.
        U128 r0 = make_u128(splitmix64(P0.rng), splitmix64(P0.rng));
        U128 w1 = F.ot(r0, r0 ^ Z0, a1);       // = r0 ^ a1*Z0
        // Shares of a_i*Z, then of sCW; both parties open their sCW share.
        U128 sCW_sh0 = Sside[0][1] ^ (a0 ? Z0 : (U128)0) ^ w0 ^ r0;
        U128 sCW_sh1 = Sside[1][1] ^ (a1 ? Z1 : (U128)0) ^ w1 ^ r1;
        U128 sCW = sCW_sh0 ^ sCW_sh1;          // opened (public key material)
        F.costs->opened_bits += 2 * 128;

        // Flag CWs: linear — open shares directly.
        uint8_t tLCW = (uint8_t)(((Tside[0][0] ^ a0 ^ 1) ^ (Tside[1][0] ^ a1)) & 1);
        uint8_t tRCW = (uint8_t)(((Tside[0][1] ^ a0) ^ (Tside[1][1] ^ a1)) & 1);
        F.costs->opened_bits += 2 * 2;

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
    F.ole(P0.beta_factor, P1.beta_factor, p, gamma0, gamma1);
    const Word d0 = mod_sub(gamma0, A0, p);
    const Word d1 = mod_sub(gamma1, A1, p);
    const Word s0 = F0;
    const Word s1 = F1;

    Word cross01_0, cross01_1;
    Word cross10_0, cross10_1;
    F.ole(d0, s1, p, cross01_0, cross01_1);
    F.ole(s0, d1, p, cross10_0, cross10_1);
    const Word w0 = mod_add(
        mod_add(mod_mul(d0, s0, p), cross01_0, p), cross10_0, p);
    const Word w1 = mod_add(
        mod_add(mod_mul(d1, s1, p), cross01_1, p), cross10_1, p);

    // The old transcript opened s0 and s1, exposing their +/-1 sum. Marginal
    // independence of that sign from alpha is insufficient once conditioned
    // on a party's leaf control-bit vector: the sign selects that party's
    // alpha-containing control-bit class. Open only the standard public key
    // material finalCW, never d0, d1, s0, s1, or their sign.
    const Word finalCW = mod_add(w0, w1, p);
    F.costs->opened_bits += 2 * 62;
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
    if (args.log_domain < 2 || args.log_domain > 20 || args.trees < 1 ||
        (args.modulus_idx != 0 && args.modulus_idx != 1)) {
        std::fprintf(stderr, "unsupported parameters (log-domain 2..20, trees >= 1, modulus-idx 0|1)\n");
        return 2;
    }
    const Word p = args.modulus_idx == 0 ? kPrime62 : kPrime62Crt2;
    const int L = args.log_domain;
    const uint64_t half = 1ULL << (L - 1);

    Costs costs;
    Functionalities F{args.seed * 0x5851F42D4C957F2DULL + 0x1405ULL, &costs};
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
        // two trees deterministically cover both position and payload edges.
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
        } else {
            P0.off = splitmix64(harness_rng) % half;
            P1.off = splitmix64(harness_rng) % half;
            P0.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
            P1.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
        }
        uint64_t alpha = P0.off + P1.off;
        Word beta = mod_mul(P0.beta_factor, P1.beta_factor, p);

        bool ok = distributed_dpf_gen(L, p, P0, P1, F, K0, K1);
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

    // Negative control: a corrupted correction word must fail validation.
    bool negctrl_ok = false;
    {
        P0.off = splitmix64(harness_rng) % half;
        P1.off = splitmix64(harness_rng) % half;
        P0.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
        P1.beta_factor = 1 + splitmix64(harness_rng) % (p - 1);
        uint64_t alpha = P0.off + P1.off;
        Word beta = mod_mul(P0.beta_factor, P1.beta_factor, p);
        if (distributed_dpf_gen(L, p, P0, P1, F, K0, K1)) {
            K1.sCW[L / 2] ^= ((U128)1 << 33);
            negctrl_ok = !validate_pair(K0, K1, alpha, beta, p, e0, e1);
        }
    }

    const uint64_t trees = (uint64_t)args.trees;
    const bool old_sign_control_ok =
        old_sign_invariant_ok && old_sign_proper_subset_observed;
    if (args.csv_header) {
        std::printf("modulus,log_domain,trees,pass,fail,centralized_ref_pass,negctrl_expected_fail,"
                    "old_sign_opening_leak_control,string_ots_per_tree,bit_triples_per_tree,"
                    "scalar_oles_per_tree,opened_bits_per_tree,keygen_plus_eval_us_per_tree,"
                    "validation\n");
    }
    bool all_ok = fail == 0 && pass == args.trees && centralized_pass == 8 &&
                  negctrl_ok && old_sign_control_ok;
    std::printf("q62%s,%d,%d,%d,%d,%d,%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%s\n",
                args.modulus_idx == 0 ? "" : "b", L, args.trees, pass, fail,
                centralized_pass, negctrl_ok ? "yes" : "NO",
                old_sign_control_ok ? "yes" : "NO",
                (double)costs.string_ots / (trees + 1),
                (double)costs.bit_triples / (trees + 1),
                (double)costs.scalar_oles / (trees + 1),
                (double)costs.opened_bits / (trees + 1),
                total_us / trees,
                all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[distributed-dpf] L=%d trees=%d: %d/%d pass, centralized ref %d/8, "
                 "negative control %s, old-sign leak regression %s; per tree: "
                 "%.0f string-OTs, %.0f bit-triples, %.0f OLE, %.0f opened bits; "
                 "%.0f us/tree (incl. validation eval %.0f us)\n",
                 L, args.trees, pass, args.trees, centralized_pass,
                 negctrl_ok ? "failed as expected" : "DID NOT FAIL",
                 old_sign_control_ok ? "observed" : "NOT OBSERVED",
                 (double)costs.string_ots / (trees + 1),
                 (double)costs.bit_triples / (trees + 1),
                 (double)costs.scalar_oles / (trees + 1),
                 (double)costs.opened_bits / (trees + 1),
                 total_us / trees, eval_us / trees);
    return all_ok ? 0 : 1;
}
