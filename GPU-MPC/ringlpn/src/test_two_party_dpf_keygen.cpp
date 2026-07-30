// Two-PROCESS distributed DPF key generation over real sockets and real OT,
// with LEVEL-SYNCHRONOUS BATCHING across trees.
//
// This is the transport realization of the protocol whose logic was frozen in
// results/reports/dealerless_orca_fc_security_contract_2026_07_29.md and
// prototyped with ideal functionalities in test_distributed_dpf_keygen.cpp.
// Differences from that prototype, all in the direction of less idealization:
//
//   * two OS processes, two TCP sockets, no shared memory or shared seed;
//   * real 1-of-2 OT (IKNP extension over Naor-Pinkas base OTs, unmodified
//     SCI code) instead of an ideal OT interface;
//   * real Gilboa Z_p OLE and real OT-based boolean AND triples instead of
//     ideal correlation oracles;
//   * party root seeds from the OS CSPRNG instead of a benchmark seed;
//   * measured wire bytes and direction switches per party, which the
//     single-process prototype could not report;
//   * all trees in a batch advance together, so the number of communication
//     stages depends on the tree DEPTH only, not on the batch size. Per-tree
//     correlation and opening counts are unchanged by batching.
//
// UNCHANGED and still NOT claimed: the DPF expansion PRG is spfss_host's
// non-cryptographic splitmix64 (the independent consumer dpfEvalAll is
// unmodified), so no 128-bit DPF-security claim is made here. Obligations
// D-SEED / P-RNG / P-KEY in the contract remain open.
//
// Each party writes only its OWN key file. Correctness is checked afterwards by
// the separate, explicitly TEST-ONLY checker test_two_party_dpf_validate, which
// reads both key files offline; no party ever reads the other party's state
// during the protocol.

#include "dpf_key_io.h"
#include "spfss_host.h"
#include "two_party_ot.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

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

// ----- the batched party-local protocol -------------------------------------
//
// Every cross-party value goes through OT or through an explicitly counted
// opening. Trees are processed level-synchronously: one OT batch and one
// opening per level for the whole batch.

bool two_party_dpf_gen_batch(int party, int log_domain, Word p,
                             const std::vector<uint64_t> &offs,
                             const std::vector<Word> &beta_factors,
                             PartyChannel &ch, PartyRandom &rng,
                             std::vector<spfss_host::DPFKey> &keys) {
    const int L = log_domain;
    const size_t B = offs.size();
    if (B == 0 || beta_factors.size() != B) return false;
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

    // --- root seeds: this party's own OS-CSPRNG draws ------------------------
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
                prg_expand(nodes[b][k].s, expL[k], expTL[k], expR[k], expTR[k]);
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
                prg_expand(nodes[b][k].s, sL, tl, sR, tr);
                const uint8_t t = nodes[b][k].t;
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

// ----- primitive self-tests (TEST-ONLY: they open private values) -----------

bool selftest_primitives(Word p, int rounds, PartyChannel &ch, PartyRandom &rng,
                         int &triple_fail, int &ole_fail) {
    triple_fail = 0;
    ole_fail = 0;
    std::vector<BitTriple> triples;
    ringlpn_2pc::generate_bit_triples(ch, rounds, rng, triples);
    std::vector<uint8_t> mine(3 * (size_t)rounds), theirs(3 * (size_t)rounds);
    for (int i = 0; i < rounds; ++i) {
        mine[3 * (size_t)i] = triples[(size_t)i].a;
        mine[3 * (size_t)i + 1] = triples[(size_t)i].b;
        mine[3 * (size_t)i + 2] = triples[(size_t)i].c;
    }
    ch.exchange_bytes(mine.data(), theirs.data(), mine.size());
    for (int i = 0; i < rounds; ++i) {
        const uint8_t a = (uint8_t)(mine[3 * (size_t)i] ^ theirs[3 * (size_t)i]);
        const uint8_t b =
            (uint8_t)(mine[3 * (size_t)i + 1] ^ theirs[3 * (size_t)i + 1]);
        const uint8_t c =
            (uint8_t)(mine[3 * (size_t)i + 2] ^ theirs[3 * (size_t)i + 2]);
        if (((a & b) & 1) != (c & 1)) ++triple_fail;
    }

    std::vector<Word> x((size_t)rounds);
    for (int i = 0; i < rounds; ++i) x[(size_t)i] = rng.field(p);
    const std::vector<Word> share =
        ringlpn_2pc::ole_batch_p0_sender(ch, x, p, rng);
    std::vector<uint64_t> mine_f(2 * (size_t)rounds), theirs_f(2 * (size_t)rounds);
    for (int i = 0; i < rounds; ++i) {
        mine_f[2 * (size_t)i] = (uint64_t)x[(size_t)i];
        mine_f[2 * (size_t)i + 1] = (uint64_t)share[(size_t)i];
    }
    ch.exchange_bytes(reinterpret_cast<const uint8_t *>(mine_f.data()),
                      reinterpret_cast<uint8_t *>(theirs_f.data()),
                      mine_f.size() * sizeof(uint64_t));
    for (int i = 0; i < rounds; ++i) {
        const Word prod =
            mod_mul(x[(size_t)i], (Word)theirs_f[2 * (size_t)i], p);
        const Word sum = mod_add(share[(size_t)i],
                                 (Word)theirs_f[2 * (size_t)i + 1], p);
        if (prod != sum) ++ole_fail;
    }
    return triple_fail == 0 && ole_fail == 0;
}

struct Args {
    int party = 0;
    std::string host = "127.0.0.1";
    int port = 42400;
    int log_domain = 8;
    int trees = 4;
    int modulus_idx = 0;
    int selftest = 0;
    uint64_t input_seed = 1;
    std::string out_prefix = "two_party_dpf";
    bool csv_header = false;
};

Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](void) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (k == "--party") a.party = std::atoi(next().c_str());
        else if (k == "--host") a.host = next();
        else if (k == "--port") a.port = std::atoi(next().c_str());
        else if (k == "--log-domain") a.log_domain = std::atoi(next().c_str());
        else if (k == "--trees") a.trees = std::atoi(next().c_str());
        else if (k == "--modulus-idx") a.modulus_idx = std::atoi(next().c_str());
        else if (k == "--selftest") a.selftest = std::atoi(next().c_str());
        else if (k == "--input-seed") a.input_seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (k == "--out-prefix") a.out_prefix = next();
        else if (k == "--csv-header") a.csv_header = true;
        else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.party != 0 && a.party != 1) {
        std::fprintf(stderr, "--party must be 0 or 1\n");
        std::exit(2);
    }
    if (a.log_domain < 2 || a.log_domain > 20) {
        std::fprintf(stderr, "--log-domain out of range\n");
        std::exit(2);
    }
    if (a.trees < 1) {
        std::fprintf(stderr, "--trees must be >= 1\n");
        std::exit(2);
    }
    return a;
}

}  // namespace

int main(int argc, char **argv) {
    const Args args = parse_args(argc, argv);
    const Word p = args.modulus_idx == 0 ? kPrime62 : kPrime62Crt2;
    const int L = args.log_domain;
    const int field_bits = ringlpn_2pc::field_bits(p);
    const uint64_t half_domain = 1ULL << (L - 1);
    const size_t B = (size_t)args.trees;

    PartyChannel ch(args.party, args.host, args.port);
    PartyRandom rng;  // OS CSPRNG: protocol randomness and the root seeds
    // Per-party test inputs come from a documented input seed so the offline
    // checker can recompute alpha and beta. They are protocol INPUTS, not
    // protocol randomness.
    PartyRandom input_rng(args.input_seed * 1000003ULL + (uint64_t)args.party);

    int triple_fail = 0, ole_fail = 0;
    bool selftest_ok = true;
    if (args.selftest > 0) {
        selftest_ok = selftest_primitives(p, args.selftest, ch, rng, triple_fail,
                                          ole_fail);
        // Self-test traffic is diagnostic, not protocol cost; the base-OT count
        // belongs to setup and is preserved.
        const uint64_t base_ots = ch.costs.base_ots;
        ch.costs = ringlpn_2pc::Counters{};
        ch.costs.base_ots = base_ots;
    }
    const uint64_t bytes_before = ch.bytes_sent();
    const uint64_t switches_before = ch.direction_switches();

    std::vector<uint64_t> offs(B);
    std::vector<Word> beta_factors(B);
    for (size_t b = 0; b < B; ++b) {
        if (b == 0) {  // deterministic edge: alpha = 0, beta = p-1
            offs[b] = 0;
            beta_factors[b] = (args.party == 0) ? 1 : (p - 1);
        } else if (b == 1) {  // deterministic edge: alpha = 2^L-2, beta = 1
            offs[b] = half_domain - 1;
            beta_factors[b] = p - 1;
        } else {
            offs[b] = input_rng.u64() % half_domain;
            beta_factors[b] = 1 + input_rng.field(p - 1);
        }
    }

    const auto t_start = std::chrono::steady_clock::now();
    std::vector<spfss_host::DPFKey> keys;
    const bool batch_ok = two_party_dpf_gen_batch(args.party, L, p, offs,
                                                  beta_factors, ch, rng, keys);
    const double total_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_start)
                                .count();

    const uint64_t protocol_bytes = ch.bytes_sent() - bytes_before;
    const uint64_t protocol_switches = ch.direction_switches() - switches_before;
    const int generated = batch_ok ? (int)keys.size() : 0;

    std::vector<ringlpn_keyio::TestInput> inputs(B);
    for (size_t b = 0; b < B; ++b) {
        inputs[b] = ringlpn_keyio::TestInput{offs[b], (uint64_t)beta_factors[b]};
    }
    const std::string key_path =
        args.out_prefix + "_p" + std::to_string(args.party) + ".key";
    const std::string meta_path =
        args.out_prefix + "_p" + std::to_string(args.party) + ".testmeta";
    const bool wrote_keys =
        batch_ok && ringlpn_keyio::write_keys(key_path, args.party, keys);
    const bool wrote_meta =
        batch_ok && ringlpn_keyio::write_test_inputs(meta_path, args.party, inputs);

    // Closed forms the frozen contract fixes per tree, now measured on a real
    // transport by each party independently. Batching must not change them.
    const uint64_t trees = (uint64_t)B;
    const ringlpn_2pc::Counters &c = ch.costs;
    const bool accounting_ok =
        c.string_ots_128 == trees * 2ULL * (uint64_t)L &&
        c.triple_ots == trees * 2ULL * (uint64_t)(L - 1) &&
        c.ole_ots == trees * 3ULL * (uint64_t)field_bits &&
        c.bit_triples == trees * (uint64_t)(L - 1) &&
        c.scalar_oles == trees * 3ULL &&
        c.phase_a.logical_bits == trees * 2ULL * (uint64_t)(L - 1) &&
        c.phase_b.logical_bits == trees * 130ULL * (uint64_t)L &&
        c.phase_c.logical_bits == trees * (uint64_t)field_bits &&
        c.revealed_share_bits() ==
            trees * (4ULL * (uint64_t)(L - 1) + 260ULL * (uint64_t)L +
                     2ULL * (uint64_t)field_bits);

    const bool all_ok = batch_ok && generated == args.trees && wrote_keys &&
                        wrote_meta && accounting_ok && selftest_ok;

    ch.sync();  // both parties leave together, so neither sees a torn socket

    auto per_tree = [trees](uint64_t v) { return (double)v / (double)trees; };
    if (args.csv_header) {
        std::printf(
            "party,modulus,log_domain,batch_trees,generated,selftest_rounds,"
            "selftest_triple_fail,selftest_ole_fail,string_ots_128_per_tree,"
            "triple_ots_per_tree,ole_ots_per_tree,bit_triples_per_tree,"
            "scalar_oles_per_tree,phase_a_logical_bits_per_tree,"
            "phase_b_logical_bits_per_tree,phase_c_logical_bits_per_tree,"
            "logical_opened_bits_per_tree,revealed_share_bits_per_tree,"
            "base_ots,setup_bytes_sent,setup_direction_switches,"
            "protocol_bytes_sent_batch,protocol_bytes_sent_per_tree,"
            "protocol_direction_switches_batch,us_batch,us_per_tree,"
            "transcript_accounting,selftest,status\n");
    }
    std::printf(
        "%d,q62%s,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
        "%.1f,%.1f,%llu,%llu,%llu,%llu,%.1f,%llu,%.1f,%.1f,%s,%s,%s\n",
        args.party, args.modulus_idx == 0 ? "" : "b", L, args.trees, generated,
        args.selftest, triple_fail, ole_fail, per_tree(c.string_ots_128),
        per_tree(c.triple_ots), per_tree(c.ole_ots), per_tree(c.bit_triples),
        per_tree(c.scalar_oles), per_tree(c.phase_a.logical_bits),
        per_tree(c.phase_b.logical_bits), per_tree(c.phase_c.logical_bits),
        per_tree(c.logical_opened_bits()), per_tree(c.revealed_share_bits()),
        (unsigned long long)c.base_ots,
        (unsigned long long)ch.setup_bytes_sent(),
        (unsigned long long)ch.setup_rounds(),
        (unsigned long long)protocol_bytes, per_tree(protocol_bytes),
        (unsigned long long)protocol_switches, total_us,
        total_us / (double)trees, accounting_ok ? "pass" : "FAIL",
        args.selftest > 0 ? (selftest_ok ? "pass" : "FAIL") : "skipped",
        all_ok ? "pass" : "FAIL");

    std::fprintf(stderr,
                 "[two-party-dpf] party %d L=%d batch=%d: %.0f string-OTs, "
                 "%.0f triple-OTs, %.0f OLE-OTs, %.0f logical-open bits, "
                 "%.0f revealed-share bits per tree; batch %llu bytes sent, "
                 "%llu direction switches, %.0f us (%.0f us/tree); setup %llu "
                 "bytes / %llu switches; accounting %s; keys -> %s\n",
                 args.party, L, args.trees, per_tree(c.string_ots_128),
                 per_tree(c.triple_ots), per_tree(c.ole_ots),
                 per_tree(c.logical_opened_bits()),
                 per_tree(c.revealed_share_bits()),
                 (unsigned long long)protocol_bytes,
                 (unsigned long long)protocol_switches, total_us,
                 total_us / (double)trees,
                 (unsigned long long)ch.setup_bytes_sent(),
                 (unsigned long long)ch.setup_rounds(),
                 accounting_ok ? "pass" : "FAIL", key_path.c_str());

    return all_ok ? 0 : 1;
}
