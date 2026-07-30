// Two-PROCESS distributed DPF key generation over real sockets and real OT.
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
//     single-process prototype could not report.
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

// ----- the party-local protocol ---------------------------------------------

bool two_party_dpf_gen(int party, int log_domain, Word p, uint64_t off,
                       Word beta_factor, PartyChannel &ch, PartyRandom &rng,
                       spfss_host::DPFKey &K) {
    const int L = log_domain;
    const uint64_t half_domain = 1ULL << (L - 1);
    if (off >= half_domain || beta_factor == 0 || beta_factor >= p) {
        return false;  // local input validation, before any correlation is used
    }

    // --- Phase A: XOR-shared bits of alpha = off_0 + off_1 -------------------
    std::vector<BitTriple> triples;
    ringlpn_2pc::generate_bit_triples(ch, L - 1, rng, triples);

    std::vector<uint8_t> abit((size_t)L, 0);
    uint8_t carry = 0;
    for (int j = 0; j < L; ++j) {
        const uint8_t u = (uint8_t)((off >> j) & 1);
        abit[(size_t)j] = (uint8_t)(u ^ carry);
        if (j + 1 >= L) break;
        // x = u_0 ^ carry (P0 holds u_0^c_0, P1 holds c_1)
        // y = u_1 ^ carry (P0 holds c_0,     P1 holds u_1^c_1)
        const uint8_t x_mine = (party == 0) ? (uint8_t)(u ^ carry) : carry;
        const uint8_t y_mine = (party == 0) ? carry : (uint8_t)(u ^ carry);
        const BitTriple &t = triples[(size_t)j];
        const uint8_t d =
            ch.open_bits((uint8_t)((x_mine ^ t.a) & 1), 1, ch.costs.phase_a);
        const uint8_t e =
            ch.open_bits((uint8_t)((y_mine ^ t.b) & 1), 1, ch.costs.phase_a);
        const uint8_t and_share = (uint8_t)(
            (((party == 0) ? (uint8_t)(d & e) : (uint8_t)0) ^ (uint8_t)(d & t.b) ^
             (uint8_t)(e & t.a) ^ t.c) & 1);
        carry = (uint8_t)(and_share ^ carry);
    }

    // --- root seed: this party's own OS-CSPRNG draw --------------------------
    K.log_domain = L;
    K.modulus = p;
    K.seed = rng.u128();
    K.t0 = (uint8_t)party;
    K.sCW.assign((size_t)L, 0);
    K.tLCW.assign((size_t)L, 0);
    K.tRCW.assign((size_t)L, 0);

    std::vector<Node> nodes(1, Node{K.seed, (uint8_t)party});
    std::vector<Node> next;
    std::vector<U128> expL, expR;
    std::vector<uint8_t> expTL, expTR;

    // --- Phase B: level-synchronous walk ------------------------------------
    for (int i = 0; i < L; ++i) {
        const int bi = L - 1 - i;  // MSB-first bit of alpha at this level
        const size_t nn = nodes.size();
        expL.resize(nn);
        expR.resize(nn);
        expTL.resize(nn);
        expTR.resize(nn);
        U128 aggL = 0, aggR = 0;
        uint8_t taggL = 0, taggR = 0;
        for (size_t k = 0; k < nn; ++k) {
            prg_expand(nodes[k].s, expL[k], expTL[k], expR[k], expTR[k]);
            aggL ^= expL[k];
            aggR ^= expR[k];
            taggL ^= expTL[k];
            taggR ^= expTR[k];
        }
        const U128 Z = aggL ^ aggR;
        const uint8_t a_mine = abit[(size_t)bi];

        // Two string OTs realize a_i * Z with both operands XOR-shared. The
        // schedule is fixed: party 0 sends first, so neither side blocks.
        const U128 r = rng.u128();
        U128 w = 0;
        const std::vector<U128> m0(1, r), m1(1, (U128)(r ^ Z));
        const std::vector<uint8_t> choice(1, a_mine);
        if (party == 0) {
            ch.ot_send_128(m0, m1);
            w = ch.ot_recv_128(choice)[0];
        } else {
            w = ch.ot_recv_128(choice)[0];
            ch.ot_send_128(m0, m1);
        }

        const U128 sCW_share = aggR ^ (a_mine ? Z : (U128)0) ^ w ^ r;
        const U128 sCW = ch.open_u128(sCW_share, ch.costs.phase_b);

        const uint8_t tL_share =
            (uint8_t)((taggL ^ a_mine ^ ((party == 0) ? 1u : 0u)) & 1);
        const uint8_t tR_share = (uint8_t)((taggR ^ a_mine) & 1);
        const uint8_t flags_mine = (uint8_t)(tL_share | (uint8_t)(tR_share << 1));
        const uint8_t flags = ch.open_bits(flags_mine, 2, ch.costs.phase_b);
        const uint8_t tLCW = (uint8_t)(flags & 1);
        const uint8_t tRCW = (uint8_t)((flags >> 1) & 1);

        K.sCW[(size_t)i] = sCW;
        K.tLCW[(size_t)i] = tLCW;
        K.tRCW[(size_t)i] = tRCW;

        next.resize(nn * 2);
        for (size_t k = 0; k < nn; ++k) {
            const uint8_t t = nodes[k].t;
            const U128 sL = expL[k] ^ (t ? sCW : (U128)0);
            const U128 sR = expR[k] ^ (t ? sCW : (U128)0);
            const uint8_t tL = (uint8_t)((expTL[k] ^ (t ? tLCW : 0)) & 1);
            const uint8_t tR = (uint8_t)((expTR[k] ^ (t ? tRCW : 0)) & 1);
            next[2 * k] = Node{sL, tL};
            next[2 * k + 1] = Node{sR, tR};
        }
        nodes.swap(next);
    }

    // --- Phase C: payload correction word from three scalar OLEs -------------
    Word A = 0, Fsum = 0;
    for (const Node &nd : nodes) {
        const Word conv = convert_zp(nd.s, p);
        if (party == 0) {
            A = mod_add(A, conv, p);
            Fsum = mod_add(Fsum, nd.t, p);
        } else {
            A = mod_sub(A, conv, p);
            Fsum = mod_sub(Fsum, nd.t, p);
        }
    }

    // OLE 1: additive shares of beta = beta_0 * beta_1.
    const Word gamma = ringlpn_2pc::ole_p0_sender(ch, beta_factor, p, rng);
    const Word d = mod_sub(gamma, A, p);
    const Word s = Fsum;
    // OLE 2: shares of d_0 * s_1. OLE 3: shares of s_0 * d_1.
    const Word cross01 =
        ringlpn_2pc::ole_p0_sender(ch, (party == 0) ? d : s, p, rng);
    const Word cross10 =
        ringlpn_2pc::ole_p0_sender(ch, (party == 0) ? s : d, p, rng);
    const Word w_share =
        mod_add(mod_add(mod_mul(d, s, p), cross01, p), cross10, p);

    // Only the standard public key material is opened: never d, s, or the
    // hidden leaf-control sign (see the contract's Phase C decision).
    K.finalCW = ch.open_field(w_share, p, ch.costs.phase_c);
    return true;
}

// ----- primitive self-tests (TEST-ONLY: they open private values) -----------

bool selftest_primitives(int party, Word p, int rounds, PartyChannel &ch,
                         PartyRandom &rng, int &triple_fail, int &ole_fail) {
    triple_fail = 0;
    ole_fail = 0;
    std::vector<BitTriple> triples;
    ringlpn_2pc::generate_bit_triples(ch, rounds, rng, triples);
    for (int i = 0; i < rounds; ++i) {
        uint8_t mine[3] = {triples[(size_t)i].a, triples[(size_t)i].b,
                           triples[(size_t)i].c};
        uint8_t theirs[3] = {0, 0, 0};
        ch.exchange(mine, theirs, 3);
        const uint8_t a = (uint8_t)(mine[0] ^ theirs[0]);
        const uint8_t b = (uint8_t)(mine[1] ^ theirs[1]);
        const uint8_t c = (uint8_t)(mine[2] ^ theirs[2]);
        if (((a & b) & 1) != (c & 1)) ++triple_fail;
    }
    for (int i = 0; i < rounds; ++i) {
        const Word x = rng.field(p);
        const Word share = ringlpn_2pc::ole_p0_sender(ch, x, p, rng);
        Word mine[2] = {x, share};
        Word theirs[2] = {0, 0};
        ch.exchange(mine, theirs, sizeof(mine));
        const Word prod = mod_mul(x, theirs[0], p);
        const Word sum = mod_add(share, theirs[1], p);
        if (prod != sum) ++ole_fail;
    }
    (void)party;
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
    return a;
}

}  // namespace

int main(int argc, char **argv) {
    const Args args = parse_args(argc, argv);
    const Word p = args.modulus_idx == 0 ? kPrime62 : kPrime62Crt2;
    const int L = args.log_domain;
    const int field_bits = ringlpn_2pc::field_bits(p);
    const uint64_t half_domain = 1ULL << (L - 1);

    PartyChannel ch(args.party, args.host, args.port);
    PartyRandom rng;  // OS CSPRNG: protocol randomness and the root seed
    // Public, per-party test inputs are derived from a documented input seed so
    // the offline checker can recompute alpha and beta. They are protocol
    // INPUTS, not protocol randomness.
    PartyRandom input_rng(args.input_seed * 1000003ULL + (uint64_t)args.party);

    int triple_fail = 0, ole_fail = 0;
    bool selftest_ok = true;
    if (args.selftest > 0) {
        selftest_ok = selftest_primitives(args.party, p, args.selftest, ch, rng,
                                          triple_fail, ole_fail);
        // Self-test traffic is diagnostic, not protocol cost; the base-OT count
        // belongs to setup and is preserved.
        const uint64_t base_ots = ch.costs.base_ots;
        ch.costs = ringlpn_2pc::Counters{};
        ch.costs.base_ots = base_ots;
    }
    const uint64_t bytes_before = ch.bytes_sent();
    const uint64_t switches_before = ch.direction_switches();

    std::vector<spfss_host::DPFKey> keys;
    std::vector<ringlpn_keyio::TestInput> inputs;
    keys.reserve((size_t)args.trees);
    inputs.reserve((size_t)args.trees);
    const auto t_start = std::chrono::steady_clock::now();
    int generated = 0, rejected = 0;
    for (int tr = 0; tr < args.trees; ++tr) {
        uint64_t off;
        Word beta_factor;
        if (tr == 0) {  // deterministic edge: alpha = 0, beta = p-1
            off = 0;
            beta_factor = (args.party == 0) ? 1 : (p - 1);
        } else if (tr == 1) {  // deterministic edge: alpha = 2^L-2, beta = 1
            off = half_domain - 1;
            beta_factor = p - 1;
        } else {
            off = input_rng.u64() % half_domain;
            beta_factor = 1 + input_rng.field(p - 1);
        }
        spfss_host::DPFKey K;
        if (!two_party_dpf_gen(args.party, L, p, off, beta_factor, ch, rng, K)) {
            ++rejected;
            continue;
        }
        keys.push_back(std::move(K));
        inputs.push_back(ringlpn_keyio::TestInput{off, (uint64_t)beta_factor});
        ++generated;
    }
    const double total_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_start)
                                .count();

    const uint64_t protocol_bytes = ch.bytes_sent() - bytes_before;
    const uint64_t protocol_switches = ch.direction_switches() - switches_before;

    const std::string key_path =
        args.out_prefix + "_p" + std::to_string(args.party) + ".key";
    const std::string meta_path =
        args.out_prefix + "_p" + std::to_string(args.party) + ".testmeta";
    const bool wrote_keys = ringlpn_keyio::write_keys(key_path, args.party, keys);
    const bool wrote_meta =
        ringlpn_keyio::write_test_inputs(meta_path, args.party, inputs);

    // Closed forms the frozen contract fixes for one tree, now measured on a
    // real transport by each party independently.
    const uint64_t trees = (uint64_t)(generated > 0 ? generated : 1);
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

    const bool all_ok = generated == args.trees && rejected == 0 && wrote_keys &&
                        wrote_meta && accounting_ok && selftest_ok;

    ch.sync();  // both parties leave together, so neither sees a torn socket

    auto per_tree = [trees](uint64_t v) { return (double)v / (double)trees; };
    if (args.csv_header) {
        std::printf(
            "party,modulus,log_domain,trees,generated,rejected,selftest_rounds,"
            "selftest_triple_fail,selftest_ole_fail,string_ots_128_per_tree,"
            "triple_ots_per_tree,ole_ots_per_tree,bit_triples_per_tree,"
            "scalar_oles_per_tree,phase_a_logical_bits_per_tree,"
            "phase_b_logical_bits_per_tree,phase_c_logical_bits_per_tree,"
            "logical_opened_bits_per_tree,revealed_share_bits_per_tree,"
            "base_ots,setup_bytes_sent,setup_direction_switches,"
            "protocol_bytes_sent_per_tree,protocol_direction_switches_per_tree,"
            "us_per_tree,transcript_accounting,selftest,status\n");
    }
    std::printf(
        "%d,q62%s,%d,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
        "%.1f,%.1f,%llu,%llu,%llu,%.1f,%.1f,%.1f,%s,%s,%s\n",
        args.party, args.modulus_idx == 0 ? "" : "b", L, args.trees, generated,
        rejected, args.selftest, triple_fail, ole_fail,
        per_tree(c.string_ots_128), per_tree(c.triple_ots), per_tree(c.ole_ots),
        per_tree(c.bit_triples), per_tree(c.scalar_oles),
        per_tree(c.phase_a.logical_bits), per_tree(c.phase_b.logical_bits),
        per_tree(c.phase_c.logical_bits), per_tree(c.logical_opened_bits()),
        per_tree(c.revealed_share_bits()),
        (unsigned long long)c.base_ots,
        (unsigned long long)ch.setup_bytes_sent(),
        (unsigned long long)ch.setup_rounds(), per_tree(protocol_bytes),
        per_tree(protocol_switches), total_us / (double)trees,
        accounting_ok ? "pass" : "FAIL",
        args.selftest > 0 ? (selftest_ok ? "pass" : "FAIL") : "skipped",
        all_ok ? "pass" : "FAIL");

    std::fprintf(stderr,
                 "[two-party-dpf] party %d L=%d trees=%d/%d: %.0f string-OTs, "
                 "%.0f triple-OTs, %.0f OLE-OTs, %.0f logical-open bits, "
                 "%.0f revealed-share bits, %.0f protocol bytes sent, "
                 "%.0f direction switches, %.0f us per tree; setup %llu bytes / "
                 "%llu switches; accounting %s; keys -> %s\n",
                 args.party, L, generated, args.trees, per_tree(c.string_ots_128),
                 per_tree(c.triple_ots), per_tree(c.ole_ots),
                 per_tree(c.logical_opened_bits()),
                 per_tree(c.revealed_share_bits()), per_tree(protocol_bytes),
                 per_tree(protocol_switches), total_us / (double)trees,
                 (unsigned long long)ch.setup_bytes_sent(),
                 (unsigned long long)ch.setup_rounds(),
                 accounting_ok ? "pass" : "FAIL", key_path.c_str());

    return all_ok ? 0 : 1;
}
