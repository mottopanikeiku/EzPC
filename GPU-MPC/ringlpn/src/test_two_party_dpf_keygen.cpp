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
#include "two_party_dpf_protocol.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The protocol itself lives in two_party_dpf_protocol.h so the Ring-LPN SPFSS
// keygen runs the identical code path.
using ringlpn_2pdpf::Node;
using ringlpn_2pdpf::PrgMode;
using ringlpn_2pdpf::kPrime62;
using ringlpn_2pdpf::kPrime62Crt2;
using ringlpn_2pdpf::two_party_dpf_gen_batch;
using ringlpn_2pc::BitTriple;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using ringlpn_2pc::U128;
using ringlpn_2pc::Word;
using ringlpn_2pc::mod_add;
using ringlpn_2pc::mod_mul;
using ringlpn_2pc::mod_sub;


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
    std::string prg = "splitmix";
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
        else if (k == "--prg") a.prg = next();
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
    if (a.prg != "splitmix" && a.prg != "gpu-aes") {
        std::fprintf(stderr, "--prg must be splitmix or gpu-aes\n");
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
    const PrgMode prg_mode =
        (args.prg == "gpu-aes") ? PrgMode::kGpuAes : PrgMode::kSplitmix;
    const bool batch_ok = two_party_dpf_gen_batch(
        args.party, L, p, prg_mode, offs, beta_factors, ch, rng, keys);
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
            "party,modulus,prg,log_domain,batch_trees,generated,selftest_rounds,"
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
        "%d,q62%s,%s,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
        "%.1f,%.1f,%llu,%llu,%llu,%llu,%.1f,%llu,%.1f,%.1f,%s,%s,%s\n",
        args.party, args.modulus_idx == 0 ? "" : "b", args.prg.c_str(), L,
        args.trees, generated,
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
