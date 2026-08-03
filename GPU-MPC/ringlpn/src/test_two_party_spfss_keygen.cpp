// Two-PROCESS SPFSS key generation for the Figure 2 Ring-LPN OLE engine.
//
// This is the dealerless replacement for `build_spfss_keys()` in
// src/bench_ole_ringlpn_cuda.cu, which is the pipeline's centralized-keygen
// oracle boundary. The structure the OLE engine needs is exactly what the
// two-party DPF protocol provides:
//
//   for each polynomial pair (i,j) and noise group, the sparse product has
//   points  alpha = a_(i,k) + b_(j,l)  with payload  u_(i,k) * v_(j,l),
//
// i.e. an ADDITIVELY shared position and a MULTIPLICATIVELY shared payload,
// with party 0 holding (a, u) and party 1 holding (b, v) privately. Each party
// runs this program against its own noise record; neither reads the other's.
//
// The protocol code is the shared one in two_party_dpf_protocol.h (the same
// implementation the standalone keygen artifact gates), run with the GPU
// expansion PRG so the emitted keys are consumable by the unmodified GPU
// evaluator. All trees of all groups go into one level-synchronous batch; the
// measured direction-switch count depends on tree depth, not tree count.
//
// No security claim: transports are real OT but not silent OT, the exact DPF
// distribution/security reduction is still open, and the noise itself is
// sampled by the benchmark (labelled), not by a distributed sampler.

#include "dpf_key_io.h"
#include "spfss_host.h"
#include "two_party_dpf_protocol.h"

#include <chrono>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using ringlpn_2pc::Word;
using ringlpn_2pdpf::PrgMode;

struct Args {
    int party = 0;
    std::string host = "127.0.0.1";
    int port = 45600;
    std::string noise;      // this party's own noise record
    std::string out;        // this party's own SPFSS key file
    bool csv_header = false;
};

Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (k == "--party") a.party = std::atoi(next().c_str());
        else if (k == "--host") a.host = next();
        else if (k == "--port") a.port = std::atoi(next().c_str());
        else if (k == "--noise") a.noise = next();
        else if (k == "--out") a.out = next();
        else if (k == "--csv-header") a.csv_header = true;
        else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            std::exit(2);
        }
    }
    if ((a.party != 0 && a.party != 1) || a.noise.empty() || a.out.empty()) {
        std::fprintf(stderr,
                     "usage: --party 0|1 --noise <file> --out <file> "
                     "[--host h] [--port p] [--csv-header]\n");
        std::exit(2);
    }
    return a;
}

}  // namespace

int main(int argc, char **argv) {
    const Args args = parse_args(argc, argv);

    ringlpn_keyio::spfss_groups::NoiseRecord noise;
    if (!ringlpn_keyio::spfss_groups::read_noise(args.noise, noise)) {
        std::fprintf(stderr, "[two-party-spfss] cannot read %s\n",
                     args.noise.c_str());
        return 2;
    }
    if (noise.party != args.party) {
        std::fprintf(stderr,
                     "[two-party-spfss] noise record is for party %d, not %d\n",
                     noise.party, args.party);
        return 2;
    }

    const int c = noise.c;
    const int t = noise.t;
    const int L = noise.log_domain;
    const Word p = (Word)noise.modulus;
    const uint64_t ct = uint64_t(c) * uint64_t(t);
    const uint64_t max_int = uint64_t(std::numeric_limits<int>::max());
    if (c <= 0 || t <= 0 || L < 2 || L > 20 ||
        (p != ringlpn_2pdpf::kPrime62 &&
         p != ringlpn_2pdpf::kPrime62Crt2) ||
        ct == 0 || ct > max_int / ct ||
        ct * ct > max_int / uint64_t(L - 1)) {
        std::fprintf(stderr,
                     "[two-party-spfss] unsupported public parameter set\n");
        return 2;
    }
    const int groups = noise.regular ? (2 * t - 1) : 1;
    const uint64_t half_domain = 1ULL << (L - 1);
    if ((noise.regular &&
         (noise.bucket <= 0 || uint64_t(noise.bucket) != half_domain)) ||
        (!noise.regular && noise.bucket != 0)) {
        std::fprintf(stderr, "[two-party-spfss] inconsistent noise layout\n");
        return 2;
    }

    // Build this party's private per-tree inputs in exactly the order the OLE
    // engine indexes its SPFSS keys: key_idx = (i + j*c) * groups + group.
    std::vector<uint64_t> offs;
    std::vector<Word> betas;
    std::vector<size_t> group_sizes;
    group_sizes.reserve((size_t)c * c * groups);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            // Party 0 contributes polynomial i, party 1 contributes polynomial j.
            const int poly = (args.party == 0) ? i : j;
            const uint64_t *pos = &noise.positions[(size_t)poly * (size_t)t];
            const uint64_t *val = &noise.values[(size_t)poly * (size_t)t];
            for (int group = 0; group < groups; ++group) {
                size_t points = 0;
                if (noise.regular) {
                    for (int k = 0; k < t; ++k) {
                        const int l = group - k;
                        if (l < 0 || l >= t) continue;
                        // party 0 walks k, party 1 walks l = group - k
                        const int own = (args.party == 0) ? k : l;
                        const uint64_t off =
                            pos[own] - (uint64_t)own * (uint64_t)noise.bucket;
                        offs.push_back(off);
                        betas.push_back((Word)val[own]);
                        ++points;
                    }
                } else {
                    for (int k = 0; k < t; ++k) {
                        for (int l = 0; l < t; ++l) {
                            const int own = (args.party == 0) ? k : l;
                            offs.push_back(pos[own]);
                            betas.push_back((Word)val[own]);
                            ++points;
                        }
                    }
                }
                group_sizes.push_back(points);
            }
        }
    }

    for (size_t b = 0; b < offs.size(); ++b) {
        if (offs[b] >= half_domain || betas[b] == 0 || betas[b] >= p) {
            std::fprintf(stderr,
                         "[two-party-spfss] input %zu out of range (off=%llu "
                         "beta=%llu half=%llu)\n",
                         b, (unsigned long long)offs[b],
                         (unsigned long long)betas[b],
                         (unsigned long long)half_domain);
            return 2;
        }
    }

    PartyChannel ch(args.party, args.host, args.port);
    PartyRandom rng;

    const uint64_t setup_bytes = ch.setup_bytes_sent();
    const uint64_t setup_switches = ch.setup_direction_switches();
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<spfss_host::DPFKey> keys;
    const bool ok = ringlpn_2pdpf::two_party_dpf_gen_batch(
        args.party, L, p, PrgMode::kGpuAes, offs, betas, ch, rng, keys);
    const double total_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_start)
                                .count();
    const uint64_t protocol_bytes = ch.bytes_sent() - setup_bytes;
    const uint64_t protocol_switches = ch.direction_switches() - setup_switches;

    // Split the flat batch back into the engine's per-key_idx groups.
    std::vector<std::vector<spfss_host::DPFKey>> grouped;
    bool split_ok = ok && keys.size() == offs.size();
    if (split_ok) {
        size_t cursor = 0;
        grouped.reserve(group_sizes.size());
        for (size_t g : group_sizes) {
            if (cursor + g > keys.size()) { split_ok = false; break; }
            grouped.emplace_back(keys.begin() + (long)cursor,
                                 keys.begin() + (long)(cursor + g));
            cursor += g;
        }
        if (split_ok && cursor != keys.size()) split_ok = false;
    }

    const bool wrote =
        split_ok && ringlpn_keyio::spfss_groups::write(
                        args.out, args.party, L, (uint64_t)p, grouped);

    const size_t trees = offs.size() == 0 ? 1 : offs.size();
    const ringlpn_2pc::Counters &cost = ch.costs;
    const int field_bits = ringlpn_2pc::field_bits(p);
    const bool accounting_ok =
        cost.string_ots_128 == (uint64_t)trees * 2ULL * (uint64_t)L &&
        cost.bit_triples == (uint64_t)trees * (uint64_t)(L - 1) &&
        cost.scalar_oles == (uint64_t)trees * 3ULL &&
        cost.logical_opened_bits() ==
            (uint64_t)trees * (2ULL * (uint64_t)(L - 1) +
                               130ULL * (uint64_t)L + (uint64_t)field_bits);
    const bool all_ok = ok && split_ok && wrote && accounting_ok;

    ch.sync();

    if (args.csv_header) {
        std::printf("party,c,t,noise_mode,log_domain,groups,trees,"
                    "string_ots_per_tree,bit_triples_per_tree,"
                    "scalar_oles_per_tree,logical_opened_bits_per_tree,"
                    "meaningful_share_bits_per_tree,base_ots,setup_bytes_sent,"
                    "protocol_bytes_sent,protocol_direction_switches,"
                    "keygen_us,transcript_accounting,status\n");
    }
    auto per_tree = [trees](uint64_t v) { return (double)v / (double)trees; };
    std::printf("%d,%d,%d,%s,%d,%d,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%llu,%llu,%llu,"
                "%llu,%.1f,%s,%s\n",
                args.party, c, t, noise.regular ? "regular" : "uniform", L,
                groups, trees, per_tree(cost.string_ots_128),
                per_tree(cost.bit_triples), per_tree(cost.scalar_oles),
                per_tree(cost.logical_opened_bits()),
                per_tree(cost.meaningful_share_bits()),
                (unsigned long long)cost.base_ots,
                (unsigned long long)setup_bytes,
                (unsigned long long)protocol_bytes,
                (unsigned long long)protocol_switches, total_us,
                accounting_ok ? "pass" : "FAIL", all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[two-party-spfss] party %d: %zu trees in %zu groups at L=%d, "
                 "%llu protocol bytes, %llu direction switches, %.0f us -> %s (%s)\n",
                 args.party, trees, group_sizes.size(), L,
                 (unsigned long long)protocol_bytes,
                 (unsigned long long)protocol_switches, total_us,
                 args.out.c_str(), all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}
