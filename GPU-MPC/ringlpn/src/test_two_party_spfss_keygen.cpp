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
// with party 0 holding (a, u) and party 1 holding (b, v) privately. By default
// each process samples only its own noise in memory; external noise-record
// ingestion is retained solely as a labelled TEST-ONLY artifact path.
//
// Party-local preparation and grouping live in two_party_spfss.h. This
// standalone host executable intentionally selects the explicit CPU baseline
// with the GPU-AES-compatible PRG; the live FC path selects GPU-batched keygen.
// All groups use one level-synchronous batch; direction switches depend on tree
// depth, not tree count.
//
// No security claim: transports are real OT but not silent OT, and the exact
// DPF distribution/security reduction is still open. In independent-sampling
// mode, each process samples only its own noise from OpenSSL's private DRBG.

#include "two_party_spfss.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using ringlpn_2pc::Word;
using ringlpn_spfss::SpfssPartyBatch;
using ringlpn_spfss::SpfssPublicParams;
using ringlpn_spfss::SpfssWork;

struct Args {
    int party = 0;
    std::string host = "127.0.0.1";
    int port = 45600;
    std::string noise;      // sample output, or TEST-ONLY external input
    std::string out;        // this party's own SPFSS key file
    int c = 0;
    int t = 0;
    int log_domain = 0;
    Word modulus = 0;
    uint64_t sid = 0;
    std::string noise_mode;
    bool sample_independent = false;
    bool csv_header = false;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: --party 0|1 --out <file> [--host h] [--port p] "
                 "[--sid N] [--csv-header] "
                 "(--c N --t N --log-domain L --modulus P "
                 "--noise-mode uniform|regular [--sample-independent] "
                 "[--noise <sample-output-record>] | "
                 "--noise <TEST-ONLY-existing-record>)\n");
    std::exit(2);
}

uint64_t parse_u64(const std::string &text, const char *name) {
    if (text.empty() || text[0] == '-') {
        std::fprintf(stderr, "invalid value for %s\n", name);
        usage();
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        std::fprintf(stderr, "invalid value for %s\n", name);
        usage();
    }
    return static_cast<uint64_t>(value);
}

int parse_int(const std::string &text, const char *name) {
    const uint64_t value = parse_u64(text, name);
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "value for %s is too large\n", name);
        usage();
    }
    return static_cast<int>(value);
}

Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                usage();
            }
            return std::string(argv[++i]);
        };
        if (k == "--party") a.party = parse_int(next(), "--party");
        else if (k == "--host") a.host = next();
        else if (k == "--port") a.port = parse_int(next(), "--port");
        else if (k == "--noise") a.noise = next();
        else if (k == "--out") a.out = next();
        else if (k == "--c") a.c = parse_int(next(), "--c");
        else if (k == "--t") a.t = parse_int(next(), "--t");
        else if (k == "--log-domain") a.log_domain = parse_int(next(), "--log-domain");
        else if (k == "--modulus") a.modulus = parse_u64(next(), "--modulus");
        else if (k == "--sid") a.sid = parse_u64(next(), "--sid");
        else if (k == "--noise-mode") a.noise_mode = next();
        else if (k == "--sample-independent") a.sample_independent = true;
        else if (k == "--csv-header") a.csv_header = true;
        else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            usage();
        }
    }
    if ((a.party != 0 && a.party != 1) || a.out.empty() ||
        a.port <= 0 || a.port > 65534) {
        usage();
    }
    const bool sampling_parameters =
        a.c != 0 || a.t != 0 || a.log_domain != 0 || a.modulus != 0 ||
        !a.noise_mode.empty();
    if (a.sample_independent || sampling_parameters) {
        a.sample_independent = true;
        if (a.c <= 0 || a.t <= 0 || a.log_domain <= 0 || a.modulus == 0 ||
            a.noise.empty() ||
            (a.noise_mode != "uniform" && a.noise_mode != "regular")) {
            usage();
        }
    } else if (a.noise.empty()) {
        usage();
    }
    return a;
}

std::string digest_hex(const ringlpn_spfss::SpfssDigest &digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string text(digest.size() * 2, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        text[2 * i] = kHex[digest[i] >> 4];
        text[2 * i + 1] = kHex[digest[i] & 0x0f];
    }
    return text;
}


}  // namespace

int main(int argc, char **argv) {
    const Args args = parse_args(argc, argv);
    const std::string temporary_out = args.out + ".tmp";
    std::remove(args.out.c_str());
    std::remove(temporary_out.c_str());
    const std::string temporary_noise =
        args.sample_independent && !args.noise.empty()
            ? args.noise + ".tmp"
            : std::string();
    if (!temporary_noise.empty()) {
        std::remove(args.noise.c_str());
        std::remove(temporary_noise.c_str());
    }

    PartyRandom rng;  // OpenSSL private DRBG for both local noise and protocol
    ringlpn_keyio::spfss_groups::NoiseRecord noise;
    SpfssPublicParams params;
    bool external_read_ok = true;
    if (args.sample_independent) {
        params.c = args.c;
        params.t = args.t;
        params.log_domain = args.log_domain;
        params.modulus = args.modulus;
        params.regular = args.noise_mode == "regular";
    } else {
        std::fprintf(stderr,
                     "[two-party-spfss] noise source is "
                     "TEST-ONLY external record ingestion\n");
        if (!ringlpn_keyio::spfss_groups::read_noise(args.noise, noise)) {
            std::fprintf(stderr, "[two-party-spfss] cannot read %s\n",
                         args.noise.c_str());
            external_read_ok = false;
        } else {
            params.c = noise.c;
            params.t = noise.t;
            params.log_domain = noise.log_domain;
            params.modulus = noise.modulus;
            params.regular = noise.regular;
        }
    }
    params.sid = args.sid;

    PartyChannel ch(args.party, args.host, args.port, /*defer_ot_setup=*/true);
    const uint64_t channel_open_bytes = ch.bytes_sent();
    const uint64_t channel_open_switches = ch.direction_switches();
    SpfssWork work;
    bool local_valid =
        external_read_ok && ringlpn_spfss::derive_spfss_work(params, work);
    if (!local_valid) {
        std::fprintf(stderr,
                     "[two-party-spfss] unsupported or unavailable local "
                     "parameter/input work set\n");
    }
    if (args.sample_independent && local_valid) {
        local_valid =
            ringlpn_spfss::sample_party_noise(params, args.party, rng, noise);
        if (!local_valid) {
            std::fprintf(stderr,
                         "[two-party-spfss] cannot sample party noise\n");
        } else if (!temporary_noise.empty() &&
                   !ringlpn_keyio::spfss_groups::write_noise(temporary_noise,
                                                              noise)) {
            std::fprintf(stderr, "[two-party-spfss] cannot stage %s\n",
                         args.noise.c_str());
            local_valid = false;
        }
    } else if (!args.sample_independent && local_valid) {
        local_valid =
            ringlpn_spfss::validate_party_noise(noise, args.party, params);
        if (!local_valid) {
            std::fprintf(stderr,
                         "[two-party-spfss] invalid TEST-ONLY noise record\n");
        }
    }

    SpfssPartyBatch batch;
    if (local_valid &&
        !ringlpn_spfss::make_party_spfss_batch(args.party, params, noise,
                                                batch)) {
        std::fprintf(stderr,
                     "[two-party-spfss] cannot construct party batch\n");
        local_valid = false;
    }

    const char *noise_source =
        args.sample_independent ? "independent_os_drbg"
                                : "TEST_ONLY_external_noise_record";
    ringlpn_spfss::SpfssDigest noise_digest{};
    ringlpn_spfss::SpfssDigest noise_content_digest{};
    ringlpn_spfss::SpfssDigest manifest_digest{};
    std::vector<uint64_t> binding;
    if (!ringlpn_spfss::digest_spfss_public_manifest(params,
                                                      manifest_digest)) {
        local_valid = false;
    }
    if (local_valid &&
        (!ringlpn_spfss::digest_party_noise(noise, args.party, params,
                                             noise_digest) ||
         !ringlpn_spfss::digest_noise_content(noise, args.party, params,
                                               noise_content_digest) ||
         !ringlpn_spfss::party_noise_binding(noise, args.party, params,
                                              binding))) {
        std::fprintf(stderr,
                     "[two-party-spfss] cannot construct local provenance\n");
        local_valid = false;
    }

    const int c = params.c;
    const int t = params.t;
    const int L = params.log_domain;
    const Word p = static_cast<Word>(params.modulus);
    const int groups = work.groups_per_pair;

    const uint64_t agreement_begin_bytes = ch.bytes_sent();
    const uint64_t agreement_begin_switches = ch.direction_switches();
    if (!ringlpn_spfss::agree_spfss_public_manifest(ch, params, local_valid)) {
        std::fprintf(stderr,
                     "[two-party-spfss] public-parameter/local-validation "
                     "mismatch; aborting before OT setup/output\n");
        std::remove(temporary_out.c_str());
        if (!temporary_noise.empty()) {
            std::remove(temporary_noise.c_str());
        }
        return 2;
    }
    const uint64_t agreement_bytes = ch.bytes_sent() - agreement_begin_bytes;
    const uint64_t agreement_switches =
        ch.direction_switches() - agreement_begin_switches;
    ch.setup_ots();
    const uint64_t setup_bytes =
        ch.bytes_sent() - agreement_begin_bytes - agreement_bytes;
    const uint64_t setup_switches =
        ch.direction_switches() - agreement_begin_switches - agreement_switches;
    const uint64_t protocol_begin_bytes = ch.bytes_sent();
    const uint64_t protocol_begin_switches = ch.direction_switches();
    const auto t_start = std::chrono::steady_clock::now();
    ringlpn_spfss::GroupedHostKeys grouped;
    ringlpn_spfss::DpfCounters dpf_counters;
    const bool ok = ringlpn_spfss::generate_party_spfss_keys_cpu_baseline(
        args.party, params, batch, ch, rng, grouped, dpf_counters);
    const double total_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_start)
                                .count();
    const uint64_t protocol_bytes = ch.bytes_sent() - protocol_begin_bytes;
    const uint64_t protocol_switches =
        ch.direction_switches() - protocol_begin_switches;

    ringlpn_spfss::SpfssDigest common_cw_digest{};
    const bool common_digest_ok =
        ok && ringlpn_spfss::digest_common_cw_transcript(
                  args.party, params, grouped, common_cw_digest);
    const size_t trees = static_cast<size_t>(work.tree_count);
    const ringlpn_2pc::Counters &cost = ch.costs;
    const int field_bits = ringlpn_2pc::field_bits(p);
    const bool accounting_ok =
        dpf_counters.string_ots_128 ==
            static_cast<uint64_t>(trees) * 2ULL * static_cast<uint64_t>(L) &&
        dpf_counters.bit_triples ==
            static_cast<uint64_t>(trees) * static_cast<uint64_t>(L - 1) &&
        dpf_counters.scalar_oles == static_cast<uint64_t>(trees) * 3ULL &&
        dpf_counters.logical_opened_bits ==
            static_cast<uint64_t>(trees) *
                (2ULL * static_cast<uint64_t>(L - 1) +
                 130ULL * static_cast<uint64_t>(L) +
                 static_cast<uint64_t>(field_bits)) &&
        dpf_counters.meaningful_share_bits ==
            2ULL * dpf_counters.logical_opened_bits;
    const bool staged =
        ok && accounting_ok && common_digest_ok &&
        ringlpn_keyio::spfss_groups::write(
            temporary_out, args.party, L, static_cast<uint64_t>(p), binding,
            grouped);
    const uint64_t final_sync_begin_bytes = ch.bytes_sent();
    const uint64_t final_sync_begin_switches = ch.direction_switches();
    ch.sync();
    const uint8_t mine_publishable = staged ? 1 : 0;
    uint8_t peer_publishable = 0;
    ch.exchange_bytes(&mine_publishable, &peer_publishable, 1);
    const bool may_rename = staged && peer_publishable == 1;
    const bool key_renamed =
        may_rename &&
        std::rename(temporary_out.c_str(), args.out.c_str()) == 0;
    const bool noise_renamed =
        key_renamed &&
        (temporary_noise.empty() ||
         std::rename(temporary_noise.c_str(), args.noise.c_str()) == 0);
    const uint8_t mine_renamed = key_renamed && noise_renamed ? 1 : 0;
    uint8_t peer_renamed = 0;
    ch.exchange_bytes(&mine_renamed, &peer_renamed, 1);
    const uint64_t final_sync_bytes =
        ch.bytes_sent() - final_sync_begin_bytes;
    const uint64_t final_sync_switches =
        ch.direction_switches() - final_sync_begin_switches;
    const bool wrote = mine_renamed == 1 && peer_renamed == 1;
    if (!wrote) {
        std::remove(temporary_out.c_str());
        std::remove(args.out.c_str());
        if (!temporary_noise.empty()) {
            std::remove(temporary_noise.c_str());
            std::remove(args.noise.c_str());
        }
    }
    const bool all_ok = ok && accounting_ok && common_digest_ok && wrote;
    const uint64_t transport_bytes = ch.bytes_sent();
    const uint64_t transport_switches = ch.direction_switches();

    if (args.csv_header) {
        std::printf(
            "party,c,t,noise_mode,log_domain,groups,trees,"
            "string_ots_per_tree,bit_triples_per_tree,"
            "scalar_oles_per_tree,logical_opened_bits_per_tree,"
            "meaningful_share_bits_per_tree,base_ots,"
            "channel_open_bytes_sent,channel_open_direction_switches,"
            "agreement_bytes_sent,agreement_direction_switches,"
            "setup_bytes_sent,setup_direction_switches,"
            "protocol_bytes_sent,protocol_direction_switches,"
            "final_sync_bytes_sent,final_sync_direction_switches,"
            "transport_bytes_sent,transport_direction_switches,"
            "keygen_us,sid,noise_source,local_noise_sha256,"
            "noise_content_sha256,public_manifest_sha256,common_cw_sha256,"
            "transcript_accounting,status\n");
    }
    auto per_tree = [trees](uint64_t v) { return (double)v / (double)trees; };
    std::printf(
        "%d,%d,%d,%s,%d,%d,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%llu,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%.1f,%llu,%s,%s,%s,%s,%s,%s,%s\n",
        args.party, c, t, noise.regular ? "regular" : "uniform", L,
        groups, trees, per_tree(cost.string_ots_128),
        per_tree(cost.bit_triples), per_tree(cost.scalar_oles),
        per_tree(cost.logical_opened_bits()),
        per_tree(cost.meaningful_share_bits()),
        (unsigned long long)cost.base_ots,
        (unsigned long long)channel_open_bytes,
        (unsigned long long)channel_open_switches,
        (unsigned long long)agreement_bytes,
        (unsigned long long)agreement_switches,
        (unsigned long long)setup_bytes,
        (unsigned long long)setup_switches,
        (unsigned long long)protocol_bytes,
        (unsigned long long)protocol_switches,
        (unsigned long long)final_sync_bytes,
        (unsigned long long)final_sync_switches,
        (unsigned long long)transport_bytes,
        (unsigned long long)transport_switches, total_us,
        (unsigned long long)params.sid, noise_source,
        digest_hex(noise_digest).c_str(),
        digest_hex(noise_content_digest).c_str(),
        digest_hex(manifest_digest).c_str(),
        digest_hex(common_cw_digest).c_str(),
        accounting_ok ? "pass" : "FAIL", all_ok ? "pass" : "FAIL");
    std::fprintf(
        stderr,
        "[two-party-spfss-provenance] party=%d sid=%llu noise_source=%s "
        "local_noise_sha256=%s noise_content_sha256=%s "
        "public_manifest_sha256=%s common_cw_sha256=%s\n",
        args.party, static_cast<unsigned long long>(params.sid), noise_source,
        digest_hex(noise_digest).c_str(),
        digest_hex(noise_content_digest).c_str(),
        digest_hex(manifest_digest).c_str(),
        digest_hex(common_cw_digest).c_str());
    std::fprintf(stderr,
                 "[two-party-spfss] party %d: %zu trees in %zu groups at L=%d, "
                 "%llu protocol bytes, %llu direction switches, %.0f us -> %s (%s)\n",
                 args.party, trees, batch.group_sizes.size(), L,
                 (unsigned long long)protocol_bytes,
                 (unsigned long long)protocol_switches, total_us,
                 args.out.c_str(), all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}
