// Two-process correctness and transcript-accounting gate for secure stochastic
// truncation. Each live call receives only one party's additive mask shares;
// TEST-ONLY output files let --check reconstruct masks and output shares after
// both processes exit. This is an OT-hybrid semi-honest artifact over plain
// unauthenticated loopback TCP, not a privacy proof or deployment transport.

#include "correlation_freshness.h"
#include "secure_truncate.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;

constexpr size_t kBoundaryCount = 16;
constexpr size_t kMaxTrials = 65520;

uint64_t mask(int bits) {
    return bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
}

struct Args {
    int party = -1;
    std::string host = "127.0.0.1";
    int port = 42700;
    int bw = 16;
    int shift = 8;
    size_t trials = 128;
    uint64_t seed = 1;
    std::string prefix = "two_party_secure_truncate";
    bool check = false;
    bool csv_header = false;
    bool expect_reject = false;
    bool mismatch_shift = false;
    bool mismatch_correlation = false;
    int invalid_share_party = -1;
};

bool parse_size(const std::string &text, size_t &out) {
    try {
        size_t used = 0;
        const unsigned long long value = std::stoull(text, &used, 10);
        if (used != text.size() ||
            value > std::numeric_limits<size_t>::max()) {
            return false;
        }
        out = size_t(value);
        return true;
    } catch (...) {
        return false;
    }
}

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", key.c_str());
                std::exit(2);
            }
            return argv[i];
        };
        if (key == "--party") {
            a.party = std::atoi(next().c_str());
        } else if (key == "--host") {
            a.host = next();
        } else if (key == "--port") {
            a.port = std::atoi(next().c_str());
        } else if (key == "--bw") {
            a.bw = std::atoi(next().c_str());
        } else if (key == "--shift") {
            a.shift = std::atoi(next().c_str());
        } else if (key == "--trials") {
            if (!parse_size(next(), a.trials)) std::exit(2);
        } else if (key == "--seed") {
            a.seed = std::strtoull(next().c_str(), nullptr, 10);
        } else if (key == "--out-prefix") {
            a.prefix = next();
        } else if (key == "--invalid-share-party") {
            a.invalid_share_party = std::atoi(next().c_str());
        } else if (key == "--check") {
            a.check = true;
        } else if (key == "--csv-header") {
            a.csv_header = true;
        } else if (key == "--expect-reject") {
            a.expect_reject = true;
        } else if (key == "--mismatch-shift") {
            a.mismatch_shift = true;
        } else if (key == "--mismatch-correlation") {
            a.mismatch_correlation = true;
        } else {
            std::fprintf(stderr, "unknown option %s\n", key.c_str());
            std::exit(2);
        }
    }
    if ((!a.check && a.party != 0 && a.party != 1) || a.port <= 0 ||
        a.port > 65535 || a.bw <= 2 || a.bw > 32 || a.shift <= 0 ||
        a.shift >= a.bw || a.trials == 0 || a.trials > kMaxTrials ||
        a.seed == 0 || (a.invalid_share_party < -1 ||
                        a.invalid_share_party > 1) ||
        (a.expect_reject != (a.mismatch_shift || a.mismatch_correlation ||
                             a.invalid_share_party >= 0)) ||
        int(a.mismatch_shift) + int(a.mismatch_correlation) +
                int(a.invalid_share_party >= 0) >
            1) {
        std::fprintf(stderr, "invalid secure-truncate arguments\n");
        std::exit(2);
    }
    return a;
}

struct Input {
    uint64_t x = 0;
    uint64_t y = 0;
    uint64_t r0 = 0;
    uint64_t r1 = 0;
    uint64_t s0 = 0;
    uint64_t s1 = 0;
};

std::vector<Input> make_inputs(const Args &a) {
    const uint64_t ring_mask = mask(a.bw);
    const uint64_t low_mask = mask(a.shift);
    const uint64_t F = uint64_t(1) << a.shift;
    const uint64_t B = uint64_t(1) << a.bw;
    const uint64_t output_mask = mask(a.bw - a.shift);
    const uint64_t boundary_x[kBoundaryCount] = {
        0,
        1,
        low_mask,
        F,
        F + 1,
        ring_mask >> 1,
        (ring_mask >> 1) + 1,
        ring_mask - F,
        ring_mask - F + 1,
        ring_mask - 1,
        ring_mask,
        low_mask > 1 ? low_mask - 1 : 0,
        F + low_mask,
        (ring_mask & ~low_mask),
        (ring_mask & ~low_mask) - F,
        B - (F >> 1),
    };
    const uint64_t boundary_r0[kBoundaryCount] = {
        0, ring_mask, low_mask, F, F - 1, ring_mask - F + 1, 1, low_mask,
        ring_mask, F + 1, ring_mask >> 1, low_mask - 1, F - 1, 0,
        ring_mask - low_mask, ring_mask,
    };
    const uint64_t boundary_r1[kBoundaryCount] = {
        0, 1, 1, ring_mask - F + 1, low_mask, F, ring_mask, ring_mask,
        low_mask, ring_mask - F, (ring_mask >> 1) + 1, 2, ring_mask,
        ring_mask, low_mask, (F >> 1) + 1,
    };

    std::vector<Input> out;
    out.reserve(kBoundaryCount + a.trials);
    for (size_t i = 0; i < kBoundaryCount; ++i) {
        const uint64_t x = boundary_x[i] & ring_mask;
        const uint64_t r0 = boundary_r0[i] & ring_mask;
        const uint64_t r1 = boundary_r1[i] & ring_mask;
        const uint64_t r = (r0 + r1) & ring_mask;
        out.push_back({x, (x + r) & ring_mask, r0, r1,
                       boundary_r0[i] & output_mask,
                       boundary_r1[i] & output_mask});
    }

    std::mt19937_64 gen(a.seed);
    for (size_t i = 0; i < a.trials; ++i) {
        const uint64_t x = gen() & ring_mask;
        const uint64_t r0 = gen() & ring_mask;
        const uint64_t r1 = gen() & ring_mask;
        const uint64_t s0 = gen() & output_mask;
        const uint64_t s1 = gen() & output_mask;
        const uint64_t r = (r0 + r1) & ring_mask;
        out.push_back({x, (x + r) & ring_mask, r0, r1, s0, s1});
    }
    return out;
}

bool derive_id(const Args &a, std::array<uint8_t, 32> &id) {
    ringlpn_freshness::InvocationId invocation{};
    for (size_t byte = 0; byte < sizeof(a.seed); ++byte) {
        invocation[byte] = uint8_t(a.seed >> (8 * byte));
    }
    ringlpn_freshness::Digest layer{};
    static constexpr uint8_t domain[] =
        "RINGLPN-SECURE-TRUNCATE-CORRECTNESS-BASELINE";
    if (!ringlpn_freshness::digest(domain, sizeof(domain) - 1, layer)) {
        return false;
    }
    ringlpn_freshness::Coordinates coordinates;
    coordinates.kind = ringlpn_freshness::Kind::kConversionEdabit;
    coordinates.phase = ringlpn_freshness::Phase::kConvertCorrelation;
    coordinates.conversion_chunk = uint32_t(a.shift);
    coordinates.primitive_ordinal = 1;
    return ringlpn_freshness::derive_correlation_id(invocation, layer,
                                                    coordinates, id);
}

struct RecordFile {
    int party = -1;
    int bw = 0;
    int shift = 0;
    uint64_t seed = 0;
    size_t boundary_count = 0;
    std::vector<uint64_t> y;
    std::vector<uint64_t> r;
    std::vector<uint64_t> output;
    std::vector<uint64_t> next_mask;
    std::vector<uint64_t> next_state;
    std::vector<uint64_t> opened_t;
};

bool write_file(const std::string &path, const Args &a,
                const std::vector<Input> &input,
                const std::vector<uint64_t> &output,
                const std::vector<uint64_t> &next_state,
                const std::vector<uint64_t> &opened_t) {
    std::ofstream file(path, std::ios::trunc);
    if (!file || output.size() != input.size() ||
        next_state.size() != input.size() || opened_t.size() != input.size()) {
        return false;
    }
    file << "RLPTRUNC3 " << a.party << ' ' << a.bw << ' ' << a.shift << ' '
         << a.seed << ' ' << kBoundaryCount << ' ' << input.size() << '\n';
    for (size_t i = 0; i < input.size(); ++i) {
        file << input[i].y << ' ' << (a.party == 0 ? input[i].r0 : input[i].r1)
             << ' ' << output[i] << ' '
             << (a.party == 0 ? input[i].s0 : input[i].s1) << ' '
             << next_state[i] << ' ' << opened_t[i] << '\n';
    }
    return bool(file);
}

bool read_file(const std::string &path, RecordFile &out) {
    std::ifstream file(path);
    std::string magic;
    size_t count = 0;
    if (!(file >> magic >> out.party >> out.bw >> out.shift >> out.seed >>
          out.boundary_count >> count) ||
        magic != "RLPTRUNC3" || (out.party != 0 && out.party != 1) ||
        out.bw <= 2 || out.bw > 32 || out.shift <= 0 ||
        out.shift >= out.bw || out.seed == 0 ||
        out.boundary_count != kBoundaryCount || count < kBoundaryCount ||
        count > kBoundaryCount + kMaxTrials) {
        return false;
    }
    const uint64_t ring_mask = mask(out.bw);
    const uint64_t output_mask = mask(out.bw - out.shift);
    out.y.resize(count);
    out.r.resize(count);
    out.output.resize(count);
    out.next_mask.resize(count);
    out.next_state.resize(count);
    out.opened_t.resize(count);
    for (size_t i = 0; i < count; ++i) {
        if (!(file >> out.y[i] >> out.r[i] >> out.output[i] >>
              out.next_mask[i] >> out.next_state[i] >> out.opened_t[i]) ||
            (out.y[i] & ~ring_mask) != 0 || (out.r[i] & ~ring_mask) != 0 ||
            (out.output[i] & ~output_mask) != 0 ||
            (out.next_mask[i] & ~output_mask) != 0 ||
            (out.next_state[i] & ~output_mask) != 0 ||
            (out.opened_t[i] & ~mask(out.shift)) != 0) {
            return false;
        }
    }
    std::string trailing;
    return !(file >> trailing);
}

bool same_header(const RecordFile &p0, const RecordFile &p1) {
    return p0.party == 0 && p1.party == 1 && p0.bw == p1.bw &&
           p0.shift == p1.shift && p0.seed == p1.seed &&
           p0.boundary_count == p1.boundary_count &&
           p0.y.size() == p1.y.size();
}

struct Validation {
    uint64_t exact_integer = 0;
    uint64_t rounded_down = 0;
    uint64_t rounded_up = 0;
    uint64_t former_predicate_witness = 0;
    uint64_t mismatch = 0;
};

Validation validate_records(const RecordFile &p0, const RecordFile &p1,
                            bool corrupt) {
    Validation result;
    const uint64_t ring_mask = mask(p0.bw);
    const uint64_t low_mask = mask(p0.shift);
    const uint64_t output_mask = mask(p0.bw - p0.shift);
    for (size_t i = 0; i < p0.y.size(); ++i) {
        bool bad = p0.y[i] != p1.y[i] ||
                   p0.next_state[i] != p1.next_state[i] ||
                   p0.opened_t[i] != p1.opened_t[i];
        const uint64_t r = (p0.r[i] + p1.r[i]) & ring_mask;
        const uint64_t r_low = r & low_mask;
        const uint64_t x = (p0.y[i] - r) & ring_mask;
        const uint64_t u = (p0.opened_t[i] - r_low) & low_mask;
        const uint64_t rho = low_mask - u;
        const uint64_t p1_output =
            p1.output[i] ^ ((corrupt && i == 0) ? uint64_t(1) : uint64_t(0));
        const uint64_t output = (p0.output[i] + p1_output) & output_mask;
        const uint64_t next_mask =
            (p0.next_mask[i] + p1.next_mask[i]) & output_mask;
        const uint64_t remasked_clear =
            (p0.next_state[i] - next_mask) & output_mask;
        const uint64_t expected = ((x + rho) >> p0.shift) & output_mask;
        const uint64_t down = x >> p0.shift;
        const uint64_t up = (down + 1) & output_mask;
        bad = bad || remasked_clear != output || output != expected;
        if ((x & low_mask) == 0) {
            ++result.exact_integer;
        } else if (expected == down) {
            ++result.rounded_down;
        } else if (expected == up) {
            ++result.rounded_up;
        } else {
            bad = true;
        }
        if (i + 1 == p0.boundary_count) {
            const uint64_t B = uint64_t(1) << p0.bw;
            const uint64_t old_wrapped =
                (low_mask - p0.opened_t[i]) < r_low ? 1 : 0;
            const uint64_t wrong_output =
                ((p0.y[i] >> p0.shift) - (r >> p0.shift) +
                 ((p0.y[i] & low_mask) > p0.opened_t[i] ? 1 : 0) -
                 old_wrapped) &
                output_mask;
            const bool witness =
                p0.y[i] == 0 && p0.r[i] == B - 1 &&
                p1.r[i] == (uint64_t(1) << (p0.shift - 1)) + 1 &&
                x == B - (uint64_t(1) << (p0.shift - 1)) &&
                output == expected && wrong_output != expected;
            if (witness) ++result.former_predicate_witness;
            bad = bad || !witness;
        }
        if (bad) ++result.mismatch;
    }
    return result;
}

int check(const Args &a) {
    RecordFile p0;
    RecordFile p1;
    const bool headers =
        read_file(a.prefix + "_p0.truncate", p0) &&
        read_file(a.prefix + "_p1.truncate", p1) && same_header(p0, p1);
    Validation exact;
    Validation corrupt;
    if (headers) {
        exact = validate_records(p0, p1, false);
        corrupt = validate_records(p0, p1, true);
    }
    const bool boundary =
        headers && p0.y.size() >= kBoundaryCount && exact.mismatch == 0;
    const bool corruption = headers && corrupt.mismatch > exact.mismatch;
    const bool former_predicate =
        headers && exact.former_predicate_witness == 1;
    const bool all = headers && boundary && corruption && former_predicate;
    if (a.csv_header) {
        std::printf(
            "bw,shift,records,boundary_records,exact_integer,rounded_down,"
            "rounded_up,mismatch,headers,boundaries,corruption_control,"
            "former_predicate_control,status\n");
    }
    std::printf("%d,%d,%zu,%zu,%llu,%llu,%llu,%llu,%s,%s,%s,%s,%s\n",
                headers ? p0.bw : 0, headers ? p0.shift : 0,
                headers ? p0.y.size() : 0,
                headers ? p0.boundary_count : 0,
                (unsigned long long)exact.exact_integer,
                (unsigned long long)exact.rounded_down,
                (unsigned long long)exact.rounded_up,
                (unsigned long long)exact.mismatch,
                headers ? "pass" : "FAIL", boundary ? "pass" : "FAIL",
                corruption ? "pass" : "FAIL",
                former_predicate ? "pass" : "FAIL", all ? "pass" : "FAIL");
    return all ? 0 : 1;
}

int party(const Args &a) {
    const std::string path =
        a.prefix + "_p" + std::to_string(a.party) + ".truncate";
    const std::string temporary = path + ".tmp";
    std::remove(path.c_str());
    std::remove(temporary.c_str());

    const std::vector<Input> input = make_inputs(a);
    std::vector<uint64_t> y(input.size());
    std::vector<uint64_t> own_mask(input.size());
    std::vector<uint64_t> own_next_mask(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        y[i] = input[i].y;
        own_mask[i] = a.party == 0 ? input[i].r0 : input[i].r1;
        own_next_mask[i] = a.party == 0 ? input[i].s0 : input[i].s1;
    }
    if (a.invalid_share_party == a.party) {
        own_mask[0] |= uint64_t(1) << a.bw;
    }

    ringlpn_2pc::SecureTruncateParams params;
    params.sid = a.seed;
    params.bw = a.bw;
    params.shift = a.shift + ((a.mismatch_shift && a.party == 1) ? 1 : 0);
    params.count = input.size();
    if (!derive_id(a, params.correlation_id)) return 1;
    if (a.mismatch_correlation && a.party == 1) params.correlation_id[0] ^= 1;

    PartyChannel channel(a.party, a.host, a.port, /*defer_ot_setup=*/true);
    PartyRandom random;
    std::vector<uint64_t> output;
    std::vector<uint64_t> next_state;
    std::vector<uint64_t> opened_t;
    ringlpn_2pc::SecureTruncateCounters counters;
    const bool truncated = ringlpn_2pc::secure_stochastic_truncate_batch(
        params, y, own_mask, own_next_mask, channel, random, output,
        next_state, counters, &opened_t);

    if (a.expect_reject) {
        const bool clean_reject =
            !truncated && output.empty() && next_state.empty() &&
            channel.costs.base_ots == 0 &&
            channel.costs.string_ots_128 == 0 &&
            channel.costs.bit_triples == 0 &&
            counters.preflight_bytes_sent == 57;
        std::fprintf(stderr,
                     "[secure-truncate] party %d negative preflight: %s\n",
                     a.party, clean_reject ? "rejected before OT" : "FAIL");
        return clean_reject ? 0 : 1;
    }

    const uint64_t n = input.size();
    const uint64_t expected_triples = n * uint64_t(2 * a.shift - 1);
    const uint64_t expected_edabit_bits = n * uint64_t(a.shift);
    const uint64_t expected_dabits = 2 * n;
    const uint64_t output_bits = uint64_t(a.bw - a.shift);
    const bool accounting =
        truncated && counters.truncations == n && counters.handoffs == n &&
        counters.edabit_bits == expected_edabit_bits &&
        counters.dabits == expected_dabits &&
        counters.triples == expected_triples &&
        counters.logical_opened_bits ==
            n * (5 * uint64_t(a.shift) + output_bits) &&
        counters.meaningful_share_bits ==
            2 * n * (5 * uint64_t(a.shift) + output_bits) &&
        counters.online_dependency_rounds == uint64_t(2 * a.shift + 2) &&
        counters.post_mask_dependencies == n * uint64_t(2 * a.shift + 3) &&
        channel.costs.string_ots_128 == expected_edabit_bits + expected_dabits &&
        channel.costs.bit_triples == expected_triples &&
        channel.costs.triple_ots == 2 * expected_triples;

    if (truncated) channel.finish_ots();
    const bool staged =
        accounting &&
        write_file(temporary, a, input, output, next_state, opened_t);
    channel.sync();
    const uint8_t mine_staged = staged ? 1 : 0;
    uint8_t peer_staged = 0;
    channel.exchange_bytes(&mine_staged, &peer_staged, 1);
    const bool renamed = staged && peer_staged == 1 &&
                         std::rename(temporary.c_str(), path.c_str()) == 0;
    const uint8_t mine_renamed = renamed ? 1 : 0;
    uint8_t peer_renamed = 0;
    channel.exchange_bytes(&mine_renamed, &peer_renamed, 1);
    const bool published = renamed && peer_renamed == 1;
    if (!published) {
        std::remove(temporary.c_str());
        std::remove(path.c_str());
    }

    std::fprintf(
        stderr,
        "[secure-truncate] party %d bw=%d shift=%d n=%zu: %llu triples, "
        "%llu edaBit bits, %llu daBits, %llu logical opened bits, "
        "%llu online rounds; accounting %s; output %s\n",
        a.party, a.bw, a.shift, input.size(),
        (unsigned long long)counters.triples,
        (unsigned long long)counters.edabit_bits,
        (unsigned long long)counters.dabits,
        (unsigned long long)counters.logical_opened_bits,
        (unsigned long long)counters.online_dependency_rounds,
        accounting ? "pass" : "FAIL", published ? path.c_str() : "FAIL");
    return published ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    const Args args = parse(argc, argv);
    return args.check ? check(args) : party(args);
}
