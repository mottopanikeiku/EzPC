// Two-process secure Z_M -> Z_{2^bw} conversion over the existing SCI/IKNP
// transport. P_i holds z_i in [0,M) and receives r_i such that
// r_0+r_1 = (z_0+z_1 mod M) mod 2^bw. The wrap bit is never opened.
//
// Correlations are real OT-backed correlations, not dealer records. For a daBit
// over Z_{2^k}, P0 independently samples d0 and uniform a0, then sends by OT
// m0=d0-a0 and m1=(1-d0)-a0. P1 selects with independently sampled d1 and
// takes a1=m[d1]. Thus a0+a1=d0 XOR d1, with each party's Boolean/arithmetic
// shares distributed exactly as F_DABIT requires. An edaBit is ell such daBits
// over Z_{2^ell}, combined arithmetically as sum_j 2^j*a_{i,j}; a0 of bit zero
// makes the first arithmetic share uniform independently of all Boolean shares.
// Boolean triples use generate_bit_triples (two one-bit OTs per triple).
//
// Security boundary: semi-honest OT-hybrid protocol. SCI IKNP is OT extension,
// not silent OT; NetIO is plain unauthenticated TCP. This is an OT-backed partial
// S6 loopback artifact, not PCG-backed M3, prefix/log-round conversion, production
// integration, authenticated deployment, end-to-end realization, or malicious
// security. --check is TEST-ONLY and reads both post-protocol party files.

#include "secure_convert.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using U128 = ringlpn_2pc::U128;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;

constexpr uint64_t P0 = 4611686018326724609ULL;
constexpr uint64_t P1 = 4611686018309947393ULL;
constexpr char MAGIC[8] = {'R', 'L', 'P', 'N', 'C', 'V', 'T', '1'};
constexpr int kMaxTrials = 4096;
constexpr int kMaxInner = 4096;
constexpr int kMaxSelftest = 1024;
constexpr uint64_t kMaxRecords =
    uint64_t(2) * kMaxTrials + kMaxTrials + 4;
constexpr uint64_t kRecordSize = 1 + 16 + 8;

enum class Mode : uint8_t {
    Boundary = 0,
    Random = 1,
    Forced = 2,
    Layer = 3,
};

U128 modulus128() {
    return U128(P0) * U128(P1);
}
// This no-wrap condition applies only to the harness's generated Layer/FC
// workload. It is not a precondition of generic Z_M conversion, whose shares
// may be arbitrary elements of [0, M).
bool layer_workload_admissible(int qbits, int bw, uint64_t inner) {
    if ((qbits != 64 && qbits != 128) || bw <= 2 || bw > 32 ||
        inner == 0) {
        return false;
    }
    const U128 M = qbits == 128 ? modulus128() : U128(P0);
    const unsigned shift = unsigned(2 * bw + 2);
    const U128 scale = U128(1) << shift;
    return U128(inner) <= (M - 1) / scale;
}

int bitlen(U128 x) {
    int n = 0;
    while (x) {
        ++n;
        x >>= 1;
    }
    return n;
}

U128 mask(int k) {
    return k == 128 ? ~U128(0) : (U128(1) << k) - 1;
}

U128 red(U128 x, int k) {
    return x & mask(k);
}


uint64_t red64(U128 x, int k) {
    return uint64_t(red(x, k));
}

uint64_t add64(uint64_t a, uint64_t b, int k) {
    return red64(U128(a) + b, k);
}

U128 uniform(U128 modulus, PartyRandom &rng) {
    if (modulus == 0) {
        std::fprintf(stderr, "uniform modulus is zero\n");
        std::exit(2);
    }
    const U128 threshold = (U128(0) - modulus) % modulus;
    U128 x = 0;
    do {
        x = rng.u128();
    } while (x < threshold);
    return x % modulus;
}

struct Args {
    int party = 0;
    int port = 42600;
    int qbits = 64;
    int bw = 16;
    int trials = 32;
    int forced = 8;
    int inner = 8;
    int selftest = 4;
    uint64_t bound = 255;
    uint64_t input_seed = 1;
    std::string host = "127.0.0.1";
    std::string prefix = "two_party_secure_convert";
    bool check = false;
    bool header = false;
};

Args parse(int ac, char **av) {
    Args a;
    for (int i = 1; i < ac; ++i) {
        std::string k(av[i]);
        auto next = [&]() {
            if (++i >= ac) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return std::string(av[i]);
        };
        if (k == "--party") {
            a.party = std::atoi(next().c_str());
        } else if (k == "--host") {
            a.host = next();
        } else if (k == "--port") {
            a.port = std::atoi(next().c_str());
        } else if (k == "--qbits") {
            a.qbits = std::atoi(next().c_str());
        } else if (k == "--bw") {
            a.bw = std::atoi(next().c_str());
        } else if (k == "--trials") {
            a.trials = std::atoi(next().c_str());
        } else if (k == "--forced-wraps") {
            a.forced = std::atoi(next().c_str());
        } else if (k == "--inner") {
            a.inner = std::atoi(next().c_str());
        } else if (k == "--value-bound") {
            a.bound = std::strtoull(next().c_str(), nullptr, 10);
        } else if (k == "--input-seed") {
            a.input_seed = std::strtoull(next().c_str(), nullptr, 10);
        } else if (k == "--selftest") {
            a.selftest = std::atoi(next().c_str());
        } else if (k == "--out-prefix") {
            a.prefix = next();
        } else if (k == "--check") {
            a.check = true;
        } else if (k == "--csv-header") {
            a.header = true;
        } else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            std::exit(2);
        }
    }
    const uint64_t max_value =
        (a.bw >= 3 && a.bw <= 32) ? ((uint64_t(1) << a.bw) - 1) : 0;
    if ((a.party != 0 && a.party != 1) || a.port < 1 || a.port > 65534 ||
        a.host.empty() || a.prefix.empty() ||
        (a.qbits != 64 && a.qbits != 128) || a.bw <= 2 || a.bw > 32 ||
        a.trials < 1 || a.trials > kMaxTrials ||
        a.forced < 1 || a.forced > kMaxTrials ||
        a.inner < 1 || a.inner > kMaxInner ||
        a.selftest < 0 || a.selftest > kMaxSelftest ||
        a.bound > max_value) {
        std::fprintf(stderr, "invalid arguments\n");
        std::exit(2);
    }
    if (!layer_workload_admissible(a.qbits, a.bw, uint64_t(a.inner))) {
        std::fprintf(
            stderr,
            "invalid harness Layer/FC workload: require "
            "inner*2^(2*bw+2)<M\n");
        std::exit(2);
    }
    return a;
}
bool agree_public_parameters(PartyChannel &ch, const Args &a) {
    // Canonical fixed-width little-endian encoding, in protocol-work order:
    // qbits,bw,trials,forced,inner,bound,input_seed,selftest.
    uint8_t mine[40] = {};
    uint8_t theirs[40] = {};
    size_t pos = 0;
    auto encode32 = [&](uint32_t x) {
        for (int i = 0; i < 4; ++i) {
            mine[pos++] = uint8_t(x >> (8 * i));
        }
    };
    auto encode64 = [&](uint64_t x) {
        for (int i = 0; i < 8; ++i) {
            mine[pos++] = uint8_t(x >> (8 * i));
        }
    };
    encode32(uint32_t(a.qbits));
    encode32(uint32_t(a.bw));
    encode32(uint32_t(a.trials));
    encode32(uint32_t(a.forced));
    encode32(uint32_t(a.inner));
    encode64(a.bound);
    encode64(a.input_seed);
    encode32(uint32_t(a.selftest));
    ch.exchange_bytes(mine, theirs, sizeof(mine));
    return std::memcmp(mine, theirs, sizeof(mine)) == 0;
}

struct Input {
    Mode mode;
    U128 z;
};

// TEST-ONLY deterministic share generation. The public input seed and each
// party's share are written to separate records solely for the offline checker;
// they are not protocol randomness or a private-input deployment interface.
std::vector<Input> make_inputs(const Args &a, U128 M) {
    PartyRandom r(a.input_seed * 1000003ULL + uint64_t(a.party));
    std::vector<Input> v;
    v.reserve(size_t(2 * a.trials + a.forced + 4));
    U128 x0[4] = {0, M - 1, 1, M - 1};
    U128 x1[4] = {0, 0, M - 1, M - 1};
    for (int i = 0; i < 4; ++i) {
        v.push_back({Mode::Boundary, a.party ? x1[i] : x0[i]});
    }
    for (int i = 0; i < a.trials; ++i) {
        v.push_back({Mode::Random, uniform(M, r)});
    }
    U128 hi = (M + 1) / 2;
    for (int i = 0; i < a.forced; ++i) {
        v.push_back({Mode::Forced, hi + uniform(M - hi, r)});
    }
    for (int i = 0; i < a.trials; ++i) {
        U128 z = 0;
        for (int j = 0; j < a.inner; ++j) {
            uint64_t x = uint64_t(uniform(U128(a.bound) + 1, r));
            uint64_t y = uint64_t(uniform(U128(a.bound) + 1, r));
            z = (z + U128(x) * y) % M;
        }
        v.push_back({Mode::Layer, z});
    }
    return v;
}


bool correlation_selftest(PartyChannel &ch, int rounds, int ell, int bw,
                          PartyRandom &r) {
    if (rounds == 0) {
        return true;
    }
    auto e = ringlpn_2pc::secure_convert_detail::generate_edabits(
        ch, size_t(rounds), ell, r);
    auto d = ringlpn_2pc::secure_convert_detail::generate_dabits(
        ch, size_t(rounds), bw, r);
    std::vector<U128> mine(size_t(rounds) * 2);
    std::vector<U128> theirs(size_t(rounds) * 2);
    for (int i = 0; i < rounds; ++i) {
        mine[size_t(i) * 2] = e[size_t(i)].arithmetic;
        mine[size_t(i) * 2 + 1] = d[size_t(i)].arithmetic;
    }
    ch.exchange_bytes(reinterpret_cast<uint8_t *>(mine.data()),
                      reinterpret_cast<uint8_t *>(theirs.data()),
                      mine.size() * sizeof(U128));
    std::vector<uint8_t> bm(size_t(rounds) * (size_t(ell) + 1));
    std::vector<uint8_t> bt(bm.size());
    for (int i = 0; i < rounds; ++i) {
        for (int j = 0; j < ell; ++j) {
            bm[size_t(i) * (size_t(ell) + 1) + size_t(j)] =
                e[size_t(i)].bits[size_t(j)];
        }
        bm[size_t(i) * (size_t(ell) + 1) + size_t(ell)] = d[size_t(i)].bit;
    }
    ch.exchange_bytes(bm.data(), bt.data(), bm.size());
    for (int i = 0; i < rounds; ++i) {
        U128 R = 0;
        for (int j = 0; j < ell; ++j) {
            R |= U128((bm[size_t(i) * (size_t(ell) + 1) + size_t(j)] ^
                       bt[size_t(i) * (size_t(ell) + 1) + size_t(j)]) &
                      1)
                 << j;
        }
        if (red(mine[size_t(i) * 2] + theirs[size_t(i) * 2], ell) != R) {
            return false;
        }
        uint8_t bit =
            (bm[size_t(i) * (size_t(ell) + 1) + size_t(ell)] ^
             bt[size_t(i) * (size_t(ell) + 1) + size_t(ell)]) &
            1;
        if (red(mine[size_t(i) * 2 + 1] + theirs[size_t(i) * 2 + 1], bw) !=
            bit) {
            return false;
        }
    }
    return true;
}


void put32(std::ostream &o, uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        o.put(char((x >> (8 * i)) & 255));
    }
}

void put64(std::ostream &o, uint64_t x) {
    for (int i = 0; i < 8; ++i) {
        o.put(char((x >> (8 * i)) & 255));
    }
}

void put128(std::ostream &o, U128 x) {
    put64(o, uint64_t(x));
    put64(o, uint64_t(x >> 64));
}

bool get32(std::istream &f, uint32_t &x) {
    x = 0;
    for (int i = 0; i < 4; ++i) {
        int c = f.get();
        if (c == EOF) {
            return false;
        }
        x |= uint32_t(uint8_t(c)) << (8 * i);
    }
    return true;
}

bool get64(std::istream &f, uint64_t &x) {
    x = 0;
    for (int i = 0; i < 8; ++i) {
        int c = f.get();
        if (c == EOF) {
            return false;
        }
        x |= uint64_t(uint8_t(c)) << (8 * i);
    }
    return true;
}

bool get128(std::istream &f, U128 &x) {
    uint64_t l, h;
    if (!get64(f, l) || !get64(f, h)) {
        return false;
    }
    x = U128(l) | (U128(h) << 64);
    return true;
}

struct Record {
    Mode mode;
    U128 z;
    uint64_t r;
};

struct File {
    uint32_t party = 0;
    uint32_t qbits = 0;
    uint32_t bw = 0;
    uint32_t ell = 0;
    uint32_t trials = 0;
    uint32_t forced = 0;
    uint32_t inner = 0;
    uint64_t bound = 0;
    uint64_t seed = 0;
    std::vector<Record> records;
};

bool write_file(const std::string &p, const Args &a, int ell,
                const std::vector<Input> &in,
                const std::vector<uint64_t> &o) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f.write(MAGIC, 8);
    put32(f, 1);
    put32(f, a.party);
    put32(f, a.qbits);
    put32(f, a.bw);
    put32(f, ell);
    put32(f, a.trials);
    put32(f, a.forced);
    put32(f, a.inner);
    put64(f, a.bound);
    put64(f, a.input_seed);
    put64(f, in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        f.put(char(in[i].mode));
        put128(f, in[i].z);
        put64(f, o[i]);
    }
    return bool(f);
}

bool read_file(const std::string &p, File &x) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        return false;
    }
    char magic[8];
    f.read(magic, 8);
    uint32_t v;
    uint64_t n;
    if (!f || std::memcmp(magic, MAGIC, 8) || !get32(f, v) || v != 1 ||
        !get32(f, x.party) || !get32(f, x.qbits) || !get32(f, x.bw) ||
        !get32(f, x.ell) || !get32(f, x.trials) || !get32(f, x.forced) ||
        !get32(f, x.inner) || !get64(f, x.bound) || !get64(f, x.seed) ||
        !get64(f, n)) {
        return false;
    }
    if (x.party > 1 || (x.qbits != 64 && x.qbits != 128) || x.bw <= 2 ||
        x.bw > 32 || x.trials == 0 || x.trials > kMaxTrials ||
        x.forced == 0 || x.forced > kMaxTrials || x.inner == 0 ||
        x.inner > kMaxInner) {
        return false;
    }
    const uint64_t max_value = (uint64_t(1) << x.bw) - 1;
    if (x.bound > max_value ||
        !layer_workload_admissible(int(x.qbits), int(x.bw), x.inner)) {
        return false;
    }
    const U128 M = x.qbits == 128 ? modulus128() : U128(P0);
    // The maxima above make this closed form overflow-free. The independent
    // cap ensures no serialized count can request more records than parse()
    // could generate.
    const uint64_t expected =
        uint64_t(2) * uint64_t(x.trials) + uint64_t(x.forced) + 4;
    if (x.ell != uint32_t(bitlen(2 * M - 1)) || n != expected ||
        n > kMaxRecords) {
        return false;
    }
    const std::streampos records_begin = f.tellg();
    f.seekg(0, std::ios::end);
    const std::streampos records_end = f.tellg();
    if (records_begin == std::streampos(-1) ||
        records_end == std::streampos(-1)) {
        return false;
    }
    const std::streamoff remaining = records_end - records_begin;
    if (remaining < 0 || uint64_t(remaining) != n * kRecordSize) {
        return false;
    }
    f.seekg(records_begin);
    if (!f) {
        return false;
    }
    x.records.resize(size_t(n));
    for (auto &r : x.records) {
        int m = f.get();
        if (m < 0 || m > 3 || !get128(f, r.z) || !get64(f, r.r) ||
            r.z >= M || r.r != red64(r.r, int(x.bw))) {
            return false;
        }
        r.mode = Mode(m);
    }
    return f.peek() == EOF;
}

bool same(const File &a, const File &b) {
    return a.party == 0 && b.party == 1 && a.qbits == b.qbits &&
           a.bw == b.bw && a.ell == b.ell && a.trials == b.trials &&
           a.forced == b.forced && a.inner == b.inner && a.bound == b.bound &&
           a.seed == b.seed && a.records.size() == b.records.size();
}

struct Stats {
    uint64_t count[4] = {0, 0, 0, 0};
    uint64_t mismatch[4] = {0, 0, 0, 0};
};

Stats validate(const File &a, const File &b, U128 M, bool corrupt) {
    Stats s;
    for (size_t i = 0; i < a.records.size(); ++i) {
        size_t m = size_t(a.records[i].mode);
        ++s.count[m];
        bool bad = a.records[i].mode != b.records[i].mode ||
                   a.records[i].z >= M || b.records[i].z >= M;
        if (!bad) {
            U128 sum = a.records[i].z + b.records[i].z;
            U128 clear = sum >= M ? sum - M : sum;
            uint64_t r1 = b.records[i].r ^ ((corrupt && i == 0) ? 1ULL : 0ULL);
            bad = red64(clear, a.bw) != add64(a.records[i].r, r1, a.bw);
        }
        if (bad) {
            ++s.mismatch[m];
        }
    }
    return s;
}

int check(const Args &a) {
    File p0, p1;
    bool ok = read_file(a.prefix + "_p0.convert", p0) &&
              read_file(a.prefix + "_p1.convert", p1) && same(p0, p1);
    Stats s, c;
    if (ok) {
        U128 M = p0.qbits == 128 ? modulus128() : U128(P0);
        s = validate(p0, p1, M, false);
        c = validate(p0, p1, M, true);
    }
    uint64_t mis = 0, cmis = 0;
    for (int i = 0; i < 4; ++i) {
        mis += s.mismatch[i];
        cmis += c.mismatch[i];
    }
    bool counts = ok && s.count[0] == 4 && s.count[1] == p0.trials &&
                  s.count[2] == p0.forced && s.count[3] == p0.trials;
    bool corrupt = ok && cmis > mis;
    bool exact = ok && counts && mis == 0;
    bool all = exact && corrupt;
    if (a.header) {
        std::printf(
            "requested_qbits,actual_qbits,bw,ell,trials,forced_wraps,inner,"
            "boundary_trials,boundary_mismatch,random_trials,random_mismatch,"
            "forced_trials,forced_mismatch,layer_trials,layer_mismatch,"
            "conversions,headers,corruption_control,bit_exact_match,status\n");
    }
    std::printf(
        "%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%llu,%s,%s,%s,%s\n",
        ok ? p0.qbits : 0, ok ? (p0.qbits == 128 ? 124 : 62) : 0,
        ok ? p0.bw : 0, ok ? p0.ell : 0, ok ? p0.trials : 0,
        ok ? p0.forced : 0, ok ? p0.inner : 0,
        (unsigned long long)s.count[0], (unsigned long long)s.mismatch[0],
        (unsigned long long)s.count[1], (unsigned long long)s.mismatch[1],
        (unsigned long long)s.count[2], (unsigned long long)s.mismatch[2],
        (unsigned long long)s.count[3], (unsigned long long)s.mismatch[3],
        (unsigned long long)(ok ? p0.records.size() : 0), ok ? "pass" : "FAIL",
        corrupt ? "pass" : "FAIL", exact ? "pass" : "FAIL",
        all ? "pass" : "FAIL");
    return all ? 0 : 1;
}

int party(const Args &a) {
    U128 M = a.qbits == 128 ? modulus128() : U128(P0);
    int ell = bitlen(2 * M - 1);
    const std::string path =
        a.prefix + "_p" + std::to_string(a.party) + ".convert";
    const std::string temporary_path = path + ".tmp";
    std::remove(path.c_str());
    std::remove(temporary_path.c_str());


    // Open only the two raw sockets first. Public work parameters must agree
    // before base OTs, TEST-ONLY selftest correlations, production
    // correlations, or output generation.
    PartyChannel ch(a.party, a.host, a.port, /*defer_ot_setup=*/true);
    const uint64_t channel_open_bytes = ch.bytes_sent();
    const uint64_t channel_open_switches = ch.direction_switches();
    const uint64_t ab = ch.bytes_sent();
    const uint64_t as = ch.direction_switches();
    const bool agreed = agree_public_parameters(ch, a);
    auto in = make_inputs(a, M);
    const size_t n = in.size();
    std::vector<U128> own_z(n);
    for (size_t i = 0; i < n; ++i) {
        own_z[i] = in[i].z;
    }
    ringlpn_2pc::SecureConvertParams params{
        a.input_seed, a.qbits, a.bw, n};
    const bool locally_valid =
        ringlpn_2pc::validate_secure_convert_inputs(params, own_z);
    const uint8_t mine_valid = locally_valid ? 1 : 0;
    uint8_t peer_valid = 0;
    ch.exchange_bytes(&mine_valid, &peer_valid, 1);
    const bool inputs_valid = locally_valid && peer_valid == 1;
    const uint64_t abytes = ch.bytes_sent() - ab;
    const uint64_t asw = ch.direction_switches() - as;
    if (!agreed || !inputs_valid) {
        std::fprintf(stderr,
                     "[two-party-convert] public-parameter or local-input "
                     "validation mismatch; aborting before OT setup and "
                     "output\n");
        return 2;
    }

    ch.setup_ots();
    const uint64_t setup_bytes =
        channel_open_bytes + ch.setup_bytes_sent();
    const uint64_t setup_switches =
        channel_open_switches + ch.setup_direction_switches();

    PartyRandom rng;
    const uint64_t sb = ch.bytes_sent();
    const uint64_t ss = ch.direction_switches();
    const auto st = std::chrono::steady_clock::now();
    const bool self = correlation_selftest(ch, a.selftest, ell, a.bw, rng);
    const double sus = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - st)
                           .count();
    const uint64_t sbytes = ch.bytes_sent() - sb;
    const uint64_t ssw = ch.direction_switches() - ss;

    // Production accounting deliberately excludes the TEST-ONLY selftest,
    // whose complete transport and wall time remain reported above.
    const uint64_t base = ch.costs.base_ots;
    ch.costs = ringlpn_2pc::Counters{};
    ch.costs.base_ots = base;

    ringlpn_2pc::SecureConvertCounters cost;
    std::vector<uint64_t> out;
    const bool converted = ringlpn_2pc::secure_convert_batch(
        params, own_z, ch, rng, out, cost);
    const uint64_t pbytes = cost.preflight_bytes_sent;
    const uint64_t psw = cost.preflight_direction_switches;
    const double cus = cost.correlation_microseconds;
    const uint64_t cbytes = cost.correlation_bytes_sent;
    const uint64_t csw = cost.correlation_direction_switches;
    const double ous = cost.online_microseconds;
    const uint64_t obytes = cost.online_bytes_sent;
    const uint64_t osw = cost.online_direction_switches;

    const uint64_t et = n * size_t(2 * ell - 2);
    const uint64_t elog = n * size_t(5 * ell - 3);
    const uint64_t erev = n * size_t(10 * ell - 6);
    const uint64_t epost = n * size_t(2 * ell - 1);
    const uint64_t edot = n * size_t(ell + 1);
    bool accounting =
        ch.costs.string_ots_128 == edot && ch.costs.triple_ots == 2 * et &&
        ch.costs.bit_triples == et && cost.conversions == n &&
        cost.edabit_bits == n * size_t(ell) && cost.dabits == n &&
        cost.triples == et && cost.logical_opened_bits == elog &&
        cost.meaningful_share_bits == erev &&
        cost.post_mask_dependencies == epost;
    const bool staged =
        agreed && self && converted && accounting &&
        write_file(temporary_path, a, ell, in, out);

    const uint64_t fb = ch.bytes_sent();
    const uint64_t fs = ch.direction_switches();
    ch.sync();
    const uint8_t mine_publishable = staged ? 1 : 0;
    uint8_t peer_publishable = 0;
    ch.exchange_bytes(&mine_publishable, &peer_publishable, 1);
    const bool may_rename = staged && peer_publishable == 1;
    const bool locally_renamed =
        may_rename &&
        std::rename(temporary_path.c_str(), path.c_str()) == 0;
    const uint8_t mine_renamed = locally_renamed ? 1 : 0;
    uint8_t peer_renamed = 0;
    ch.exchange_bytes(&mine_renamed, &peer_renamed, 1);
    const uint64_t fbytes = ch.bytes_sent() - fb;
    const uint64_t fsw = ch.direction_switches() - fs;
    const bool wrote = locally_renamed && peer_renamed == 1;
    if (!wrote) {
        std::remove(temporary_path.c_str());
        std::remove(path.c_str());
    }


    // "protocol" is agreement + selftest + conversion preflight +
    // correlation + online + final synchronization/publication agreement.
    // "transport" is setup + protocol; both sums are explicit and checked
    // against the channel's monotonic counters.
    const uint64_t protocol_bytes =
        abytes + sbytes + pbytes + cbytes + obytes + fbytes;
    const uint64_t protocol_switches =
        asw + ssw + psw + csw + osw + fsw;
    const uint64_t transport_bytes = setup_bytes + protocol_bytes;
    const uint64_t transport_switches = setup_switches + protocol_switches;
    accounting =
        accounting && transport_bytes == ch.bytes_sent() &&
        transport_switches == ch.direction_switches();
    const bool all = agreed && self && converted && accounting && wrote;
    const double d = double(n);

    if (a.header) {
        std::printf(
            "party,requested_qbits,actual_qbits,bw,ell,conversions,"
            "and_triples_per_conv,edabit_bits_per_conv,dabits_per_conv,"
            "dabit_ots_per_conv,triple_ots_per_conv,"
            "logical_opened_bits_per_conv,meaningful_share_bits_per_conv,"
            "post_mask_dependency_stages_per_conv,base_ots,setup_bytes_sent,"
            "setup_direction_switches,agreement_bytes_sent_batch,"
            "agreement_direction_switches_batch,selftest_bytes_sent_batch,"
            "selftest_direction_switches_batch,preflight_bytes_sent_batch,"
            "preflight_direction_switches_batch,correlation_bytes_sent_batch,"
            "correlation_direction_switches_batch,online_bytes_sent_batch,"
            "online_direction_switches_batch,final_sync_bytes_sent_batch,"
            "final_sync_direction_switches_batch,protocol_bytes_sent_batch,"
            "protocol_direction_switches_batch,transport_bytes_sent_batch,"
            "transport_direction_switches_batch,selftest_us_batch,"
            "correlation_us_batch,online_us_batch,public_parameter_agreement,"
            "correlation_selftest,transcript_accounting,status\n");
    }
    std::printf(
        "%d,%d,%d,%d,%d,%llu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%.1f,%.1f,%.1f,%s,%s,%s,%s\n",
        a.party, a.qbits, a.qbits == 128 ? 124 : 62, a.bw, ell,
        (unsigned long long)n, cost.triples / d, cost.edabit_bits / d,
        cost.dabits / d, ch.costs.string_ots_128 / d,
        ch.costs.triple_ots / d, cost.logical_opened_bits / d,
        cost.meaningful_share_bits / d, cost.post_mask_dependencies / d,
        (unsigned long long)ch.costs.base_ots,
        (unsigned long long)setup_bytes,
        (unsigned long long)setup_switches,
        (unsigned long long)abytes,
        (unsigned long long)asw,
        (unsigned long long)sbytes,
        (unsigned long long)ssw,
        (unsigned long long)pbytes,
        (unsigned long long)psw,
        (unsigned long long)cbytes,
        (unsigned long long)csw,
        (unsigned long long)obytes,
        (unsigned long long)osw,
        (unsigned long long)fbytes,
        (unsigned long long)fsw,
        (unsigned long long)protocol_bytes,
        (unsigned long long)protocol_switches,
        (unsigned long long)transport_bytes,
        (unsigned long long)transport_switches, sus, cus, ous,
        agreed ? "pass" : "FAIL", self ? "pass" : "FAIL",
        accounting ? "pass" : "FAIL", all ? "pass" : "FAIL");
    std::fprintf(
        stderr,
        "[two-party-convert] party %d q%d bw=%d n=%zu: "
        "setup %llu/%llu; agreement %llu/%llu; selftest %llu/%llu; "
        "preflight %llu/%llu; correlations %llu/%llu; online %llu/%llu; "
        "final-sync %llu/%llu; transport %llu/%llu; selftest %s "
        "accounting %s; output %s\n",
        a.party, a.qbits, a.bw, n,
        (unsigned long long)setup_bytes,
        (unsigned long long)setup_switches,
        (unsigned long long)abytes,
        (unsigned long long)asw,
        (unsigned long long)sbytes,
        (unsigned long long)ssw,
        (unsigned long long)pbytes,
        (unsigned long long)psw,
        (unsigned long long)cbytes,
        (unsigned long long)csw,
        (unsigned long long)obytes,
        (unsigned long long)osw,
        (unsigned long long)fbytes,
        (unsigned long long)fsw,
        (unsigned long long)transport_bytes,
        (unsigned long long)transport_switches, self ? "pass" : "FAIL",
        accounting ? "pass" : "FAIL", path.c_str());
    return all ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    Args a = parse(argc, argv);
    return a.check ? check(a) : party(a);
}
