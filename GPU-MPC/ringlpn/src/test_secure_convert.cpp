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

#include "two_party_ot.h"

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
using ringlpn_2pc::BitTriple;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;

constexpr uint64_t P0 = 4611686018326724609ULL;
constexpr uint64_t P1 = 4611686018309947393ULL;
constexpr char MAGIC[8] = {'R', 'L', 'P', 'N', 'C', 'V', 'T', '1'};

enum class Mode : uint8_t {
    Boundary = 0,
    Random = 1,
    Forced = 2,
    Layer = 3,
};

U128 modulus128() {
    return U128(P0) * U128(P1);
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

U128 sub(U128 a, U128 b, int k) {
    return red(a - b, k);
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
    constexpr int kMaxTrials = 4096;
    constexpr int kMaxInner = 4096;
    constexpr int kMaxSelftest = 1024;
    const uint64_t max_value =
        (a.bw >= 3 && a.bw <= 32) ? ((uint64_t(1) << a.bw) - 1) : 0;
    if ((a.party != 0 && a.party != 1) || a.port < 1 || a.port > 65535 ||
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
    return a;
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

struct Dabit {
    uint8_t bit = 0;
    U128 arithmetic = 0;
};

std::vector<Dabit> gen_dabits(PartyChannel &ch, size_t n, int k,
                              PartyRandom &r) {
    std::vector<Dabit> o(n);
    std::vector<uint8_t> b(n);
    for (size_t i = 0; i < n; ++i) {
        b[i] = r.bit();
    }
    if (ch.is_p0()) {
        std::vector<U128> m0(n), m1(n);
        for (size_t i = 0; i < n; ++i) {
            U128 a0 = red(r.u128(), k);
            o[i] = {b[i], a0};
            m0[i] = sub(U128(b[i]), a0, k);
            m1[i] = sub(U128(1 - b[i]), a0, k);
        }
        ch.ot_send_128(m0, m1);
    } else {
        auto a1 = ch.ot_recv_128(b);
        for (size_t i = 0; i < n; ++i) {
            o[i] = {b[i], red(a1[i], k)};
        }
    }
    return o;
}

struct Edabit {
    U128 arithmetic = 0;
    std::vector<uint8_t> bits;
};

std::vector<Edabit> gen_edabits(PartyChannel &ch, size_t n, int ell,
                                PartyRandom &r) {
    auto d = gen_dabits(ch, n * size_t(ell), ell, r);
    std::vector<Edabit> o(n);
    for (size_t i = 0; i < n; ++i) {
        o[i].bits.resize(size_t(ell));
        for (int j = 0; j < ell; ++j) {
            const auto &x = d[i * size_t(ell) + size_t(j)];
            o[i].bits[size_t(j)] = x.bit;
            o[i].arithmetic =
                red(o[i].arithmetic + (x.arithmetic << j), ell);
        }
    }
    return o;
}

bool correlation_selftest(PartyChannel &ch, int rounds, int ell, int bw,
                          PartyRandom &r) {
    if (rounds == 0) {
        return true;
    }
    auto e = gen_edabits(ch, size_t(rounds), ell, r);
    auto d = gen_dabits(ch, size_t(rounds), bw, r);
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

struct Costs {
    uint64_t conversions = 0;
    uint64_t edabit_bits = 0;
    uint64_t triples = 0;
    uint64_t dabits = 0;
    uint64_t logical = 0;
    uint64_t sent = 0;
    uint64_t recv = 0;
    uint64_t post = 0;

    uint64_t meaningful_share_bits() const {
        return sent + recv;
    }
};

std::vector<uint8_t> and_batch(const std::vector<uint8_t> &x,
                               const std::vector<uint8_t> &y,
                               const std::vector<BitTriple> &t, size_t &pos,
                               PartyChannel &ch, Costs &c) {
    size_t n = x.size();
    std::vector<uint8_t> m(n), q(n), o(n);
    for (size_t i = 0; i < n; ++i) {
        m[i] = uint8_t(((x[i] ^ t[pos + i].a) & 1) |
                       (((y[i] ^ t[pos + i].b) & 1) << 1));
    }
    ch.exchange_bytes(m.data(), q.data(), n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t z = m[i] ^ q[i];
        uint8_t d = z & 1;
        uint8_t e = (z >> 1) & 1;
        o[i] = uint8_t((t[pos + i].c ^ (d & t[pos + i].b) ^
                        (e & t[pos + i].a) ^
                        (ch.is_p0() ? (d & e) : 0)) &
                       1);
    }
    pos += n;
    c.triples += n;
    c.logical += 2 * n;
    c.sent += 2 * n;
    c.recv += 2 * n;
    c.post += n;
    return o;
}

struct Ripple {
    std::vector<uint8_t> carry;
    std::vector<uint8_t> sum;
};

Ripple ripple(const std::vector<U128> &k, const std::vector<uint8_t> &x,
              int cin, int ell, bool want, const std::vector<BitTriple> &t,
              size_t &pos, PartyChannel &ch, Costs &c) {
    size_t n = k.size();
    Ripple r;
    r.carry.assign(n, ch.is_p0() ? uint8_t(cin) : 0);
    if (want) {
        r.sum.assign(n * size_t(ell), 0);
    }
    for (int j = 0; j < ell; ++j) {
        std::vector<uint8_t> xj(n), old = r.carry, next;
        for (size_t i = 0; i < n; ++i) {
            xj[i] = x[i * size_t(ell) + size_t(j)];
            uint8_t b = uint8_t((k[i] >> j) & 1);
            if (want) {
                r.sum[i * size_t(ell) + size_t(j)] =
                    uint8_t(xj[i] ^ old[i] ^ (ch.is_p0() ? b : 0));
            }
        }
        if (j == 0) {
            next.resize(n);
            for (size_t i = 0; i < n; ++i) {
                next[i] = uint8_t(xj[i] & cin);
            }
        } else {
            next = and_batch(xj, old, t, pos, ch, c);
        }
        for (size_t i = 0; i < n; ++i) {
            if ((k[i] >> j) & 1) {
                next[i] ^= uint8_t(xj[i] ^ old[i]);
            }
        }
        r.carry.swap(next);
    }
    return r;
}

std::vector<uint64_t> convert(const Args &a, U128 M,
                              const std::vector<Input> &in,
                              const std::vector<Edabit> &eda,
                              const std::vector<Dabit> &da,
                              const std::vector<BitTriple> &t,
                              PartyChannel &ch, Costs &c) {
    size_t n = in.size();
    int ell = bitlen(2 * M - 1);
    c.conversions = n;
    c.edabit_bits = n * size_t(ell);
    c.dabits = n;
    std::vector<U128> m(n), q(n), A(n);
    for (size_t i = 0; i < n; ++i) {
        m[i] = red(in[i].z + eda[i].arithmetic, ell);
    }
    ch.exchange_bytes(reinterpret_cast<uint8_t *>(m.data()),
                      reinterpret_cast<uint8_t *>(q.data()), n * sizeof(U128));
    for (size_t i = 0; i < n; ++i) {
        A[i] = red(m[i] + q[i], ell);
    }
    c.logical += n * ell;
    c.sent += n * ell;
    c.recv += n * ell;

    std::vector<uint8_t> nr(n * size_t(ell));
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < ell; ++j) {
            nr[i * size_t(ell) + size_t(j)] =
                uint8_t(eda[i].bits[size_t(j)] ^ (ch.is_p0() ? 1 : 0));
        }
    }
    size_t pos = 0;
    auto s = ripple(A, nr, 1, ell, true, t, pos, ch, c);
    std::vector<U128> threshold(n, red(-M, ell));
    auto w = ripple(threshold, s.sum, 0, ell, false, t, pos, ch, c);

    std::vector<uint8_t> em(n), et(n);
    for (size_t i = 0; i < n; ++i) {
        em[i] = uint8_t((w.carry[i] ^ da[i].bit) & 1);
    }
    ch.exchange_bytes(em.data(), et.data(), n);
    c.logical += n;
    c.sent += n;
    c.recv += n;
    c.post += n;

    std::vector<uint64_t> o(n);
    U128 Mbw = red(M, a.bw);
    for (size_t i = 0; i < n; ++i) {
        uint8_t e = uint8_t((em[i] ^ et[i]) & 1);
        U128 wa = e == 0 ? da[i].arithmetic
                         : (ch.is_p0() ? sub(1, da[i].arithmetic, a.bw)
                                       : sub(0, da[i].arithmetic, a.bw));
        o[i] = red64(sub(in[i].z, red(Mbw * wa, a.bw), a.bw), a.bw);
    }
    if (pos != t.size()) {
        std::fprintf(stderr, "triple consumption mismatch\n");
        std::exit(2);
    }
    return o;
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
        x.bw > 32 || x.trials == 0 || x.forced == 0 || x.inner == 0) {
        return false;
    }
    const U128 M = x.qbits == 128 ? modulus128() : U128(P0);
    if (x.ell != uint32_t(bitlen(2 * M - 1)) ||
        n != uint64_t(2) * x.trials + x.forced + 4) {
        return false;
    }
    x.records.resize(n);
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
    auto in = make_inputs(a, M);
    size_t n = in.size();
    PartyChannel ch(a.party, a.host, a.port);
    PartyRandom rng;
    bool self = correlation_selftest(ch, a.selftest, ell, a.bw, rng);
    uint64_t base = ch.costs.base_ots;
    ch.costs = ringlpn_2pc::Counters{};
    ch.costs.base_ots = base;

    uint64_t cb = ch.bytes_sent();
    uint64_t cs = ch.direction_switches();
    auto ct = std::chrono::steady_clock::now();
    auto eda = gen_edabits(ch, n, ell, rng);
    auto da = gen_dabits(ch, n, a.bw, rng);
    std::vector<BitTriple> t;
    size_t tn = n * size_t(2 * ell - 2);
    ringlpn_2pc::generate_bit_triples(ch, int(tn), rng, t);
    double cus = std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - ct)
                     .count();
    uint64_t cbytes = ch.bytes_sent() - cb;
    uint64_t csw = ch.direction_switches() - cs;

    uint64_t ob = ch.bytes_sent();
    uint64_t os = ch.direction_switches();
    auto ot = std::chrono::steady_clock::now();
    Costs cost;
    auto out = convert(a, M, in, eda, da, t, ch, cost);
    double ous = std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - ot)
                     .count();
    uint64_t obytes = ch.bytes_sent() - ob;
    uint64_t osw = ch.direction_switches() - os;

    uint64_t et = n * size_t(2 * ell - 2);
    uint64_t elog = n * size_t(5 * ell - 3);
    uint64_t erev = n * size_t(10 * ell - 6);
    uint64_t epost = n * size_t(2 * ell - 1);
    uint64_t edot = n * size_t(ell + 1);
    bool accounting =
        ch.costs.string_ots_128 == edot && ch.costs.triple_ots == 2 * et &&
        ch.costs.bit_triples == et && cost.conversions == n &&
        cost.edabit_bits == n * size_t(ell) && cost.dabits == n &&
        cost.triples == et && cost.logical == elog &&
        cost.meaningful_share_bits() == erev &&
        cost.post == epost;
    std::string path =
        a.prefix + "_p" + std::to_string(a.party) + ".convert";
    bool wrote = write_file(path, a, ell, in, out);
    bool all = self && accounting && wrote;
    ch.sync();
    double d = double(n);

    if (a.header) {
        std::printf(
            "party,requested_qbits,actual_qbits,bw,ell,conversions,"
            "and_triples_per_conv,edabit_bits_per_conv,dabits_per_conv,"
            "dabit_ots_per_conv,triple_ots_per_conv,"
            "logical_opened_bits_per_conv,meaningful_share_bits_per_conv,"
            "post_mask_dependency_rounds_per_conv,base_ots,setup_bytes_sent,"
            "setup_direction_switches,correlation_bytes_sent_batch,"
            "correlation_direction_switches_batch,online_bytes_sent_batch,"
            "online_direction_switches_batch,protocol_bytes_sent_batch,"
            "protocol_direction_switches_batch,correlation_us_batch,"
            "online_us_batch,correlation_selftest,transcript_accounting,status\n");
    }
    std::printf(
        "%d,%d,%d,%d,%d,%llu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.1f,%.1f,%s,%s,%s\n",
        a.party, a.qbits, a.qbits == 128 ? 124 : 62, a.bw, ell,
        (unsigned long long)n, cost.triples / d, cost.edabit_bits / d,
        cost.dabits / d, ch.costs.string_ots_128 / d,
        ch.costs.triple_ots / d, cost.logical / d,
        cost.meaningful_share_bits() / d,
        cost.post / d, (unsigned long long)ch.costs.base_ots,
        (unsigned long long)ch.setup_bytes_sent(),
        (unsigned long long)ch.setup_direction_switches(),
        (unsigned long long)cbytes,
        (unsigned long long)csw, (unsigned long long)obytes,
        (unsigned long long)osw, (unsigned long long)(cbytes + obytes),
        (unsigned long long)(csw + osw), cus, ous, self ? "pass" : "FAIL",
        accounting ? "pass" : "FAIL", all ? "pass" : "FAIL");
    std::fprintf(
        stderr,
        "[two-party-convert] party %d q%d bw=%d n=%zu: setup %llu/%llu; "
        "correlations %llu/%llu; online %llu/%llu; selftest %s accounting %s; "
        "output %s\n",
        a.party, a.qbits, a.bw, n,
        (unsigned long long)ch.setup_bytes_sent(),
        (unsigned long long)ch.setup_direction_switches(),
        (unsigned long long)cbytes,
        (unsigned long long)csw, (unsigned long long)obytes,
        (unsigned long long)osw, self ? "pass" : "FAIL",
        accounting ? "pass" : "FAIL", path.c_str());
    return all ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    Args a = parse(argc, argv);
    return a.check ? check(a) : party(a);
}
