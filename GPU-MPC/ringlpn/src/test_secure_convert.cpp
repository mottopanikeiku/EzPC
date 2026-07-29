// Secure Z_M -> Z_{2^bw} share-conversion prototype (Step 2 of the dealerless
// roadmap).
//
// The carry-corrected conversion exact_zm_to_ring_shares() is a dealer/oracle:
// it reads BOTH additive shares z0, z1 to compute the wrap bit
// m = [z0 + z1 >= M]. In a dealerless protocol neither party may learn the
// other's share, so the wrap bit must be computed by a secure two-party
// sub-protocol. This file implements and validates such a protocol and reports
// its cost, so the Route-A (prime-field OLE + secure conversion) overhead can be
// compared against the matmul before committing to Route A vs the Z_2^k-native
// Route B.
//
// Protocol (semi-honest, two parties P0 holds z0, P1 holds z1, both in [0, M)):
//   1. edaBit-mask open: with a shared random R in [0, L) (L = 2^ceil(log2 2M))
//      held both as arithmetic shares over Z_L and as boolean shares of its bits,
//      open A = (z0 + z1 + R) mod L. R perfectly hides S = z0 + z1 < 2M <= L.
//   2. Boolean wrap circuit: S = (A - R) mod L is recovered as boolean shares via
//      a ripple adder (public A + boolean-shared two's complement of R). A second
//      ripple adder computes the carry-out of S + (L - M), which equals the wrap
//      bit w = [S >= M]. AND gates use boolean Beaver triples (party-separated).
//   3. B2A: a daBit converts the boolean-shared w to arithmetic shares over
//      Z_{2^bw}.
//   4. Local correction: r_i = (z_i - M * w_i) mod 2^bw. Then
//      r0 + r1 == (z0 + z1 - w*M) == v (mod 2^bw), matching the oracle.
//
// Ideal/prototype pieces (the honest scope boundary): the edaBits, daBits and
// boolean triples are generated here by a labeled offline "dealer". In the full
// dealerless system these correlations are produced silently from PCG/OT (silent
// OT -> edaBits is a standard pipeline); the cost counters below report exactly
// how much such correlated randomness each conversion consumes.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using u128 = unsigned __int128;

constexpr uint64_t kPrime62 = 4611686018326724609ULL;
constexpr uint64_t kPrime62Crt2 = 4611686018309947393ULL;

// ----- ring / modular helpers (host copies, matching the oracle reference) -----

static uint64_t ring_reduce(u128 x, int bw) {
    if (bw == 64) {
        return static_cast<uint64_t>(x);
    }
    return static_cast<uint64_t>(x & ((u128(1) << bw) - 1));
}

static uint64_t ring_add(uint64_t a, uint64_t b, int bw) {
    return ring_reduce(u128(a) + b, bw);
}

static uint64_t ring_sub(uint64_t a, uint64_t b, int bw) {
    u128 r = u128(1) << bw;
    return ring_reduce(u128(a) + r - ring_reduce(b, bw), bw);
}

static u128 mod_sub(u128 a, u128 b, u128 modulus) {
    return a >= b ? a - b : a + modulus - b;
}

static u128 uniform_mod(u128 modulus, std::mt19937_64 &rng) {
    u128 x = (u128(rng()) << 64) ^ u128(rng());
    return x % modulus;
}

// Oracle reference: identical carry correction to test_orca_zp_bridge.cpp and
// orca_fc_ringlpn_keywriter.cuh::exactZmToRingShares.
static void exact_zm_to_ring_shares(u128 z0, u128 z1, u128 modulus, int bw,
                                    uint64_t &r0, uint64_t &r1) {
    const bool carry = z0 + z1 >= modulus;
    r0 = ring_reduce(z0, bw);
    r1 = ring_reduce(z1, bw);
    if (carry) {
        r1 = ring_sub(r1, ring_reduce(modulus, bw), bw);
    }
}

// ----- correlated randomness (prototype offline dealer) -----

struct BitShare {
    uint8_t s0 = 0;
    uint8_t s1 = 0;
    uint8_t clear() const { return s0 ^ s1; }
};

struct CostCounters {
    uint64_t conversions = 0;
    uint64_t edabit_bits = 0;       // boolean-shared random bits consumed
    uint64_t and_triples = 0;       // boolean Beaver triples consumed
    uint64_t dabits = 0;            // daBits consumed for B2A
    uint64_t logical_opened_bits = 0;
    uint64_t revealed_share_bits = 0;
    uint64_t post_mask_dependency_rounds = 0;  // excludes the initial masked-value opening
};

// One boolean AND triple, party-separated.
struct AndTriple {
    BitShare a, b, c;  // (a)&(b) == (c) in the clear
};

static AndTriple make_and_triple(std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> bit(0, 1);
    uint8_t a = bit(rng), b = bit(rng), c = a & b;
    AndTriple t;
    t.a.s0 = bit(rng); t.a.s1 = a ^ t.a.s0;
    t.b.s0 = bit(rng); t.b.s1 = b ^ t.b.s0;
    t.c.s0 = bit(rng); t.c.s1 = c ^ t.c.s0;
    return t;
}

// Secure boolean AND of two shared bits via a Beaver triple. Counts one triple,
// one 2-bit opening, one round.
static BitShare secure_and(const BitShare &x, const BitShare &y,
                           std::mt19937_64 &rng, CostCounters &cost) {
    AndTriple t = make_and_triple(rng);
    const uint8_t d = (x.s0 ^ t.a.s0) ^ (x.s1 ^ t.a.s1);  // open x ^ a
    const uint8_t e = (y.s0 ^ t.b.s0) ^ (y.s1 ^ t.b.s1);  // open y ^ b
    cost.and_triples += 1;
    cost.logical_opened_bits += 2;
    cost.revealed_share_bits += 4;
    cost.post_mask_dependency_rounds += 1;
    BitShare z;
    // z = c ^ (d & b) ^ (e & a) ^ (d & e), with d&e placed on party 0 only.
    z.s0 = t.c.s0 ^ (d & t.b.s0) ^ (e & t.a.s0) ^ (d & e);
    z.s1 = t.c.s1 ^ (d & t.b.s1) ^ (e & t.a.s1);
    return z;
}

static BitShare xor_bs(const BitShare &x, const BitShare &y) {
    return BitShare{static_cast<uint8_t>(x.s0 ^ y.s0), static_cast<uint8_t>(x.s1 ^ y.s1)};
}

// AND of a shared bit with a public bit (local, no triple).
static BitShare and_public(const BitShare &x, uint8_t p) {
    return BitShare{static_cast<uint8_t>(x.s0 & p), static_cast<uint8_t>(x.s1 & p)};
}

// XOR of a shared bit with a public bit (on party 0 only).
static BitShare xor_public(const BitShare &x, uint8_t p) {
    return BitShare{static_cast<uint8_t>(x.s0 ^ p), x.s1};
}

struct Edabit {
    u128 r0 = 0, r1 = 0;            // arithmetic shares over Z_L (R = (r0+r1) mod L)
    std::vector<BitShare> bits;    // boolean shares of each bit of R
};

static Edabit make_edabit(int ell, u128 L, std::mt19937_64 &rng, CostCounters &cost) {
    Edabit e;
    u128 R = uniform_mod(L, rng);
    e.r0 = uniform_mod(L, rng);
    e.r1 = mod_sub(R, e.r0, L);
    e.bits.resize(ell);
    std::uniform_int_distribution<int> bit(0, 1);
    for (int j = 0; j < ell; ++j) {
        uint8_t rb = static_cast<uint8_t>((R >> j) & 1);
        e.bits[j].s0 = bit(rng);
        e.bits[j].s1 = rb ^ e.bits[j].s0;
    }
    cost.edabit_bits += ell;
    return e;
}

struct Dabit {
    BitShare b;          // boolean shares of bit d
    uint64_t a0 = 0, a1 = 0;  // arithmetic shares over Z_{2^bw} with (a0+a1) mod 2^bw = d
};

static Dabit make_dabit(int bw, std::mt19937_64 &rng, CostCounters &cost) {
    Dabit d;
    std::uniform_int_distribution<int> bit(0, 1);
    uint8_t db = bit(rng);
    d.b.s0 = bit(rng);
    d.b.s1 = db ^ d.b.s0;
    std::uniform_int_distribution<uint64_t> ring(0, (bw == 64) ? UINT64_MAX : ((uint64_t(1) << bw) - 1));
    d.a0 = ring(rng);
    d.a1 = ring_sub(db, d.a0, bw);  // (a0 + a1) mod 2^bw = db
    cost.dabits += 1;
    return d;
}

// ----- ripple adder: public ell-bit constant C + boolean-shared X + carry_in -----
// Returns boolean-shared sum bits (if want_sum) and the final carry-out.

static BitShare ripple_add(u128 C, const std::vector<BitShare> &X, uint8_t carry_in,
                           int ell, bool want_sum, std::vector<BitShare> *sum_out,
                           std::mt19937_64 &rng, CostCounters &cost) {
    // carry starts public (carry_in); becomes shared after bit 0.
    BitShare carry{carry_in, 0};
    bool carry_public = true;
    uint8_t carry_pub_val = carry_in;
    if (want_sum) {
        sum_out->assign(ell, BitShare{});
    }
    for (int j = 0; j < ell; ++j) {
        const uint8_t cj = static_cast<uint8_t>((C >> j) & 1);
        const BitShare &xj = X[j];
        // sum_j = x_j ^ c_j ^ carry_j  (all linear)
        if (want_sum) {
            (*sum_out)[j] = xor_public(carry, cj);
            (*sum_out)[j] = xor_bs((*sum_out)[j], xj);
        }
        // carry_{j+1} = (x_j & carry_j) ^ (c_j & (x_j ^ carry_j))
        BitShare x_and_carry;
        if (carry_public) {
            x_and_carry = and_public(xj, carry_pub_val);  // local
        } else {
            x_and_carry = secure_and(xj, carry, rng, cost);
        }
        BitShare next = x_and_carry;
        if (cj) {
            BitShare x_xor_carry = carry_public ? xor_public(xj, carry_pub_val) : xor_bs(xj, carry);
            next = xor_bs(next, x_xor_carry);  // ^ (c_j & (x^carry)), c_j=1
        }
        carry = next;
        carry_public = false;  // after combining with shared x_j, carry is shared
    }
    return carry;  // carry-out
}

// ----- the secure conversion -----

struct ConvOut {
    uint64_t r0 = 0, r1 = 0;
    uint8_t wrap = 0;  // recovered wrap bit (clear, for diagnostics only)
};

static ConvOut secure_convert(u128 z0, u128 z1, u128 modulus, int bw, int ell, u128 L,
                              std::mt19937_64 &rng, CostCounters &cost) {
    cost.conversions += 1;

    Edabit eda = make_edabit(ell, L, rng, cost);

    // Step 1: open A = (z0 + z1 + R) mod L.
    u128 y0 = (z0 + eda.r0) % L;
    u128 y1 = (z1 + eda.r1) % L;
    u128 A = (y0 + y1) % L;
    cost.logical_opened_bits += ell;
    cost.revealed_share_bits += 2ull * ell;
    // This initial masked-value opening is excluded from the post-mask counter.

    // Step 2a: S = (A - R) mod L = A + NOT(R) + 1 (mod L). Boolean-shared sum bits.
    std::vector<BitShare> notR(ell);
    for (int j = 0; j < ell; ++j) {
        notR[j] = xor_public(eda.bits[j], 1);  // NOT(R) bit
    }
    std::vector<BitShare> Sbits;
    ripple_add(A, notR, /*carry_in=*/1, ell, /*want_sum=*/true, &Sbits, rng, cost);

    // Step 2b: w = carry-out of S + (L - M).  (S >= M  <=>  S + (L-M) >= L)
    u128 LminusM = (L - (modulus % L)) % L;
    BitShare wrap = ripple_add(LminusM, Sbits, /*carry_in=*/0, ell, /*want_sum=*/false,
                               nullptr, rng, cost);

    // Step 3: B2A of wrap bit -> arithmetic shares over Z_{2^bw}.
    Dabit da = make_dabit(bw, rng, cost);
    const uint8_t e = wrap.clear() ^ da.b.clear();  // open w ^ d
    cost.logical_opened_bits += 1;
    cost.revealed_share_bits += 2;
    cost.post_mask_dependency_rounds += 1;
    // w_arith = e + d - 2*e*d : party0 carries the public e term.
    uint64_t wa0 = ring_reduce(u128(e) + da.a0 - u128(2) * e * da.a0 + (u128(1) << bw) * 4, bw);
    uint64_t wa1 = ring_reduce(u128(da.a1) - u128(2) * e * da.a1 + (u128(1) << bw) * 4, bw);

    // Step 4: local correction r_i = (z_i - M * w_i) mod 2^bw.
    const uint64_t Mbw = ring_reduce(modulus, bw);
    ConvOut out;
    out.wrap = wrap.clear();
    out.r0 = ring_sub(ring_reduce(z0, bw), ring_reduce(u128(Mbw) * wa0, bw), bw);
    out.r1 = ring_sub(ring_reduce(z1, bw), ring_reduce(u128(Mbw) * wa1, bw), bw);
    return out;
}

// ----- tests -----

struct Args {
    int qbits = 64;
    u128 modulus = kPrime62;
    int bw = 16;
    int trials = 2000;
    int forced_wraps = 256;
    int inner = 8;          // layer-shaped dot length
    uint64_t value_bound = 255;
    uint64_t seed = 1;
    bool csv_header = false;
};

static int bitlen(u128 x) {
    int b = 0;
    while (x > 0) { ++b; x >>= 1; }
    return b;
}

struct Stats {
    int trials = 0;
    int wrap_mismatch = 0;      // recovered wrap != true wrap
    int convert_mismatch = 0;   // reconstruction != target
    int oracle_mismatch = 0;    // secure result != oracle result
};

static Stats run_random_trials(const Args &args, int ell, u128 L, std::mt19937_64 &rng,
                               CostCounters &cost) {
    Stats st;
    st.trials = args.trials + args.forced_wraps;
    for (int i = 0; i < st.trials; ++i) {
        u128 clear = uniform_mod(args.modulus - 1, rng);  // value v < M
        u128 z0 = (i < args.forced_wraps) ? (clear + 1) : uniform_mod(args.modulus, rng);
        u128 z1 = mod_sub(clear, z0, args.modulus);

        const uint64_t target = ring_reduce(clear, args.bw);
        const uint8_t true_wrap = (z0 + z1 >= args.modulus) ? 1 : 0;

        ConvOut sc = secure_convert(z0, z1, args.modulus, args.bw, ell, L, rng, cost);
        if (sc.wrap != true_wrap) ++st.wrap_mismatch;
        if (ring_add(sc.r0, sc.r1, args.bw) != target) ++st.convert_mismatch;

        uint64_t o0 = 0, o1 = 0;
        exact_zm_to_ring_shares(z0, z1, args.modulus, args.bw, o0, o1);
        if (ring_add(sc.r0, sc.r1, args.bw) != ring_add(o0, o1, args.bw)) ++st.oracle_mismatch;
    }
    return st;
}

static Stats run_boundary_trials(const Args &args, int ell, u128 L,
                                 std::mt19937_64 &rng, CostCounters &cost) {
    Stats st;
    constexpr int kCases = 4;
    const u128 z0[kCases] = {0, args.modulus - 1, 1, args.modulus - 1};
    const u128 z1[kCases] = {0, 0, args.modulus - 1, args.modulus - 1};
    st.trials = kCases;
    for (int i = 0; i < kCases; ++i) {
        const u128 sum = z0[i] + z1[i];
        const u128 clear = sum % args.modulus;
        const uint8_t true_wrap = sum >= args.modulus ? 1 : 0;
        const uint64_t target = ring_reduce(clear, args.bw);

        ConvOut sc = secure_convert(
            z0[i], z1[i], args.modulus, args.bw, ell, L, rng, cost);
        if (sc.wrap != true_wrap) ++st.wrap_mismatch;
        if (ring_add(sc.r0, sc.r1, args.bw) != target) {
            ++st.convert_mismatch;
        }
        uint64_t o0 = 0, o1 = 0;
        exact_zm_to_ring_shares(
            z0[i], z1[i], args.modulus, args.bw, o0, o1);
        if (ring_add(sc.r0, sc.r1, args.bw) !=
            ring_add(o0, o1, args.bw)) {
            ++st.oracle_mismatch;
        }
    }
    return st;
}

// Layer-shaped: v is a realistic bounded dot product (sum of K bounded products),
// shared over Z_M, then converted. Mirrors the transcript's per-entry conversion.
static Stats run_layer_trials(const Args &args, int ell, u128 L, std::mt19937_64 &rng,
                              CostCounters &cost) {
    Stats st;
    st.trials = args.trials;
    std::uniform_int_distribution<uint64_t> bounded(0, args.value_bound);
    for (int i = 0; i < args.trials; ++i) {
        u128 v = 0;
        for (int k = 0; k < args.inner; ++k) {
            v += u128(bounded(rng)) * bounded(rng);
        }
        v %= args.modulus;
        u128 z0 = uniform_mod(args.modulus, rng);
        u128 z1 = mod_sub(v, z0, args.modulus);

        const uint64_t target = ring_reduce(v, args.bw);
        const uint8_t true_wrap = (z0 + z1 >= args.modulus) ? 1 : 0;

        ConvOut sc = secure_convert(z0, z1, args.modulus, args.bw, ell, L, rng, cost);
        if (sc.wrap != true_wrap) ++st.wrap_mismatch;
        if (ring_add(sc.r0, sc.r1, args.bw) != target) ++st.convert_mismatch;
        uint64_t o0 = 0, o1 = 0;
        exact_zm_to_ring_shares(z0, z1, args.modulus, args.bw, o0, o1);
        if (ring_add(sc.r0, sc.r1, args.bw) != ring_add(o0, o1, args.bw)) ++st.oracle_mismatch;
    }
    return st;
}

static u128 q128_modulus() { return u128(kPrime62) * u128(kPrime62Crt2); }

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--qbits") && i + 1 < argc) {
            args.qbits = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--bw") && i + 1 < argc) {
            args.bw = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) {
            args.trials = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--forced-wraps") && i + 1 < argc) {
            args.forced_wraps = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--inner") && i + 1 < argc) {
            args.inner = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--value-bound") && i + 1 < argc) {
            args.value_bound = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            std::cerr << "Unknown arg: " << argv[i] << "\n";
            std::exit(1);
        }
    }
    if (args.qbits != 64 && args.qbits != 128) {
        std::cerr << "qbits must be 64 or 128\n";
        std::exit(1);
    }
    args.modulus = (args.qbits == 128) ? q128_modulus() : u128(kPrime62);
    if (args.bw <= 2 || args.bw > 32) {
        std::cerr << "bw must be in (2, 32]\n";
        std::exit(1);
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    std::mt19937_64 rng(args.seed);

    const int ell = bitlen(args.modulus * 2 - 1);  // L = 2^ell >= 2M
    const u128 L = u128(1) << ell;

    CostCounters cost;
    Stats boundary = run_boundary_trials(args, ell, L, rng, cost);
    Stats rnd = run_random_trials(args, ell, L, rng, cost);
    Stats layer = run_layer_trials(args, ell, L, rng, cost);

    const bool exact_match =
        boundary.wrap_mismatch == 0 && boundary.convert_mismatch == 0 &&
        boundary.oracle_mismatch == 0 &&
        rnd.wrap_mismatch == 0 && rnd.convert_mismatch == 0 &&
        rnd.oracle_mismatch == 0 && layer.wrap_mismatch == 0 &&
        layer.convert_mismatch == 0 && layer.oracle_mismatch == 0;
    const uint64_t expected_triples =
        cost.conversions * static_cast<uint64_t>(2 * ell - 2);
    const uint64_t expected_logical_opened =
        cost.conversions * static_cast<uint64_t>(5 * ell - 3);
    const uint64_t expected_revealed_shares =
        cost.conversions * static_cast<uint64_t>(10 * ell - 6);
    const uint64_t expected_post_mask_rounds =
        cost.conversions * static_cast<uint64_t>(2 * ell - 1);
    const bool transcript_accounting =
        cost.and_triples == expected_triples &&
        cost.edabit_bits == cost.conversions * static_cast<uint64_t>(ell) &&
        cost.dabits == cost.conversions &&
        cost.logical_opened_bits == expected_logical_opened &&
        cost.revealed_share_bits == expected_revealed_shares &&
        cost.post_mask_dependency_rounds == expected_post_mask_rounds;

    const double conv = static_cast<double>(cost.conversions);
    const double and_per = conv ? cost.and_triples / conv : 0.0;
    const double eda_per = conv ? cost.edabit_bits / conv : 0.0;
    const double logical_per =
        conv ? cost.logical_opened_bits / conv : 0.0;
    const double revealed_per =
        conv ? cost.revealed_share_bits / conv : 0.0;
    const double post_mask_rounds_per =
        conv ? cost.post_mask_dependency_rounds / conv : 0.0;

    if (args.csv_header) {
        std::cout << "mode,requested_qbits,actual_qbits,bw,ell,trials,forced_wraps,inner,"
                  << "boundary_trials,boundary_wrap_mismatch,"
                  << "boundary_convert_mismatch,boundary_oracle_mismatch,"
                  << "rand_wrap_mismatch,rand_convert_mismatch,rand_oracle_mismatch,"
                  << "layer_wrap_mismatch,layer_convert_mismatch,layer_oracle_mismatch,"
                  << "conversions,and_triples_per_conv,edabit_bits_per_conv,"
                  << "dabits_per_conv,logical_opened_bits_per_conv,"
                  << "revealed_share_bits_per_conv,post_mask_dependency_rounds_per_conv,"
                  << "transcript_accounting,bit_exact_match\n";
    }
    std::cout << "secure_convert_edabit_ripple," << args.qbits << ","
              << (args.qbits == 128 ? 124 : 62) << "," << args.bw << "," << ell << ","
              << args.trials << "," << args.forced_wraps << "," << args.inner << ","
              << boundary.trials << "," << boundary.wrap_mismatch << ","
              << boundary.convert_mismatch << "," << boundary.oracle_mismatch << ","
              << rnd.wrap_mismatch << "," << rnd.convert_mismatch << "," << rnd.oracle_mismatch << ","
              << layer.wrap_mismatch << "," << layer.convert_mismatch << "," << layer.oracle_mismatch << ","
              << cost.conversions << "," << and_per << "," << eda_per << ","
              << (conv ? cost.dabits / conv : 0.0) << "," << logical_per << ","
              << revealed_per << "," << post_mask_rounds_per << ","
              << (transcript_accounting ? "pass" : "fail") << ","
              << (exact_match ? "pass" : "fail") << "\n";

    return exact_match && transcript_accounting ? 0 : 2;
}
