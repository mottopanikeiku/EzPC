#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using u128 = unsigned __int128;

constexpr uint64_t kPrime62 = 4611686018326724609ULL;

struct Args {
    uint64_t prime = kPrime62;
    int bw = 16;
    int rows = 2;
    int inner = 2;
    int cols = 2;
    uint64_t value_bound = 255;
    int trials = 1000;
    int forced_wraps = 128;
    uint64_t seed = 1;
    bool csv_header = false;
};

struct ConversionStats {
    int trials = 0;
    int forced_wraps = 0;
    int naive_failures = 0;
    int corrected_failures = 0;
};

struct MatmulStats {
    bool bound_ok = false;
    bool validation_ok = false;
    bool counterexample_found = false;
};

static u128 ring_modulus(int bw) {
    if (bw <= 0 || bw > 64) {
        std::cerr << "bw must be in [1, 64]\n";
        std::exit(1);
    }
    return u128(1) << bw;
}

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
    u128 r = ring_modulus(bw);
    return ring_reduce(u128(a) + r - ring_reduce(b, bw), bw);
}

static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t p) {
    u128 s = u128(a) + b;
    if (s >= p) {
        s -= p;
    }
    return static_cast<uint64_t>(s);
}

static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t p) {
    return a >= b ? a - b : static_cast<uint64_t>(u128(a) + p - b);
}

static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p) {
    return static_cast<uint64_t>((u128(a) * b) % p);
}

static uint64_t uniform_mod(uint64_t p, std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, p - 1);
    return dist(rng);
}

static void exact_zp_to_ring_shares(uint64_t z0,
                                    uint64_t z1,
                                    uint64_t p,
                                    int bw,
                                    uint64_t &r0,
                                    uint64_t &r1) {
    const bool carry = u128(z0) + z1 >= p;
    r0 = ring_reduce(z0, bw);
    r1 = ring_reduce(z1, bw);
    if (carry) {
        r1 = ring_sub(r1, ring_reduce(p, bw), bw);
    }
}

static ConversionStats run_conversion_trials(const Args &args) {
    std::mt19937_64 rng(args.seed);
    ConversionStats stats;
    stats.trials = args.trials;
    stats.forced_wraps = args.forced_wraps;

    for (int i = 0; i < args.trials + args.forced_wraps; ++i) {
        uint64_t clear = uniform_mod(args.prime - 1, rng);
        uint64_t z0 = 0;
        if (i < args.forced_wraps) {
            z0 = clear + 1;
        } else {
            z0 = uniform_mod(args.prime, rng);
        }
        uint64_t z1 = mod_sub(clear, z0, args.prime);

        uint64_t target = ring_reduce(clear, args.bw);
        uint64_t naive = ring_add(ring_reduce(z0, args.bw), ring_reduce(z1, args.bw), args.bw);
        if (naive != target) {
            ++stats.naive_failures;
        }

        uint64_t r0 = 0;
        uint64_t r1 = 0;
        exact_zp_to_ring_shares(z0, z1, args.prime, args.bw, r0, r1);
        uint64_t corrected = ring_add(r0, r1, args.bw);
        if (corrected != target) {
            ++stats.corrected_failures;
        }
    }
    return stats;
}

static bool no_prime_wrap_bound(const Args &args) {
    u128 bound = u128(args.inner) * args.value_bound * args.value_bound;
    return bound < args.prime;
}

static MatmulStats run_constant_polynomial_scalar_matmul(const Args &args) {
    MatmulStats stats;
    stats.bound_ok = no_prime_wrap_bound(args);

    std::mt19937_64 rng(args.seed ^ 0xD1B54A32D192ED03ULL);
    std::uniform_int_distribution<uint64_t> dist(0, args.value_bound);
    std::vector<uint64_t> a(static_cast<size_t>(args.rows) * args.inner);
    std::vector<uint64_t> b(static_cast<size_t>(args.inner) * args.cols);
    for (uint64_t &v : a) {
        v = dist(rng);
    }
    for (uint64_t &v : b) {
        v = dist(rng);
    }

    bool ok = true;
    for (int r = 0; r < args.rows; ++r) {
        for (int c = 0; c < args.cols; ++c) {
            u128 integer_dot = 0;
            uint64_t field_dot = 0;
            for (int k = 0; k < args.inner; ++k) {
                uint64_t av = a[static_cast<size_t>(r) * args.inner + k];
                uint64_t bv = b[static_cast<size_t>(k) * args.cols + c];
                integer_dot += u128(av) * bv;
                field_dot = mod_add(field_dot, mod_mul(av, bv, args.prime), args.prime);
            }

            uint64_t target = ring_reduce(integer_dot, args.bw);
            uint64_t z0 = uniform_mod(args.prime, rng);
            uint64_t z1 = mod_sub(field_dot, z0, args.prime);
            uint64_t r0 = 0;
            uint64_t r1 = 0;
            exact_zp_to_ring_shares(z0, z1, args.prime, args.bw, r0, r1);
            uint64_t converted = ring_add(r0, r1, args.bw);
            if (stats.bound_ok && converted != target) {
                ok = false;
            }
        }
    }
    stats.validation_ok = stats.bound_ok && ok;

    if (!stats.bound_ok) {
        uint64_t v = args.value_bound;
        uint64_t direct = ring_reduce(u128(v) * v, args.bw);
        uint64_t field = mod_mul(v % args.prime, v % args.prime, args.prime);
        stats.counterexample_found = ring_reduce(field, args.bw) != direct;
    }
    return stats;
}

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--prime P] [--bw N] [--rows M] [--inner K] [--cols N]"
              << " [--value-bound B] [--trials N] [--forced-wraps N]"
              << " [--seed N] [--csv-header]\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prime") && i + 1 < argc) {
            args.prime = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--bw") && i + 1 < argc) {
            args.bw = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) {
            args.rows = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--inner") && i + 1 < argc) {
            args.inner = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--cols") && i + 1 < argc) {
            args.cols = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--value-bound") && i + 1 < argc) {
            args.value_bound = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) {
            args.trials = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--forced-wraps") && i + 1 < argc) {
            args.forced_wraps = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }

    if (args.prime < 3 || args.prime >= (uint64_t(1) << 63) || args.bw <= 0 ||
        args.bw > 64 || args.rows <= 0 || args.inner <= 0 || args.cols <= 0 ||
        args.trials < 0 || args.forced_wraps < 0 || args.value_bound >= args.prime ||
        u128(args.value_bound) >= ring_modulus(args.bw)) {
        usage(argv[0]);
        std::exit(1);
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    ConversionStats conv = run_conversion_trials(args);
    MatmulStats matmul = run_constant_polynomial_scalar_matmul(args);

    if (args.csv_header) {
        std::cout << "device,input_mode,prime,bw,rows,inner,cols,value_bound,"
                  << "share_trials,forced_wrap_trials,naive_share_failures,"
                  << "corrected_share_failures,no_prime_wrap_bound,"
                  << "constant_scalar_matmul_validation,counterexample_found\n";
    }
    std::cout << "host_orca_zp_bridge,constant_polynomial_scalar,"
              << args.prime << "," << args.bw << ","
              << args.rows << "," << args.inner << "," << args.cols << ","
              << args.value_bound << "," << conv.trials << ","
              << conv.forced_wraps << "," << conv.naive_failures << ","
              << conv.corrected_failures << ","
              << (matmul.bound_ok ? 1 : 0) << ","
              << (matmul.validation_ok ? "pass" : "not_claimed") << ","
              << (matmul.counterexample_found ? 1 : 0) << "\n";

    if (conv.corrected_failures != 0) {
        return 2;
    }
    if (matmul.bound_ok && !matmul.validation_ok) {
        return 3;
    }
    if (!matmul.bound_ok && !matmul.counterexample_found) {
        return 4;
    }
    return 0;
}
