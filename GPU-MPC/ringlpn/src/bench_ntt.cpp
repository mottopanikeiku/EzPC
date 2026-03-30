#include <nfl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <gmpxx.h>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

struct Stats {
    double mean_us;
    double stddev_us;
};

struct ResolvedConfig {
    int requested_qbits;
    int actual_qbits;
    int limb_bits;
    const char *limb_label;
};

static Stats compute_stats(const std::vector<double> &samples) {
    Stats s{0.0, 0.0};
    if (samples.empty()) {
        return s;
    }
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / static_cast<double>(samples.size());
    double var = 0.0;
    for (double v : samples) {
        double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(samples.size());
    s.mean_us = mean;
    s.stddev_us = std::sqrt(var);
    return s;
}

struct Args {
    int n = 0;
    int qbits = 0;
    int iters = 10000;
    int warmup = 1000;
    bool csv_header = false;
    bool skip_validation = false;
};

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " --n <deg> --qbits <bits> [--iters N] [--warmup N] [--csv-header] [--skip-validation]\n";
}

static Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            a.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--qbits") == 0 && i + 1 < argc) {
            a.qbits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            a.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv-header") == 0) {
            a.csv_header = true;
        } else if (std::strcmp(argv[i], "--skip-validation") == 0) {
            a.skip_validation = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (a.n == 0 || a.qbits == 0) {
        usage(argv[0]);
        std::exit(1);
    }
    return a;
}

static bool is_power_of_two(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

static int ilog2_exact(int n) {
    int logn = 0;
    while ((1 << logn) < n) {
        logn++;
    }
    return logn;
}

static void maximize_stack_limit() {
    struct rlimit limit;
    if (getrlimit(RLIMIT_STACK, &limit) != 0) {
        return;
    }

    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur == limit.rlim_max) {
        return;
    }

    struct rlimit updated = limit;
    updated.rlim_cur = limit.rlim_max;
    setrlimit(RLIMIT_STACK, &updated);
}

static ResolvedConfig resolve_config(int n, int requested_qbits) {
    if (!is_power_of_two(n) || n < 1024 || n > 1048576) {
        throw std::invalid_argument("Unsupported degree: n must be a power of two in [1024, 1048576]");
    }

    switch (requested_qbits) {
    case 30:
        if (n > 32768) {
            throw std::invalid_argument("Unsupported config: 30-bit modulus is limited to n <= 32768 in NFLLib uint32_t mode");
        }
        return {30, 30, 30, "u32"};
    case 32:
        if (n > 32768) {
            throw std::invalid_argument("Unsupported config: requested qbits=32 maps to actual qbits=30, which is limited to n <= 32768 in NFLLib uint32_t mode");
        }
        return {32, 30, 30, "u32"};
    case 60:
        if (n > 32768) {
            throw std::invalid_argument("Unsupported config: 60-bit modulus is limited to n <= 32768 in NFLLib uint32_t mode");
        }
        return {60, 60, 30, "u32"};
    case 62:
        return {62, 62, 62, "u64"};
    case 64:
        return {64, 62, 62, "u64"};
    case 124:
        return {124, 124, 62, "u64"};
    case 128:
        return {128, 124, 62, "u64"};
    default:
        throw std::invalid_argument("Unsupported qbits request: expected one of 30, 32, 60, 62, 64, 124, 128");
    }
}

template <typename Poly>
struct PolyDeleter {
    void operator()(Poly *poly) const {
        if (poly == nullptr) {
            return;
        }
        poly->~Poly();
        std::free(poly);
    }
};

template <typename Poly>
using PolyPtr = std::unique_ptr<Poly, PolyDeleter<Poly>>;

template <typename Poly, typename... Args>
static PolyPtr<Poly> make_aligned_poly(Args&&... args) {
    void *storage = nullptr;
    const size_t alignment = std::max<size_t>(32, alignof(Poly));
    if (posix_memalign(&storage, alignment, sizeof(Poly)) != 0) {
        throw std::bad_alloc();
    }

    try {
        return PolyPtr<Poly>(new (storage) Poly(std::forward<Args>(args)...));
    } catch (...) {
        std::free(storage);
        throw;
    }
}

template <typename Poly>
static std::vector<std::pair<size_t, uint64_t>> make_sparse_terms(size_t salt) {
    const size_t degree = Poly::degree;
    std::vector<size_t> positions;
    positions.push_back(0);
    positions.push_back((1 + salt) % degree);
    positions.push_back((7 + 2 * salt) % degree);
    positions.push_back((31 + 3 * salt) % degree);
    positions.push_back((degree / 8 + 5 * salt) % degree);
    positions.push_back((degree / 4 + 7 * salt) % degree);
    positions.push_back((degree / 2 + 11 * salt) % degree);
    positions.push_back((degree - (3 + salt % 11)) % degree);

    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    while (positions.size() < 8) {
        positions.push_back((positions.back() + 13) % degree);
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    }

    std::vector<std::pair<size_t, uint64_t>> terms;
    for (size_t index = 0; index < 8; index++) {
        terms.push_back(std::make_pair(positions[index], static_cast<uint64_t>(3 + salt + index * 2)));
    }
    return terms;
}

template <typename Poly>
static void fill_sparse_poly(Poly &poly, const std::vector<std::pair<size_t, uint64_t>> &terms) {
    poly = static_cast<typename Poly::value_type>(0);
    for (size_t term_index = 0; term_index < terms.size(); term_index++) {
        const size_t position = terms[term_index].first;
        const typename Poly::value_type value = static_cast<typename Poly::value_type>(terms[term_index].second);
        for (size_t modulus_index = 0; modulus_index < Poly::nmoduli; modulus_index++) {
            poly(modulus_index, position) = value;
        }
    }
}

template <typename Poly>
static std::map<size_t, long long> reference_negacyclic_product(
    const std::vector<std::pair<size_t, uint64_t>> &lhs,
    const std::vector<std::pair<size_t, uint64_t>> &rhs) {
    std::map<size_t, long long> result;
    for (size_t lhs_index = 0; lhs_index < lhs.size(); lhs_index++) {
        for (size_t rhs_index = 0; rhs_index < rhs.size(); rhs_index++) {
            size_t position = lhs[lhs_index].first + rhs[rhs_index].first;
            long long contribution = static_cast<long long>(lhs[lhs_index].second * rhs[rhs_index].second);
            if (position >= Poly::degree) {
                position -= Poly::degree;
                contribution = -contribution;
            }
            result[position] += contribution;
        }
    }
    return result;
}

template <typename Poly>
static bool same_coefficients(const Poly &lhs, const Poly &rhs, std::string *reason) {
    for (size_t modulus_index = 0; modulus_index < Poly::nmoduli; modulus_index++) {
        for (size_t coefficient_index = 0; coefficient_index < Poly::degree; coefficient_index++) {
            if (lhs(modulus_index, coefficient_index) != rhs(modulus_index, coefficient_index)) {
                if (reason != nullptr) {
                    std::ostringstream stream;
                    stream << "Mismatch at modulus " << modulus_index
                           << ", coefficient " << coefficient_index
                           << ": got " << lhs(modulus_index, coefficient_index)
                           << ", expected " << rhs(modulus_index, coefficient_index);
                    *reason = stream.str();
                }
                return false;
            }
        }
    }
    return true;
}

template <typename Poly>
static bool matches_reference_product(const Poly &poly, const std::map<size_t, long long> &expected, std::string *reason) {
    for (size_t modulus_index = 0; modulus_index < Poly::nmoduli; modulus_index++) {
        const unsigned long long modulus = static_cast<unsigned long long>(Poly::get_modulus(modulus_index));
        for (size_t coefficient_index = 0; coefficient_index < Poly::degree; coefficient_index++) {
            unsigned long long expected_value = 0;
            std::map<size_t, long long>::const_iterator it = expected.find(coefficient_index);
            if (it != expected.end()) {
                long long residue = it->second % static_cast<long long>(modulus);
                if (residue < 0) {
                    residue += static_cast<long long>(modulus);
                }
                expected_value = static_cast<unsigned long long>(residue);
            }

            if (poly(modulus_index, coefficient_index) != expected_value) {
                if (reason != nullptr) {
                    std::ostringstream stream;
                    stream << "Mismatch at modulus " << modulus_index
                           << ", coefficient " << coefficient_index
                           << ": got " << poly(modulus_index, coefficient_index)
                           << ", expected " << expected_value;
                    *reason = stream.str();
                }
                return false;
            }
        }
    }
    return true;
}

template <typename Poly>
static void validate_config(const ResolvedConfig &config) {
    const std::vector<std::pair<size_t, uint64_t>> lhs_terms = make_sparse_terms<Poly>(1);
    const std::vector<std::pair<size_t, uint64_t>> rhs_terms = make_sparse_terms<Poly>(5);

    PolyPtr<Poly> lhs = make_aligned_poly<Poly>(0);
    PolyPtr<Poly> rhs = make_aligned_poly<Poly>(0);
    fill_sparse_poly(*lhs, lhs_terms);
    fill_sparse_poly(*rhs, rhs_terms);

    PolyPtr<Poly> roundtrip = make_aligned_poly<Poly>(*lhs);
    roundtrip->ntt_pow_phi();
    roundtrip->invntt_pow_invphi();

    std::string reason;
    if (!same_coefficients(*roundtrip, *lhs, &reason)) {
        std::ostringstream stream;
        stream << "Validation failed for requested qbits=" << config.requested_qbits
               << " (actual qbits=" << config.actual_qbits << "): NTT roundtrip mismatch. "
               << reason;
        throw std::runtime_error(stream.str());
    }

    PolyPtr<Poly> lhs_ntt = make_aligned_poly<Poly>(*lhs);
    PolyPtr<Poly> rhs_ntt = make_aligned_poly<Poly>(*rhs);
    lhs_ntt->ntt_pow_phi();
    rhs_ntt->ntt_pow_phi();
    PolyPtr<Poly> product = make_aligned_poly<Poly>((*lhs_ntt) * (*rhs_ntt));
    product->invntt_pow_invphi();

    const std::map<size_t, long long> expected = reference_negacyclic_product<Poly>(lhs_terms, rhs_terms);
    if (!matches_reference_product(*product, expected, &reason)) {
        std::ostringstream stream;
        stream << "Validation failed for requested qbits=" << config.requested_qbits
               << " (actual qbits=" << config.actual_qbits << "): negacyclic product mismatch. "
               << reason;
        throw std::runtime_error(stream.str());
    }
}

template <typename Poly>
static void run_bench(const Args &args, const ResolvedConfig &config) {
    if (!args.skip_validation) {
        validate_config<Poly>(config);
    }

    PolyPtr<Poly> a = make_aligned_poly<Poly>(nfl::uniform());
    PolyPtr<Poly> b = make_aligned_poly<Poly>(nfl::uniform());
    PolyPtr<Poly> c = make_aligned_poly<Poly>();
    PolyPtr<Poly> x = make_aligned_poly<Poly>(*a);
    PolyPtr<Poly> y = make_aligned_poly<Poly>(*b);

    // Warmup
    for (int i = 0; i < args.warmup; i++) {
        *x = *a;
        x->ntt_pow_phi();
        x->invntt_pow_invphi();
        *c = (*a) * (*b);
    }

    std::vector<double> ntt_samples;
    std::vector<double> intt_samples;
    std::vector<double> mul_samples;
    ntt_samples.reserve(args.iters);
    intt_samples.reserve(args.iters);
    mul_samples.reserve(args.iters);

    // Forward NTT timing
    for (int i = 0; i < args.iters; i++) {
        *x = *a;
        auto start = std::chrono::high_resolution_clock::now();
        x->ntt_pow_phi();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        ntt_samples.push_back(elapsed.count());
        x->invntt_pow_invphi();
    }

    // Inverse NTT timing (prepare x in NTT domain once per iteration)
    for (int i = 0; i < args.iters; i++) {
        *x = *a;
        x->ntt_pow_phi();
        auto start = std::chrono::high_resolution_clock::now();
        x->invntt_pow_invphi();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        intt_samples.push_back(elapsed.count());
    }

    // Polynomial multiplication timing (NTT(a) + NTT(b) + pointwise + INTT)
    for (int i = 0; i < args.iters; i++) {
        *x = *a;
        *y = *b;
        auto start = std::chrono::high_resolution_clock::now();
        x->ntt_pow_phi();
        y->ntt_pow_phi();
        *c = (*x) * (*y);
        c->invntt_pow_invphi();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        mul_samples.push_back(elapsed.count());
    }

    Stats ntt = compute_stats(ntt_samples);
    Stats intt = compute_stats(intt_samples);
    Stats mul = compute_stats(mul_samples);

    std::cout << args.n << ","
              << ilog2_exact(args.n) << ","
              << config.requested_qbits << ","
              << config.actual_qbits << ","
              << config.limb_bits << ","
              << sizeof(typename Poly::value_type) << ","
              << config.limb_label << ","
              << args.iters << ","
              << (args.skip_validation ? "skipped" : "pass") << ","
              << ntt.mean_us << "," << ntt.stddev_us << ","
              << intt.mean_us << "," << intt.stddev_us << ","
              << mul.mean_us << "," << mul.stddev_us << "\n";
}

#define DISPATCH_CASE(TYPE, DEGREE, QBITS) \
    case DEGREE: \
        run_bench<nfl::poly_from_modulus<TYPE, DEGREE, QBITS>>(args, config); \
        return

template <int QBITS>
static void dispatch_u32(const Args &args, const ResolvedConfig &config) {
    switch (args.n) {
    DISPATCH_CASE(uint32_t, 1024, QBITS);
    DISPATCH_CASE(uint32_t, 2048, QBITS);
    DISPATCH_CASE(uint32_t, 4096, QBITS);
    DISPATCH_CASE(uint32_t, 8192, QBITS);
    DISPATCH_CASE(uint32_t, 16384, QBITS);
    DISPATCH_CASE(uint32_t, 32768, QBITS);
    default:
        throw std::invalid_argument("Unsupported degree for NFLLib uint32_t mode");
    }
}

template <int QBITS>
static void dispatch_u64(const Args &args, const ResolvedConfig &config) {
    switch (args.n) {
    DISPATCH_CASE(uint64_t, 1024, QBITS);
    DISPATCH_CASE(uint64_t, 2048, QBITS);
    DISPATCH_CASE(uint64_t, 4096, QBITS);
    DISPATCH_CASE(uint64_t, 8192, QBITS);
    DISPATCH_CASE(uint64_t, 16384, QBITS);
    DISPATCH_CASE(uint64_t, 32768, QBITS);
    DISPATCH_CASE(uint64_t, 65536, QBITS);
    DISPATCH_CASE(uint64_t, 131072, QBITS);
    DISPATCH_CASE(uint64_t, 262144, QBITS);
    DISPATCH_CASE(uint64_t, 524288, QBITS);
    DISPATCH_CASE(uint64_t, 1048576, QBITS);
    default:
        throw std::invalid_argument("Unsupported degree for NFLLib uint64_t mode");
    }
}

#undef DISPATCH_CASE

static void dispatch(const Args &args) {
    const ResolvedConfig config = resolve_config(args.n, args.qbits);

    if (config.actual_qbits == 30) {
        dispatch_u32<30>(args, config);
        return;
    }

    if (config.actual_qbits == 60) {
        dispatch_u32<60>(args, config);
        return;
    }

    if (config.actual_qbits == 62) {
        dispatch_u64<62>(args, config);
        return;
    }

    if (config.actual_qbits == 124) {
        dispatch_u64<124>(args, config);
        return;
    }

    throw std::invalid_argument("Unsupported resolved configuration");
}

int main(int argc, char **argv) {
    try {
        maximize_stack_limit();
        Args args = parse_args(argc, argv);
        if (args.csv_header) {
            std::cout << "n,logn,requested_qbits,actual_qbits,limb_bits,limb_bytes,limb_type,iters,validation,ntt_mean_us,ntt_std_us,intt_mean_us,intt_std_us,poly_mul_mean_us,poly_mul_std_us\n";
        }
        dispatch(args);
        return 0;
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << "\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 3;
    }
}
