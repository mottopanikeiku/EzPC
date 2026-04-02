#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace {

#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_cheddar"
#endif

constexpr int kMinDegree = 8192;
constexpr int kMaxDegree = 1048576;
constexpr int kMinLogDegree = 13;
constexpr int kMaxLogDegree = 20;
constexpr uint32_t kModulus = 1004535809u;
constexpr uint32_t kPrimitiveGenerator = 3u;
constexpr int kActualQBits = 30;
constexpr int kMaxValidationBatches = 4;
constexpr int kLsbSize = 32;

struct Stats {
    double mean_us;
    double stddev_us;
};

struct Args {
    int n = kMinDegree;
    int requested_qbits = 32;
    int batch_size = 1;
    int iters = 10000;
    int warmup = 1000;
    bool csv_header = false;
    bool skip_validation = false;
};

struct HostTables {
    std::vector<uint32_t> primes;
    std::vector<int32_t> inv_primes;
    std::vector<uint32_t> fwd_twiddles_mont;
    std::vector<uint32_t> fwd_twiddles_msb;
    std::vector<uint32_t> inv_twiddles_mont;
    std::vector<uint32_t> inv_twiddles_msb;
    std::vector<uint32_t> inv_degree;
    std::vector<uint32_t> inv_degree_mont;
    std::vector<uint32_t> montgomery_converter;
};

struct DeviceTables {
    uint32_t *d_primes = nullptr;
    int32_t *d_inv_primes = nullptr;
    uint32_t *d_fwd_twiddles = nullptr;
    uint32_t *d_fwd_twiddles_msb = nullptr;
    uint32_t *d_inv_twiddles = nullptr;
    uint32_t *d_inv_twiddles_msb = nullptr;
    uint32_t *d_inv_degree = nullptr;
    uint32_t *d_inv_degree_mont = nullptr;
    uint32_t *d_montgomery_converter = nullptr;
};

static void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

static void check_launch(const char *msg) {
    check(cudaGetLastError(), msg);
}

static bool is_power_of_two(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --n <deg> [--qbits 30|32] [--batch N] [--iters N] [--warmup N]"
              << " [--csv-header] [--skip-validation]\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--qbits") == 0 && i + 1 < argc) {
            args.requested_qbits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            args.batch_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv-header") == 0) {
            args.csv_header = true;
        } else if (std::strcmp(argv[i], "--skip-validation") == 0) {
            args.skip_validation = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }

    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree) {
        usage(argv[0]);
        std::exit(1);
    }

    int log_degree = 0;
    int temp_n = args.n;
    while (temp_n > 1) {
        temp_n >>= 1;
        log_degree++;
    }
    if (log_degree < kMinLogDegree || log_degree > kMaxLogDegree) {
        std::cerr << "Unsupported degree for cheddar extract: expected log2(n) in ["
                  << kMinLogDegree << ", " << kMaxLogDegree << "]\n";
        std::exit(1);
    }

    if (args.requested_qbits != 30 && args.requested_qbits != 32) {
        std::cerr << "Unsupported qbits request: expected one of 30 or 32\n";
        std::exit(1);
    }

    if (args.batch_size <= 0 || args.iters <= 0 || args.warmup < 0) {
        usage(argv[0]);
        std::exit(1);
    }

    if (args.batch_size > 65535) {
        std::cerr << "Unsupported batch size: grid.y is capped at 65535 in this extracted kernel path\n";
        std::exit(1);
    }

    if (((kModulus - 1ULL) % (2ULL * static_cast<uint64_t>(args.n))) != 0ULL) {
        std::cerr << "Unsupported degree for selected 30-bit prime: need 2n to divide q-1\n";
        std::exit(1);
    }

    return args;
}

static int int_log2(uint32_t x) {
    int p = 0;
    while (x > 1) {
        x >>= 1;
        p++;
    }
    return p;
}

enum class InputPattern {
    Zero,
    One,
    Impulse,
    Max,
    Random,
};

static Stats compute_stats(const std::vector<double> &samples) {
    Stats s{0.0, 0.0};
    if (samples.empty()) {
        return s;
    }
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.mean_us = sum / static_cast<double>(samples.size());
    double var = 0.0;
    for (double sample : samples) {
        double delta = sample - s.mean_us;
        var += delta * delta;
    }
    var /= static_cast<double>(samples.size());
    s.stddev_us = std::sqrt(var);
    return s;
}

static uint32_t mod_add(uint32_t a, uint32_t b) {
    uint64_t sum = static_cast<uint64_t>(a) + b;
    if (sum >= kModulus) {
        sum -= kModulus;
    }
    return static_cast<uint32_t>(sum);
}

static uint32_t mod_sub(uint32_t a, uint32_t b) {
    return a >= b ? (a - b) : (a + kModulus - b);
}

static uint32_t mod_mul_host(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % kModulus);
}

static uint32_t mod_pow(uint32_t base, uint64_t exp) {
    uint32_t result = 1;
    uint32_t cur = base;
    while (exp > 0) {
        if (exp & 1ULL) {
            result = mod_mul_host(result, cur);
        }
        cur = mod_mul_host(cur, cur);
        exp >>= 1ULL;
    }
    return result;
}

static uint32_t mod_inv(uint32_t x) {
    return mod_pow(x, kModulus - 2ULL);
}

static int64_t gcd_extended(int64_t a, int64_t b, int64_t &x, int64_t &y) {
    if (a == 0) {
        x = 0;
        y = 1;
        return b;
    }
    int64_t x1 = 0;
    int64_t y1 = 0;
    int64_t gcd = gcd_extended(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return gcd;
}

static int32_t inv_mod_base(uint32_t q) {
    uint32_t base_minus_one = ~static_cast<uint32_t>(0);
    uint32_t quotient = base_minus_one / q;
    uint32_t remainder = (base_minus_one % q) + 1;
    int64_t x = 0;
    int64_t y = 0;
    gcd_extended(static_cast<int64_t>(remainder), static_cast<int64_t>(q), x, y);
    int64_t inv_q = y - static_cast<int64_t>(quotient) * x;
    return static_cast<int32_t>(inv_q);
}

static uint32_t to_montgomery_host(uint32_t x) {
    uint64_t wide = static_cast<uint64_t>(x) << 32;
    return static_cast<uint32_t>(wide % kModulus);
}

static uint32_t bit_reverse(uint32_t x, int log_degree) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - log_degree);
}

static void bit_reverse_vector(std::vector<uint32_t> &data) {
    const int n = static_cast<int>(data.size());
    const int log_degree = int_log2(static_cast<uint32_t>(n));
    for (int i = 0; i < n; i++) {
        int j = static_cast<int>(bit_reverse(static_cast<uint32_t>(i), log_degree));
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

static uint32_t compute_phi_for_n(int n) {
    return mod_pow(kPrimitiveGenerator, (kModulus - 1ULL) / (2ULL * static_cast<uint64_t>(n)));
}

static void compute_reference_vectors(std::vector<uint32_t> &phi_norm,
                                      std::vector<uint32_t> &post_norm,
                                      int n) {
    phi_norm.resize(n);
    post_norm.resize(n);

    uint32_t phi = compute_phi_for_n(n);
    uint32_t invphi = mod_inv(phi);
    uint32_t inv_n = mod_inv(static_cast<uint32_t>(n));

    uint32_t cur = 1;
    for (int i = 0; i < n; i++) {
        phi_norm[i] = cur;
        cur = mod_mul_host(cur, phi);
    }

    cur = inv_n;
    for (int i = 0; i < n; i++) {
        post_norm[i] = cur;
        cur = mod_mul_host(cur, invphi);
    }
}

static void compute_cheddar_tables(HostTables &tables, int n) {
    const int msb_size = n / kLsbSize;
    uint32_t psi = compute_phi_for_n(n);
    uint32_t psi_inv = mod_inv(psi);
    uint32_t inv_n = mod_inv(static_cast<uint32_t>(n));
    uint32_t one_mont = to_montgomery_host(1u);

    std::vector<uint32_t> psi_rev(n);
    std::vector<uint32_t> psi_inv_rev(n);
    psi_rev[0] = 1u;
    psi_inv_rev[0] = 1u;
    for (int i = 1; i < n; i++) {
        psi_rev[i] = mod_mul_host(psi_rev[i - 1], psi);
        psi_inv_rev[i] = mod_mul_host(psi_inv_rev[i - 1], psi_inv);
    }
    bit_reverse_vector(psi_rev);
    bit_reverse_vector(psi_inv_rev);

    tables.primes = {kModulus};
    tables.inv_primes = {inv_mod_base(kModulus)};
    tables.fwd_twiddles_mont.resize(n);
    tables.inv_twiddles_mont.resize(n);
    tables.fwd_twiddles_msb.resize(msb_size);
    tables.inv_twiddles_msb.resize(msb_size);
    tables.inv_degree = {inv_n};
    tables.inv_degree_mont = {to_montgomery_host(inv_n)};
    tables.montgomery_converter = {to_montgomery_host(one_mont)};

    for (int i = 0; i < n; i++) {
        tables.fwd_twiddles_mont[i] = to_montgomery_host(psi_rev[i]);
        tables.inv_twiddles_mont[i] = to_montgomery_host(psi_inv_rev[i]);
    }

    for (int i = 0; i < msb_size; i++) {
        tables.fwd_twiddles_msb[i] = tables.fwd_twiddles_mont[i * kLsbSize];
        tables.inv_twiddles_msb[i] = tables.inv_twiddles_mont[i * kLsbSize];
    }
}

static void host_forward_ntt(std::vector<uint32_t> &a,
                             const std::vector<uint32_t> &phi,
                             int n,
                             int log_degree) {
    std::vector<uint32_t> tmp(n);
    for (int i = 0; i < n; i++) {
        tmp[bit_reverse(static_cast<uint32_t>(i), log_degree)] = mod_mul_host(a[i], phi[i]);
    }
    a.swap(tmp);

    uint32_t phi_root = compute_phi_for_n(n);
    uint32_t omega = mod_mul_host(phi_root, phi_root);

    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        uint32_t wlen = mod_pow(omega, static_cast<uint64_t>(n / len));
        for (int start = 0; start < n; start += len) {
            uint32_t w = 1;
            for (int j = 0; j < half; j++) {
                uint32_t u = a[start + j];
                uint32_t v = mod_mul_host(a[start + j + half], w);
                a[start + j] = mod_add(u, v);
                a[start + j + half] = mod_sub(u, v);
                w = mod_mul_host(w, wlen);
            }
        }
    }
}

static void host_inverse_ntt(std::vector<uint32_t> &a,
                             const std::vector<uint32_t> &post_scale,
                             int n,
                             int log_degree) {
    std::vector<uint32_t> tmp(n);
    for (int i = 0; i < n; i++) {
        tmp[bit_reverse(static_cast<uint32_t>(i), log_degree)] = a[i];
    }
    a.swap(tmp);

    uint32_t phi_root = compute_phi_for_n(n);
    uint32_t omega = mod_mul_host(phi_root, phi_root);
    uint32_t invomega = mod_inv(omega);

    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        uint32_t wlen = mod_pow(invomega, static_cast<uint64_t>(n / len));
        for (int start = 0; start < n; start += len) {
            uint32_t w = 1;
            for (int j = 0; j < half; j++) {
                uint32_t u = a[start + j];
                uint32_t v = mod_mul_host(a[start + j + half], w);
                a[start + j] = mod_add(u, v);
                a[start + j + half] = mod_sub(u, v);
                w = mod_mul_host(w, wlen);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        a[i] = mod_mul_host(a[i], post_scale[i]);
    }
}

static std::vector<uint32_t> make_input_pattern(int n, InputPattern pattern, uint32_t seed = 0) {
    std::vector<uint32_t> values(n, 0);
    switch (pattern) {
    case InputPattern::Zero:
        break;
    case InputPattern::One:
        std::fill(values.begin(), values.end(), 1u);
        break;
    case InputPattern::Impulse:
        if (n > 0) {
            values[0] = 1u;
        }
        break;
    case InputPattern::Max:
        std::fill(values.begin(), values.end(), kModulus - 1u);
        break;
    case InputPattern::Random: {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist(0, kModulus - 1);
        for (int i = 0; i < n; i++) {
            values[i] = dist(rng);
        }
        break;
    }
    }
    return values;
}

static std::vector<uint32_t> make_batched_pattern(int batch_count,
                                                  int n,
                                                  InputPattern pattern,
                                                  uint32_t seed_base = 0) {
    std::vector<uint32_t> values(static_cast<size_t>(batch_count) * static_cast<size_t>(n));
    for (int batch = 0; batch < batch_count; batch++) {
        std::vector<uint32_t> lane =
            make_input_pattern(n, pattern, seed_base + static_cast<uint32_t>(batch * 17));
        std::copy(lane.begin(), lane.end(),
                  values.begin() + static_cast<size_t>(batch) * static_cast<size_t>(n));
    }
    return values;
}

static bool compare_vectors(const std::vector<uint32_t> &expected,
                            const std::vector<uint32_t> &actual,
                            int n,
                            const char *label) {
    if (expected.size() != actual.size()) {
        std::cerr << label << " size mismatch: expected " << expected.size()
                  << " got " << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (expected[i] != actual[i]) {
            std::cerr << label << " mismatch at batch " << (i / static_cast<size_t>(n))
                      << ", index " << (i % static_cast<size_t>(n))
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << "\n";
            return false;
        }
    }
    return true;
}

static std::vector<uint32_t> host_polymul_reference(const std::vector<uint32_t> &lhs,
                                                    const std::vector<uint32_t> &rhs,
                                                    const std::vector<uint32_t> &phi_norm,
                                                    const std::vector<uint32_t> &post_norm,
                                                    int n,
                                                    int log_degree) {
    std::vector<uint32_t> host_a = lhs;
    std::vector<uint32_t> host_b = rhs;
    host_forward_ntt(host_a, phi_norm, n, log_degree);
    host_forward_ntt(host_b, phi_norm, n, log_degree);
    std::vector<uint32_t> host_c(n);
    for (int i = 0; i < n; i++) {
        host_c[i] = mod_mul_host(host_a[i], host_b[i]);
    }
    host_inverse_ntt(host_c, post_norm, n, log_degree);
    return host_c;
}

template <int Start, int End, int Inc = 1, class Func>
constexpr void constexpr_for(Func &&func) {
    if constexpr (Start < End) {
        func(std::integral_constant<int, Start>());
        constexpr_for<Start + Inc, End, Inc>(std::forward<Func>(func));
    }
}

namespace cheddar_extract {

using std::make_signed_t;

constexpr bool kExtendedOT = true;

enum class NTTType { NTT, INTT };
enum class Phase { Phase1, Phase2 };

template <typename word>
__device__ __inline__ word StreamingLoad(const word *src) {
    return *src;
}

template <typename word>
__device__ __inline__ word StreamingLoadConst(const word *src) {
    return *src;
}

template <typename word, int size>
__device__ __inline__ void VectorizedMove(word *dst, const word *src) {
#pragma unroll
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

template <typename word>
__device__ __inline__ make_signed_t<word> MultMontgomeryLazy(
    make_signed_t<word> a,
    make_signed_t<word> b,
    word q,
    make_signed_t<word> q_inv);

template <>
__device__ __inline__ int32_t MultMontgomeryLazy<uint32_t>(
    int32_t a,
    int32_t b,
    uint32_t q,
    int32_t q_inv) {
    int64_t mult;
    asm("mul.wide.s32 %0, %1, %2;" : "=l"(mult) : "r"(a), "r"(b));
    int32_t lo;
    int32_t hi;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(mult));
    int32_t temp = lo * q_inv;
    temp = __mulhi(temp, static_cast<int32_t>(q));
    return hi - temp;
}

template <typename word>
__device__ __inline__ word MultMontgomery(word a,
                                          word b,
                                          word q,
                                          make_signed_t<word> q_inv) {
    auto res = MultMontgomeryLazy<word>(static_cast<make_signed_t<word>>(a),
                                        static_cast<make_signed_t<word>>(b),
                                        q,
                                        q_inv);
    if (res < 0) {
        res += q;
    }
    return static_cast<word>(res);
}

template <typename word>
__device__ __inline__ void ButterflyNTT(make_signed_t<word> &a,
                                        make_signed_t<word> &b,
                                        word w,
                                        word q,
                                        make_signed_t<word> q_inv) {
    if (a < 0) {
        a += q;
    }
    make_signed_t<word> mult =
        MultMontgomeryLazy<word>(b, static_cast<make_signed_t<word>>(w), q, q_inv);
    if (mult < 0) {
        mult += q;
    }
    b = a - mult;
    a = (a - static_cast<make_signed_t<word>>(q)) + mult;
}

template <typename word>
__device__ __inline__ void ButterflyINTT(make_signed_t<word> &a,
                                         make_signed_t<word> &b,
                                         word w,
                                         word q,
                                         make_signed_t<word> q_inv) {
    if (a < 0) {
        a += q;
    }
    if (b < 0) {
        b += q;
    }
    make_signed_t<word> diff = a - b;
    a = (a - static_cast<make_signed_t<word>>(q)) + b;
    b = MultMontgomeryLazy<word>(diff, static_cast<make_signed_t<word>>(w), q, q_inv);
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixNTTFirst(make_signed_t<word> *local,
                                              int tw_idx,
                                              const word *w,
                                              word q,
                                              make_signed_t<word> q_inv) {
    if constexpr (stage > 1) {
        MultiRadixNTTFirst<word, radix, stage - 1>(local, tw_idx, w, q, q_inv);
    }

    constexpr int num_tw = 1 << (stage - 1);
    constexpr int stride = radix / (1 << stage);

    word w_vec[num_tw];
    VectorizedMove<word, num_tw>(w_vec, w + (tw_idx << (stage - 1)));

#pragma unroll
    for (int i = 0; i < num_tw; i++) {
#pragma unroll
        for (int j = 0; j < stride; j++) {
            ButterflyNTT<word>(local[i * 2 * stride + j],
                               local[i * 2 * stride + j + stride],
                               w_vec[i],
                               q,
                               q_inv);
        }
    }
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixINTTLast(make_signed_t<word> *local,
                                              int tw_idx,
                                              const word *w,
                                              word q,
                                              make_signed_t<word> q_inv) {
    constexpr int num_tw = 1 << (stage - 1);
    constexpr int stride = radix / (1 << stage);

    word w_vec[num_tw];
    VectorizedMove<word, num_tw>(w_vec, w + (tw_idx << (stage - 1)));

#pragma unroll
    for (int i = 0; i < num_tw; i++) {
#pragma unroll
        for (int j = 0; j < stride; j++) {
            ButterflyINTT<word>(local[i * 2 * stride + j],
                                local[i * 2 * stride + j + stride],
                                w_vec[i],
                                q,
                                q_inv);
        }
    }

    if constexpr (stage > 1) {
        MultiRadixINTTLast<word, radix, stage - 1>(local, tw_idx, w, q, q_inv);
    }
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixNTT(make_signed_t<word> *local,
                                         int tw_idx,
                                         const word *w,
                                         word q,
                                         make_signed_t<word> q_inv) {
    constexpr int num_tw = radix / (1 << stage);
    constexpr int stride = 1 << (stage - 1);

    word w_vec[num_tw];
    VectorizedMove<word, num_tw>(w_vec, w + tw_idx);

#pragma unroll
    for (int i = 0; i < num_tw; i++) {
#pragma unroll
        for (int j = 0; j < stride; j++) {
            ButterflyNTT<word>(local[i * 2 * stride + j],
                               local[i * 2 * stride + j + stride],
                               w_vec[i],
                               q,
                               q_inv);
        }
    }

    if constexpr (stage > 1) {
        MultiRadixNTT<word, radix, stage - 1>(local, 2 * tw_idx, w, q, q_inv);
    }
}

template <typename word, int radix, int stage, int lsb_size>
__device__ __inline__ void MultiRadixNTT_OT(make_signed_t<word> *local,
                                            int tw_idx,
                                            const word *w,
                                            const word *w_msb,
                                            word q,
                                            make_signed_t<word> q_inv) {
    using signed_word = make_signed_t<word>;
    int last_tw_idx = (1 << (stage - 1)) * tw_idx;
    int msb_idx = last_tw_idx / lsb_size;
    int lsb_idx = last_tw_idx % lsb_size;

    constexpr int num_outer_blocks = radix / (1 << stage);
    constexpr int accumed_tw_num = (1 << stage) - 1;
    word twiddle_factor_set[accumed_tw_num * num_outer_blocks];

    constexpr int num_tw_factor = (1 << (stage - 1)) * num_outer_blocks;
    constexpr int offset = ((1 << (stage - 1)) - 1) * num_outer_blocks;

    if constexpr (kExtendedOT) {
#pragma unroll
        for (int i = 0; i < num_tw_factor; i++) {
            twiddle_factor_set[i + offset] =
                MultMontgomeryLazy<word>(static_cast<signed_word>(w[lsb_idx + i]),
                                         static_cast<signed_word>(w_msb[msb_idx]),
                                         q,
                                         q_inv);
        }

#pragma unroll
        for (int curr_stage = stage; curr_stage > 1; curr_stage--) {
            int src_offset = ((1 << (curr_stage - 1)) - 1) * num_outer_blocks;
            int dst_offset = ((1 << (curr_stage - 2)) - 1) * num_outer_blocks;
            int curr_stage_tw_num = (1 << (curr_stage - 1)) * num_outer_blocks;
#pragma unroll
            for (int i = 0; i < curr_stage_tw_num / 2; i++) {
                word operand = twiddle_factor_set[src_offset + i * 2];
                twiddle_factor_set[dst_offset + i] =
                    MultMontgomeryLazy<word>(static_cast<signed_word>(operand),
                                             static_cast<signed_word>(operand),
                                             q,
                                             q_inv);
            }
        }

#pragma unroll
        for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
            int block_size = 1 << (stage - curr_stage + 1);
            int num_blocks = radix / block_size;
            int tw_offset = ((1 << (curr_stage - 1)) - 1) * num_outer_blocks;
#pragma unroll
            for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
                int stride = block_size / 2;
#pragma unroll
                for (int i = 0; i < stride; i++) {
                    ButterflyNTT<word>(local[curr_block * block_size + i],
                                       local[curr_block * block_size + i + stride],
                                       twiddle_factor_set[tw_offset + curr_block],
                                       q,
                                       q_inv);
                }
            }
        }
    } else {
#pragma unroll
        for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
            int block_size = 1 << (stage - curr_stage + 1);
            int num_blocks = radix / block_size;
            if (curr_stage == stage) {
                word ot_factors[num_tw_factor];
#pragma unroll
                for (int i = 0; i < num_tw_factor; i++) {
                    ot_factors[i] = MultMontgomeryLazy<word>(
                        static_cast<signed_word>(w[lsb_idx + i]),
                        static_cast<signed_word>(w_msb[msb_idx]),
                        q,
                        q_inv);
                }
#pragma unroll
                for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
                    int stride = block_size / 2;
#pragma unroll
                    for (int i = 0; i < stride; i++) {
                        ButterflyNTT<word>(local[curr_block * block_size + i],
                                           local[curr_block * block_size + i + stride],
                                           ot_factors[curr_block],
                                           q,
                                           q_inv);
                    }
                }
                continue;
            }
#pragma unroll
            for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
                int stride = block_size / 2;
#pragma unroll
                for (int i = 0; i < stride; i++) {
                    ButterflyNTT<word>(local[curr_block * block_size + i],
                                       local[curr_block * block_size + i + stride],
                                       w[(1 << (curr_stage - 1)) * tw_idx + curr_block],
                                       q,
                                       q_inv);
                }
            }
        }
    }
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixINTT(make_signed_t<word> *local,
                                          int tw_idx,
                                          const word *w,
                                          word q,
                                          make_signed_t<word> q_inv) {
    if constexpr (stage > 1) {
        MultiRadixINTT<word, radix, stage - 1>(local, 2 * tw_idx, w, q, q_inv);
    }

    constexpr int num_tw = radix / (1 << stage);
    constexpr int stride = 1 << (stage - 1);

    word w_vec[num_tw];
    VectorizedMove<word, num_tw>(w_vec, w + tw_idx);

#pragma unroll
    for (int i = 0; i < num_tw; i++) {
#pragma unroll
        for (int j = 0; j < stride; j++) {
            ButterflyINTT<word>(local[i * 2 * stride + j],
                                local[i * 2 * stride + j + stride],
                                w_vec[i],
                                q,
                                q_inv);
        }
    }
}

template <typename word, int radix, int stage, int lsb_size>
__device__ __inline__ void MultiRadixINTT_OT(make_signed_t<word> *local,
                                             int tw_idx,
                                             const word *w,
                                             const word *w_msb,
                                             word q,
                                             make_signed_t<word> q_inv) {
    using signed_word = make_signed_t<word>;
    int first_tw_idx = (1 << (stage - 1)) * tw_idx;
    int msb_idx = first_tw_idx / lsb_size;
    int lsb_idx = first_tw_idx % lsb_size;

    constexpr int num_outer_blocks = radix / (1 << stage);
    constexpr int accumed_tw_num = (1 << stage) - 1;
    word twiddle_factor_set[accumed_tw_num * num_outer_blocks];

    constexpr int num_tw_factor = (1 << (stage - 1)) * num_outer_blocks;
    if constexpr (kExtendedOT) {
#pragma unroll
        for (int i = 0; i < num_tw_factor; i++) {
            twiddle_factor_set[i] =
                MultMontgomeryLazy<word>(static_cast<signed_word>(w[lsb_idx + i]),
                                         static_cast<signed_word>(w_msb[msb_idx]),
                                         q,
                                         q_inv);
        }

        int accum = 0;
#pragma unroll
        for (int curr_stage = 1; curr_stage < stage; curr_stage++) {
            int curr_stage_tw_num = num_outer_blocks * (1 << (stage - curr_stage));
#pragma unroll
            for (int i = 0; i < curr_stage_tw_num / 2; i++) {
                word operand = twiddle_factor_set[accum + i * 2];
                twiddle_factor_set[curr_stage_tw_num + accum + i] =
                    MultMontgomeryLazy<word>(static_cast<signed_word>(operand),
                                             static_cast<signed_word>(operand),
                                             q,
                                             q_inv);
            }
            accum += curr_stage_tw_num;
        }

        accum = 0;
#pragma unroll
        for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
            int block_size = 1 << curr_stage;
            int num_blocks = radix / block_size;
#pragma unroll
            for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
                int stride = block_size / 2;
#pragma unroll
                for (int i = 0; i < stride; i++) {
                    ButterflyINTT<word>(local[curr_block * block_size + i],
                                        local[curr_block * block_size + i + stride],
                                        twiddle_factor_set[accum + curr_block],
                                        q,
                                        q_inv);
                }
            }
            accum += num_blocks;
        }
    } else {
#pragma unroll
        for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
            int block_size = 1 << curr_stage;
            int num_blocks = radix / block_size;
            if (curr_stage == 1) {
                word ot_factors[num_tw_factor];
#pragma unroll
                for (int i = 0; i < num_tw_factor; i++) {
                    ot_factors[i] = MultMontgomeryLazy<word>(
                        static_cast<signed_word>(w[lsb_idx + i]),
                        static_cast<signed_word>(w_msb[msb_idx]),
                        q,
                        q_inv);
                }
#pragma unroll
                for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
                    int stride = block_size / 2;
#pragma unroll
                    for (int i = 0; i < stride; i++) {
                        ButterflyINTT<word>(local[curr_block * block_size + i],
                                            local[curr_block * block_size + i + stride],
                                            ot_factors[curr_block],
                                            q,
                                            q_inv);
                    }
                }
                continue;
            }
#pragma unroll
            for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
                int stride = block_size / 2;
#pragma unroll
                for (int i = 0; i < stride; i++) {
                    ButterflyINTT<word>(local[curr_block * block_size + i],
                                        local[curr_block * block_size + i + stride],
                                        w[(1 << (stage - curr_stage)) * tw_idx + curr_block],
                                        q,
                                        q_inv);
                }
            }
        }
    }
}

template <int log_degree, NTTType type, Phase phase>
struct NTTLaunchConfig {
    __host__ __device__ static constexpr int RadixStages() {
        if ((type == NTTType::NTT && phase == Phase::Phase1) ||
            (type == NTTType::INTT && phase == Phase::Phase2)) {
            if (log_degree == 16) {
                return 7;
            }
            return log_degree - 9;
        }
        return 9;
    }

    __host__ __device__ static constexpr int StageMerging() {
        if ((type == NTTType::NTT && phase == Phase::Phase1) ||
            (type == NTTType::INTT && phase == Phase::Phase2)) {
            if (log_degree == 16) {
                return 4;
            }
        }
        return 3;
    }

    __host__ __device__ static constexpr int LogWarpBatching() {
        if ((type == NTTType::NTT && phase == Phase::Phase1) ||
            (type == NTTType::INTT && phase == Phase::Phase2)) {
            if (log_degree == 16) {
                return 4;
            }
        }
        return 0;
    }

    __host__ __device__ static constexpr int LsbSize() {
        return 32;
    }

    __host__ __device__ static constexpr bool OFTwiddle() {
        return true;
    }

    __host__ static constexpr int BlockDim() {
        return 1 << (RadixStages() - StageMerging() + LogWarpBatching());
    }
};

}  // namespace cheddar_extract

template <int log_degree>
__global__ void NTTPhase1SinglePrime(int32_t *dst,
                                     const uint32_t *prime_ptr,
                                     const int32_t *inv_prime_ptr,
                                     const uint32_t *twiddle_factors,
                                     int batch_count,
                                     const int32_t *src,
                                     const uint32_t *src_const) {
    extern __shared__ char shared_mem[];
    int32_t *temp = reinterpret_cast<int32_t *>(shared_mem);

    using Config = cheddar_extract::NTTLaunchConfig<log_degree,
                                                    cheddar_extract::NTTType::NTT,
                                                    cheddar_extract::Phase::Phase1>;
    constexpr int kNumStages = Config::RadixStages();
    constexpr int kStageMerging = Config::StageMerging();
    constexpr int kPerThreadElems = 1 << kStageMerging;
    constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
    constexpr int kLogWarpBatching = Config::LogWarpBatching();

    int poly_idx = blockIdx.y;
    if (poly_idx >= batch_count) {
        return;
    }

    uint32_t prime = cheddar_extract::StreamingLoadConst(prime_ptr);
    int32_t inv_prime = cheddar_extract::StreamingLoadConst(inv_prime_ptr);
    const uint32_t *w = twiddle_factors;
    const int32_t *src_limb = src + (static_cast<size_t>(poly_idx) << log_degree);
    int32_t *dst_limb = dst + (static_cast<size_t>(poly_idx) << log_degree);

    int32_t local[kPerThreadElems];
    int stage_group_idx = threadIdx.x >> kLogWarpBatching;
    int batch_lane = threadIdx.x & ((1 << kLogWarpBatching) - 1);
    const int32_t *load_ptr = src_limb + batch_lane +
                              (blockIdx.x << kLogWarpBatching) +
                              (stage_group_idx << (log_degree - kNumStages));
    for (int i = 0; i < kPerThreadElems; i++) {
        local[i] = cheddar_extract::StreamingLoad(load_ptr + (i << (log_degree - kStageMerging)));
    }

    if (src_const != nullptr) {
        uint32_t src_const_value = cheddar_extract::StreamingLoadConst(src_const);
        for (int i = 0; i < kPerThreadElems; i++) {
            local[i] = cheddar_extract::MultMontgomeryLazy<uint32_t>(
                local[i],
                static_cast<int32_t>(src_const_value),
                prime,
                inv_prime);
        }
    }

    int final_tw_idx = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
    int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
    int sm_log_stride = kNumStages - kStageMerging + kLogWarpBatching;

    cheddar_extract::MultiRadixNTTFirst<uint32_t, kPerThreadElems, kTailStages>(
        local, tw_idx, w, prime, inv_prime);
    for (int j = 0; j < kPerThreadElems; j++) {
        temp[threadIdx.x + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kTailStages;

    constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
    for (int i = num_main_iters - 1; i >= 0; i--) {
        int sm_idx =
            ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
            (threadIdx.x & ((1 << sm_log_stride) - 1));
        for (int j = 0; j < kPerThreadElems; j++) {
            local[j] = temp[sm_idx + (j << sm_log_stride)];
        }

        int iter_tw_idx = final_tw_idx >> (kStageMerging * i);
        cheddar_extract::MultiRadixNTT<uint32_t, kPerThreadElems, kStageMerging>(
            local, iter_tw_idx, w, prime, inv_prime);
        if (i == 0) {
            break;
        }
        for (int j = 0; j < kPerThreadElems; j++) {
            temp[sm_idx + (j << sm_log_stride)] = local[j];
        }
        __syncthreads();
        sm_log_stride -= kStageMerging;
    }

    int dst_idx = batch_lane +
                  (stage_group_idx << ((log_degree - kNumStages) + kStageMerging)) +
                  (blockIdx.x << kLogWarpBatching);
    for (int i = 0; i < kPerThreadElems; i++) {
        dst_limb[dst_idx + (i << (log_degree - kNumStages))] = local[i];
    }
}

template <int log_degree>
__global__ void NTTPhase2SinglePrime(int32_t *dst,
                                     const uint32_t *prime_ptr,
                                     const int32_t *inv_prime_ptr,
                                     const uint32_t *twiddle_factors,
                                     const uint32_t *twiddle_factors_msb,
                                     int batch_count,
                                     const int32_t *src) {
    extern __shared__ char shared_mem[];
    int32_t *temp = reinterpret_cast<int32_t *>(shared_mem);

    using Config = cheddar_extract::NTTLaunchConfig<log_degree,
                                                    cheddar_extract::NTTType::NTT,
                                                    cheddar_extract::Phase::Phase2>;
    constexpr int kNumStages = Config::RadixStages();
    constexpr int kStageMerging = Config::StageMerging();
    constexpr int kPerThreadElems = 1 << kStageMerging;
    constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
    constexpr int kLsbSize = Config::LsbSize();
    constexpr int kOFTwiddle = Config::OFTwiddle();
    constexpr int kLogWarpBatching = Config::LogWarpBatching();

    int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
    int batch_lane = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
    temp += row_idx << kNumStages;

    int poly_idx = blockIdx.y;
    if (poly_idx >= batch_count) {
        return;
    }

    uint32_t prime = cheddar_extract::StreamingLoadConst(prime_ptr);
    int32_t inv_prime = cheddar_extract::StreamingLoadConst(inv_prime_ptr);
    const int32_t *src_limb = src + (static_cast<size_t>(poly_idx) << log_degree);
    int32_t *dst_limb = dst + (static_cast<size_t>(poly_idx) << log_degree);
    const uint32_t *w = twiddle_factors;
    const uint32_t *w_msb = twiddle_factors_msb;

    int32_t local[kPerThreadElems];
    int log_stride = kNumStages - kStageMerging;
    const int32_t *load_ptr =
        src_limb + batch_lane + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
        (row_idx << kNumStages);
    for (int i = 0; i < kPerThreadElems; i++) {
        local[i] = cheddar_extract::StreamingLoad(load_ptr + (i << log_stride));
    }

    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int final_tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
    int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
    int sm_log_stride = log_stride;

    cheddar_extract::MultiRadixNTTFirst<uint32_t, kPerThreadElems, kTailStages>(
        local, tw_idx, w, prime, inv_prime);
    for (int j = 0; j < kPerThreadElems; j++) {
        temp[batch_lane + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kTailStages;

    constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
    for (int i = num_main_iters - 1; i >= 0; i--) {
        int sm_idx =
            ((batch_lane >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
            (batch_lane & ((1 << sm_log_stride) - 1));
        for (int j = 0; j < kPerThreadElems; j++) {
            local[j] = temp[sm_idx + (j << sm_log_stride)];
        }

        int iter_tw_idx = final_tw_idx >> (kStageMerging * i);
        if (i == 0) {
            if constexpr (kOFTwiddle) {
                cheddar_extract::MultiRadixNTT_OT<uint32_t,
                                                  kPerThreadElems,
                                                  kStageMerging,
                                                  kLsbSize>(local,
                                                            iter_tw_idx,
                                                            w,
                                                            w_msb,
                                                            prime,
                                                            inv_prime);
            } else {
                cheddar_extract::MultiRadixNTT<uint32_t, kPerThreadElems, kStageMerging>(
                    local, iter_tw_idx, w, prime, inv_prime);
            }
        } else {
            if constexpr (kOFTwiddle && !cheddar_extract::kExtendedOT) {
                cheddar_extract::MultiRadixNTT_OT<uint32_t,
                                                  kPerThreadElems,
                                                  kStageMerging,
                                                  kLsbSize>(local,
                                                            iter_tw_idx,
                                                            w,
                                                            w_msb,
                                                            prime,
                                                            inv_prime);
            } else {
                cheddar_extract::MultiRadixNTT<uint32_t, kPerThreadElems, kStageMerging>(
                    local, iter_tw_idx, w, prime, inv_prime);
            }
        }

        if (i == 0) {
            break;
        }
        for (int j = 0; j < kPerThreadElems; j++) {
            temp[sm_idx + (j << sm_log_stride)] = local[j];
        }
        __syncthreads();
        sm_log_stride -= kStageMerging;
    }

    for (int i = 0; i < kPerThreadElems; i++) {
        if (local[i] < 0) {
            local[i] += prime;
        }
    }

    int32_t *dst_ptr = dst_limb + (x_idx << kStageMerging);
    cheddar_extract::VectorizedMove<int32_t, kPerThreadElems>(dst_ptr, local);
}

template <int log_degree>
__global__ void INTTPhase1SinglePrime(int32_t *dst,
                                      const uint32_t *prime_ptr,
                                      const int32_t *inv_prime_ptr,
                                      const uint32_t *twiddle_factors,
                                      const uint32_t *twiddle_factors_msb,
                                      int batch_count,
                                      const int32_t *src) {
    extern __shared__ char shared_mem[];
    int32_t *temp = reinterpret_cast<int32_t *>(shared_mem);

    using Config = cheddar_extract::NTTLaunchConfig<log_degree,
                                                    cheddar_extract::NTTType::INTT,
                                                    cheddar_extract::Phase::Phase1>;
    constexpr int kNumStages = Config::RadixStages();
    constexpr int kStageMerging = Config::StageMerging();
    constexpr int kPerThreadElems = 1 << kStageMerging;
    constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
    constexpr int kLsbSize = Config::LsbSize();
    constexpr int kOFTwiddle = Config::OFTwiddle();
    constexpr int kLogWarpBatching = Config::LogWarpBatching();

    int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
    int batch_lane = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
    temp += row_idx << kNumStages;

    int poly_idx = blockIdx.y;
    if (poly_idx >= batch_count) {
        return;
    }

    uint32_t prime = cheddar_extract::StreamingLoadConst(prime_ptr);
    int32_t inv_prime = cheddar_extract::StreamingLoadConst(inv_prime_ptr);
    const int32_t *src_limb = src + (static_cast<size_t>(poly_idx) << log_degree);
    int32_t *dst_limb = dst + (static_cast<size_t>(poly_idx) << log_degree);
    const uint32_t *w = twiddle_factors;
    const uint32_t *w_msb = twiddle_factors_msb;

    int32_t local[kPerThreadElems];
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t *load_ptr = src_limb + (x_idx << kStageMerging);
    cheddar_extract::VectorizedMove<int32_t, kPerThreadElems>(local, load_ptr);

    int tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
    int sm_log_stride = 0;
    int sm_idx = batch_lane << kStageMerging;

    constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
    for (int i = 0; i < num_main_iters; i++) {
        if (i == 0) {
            if constexpr (kOFTwiddle) {
                cheddar_extract::MultiRadixINTT_OT<uint32_t,
                                                   kPerThreadElems,
                                                   kStageMerging,
                                                   kLsbSize>(local,
                                                             tw_idx,
                                                             w,
                                                             w_msb,
                                                             prime,
                                                             inv_prime);
            } else {
                cheddar_extract::MultiRadixINTT<uint32_t, kPerThreadElems, kStageMerging>(
                    local, tw_idx, w, prime, inv_prime);
            }
        } else {
            if constexpr (kOFTwiddle && !cheddar_extract::kExtendedOT) {
                cheddar_extract::MultiRadixINTT_OT<uint32_t,
                                                   kPerThreadElems,
                                                   kStageMerging,
                                                   kLsbSize>(local,
                                                             tw_idx,
                                                             w,
                                                             w_msb,
                                                             prime,
                                                             inv_prime);
            } else {
                cheddar_extract::MultiRadixINTT<uint32_t, kPerThreadElems, kStageMerging>(
                    local, tw_idx, w, prime, inv_prime);
            }
        }

        for (int j = 0; j < kPerThreadElems; j++) {
            temp[sm_idx + (j << sm_log_stride)] = local[j];
        }
        __syncthreads();

        if (i == num_main_iters - 1) {
            tw_idx >>= kTailStages;
            sm_log_stride += kTailStages;
        } else {
            tw_idx >>= kStageMerging;
            sm_log_stride += kStageMerging;
        }

        sm_idx = (batch_lane & ((1 << sm_log_stride) - 1)) +
                 ((batch_lane >> sm_log_stride) << (sm_log_stride + kStageMerging));
        for (int j = 0; j < kPerThreadElems; j++) {
            local[j] = temp[sm_idx + (j << sm_log_stride)];
        }
    }

    cheddar_extract::MultiRadixINTTLast<uint32_t, kPerThreadElems, kTailStages>(
        local, tw_idx, w, prime, inv_prime);

    int dst_idx = batch_lane + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
                  (row_idx << kNumStages);
    for (int j = 0; j < kPerThreadElems; j++) {
        dst_limb[dst_idx + (j << (kNumStages - kStageMerging))] = local[j];
    }
}

template <int log_degree>
__global__ void INTTPhase2SinglePrime(int32_t *dst,
                                      const uint32_t *prime_ptr,
                                      const int32_t *inv_prime_ptr,
                                      const uint32_t *twiddle_factors,
                                      int batch_count,
                                      const int32_t *src,
                                      const uint32_t *src_const) {
    extern __shared__ char shared_mem[];
    int32_t *temp = reinterpret_cast<int32_t *>(shared_mem);

    using Config = cheddar_extract::NTTLaunchConfig<log_degree,
                                                    cheddar_extract::NTTType::INTT,
                                                    cheddar_extract::Phase::Phase2>;
    constexpr int kNumStages = Config::RadixStages();
    constexpr int kStageMerging = Config::StageMerging();
    constexpr int kPerThreadElems = 1 << kStageMerging;
    constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
    constexpr int kLogWarpBatching = Config::LogWarpBatching();

    int poly_idx = blockIdx.y;
    if (poly_idx >= batch_count) {
        return;
    }

    uint32_t prime = cheddar_extract::StreamingLoadConst(prime_ptr);
    int32_t inv_prime = cheddar_extract::StreamingLoadConst(inv_prime_ptr);
    uint32_t src_const_value = cheddar_extract::StreamingLoadConst(src_const);
    const int32_t *src_limb = src + (static_cast<size_t>(poly_idx) << log_degree);
    int32_t *dst_limb = dst + (static_cast<size_t>(poly_idx) << log_degree);
    const uint32_t *w = twiddle_factors;

    int32_t local[kPerThreadElems];
    constexpr int initial_log_stride = log_degree - kNumStages;
    int stage_group_idx = threadIdx.x >> kLogWarpBatching;
    int batch_lane = threadIdx.x & ((1 << kLogWarpBatching) - 1);
    const int32_t *load_ptr =
        src_limb + (stage_group_idx << (initial_log_stride + kStageMerging)) +
        batch_lane + (blockIdx.x << kLogWarpBatching);
    for (int i = 0; i < kPerThreadElems; i++) {
        local[i] = cheddar_extract::StreamingLoad(load_ptr + (i << initial_log_stride));
    }

    int tw_idx = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
    int sm_log_stride = kLogWarpBatching;
    int sm_idx = (threadIdx.x & ((1 << sm_log_stride) - 1)) +
                 ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));

    constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
    for (int i = 0; i < num_main_iters; i++) {
        cheddar_extract::MultiRadixINTT<uint32_t, kPerThreadElems, kStageMerging>(
            local, tw_idx, w, prime, inv_prime);

        for (int j = 0; j < kPerThreadElems; j++) {
            temp[sm_idx + (j << sm_log_stride)] = local[j];
        }
        __syncthreads();

        if (i == num_main_iters - 1) {
            tw_idx >>= kTailStages;
            sm_log_stride += kTailStages;
        } else {
            tw_idx >>= kStageMerging;
            sm_log_stride += kStageMerging;
        }

        sm_idx = (threadIdx.x & ((1 << sm_log_stride) - 1)) +
                 ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));
        for (int j = 0; j < kPerThreadElems; j++) {
            local[j] = temp[sm_idx + (j << sm_log_stride)];
        }
    }

    cheddar_extract::MultiRadixINTTLast<uint32_t, kPerThreadElems, kTailStages>(
        local, tw_idx, w, prime, inv_prime);

    int dst_idx = batch_lane + (stage_group_idx << initial_log_stride) +
                  (blockIdx.x << kLogWarpBatching);
    for (int j = 0; j < kPerThreadElems; j++) {
        int32_t temp_val = cheddar_extract::MultMontgomeryLazy<uint32_t>(
            local[j],
            static_cast<int32_t>(src_const_value),
            prime,
            inv_prime);
        if (temp_val < 0) {
            temp_val += prime;
        }
        dst_limb[dst_idx + (j << (log_degree - kStageMerging))] = temp_val;
    }
}

__global__ void pointwise_mul_kernel(const uint32_t *a,
                                     const uint32_t *b,
                                     uint32_t *out,
                                     size_t total_coeffs,
                                     const uint32_t *prime_ptr,
                                     const int32_t *inv_prime_ptr) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) {
        return;
    }
    uint32_t prime = cheddar_extract::StreamingLoadConst(prime_ptr);
    int32_t inv_prime = cheddar_extract::StreamingLoadConst(inv_prime_ptr);
    out[idx] = cheddar_extract::MultMontgomery<uint32_t>(a[idx], b[idx], prime, inv_prime);
}

static unsigned int grid_size(size_t work_items, unsigned int block_size) {
    return static_cast<unsigned int>((work_items + block_size - 1) / block_size);
}

template <int log_degree>
static void run_forward_only_impl(uint32_t *d_input,
                                  uint32_t *d_work,
                                  const DeviceTables &tables,
                                  int batch_count,
                                  bool convert_to_montgomery) {
    using Config1 = cheddar_extract::NTTLaunchConfig<log_degree,
                                                     cheddar_extract::NTTType::NTT,
                                                     cheddar_extract::Phase::Phase1>;
    using Config2 = cheddar_extract::NTTLaunchConfig<log_degree,
                                                     cheddar_extract::NTTType::NTT,
                                                     cheddar_extract::Phase::Phase2>;

    constexpr int degree = 1 << log_degree;
    constexpr int block_dim_1 = Config1::BlockDim();
    constexpr int stage_merging_1 = Config1::StageMerging();
    constexpr int block_dim_2 = Config2::BlockDim();
    constexpr int stage_merging_2 = Config2::StageMerging();
    constexpr int shared_mem_1 = block_dim_1 * (1 << stage_merging_1) * sizeof(uint32_t);
    constexpr int shared_mem_2 = block_dim_2 * (1 << stage_merging_2) * sizeof(uint32_t);

    dim3 grid_phase1(degree / (1 << stage_merging_1) / block_dim_1,
                     static_cast<unsigned int>(batch_count));
    dim3 grid_phase2(degree / (1 << stage_merging_2) / block_dim_2,
                     static_cast<unsigned int>(batch_count));
    const uint32_t *src_const = convert_to_montgomery ? tables.d_montgomery_converter : nullptr;

    NTTPhase1SinglePrime<log_degree><<<grid_phase1, block_dim_1, shared_mem_1>>>(
        reinterpret_cast<int32_t *>(d_work),
        tables.d_primes,
        tables.d_inv_primes,
        tables.d_fwd_twiddles,
        batch_count,
        reinterpret_cast<const int32_t *>(d_input),
        src_const);
    check_launch("launch cheddar NTTPhase1SinglePrime");

    NTTPhase2SinglePrime<log_degree><<<grid_phase2, block_dim_2, shared_mem_2>>>(
        reinterpret_cast<int32_t *>(d_work),
        tables.d_primes,
        tables.d_inv_primes,
        tables.d_fwd_twiddles,
        tables.d_fwd_twiddles_msb,
        batch_count,
        reinterpret_cast<const int32_t *>(d_work));
    check_launch("launch cheddar NTTPhase2SinglePrime");
}

static void run_forward_only(uint32_t *d_input,
                             uint32_t *d_work,
                             const DeviceTables &tables,
                             int n,
                             int batch_count,
                             int log_degree,
                             bool convert_to_montgomery = true) {
    bool launched = false;
    constexpr_for<kMinLogDegree, kMaxLogDegree + 1>([&](auto degree_c) {
        if (log_degree != degree_c) {
            return;
        }
        run_forward_only_impl<degree_c>(d_input, d_work, tables, batch_count, convert_to_montgomery);
        launched = true;
    });
    if (!launched) {
        std::cerr << "Unsupported log_degree in run_forward_only: " << log_degree << "\n";
        std::exit(1);
    }
}

template <int log_degree>
static void run_inverse_only_impl(uint32_t *d_ntt_input,
                                  uint32_t *d_out,
                                  const DeviceTables &tables,
                                  int batch_count,
                                  bool convert_from_montgomery) {
    using Config1 = cheddar_extract::NTTLaunchConfig<log_degree,
                                                     cheddar_extract::NTTType::INTT,
                                                     cheddar_extract::Phase::Phase1>;
    using Config2 = cheddar_extract::NTTLaunchConfig<log_degree,
                                                     cheddar_extract::NTTType::INTT,
                                                     cheddar_extract::Phase::Phase2>;

    constexpr int degree = 1 << log_degree;
    constexpr int block_dim_1 = Config1::BlockDim();
    constexpr int stage_merging_1 = Config1::StageMerging();
    constexpr int block_dim_2 = Config2::BlockDim();
    constexpr int stage_merging_2 = Config2::StageMerging();
    constexpr int shared_mem_1 = block_dim_1 * (1 << stage_merging_1) * sizeof(uint32_t);
    constexpr int shared_mem_2 = block_dim_2 * (1 << stage_merging_2) * sizeof(uint32_t);

    dim3 grid_phase1(degree / (1 << stage_merging_1) / block_dim_1,
                     static_cast<unsigned int>(batch_count));
    dim3 grid_phase2(degree / (1 << stage_merging_2) / block_dim_2,
                     static_cast<unsigned int>(batch_count));
    const uint32_t *src_const = convert_from_montgomery ? tables.d_inv_degree : tables.d_inv_degree_mont;

    INTTPhase1SinglePrime<log_degree><<<grid_phase1, block_dim_1, shared_mem_1>>>(
        reinterpret_cast<int32_t *>(d_out),
        tables.d_primes,
        tables.d_inv_primes,
        tables.d_inv_twiddles,
        tables.d_inv_twiddles_msb,
        batch_count,
        reinterpret_cast<const int32_t *>(d_ntt_input));
    check_launch("launch cheddar INTTPhase1SinglePrime");

    INTTPhase2SinglePrime<log_degree><<<grid_phase2, block_dim_2, shared_mem_2>>>(
        reinterpret_cast<int32_t *>(d_out),
        tables.d_primes,
        tables.d_inv_primes,
        tables.d_inv_twiddles,
        batch_count,
        reinterpret_cast<const int32_t *>(d_out),
        src_const);
    check_launch("launch cheddar INTTPhase2SinglePrime");
}

static void run_inverse_only(uint32_t *d_ntt_input,
                             uint32_t *d_out,
                             const DeviceTables &tables,
                             int n,
                             int batch_count,
                             int log_degree,
                             bool convert_from_montgomery = true) {
    bool launched = false;
    constexpr_for<kMinLogDegree, kMaxLogDegree + 1>([&](auto degree_c) {
        if (log_degree != degree_c) {
            return;
        }
        run_inverse_only_impl<degree_c>(d_ntt_input, d_out, tables, batch_count, convert_from_montgomery);
        launched = true;
    });
    if (!launched) {
        std::cerr << "Unsupported log_degree in run_inverse_only: " << log_degree << "\n";
        std::exit(1);
    }
}

static void run_full_polymul(uint32_t *d_a,
                             uint32_t *d_b,
                             uint32_t *d_a_work,
                             uint32_t *d_b_work,
                             uint32_t *d_c_work,
                             uint32_t *d_out,
                             const DeviceTables &tables,
                             int n,
                             int batch_count,
                             int log_degree) {
    run_forward_only(d_a, d_a_work, tables, n, batch_count, log_degree, true);
    run_forward_only(d_b, d_b_work, tables, n, batch_count, log_degree, true);

    dim3 block(256);
    size_t total_coeffs = static_cast<size_t>(batch_count) * static_cast<size_t>(n);
    dim3 grid(grid_size(total_coeffs, block.x));
    pointwise_mul_kernel<<<grid, block>>>(
        d_a_work, d_b_work, d_c_work, total_coeffs, tables.d_primes, tables.d_inv_primes);
    check_launch("launch cheddar pointwise_mul_kernel");

    run_inverse_only(d_c_work, d_out, tables, n, batch_count, log_degree, true);
}

static bool validate_roundtrip_case(const char *label,
                                    const std::vector<uint32_t> &input,
                                    int batch_count,
                                    uint32_t *d_input,
                                    uint32_t *d_work,
                                    uint32_t *d_out,
                                    const DeviceTables &tables,
                                    int n,
                                    int log_degree) {
    check(cudaMemcpy(d_input,
                     input.data(),
                     sizeof(uint32_t) * input.size(),
                     cudaMemcpyHostToDevice),
          "copy roundtrip input");
    run_forward_only(d_input, d_work, tables, n, batch_count, log_degree, true);
    run_inverse_only(d_work, d_out, tables, n, batch_count, log_degree, true);
    check(cudaDeviceSynchronize(), "sync roundtrip validation");
    std::vector<uint32_t> output(input.size());
    check(cudaMemcpy(output.data(),
                     d_out,
                     sizeof(uint32_t) * output.size(),
                     cudaMemcpyDeviceToHost),
          "copy roundtrip output");
    return compare_vectors(input, output, n, label);
}

static bool validate_polymul_case(const char *label,
                                  const std::vector<uint32_t> &lhs,
                                  const std::vector<uint32_t> &rhs,
                                  int batch_count,
                                  uint32_t *d_a,
                                  uint32_t *d_b,
                                  uint32_t *d_a_work,
                                  uint32_t *d_b_work,
                                  uint32_t *d_c_work,
                                  uint32_t *d_out,
                                  const DeviceTables &tables,
                                  const std::vector<uint32_t> &phi_norm,
                                  const std::vector<uint32_t> &post_norm,
                                  int n,
                                  int log_degree) {
    std::vector<uint32_t> expected(lhs.size());
    for (int batch = 0; batch < batch_count; batch++) {
        std::vector<uint32_t> lhs_poly(lhs.begin() + static_cast<size_t>(batch) * static_cast<size_t>(n),
                                       lhs.begin() + static_cast<size_t>(batch + 1) * static_cast<size_t>(n));
        std::vector<uint32_t> rhs_poly(rhs.begin() + static_cast<size_t>(batch) * static_cast<size_t>(n),
                                       rhs.begin() + static_cast<size_t>(batch + 1) * static_cast<size_t>(n));
        std::vector<uint32_t> poly =
            host_polymul_reference(lhs_poly, rhs_poly, phi_norm, post_norm, n, log_degree);
        std::copy(poly.begin(), poly.end(),
                  expected.begin() + static_cast<size_t>(batch) * static_cast<size_t>(n));
    }

    check(cudaMemcpy(d_a, lhs.data(), sizeof(uint32_t) * lhs.size(), cudaMemcpyHostToDevice),
          "copy polymul lhs");
    check(cudaMemcpy(d_b, rhs.data(), sizeof(uint32_t) * rhs.size(), cudaMemcpyHostToDevice),
          "copy polymul rhs");
    run_full_polymul(d_a,
                     d_b,
                     d_a_work,
                     d_b_work,
                     d_c_work,
                     d_out,
                     tables,
                     n,
                     batch_count,
                     log_degree);
    check(cudaDeviceSynchronize(), "sync polymul validation");

    std::vector<uint32_t> actual(lhs.size());
    check(cudaMemcpy(actual.data(), d_out, sizeof(uint32_t) * actual.size(), cudaMemcpyDeviceToHost),
          "copy polymul output");
    return compare_vectors(expected, actual, n, label);
}

static bool run_validation_suite(uint32_t *d_a,
                                 uint32_t *d_b,
                                 uint32_t *d_a_work,
                                 uint32_t *d_b_work,
                                 uint32_t *d_c_work,
                                 uint32_t *d_out,
                                 const DeviceTables &tables,
                                 const std::vector<uint32_t> &phi_norm,
                                 const std::vector<uint32_t> &post_norm,
                                 int n,
                                 int batch_size,
                                 int log_degree) {
    const int validation_batches = std::min(batch_size, kMaxValidationBatches);
    bool ok = true;

    ok = validate_roundtrip_case("roundtrip zeros",
                                 make_batched_pattern(validation_batches, n, InputPattern::Zero),
                                 validation_batches,
                                 d_a,
                                 d_a_work,
                                 d_out,
                                 tables,
                                 n,
                                 log_degree) && ok;
    ok = validate_roundtrip_case("roundtrip ones",
                                 make_batched_pattern(validation_batches, n, InputPattern::One),
                                 validation_batches,
                                 d_a,
                                 d_a_work,
                                 d_out,
                                 tables,
                                 n,
                                 log_degree) && ok;
    ok = validate_roundtrip_case("roundtrip impulse",
                                 make_batched_pattern(validation_batches, n, InputPattern::Impulse),
                                 validation_batches,
                                 d_a,
                                 d_a_work,
                                 d_out,
                                 tables,
                                 n,
                                 log_degree) && ok;
    ok = validate_roundtrip_case("roundtrip max",
                                 make_batched_pattern(validation_batches, n, InputPattern::Max),
                                 validation_batches,
                                 d_a,
                                 d_a_work,
                                 d_out,
                                 tables,
                                 n,
                                 log_degree) && ok;
    ok = validate_roundtrip_case("roundtrip random",
                                 make_batched_pattern(validation_batches, n, InputPattern::Random, 12345u),
                                 validation_batches,
                                 d_a,
                                 d_a_work,
                                 d_out,
                                 tables,
                                 n,
                                 log_degree) && ok;

    ok = validate_polymul_case("polymul zero*zero",
                               make_batched_pattern(validation_batches, n, InputPattern::Zero),
                               make_batched_pattern(validation_batches, n, InputPattern::Zero),
                               validation_batches,
                               d_a,
                               d_b,
                               d_a_work,
                               d_b_work,
                               d_c_work,
                               d_out,
                               tables,
                               phi_norm,
                               post_norm,
                               n,
                               log_degree) && ok;
    ok = validate_polymul_case("polymul one*one",
                               make_batched_pattern(validation_batches, n, InputPattern::One),
                               make_batched_pattern(validation_batches, n, InputPattern::One),
                               validation_batches,
                               d_a,
                               d_b,
                               d_a_work,
                               d_b_work,
                               d_c_work,
                               d_out,
                               tables,
                               phi_norm,
                               post_norm,
                               n,
                               log_degree) && ok;
    ok = validate_polymul_case("polymul impulse*ones",
                               make_batched_pattern(validation_batches, n, InputPattern::Impulse),
                               make_batched_pattern(validation_batches, n, InputPattern::One),
                               validation_batches,
                               d_a,
                               d_b,
                               d_a_work,
                               d_b_work,
                               d_c_work,
                               d_out,
                               tables,
                               phi_norm,
                               post_norm,
                               n,
                               log_degree) && ok;
    ok = validate_polymul_case("polymul max*max",
                               make_batched_pattern(validation_batches, n, InputPattern::Max),
                               make_batched_pattern(validation_batches, n, InputPattern::Max),
                               validation_batches,
                               d_a,
                               d_b,
                               d_a_work,
                               d_b_work,
                               d_c_work,
                               d_out,
                               tables,
                               phi_norm,
                               post_norm,
                               n,
                               log_degree) && ok;
    ok = validate_polymul_case("polymul random",
                               make_batched_pattern(validation_batches, n, InputPattern::Random, 1u),
                               make_batched_pattern(validation_batches, n, InputPattern::Random, 2u),
                               validation_batches,
                               d_a,
                               d_b,
                               d_a_work,
                               d_b_work,
                               d_c_work,
                               d_out,
                               tables,
                               phi_norm,
                               post_norm,
                               n,
                               log_degree) && ok;

    return ok;
}

static void alloc_and_copy(DeviceTables &tables, const HostTables &host_tables) {
    check(cudaMalloc(&tables.d_primes, sizeof(uint32_t) * host_tables.primes.size()), "malloc primes");
    check(cudaMalloc(&tables.d_inv_primes, sizeof(int32_t) * host_tables.inv_primes.size()), "malloc inv primes");
    check(cudaMalloc(&tables.d_fwd_twiddles,
                     sizeof(uint32_t) * host_tables.fwd_twiddles_mont.size()),
          "malloc forward twiddles");
    check(cudaMalloc(&tables.d_fwd_twiddles_msb,
                     sizeof(uint32_t) * host_tables.fwd_twiddles_msb.size()),
          "malloc forward twiddles msb");
    check(cudaMalloc(&tables.d_inv_twiddles,
                     sizeof(uint32_t) * host_tables.inv_twiddles_mont.size()),
          "malloc inverse twiddles");
    check(cudaMalloc(&tables.d_inv_twiddles_msb,
                     sizeof(uint32_t) * host_tables.inv_twiddles_msb.size()),
          "malloc inverse twiddles msb");
    check(cudaMalloc(&tables.d_inv_degree, sizeof(uint32_t) * host_tables.inv_degree.size()),
          "malloc inverse degree");
    check(cudaMalloc(&tables.d_inv_degree_mont,
                     sizeof(uint32_t) * host_tables.inv_degree_mont.size()),
          "malloc inverse degree mont");
    check(cudaMalloc(&tables.d_montgomery_converter,
                     sizeof(uint32_t) * host_tables.montgomery_converter.size()),
          "malloc montgomery converter");

    check(cudaMemcpy(tables.d_primes,
                     host_tables.primes.data(),
                     sizeof(uint32_t) * host_tables.primes.size(),
                     cudaMemcpyHostToDevice),
          "copy primes");
    check(cudaMemcpy(tables.d_inv_primes,
                     host_tables.inv_primes.data(),
                     sizeof(int32_t) * host_tables.inv_primes.size(),
                     cudaMemcpyHostToDevice),
          "copy inv primes");
    check(cudaMemcpy(tables.d_fwd_twiddles,
                     host_tables.fwd_twiddles_mont.data(),
                     sizeof(uint32_t) * host_tables.fwd_twiddles_mont.size(),
                     cudaMemcpyHostToDevice),
          "copy forward twiddles");
    check(cudaMemcpy(tables.d_fwd_twiddles_msb,
                     host_tables.fwd_twiddles_msb.data(),
                     sizeof(uint32_t) * host_tables.fwd_twiddles_msb.size(),
                     cudaMemcpyHostToDevice),
          "copy forward twiddles msb");
    check(cudaMemcpy(tables.d_inv_twiddles,
                     host_tables.inv_twiddles_mont.data(),
                     sizeof(uint32_t) * host_tables.inv_twiddles_mont.size(),
                     cudaMemcpyHostToDevice),
          "copy inverse twiddles");
    check(cudaMemcpy(tables.d_inv_twiddles_msb,
                     host_tables.inv_twiddles_msb.data(),
                     sizeof(uint32_t) * host_tables.inv_twiddles_msb.size(),
                     cudaMemcpyHostToDevice),
          "copy inverse twiddles msb");
    check(cudaMemcpy(tables.d_inv_degree,
                     host_tables.inv_degree.data(),
                     sizeof(uint32_t) * host_tables.inv_degree.size(),
                     cudaMemcpyHostToDevice),
          "copy inverse degree");
    check(cudaMemcpy(tables.d_inv_degree_mont,
                     host_tables.inv_degree_mont.data(),
                     sizeof(uint32_t) * host_tables.inv_degree_mont.size(),
                     cudaMemcpyHostToDevice),
          "copy inverse degree mont");
    check(cudaMemcpy(tables.d_montgomery_converter,
                     host_tables.montgomery_converter.data(),
                     sizeof(uint32_t) * host_tables.montgomery_converter.size(),
                     cudaMemcpyHostToDevice),
          "copy montgomery converter");
}

static void free_tables(DeviceTables &tables) {
    cudaFree(tables.d_primes);
    cudaFree(tables.d_inv_primes);
    cudaFree(tables.d_fwd_twiddles);
    cudaFree(tables.d_fwd_twiddles_msb);
    cudaFree(tables.d_inv_twiddles);
    cudaFree(tables.d_inv_twiddles_msb);
    cudaFree(tables.d_inv_degree);
    cudaFree(tables.d_inv_degree_mont);
    cudaFree(tables.d_montgomery_converter);
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,n,logn,requested_qbits,actual_qbits,batch_size,iters,validation,ntt_mean_us,ntt_std_us,intt_mean_us,intt_std_us,poly_mul_mean_us,poly_mul_std_us,correct\n";
    }

    const int n = args.n;
    const int log_degree = int_log2(static_cast<uint32_t>(n));
    const size_t total_coeffs = static_cast<size_t>(args.batch_size) * static_cast<size_t>(n);

    HostTables host_tables;
    compute_cheddar_tables(host_tables, n);

    DeviceTables tables;
    alloc_and_copy(tables, host_tables);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, kModulus - 1);
    std::vector<uint32_t> a(total_coeffs);
    std::vector<uint32_t> b(total_coeffs);
    for (size_t i = 0; i < total_coeffs; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    uint32_t *d_a = nullptr;
    uint32_t *d_b = nullptr;
    uint32_t *d_a_work = nullptr;
    uint32_t *d_b_work = nullptr;
    uint32_t *d_c_work = nullptr;
    uint32_t *d_out = nullptr;
    check(cudaMalloc(&d_a, sizeof(uint32_t) * total_coeffs), "malloc d_a");
    check(cudaMalloc(&d_b, sizeof(uint32_t) * total_coeffs), "malloc d_b");
    check(cudaMalloc(&d_a_work, sizeof(uint32_t) * total_coeffs), "malloc d_a_work");
    check(cudaMalloc(&d_b_work, sizeof(uint32_t) * total_coeffs), "malloc d_b_work");
    check(cudaMalloc(&d_c_work, sizeof(uint32_t) * total_coeffs), "malloc d_c_work");
    check(cudaMalloc(&d_out, sizeof(uint32_t) * total_coeffs), "malloc d_out");
    check(cudaMemcpy(d_a, a.data(), sizeof(uint32_t) * total_coeffs, cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(d_b, b.data(), sizeof(uint32_t) * total_coeffs, cudaMemcpyHostToDevice), "copy b");

    std::vector<uint32_t> phi_norm;
    std::vector<uint32_t> post_norm;
    compute_reference_vectors(phi_norm, post_norm, n);

    bool correct = true;
    if (!args.skip_validation) {
        correct = run_validation_suite(d_a,
                                       d_b,
                                       d_a_work,
                                       d_b_work,
                                       d_c_work,
                                       d_out,
                                       tables,
                                       phi_norm,
                                       post_norm,
                                       n,
                                       args.batch_size,
                                       log_degree);
    }

    check(cudaMemcpy(d_a, a.data(), sizeof(uint32_t) * total_coeffs, cudaMemcpyHostToDevice), "restore benchmark a");
    check(cudaMemcpy(d_b, b.data(), sizeof(uint32_t) * total_coeffs, cudaMemcpyHostToDevice), "restore benchmark b");

    std::vector<double> ntt_samples;
    std::vector<double> intt_samples;
    std::vector<double> poly_samples;
    ntt_samples.reserve(args.iters);
    intt_samples.reserve(args.iters);
    poly_samples.reserve(args.iters);

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
    check(cudaEventCreate(&start_evt), "create start event");
    check(cudaEventCreate(&stop_evt), "create stop event");

    run_forward_only(d_a, d_a_work, tables, n, args.batch_size, log_degree, true);
    check(cudaDeviceSynchronize(), "sync prep forward");

    for (int i = 0; i < args.warmup; i++) {
        run_forward_only(d_a, d_a_work, tables, n, args.batch_size, log_degree, true);
        run_inverse_only(d_a_work, d_out, tables, n, args.batch_size, log_degree, true);
        run_full_polymul(d_a,
                         d_b,
                         d_a_work,
                         d_b_work,
                         d_c_work,
                         d_out,
                         tables,
                         n,
                         args.batch_size,
                         log_degree);
        check(cudaDeviceSynchronize(), "sync warmup");
    }

    for (int i = 0; i < args.iters; i++) {
        check(cudaEventRecord(start_evt), "record start ntt");
        run_forward_only(d_a, d_a_work, tables, n, args.batch_size, log_degree, true);
        check(cudaEventRecord(stop_evt), "record stop ntt");
        check(cudaEventSynchronize(stop_evt), "sync stop ntt");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_evt, stop_evt), "elapsed ntt");
        ntt_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(start_evt), "record start intt");
        run_inverse_only(d_a_work, d_out, tables, n, args.batch_size, log_degree, true);
        check(cudaEventRecord(stop_evt), "record stop intt");
        check(cudaEventSynchronize(stop_evt), "sync stop intt");
        check(cudaEventElapsedTime(&ms, start_evt, stop_evt), "elapsed intt");
        intt_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(start_evt), "record start polymul");
        run_full_polymul(d_a,
                         d_b,
                         d_a_work,
                         d_b_work,
                         d_c_work,
                         d_out,
                         tables,
                         n,
                         args.batch_size,
                         log_degree);
        check(cudaEventRecord(stop_evt), "record stop polymul");
        check(cudaEventSynchronize(stop_evt), "sync stop polymul");
        check(cudaEventElapsedTime(&ms, start_evt, stop_evt), "elapsed polymul");
        poly_samples.push_back(ms * 1000.0);
    }

    Stats ntt = compute_stats(ntt_samples);
    Stats intt = compute_stats(intt_samples);
    Stats poly = compute_stats(poly_samples);
    const char *validation = args.skip_validation ? "skipped" : (correct ? "pass" : "fail");
    const int correct_flag = args.skip_validation ? -1 : (correct ? 1 : 0);

    std::cout << RINGLPN_DEVICE_LABEL << "," << n << "," << log_degree << ","
              << args.requested_qbits << "," << kActualQBits << ","
              << args.batch_size << "," << args.iters << ","
              << validation << ","
              << ntt.mean_us << "," << ntt.stddev_us << ","
              << intt.mean_us << "," << intt.stddev_us << ","
              << poly.mean_us << "," << poly.stddev_us << ","
              << correct_flag << "\n";

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    free_tables(tables);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_work);
    cudaFree(d_b_work);
    cudaFree(d_c_work);
    cudaFree(d_out);
    return (args.skip_validation || correct) ? 0 : 2;
}