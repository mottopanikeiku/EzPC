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
#include <vector>

namespace {

constexpr int kMaxDegree = 16384;
constexpr uint32_t kModulus = 1073479681u;
constexpr uint32_t kPrimitiveRootMax = 31849551u;
constexpr int kMaxPolyDegree = 32768;
constexpr int kFusedStages = 8;

struct Stats {
    double mean_us;
    double stddev_us;
};

struct Args {
    int n = 4096;
    int iters = 10000;
    int warmup = 1000;
    bool csv_header = false;
};

struct DeviceTables {
    uint32_t *d_phi = nullptr;
    uint32_t *d_fwd_twiddles = nullptr;
    uint32_t *d_inv_twiddles = nullptr;
    uint32_t *d_stage_offsets = nullptr;
    uint32_t *d_post_scale = nullptr;
};

static void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

static void check_launch(const char *msg) {
    // Per CUDA Runtime API docs, cudaGetLastError reports launch/configuration errors
    // for prior runtime calls and resets the thread-local runtime error state.
    check(cudaGetLastError(), msg);
}

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " [--n 4096|16384] [--iters N] [--warmup N] [--csv-header]\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv-header") == 0) {
            args.csv_header = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.n != 4096 && args.n != 16384) {
        usage(argv[0]);
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

static uint32_t mont_nprime() {
    uint32_t inv = 1;
    for (int i = 0; i < 5; i++) {
        inv *= 2u - kModulus * inv;
    }
    return static_cast<uint32_t>(0u - inv);
}

static uint32_t to_mont_host(uint32_t x, uint32_t nprime, uint32_t r2_mod_q) {
    uint64_t t = static_cast<uint64_t>(x) * r2_mod_q;
    uint32_t m = static_cast<uint32_t>(t) * nprime;
    uint64_t u = (t + static_cast<uint64_t>(m) * kModulus) >> 32;
    if (u >= kModulus) {
        u -= kModulus;
    }
    return static_cast<uint32_t>(u);
}

static uint32_t bit_reverse(uint32_t x, int log_degree) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - log_degree);
}

static uint32_t compute_phi_for_n(int n) {
    uint32_t phi = kPrimitiveRootMax;
    int ratio = kMaxPolyDegree / n;
    while (ratio > 1) {
        phi = mod_mul_host(phi, phi);
        ratio >>= 1;
    }
    return phi;
}

static void compute_tables(
    std::vector<uint32_t> &phi_mont,
    std::vector<uint32_t> &post_scale_mont,
    std::vector<uint32_t> &fwd_twiddles_mont,
    std::vector<uint32_t> &inv_twiddles_mont,
    std::vector<uint32_t> &stage_offsets,
    int n,
    uint32_t &nprime,
    uint32_t &r2_mod_q
) {
    nprime = mont_nprime();
    uint32_t r_mod_q = static_cast<uint32_t>((1ULL << 32) % kModulus);
    r2_mod_q = static_cast<uint32_t>((static_cast<uint64_t>(r_mod_q) * r_mod_q) % kModulus);

    uint32_t phi = compute_phi_for_n(n);
    uint32_t invphi = mod_inv(phi);
    uint32_t omega = mod_mul_host(phi, phi);
    uint32_t invomega = mod_inv(omega);
    uint32_t inv_n = mod_inv(static_cast<uint32_t>(n));

    phi_mont.resize(n);
    post_scale_mont.resize(n);

    uint32_t cur = 1;
    for (int i = 0; i < n; i++) {
        phi_mont[i] = to_mont_host(cur, nprime, r2_mod_q);
        cur = mod_mul_host(cur, phi);
    }

    cur = inv_n;
    for (int i = 0; i < n; i++) {
        post_scale_mont[i] = to_mont_host(cur, nprime, r2_mod_q);
        cur = mod_mul_host(cur, invphi);
    }

    stage_offsets.clear();
    stage_offsets.push_back(0);
    for (int len = 2; len <= n; len <<= 1) {
        stage_offsets.push_back(stage_offsets.back() + len / 2);
    }
    fwd_twiddles_mont.resize(stage_offsets.back());
    inv_twiddles_mont.resize(stage_offsets.back());

    int stage = 0;
    for (int len = 2; len <= n; len <<= 1, stage++) {
        int half = len / 2;
        uint32_t wlen = mod_pow(omega, static_cast<uint64_t>(n / len));
        uint32_t iwlen = mod_pow(invomega, static_cast<uint64_t>(n / len));
        uint32_t w = 1;
        uint32_t iw = 1;
        int base = stage_offsets[stage];
        for (int j = 0; j < half; j++) {
            fwd_twiddles_mont[base + j] = to_mont_host(w, nprime, r2_mod_q);
            inv_twiddles_mont[base + j] = to_mont_host(iw, nprime, r2_mod_q);
            w = mod_mul_host(w, wlen);
            iw = mod_mul_host(iw, iwlen);
        }
    }
}

static void host_forward_ntt(std::vector<uint32_t> &a, const std::vector<uint32_t> &phi, int n, int log_degree) {
    std::vector<uint32_t> tmp(n);
    for (int i = 0; i < n; i++) {
        tmp[bit_reverse(i, log_degree)] = mod_mul_host(a[i], phi[i]);
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

static void host_inverse_ntt(std::vector<uint32_t> &a, const std::vector<uint32_t> &post_scale, int n, int log_degree) {
    std::vector<uint32_t> tmp(n);
    for (int i = 0; i < n; i++) {
        tmp[bit_reverse(i, log_degree)] = a[i];
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

static bool compare_vectors(const std::vector<uint32_t> &expected,
                            const std::vector<uint32_t> &actual,
                            const char *label) {
    if (expected.size() != actual.size()) {
        std::cerr << label << " size mismatch: expected " << expected.size()
                  << " got " << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (expected[i] != actual[i]) {
            std::cerr << label << " mismatch at index " << i
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

static bool validate_roundtrip_case(const char *label,
                                    const std::vector<uint32_t> &input,
                                    uint32_t *d_input,
                                    uint32_t *d_work,
                                    uint32_t *d_tmp,
                                    uint32_t *d_out,
                                    const DeviceTables &tables,
                                    const std::vector<uint32_t> &stage_offsets,
                                    int n,
                                    int log_degree,
                                    uint32_t nprime,
                                    uint32_t r2_mod_q) {
    check(cudaMemcpy(d_input, input.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice),
          "copy roundtrip input");
    run_forward_only(d_input, d_work, tables, stage_offsets, n, log_degree, nprime, r2_mod_q);
    run_inverse_only(d_work, d_tmp, d_out, tables, stage_offsets, n, log_degree, nprime);
    check(cudaDeviceSynchronize(), "sync roundtrip validation");
    std::vector<uint32_t> output(n);
    check(cudaMemcpy(output.data(), d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
          "copy roundtrip output");
    return compare_vectors(input, output, label);
}

static bool validate_polymul_case(const char *label,
                                  const std::vector<uint32_t> &lhs,
                                  const std::vector<uint32_t> &rhs,
                                  uint32_t *d_a,
                                  uint32_t *d_b,
                                  uint32_t *d_a_work,
                                  uint32_t *d_b_work,
                                  uint32_t *d_c_work,
                                  uint32_t *d_tmp,
                                  uint32_t *d_out,
                                  const DeviceTables &tables,
                                  const std::vector<uint32_t> &stage_offsets,
                                  const std::vector<uint32_t> &phi_norm,
                                  const std::vector<uint32_t> &post_norm,
                                  int n,
                                  int log_degree,
                                  uint32_t nprime,
                                  uint32_t r2_mod_q) {
    const std::vector<uint32_t> expected =
        host_polymul_reference(lhs, rhs, phi_norm, post_norm, n, log_degree);

    check(cudaMemcpy(d_a, lhs.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice),
          "copy polymul lhs");
    check(cudaMemcpy(d_b, rhs.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice),
          "copy polymul rhs");
    run_full_polymul(d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                     tables, stage_offsets, n, log_degree, nprime, r2_mod_q);
    check(cudaDeviceSynchronize(), "sync polymul validation");

    std::vector<uint32_t> actual(n);
    check(cudaMemcpy(actual.data(), d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
          "copy polymul output");
    return compare_vectors(expected, actual, label);
}

static bool run_validation_suite(uint32_t *d_a,
                                 uint32_t *d_b,
                                 uint32_t *d_a_work,
                                 uint32_t *d_b_work,
                                 uint32_t *d_c_work,
                                 uint32_t *d_tmp,
                                 uint32_t *d_out,
                                 const DeviceTables &tables,
                                 const std::vector<uint32_t> &stage_offsets,
                                 const std::vector<uint32_t> &phi_norm,
                                 const std::vector<uint32_t> &post_norm,
                                 int n,
                                 int log_degree,
                                 uint32_t nprime,
                                 uint32_t r2_mod_q) {
    bool ok = true;

    ok = validate_roundtrip_case("roundtrip zeros",
                                 make_input_pattern(n, InputPattern::Zero),
                                 d_a, d_a_work, d_tmp, d_out, tables,
                                 stage_offsets, n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_roundtrip_case("roundtrip ones",
                                 make_input_pattern(n, InputPattern::One),
                                 d_a, d_a_work, d_tmp, d_out, tables,
                                 stage_offsets, n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_roundtrip_case("roundtrip impulse",
                                 make_input_pattern(n, InputPattern::Impulse),
                                 d_a, d_a_work, d_tmp, d_out, tables,
                                 stage_offsets, n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_roundtrip_case("roundtrip max",
                                 make_input_pattern(n, InputPattern::Max),
                                 d_a, d_a_work, d_tmp, d_out, tables,
                                 stage_offsets, n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_roundtrip_case("roundtrip random(12345)",
                                 make_input_pattern(n, InputPattern::Random, 12345u),
                                 d_a, d_a_work, d_tmp, d_out, tables,
                                 stage_offsets, n, log_degree, nprime, r2_mod_q) && ok;

    ok = validate_polymul_case("polymul zero*zero",
                               make_input_pattern(n, InputPattern::Zero),
                               make_input_pattern(n, InputPattern::Zero),
                               d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                               tables, stage_offsets, phi_norm, post_norm,
                               n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_polymul_case("polymul one*one",
                               make_input_pattern(n, InputPattern::One),
                               make_input_pattern(n, InputPattern::One),
                               d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                               tables, stage_offsets, phi_norm, post_norm,
                               n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_polymul_case("polymul impulse*ones",
                               make_input_pattern(n, InputPattern::Impulse),
                               make_input_pattern(n, InputPattern::One),
                               d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                               tables, stage_offsets, phi_norm, post_norm,
                               n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_polymul_case("polymul max*max",
                               make_input_pattern(n, InputPattern::Max),
                               make_input_pattern(n, InputPattern::Max),
                               d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                               tables, stage_offsets, phi_norm, post_norm,
                               n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_polymul_case("polymul random(1)*random(2)",
                               make_input_pattern(n, InputPattern::Random, 1u),
                               make_input_pattern(n, InputPattern::Random, 2u),
                               d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                               tables, stage_offsets, phi_norm, post_norm,
                               n, log_degree, nprime, r2_mod_q) && ok;
    ok = validate_polymul_case("polymul random(3)*random(4)",
                               make_input_pattern(n, InputPattern::Random, 3u),
                               make_input_pattern(n, InputPattern::Random, 4u),
                               d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
                               tables, stage_offsets, phi_norm, post_norm,
                               n, log_degree, nprime, r2_mod_q) && ok;

    return ok;
}

__device__ __forceinline__ uint32_t mont_mul(uint32_t a, uint32_t b, uint32_t q, uint32_t nprime) {
    uint64_t t = static_cast<uint64_t>(a) * b;
    uint32_t m = static_cast<uint32_t>(t) * nprime;
    uint64_t u = (t + static_cast<uint64_t>(m) * q) >> 32;
    if (u >= q) {
        u -= q;
    }
    return static_cast<uint32_t>(u);
}

__device__ __forceinline__ uint32_t add_mod(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t sum = a + b;
    if (sum >= q) {
        sum -= q;
    }
    return sum;
}

__device__ __forceinline__ uint32_t sub_mod(uint32_t a, uint32_t b, uint32_t q) {
    return a >= b ? (a - b) : (a + q - b);
}

__device__ __forceinline__ uint32_t device_bit_reverse(uint32_t x, int log_degree) {
    return __brev(x) >> (32 - log_degree);
}

__global__ void preprocess_phi_kernel(const uint32_t *in, uint32_t *out, const uint32_t *phi_mont, int n, int log_degree, uint32_t q, uint32_t nprime, uint32_t r2_mod_q) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<uint32_t>(n)) {
        return;
    }
    uint32_t rev = device_bit_reverse(idx, log_degree);
    uint32_t x_mont = mont_mul(in[idx], r2_mod_q, q, nprime);
    out[rev] = mont_mul(x_mont, phi_mont[idx], q, nprime);
}

__global__ void bit_reverse_copy_kernel(const uint32_t *in, uint32_t *out, int n, int log_degree) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<uint32_t>(n)) {
        return;
    }
    uint32_t rev = device_bit_reverse(idx, log_degree);
    out[rev] = in[idx];
}

__global__ void ntt_stage_kernel(uint32_t *data, const uint32_t *twiddles, uint32_t stage_offset, uint32_t len, int n, uint32_t q, uint32_t nprime) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = len >> 1;
    if (idx >= static_cast<uint32_t>(n / 2)) {
        return;
    }
    uint32_t group = idx / half;
    uint32_t j = idx % half;
    uint32_t base = group * len + j;
    uint32_t w = twiddles[stage_offset + j];
    uint32_t u = data[base];
    uint32_t v = mont_mul(data[base + half], w, q, nprime);
    data[base] = add_mod(u, v, q);
    data[base + half] = sub_mod(u, v, q);
}

__global__ void ntt_first8_stages_kernel(
    uint32_t *data,
    const uint32_t *twiddles,
    const uint32_t *stage_offsets,
    int n,
    int log_degree,
    uint32_t q,
    uint32_t nprime
) {
    extern __shared__ uint32_t s[];
    const uint32_t tid = threadIdx.x;
    const uint32_t segment_base = blockIdx.x * 256;
    if (segment_base >= static_cast<uint32_t>(n)) {
        return;
    }

    s[tid] = data[segment_base + tid];
    s[tid + 128] = data[segment_base + tid + 128];
    __syncthreads();

    const int fused = (log_degree < kFusedStages) ? log_degree : kFusedStages;
    for (int stage = 0; stage < fused; stage++) {
        const uint32_t len = 2u << stage;
        const uint32_t half = len >> 1;
        const uint32_t group = tid / half;
        const uint32_t j = tid % half;
        const uint32_t base = group * len + j;
        const uint32_t w = twiddles[stage_offsets[stage] + j];
        const uint32_t u = s[base];
        const uint32_t v = mont_mul(s[base + half], w, q, nprime);
        s[base] = add_mod(u, v, q);
        s[base + half] = sub_mod(u, v, q);
        __syncthreads();
    }

    data[segment_base + tid] = s[tid];
    data[segment_base + tid + 128] = s[tid + 128];
}

__global__ void pointwise_mul_kernel(const uint32_t *a, const uint32_t *b, uint32_t *out, int n, uint32_t q, uint32_t nprime) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<uint32_t>(n)) {
        return;
    }
    out[idx] = mont_mul(a[idx], b[idx], q, nprime);
}

__global__ void postprocess_inv_kernel(const uint32_t *in, uint32_t *out, const uint32_t *post_scale_mont, int n, uint32_t q, uint32_t nprime) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<uint32_t>(n)) {
        return;
    }
    uint32_t scaled = mont_mul(in[idx], post_scale_mont[idx], q, nprime);
    out[idx] = mont_mul(scaled, 1u, q, nprime);
}

static void run_forward_only(uint32_t *d_input, uint32_t *d_work, const DeviceTables &tables, const std::vector<uint32_t> &stage_offsets, int n, int log_degree, uint32_t nprime, uint32_t r2_mod_q) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    preprocess_phi_kernel<<<grid, block>>>(d_input, d_work, tables.d_phi, n, log_degree, kModulus, nprime, r2_mod_q);
    check_launch("launch preprocess_phi_kernel (forward)");

    dim3 fused_block(128);
    dim3 fused_grid((n + 255) / 256);
    ntt_first8_stages_kernel<<<fused_grid, fused_block, 256 * sizeof(uint32_t)>>>(
        d_work, tables.d_fwd_twiddles, tables.d_stage_offsets, n, log_degree, kModulus, nprime);
    check_launch("launch ntt_first8_stages_kernel (forward)");

    for (int stage = kFusedStages; stage < log_degree; stage++) {
        int len = 2 << stage;
        dim3 stage_grid(((n / 2) + block.x - 1) / block.x);
        ntt_stage_kernel<<<stage_grid, block>>>(d_work, tables.d_fwd_twiddles, stage_offsets[stage], len, n, kModulus, nprime);
        check_launch("launch ntt_stage_kernel (forward tail)");
    }
}

static void run_inverse_only(uint32_t *d_ntt_input, uint32_t *d_tmp, uint32_t *d_out, const DeviceTables &tables, const std::vector<uint32_t> &stage_offsets, int n, int log_degree, uint32_t nprime) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    bit_reverse_copy_kernel<<<grid, block>>>(d_ntt_input, d_tmp, n, log_degree);
    check_launch("launch bit_reverse_copy_kernel (inverse)");

    dim3 fused_block(128);
    dim3 fused_grid((n + 255) / 256);
    ntt_first8_stages_kernel<<<fused_grid, fused_block, 256 * sizeof(uint32_t)>>>(
        d_tmp, tables.d_inv_twiddles, tables.d_stage_offsets, n, log_degree, kModulus, nprime);
    check_launch("launch ntt_first8_stages_kernel (inverse)");

    for (int stage = kFusedStages; stage < log_degree; stage++) {
        int len = 2 << stage;
        dim3 stage_grid(((n / 2) + block.x - 1) / block.x);
        ntt_stage_kernel<<<stage_grid, block>>>(d_tmp, tables.d_inv_twiddles, stage_offsets[stage], len, n, kModulus, nprime);
        check_launch("launch ntt_stage_kernel (inverse tail)");
    }
    postprocess_inv_kernel<<<grid, block>>>(d_tmp, d_out, tables.d_post_scale, n, kModulus, nprime);
    check_launch("launch postprocess_inv_kernel");
}

static void run_full_polymul(uint32_t *d_a, uint32_t *d_b, uint32_t *d_a_work, uint32_t *d_b_work, uint32_t *d_c_work, uint32_t *d_tmp, uint32_t *d_out, const DeviceTables &tables, const std::vector<uint32_t> &stage_offsets, int n, int log_degree, uint32_t nprime, uint32_t r2_mod_q) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    preprocess_phi_kernel<<<grid, block>>>(d_a, d_a_work, tables.d_phi, n, log_degree, kModulus, nprime, r2_mod_q);
    check_launch("launch preprocess_phi_kernel (polymul lhs)");
    preprocess_phi_kernel<<<grid, block>>>(d_b, d_b_work, tables.d_phi, n, log_degree, kModulus, nprime, r2_mod_q);
    check_launch("launch preprocess_phi_kernel (polymul rhs)");

    dim3 fused_block(128);
    dim3 fused_grid((n + 255) / 256);
    ntt_first8_stages_kernel<<<fused_grid, fused_block, 256 * sizeof(uint32_t)>>>(
        d_a_work, tables.d_fwd_twiddles, tables.d_stage_offsets, n, log_degree, kModulus, nprime);
    check_launch("launch ntt_first8_stages_kernel (polymul lhs)");
    ntt_first8_stages_kernel<<<fused_grid, fused_block, 256 * sizeof(uint32_t)>>>(
        d_b_work, tables.d_fwd_twiddles, tables.d_stage_offsets, n, log_degree, kModulus, nprime);
    check_launch("launch ntt_first8_stages_kernel (polymul rhs)");

    for (int stage = kFusedStages; stage < log_degree; stage++) {
        int len = 2 << stage;
        dim3 stage_grid(((n / 2) + block.x - 1) / block.x);
        ntt_stage_kernel<<<stage_grid, block>>>(d_a_work, tables.d_fwd_twiddles, stage_offsets[stage], len, n, kModulus, nprime);
        check_launch("launch ntt_stage_kernel (polymul lhs tail)");
        ntt_stage_kernel<<<stage_grid, block>>>(d_b_work, tables.d_fwd_twiddles, stage_offsets[stage], len, n, kModulus, nprime);
        check_launch("launch ntt_stage_kernel (polymul rhs tail)");
    }
    pointwise_mul_kernel<<<grid, block>>>(d_a_work, d_b_work, d_c_work, n, kModulus, nprime);
    check_launch("launch pointwise_mul_kernel");
    bit_reverse_copy_kernel<<<grid, block>>>(d_c_work, d_tmp, n, log_degree);
    check_launch("launch bit_reverse_copy_kernel (polymul inverse prep)");

    ntt_first8_stages_kernel<<<fused_grid, fused_block, 256 * sizeof(uint32_t)>>>(
        d_tmp, tables.d_inv_twiddles, tables.d_stage_offsets, n, log_degree, kModulus, nprime);
    check_launch("launch ntt_first8_stages_kernel (polymul inverse)");

    for (int stage = kFusedStages; stage < log_degree; stage++) {
        int len = 2 << stage;
        dim3 stage_grid(((n / 2) + block.x - 1) / block.x);
        ntt_stage_kernel<<<stage_grid, block>>>(d_tmp, tables.d_inv_twiddles, stage_offsets[stage], len, n, kModulus, nprime);
        check_launch("launch ntt_stage_kernel (polymul inverse tail)");
    }
    postprocess_inv_kernel<<<grid, block>>>(d_tmp, d_out, tables.d_post_scale, n, kModulus, nprime);
    check_launch("launch postprocess_inv_kernel (polymul)");
}

static void alloc_and_copy(DeviceTables &tables,
                           const std::vector<uint32_t> &phi_mont,
                           const std::vector<uint32_t> &post_scale_mont,
                           const std::vector<uint32_t> &fwd_twiddles_mont,
                           const std::vector<uint32_t> &inv_twiddles_mont,
                           const std::vector<uint32_t> &stage_offsets) {
    check(cudaMalloc(&tables.d_phi, sizeof(uint32_t) * phi_mont.size()), "malloc phi");
    check(cudaMalloc(&tables.d_post_scale, sizeof(uint32_t) * post_scale_mont.size()), "malloc post scale");
    check(cudaMalloc(&tables.d_fwd_twiddles, sizeof(uint32_t) * fwd_twiddles_mont.size()), "malloc fwd twiddles");
    check(cudaMalloc(&tables.d_inv_twiddles, sizeof(uint32_t) * inv_twiddles_mont.size()), "malloc inv twiddles");
    check(cudaMalloc(&tables.d_stage_offsets, sizeof(uint32_t) * stage_offsets.size()), "malloc stage offsets");
    check(cudaMemcpy(tables.d_phi, phi_mont.data(), sizeof(uint32_t) * phi_mont.size(), cudaMemcpyHostToDevice), "copy phi");
    check(cudaMemcpy(tables.d_post_scale, post_scale_mont.data(), sizeof(uint32_t) * post_scale_mont.size(), cudaMemcpyHostToDevice), "copy post scale");
    check(cudaMemcpy(tables.d_fwd_twiddles, fwd_twiddles_mont.data(), sizeof(uint32_t) * fwd_twiddles_mont.size(), cudaMemcpyHostToDevice), "copy fwd twiddles");
    check(cudaMemcpy(tables.d_inv_twiddles, inv_twiddles_mont.data(), sizeof(uint32_t) * inv_twiddles_mont.size(), cudaMemcpyHostToDevice), "copy inv twiddles");
    check(cudaMemcpy(tables.d_stage_offsets, stage_offsets.data(), sizeof(uint32_t) * stage_offsets.size(), cudaMemcpyHostToDevice), "copy stage offsets");
}

static void free_tables(DeviceTables &tables) {
    cudaFree(tables.d_phi);
    cudaFree(tables.d_fwd_twiddles);
    cudaFree(tables.d_inv_twiddles);
    cudaFree(tables.d_stage_offsets);
    cudaFree(tables.d_post_scale);
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,n,qbits,iters,ntt_mean_us,ntt_std_us,intt_mean_us,intt_std_us,poly_mul_mean_us,poly_mul_std_us,correct\n";
    }

    const int n = args.n;
    const int log_degree = int_log2(static_cast<uint32_t>(n));

    uint32_t nprime = 0;
    uint32_t r2_mod_q = 0;
    std::vector<uint32_t> phi_mont;
    std::vector<uint32_t> post_scale_mont;
    std::vector<uint32_t> fwd_twiddles_mont;
    std::vector<uint32_t> inv_twiddles_mont;
    std::vector<uint32_t> stage_offsets;
    compute_tables(phi_mont, post_scale_mont, fwd_twiddles_mont, inv_twiddles_mont, stage_offsets, n, nprime, r2_mod_q);

    DeviceTables tables;
    alloc_and_copy(tables, phi_mont, post_scale_mont, fwd_twiddles_mont, inv_twiddles_mont, stage_offsets);
    const std::vector<uint32_t> &stage_offsets_host = stage_offsets;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, kModulus - 1);
    std::vector<uint32_t> a(n);
    std::vector<uint32_t> b(n);
    for (int i = 0; i < n; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    uint32_t *d_a = nullptr;
    uint32_t *d_b = nullptr;
    uint32_t *d_a_work = nullptr;
    uint32_t *d_b_work = nullptr;
    uint32_t *d_c_work = nullptr;
    uint32_t *d_tmp = nullptr;
    uint32_t *d_out = nullptr;
    check(cudaMalloc(&d_a, sizeof(uint32_t) * n), "malloc d_a");
    check(cudaMalloc(&d_b, sizeof(uint32_t) * n), "malloc d_b");
    check(cudaMalloc(&d_a_work, sizeof(uint32_t) * n), "malloc d_a_work");
    check(cudaMalloc(&d_b_work, sizeof(uint32_t) * n), "malloc d_b_work");
    check(cudaMalloc(&d_c_work, sizeof(uint32_t) * n), "malloc d_c_work");
    check(cudaMalloc(&d_tmp, sizeof(uint32_t) * n), "malloc d_tmp");
    check(cudaMalloc(&d_out, sizeof(uint32_t) * n), "malloc d_out");
    check(cudaMemcpy(d_a, a.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(d_b, b.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice), "copy b");

    // Build host reference tables once, then run a validation suite before timing.
    std::vector<uint32_t> phi_norm(n);
    std::vector<uint32_t> post_norm(n);
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

    bool correct = run_validation_suite(
        d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out,
        tables, stage_offsets_host, phi_norm, post_norm,
        n, log_degree, nprime, r2_mod_q);

    check(cudaMemcpy(d_a, a.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice), "restore benchmark a");
    check(cudaMemcpy(d_b, b.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice), "restore benchmark b");

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

    // prepare an NTT-domain sample for inverse timing
    run_forward_only(d_a, d_a_work, tables, stage_offsets_host, n, log_degree, nprime, r2_mod_q);
    check(cudaDeviceSynchronize(), "sync prep forward");

    for (int i = 0; i < args.warmup; i++) {
        run_forward_only(d_a, d_a_work, tables, stage_offsets_host, n, log_degree, nprime, r2_mod_q);
        run_inverse_only(d_a_work, d_tmp, d_out, tables, stage_offsets_host, n, log_degree, nprime);
        run_full_polymul(d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out, tables, stage_offsets_host, n, log_degree, nprime, r2_mod_q);
        check(cudaDeviceSynchronize(), "sync warmup");
    }

    for (int i = 0; i < args.iters; i++) {
        check(cudaEventRecord(start_evt), "record start ntt");
        run_forward_only(d_a, d_a_work, tables, stage_offsets_host, n, log_degree, nprime, r2_mod_q);
        check(cudaEventRecord(stop_evt), "record stop ntt");
        check(cudaEventSynchronize(stop_evt), "sync stop ntt");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_evt, stop_evt), "elapsed ntt");
        ntt_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(start_evt), "record start intt");
        run_inverse_only(d_a_work, d_tmp, d_out, tables, stage_offsets_host, n, log_degree, nprime);
        check(cudaEventRecord(stop_evt), "record stop intt");
        check(cudaEventSynchronize(stop_evt), "sync stop intt");
        check(cudaEventElapsedTime(&ms, start_evt, stop_evt), "elapsed intt");
        intt_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(start_evt), "record start polymul");
        run_full_polymul(d_a, d_b, d_a_work, d_b_work, d_c_work, d_tmp, d_out, tables, stage_offsets_host, n, log_degree, nprime, r2_mod_q);
        check(cudaEventRecord(stop_evt), "record stop polymul");
        check(cudaEventSynchronize(stop_evt), "sync stop polymul");
        check(cudaEventElapsedTime(&ms, start_evt, stop_evt), "elapsed polymul");
        poly_samples.push_back(ms * 1000.0);
    }

    Stats ntt = compute_stats(ntt_samples);
    Stats intt = compute_stats(intt_samples);
    Stats poly = compute_stats(poly_samples);

    std::cout << "cuda," << n << ",30," << args.iters << ","
              << ntt.mean_us << "," << ntt.stddev_us << ","
              << intt.mean_us << "," << intt.stddev_us << ","
              << poly.mean_us << "," << poly.stddev_us << ","
              << (correct ? 1 : 0) << "\n";

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    free_tables(tables);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_work);
    cudaFree(d_b_work);
    cudaFree(d_c_work);
    cudaFree(d_tmp);
    cudaFree(d_out);
    return correct ? 0 : 2;
}
