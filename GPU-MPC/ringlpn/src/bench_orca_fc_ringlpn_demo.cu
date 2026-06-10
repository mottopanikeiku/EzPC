#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "fss/gpu_matmul.h"
#include "ringlpn/src/orca_fc_ringlpn_keywriter.cuh"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

namespace {

using u128 = unsigned __int128;
using T = u64;

constexpr uint64_t kPrime62 = 4611686018326724609ULL;

struct Args {
    int rows = 2;
    int inner = 2;
    int cols = 2;
    int bw = 16;
    int qbits = 64;
    uint64_t value_bound = 255;
    int poly_n = 8192;
    int c = 2;
    int t = 8;
    uint64_t seed = 1;
    uint64_t second_seed = 2;
    bool compare_baseline = true;
    bool csv_header = false;
};

struct DealerOutput {
    std::vector<T> input;
    std::vector<T> weight;
    std::vector<T> mask_a;
    std::vector<T> mask_b;
    std::vector<T> masked_input;
    std::vector<T> masked_weight;
    std::vector<T> output_mask;
    std::vector<T> expected_masked_output;
    std::vector<uint8_t> key0;
    std::vector<uint8_t> key1;
    bool conversion_ok = true;
};

struct DemoResult {
    DealerOutput dealer;
    std::vector<T> reconstructed;
    bool online_ok = false;
    bool key_order_ok = false;
};

struct BaselineResult {
    std::vector<uint8_t> key0;
    std::vector<uint8_t> key1;
    std::vector<T> reconstructed;
    bool keygen_ok = false;
    bool online_ok = false;
    bool matches_ringlpn = false;
};

struct ContractResult {
    bool online_ok = false;
    bool key_order_ok = false;
};

static uint64_t ring_mask(int bw) {
    return bw == 64 ? UINT64_MAX : ((uint64_t(1) << bw) - 1);
}

static uint64_t ring_reduce(uint64_t x, int bw) {
    return x & ring_mask(bw);
}

static uint64_t ring_reduce128(u128 x, int bw) {
    if (bw == 64) {
        return static_cast<uint64_t>(x);
    }
    return static_cast<uint64_t>(x & ((u128(1) << bw) - 1));
}

static uint64_t ring_add(uint64_t a, uint64_t b, int bw) {
    return ring_reduce128(u128(a) + b, bw);
}

static bool no_prime_wrap_bound(const Args &args) {
    return u128(args.inner) * args.value_bound * args.value_bound <
           ringlpn_orca::modulusForQbits(args.qbits);
}

static uint64_t uniform_mod(uint64_t p, std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, p - 1);
    return dist(rng);
}

static MatmulParams make_matmul_params(const Args &args) {
    MatmulParams p;
    p.batchSz = 1;
    p.M = args.rows;
    p.K = args.inner;
    p.N = args.cols;
    stdInit(p, args.bw, 0);
    return p;
}

static void copy_to_gpu(const std::vector<T> &src, T **dst);
static bool write_demo_keys_with_helper(const Args &args, DealerOutput &out);

static uint64_t clear_matmul_entry(const std::vector<T> &a,
                                   const std::vector<T> &b,
                                   int rows,
                                   int inner,
                                   int cols,
                                   int row,
                                   int col,
                                   int bw) {
    u128 acc = 0;
    for (int k = 0; k < inner; ++k) {
        acc += u128(a[static_cast<size_t>(row) * inner + k]) *
               b[static_cast<size_t>(k) * cols + col];
    }
    (void)rows;
    return ring_reduce128(acc, bw);
}

static bool validate_args(const Args &args) {
    if (args.rows <= 0 || args.inner <= 0 || args.cols <= 0 ||
        args.rows > 8 || args.inner > 8 || args.cols > 8 ||
        args.bw <= 0 || args.bw > 32 || args.value_bound >= kPrime62 ||
        (args.qbits != 64 && args.qbits != 128) ||
        args.poly_n != 8192 || args.c != 2 || args.t != 8 ||
        args.seed == args.second_seed) {
        return false;
    }
    if (args.bw < 64 && args.value_bound >= (uint64_t(1) << args.bw)) {
        return false;
    }
    return no_prime_wrap_bound(args);
}

static DealerOutput generate_dealer_output(const Args &args, uint64_t seed) {
    DealerOutput out;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> bounded(0, args.value_bound);

    const size_t size_a = static_cast<size_t>(args.rows) * args.inner;
    const size_t size_b = static_cast<size_t>(args.inner) * args.cols;
    const size_t size_c = static_cast<size_t>(args.rows) * args.cols;
    out.input.resize(size_a);
    out.weight.resize(size_b);
    out.mask_a.resize(size_a);
    out.mask_b.resize(size_b);
    out.masked_input.resize(size_a);
    out.masked_weight.resize(size_b);
    out.output_mask.resize(size_c);
    out.expected_masked_output.resize(size_c);

    for (size_t i = 0; i < size_a; ++i) {
        out.input[i] = static_cast<T>(bounded(rng));
        out.mask_a[i] = static_cast<T>(bounded(rng));
        out.masked_input[i] = ring_add(out.input[i], out.mask_a[i], args.bw);
    }
    for (size_t i = 0; i < size_b; ++i) {
        out.weight[i] = static_cast<T>(bounded(rng));
        out.mask_b[i] = static_cast<T>(bounded(rng));
        out.masked_weight[i] = ring_add(out.weight[i], out.mask_b[i], args.bw);
    }

    for (int r = 0; r < args.rows; ++r) {
        for (int col = 0; col < args.cols; ++col) {
            const size_t idx = static_cast<size_t>(r) * args.cols + col;
            out.output_mask[idx] = static_cast<T>(uniform_mod(uint64_t(1) << args.bw, rng));
            uint64_t clear = clear_matmul_entry(
                out.input, out.weight, args.rows, args.inner, args.cols, r, col, args.bw);
            out.expected_masked_output[idx] = ring_add(clear, out.output_mask[idx], args.bw);
        }
    }

    out.conversion_ok = write_demo_keys_with_helper(args, out);
    return out;
}

static std::vector<T> key_span_to_vector(T *ptr, int count) {
    return std::vector<T>(ptr, ptr + count);
}

static void copy_to_gpu(const std::vector<T> &src, T **dst) {
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(dst), src.size() * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
    err = cudaMemcpy(*dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

static bool write_demo_keys_with_helper(const Args &args, DealerOutput &out) {
    MatmulParams p = make_matmul_params(args);
    const size_t key_bytes =
        (static_cast<size_t>(p.size_A) + p.size_B + p.size_C) * sizeof(T);
    out.key0.assign(key_bytes, 0);
    out.key1.assign(key_bytes, 0);

    T *d_mask_a = nullptr;
    T *d_mask_b = nullptr;
    T *d_output_mask = nullptr;
    copy_to_gpu(out.mask_a, &d_mask_a);
    copy_to_gpu(out.mask_b, &d_mask_b);
    copy_to_gpu(out.output_mask, &d_output_mask);

    uint8_t *ptr0 = out.key0.data();
    uint8_t *ptr1 = out.key1.data();
    bool ok0 = ringlpn_orca::writeMatmulKey<T>(
        &ptr0, SERVER0, p, d_mask_a, d_mask_b, d_output_mask, args.qbits, args.seed);
    bool ok1 = ringlpn_orca::writeMatmulKey<T>(
        &ptr1, SERVER1, p, d_mask_a, d_mask_b, d_output_mask, args.qbits, args.seed);

    cudaFree(d_mask_a);
    cudaFree(d_mask_b);
    cudaFree(d_output_mask);
    return ok0 && ok1 && ptr0 == out.key0.data() + out.key0.size() &&
           ptr1 == out.key1.data() + out.key1.size();
}

static std::vector<T> copy_from_gpu(T *src, size_t count) {
    std::vector<T> out(count);
    cudaError_t err = cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
    return out;
}

static DemoResult run_online_case(const Args &args,
                                  const DealerOutput &dealer,
                                  const std::vector<uint8_t> &key_buf0,
                                  const std::vector<uint8_t> &key_buf1) {
    DemoResult result;
    result.dealer = dealer;
    MatmulParams p = make_matmul_params(args);

    uint8_t *key_ptr0 = const_cast<uint8_t *>(key_buf0.data());
    uint8_t *key_ptr1 = const_cast<uint8_t *>(key_buf1.data());
    GPUMatmulKey<T> key0 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr0);
    GPUMatmulKey<T> key1 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr1);
    result.key_order_ok =
        key_ptr0 == key_buf0.data() + key_buf0.size() &&
        key_ptr1 == key_buf1.data() + key_buf1.size();

    T *d_x = nullptr;
    T *d_w = nullptr;
    T *d_a0 = nullptr;
    T *d_a1 = nullptr;
    T *d_b0 = nullptr;
    T *d_b1 = nullptr;
    copy_to_gpu(result.dealer.masked_input, &d_x);
    copy_to_gpu(result.dealer.masked_weight, &d_w);
    copy_to_gpu(key_span_to_vector(key0.A, p.size_A), &d_a0);
    copy_to_gpu(key_span_to_vector(key1.A, p.size_A), &d_a1);
    copy_to_gpu(key_span_to_vector(key0.B, p.size_B), &d_b0);
    copy_to_gpu(key_span_to_vector(key1.B, p.size_B), &d_b1);

    Stats stats0;
    Stats stats1;
    T *d_o0 = gpuMatmulBeaver<T>(p, key0, SERVER0, d_x, d_w, d_a0, d_b0, nullptr, &stats0);
    T *d_o1 = gpuMatmulBeaver<T>(p, key1, SERVER1, d_x, d_w, d_a1, d_b1, nullptr, &stats1);
    std::vector<T> o0 = copy_from_gpu(d_o0, p.size_C);
    std::vector<T> o1 = copy_from_gpu(d_o1, p.size_C);
    result.reconstructed.resize(p.size_C);
    for (int i = 0; i < p.size_C; ++i) {
        result.reconstructed[i] = ring_add(o0[i], o1[i], args.bw);
    }

    result.online_ok = result.key_order_ok && result.dealer.conversion_ok &&
                       result.reconstructed == result.dealer.expected_masked_output;

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_b0);
    cudaFree(d_b1);
    gpuFree(d_o0);
    gpuFree(d_o1);
    return result;
}

static uint64_t contract_matmul_entry(const MatmulParams &p,
                                      const std::vector<T> &a,
                                      const std::vector<T> &b,
                                      int batch,
                                      int row,
                                      int col) {
    size_t a_base = static_cast<size_t>(batch) * p.stride_A;
    size_t b_base = static_cast<size_t>(batch) * p.stride_B;
    u128 acc = 0;
    for (int k = 0; k < p.K; ++k) {
        uint64_t av = ringlpn_orca::matrixValue(a, a_base, p.M, p.K,
                                                p.rowMaj_A, row, k, p.bw);
        uint64_t bv = ringlpn_orca::matrixValue(b, b_base, p.K, p.N,
                                                p.rowMaj_B, k, col, p.bw);
        acc += u128(av) * bv;
    }
    return ring_reduce128(acc, p.bw);
}

static std::vector<T> add_ring_vectors(const std::vector<T> &a,
                                       const std::vector<T> &b,
                                       int bw) {
    std::vector<T> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<T>(ring_add(a[i], b[i], bw));
    }
    return out;
}

static bool write_piecewise_matmul_keys(const MatmulParams &p,
                                        int qbits,
                                        uint64_t tag,
                                        const std::vector<T> &mask_a,
                                        const std::vector<T> &mask_b,
                                        const std::vector<T> &mask_c,
                                        std::vector<uint8_t> &key0,
                                        std::vector<uint8_t> &key1) {
    const size_t key_bytes =
        (static_cast<size_t>(p.size_A) + p.size_B + p.size_C) * sizeof(T);
    key0.assign(key_bytes, 0);
    key1.assign(key_bytes, 0);

    T *d_mask_a = nullptr;
    T *d_mask_b = nullptr;
    T *d_mask_c = nullptr;
    copy_to_gpu(mask_a, &d_mask_a);
    copy_to_gpu(mask_b, &d_mask_b);
    copy_to_gpu(mask_c, &d_mask_c);

    uint8_t *ptr0 = key0.data();
    uint8_t *ptr1 = key1.data();
    bool ok0 = ringlpn_orca::writeValueShares<T>(
                   &ptr0, SERVER0, p.size_A, d_mask_a, p.bw, qbits, tag + 1) &&
               ringlpn_orca::writeValueShares<T>(
                   &ptr0, SERVER0, p.size_B, d_mask_b, p.bw, qbits, tag + 2) &&
               ringlpn_orca::writeMatmulCShare<T>(
                   &ptr0, SERVER0, p, d_mask_a, d_mask_b, d_mask_c, qbits, tag + 3);
    bool ok1 = ringlpn_orca::writeValueShares<T>(
                   &ptr1, SERVER1, p.size_A, d_mask_a, p.bw, qbits, tag + 1) &&
               ringlpn_orca::writeValueShares<T>(
                   &ptr1, SERVER1, p.size_B, d_mask_b, p.bw, qbits, tag + 2) &&
               ringlpn_orca::writeMatmulCShare<T>(
                   &ptr1, SERVER1, p, d_mask_a, d_mask_b, d_mask_c, qbits, tag + 3);

    cudaFree(d_mask_a);
    cudaFree(d_mask_b);
    cudaFree(d_mask_c);
    return ok0 && ok1 && ptr0 == key0.data() + key0.size() &&
           ptr1 == key1.data() + key1.size();
}

static ContractResult run_piecewise_matmul_contract(const MatmulParams &p,
                                                    int qbits,
                                                    uint64_t tag,
                                                    const std::vector<T> &clear_a,
                                                    const std::vector<T> &clear_b,
                                                    const std::vector<T> &mask_a,
                                                    const std::vector<T> &mask_b,
                                                    const std::vector<T> &mask_c) {
    ContractResult result;
    std::vector<uint8_t> key_buf0;
    std::vector<uint8_t> key_buf1;
    if (!write_piecewise_matmul_keys(
            p, qbits, tag, mask_a, mask_b, mask_c, key_buf0, key_buf1)) {
        return result;
    }

    uint8_t *key_ptr0 = key_buf0.data();
    uint8_t *key_ptr1 = key_buf1.data();
    GPUMatmulKey<T> key0 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr0);
    GPUMatmulKey<T> key1 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr1);
    result.key_order_ok =
        key_ptr0 == key_buf0.data() + key_buf0.size() &&
        key_ptr1 == key_buf1.data() + key_buf1.size();

    std::vector<T> masked_a = add_ring_vectors(clear_a, mask_a, p.bw);
    std::vector<T> masked_b = add_ring_vectors(clear_b, mask_b, p.bw);
    std::vector<T> expected(p.size_C);
    for (int batch = 0; batch < p.batchSz; ++batch) {
        size_t c_base = static_cast<size_t>(batch) * p.stride_C;
        for (int row = 0; row < p.M; ++row) {
            for (int col = 0; col < p.N; ++col) {
                size_t c_idx = c_base + static_cast<size_t>(row) * p.N + col;
                expected[c_idx] = static_cast<T>(
                    ring_add(contract_matmul_entry(p, clear_a, clear_b, batch, row, col),
                             mask_c[c_idx], p.bw));
            }
        }
    }

    T *d_a = nullptr;
    T *d_b = nullptr;
    T *d_a0 = nullptr;
    T *d_a1 = nullptr;
    T *d_b0 = nullptr;
    T *d_b1 = nullptr;
    copy_to_gpu(masked_a, &d_a);
    copy_to_gpu(masked_b, &d_b);
    copy_to_gpu(key_span_to_vector(key0.A, p.size_A), &d_a0);
    copy_to_gpu(key_span_to_vector(key1.A, p.size_A), &d_a1);
    copy_to_gpu(key_span_to_vector(key0.B, p.size_B), &d_b0);
    copy_to_gpu(key_span_to_vector(key1.B, p.size_B), &d_b1);

    Stats stats0;
    Stats stats1;
    T *d_o0 = gpuMatmulBeaver<T>(p, key0, SERVER0, d_a, d_b, d_a0, d_b0, nullptr, &stats0);
    T *d_o1 = gpuMatmulBeaver<T>(p, key1, SERVER1, d_a, d_b, d_a1, d_b1, nullptr, &stats1);
    std::vector<T> o0 = copy_from_gpu(d_o0, p.size_C);
    std::vector<T> o1 = copy_from_gpu(d_o1, p.size_C);
    std::vector<T> reconstructed(p.size_C);
    for (int i = 0; i < p.size_C; ++i) {
        reconstructed[i] = static_cast<T>(ring_add(o0[i], o1[i], p.bw));
    }
    result.online_ok = result.key_order_ok && reconstructed == expected;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_b0);
    cudaFree(d_b1);
    gpuFree(d_o0);
    gpuFree(d_o1);
    return result;
}

static MatmulParams make_dW_params(const Args &args) {
    MatmulParams pdW;
    pdW.batchSz = 1;
    pdW.M = args.inner;
    pdW.K = args.rows;
    pdW.N = args.cols;
    stdInit(pdW, args.bw, 0);
    pdW.rowMaj_A = false;
    return pdW;
}

static MatmulParams make_dX_params(const Args &args) {
    MatmulParams pdX;
    pdX.batchSz = 1;
    pdX.M = args.rows;
    pdX.K = args.cols;
    pdX.N = args.inner;
    stdInit(pdX, args.bw, 0);
    pdX.rowMaj_B = false;
    return pdX;
}

static std::vector<T> random_bounded_vector(size_t count,
                                            uint64_t bound,
                                            std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, bound);
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<T>(dist(rng));
    }
    return out;
}

static std::vector<T> random_ring_vector(size_t count, int bw, std::mt19937_64 &rng) {
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<T>(uniform_mod(uint64_t(1) << bw, rng));
    }
    return out;
}

static std::pair<ContractResult, ContractResult> run_backward_contracts(
    const Args &args,
    const DealerOutput &dealer) {
    std::mt19937_64 rng(args.seed ^ 0xBADC0FFEEULL);
    std::vector<T> grad =
        random_bounded_vector(static_cast<size_t>(args.rows) * args.cols,
                              args.value_bound, rng);
    std::vector<T> mask_grad =
        random_bounded_vector(static_cast<size_t>(args.rows) * args.cols,
                              args.value_bound, rng);
    std::vector<T> mask_dW =
        random_ring_vector(static_cast<size_t>(args.inner) * args.cols, args.bw, rng);
    std::vector<T> mask_dX =
        random_ring_vector(static_cast<size_t>(args.rows) * args.inner, args.bw, rng);

    ContractResult dW = run_piecewise_matmul_contract(
        make_dW_params(args), args.qbits, args.seed ^ 0xD00D00ULL,
        dealer.input, grad, dealer.mask_a, mask_grad, mask_dW);
    ContractResult dX = run_piecewise_matmul_contract(
        make_dX_params(args), args.qbits, args.seed ^ 0xD00D10ULL,
        grad, dealer.weight, mask_grad, dealer.mask_b, mask_dX);
    return {dW, dX};
}

static DemoResult run_demo_case(const Args &args, uint64_t seed) {
    DealerOutput dealer = generate_dealer_output(args, seed);
    return run_online_case(args, dealer, dealer.key0, dealer.key1);
}

static BaselineResult run_baseline_case(const Args &args,
                                        const DemoResult &ringlpn_result) {
    BaselineResult baseline;
    const DealerOutput &dealer = ringlpn_result.dealer;
    MatmulParams p = make_matmul_params(args);
    const size_t key_bytes =
        (static_cast<size_t>(p.size_A) + p.size_B + p.size_C) * sizeof(T);
    baseline.key0.assign(key_bytes, 0);
    baseline.key1.assign(key_bytes, 0);

    T *d_mask_a = nullptr;
    T *d_mask_b = nullptr;
    T *d_output_mask = nullptr;
    copy_to_gpu(dealer.mask_a, &d_mask_a);
    copy_to_gpu(dealer.mask_b, &d_mask_b);
    copy_to_gpu(dealer.output_mask, &d_output_mask);

    uint8_t *ptr0 = baseline.key0.data();
    initGPURandomness();
    T *d_return0 = gpuKeygenMatmul<T>(
        &ptr0, SERVER0, p, d_mask_a, d_mask_b, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
    destroyGPURandomness();

    uint8_t *ptr1 = baseline.key1.data();
    initGPURandomness();
    T *d_return1 = gpuKeygenMatmul<T>(
        &ptr1, SERVER1, p, d_mask_a, d_mask_b, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
    destroyGPURandomness();

    baseline.keygen_ok =
        ptr0 == baseline.key0.data() + baseline.key0.size() &&
        ptr1 == baseline.key1.data() + baseline.key1.size() &&
        d_return0 == d_output_mask && d_return1 == d_output_mask;

    DemoResult online = run_online_case(args, dealer, baseline.key0, baseline.key1);
    baseline.reconstructed = online.reconstructed;
    baseline.online_ok = baseline.keygen_ok && online.online_ok;
    baseline.matches_ringlpn = baseline.online_ok &&
                               baseline.reconstructed == ringlpn_result.reconstructed;

    cudaFree(d_mask_a);
    cudaFree(d_mask_b);
    cudaFree(d_output_mask);
    return baseline;
}

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--rows M] [--inner K] [--cols N] [--bw N]"
              << " [--qbits 64|128] [--value-bound B] [--poly-n N] [--c N] [--t N]"
              << " [--seed N] [--second-seed N] [--skip-baseline]"
              << " [--csv-header]\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) {
            args.rows = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--inner") && i + 1 < argc) {
            args.inner = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--cols") && i + 1 < argc) {
            args.cols = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--bw") && i + 1 < argc) {
            args.bw = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--qbits") && i + 1 < argc) {
            args.qbits = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--value-bound") && i + 1 < argc) {
            args.value_bound = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--poly-n") && i + 1 < argc) {
            args.poly_n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--c") && i + 1 < argc) {
            args.c = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--t") && i + 1 < argc) {
            args.t = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--second-seed") && i + 1 < argc) {
            args.second_seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--skip-baseline")) {
            args.compare_baseline = false;
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (!validate_args(args)) {
        usage(argv[0]);
        std::exit(1);
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    initGPUMemPool();

    DemoResult first = run_demo_case(args, args.seed);
    DemoResult replay = run_demo_case(args, args.seed);
    DemoResult second = run_demo_case(args, args.second_seed);
    auto backward = run_backward_contracts(args, first.dealer);
    BaselineResult baseline;
    if (args.compare_baseline) {
        baseline = run_baseline_case(args, first);
    }

    const bool replay_ok =
        first.online_ok && replay.online_ok &&
        first.dealer.key0 == replay.dealer.key0 &&
        first.dealer.key1 == replay.dealer.key1 &&
        first.reconstructed == replay.reconstructed;
    const bool second_ok = second.online_ok;
    const bool second_differs =
        first.dealer.key0 != second.dealer.key0 || first.dealer.key1 != second.dealer.key1;
    const bool backward_dW_ok = backward.first.online_ok;
    const bool backward_dX_ok = backward.second.online_ok;
    const bool validation =
        replay_ok && second_ok && second_differs && backward_dW_ok && backward_dX_ok;
    const bool baseline_ok =
        !args.compare_baseline ||
        (baseline.online_ok && baseline.matches_ringlpn &&
         baseline.key0.size() == first.dealer.key0.size() &&
         baseline.key1.size() == first.dealer.key1.size());
    const bool all_ok = validation && baseline_ok;

    if (args.csv_header) {
        std::cout << "device,input_mode,requested_qbits,actual_qbits,seed,second_seed,rows,inner,cols,bw,value_bound,"
                  << "no_prime_wrap_bound,poly_n,c,t,noise,tf,key_bytes_per_party,"
                  << "baseline_key_bytes_per_party,corrected_carry_conversion,"
                  << "deterministic_replay,second_seed_validation,second_seed_distinct,"
                  << "online_contract,backward_dW_contract,backward_dX_contract,"
                  << "baseline_online_contract,baseline_matches_ringlpn,"
                  << "validation\n";
    }
    std::cout << "cuda_orca_fc_ringlpn_demo,bounded_q" << args.qbits
              << "_constant_polynomial,"
              << args.qbits << "," << ringlpn_orca::actualQbitsForQbits(args.qbits) << ","
              << args.seed << "," << args.second_seed << ","
              << args.rows << "," << args.inner << "," << args.cols << ","
              << args.bw << "," << args.value_bound << ","
              << (no_prime_wrap_bound(args) ? 1 : 0) << ","
              << args.poly_n << "," << args.c << "," << args.t << ","
              << "regular,None,"
              << first.dealer.key0.size() << ","
              << (args.compare_baseline ? baseline.key0.size() : 0) << ","
              << (first.dealer.conversion_ok ? 1 : 0) << ","
              << (replay_ok ? 1 : 0) << ","
              << (second_ok ? 1 : 0) << ","
              << (second_differs ? 1 : 0) << ","
              << (first.online_ok ? "pass" : "fail") << ","
              << (backward_dW_ok ? "pass" : "fail") << ","
              << (backward_dX_ok ? "pass" : "fail") << ","
              << (!args.compare_baseline ? "skipped" : (baseline.online_ok ? "pass" : "fail")) << ","
              << (!args.compare_baseline ? -1 : (baseline.matches_ringlpn ? 1 : 0)) << ","
              << (all_ok ? "pass" : "fail") << "\n";

    return all_ok ? 0 : 2;
}
