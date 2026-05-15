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
#include "utils/gpu_mem.h"

namespace {

using u128 = unsigned __int128;
using T = u64;

constexpr uint64_t kPrime62 = 4611686018326724609ULL;

struct Args {
    int rows = 2;
    int inner = 2;
    int cols = 2;
    int bw = 16;
    uint64_t value_bound = 255;
    int poly_n = 8192;
    int c = 2;
    int t = 8;
    uint64_t seed = 1;
    uint64_t second_seed = 2;
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

static uint64_t ring_sub(uint64_t a, uint64_t b, int bw) {
    return ring_reduce128(u128(a) + (u128(1) << bw) - ring_reduce(b, bw), bw);
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

static std::pair<T, T> share_zp_value_to_ring(uint64_t value,
                                              int bw,
                                              std::mt19937_64 &rng,
                                              bool &conversion_ok) {
    uint64_t z0 = uniform_mod(kPrime62, rng);
    uint64_t z1 = mod_sub(value % kPrime62, z0, kPrime62);
    uint64_t r0 = 0;
    uint64_t r1 = 0;
    exact_zp_to_ring_shares(z0, z1, kPrime62, bw, r0, r1);
    conversion_ok = conversion_ok && (ring_add(r0, r1, bw) == ring_reduce(value, bw));
    return {static_cast<T>(r0), static_cast<T>(r1)};
}

static std::pair<T, T> share_ring_value(uint64_t value,
                                        int bw,
                                        std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, ring_mask(bw));
    uint64_t r0 = dist(rng);
    uint64_t r1 = ring_sub(value, r0, bw);
    return {static_cast<T>(r0), static_cast<T>(r1)};
}

static void append_raw(std::vector<uint8_t> &dst, const std::vector<T> &src) {
    const size_t old = dst.size();
    dst.resize(old + src.size() * sizeof(T));
    std::memcpy(dst.data() + old, src.data(), src.size() * sizeof(T));
}

static void serialize_fc_key(const std::vector<T> &a,
                             const std::vector<T> &b,
                             const std::vector<T> &c,
                             std::vector<uint8_t> &out) {
    out.clear();
    append_raw(out, a);
    append_raw(out, b);
    append_raw(out, c);
}

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
    return args.rows == 2 && args.inner == 2 && args.cols == 2 &&
           args.bw == 16 && args.value_bound == 255 && args.poly_n == 8192 &&
           args.c == 2 && args.t == 8;
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

    std::vector<T> a0(size_a), a1(size_a);
    std::vector<T> b0(size_b), b1(size_b);
    std::vector<T> c0(size_c), c1(size_c);

    for (size_t i = 0; i < size_a; ++i) {
        out.input[i] = static_cast<T>(bounded(rng));
        out.mask_a[i] = static_cast<T>(bounded(rng));
        auto shares = share_zp_value_to_ring(out.mask_a[i], args.bw, rng, out.conversion_ok);
        a0[i] = shares.first;
        a1[i] = shares.second;
        out.masked_input[i] = ring_add(out.input[i], out.mask_a[i], args.bw);
    }
    for (size_t i = 0; i < size_b; ++i) {
        out.weight[i] = static_cast<T>(bounded(rng));
        out.mask_b[i] = static_cast<T>(bounded(rng));
        auto shares = share_zp_value_to_ring(out.mask_b[i], args.bw, rng, out.conversion_ok);
        b0[i] = shares.first;
        b1[i] = shares.second;
        out.masked_weight[i] = ring_add(out.weight[i], out.mask_b[i], args.bw);
    }

    for (int r = 0; r < args.rows; ++r) {
        for (int col = 0; col < args.cols; ++col) {
            const size_t idx = static_cast<size_t>(r) * args.cols + col;
            uint64_t ab_field = 0;
            for (int k = 0; k < args.inner; ++k) {
                uint64_t av = out.mask_a[static_cast<size_t>(r) * args.inner + k];
                uint64_t bv = out.mask_b[static_cast<size_t>(k) * args.cols + col];
                uint64_t term = mod_mul(av, bv, kPrime62);
                ab_field += term;
                if (ab_field >= kPrime62) {
                    ab_field -= kPrime62;
                }
            }
            auto ab_shares = share_zp_value_to_ring(ab_field, args.bw, rng, out.conversion_ok);
            out.output_mask[idx] = static_cast<T>(uniform_mod(uint64_t(1) << args.bw, rng));
            auto mask_shares = share_ring_value(out.output_mask[idx], args.bw, rng);
            c0[idx] = ring_add(ab_shares.first, mask_shares.first, args.bw);
            c1[idx] = ring_add(ab_shares.second, mask_shares.second, args.bw);
            uint64_t clear = clear_matmul_entry(
                out.input, out.weight, args.rows, args.inner, args.cols, r, col, args.bw);
            out.expected_masked_output[idx] = ring_add(clear, out.output_mask[idx], args.bw);
        }
    }

    serialize_fc_key(a0, b0, c0, out.key0);
    serialize_fc_key(a1, b1, c1, out.key1);
    return out;
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

static std::vector<T> copy_from_gpu(T *src, size_t count) {
    std::vector<T> out(count);
    cudaError_t err = cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
    return out;
}

static DemoResult run_demo_case(const Args &args, uint64_t seed) {
    DemoResult result;
    result.dealer = generate_dealer_output(args, seed);

    MatmulParams p;
    p.batchSz = 1;
    p.M = args.rows;
    p.K = args.inner;
    p.N = args.cols;
    stdInit(p, args.bw, 0);

    uint8_t *key_ptr0 = result.dealer.key0.data();
    uint8_t *key_ptr1 = result.dealer.key1.data();
    GPUMatmulKey<T> key0 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr0);
    GPUMatmulKey<T> key1 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr1);
    const bool key_order_ok =
        key_ptr0 == result.dealer.key0.data() + result.dealer.key0.size() &&
        key_ptr1 == result.dealer.key1.data() + result.dealer.key1.size();

    T *d_x = nullptr;
    T *d_w = nullptr;
    T *d_a0 = nullptr;
    T *d_a1 = nullptr;
    T *d_b0 = nullptr;
    T *d_b1 = nullptr;
    copy_to_gpu(result.dealer.masked_input, &d_x);
    copy_to_gpu(result.dealer.masked_weight, &d_w);
    copy_to_gpu(std::vector<T>(key0.A, key0.A + p.size_A), &d_a0);
    copy_to_gpu(std::vector<T>(key1.A, key1.A + p.size_A), &d_a1);
    copy_to_gpu(std::vector<T>(key0.B, key0.B + p.size_B), &d_b0);
    copy_to_gpu(std::vector<T>(key1.B, key1.B + p.size_B), &d_b1);

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

    result.online_ok = key_order_ok && result.dealer.conversion_ok &&
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

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--seed N] [--second-seed N] [--csv-header]\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--second-seed") && i + 1 < argc) {
            args.second_seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (!validate_args(args) || args.seed == args.second_seed) {
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

    const bool replay_ok =
        first.online_ok && replay.online_ok &&
        first.dealer.key0 == replay.dealer.key0 &&
        first.dealer.key1 == replay.dealer.key1 &&
        first.reconstructed == replay.reconstructed;
    const bool second_ok = second.online_ok;
    const bool second_differs =
        first.dealer.key0 != second.dealer.key0 || first.dealer.key1 != second.dealer.key1;
    const bool validation = replay_ok && second_ok && second_differs;

    if (args.csv_header) {
        std::cout << "device,input_mode,seed,second_seed,rows,inner,cols,bw,value_bound,"
                  << "poly_n,c,t,noise,tf,key_bytes_per_party,corrected_carry_conversion,"
                  << "deterministic_replay,second_seed_validation,second_seed_distinct,"
                  << "online_contract,validation\n";
    }
    std::cout << "cuda_orca_fc_ringlpn_demo,bounded_q62_constant_polynomial,"
              << args.seed << "," << args.second_seed << ","
              << args.rows << "," << args.inner << "," << args.cols << ","
              << args.bw << "," << args.value_bound << ","
              << args.poly_n << "," << args.c << "," << args.t << ","
              << "regular,None,"
              << first.dealer.key0.size() << ","
              << (first.dealer.conversion_ok ? 1 : 0) << ","
              << (replay_ok ? 1 : 0) << ","
              << (second_ok ? 1 : 0) << ","
              << (second_differs ? 1 : 0) << ","
              << (first.online_ok ? "pass" : "fail") << ","
              << (validation ? "pass" : "fail") << "\n";

    return validation ? 0 : 2;
}
