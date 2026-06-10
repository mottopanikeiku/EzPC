// Ideal-OLE dealerless FC transcript test (Step 1 of the dealerless roadmap).
//
// Builds party-local Orca FC key buffers (A_i || B_i || C_i) from an ideal-OLE
// transcript instead of the centralized plaintext-product keywriter, then drives
// the unchanged gpuMatmulBeaver online path and checks that the reconstruction
// equals the clear FC output plus the output mask. Reports the OLE and conversion
// counts so the OLE-to-Beaver cost of the reduction is explicit.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "fss/gpu_matmul.h"
#include "ringlpn/src/orca_fc_ideal_ole_transcript.cuh"
#include "ringlpn/src/orca_fc_ringlpn_keywriter.cuh"
#include "utils/gpu_mem.h"

namespace {

using u128 = unsigned __int128;
using T = u64;

struct Args {
    int rows = 2;
    int inner = 2;
    int cols = 2;
    int bw = 16;
    int qbits = 64;
    uint64_t value_bound = 255;
    uint64_t seed = 1;
    bool csv_header = false;
};

static uint64_t ring_mask(int bw) {
    return bw == 64 ? UINT64_MAX : ((uint64_t(1) << bw) - 1);
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

static MatmulParams make_matmul_params(const Args &args) {
    MatmulParams p;
    p.batchSz = 1;
    p.M = args.rows;
    p.K = args.inner;
    p.N = args.cols;
    stdInit(p, args.bw, 0);
    return p;
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

static std::vector<T> key_span_to_vector(T *ptr, int count) {
    return std::vector<T>(ptr, ptr + count);
}

static void append_T(std::vector<uint8_t> &buf, const std::vector<T> &src) {
    const size_t off = buf.size();
    buf.resize(off + src.size() * sizeof(T));
    std::memcpy(buf.data() + off, src.data(), src.size() * sizeof(T));
}

static uint64_t clear_matmul_entry(const std::vector<T> &a,
                                   const std::vector<T> &b,
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
    return ring_reduce128(acc, bw);
}

struct CaseResult {
    bool transcript_ok = false;
    bool key_order_ok = false;
    bool masks_consistent = false;
    bool online_ok = false;
    ringlpn_orca::TranscriptCounters counters;
    uint64_t expected_ole = 0;
    uint64_t expected_conv = 0;
};

static CaseResult run_case(const Args &args) {
    CaseResult result;
    MatmulParams p = make_matmul_params(args);

    std::vector<T> a0, a1, b0, b1, c0, c1, mask_a, mask_b, mask_y;
    result.transcript_ok = ringlpn_orca::buildIdealOleTranscript<T>(
        p, args.qbits, args.seed, a0, a1, b0, b1, c0, c1, mask_a, mask_b, mask_y,
        result.counters);
    if (!result.transcript_ok) {
        return result;
    }

    result.expected_ole = static_cast<uint64_t>(2) * p.batchSz * p.M * p.K * p.N;
    result.expected_conv = static_cast<uint64_t>(p.batchSz) * p.M * p.N;

    // Sanity: the sampled shares reconstruct the masks they claim to.
    result.masks_consistent = true;
    for (int i = 0; i < p.size_A; ++i) {
        if (ring_add(a0[i], a1[i], args.bw) != ring_add(mask_a[i], 0, args.bw)) {
            result.masks_consistent = false;
        }
    }

    // Assemble party-local key buffers in Orca's A || B || C byte order.
    std::vector<uint8_t> key0;
    std::vector<uint8_t> key1;
    append_T(key0, a0);
    append_T(key0, b0);
    append_T(key0, c0);
    append_T(key1, a1);
    append_T(key1, b1);
    append_T(key1, c1);

    // Choose clear inputs and form the masked online inputs from the masks.
    std::mt19937_64 rng(args.seed ^ 0x0DEA11E5ULL);
    std::uniform_int_distribution<uint64_t> ring_dist(0, ring_mask(args.bw));
    std::vector<T> input(p.size_A);
    std::vector<T> weight(p.size_B);
    std::vector<T> masked_input(p.size_A);
    std::vector<T> masked_weight(p.size_B);
    for (int i = 0; i < p.size_A; ++i) {
        input[i] = static_cast<T>(ring_dist(rng));
        masked_input[i] = static_cast<T>(ring_add(input[i], mask_a[i], args.bw));
    }
    for (int i = 0; i < p.size_B; ++i) {
        weight[i] = static_cast<T>(ring_dist(rng));
        masked_weight[i] = static_cast<T>(ring_add(weight[i], mask_b[i], args.bw));
    }

    std::vector<T> expected(p.size_C);
    for (int r = 0; r < args.rows; ++r) {
        for (int col = 0; col < args.cols; ++col) {
            const size_t idx = static_cast<size_t>(r) * args.cols + col;
            const uint64_t clear =
                clear_matmul_entry(input, weight, args.inner, args.cols, r, col, args.bw);
            expected[idx] = static_cast<T>(ring_add(clear, mask_y[idx], args.bw));
        }
    }

    // Read keys back and run the unchanged Beaver matmul online path.
    uint8_t *key_ptr0 = key0.data();
    uint8_t *key_ptr1 = key1.data();
    GPUMatmulKey<T> gkey0 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr0);
    GPUMatmulKey<T> gkey1 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr1);
    result.key_order_ok = key_ptr0 == key0.data() + key0.size() &&
                          key_ptr1 == key1.data() + key1.size();

    T *d_x = nullptr;
    T *d_w = nullptr;
    T *d_a0 = nullptr;
    T *d_a1 = nullptr;
    T *d_b0 = nullptr;
    T *d_b1 = nullptr;
    copy_to_gpu(masked_input, &d_x);
    copy_to_gpu(masked_weight, &d_w);
    copy_to_gpu(key_span_to_vector(gkey0.A, p.size_A), &d_a0);
    copy_to_gpu(key_span_to_vector(gkey1.A, p.size_A), &d_a1);
    copy_to_gpu(key_span_to_vector(gkey0.B, p.size_B), &d_b0);
    copy_to_gpu(key_span_to_vector(gkey1.B, p.size_B), &d_b1);

    Stats stats0;
    Stats stats1;
    T *d_o0 = gpuMatmulBeaver<T>(p, gkey0, SERVER0, d_x, d_w, d_a0, d_b0, nullptr, &stats0);
    T *d_o1 = gpuMatmulBeaver<T>(p, gkey1, SERVER1, d_x, d_w, d_a1, d_b1, nullptr, &stats1);
    std::vector<T> o0 = copy_from_gpu(d_o0, p.size_C);
    std::vector<T> o1 = copy_from_gpu(d_o1, p.size_C);

    std::vector<T> reconstructed(p.size_C);
    for (int i = 0; i < p.size_C; ++i) {
        reconstructed[i] = static_cast<T>(ring_add(o0[i], o1[i], args.bw));
    }

    result.online_ok = result.key_order_ok && result.masks_consistent &&
                       result.counters.ole_calls == result.expected_ole &&
                       result.counters.conversions == result.expected_conv &&
                       reconstructed == expected;

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
              << " [--rows M] [--inner K] [--cols N] [--bw N] [--qbits 64]"
              << " [--value-bound B] [--seed N] [--csv-header]\n";
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
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.rows <= 0 || args.inner <= 0 || args.cols <= 0 || args.rows > 8 ||
        args.inner > 8 || args.cols > 8 || args.bw <= 2 || args.bw > 30 ||
        args.qbits != 64) {
        usage(argv[0]);
        std::exit(1);
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    initGPUMemPool();

    CaseResult r = run_case(args);

    if (args.csv_header) {
        std::cout << "device,mode,requested_qbits,actual_qbits,seed,rows,inner,cols,bw,"
                  << "no_prime_wrap_bound,ole_calls,expected_ole,conversions,expected_conversions,"
                  << "transcript_built,masks_consistent,key_order,online_contract,validation\n";
    }
    std::cout << "cuda_orca_fc_ideal_ole_transcript,ideal_ole_q" << args.qbits
              << "_constant_polynomial," << args.qbits << ","
              << ringlpn_orca::actualQbitsForQbits(args.qbits) << "," << args.seed << ","
              << args.rows << "," << args.inner << "," << args.cols << "," << args.bw << ","
              << (r.counters.bound_ok ? 1 : 0) << "," << r.counters.ole_calls << ","
              << r.expected_ole << "," << r.counters.conversions << "," << r.expected_conv << ","
              << (r.transcript_ok ? 1 : 0) << "," << (r.masks_consistent ? 1 : 0) << ","
              << (r.key_order_ok ? 1 : 0) << "," << (r.online_ok ? "pass" : "fail") << ","
              << (r.online_ok ? "pass" : "fail") << "\n";

    return r.online_ok ? 0 : 2;
}
