
#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "fss/gpu_dpf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using T = u64;

namespace {

constexpr size_t kOneGB = 1024ULL * 1024ULL * 1024ULL;
constexpr int kDefaultBin = 16;
constexpr int kDefaultChunkSize = 8192;

struct Args {
    int bin = kDefaultBin;
    int n = kDefaultChunkSize;
    int chunk_size = kDefaultChunkSize;
    int iters = 100;
    int warmup = 10;
    bool csv_header = false;
};

struct SummaryStats {
    double mean_us;
    double stddev_us;
};

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --bin <bits> --n <N> [--chunk-size N] [--iters N] [--warmup N] [--csv-header]\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--bin") == 0 && i + 1 < argc) {
            args.bin = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            args.chunk_size = std::atoi(argv[++i]);
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

    if (args.bin <= LOG_AES_BLOCK_LEN || args.bin > 63) {
        std::cerr << "Unsupported bin: expected bin in [" << (LOG_AES_BLOCK_LEN + 1)
                  << ", 63]\n";
        std::exit(1);
    }
    if (args.n <= 0 || args.chunk_size <= 0 || args.iters <= 0 || args.warmup < 0) {
        usage(argv[0]);
        std::exit(1);
    }
    if (args.chunk_size > args.n) {
        args.chunk_size = args.n;
    }
    return args;
}

static SummaryStats compute_stats(const std::vector<double> &samples) {
    if (samples.empty()) {
        return {0.0, 0.0};
    }
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / static_cast<double>(samples.size());
    double var = 0.0;
    for (double sample : samples) {
        double delta = sample - mean;
        var += delta * delta;
    }
    var /= static_cast<double>(samples.size());
    return {mean, std::sqrt(var)};
}

static size_t align32(size_t value) {
    return value - (value % 32ULL);
}

static size_t estimate_dpf_key_bytes_single_party(int bin, int n, bool eval_all) {
    const size_t top_level_header = 3 * sizeof(int);
    const size_t per_tree_header = 3 * sizeof(int);

    const size_t mem_sz_one_k = static_cast<size_t>(bin - LOG_AES_BLOCK_LEN + 2) * sizeof(AESBlock);
    size_t batch_capacity = (24ULL * kOneGB) / mem_sz_one_k;
    batch_capacity = align32(batch_capacity);
    if (batch_capacity == 0) {
        batch_capacity = 32;
    }

    size_t total_bytes = top_level_header;
    int remaining = n;
    while (remaining > 0) {
        int cur_n = std::min<int>(remaining, static_cast<int>(batch_capacity));
        size_t mem_size_k = static_cast<size_t>(cur_n) * static_cast<size_t>(bin - LOG_AES_BLOCK_LEN) * sizeof(AESBlock);
        size_t mem_size_l = static_cast<size_t>(cur_n) * sizeof(AESBlock);
        size_t mem_size_t = 0;
        if (eval_all) {
            mem_size_t = static_cast<size_t>(cur_n) * sizeof(u32);
        } else {
            mem_size_t = static_cast<size_t>(((cur_n - 1) / PACKING_SIZE) + 1) * sizeof(PACK_TYPE) * static_cast<size_t>(bin - LOG_AES_BLOCK_LEN);
        }
        total_bytes += per_tree_header + mem_size_k + 2 * mem_size_l + mem_size_t;
        remaining -= cur_n;
    }
    return total_bytes;
}

static bool validate_key_layout(u8 *buffer, size_t bytes, int bin, int n) {
    u8 *ptr = buffer;
    GPUDPFKey key = readGPUDPFKey(&ptr);
    bool ok = true;
    ok = ok && key.bin == bin;
    ok = ok && key.M == n;
    ok = ok && key.memSzOut > 0;
    ok = ok && static_cast<size_t>(ptr - buffer) == bytes;
    if (key.bin > LOG_AES_BLOCK_LEN) {
        delete[] key.dpfTreeKey;
    }
    return ok;
}

template <typename Func>
static SummaryStats benchmark(Func &&func, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        func();
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        samples.push_back(static_cast<double>(elapsed.count()));
    }
    return compute_stats(samples);
}

static size_t generate_pair_full(u8 *buf0,
                                 u8 *buf1,
                                 int bin,
                                 int n,
                                 T *d_rin,
                                 AESGlobalContext *gaes) {
    u8 *ptr0 = buf0;
    u8 *ptr1 = buf1;
    gpuKeyGenDPF(&ptr0, SERVER0, bin, n, d_rin, gaes, true);
    gpuKeyGenDPF(&ptr1, SERVER1, bin, n, d_rin, gaes, true);
    return static_cast<size_t>(ptr0 - buf0) + static_cast<size_t>(ptr1 - buf1);
}

static void generate_pair_partial(u8 *chunk_buf0,
                                  u8 *chunk_buf1,
                                  int bin,
                                  int n,
                                  int chunk_size,
                                  T *d_rin,
                                  AESGlobalContext *gaes,
                                  size_t &peak_pair_bytes,
                                  size_t &total_pair_bytes) {
    peak_pair_bytes = 0;
    total_pair_bytes = 0;
    for (int offset = 0; offset < n; offset += chunk_size) {
        int cur_n = std::min(chunk_size, n - offset);
        u8 *ptr0 = chunk_buf0;
        u8 *ptr1 = chunk_buf1;
        gpuKeyGenDPF(&ptr0, SERVER0, bin, cur_n, d_rin + offset, gaes, true);
        gpuKeyGenDPF(&ptr1, SERVER1, bin, cur_n, d_rin + offset, gaes, true);
        size_t pair_bytes = static_cast<size_t>(ptr0 - chunk_buf0) + static_cast<size_t>(ptr1 - chunk_buf1);
        peak_pair_bytes = std::max(peak_pair_bytes, pair_bytes);
        total_pair_bytes += pair_bytes;
    }
}

static int run_benchmark(const Args &args) {
    AESGlobalContext gaes;
    initAESContext(&gaes);
    initGPUMemPool();
    initGPURandomness();

    T *d_rin = randomGEOnGpu<T>(args.n, args.bin);

    const size_t full_single_bytes = estimate_dpf_key_bytes_single_party(args.bin, args.n, true);
    const size_t chunk_single_bytes = estimate_dpf_key_bytes_single_party(args.bin, args.chunk_size, true);
    u8 *full_buf0 = cpuMalloc(full_single_bytes, true);
    u8 *full_buf1 = cpuMalloc(full_single_bytes, true);
    u8 *chunk_buf0 = cpuMalloc(chunk_single_bytes, true);
    u8 *chunk_buf1 = cpuMalloc(chunk_single_bytes, true);

    const size_t full_pair_key_bytes = generate_pair_full(full_buf0, full_buf1, args.bin, args.n, d_rin, &gaes);
    bool valid = true;
    valid = valid && validate_key_layout(full_buf0, full_pair_key_bytes / 2, args.bin, args.n);
    valid = valid && validate_key_layout(full_buf1, full_pair_key_bytes / 2, args.bin, args.n);

    size_t partial_peak_pair_bytes = 0;
    size_t partial_total_pair_bytes = 0;
    generate_pair_partial(chunk_buf0,
                          chunk_buf1,
                          args.bin,
                          args.n,
                          args.chunk_size,
                          d_rin,
                          &gaes,
                          partial_peak_pair_bytes,
                          partial_total_pair_bytes);

    for (int offset = 0; offset < args.n; offset += args.chunk_size) {
        int cur_n = std::min(args.chunk_size, args.n - offset);
        size_t cur_bytes = estimate_dpf_key_bytes_single_party(args.bin, cur_n, true);
        u8 *ptr0 = chunk_buf0;
        u8 *ptr1 = chunk_buf1;
        gpuKeyGenDPF(&ptr0, SERVER0, args.bin, cur_n, d_rin + offset, &gaes, true);
        gpuKeyGenDPF(&ptr1, SERVER1, args.bin, cur_n, d_rin + offset, &gaes, true);
        valid = valid && validate_key_layout(chunk_buf0, static_cast<size_t>(ptr0 - chunk_buf0), args.bin, cur_n);
        valid = valid && validate_key_layout(chunk_buf1, static_cast<size_t>(ptr1 - chunk_buf1), args.bin, cur_n);
        valid = valid && cur_bytes == static_cast<size_t>(ptr0 - chunk_buf0);
        valid = valid && cur_bytes == static_cast<size_t>(ptr1 - chunk_buf1);
    }

    SummaryStats full_stats = benchmark(
        [&]() {
            generate_pair_full(full_buf0, full_buf1, args.bin, args.n, d_rin, &gaes);
        },
        args.warmup,
        args.iters);

    SummaryStats partial_stats = benchmark(
        [&]() {
            size_t ignored_peak = 0;
            size_t ignored_total = 0;
            generate_pair_partial(chunk_buf0,
                                  chunk_buf1,
                                  args.bin,
                                  args.n,
                                  args.chunk_size,
                                  d_rin,
                                  &gaes,
                                  ignored_peak,
                                  ignored_total);
        },
        args.warmup,
        args.iters);

    const char *validation = valid ? "pass" : "fail";
    const double peak_reduction = partial_peak_pair_bytes > 0
                                      ? static_cast<double>(full_pair_key_bytes) / static_cast<double>(partial_peak_pair_bytes)
                                      : 0.0;
    const double total_bytes_multiplier = full_pair_key_bytes > 0
                                              ? static_cast<double>(partial_total_pair_bytes) / static_cast<double>(full_pair_key_bytes)
                                              : 0.0;
    const double keygen_time_overhead = full_stats.mean_us > 0.0
                                            ? partial_stats.mean_us / full_stats.mean_us
                                            : 0.0;

    std::cout << "cuda_dpf_online,eval_all," << args.bin << "," << args.n << "," << args.chunk_size
              << "," << args.iters << "," << validation << ","
              << full_pair_key_bytes << "," << partial_peak_pair_bytes << ","
              << partial_total_pair_bytes << "," << peak_reduction << ","
              << total_bytes_multiplier << "," << full_stats.mean_us << ","
              << full_stats.stddev_us << "," << partial_stats.mean_us << ","
              << partial_stats.stddev_us << "," << keygen_time_overhead << ","
              << (valid ? 1 : 0) << "\n";

    cpuFree(full_buf0, true);
    cpuFree(full_buf1, true);
    cpuFree(chunk_buf0, true);
    cpuFree(chunk_buf1, true);
    gpuFree(d_rin);
    destroyGPURandomness();
    return valid ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,bin,n,chunk_size,iters,validation,full_pair_key_bytes,partial_peak_pair_key_bytes,partial_total_pair_key_bytes,peak_reduction,total_bytes_multiplier,full_pair_keygen_mean_us,full_pair_keygen_std_us,partial_pair_keygen_mean_us,partial_pair_keygen_std_us,keygen_time_overhead,correct\n";
    }
    return run_benchmark(args);
}