#include "gpu_spfss_zp.cuh"

#define RINGLPN_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_ole"
#endif
#define Stats RingLpnNttStats
#include "bench_ntt_cuda_cheddar.cu"
#undef Stats

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Word = uint64_t;
using Clock = std::chrono::high_resolution_clock;

struct OleArgs {
    int n = kMinDegree;
    int c = 2;
    int t = 64;
    int qbits = 64;
    int iters = 3;
    int warmup = 1;
    int chunk_size = 8192;
    uint64_t seed = 1;
    std::string noise = "uniform";
    bool csv_header = false;
    bool skip_validation = false;
};

struct SparsePoly {
    std::vector<int> positions;
    std::vector<Word> values;
};

struct SummaryStats {
    double mean_us = 0.0;
    double stddev_us = 0.0;
};

__device__ __forceinline__ Word mod_add_device(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? s - modulus : s;
}

__device__ __forceinline__ Word mod_sub_device(Word a, Word b, Word modulus) {
    return a >= b ? a - b : modulus - (b - a);
}

__global__ void reduce_batches_kernel(const Word *batches,
                                      Word *out,
                                      int batch_count,
                                      int n,
                                      Word modulus) {
    size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= static_cast<size_t>(n)) {
        return;
    }
    Word acc = 0;
    for (int batch = 0; batch < batch_count; ++batch) {
        acc = mod_add_device(acc, batches[static_cast<size_t>(batch) * n + coeff], modulus);
    }
    out[coeff] = acc;
}

__global__ void add_vectors_kernel(const Word *a,
                                   const Word *b,
                                   Word *out,
                                   size_t count,
                                   Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = mod_add_device(a[idx], b[idx], modulus);
    }
}

__global__ void fold_2n_to_n_kernel(const Word *in,
                                    Word *out,
                                    int n,
                                    Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        out[idx] = mod_sub_device(in[idx], in[idx + static_cast<size_t>(n)], modulus);
    }
}

__global__ void scatter_regular_group_kernel(const Word *group,
                                             Word *full,
                                             int group_domain,
                                             int base,
                                             Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(group_domain)) {
        Word value = group[idx];
        if (value != 0) {
            size_t out_idx = static_cast<size_t>(base) + idx;
            full[out_idx] = mod_add_device(full[out_idx], value, modulus);
        }
    }
}

static void ole_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --n <deg> [--qbits 64|128] [--c N] [--t N] [--seed N]"
              << " [--iters N] [--warmup N] [--chunk-size N]"
              << " [--noise uniform|regular] [--csv-header] [--skip-validation]\n";
}

static OleArgs parse_ole_args(int argc, char **argv) {
    OleArgs args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--qbits") && i + 1 < argc) {
            args.qbits = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--c") && i + 1 < argc) {
            args.c = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--t") && i + 1 < argc) {
            args.t = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--noise") && i + 1 < argc) {
            args.noise = argv[++i];
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--chunk-size") && i + 1 < argc) {
            args.chunk_size = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else if (!std::strcmp(argv[i], "--skip-validation")) {
            args.skip_validation = true;
        } else {
            ole_usage(argv[0]);
            std::exit(1);
        }
    }

    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree) {
        ole_usage(argv[0]);
        std::exit(1);
    }
    if ((args.qbits != 64 && args.qbits != 128) ||
        args.c <= 0 || args.t <= 0 || args.t > args.n ||
        args.iters <= 0 || args.warmup < 0 || args.chunk_size <= 0) {
        ole_usage(argv[0]);
        std::exit(1);
    }
    if (args.noise != "uniform" && args.noise != "regular") {
        ole_usage(argv[0]);
        std::exit(1);
    }
    if (args.noise == "regular" &&
        (args.n % args.t != 0 || !is_power_of_two(args.t))) {
        std::cerr << "Regular noise requires power-of-two t dividing n for the first paper-compatible GPU artifact\n";
        std::exit(1);
    }
    if (static_cast<uint64_t>(args.c) * static_cast<uint64_t>(args.c) > 65535ULL) {
        std::cerr << "Unsupported c: c*c must fit CUDA grid.y limit\n";
        std::exit(1);
    }
    return args;
}

static std::vector<ModulusConfig<Word>> ole_modulus_configs(int qbits) {
    if (qbits == 128) {
        return {kConfig62, kConfig62Crt2};
    }
    return {kConfig62};
}

static int ole_actual_qbits(const std::vector<ModulusConfig<Word>> &configs) {
    int total = 0;
    for (const auto &config : configs) {
        total += config.actual_qbits;
    }
    return total;
}

static uint64_t mix_seed(uint64_t seed, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL + (tag << 6) + (tag >> 2);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static bool use_regular_noise(const OleArgs &args) {
    return args.noise == "regular";
}

static int regular_bucket_size(const OleArgs &args) {
    return args.n / args.t;
}

static int spfss_domain_size(const OleArgs &args) {
    return use_regular_noise(args) ? 2 * regular_bucket_size(args) : 2 * args.n;
}

static int spfss_group_count(const OleArgs &args) {
    return use_regular_noise(args) ? 2 * args.t - 1 : 1;
}

static SummaryStats summarize(const std::vector<double> &samples) {
    SummaryStats s;
    if (samples.empty()) {
        return s;
    }
    s.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) /
                static_cast<double>(samples.size());
    double var = 0.0;
    for (double sample : samples) {
        double delta = sample - s.mean_us;
        var += delta * delta;
    }
    s.stddev_us = std::sqrt(var / static_cast<double>(samples.size()));
    return s;
}

static int log2i(int n) {
    int out = 0;
    while (n > 1) {
        n >>= 1;
        ++out;
    }
    return out;
}

static Word host_mod_add(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? s - modulus : s;
}

static SparsePoly sample_sparse(int n, int t, Word modulus, std::mt19937_64 &rng) {
    SparsePoly out;
    out.positions.reserve(t);
    out.values.reserve(t);
    std::uniform_int_distribution<int> pos_dist(0, n - 1);
    std::uniform_int_distribution<Word> value_dist(1, modulus - 1);
    std::unordered_set<int> used;
    used.reserve(static_cast<size_t>(t) * 2 + 1);
    while (static_cast<int>(used.size()) < t) {
        int pos = pos_dist(rng);
        if (!used.insert(pos).second) {
            continue;
        }
        out.positions.push_back(pos);
        out.values.push_back(value_dist(rng));
    }
    return out;
}

static SparsePoly sample_sparse_regular(int n, int t, Word modulus, std::mt19937_64 &rng) {
    SparsePoly out;
    out.positions.reserve(t);
    out.values.reserve(t);
    int bucket = n / t;
    std::uniform_int_distribution<Word> value_dist(1, modulus - 1);
    for (int b = 0; b < t; ++b) {
        std::uniform_int_distribution<int> offset_dist(0, bucket - 1);
        out.positions.push_back(b * bucket + offset_dist(rng));
        out.values.push_back(value_dist(rng));
    }
    return out;
}

static std::vector<Word> sample_dense(int n, Word modulus, std::mt19937_64 &rng) {
    std::uniform_int_distribution<Word> dist(0, modulus - 1);
    std::vector<Word> out(n);
    for (Word &v : out) {
        v = dist(rng);
    }
    return out;
}

static void add_sparse_to_dense(const SparsePoly &s, std::vector<Word> &out, int n, Word modulus) {
    for (size_t i = 0; i < s.positions.size(); ++i) {
        out[s.positions[i]] = host_mod_add(out[s.positions[i]], s.values[i], modulus);
    }
}

static std::vector<Word> flatten_dense(const std::vector<std::vector<Word>> &polys, int n) {
    std::vector<Word> out(static_cast<size_t>(polys.size()) * static_cast<size_t>(n));
    for (size_t i = 0; i < polys.size(); ++i) {
        std::copy(polys[i].begin(), polys[i].end(),
                  out.begin() + i * static_cast<size_t>(n));
    }
    return out;
}

static void copy_to_device(Word **dst, const std::vector<Word> &src, const char *label) {
    check(cudaMalloc(reinterpret_cast<void **>(dst), src.size() * sizeof(Word)), label);
    check(cudaMemcpy(*dst, src.data(), src.size() * sizeof(Word), cudaMemcpyHostToDevice), label);
}

static void alloc_device(Word **dst, size_t count, const char *label) {
    check(cudaMalloc(reinterpret_cast<void **>(dst), count * sizeof(Word)), label);
}

static void reduce_batches(const Word *d_batches,
                           Word *d_out,
                           int batch_count,
                           int n,
                           Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    reduce_batches_kernel<<<grid, block>>>(d_batches, d_out, batch_count, n, modulus);
    check(cudaGetLastError(), "launch reduce_batches_kernel");
}

static void add_vectors(const Word *d_a, const Word *d_b, Word *d_out, int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    add_vectors_kernel<<<grid, block>>>(d_a, d_b, d_out, static_cast<size_t>(n), modulus);
    check(cudaGetLastError(), "launch add_vectors_kernel");
}

static void fold_2n_to_n(const Word *d_in, Word *d_out, int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    fold_2n_to_n_kernel<<<grid, block>>>(d_in, d_out, n, modulus);
    check(cudaGetLastError(), "launch fold_2n_to_n_kernel");
}

struct OleState {
    OleArgs args;
    int log_degree = 0;
    int log_domain = 0;
    int limb_index = 0;
    ModulusConfig<Word> config = kConfig62;
    Word modulus = kConfig62.modulus;
    HostTables<Word> host_tables;
    DeviceTables<Word> tables;
    std::vector<Word> phi_norm;
    std::vector<Word> post_norm;
    std::vector<std::vector<Word>> a;
    std::vector<std::vector<SparsePoly>> e;
    std::vector<ringlpn_spfss_zp::GPUDPFZpKey> keys0;
    std::vector<ringlpn_spfss_zp::GPUDPFZpKey> keys1;
    size_t spfss_pair_key_bytes = 0;

    Word *d_a = nullptr;
    Word *d_e0 = nullptr;
    Word *d_e1 = nullptr;
    Word *d_aa_lhs = nullptr;
    Word *d_aa_rhs = nullptr;
    Word *d_aa = nullptr;
    Word *d_aw = nullptr;
    Word *d_bw = nullptr;
    Word *d_cw = nullptr;
    Word *d_terms = nullptr;
    Word *d_x0 = nullptr;
    Word *d_x1 = nullptr;
    Word *d_u0 = nullptr;
    Word *d_u1 = nullptr;
    Word *d_u2n0 = nullptr;
    Word *d_u2n1 = nullptr;
    Word *d_group0 = nullptr;
    Word *d_group1 = nullptr;
    Word *d_z0 = nullptr;
    Word *d_z1 = nullptr;
    Word *d_zsum = nullptr;
    Word *d_expected = nullptr;

    void cleanup() {
        free_tables(tables);
        cudaFree(d_a);
        cudaFree(d_e0);
        cudaFree(d_e1);
        cudaFree(d_aa_lhs);
        cudaFree(d_aa_rhs);
        cudaFree(d_aa);
        cudaFree(d_aw);
        cudaFree(d_bw);
        cudaFree(d_cw);
        cudaFree(d_terms);
        cudaFree(d_x0);
        cudaFree(d_x1);
        cudaFree(d_u0);
        cudaFree(d_u1);
        cudaFree(d_u2n0);
        cudaFree(d_u2n1);
        cudaFree(d_group0);
        cudaFree(d_group1);
        cudaFree(d_z0);
        cudaFree(d_z1);
        cudaFree(d_zsum);
        cudaFree(d_expected);
    }
};

struct OleLimbResult {
    bool correct = false;
    bool host_validation_ran = false;
    size_t key_bytes = 0;
    double keygen_us = 0.0;
};

static void build_inputs(OleState &state) {
    const int n = state.args.n;
    const int c = state.args.c;
    const Word modulus = state.modulus;
    std::mt19937_64 rng(state.args.seed);

    state.a.resize(c);
    state.a[0].assign(n, 0);
    state.a[0][0] = 1;
    for (int i = 1; i < c; ++i) {
        state.a[i] = sample_dense(n, modulus, rng);
    }

    state.e.assign(2, std::vector<SparsePoly>(c));
    for (int party = 0; party < 2; ++party) {
        for (int i = 0; i < c; ++i) {
            state.e[party][i] = use_regular_noise(state.args)
                                    ? sample_sparse_regular(n, state.args.t, modulus, rng)
                                    : sample_sparse(n, state.args.t, modulus, rng);
        }
    }

    std::vector<std::vector<Word>> e0_dense(c, std::vector<Word>(n, 0));
    std::vector<std::vector<Word>> e1_dense(c, std::vector<Word>(n, 0));
    for (int i = 0; i < c; ++i) {
        add_sparse_to_dense(state.e[0][i], e0_dense[i], n, modulus);
        add_sparse_to_dense(state.e[1][i], e1_dense[i], n, modulus);
    }

    copy_to_device(&state.d_a, flatten_dense(state.a, n), "copy a");
    copy_to_device(&state.d_e0, flatten_dense(e0_dense, n), "copy e0");
    copy_to_device(&state.d_e1, flatten_dense(e1_dense, n), "copy e1");

    const int cc = c * c;
    std::vector<std::vector<Word>> aa_lhs(cc);
    std::vector<std::vector<Word>> aa_rhs(cc);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * c;
            aa_lhs[idx] = state.a[i];
            aa_rhs[idx] = state.a[j];
        }
    }
    copy_to_device(&state.d_aa_lhs, flatten_dense(aa_lhs, n), "copy aa lhs");
    copy_to_device(&state.d_aa_rhs, flatten_dense(aa_rhs, n), "copy aa rhs");

    size_t c_coeffs = static_cast<size_t>(c) * n;
    size_t cc_coeffs = static_cast<size_t>(cc) * n;
    alloc_device(&state.d_aw, std::max(c_coeffs, cc_coeffs), "alloc work a");
    alloc_device(&state.d_bw, std::max(c_coeffs, cc_coeffs), "alloc work b");
    alloc_device(&state.d_cw, std::max(c_coeffs, cc_coeffs), "alloc work c");
    alloc_device(&state.d_terms, cc_coeffs, "alloc terms");
    alloc_device(&state.d_aa, cc_coeffs, "alloc aa");
    alloc_device(&state.d_x0, n, "alloc x0");
    alloc_device(&state.d_x1, n, "alloc x1");
    alloc_device(&state.d_u0, cc_coeffs, "alloc u0");
    alloc_device(&state.d_u1, cc_coeffs, "alloc u1");
    alloc_device(&state.d_u2n0, static_cast<size_t>(2) * n, "alloc u2n0");
    alloc_device(&state.d_u2n1, static_cast<size_t>(2) * n, "alloc u2n1");
    if (use_regular_noise(state.args)) {
        alloc_device(&state.d_group0, spfss_domain_size(state.args), "alloc regular group0");
        alloc_device(&state.d_group1, spfss_domain_size(state.args), "alloc regular group1");
    }
    alloc_device(&state.d_z0, n, "alloc z0");
    alloc_device(&state.d_z1, n, "alloc z1");
    alloc_device(&state.d_zsum, n, "alloc zsum");
    alloc_device(&state.d_expected, n, "alloc expected");

    run_full_polymul(state.d_aa_lhs,
                     state.d_aa_rhs,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_aa,
                     state.tables,
                     n,
                     cc,
                     state.log_degree);
    check(cudaDeviceSynchronize(), "sync aa precompute");
}

static double build_spfss_keys(OleState &state, AESGlobalContext *gaes) {
    const int c = state.args.c;
    const int t = state.args.t;
    const Word modulus = state.modulus;
    const int cc = c * c;
    const int groups = spfss_group_count(state.args);
    const int bucket = use_regular_noise(state.args) ? regular_bucket_size(state.args) : 0;
    state.keys0.resize(static_cast<size_t>(cc) * groups);
    state.keys1.resize(static_cast<size_t>(cc) * groups);
    state.spfss_pair_key_bytes = 0;

    auto start = Clock::now();
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            const SparsePoly &e0 = state.e[0][i];
            const SparsePoly &e1 = state.e[1][j];
            size_t matrix_idx = static_cast<size_t>(i) + static_cast<size_t>(j) * c;
            if (use_regular_noise(state.args)) {
                for (int group = 0; group < groups; ++group) {
                    std::vector<Word> alphas;
                    std::vector<Word> betas;
                    alphas.reserve(t);
                    betas.reserve(t);
                    for (int k = 0; k < t; ++k) {
                        int l = group - k;
                        if (l < 0 || l >= t) {
                            continue;
                        }
                        int off0 = e0.positions[k] - k * bucket;
                        int off1 = e1.positions[l] - l * bucket;
                        alphas.push_back(static_cast<Word>(off0 + off1));
                        betas.push_back(mod_mul_host(e0.values[k], e1.values[l], modulus));
                    }
                    size_t key_idx = matrix_idx * static_cast<size_t>(groups) +
                                     static_cast<size_t>(group);
                    ringlpn_spfss_zp::gpuKeyGenDPFZpPair(
                        alphas,
                        betas,
                        state.log_domain,
                        modulus,
                        state.args.seed ^ (0xA24BAED4963EE407ULL +
                                           key_idx * 0x9E3779B97F4A7C15ULL),
                        gaes,
                        state.keys0[key_idx],
                        state.keys1[key_idx]);
                    state.spfss_pair_key_bytes +=
                        ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys0[key_idx]) +
                        ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys1[key_idx]);
                }
            } else {
                std::vector<Word> alphas;
                std::vector<Word> betas;
                alphas.reserve(static_cast<size_t>(t) * t);
                betas.reserve(static_cast<size_t>(t) * t);
                for (int k = 0; k < t; ++k) {
                    for (int l = 0; l < t; ++l) {
                        alphas.push_back(static_cast<Word>(e0.positions[k] + e1.positions[l]));
                        betas.push_back(mod_mul_host(e0.values[k], e1.values[l], modulus));
                    }
                }
                size_t key_idx = matrix_idx;
                ringlpn_spfss_zp::gpuKeyGenDPFZpPair(
                    alphas,
                    betas,
                    state.log_domain,
                    modulus,
                    state.args.seed ^ (0xA24BAED4963EE407ULL +
                                       key_idx * 0x9E3779B97F4A7C15ULL),
                    gaes,
                    state.keys0[key_idx],
                    state.keys1[key_idx]);
                state.spfss_pair_key_bytes +=
                    ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys0[key_idx]) +
                    ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys1[key_idx]);
            }
        }
    }
    check(cudaDeviceSynchronize(), "sync SPFSS keygen");
    auto end = Clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

static void run_x_phase(OleState &state) {
    int n = state.args.n;
    int c = state.args.c;
    run_full_polymul(state.d_a,
                     state.d_e0,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_terms,
                     state.tables,
                     n,
                     c,
                     state.log_degree);
    reduce_batches(state.d_terms, state.d_x0, c, n, state.modulus);

    run_full_polymul(state.d_a,
                     state.d_e1,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_terms,
                     state.tables,
                     n,
                     c,
                     state.log_degree);
    reduce_batches(state.d_terms, state.d_x1, c, n, state.modulus);
}

static void run_spfss_eval_phase(OleState &state, AESGlobalContext *gaes) {
    const int n = state.args.n;
    const int cc = state.args.c * state.args.c;
    const int groups = spfss_group_count(state.args);
    for (int idx = 0; idx < cc; ++idx) {
        if (use_regular_noise(state.args)) {
            const int bucket = regular_bucket_size(state.args);
            const int group_domain = spfss_domain_size(state.args);
            check(cudaMemset(state.d_u2n0, 0, static_cast<size_t>(2) * n * sizeof(Word)),
                  "zero regular full u2n0");
            check(cudaMemset(state.d_u2n1, 0, static_cast<size_t>(2) * n * sizeof(Word)),
                  "zero regular full u2n1");
            for (int group = 0; group < groups; ++group) {
                size_t key_idx = static_cast<size_t>(idx) * groups + group;
                ringlpn_spfss_zp::gpuDpfZpFullEvalSum(state.keys0[key_idx], state.d_group0, gaes);
                ringlpn_spfss_zp::gpuDpfZpFullEvalSum(state.keys1[key_idx], state.d_group1, gaes);
                dim3 block(256);
                dim3 grid(grid_size(static_cast<size_t>(group_domain), block.x));
                int base = group * bucket;
                scatter_regular_group_kernel<<<grid, block>>>(
                    state.d_group0, state.d_u2n0, group_domain, base, state.modulus);
                check(cudaGetLastError(), "launch scatter regular group0");
                scatter_regular_group_kernel<<<grid, block>>>(
                    state.d_group1, state.d_u2n1, group_domain, base, state.modulus);
                check(cudaGetLastError(), "launch scatter regular group1");
            }
        } else {
            ringlpn_spfss_zp::gpuDpfZpFullEvalSum(state.keys0[idx], state.d_u2n0, gaes);
            ringlpn_spfss_zp::gpuDpfZpFullEvalSum(state.keys1[idx], state.d_u2n1, gaes);
        }
        fold_2n_to_n(state.d_u2n0, state.d_u0 + static_cast<size_t>(idx) * n, n, state.modulus);
        fold_2n_to_n(state.d_u2n1, state.d_u1 + static_cast<size_t>(idx) * n, n, state.modulus);
    }
}

static void run_z_phase(OleState &state) {
    int n = state.args.n;
    int cc = state.args.c * state.args.c;
    run_full_polymul(state.d_aa,
                     state.d_u0,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_terms,
                     state.tables,
                     n,
                     cc,
                     state.log_degree);
    reduce_batches(state.d_terms, state.d_z0, cc, n, state.modulus);

    run_full_polymul(state.d_aa,
                     state.d_u1,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_terms,
                     state.tables,
                     n,
                     cc,
                     state.log_degree);
    reduce_batches(state.d_terms, state.d_z1, cc, n, state.modulus);
}

static bool validate_outputs(OleState &state, bool &host_validation_ran) {
    host_validation_ran = false;
    if (state.args.skip_validation) {
        return true;
    }
    int n = state.args.n;
    add_vectors(state.d_z0, state.d_z1, state.d_zsum, n, state.modulus);
    run_full_polymul(state.d_x0,
                     state.d_x1,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_expected,
                     state.tables,
                     n,
                     1,
                     state.log_degree);
    check(cudaDeviceSynchronize(), "sync validation");

    std::vector<Word> zsum(n);
    std::vector<Word> expected(n);
    check(cudaMemcpy(zsum.data(), state.d_zsum, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy zsum");
    check(cudaMemcpy(expected.data(), state.d_expected, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy expected");
    bool gpu_ok = compare_vectors(expected, zsum, n, "OLE z0+z1");

    if (n <= 8192) {
        std::vector<Word> x0(n);
        std::vector<Word> x1(n);
        check(cudaMemcpy(x0.data(), state.d_x0, sizeof(Word) * n, cudaMemcpyDeviceToHost),
              "copy x0");
        check(cudaMemcpy(x1.data(), state.d_x1, sizeof(Word) * n, cudaMemcpyDeviceToHost),
              "copy x1");
        std::vector<Word> host_expected =
            host_polymul_reference(x0,
                                   x1,
                                   state.phi_norm,
                                   state.post_norm,
                                   state.config,
                                   n,
                                   state.log_degree);
        host_validation_ran = true;
        gpu_ok = compare_vectors(host_expected, zsum, n, "host OLE oracle") && gpu_ok;
    }
    return gpu_ok;
}

static void init_ole_state(OleState &state,
                           const OleArgs &args,
                           const ModulusConfig<Word> &config,
                           int limb_index) {
    state.args = args;
    state.args.seed = mix_seed(args.seed, static_cast<uint64_t>(limb_index));
    state.log_degree = log2i(args.n);
    state.log_domain = log2i(spfss_domain_size(args));
    state.limb_index = limb_index;
    state.config = config;
    state.modulus = config.modulus;
    compute_cheddar_tables(state.host_tables, args.n, config);
    alloc_and_copy(state.tables, state.host_tables);
    compute_reference_vectors(state.phi_norm, state.post_norm, args.n, config);
    build_inputs(state);
}

static OleLimbResult run_initial_ole_limb(OleState &state, AESGlobalContext *gaes) {
    OleLimbResult result;
    result.keygen_us = build_spfss_keys(state, gaes);
    result.key_bytes = state.spfss_pair_key_bytes;

    run_x_phase(state);
    run_spfss_eval_phase(state, gaes);
    run_z_phase(state);
    check(cudaDeviceSynchronize(), "sync initial OLE run");
    result.correct = validate_outputs(state, result.host_validation_ran);
    return result;
}

static int run_benchmark(const OleArgs &args) {
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);

    std::vector<ModulusConfig<Word>> configs = ole_modulus_configs(args.qbits);
    std::vector<std::unique_ptr<OleState>> states;
    states.reserve(configs.size());
    bool correct = true;
    bool host_validation_ran = true;
    size_t key_bytes = 0;
    double keygen_us = 0.0;

    for (size_t limb = 0; limb < configs.size(); ++limb) {
        auto state = std::make_unique<OleState>();
        init_ole_state(*state, args, configs[limb], static_cast<int>(limb));
        OleLimbResult limb_result = run_initial_ole_limb(*state, &gaes);
        correct = limb_result.correct && correct;
        host_validation_ran = limb_result.host_validation_ran && host_validation_ran;
        key_bytes += limb_result.key_bytes;
        keygen_us += limb_result.keygen_us;
        states.push_back(std::move(state));
    }

    for (int iter = 0; iter < args.warmup; ++iter) {
        for (auto &state : states) {
            run_x_phase(*state);
            run_spfss_eval_phase(*state, &gaes);
            run_z_phase(*state);
        }
        check(cudaDeviceSynchronize(), "sync OLE warmup");
    }

    std::vector<double> samples;
    samples.reserve(args.iters);
    for (int iter = 0; iter < args.iters; ++iter) {
        auto start = Clock::now();
        for (auto &state : states) {
            run_x_phase(*state);
            run_spfss_eval_phase(*state, &gaes);
            run_z_phase(*state);
        }
        check(cudaDeviceSynchronize(), "sync OLE iter");
        auto end = Clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
    }
    SummaryStats ole = summarize(samples);

    const char *validation =
        args.skip_validation ? "skipped" : (correct ? "pass" : "fail");
    const char *host_validation =
        args.skip_validation ? "skipped" : (host_validation_ran ? "pass" : "skipped");

    std::cout << RINGLPN_DEVICE_LABEL << ",figure2_spfss_" << args.noise << ","
              << args.n << "," << log2i(args.n) << "," << log2i(spfss_domain_size(args)) << ","
              << args.qbits << "," << ole_actual_qbits(configs) << ","
              << args.noise << "," << spfss_domain_size(args) << ","
              << args.c << "," << args.t << "," << args.chunk_size << ","
              << args.iters << "," << validation << "," << host_validation << ","
              << key_bytes << "," << keygen_us << ","
              << ole.mean_us << "," << ole.stddev_us << ","
              << (args.skip_validation ? -1 : (correct ? 1 : 0)) << "\n";

    for (auto &state : states) {
        state->cleanup();
    }
    freeAESGlobalContext(&gaes);
    check(cudaDeviceSynchronize(), "sync cleanup");
    return (args.skip_validation || correct) ? 0 : 2;
}

}  // namespace

#ifndef RINGLPN_OLE_DISABLE_MAIN
int main(int argc, char **argv) {
    OleArgs args = parse_ole_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,n,logn,log_domain,requested_qbits,actual_qbits,noise_mode,spfss_domain,c,t,chunk_size,iters,validation,host_validation,spfss_pair_key_bytes,spfss_keygen_us,ole_expand_mean_us,ole_expand_std_us,correct\n";
    }
    return run_benchmark(args);
}
#endif
