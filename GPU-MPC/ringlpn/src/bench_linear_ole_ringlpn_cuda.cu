#define RINGLPN_OLE_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_linear_ole"
#endif
#include "bench_ole_ringlpn_cuda.cu"

#include <memory>

namespace {

struct LinearArgs {
    int n = kMinDegree;
    int rows = 2;
    int inner = 2;
    int cols = 2;
    int c = 2;
    int t = 8;
    int qbits = 64;
    int iters = 1;
    int warmup = 0;
    int chunk_size = 8192;
    uint64_t seed = 1;
    bool csv_header = false;
    bool skip_validation = false;
};

struct LinearProduct {
    int out_slot = 0;
    std::unique_ptr<OleState> a0_b1;
    std::unique_ptr<OleState> a1_b0;
};

struct LinearRunState {
    LinearArgs args;
    std::vector<LinearProduct> products;
    size_t spfss_pair_key_bytes = 0;
    double keygen_us = 0.0;

    Word *d_c0 = nullptr;
    Word *d_c1 = nullptr;
    Word *d_csum = nullptr;
    Word *d_expected = nullptr;
    Word *d_tmp_a = nullptr;
    Word *d_tmp_b = nullptr;
    Word *d_local0 = nullptr;
    Word *d_local1 = nullptr;
    Word *d_expected_term = nullptr;

    void cleanup() {
        for (auto &p : products) {
            if (p.a0_b1) {
                p.a0_b1->cleanup();
            }
            if (p.a1_b0) {
                p.a1_b0->cleanup();
            }
        }
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFree(d_csum);
        cudaFree(d_expected);
        cudaFree(d_tmp_a);
        cudaFree(d_tmp_b);
        cudaFree(d_local0);
        cudaFree(d_local1);
        cudaFree(d_expected_term);
    }
};

__device__ __forceinline__ Word linear_mod_add_device(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? s - modulus : s;
}

__global__ void linear_add_pair_kernel(const Word *a,
                                       const Word *b,
                                       Word *out,
                                       int n,
                                       Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        out[idx] = linear_mod_add_device(a[idx], b[idx], modulus);
    }
}

__global__ void linear_add_matrix_kernel(const Word *a,
                                         const Word *b,
                                         Word *out,
                                         size_t count,
                                         Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = linear_mod_add_device(a[idx], b[idx], modulus);
    }
}

__global__ void linear_accumulate_one_kernel(Word *slot,
                                             const Word *a,
                                             int n,
                                             Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        slot[idx] = linear_mod_add_device(slot[idx], a[idx], modulus);
    }
}

__global__ void linear_accumulate_three_kernel(Word *slot,
                                               const Word *a,
                                               const Word *b,
                                               const Word *c,
                                               int n,
                                               Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        Word acc = linear_mod_add_device(slot[idx], a[idx], modulus);
        acc = linear_mod_add_device(acc, b[idx], modulus);
        slot[idx] = linear_mod_add_device(acc, c[idx], modulus);
    }
}

static void linear_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --n <deg> [--qbits 64] [--rows M] [--inner K] [--cols N]"
              << " [--c N] [--t N] [--seed N] [--iters N] [--warmup N]"
              << " [--chunk-size N] [--csv-header] [--skip-validation]\n";
}

static LinearArgs parse_linear_args(int argc, char **argv) {
    LinearArgs args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--qbits") && i + 1 < argc) {
            args.qbits = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) {
            args.rows = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--inner") && i + 1 < argc) {
            args.inner = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--cols") && i + 1 < argc) {
            args.cols = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--c") && i + 1 < argc) {
            args.c = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--t") && i + 1 < argc) {
            args.t = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
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
            linear_usage(argv[0]);
            std::exit(1);
        }
    }

    const int max_products = 64;
    int ring_products = args.rows * args.inner * args.cols;
    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree ||
        args.qbits != 64 || args.rows <= 0 || args.inner <= 0 || args.cols <= 0 ||
        args.c <= 0 || args.t <= 0 || args.t > args.n || args.iters <= 0 ||
        args.warmup < 0 || args.chunk_size <= 0 || ring_products <= 0 ||
        ring_products > max_products) {
        linear_usage(argv[0]);
        std::exit(1);
    }
    if (static_cast<uint64_t>(args.c) * static_cast<uint64_t>(args.c) > 65535ULL) {
        std::cerr << "Unsupported c: c*c must fit CUDA grid.y limit\n";
        std::exit(1);
    }
    return args;
}

static uint64_t linear_mix_seed(uint64_t seed, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL + (tag << 6) + (tag >> 2);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static std::unique_ptr<OleState> make_ole_state(const LinearArgs &args,
                                                uint64_t seed,
                                                AESGlobalContext *gaes,
                                                double &keygen_us,
                                                size_t &key_bytes) {
    auto state = std::make_unique<OleState>();
    state->args.n = args.n;
    state->args.c = args.c;
    state->args.t = args.t;
    state->args.qbits = args.qbits;
    state->args.iters = 1;
    state->args.warmup = 0;
    state->args.chunk_size = args.chunk_size;
    state->args.seed = seed;
    state->args.skip_validation = true;
    state->log_degree = log2i(args.n);
    state->log_domain = log2i(2 * args.n);
    compute_cheddar_tables(state->host_tables, args.n, kConfig62);
    alloc_and_copy(state->tables, state->host_tables);
    compute_reference_vectors(state->phi_norm, state->post_norm, args.n, kConfig62);
    build_inputs(*state);
    double us = build_spfss_keys(*state, gaes);
    keygen_us += us;
    key_bytes += state->spfss_pair_key_bytes;
    return state;
}

static void alloc_linear_buffers(LinearRunState &state) {
    const size_t out_coeffs =
        static_cast<size_t>(state.args.rows) * state.args.cols * state.args.n;
    alloc_device(&state.d_c0, out_coeffs, "alloc linear c0");
    alloc_device(&state.d_c1, out_coeffs, "alloc linear c1");
    alloc_device(&state.d_csum, out_coeffs, "alloc linear csum");
    alloc_device(&state.d_expected, out_coeffs, "alloc linear expected");
    alloc_device(&state.d_tmp_a, state.args.n, "alloc linear tmp a");
    alloc_device(&state.d_tmp_b, state.args.n, "alloc linear tmp b");
    alloc_device(&state.d_local0, state.args.n, "alloc linear local0");
    alloc_device(&state.d_local1, state.args.n, "alloc linear local1");
    alloc_device(&state.d_expected_term, state.args.n, "alloc linear expected term");
}

static void reset_linear_outputs(LinearRunState &state, bool reset_expected) {
    const size_t out_bytes =
        static_cast<size_t>(state.args.rows) * state.args.cols * state.args.n * sizeof(Word);
    check(cudaMemset(state.d_c0, 0, out_bytes), "clear linear c0");
    check(cudaMemset(state.d_c1, 0, out_bytes), "clear linear c1");
    check(cudaMemset(state.d_csum, 0, out_bytes), "clear linear csum");
    if (reset_expected) {
        check(cudaMemset(state.d_expected, 0, out_bytes), "clear linear expected");
    }
}

static void add_pair(const Word *a, const Word *b, Word *out, int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    linear_add_pair_kernel<<<grid, block>>>(a, b, out, n, modulus);
    check(cudaGetLastError(), "launch linear_add_pair_kernel");
}

static void accumulate_one(Word *slot, const Word *a, int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    linear_accumulate_one_kernel<<<grid, block>>>(slot, a, n, modulus);
    check(cudaGetLastError(), "launch linear_accumulate_one_kernel");
}

static void accumulate_three(Word *slot,
                             const Word *a,
                             const Word *b,
                             const Word *c,
                             int n,
                             Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    linear_accumulate_three_kernel<<<grid, block>>>(slot, a, b, c, n, modulus);
    check(cudaGetLastError(), "launch linear_accumulate_three_kernel");
}

static void run_one_linear_product(LinearRunState &linear,
                                   LinearProduct &product,
                                   AESGlobalContext *gaes,
                                   bool build_expected) {
    OleState &a0_b1 = *product.a0_b1;
    OleState &a1_b0 = *product.a1_b0;
    const int n = linear.args.n;
    const Word modulus = kConfig62.modulus;
    Word *c0_slot = linear.d_c0 + static_cast<size_t>(product.out_slot) * n;
    Word *c1_slot = linear.d_c1 + static_cast<size_t>(product.out_slot) * n;
    Word *expected_slot = linear.d_expected + static_cast<size_t>(product.out_slot) * n;

    run_x_phase(a0_b1);
    run_spfss_eval_phase(a0_b1, gaes);
    run_z_phase(a0_b1);

    run_x_phase(a1_b0);
    run_spfss_eval_phase(a1_b0, gaes);
    run_z_phase(a1_b0);

    run_full_polymul(a0_b1.d_x0,
                     a1_b0.d_x1,
                     a0_b1.d_aw,
                     a0_b1.d_bw,
                     a0_b1.d_cw,
                     linear.d_local0,
                     a0_b1.tables,
                     n,
                     1,
                     a0_b1.log_degree);
    run_full_polymul(a1_b0.d_x0,
                     a0_b1.d_x1,
                     a0_b1.d_aw,
                     a0_b1.d_bw,
                     a0_b1.d_cw,
                     linear.d_local1,
                     a0_b1.tables,
                     n,
                     1,
                     a0_b1.log_degree);

    accumulate_three(c0_slot, linear.d_local0, a0_b1.d_z0, a1_b0.d_z0, n, modulus);
    accumulate_three(c1_slot, linear.d_local1, a0_b1.d_z1, a1_b0.d_z1, n, modulus);

    if (build_expected) {
        add_pair(a0_b1.d_x0, a1_b0.d_x0, linear.d_tmp_a, n, modulus);
        add_pair(a1_b0.d_x1, a0_b1.d_x1, linear.d_tmp_b, n, modulus);
        run_full_polymul(linear.d_tmp_a,
                         linear.d_tmp_b,
                         a0_b1.d_aw,
                         a0_b1.d_bw,
                         a0_b1.d_cw,
                         linear.d_expected_term,
                         a0_b1.tables,
                         n,
                         1,
                         a0_b1.log_degree);
        accumulate_one(expected_slot, linear.d_expected_term, n, modulus);
    }
}

static void run_linear_expand(LinearRunState &linear,
                              AESGlobalContext *gaes,
                              bool build_expected) {
    reset_linear_outputs(linear, build_expected);
    for (auto &product : linear.products) {
        run_one_linear_product(linear, product, gaes, build_expected);
    }
    check(cudaDeviceSynchronize(), "sync linear expand");
}

static bool validate_linear_outputs(LinearRunState &linear) {
    if (linear.args.skip_validation) {
        return true;
    }
    const size_t out_coeffs =
        static_cast<size_t>(linear.args.rows) * linear.args.cols * linear.args.n;
    dim3 block(256);
    dim3 grid(grid_size(out_coeffs, block.x));
    linear_add_matrix_kernel<<<grid, block>>>(
        linear.d_c0, linear.d_c1, linear.d_csum, out_coeffs, kConfig62.modulus);
    check(cudaGetLastError(), "launch linear_add_matrix_kernel");
    check(cudaDeviceSynchronize(), "sync linear validation");

    std::vector<Word> csum(out_coeffs);
    std::vector<Word> expected(out_coeffs);
    check(cudaMemcpy(csum.data(), linear.d_csum, out_coeffs * sizeof(Word), cudaMemcpyDeviceToHost),
          "copy linear csum");
    check(cudaMemcpy(expected.data(), linear.d_expected, out_coeffs * sizeof(Word), cudaMemcpyDeviceToHost),
          "copy linear expected");
    return compare_vectors(expected, csum, linear.args.n, "linear OLE Beaver matrix");
}

static void build_linear_products(LinearRunState &linear, AESGlobalContext *gaes) {
    const LinearArgs &args = linear.args;
    linear.products.reserve(static_cast<size_t>(args.rows) * args.inner * args.cols);
    uint64_t tag = 0;
    for (int r = 0; r < args.rows; ++r) {
        for (int k = 0; k < args.inner; ++k) {
            for (int col = 0; col < args.cols; ++col) {
                LinearProduct product;
                product.out_slot = r * args.cols + col;
                product.a0_b1 = make_ole_state(
                    args, linear_mix_seed(args.seed, tag++), gaes,
                    linear.keygen_us, linear.spfss_pair_key_bytes);
                product.a1_b0 = make_ole_state(
                    args, linear_mix_seed(args.seed, tag++), gaes,
                    linear.keygen_us, linear.spfss_pair_key_bytes);
                linear.products.push_back(std::move(product));
            }
        }
    }
}

static int run_linear_benchmark(const LinearArgs &args) {
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);

    LinearRunState linear;
    linear.args = args;
    alloc_linear_buffers(linear);
    build_linear_products(linear, &gaes);

    run_linear_expand(linear, &gaes, !args.skip_validation);
    bool correct = validate_linear_outputs(linear);

    for (int iter = 0; iter < args.warmup; ++iter) {
        run_linear_expand(linear, &gaes, false);
    }

    std::vector<double> samples;
    samples.reserve(args.iters);
    for (int iter = 0; iter < args.iters; ++iter) {
        auto start = Clock::now();
        run_linear_expand(linear, &gaes, false);
        auto end = Clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
    }
    SummaryStats stats = summarize(samples);

    const int ring_products = args.rows * args.inner * args.cols;
    const int ole_instances = 2 * ring_products;
    const char *validation =
        args.skip_validation ? "skipped" : (correct ? "pass" : "fail");
    std::cout << RINGLPN_DEVICE_LABEL << ",ring_beaver_two_ole_uniform,"
              << args.n << "," << log2i(args.n) << "," << log2i(2 * args.n) << ","
              << args.qbits << "," << kConfig62.actual_qbits << ","
              << args.rows << "," << args.inner << "," << args.cols << ","
              << args.c << "," << args.t << "," << args.chunk_size << ","
              << ring_products << "," << ole_instances << "," << args.iters << ","
              << validation << "," << linear.spfss_pair_key_bytes << ","
              << linear.keygen_us << "," << stats.mean_us << "," << stats.stddev_us << ","
              << (args.skip_validation ? -1 : (correct ? 1 : 0)) << "\n";

    linear.cleanup();
    freeAESGlobalContext(&gaes);
    check(cudaDeviceSynchronize(), "sync linear cleanup");
    return (args.skip_validation || correct) ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
    LinearArgs args = parse_linear_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,n,logn,log_domain,requested_qbits,actual_qbits,rows,inner,cols,c,t,chunk_size,ring_products,ole_instances,iters,validation,spfss_pair_key_bytes,spfss_keygen_us,linear_expand_mean_us,linear_expand_std_us,correct\n";
    }
    return run_linear_benchmark(args);
}
