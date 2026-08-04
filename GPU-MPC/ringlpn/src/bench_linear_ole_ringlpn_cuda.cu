#define RINGLPN_OLE_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_linear_ole"
#endif
#include "bench_ole_ringlpn_cuda.cu"

#include <algorithm>
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
    std::string noise = "uniform";
    bool csv_header = false;
    bool skip_validation = false;
};

struct LinearProduct {
    int row = 0;
    int inner_idx = 0;
    int col = 0;
    int out_slot = 0;
    std::unique_ptr<OleState> a0_b1;
    std::unique_ptr<OleState> a1_b0;
};

struct LinearOperandShare {
    std::vector<SparsePoly> p0;
    std::vector<SparsePoly> p1;
};

struct LinearSharedInputs {
    std::vector<std::vector<Word>> a;
    std::vector<LinearOperandShare> a_entries;
    std::vector<LinearOperandShare> b_entries;
};

struct LinearRunState {
    LinearArgs args;
    ModulusConfig<Word> config = kConfig62;
    std::vector<LinearProduct> products;
    size_t spfss_pair_key_bytes = 0;
    double keygen_us = 0.0;
    bool shared_operand_check = false;

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

struct LinearLimbResult {
    bool correct = false;
    bool shared_operand_check = false;
    size_t key_bytes = 0;
    double keygen_us = 0.0;
    SummaryStats expand;
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
              << " --n <deg> [--qbits 64|128] [--rows M] [--inner K] [--cols N]"
              << " [--c N] [--t N] [--seed N] [--iters N] [--warmup N]"
              << " [--chunk-size N] [--noise uniform|regular]"
              << " [--csv-header] [--skip-validation]\n";
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
            linear_usage(argv[0]);
            std::exit(1);
        }
    }

    const int max_products = 64;
    int ring_products = args.rows * args.inner * args.cols;
    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree ||
        (args.qbits != 64 && args.qbits != 128) ||
        args.rows <= 0 || args.inner <= 0 || args.cols <= 0 ||
        args.c <= 0 || args.t <= 0 || args.t > args.n || args.iters <= 0 ||
        args.warmup < 0 || args.chunk_size <= 0 || ring_products <= 0 ||
        ring_products > max_products) {
        linear_usage(argv[0]);
        std::exit(1);
    }
    if (args.noise != "uniform" && args.noise != "regular") {
        linear_usage(argv[0]);
        std::exit(1);
    }
    if (args.noise == "regular" &&
        (args.n % args.t != 0 || !is_power_of_two(args.t))) {
        std::cerr << "Regular noise requires power-of-two t dividing n for the first linear GPU artifact\n";
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

static int linear_spfss_domain_size(const LinearArgs &args) {
    return args.noise == "regular" ? 2 * (args.n / args.t) : 2 * args.n;
}

static std::vector<Word> sample_dense(int n,
                                      Word modulus,
                                      std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, modulus - 1);
    std::vector<Word> out(static_cast<size_t>(n));
    for (Word &coefficient : out) {
        coefficient = dist(rng);
    }
    return out;
}


static std::vector<SparsePoly> sample_operand_share(const LinearArgs &args,
                                                    Word modulus,
                                                    std::mt19937_64 &rng) {
    std::vector<SparsePoly> out(args.c);
    for (int i = 0; i < args.c; ++i) {
        out[i] = args.noise == "regular"
                     ? sample_sparse_regular(args.n, args.t, modulus, rng)
                     : sample_sparse(args.n, args.t, modulus, rng);
    }
    return out;
}

static LinearSharedInputs build_shared_inputs(const LinearArgs &args,
                                              const ModulusConfig<Word> &config) {
    LinearSharedInputs shared;
    std::mt19937_64 rng(args.seed);
    const Word modulus = config.modulus;

    shared.a.resize(args.c);
    shared.a[0].assign(args.n, 0);
    shared.a[0][0] = 1;
    for (int i = 1; i < args.c; ++i) {
        shared.a[i] = sample_dense(args.n, modulus, rng);
    }

    shared.a_entries.resize(static_cast<size_t>(args.rows) * args.inner);
    for (auto &entry : shared.a_entries) {
        entry.p0 = sample_operand_share(args, modulus, rng);
        entry.p1 = sample_operand_share(args, modulus, rng);
    }

    shared.b_entries.resize(static_cast<size_t>(args.inner) * args.cols);
    for (auto &entry : shared.b_entries) {
        entry.p0 = sample_operand_share(args, modulus, rng);
        entry.p1 = sample_operand_share(args, modulus, rng);
    }
    return shared;
}

static bool same_sparse_poly(const SparsePoly &a, const SparsePoly &b) {
    return a.positions == b.positions && a.values == b.values;
}

static bool same_sparse_vec(const std::vector<SparsePoly> &a,
                            const std::vector<SparsePoly> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!same_sparse_poly(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

static void build_inputs_from_shared(OleState &state,
                                     const std::vector<std::vector<Word>> &a,
                                     const std::vector<SparsePoly> &e0,
                                     const std::vector<SparsePoly> &e1) {
    const int n = state.args.n;
    const int c = state.args.c;
    const Word modulus = state.modulus;
    if (static_cast<int>(a.size()) != c || static_cast<int>(e0.size()) != c ||
        static_cast<int>(e1.size()) != c) {
        std::cerr << "shared linear input shape mismatch\n";
        std::exit(1);
    }

    state.a = a;
    state.e.assign(2, std::vector<SparsePoly>(c));
    state.e[0] = e0;
    state.e[1] = e1;

    std::vector<std::vector<Word>> e0_dense(c, std::vector<Word>(n, 0));
    std::vector<std::vector<Word>> e1_dense(c, std::vector<Word>(n, 0));
    for (int i = 0; i < c; ++i) {
        add_sparse_to_dense(state.e[0][i], e0_dense[i], n, modulus);
        add_sparse_to_dense(state.e[1][i], e1_dense[i], n, modulus);
    }

    copy_to_device(&state.d_a, flatten_dense(state.a, n), "copy shared a");
    copy_to_device(&state.d_e0, flatten_dense(e0_dense, n), "copy shared e0");
    copy_to_device(&state.d_e1, flatten_dense(e1_dense, n), "copy shared e1");

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
    copy_to_device(&state.d_aa_lhs, flatten_dense(aa_lhs, n), "copy shared aa lhs");
    copy_to_device(&state.d_aa_rhs, flatten_dense(aa_rhs, n), "copy shared aa rhs");

    size_t c_coeffs = static_cast<size_t>(c) * n;
    size_t cc_coeffs = static_cast<size_t>(cc) * n;
    alloc_device(&state.d_aw, std::max(c_coeffs, cc_coeffs), "alloc shared work a");
    alloc_device(&state.d_bw, std::max(c_coeffs, cc_coeffs), "alloc shared work b");
    alloc_device(&state.d_cw, std::max(c_coeffs, cc_coeffs), "alloc shared work c");
    alloc_device(&state.d_terms, cc_coeffs, "alloc shared terms");
    alloc_device(&state.d_aa, cc_coeffs, "alloc shared aa");
    alloc_device(&state.d_x0, n, "alloc shared x0");
    alloc_device(&state.d_x1, n, "alloc shared x1");
    alloc_device(&state.d_u0, cc_coeffs, "alloc shared u0");
    alloc_device(&state.d_u1, cc_coeffs, "alloc shared u1");
    alloc_device(&state.d_u2n0, static_cast<size_t>(2) * n, "alloc shared u2n0");
    alloc_device(&state.d_u2n1, static_cast<size_t>(2) * n, "alloc shared u2n1");
    if (use_regular_noise(state.args)) {
        alloc_device(&state.d_group0, spfss_domain_size(state.args), "alloc shared regular group0");
        alloc_device(&state.d_group1, spfss_domain_size(state.args), "alloc shared regular group1");
    }
    alloc_device(&state.d_z0, n, "alloc shared z0");
    alloc_device(&state.d_z1, n, "alloc shared z1");
    alloc_device(&state.d_zsum, n, "alloc shared zsum");
    alloc_device(&state.d_expected, n, "alloc shared expected");

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
    check(cudaDeviceSynchronize(), "sync shared aa precompute");
}

static std::unique_ptr<OleState> make_ole_state(const LinearArgs &args,
                                                const ModulusConfig<Word> &config,
                                                uint64_t seed,
                                                const std::vector<std::vector<Word>> &a,
                                                const std::vector<SparsePoly> &e0,
                                                const std::vector<SparsePoly> &e1,
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
    state->args.noise = args.noise;
    state->args.skip_validation = true;
    state->log_degree = log2i(args.n);
    state->log_domain = log2i(spfss_domain_size(state->args));
    state->config = config;
    state->modulus = config.modulus;
    compute_cheddar_tables(state->host_tables, args.n, config);
    alloc_and_copy(state->tables, state->host_tables);
    compute_reference_vectors(state->phi_norm, state->post_norm, args.n, config);
    build_inputs_from_shared(*state, a, e0, e1);
    double us = build_spfss_keys(*state, gaes);
    keygen_us += us;
    key_bytes += state->spfss_pair_key_bytes;
    return state;
}

static bool check_shared_operand_reuse(const LinearRunState &linear) {
    const LinearArgs &args = linear.args;
    std::vector<const std::vector<SparsePoly> *> a0_ref(
        static_cast<size_t>(args.rows) * args.inner, nullptr);
    std::vector<const std::vector<SparsePoly> *> a1_ref(
        static_cast<size_t>(args.rows) * args.inner, nullptr);
    std::vector<const std::vector<SparsePoly> *> b0_ref(
        static_cast<size_t>(args.inner) * args.cols, nullptr);
    std::vector<const std::vector<SparsePoly> *> b1_ref(
        static_cast<size_t>(args.inner) * args.cols, nullptr);

    for (const auto &product : linear.products) {
        const size_t a_idx =
            static_cast<size_t>(product.row) * args.inner + product.inner_idx;
        const size_t b_idx =
            static_cast<size_t>(product.inner_idx) * args.cols + product.col;
        const auto &a0 = product.a0_b1->e[0];
        const auto &a1 = product.a1_b0->e[0];
        const auto &b0 = product.a1_b0->e[1];
        const auto &b1 = product.a0_b1->e[1];

        if (!a0_ref[a_idx]) {
            a0_ref[a_idx] = &a0;
            a1_ref[a_idx] = &a1;
        } else if (!same_sparse_vec(*a0_ref[a_idx], a0) ||
                   !same_sparse_vec(*a1_ref[a_idx], a1)) {
            return false;
        }

        if (!b0_ref[b_idx]) {
            b0_ref[b_idx] = &b0;
            b1_ref[b_idx] = &b1;
        } else if (!same_sparse_vec(*b0_ref[b_idx], b0) ||
                   !same_sparse_vec(*b1_ref[b_idx], b1)) {
            return false;
        }
    }
    return std::all_of(a0_ref.begin(), a0_ref.end(), [](const auto *p) { return p != nullptr; }) &&
           std::all_of(a1_ref.begin(), a1_ref.end(), [](const auto *p) { return p != nullptr; }) &&
           std::all_of(b0_ref.begin(), b0_ref.end(), [](const auto *p) { return p != nullptr; }) &&
           std::all_of(b1_ref.begin(), b1_ref.end(), [](const auto *p) { return p != nullptr; });
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
    const Word modulus = linear.config.modulus;
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
        linear.d_c0, linear.d_c1, linear.d_csum, out_coeffs, linear.config.modulus);
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
    LinearSharedInputs shared = build_shared_inputs(args, linear.config);
    uint64_t tag = 0;
    for (int r = 0; r < args.rows; ++r) {
        for (int k = 0; k < args.inner; ++k) {
            for (int col = 0; col < args.cols; ++col) {
                const auto &a_entry = shared.a_entries[static_cast<size_t>(r) * args.inner + k];
                const auto &b_entry = shared.b_entries[static_cast<size_t>(k) * args.cols + col];
                LinearProduct product;
                product.row = r;
                product.inner_idx = k;
                product.col = col;
                product.out_slot = r * args.cols + col;
                product.a0_b1 = make_ole_state(
                    args, linear.config, linear_mix_seed(args.seed, tag++),
                    shared.a, a_entry.p0, b_entry.p1, gaes,
                    linear.keygen_us, linear.spfss_pair_key_bytes);
                product.a1_b0 = make_ole_state(
                    args, linear.config, linear_mix_seed(args.seed, tag++),
                    shared.a, a_entry.p1, b_entry.p0, gaes,
                    linear.keygen_us, linear.spfss_pair_key_bytes);
                linear.products.push_back(std::move(product));
            }
        }
    }
    linear.shared_operand_check = check_shared_operand_reuse(linear);
}

static LinearLimbResult run_linear_limb(const LinearArgs &args,
                                        const ModulusConfig<Word> &config,
                                        AESGlobalContext *gaes,
                                        int limb_index) {
    LinearLimbResult result;
    LinearRunState linear;
    linear.args = args;
    linear.args.seed = mix_seed(args.seed, static_cast<uint64_t>(limb_index));
    linear.config = config;
    alloc_linear_buffers(linear);
    build_linear_products(linear, gaes);

    run_linear_expand(linear, gaes, !args.skip_validation);
    result.correct = linear.shared_operand_check && validate_linear_outputs(linear);

    for (int iter = 0; iter < args.warmup; ++iter) {
        run_linear_expand(linear, gaes, false);
    }

    std::vector<double> samples;
    samples.reserve(args.iters);
    for (int iter = 0; iter < args.iters; ++iter) {
        auto start = Clock::now();
        run_linear_expand(linear, gaes, false);
        auto end = Clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
    }
    result.expand = summarize(samples);
    result.shared_operand_check = linear.shared_operand_check;
    result.key_bytes = linear.spfss_pair_key_bytes;
    result.keygen_us = linear.keygen_us;

    linear.cleanup();
    return result;
}

static int run_linear_benchmark(const LinearArgs &args) {
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);

    std::vector<ModulusConfig<Word>> configs = ole_modulus_configs(args.qbits);
    bool correct = true;
    bool shared_operand_check = true;
    size_t key_bytes = 0;
    double keygen_us = 0.0;
    SummaryStats stats;
    for (size_t limb = 0; limb < configs.size(); ++limb) {
        LinearLimbResult limb_result =
            run_linear_limb(args, configs[limb], &gaes, static_cast<int>(limb));
        correct = limb_result.correct && correct;
        shared_operand_check = limb_result.shared_operand_check && shared_operand_check;
        key_bytes += limb_result.key_bytes;
        keygen_us += limb_result.keygen_us;
        stats.mean_us += limb_result.expand.mean_us;
        stats.stddev_us += limb_result.expand.stddev_us;
    }

    const int ring_products = args.rows * args.inner * args.cols;
    const int ole_instances = 2 * ring_products;
    const char *validation =
        args.skip_validation ? "skipped" : (correct ? "pass" : "fail");
    std::cout << RINGLPN_DEVICE_LABEL << ",ring_beaver_two_ole_" << args.noise << ","
              << args.n << "," << log2i(args.n) << "," << log2i(linear_spfss_domain_size(args)) << ","
              << args.qbits << "," << ole_actual_qbits(configs) << ","
              << args.noise << "," << linear_spfss_domain_size(args) << ","
              << args.rows << "," << args.inner << "," << args.cols << ","
              << args.c << "," << args.t << "," << args.chunk_size << ","
              << ring_products << "," << ole_instances << "," << args.iters << ","
              << validation << "," << key_bytes << ","
              << keygen_us << "," << stats.mean_us << "," << stats.stddev_us << ","
              << (shared_operand_check ? 1 : 0) << ","
              << (args.skip_validation ? -1 : (correct ? 1 : 0)) << "\n";

    freeAESGlobalContext(&gaes);
    check(cudaDeviceSynchronize(), "sync linear cleanup");
    return (args.skip_validation || correct) ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
    LinearArgs args = parse_linear_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,n,logn,log_domain,requested_qbits,actual_qbits,noise_mode,spfss_domain,rows,inner,cols,c,t,chunk_size,ring_products,ole_instances,iters,validation,spfss_pair_key_bytes,spfss_keygen_us,linear_expand_mean_us,linear_expand_std_us,shared_operands,correct\n";
    }
    return run_linear_benchmark(args);
}
