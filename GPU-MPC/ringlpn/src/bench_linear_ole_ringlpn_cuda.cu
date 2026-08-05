#define RINGLPN_OLE_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_linear_ole"
#endif
#include "bench_ole_ringlpn_cuda.cu"

#include <algorithm>
#include <memory>
#include <limits>

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

struct LinearDeviceBytes {
    size_t aes = 0;
    size_t tables = 0;
    size_t public_ntt = 0;
    size_t streamed_input = 0;
    size_t ntt_work = 0;
    size_t x_outputs = 0;
    size_t spfss_stream = 0;
    size_t z_outputs = 0;
    size_t saved_product = 0;
    size_t matrix_outputs = 0;
    size_t transient = 0;
    size_t setup_peak = 0;
    size_t expand_peak = 0;
    size_t total_peak = 0;
    size_t required_after_aes = 0;
};

static size_t linear_checked_size(unsigned __int128 value, const char *label) {
    if (value > static_cast<unsigned __int128>(
                    std::numeric_limits<size_t>::max())) {
        std::cerr << "Ring-LPN GPU memory accounting overflow at " << label
                  << "\n";
        std::exit(1);
    }
    return static_cast<size_t>(value);
}

static LinearDeviceBytes linear_device_bytes(const LinearArgs &args) {
    using U128 = unsigned __int128;
    const U128 n = static_cast<unsigned>(args.n);
    const U128 word = sizeof(Word);
    const U128 out_polys =
        static_cast<unsigned>(args.rows) * static_cast<unsigned>(args.cols);
    const U128 domain = static_cast<unsigned>(
        args.noise == "regular" ? 2 * (args.n / args.t) : 2 * args.n);
    const int log_domain = log2i(static_cast<int>(domain));
    const U128 max_points =
        args.noise == "regular"
            ? static_cast<U128>(static_cast<unsigned>(args.t))
            : static_cast<U128>(static_cast<unsigned>(args.t)) *
                  static_cast<U128>(static_cast<unsigned>(args.t));

    LinearDeviceBytes bytes;
    bytes.aes = 5 * AES_128_TABLE_SIZE * sizeof(u32) + 256 * sizeof(u8);
    bytes.tables = linear_checked_size(
        (2 * n + 2 * (n / kLsbSize) + 5) * word, "NTT tables");
    bytes.public_ntt =
        linear_checked_size(static_cast<U128>(args.c) * n * word,
                            "public NTT cache");
    bytes.streamed_input = linear_checked_size(n * word, "streamed input");
    bytes.ntt_work = linear_checked_size(4 * n * word, "NTT work");
    bytes.x_outputs = linear_checked_size(2 * n * word, "x outputs");
    bytes.spfss_stream = linear_checked_size(
        (3 * n + (args.noise == "regular" ? domain : 0)) * word,
        "SPFSS stream");
    bytes.z_outputs = linear_checked_size(2 * n * word, "z outputs");
    bytes.saved_product =
        linear_checked_size(5 * n * word, "saved product");
    bytes.matrix_outputs =
        linear_checked_size(3 * out_polys * n * word, "matrix outputs");

    // gpuKeyGenDPFZpPair has two Word inputs, two seed arrays, one seed-CW
    // array, two bit-CW arrays, and one final-CW array live simultaneously.
    // Loaded keys skip keygen, whose transient is larger than evaluation.
    const U128 keygen_transient =
        max_points * static_cast<U128>(56 + 18 * log_domain);
    const U128 eval_transient =
        max_points * static_cast<U128>(24 + 18 * log_domain);
    bytes.transient = linear_checked_size(
        std::getenv("RINGLPN_OLE_SPFSS_KEYS") ? eval_transient
                                               : keygen_transient,
        "SPFSS transient");

    const U128 linear =
        static_cast<U128>(bytes.saved_product) + bytes.matrix_outputs;
    const U128 setup = static_cast<U128>(bytes.aes) + linear + bytes.tables +
                       2 * static_cast<U128>(bytes.public_ntt);
    const U128 expand =
        static_cast<U128>(bytes.aes) + linear + bytes.tables +
        bytes.public_ntt + bytes.streamed_input + bytes.ntt_work +
        bytes.x_outputs + bytes.spfss_stream + bytes.z_outputs +
        bytes.transient;
    bytes.setup_peak = linear_checked_size(setup, "setup peak");
    bytes.expand_peak = linear_checked_size(expand, "expand peak");
    bytes.total_peak = std::max(bytes.setup_peak, bytes.expand_peak);
    bytes.required_after_aes = bytes.total_peak - bytes.aes;
    return bytes;
}

static void linear_memory_preflight(const LinearDeviceBytes &bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes),
          "query Ring-LPN GPU memory");
    (void)total_bytes;
    if (bytes.required_after_aes > free_bytes) {
        std::cerr << "Ring-LPN GPU memory preflight failed: required_bytes="
                  << bytes.required_after_aes
                  << " available_bytes=" << free_bytes
                  << " total_peak_bytes=" << bytes.total_peak
                  << " setup_peak_bytes=" << bytes.setup_peak
                  << " expand_peak_bytes=" << bytes.expand_peak << "\n";
        std::exit(1);
    }
}

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
    std::unique_ptr<OleState> workspace;
    std::vector<Word> h_streamed_input;
    size_t spfss_pair_key_bytes = 0;
    double keygen_us = 0.0;
    bool shared_operand_check = false;

    Word *d_c0 = nullptr;
    Word *d_c1 = nullptr;
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
        if (workspace) {
            workspace->cleanup();
        }
        cudaFree(d_c0);
        cudaFree(d_c1);
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

    const uint64_t max_products = 64;
    const uint64_t ring_products =
        static_cast<uint64_t>(static_cast<unsigned>(args.rows)) *
        static_cast<uint64_t>(static_cast<unsigned>(args.inner)) *
        static_cast<uint64_t>(static_cast<unsigned>(args.cols));
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

static std::unique_ptr<OleState> make_linear_workspace(
    const LinearArgs &args,
    const ModulusConfig<Word> &config,
    const std::vector<std::vector<Word>> &a) {
    if (static_cast<int>(a.size()) != args.c) {
        std::cerr << "shared linear public input shape mismatch\n";
        std::exit(1);
    }
    auto state = std::make_unique<OleState>();
    state->args.n = args.n;
    state->args.c = args.c;
    state->args.t = args.t;
    state->args.qbits = args.qbits;
    state->args.iters = 1;
    state->args.warmup = 0;
    state->args.chunk_size = args.chunk_size;
    state->args.seed = args.seed;
    state->args.noise = args.noise;
    state->args.skip_validation = true;
    state->log_degree = log2i(args.n);
    state->log_domain = log2i(spfss_domain_size(state->args));
    state->config = config;
    state->modulus = config.modulus;
    compute_cheddar_tables(state->host_tables, args.n, config);
    alloc_and_copy(state->tables, state->host_tables);

    const size_t n = static_cast<size_t>(args.n);
    const size_t c_coeffs = static_cast<size_t>(args.c) * n;
    copy_to_device(&state->d_a, flatten_dense(a, args.n),
                   "copy shared public a");
    alloc_device(&state->d_a_ntt, c_coeffs, "alloc shared public a NTT");
    run_forward_only(state->d_a, state->d_a_ntt, state->tables, args.n,
                     args.c, state->log_degree);
    check(cudaDeviceSynchronize(), "sync shared public a NTT");
    cudaFree(state->d_a);
    state->d_a = nullptr;

    // All coefficient and c^2 product work is streamed through one polynomial.
    // d_aw holds NTT(a_i)*NTT(a_j), while d_bw/d_cw/d_terms are the prepared
    // polymul work/output buffers.
    alloc_device(&state->d_e0, n, "alloc streamed shared input");
    alloc_device(&state->d_aw, n, "alloc streamed shared work a");
    alloc_device(&state->d_bw, n, "alloc streamed shared work b");
    alloc_device(&state->d_cw, n, "alloc streamed shared work c");
    alloc_device(&state->d_terms, n, "alloc streamed shared term");
    alloc_device(&state->d_x0, n, "alloc shared x0");
    alloc_device(&state->d_x1, n, "alloc shared x1");
    alloc_device(&state->d_u0, n, "alloc streamed shared u");
    alloc_device(&state->d_u2n0, 2 * n, "alloc streamed shared u2n");
    if (use_regular_noise(state->args)) {
        alloc_device(&state->d_group0, spfss_domain_size(state->args),
                     "alloc streamed regular group");
    }
    alloc_device(&state->d_z0, n, "alloc shared z0");
    alloc_device(&state->d_z1, n, "alloc shared z1");
    return state;
}

static std::unique_ptr<OleState> make_ole_state(const LinearArgs &args,
                                                const ModulusConfig<Word> &config,
                                                uint64_t seed,
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
    state->e.assign(2, std::vector<SparsePoly>(args.c));
    state->e[0] = e0;
    state->e[1] = e1;
    state->noise_binding[0] = ringlpn_keyio::spfss_groups::noise_binding(
        party_noise_record(*state, 0));
    state->noise_binding[1] = ringlpn_keyio::spfss_groups::noise_binding(
        party_noise_record(*state, 1));
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
    alloc_device(&state.d_expected, out_coeffs, "alloc linear expected");
    alloc_device(&state.d_tmp_a, state.args.n, "alloc linear saved x0");
    alloc_device(&state.d_tmp_b, state.args.n, "alloc linear saved x1");
    alloc_device(&state.d_local0, state.args.n, "alloc linear saved z0");
    alloc_device(&state.d_local1, state.args.n, "alloc linear saved z1");
    alloc_device(&state.d_expected_term, state.args.n,
                 "alloc linear streamed product");
    state.h_streamed_input.assign(static_cast<size_t>(state.args.n), 0);
}

static void reset_linear_outputs(LinearRunState &state, bool reset_expected) {
    const size_t out_bytes =
        static_cast<size_t>(state.args.rows) * state.args.cols * state.args.n * sizeof(Word);
    check(cudaMemset(state.d_c0, 0, out_bytes), "clear linear c0");
    check(cudaMemset(state.d_c1, 0, out_bytes), "clear linear c1");
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

static void run_streamed_x(LinearRunState &linear,
                           const OleState &source,
                           int party,
                           Word *d_out) {
    OleState &work = *linear.workspace;
    const int n = linear.args.n;
    const size_t n_bytes = static_cast<size_t>(n) * sizeof(Word);
    check(cudaMemset(d_out, 0, n_bytes), "clear streamed x");
    for (int i = 0; i < linear.args.c; ++i) {
        std::fill(linear.h_streamed_input.begin(),
                  linear.h_streamed_input.end(), 0);
        add_sparse_to_dense(source.e[party][i], linear.h_streamed_input, n,
                            linear.config.modulus);
        check(cudaMemcpy(work.d_e0, linear.h_streamed_input.data(), n_bytes,
                         cudaMemcpyHostToDevice),
              "copy streamed sparse polynomial");
        run_polymul_prepared_lhs(
            work.d_a_ntt + static_cast<size_t>(i) * n, work.d_e0, work.d_bw,
            work.d_cw, work.d_terms, work.tables, n, 1, work.log_degree);
        accumulate_one(d_out, work.d_terms, n, linear.config.modulus);
    }
}

static void eval_streamed_spfss_index(
    const ringlpn_ole_party::RingOlePublicParams &params,
    const std::vector<ringlpn_spfss_zp::GPUDPFZpKey> &keys,
    int matrix_idx,
    OleState &work,
    Word *d_z,
    AESGlobalContext *gaes) {
    if (params.regular) {
        const int groups = ringlpn_ole_party::group_count(params);
        const int bucket = ringlpn_ole_party::bucket_size(params);
        const int group_domain = ringlpn_ole_party::domain_size(params);
        check(cudaMemset(work.d_u2n0, 0,
                         static_cast<size_t>(2) * params.n * sizeof(Word)),
              "zero streamed regular u2n");
        for (int group = 0; group < groups; ++group) {
            const size_t key_idx =
                static_cast<size_t>(matrix_idx) * groups + group;
            ringlpn_spfss_zp::gpuDpfZpFullEvalSum(keys[key_idx],
                                                  work.d_group0, gaes);
            dim3 block(256);
            dim3 grid(grid_size(static_cast<size_t>(group_domain), block.x));
            ringlpn_ole_party::party_scatter_regular_group_kernel<<<grid, block>>>(
                work.d_group0, work.d_u2n0, group_domain, group * bucket,
                params.modulus);
            check(cudaGetLastError(), "launch streamed regular scatter");
        }
    } else {
        ringlpn_spfss_zp::gpuDpfZpFullEvalSum(
            keys[static_cast<size_t>(matrix_idx)], work.d_u2n0, gaes);
    }
    ringlpn_ole_party::fold_2n_to_n(work.d_u2n0, work.d_u0, params.n,
                                    params.modulus);
    run_polymul_prepared_lhs(work.d_aw, work.d_u0, work.d_bw, work.d_cw,
                             work.d_terms, work.tables, params.n, 1,
                             work.log_degree);
    accumulate_one(d_z, work.d_terms, params.n, params.modulus);
}

static void run_streamed_spfss_z(LinearRunState &linear,
                                 const OleState &source,
                                 AESGlobalContext *gaes) {
    OleState &work = *linear.workspace;
    const int n = linear.args.n;
    const size_t n_bytes = static_cast<size_t>(n) * sizeof(Word);
    check(cudaMemset(work.d_z0, 0, n_bytes), "clear streamed z0");
    check(cudaMemset(work.d_z1, 0, n_bytes), "clear streamed z1");
    const auto params = party_public_params(source);
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    for (int j = 0; j < linear.args.c; ++j) {
        for (int i = 0; i < linear.args.c; ++i) {
            const int matrix_idx = i + j * linear.args.c;
            pointwise_mul_kernel<Word><<<grid, block>>>(
                work.d_a_ntt + static_cast<size_t>(i) * n,
                work.d_a_ntt + static_cast<size_t>(j) * n, work.d_aw,
                static_cast<size_t>(n), n, 1, work.tables.d_primes,
                work.tables.d_inv_primes);
            check(cudaGetLastError(), "launch streamed public product");
            eval_streamed_spfss_index(params, source.keys0, matrix_idx, work,
                                      work.d_z0, gaes);
            eval_streamed_spfss_index(params, source.keys1, matrix_idx, work,
                                      work.d_z1, gaes);
        }
    }
}

static void run_streamed_ole(LinearRunState &linear,
                             const OleState &source,
                             AESGlobalContext *gaes) {
    OleState &work = *linear.workspace;
    run_streamed_x(linear, source, 0, work.d_x0);
    run_streamed_x(linear, source, 1, work.d_x1);
    run_streamed_spfss_z(linear, source, gaes);
}

static void run_one_linear_product(LinearRunState &linear,
                                   LinearProduct &product,
                                   AESGlobalContext *gaes,
                                   bool build_expected) {
    OleState &work = *linear.workspace;
    const int n = linear.args.n;
    const size_t n_bytes = static_cast<size_t>(n) * sizeof(Word);
    const Word modulus = linear.config.modulus;
    Word *c0_slot = linear.d_c0 + static_cast<size_t>(product.out_slot) * n;
    Word *c1_slot = linear.d_c1 + static_cast<size_t>(product.out_slot) * n;
    Word *expected_slot =
        linear.d_expected + static_cast<size_t>(product.out_slot) * n;

    run_streamed_ole(linear, *product.a0_b1, gaes);
    check(cudaMemcpy(linear.d_tmp_a, work.d_x0, n_bytes,
                     cudaMemcpyDeviceToDevice), "save streamed x0");
    check(cudaMemcpy(linear.d_tmp_b, work.d_x1, n_bytes,
                     cudaMemcpyDeviceToDevice), "save streamed x1");
    check(cudaMemcpy(linear.d_local0, work.d_z0, n_bytes,
                     cudaMemcpyDeviceToDevice), "save streamed z0");
    check(cudaMemcpy(linear.d_local1, work.d_z1, n_bytes,
                     cudaMemcpyDeviceToDevice), "save streamed z1");

    run_streamed_ole(linear, *product.a1_b0, gaes);
    run_full_polymul(linear.d_tmp_a, work.d_x1, work.d_aw, work.d_bw,
                     work.d_cw, linear.d_expected_term, work.tables, n, 1,
                     work.log_degree);
    accumulate_three(c0_slot, linear.d_expected_term, linear.d_local0,
                     work.d_z0, n, modulus);
    run_full_polymul(work.d_x0, linear.d_tmp_b, work.d_aw, work.d_bw,
                     work.d_cw, linear.d_expected_term, work.tables, n, 1,
                     work.log_degree);
    accumulate_three(c1_slot, linear.d_expected_term, linear.d_local1,
                     work.d_z1, n, modulus);

    if (build_expected) {
        add_pair(linear.d_tmp_a, work.d_x0, linear.d_local0, n, modulus);
        add_pair(linear.d_tmp_b, work.d_x1, linear.d_local1, n, modulus);
        run_full_polymul(linear.d_local0, linear.d_local1, work.d_aw,
                         work.d_bw, work.d_cw, linear.d_expected_term,
                         work.tables, n, 1, work.log_degree);
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
        linear.d_c0, linear.d_c1, linear.d_c0, out_coeffs,
        linear.config.modulus);
    check(cudaGetLastError(), "launch linear_add_matrix_kernel");
    check(cudaDeviceSynchronize(), "sync linear validation");

    std::vector<Word> csum(out_coeffs);
    std::vector<Word> expected(out_coeffs);
    check(cudaMemcpy(csum.data(), linear.d_c0, out_coeffs * sizeof(Word),
                     cudaMemcpyDeviceToHost), "copy linear csum");
    check(cudaMemcpy(expected.data(), linear.d_expected, out_coeffs * sizeof(Word), cudaMemcpyDeviceToHost),
          "copy linear expected");
    return compare_vectors(expected, csum, linear.args.n, "linear OLE Beaver matrix");
}

static void build_linear_products(LinearRunState &linear, AESGlobalContext *gaes) {
    const LinearArgs &args = linear.args;
    linear.products.reserve(static_cast<size_t>(args.rows) * args.inner * args.cols);
    LinearSharedInputs shared = build_shared_inputs(args, linear.config);
    linear.workspace =
        make_linear_workspace(args, linear.config, shared.a);
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
                    a_entry.p0, b_entry.p1, gaes, linear.keygen_us,
                    linear.spfss_pair_key_bytes);
                product.a1_b0 = make_ole_state(
                    args, linear.config, linear_mix_seed(args.seed, tag++),
                    a_entry.p1, b_entry.p0, gaes, linear.keygen_us,
                    linear.spfss_pair_key_bytes);
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
    // This executable uses legacy cudaMalloc for its large explicit buffers.
    // Do not pre-reserve 25 GiB in the cudaMallocAsync default pool: that
    // reservation is unavailable to these allocations on the target driver.
    const LinearDeviceBytes memory = linear_device_bytes(args);
    AESGlobalContext gaes;
    initAESContext(&gaes);
    linear_memory_preflight(memory);
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
              << (args.skip_validation ? -1 : (correct ? 1 : 0)) << ","
              << memory.required_after_aes << "," << memory.total_peak << "\n";

    freeAESGlobalContext(&gaes);
    check(cudaDeviceSynchronize(), "sync linear cleanup");
    return (args.skip_validation || correct) ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
    LinearArgs args = parse_linear_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,n,logn,log_domain,requested_qbits,actual_qbits,noise_mode,spfss_domain,rows,inner,cols,c,t,chunk_size,ring_products,ole_instances,iters,validation,spfss_pair_key_bytes,spfss_keygen_us,linear_expand_mean_us,linear_expand_std_us,shared_operands,correct,gpu_required_after_aes_bytes,gpu_static_peak_bytes\n";
    }
    return run_linear_benchmark(args);
}
