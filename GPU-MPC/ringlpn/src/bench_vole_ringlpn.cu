#define RINGLPN_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_vole"
#endif

#include "bench_ntt_cuda_cheddar.cu"

#include <unordered_set>

namespace {

constexpr int kDefaultOutputs = 4;
constexpr int kDefaultLanes = 2;
constexpr int kDefaultNoiseWeight = 64;

struct VoleArgs {
    int n = kMinDegree;
    int requested_qbits = 32;
    int outputs = kDefaultOutputs;
    int lanes = kDefaultLanes;
    int noise_weight = kDefaultNoiseWeight;
    int iters = 100;
    int warmup = 10;
    uint64_t seed = 12345;
    bool csv_header = false;
    bool skip_validation = false;
};

static void vole_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --n <deg> [--qbits 32|64] [--m N] [--c N] [--noise-weight N]"
              << " [--iters N] [--warmup N] [--seed N] [--csv-header] [--skip-validation]\n";
}

static VoleArgs parse_vole_args(int argc, char **argv) {
    VoleArgs args;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--qbits") == 0 && i + 1 < argc) {
            args.requested_qbits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            args.outputs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--c") == 0 && i + 1 < argc) {
            args.lanes = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--noise-weight") == 0 && i + 1 < argc) {
            args.noise_weight = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--csv-header") == 0) {
            args.csv_header = true;
        } else if (std::strcmp(argv[i], "--skip-validation") == 0) {
            args.skip_validation = true;
        } else {
            vole_usage(argv[0]);
            std::exit(1);
        }
    }

    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree) {
        vole_usage(argv[0]);
        std::exit(1);
    }

    int log_degree = 0;
    int temp_n = args.n;
    while (temp_n > 1) {
        temp_n >>= 1;
        log_degree++;
    }
    if (log_degree < kMinLogDegree || log_degree > kMaxLogDegree) {
        std::cerr << "Unsupported degree: expected log2(n) in ["
                  << kMinLogDegree << ", " << kMaxLogDegree << "]\n";
        std::exit(1);
    }

    if (args.requested_qbits != 32 && args.requested_qbits != 64) {
        std::cerr << "Unsupported qbits request: expected one of 32 or 64\n";
        std::exit(1);
    }

    if (args.outputs <= 0 || args.lanes <= 0 || args.noise_weight < 0 || args.iters <= 0 ||
        args.warmup < 0) {
        vole_usage(argv[0]);
        std::exit(1);
    }

    if (static_cast<uint64_t>(args.outputs) * static_cast<uint64_t>(args.lanes) > 65535ULL) {
        std::cerr << "Unsupported m*c: pair batch must fit within the extracted kernel grid.y limit\n";
        std::exit(1);
    }

    if (args.noise_weight > args.n) {
        std::cerr << "Noise weight cannot exceed polynomial degree\n";
        std::exit(1);
    }

    return args;
}

template <typename Word>
static size_t estimate_required_device_bytes(const VoleArgs &args) {
    const size_t pair_coeffs = static_cast<size_t>(args.outputs) *
                               static_cast<size_t>(args.lanes) *
                               static_cast<size_t>(args.n);
    const size_t output_coeffs = static_cast<size_t>(args.outputs) * static_cast<size_t>(args.n);
    const size_t pair_arrays = 8;
    const size_t output_arrays = 7;
    return sizeof(Word) * (pair_arrays * pair_coeffs + output_arrays * output_coeffs);
}

static bool has_device_capacity(size_t required_bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    return required_bytes <= free_bytes;
}

template <typename Word>
static std::vector<Word> sample_uniform_polys(int poly_count,
                                              int n,
                                              Word modulus,
                                              uint64_t seed) {
    using Rng = std::conditional_t<(sizeof(Word) <= 4), std::mt19937, std::mt19937_64>;
    Rng rng(static_cast<typename Rng::result_type>(seed));
    std::uniform_int_distribution<Word> dist(0, static_cast<Word>(modulus - 1));

    std::vector<Word> out(static_cast<size_t>(poly_count) * static_cast<size_t>(n));
    for (Word &value : out) {
        value = dist(rng);
    }
    return out;
}

template <typename Word>
static std::vector<Word> sample_sparse_noise_polys(int poly_count,
                                                   int n,
                                                   Word modulus,
                                                   int weight,
                                                   uint64_t seed) {
    using Rng = std::conditional_t<(sizeof(Word) <= 4), std::mt19937, std::mt19937_64>;
    Rng rng(static_cast<typename Rng::result_type>(seed));
    std::uniform_int_distribution<int> pos_dist(0, n - 1);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    const int capped_weight = std::min(weight, n);
    std::vector<Word> out(static_cast<size_t>(poly_count) * static_cast<size_t>(n), 0);
    for (int poly_idx = 0; poly_idx < poly_count; poly_idx++) {
        std::unordered_set<int> used;
        used.reserve(static_cast<size_t>(capped_weight) * 2 + 1);
        while (static_cast<int>(used.size()) < capped_weight) {
            int pos = pos_dist(rng);
            if (!used.insert(pos).second) {
                continue;
            }
            out[static_cast<size_t>(poly_idx) * static_cast<size_t>(n) + static_cast<size_t>(pos)] =
                sign_dist(rng) == 0 ? static_cast<Word>(1) : static_cast<Word>(modulus - 1);
        }
    }
    return out;
}

template <typename Word>
static std::vector<Word> scalar_mul_add_batches(const std::vector<Word> &base,
                                                const std::vector<Word> &noise,
                                                Word scalar,
                                                Word modulus) {
    std::vector<Word> out(base.size(), 0);
    for (size_t idx = 0; idx < base.size(); idx++) {
        out[idx] = mod_add(base[idx], mod_mul_host(noise[idx], scalar, modulus), modulus);
    }
    return out;
}

template <typename Word>
static std::vector<Word> repeat_rhs_for_outputs(const std::vector<Word> &rhs,
                                                int outputs,
                                                int lanes,
                                                int n) {
    std::vector<Word> out(static_cast<size_t>(outputs) * static_cast<size_t>(lanes) *
                          static_cast<size_t>(n));
    const size_t block_size = static_cast<size_t>(lanes) * static_cast<size_t>(n);
    for (int output_idx = 0; output_idx < outputs; output_idx++) {
        std::copy(rhs.begin(), rhs.end(), out.begin() + static_cast<size_t>(output_idx) * block_size);
    }
    return out;
}

template <typename Word>
__device__ __forceinline__ Word mod_add_device(Word a, Word b, Word modulus) {
    return a >= modulus - b ? static_cast<Word>(a - (modulus - b)) : static_cast<Word>(a + b);
}

template <typename Word>
__global__ void reduce_poly_batches_kernel(const Word *pairwise,
                                           Word *reduced,
                                           int n,
                                           int lanes,
                                           Word modulus) {
    size_t coeff_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int output_idx = static_cast<int>(blockIdx.y);
    if (coeff_idx >= static_cast<size_t>(n)) {
        return;
    }

    size_t base = static_cast<size_t>(output_idx) * static_cast<size_t>(lanes) *
                  static_cast<size_t>(n) + coeff_idx;
    Word acc = 0;
    for (int lane_idx = 0; lane_idx < lanes; lane_idx++) {
        acc = mod_add_device(acc,
                             pairwise[base + static_cast<size_t>(lane_idx) * static_cast<size_t>(n)],
                             modulus);
    }
    reduced[static_cast<size_t>(output_idx) * static_cast<size_t>(n) + coeff_idx] = acc;
}

template <typename Word>
__global__ void add_poly_batches_kernel(const Word *lhs,
                                        const Word *rhs,
                                        Word *out,
                                        size_t total_coeffs,
                                        Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) {
        return;
    }
    out[idx] = mod_add_device(lhs[idx], rhs[idx], modulus);
}

template <typename Word>
static void run_inner_product_phase(Word *d_a_pairs,
                                    Word *d_rhs_pairs,
                                    Word *d_a_work,
                                    Word *d_b_work,
                                    Word *d_c_work,
                                    Word *d_pairwise_out,
                                    Word *d_reduced,
                                    Word *d_offset,
                                    Word *d_result,
                                    const DeviceTables<Word> &tables,
                                    int n,
                                    int outputs,
                                    int lanes,
                                    int log_degree,
                                    Word modulus) {
    const int pair_batch = outputs * lanes;
    run_full_polymul(d_a_pairs,
                     d_rhs_pairs,
                     d_a_work,
                     d_b_work,
                     d_c_work,
                     d_pairwise_out,
                     tables,
                     n,
                     pair_batch,
                     log_degree);

    dim3 block(256);
    dim3 reduce_grid(grid_size(static_cast<size_t>(n), block.x), static_cast<unsigned int>(outputs));
    reduce_poly_batches_kernel<Word><<<reduce_grid, block>>>(
        d_pairwise_out, d_reduced, n, lanes, modulus);
    check_launch("launch reduce_poly_batches_kernel");

    const size_t total_output_coeffs = static_cast<size_t>(outputs) * static_cast<size_t>(n);
    dim3 add_grid(grid_size(total_output_coeffs, block.x));
    add_poly_batches_kernel<Word><<<add_grid, block>>>(
        d_reduced, d_offset, d_result, total_output_coeffs, modulus);
    check_launch("launch add_poly_batches_kernel");
}

template <typename Word>
static std::vector<Word> host_expand_phase(const std::vector<Word> &a_pairs,
                                           const std::vector<Word> &rhs,
                                           const std::vector<Word> &offset,
                                           const std::vector<Word> &phi_norm,
                                           const std::vector<Word> &post_norm,
                                           const ModulusConfig<Word> &config,
                                           int outputs,
                                           int lanes,
                                           int n,
                                           int log_degree) {
    std::vector<Word> out(static_cast<size_t>(outputs) * static_cast<size_t>(n), 0);
    for (int output_idx = 0; output_idx < outputs; output_idx++) {
        std::vector<Word> acc(n, 0);
        for (int lane_idx = 0; lane_idx < lanes; lane_idx++) {
            const size_t pair_base =
                (static_cast<size_t>(output_idx) * static_cast<size_t>(lanes) +
                 static_cast<size_t>(lane_idx)) *
                static_cast<size_t>(n);
            const size_t rhs_base = static_cast<size_t>(lane_idx) * static_cast<size_t>(n);
            std::vector<Word> lhs_poly(a_pairs.begin() + pair_base,
                                       a_pairs.begin() + pair_base + static_cast<size_t>(n));
            std::vector<Word> rhs_poly(rhs.begin() + rhs_base,
                                       rhs.begin() + rhs_base + static_cast<size_t>(n));
            std::vector<Word> product = host_polymul_reference(
                lhs_poly, rhs_poly, phi_norm, post_norm, config, n, log_degree);
            for (int coeff_idx = 0; coeff_idx < n; coeff_idx++) {
                acc[coeff_idx] = mod_add(acc[coeff_idx], product[coeff_idx], config.modulus);
            }
        }

        const size_t out_base = static_cast<size_t>(output_idx) * static_cast<size_t>(n);
        for (int coeff_idx = 0; coeff_idx < n; coeff_idx++) {
            out[out_base + static_cast<size_t>(coeff_idx)] =
                mod_add(acc[coeff_idx], offset[out_base + static_cast<size_t>(coeff_idx)], config.modulus);
        }
    }
    return out;
}

template <typename Word>
static bool validate_vole_relation(const std::vector<Word> &x,
                                   const std::vector<Word> &y,
                                   const std::vector<Word> &z,
                                   Word delta,
                                   int n,
                                   Word modulus) {
    for (size_t idx = 0; idx < z.size(); idx++) {
        Word rhs = mod_add(y[idx], mod_mul_host(x[idx], delta, modulus), modulus);
        if (rhs != z[idx]) {
            std::cerr << "VOLE relation mismatch at output " << (idx / static_cast<size_t>(n))
                      << ", index " << (idx % static_cast<size_t>(n))
                      << ": expected " << rhs << ", got " << z[idx] << "\n";
            return false;
        }
    }
    return true;
}

template <typename Word>
static int run_vole_benchmark(const VoleArgs &args, const ModulusConfig<Word> &config) {
    if (((static_cast<WideWord<Word>>(config.modulus) - 1ULL) %
         (2ULL * static_cast<uint64_t>(args.n))) != 0ULL) {
        std::cerr << "Unsupported degree for selected prime: need 2n to divide q-1\n";
        return 1;
    }

    const size_t required_bytes = estimate_required_device_bytes<Word>(args);
    if (!has_device_capacity(required_bytes)) {
        std::cerr << "Insufficient GPU memory for requested configuration; need about "
                  << required_bytes << " bytes\n";
        return 1;
    }

    const int n = args.n;
    const int log_degree = int_log2(static_cast<uint32_t>(n));
    const int pair_batch = args.outputs * args.lanes;
    const size_t pair_coeffs = static_cast<size_t>(pair_batch) * static_cast<size_t>(n);
    const size_t output_coeffs = static_cast<size_t>(args.outputs) * static_cast<size_t>(n);

    HostTables<Word> host_tables;
    compute_cheddar_tables(host_tables, n, config);

    DeviceTables<Word> tables;
    alloc_and_copy(tables, host_tables);

    std::vector<Word> phi_norm;
    std::vector<Word> post_norm;
    compute_reference_vectors(phi_norm, post_norm, n, config);

    using Rng = std::conditional_t<(sizeof(Word) <= 4), std::mt19937, std::mt19937_64>;
    Rng delta_rng(static_cast<typename Rng::result_type>(args.seed + 1000));
    std::uniform_int_distribution<Word> delta_dist(1, static_cast<Word>(config.modulus - 1));
    Word delta = delta_dist(delta_rng);

    std::vector<Word> a_pairs = sample_uniform_polys(
        pair_batch, n, config.modulus, args.seed + 11);
    std::vector<Word> e = sample_sparse_noise_polys(
        args.lanes, n, config.modulus, args.noise_weight, args.seed + 21);
    std::vector<Word> vm = sample_uniform_polys(
        args.lanes, n, config.modulus, args.seed + 31);
    std::vector<Word> f = sample_sparse_noise_polys(
        args.outputs, n, config.modulus, args.noise_weight, args.seed + 41);
    std::vector<Word> vj = sample_uniform_polys(
        args.outputs, n, config.modulus, args.seed + 51);
    std::vector<Word> wm = scalar_mul_add_batches(vm, e, delta, config.modulus);
    std::vector<Word> wj = scalar_mul_add_batches(vj, f, delta, config.modulus);

    std::vector<Word> e_pairs = repeat_rhs_for_outputs(e, args.outputs, args.lanes, n);
    std::vector<Word> vm_pairs = repeat_rhs_for_outputs(vm, args.outputs, args.lanes, n);
    std::vector<Word> wm_pairs = repeat_rhs_for_outputs(wm, args.outputs, args.lanes, n);

    Word *d_a_pairs = nullptr;
    Word *d_e_pairs = nullptr;
    Word *d_vm_pairs = nullptr;
    Word *d_wm_pairs = nullptr;
    Word *d_a_work = nullptr;
    Word *d_b_work = nullptr;
    Word *d_c_work = nullptr;
    Word *d_pairwise_out = nullptr;
    Word *d_reduced = nullptr;
    Word *d_f = nullptr;
    Word *d_vj = nullptr;
    Word *d_wj = nullptr;
    Word *d_x = nullptr;
    Word *d_y = nullptr;
    Word *d_z = nullptr;

    auto alloc_and_upload = [&](Word **dst, const std::vector<Word> &src, const char *label) {
        check(cudaMalloc(dst, sizeof(Word) * src.size()), label);
        check(cudaMemcpy(*dst, src.data(), sizeof(Word) * src.size(), cudaMemcpyHostToDevice), label);
    };
    auto alloc_only = [&](Word **dst, size_t count, const char *label) {
        check(cudaMalloc(dst, sizeof(Word) * count), label);
    };

    alloc_and_upload(&d_a_pairs, a_pairs, "alloc/copy d_a_pairs");
    alloc_and_upload(&d_e_pairs, e_pairs, "alloc/copy d_e_pairs");
    alloc_and_upload(&d_vm_pairs, vm_pairs, "alloc/copy d_vm_pairs");
    alloc_and_upload(&d_wm_pairs, wm_pairs, "alloc/copy d_wm_pairs");
    alloc_only(&d_a_work, pair_coeffs, "alloc d_a_work");
    alloc_only(&d_b_work, pair_coeffs, "alloc d_b_work");
    alloc_only(&d_c_work, pair_coeffs, "alloc d_c_work");
    alloc_only(&d_pairwise_out, pair_coeffs, "alloc d_pairwise_out");
    alloc_only(&d_reduced, output_coeffs, "alloc d_reduced");
    alloc_and_upload(&d_f, f, "alloc/copy d_f");
    alloc_and_upload(&d_vj, vj, "alloc/copy d_vj");
    alloc_and_upload(&d_wj, wj, "alloc/copy d_wj");
    alloc_only(&d_x, output_coeffs, "alloc d_x");
    alloc_only(&d_y, output_coeffs, "alloc d_y");
    alloc_only(&d_z, output_coeffs, "alloc d_z");

    bool correct = true;
    run_inner_product_phase(d_a_pairs,
                            d_e_pairs,
                            d_a_work,
                            d_b_work,
                            d_c_work,
                            d_pairwise_out,
                            d_reduced,
                            d_f,
                            d_x,
                            tables,
                            n,
                            args.outputs,
                            args.lanes,
                            log_degree,
                            config.modulus);
    run_inner_product_phase(d_a_pairs,
                            d_vm_pairs,
                            d_a_work,
                            d_b_work,
                            d_c_work,
                            d_pairwise_out,
                            d_reduced,
                            d_vj,
                            d_y,
                            tables,
                            n,
                            args.outputs,
                            args.lanes,
                            log_degree,
                            config.modulus);
    run_inner_product_phase(d_a_pairs,
                            d_wm_pairs,
                            d_a_work,
                            d_b_work,
                            d_c_work,
                            d_pairwise_out,
                            d_reduced,
                            d_wj,
                            d_z,
                            tables,
                            n,
                            args.outputs,
                            args.lanes,
                            log_degree,
                            config.modulus);
    check(cudaDeviceSynchronize(), "sync initial VOLE run");

    if (!args.skip_validation) {
        const int validation_outputs = std::min(args.outputs, kMaxValidationBatches);
        const size_t validation_coeffs = static_cast<size_t>(validation_outputs) * static_cast<size_t>(n);
        std::vector<Word> gpu_x(validation_coeffs);
        std::vector<Word> gpu_y(validation_coeffs);
        std::vector<Word> gpu_z(validation_coeffs);
        check(cudaMemcpy(gpu_x.data(), d_x, sizeof(Word) * validation_coeffs, cudaMemcpyDeviceToHost),
              "copy gpu_x");
        check(cudaMemcpy(gpu_y.data(), d_y, sizeof(Word) * validation_coeffs, cudaMemcpyDeviceToHost),
              "copy gpu_y");
        check(cudaMemcpy(gpu_z.data(), d_z, sizeof(Word) * validation_coeffs, cudaMemcpyDeviceToHost),
              "copy gpu_z");

        std::vector<Word> host_x = host_expand_phase(a_pairs,
                                                     e,
                                                     f,
                                                     phi_norm,
                                                     post_norm,
                                                     config,
                                                     validation_outputs,
                                                     args.lanes,
                                                     n,
                                                     log_degree);
        std::vector<Word> host_y = host_expand_phase(a_pairs,
                                                     vm,
                                                     vj,
                                                     phi_norm,
                                                     post_norm,
                                                     config,
                                                     validation_outputs,
                                                     args.lanes,
                                                     n,
                                                     log_degree);
        std::vector<Word> host_z = host_expand_phase(a_pairs,
                                                     wm,
                                                     wj,
                                                     phi_norm,
                                                     post_norm,
                                                     config,
                                                     validation_outputs,
                                                     args.lanes,
                                                     n,
                                                     log_degree);

        correct = compare_vectors(host_x, gpu_x, n, "VOLE x") && correct;
        correct = compare_vectors(host_y, gpu_y, n, "VOLE y") && correct;
        correct = compare_vectors(host_z, gpu_z, n, "VOLE z") && correct;
        correct = validate_vole_relation(gpu_x, gpu_y, gpu_z, delta, n, config.modulus) && correct;
    }

    std::vector<double> x_samples;
    std::vector<double> y_samples;
    std::vector<double> z_samples;
    std::vector<double> total_samples;
    x_samples.reserve(args.iters);
    y_samples.reserve(args.iters);
    z_samples.reserve(args.iters);
    total_samples.reserve(args.iters);

    cudaEvent_t phase_start_evt;
    cudaEvent_t phase_stop_evt;
    cudaEvent_t total_start_evt;
    cudaEvent_t total_stop_evt;
    check(cudaEventCreate(&phase_start_evt), "create phase start event");
    check(cudaEventCreate(&phase_stop_evt), "create phase stop event");
    check(cudaEventCreate(&total_start_evt), "create total start event");
    check(cudaEventCreate(&total_stop_evt), "create total stop event");

    for (int iter = 0; iter < args.warmup; iter++) {
        run_inner_product_phase(d_a_pairs,
                                d_e_pairs,
                                d_a_work,
                                d_b_work,
                                d_c_work,
                                d_pairwise_out,
                                d_reduced,
                                d_f,
                                d_x,
                                tables,
                                n,
                                args.outputs,
                                args.lanes,
                                log_degree,
                                config.modulus);
        run_inner_product_phase(d_a_pairs,
                                d_vm_pairs,
                                d_a_work,
                                d_b_work,
                                d_c_work,
                                d_pairwise_out,
                                d_reduced,
                                d_vj,
                                d_y,
                                tables,
                                n,
                                args.outputs,
                                args.lanes,
                                log_degree,
                                config.modulus);
        run_inner_product_phase(d_a_pairs,
                                d_wm_pairs,
                                d_a_work,
                                d_b_work,
                                d_c_work,
                                d_pairwise_out,
                                d_reduced,
                                d_wj,
                                d_z,
                                tables,
                                n,
                                args.outputs,
                                args.lanes,
                                log_degree,
                                config.modulus);
        check(cudaDeviceSynchronize(), "sync VOLE warmup");
    }

    for (int iter = 0; iter < args.iters; iter++) {
        float ms = 0.0f;

        check(cudaEventRecord(total_start_evt), "record total start");

        check(cudaEventRecord(phase_start_evt), "record x start");
        run_inner_product_phase(d_a_pairs,
                                d_e_pairs,
                                d_a_work,
                                d_b_work,
                                d_c_work,
                                d_pairwise_out,
                                d_reduced,
                                d_f,
                                d_x,
                                tables,
                                n,
                                args.outputs,
                                args.lanes,
                                log_degree,
                                config.modulus);
        check(cudaEventRecord(phase_stop_evt), "record x stop");
        check(cudaEventSynchronize(phase_stop_evt), "sync x stop");
        check(cudaEventElapsedTime(&ms, phase_start_evt, phase_stop_evt), "elapsed x");
        x_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(phase_start_evt), "record y start");
        run_inner_product_phase(d_a_pairs,
                                d_vm_pairs,
                                d_a_work,
                                d_b_work,
                                d_c_work,
                                d_pairwise_out,
                                d_reduced,
                                d_vj,
                                d_y,
                                tables,
                                n,
                                args.outputs,
                                args.lanes,
                                log_degree,
                                config.modulus);
        check(cudaEventRecord(phase_stop_evt), "record y stop");
        check(cudaEventSynchronize(phase_stop_evt), "sync y stop");
        check(cudaEventElapsedTime(&ms, phase_start_evt, phase_stop_evt), "elapsed y");
        y_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(phase_start_evt), "record z start");
        run_inner_product_phase(d_a_pairs,
                                d_wm_pairs,
                                d_a_work,
                                d_b_work,
                                d_c_work,
                                d_pairwise_out,
                                d_reduced,
                                d_wj,
                                d_z,
                                tables,
                                n,
                                args.outputs,
                                args.lanes,
                                log_degree,
                                config.modulus);
        check(cudaEventRecord(phase_stop_evt), "record z stop");
        check(cudaEventSynchronize(phase_stop_evt), "sync z stop");
        check(cudaEventElapsedTime(&ms, phase_start_evt, phase_stop_evt), "elapsed z");
        z_samples.push_back(ms * 1000.0);

        check(cudaEventRecord(total_stop_evt), "record total stop");
        check(cudaEventSynchronize(total_stop_evt), "sync total stop");
        check(cudaEventElapsedTime(&ms, total_start_evt, total_stop_evt), "elapsed total");
        total_samples.push_back(ms * 1000.0);
    }

    Stats x_stats = compute_stats(x_samples);
    Stats y_stats = compute_stats(y_samples);
    Stats z_stats = compute_stats(z_samples);
    Stats total_stats = compute_stats(total_samples);
    const char *validation = args.skip_validation ? "skipped" : (correct ? "pass" : "fail");
    const int correct_flag = args.skip_validation ? -1 : (correct ? 1 : 0);

    std::cout << RINGLPN_DEVICE_LABEL << ",synthetic_mpvole," << n << "," << log_degree << ","
              << args.requested_qbits << "," << config.actual_qbits << ","
              << args.outputs << "," << args.lanes << "," << args.noise_weight << ","
              << args.iters << "," << validation << ","
              << x_stats.mean_us << "," << x_stats.stddev_us << ","
              << y_stats.mean_us << "," << y_stats.stddev_us << ","
              << z_stats.mean_us << "," << z_stats.stddev_us << ","
              << total_stats.mean_us << "," << total_stats.stddev_us << ","
              << correct_flag << "\n";

    cudaEventDestroy(phase_start_evt);
    cudaEventDestroy(phase_stop_evt);
    cudaEventDestroy(total_start_evt);
    cudaEventDestroy(total_stop_evt);
    free_tables(tables);
    cudaFree(d_a_pairs);
    cudaFree(d_e_pairs);
    cudaFree(d_vm_pairs);
    cudaFree(d_wm_pairs);
    cudaFree(d_a_work);
    cudaFree(d_b_work);
    cudaFree(d_c_work);
    cudaFree(d_pairwise_out);
    cudaFree(d_reduced);
    cudaFree(d_f);
    cudaFree(d_vj);
    cudaFree(d_wj);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return (args.skip_validation || correct) ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
    VoleArgs args = parse_vole_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,n,logn,requested_qbits,actual_qbits,m,c,noise_weight,iters,validation,x_mean_us,x_std_us,y_mean_us,y_std_us,z_mean_us,z_std_us,expand_mean_us,expand_std_us,correct\n";
    }

    if (args.requested_qbits == 64) {
        return run_vole_benchmark<uint64_t>(args, kConfig62);
    }
    return run_vole_benchmark<uint32_t>(args, kConfig30);
}