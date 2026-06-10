// External-baseline NTT comparison: GPU-NTT (Ozcan-Savas, eprint 2023/1410)
// merge-NTT versus this project's cheddar-derived backend, same process, same
// prime, same psi, same operation (negacyclic polymul in Z_p[X]/(X^N+1)),
// cross-validated: both GPU polymul outputs must equal the host oracle
// (host_polymul_reference) elementwise.
//
// Modulus caveat (measured, load-bearing): GPU-NTT's 64-bit Barrett reduction
// does not support the project's 62-bit primes -- the identical call sequence
// round-trips correctly at its <=60-bit default-pool prime and produces
// out-of-range values at p62 (no headroom for 62-bit moduli). The comparison
// therefore runs at the 60-bit pool prime q' = 576460756061519873
// (v2(q'-1) = 29, NTT-friendly far past 2^20), which both backends support;
// rows at --prime p62 report the GPU-NTT side as "unsupported" and still time
// the cheddar side. GPU-NTT's 4-step variant is also absent: upstream's
// custom-prime NTTParameters4Step constructor is commented out, so it cannot
// run an externally chosen prime at all.
//
// Benchmark-only external dependency: set GPU_NTT_HOME (default
// /home/fatih/GPU-NTT) and build with scripts/build_ntt_gpu_ntt_baseline.sh.
// Not part of the paper-checkpoint gate.

#include "gpuntt/ntt_merge/ntt.cuh"

#define RINGLPN_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "ntt_baseline_compare"
#endif
#define Stats RingLpnNttStats
#include "ringlpn/src/bench_ntt_cuda_cheddar.cu"
#undef Stats

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using BClock = std::chrono::high_resolution_clock;
using Word = uint64_t;

// 60-bit GPU-NTT default-pool prime, with a primitive 2^21-th root so the
// cheddar tables (kMaxDegree = 2^20) can be built for it.
constexpr ModulusConfig<Word> kConfigPool60 = {
    576460756061519873ULL,
    0ULL,
    455708031105737486ULL,
    60,
};

struct BaselineArgs {
    int n = 8192;
    std::string prime = "pool60";  // pool60 | p62
    int batch = 4;
    int iters = 100;
    int warmup = 10;
    bool csv_header = false;
};

__global__ void baseline_pointwise_mul(Data64 *a,
                                       const Data64 *b,
                                       Modulus<Data64> mod,
                                       size_t count) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        a[idx] = OPERATOR_GPU<Data64>::mult(a[idx], b[idx], mod);
    }
}

static void baseline_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--n N] [--prime pool60|p62] [--batch B] [--iters I]"
              << " [--warmup W] [--csv-header]\n";
}

static BaselineArgs parse_baseline_args(int argc, char **argv) {
    BaselineArgs args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--prime") && i + 1 < argc) {
            args.prime = argv[++i];
        } else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) {
            args.batch = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            baseline_usage(argv[0]);
            std::exit(1);
        }
    }
    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree ||
        (args.prime != "pool60" && args.prime != "p62") || args.batch <= 0 ||
        args.iters <= 0 || args.warmup < 0) {
        baseline_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

static RingLpnNttStats time_loop(const std::function<void()> &body,
                                 int iters,
                                 int warmup) {
    for (int i = 0; i < warmup; ++i) {
        body();
    }
    check(cudaDeviceSynchronize(), "sync baseline warmup");
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto start = BClock::now();
        body();
        check(cudaDeviceSynchronize(), "sync baseline iter");
        auto end = BClock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count()));
    }
    return compute_stats(samples);
}

struct BackendResult {
    bool supported = false;
    bool correct = false;
    RingLpnNttStats fwd{};
    RingLpnNttStats polymul{};
};

// --- GPU-NTT merge backend -------------------------------------------------

static BackendResult run_gpuntt_backend(const BaselineArgs &args,
                                        const ModulusConfig<Word> &config,
                                        const std::vector<Word> &h_a,
                                        const std::vector<Word> &h_b,
                                        const std::vector<Word> &expected) {
    BackendResult out;
    if (config.modulus >> 60) {
        // 62-bit modulus: Barrett headroom failure, measured (see header).
        return out;
    }
    out.supported = true;

    const int n = args.n;
    const int log_degree = int_log2(static_cast<uint32_t>(n));
    const size_t total = static_cast<size_t>(args.batch) * n;

    const Word psi = compute_phi_for_n(config, n);
    const Word omega = mod_mul_host<Word>(psi, psi, config.modulus);

    Modulus<Data64> mod(config.modulus);
    gpuntt::NTTFactors<Data64> factors(mod, omega, psi);
    gpuntt::NTTParameters<Data64> params(
        log_degree, factors, gpuntt::ReductionPolynomial::X_N_plus);

    auto fwd_host =
        params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
    auto inv_host =
        params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
    Root<Data64> *d_fwd_table = nullptr;
    Root<Data64> *d_inv_table = nullptr;
    check(cudaMalloc(&d_fwd_table, fwd_host.size() * sizeof(fwd_host[0])),
          "alloc gpuntt fwd table");
    check(cudaMalloc(&d_inv_table, inv_host.size() * sizeof(inv_host[0])),
          "alloc gpuntt inv table");
    check(cudaMemcpy(d_fwd_table, fwd_host.data(),
                     fwd_host.size() * sizeof(fwd_host[0]), cudaMemcpyHostToDevice),
          "copy gpuntt fwd table");
    check(cudaMemcpy(d_inv_table, inv_host.data(),
                     inv_host.size() * sizeof(inv_host[0]), cudaMemcpyHostToDevice),
          "copy gpuntt inv table");

    Data64 *d_a = nullptr;
    Data64 *d_b = nullptr;
    check(cudaMalloc(&d_a, total * sizeof(Word)), "alloc gpuntt a");
    check(cudaMalloc(&d_b, total * sizeof(Word)), "alloc gpuntt b");

    gpuntt::ntt_configuration<Data64> cfg_ntt = {};
    cfg_ntt.n_power = log_degree;
    cfg_ntt.ntt_type = gpuntt::FORWARD;
    cfg_ntt.ntt_layout = gpuntt::PerPolynomial;
    cfg_ntt.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
    cfg_ntt.zero_padding = false;
    cfg_ntt.stream = 0;
    gpuntt::ntt_configuration<Data64> cfg_intt = cfg_ntt;
    cfg_intt.ntt_type = gpuntt::INVERSE;
    cfg_intt.mod_inverse = params.n_inv;

    const dim3 mul_block(256);
    const dim3 mul_grid(grid_size(total, mul_block.x));

    auto upload = [&]() {
        check(cudaMemcpy(d_a, h_a.data(), total * sizeof(Word), cudaMemcpyHostToDevice),
              "upload gpuntt a");
        check(cudaMemcpy(d_b, h_b.data(), total * sizeof(Word), cudaMemcpyHostToDevice),
              "upload gpuntt b");
    };
    auto polymul_once = [&]() {
        gpuntt::GPU_NTT_Inplace(d_a, d_fwd_table, mod, cfg_ntt, args.batch);
        gpuntt::GPU_NTT_Inplace(d_b, d_fwd_table, mod, cfg_ntt, args.batch);
        baseline_pointwise_mul<<<mul_grid, mul_block>>>(d_a, d_b, mod, total);
        check_launch("baseline pointwise mul");
        gpuntt::GPU_INTT_Inplace(d_a, d_inv_table, mod, cfg_intt, args.batch);
    };

    upload();
    polymul_once();
    check(cudaDeviceSynchronize(), "sync gpuntt validation");
    std::vector<Word> got(n);
    check(cudaMemcpy(got.data(), d_a, n * sizeof(Word), cudaMemcpyDeviceToHost),
          "copy gpuntt result");
    out.correct = (got == expected);

    upload();
    out.fwd = time_loop(
        [&]() { gpuntt::GPU_NTT_Inplace(d_a, d_fwd_table, mod, cfg_ntt, args.batch); },
        args.iters, args.warmup);
    upload();
    out.polymul = time_loop(polymul_once, args.iters, args.warmup);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_fwd_table);
    cudaFree(d_inv_table);
    return out;
}

// --- cheddar backend --------------------------------------------------------

static BackendResult run_cheddar_backend(const BaselineArgs &args,
                                         const ModulusConfig<Word> &config,
                                         const std::vector<Word> &h_a,
                                         const std::vector<Word> &h_b,
                                         const std::vector<Word> &expected) {
    BackendResult out;
    out.supported = true;

    const int n = args.n;
    const int log_degree = int_log2(static_cast<uint32_t>(n));
    const size_t total = static_cast<size_t>(args.batch) * n;

    HostTables<Word> host_tables;
    compute_cheddar_tables(host_tables, n, config);
    DeviceTables<Word> tables;
    alloc_and_copy(tables, host_tables);

    Word *d_a = nullptr;
    Word *d_b = nullptr;
    Word *d_aw = nullptr;
    Word *d_bw = nullptr;
    Word *d_cw = nullptr;
    Word *d_out = nullptr;
    check(cudaMalloc(&d_a, total * sizeof(Word)), "alloc cheddar a");
    check(cudaMalloc(&d_b, total * sizeof(Word)), "alloc cheddar b");
    check(cudaMalloc(&d_aw, total * sizeof(Word)), "alloc cheddar aw");
    check(cudaMalloc(&d_bw, total * sizeof(Word)), "alloc cheddar bw");
    check(cudaMalloc(&d_cw, total * sizeof(Word)), "alloc cheddar cw");
    check(cudaMalloc(&d_out, total * sizeof(Word)), "alloc cheddar out");

    auto upload = [&]() {
        check(cudaMemcpy(d_a, h_a.data(), total * sizeof(Word), cudaMemcpyHostToDevice),
              "upload cheddar a");
        check(cudaMemcpy(d_b, h_b.data(), total * sizeof(Word), cudaMemcpyHostToDevice),
              "upload cheddar b");
    };
    auto polymul_once = [&]() {
        run_full_polymul(d_a, d_b, d_aw, d_bw, d_cw, d_out, tables, n, args.batch,
                         log_degree);
    };

    upload();
    polymul_once();
    check(cudaDeviceSynchronize(), "sync cheddar validation");
    std::vector<Word> got(n);
    check(cudaMemcpy(got.data(), d_out, n * sizeof(Word), cudaMemcpyDeviceToHost),
          "copy cheddar result");
    out.correct = (got == expected);

    upload();
    out.fwd = time_loop(
        [&]() {
            run_forward_only(d_a, d_aw, tables, n, args.batch, log_degree);
        },
        args.iters, args.warmup);
    upload();
    out.polymul = time_loop(polymul_once, args.iters, args.warmup);

    free_tables(tables);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_aw);
    cudaFree(d_bw);
    cudaFree(d_cw);
    cudaFree(d_out);
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    BaselineArgs args = parse_baseline_args(argc, argv);
    const ModulusConfig<Word> config =
        args.prime == "p62" ? kConfig62 : kConfigPool60;

    const int n = args.n;
    const int log_degree = int_log2(static_cast<uint32_t>(n));
    const size_t total = static_cast<size_t>(args.batch) * n;

    std::mt19937_64 rng(1);
    std::uniform_int_distribution<uint64_t> dist(0, config.modulus - 1);
    std::vector<Word> h_a(total);
    std::vector<Word> h_b(total);
    for (size_t i = 0; i < total; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    std::vector<Word> phi_norm;
    std::vector<Word> post_norm;
    compute_reference_vectors(phi_norm, post_norm, n, config);
    std::vector<Word> lhs(h_a.begin(), h_a.begin() + n);
    std::vector<Word> rhs(h_b.begin(), h_b.begin() + n);
    std::vector<Word> expected =
        host_polymul_reference(lhs, rhs, phi_norm, post_norm, config, n, log_degree);

    BackendResult gpuntt_r = run_gpuntt_backend(args, config, h_a, h_b, expected);
    BackendResult cheddar_r = run_cheddar_backend(args, config, h_a, h_b, expected);

    auto verdict = [](const BackendResult &r) {
        return !r.supported ? "unsupported" : (r.correct ? "pass" : "fail");
    };

    if (args.csv_header) {
        std::cout << "device,n,logn,prime_label,modulus,batch,iters,"
                  << "gpuntt_validation,cheddar_validation,"
                  << "gpuntt_ntt_mean_us,gpuntt_ntt_std_us,"
                  << "cheddar_ntt_mean_us,cheddar_ntt_std_us,"
                  << "gpuntt_poly_mul_mean_us,gpuntt_poly_mul_std_us,"
                  << "cheddar_poly_mul_mean_us,cheddar_poly_mul_std_us\n";
    }
    std::cout << RINGLPN_DEVICE_LABEL << "," << n << "," << log_degree << ","
              << args.prime << "," << config.modulus << "," << args.batch << ","
              << args.iters << "," << verdict(gpuntt_r) << "," << verdict(cheddar_r)
              << "," << gpuntt_r.fwd.mean_us << "," << gpuntt_r.fwd.stddev_us << ","
              << cheddar_r.fwd.mean_us << "," << cheddar_r.fwd.stddev_us << ","
              << gpuntt_r.polymul.mean_us << "," << gpuntt_r.polymul.stddev_us << ","
              << cheddar_r.polymul.mean_us << "," << cheddar_r.polymul.stddev_us
              << "\n";

    const bool gate = (!gpuntt_r.supported || gpuntt_r.correct) && cheddar_r.correct;
    return gate ? 0 : 2;
}
