#include "gpu_spfss_zp.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using ringlpn_spfss_zp::GPUDPFZpKey;
using ringlpn_spfss_zp::Word;

namespace {

constexpr Word kModulus62 = 4611686018326724609ULL;

struct Case {
    const char *name;
    int log_domain;
    std::vector<Word> alphas;
    std::vector<Word> betas;
};

Word mod_add_host(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? s - modulus : s;
}

bool run_case(const Case &tc, AESGlobalContext *gaes) {
    GPUDPFZpKey k0;
    GPUDPFZpKey k1;
    ringlpn_spfss_zp::gpuKeyGenDPFZpPair(tc.alphas,
                                         tc.betas,
                                         tc.log_domain,
                                         kModulus62,
                                         0xC0FFEE1234ULL + static_cast<uint64_t>(tc.log_domain),
                                         gaes,
                                         k0,
                                         k1);

    const size_t domain = size_t(1) << tc.log_domain;
    Word *d_out0 = nullptr;
    Word *d_out1 = nullptr;
    ringlpn_spfss_zp::cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_out0),
                                            domain * sizeof(Word)),
                                 "alloc test out0");
    ringlpn_spfss_zp::cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_out1),
                                            domain * sizeof(Word)),
                                 "alloc test out1");
    ringlpn_spfss_zp::gpuDpfZpFullEvalSum(k0, d_out0, gaes);
    ringlpn_spfss_zp::gpuDpfZpFullEvalSum(k1, d_out1, gaes);

    std::vector<Word> out0(domain);
    std::vector<Word> out1(domain);
    ringlpn_spfss_zp::cuda_check(cudaMemcpy(out0.data(), d_out0, domain * sizeof(Word),
                                            cudaMemcpyDeviceToHost),
                                 "copy test out0");
    ringlpn_spfss_zp::cuda_check(cudaMemcpy(out1.data(), d_out1, domain * sizeof(Word),
                                            cudaMemcpyDeviceToHost),
                                 "copy test out1");
    cudaFree(d_out0);
    cudaFree(d_out1);

    std::vector<Word> expected(domain, 0);
    for (size_t i = 0; i < tc.alphas.size(); ++i) {
        expected[tc.alphas[i]] = mod_add_host(expected[tc.alphas[i]], tc.betas[i], kModulus62);
    }

    bool ok = true;
    size_t first_bad = 0;
    Word first_got = 0;
    Word first_expected = 0;
    for (size_t x = 0; x < domain; ++x) {
        Word got = mod_add_host(out0[x], out1[x], kModulus62);
        if (got != expected[x]) {
            ok = false;
            first_bad = x;
            first_got = got;
            first_expected = expected[x];
            break;
        }
    }

    std::cout << tc.name << ",log_domain=" << tc.log_domain
              << ",points=" << tc.alphas.size()
              << ",spfss_pass=" << (ok ? 1 : 0);
    if (!ok) {
        std::cout << ",first_bad=" << first_bad
                  << ",got=" << first_got
                  << ",expected=" << first_expected;
    }
    std::cout << "\n";
    return ok;
}

}  // namespace

int main() {
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);

    std::vector<Case> cases = {
        {"single_point", 6, {17}, {1234567}},
        {"multiple_points", 7, {3, 17, 42, 96}, {5, 11, 19, 23}},
        {"colliding_alphas", 7, {9, 9, 9, 31}, {7, 13, 29, 37}},
        {"edge_alphas", 8, {0, 255}, {111, 222}},
    };

    bool ok = true;
    for (const auto &tc : cases) {
        ok = run_case(tc, &gaes) && ok;
    }

    freeAESGlobalContext(&gaes);
    cudaDeviceSynchronize();
    return ok ? 0 : 1;
}
