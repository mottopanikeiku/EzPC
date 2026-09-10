#include "gpu_spfss_zp.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using ringlpn_spfss_zp::GPUDPFZpKey;
using ringlpn_spfss_zp::Word;

namespace {

constexpr Word kModulus62 = 4611686018326724609ULL;
constexpr Word kModulus62Crt2 = 4611686018309947393ULL;

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


struct BatchSlice {
    size_t offset;
    size_t count;
};

bool make_key_pairs(const std::vector<int> &counts, int log_domain,
                    Word modulus, AESGlobalContext *gaes,
                    std::vector<GPUDPFZpKey> &keys0,
                    std::vector<GPUDPFZpKey> &keys1) {
    const size_t domain = size_t(1) << log_domain;
    keys0.clear();
    keys1.clear();
    keys0.reserve(counts.size());
    keys1.reserve(counts.size());
    for (size_t key_index = 0; key_index < counts.size(); ++key_index) {
        if (counts[key_index] <= 0) return false;
        std::vector<Word> alphas(static_cast<size_t>(counts[key_index]));
        std::vector<Word> betas(alphas.size());
        for (size_t point = 0; point < alphas.size(); ++point) {
            alphas[point] = static_cast<Word>(
                (11 * key_index + 7 * point + point * point) % domain);
            betas[point] = static_cast<Word>(
                (0x12345ULL + 97 * key_index + 31 * point) % modulus);
        }
        GPUDPFZpKey key0;
        GPUDPFZpKey key1;
        ringlpn_spfss_zp::gpuKeyGenDPFZpPair(
            alphas, betas, log_domain, modulus,
            0xB47C0000ULL + static_cast<uint64_t>(key_index) +
                static_cast<uint64_t>(modulus & 0xffff),
            gaes, key0, key1);
        keys0.push_back(std::move(key0));
        keys1.push_back(std::move(key1));
    }
    return true;
}

bool reference_prepared_outputs(
    const ringlpn_spfss_zp::DeviceGPUDPFZpKeyBatch &keys,
    AESGlobalContext *gaes, std::vector<Word> &reference) {
    const size_t domain = size_t(1) << keys.log_domain();
    if (keys.size() >
        std::numeric_limits<size_t>::max() / domain) {
        return false;
    }
    reference.resize(keys.size() * domain);
    Word *d_one = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&d_one),
                   domain * sizeof(Word)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    for (size_t index = 0; index < keys.size(); ++index) {
        ringlpn_spfss_zp::gpuDpfZpFullEvalSumPrepared(
            keys.at(index), d_one, gaes);
        ringlpn_spfss_zp::cuda_check(
            cudaMemcpy(reference.data() + index * domain, d_one,
                       domain * sizeof(Word), cudaMemcpyDeviceToHost),
            "copy prepared parity reference");
    }
    cudaFree(d_one);
    return true;
}

bool verify_batch_slice(
    const char *layout, Word modulus, int party, bool enable_breadth,
    const ringlpn_spfss_zp::DeviceGPUDPFZpKeyBatch &keys,
    const std::vector<Word> &reference, const BatchSlice &slice,
    AESGlobalContext *gaes) {
    const size_t domain = size_t(1) << keys.log_domain();
    size_t output_words = 0;
    if (!ringlpn_spfss_zp::checked_size_product(
            slice.count, domain, output_words) ||
        slice.offset > keys.size() ||
        slice.count > keys.size() - slice.offset) {
        return false;
    }
    constexpr size_t guard_words = 8;
    if (output_words >
        std::numeric_limits<size_t>::max() - 2 * guard_words) {
        return false;
    }
    const size_t allocation_words = output_words + 2 * guard_words;
    const Word sentinel = 0xD15EA5E5D15EA5E5ULL;
    std::vector<Word> guarded(allocation_words, sentinel);
    Word *d_guarded = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&d_guarded),
                   allocation_words * sizeof(Word)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(d_guarded, guarded.data(),
                   allocation_words * sizeof(Word), cudaMemcpyHostToDevice),
        "copy batch parity guards");
    Word *d_output = d_guarded + guard_words;
    ringlpn_spfss_zp::DpfEvaluationPath path;
    const bool offset_rejected =
        !ringlpn_spfss_zp::gpuDpfZpFullEvalSumBatchPrepared(
            keys, keys.size(), 1, d_output, output_words, gaes, &path);
    const bool capacity_rejected =
        !ringlpn_spfss_zp::gpuDpfZpFullEvalSumBatchPrepared(
            keys, slice.offset, slice.count, d_output, output_words - 1, gaes,
            &path);
    const bool evaluated =
        ringlpn_spfss_zp::gpuDpfZpFullEvalSumBatchPrepared(
            keys, slice.offset, slice.count, d_output, output_words, gaes,
            &path);
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(guarded.data(), d_guarded,
                   allocation_words * sizeof(Word), cudaMemcpyDeviceToHost),
        "copy batch parity output");
    cudaFree(d_guarded);

    bool guards_ok = true;
    for (size_t index = 0; index < guard_words; ++index) {
        guards_ok = guards_ok && guarded[index] == sentinel &&
                    guarded[guard_words + output_words + index] == sentinel;
    }
    const Word *expected =
        reference.data() + slice.offset * domain;
    const bool parity =
        evaluated &&
        std::equal(guarded.begin() + static_cast<std::ptrdiff_t>(guard_words),
                   guarded.begin() + static_cast<std::ptrdiff_t>(
                                         guard_words + output_words),
                   expected);
    const bool modulo_outputs =
        evaluated &&
        std::all_of(
            guarded.begin() + static_cast<std::ptrdiff_t>(guard_words),
            guarded.begin() + static_cast<std::ptrdiff_t>(
                                  guard_words + output_words),
            [modulus](Word value) { return value < modulus; });
    const ringlpn_spfss_zp::DpfEvaluationPath expected_path =
        enable_breadth
            ? ringlpn_spfss_zp::DpfEvaluationPath::kBreadth
            : ringlpn_spfss_zp::DpfEvaluationPath::kRootToLeaf;
    const bool path_ok = evaluated && path == expected_path;
    const bool ok = offset_rejected && capacity_rejected && guards_ok &&
                    parity && modulo_outputs && path_ok;
    std::cout << "batch_parity,layout=" << layout
              << ",prime=" << modulus << ",party=" << party
              << ",path=" << (enable_breadth ? "breadth" : "root_to_leaf")
              << ",offset=" << slice.offset << ",count=" << slice.count
              << ",capacity_rejected=" << (capacity_rejected ? 1 : 0)
              << ",offset_rejected=" << (offset_rejected ? 1 : 0)
              << ",guards=" << (guards_ok ? 1 : 0)
              << ",exact=" << (parity ? 1 : 0)
              << ",modulo_p=" << (modulo_outputs ? 1 : 0)
              << ",pass=" << (ok ? 1 : 0) << "\n";
    return ok;
}

bool run_batch_layout(const char *layout, const std::vector<int> &counts,
                      const std::vector<BatchSlice> &slices,
                      int expected_max_count, Word modulus,
                      AESGlobalContext *gaes) {
    constexpr int log_domain = 6;
    std::vector<GPUDPFZpKey> keys0;
    std::vector<GPUDPFZpKey> keys1;
    if (!make_key_pairs(counts, log_domain, modulus, gaes, keys0, keys1)) {
        return false;
    }
    bool ok = true;
    for (int party = 0; party < 2; ++party) {
        const std::vector<GPUDPFZpKey> &host_keys =
            party == 0 ? keys0 : keys1;
        for (bool enable_breadth : {true, false}) {
            ringlpn_spfss_zp::DeviceGPUDPFZpKeyBatch prepared;
            if (!prepared.initialize(host_keys, enable_breadth) ||
                prepared.max_count() != expected_max_count ||
                prepared.breadth_state_count() >
                    ringlpn_spfss_zp::kMaxBreadthStates ||
                prepared.breadth_ready() != enable_breadth) {
                return false;
            }
            std::vector<Word> reference;
            if (!reference_prepared_outputs(prepared, gaes, reference)) {
                return false;
            }
            for (const BatchSlice &slice : slices) {
                ok = verify_batch_slice(layout, modulus, party,
                                        enable_breadth, prepared, reference,
                                        slice, gaes) &&
                     ok;
            }
        }
    }
    return ok;
}
}  // namespace

int main() {
    // This focused smoke needs only its explicit buffers. Do not invoke
    // Orca's 25-GiB eager async-pool reservation: it can fail on an otherwise
    // sufficient shared GPU before the test allocates any protocol state.
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

    std::vector<int> regular_counts;
    regular_counts.reserve(60);
    for (int matrix = 0; matrix < 4; ++matrix) {
        for (int group = 0; group < 15; ++group) {
            regular_counts.push_back(
                std::min(std::min(group + 1, 15 - group), 8));
        }
    }
    const std::vector<BatchSlice> regular_slices = {
        {0, 1}, {1, 14}, {15, 15}, {0, 60},
    };
    const std::vector<int> uniform_counts(4, 64);
    const std::vector<BatchSlice> uniform_slices = {
        {0, 1}, {1, 3},
    };
    for (Word modulus : {kModulus62, kModulus62Crt2}) {
        ok = run_batch_layout("regular", regular_counts, regular_slices, 8,
                              modulus, &gaes) &&
             ok;
        ok = run_batch_layout("uniform", uniform_counts, uniform_slices, 64,
                              modulus, &gaes) &&
             ok;
    }

    freeAESGlobalContext(&gaes);
    cudaDeviceSynchronize();
    return ok ? 0 : 1;
}
