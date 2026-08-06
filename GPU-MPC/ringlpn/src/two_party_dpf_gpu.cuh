#pragma once

#include "gpu_spfss_zp.cuh"
#include "two_party_dpf_protocol.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ringlpn_2pdpf {
namespace gpu_detail {

constexpr unsigned int kTreeReductionThreads = 256;


__global__ void expand_party_frontier_kernel(
    size_t width, size_t batch, const AESBlock *current_seeds,
    AESBlock *next_seeds,
    uint8_t *next_control, AESBlock *aggregate_left,
    AESBlock *aggregate_right, unsigned int *aggregate_t_left,
    unsigned int *aggregate_t_right, AESGlobalContext gaes) {
    AESSharedContext aes;
    loadSbox(&gaes, &aes);
    const size_t tree = blockIdx.x;
    if (tree >= batch) return;

    AESBlock left_xor = 0;
    AESBlock right_xor = 0;
    unsigned int t_left_xor = 0;
    unsigned int t_right_xor = 0;
    const size_t input_base = tree * width;
    const size_t output_base = tree * (2 * width);
    for (size_t node = threadIdx.x; node < width; node += blockDim.x) {
        AESBlock left, right;
        uint8_t t_left, t_right;
        ringlpn_spfss_zp::aes_prg_expand(current_seeds[input_base + node], &aes,
                                         left, t_left, right, t_right);
        next_seeds[output_base + 2 * node] = left;
        next_seeds[output_base + 2 * node + 1] = right;
        next_control[output_base + 2 * node] = t_left;
        next_control[output_base + 2 * node + 1] = t_right;
        left_xor ^= left;
        right_xor ^= right;
        t_left_xor ^= static_cast<unsigned int>(t_left);
        t_right_xor ^= static_cast<unsigned int>(t_right);
    }

    unsigned long long left_lo = static_cast<unsigned long long>(left_xor);
    unsigned long long left_hi =
        static_cast<unsigned long long>(left_xor >> 64);
    unsigned long long right_lo = static_cast<unsigned long long>(right_xor);
    unsigned long long right_hi =
        static_cast<unsigned long long>(right_xor >> 64);
    constexpr int kWarpSize = 32;
    constexpr unsigned int kFullWarp = 0xffffffffU;
    for (int offset = kWarpSize / 2; offset != 0; offset /= 2) {
        left_lo ^= __shfl_down_sync(kFullWarp, left_lo, offset);
        left_hi ^= __shfl_down_sync(kFullWarp, left_hi, offset);
        right_lo ^= __shfl_down_sync(kFullWarp, right_lo, offset);
        right_hi ^= __shfl_down_sync(kFullWarp, right_hi, offset);
        t_left_xor ^= __shfl_down_sync(kFullWarp, t_left_xor, offset);
        t_right_xor ^= __shfl_down_sync(kFullWarp, t_right_xor, offset);
    }
    constexpr unsigned int kWarps =
        kTreeReductionThreads / static_cast<unsigned int>(kWarpSize);
    __shared__ unsigned long long left_lo_partial[kWarps];
    __shared__ unsigned long long left_hi_partial[kWarps];
    __shared__ unsigned long long right_lo_partial[kWarps];
    __shared__ unsigned long long right_hi_partial[kWarps];
    __shared__ unsigned int t_left_partial[kWarps];
    __shared__ unsigned int t_right_partial[kWarps];
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        left_lo_partial[warp] = left_lo;
        left_hi_partial[warp] = left_hi;
        right_lo_partial[warp] = right_lo;
        right_hi_partial[warp] = right_hi;
        t_left_partial[warp] = t_left_xor;
        t_right_partial[warp] = t_right_xor;
    }
    __syncthreads();
    if (warp == 0) {
        left_lo = lane < kWarps ? left_lo_partial[lane] : 0;
        left_hi = lane < kWarps ? left_hi_partial[lane] : 0;
        right_lo = lane < kWarps ? right_lo_partial[lane] : 0;
        right_hi = lane < kWarps ? right_hi_partial[lane] : 0;
        t_left_xor = lane < kWarps ? t_left_partial[lane] : 0;
        t_right_xor = lane < kWarps ? t_right_partial[lane] : 0;
        for (int offset = kWarpSize / 2; offset != 0; offset /= 2) {
            left_lo ^= __shfl_down_sync(kFullWarp, left_lo, offset);
            left_hi ^= __shfl_down_sync(kFullWarp, left_hi, offset);
            right_lo ^= __shfl_down_sync(kFullWarp, right_lo, offset);
            right_hi ^= __shfl_down_sync(kFullWarp, right_hi, offset);
            t_left_xor ^=
                __shfl_down_sync(kFullWarp, t_left_xor, offset);
            t_right_xor ^=
                __shfl_down_sync(kFullWarp, t_right_xor, offset);
        }
        if (lane == 0) {
            aggregate_left[tree] =
                static_cast<AESBlock>(left_lo) |
                (static_cast<AESBlock>(left_hi) << 64);
            aggregate_right[tree] =
                static_cast<AESBlock>(right_lo) |
                (static_cast<AESBlock>(right_hi) << 64);
            aggregate_t_left[tree] = t_left_xor;
            aggregate_t_right[tree] = t_right_xor;
        }
    }
}

__global__ void apply_party_correction_kernel(
    size_t width, size_t batch, const uint8_t *current_control,
    AESBlock *next_seeds, uint8_t *next_control, const AESBlock *seed_cw,
    const uint8_t *t_left_cw, const uint8_t *t_right_cw) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t child_count = batch * width * 2;
    if (index >= child_count) return;
    const size_t child_width = width * 2;
    const size_t tree = index / child_width;
    const size_t child = index - tree * child_width;
    const size_t parent = tree * width + child / 2;
    if (current_control[parent] == 0) return;
    next_seeds[index] ^= seed_cw[tree];
    next_control[index] = static_cast<uint8_t>(
        next_control[index] ^
        ((child & 1) == 0 ? t_left_cw[tree] : t_right_cw[tree]));
}

__global__ void sum_party_leaves_kernel(
    int party, size_t width, size_t batch, Word modulus,
    const AESBlock *seeds, const uint8_t *control, Word *seed_sum,
    Word *control_sum) {
    const size_t tree = blockIdx.x;
    if (tree >= batch) return;
    Word local_seed = 0;
    Word local_control = 0;
    const size_t base = tree * width;
    for (size_t node = threadIdx.x; node < width; node += blockDim.x) {
        local_seed = ringlpn_spfss_zp::mod_add(
            local_seed,
            ringlpn_spfss_zp::convert_zp(seeds[base + node], modulus),
            modulus);
        local_control = ringlpn_spfss_zp::mod_add(
            local_control, static_cast<Word>(control[base + node] & 1),
            modulus);
    }
    __shared__ Word seed_partial[kTreeReductionThreads];
    __shared__ Word control_partial[kTreeReductionThreads];
    seed_partial[threadIdx.x] = local_seed;
    control_partial[threadIdx.x] = local_control;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            seed_partial[threadIdx.x] = ringlpn_spfss_zp::mod_add(
                seed_partial[threadIdx.x],
                seed_partial[threadIdx.x + stride], modulus);
            control_partial[threadIdx.x] = ringlpn_spfss_zp::mod_add(
                control_partial[threadIdx.x],
                control_partial[threadIdx.x + stride], modulus);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        seed_sum[tree] = party == 0
            ? seed_partial[0]
            : ringlpn_spfss_zp::mod_sub(0, seed_partial[0], modulus);
        control_sum[tree] = party == 0
            ? control_partial[0]
            : ringlpn_spfss_zp::mod_sub(0, control_partial[0], modulus);
    }
}

inline bool checked_bytes(size_t count, size_t element_size, size_t &bytes) {
    if (count != 0 && element_size >
                          std::numeric_limits<size_t>::max() / count) {
        return false;
    }
    bytes = count * element_size;
    return true;
}

inline bool cuda_success(cudaError_t status) { return status == cudaSuccess; }

}  // namespace gpu_detail

// One consume-once, party-owned GPU frontier. The object allocates a fixed pair
// of batch buffers; it never allocates per tree or per level. One CUDA block
// owns each tree's XOR/modular reductions, avoiding same-address atomics. Only
// OT aggregates and final party-local sums cross back to the host.
class GpuBatchedPartyTreeBatchState final : public PartyTreeBatchState {
  public:
    explicit GpuBatchedPartyTreeBatchState(AESGlobalContext *gaes)
        : gaes_(gaes) {}

    ~GpuBatchedPartyTreeBatchState() override { release(); }

    GpuBatchedPartyTreeBatchState(const GpuBatchedPartyTreeBatchState &) = delete;
    GpuBatchedPartyTreeBatchState &operator=(
        const GpuBatchedPartyTreeBatchState &) = delete;

    bool initialize(int party, int log_domain, Word modulus, PrgMode prg,
                    const std::vector<U128> &roots) override {
        if (consumed_ || gaes_ == nullptr || (party != 0 && party != 1) ||
            log_domain < 2 || log_domain > 20 || roots.empty() ||
            prg != PrgMode::kGpuAes) {
            return false;
        }
        consumed_ = true;
        party_ = party;
        log_domain_ = log_domain;
        modulus_ = modulus;
        batch_ = roots.size();
        width_ = 1;
        level_ = 0;
        if (batch_ > (std::numeric_limits<size_t>::max() >> log_domain_)) {
            return false;
        }
        capacity_ = batch_ << log_domain_;
        size_t seed_bytes = 0, control_bytes = 0, block_batch_bytes = 0;
        size_t word_batch_bytes = 0, uint_batch_bytes = 0;
        if (!gpu_detail::checked_bytes(capacity_, sizeof(AESBlock), seed_bytes) ||
            !gpu_detail::checked_bytes(capacity_, sizeof(uint8_t),
                                       control_bytes) ||
            !gpu_detail::checked_bytes(batch_, sizeof(AESBlock),
                                       block_batch_bytes) ||
            !gpu_detail::checked_bytes(batch_, sizeof(Word), word_batch_bytes) ||
            !gpu_detail::checked_bytes(batch_, sizeof(unsigned int),
                                       uint_batch_bytes)) {
            return false;
        }

        if (!allocate(reinterpret_cast<void **>(&current_seeds_), seed_bytes) ||
            !allocate(reinterpret_cast<void **>(&next_seeds_), seed_bytes) ||
            !allocate(reinterpret_cast<void **>(&current_control_),
                      control_bytes) ||
            !allocate(reinterpret_cast<void **>(&next_control_), control_bytes) ||
            !allocate(reinterpret_cast<void **>(&aggregate_left_),
                      block_batch_bytes) ||
            !allocate(reinterpret_cast<void **>(&aggregate_right_),
                      block_batch_bytes) ||
            !allocate(reinterpret_cast<void **>(&aggregate_t_left_),
                      uint_batch_bytes) ||
            !allocate(reinterpret_cast<void **>(&aggregate_t_right_),
                      uint_batch_bytes) ||
            !allocate(reinterpret_cast<void **>(&seed_cw_), block_batch_bytes) ||
            !allocate(reinterpret_cast<void **>(&t_left_cw_), batch_) ||
            !allocate(reinterpret_cast<void **>(&t_right_cw_), batch_) ||
            !allocate(reinterpret_cast<void **>(&seed_sum_), word_batch_bytes) ||
            !allocate(reinterpret_cast<void **>(&control_sum_),
                      word_batch_bytes)) {
            return false;
        }
        static_assert(sizeof(U128) == sizeof(AESBlock),
                      "DPF roots must retain all 128 bits");
        if (!gpu_detail::cuda_success(cudaMemcpy(
                current_seeds_, roots.data(), block_batch_bytes,
                cudaMemcpyHostToDevice)) ||
            !gpu_detail::cuda_success(cudaMemset(current_control_, party_,
                                                 batch_))) {
            return false;
        }
        counters_.gpu_h2d_bytes += block_batch_bytes;
        host_aggregate_left_.resize(batch_);
        host_aggregate_right_.resize(batch_);
        host_aggregate_t_left_.resize(batch_);
        host_aggregate_t_right_.resize(batch_);
        return true;
    }
    bool consumed() const override { return consumed_; }

    bool expand_level(int level, std::vector<U128> &aggregate_left,
                      std::vector<U128> &aggregate_right,
                      std::vector<uint8_t> &aggregate_t_left,
                      std::vector<uint8_t> &aggregate_t_right) override {
        if (level != level_ || level_ >= log_domain_ || width_ == 0 ||
            batch_ > capacity_ / width_) {
            return false;
        }
        const size_t block_bytes = batch_ * sizeof(AESBlock);
        const size_t uint_bytes = batch_ * sizeof(unsigned int);
        if (batch_ > std::numeric_limits<unsigned int>::max()) return false;
        gpu_detail::expand_party_frontier_kernel<<<
            static_cast<unsigned int>(batch_),
            gpu_detail::kTreeReductionThreads>>>(
            width_, batch_, current_seeds_, next_seeds_, next_control_,
            aggregate_left_, aggregate_right_, aggregate_t_left_,
            aggregate_t_right_, *gaes_);
        ++counters_.gpu_kernel_launches;
        if (!gpu_detail::cuda_success(cudaGetLastError())) return false;

        if (!gpu_detail::cuda_success(cudaMemcpy(
                host_aggregate_left_.data(), aggregate_left_, block_bytes,
                cudaMemcpyDeviceToHost)) ||
            !gpu_detail::cuda_success(cudaMemcpy(
                host_aggregate_right_.data(), aggregate_right_, block_bytes,
                cudaMemcpyDeviceToHost)) ||
            !gpu_detail::cuda_success(cudaMemcpy(
                host_aggregate_t_left_.data(), aggregate_t_left_, uint_bytes,
                cudaMemcpyDeviceToHost)) ||
            !gpu_detail::cuda_success(cudaMemcpy(
                host_aggregate_t_right_.data(), aggregate_t_right_, uint_bytes,
                cudaMemcpyDeviceToHost))) {
            return false;
        }
        counters_.gpu_d2h_bytes += 2 * block_bytes + 2 * uint_bytes;
        aggregate_left.assign(host_aggregate_left_.begin(),
                              host_aggregate_left_.end());
        aggregate_right.assign(host_aggregate_right_.begin(),
                               host_aggregate_right_.end());
        aggregate_t_left.resize(batch_);
        aggregate_t_right.resize(batch_);
        for (size_t tree = 0; tree < batch_; ++tree) {
            aggregate_t_left[tree] =
                static_cast<uint8_t>(host_aggregate_t_left_[tree] & 1);
            aggregate_t_right[tree] =
                static_cast<uint8_t>(host_aggregate_t_right_[tree] & 1);
        }
        return true;
    }

    bool apply_level_correction(
        int level, const std::vector<U128> &seed_cw,
        const std::vector<uint8_t> &t_left_cw,
        const std::vector<uint8_t> &t_right_cw) override {
        if (level != level_ || seed_cw.size() != batch_ ||
            t_left_cw.size() != batch_ || t_right_cw.size() != batch_ ||
            width_ > capacity_ / (2 * batch_)) {
            return false;
        }
        const size_t seed_bytes = batch_ * sizeof(AESBlock);
        if (!gpu_detail::cuda_success(cudaMemcpy(
                seed_cw_, seed_cw.data(), seed_bytes,
                cudaMemcpyHostToDevice)) ||
            !gpu_detail::cuda_success(cudaMemcpy(
                t_left_cw_, t_left_cw.data(), batch_,
                cudaMemcpyHostToDevice)) ||
            !gpu_detail::cuda_success(cudaMemcpy(
                t_right_cw_, t_right_cw.data(), batch_,
                cudaMemcpyHostToDevice))) {
            return false;
        }
        counters_.gpu_h2d_bytes += seed_bytes + 2 * batch_;
        const size_t children = batch_ * width_ * 2;
        constexpr unsigned int threads = 256;
        const size_t blocks_size = (children + threads - 1) / threads;
        if (blocks_size > std::numeric_limits<unsigned int>::max()) return false;
        gpu_detail::apply_party_correction_kernel<<<
            static_cast<unsigned int>(blocks_size), threads>>>(
            width_, batch_, current_control_, next_seeds_, next_control_, seed_cw_,
            t_left_cw_, t_right_cw_);
        ++counters_.gpu_kernel_launches;
        if (!gpu_detail::cuda_success(cudaGetLastError()) ||
            !gpu_detail::cuda_success(cudaDeviceSynchronize())) {
            return false;
        }
        ++counters_.level_synchronizations;
        std::swap(current_seeds_, next_seeds_);
        std::swap(current_control_, next_control_);
        width_ *= 2;
        ++level_;
        return true;
    }

    bool final_sums(std::vector<Word> &seed_sum,
                    std::vector<Word> &control_sum) override {
        if (level_ != log_domain_ || width_ != (size_t{1} << log_domain_)) {
            return false;
        }
        const size_t sum_bytes = batch_ * sizeof(Word);
        if (batch_ > std::numeric_limits<unsigned int>::max()) return false;
        gpu_detail::sum_party_leaves_kernel<<<
            static_cast<unsigned int>(batch_),
            gpu_detail::kTreeReductionThreads>>>(
            party_, width_, batch_, modulus_, current_seeds_, current_control_,
            seed_sum_, control_sum_);
        ++counters_.gpu_kernel_launches;
        if (!gpu_detail::cuda_success(cudaGetLastError())) return false;
        seed_sum.resize(batch_);
        control_sum.resize(batch_);
        if (!gpu_detail::cuda_success(cudaMemcpy(
                seed_sum.data(), seed_sum_, sum_bytes,
                cudaMemcpyDeviceToHost)) ||
            !gpu_detail::cuda_success(cudaMemcpy(
                control_sum.data(), control_sum_, sum_bytes,
                cudaMemcpyDeviceToHost))) {
            return false;
        }
        counters_.gpu_d2h_bytes += 2 * sum_bytes;
        return true;
    }

    void add_backend_counters(DpfStageCounters &out) const override {
        out.gpu_kernel_launches += counters_.gpu_kernel_launches;
        out.gpu_h2d_bytes += counters_.gpu_h2d_bytes;
        out.gpu_d2h_bytes += counters_.gpu_d2h_bytes;
        if (out.gpu_peak_bytes < counters_.gpu_peak_bytes) {
            out.gpu_peak_bytes = counters_.gpu_peak_bytes;
        }
        out.level_synchronizations += counters_.level_synchronizations;
    }

  private:
    bool allocate(void **address, size_t bytes) {
        if (bytes == 0 || !gpu_detail::cuda_success(cudaMalloc(address, bytes))) {
            return false;
        }
        allocated_bytes_ += bytes;
        counters_.gpu_peak_bytes = allocated_bytes_;
        return true;
    }

    void release() {
        cudaFree(current_seeds_);
        cudaFree(next_seeds_);
        cudaFree(current_control_);
        cudaFree(next_control_);
        cudaFree(aggregate_left_);
        cudaFree(aggregate_right_);
        cudaFree(aggregate_t_left_);
        cudaFree(aggregate_t_right_);
        cudaFree(seed_cw_);
        cudaFree(t_left_cw_);
        cudaFree(t_right_cw_);
        cudaFree(seed_sum_);
        cudaFree(control_sum_);
    }

    AESGlobalContext *gaes_ = nullptr;
    bool consumed_ = false;
    int party_ = -1;
    int log_domain_ = 0;
    int level_ = 0;
    Word modulus_ = 0;
    size_t batch_ = 0;
    size_t width_ = 0;
    size_t capacity_ = 0;
    size_t allocated_bytes_ = 0;
    AESBlock *current_seeds_ = nullptr;
    AESBlock *next_seeds_ = nullptr;
    uint8_t *current_control_ = nullptr;
    uint8_t *next_control_ = nullptr;
    AESBlock *aggregate_left_ = nullptr;
    AESBlock *aggregate_right_ = nullptr;
    unsigned int *aggregate_t_left_ = nullptr;
    unsigned int *aggregate_t_right_ = nullptr;
    AESBlock *seed_cw_ = nullptr;
    uint8_t *t_left_cw_ = nullptr;
    uint8_t *t_right_cw_ = nullptr;
    Word *seed_sum_ = nullptr;
    Word *control_sum_ = nullptr;
    std::vector<AESBlock> host_aggregate_left_;
    std::vector<AESBlock> host_aggregate_right_;
    std::vector<unsigned int> host_aggregate_t_left_;
    std::vector<unsigned int> host_aggregate_t_right_;
    DpfStageCounters counters_;
};

inline bool two_party_dpf_gen_batch_gpu_batched(
    int party, int log_domain, Word p, const std::vector<uint64_t> &offs,
    const std::vector<Word> &beta_factors, PartyChannel &ch, PartyRandom &rng,
    AESGlobalContext *gaes, std::vector<spfss_host::DPFKey> &keys,
    PhaseCOleSource *phase_c_ole_source,
    DpfStageCounters *stage_counters = nullptr) {
    if (gaes == nullptr) {
        keys.clear();
        if (stage_counters != nullptr) *stage_counters = DpfStageCounters{};
        return false;
    }
    GpuBatchedPartyTreeBatchState state(gaes);
    return two_party_dpf_gen_batch_with_state(
        party, log_domain, p, PrgMode::kGpuAes, offs, beta_factors, ch, rng,
        state, keys, phase_c_ole_source, stage_counters);
}

}  // namespace ringlpn_2pdpf
