#pragma once

// One-record terminal Orca backend. It consumes one party-local bound
// record/state pair and rejects every graph callback except its matching linear
// operation followed by output. Arbitrary graph mask chaining is intentionally
// outside this boundary.
#include "linear_preprocess.h"

#include "backend/orca.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ringlpn_orca {
namespace terminal_detail {

constexpr size_t kPreflightBytes = 288;
constexpr std::array<uint8_t, 8> kPreflightMagic = {
    'R', 'L', 'P', 'O', 'R', 'C', 'A', '1'};
constexpr uint32_t kPreflightVersion = 1;

inline void put_u32(std::array<uint8_t, kPreflightBytes> &out, size_t offset,
                    uint32_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline void put_u64(std::array<uint8_t, kPreflightBytes> &out, size_t offset,
                    uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline uint32_t get_u32(
    const std::array<uint8_t, kPreflightBytes> &in, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint32_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

inline bool same_protocol(const ringlpn_linear::ProtocolParameters &left,
                          const ringlpn_linear::ProtocolParameters &right) {
    return left.qbits == right.qbits && left.bw == right.bw &&
           left.ole_n == right.ole_n && left.ole_c == right.ole_c &&
           left.ole_t == right.ole_t &&
           left.regular_noise == right.regular_noise &&
           left.channel == right.channel &&
           left.ot_backend == right.ot_backend;
}

inline bool same_plan(const ringlpn_linear::LinearPlan &left,
                      const ringlpn_linear::LinearPlan &right) {
    if (left.kind != right.kind ||
        !same_protocol(left.protocol, right.protocol) ||
        left.input_words != right.input_words ||
        left.weight_words != right.weight_words ||
        left.output_words != right.output_words ||
        left.cross_terms != right.cross_terms ||
        left.ring_batches != right.ring_batches ||
        left.ring_application_slots != right.ring_application_slots ||
        left.ring_bootstrap_slots != right.ring_bootstrap_slots ||
        left.output_h != right.output_h ||
        left.output_w != right.output_w) {
        return false;
    }
    if (left.kind == ringlpn_linear::LinearKind::Fc) {
        return left.fc.rows == right.fc.rows &&
               left.fc.inner == right.fc.inner &&
               left.fc.cols == right.fc.cols;
    }
    return left.conv.n == right.conv.n && left.conv.h == right.conv.h &&
           left.conv.w == right.conv.w && left.conv.ci == right.conv.ci &&
           left.conv.fh == right.conv.fh &&
           left.conv.fw == right.conv.fw && left.conv.co == right.conv.co &&
           left.conv.padding == right.conv.padding &&
           left.conv.stride == right.conv.stride;
}

inline bool digest_nonzero(const ringlpn_linear::Digest &digest) {
    return std::any_of(digest.begin(), digest.end(),
                       [](uint8_t byte) { return byte != 0; });
}

inline std::array<uint8_t, kPreflightBytes> encode_preflight(
    int party, bool local_valid,
    const ringlpn_linear::LayerExpectation &expected,
    const ringlpn_linear::OwnedLayerMaterial &material) {
    std::array<uint8_t, kPreflightBytes> out{};
    std::copy(kPreflightMagic.begin(), kPreflightMagic.end(), out.begin());
    put_u32(out, 8, kPreflightVersion);
    put_u32(out, 12, static_cast<uint32_t>(party));
    put_u32(out, 16, local_valid ? 1U : 0U);
    put_u32(out, 20,
            expected.plan.kind == ringlpn_linear::LinearKind::Fc ? 1U : 2U);
    put_u32(out, 24, static_cast<uint32_t>(expected.plan.protocol.qbits));
    put_u32(out, 28, static_cast<uint32_t>(expected.plan.protocol.bw));
    put_u32(out, 32, static_cast<uint32_t>(expected.plan.protocol.ole_n));
    put_u32(out, 36, static_cast<uint32_t>(expected.plan.protocol.ole_c));
    put_u32(out, 40, static_cast<uint32_t>(expected.plan.protocol.ole_t));
    put_u32(out, 44, expected.plan.protocol.regular_noise ? 1U : 0U);
    put_u32(out, 48,
            expected.plan.protocol.ot_backend == "sci-iknp" ? 0U : 1U);
    put_u64(out, 56, expected.sid);
    put_u64(out, 64, expected.layer_ordinal);
    put_u64(out, 72, expected.plan.input_words);
    put_u64(out, 80, expected.plan.weight_words);
    put_u64(out, 88, expected.plan.output_words);
    put_u64(out, 96, expected.plan.cross_terms);
    put_u64(out, 104, expected.plan.ring_batches);
    put_u64(out, 112, expected.plan.ring_application_slots);
    put_u64(out, 120, expected.plan.ring_bootstrap_slots);
    put_u32(out, 128, static_cast<uint32_t>(expected.plan.output_h));
    put_u32(out, 132, static_cast<uint32_t>(expected.plan.output_w));
    std::array<uint64_t, 9> shape{};
    if (expected.plan.kind == ringlpn_linear::LinearKind::Fc) {
        shape[0] = static_cast<uint64_t>(expected.plan.fc.rows);
        shape[1] = static_cast<uint64_t>(expected.plan.fc.inner);
        shape[2] = static_cast<uint64_t>(expected.plan.fc.cols);
    } else {
        shape = {static_cast<uint64_t>(expected.plan.conv.n),
                 static_cast<uint64_t>(expected.plan.conv.h),
                 static_cast<uint64_t>(expected.plan.conv.w),
                 static_cast<uint64_t>(expected.plan.conv.ci),
                 static_cast<uint64_t>(expected.plan.conv.fh),
                 static_cast<uint64_t>(expected.plan.conv.fw),
                 static_cast<uint64_t>(expected.plan.conv.co),
                 static_cast<uint64_t>(expected.plan.conv.padding),
                 static_cast<uint64_t>(expected.plan.conv.stride)};
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        put_u64(out, 136 + i * sizeof(uint64_t), shape[i]);
    }
    std::copy(expected.invocation_id.begin(), expected.invocation_id.end(),
              out.begin() + 208);
    if (!material.empty()) {
        std::copy(material.layer_identity().begin(),
                  material.layer_identity().end(), out.begin() + 224);
        std::copy(material.record_metadata().ledger_digest.begin(),
                  material.record_metadata().ledger_digest.end(),
                  out.begin() + 256);
    }
    return out;
}

inline bool common_preflight_matches(
    const std::array<uint8_t, kPreflightBytes> &local,
    const std::array<uint8_t, kPreflightBytes> &remote) {
    return std::equal(local.begin(), local.begin() + 12, remote.begin()) &&
           std::equal(local.begin() + 20, local.end(), remote.begin() + 20);
}

}  // namespace terminal_detail

template <typename T>
class TerminalLinearBackend final : public Orca<T> {
    static_assert(std::is_same_v<T, uint64_t>,
                  "Ring-LPN linear records contain uint64_t words");

  public:
    TerminalLinearBackend(
        int party, const std::string &peer_ip, int bw,
        ringlpn_linear::LayerExpectation expected,
        ringlpn_linear::OwnedLayerMaterial material, bool local_valid)
        : Orca<T>(), expected_(std::move(expected)),
          material_(std::move(material)) {
        this->party = party;
        this->bw = bw;
        this->scale = 0;
        this->s.reset();

        bool effective_valid =
            local_valid && (party == SERVER0 || party == SERVER1) &&
            !peer_ip.empty() && bw == expected_.plan.protocol.bw &&
            expected_.party == party && expected_.sid != 0 &&
            expected_.layer_ordinal != 0 && expected_.require_invocation &&
            expected_.require_digests && !material_.empty() &&
            terminal_detail::same_plan(material_.plan(), expected_.plan) &&
            material_.record_metadata().party == party &&
            material_.record_metadata().sid == expected_.sid &&
            material_.record_metadata().invocation_id ==
                expected_.invocation_id &&
            material_.record_metadata().digest == expected_.record_digest &&
            material_.state_digest() == expected_.state_digest &&
            material_.layer_ordinal() == expected_.layer_ordinal &&
            terminal_detail::digest_nonzero(material_.layer_identity()) &&
            terminal_detail::digest_nonzero(
                material_.record_metadata().ledger_digest);
        const ringlpn_linear::Status gpu_status =
            expected_.plan.kind == ringlpn_linear::LinearKind::Fc
                ? ringlpn_linear::FcPreprocessor::initialize_gpu()
                : ringlpn_linear::Conv2dPreprocessor::initialize_gpu();
        effective_valid = effective_valid &&
                          gpu_status == ringlpn_linear::Status::Ok;

        OneGB = size_t{2} << 20;
        this->peer = new GpuPeer(false);
        this->peer->connect(party, peer_ip);

        const auto local = terminal_detail::encode_preflight(
            party, effective_valid, expected_, material_);
        std::array<uint8_t, terminal_detail::kPreflightBytes> remote{};
        if (party == SERVER0) {
            this->peer->sendBytes(local.data(), local.size());
            this->peer->recvBytes(remote.data(), remote.size());
        } else {
            this->peer->recvBytes(remote.data(), remote.size());
            this->peer->sendBytes(local.data(), local.size());
        }
        const uint32_t remote_party = terminal_detail::get_u32(remote, 12);
        const uint32_t remote_valid = terminal_detail::get_u32(remote, 16);
        if (!effective_valid || remote_party != static_cast<uint32_t>(1 - party) ||
            remote_valid != 1 ||
            !terminal_detail::common_preflight_matches(local, remote)) {
            material_.reset();
            shutdown_peer();
            throw std::runtime_error(
                "Ring-LPN bilateral linear preflight rejected");
        }
    }

    ~TerminalLinearBackend() {
        if (!material_.empty()) {
            std::fprintf(stderr,
                         "[ringlpn-orca] unconsumed terminal material\n");
            material_.reset();
            shutdown_peer();
            std::abort();
        }
        shutdown_peer();
    }

    TerminalLinearBackend(const TerminalLinearBackend &) = delete;
    TerminalLinearBackend &operator=(const TerminalLinearBackend &) = delete;

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co,
                const Tensor4D<T> &input, const Tensor2D<T> &filter,
                Tensor4D<T> &output, bool is_first) override {
        Tensor1D<T> unused_bias(co);
        conv2D(fh, fw, padding, stride, ci, co, input, filter, false,
               unused_bias, output, is_first);
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co,
                const Tensor4D<T> &input, const Tensor2D<T> &filter,
                bool use_bias, const Tensor1D<T> &bias,
                Tensor4D<T> &output, bool is_first) override {
        const auto comm_start = this->s.comm_time;
        const auto start = std::chrono::high_resolution_clock::now();
        const auto &plan = expected_.plan;
        if (linear_consumed_ || material_.empty() ||
            plan.kind != ringlpn_linear::LinearKind::Conv2d || !is_first ||
            fh != static_cast<u64>(plan.conv.fh) ||
            fw != static_cast<u64>(plan.conv.fw) ||
            padding != static_cast<u64>(plan.conv.padding) ||
            stride != static_cast<u64>(plan.conv.stride) ||
            ci != static_cast<u64>(plan.conv.ci) ||
            co != static_cast<u64>(plan.conv.co) ||
            input.d1 != static_cast<u64>(plan.conv.n) ||
            input.d2 != static_cast<u64>(plan.conv.h) ||
            input.d3 != static_cast<u64>(plan.conv.w) ||
            input.d4 != static_cast<u64>(plan.conv.ci) ||
            filter.d1 != static_cast<u64>(plan.conv.co) ||
            filter.d2 != static_cast<u64>(plan.conv.fh * plan.conv.fw *
                                          plan.conv.ci) ||
            output.d1 != static_cast<u64>(plan.conv.n) ||
            output.d2 != static_cast<u64>(plan.output_h) ||
            output.d3 != static_cast<u64>(plan.output_w) ||
            output.d4 != static_cast<u64>(plan.conv.co) ||
            (use_bias && bias.d1 != static_cast<u64>(plan.conv.co))) {
            reject("Conv2D callback does not match bound material");
        }

        GPUConv2DKey<T> key;
        key.p = {this->bw,
                 this->bw,
                 plan.conv.n,
                 plan.conv.h,
                 plan.conv.w,
                 plan.conv.ci,
                 plan.conv.fh,
                 plan.conv.fw,
                 plan.conv.co,
                 plan.conv.padding,
                 plan.conv.padding,
                 plan.conv.padding,
                 plan.conv.padding,
                 plan.conv.stride,
                 plan.conv.stride,
                 plan.output_h,
                 plan.output_w,
                 static_cast<size_t>(plan.input_words),
                 static_cast<size_t>(plan.weight_words),
                 static_cast<size_t>(plan.output_words)};
        const auto input_mask = material_.input_mask_share();
        const auto weight_mask = material_.weight_mask_share();
        const auto correction = material_.output_correction_share();
        if (input_mask.size != key.p.size_I ||
            weight_mask.size != key.p.size_F ||
            correction.size != key.p.size_O) {
            reject("Conv2D material length mismatch");
        }
        key.mem_size_I = key.p.size_I * sizeof(T);
        key.mem_size_F = key.p.size_F * sizeof(T);
        key.mem_size_O = key.p.size_O * sizeof(T);
        key.I = const_cast<T *>(reinterpret_cast<const T *>(input_mask.data));
        key.F = const_cast<T *>(reinterpret_cast<const T *>(weight_mask.data));
        key.O = const_cast<T *>(reinterpret_cast<const T *>(correction.data));

        reconstruct_masked_input(input.d_data, key.p.size_I, input_mask);
        T *d_online_filter = reconstruct_masked_weight(
            filter.data, key.p.size_F, weight_mask);
        this->runConv2DWithKey(key, input, d_online_filter, use_bias, bias,
                               output, false);
        gpuFree(d_online_filter);
        linear_consumed_ = true;
        terminal_output_ = output.d_data;
        this->s.conv_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
        this->s.conv_comm_time += this->s.comm_time - comm_start;
    }

    void matmul(const Tensor2D<T> &, const Tensor2D<T> &,
                Tensor2D<T> &) override {
        reject("terminal Ring-LPN backend requires the Sytorch callback");
    }

    void matmul(const Tensor2D<T> &input, const Tensor2D<T> &weight,
                Tensor2D<T> &output, bool use_bias, Tensor1D<T> &bias,
                bool is_first) override {
        const auto comm_start = this->s.comm_time;
        const auto start = std::chrono::high_resolution_clock::now();
        const auto &plan = expected_.plan;
        if (linear_consumed_ || material_.empty() ||
            plan.kind != ringlpn_linear::LinearKind::Fc || !is_first ||
            input.d1 != static_cast<u64>(plan.fc.rows) ||
            input.d2 != static_cast<u64>(plan.fc.inner) ||
            weight.d1 != static_cast<u64>(plan.fc.inner) ||
            weight.d2 != static_cast<u64>(plan.fc.cols) ||
            output.d1 != static_cast<u64>(plan.fc.rows) ||
            output.d2 != static_cast<u64>(plan.fc.cols) ||
            (use_bias && bias.d1 != static_cast<u64>(plan.fc.cols))) {
            reject("matmul callback does not match bound material");
        }

        MatmulParams params;
        params.M = plan.fc.rows;
        params.K = plan.fc.inner;
        params.N = plan.fc.cols;
        params.batchSz = 1;
        stdInit(params, this->bw, 0);
        const auto input_mask = material_.input_mask_share();
        const auto weight_mask = material_.weight_mask_share();
        const auto correction = material_.output_correction_share();
        if (input_mask.size != static_cast<size_t>(params.size_A) ||
            weight_mask.size != static_cast<size_t>(params.size_B) ||
            correction.size != static_cast<size_t>(params.size_C)) {
            reject("matmul material length mismatch");
        }
        GPUMatmulKey<T> key;
        key.mem_size_A = params.size_A * sizeof(T);
        key.mem_size_B = params.size_B * sizeof(T);
        key.mem_size_C = params.size_C * sizeof(T);
        key.A = const_cast<T *>(reinterpret_cast<const T *>(input_mask.data));
        key.B = const_cast<T *>(reinterpret_cast<const T *>(weight_mask.data));
        key.C = const_cast<T *>(reinterpret_cast<const T *>(correction.data));

        reconstruct_masked_input(input.d_data, params.size_A, input_mask);
        T *d_online_weight = reconstruct_masked_weight(
            weight.data, params.size_B, weight_mask);
        this->runMatmulWithKey(params, key, input, d_online_weight, use_bias,
                               bias, output, false);
        gpuFree(d_online_weight);
        linear_consumed_ = true;
        terminal_output_ = output.d_data;
        this->s.matmul_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
        this->s.matmul_comm_time += this->s.comm_time - comm_start;
    }

    void output(Tensor<T> &value) override {
        if (!linear_consumed_ || output_consumed_ || material_.empty() ||
            value.data == nullptr || value.d_data == nullptr ||
            value.d_data != terminal_output_ ||
            value.size() != expected_.plan.output_words) {
            reject("output requires one matching terminal linear callback");
        }
        const auto output_mask = material_.output_mask_share();
        if (output_mask.size != static_cast<size_t>(value.size())) {
            reject("terminal output-mask length mismatch");
        }
        T *d_output_mask = reinterpret_cast<T *>(moveToGPU(
            reinterpret_cast<uint8_t *>(
                const_cast<uint64_t *>(output_mask.data)),
            output_mask.size * sizeof(T), &this->s));
        if (this->party == SERVER0) {
            gpuLinearComb(this->bw, value.size(), value.d_data, T(1),
                          value.d_data, -T(1), d_output_mask);
        } else {
            gpuLinearComb(this->bw, value.size(), value.d_data, T(0),
                          value.d_data, -T(1), d_output_mask);
        }
        gpuFree(d_output_mask);
        this->peer->reconstructInPlace(value.d_data, this->bw, value.size(),
                                       &this->s);
        moveIntoCPUMem(reinterpret_cast<uint8_t *>(value.data),
                       reinterpret_cast<uint8_t *>(value.d_data),
                       value.size() * sizeof(T), &this->s);
        material_.reset();
        output_consumed_ = true;
        terminal_output_ = nullptr;
    }

    void truncateForward(Tensor<T> &, u64, u8 = 0) override {
        reject("terminal Ring-LPN backend rejects truncation");
    }

    void relu(Tensor<T> &, Tensor<T> &, const Tensor<T> &, u64,
              int) override {
        reject("terminal Ring-LPN backend rejects ReLU");
    }

    void maxPool2D(u64, u64, u64, const Tensor4D<T> &, Tensor4D<T> &,
                   Tensor4D<u64> &, u64, u8) override {
        reject("terminal Ring-LPN backend rejects MaxPool2D");
    }

    void avgPool2D(u64, u64, u64, const Tensor4D<T> &, Tensor4D<T> &,
                   u64) override {
        reject("terminal Ring-LPN backend rejects AvgPool2D");
    }

    void sumPool2D(u64, u64, u64, const Tensor4D<T> &,
                   Tensor4D<T> &) override {
        reject("terminal Ring-LPN backend rejects SumPool2D");
    }

    void addbias(Tensor<T> &, const Tensor1D<T> &) override {
        reject("terminal Ring-LPN backend rejects standalone bias");
    }

    void rotary_embedding(Tensor<T> &, Tensor<T> &, u64, u64 = 2048,
                          u64 = 10000) override {
        reject("terminal Ring-LPN backend rejects rotary embedding");
    }

    void mha(int, int, int, bool, bool, bool, const Tensor2D<T> &,
             const Tensor1D<T> &, const Tensor2D<T> &, const Tensor1D<T> &,
             const Tensor2D<T> &, Tensor2D<T> &) override {
        reject("terminal Ring-LPN backend rejects attention");
    }

    void add(const std::vector<Tensor<T> *> &, Tensor<T> &) override {
        reject("terminal Ring-LPN backend rejects residual add");
    }

    void signext(Tensor<T> &, u64) override {
        reject("terminal Ring-LPN backend rejects sign extension");
    }

    void close() override {
        if (!output_consumed_) {
            reject("terminal Ring-LPN backend closed before output");
        }
        shutdown_peer();
    }

  private:
    [[noreturn]] void reject(const char *message) {
        material_.reset();
        throw std::runtime_error(message);
    }

    void shutdown_peer() noexcept {
        if (this->peer != nullptr) {
            this->peer->close();
            delete this->peer;
            this->peer = nullptr;
        }
    }

    void reconstruct_masked_input(
        T *d_input_share, size_t words,
        ringlpn_linear::WordView input_mask) {
        if (d_input_share == nullptr || input_mask.size != words) {
            reject("terminal input share length mismatch");
        }
        T *d_mask = reinterpret_cast<T *>(moveToGPU(
            reinterpret_cast<uint8_t *>(
                const_cast<uint64_t *>(input_mask.data)),
            words * sizeof(T), &this->s));
        gpuLinearComb(this->bw, words, d_input_share, T(1), d_input_share,
                      T(1), d_mask);
        gpuFree(d_mask);
        this->peer->reconstructInPlace(d_input_share, this->bw, words,
                                       &this->s);
    }

    T *reconstruct_masked_weight(
        const T *local_weight_share, size_t words,
        ringlpn_linear::WordView weight_mask) {
        if (local_weight_share == nullptr || weight_mask.size != words) {
            reject("terminal weight share length mismatch");
        }
        T *d_online_weight = reinterpret_cast<T *>(moveToGPU(
            reinterpret_cast<uint8_t *>(const_cast<T *>(local_weight_share)),
            words * sizeof(T), &this->s));
        T *d_mask = reinterpret_cast<T *>(moveToGPU(
            reinterpret_cast<uint8_t *>(
                const_cast<uint64_t *>(weight_mask.data)),
            words * sizeof(T), &this->s));
        gpuLinearComb(this->bw, words, d_online_weight, T(1), d_online_weight,
                      T(1), d_mask);
        gpuFree(d_mask);
        this->peer->reconstructInPlace(d_online_weight, this->bw, words,
                                       &this->s);
        return d_online_weight;
    }

    ringlpn_linear::LayerExpectation expected_;
    ringlpn_linear::OwnedLayerMaterial material_;
    T *terminal_output_ = nullptr;
    bool linear_consumed_ = false;
    bool output_consumed_ = false;
};

}  // namespace ringlpn_orca
