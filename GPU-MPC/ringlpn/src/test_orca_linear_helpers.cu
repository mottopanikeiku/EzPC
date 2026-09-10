// Focused two-process regression for OrcaBase's extracted linear helpers.
// Stock trusted keygen supplies the test-only masks; the oracle is independent
// clear ring arithmetic. This is not Ring-LPN preprocessing evidence.

#include "backend/orca_base.h"
#include "utils/gpu_random.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Word = uint64_t;
constexpr int kBw = 32;
constexpr Word kMask = (Word{1} << kBw) - 1;

Word ring_add(Word left, Word right) {
    return (left + right) & kMask;
}

Word ring_sub(Word left, Word right) {
    return (left - right) & kMask;
}

template <typename T>
T *to_gpu(const std::vector<T> &values, Stats *stats = nullptr) {
    return reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(
            const_cast<T *>(values.data())),
        values.size() * sizeof(T), stats));
}

template <typename T>
std::vector<T> from_gpu(T *values, size_t count, Stats *stats = nullptr) {
    std::vector<T> host(count);
    moveIntoCPUMem(reinterpret_cast<uint8_t *>(host.data()),
                   reinterpret_cast<uint8_t *>(values), count * sizeof(T),
                   stats);
    return host;
}

std::vector<Word> additive_share(const std::vector<Word> &clear,
                                 int party, Word tag) {
    std::vector<Word> share(clear.size());
    for (size_t i = 0; i < clear.size(); ++i) {
        const Word party_zero =
            (tag + Word{0x9e3779b9} * static_cast<Word>(i + 1)) & kMask;
        share[i] = party == SERVER0 ? party_zero
                                    : ring_sub(clear[i], party_zero);
    }
    return share;
}

std::vector<Word> add_vectors(const std::vector<Word> &left,
                              const std::vector<Word> &right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("vector size mismatch");
    }
    std::vector<Word> out(left.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = ring_add(left[i], right[i]);
    }
    return out;
}

void configure_gpu_pool() {
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    cudaMemPool_t pool;
    checkCudaErrors(cudaDeviceGetDefaultMemPool(&pool, device));
    uint64_t threshold = UINT64_MAX;
    checkCudaErrors(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &threshold));
}

class HelperHarness final : public OrcaBase<Word> {
  public:
    HelperHarness(int party_value, const std::string &peer_ip) {
        configure_gpu_pool();
        party = party_value;
        bw = kBw;
        scale = 0;
        s.reset();
        peer = new GpuPeer(false);
        peer->connect(party, peer_ip);
    }

    ~HelperHarness() {
        if (peer != nullptr) {
            peer->close();
            delete peer;
            peer = nullptr;
        }
    }

    using OrcaBase<Word>::runConv2DWithKey;
    using OrcaBase<Word>::runMatmulWithKey;

    void finish() {
        if (peer != nullptr) peer->sync();
    }
};

bool run_matmul_case(HelperHarness &harness, int party) {
    constexpr int rows = 2;
    constexpr int inner = 3;
    constexpr int cols = 2;
    MatmulParams params;
    params.M = rows;
    params.K = inner;
    params.N = cols;
    params.batchSz = 1;
    stdInit(params, kBw, 0);

    // Canonical ring encodings include -1, INT32_MIN and wrapping products.
    const std::vector<Word> clear_input = {
        3, kMask, Word{1} << 31, 11, 13, 17};
    const std::vector<Word> clear_weight = {
        kMask - 1, 23, 29, Word{1} << 31, 37, 41};
    const std::vector<Word> input_mask = {101, 103, 107, 109, 113, 127};
    const std::vector<Word> weight_mask = {131, 137, 139, 149, 151, 157};
    const std::vector<Word> output_mask = {163, 167, 173, 179};
    const std::vector<Word> bias_values = {kMask - 180, 191};
    const std::vector<Word> masked_weight =
        add_vectors(clear_weight, weight_mask);

    const size_t key_bytes =
        static_cast<size_t>(params.size_A + params.size_B + params.size_C) *
        sizeof(Word);
    std::vector<uint8_t> key_storage(key_bytes, 0);
    uint8_t *cursor = key_storage.data();
    Word *d_input_mask = to_gpu(input_mask);
    Word *d_weight_mask = to_gpu(weight_mask);
    Word *d_output_mask = to_gpu(output_mask);
    Word *returned_mask = gpuKeygenMatmul<Word>(
        &cursor, party, params, d_input_mask, d_weight_mask, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
    gpuFree(d_input_mask);
    gpuFree(d_weight_mask);
    if (cursor != key_storage.data() + key_storage.size() ||
        returned_mask != d_output_mask) {
        gpuFree(d_output_mask);
        return false;
    }
    gpuFree(d_output_mask);

    cursor = key_storage.data();
    GPUMatmulKey<Word> key =
        readGPUMatmulKey<Word>(params, TruncateType::None, &cursor);
    if (cursor != key_storage.data() + key_storage.size()) return false;

    Tensor2D<Word> input(rows, inner);
    const std::vector<Word> input_share =
        additive_share(clear_input, party, 0x12345678);
    std::copy(input_share.begin(), input_share.end(), input.data);
    input.d_data = to_gpu(input_share, &harness.s);
    Word *d_online_weight = to_gpu(masked_weight, &harness.s);
    Tensor1D<Word> bias(cols);
    std::copy(bias_values.begin(), bias_values.end(), bias.data);
    Tensor2D<Word> output(rows, cols);

    harness.runMatmulWithKey(params, key, input, d_online_weight, true, bias,
                             output, true);
    const std::vector<Word> masked_output =
        from_gpu(output.d_data, output_mask.size(), &harness.s);
    gpuFree(input.d_data);
    input.d_data = nullptr;
    gpuFree(d_online_weight);
    gpuFree(output.d_data);
    output.d_data = nullptr;

    std::vector<Word> expected(output_mask.size(), 0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            unsigned __int128 accumulator = bias_values[col];
            for (int index = 0; index < inner; ++index) {
                accumulator +=
                    static_cast<unsigned __int128>(
                        clear_input[row * inner + index]) *
                    clear_weight[index * cols + col];
            }
            expected[row * cols + col] =
                static_cast<Word>(accumulator) & kMask;
        }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (ring_sub(masked_output[i], output_mask[i]) != expected[i]) {
            return false;
        }
    }
    return true;
}

size_t input_index(int n, int h, int w, int ci, int height, int width,
                   int channels) {
    return ((static_cast<size_t>(n) * height + h) * width + w) * channels +
           ci;
}

size_t filter_index(int co, int fh, int fw, int ci, int filter_h,
                    int filter_w, int channels) {
    return ((static_cast<size_t>(co) * filter_h + fh) * filter_w + fw) *
               channels +
           ci;
}

bool run_conv_case(HelperHarness &harness, int party) {
    constexpr int batch = 1;
    constexpr int height = 4;
    constexpr int width = 4;
    constexpr int channels = 1;
    constexpr int filter_h = 3;
    constexpr int filter_w = 3;
    constexpr int output_channels = 2;
    constexpr int padding = 1;
    constexpr int stride = 1;

    GPUConv2DKey<Word> key_template;
    key_template.p = {kBw,
                      kBw,
                      batch,
                      height,
                      width,
                      channels,
                      filter_h,
                      filter_w,
                      output_channels,
                      padding,
                      padding,
                      padding,
                      padding,
                      stride,
                      stride,
                      0,
                      0,
                      0,
                      0,
                      0};
    fillConv2DParams(&key_template.p);
    key_template.mem_size_I = key_template.p.size_I * sizeof(Word);
    key_template.mem_size_F = key_template.p.size_F * sizeof(Word);
    key_template.mem_size_O = key_template.p.size_O * sizeof(Word);

    std::vector<Word> clear_input(key_template.p.size_I);
    std::vector<Word> clear_filter(key_template.p.size_F);
    std::vector<Word> input_mask(key_template.p.size_I);
    std::vector<Word> filter_mask(key_template.p.size_F);
    std::vector<Word> output_mask(key_template.p.size_O);
    for (size_t i = 0; i < clear_input.size(); ++i) {
        clear_input[i] = i % 3 == 0 ? ring_sub(0, static_cast<Word>(i + 1))
                                     : static_cast<Word>(i + 1);
        input_mask[i] = static_cast<Word>(211 + 2 * i);
    }
    for (size_t i = 0; i < clear_filter.size(); ++i) {
        clear_filter[i] = i % 2 == 0
                              ? ring_sub(0, static_cast<Word>(3 + (i % 7)))
                              : static_cast<Word>(3 + (i % 7));
        filter_mask[i] = static_cast<Word>(307 + 3 * i);
    }
    for (size_t i = 0; i < output_mask.size(); ++i) {
        output_mask[i] = static_cast<Word>(401 + 5 * i);
    }
    const std::vector<Word> bias_values = {kMask - 16, Word{1} << 31};
    const std::vector<Word> masked_filter =
        add_vectors(clear_filter, filter_mask);

    const size_t key_bytes = key_template.mem_size_I +
                             key_template.mem_size_F +
                             key_template.mem_size_O;
    std::vector<uint8_t> key_storage(key_bytes, 0);
    uint8_t *cursor = key_storage.data();
    Word *d_input_mask = to_gpu(input_mask);
    Word *d_output_mask = to_gpu(output_mask);
    Word *returned_mask = gpuKeygenConv2D<Word>(
        &cursor, party, key_template, d_input_mask, filter_mask.data(), true,
        d_output_mask);
    gpuFree(d_input_mask);
    if (cursor != key_storage.data() + key_storage.size() ||
        returned_mask != d_output_mask) {
        gpuFree(d_output_mask);
        return false;
    }
    gpuFree(d_output_mask);

    GPUConv2DKey<Word> key = key_template;
    cursor = key_storage.data();
    key.I = reinterpret_cast<Word *>(cursor);
    cursor += key.mem_size_I;
    key.F = reinterpret_cast<Word *>(cursor);
    cursor += key.mem_size_F;
    key.O = reinterpret_cast<Word *>(cursor);
    cursor += key.mem_size_O;
    if (cursor != key_storage.data() + key_storage.size()) return false;

    Tensor4D<Word> input(batch, height, width, channels);
    const std::vector<Word> input_share =
        additive_share(clear_input, party, 0x76543210);
    std::copy(input_share.begin(), input_share.end(), input.data);
    input.d_data = to_gpu(input_share, &harness.s);
    Word *d_online_filter = to_gpu(masked_filter, &harness.s);
    Tensor1D<Word> bias(output_channels);
    std::copy(bias_values.begin(), bias_values.end(), bias.data);
    Tensor4D<Word> output(batch, key.p.OH, key.p.OW, output_channels);

    harness.runConv2DWithKey(key, input, d_online_filter, true, bias, output,
                             true);
    const std::vector<Word> masked_output =
        from_gpu(output.d_data, output_mask.size(), &harness.s);
    gpuFree(input.d_data);
    input.d_data = nullptr;
    gpuFree(d_online_filter);
    gpuFree(output.d_data);
    output.d_data = nullptr;

    std::vector<Word> expected(output_mask.size(), 0);
    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < key.p.OH; ++oh) {
            for (int ow = 0; ow < key.p.OW; ++ow) {
                for (int co = 0; co < output_channels; ++co) {
                    unsigned __int128 accumulator = bias_values[co];
                    for (int fh = 0; fh < filter_h; ++fh) {
                        const int ih = oh * stride + fh - padding;
                        if (ih < 0 || ih >= height) continue;
                        for (int fw = 0; fw < filter_w; ++fw) {
                            const int iw = ow * stride + fw - padding;
                            if (iw < 0 || iw >= width) continue;
                            for (int ci = 0; ci < channels; ++ci) {
                                accumulator +=
                                    static_cast<unsigned __int128>(clear_input[
                                        input_index(n, ih, iw, ci, height,
                                                    width, channels)]) *
                                    clear_filter[filter_index(
                                        co, fh, fw, ci, filter_h, filter_w,
                                        channels)];
                            }
                        }
                    }
                    const size_t index =
                        ((static_cast<size_t>(n) * key.p.OH + oh) * key.p.OW +
                         ow) *
                            output_channels +
                        co;
                    expected[index] = static_cast<Word>(accumulator) & kMask;
                }
            }
        }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (ring_sub(masked_output[i], output_mask[i]) != expected[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s PARTY PEER_IP\n", argv[0]);
        return 2;
    }
    const int party = std::atoi(argv[1]);
    if ((party != SERVER0 && party != SERVER1) || argv[2][0] == '\0') {
        std::fprintf(stderr, "[orca-linear-helpers] invalid arguments\n");
        return 2;
    }

    try {
        OneGB = size_t{2} << 20;
        HelperHarness harness(party, argv[2]);
        initGPURandomness();
        const bool matmul_ok = run_matmul_case(harness, party);
        const bool conv_ok = run_conv_case(harness, party);
        destroyGPURandomness();
        harness.finish();
        if (!matmul_ok || !conv_ok) {
            std::fprintf(stderr,
                         "[orca-linear-helpers] matmul=%d conv2d=%d\n",
                         matmul_ok ? 1 : 0, conv_ok ? 1 : 0);
            return 1;
        }
        std::puts("orca-linear-helpers,pass");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[orca-linear-helpers] %s\n", error.what());
        return 1;
    }
}
