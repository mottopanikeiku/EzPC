// TEST-ONLY trusted compatibility adapter for the exact full ResNet18 graph.
//
// The adapter reads both parties' 21 source-bound mask-state records, samples
// the live secure-truncation successor masks, and generates Orca's unchanged
// MaxPool/ReLU/sign-extension key bytes.  It emits one private, digest-bound,
// memory-mappable record per party.  This process sees both parties' masks and
// is deliberately not dealerless nonlinear preprocessing, private inference,
// or concrete-security evidence.

#define BUF_MEM LLAMA_BUF_MEM
#include "utils/gpu_comms.h"
#undef BUF_MEM
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/helper_cuda.h"

#include "fss/dcf/gpu_maxpool.h"
#include "fss/dcf/gpu_relu.h"
#include "fss/dcf/gpu_truncate.h"
#include "graph_mask_state.h"
#include "resnet18_graph_contract.h"
#include "stock_nonlinear_full_record.h"

#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace contract = ringlpn_resnet18;
namespace record = ringlpn_full_graph_record;
using T = uint64_t;
using Digest = ringlpn_freshness::Digest;
using MaskState = ringlpn_graph::MaskStateRecord<T>;
using Clock = std::chrono::steady_clock;

constexpr size_t kGuardBytes = 4096;
constexpr uint8_t kGuardValue = 0xa7;
constexpr size_t kIoChunkBytes = size_t(1) << 20;
constexpr uint64_t kFullMask = std::numeric_limits<uint32_t>::max();

struct Args {
    bool csv_header = false;
    bool publication_control = false;
    int gpu = 0;
    std::string invocation_text;
    ringlpn_freshness::InvocationId invocation{};
    Digest manifest_digest{};
    Digest record_set_digest{};
    std::string record_set_root;
    std::string p0_output;
    std::string p1_output;
    std::array<ringlpn_graph::LinearArtifactBindings,
               contract::kLinearCount> linear_artifacts;
};

struct GeneratedKey {
    std::unique_ptr<uint8_t[]> raw;
    size_t raw_bytes = 0;
    std::vector<T> output_mask;
};

struct RawKeyMeta {
    uint64_t bytes = 0;
    Digest digest{};
};

struct PartyInputs {
    std::array<MaskState, contract::kLinearCount> states;
    std::array<Digest, contract::kLinearCount> record_digests{};
    std::array<Digest, contract::kLinearCount> state_digests{};
};

bool parse_hex(const std::string &text, uint8_t *out, size_t bytes) {
    if (text.size() != 2 * bytes) return false;
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < bytes; ++i) {
        const int high = nibble(text[2 * i]);
        const int low = nibble(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        out[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

bool parse_int(const std::string &text, int &out) {
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used, 10);
        if (used != text.size() || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            return false;
        }
        out = static_cast<int>(value);
        return true;
    } catch (...) {
        return false;
    }
}


bool artifact_arguments_complete(const Args &args) {
    return std::all_of(
        args.linear_artifacts.begin(), args.linear_artifacts.end(),
        [](const ringlpn_graph::LinearArtifactBindings &bindings) {
            return ringlpn_graph::artifact_bindings_complete(bindings);
        });
}
bool parse_args(int argc, char **argv, Args &args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char * {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        const char *value = nullptr;
        if (key == "--csv-header") {
            args.csv_header = true;
        } else if (key == "--publication-control") {
            args.publication_control = true;
        } else if (key == "--gpu" && (value = next())) {
            if (!parse_int(value, args.gpu)) return false;
        } else if (key == "--invocation" && (value = next())) {
            args.invocation_text = value;
        } else if (key == "--manifest-digest" && (value = next())) {
            if (!parse_hex(value, args.manifest_digest.data(),
                           args.manifest_digest.size())) {
                return false;
            }
        } else if (key == "--record-set-digest" && (value = next())) {
            if (!parse_hex(value, args.record_set_digest.data(),
                           args.record_set_digest.size())) {
                return false;
            }
        } else if (key == "--record-set-root" && (value = next())) {
            args.record_set_root = value;
        } else if (key == "--linear-artifact") {
            if (i + 5 >= argc) return false;
            int order = 0;
            const std::string order_text = argv[++i];
            const std::string label = argv[++i];
            const std::string path = argv[++i];
            const std::string bytes = argv[++i];
            const std::string sha256 = argv[++i];
            if (!parse_int(order_text, order) || order <= 0 ||
                static_cast<size_t>(order) > contract::kLinearCount ||
                !ringlpn_graph::set_artifact_binding(
                    args.linear_artifacts[static_cast<size_t>(order - 1)],
                    label, path, bytes, sha256)) {
                return false;
            }
        } else if (key == "--p0-output" && (value = next())) {
            args.p0_output = value;
        } else if (key == "--p1-output" && (value = next())) {
            args.p1_output = value;
        } else {
            return false;
        }
    }
    const bool outputs_valid = !args.p0_output.empty() &&
                               !args.p1_output.empty() &&
                               args.p0_output != args.p1_output;
    if (args.publication_control) {
        return outputs_valid && args.record_set_root.empty() &&
               args.invocation_text.empty() &&
               !record::nonzero(args.manifest_digest.data(),
                                args.manifest_digest.size()) &&
               !record::nonzero(args.record_set_digest.data(),
                                args.record_set_digest.size());
    }
    return args.gpu >= 0 && outputs_valid && !args.record_set_root.empty() &&
           artifact_arguments_complete(args) &&
           ringlpn_freshness::parse_invocation_id(args.invocation_text,
                                                  args.invocation) &&
           record::nonzero(args.manifest_digest.data(),
                           args.manifest_digest.size()) &&
           record::nonzero(args.record_set_digest.data(),
                           args.record_set_digest.size()) &&
           contract::contract_valid();
}

std::string hex_digest(const Digest &digest) {
    return ringlpn_freshness::hex(digest);
}

size_t packed_bytes(int bits, size_t count) {
    if (bits <= 0 || count == 0 ||
        count > (std::numeric_limits<size_t>::max() - PACKING_SIZE) /
                    static_cast<size_t>(bits)) {
        return 0;
    }
    return ((static_cast<size_t>(bits) * count - 1) / PACKING_SIZE + 1) *
           sizeof(PACK_TYPE);
}

size_t dcf_key_bytes(int bin, int bout, size_t count) {
    const int elems_per_block = AES_BLOCK_LEN_IN_BITS / bout;
    int packed_levels = 0;
    for (int value = elems_per_block; value > 1; value >>= 1) {
        ++packed_levels;
    }
    const int new_bin = bin - packed_levels;
    if (new_bin <= 1) return 0;
    return 7 * sizeof(int) +
           count * static_cast<size_t>(new_bin) * sizeof(AESBlock) +
           2 * count * sizeof(AESBlock) +
           packed_bytes(bout, count) * static_cast<size_t>(new_bin - 1);
}

size_t expected_maxpool_key_bytes() {
    const size_t n = static_cast<size_t>(
        contract::kStockKeySpecs[0].output_words);
    const size_t one_relu = 3 * sizeof(int) +
                            dcf_key_bytes(contract::kTruncatedBw, 1, n) +
                            packed_bytes(1, n) + 5 * n * sizeof(T);
    return 8 * one_relu;
}

size_t expected_relu_key_bytes(size_t count) {
    return 3 * sizeof(int) +
           dcf_key_bytes(contract::kTruncatedBw, 2, count) +
           2 * packed_bytes(2, count) + 6 * count * sizeof(T);
}

size_t expected_signextend_key_bytes(size_t count) {
    return 3 * sizeof(int) +
           dcf_key_bytes(contract::kTruncatedBw, 1, count) +
           packed_bytes(1, count) + 3 * count * sizeof(T);
}

bool configure_gpu_pool(int gpu) {
    if (cudaSetDevice(gpu) != cudaSuccess) return false;
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, gpu) != cudaSuccess) return false;
    uint64_t threshold = UINT64_MAX;
    return cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                   &threshold) == cudaSuccess;
}

bool fill_random(std::vector<T> &values, int bits) {
    if (values.empty() || bits <= 0 || bits > 32 ||
        values.size() > static_cast<size_t>(std::numeric_limits<int>::max()) /
                            sizeof(T) ||
        RAND_priv_bytes(reinterpret_cast<unsigned char *>(values.data()),
                        static_cast<int>(values.size() * sizeof(T))) != 1) {
        return false;
    }
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    for (T &value : values) value &= mask;
    return true;
}

bool fill_random(Digest &value) {
    return RAND_priv_bytes(value.data(), static_cast<int>(value.size())) == 1 &&
           record::nonzero(value.data(), value.size());
}

std::vector<T> add_masks(const std::vector<T> &left,
                         const std::vector<T> &right, int bits) {
    if (left.size() != right.size() || bits <= 0 || bits > 32) return {};
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    std::vector<T> result(left.size());
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] = (left[i] + right[i]) & mask;
    }
    return result;
}

bool split_mask(const std::vector<T> &full, int bits,
                std::vector<T> &share0, std::vector<T> &share1) {
    share0.resize(full.size());
    share1.resize(full.size());
    if (!fill_random(share0, bits)) return false;
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    for (size_t i = 0; i < full.size(); ++i) {
        share1[i] = (full[i] - share0[i]) & mask;
    }
    return true;
}

MaxpoolParams exact_maxpool_params() {
    MaxpoolParams params = {
        contract::kTruncatedBw, contract::kTruncatedBw, 0, 0,
        contract::kFullBw,
        1, 112, 112, 64,
        3, 3,
        2, 2,
        1, 1,
        1, 1,
        0, 0, false};
    initPoolParams(params);
    return params;
}

bool guarded_buffer(size_t bytes, GeneratedKey &generated) {
    if (bytes == 0 || bytes > record::kMaxRecordBytes - kGuardBytes) {
        return false;
    }
    generated.raw.reset(new (std::nothrow) uint8_t[bytes + kGuardBytes]);
    if (!generated.raw) return false;
    std::memset(generated.raw.get() + bytes, kGuardValue, kGuardBytes);
    generated.raw_bytes = bytes;
    return true;
}

bool guard_ok(const GeneratedKey &generated) {
    return generated.raw &&
           std::all_of(generated.raw.get() + generated.raw_bytes,
                       generated.raw.get() + generated.raw_bytes + kGuardBytes,
                       [](uint8_t value) { return value == kGuardValue; });
}

bool generate_maxpool(int party, const std::vector<T> &input_mask,
                      GeneratedKey &generated) {
    const size_t expected = expected_maxpool_key_bytes();
    if (input_mask.size() != contract::kStockKeySpecs[0].input_words ||
        !guarded_buffer(expected, generated)) {
        return false;
    }
    initGPURandomness();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(input_mask.data())),
        input_mask.size() * sizeof(T), nullptr));
    uint8_t *cursor = generated.raw.get();
    MaxpoolParams params = exact_maxpool_params();
    T *d_output = dcf::gpuKeygenMaxpool(
        &cursor, party, params, d_input, static_cast<uint8_t *>(nullptr),
        &gaes);
    generated.output_mask.resize(
        static_cast<size_t>(contract::kStockKeySpecs[0].output_words));
    const cudaError_t copied = cudaMemcpy(
        generated.output_mask.data(), d_output,
        generated.output_mask.size() * sizeof(T), cudaMemcpyDeviceToHost);
    gpuFree(d_output);
    gpuFree(d_input);
    const cudaError_t synchronized = cudaDeviceSynchronize();
    destroyGPURandomness();
    return copied == cudaSuccess && synchronized == cudaSuccess &&
           static_cast<size_t>(cursor - generated.raw.get()) == expected &&
           guard_ok(generated);
}

bool generate_relu(int party, const contract::StockKeySpec &spec,
                   const std::vector<T> &input_mask,
                   GeneratedKey &generated) {
    const size_t count = static_cast<size_t>(spec.input_words);
    const size_t expected = expected_relu_key_bytes(count);
    if (spec.kind != contract::StockKeyKind::ReluExtend ||
        input_mask.size() != count || spec.output_words != spec.input_words ||
        !guarded_buffer(expected, generated)) {
        return false;
    }
    initGPURandomness();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(input_mask.data())),
        input_mask.size() * sizeof(T), nullptr));
    uint8_t *cursor = generated.raw.get();
    std::pair<uint8_t *, T *> output = dcf::gpuKeygenReluExtend<T>(
        &cursor, party, spec.input_bw, spec.output_bw,
        static_cast<int>(count), d_input, &gaes);
    generated.output_mask.resize(count);
    const cudaError_t copied = cudaMemcpy(
        generated.output_mask.data(), output.second, count * sizeof(T),
        cudaMemcpyDeviceToHost);
    gpuFree(output.first);
    gpuFree(output.second);
    gpuFree(d_input);
    const cudaError_t synchronized = cudaDeviceSynchronize();
    destroyGPURandomness();
    return copied == cudaSuccess && synchronized == cudaSuccess &&
           static_cast<size_t>(cursor - generated.raw.get()) == expected &&
           guard_ok(generated);
}

bool generate_signextend(int party, const contract::StockKeySpec &spec,
                         const std::vector<T> &input_mask,
                         GeneratedKey &generated) {
    const size_t count = static_cast<size_t>(spec.input_words);
    const size_t expected = expected_signextend_key_bytes(count);
    if (spec.kind != contract::StockKeyKind::SignExtend ||
        input_mask.size() != count || spec.output_words != spec.input_words ||
        !guarded_buffer(expected, generated)) {
        return false;
    }
    initGPURandomness();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(input_mask.data())),
        input_mask.size() * sizeof(T), nullptr));
    uint8_t *cursor = generated.raw.get();
    T *d_output = dcf::genSignExtendKey<T>(
        &cursor, party, spec.input_bw, spec.output_bw,
        static_cast<int>(count), d_input, &gaes);
    generated.output_mask.resize(count);
    const cudaError_t copied = cudaMemcpy(
        generated.output_mask.data(), d_output, count * sizeof(T),
        cudaMemcpyDeviceToHost);
    gpuFree(d_output);
    gpuFree(d_input);
    const cudaError_t synchronized = cudaDeviceSynchronize();
    destroyGPURandomness();
    return copied == cudaSuccess && synchronized == cudaSuccess &&
           static_cast<size_t>(cursor - generated.raw.get()) == expected &&
           guard_ok(generated);
}

bool write_all(int fd, const uint8_t *data, size_t size) {
    while (size != 0) {
        const size_t chunk = std::min(size, kIoChunkBytes);
        const ssize_t written = ::write(fd, data, chunk);
        if (written <= 0) return false;
        data += static_cast<size_t>(written);
        size -= static_cast<size_t>(written);
    }
    return true;
}

bool append_key(int fd, const GeneratedKey &key, RawKeyMeta &meta) {
    if (!key.raw || key.raw_bytes == 0 ||
        !ringlpn_freshness::digest(key.raw.get(), key.raw_bytes,
                                   meta.digest) ||
        !write_all(fd, key.raw.get(), key.raw_bytes)) {
        return false;
    }
    meta.bytes = key.raw_bytes;
    return true;
}

bool generate_pair(int p0_fd, int p1_fd, size_t key_index,
                   const std::vector<T> &input_mask,
                   std::array<RawKeyMeta, contract::kStockKeyCount> &p0_meta,
                   std::array<RawKeyMeta, contract::kStockKeyCount> &p1_meta,
                   std::vector<T> &output_mask) {
    const contract::StockKeySpec &spec = contract::kStockKeySpecs[key_index];
    auto generate = [&](int party, GeneratedKey &key) {
        if (spec.kind == contract::StockKeyKind::MaxPool) {
            return generate_maxpool(party, input_mask, key);
        }
        if (spec.kind == contract::StockKeyKind::ReluExtend) {
            return generate_relu(party, spec, input_mask, key);
        }
        return generate_signextend(party, spec, input_mask, key);
    };
    GeneratedKey p0;
    if (!generate(0, p0) || !append_key(p0_fd, p0, p0_meta[key_index])) {
        return false;
    }
    output_mask = p0.output_mask;
    p0 = GeneratedKey{};
    GeneratedKey p1;
    if (!generate(1, p1) || p1.output_mask != output_mask ||
        !append_key(p1_fd, p1, p1_meta[key_index]) ||
        p0_meta[key_index].bytes != p1_meta[key_index].bytes) {
        return false;
    }
    return true;
}


std::filesystem::path layer_directory(const std::filesystem::path &root,
                                      size_t index) {
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "%02zu_%.*s", index + 1,
                  static_cast<int>(contract::kLinearSpecs[index].name.size()),
                  contract::kLinearSpecs[index].name.data());
    return root / prefix;
}

bool manifest_artifact_paths_match(const Args &args) {
    const std::filesystem::path root(args.record_set_root);
    for (size_t index = 0; index < contract::kLinearCount; ++index) {
        const contract::LinearSpec &spec = contract::kLinearSpecs[index];
        const std::filesystem::path directory = layer_directory(root, index);
        const char *suffix =
            spec.kind == contract::LinearKind::Conv2D ? ".conv" : ".fc";
        const ringlpn_graph::LinearArtifactBindings &bindings =
            args.linear_artifacts[index];
        const std::array<const char *, 4> labels = {
            "p0_record", "p1_record", "p0_state", "p1_state",
        };
        const std::array<const ringlpn_graph::ArtifactBinding *, 4> bound = {
            &bindings.p0_record, &bindings.p1_record,
            &bindings.p0_state, &bindings.p1_state,
        };
        const std::array<std::filesystem::path, 4> expected = {
            directory / (std::string("party0/key_p0") + suffix),
            directory / (std::string("party1/key_p1") + suffix),
            directory / "party0/mask.state",
            directory / "party1/mask.state",
        };
        for (size_t artifact = 0; artifact < expected.size(); ++artifact) {
            if (bound[artifact]->path != expected[artifact]) {
                std::fprintf(stderr,
                             "[full-graph adapter] layer %zu %s path mismatch\n",
                             index + 1, labels[artifact]);
                return false;
            }
        }
    }
    return true;
}

bool verify_record_digest(
    const ringlpn_graph::ArtifactBinding &binding,
    uint64_t format_expected_bytes, const Digest &expected_internal_digest) {
    if (!binding.present || binding.bytes != format_expected_bytes ||
        binding.bytes < record::kDigestBytes ||
        binding.bytes > std::numeric_limits<size_t>::max()) {
        return false;
    }
    const int fd = ringlpn_graph::open_regular_nofollow(binding.path);
    if (fd < 0) return false;
    struct stat metadata {};
    bool ok = ::fstat(fd, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
              metadata.st_size >= 0 &&
              static_cast<uint64_t>(metadata.st_size) == binding.bytes;
    const size_t size = ok ? static_cast<size_t>(binding.bytes) : 0;
    void *mapped = ok ? ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0)
                      : MAP_FAILED;
    if (mapped == MAP_FAILED) ok = false;
    if (ok) {
        const uint8_t *bytes = static_cast<const uint8_t *>(mapped);
        Digest internal_digest{};
        Digest manifest_digest{};
        ok = ringlpn_freshness::digest(
                 bytes, size - record::kDigestBytes, internal_digest) &&
             ringlpn_freshness::digest(bytes, size, manifest_digest) &&
             internal_digest == expected_internal_digest &&
             manifest_digest == binding.sha256 &&
             std::equal(internal_digest.begin(), internal_digest.end(),
                        bytes + size - record::kDigestBytes);
    }
    if (mapped != MAP_FAILED) ::munmap(mapped, size);
    ::close(fd);
    return ok;
}

bool load_party_inputs(
    const std::filesystem::path &root,
    const std::array<ringlpn_graph::LinearArtifactBindings,
                     contract::kLinearCount> &artifacts,
    int party, PartyInputs &inputs) {
    for (size_t i = 0; i < contract::kLinearCount; ++i) {
        const contract::LinearSpec &spec = contract::kLinearSpecs[i];
        const std::filesystem::path directory = layer_directory(root, i);
        const std::filesystem::path expected_state_path =
            directory / (party == 0 ? "party0/mask.state" :
                                      "party1/mask.state");
        const char *suffix =
            spec.kind == contract::LinearKind::Conv2D ? ".conv" : ".fc";
        const std::filesystem::path expected_record_path =
            directory /
            (party == 0 ? std::string("party0/key_p0") + suffix
                        : std::string("party1/key_p1") + suffix);
        const ringlpn_graph::ArtifactBinding &state_binding =
            ringlpn_graph::party_state_binding(artifacts[i], party);
        const ringlpn_graph::ArtifactBinding &record_binding =
            ringlpn_graph::party_record_binding(artifacts[i], party);
        if (state_binding.path != expected_state_path) {
            std::fprintf(stderr,
                         "[full-graph adapter] layer %zu %s path mismatch\n",
                         i + 1, party == 0 ? "p0_state" : "p1_state");
            return false;
        }
        if (!ringlpn_graph::read_mask_state(state_binding, inputs.states[i])) {
            std::fprintf(stderr,
                         "[full-graph adapter] layer %zu %s bytes/SHA rejected\n",
                         i + 1, party == 0 ? "p0_state" : "p1_state");
            return false;
        }
        const MaskState &state = inputs.states[i];
        const uint64_t header_bytes =
            spec.kind == contract::LinearKind::Conv2D ? 224 : 176;
        const uint64_t expected_record_bytes =
            header_bytes +
            (spec.input_words + spec.weight_words + spec.output_words) *
                sizeof(T) +
            record::kDigestBytes;
        if (record_binding.path != expected_record_path) {
            std::fprintf(stderr,
                         "[full-graph adapter] layer %zu %s path mismatch\n",
                         i + 1, party == 0 ? "p0_record" : "p1_record");
            return false;
        }
        if (!verify_record_digest(record_binding, expected_record_bytes,
                                  state.header.linear_record_digest)) {
            std::fprintf(stderr,
                         "[full-graph adapter] layer %zu %s bytes/SHA rejected\n",
                         i + 1, party == 0 ? "p0_record" : "p1_record");
            return false;
        }
        if (state.header.party != party ||
            state.header.layer_ordinal != i + 1 ||
            state.header.input_bw != contract::kFullBw ||
            state.header.output_bw != contract::kFullBw ||
            state.header.input_words != spec.input_words ||
            state.header.output_words != spec.output_words ||
            state.input_mask_share.size() != spec.input_words ||
            state.output_mask_share.size() != spec.output_words) {
            std::fprintf(stderr,
                         "[full-graph adapter] layer %zu %s header rejected\n",
                         i + 1, party == 0 ? "p0_state" : "p1_state");
            return false;
        }
        inputs.record_digests[i] = state.header.linear_record_digest;
        inputs.state_digests[i] = state.digest;
    }
    return true;
}

bool load_inputs(const Args &args, PartyInputs &p0, PartyInputs &p1,
                 std::array<std::vector<T>, contract::kLinearCount> &input_masks,
                 std::array<std::vector<T>, contract::kLinearCount> &output_masks) {
    if (!manifest_artifact_paths_match(args)) return false;
    const std::filesystem::path root(args.record_set_root);
    if (!load_party_inputs(root, args.linear_artifacts, 0, p0) ||
        !load_party_inputs(root, args.linear_artifacts, 1, p1)) {
        return false;
    }
    for (size_t i = 0; i < contract::kLinearCount; ++i) {
        if (!ringlpn_graph::mask_state_headers_match(p0.states[i].header,
                                                     p1.states[i].header)) {
            return false;
        }
        input_masks[i] = add_masks(p0.states[i].input_mask_share,
                                   p1.states[i].input_mask_share,
                                   contract::kFullBw);
        output_masks[i] = add_masks(p0.states[i].output_mask_share,
                                    p1.states[i].output_mask_share,
                                    contract::kFullBw);
        if (input_masks[i].size() != contract::kLinearSpecs[i].input_words ||
            output_masks[i].size() != contract::kLinearSpecs[i].output_words) {
            return false;
        }
    }
    return true;
}

std::vector<T> add_low(const std::vector<T> &left,
                       const std::vector<T> &right) {
    return add_masks(left, right, contract::kTruncatedBw);
}

std::vector<T> global_pool_mask(const std::vector<T> &input) {
    if (input.size() != contract::kGlobalPoolInputWords) return {};
    std::vector<T> output(contract::kGlobalPoolOutputWords, 0);
    for (size_t channel = 0; channel < output.size(); ++channel) {
        uint64_t sum = 0;
        for (size_t spatial = 0; spatial < 49; ++spatial) {
            sum = (sum + input[spatial * output.size() + channel]) & kFullMask;
        }
        output[channel] = (sum * contract::kGlobalPoolMultiplier) & kFullMask;
    }
    return output;
}

const std::vector<T> *mask_source(
    contract::MaskSource source,
    const std::array<std::vector<T>, contract::kStockKeyCount> &stock_masks) {
    switch (source) {
        case contract::MaskSource::Relu2: return &stock_masks[1];
        case contract::MaskSource::Relu4: return &stock_masks[2];
        case contract::MaskSource::Relu7: return &stock_masks[3];
        case contract::MaskSource::Relu9: return &stock_masks[4];
        case contract::MaskSource::Relu12: return &stock_masks[5];
        case contract::MaskSource::Relu14: return &stock_masks[6];
        case contract::MaskSource::Relu18: return &stock_masks[7];
        case contract::MaskSource::Relu20: return &stock_masks[8];
        case contract::MaskSource::Relu23: return &stock_masks[9];
        case contract::MaskSource::Relu25: return &stock_masks[10];
        case contract::MaskSource::Relu29: return &stock_masks[11];
        case contract::MaskSource::Relu31: return &stock_masks[12];
        case contract::MaskSource::Relu34: return &stock_masks[13];
        case contract::MaskSource::Relu36: return &stock_masks[14];
        case contract::MaskSource::Relu40: return &stock_masks[15];
        case contract::MaskSource::Relu42: return &stock_masks[16];
        case contract::MaskSource::SignExtend: return &stock_masks[18];
    }
    return nullptr;
}

bool build_layout(
    record::Header &header,
    const std::array<RawKeyMeta, contract::kStockKeyCount> &keys,
    std::array<record::TruncBinding, contract::kTruncationCount> &trunc,
    std::array<record::StockBinding, contract::kStockKeyCount> &stock,
    std::array<record::RemaskBinding, contract::kRemaskCount> &remask) {
    uint64_t cursor = record::kHeaderBytes;
    header.trunc_table_offset = cursor;
    if (!record::checked_add(
            cursor, contract::kTruncationCount * record::kTruncBindingBytes)) {
        return false;
    }
    header.stock_table_offset = cursor;
    if (!record::checked_add(
            cursor, contract::kStockKeyCount * record::kStockBindingBytes)) {
        return false;
    }
    header.remask_table_offset = cursor;
    if (!record::checked_add(
            cursor, contract::kRemaskCount * record::kRemaskBindingBytes)) {
        return false;
    }
    header.trunc_data_offset = cursor;
    for (size_t i = 0; i < trunc.size(); ++i) {
        trunc[i].stream_position = contract::kTruncationSpecs[i].stream_position;
        trunc[i].linear_index = contract::kTruncationSpecs[i].linear_index;
        trunc[i].words = contract::kTruncationSpecs[i].words;
        trunc[i].data_offset = cursor;
        uint64_t bytes = 0;
        if (!record::checked_words_bytes(trunc[i].words, bytes) ||
            !record::checked_add(cursor, bytes)) {
            return false;
        }
    }
    header.stock_output_mask_offset = cursor;
    for (size_t i = 0; i < stock.size(); ++i) {
        const contract::StockKeySpec &spec = contract::kStockKeySpecs[i];
        stock[i].stream_position = spec.stream_position;
        stock[i].kind = spec.kind;
        stock[i].input_bw = spec.input_bw;
        stock[i].output_bw = spec.output_bw;
        stock[i].input_words = spec.input_words;
        stock[i].output_words = spec.output_words;
        stock[i].output_mask_offset = cursor;
        stock[i].output_mask_words = spec.output_words;
        stock[i].key_bytes = keys[i].bytes;
        stock[i].key_digest = keys[i].digest;
        uint64_t bytes = 0;
        if (!record::checked_words_bytes(spec.output_words, bytes) ||
            !record::checked_add(cursor, bytes)) {
            return false;
        }
    }
    header.remask_data_offset = cursor;
    for (size_t i = 0; i < remask.size(); ++i) {
        const contract::RemaskSpec &spec = contract::kRemaskSpecs[i];
        remask[i].target_linear_index = spec.target_linear_index;
        remask[i].source = spec.source;
        remask[i].words = spec.words;
        remask[i].data_offset = cursor;
        uint64_t bytes = 0;
        if (!record::checked_words_bytes(spec.words, bytes) ||
            !record::checked_add(cursor, bytes)) {
            return false;
        }
    }
    header.terminal_mask_offset = cursor;
    header.terminal_words = contract::kClassifierOutputWords;
    uint64_t terminal_bytes = 0;
    if (!record::checked_words_bytes(header.terminal_words, terminal_bytes) ||
        !record::checked_add(cursor, terminal_bytes)) {
        return false;
    }
    header.raw_key_offset = cursor;
    header.raw_key_bytes = 0;
    for (size_t i = 0; i < stock.size(); ++i) {
        stock[i].key_offset = cursor;
        if (!record::checked_add(cursor, stock[i].key_bytes) ||
            !record::checked_add(header.raw_key_bytes, stock[i].key_bytes)) {
            return false;
        }
    }
    if (!record::checked_add(cursor, record::kDigestBytes) ||
        cursor > record::kMaxRecordBytes) {
        return false;
    }
    header.file_bytes = cursor;
    return record::valid_header_identity(header);
}

bool hash_and_write(int fd, EVP_MD_CTX *ctx, const uint8_t *data, size_t size) {
    return EVP_DigestUpdate(ctx, data, size) == 1 &&
           write_all(fd, data, size);
}

bool write_words(int fd, EVP_MD_CTX *ctx,
                 const std::vector<T> &values) {
    constexpr size_t kWordsPerChunk = 1 << 15;
    std::vector<uint8_t> encoded(kWordsPerChunk * sizeof(T));
    size_t cursor = 0;
    while (cursor < values.size()) {
        const size_t count = std::min(kWordsPerChunk, values.size() - cursor);
        for (size_t i = 0; i < count; ++i) {
            for (size_t byte = 0; byte < sizeof(T); ++byte) {
                encoded[i * sizeof(T) + byte] =
                    static_cast<uint8_t>(values[cursor + i] >> (8 * byte));
            }
        }
        if (!hash_and_write(fd, ctx, encoded.data(), count * sizeof(T))) {
            return false;
        }
        cursor += count;
    }
    return true;
}

bool copy_raw_keys(int output_fd, EVP_MD_CTX *ctx,
                   const std::string &raw_path, uint64_t expected_bytes) {
    const int input_fd = ::open(raw_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (input_fd < 0) return false;
    std::vector<uint8_t> buffer(kIoChunkBytes);
    uint64_t copied = 0;
    bool ok = true;
    while (copied < expected_bytes) {
        const size_t wanted = static_cast<size_t>(std::min<uint64_t>(
            buffer.size(), expected_bytes - copied));
        const ssize_t got = ::read(input_fd, buffer.data(), wanted);
        if (got <= 0 || !hash_and_write(output_fd, ctx, buffer.data(),
                                         static_cast<size_t>(got))) {
            ok = false;
            break;
        }
        copied += static_cast<uint64_t>(got);
    }
    uint8_t extra = 0;
    if (ok && (::read(input_fd, &extra, 1) != 0 || copied != expected_bytes)) {
        ok = false;
    }
    ::close(input_fd);
    return ok;
}

bool write_record(
    const std::string &path, const record::Header &header,
    const std::array<record::TruncBinding, contract::kTruncationCount> &trunc,
    const std::array<record::StockBinding, contract::kStockKeyCount> &stock,
    const std::array<record::RemaskBinding, contract::kRemaskCount> &remask,
    const std::array<std::vector<T>, contract::kTruncationCount> &trunc_shares,
    const std::array<std::vector<T>, contract::kStockKeyCount> &stock_shares,
    const std::array<std::vector<T>, contract::kRemaskCount> &remask_deltas,
    const std::vector<T> &terminal_mask, const std::string &raw_path,
    Digest &digest) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
    if (fd < 0) return false;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    bool ok = ctx != nullptr && EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1;
    const std::vector<uint8_t> encoded_header = record::encode_header(header);
    const std::vector<uint8_t> encoded_trunc = record::encode_trunc_table(trunc);
    const std::vector<uint8_t> encoded_stock = record::encode_stock_table(stock);
    const std::vector<uint8_t> encoded_remask = record::encode_remask_table(remask);
    ok = ok && hash_and_write(fd, ctx, encoded_header.data(), encoded_header.size()) &&
         hash_and_write(fd, ctx, encoded_trunc.data(), encoded_trunc.size()) &&
         hash_and_write(fd, ctx, encoded_stock.data(), encoded_stock.size()) &&
         hash_and_write(fd, ctx, encoded_remask.data(), encoded_remask.size());
    for (const auto &values : trunc_shares) ok = ok && write_words(fd, ctx, values);
    for (const auto &values : stock_shares) ok = ok && write_words(fd, ctx, values);
    for (const auto &values : remask_deltas) ok = ok && write_words(fd, ctx, values);
    ok = ok && write_words(fd, ctx, terminal_mask) &&
         copy_raw_keys(fd, ctx, raw_path, header.raw_key_bytes);
    unsigned int written = 0;
    ok = ok && EVP_DigestFinal_ex(ctx, digest.data(), &written) == 1 &&
         written == digest.size() && write_all(fd, digest.data(), digest.size()) &&
         ::fsync(fd) == 0;
    if (ctx != nullptr) EVP_MD_CTX_free(ctx);
    const bool close_ok = ::close(fd) == 0;
    if (!ok || !close_ok) {
        std::remove(path.c_str());
        return false;
    }
    std::error_code error;
    return std::filesystem::file_size(path, error) == header.file_bytes && !error;
}

bool fsync_parent(const std::string &path) {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) parent = ".";
    const int fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

bool publish_pair(const std::string &p0_temp, const std::string &p1_temp,
                  const std::string &p0_output,
                  const std::string &p1_output) {
    const bool first = std::rename(p0_temp.c_str(), p0_output.c_str()) == 0;
    const bool second = first &&
        std::rename(p1_temp.c_str(), p1_output.c_str()) == 0;
    if (!second) {
        std::remove(p0_temp.c_str());
        std::remove(p1_temp.c_str());
        if (first) std::remove(p0_output.c_str());
        if (second) std::remove(p1_output.c_str());
        return false;
    }
    return true;
}

int run_publication_control(const Args &args) {
    const std::string p0_temp = args.p0_output + ".tmp";
    const std::string p1_temp = args.p1_output + ".tmp";
    auto absent = [](const std::string &path) {
        std::error_code error;
        return !std::filesystem::exists(path, error) && !error;
    };
    std::error_code blocker_error;
    const auto blocker_status =
        std::filesystem::symlink_status(args.p1_output, blocker_error);
    const bool p1_blocker = !blocker_error &&
        std::filesystem::is_directory(blocker_status) &&
        !std::filesystem::is_symlink(blocker_status);
    if (!absent(args.p0_output) || !p1_blocker ||
        !absent(p0_temp) || !absent(p1_temp)) {
        return 2;
    }
    const uint8_t p0 = 0;
    const uint8_t p1 = 1;
    int fd0 = ::open(p0_temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                     S_IRUSR | S_IWUSR);
    int fd1 = ::open(p1_temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                     S_IRUSR | S_IWUSR);
    const bool staged = fd0 >= 0 && fd1 >= 0 &&
                        write_all(fd0, &p0, 1) && write_all(fd1, &p1, 1) &&
                        ::fsync(fd0) == 0 && ::fsync(fd1) == 0;
    if (fd0 >= 0) ::close(fd0);
    if (fd1 >= 0) ::close(fd1);
    if (!staged) {
        std::remove(p0_temp.c_str());
        std::remove(p1_temp.c_str());
        return 1;
    }
    // The caller supplies a directory at p1_output.  First rename succeeds,
    // second rename fails, and bilateral cleanup must remove the first output.
    const bool rejected = !publish_pair(p0_temp, p1_temp,
                                        args.p0_output, args.p1_output) &&
                          absent(args.p0_output) && absent(p0_temp) &&
                          absent(p1_temp);
    if (args.csv_header) {
        std::printf("control,p0_absent,p1_temp_absent,status\n");
    }
    std::printf("forced_second_rename,%s,%s,%s\n",
                absent(args.p0_output) ? "pass" : "FAIL",
                absent(p1_temp) ? "pass" : "FAIL",
                rejected ? "pass" : "FAIL");
    return rejected ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        std::fprintf(stderr, "invalid full-graph trusted-adapter arguments\n");
        return 2;
    }
    if (args.publication_control) return run_publication_control(args);

    const std::string p0_temp = args.p0_output + ".tmp";
    const std::string p1_temp = args.p1_output + ".tmp";
    const std::string p0_raw = args.p0_output + ".raw.tmp";
    const std::string p1_raw = args.p1_output + ".raw.tmp";
    const std::array<std::string, 6> outputs = {
        args.p0_output, args.p1_output, p0_temp, p1_temp, p0_raw, p1_raw};
    for (const std::string &path : outputs) {
        std::error_code error;
        if (std::filesystem::exists(path, error) || error) {
            std::fprintf(stderr, "full-graph adapter output already exists\n");
            return 2;
        }
    }

    PartyInputs p0_inputs;
    PartyInputs p1_inputs;
    std::array<std::vector<T>, contract::kLinearCount> input_masks;
    std::array<std::vector<T>, contract::kLinearCount> output_masks;
    if (!load_inputs(args, p0_inputs, p1_inputs, input_masks, output_masks) ||
        !configure_gpu_pool(args.gpu)) {
        std::fprintf(stderr, "full-graph source-state/GPU validation failed\n");
        return 2;
    }

    std::array<std::vector<T>, contract::kTruncationCount> trunc_share0;
    std::array<std::vector<T>, contract::kTruncationCount> trunc_share1;
    std::array<std::vector<T>, contract::kTruncationCount> trunc_full;
    for (size_t i = 0; i < contract::kTruncationCount; ++i) {
        const size_t words =
            static_cast<size_t>(contract::kTruncationSpecs[i].words);
        trunc_share0[i].resize(words);
        trunc_share1[i].resize(words);
        if (!fill_random(trunc_share0[i], contract::kTruncatedBw) ||
            !fill_random(trunc_share1[i], contract::kTruncatedBw)) {
            std::fprintf(stderr, "full-graph truncation CSPRNG failure\n");
            return 1;
        }
        trunc_full[i] = add_low(trunc_share0[i], trunc_share1[i]);
    }

    const int p0_raw_fd = ::open(p0_raw.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    const int p1_raw_fd = ::open(p1_raw.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (p0_raw_fd < 0 || p1_raw_fd < 0) {
        if (p0_raw_fd >= 0) ::close(p0_raw_fd);
        if (p1_raw_fd >= 0) ::close(p1_raw_fd);
        std::remove(p0_raw.c_str());
        std::remove(p1_raw.c_str());
        return 1;
    }

    const auto started = Clock::now();
    std::array<RawKeyMeta, contract::kStockKeyCount> p0_meta;
    std::array<RawKeyMeta, contract::kStockKeyCount> p1_meta;
    std::array<std::vector<T>, contract::kStockKeyCount> stock_full;
    bool ok = generate_pair(p0_raw_fd, p1_raw_fd, 0, trunc_full[0],
                            p0_meta, p1_meta, stock_full[0]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 1, stock_full[0],
                             p0_meta, p1_meta, stock_full[1]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 2, trunc_full[1],
                             p0_meta, p1_meta, stock_full[2]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 3,
                             add_low(trunc_full[2], stock_full[1]),
                             p0_meta, p1_meta, stock_full[3]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 4, trunc_full[3],
                             p0_meta, p1_meta, stock_full[4]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 5,
                             add_low(trunc_full[4], stock_full[3]),
                             p0_meta, p1_meta, stock_full[5]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 6, trunc_full[5],
                             p0_meta, p1_meta, stock_full[6]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 7,
                             add_low(trunc_full[6], trunc_full[7]),
                             p0_meta, p1_meta, stock_full[7]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 8, trunc_full[8],
                             p0_meta, p1_meta, stock_full[8]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 9,
                             add_low(trunc_full[9], stock_full[7]),
                             p0_meta, p1_meta, stock_full[9]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 10, trunc_full[10],
                             p0_meta, p1_meta, stock_full[10]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 11,
                             add_low(trunc_full[11], trunc_full[12]),
                             p0_meta, p1_meta, stock_full[11]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 12, trunc_full[13],
                             p0_meta, p1_meta, stock_full[12]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 13,
                             add_low(trunc_full[14], stock_full[11]),
                             p0_meta, p1_meta, stock_full[13]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 14, trunc_full[15],
                             p0_meta, p1_meta, stock_full[14]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 15,
                             add_low(trunc_full[16], trunc_full[17]),
                             p0_meta, p1_meta, stock_full[15]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 16, trunc_full[18],
                             p0_meta, p1_meta, stock_full[16]);
    ok = ok && generate_pair(p0_raw_fd, p1_raw_fd, 17,
                             add_low(trunc_full[19], stock_full[15]),
                             p0_meta, p1_meta, stock_full[17]);
    const std::vector<T> pooled_mask = global_pool_mask(stock_full[17]);
    ok = ok && pooled_mask.size() == contract::kGlobalPoolOutputWords &&
         generate_pair(p0_raw_fd, p1_raw_fd, 18, trunc_full[20],
                       p0_meta, p1_meta, stock_full[18]);
    ok = ok && ::fsync(p0_raw_fd) == 0 && ::fsync(p1_raw_fd) == 0;
    const bool raw_close_ok = ::close(p0_raw_fd) == 0 && ::close(p1_raw_fd) == 0;
    ok = ok && raw_close_ok;
    if (!ok) {
        std::remove(p0_raw.c_str());
        std::remove(p1_raw.c_str());
        std::fprintf(stderr, "full-graph stock key generation failed\n");
        return 1;
    }

    std::array<std::vector<T>, contract::kStockKeyCount> stock_share0;
    std::array<std::vector<T>, contract::kStockKeyCount> stock_share1;
    for (size_t i = 0; ok && i < contract::kStockKeyCount; ++i) {
        ok = split_mask(stock_full[i], contract::kStockKeySpecs[i].output_bw,
                        stock_share0[i], stock_share1[i]);
    }
    std::array<std::vector<T>, contract::kRemaskCount> remask_deltas;
    for (size_t i = 0; ok && i < contract::kRemaskCount; ++i) {
        const contract::RemaskSpec &spec = contract::kRemaskSpecs[i];
        const std::vector<T> *source = mask_source(spec.source, stock_full);
        const std::vector<T> &target = input_masks[spec.target_linear_index];
        if (source == nullptr || source->size() != spec.words ||
            target.size() != spec.words) {
            ok = false;
            break;
        }
        remask_deltas[i].resize(static_cast<size_t>(spec.words));
        for (size_t j = 0; j < remask_deltas[i].size(); ++j) {
            remask_deltas[i][j] = (target[j] - (*source)[j]) & kFullMask;
        }
    }
    const std::vector<T> terminal_mask = output_masks.back();
    Digest bundle_id{};
    ok = ok && terminal_mask.size() == contract::kClassifierOutputWords &&
         fill_random(bundle_id);

    record::Header p0_header;
    p0_header.party = 0;
    p0_header.scope = record::kTrustedKnownZeroScope;
    p0_header.full_bw = contract::kFullBw;
    p0_header.truncated_bw = contract::kTruncatedBw;
    p0_header.scale = contract::kScale;
    p0_header.invocation = args.invocation;
    p0_header.manifest_digest = args.manifest_digest;
    p0_header.record_set_digest = args.record_set_digest;
    p0_header.bundle_id = bundle_id;
    p0_header.linear_record_digests = p0_inputs.record_digests;
    p0_header.mask_state_digests = p0_inputs.state_digests;
    record::Header p1_header = p0_header;
    p1_header.party = 1;
    p1_header.linear_record_digests = p1_inputs.record_digests;
    p1_header.mask_state_digests = p1_inputs.state_digests;

    std::array<record::TruncBinding, contract::kTruncationCount> p0_trunc;
    std::array<record::StockBinding, contract::kStockKeyCount> p0_stock;
    std::array<record::RemaskBinding, contract::kRemaskCount> p0_remask;
    std::array<record::TruncBinding, contract::kTruncationCount> p1_trunc;
    std::array<record::StockBinding, contract::kStockKeyCount> p1_stock;
    std::array<record::RemaskBinding, contract::kRemaskCount> p1_remask;
    ok = ok && build_layout(p0_header, p0_meta, p0_trunc, p0_stock, p0_remask) &&
         build_layout(p1_header, p1_meta, p1_trunc, p1_stock, p1_remask) &&
         p0_header.file_bytes == p1_header.file_bytes &&
         p0_header.raw_key_bytes == p1_header.raw_key_bytes;

    Digest p0_digest{};
    Digest p1_digest{};
    ok = ok && write_record(p0_temp, p0_header, p0_trunc, p0_stock,
                            p0_remask, trunc_share0, stock_share0,
                            remask_deltas, terminal_mask, p0_raw, p0_digest) &&
         write_record(p1_temp, p1_header, p1_trunc, p1_stock,
                      p1_remask, trunc_share1, stock_share1,
                      remask_deltas, terminal_mask, p1_raw, p1_digest);
    std::remove(p0_raw.c_str());
    std::remove(p1_raw.c_str());
    if (ok) {
        record::MappedRecord checked0;
        record::MappedRecord checked1;
        ok = checked0.open(p0_temp) && checked1.open(p1_temp) &&
             checked0.digest() == p0_digest && checked1.digest() == p1_digest &&
             record::public_headers_match(checked0.header(), checked1.header());
        for (size_t i = 0; ok && i < contract::kRemaskCount; ++i) {
            ok = std::equal(checked0.remask_delta(i),
                            checked0.remask_delta(i) +
                                checked0.remask_binding(i).words,
                            checked1.remask_delta(i));
        }
        ok = ok && std::equal(checked0.terminal_mask(),
                              checked0.terminal_mask() +
                                  checked0.header().terminal_words,
                              checked1.terminal_mask());
    }
    ok = ok && publish_pair(p0_temp, p1_temp,
                            args.p0_output, args.p1_output) &&
         fsync_parent(args.p0_output) && fsync_parent(args.p1_output);
    if (!ok) {
        std::remove(p0_temp.c_str());
        std::remove(p1_temp.c_str());
        std::remove(args.p0_output.c_str());
        std::remove(args.p1_output.c_str());
        std::fprintf(stderr, "full-graph record publication failed\n");
        return 1;
    }

    const uint64_t total_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(Clock::now() - started)
            .count());
    if (args.csv_header) {
        std::printf(
            "key_source,stock_key_items,truncation_items,remask_edges,"
            "raw_key_bytes_per_party,record_bytes_per_party,p0_record_digest,"
            "p1_record_digest,total_us,status\n");
    }
    std::printf(
        "stock_trusted_dealer_test_only,%zu,%zu,%zu,%llu,%llu,%s,%s,%llu,pass\n",
        contract::kStockKeyCount, contract::kTruncationCount,
        contract::kRemaskCount,
        static_cast<unsigned long long>(p0_header.raw_key_bytes),
        static_cast<unsigned long long>(p0_header.file_bytes),
        hex_digest(p0_digest).c_str(), hex_digest(p1_digest).c_str(),
        static_cast<unsigned long long>(total_us));
    return 0;
}
