// TEST-ONLY trusted compatibility adapter for the exact ResNet18 prefix.
//
// This process reads both parties' source-bound linear mask states, creates the
// post-truncation mask, and invokes Orca's stock trusted-dealer MaxPool/ReLU
// keygen twice with its deterministic party split.  It publishes one private
// record per party.  The live graph processes still read only their own record
// and execute unchanged stock consumers.  This is deliberately not a
// distributed DCF protocol and must never be cited as dealerless nonlinear
// preprocessing or private/trained model evidence.

// Llama's legacy unscoped BUF_MEM enumerator collides with OpenSSL's BUF_MEM
// typedef.  Rename only the header token; enum values are ABI-identical.
#define BUF_MEM LLAMA_BUF_MEM
#include "utils/gpu_comms.h"
#undef BUF_MEM
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/helper_cuda.h"

#include "fss/dcf/gpu_maxpool.h"
#include "fss/dcf/gpu_relu.h"
#include "graph_mask_state.h"
#include "stock_nonlinear_prefix_record.h"

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
#include <vector>

namespace {

namespace nonlinear = ringlpn_nonlinear_prefix;
using T = uint64_t;
using Digest = ringlpn_freshness::Digest;
using InvocationId = ringlpn_freshness::InvocationId;
using MaskState = ringlpn_graph::MaskStateRecord<T>;

constexpr size_t kGuardBytes = 4096;
constexpr uint8_t kGuardValue = 0xA7;
constexpr size_t kIoChunkBytes = size_t(32) << 20;

struct Args {
    bool csv_header = false;
    bool publication_control = false;
    int gpu = 0;
    std::string invocation_text;
    InvocationId invocation{};
    Digest manifest_digest{};
    std::string p0_conv0_state;
    std::string p1_conv0_state;
    std::string p0_conv3_state;
    std::string p1_conv3_state;
    std::string p0_output;
    std::string p1_output;
};

struct GeneratedKeys {
    std::unique_ptr<uint8_t[]> raw;
    size_t raw_bytes = 0;
    size_t maxpool_key_bytes = 0;
    size_t relu_key_bytes = 0;
    std::vector<T> maxpool_mask;
    std::vector<T> relu_mask;
};

bool parse_hex(const std::string &text, uint8_t *out, size_t bytes) {
    if (text.size() != 2 * bytes) return false;
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
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

bool parse_args(int argc, char **argv, Args &args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        const char *value = nullptr;
        if (key == "--csv-header") {
            args.csv_header = true;
        } else if (key == "--publication-control") {
            args.publication_control = true;
        } else if (key == "--gpu" && (value = next())) {
            if (!parse_int(value, args.gpu)) return false;
        } else if (key == "--invocation-id" && (value = next())) {
            args.invocation_text = value;
        } else if (key == "--manifest-digest" && (value = next())) {
            if (!parse_hex(value, args.manifest_digest.data(),
                           args.manifest_digest.size())) {
                return false;
            }
        } else if (key == "--p0-conv0-state" && (value = next())) {
            args.p0_conv0_state = value;
        } else if (key == "--p1-conv0-state" && (value = next())) {
            args.p1_conv0_state = value;
        } else if (key == "--p0-conv3-state" && (value = next())) {
            args.p0_conv3_state = value;
        } else if (key == "--p1-conv3-state" && (value = next())) {
            args.p1_conv3_state = value;
        } else if (key == "--p0-output" && (value = next())) {
            args.p0_output = value;
        } else if (key == "--p1-output" && (value = next())) {
            args.p1_output = value;
        } else {
            return false;
        }
    }
    const bool manifest_nonzero = std::any_of(
        args.manifest_digest.begin(), args.manifest_digest.end(),
        [](uint8_t byte) { return byte != 0; });
    const bool outputs_valid =
        !args.p0_output.empty() && !args.p1_output.empty() &&
        args.p0_output != args.p1_output &&
        args.p0_output + ".tmp" != args.p1_output &&
        args.p1_output + ".tmp" != args.p0_output;
    if (args.publication_control) {
        return args.invocation_text.empty() && !manifest_nonzero &&
               args.p0_conv0_state.empty() && args.p1_conv0_state.empty() &&
               args.p0_conv3_state.empty() && args.p1_conv3_state.empty() &&
               outputs_valid;
    }
    return args.gpu >= 0 && manifest_nonzero &&
           ringlpn_freshness::parse_invocation_id(args.invocation_text,
                                                  args.invocation) &&
           !args.p0_conv0_state.empty() && !args.p1_conv0_state.empty() &&
           !args.p0_conv3_state.empty() && !args.p1_conv3_state.empty() &&
           outputs_valid;
}

size_t packed_bytes(int bits, size_t count) {
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
    // The exact prefix batch is far below Orca's 24-GiB per-tree split limit,
    // hence one outer DCF header and one tree record.
    return 4 * sizeof(int) + 3 * sizeof(int) +
           count * static_cast<size_t>(new_bin) * sizeof(AESBlock) +
           2 * count * sizeof(AESBlock) +
           packed_bytes(bout, count) * static_cast<size_t>(new_bin - 1);
}

size_t expected_maxpool_key_bytes() {
    const size_t n = static_cast<size_t>(nonlinear::kNonlinearWords);
    const size_t one_relu = 3 * sizeof(int) +
                            dcf_key_bytes(nonlinear::kTruncatedBw, 1, n) +
                            packed_bytes(1, n) + 5 * n * sizeof(T);
    return 8 * one_relu;
}

size_t expected_relu_key_bytes() {
    const size_t n = static_cast<size_t>(nonlinear::kNonlinearWords);
    return 3 * sizeof(int) +
           dcf_key_bytes(nonlinear::kTruncatedBw, 2, n) +
           2 * packed_bytes(2, n) + 6 * n * sizeof(T);
}

MaxpoolParams exact_maxpool_params() {
    MaxpoolParams params = {
        nonlinear::kTruncatedBw, nonlinear::kTruncatedBw, 0, 0,
        nonlinear::kFullBw,
        1, 112, 112, 64,
        3, 3,
        2, 2,
        1, 1,
        1, 1,
        0, 0, false};
    initPoolParams(params);
    return params;
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
                            sizeof(T)) {
        return false;
    }
    if (RAND_priv_bytes(reinterpret_cast<unsigned char *>(values.data()),
                        static_cast<int>(values.size() * sizeof(T))) != 1) {
        return false;
    }
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    for (T &value : values) value &= mask;
    return true;
}

bool fill_random(Digest &value) {
    return RAND_priv_bytes(value.data(), static_cast<int>(value.size())) == 1 &&
           std::any_of(value.begin(), value.end(),
                       [](uint8_t byte) { return byte != 0; });
}

std::vector<T> add_masks(const std::vector<T> &left,
                         const std::vector<T> &right, int bits) {
    if (left.size() != right.size()) return {};
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

bool generate_keys(int party, const std::vector<T> &input_mask,
                   GeneratedKeys &generated) {
    const size_t expected_maxpool = expected_maxpool_key_bytes();
    const size_t expected_relu = expected_relu_key_bytes();
    const size_t expected_total = expected_maxpool + expected_relu;
    if (expected_total > nonlinear::kMaxRawKeyBytes ||
        input_mask.size() != nonlinear::kTruncWords) {
        return false;
    }
    std::unique_ptr<uint8_t[]> raw(
        new (std::nothrow) uint8_t[expected_total + kGuardBytes]);
    if (!raw) return false;
    std::memset(raw.get() + expected_total, kGuardValue, kGuardBytes);

    initGPURandomness();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(input_mask.data())),
        input_mask.size() * sizeof(T), nullptr));
    uint8_t *cursor = raw.get();
    MaxpoolParams params = exact_maxpool_params();
    T *d_maxpool = dcf::gpuKeygenMaxpool(
        &cursor, party, params, d_input, static_cast<uint8_t *>(nullptr),
        &gaes);
    const size_t maxpool_bytes = static_cast<size_t>(cursor - raw.get());
    std::vector<T> maxpool_mask(
        static_cast<size_t>(nonlinear::kNonlinearWords));
    const cudaError_t maxpool_copy = cudaMemcpy(
        maxpool_mask.data(), d_maxpool, maxpool_mask.size() * sizeof(T),
        cudaMemcpyDeviceToHost);

    std::pair<uint8_t *, T *> relu = dcf::gpuKeygenReluExtend<T>(
        &cursor, party, nonlinear::kTruncatedBw, nonlinear::kFullBw,
        static_cast<int>(nonlinear::kNonlinearWords), d_maxpool, &gaes);
    gpuFree(relu.first);
    std::vector<T> relu_mask(
        static_cast<size_t>(nonlinear::kNonlinearWords));
    const cudaError_t relu_copy = cudaMemcpy(
        relu_mask.data(), relu.second, relu_mask.size() * sizeof(T),
        cudaMemcpyDeviceToHost);
    gpuFree(relu.second);
    gpuFree(d_maxpool);
    gpuFree(d_input);
    const cudaError_t synchronized = cudaDeviceSynchronize();
    destroyGPURandomness();

    const size_t total_bytes = static_cast<size_t>(cursor - raw.get());
    const bool guard_ok = std::all_of(
        raw.get() + expected_total, raw.get() + expected_total + kGuardBytes,
        [](uint8_t byte) { return byte == kGuardValue; });
    if (maxpool_copy != cudaSuccess || relu_copy != cudaSuccess ||
        synchronized != cudaSuccess || maxpool_bytes != expected_maxpool ||
        total_bytes != expected_total || !guard_ok) {
        return false;
    }
    generated.raw = std::move(raw);
    generated.raw_bytes = total_bytes;
    generated.maxpool_key_bytes = maxpool_bytes;
    generated.relu_key_bytes = total_bytes - maxpool_bytes;
    generated.maxpool_mask = std::move(maxpool_mask);
    generated.relu_mask = std::move(relu_mask);
    return true;
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

bool hash_and_write(int fd, EVP_MD_CTX *ctx, const uint8_t *data,
                    size_t size) {
    return EVP_DigestUpdate(ctx, data, size) == 1 &&
           write_all(fd, data, size);
}

bool write_words(int fd, EVP_MD_CTX *ctx,
                 const std::vector<T> &values) {
    constexpr size_t kWordsPerChunk = 1 << 15;
    std::vector<uint8_t> encoded(kWordsPerChunk * sizeof(T));
    size_t cursor = 0;
    while (cursor < values.size()) {
        const size_t count =
            std::min(kWordsPerChunk, values.size() - cursor);
        for (size_t i = 0; i < count; ++i) {
            const uint64_t value = values[cursor + i];
            for (size_t byte = 0; byte < sizeof(T); ++byte) {
                encoded[i * sizeof(T) + byte] =
                    static_cast<uint8_t>(value >> (8 * byte));
            }
        }
        if (!hash_and_write(fd, ctx, encoded.data(), count * sizeof(T))) {
            return false;
        }
        cursor += count;
    }
    return true;
}

bool write_record(const std::string &path,
                  const nonlinear::Header &header,
                  const std::vector<T> &trunc_share,
                  const std::vector<T> &maxpool_share,
                  const std::vector<T> &relu_share,
                  const std::vector<T> &delta,
                  const GeneratedKeys &generated, Digest &digest) {
    if (!nonlinear::valid_header(header) ||
        trunc_share.size() != nonlinear::kTruncWords ||
        maxpool_share.size() != nonlinear::kNonlinearWords ||
        relu_share.size() != nonlinear::kNonlinearWords ||
        delta.size() != nonlinear::kNonlinearWords ||
        generated.raw_bytes != header.maxpool_key_bytes +
                                   header.relu_key_bytes ||
        generated.maxpool_key_bytes != header.maxpool_key_bytes ||
        generated.relu_key_bytes != header.relu_key_bytes) {
        return false;
    }
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
    if (fd < 0) return false;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    bool ok = ctx != nullptr && EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1;
    const std::vector<uint8_t> encoded_header = nonlinear::encode_header(header);
    ok = ok && hash_and_write(fd, ctx, encoded_header.data(),
                              encoded_header.size());
    ok = ok && write_words(fd, ctx, trunc_share);
    ok = ok && write_words(fd, ctx, maxpool_share);
    ok = ok && write_words(fd, ctx, relu_share);
    ok = ok && write_words(fd, ctx, delta);
    ok = ok && hash_and_write(fd, ctx, generated.raw.get(),
                              generated.raw_bytes);
    unsigned int digest_size = 0;
    ok = ok && EVP_DigestFinal_ex(ctx, digest.data(), &digest_size) == 1 &&
         digest_size == digest.size();
    if (ctx != nullptr) EVP_MD_CTX_free(ctx);
    ok = ok && write_all(fd, digest.data(), digest.size()) && ::fsync(fd) == 0;
    const int close_rc = ::close(fd);
    if (!ok || close_rc != 0) {
        std::remove(path.c_str());
        return false;
    }
    return true;
}

bool fsync_parent(const std::string &path) {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) parent = ".";
    const int fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    const int close_rc = ::close(fd);
    return ok && close_rc == 0;
}

bool publish_pair(const std::string &p0_temp, const std::string &p1_temp,
                  const std::string &p0_output,
                  const std::string &p1_output) {
    bool published = std::rename(p0_temp.c_str(), p0_output.c_str()) == 0;
    if (published) {
        published = std::rename(p1_temp.c_str(), p1_output.c_str()) == 0;
    }
    if (!published) {
        std::remove(p0_temp.c_str());
        std::remove(p1_temp.c_str());
        std::remove(p0_output.c_str());
        std::remove(p1_output.c_str());
    }
    return published;
}

bool stage_publication_control_file(const std::string &path, uint8_t party) {
    const int fd = ::open(path.c_str(),
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
    if (fd < 0) return false;
    const std::array<uint8_t, 8> marker = {
        'R', 'L', 'P', 'N', 'P', 'U', 'B', party};
    const bool ok = write_all(fd, marker.data(), marker.size()) &&
                    ::fsync(fd) == 0;
    const int close_rc = ::close(fd);
    if (!ok || close_rc != 0) {
        std::remove(path.c_str());
        return false;
    }
    return true;
}

int run_publication_control(const Args &args) {
    const std::string p0_temp = args.p0_output + ".tmp";
    const std::string p1_temp = args.p1_output + ".tmp";
    const auto absent = [](const std::string &path) {
        std::error_code error;
        const bool exists = std::filesystem::exists(path, error);
        return !error && !exists;
    };
    const auto nonempty_directory = [](const std::string &path) {
        std::error_code error;
        const bool directory = std::filesystem::is_directory(path, error);
        if (error || !directory) return false;
        const bool empty = std::filesystem::is_empty(path, error);
        return !error && !empty;
    };
    const bool prepared =
        absent(args.p0_output) && absent(p0_temp) && absent(p1_temp) &&
        nonempty_directory(args.p1_output) &&
        stage_publication_control_file(p0_temp, 0) &&
        stage_publication_control_file(p1_temp, 1);
    const bool published =
        prepared && publish_pair(p0_temp, p1_temp, args.p0_output,
                                 args.p1_output);
    const bool passed =
        prepared && !published && absent(args.p0_output) && absent(p0_temp) &&
        absent(p1_temp) && nonempty_directory(args.p1_output);
    if (args.csv_header) {
        std::printf("control,expected,observed,status\n");
    }
    std::printf("forced_second_rename_failure,bilateral_rollback,%s,%s\n",
                passed ? "both_outputs_absent" : "partial_output",
                passed ? "pass" : "FAIL");
    return passed ? 0 : 1;
}

std::string hex_digest(const Digest &digest) {
    static constexpr char chars[] = "0123456789abcdef";
    std::string out(2 * digest.size(), '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = chars[digest[i] >> 4];
        out[2 * i + 1] = chars[digest[i] & 15];
    }
    return out;
}

bool exact_states(const MaskState &conv0_p0, const MaskState &conv0_p1,
                  const MaskState &conv3_p0, const MaskState &conv3_p1) {
    return ringlpn_graph::mask_state_headers_match(conv0_p0.header,
                                                   conv0_p1.header) &&
           ringlpn_graph::mask_state_headers_match(conv3_p0.header,
                                                   conv3_p1.header) &&
           conv0_p0.header.layer_ordinal == 1 &&
           conv0_p0.header.input_bw == nonlinear::kFullBw &&
           conv0_p0.header.output_bw == nonlinear::kFullBw &&
           conv0_p0.header.output_words == nonlinear::kTruncWords &&
           conv3_p0.header.layer_ordinal == 2 &&
           conv3_p0.header.input_bw == nonlinear::kFullBw &&
           conv3_p0.header.output_bw == nonlinear::kFullBw &&
           conv3_p0.header.input_words == nonlinear::kNonlinearWords &&
           conv3_p0.input_mask_share.size() == nonlinear::kNonlinearWords &&
           conv3_p1.input_mask_share.size() == nonlinear::kNonlinearWords;
}

}  // namespace

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        std::fprintf(stderr, "invalid stock nonlinear adapter arguments\n");
        return 2;
    }
    if (args.publication_control) {
        return run_publication_control(args);
    }
    const std::string p0_temp = args.p0_output + ".tmp";
    const std::string p1_temp = args.p1_output + ".tmp";
    std::error_code error;
    if (std::filesystem::exists(args.p0_output, error) || error ||
        std::filesystem::exists(args.p1_output, error) || error ||
        std::filesystem::exists(p0_temp, error) || error ||
        std::filesystem::exists(p1_temp, error) || error) {
        std::fprintf(stderr, "stock nonlinear output already exists\n");
        return 2;
    }

    MaskState conv0_p0, conv0_p1, conv3_p0, conv3_p1;
    if (!ringlpn_graph::read_mask_state(args.p0_conv0_state, conv0_p0) ||
        !ringlpn_graph::read_mask_state(args.p1_conv0_state, conv0_p1) ||
        !ringlpn_graph::read_mask_state(args.p0_conv3_state, conv3_p0) ||
        !ringlpn_graph::read_mask_state(args.p1_conv3_state, conv3_p1) ||
        !exact_states(conv0_p0, conv0_p1, conv3_p0, conv3_p1)) {
        std::fprintf(stderr, "stock nonlinear source-state validation failed\n");
        return 2;
    }

    std::array<Digest, 4> linear_digests = {
        conv0_p0.header.linear_record_digest,
        conv0_p1.header.linear_record_digest,
        conv3_p0.header.linear_record_digest,
        conv3_p1.header.linear_record_digest};
    Digest bundle_id{};
    std::vector<T> trunc_share0(static_cast<size_t>(nonlinear::kTruncWords));
    std::vector<T> trunc_share1(static_cast<size_t>(nonlinear::kTruncWords));
    if (!fill_random(trunc_share0, nonlinear::kTruncatedBw) ||
        !fill_random(trunc_share1, nonlinear::kTruncatedBw) ||
        !fill_random(bundle_id)) {
        std::fprintf(stderr, "stock nonlinear CSPRNG failure\n");
        return 1;
    }
    const std::vector<T> full_trunc = add_masks(
        trunc_share0, trunc_share1, nonlinear::kTruncatedBw);
    const std::vector<T> conv3_input_mask = add_masks(
        conv3_p0.input_mask_share, conv3_p1.input_mask_share,
        nonlinear::kFullBw);
    if (full_trunc.size() != nonlinear::kTruncWords ||
        conv3_input_mask.size() != nonlinear::kNonlinearWords ||
        !configure_gpu_pool(args.gpu)) {
        std::fprintf(stderr, "stock nonlinear GPU/setup failure\n");
        return 1;
    }

    const auto started = std::chrono::steady_clock::now();
    GeneratedKeys p0_keys;
    if (!generate_keys(0, full_trunc, p0_keys)) {
        std::fprintf(stderr, "party-0 stock nonlinear keygen failed\n");
        return 1;
    }
    std::vector<T> maxpool_share0, maxpool_share1;
    std::vector<T> relu_share0, relu_share1;
    if (!split_mask(p0_keys.maxpool_mask, nonlinear::kTruncatedBw,
                    maxpool_share0, maxpool_share1) ||
        !split_mask(p0_keys.relu_mask, nonlinear::kFullBw,
                    relu_share0, relu_share1)) {
        std::fprintf(stderr, "stock nonlinear output-mask split failed\n");
        return 1;
    }
    std::vector<T> delta(static_cast<size_t>(nonlinear::kNonlinearWords));
    const uint64_t full_mask = (uint64_t(1) << nonlinear::kFullBw) - 1;
    for (size_t i = 0; i < delta.size(); ++i) {
        delta[i] = (conv3_input_mask[i] - p0_keys.relu_mask[i]) & full_mask;
    }

    nonlinear::Header p0_header;
    p0_header.party = 0;
    p0_header.scope = nonlinear::kTrustedStockDealerKnownZeroScope;
    p0_header.trunc_bw = nonlinear::kTruncatedBw;
    p0_header.full_bw = nonlinear::kFullBw;
    p0_header.trunc_words = nonlinear::kTruncWords;
    p0_header.nonlinear_words = nonlinear::kNonlinearWords;
    p0_header.maxpool_key_bytes = p0_keys.maxpool_key_bytes;
    p0_header.relu_key_bytes = p0_keys.relu_key_bytes;
    p0_header.invocation = args.invocation;
    p0_header.manifest_digest = args.manifest_digest;
    p0_header.linear_record_digests = linear_digests;
    p0_header.bundle_id = bundle_id;
    Digest p0_digest{};
    if (!write_record(p0_temp, p0_header, trunc_share0, maxpool_share0,
                      relu_share0, delta, p0_keys, p0_digest)) {
        std::fprintf(stderr, "party-0 stock nonlinear record write failed\n");
        return 1;
    }
    const std::vector<T> expected_maxpool_mask = p0_keys.maxpool_mask;
    const std::vector<T> expected_relu_mask = p0_keys.relu_mask;
    const size_t maxpool_key_bytes = p0_keys.maxpool_key_bytes;
    const size_t relu_key_bytes = p0_keys.relu_key_bytes;
    p0_keys = GeneratedKeys{};

    GeneratedKeys p1_keys;
    if (!generate_keys(1, full_trunc, p1_keys) ||
        p1_keys.maxpool_key_bytes != maxpool_key_bytes ||
        p1_keys.relu_key_bytes != relu_key_bytes ||
        p1_keys.maxpool_mask != expected_maxpool_mask ||
        p1_keys.relu_mask != expected_relu_mask) {
        std::remove(p0_temp.c_str());
        std::fprintf(stderr, "party stock key streams did not share masks\n");
        return 1;
    }
    nonlinear::Header p1_header = p0_header;
    p1_header.party = 1;
    Digest p1_digest{};
    if (!write_record(p1_temp, p1_header, trunc_share1, maxpool_share1,
                      relu_share1, delta, p1_keys, p1_digest)) {
        std::remove(p0_temp.c_str());
        std::fprintf(stderr, "party-1 stock nonlinear record write failed\n");
        return 1;
    }
    p1_keys = GeneratedKeys{};

    if (!publish_pair(p0_temp, p1_temp, args.p0_output, args.p1_output)) {
        std::fprintf(stderr, "bilateral stock nonlinear publish failed\n");
        return 1;
    }
    if (!fsync_parent(args.p0_output) || !fsync_parent(args.p1_output)) {
        std::remove(args.p0_output.c_str());
        std::remove(args.p1_output.c_str());
        std::fprintf(stderr, "stock nonlinear directory sync failed\n");
        return 1;
    }

    const uint64_t total_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - started)
            .count());
    if (args.csv_header) {
        std::printf(
            "key_source,maxpool_key_bytes_per_party,relu_key_bytes_per_party,"
            "p0_record_digest,p1_record_digest,total_us,status\n");
    }
    std::printf("stock_trusted_dealer_test_only,%zu,%zu,%s,%s,%llu,pass\n",
                maxpool_key_bytes, relu_key_bytes,
                hex_digest(p0_digest).c_str(), hex_digest(p1_digest).c_str(),
                static_cast<unsigned long long>(total_us));
    return 0;
}
