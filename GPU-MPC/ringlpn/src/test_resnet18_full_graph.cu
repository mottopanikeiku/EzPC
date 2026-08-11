// Exact known-zero ResNet18 graph execution for the source-bound Ring-LPN
// linear-record set.  All 21 Conv2D/FC records use Orca's unchanged Beaver
// consumers; all 21 stochastic truncations use the live two-party protocol;
// and the 19 stock MaxPool/ReLU/sign-extension keys come from the explicitly
// TEST-ONLY trusted compatibility adapter.  The output is a digest trace of
// every graph edge, checked independently against reconstructed mask state.
// This is systems evidence, not private/trained inference or dealerless
// nonlinear preprocessing.

#define BUF_MEM LLAMA_BUF_MEM
#include "utils/gpu_comms.h"
#undef BUF_MEM
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/helper_cuda.h"
#include "utils/misc_utils.h"

#include "fss/dcf/gpu_maxpool.h"
#include "fss/dcf/gpu_relu.h"
#include "fss/dcf/gpu_truncate.h"
#include "fss/gpu_avgpool.h"
#include "fss/gpu_local_truncate.h"
#include "fss/gpu_matmul.h"
#include "fss/gpu_conv2d.h"
#include "graph_mask_state.h"
#include "private_file.h"
#include "resnet18_graph_contract.h"
#include "secure_truncate.h"
#include "stock_nonlinear_full_record.h"
#include "linear_preprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace contract = ringlpn_resnet18;
namespace full_record = ringlpn_full_graph_record;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using T = uint64_t;
using Digest = ringlpn_freshness::Digest;
using InvocationId = ringlpn_freshness::InvocationId;
using MaskState = ringlpn_graph::MaskStateRecord<T>;
using Clock = std::chrono::steady_clock;

constexpr int kFullBw = contract::kFullBw;
constexpr int kShift = contract::kScale;
constexpr int kTruncatedBw = contract::kTruncatedBw;
constexpr uint64_t kFullMask = std::numeric_limits<uint32_t>::max();
constexpr size_t kStockCommOneGB = size_t(2) << 20;
constexpr size_t kRunHeaderBytes = 512;
constexpr size_t kRunDigestBytes = 32;
constexpr size_t kTraceLinearInput = 0;
constexpr size_t kTraceLinearOutput =
    kTraceLinearInput + contract::kLinearCount;
constexpr size_t kTraceTruncOutput =
    kTraceLinearOutput + contract::kLinearCount;
constexpr size_t kTraceStockInput =
    kTraceTruncOutput + contract::kTruncationCount;
constexpr size_t kTraceStockOutput =
    kTraceStockInput + contract::kStockKeyCount;
constexpr size_t kTraceResidualOutput =
    kTraceStockOutput + contract::kStockKeyCount;
constexpr size_t kTraceGlobalPoolOutput =
    kTraceResidualOutput + contract::kResidualCount;
constexpr size_t kTraceTerminalOutput = kTraceGlobalPoolOutput + 1;
constexpr size_t kTraceCount = kTraceTerminalOutput + 1;
constexpr size_t kMaxRunBytes = 1 << 20;
constexpr std::array<uint8_t, 8> kRunMagic = {
    'R', 'L', 'P', 'N', 'F', 'G', 'R', '1'};
constexpr uint32_t kRunVersion = 1;
static_assert(kTraceCount == 111);

struct GraphArgs {
    bool check = false;
    bool csv_header = false;
    int party = -1;
    int gpu = 0;
    std::string host = "127.0.0.1";
    int port = 58600;
    std::string ledger;
    std::string invocation_text;
    InvocationId invocation{};
    Digest manifest_digest{};
    Digest record_set_digest{};
    std::array<Digest, 2> graph_record_digests{};
    std::string record_set_root;
    std::string graph_record;
    std::string output;
    std::string p0_graph_record;
    std::string p1_graph_record;
    std::string p0_output;
    std::string p1_output;
    std::array<ringlpn_graph::LinearArtifactBindings,
               contract::kLinearCount> linear_artifacts;
};

struct Totals {
    uint64_t total_us = 0;
    uint64_t linear_us = 0;
    uint64_t truncation_us = 0;
    uint64_t stock_us = 0;
    uint64_t global_pool_us = 0;
    uint64_t terminal_us = 0;
    uint64_t party_channel_bytes_sent = 0;
    uint64_t stock_bytes_sent = 0;
    uint64_t stock_bytes_received = 0;
    uint64_t party_channel_direction_switches = 0;
    uint64_t truncations = 0;
    uint64_t handoffs = 0;
    uint64_t dabits = 0;
    uint64_t edabits = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t triples = 0;
    uint64_t raw_key_bytes = 0;
};

struct RunHeader {
    int party = -1;
    InvocationId invocation{};
    Digest manifest_digest{};
    Digest record_set_digest{};
    Digest graph_record_digest{};
    Digest bundle_id{};
    Digest layer_identity{};
    Digest plan_digest{};
    Totals totals;
};

struct RunRecord {
    RunHeader header;
    std::array<Digest, kTraceCount> traces{};
    Digest digest{};
};

struct TruncationTotals {
    uint64_t truncations = 0;
    uint64_t handoffs = 0;
    uint64_t edabit_bits = 0;
    uint64_t dabits = 0;
    uint64_t triples = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t meaningful_share_bits = 0;
    uint64_t online_dependency_rounds = 0;
    uint64_t post_mask_dependencies = 0;
    uint64_t bytes_sent = 0;
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

bool artifact_arguments_complete(const GraphArgs &args) {
    return std::all_of(
        args.linear_artifacts.begin(), args.linear_artifacts.end(),
        [](const ringlpn_graph::LinearArtifactBindings &bindings) {
            return ringlpn_graph::artifact_bindings_complete(bindings);
        });
}

bool parse_args(int argc, char **argv, GraphArgs &args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char * {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        const char *value = nullptr;
        if (key == "--check") {
            args.check = true;
        } else if (key == "--csv-header") {
            args.csv_header = true;
        } else if (key == "--party" && (value = next())) {
            if (!parse_int(value, args.party)) return false;
        } else if (key == "--gpu" && (value = next())) {
            if (!parse_int(value, args.gpu)) return false;
        } else if (key == "--host" && (value = next())) {
            args.host = value;
        } else if (key == "--port" && (value = next())) {
            if (!parse_int(value, args.port)) return false;
        } else if (key == "--ledger" && (value = next())) {
            args.ledger = value;
        } else if (key == "--invocation-id" && (value = next())) {
            args.invocation_text = value;
        } else if (key == "--manifest-digest" && (value = next())) {
            if (!parse_hex(value, args.manifest_digest.data(), 32)) return false;
        } else if (key == "--record-set-digest" && (value = next())) {
            if (!parse_hex(value, args.record_set_digest.data(), 32)) return false;
        } else if (key == "--p0-graph-digest" && (value = next())) {
            if (!parse_hex(value, args.graph_record_digests[0].data(), 32)) {
                return false;
            }
        } else if (key == "--p1-graph-digest" && (value = next())) {
            if (!parse_hex(value, args.graph_record_digests[1].data(), 32)) {
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
        } else if (key == "--graph-record" && (value = next())) {
            args.graph_record = value;
        } else if (key == "--output" && (value = next())) {
            args.output = value;
        } else if (key == "--p0-graph-record" && (value = next())) {
            args.p0_graph_record = value;
        } else if (key == "--p1-graph-record" && (value = next())) {
            args.p1_graph_record = value;
        } else if (key == "--p0-output" && (value = next())) {
            args.p0_output = value;
        } else if (key == "--p1-output" && (value = next())) {
            args.p1_output = value;
        } else {
            return false;
        }
    }
    const auto nonzero = [](const Digest &digest) {
        return full_record::nonzero(digest.data(), digest.size());
    };
    const bool public_valid =
        contract::contract_valid() &&
        ringlpn_freshness::parse_invocation_id(args.invocation_text,
                                               args.invocation) &&
        nonzero(args.manifest_digest) && nonzero(args.record_set_digest) &&
        nonzero(args.graph_record_digests[0]) &&
        nonzero(args.graph_record_digests[1]) &&
        !args.record_set_root.empty() && artifact_arguments_complete(args);
    if (!public_valid) return false;
    if (args.check) {
        return args.party == -1 && !args.p0_graph_record.empty() &&
               !args.p1_graph_record.empty() && !args.p0_output.empty() &&
               !args.p1_output.empty() && args.graph_record.empty() &&
               args.output.empty() && args.ledger.empty();
    }
    return (args.party == 0 || args.party == 1) && args.gpu >= 0 &&
           args.port > 0 && args.port < 65533 && !args.ledger.empty() &&
           args.ledger.front() == '/' && !args.graph_record.empty() &&
           !args.output.empty() && args.p0_graph_record.empty() &&
           args.p1_graph_record.empty() && args.p0_output.empty() &&
           args.p1_output.empty();
}

void put_u32(std::vector<uint8_t> &out, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void put_u64(std::vector<uint8_t> &out, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

uint32_t get_u32(const uint8_t *in, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const uint8_t *in, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

bool add_u64(uint64_t &target, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - target) return false;
    target += value;
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

std::filesystem::path linear_record_path(const std::filesystem::path &root,
                                         size_t index, int party) {
    const char *suffix = contract::kLinearSpecs[index].kind ==
                                 contract::LinearKind::Conv2D
                             ? ".conv"
                             : ".fc";
    return layer_directory(root, index) /
           (party == 0 ? std::string("party0/key_p0") + suffix
                       : std::string("party1/key_p1") + suffix);
}

std::filesystem::path state_path(const std::filesystem::path &root,
                                 size_t index, int party) {
    return layer_directory(root, index) /
           (party == 0 ? "party0/mask.state" : "party1/mask.state");
}

bool manifest_artifact_paths_match(const GraphArgs &args) {
    const std::filesystem::path root(args.record_set_root);
    for (size_t index = 0; index < contract::kLinearCount; ++index) {
        const ringlpn_graph::LinearArtifactBindings &bindings =
            args.linear_artifacts[index];
        for (int party = 0; party < 2; ++party) {
            const char *record_label =
                party == 0 ? "p0_record" : "p1_record";
            const char *state_label = party == 0 ? "p0_state" : "p1_state";
            if (ringlpn_graph::party_record_binding(bindings, party).path !=
                linear_record_path(root, index, party)) {
                std::fprintf(stderr,
                             "[full-graph] layer %zu %s path mismatch\n",
                             index + 1, record_label);
                return false;
            }
            if (ringlpn_graph::party_state_binding(bindings, party).path !=
                state_path(root, index, party)) {
                std::fprintf(stderr,
                             "[full-graph] layer %zu %s path mismatch\n",
                             index + 1, state_label);
                return false;
            }
        }
    }
    return true;
}

class LinearMappedRecord {
  public:
    LinearMappedRecord() = default;
    LinearMappedRecord(const LinearMappedRecord &) = delete;
    LinearMappedRecord &operator=(const LinearMappedRecord &) = delete;
    ~LinearMappedRecord() { close(); }

    bool open(const ringlpn_graph::ArtifactBinding &binding,
              size_t index, int party) {
        close();
        if (index >= contract::kLinearCount || !binding.present) return false;
        const contract::LinearSpec &spec = contract::kLinearSpecs[index];
        header_bytes_ = spec.kind == contract::LinearKind::Conv2D ? 224 : 176;
        fd_ = ringlpn_graph::open_regular_nofollow(binding.path);
        if (fd_ < 0) return false;
        struct stat metadata {};
        const uint64_t expected_bytes =
            header_bytes_ +
            (spec.input_words + spec.weight_words + spec.output_words) *
                sizeof(T) +
            32;
        if (binding.bytes != expected_bytes ||
            binding.bytes > std::numeric_limits<size_t>::max() ||
            ::fstat(fd_, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
            metadata.st_size <= 0 ||
            static_cast<uint64_t>(metadata.st_size) != binding.bytes) {
            close();
            return false;
        }
        size_ = static_cast<size_t>(binding.bytes);
        mapped_ = static_cast<const uint8_t *>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_ == MAP_FAILED) {
            mapped_ = nullptr;
            close();
            return false;
        }
        const std::array<uint8_t, 8> conv_magic = {
            'R','L','P','N','C','V','2','P'};
        const std::array<uint8_t, 8> fc_magic = {
            'R','L','P','N','F','C','2','P'};
        const auto &magic = spec.kind == contract::LinearKind::Conv2D
                                ? conv_magic
                                : fc_magic;
        Digest calculated{};
        Digest manifest_digest{};
        sid_ = get_u64(mapped_, 16);
        bool ok = std::equal(magic.begin(), magic.end(), mapped_) &&
                  get_u32(mapped_, 8) == 3 &&
                  get_u32(mapped_, 12) == static_cast<uint32_t>(party) &&
                  sid_ != 0 &&
                  get_u32(mapped_, 24) == 128 &&
                  get_u32(mapped_, 28) == kFullBw &&
                  get_u64(mapped_, 60) != 0 &&
                  get_u64(mapped_, 68) ==
                      spec.input_words + spec.weight_words + spec.output_words &&
                  ringlpn_freshness::digest(mapped_, size_ - 32, calculated) &&
                  ringlpn_freshness::digest(mapped_, size_, manifest_digest) &&
                  manifest_digest == binding.sha256 &&
                  std::equal(calculated.begin(), calculated.end(),
                             mapped_ + size_ - 32);
        if (spec.kind == contract::LinearKind::Conv2D) {
            ok = ok && get_u32(mapped_, 32) == spec.n &&
                 get_u32(mapped_, 36) == spec.h &&
                 get_u32(mapped_, 40) == spec.w &&
                 get_u32(mapped_, 76) == spec.ci &&
                 get_u32(mapped_, 80) == spec.fh &&
                 get_u32(mapped_, 84) == spec.fw &&
                 get_u32(mapped_, 88) == spec.co &&
                 get_u32(mapped_, 92) == spec.padding &&
                 get_u32(mapped_, 96) == spec.stride &&
                 std::all_of(mapped_ + 100, mapped_ + 128,
                             [](uint8_t byte) { return byte == 0; });
            std::copy(mapped_ + 128, mapped_ + 144, invocation_.begin());
            std::copy(mapped_ + 144, mapped_ + 176, ledger_digest_.begin());
        } else {
            ok = ok && get_u32(mapped_, 32) == spec.rows &&
                 get_u32(mapped_, 36) == spec.inner &&
                 get_u32(mapped_, 40) == spec.cols &&
                 get_u32(mapped_, 76) == 0;
            std::copy(mapped_ + 80, mapped_ + 96, invocation_.begin());
            std::copy(mapped_ + 96, mapped_ + 128, ledger_digest_.begin());
        }
        ok = ok && full_record::nonzero(invocation_.data(), invocation_.size()) &&
             full_record::nonzero(ledger_digest_.data(), ledger_digest_.size());
        if (!ok) {
            close();
            return false;
        }
        digest_ = calculated;
        payload_ = reinterpret_cast<const T *>(mapped_ + header_bytes_);
        return true;
    }

    void close() {
        if (mapped_ != nullptr) ::munmap(const_cast<uint8_t *>(mapped_), size_);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        mapped_ = nullptr;
        size_ = 0;
        header_bytes_ = 0;
        payload_ = nullptr;
        sid_ = 0;
    }

    const T *payload() const { return payload_; }
    const Digest &digest() const { return digest_; }
    const InvocationId &invocation() const { return invocation_; }
    uint64_t sid() const { return sid_; }

  private:
    int fd_ = -1;
    const uint8_t *mapped_ = nullptr;
    size_t size_ = 0;
    size_t header_bytes_ = 0;
    const T *payload_ = nullptr;
    uint64_t sid_ = 0;
    Digest digest_{};
    Digest ledger_digest_{};
    InvocationId invocation_{};
};

bool load_local_linear(const GraphArgs &args,
                       const full_record::MappedRecord &graph,
                       size_t index, LinearMappedRecord &linear,
                       MaskState &state) {
    const contract::LinearSpec &spec = contract::kLinearSpecs[index];
    const std::filesystem::path root(args.record_set_root);
    const std::filesystem::path expected_linear_path =
        linear_record_path(root, index, args.party);
    const std::filesystem::path expected_mask_path =
        state_path(root, index, args.party);
    const ringlpn_graph::ArtifactBinding &linear_binding =
        ringlpn_graph::party_record_binding(args.linear_artifacts[index],
                                            args.party);
    const ringlpn_graph::ArtifactBinding &state_binding =
        ringlpn_graph::party_state_binding(args.linear_artifacts[index],
                                           args.party);
    const char *linear_label = args.party == 0 ? "p0_record" : "p1_record";
    const char *state_label = args.party == 0 ? "p0_state" : "p1_state";
    if (linear_binding.path != expected_linear_path) {
        std::fprintf(stderr, "[full-graph] layer %zu %s path mismatch\n",
                     index + 1, linear_label);
        return false;
    }
    if (state_binding.path != expected_mask_path) {
        std::fprintf(stderr, "[full-graph] layer %zu %s path mismatch\n",
                     index + 1, state_label);
        return false;
    }
    const bool linear_opened =
        linear.open(linear_binding, index, args.party);
    const bool state_read =
        ringlpn_graph::read_mask_state(state_binding, state);
    if (!linear_opened || !state_read) {
        std::fprintf(stderr,
                     "[full-graph] layer %zu artifact rejected "
                     "(%s=%d %s=%d)\n",
                     index + 1, linear_label, linear_opened ? 1 : 0,
                     state_label, state_read ? 1 : 0);
        return false;
    }
    const T *payload = linear.payload();
    const bool linear_graph_digest =
        linear.digest() == graph.header().linear_record_digests[index];
    const bool state_graph_digest =
        state.digest == graph.header().mask_state_digests[index];
    const bool header_matches =
        state.header.party == args.party &&
        state.header.sid == linear.sid() &&
        state.header.layer_ordinal == index + 1 &&
        state.header.input_bw == kFullBw &&
        state.header.output_bw == kFullBw &&
        state.header.input_words == spec.input_words &&
        state.header.output_words == spec.output_words;
    const bool invocation_matches =
        state.header.invocation_id == linear.invocation();
    const bool record_digest_matches =
        state.header.linear_record_digest == linear.digest();
    const bool sizes_match =
        state.input_mask_share.size() == spec.input_words &&
        state.output_mask_share.size() == spec.output_words;
    const bool input_matches =
        sizes_match &&
        std::equal(state.input_mask_share.begin(),
                   state.input_mask_share.end(), payload);
    // The state output is the independently generated y-share, not the
    // serialized Beaver c-share. Its state digest and atomic linear-record
    // binding above are the integrity relation.
    const bool ok = linear_graph_digest && state_graph_digest &&
                    header_matches && invocation_matches &&
                    record_digest_matches && sizes_match && input_matches;
    if (!ok) {
        std::fprintf(
            stderr,
            "[full-graph] local record binding rejected at order %zu "
            "(linear_graph=%d state_graph=%d header=%d invocation=%d "
            "record_digest=%d sizes=%d input=%d)\n",
            index + 1, linear_graph_digest ? 1 : 0,
            state_graph_digest ? 1 : 0, header_matches ? 1 : 0,
            invocation_matches ? 1 : 0, record_digest_matches ? 1 : 0,
            sizes_match ? 1 : 0, input_matches ? 1 : 0);
    }
    return ok;
}

bool derive_run_identities(const GraphArgs &args, Digest &layer_identity,
                           Digest &plan_digest) {
    static constexpr uint8_t layer_domain[] =
        "RINGLPN-RESNET18-FULL-KNOWN-ZERO-GRAPH-V1";
    std::vector<uint8_t> layer(layer_domain,
                               layer_domain + sizeof(layer_domain) - 1);
    layer.insert(layer.end(), args.manifest_digest.begin(),
                 args.manifest_digest.end());
    layer.insert(layer.end(), args.record_set_digest.begin(),
                 args.record_set_digest.end());
    for (const Digest &digest : args.graph_record_digests) {
        layer.insert(layer.end(), digest.begin(), digest.end());
    }
    std::vector<uint8_t> counts(8 * sizeof(uint64_t), 0);
    put_u64(counts, 0, contract::kLinearCount);
    put_u64(counts, 8, contract::kTruncationCount);
    put_u64(counts, 16, contract::kStockKeyCount);
    put_u64(counts, 24, contract::kResidualCount);
    put_u64(counts, 32, contract::kRemaskCount);
    put_u64(counts, 40, contract::kLinearCount +
                            contract::kTruncationCount +
                            contract::kStockKeyCount + 1);
    put_u64(counts, 48, kTraceCount);
    put_u64(counts, 56, contract::kClassifierOutputWords);
    layer.insert(layer.end(), counts.begin(), counts.end());
    if (!ringlpn_freshness::digest(layer.data(), layer.size(),
                                   layer_identity)) {
        return false;
    }
    static constexpr uint8_t plan_domain[] =
        "RINGLPN-RESNET18-FULL-GRAPH-TRACE-PLAN-V1";
    std::vector<uint8_t> plan(plan_domain,
                              plan_domain + sizeof(plan_domain) - 1);
    plan.insert(plan.end(), layer_identity.begin(), layer_identity.end());
    for (const contract::LinearSpec &spec : contract::kLinearSpecs) {
        std::vector<uint8_t> row(7 * sizeof(uint64_t), 0);
        put_u64(row, 0, spec.input_words);
        put_u64(row, 8, spec.weight_words);
        put_u64(row, 16, spec.output_words);
        put_u64(row, 24, spec.stream_position);
        put_u64(row, 32, static_cast<uint64_t>(spec.kind));
        put_u64(row, 40, spec.padding);
        put_u64(row, 48, spec.stride);
        plan.insert(plan.end(), row.begin(), row.end());
    }
    for (const contract::StockKeySpec &spec : contract::kStockKeySpecs) {
        std::vector<uint8_t> row(5 * sizeof(uint64_t), 0);
        put_u64(row, 0, spec.stream_position);
        put_u64(row, 8, static_cast<uint64_t>(spec.kind));
        put_u64(row, 16, spec.input_words);
        put_u64(row, 24, spec.output_words);
        put_u64(row, 32,
                (static_cast<uint64_t>(spec.input_bw) << 32) |
                    static_cast<uint32_t>(spec.output_bw));
        plan.insert(plan.end(), row.begin(), row.end());
    }
    return ringlpn_freshness::digest(plan.data(), plan.size(), plan_digest);
}

bool exchange_preflight(PartyChannel &channel, const GraphArgs &args,
                        const Digest &layer_identity, bool local_valid) {
    std::array<uint8_t, 129> mine{};
    std::array<uint8_t, 129> peer{};
    static constexpr uint8_t magic[16] = {
        'R','L','P','N','F','U','L','L','G','R','A','P','H','P','F','1'};
    std::copy(std::begin(magic), std::end(magic), mine.begin());
    std::copy(args.invocation.begin(), args.invocation.end(), mine.begin() + 16);
    std::copy(layer_identity.begin(), layer_identity.end(), mine.begin() + 32);
    std::copy(args.manifest_digest.begin(), args.manifest_digest.end(),
              mine.begin() + 64);
    std::copy(args.record_set_digest.begin(), args.record_set_digest.end(),
              mine.begin() + 96);
    mine[128] = local_valid ? 1 : 0;
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    return std::equal(mine.begin(), mine.begin() + 128, peer.begin()) &&
           mine[128] == 1 && peer[128] == 1;
}

std::vector<T> reconstruct_additive(const T *mine, size_t words, int bits,
                                    PartyChannel &channel) {
    std::vector<T> local(mine, mine + words);
    std::vector<T> peer(words);
    channel.exchange_bytes(reinterpret_cast<const uint8_t *>(local.data()),
                           reinterpret_cast<uint8_t *>(peer.data()),
                           words * sizeof(T));
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    for (size_t i = 0; i < words; ++i) {
        local[i] = (local[i] + peer[i]) & mask;
    }
    return local;
}

std::vector<T> reconstruct_additive(const std::vector<T> &mine, int bits,
                                    PartyChannel &channel) {
    return reconstruct_additive(mine.data(), mine.size(), bits, channel);
}

bool trace_values(const std::vector<T> &values, Digest &digest) {
    if (values.empty() ||
        values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return false;
    }
    std::vector<uint8_t> encoded(values.size() * sizeof(T));
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t byte = 0; byte < sizeof(T); ++byte) {
            encoded[i * sizeof(T) + byte] =
                static_cast<uint8_t>(values[i] >> (8 * byte));
        }
    }
    return ringlpn_freshness::digest(encoded.data(), encoded.size(), digest);
}

bool copy_to_gpu(const T *source, size_t words, T **destination,
                 const char *label) {
    cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(destination), words * sizeof(T));
    if (status == cudaSuccess) {
        status = cudaMemcpy(*destination, source, words * sizeof(T),
                            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[full-graph] CUDA %s failed: %s\n", label,
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool execute_linear_kernel(const contract::LinearSpec &spec,
                           const LinearMappedRecord &linear, int party,
                           const std::vector<T> &public_input,
                           const std::vector<T> &public_weight,
                           std::vector<T> &own_output) {
    if (public_input.size() != spec.input_words ||
        public_weight.size() != spec.weight_words) {
        std::fprintf(stderr,
                     "[full-graph] linear shape mismatch "
                     "(input=%zu expected_input=%llu weight=%zu "
                     "expected_weight=%llu)\n",
                     public_input.size(),
                     static_cast<unsigned long long>(spec.input_words),
                     public_weight.size(),
                     static_cast<unsigned long long>(spec.weight_words));
        return false;
    }
    const T *payload = linear.payload();
    T *d_input = nullptr;
    T *d_weight = nullptr;
    T *d_a = nullptr;
    T *d_b = nullptr;
    T *d_output = nullptr;
    bool ok = copy_to_gpu(public_input.data(), public_input.size(), &d_input,
                          "linear public-input copy") &&
              copy_to_gpu(public_weight.data(), public_weight.size(), &d_weight,
                          "linear public-weight copy") &&
              copy_to_gpu(payload, spec.input_words, &d_a,
                          "linear input-mask copy") &&
              copy_to_gpu(payload + spec.input_words, spec.weight_words, &d_b,
                          "linear weight-mask copy");
    Stats stats;
    if (ok && spec.kind == contract::LinearKind::Conv2D) {
        ringlpn_linear::ProtocolParameters protocol;
        protocol.qbits = 128;
        protocol.bw = kFullBw;
        const ringlpn_linear::Conv2dShape shape{
            spec.n, spec.h, spec.w, spec.ci, spec.fh,
            spec.fw, spec.co, spec.padding, spec.stride};
        ringlpn_linear::LinearPlan plan;
        const ringlpn_linear::Status plan_status =
            ringlpn_linear::Conv2dPreprocessor::plan(shape, protocol, plan);
        ok = plan_status == ringlpn_linear::Status::Ok &&
             plan.input_words == spec.input_words &&
             plan.weight_words == spec.weight_words &&
             plan.output_words == spec.output_words;
        if (!ok) {
            std::fprintf(stderr,
                         "[full-graph] convolution contract mismatch "
                         "(status=%s A=%llu/%llu B=%llu/%llu C=%llu/%llu)\n",
                         ringlpn_linear::status_name(plan_status),
                         static_cast<unsigned long long>(plan.input_words),
                         static_cast<unsigned long long>(spec.input_words),
                         static_cast<unsigned long long>(plan.weight_words),
                         static_cast<unsigned long long>(spec.weight_words),
                         static_cast<unsigned long long>(plan.output_words),
                         static_cast<unsigned long long>(spec.output_words));
        }
        if (ok) {
            GPUConv2DKey<T> key{};
            key.p = {kFullBw, kFullBw, spec.n, spec.h, spec.w, spec.ci,
                     spec.fh, spec.fw, spec.co, spec.padding, spec.padding,
                     spec.padding, spec.padding, spec.stride, spec.stride,
                     plan.output_h, plan.output_w};
            key.p.size_I = static_cast<size_t>(plan.input_words);
            key.p.size_F = static_cast<size_t>(plan.weight_words);
            key.p.size_O = static_cast<size_t>(plan.output_words);
            key.mem_size_I = key.p.size_I * sizeof(T);
            key.mem_size_F = key.p.size_F * sizeof(T);
            key.mem_size_O = key.p.size_O * sizeof(T);
            key.I = const_cast<T *>(payload);
            key.F = const_cast<T *>(payload + spec.input_words);
            key.O = const_cast<T *>(payload + spec.input_words +
                                    spec.weight_words);
            d_output = gpuConv2DBeaver<T>(key, party, d_input, d_weight,
                                          d_a, d_b, nullptr, &stats, 0);
        }
    } else if (ok) {
        MatmulParams params;
        params.batchSz = 1;
        params.M = spec.rows;
        params.K = spec.inner;
        params.N = spec.cols;
        stdInit(params, kFullBw, 0);
        ok = params.size_A == spec.input_words &&
             params.size_B == spec.weight_words &&
             params.size_C == spec.output_words;
        if (ok) {
            GPUMatmulKey<T> key;
            key.mem_size_A = spec.input_words * sizeof(T);
            key.mem_size_B = spec.weight_words * sizeof(T);
            key.mem_size_C = spec.output_words * sizeof(T);
            key.A = const_cast<T *>(payload);
            key.B = const_cast<T *>(payload + spec.input_words);
            key.C = const_cast<T *>(payload + spec.input_words +
                                    spec.weight_words);
            d_output = gpuMatmulBeaver<T>(params, key, party, d_input,
                                          d_weight, d_a, d_b, nullptr, &stats);
        }
    }
    if (ok && d_output != nullptr) {
        own_output.resize(static_cast<size_t>(spec.output_words));
        const cudaError_t copied =
            cudaMemcpy(own_output.data(), d_output,
                       own_output.size() * sizeof(T), cudaMemcpyDeviceToHost);
        const cudaError_t synchronized =
            copied == cudaSuccess ? cudaDeviceSynchronize() : copied;
        ok = copied == cudaSuccess && synchronized == cudaSuccess;
        if (!ok) {
            std::fprintf(stderr,
                         "[full-graph] CUDA linear output failed "
                         "(copy=%s synchronize=%s)\n",
                         cudaGetErrorString(copied),
                         cudaGetErrorString(synchronized));
        }
    } else {
        if (ok) {
            std::fprintf(stderr,
                         "[full-graph] linear consumer returned null output\n");
        }
        ok = false;
    }
    if (d_input != nullptr) cudaFree(d_input);
    if (d_weight != nullptr) cudaFree(d_weight);
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_output != nullptr) cudaFree(d_output);
    return ok;
}

bool add_truncation_totals(
    TruncationTotals &total,
    const ringlpn_2pc::SecureTruncateCounters &one) {
    uint64_t bytes = 0;
    return add_u64(bytes, one.preflight_bytes_sent) &&
           add_u64(bytes, one.correlation_bytes_sent) &&
           add_u64(bytes, one.online_bytes_sent) &&
           add_u64(total.truncations, one.truncations) &&
           add_u64(total.handoffs, one.handoffs) &&
           add_u64(total.edabit_bits, one.edabit_bits) &&
           add_u64(total.dabits, one.dabits) &&
           add_u64(total.triples, one.triples) &&
           add_u64(total.logical_opened_bits, one.logical_opened_bits) &&
           add_u64(total.meaningful_share_bits,
                   one.meaningful_share_bits) &&
           add_u64(total.online_dependency_rounds,
                   one.online_dependency_rounds) &&
           add_u64(total.post_mask_dependencies,
                   one.post_mask_dependencies) &&
           add_u64(total.bytes_sent, bytes);
}

bool secure_truncate(size_t index, const std::vector<T> &public_input,
                     const std::vector<T> &input_mask_share,
                     const T *next_mask_share, const GraphArgs &args,
                     const Digest &layer_identity, PartyChannel &channel,
                     PartyRandom &random, std::vector<T> &public_output,
                     TruncationTotals &totals) {
    if (index >= contract::kTruncationCount ||
        public_input.size() != contract::kTruncationSpecs[index].words ||
        input_mask_share.size() != public_input.size()) {
        return false;
    }
    size_t cursor = 0;
    uint64_t chunk = 0;
    while (cursor < public_input.size()) {
        const size_t count = std::min(
            ringlpn_2pc::kMaxSecureTruncateBatch,
            public_input.size() - cursor);
        ringlpn_2pc::SecureTruncateParams params;
        params.bw = kFullBw;
        params.shift = kShift;
        params.count = count;
        ringlpn_freshness::Coordinates coordinates;
        coordinates.kind = ringlpn_freshness::Kind::kConversionEdabit;
        coordinates.layer = index + 1;
        coordinates.phase = ringlpn_freshness::Phase::kConvertCorrelation;
        coordinates.primitive_ordinal =
            contract::kTruncationSpecs[index].stream_position;
        coordinates.conversion_chunk = chunk;
        if (!ringlpn_freshness::derive_correlation_id(
                args.invocation, layer_identity, coordinates,
                params.correlation_id)) {
            return false;
        }
        params.sid = ringlpn_freshness::compatibility_handle(
            params.correlation_id);
        std::vector<T> masked_chunk(public_input.begin() + cursor,
                                    public_input.begin() + cursor + count);
        std::vector<T> mask_chunk(input_mask_share.begin() + cursor,
                                  input_mask_share.begin() + cursor + count);
        std::vector<T> next_chunk(next_mask_share + cursor,
                                  next_mask_share + cursor + count);
        std::vector<T> output_share;
        std::vector<T> public_chunk;
        ringlpn_2pc::SecureTruncateCounters one;
        if (!ringlpn_2pc::secure_stochastic_truncate_batch(
                params, masked_chunk, mask_chunk, next_chunk, channel,
                random, output_share, public_chunk, one) ||
            !add_truncation_totals(totals, one)) {
            return false;
        }
        public_output.insert(public_output.end(), public_chunk.begin(),
                             public_chunk.end());
        cursor += count;
        ++chunk;
    }
    return public_output.size() == public_input.size();
}

MaxpoolParams exact_maxpool_params() {
    MaxpoolParams params = {
        kTruncatedBw, kTruncatedBw, 0, 0, kFullBw,
        1, 112, 112, 64,
        3, 3,
        2, 2,
        1, 1,
        1, 1,
        0, 0, false};
    initPoolParams(params);
    return params;
}

void release_maxpool_key(dcf::GPUMaxpoolKey<T> &key) {
    for (int round = 1; round <= 8; ++round) {
        delete[] key.reluKey[round].dreluKey.dcfKey.dcfTreeKey;
    }
    delete[] key.reluKey;
    delete[] key.andKey;
}

void release_relu_key(dcf::GPUReluExtendKey<T> &key) {
    delete[] key.dReluKey.dcfKey.dcfTreeKey;
}

void release_sign_key(dcf::GPUSignExtendKey<T> &key) {
    delete[] key.dcfKey.dcfKey.dcfTreeKey;
}

class ScopedStdoutToStderr {
  public:
    ScopedStdoutToStderr() {
        std::fflush(stdout);
        saved_ = ::dup(STDOUT_FILENO);
        ok_ = saved_ >= 0 && ::dup2(STDERR_FILENO, STDOUT_FILENO) >= 0;
    }
    ScopedStdoutToStderr(const ScopedStdoutToStderr &) = delete;
    ScopedStdoutToStderr &operator=(const ScopedStdoutToStderr &) = delete;
    ~ScopedStdoutToStderr() { restore(); }
    void restore() {
        std::fflush(stdout);
        if (saved_ >= 0) {
            ::dup2(saved_, STDOUT_FILENO);
            ::close(saved_);
            saved_ = -1;
        }
    }
    bool ok() const { return ok_; }

  private:
    int saved_ = -1;
    bool ok_ = false;
};

bool execute_stock(size_t index, int party,
                   const full_record::MappedRecord &graph, GpuPeer &peer,
                   AESGlobalContext &gaes, Stats &stats,
                   const std::vector<T> &public_input,
                   std::vector<T> &public_output) {
    if (index >= contract::kStockKeyCount ||
        public_input.size() != contract::kStockKeySpecs[index].input_words) {
        return false;
    }
    const contract::StockKeySpec &spec = contract::kStockKeySpecs[index];
    const full_record::StockBinding &binding = graph.stock_binding(index);
    uint8_t *cursor = const_cast<uint8_t *>(graph.raw_key(index));
    uint8_t *const key_end = cursor + binding.key_bytes;
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(public_input.data())),
        public_input.size() * sizeof(T), nullptr));
    T *d_output = nullptr;
    bool ok = d_input != nullptr;
    u32 *d_relu_bits = nullptr;
    if (ok && spec.kind == contract::StockKeyKind::MaxPool) {
        MaxpoolParams params = exact_maxpool_params();
        dcf::GPUMaxpoolKey<T> key =
            dcf::readGPUMaxpoolKey<T>(params, &cursor);
        key.andKey = new GPUAndKey[9]();
        ok = cursor == key_end &&
             std::all_of(key.reluKey + 1, key.reluKey + 9,
                         [&](const dcf::GPU2RoundReLUKey<T> &one) {
                             return one.bin == spec.input_bw &&
                                    one.bout == spec.input_bw &&
                                    one.N == static_cast<int>(spec.output_words);
                         });
        if (ok) {
            d_output = dcf::gpuMaxPool<T>(&peer, party, params, key,
                                          d_input, nullptr, &gaes, &stats);
        }
        release_maxpool_key(key);
    } else if (ok && spec.kind == contract::StockKeyKind::ReluExtend) {
        dcf::GPUReluExtendKey<T> key =
            dcf::readGPUReluExtendKey<T>(&cursor);
        ok = cursor == key_end && key.bin == spec.input_bw &&
             key.bout == spec.output_bw &&
             key.N == static_cast<int>(spec.output_words);
        if (ok) {
            std::pair<u32 *, T *> result = dcf::gpuReluExtend<T>(
                &peer, party, key, d_input, &gaes, &stats);
            d_relu_bits = result.first;
            d_output = result.second;
        }
        release_relu_key(key);
    } else if (ok) {
        dcf::GPUSignExtendKey<T> key =
            dcf::readGPUSignExtendKey<T>(&cursor);
        ok = cursor == key_end && key.bin == spec.input_bw &&
             key.bout == spec.output_bw &&
             key.N == static_cast<int>(spec.output_words);
        if (ok) {
            dcf::gpuSignExtend<T>(key, party, &peer, d_input, &gaes, &stats);
            d_output = d_input;
            d_input = nullptr;
        }
        release_sign_key(key);
    }
    if (ok && d_output != nullptr) {
        public_output.resize(static_cast<size_t>(spec.output_words));
        ok = cudaMemcpy(public_output.data(), d_output,
                        public_output.size() * sizeof(T),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaDeviceSynchronize() == cudaSuccess;
    } else {
        ok = false;
    }
    if (d_relu_bits != nullptr) gpuFree(d_relu_bits);
    if (d_output != nullptr) gpuFree(d_output);
    if (d_input != nullptr) gpuFree(d_input);
    return ok;
}

std::vector<T> add_values(const std::vector<T> &left,
                          const std::vector<T> &right, int bits) {
    if (left.size() != right.size() || bits <= 0 || bits > 32) return {};
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    std::vector<T> output(left.size());
    for (size_t i = 0; i < left.size(); ++i) {
        output[i] = (left[i] + right[i]) & mask;
    }
    return output;
}

using PublicTruncationMap = std::array<
    const std::vector<T> *, contract::kTruncationCount>;
using PublicStockMap = std::array<
    const std::vector<T> *, contract::kStockKeyCount>;
using PublicResidualMap = std::array<
    std::vector<T>, contract::kResidualCount>;

const std::vector<T> *resolve_value(
    contract::ValueSource source,
    const PublicTruncationMap &public_truncations,
    const PublicStockMap &public_stock,
    const PublicResidualMap &public_residuals) {
    if (source.kind == contract::ValueSourceKind::Truncation &&
        source.index < public_truncations.size()) {
        return public_truncations[source.index];
    }
    if (source.kind == contract::ValueSourceKind::Stock &&
        source.index < public_stock.size()) {
        return public_stock[source.index];
    }
    if (source.kind == contract::ValueSourceKind::Residual &&
        source.index < public_residuals.size() &&
        !public_residuals[source.index].empty()) {
        return &public_residuals[source.index];
    }
    return nullptr;
}

std::vector<T> remask_for_linear(const full_record::MappedRecord &graph,
                                 size_t target,
                                 const PublicStockMap &public_stock) {
    if (target == 0 || target >= contract::kLinearCount) {
        return {};
    }
    const size_t remask_index = target - 1;
    const contract::RemaskSpec &spec =
        contract::kRemaskSpecs[remask_index];
    const size_t source_index = contract::stock_index(spec.source);
    const std::vector<T> *source = source_index < public_stock.size()
                                      ? public_stock[source_index]
                                      : nullptr;
    const full_record::RemaskBinding &binding =
        graph.remask_binding(remask_index);
    if (source == nullptr ||
        source->size() != contract::kLinearSpecs[target].input_words ||
        spec.target_linear_index != target ||
        spec.words != source->size() ||
        binding.target_linear_index != target ||
        binding.source != spec.source ||
        binding.words != source->size()) {
        return {};
    }
    const T *delta = graph.remask_delta(remask_index);
    std::vector<T> output(source->size());
    for (size_t i = 0; i < source->size(); ++i) {
        output[i] = ((*source)[i] + delta[i]) & kFullMask;
    }
    return output;
}

std::vector<T> global_pool_host(const T *input, size_t words) {
    if (words != contract::kGlobalPoolInputWords) return {};
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

bool execute_global_pool(const std::vector<T> &input,
                         std::vector<T> &output) {
    if (input.size() != contract::kGlobalPoolInputWords) return false;
    AvgPoolParams params = {
        kFullBw, kFullBw, kShift, kShift, 0,
        1, 7, 7, 512,
        7, 7,
        1, 1,
        0, 0,
        0, 0,
        0, 0, false};
    initPoolParams(params);
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(input.data())),
        input.size() * sizeof(T), nullptr));
    Stats stats;
    T *d_output = gpuAddPool<T>(params, d_input, &stats);
    output.resize(contract::kGlobalPoolOutputWords);
    const bool ok = cudaMemcpy(output.data(), d_output,
                               output.size() * sizeof(T),
                               cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess;
    gpuFree(d_input);
    gpuFree(d_output);
    return ok;
}

bool execute_terminal(int party, const std::vector<T> &public_input,
                      const T *terminal_mask,
                      std::vector<T> &output) {
    if (public_input.size() != contract::kClassifierOutputWords) return false;
    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(public_input.data())),
        public_input.size() * sizeof(T), nullptr));
    Stats stats;
    unmaskValues(kFullBw, static_cast<int>(public_input.size()), d_input,
                 const_cast<T *>(terminal_mask), &stats);
    gpuLocalTr<T, T, ars>(party, kFullBw, kShift,
                          static_cast<int>(public_input.size()), d_input, true);
    output.resize(public_input.size());
    const bool ok = cudaMemcpy(output.data(), d_input,
                               output.size() * sizeof(T),
                               cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess;
    gpuFree(d_input);
    return ok;
}

std::vector<uint8_t> encode_run_header(const RunHeader &header) {
    std::vector<uint8_t> out(kRunHeaderBytes, 0);
    std::copy(kRunMagic.begin(), kRunMagic.end(), out.begin());
    put_u32(out, 8, kRunVersion);
    put_u32(out, 12, static_cast<uint32_t>(header.party));
    put_u32(out, 16, kFullBw);
    put_u32(out, 20, kShift);
    put_u32(out, 24, kTraceCount);
    std::copy(header.invocation.begin(), header.invocation.end(),
              out.begin() + 32);
    std::copy(header.manifest_digest.begin(), header.manifest_digest.end(),
              out.begin() + 48);
    std::copy(header.record_set_digest.begin(),
              header.record_set_digest.end(), out.begin() + 80);
    std::copy(header.graph_record_digest.begin(),
              header.graph_record_digest.end(), out.begin() + 112);
    std::copy(header.bundle_id.begin(), header.bundle_id.end(),
              out.begin() + 144);
    std::copy(header.layer_identity.begin(), header.layer_identity.end(),
              out.begin() + 176);
    std::copy(header.plan_digest.begin(), header.plan_digest.end(),
              out.begin() + 208);
    const Totals &t = header.totals;
    put_u64(out, 240, t.total_us);
    put_u64(out, 248, t.linear_us);
    put_u64(out, 256, t.truncation_us);
    put_u64(out, 264, t.stock_us);
    put_u64(out, 272, t.global_pool_us);
    put_u64(out, 280, t.terminal_us);
    put_u64(out, 288, t.party_channel_bytes_sent);
    put_u64(out, 296, t.stock_bytes_sent);
    put_u64(out, 304, t.stock_bytes_received);
    put_u64(out, 312, t.party_channel_direction_switches);
    put_u64(out, 320, t.truncations);
    put_u64(out, 328, t.handoffs);
    put_u64(out, 336, t.dabits);
    put_u64(out, 344, t.edabits);
    put_u64(out, 352, t.logical_opened_bits);
    put_u64(out, 360, t.triples);
    put_u64(out, 368, t.raw_key_bytes);
    return out;
}

bool decode_run_header(const uint8_t *in, size_t size, RunHeader &header) {
    if (size < kRunHeaderBytes ||
        !std::equal(kRunMagic.begin(), kRunMagic.end(), in) ||
        get_u32(in, 8) != kRunVersion || get_u32(in, 16) != kFullBw ||
        get_u32(in, 20) != kShift || get_u32(in, 24) != kTraceCount ||
        get_u32(in, 28) != 0 ||
        !std::all_of(in + 376, in + kRunHeaderBytes,
                     [](uint8_t byte) { return byte == 0; })) {
        return false;
    }
    header.party = static_cast<int>(get_u32(in, 12));
    std::copy(in + 32, in + 48, header.invocation.begin());
    std::copy(in + 48, in + 80, header.manifest_digest.begin());
    std::copy(in + 80, in + 112, header.record_set_digest.begin());
    std::copy(in + 112, in + 144, header.graph_record_digest.begin());
    std::copy(in + 144, in + 176, header.bundle_id.begin());
    std::copy(in + 176, in + 208, header.layer_identity.begin());
    std::copy(in + 208, in + 240, header.plan_digest.begin());
    Totals &t = header.totals;
    t.total_us = get_u64(in, 240);
    t.linear_us = get_u64(in, 248);
    t.truncation_us = get_u64(in, 256);
    t.stock_us = get_u64(in, 264);
    t.global_pool_us = get_u64(in, 272);
    t.terminal_us = get_u64(in, 280);
    t.party_channel_bytes_sent = get_u64(in, 288);
    t.stock_bytes_sent = get_u64(in, 296);
    t.stock_bytes_received = get_u64(in, 304);
    t.party_channel_direction_switches = get_u64(in, 312);
    t.truncations = get_u64(in, 320);
    t.handoffs = get_u64(in, 328);
    t.dabits = get_u64(in, 336);
    t.edabits = get_u64(in, 344);
    t.logical_opened_bits = get_u64(in, 352);
    t.triples = get_u64(in, 360);
    t.raw_key_bytes = get_u64(in, 368);
    return (header.party == 0 || header.party == 1) &&
           full_record::nonzero(header.invocation.data(),
                                header.invocation.size()) &&
           full_record::nonzero(header.manifest_digest.data(), 32) &&
           full_record::nonzero(header.record_set_digest.data(), 32) &&
           full_record::nonzero(header.graph_record_digest.data(), 32) &&
           full_record::nonzero(header.bundle_id.data(), 32) &&
           full_record::nonzero(header.layer_identity.data(), 32) &&
           full_record::nonzero(header.plan_digest.data(), 32);
}

bool serialize_run_record(const RunRecord &record,
                          std::vector<uint8_t> &bytes, Digest &digest) {
    bytes = encode_run_header(record.header);
    for (const Digest &trace : record.traces) {
        if (!full_record::nonzero(trace.data(), trace.size())) return false;
        bytes.insert(bytes.end(), trace.begin(), trace.end());
    }
    if (!ringlpn_freshness::digest(bytes.data(), bytes.size(), digest)) {
        return false;
    }
    bytes.insert(bytes.end(), digest.begin(), digest.end());
    return bytes.size() ==
           kRunHeaderBytes + kTraceCount * 32 + kRunDigestBytes;
}

bool read_no_follow(const std::string &path, std::vector<uint8_t> &bytes) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat metadata {};
    bool ok = ::fstat(fd, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
              metadata.st_size > 0 &&
              metadata.st_size <= static_cast<off_t>(kMaxRunBytes);
    if (ok) bytes.resize(static_cast<size_t>(metadata.st_size));
    size_t cursor = 0;
    while (ok && cursor < bytes.size()) {
        const ssize_t got = ::read(fd, bytes.data() + cursor,
                                   bytes.size() - cursor);
        if (got <= 0) {
            ok = false;
        } else {
            cursor += static_cast<size_t>(got);
        }
    }
    uint8_t extra = 0;
    ok = ok && ::read(fd, &extra, 1) == 0;
    ::close(fd);
    return ok;
}

bool read_run_record(const std::string &path, RunRecord &record) {
    std::vector<uint8_t> bytes;
    const size_t expected_size =
        kRunHeaderBytes + kTraceCount * 32 + kRunDigestBytes;
    if (!read_no_follow(path, bytes) || bytes.size() != expected_size) {
        return false;
    }
    RunRecord parsed;
    if (!decode_run_header(bytes.data(), bytes.size(), parsed.header) ||
        !ringlpn_freshness::digest(bytes.data(), bytes.size() - 32,
                                   parsed.digest) ||
        !std::equal(parsed.digest.begin(), parsed.digest.end(),
                    bytes.end() - 32)) {
        return false;
    }
    size_t cursor = kRunHeaderBytes;
    for (Digest &trace : parsed.traces) {
        std::copy(bytes.begin() + cursor, bytes.begin() + cursor + 32,
                  trace.begin());
        cursor += 32;
    }
    record = std::move(parsed);
    return true;
}

bool write_private_atomic_pair(const GraphArgs &args, PartyChannel &channel,
                               const std::vector<uint8_t> &bytes) {
    ringlpn_private_file::AtomicWriter writer;
    const bool staged = writer.stage(args.output, bytes);
    uint8_t mine = staged ? 1 : 0;
    uint8_t peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!staged || peer != 1) return false;

    const bool published = writer.publish();
    mine = published ? 1 : 0;
    peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!published || peer != 1) return false;

    writer.commit();
    return true;
}

bool graph_record_matches_args(const GraphArgs &args,
                               const full_record::MappedRecord &graph,
                               int party) {
    const full_record::Header &header = graph.header();
    return header.party == party &&
           header.scope == full_record::kTrustedKnownZeroScope &&
           header.invocation == args.invocation &&
           header.manifest_digest == args.manifest_digest &&
           header.record_set_digest == args.record_set_digest &&
           graph.digest() == args.graph_record_digests[party] &&
           header.full_bw == kFullBw &&
           header.truncated_bw == kTruncatedBw &&
           header.scale == kShift;
}

int run_party(const GraphArgs &args) {
    const auto started = Clock::now();
    full_record::MappedRecord graph;
    Digest layer_identity{};
    Digest plan_digest{};
    std::error_code output_error;
    const bool output_absent =
        !std::filesystem::exists(args.output, output_error) &&
        !std::filesystem::exists(args.output + ".tmp", output_error);
    const bool paths_distinct =
        std::filesystem::path(args.output).lexically_normal() !=
            std::filesystem::path(args.ledger).lexically_normal() &&
        std::filesystem::path(args.output).lexically_normal() !=
            std::filesystem::path(args.graph_record).lexically_normal();
    const bool output_valid = !output_error && output_absent;
    const bool graph_opened = graph.open(args.graph_record);
    const bool graph_bound =
        graph_opened && graph_record_matches_args(args, graph, args.party);
    const bool identity_ok =
        derive_run_identities(args, layer_identity, plan_digest);
    const bool cuda_ready = cudaSetDevice(args.gpu) == cudaSuccess;
    bool local_valid = output_valid && paths_distinct && graph_bound &&
                       identity_ok && cuda_ready;
    if (!local_valid) {
        std::fprintf(
            stderr,
            "[full-graph] local static preflight rejected "
            "(output=%d paths=%d graph_opened=%d graph_bound=%d "
            "identity=%d cuda=%d)\n",
            output_valid ? 1 : 0, paths_distinct ? 1 : 0,
            graph_opened ? 1 : 0, graph_bound ? 1 : 0,
            identity_ok ? 1 : 0, cuda_ready ? 1 : 0);
    }
    // Validate all source records before any peer setup. Records are then
    // reopened one at a time for bounded-memory execution.
    for (size_t i = 0; local_valid && i < contract::kLinearCount; ++i) {
        LinearMappedRecord linear;
        MaskState state;
        if (!load_local_linear(args, graph, i, linear, state)) {
            std::fprintf(stderr,
                         "[full-graph] local linear preflight rejected at "
                         "order %zu\n",
                         i + 1);
            local_valid = false;
        }
    }
    ringlpn_freshness::Claim claim;
    const bool claim_ok = local_valid &&
        ringlpn_freshness::claim_namespace_once(
            args.ledger, args.party, args.invocation, layer_identity,
            plan_digest, claim);
    if (!local_valid || !claim_ok) {
        std::remove((args.output + ".tmp").c_str());
        std::fprintf(stderr,
                     "[full-graph] local preflight/claim rejected "
                     "(local=%d claim=%d)\n",
                     local_valid ? 1 : 0, claim_ok ? 1 : 0);
        return 2;
    }
    PartyChannel channel(args.party, args.host, args.port,
                         /*defer_ot_setup=*/true,
                         /*require_loopback_endpoints=*/true);
    if (!exchange_preflight(channel, args, layer_identity, true)) {
        std::remove((args.output + ".tmp").c_str());
        std::fprintf(stderr, "[full-graph] public preflight rejected\n");
        return 2;
    }

    if (ringlpn_linear::Conv2dPreprocessor::initialize_gpu() !=
        ringlpn_linear::Status::Ok) {
        std::fprintf(stderr, "[full-graph] GPU initialization failed\n");
        return 2;
    }
    PartyRandom random;
    TruncationTotals truncation_totals;
    Totals totals;
    totals.raw_key_bytes = graph.header().raw_key_bytes;
    std::array<Digest, kTraceCount> traces{};
    ScopedStdoutToStderr redirect;
    bool ok = redirect.ok();
    std::string failure_stage;
    auto reject = [&](std::string stage) {
        if (failure_stage.empty()) failure_stage = std::move(stage);
        ok = false;
    };
    if (!ok) failure_stage = "stdout_redirect";
    const size_t original_one_gb = OneGB;
    OneGB = kStockCommOneGB;
    GpuPeer stock_peer(true);
    stock_peer.connect(args.party, args.host, args.port + 2);
    AESGlobalContext gaes;
    initAESContext(&gaes);
    Stats stock_stats;

    PublicTruncationMap public_truncations{};
    PublicStockMap public_stock{};
    PublicResidualMap public_residuals{};
    std::array<bool, contract::kResidualCount> residual_seen{};

    auto linear = [&](size_t index, const std::vector<T> &input,
                      std::vector<T> &output, bool truncate) {
        if (!ok) return;
        const auto stage = [&](const char *name) {
            return std::string(name) + ":" + std::to_string(index + 1);
        };
        if (index >= contract::kLinearCount ||
            !trace_values(input, traces[kTraceLinearInput + index])) {
            reject(stage("linear_input"));
            return;
        }
        LinearMappedRecord record;
        MaskState state;
        if (!load_local_linear(args, graph, index, record, state)) {
            reject(stage("linear_record"));
            return;
        }
        const contract::LinearSpec &spec = contract::kLinearSpecs[index];
        std::vector<T> public_weight = reconstruct_additive(
            record.payload() + spec.input_words,
            static_cast<size_t>(spec.weight_words), kFullBw, channel);
        std::vector<T> own_output;
        const auto linear_started = Clock::now();
        if (!execute_linear_kernel(spec, record, args.party, input,
                                   public_weight, own_output)) {
            reject(stage("linear_kernel"));
        }
        if (ok) {
            output = reconstruct_additive(own_output, kFullBw, channel);
            if (!trace_values(output, traces[kTraceLinearOutput + index])) {
                reject(stage("linear_trace"));
            }
        }
        add_u64(totals.linear_us, static_cast<uint64_t>(
            std::chrono::duration<double, std::micro>(
                Clock::now() - linear_started).count()));
        if (ok && truncate) {
            std::vector<T> truncated;
            const auto trunc_started = Clock::now();
            if (!secure_truncate(index, output, state.output_mask_share,
                                 graph.trunc_share(index), args,
                                 layer_identity, channel, random, truncated,
                                 truncation_totals) ||
                !trace_values(truncated,
                              traces[kTraceTruncOutput + index])) {
                reject(stage("truncate"));
            }
            add_u64(totals.truncation_us, static_cast<uint64_t>(
                std::chrono::duration<double, std::micro>(
                    Clock::now() - trunc_started).count()));
            output = std::move(truncated);
            if (ok) public_truncations[index] = &output;
        }
    };

    auto stock = [&](size_t index, std::vector<T> &output) {
        if (!ok) return;
        const auto stage = [&](const char *name) {
            return std::string(name) + ":" + std::to_string(index + 1);
        };
        if (index >= contract::kStockKeyCount) {
            reject(stage("stock_index"));
            return;
        }
        const contract::ValueSource source_spec =
            contract::kStockInputSources[index];
        const std::vector<T> *input =
            resolve_value(source_spec, public_truncations, public_stock,
                          public_residuals);
        if (input == nullptr ||
            input->size() != contract::kStockKeySpecs[index].input_words ||
            !trace_values(*input, traces[kTraceStockInput + index])) {
            reject(stage("stock_input"));
            return;
        }
        const auto stock_started = Clock::now();
        if (!execute_stock(index, args.party, graph, stock_peer, gaes,
                           stock_stats, *input, output)) {
            reject(stage("stock_kernel"));
        } else if (!trace_values(output,
                                 traces[kTraceStockOutput + index])) {
            reject(stage("stock_trace"));
        }
        if (ok) public_stock[index] = &output;
        add_u64(totals.stock_us, static_cast<uint64_t>(
            std::chrono::duration<double, std::micro>(
                Clock::now() - stock_started).count()));
    };

    auto residual = [&](size_t index) {
        if (!ok) return;
        const auto stage = [&](const char *name) {
            return std::string(name) + ":" + std::to_string(index + 1);
        };
        if (index >= contract::kResidualCount || residual_seen[index]) {
            reject(stage("residual_index"));
            return;
        }
        const contract::ResidualSpec &spec =
            contract::kResidualSpecs[index];
        const std::vector<T> *left =
            resolve_value(spec.main, public_truncations, public_stock,
                          public_residuals);
        const std::vector<T> *right =
            resolve_value(spec.shortcut, public_truncations, public_stock,
                          public_residuals);
        if (left == nullptr || right == nullptr ||
            left->size() != spec.words || right->size() != spec.words) {
            reject(stage("residual_input"));
            return;
        }
        public_residuals[index] =
            add_values(*left, *right, kTruncatedBw);
        if (public_residuals[index].empty() ||
            !trace_values(public_residuals[index],
                          traces[kTraceResidualOutput + index])) {
            reject(stage("residual_trace"));
            return;
        }
        residual_seen[index] = true;
    };

    // Exact source order from experiments/orca/cnn.h:487-535.
    std::vector<T> input0;
    {
        LinearMappedRecord first;
        MaskState first_state;
        if (!load_local_linear(args, graph, 0, first, first_state)) {
            reject("initial_linear_record");
        } else {
            input0 = reconstruct_additive(first_state.input_mask_share,
                                          kFullBw, channel);
        }
    }

    std::vector<T> t0, mp, r2;
    linear(0, input0, t0, true); stock(0, mp); stock(1, r2);

    std::vector<T> t1, r4, t2, r7;
    linear(1, remask_for_linear(graph, 1, public_stock), t1, true);
    stock(2, r4);
    linear(2, remask_for_linear(graph, 2, public_stock), t2, true);
    residual(0); stock(3, r7);

    std::vector<T> t3, r9, t4, r12;
    linear(3, remask_for_linear(graph, 3, public_stock), t3, true);
    stock(4, r9);
    linear(4, remask_for_linear(graph, 4, public_stock), t4, true);
    residual(1); stock(5, r12);

    std::vector<T> t5, r14, t6, t7, r18;
    linear(5, remask_for_linear(graph, 5, public_stock), t5, true);
    stock(6, r14);
    linear(6, remask_for_linear(graph, 6, public_stock), t6, true);
    linear(7, remask_for_linear(graph, 7, public_stock), t7, true);
    residual(2); stock(7, r18);

    std::vector<T> t8, r20, t9, r23;
    linear(8, remask_for_linear(graph, 8, public_stock), t8, true);
    stock(8, r20);
    linear(9, remask_for_linear(graph, 9, public_stock), t9, true);
    residual(3); stock(9, r23);

    std::vector<T> t10, r25, t11, t12, r29;
    linear(10, remask_for_linear(graph, 10, public_stock), t10, true);
    stock(10, r25);
    linear(11, remask_for_linear(graph, 11, public_stock), t11, true);
    linear(12, remask_for_linear(graph, 12, public_stock), t12, true);
    residual(4); stock(11, r29);

    std::vector<T> t13, r31, t14, r34;
    linear(13, remask_for_linear(graph, 13, public_stock), t13, true);
    stock(12, r31);
    linear(14, remask_for_linear(graph, 14, public_stock), t14, true);
    residual(5); stock(13, r34);

    std::vector<T> t15, r36, t16, t17, r40;
    linear(15, remask_for_linear(graph, 15, public_stock), t15, true);
    stock(14, r36);
    linear(16, remask_for_linear(graph, 16, public_stock), t16, true);
    linear(17, remask_for_linear(graph, 17, public_stock), t17, true);
    residual(6); stock(15, r40);

    std::vector<T> t18, r42, t19, r45;
    linear(18, remask_for_linear(graph, 18, public_stock), t18, true);
    stock(16, r42);
    linear(19, remask_for_linear(graph, 19, public_stock), t19, true);
    residual(7); stock(17, r45);

    std::vector<T> pooled;
    const auto pool_started = Clock::now();
    if (ok && (!execute_global_pool(r45, pooled) ||
               !trace_values(pooled, traces[kTraceGlobalPoolOutput]))) {
        reject("global_pool");
    }
    totals.global_pool_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(Clock::now() - pool_started)
            .count());

    std::vector<T> pool_mask_share = global_pool_host(
        graph.stock_output_mask_share(17),
        graph.stock_binding(17).output_mask_words);
    std::vector<T> t20;
    const auto pool_trunc_started = Clock::now();
    if (ok && (!secure_truncate(20, pooled, pool_mask_share,
                                graph.trunc_share(20), args,
                                layer_identity, channel, random, t20,
                                truncation_totals) ||
               !trace_values(t20, traces[kTraceTruncOutput + 20]))) {
        reject("global_pool_truncate");
    }
    if (ok) public_truncations[20] = &t20;
    add_u64(totals.truncation_us, static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            Clock::now() - pool_trunc_started).count()));

    std::vector<T> sign_extended;
    stock(18, sign_extended);
    std::vector<T> classifier_public;
    linear(20, remask_for_linear(graph, 20, public_stock),
           classifier_public, false);
    std::vector<T> terminal;
    const auto terminal_started = Clock::now();
    if (ok && (!execute_terminal(args.party, classifier_public,
                                 graph.terminal_mask(), terminal) ||
               !trace_values(terminal, traces[kTraceTerminalOutput]))) {
        reject("terminal");
    }
    totals.terminal_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            Clock::now() - terminal_started).count());

    try {
        channel.finish_ots();
        totals.party_channel_bytes_sent = channel.bytes_sent();
        totals.party_channel_direction_switches = channel.direction_switches();
    } catch (...) {
        reject("finish_ots");
    }
    totals.stock_bytes_sent = stock_peer.bytesSent();
    totals.stock_bytes_received = stock_peer.bytesReceived();
    stock_peer.close();
    stock_peer.freeCommBufs(true);
    delete static_cast<SocketBuf *>(stock_peer.peer->keyBuf);
    delete stock_peer.peer;
    OneGB = original_one_gb;
    redirect.restore();

    totals.truncations = truncation_totals.truncations;
    totals.handoffs = truncation_totals.handoffs;
    totals.dabits = truncation_totals.dabits;
    totals.edabits = truncation_totals.edabit_bits;
    totals.logical_opened_bits = truncation_totals.logical_opened_bits;
    totals.triples = truncation_totals.triples;
    uint64_t expected_truncations = 0;
    for (const contract::TruncationSpec &spec : contract::kTruncationSpecs) {
        add_u64(expected_truncations, spec.words);
    }
    if (ok &&
        (!std::all_of(residual_seen.begin(), residual_seen.end(),
                      [](bool seen) { return seen; }) ||
         totals.truncations != expected_truncations ||
         totals.handoffs != expected_truncations ||
         terminal.size() != contract::kClassifierOutputWords)) {
        reject("final_invariants");
    }
    totals.total_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(Clock::now() - started)
            .count());

    RunRecord result;
    result.header.party = args.party;
    result.header.invocation = args.invocation;
    result.header.manifest_digest = args.manifest_digest;
    result.header.record_set_digest = args.record_set_digest;
    result.header.graph_record_digest = graph.digest();
    result.header.bundle_id = graph.header().bundle_id;
    result.header.layer_identity = layer_identity;
    result.header.plan_digest = plan_digest;
    result.header.totals = totals;
    result.traces = traces;
    std::vector<uint8_t> bytes;
    Digest result_digest{};
    bool serialized = ok && serialize_run_record(result, bytes, result_digest);
    if (ok && !serialized) reject("serialize");
    uint8_t ready = serialized ? 1 : 0;
    uint8_t peer_ready = 0;
    channel.exchange_bytes(&ready, &peer_ready, 1);
    if (ok && peer_ready != 1) reject("peer_not_ready");
    if (ok && !write_private_atomic_pair(args, channel, bytes)) {
        reject("bilateral_publication");
    }
    if (!ok) {
        std::fprintf(stderr,
                     "[full-graph] execution rejected at stage %s\n",
                     failure_stage.empty() ? "unknown" :
                                             failure_stage.c_str());
    }

    std::printf(
        "%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%s\n",
        args.party,
        static_cast<unsigned long long>(totals.total_us),
        static_cast<unsigned long long>(totals.linear_us),
        static_cast<unsigned long long>(totals.truncation_us),
        static_cast<unsigned long long>(totals.stock_us),
        static_cast<unsigned long long>(totals.global_pool_us),
        static_cast<unsigned long long>(totals.terminal_us),
        static_cast<unsigned long long>(totals.party_channel_bytes_sent +
                                        totals.stock_bytes_sent),
        static_cast<unsigned long long>(totals.party_channel_bytes_sent),
        static_cast<unsigned long long>(totals.stock_bytes_sent),
        static_cast<unsigned long long>(totals.stock_bytes_received),
        static_cast<unsigned long long>(totals.truncations),
        static_cast<unsigned long long>(totals.handoffs),
        static_cast<unsigned long long>(totals.dabits),
        static_cast<unsigned long long>(totals.edabits),
        static_cast<unsigned long long>(totals.logical_opened_bits),
        static_cast<unsigned long long>(totals.triples),
        static_cast<unsigned long long>(totals.raw_key_bytes),
        ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

bool same_run_headers(const RunHeader &p0, const RunHeader &p1,
                      const GraphArgs &args) {
    return p0.party == 0 && p1.party == 1 &&
           p0.invocation == args.invocation &&
           p1.invocation == args.invocation &&
           p0.manifest_digest == args.manifest_digest &&
           p1.manifest_digest == args.manifest_digest &&
           p0.record_set_digest == args.record_set_digest &&
           p1.record_set_digest == args.record_set_digest &&
           p0.graph_record_digest == args.graph_record_digests[0] &&
           p1.graph_record_digest == args.graph_record_digests[1] &&
           p0.bundle_id == p1.bundle_id &&
           p0.layer_identity == p1.layer_identity &&
           p0.plan_digest == p1.plan_digest;
}

std::vector<T> reconstruct_mask(const T *p0, const T *p1, size_t words,
                                int bits) {
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    std::vector<T> result(words);
    for (size_t i = 0; i < words; ++i) {
        result[i] = (p0[i] + p1[i]) & mask;
    }
    return result;
}

int run_check(const GraphArgs &args) {
    RunRecord run0;
    RunRecord run1;
    full_record::MappedRecord graph0;
    full_record::MappedRecord graph1;
    if (!read_run_record(args.p0_output, run0) ||
        !read_run_record(args.p1_output, run1) ||
        !same_run_headers(run0.header, run1.header, args) ||
        !graph0.open(args.p0_graph_record) ||
        !graph1.open(args.p1_graph_record) ||
        !graph_record_matches_args(args, graph0, 0) ||
        !graph_record_matches_args(args, graph1, 1) ||
        !full_record::public_headers_match(graph0.header(), graph1.header()) ||
        run0.header.bundle_id != graph0.header().bundle_id ||
        run1.header.bundle_id != graph1.header().bundle_id ||
        run0.traces != run1.traces) {
        std::fprintf(stderr, "[full-graph-check] record parsing/binding failed\n");
        return 1;
    }
    Digest expected_layer{};
    Digest expected_plan{};
    if (!derive_run_identities(args, expected_layer, expected_plan) ||
        run0.header.layer_identity != expected_layer ||
        run0.header.plan_digest != expected_plan) {
        std::fprintf(stderr, "[full-graph-check] identity validation failed\n");
        return 1;
    }

    std::array<Digest, kTraceCount> expected{};
    std::array<std::vector<T>, contract::kLinearCount> linear_inputs;
    std::array<std::vector<T>, contract::kLinearCount> linear_outputs;
    bool source_bindings_ok = true;
    const std::filesystem::path root(args.record_set_root);
    for (size_t i = 0; source_bindings_ok && i < contract::kLinearCount; ++i) {
        MaskState p0;
        MaskState p1;
        LinearMappedRecord r0;
        LinearMappedRecord r1;
        GraphArgs a0 = args;
        GraphArgs a1 = args;
        a0.party = 0;
        a1.party = 1;
        source_bindings_ok =
            load_local_linear(a0, graph0, i, r0, p0) &&
            load_local_linear(a1, graph1, i, r1, p1) &&
            ringlpn_graph::mask_state_headers_match(p0.header, p1.header);
        if (!source_bindings_ok) break;
        linear_inputs[i] = add_values(p0.input_mask_share,
                                      p1.input_mask_share, kFullBw);
        linear_outputs[i] = add_values(p0.output_mask_share,
                                       p1.output_mask_share, kFullBw);
        source_bindings_ok =
            !linear_inputs[i].empty() && !linear_outputs[i].empty() &&
            trace_values(linear_inputs[i], expected[kTraceLinearInput + i]) &&
            trace_values(linear_outputs[i], expected[kTraceLinearOutput + i]);
    }

    std::array<std::vector<T>, contract::kTruncationCount> trunc_masks;
    bool truncation_ok = source_bindings_ok;
    for (size_t i = 0; truncation_ok && i < contract::kTruncationCount; ++i) {
        const size_t words = static_cast<size_t>(
            graph0.trunc_binding(i).words);
        trunc_masks[i] = reconstruct_mask(graph0.trunc_share(i),
                                          graph1.trunc_share(i),
                                          words, kTruncatedBw);
        truncation_ok = trace_values(
            trunc_masks[i], expected[kTraceTruncOutput + i]);
    }

    std::array<std::vector<T>, contract::kStockKeyCount> stock_masks;
    bool stock_ok = truncation_ok;
    for (size_t i = 0; stock_ok && i < contract::kStockKeyCount; ++i) {
        const size_t words = static_cast<size_t>(
            graph0.stock_binding(i).output_mask_words);
        stock_masks[i] = reconstruct_mask(
            graph0.stock_output_mask_share(i),
            graph1.stock_output_mask_share(i), words,
            contract::kStockKeySpecs[i].output_bw);
        stock_ok = trace_values(stock_masks[i],
                                expected[kTraceStockOutput + i]);
    }

    std::array<std::vector<T>, contract::kResidualCount> residuals;
    const auto resolve_mask = [&](contract::ValueSource source)
        -> const std::vector<T> * {
        if (source.kind == contract::ValueSourceKind::Truncation &&
            source.index < trunc_masks.size()) {
            return &trunc_masks[source.index];
        }
        if (source.kind == contract::ValueSourceKind::Stock &&
            source.index < stock_masks.size()) {
            return &stock_masks[source.index];
        }
        if (source.kind == contract::ValueSourceKind::Residual &&
            source.index < residuals.size()) {
            return &residuals[source.index];
        }
        return nullptr;
    };
    bool residual_ok = stock_ok;
    for (size_t i = 0; residual_ok && i < contract::kResidualCount; ++i) {
        const contract::ResidualSpec &spec = contract::kResidualSpecs[i];
        const std::vector<T> *left = resolve_mask(spec.main);
        const std::vector<T> *right = resolve_mask(spec.shortcut);
        residual_ok = left != nullptr && right != nullptr &&
                      left->size() == spec.words &&
                      right->size() == spec.words;
        if (residual_ok) {
            residuals[i] = add_values(*left, *right, kTruncatedBw);
            residual_ok = !residuals[i].empty() &&
                trace_values(residuals[i],
                             expected[kTraceResidualOutput + i]);
        }
    }

    for (size_t i = 0; stock_ok && residual_ok &&
                       i < contract::kStockKeyCount; ++i) {
        const std::vector<T> *input =
            resolve_mask(contract::kStockInputSources[i]);
        stock_ok = input != nullptr &&
                   input->size() == contract::kStockKeySpecs[i].input_words &&
                   trace_values(*input, expected[kTraceStockInput + i]);
    }

    bool state_links_ok = stock_ok && residual_ok;
    for (size_t i = 0; state_links_ok && i < contract::kRemaskCount; ++i) {
        const contract::RemaskSpec &spec = contract::kRemaskSpecs[i];
        const size_t source_index = contract::stock_index(spec.source);
        const std::vector<T> *source =
            source_index < stock_masks.size()
                ? &stock_masks[source_index]
                : nullptr;
        const T *delta0 = graph0.remask_delta(i);
        const T *delta1 = graph1.remask_delta(i);
        state_links_ok = source != nullptr &&
                         source->size() == spec.words &&
                         linear_inputs[spec.target_linear_index].size() ==
                             spec.words &&
                         std::equal(delta0, delta0 + spec.words, delta1);
        for (size_t j = 0; state_links_ok && j < spec.words; ++j) {
            state_links_ok =
                (((*source)[j] + delta0[j]) & kFullMask) ==
                linear_inputs[spec.target_linear_index][j];
        }
    }

    std::vector<T> pooled = global_pool_host(stock_masks[17].data(),
                                             stock_masks[17].size());
    bool pool_ok = state_links_ok && !pooled.empty() &&
                   trace_values(pooled, expected[kTraceGlobalPoolOutput]);
    std::vector<T> terminal(contract::kClassifierOutputWords, 0);
    bool terminal_ok = pool_ok &&
        std::equal(graph0.terminal_mask(),
                   graph0.terminal_mask() + contract::kClassifierOutputWords,
                   graph1.terminal_mask()) &&
        std::equal(graph0.terminal_mask(),
                   graph0.terminal_mask() + contract::kClassifierOutputWords,
                   linear_outputs[20].begin()) &&
        trace_values(terminal, expected[kTraceTerminalOutput]);

    uint64_t expected_truncations = 0;
    for (const contract::TruncationSpec &spec : contract::kTruncationSpecs) {
        add_u64(expected_truncations, spec.words);
    }
    const bool counters_ok =
        run0.header.totals.truncations == expected_truncations &&
        run1.header.totals.truncations == expected_truncations &&
        run0.header.totals.handoffs == expected_truncations &&
        run1.header.totals.handoffs == expected_truncations &&
        run0.header.totals.party_channel_bytes_sent > 0 &&
        run1.header.totals.party_channel_bytes_sent > 0 &&
        run0.header.totals.stock_bytes_sent > 0 &&
        run1.header.totals.stock_bytes_sent > 0 &&
        run0.header.totals.raw_key_bytes == graph0.header().raw_key_bytes &&
        run1.header.totals.raw_key_bytes == graph1.header().raw_key_bytes;

    bool trace_ok = terminal_ok;
    for (size_t i = 0; trace_ok && i < kTraceCount; ++i) {
        trace_ok = run0.traces[i] == expected[i];
    }
    const bool all_ok = source_bindings_ok && truncation_ok && stock_ok &&
                        residual_ok && state_links_ok && pool_ok && terminal_ok &&
                        counters_ok && trace_ok;
    if (args.csv_header) {
        std::printf(
            "source_bindings,linear_inputs,linear_outputs,"
            "secure_stochastic_truncations,stock_key_stream,residual_merges,"
            "global_average_pool,state_links,terminal_output,trace_contract,"
            "counter_contract,stock_nonlinear_key_source,scope,status\n");
    }
    std::printf(
        "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
        "stock_trusted_dealer_test_only,known-zero-full-resnet18,%s\n",
        source_bindings_ok ? "pass" : "FAIL",
        trace_ok ? "pass" : "FAIL",
        trace_ok ? "pass" : "FAIL",
        truncation_ok ? "pass" : "FAIL",
        stock_ok ? "pass" : "FAIL",
        residual_ok ? "pass" : "FAIL",
        pool_ok ? "pass" : "FAIL",
        state_links_ok ? "pass" : "FAIL",
        terminal_ok ? "pass" : "FAIL",
        trace_ok ? "pass" : "FAIL",
        counters_ok ? "pass" : "FAIL",
        all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    GraphArgs args;
    if (!parse_args(argc, argv, args) ||
        !manifest_artifact_paths_match(args)) {
        std::fprintf(stderr, "invalid full-graph arguments\n");
        return 2;
    }
    if (args.check) return run_check(args);
    if (args.csv_header) {
        std::printf(
            "party,total_us,linear_us,secure_truncate_us,stock_nonlinear_us,"
            "global_pool_us,terminal_us,total_protocol_bytes_sent,"
            "party_channel_bytes_sent,stock_bytes_sent,stock_bytes_received,"
            "truncations,handoffs,dabits,edabits,logical_opened_bits,triples,"
            "stock_raw_key_bytes,status\n");
    }
    return run_party(args);
}
