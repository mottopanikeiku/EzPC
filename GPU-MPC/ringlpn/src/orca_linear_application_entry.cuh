#pragma once

#if defined(ORCA_RINGLPN_LINEAR_INTEGRATION) && \
    ORCA_RINGLPN_LINEAR_INTEGRATION
#include "ringlpn/src/orca_terminal_linear_backend.cuh"
#include "ringlpn/src/private_file.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

// Role 2 installs its backend before executing any layer. Avoid constructing
// the unused cleartext backend so this source-native binary needs no separate
// libsytorch archive; graph generation itself does not execute callbacks.
template <>
Backend<InfType> *defaultBackend<InfType>() {
    return nullptr;
}

namespace ringlpn_application_detail {

using ringlpn_linear::Digest;
using ringlpn_linear::InvocationId;
using ringlpn_linear::LayerExpectation;
using ringlpn_linear::LinearKind;
using ringlpn_linear::LinearPlan;
using ringlpn_linear::OwnedLayerMaterial;
using ringlpn_linear::ProtocolParameters;
using ringlpn_linear::Status;

struct Role2Options {
    std::map<std::string, std::string> values;
    bool fc = false;
};

bool parse_u64_decimal(const std::string &text, uint64_t &value) {
    if (text.empty()) return false;
    uint64_t parsed = 0;
    for (char character : text) {
        if (character < '0' || character > '9') return false;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    value = parsed;
    return true;
}

bool parse_int_value(const std::string &text, int minimum, int &value) {
    uint64_t parsed = 0;
    if (!parse_u64_decimal(text, parsed) ||
        parsed > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        parsed < static_cast<uint64_t>(minimum)) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

template <size_t Size>
bool parse_hex(const std::string &text, std::array<uint8_t, Size> &value) {
    if (text.size() != 2 * Size) return false;
    auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') {
            return character - 'a' + 10;
        }
        return -1;
    };
    for (size_t i = 0; i < Size; ++i) {
        const int high = nibble(text[2 * i]);
        const int low = nibble(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        value[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

bool parse_role2_options(int argc, char **argv,
                         const std::string &model_name,
                         Role2Options &options) {
    options.fc = model_name == "RingLPN-FC";
    if (!options.fc && model_name != "RingLPN-Conv2D") return false;
    const std::set<std::string> common = {
        "--record",          "--state",       "--input-share",
        "--weight-share",    "--bias",        "--output",
        "--sid",             "--invocation-id",
        "--record-digest",   "--state-digest"};
    const std::set<std::string> fc_shape = {
        "--rows", "--inner", "--cols"};
    const std::set<std::string> conv_shape = {
        "--n",  "--h",  "--w",       "--ci",      "--fh",
        "--fw", "--co", "--padding", "--stride"};
    std::set<std::string> required = common;
    const auto &shape = options.fc ? fc_shape : conv_shape;
    required.insert(shape.begin(), shape.end());
    if (argc < 7 || (argc - 7) % 2 != 0) return false;
    for (int index = 7; index < argc; index += 2) {
        const std::string key = argv[index];
        if (required.count(key) != 1 ||
            !options.values.emplace(key, argv[index + 1]).second) {
            return false;
        }
    }
    return options.values.size() == required.size();
}

class ScopedBytes {
  public:
    ~ScopedBytes() {
        volatile uint8_t *data = bytes.data();
        for (size_t i = 0; i < bytes.size(); ++i) data[i] = 0;
    }
    std::vector<uint8_t> bytes;
};

void scrub_words(std::vector<uint64_t> &words) {
    volatile uint8_t *bytes =
        reinterpret_cast<volatile uint8_t *>(words.data());
    for (size_t i = 0; i < words.size() * sizeof(uint64_t); ++i) {
        bytes[i] = 0;
    }
    words.clear();
}

class ScopedWords {
  public:
    ~ScopedWords() { scrub_words(value); }
    std::vector<uint64_t> value;
};

bool read_word_file(const std::string &path, size_t expected_words,
                    bool require_private, std::vector<uint64_t> &words) {
    scrub_words(words);
    if (path.empty() ||
        expected_words >
            std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
        return false;
    }
    const size_t expected_bytes = expected_words * sizeof(uint64_t);
    const int descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) return false;
    struct stat before {};
    bool ok = ::fstat(descriptor, &before) == 0 &&
              S_ISREG(before.st_mode) && before.st_uid == ::geteuid() &&
              before.st_nlink == 1 && before.st_size >= 0 &&
              static_cast<uint64_t>(before.st_size) == expected_bytes;
    const mode_t mode = before.st_mode & 07777;
    ok = ok && (mode == 0600 || (!require_private && mode == 0644));
    ScopedBytes private_bytes;
    private_bytes.bytes.assign(ok ? expected_bytes : 0, 0);
    size_t cursor = 0;
    while (ok && cursor < private_bytes.bytes.size()) {
        const ssize_t count =
            ::read(descriptor, private_bytes.bytes.data() + cursor,
                   private_bytes.bytes.size() - cursor);
        if (count > 0) {
            cursor += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            ok = false;
        }
    }
    uint8_t extra = 0;
    ssize_t trailing = -1;
    do {
        if (ok) trailing = ::read(descriptor, &extra, 1);
    } while (ok && trailing < 0 && errno == EINTR);
    struct stat after {};
    ok = ok && trailing == 0 && ::fstat(descriptor, &after) == 0 &&
         before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_size == after.st_size && before.st_uid == after.st_uid &&
         before.st_mode == after.st_mode && before.st_nlink == after.st_nlink &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
    if (::close(descriptor) != 0) ok = false;
    if (!ok) return false;

    words.assign(expected_words, 0);
    for (size_t i = 0; i < expected_words; ++i) {
        uint64_t value = 0;
        for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
            value |= static_cast<uint64_t>(
                         private_bytes.bytes[i * sizeof(uint64_t) + byte])
                     << (8 * byte);
        }
        if (value >= (uint64_t{1} << 32)) {
            scrub_words(words);
            return false;
        }
        words[i] = value;
    }
    return true;
}

template <typename T>
class TerminalConv2DLayer final : public Layer<T> {
  public:
    TerminalConv2DLayer(u64 ci, u64 co, u64 fh, u64 fw, u64 padding,
                        u64 stride)
        : Layer<T>("Conv2D"), filter(co, fh * fw * ci), bias(co), ci_(ci),
          co_(co), fh_(fh), fw_(fw), padding_(padding), stride_(stride) {
        this->doTruncationForward = true;
        this->useBias = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) override {
        always_assert(shapes.size() == 1);
        always_assert(shapes[0].size() == 4);
        always_assert(shapes[0][3] == ci_);
    }

    void _forward(Tensor<T> &input) override {
        auto output = this->activation.as_4d();
        this->backend->conv2D(fh_, fw_, padding_, stride_, ci_, co_,
                              input.as_4d(), filter, this->useBias, bias,
                              output, this->isFirst);
        this->activation.d_data = output.d_data;
    }

    TensorRef<T> getweights() override { return filter.ref(); }
    TensorRef<T> getbias() override { return bias.ref(); }

    std::vector<u64> get_output_dims(
        const std::vector<std::vector<u64>> &shapes) override {
        always_assert(shapes.size() == 1);
        always_assert(shapes[0].size() == 4);
        always_assert(shapes[0][3] == ci_);
        return {shapes[0][0],
                (shapes[0][1] + 2 * padding_ - fh_) / stride_ + 1,
                (shapes[0][2] + 2 * padding_ - fw_) / stride_ + 1, co_};
    }

    Tensor2D<T> filter;
    Tensor1D<T> bias;

  private:
    u64 ci_;
    u64 co_;
    u64 fh_;
    u64 fw_;
    u64 padding_;
    u64 stride_;
};

template <typename T>
class TerminalLinearModule final : public SytorchModule<T> {
  public:
    explicit TerminalLinearModule(const LinearPlan &plan) : plan_(plan) {
        if (plan.kind == LinearKind::Fc) {
            fc_ = new FC<T>(plan.fc.inner, plan.fc.cols, true);
        } else {
            conv_ = new TerminalConv2DLayer<T>(
                plan.conv.ci, plan.conv.co, plan.conv.fh, plan.conv.fw,
                plan.conv.padding, plan.conv.stride);
        }
    }

    ~TerminalLinearModule() {
        delete fc_;
        delete conv_;
    }

    Tensor<T> &_forward(Tensor<T> &input) override {
        return fc_ != nullptr ? fc_->forward(input) : conv_->forward(input);
    }

    bool load_values(Tensor<T> &input,
                     const std::vector<uint64_t> &input_share,
                     const std::vector<uint64_t> &weight_share,
                     const std::vector<uint64_t> &bias) {
        if (input.size() != input_share.size()) return false;
        std::copy(input_share.begin(), input_share.end(), input.data);
        TensorRef<T> weights =
            fc_ != nullptr ? fc_->getweights() : conv_->getweights();
        TensorRef<T> biases =
            fc_ != nullptr ? fc_->getbias() : conv_->getbias();
        if (weights.size != weight_share.size() ||
            biases.size != bias.size()) {
            return false;
        }
        std::copy(weight_share.begin(), weight_share.end(), weights.data);
        std::copy(bias.begin(), bias.end(), biases.data);
        return true;
    }

  private:
    LinearPlan plan_;
    FC<T> *fc_ = nullptr;
    TerminalConv2DLayer<T> *conv_ = nullptr;
};

bool publish_output(const std::string &path,
                    const Tensor<InfType> &output) {
    if (output.data == nullptr ||
        output.size() >
            std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
        return false;
    }
    std::vector<uint8_t> bytes(output.size() * sizeof(uint64_t), 0);
    for (size_t i = 0; i < output.size(); ++i) {
        const uint64_t value = static_cast<uint64_t>(output.data[i]);
        for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
            bytes[i * sizeof(uint64_t) + byte] =
                static_cast<uint8_t>(value >> (8 * byte));
        }
    }
    ringlpn_private_file::AtomicWriter writer;
    if (!writer.stage(path, bytes) || !writer.publish()) return false;
    writer.commit();
    return true;
}

void role2_usage(const char *program) {
    std::fprintf(
        stderr,
        "Usage: %s RingLPN-FC 32 0 2 PARTY PEER_IP "
        "--record F --state F --input-share F --weight-share F "
        "--bias F --output F --sid N --invocation-id 32hex "
        "--record-digest 64hex --state-digest 64hex "
        "--rows M --inner K --cols N\\n"
        "   or: %s RingLPN-Conv2D 32 0 2 PARTY PEER_IP "
        "--record F --state F --input-share F --weight-share F "
        "--bias F --output F --sid N --invocation-id 32hex "
        "--record-digest 64hex --state-digest 64hex "
        "--n N --h H --w W --ci CI --fh FH --fw FW --co CO "
        "--padding P --stride S\\n",
        program, program);
}

int run_role2(int argc, char **argv) {
    static_assert(std::is_same_v<InfType, uint64_t>,
                  "Ring-LPN role 2 requires the default uint64_t InfType");
    if (argc < 7 || std::strcmp(argv[4], "2") != 0) {
        role2_usage(argv[0]);
        return 2;
    }
    const std::string model_name = argv[1];
    int bw = 0;
    int scale = 0;
    int party = -1;
    if (!parse_int_value(argv[2], 1, bw) ||
        !parse_int_value(argv[3], 0, scale) ||
        !parse_int_value(argv[5], 0, party) ||
        (party != SERVER0 && party != SERVER1) || argv[6][0] == '\0') {
        role2_usage(argv[0]);
        return 2;
    }
    Role2Options options;
    if (!parse_role2_options(argc, argv, model_name, options)) {
        role2_usage(argv[0]);
        return 2;
    }

    ProtocolParameters protocol;
    protocol.qbits = 128;
    protocol.bw = 32;
    protocol.ole_n = 8192;
    protocol.ole_c = 2;
    protocol.ole_t = 8;
    protocol.regular_noise = true;
    protocol.channel = "local-loopback";
    protocol.ot_backend = "sci-iknp";

    bool local_valid = bw == 32 && scale == 0;
    LinearPlan plan;
    Status plan_status = Status::InvalidPlan;
    if (options.fc) {
        ringlpn_linear::FcShape shape;
        const bool shape_ok =
            parse_int_value(options.values.at("--rows"), 1, shape.rows) &&
            parse_int_value(options.values.at("--inner"), 1, shape.inner) &&
            parse_int_value(options.values.at("--cols"), 1, shape.cols);
        local_valid = local_valid && shape_ok;
        if (shape_ok) {
            plan_status =
                ringlpn_linear::FcPreprocessor::plan(shape, protocol, plan);
        }
    } else {
        ringlpn_linear::Conv2dShape shape;
        const bool shape_ok =
            parse_int_value(options.values.at("--n"), 1, shape.n) &&
            parse_int_value(options.values.at("--h"), 1, shape.h) &&
            parse_int_value(options.values.at("--w"), 1, shape.w) &&
            parse_int_value(options.values.at("--ci"), 1, shape.ci) &&
            parse_int_value(options.values.at("--fh"), 1, shape.fh) &&
            parse_int_value(options.values.at("--fw"), 1, shape.fw) &&
            parse_int_value(options.values.at("--co"), 1, shape.co) &&
            parse_int_value(options.values.at("--padding"), 0,
                            shape.padding) &&
            parse_int_value(options.values.at("--stride"), 1, shape.stride);
        local_valid = local_valid && shape_ok;
        if (shape_ok) {
            plan_status = ringlpn_linear::Conv2dPreprocessor::plan(
                shape, protocol, plan);
        }
    }
    local_valid = local_valid && plan_status == Status::Ok;

    LayerExpectation expected;
    expected.plan = plan;
    expected.party = party;
    expected.layer_ordinal = 1;
    expected.require_invocation = true;
    expected.require_digests = true;
    const bool sid_ok =
        parse_u64_decimal(options.values.at("--sid"), expected.sid) &&
        expected.sid != 0;
    const bool invocation_ok = parse_hex(
        options.values.at("--invocation-id"), expected.invocation_id);
    const bool record_digest_ok = parse_hex(
        options.values.at("--record-digest"), expected.record_digest);
    const bool state_digest_ok = parse_hex(
        options.values.at("--state-digest"), expected.state_digest);
    local_valid = local_valid && sid_ok && invocation_ok &&
                  record_digest_ok && state_digest_ok;

    const std::string &record_path = options.values.at("--record");
    const std::string &state_path = options.values.at("--state");
    const std::string &input_path = options.values.at("--input-share");
    const std::string &weight_path = options.values.at("--weight-share");
    const std::string &bias_path = options.values.at("--bias");
    const std::string &output_path = options.values.at("--output");
    const bool output_absent =
        ringlpn_private_file::AtomicWriter::destination_absent(output_path);

    ScopedWords private_input;
    ScopedWords private_weight;
    std::vector<uint64_t> &input_share = private_input.value;
    std::vector<uint64_t> &weight_share = private_weight.value;
    std::vector<uint64_t> bias;
    bool input_ok = false;
    bool weight_ok = false;
    bool bias_ok = false;
    if (plan_status == Status::Ok) {
        input_ok = read_word_file(
            input_path, static_cast<size_t>(plan.input_words), true,
            input_share);
        weight_ok = read_word_file(
            weight_path, static_cast<size_t>(plan.weight_words), true,
            weight_share);
        const size_t bias_words =
            plan.kind == LinearKind::Fc
                ? static_cast<size_t>(plan.fc.cols)
                : static_cast<size_t>(plan.conv.co);
        bias_ok = read_word_file(bias_path, bias_words, false, bias);
    }
    local_valid =
        local_valid && output_absent && input_ok && weight_ok && bias_ok;

    OwnedLayerMaterial material;
    Status material_status = Status::RecordMismatch;
    if (plan_status == Status::Ok) {
        material_status =
            plan.kind == LinearKind::Fc
                ? ringlpn_linear::FcPreprocessor::
                      open_and_validate_layer_material(
                          record_path, state_path, expected, material)
                : ringlpn_linear::Conv2dPreprocessor::
                      open_and_validate_layer_material(
                          record_path, state_path, expected, material);
    }
    local_valid = local_valid && material_status == Status::Ok;

    auto backend =
        std::make_unique<ringlpn_orca::TerminalLinearBackend<InfType>>(
            party, argv[6], bw, expected, std::move(material), local_valid);

    std::vector<u64> input_shape;
    if (plan.kind == LinearKind::Fc) {
        input_shape = {static_cast<u64>(plan.fc.rows),
                       static_cast<u64>(plan.fc.inner)};
    } else {
        input_shape = {static_cast<u64>(plan.conv.n),
                       static_cast<u64>(plan.conv.h),
                       static_cast<u64>(plan.conv.w),
                       static_cast<u64>(plan.conv.ci)};
    }
    Tensor<InfType> input(input_shape);
    auto model = std::make_unique<TerminalLinearModule<InfType>>(plan);
    model->init(0, input);
    model->zero();
    if (!model->load_values(input, input_share, weight_share, bias)) {
        throw std::runtime_error("role-2 tensor material length mismatch");
    }
    scrub_words(input_share);
    scrub_words(weight_share);
    model->setBackend(backend.get());
    model->optimize();
    input.d_data = reinterpret_cast<InfType *>(moveToGPU(
        reinterpret_cast<uint8_t *>(input.data),
        input.size() * sizeof(InfType), &backend->s));
    Tensor<InfType> &activation = model->forward(input);
    backend->output(activation);
    if (!publish_output(output_path, activation)) {
        throw std::runtime_error("atomic role-2 output publication failed");
    }
    backend->close();
    return 0;
}

}  // namespace ringlpn_application_detail

inline int ringlpn_role2_main(int argc, char **argv) noexcept {
    try {
        return ringlpn_application_detail::run_role2(argc, argv);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[ringlpn-orca] %s\n", error.what());
        return 1;
    }
}
#endif
