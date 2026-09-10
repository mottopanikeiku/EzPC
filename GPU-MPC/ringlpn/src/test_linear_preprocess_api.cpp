// Focused public-surface contract for libringlpn_linear. This translation unit
// intentionally includes no backend or CUDA/Orca header.
#include <ringlpn/linear_preprocess.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <openssl/evp.h>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kDigestBytes = 32;
constexpr size_t kFcRecordHeaderBytes = 176;
constexpr size_t kStateHeaderBytes = 160;
constexpr size_t kStateInputWordsOffset = 40;
constexpr size_t kStateRecordDigestOffset = 104;

struct MaterialPaths {
    std::string fc_p0_record;
    std::string fc_p0_state;
    std::string fc_p1_record;
    std::string fc_p1_state;
    std::string conv_p0_record;
    std::string conv_p0_state;
    std::string conv_p1_record;
    std::string conv_p1_state;
};

struct FixtureInfo {
    ringlpn_linear::LayerExpectation expected;
    ringlpn_linear::RecordMetadata metadata;
    ringlpn_linear::Digest layer_identity{};
};

class TempRoot {
  public:
    TempRoot() {
        char path[] = "/tmp/ringlpn-linear-api-XXXXXX";
        char *created = ::mkdtemp(path);
        if (created != nullptr) root_ = created;
    }
    ~TempRoot() {
        if (!root_.empty()) {
            std::error_code error;
            std::filesystem::remove_all(root_, error);
        }
    }
    TempRoot(const TempRoot &) = delete;
    TempRoot &operator=(const TempRoot &) = delete;

    bool valid() const { return !root_.empty(); }
    std::string path(const std::string &name) const {
        return root_ + "/" + name;
    }

  private:
    std::string root_;
};

bool parse_paths(int argc, char **argv, MaterialPaths &paths) {
    if (argc == 1) return true;
    if (argc != 17) return false;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--fc-p0-record") paths.fc_p0_record = value;
        else if (key == "--fc-p0-state") paths.fc_p0_state = value;
        else if (key == "--fc-p1-record") paths.fc_p1_record = value;
        else if (key == "--fc-p1-state") paths.fc_p1_state = value;
        else if (key == "--conv-p0-record") paths.conv_p0_record = value;
        else if (key == "--conv-p0-state") paths.conv_p0_state = value;
        else if (key == "--conv-p1-record") paths.conv_p1_record = value;
        else if (key == "--conv-p1-state") paths.conv_p1_state = value;
        else return false;
    }
    return !paths.fc_p0_record.empty() && !paths.fc_p0_state.empty() &&
           !paths.fc_p1_record.empty() && !paths.fc_p1_state.empty() &&
           !paths.conv_p0_record.empty() && !paths.conv_p0_state.empty() &&
           !paths.conv_p1_record.empty() && !paths.conv_p1_state.empty();
}

bool read_file(const std::string &path, std::vector<uint8_t> &bytes) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return false;
    struct stat metadata {};
    bool ok = ::fstat(descriptor, &metadata) == 0 &&
              S_ISREG(metadata.st_mode) && metadata.st_size > 0 &&
              static_cast<uint64_t>(metadata.st_size) <= (uint64_t{1} << 30);
    bytes.assign(ok ? static_cast<size_t>(metadata.st_size) : 0, 0);
    size_t cursor = 0;
    while (ok && cursor < bytes.size()) {
        const ssize_t count =
            ::read(descriptor, bytes.data() + cursor, bytes.size() - cursor);
        if (count > 0) {
            cursor += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            ok = false;
        }
    }
    if (::close(descriptor) != 0) ok = false;
    if (!ok) bytes.clear();
    return ok;
}

bool write_private_file(const std::string &path,
                        const std::vector<uint8_t> &bytes) {
    const int descriptor =
        ::open(path.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor < 0) return false;
    bool ok = true;
    size_t cursor = 0;
    while (cursor < bytes.size()) {
        const ssize_t count =
            ::write(descriptor, bytes.data() + cursor, bytes.size() - cursor);
        if (count > 0) {
            cursor += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            ok = false;
            break;
        }
    }
    if (::close(descriptor) != 0) ok = false;
    if (!ok) ::unlink(path.c_str());
    return ok;
}

uint32_t get_u32(const std::vector<uint8_t> &bytes, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint32_t>(bytes[offset + i]) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const std::vector<uint8_t> &bytes, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
    }
    return value;
}

void put_u64(std::vector<uint8_t> &bytes, size_t offset, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

bool refresh_internal_digest(std::vector<uint8_t> &bytes) {
    if (bytes.size() < kDigestBytes) return false;
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    if (context == nullptr) return false;
    ringlpn_linear::Digest digest{};
    unsigned int written = 0;
    const size_t authenticated = bytes.size() - kDigestBytes;
    const bool ok =
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, bytes.data(), authenticated) == 1 &&
        EVP_DigestFinal_ex(context, digest.data(), &written) == 1 &&
        written == digest.size();
    EVP_MD_CTX_free(context);
    if (!ok) return false;
    std::copy(digest.begin(), digest.end(), bytes.begin() + authenticated);
    return true;
}

bool material_is_empty(
    const ringlpn_linear::OwnedLayerMaterial &material) {
    return material.empty() && material.layer_ordinal() == 0 &&
           material.input_mask_share().data == nullptr &&
           material.input_mask_share().size == 0 &&
           material.weight_mask_share().data == nullptr &&
           material.weight_mask_share().size == 0 &&
           material.output_correction_share().data == nullptr &&
           material.output_correction_share().size == 0 &&
           material.output_mask_share().data == nullptr &&
           material.output_mask_share().size == 0;
}

template <typename Preprocessor>
bool load_fixture(const std::string &record_path,
                  const std::string &state_path, int party,
                  FixtureInfo &info) {
    ringlpn_linear::OwnedRecord record;
    if (Preprocessor::open_record(record_path, record) !=
            ringlpn_linear::Status::Ok ||
        record.metadata().party != party) {
        return false;
    }

    ringlpn_linear::LayerExpectation expected;
    expected.plan = record.plan();
    expected.party = party;
    expected.sid = record.metadata().sid;
    expected.layer_ordinal = 1;
    expected.invocation_id = record.metadata().invocation_id;
    expected.record_digest = record.metadata().digest;
    expected.require_invocation = true;

    ringlpn_linear::OwnedLayerMaterial material;
    if (Preprocessor::open_and_validate_layer_material(
            record_path, state_path, expected, material) !=
        ringlpn_linear::Status::Ok) {
        return false;
    }
    expected.state_digest = material.state_digest();
    expected.require_digests = true;

    const auto record_input = record.input_mask();
    const auto material_input = material.input_mask_share();
    const bool identity_nonzero =
        std::any_of(material.layer_identity().begin(),
                    material.layer_identity().end(),
                    [](uint8_t byte) { return byte != 0; });
    bool ok =
        material.plan().kind == expected.plan.kind &&
        material.record_metadata().digest == expected.record_digest &&
        material.layer_ordinal() == expected.layer_ordinal &&
        identity_nonzero && material_input.size == record_input.size &&
        std::equal(material_input.begin(), material_input.end(),
                   record_input.begin()) &&
        material.weight_mask_share().size ==
            static_cast<size_t>(expected.plan.weight_words) &&
        material.output_correction_share().size ==
            static_cast<size_t>(expected.plan.output_words) &&
        material.output_mask_share().size ==
            static_cast<size_t>(expected.plan.output_words);

    ringlpn_linear::OwnedLayerMaterial moved(std::move(material));
    ok = ok && material_is_empty(material) && !moved.empty();
    ringlpn_linear::OwnedLayerMaterial assigned;
    assigned = std::move(moved);
    ok = ok && material_is_empty(moved) && !assigned.empty();
    assigned.reset();
    ok = ok && material_is_empty(assigned);

    if (ok && Preprocessor::open_and_validate_layer_material(
                  record_path, state_path, expected, assigned) !=
                  ringlpn_linear::Status::Ok) {
        ok = false;
    }
    if (ok) {
        info.expected = expected;
        info.metadata = record.metadata();
        info.layer_identity = assigned.layer_identity();
    }
    assigned.reset();
    record.reset();
    return ok;
}

template <typename Preprocessor>
bool expect_rejection(
    const char *label, const std::string &valid_record,
    const std::string &valid_state,
    const ringlpn_linear::LayerExpectation &valid_expected,
    const std::string &candidate_record,
    const std::string &candidate_state,
    const ringlpn_linear::LayerExpectation &candidate_expected,
    ringlpn_linear::Status wanted) {
    ringlpn_linear::OwnedLayerMaterial material;
    if (Preprocessor::open_and_validate_layer_material(
            valid_record, valid_state, valid_expected, material) !=
        ringlpn_linear::Status::Ok) {
        std::fprintf(stderr, "[linear-library-api] %s setup failed\n", label);
        return false;
    }
    const ringlpn_linear::Status actual =
        Preprocessor::open_and_validate_layer_material(
            candidate_record, candidate_state, candidate_expected, material);
    if (actual != wanted || !material_is_empty(material)) {
        std::fprintf(stderr,
                     "[linear-library-api] %s got %s, expected %s, "
                     "empty=%d\n",
                     label, ringlpn_linear::status_name(actual),
                     ringlpn_linear::status_name(wanted),
                     material_is_empty(material) ? 1 : 0);
        return false;
    }
    return true;
}

bool expect_fifo_rejection(
    const MaterialPaths &paths, const FixtureInfo &fixture,
    const std::string &fifo, bool as_record) {
    const pid_t child = ::fork();
    if (child < 0) return false;
    if (child == 0) {
        // Bound the regression: a blocking open must fail, not hang the suite.
        ::alarm(5);
        const bool rejected =
            expect_rejection<ringlpn_linear::FcPreprocessor>(
                as_record ? "record-fifo" : "state-fifo",
                paths.fc_p0_record, paths.fc_p0_state, fixture.expected,
                as_record ? fifo : paths.fc_p0_record,
                as_record ? paths.fc_p0_state : fifo, fixture.expected,
                ringlpn_linear::Status::IoError);
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    pid_t waited;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

ringlpn_linear::LayerExpectation without_digests(
    ringlpn_linear::LayerExpectation expected) {
    expected.require_digests = false;
    return expected;
}

bool validate_fixture_pairs(const MaterialPaths &paths, FixtureInfo &fc_p0,
                            FixtureInfo &fc_p1, FixtureInfo &conv_p0,
                            FixtureInfo &conv_p1) {
    if (!load_fixture<ringlpn_linear::FcPreprocessor>(
            paths.fc_p0_record, paths.fc_p0_state, 0, fc_p0) ||
        !load_fixture<ringlpn_linear::FcPreprocessor>(
            paths.fc_p1_record, paths.fc_p1_state, 1, fc_p1) ||
        !load_fixture<ringlpn_linear::Conv2dPreprocessor>(
            paths.conv_p0_record, paths.conv_p0_state, 0, conv_p0) ||
        !load_fixture<ringlpn_linear::Conv2dPreprocessor>(
            paths.conv_p1_record, paths.conv_p1_state, 1, conv_p1)) {
        return false;
    }
    return fc_p0.metadata.sid == fc_p1.metadata.sid &&
           fc_p0.metadata.invocation_id == fc_p1.metadata.invocation_id &&
           fc_p0.metadata.ledger_digest == fc_p1.metadata.ledger_digest &&
           fc_p0.metadata.emp_silent_bridge_digest ==
               fc_p1.metadata.emp_silent_bridge_digest &&
           fc_p0.layer_identity == fc_p1.layer_identity &&
           conv_p0.metadata.sid == conv_p1.metadata.sid &&
           conv_p0.metadata.invocation_id ==
               conv_p1.metadata.invocation_id &&
           conv_p0.metadata.ledger_digest ==
               conv_p1.metadata.ledger_digest &&
           conv_p0.metadata.emp_silent_bridge_digest ==
               conv_p1.metadata.emp_silent_bridge_digest &&
           conv_p0.layer_identity == conv_p1.layer_identity;
}

bool run_rejection_controls(const MaterialPaths &paths,
                            const FixtureInfo &fc_p0,
                            const FixtureInfo &conv_p0) {
    TempRoot temporary;
    if (!temporary.valid()) return false;

    bool ok = true;
    auto changed = conv_p0.expected;
    ok = expect_rejection<ringlpn_linear::Conv2dPreprocessor>(
             "wrong-kind", paths.conv_p0_record, paths.conv_p0_state,
             conv_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;

    changed = fc_p0.expected;
    ++changed.plan.fc.rows;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-shape", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    ++changed.plan.protocol.bw;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-bw", paths.fc_p0_record, paths.fc_p0_state, fc_p0.expected,
             paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    changed.party = 1;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-party", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    changed.sid = changed.sid == std::numeric_limits<uint64_t>::max()
                      ? changed.sid - 1
                      : changed.sid + 1;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-sid", paths.fc_p0_record, paths.fc_p0_state, fc_p0.expected,
             paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    changed.invocation_id[0] ^= 1;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-invocation", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    ++changed.layer_ordinal;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-ordinal", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    changed.record_digest[0] ^= 1;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-record-digest", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;
    changed = fc_p0.expected;
    changed.state_digest[0] ^= 1;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "wrong-state-digest", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;

    changed = without_digests(fc_p0.expected);
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "swapped-state", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, paths.fc_p1_state, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;

    std::vector<uint8_t> state_bytes;
    std::vector<uint8_t> record_bytes;
    if (!read_file(paths.fc_p0_state, state_bytes) ||
        !read_file(paths.fc_p0_record, record_bytes) ||
        state_bytes.size() < kStateHeaderBytes + kDigestBytes ||
        record_bytes.size() < kFcRecordHeaderBytes + kDigestBytes) {
        return false;
    }

    std::vector<uint8_t> mutated = state_bytes;
    mutated[kStateRecordDigestOffset] ^= 1;
    const std::string binding_path =
        temporary.path("state-record-binding.bin");
    if (!refresh_internal_digest(mutated) ||
        !write_private_file(binding_path, mutated)) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "state-record-binding", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, binding_path, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;

    mutated = state_bytes;
    const uint32_t state_bw = get_u32(mutated, 32);
    if (state_bw == 0 || state_bw >= 64 ||
        mutated.size() < kStateHeaderBytes + sizeof(uint64_t) + kDigestBytes) {
        return false;
    }
    const uint64_t state_mask = (uint64_t{1} << state_bw) - 1;
    const uint64_t old_input = get_u64(mutated, kStateHeaderBytes);
    put_u64(mutated, kStateHeaderBytes, (old_input + 1) & state_mask);
    const std::string input_path = temporary.path("state-input.bin");
    if (!refresh_internal_digest(mutated) ||
        !write_private_file(input_path, mutated)) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "state-input-binding", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, input_path, changed,
             ringlpn_linear::Status::RecordMismatch) &&
         ok;

    mutated = state_bytes;
    mutated.pop_back();
    const std::string truncated_path = temporary.path("state-truncated.bin");
    if (!write_private_file(truncated_path, mutated)) return false;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "truncated-state", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, truncated_path, changed,
             ringlpn_linear::Status::CorruptRecord) &&
         ok;

    mutated = state_bytes;
    mutated.back() ^= 1;
    const std::string corrupt_path = temporary.path("state-corrupt.bin");
    if (!write_private_file(corrupt_path, mutated)) return false;
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "corrupt-state", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, corrupt_path, changed,
             ringlpn_linear::Status::CorruptRecord) &&
         ok;

    mutated = state_bytes;
    const uint64_t input_words =
        get_u64(mutated, kStateInputWordsOffset);
    const size_t state_output_offset =
        kStateHeaderBytes + static_cast<size_t>(input_words) * sizeof(uint64_t);
    if (state_output_offset + sizeof(uint64_t) + kDigestBytes >
        mutated.size()) {
        return false;
    }
    put_u64(mutated, state_output_offset, uint64_t{1} << state_bw);
    const std::string noncanonical_state =
        temporary.path("state-noncanonical.bin");
    if (!refresh_internal_digest(mutated) ||
        !write_private_file(noncanonical_state, mutated)) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "noncanonical-state", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, noncanonical_state, changed,
             ringlpn_linear::Status::CorruptRecord) &&
         ok;

    mutated = record_bytes;
    const uint32_t record_bw = get_u32(mutated, 28);
    if (record_bw == 0 || record_bw >= 64 ||
        kFcRecordHeaderBytes + sizeof(uint64_t) + kDigestBytes >
            mutated.size()) {
        return false;
    }
    put_u64(mutated, kFcRecordHeaderBytes, uint64_t{1} << record_bw);
    const std::string noncanonical_record =
        temporary.path("record-noncanonical.bin");
    if (!refresh_internal_digest(mutated) ||
        !write_private_file(noncanonical_record, mutated)) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "noncanonical-record", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, noncanonical_record, paths.fc_p0_state, changed,
             ringlpn_linear::Status::CorruptRecord) &&
         ok;

    const std::string permissive = temporary.path("state-permissive.bin");
    if (!write_private_file(permissive, state_bytes) ||
        ::chmod(permissive.c_str(), 0644) != 0) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "permissive-state", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, paths.fc_p0_record, permissive, changed,
             ringlpn_linear::Status::IoError) &&
         ok;

    const std::string victim = temporary.path("state-victim.bin");
    const std::string incoming = temporary.path("state-incoming.bin");
    if (!write_private_file(victim, state_bytes) ||
        ::symlink(paths.fc_p0_state.c_str(), incoming.c_str()) != 0 ||
        ::rename(incoming.c_str(), victim.c_str()) != 0) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "state-symlink-replacement", paths.fc_p0_record,
             paths.fc_p0_state, fc_p0.expected, paths.fc_p0_record, victim,
             changed, ringlpn_linear::Status::IoError) &&
         ok;

    const std::string record_link = temporary.path("record-link.bin");
    if (::symlink(paths.fc_p0_record.c_str(), record_link.c_str()) != 0) {
        return false;
    }
    ok = expect_rejection<ringlpn_linear::FcPreprocessor>(
             "record-symlink", paths.fc_p0_record, paths.fc_p0_state,
             fc_p0.expected, record_link, paths.fc_p0_state, changed,
             ringlpn_linear::Status::IoError) &&
         ok;

    const std::string fifo = temporary.path("private-fifo");
    if (::mkfifo(fifo.c_str(), 0600) != 0) return false;
    ok = expect_fifo_rejection(paths, fc_p0, fifo, true) && ok;
    ok = expect_fifo_rejection(paths, fc_p0, fifo, false) && ok;

    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    MaterialPaths paths;
    if (!parse_paths(argc, argv, paths)) {
        std::fprintf(
            stderr,
            "Usage: %s [--fc-p0-record F --fc-p0-state F "
            "--fc-p1-record F --fc-p1-state F --conv-p0-record F "
            "--conv-p0-state F --conv-p1-record F --conv-p1-state F]\n",
            argv[0]);
        return 2;
    }

    ringlpn_linear::ProtocolParameters protocol;
    ringlpn_linear::LinearPlan fc_plan;
    ringlpn_linear::LinearPlan conv_plan;
    const ringlpn_linear::FcShape fc_shape{2, 3, 2};
    const ringlpn_linear::Conv2dShape conv_shape{
        1, 4, 4, 1, 3, 3, 2, 1, 1};
    bool ok = ringlpn_linear::FcPreprocessor::plan(
                  fc_shape, protocol, fc_plan) == ringlpn_linear::Status::Ok &&
              ringlpn_linear::Conv2dPreprocessor::plan(
                  conv_shape, protocol, conv_plan) ==
                  ringlpn_linear::Status::Ok &&
              fc_plan.input_words == 6 && fc_plan.weight_words == 6 &&
              fc_plan.output_words == 4 && conv_plan.input_words == 16 &&
              conv_plan.weight_words == 18 && conv_plan.output_words == 32;

    // At q64/bw29, the exact no-wrap bound admits inner=3 but not 4.
    protocol.qbits = 64;
    protocol.bw = 29;
    ok = (ringlpn_linear::FcPreprocessor::plan(
              {1, 3, 1}, protocol, fc_plan) == ringlpn_linear::Status::Ok &&
          ringlpn_linear::FcPreprocessor::plan(
              {1, 4, 1}, protocol, fc_plan) ==
              ringlpn_linear::Status::InvalidPlan &&
          ringlpn_linear::Conv2dPreprocessor::plan(
              {1, 1, 1, 3, 1, 1, 1, 0, 1}, protocol, conv_plan) ==
              ringlpn_linear::Status::Ok &&
          ringlpn_linear::Conv2dPreprocessor::plan(
              {1, 1, 1, 4, 1, 1, 1, 0, 1}, protocol, conv_plan) ==
              ringlpn_linear::Status::InvalidPlan) && ok;

    // Tiny tensors can still overflow stock int-coordinate arithmetic.
    protocol.qbits = 128;
    protocol.bw = 32;
    ok = (ringlpn_linear::Conv2dPreprocessor::plan(
              {1, 1, 1, 1, 1, 1, 1, std::numeric_limits<int>::max(),
               std::numeric_limits<int>::max()},
              protocol, conv_plan) ==
          ringlpn_linear::Status::InvalidPlan) && ok;

    if (!paths.fc_p0_record.empty()) {
        FixtureInfo fc_p0;
        FixtureInfo fc_p1;
        FixtureInfo conv_p0;
        FixtureInfo conv_p1;
        ok = validate_fixture_pairs(paths, fc_p0, fc_p1, conv_p0, conv_p1) &&
             run_rejection_controls(paths, fc_p0, conv_p0) && ok;
    }

    if (!ok) {
        std::fprintf(stderr, "[linear-library-api] contract failed\n");
        return 1;
    }
    std::puts("linear-library-api,ok");
    return 0;
}
