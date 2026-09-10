// Focused public-surface contract for libringlpn_linear. This translation unit
// intentionally includes no backend or CUDA/Orca header.
#include <ringlpn/linear_preprocess.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct RecordPaths {
    std::string fc_p0;
    std::string fc_p1;
    std::string conv_p0;
    std::string conv_p1;
};

bool parse_paths(int argc, char **argv, RecordPaths &paths) {
    if (argc == 1) return true;
    if (argc != 9) return false;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--fc-p0") paths.fc_p0 = value;
        else if (key == "--fc-p1") paths.fc_p1 = value;
        else if (key == "--conv-p0") paths.conv_p0 = value;
        else if (key == "--conv-p1") paths.conv_p1 = value;
        else return false;
    }
    return !paths.fc_p0.empty() && !paths.fc_p1.empty() &&
           !paths.conv_p0.empty() && !paths.conv_p1.empty();
}

template <typename Preprocessor>
bool reject_path_substitution(const std::string &record_path) {
    char directory_template[] = "/tmp/ringlpn-linear-api-XXXXXX";
    char *directory = ::mkdtemp(directory_template);
    if (directory == nullptr) return false;
    const std::string root(directory);
    const std::string direct = root + "/direct";
    const std::string victim = root + "/victim";
    const std::string incoming = root + "/incoming";
    const std::string permissive = root + "/permissive";

    bool ok = ::symlink(record_path.c_str(), direct.c_str()) == 0;
    ringlpn_linear::OwnedRecord opened;
    if (ok) {
        ok = Preprocessor::open_record(direct, opened) ==
             ringlpn_linear::Status::IoError;
    }
    const int victim_fd =
        ::open(victim.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (victim_fd < 0) ok = false;
    if (victim_fd >= 0 && ::close(victim_fd) != 0) ok = false;
    if (::symlink(record_path.c_str(), incoming.c_str()) != 0 ||
        ::rename(incoming.c_str(), victim.c_str()) != 0) {
        ok = false;
    } else {
        opened.reset();
        ok = ok && Preprocessor::open_record(victim, opened) ==
                       ringlpn_linear::Status::IoError;
    }
    const int permissive_fd =
        ::open(permissive.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (permissive_fd < 0) {
        ok = false;
    } else {
        const bool mode_changed = ::fchmod(permissive_fd, 0644) == 0;
        const bool closed = ::close(permissive_fd) == 0;
        if (!mode_changed || !closed) {
            ok = false;
        } else {
            opened.reset();
            ok = ok && Preprocessor::open_record(permissive, opened) ==
                           ringlpn_linear::Status::IoError;
        }
    }
    ::unlink(direct.c_str());
    ::unlink(incoming.c_str());
    ::unlink(victim.c_str());
    ::unlink(permissive.c_str());
    if (::rmdir(root.c_str()) != 0) ok = false;
    return ok;
}
template <typename Preprocessor>
bool validate_fixture_pair_sequentially(const std::string &party0_path,
                                        const std::string &party1_path) {
    ringlpn_linear::OwnedRecord record;
    if (Preprocessor::open_record(party0_path, record) !=
        ringlpn_linear::Status::Ok) {
        return false;
    }
    const ringlpn_linear::LinearPlan plan = record.plan();
    const ringlpn_linear::RecordMetadata party0 = record.metadata();
    ringlpn_linear::RecordExpectation expected;
    expected.plan = plan;
    expected.party = 0;
    expected.sid = party0.sid;
    expected.invocation_id = party0.invocation_id;
    expected.require_invocation = true;
    if (Preprocessor::validate_record(record, expected) !=
        ringlpn_linear::Status::Ok) {
        return false;
    }
    record.reset();

    if (Preprocessor::open_record(party1_path, record) !=
        ringlpn_linear::Status::Ok) {
        return false;
    }
    expected.party = 1;
    const bool valid =
        Preprocessor::validate_record(record, expected) ==
            ringlpn_linear::Status::Ok &&
        record.metadata().ledger_digest == party0.ledger_digest &&
        record.metadata().emp_silent_bridge_digest ==
            party0.emp_silent_bridge_digest;
    record.reset();
    return valid;
}


}  // namespace

int main(int argc, char **argv) {
    RecordPaths paths;
    if (!parse_paths(argc, argv, paths)) {
        std::fprintf(stderr,
                     "Usage: %s [--fc-p0 F --fc-p1 F --conv-p0 F "
                     "--conv-p1 F]\n",
                     argv[0]);
        return 2;
    }

    ringlpn_linear::ProtocolParameters protocol;
    ringlpn_linear::LinearPlan fc_plan;
    ringlpn_linear::LinearPlan conv_plan;
    const ringlpn_linear::FcShape fc_shape{2, 2, 2};
    const ringlpn_linear::Conv2dShape conv_shape{
        1, 4, 4, 1, 3, 3, 2, 1, 1};
    bool ok = ringlpn_linear::FcPreprocessor::plan(
                  fc_shape, protocol, fc_plan) == ringlpn_linear::Status::Ok &&
              ringlpn_linear::Conv2dPreprocessor::plan(
                  conv_shape, protocol, conv_plan) ==
                  ringlpn_linear::Status::Ok &&
              fc_plan.input_words == 4 && fc_plan.weight_words == 4 &&
              fc_plan.output_words == 4 && conv_plan.input_words == 16 &&
              conv_plan.weight_words == 18 && conv_plan.output_words == 32;

    if (!paths.fc_p0.empty()) {
        ok = ok &&
             validate_fixture_pair_sequentially<
                 ringlpn_linear::FcPreprocessor>(paths.fc_p0, paths.fc_p1) &&
             validate_fixture_pair_sequentially<
                 ringlpn_linear::Conv2dPreprocessor>(
                 paths.conv_p0, paths.conv_p1) &&
             reject_path_substitution<ringlpn_linear::FcPreprocessor>(
                 paths.fc_p0) &&
             reject_path_substitution<ringlpn_linear::Conv2dPreprocessor>(
                 paths.conv_p0);
    }

    if (!ok) {
        std::fprintf(stderr, "[linear-library-api] contract failed\n");
        return 1;
    }
    std::puts("linear-library-api,ok");
    return 0;
}
