#include "private_file.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

bool require(bool condition, const char *label) {
    if (!condition) std::fprintf(stderr, "private_file_control_fail=%s\n", label);
    return condition;
}

bool read_exact_private(const std::string &path,
                        const std::vector<uint8_t> &expected,
                        ino_t *inode = nullptr) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat metadata {};
    bool ok = ::fstat(fd, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
              metadata.st_uid == ::geteuid() && metadata.st_nlink == 1 &&
              (metadata.st_mode & 07777) == (S_IRUSR | S_IWUSR) &&
              metadata.st_size == static_cast<off_t>(expected.size());
    if (ok && inode != nullptr) *inode = metadata.st_ino;
    std::vector<uint8_t> observed(expected.size());
    size_t cursor = 0;
    while (ok && cursor < observed.size()) {
        const ssize_t count =
            ::read(fd, observed.data() + cursor, observed.size() - cursor);
        if (count > 0) {
            cursor += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            ok = false;
        }
    }
    uint8_t extra = 0;
    if (ok && ::read(fd, &extra, 1) != 0) ok = false;
    if (::close(fd) != 0) ok = false;
    return ok && observed == expected;
}

bool path_absent(const std::string &path) {
    struct stat metadata {};
    return ::lstat(path.c_str(), &metadata) != 0 && errno == ENOENT;
}

bool no_private_temps(const std::string &directory) {
    DIR *stream = ::opendir(directory.c_str());
    if (stream == nullptr) return false;
    bool clean = true;
    while (dirent *entry = ::readdir(stream)) {
        if (std::strncmp(entry->d_name, ".ringlpn-private-", 17) == 0) {
            clean = false;
            break;
        }
    }
    return ::closedir(stream) == 0 && clean;
}

bool create_blocker(const std::string &path,
                    const std::vector<uint8_t> &bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                           O_NOFOLLOW,
                          S_IRUSR | S_IWUSR);
    if (fd < 0) return false;
    const bool wrote =
        ringlpn_private_file::AtomicWriter::write_descriptor(
            fd, bytes.data(), bytes.size()) &&
        ::fsync(fd) == 0;
    const bool closed = ::close(fd) == 0;
    return wrote && closed;
}

}  // namespace

int main() {
    const mode_t previous_umask = ::umask(0000);
    char directory_template[] = "/tmp/ringlpn-private-file-XXXXXX";
    char *created = ::mkdtemp(directory_template);
    bool ok = require(created != nullptr, "mkdtemp");
    const std::string root = created == nullptr ? std::string() : created;
    const std::vector<uint8_t> payload = {
        0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff, 0x42, 0x19};
    const std::vector<uint8_t> blocker = {'b', 'l', 'o', 'c', 'k'};

    const std::string final = root + "/final.bin";
    if (ok) {
        ok = require(ringlpn_private_file::write_atomic(final, payload),
                     "atomic_write") &&
             require(read_exact_private(final, payload),
                     "mode_link_bytes_under_umask_000") &&
             require(no_private_temps(root), "no_temp_after_commit");
    }

    const std::string existing = root + "/existing.bin";
    if (ok) {
        ok = require(create_blocker(existing, blocker), "create_blocker") &&
             require(!ringlpn_private_file::write_atomic(existing, payload),
                     "reject_preexisting") &&
             require(read_exact_private(existing, blocker),
                     "preserve_preexisting") &&
             require(no_private_temps(root), "no_temp_preexisting");
    }

    const std::string symlink_path = root + "/symlink.bin";
    if (ok) {
        struct stat symlink_metadata {};
        ok = require(::symlink("final.bin", symlink_path.c_str()) == 0,
                     "create_symlink") &&
             require(!ringlpn_private_file::write_atomic(symlink_path, payload),
                     "reject_symlink") &&
             require(::lstat(symlink_path.c_str(), &symlink_metadata) == 0 &&
                         S_ISLNK(symlink_metadata.st_mode),
                     "preserve_symlink") &&
             require(no_private_temps(root), "no_temp_symlink");
    }

    const std::string escape_parent = root + "/escape-parent";
    const std::string escape_nested = escape_parent + "/nested";
    const std::string intermediate_link = root + "/intermediate-link";
    const std::string escaped_target = escape_nested + "/record.bin";
    const std::string traversed_target =
        intermediate_link + "/nested/record.bin";
    if (ok) {
        struct stat link_metadata {};
        ok = require(::mkdir(escape_parent.c_str(), S_IRWXU) == 0,
                     "create_escape_parent") &&
             require(::mkdir(escape_nested.c_str(), S_IRWXU) == 0,
                     "create_escape_nested") &&
             require(::symlink("escape-parent", intermediate_link.c_str()) == 0,
                     "create_intermediate_symlink") &&
             require(!ringlpn_private_file::write_atomic(traversed_target,
                                                         payload),
                     "reject_intermediate_parent_symlink") &&
             require(path_absent(escaped_target),
                     "intermediate_symlink_target_absent") &&
             require(::lstat(intermediate_link.c_str(), &link_metadata) == 0 &&
                         S_ISLNK(link_metadata.st_mode),
                     "preserve_intermediate_symlink") &&
             require(no_private_temps(escape_nested),
                     "no_temp_through_intermediate_symlink");
    }

    const std::string public_parent = root + "/public-parent";
    const std::string public_target = public_parent + "/record.bin";
    if (ok) {
        ok = require(::mkdir(public_parent.c_str(), S_IRWXU) == 0,
                     "create_public_parent") &&
             require(::chmod(public_parent.c_str(), 0755) == 0,
                     "make_parent_nonprivate") &&
             require(!ringlpn_private_file::write_atomic(public_target, payload),
                     "reject_nonprivate_parent") &&
             require(path_absent(public_target), "nonprivate_target_absent") &&
             require(no_private_temps(public_parent),
                     "no_temp_nonprivate_parent");
    }

    const std::string failed = root + "/failed.bin";
    if (ok) {
        ringlpn_private_file::AtomicWriter writer;
        const bool staged = writer.stage_stream(failed, [&](int fd) {
            return ringlpn_private_file::AtomicWriter::write_descriptor(
                       fd, payload.data(), payload.size() / 2) &&
                   false;
        });
        ok = require(!staged, "writer_failure_rejected") &&
             require(path_absent(failed), "writer_failure_no_final") &&
             require(no_private_temps(root), "writer_failure_no_temp");
    }

    const std::string rolled_back = root + "/rolled-back.bin";
    ino_t rolled_back_inode = 0;
    if (ok) {
        {
            ringlpn_private_file::AtomicWriter writer;
            ok = require(writer.stage(rolled_back, payload),
                         "rollback_stage") &&
                 require(writer.publish(), "rollback_publish") &&
                 require(read_exact_private(rolled_back, payload,
                                            &rolled_back_inode),
                         "rollback_published_inode");
        }
        ok = ok && require(rolled_back_inode != 0, "rollback_inode_recorded") &&
             require(path_absent(rolled_back), "uncommitted_inode_removed") &&
             require(no_private_temps(root), "rollback_no_temp");
    }

    const std::string committed = root + "/committed.bin";
    if (ok) {
        {
            ringlpn_private_file::AtomicWriter writer;
            ok = require(writer.stage(committed, payload), "commit_stage") &&
                 require(writer.publish(), "commit_publish");
            if (ok) writer.commit();
        }
        ok = ok && require(read_exact_private(committed, payload),
                           "committed_inode_persists") &&
             require(no_private_temps(root), "final_no_temp");
    }

    if (!root.empty()) {
        (void)::unlink(final.c_str());
        (void)::unlink(existing.c_str());
        (void)::unlink(symlink_path.c_str());
        (void)::unlink(failed.c_str());
        (void)::unlink(rolled_back.c_str());
        (void)::unlink(escaped_target.c_str());
        (void)::unlink(intermediate_link.c_str());
        (void)::rmdir(escape_nested.c_str());
        (void)::rmdir(escape_parent.c_str());
        (void)::unlink(committed.c_str());
        (void)::chmod(public_parent.c_str(), 0700);
        (void)::unlink(public_target.c_str());
        (void)::rmdir(public_parent.c_str());
        (void)::rmdir(root.c_str());
    }
    ::umask(previous_umask);
    if (!ok) return 1;
    std::printf("private_file_control=pass\n");
    return 0;
}
