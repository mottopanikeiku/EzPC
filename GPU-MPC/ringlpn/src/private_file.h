#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace ringlpn_private_file {

class AtomicWriter {
  public:
    AtomicWriter() = default;
    AtomicWriter(const AtomicWriter &) = delete;
    AtomicWriter &operator=(const AtomicWriter &) = delete;
    AtomicWriter(AtomicWriter &&) = delete;
    AtomicWriter &operator=(AtomicWriter &&) = delete;

    ~AtomicWriter() { cleanup(); }
    static bool destination_absent(const std::string &path) {
        std::string parent;
        std::string name;
        if (!split_path(path, parent, name)) return false;
        const int directory_fd = open_directory_no_follow(parent);
        struct stat directory_metadata {};
        struct stat destination_metadata {};
        const bool parent_safe =
            ::fstat(directory_fd, &directory_metadata) == 0 &&
            safe_private_directory(directory_metadata);
        const bool absent =
            parent_safe &&
            ::fstatat(directory_fd, name.c_str(), &destination_metadata,
                      AT_SYMLINK_NOFOLLOW) != 0 &&
            errno == ENOENT;
        const bool closed = ::close(directory_fd) == 0;
        return absent && closed;
    }


    static bool write_descriptor(int fd, const void *bytes, size_t size) {
        return (bytes != nullptr || size == 0) &&
               write_all(fd, static_cast<const uint8_t *>(bytes), size);
    }

    template <typename Write>
    bool stage_stream(const std::string &final_path, Write &&write) {
        cleanup();
        if (!split_path(final_path, parent_path_, final_name_)) {
            reset_names();
            return false;
        }

        directory_fd_ = open_directory_no_follow(parent_path_);
        struct stat directory_metadata {};
        if (directory_fd_ < 0 ||
            ::fstat(directory_fd_, &directory_metadata) != 0 ||
            !safe_private_directory(directory_metadata) ||
            !name_absent(final_name_)) {
            close_directory();
            reset_names();
            return false;
        }

        int file_fd = -1;
        for (unsigned attempt = 0; attempt < 16 && file_fd < 0; ++attempt) {
            std::string candidate;
            if (!random_temporary_name(candidate)) break;
            file_fd = ::openat(directory_fd_, candidate.c_str(),
                               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                   O_NOFOLLOW,
                               S_IRUSR | S_IWUSR);
            if (file_fd >= 0) {
                temporary_name_ = std::move(candidate);
                temporary_present_ = true;
                break;
            }
            if (errno != EEXIST) break;
        }
        if (file_fd < 0) {
            close_directory();
            reset_names();
            return false;
        }

        struct stat metadata {};
        bool ok = ::fstat(file_fd, &metadata) == 0 &&
                  safe_private_inode(metadata, true) && metadata.st_size == 0;
        if (ok) {
            device_ = metadata.st_dev;
            inode_ = metadata.st_ino;
            identity_known_ = true;
            ok = write(file_fd) && ::fsync(file_fd) == 0;
        }
        if (::close(file_fd) != 0) ok = false;
        if (ok) {
            struct stat staged {};
            ok = stat_matching(temporary_name_, staged, true) &&
                 staged.st_size >= 0;
        }
        if (!ok) {
            cleanup();
            return false;
        }
        return true;
    }

    bool stage(const std::string &final_path, const uint8_t *bytes, size_t size) {
        if ((bytes == nullptr && size != 0) ||
            size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
            return false;
        }
        return stage_stream(final_path, [&](int fd) {
            return write_all(fd, bytes, size);
        });
    }

    bool stage(const std::string &final_path,
               const std::vector<uint8_t> &bytes) {
        return stage(final_path, bytes.data(), bytes.size());
    }

    bool publish() {
        if (directory_fd_ < 0 || !temporary_present_ || final_present_ ||
            !identity_known_ || !name_absent(final_name_)) {
            return false;
        }
        struct stat staged {};
        if (!stat_matching(temporary_name_, staged, true)) return false;

#if defined(SYS_renameat2) && defined(RENAME_NOREPLACE)
        if (::syscall(SYS_renameat2, directory_fd_, temporary_name_.c_str(),
                      directory_fd_, final_name_.c_str(),
                      RENAME_NOREPLACE) == 0) {
            temporary_present_ = false;
            final_present_ = true;
        } else if (errno != ENOSYS && errno != EINVAL &&
                   errno != EOPNOTSUPP) {
            return false;
        }
#endif
        if (!final_present_) {
            if (::linkat(directory_fd_, temporary_name_.c_str(), directory_fd_,
                         final_name_.c_str(), 0) != 0) {
                return false;
            }
            final_present_ = true;
            if (::unlinkat(directory_fd_, temporary_name_.c_str(), 0) != 0) {
                return false;
            }
            temporary_present_ = false;
        }

        struct stat published {};
        if (!stat_matching(final_name_, published, true) ||
            ::fsync(directory_fd_) != 0) {
            return false;
        }
        return true;
    }

    void commit() {
        if (!final_present_ || temporary_present_) return;
        final_present_ = false;
        identity_known_ = false;
        close_directory();
        reset_names();
    }

    void cleanup() {
        bool removed = false;
        if (directory_fd_ >= 0 && identity_known_) {
            if (temporary_present_ && unlink_matching(temporary_name_)) {
                temporary_present_ = false;
                removed = true;
            }
            if (final_present_ && unlink_matching(final_name_)) {
                final_present_ = false;
                removed = true;
            }
            if (removed) (void)::fsync(directory_fd_);
        }
        identity_known_ = false;
        close_directory();
        reset_names();
    }

  private:
    static bool split_path(const std::string &path, std::string &parent,
                           std::string &name) {
        if (path.empty()) return false;
        const size_t separator = path.find_last_of('/');
        if (separator == std::string::npos) {
            parent = ".";
            name = path;
        } else {
            parent = separator == 0 ? "/" : path.substr(0, separator);
            name = path.substr(separator + 1);
        }
        return !name.empty() && name != "." && name != "..";
    }
    static int open_directory_no_follow(const std::string &path) {
        if (path.empty()) return -1;
        constexpr int flags =
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW;
        const bool absolute = path.front() == '/';
        int current_fd = ::open(absolute ? "/" : ".", flags);
        if (current_fd < 0) return -1;

        size_t cursor = absolute ? 1 : 0;
        while (cursor < path.size()) {
            while (cursor < path.size() && path[cursor] == '/') ++cursor;
            if (cursor == path.size()) break;
            const size_t separator = path.find('/', cursor);
            const size_t end =
                separator == std::string::npos ? path.size() : separator;
            const std::string component = path.substr(cursor, end - cursor);
            cursor = end;
            if (component == ".") continue;
            if (component == "..") {
                (void)::close(current_fd);
                return -1;
            }
            const int next_fd = ::openat(current_fd, component.c_str(), flags);
            const int open_error = errno;
            const bool closed = ::close(current_fd) == 0;
            if (next_fd < 0) {
                errno = open_error;
                return -1;
            }
            if (!closed) {
                (void)::close(next_fd);
                return -1;
            }
            current_fd = next_fd;
        }
        return current_fd;
    }


    static bool write_all(int fd, const uint8_t *bytes, size_t size) {
        size_t written = 0;
        while (written < size) {
            const ssize_t count =
                ::write(fd, bytes + written, size - written);
            if (count > 0) {
                written += static_cast<size_t>(count);
            } else if (count < 0 && errno == EINTR) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }

    static bool fill_random(uint8_t *bytes, size_t size) {
        size_t filled = 0;
        while (filled < size) {
            const ssize_t count = ::getrandom(bytes + filled, size - filled, 0);
            if (count > 0) {
                filled += static_cast<size_t>(count);
            } else if (count < 0 && errno == EINTR) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }

    static bool random_temporary_name(std::string &name) {
        uint8_t random[16];
        if (!fill_random(random, sizeof(random))) return false;
        static constexpr char hex[] = "0123456789abcdef";
        name = ".ringlpn-private-";
        name.reserve(name.size() + 2 * sizeof(random));
        for (uint8_t byte : random) {
            name.push_back(hex[byte >> 4]);
            name.push_back(hex[byte & 0x0f]);
        }
        return true;
    }

    static bool safe_private_directory(const struct stat &metadata) {
        return S_ISDIR(metadata.st_mode) && metadata.st_uid == ::geteuid() &&
               metadata.st_nlink >= 1 && (metadata.st_mode & 0077) == 0 &&
               (metadata.st_mode & S_IXUSR) != 0;
    }

    static bool safe_private_inode(const struct stat &metadata,
                                   bool require_single_link) {
        return S_ISREG(metadata.st_mode) && metadata.st_uid == ::geteuid() &&
               (metadata.st_mode & 07777) == (S_IRUSR | S_IWUSR) &&
               (!require_single_link || metadata.st_nlink == 1);
    }

    bool name_absent(const std::string &name) const {
        struct stat metadata {};
        if (::fstatat(directory_fd_, name.c_str(), &metadata,
                      AT_SYMLINK_NOFOLLOW) == 0) {
            return false;
        }
        return errno == ENOENT;
    }

    bool stat_matching(const std::string &name, struct stat &metadata,
                       bool require_single_link) const {
        return ::fstatat(directory_fd_, name.c_str(), &metadata,
                         AT_SYMLINK_NOFOLLOW) == 0 &&
               safe_private_inode(metadata, require_single_link) &&
               metadata.st_dev == device_ && metadata.st_ino == inode_;
    }

    bool unlink_matching(const std::string &name) const {
        struct stat metadata {};
        return stat_matching(name, metadata, false) &&
               ::unlinkat(directory_fd_, name.c_str(), 0) == 0;
    }

    void close_directory() {
        if (directory_fd_ >= 0) {
            (void)::close(directory_fd_);
            directory_fd_ = -1;
        }
    }

    void reset_names() {
        parent_path_.clear();
        final_name_.clear();
        temporary_name_.clear();
        temporary_present_ = false;
        final_present_ = false;
    }

    int directory_fd_ = -1;
    std::string parent_path_;
    std::string final_name_;
    std::string temporary_name_;
    dev_t device_ = 0;
    ino_t inode_ = 0;
    bool identity_known_ = false;
    bool temporary_present_ = false;
    bool final_present_ = false;
};

inline bool write_atomic(const std::string &path,
                         const std::vector<uint8_t> &bytes) {
    AtomicWriter writer;
    if (!writer.stage(path, bytes) || !writer.publish()) return false;
    writer.commit();
    return true;
}

}  // namespace ringlpn_private_file
