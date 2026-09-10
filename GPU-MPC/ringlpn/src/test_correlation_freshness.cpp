// Deterministically overlap two honest claims in one ledger. Linker wrappers
// pause the first writer after O_EXCL creates its pending file, before bytes
// are written. The second claimant must not mistake that live write for a crash.
#include "correlation_freshness.h"

#include <cstdlib>
#include <filesystem>
#include <sys/file.h>
#include <sys/wait.h>

namespace {
int writer_ready = -1;
int writer_release = -1;
int contender_ready = -1;
bool pause_write = false;

bool receive(int fd, char &byte) {
    ssize_t count;
    do { count = ::read(fd, &byte, 1); } while (count < 0 && errno == EINTR);
    return count == 1;
}

bool succeeded(pid_t child) {
    int status = 0;
    pid_t result;
    do { result = ::waitpid(child, &status, 0); }
    while (result < 0 && errno == EINTR);
    return result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool claim(const std::string &root, int party, uint8_t invocation_byte = 1) {
    ringlpn_freshness::InvocationId invocation{};
    ringlpn_freshness::Digest layer{}, plan{};
    invocation[0] = invocation_byte;
    layer[0] = 2;
    plan[0] = 3;
    ringlpn_freshness::Claim result;
    return ringlpn_freshness::claim_namespace_once(
        root, party, invocation, layer, plan, result);
}
}  // namespace

extern "C" ssize_t __real_write(int, const void *, size_t);
extern "C" int __real_flock(int, int);

extern "C" ssize_t __wrap_write(int fd, const void *data, size_t size) {
    if (pause_write) {
        pause_write = false;
        char byte = 'W';
        if (__real_write(writer_ready, &byte, 1) != 1 ||
            !receive(writer_release, byte)) {
            ::_exit(2);
        }
    }
    return __real_write(fd, data, size);
}

extern "C" int __wrap_flock(int fd, int operation) {
    if (contender_ready >= 0 && operation == LOCK_EX) {
        const char byte = 'L';
        if (__real_write(contender_ready, &byte, 1) != 1) ::_exit(2);
        contender_ready = -1;
    }
    return __real_flock(fd, operation);
}

int main() {
    ::alarm(20);
    char pattern[] = "/tmp/ringlpn-freshness-test-XXXXXX";
    const char *created = ::mkdtemp(pattern);
    if (created == nullptr) return 2;
    const std::string root = created;
    int ready[2], release[2], contender[2];
    if (::pipe(ready) != 0 || ::pipe(release) != 0 ||
        ::pipe(contender) != 0) return 2;
    const pid_t first = ::fork();
    if (first < 0) return 2;
    if (first == 0) {
        ::alarm(10);
        writer_ready = ready[1];
        writer_release = release[0];
        pause_write = true;
        ::_exit(claim(root, 0) ? 0 : 1);
    }
    char byte = 0;
    if (!receive(ready[0], byte) || byte != 'W') return 2;
    const pid_t second = ::fork();
    if (second < 0) return 2;
    if (second == 0) {
        ::alarm(10);
        contender_ready = contender[1];
        const bool ok = claim(root, 1);
        const char finished = 'R';
        (void)__real_write(contender[1], &finished, 1);
        ::_exit(ok ? 0 : 1);
    }
    // A fixed implementation announces lock acquisition before blocking. The
    // original implementation instead returns failure against the partial file.
    if (!receive(contender[0], byte)) return 2;
    const char resume = 'C';
    if (__real_write(release[1], &resume, 1) != 1) return 2;
    const bool first_ok = succeeded(first);
    const bool second_ok = succeeded(second);
    bool ok = first_ok && second_ok &&
              ringlpn_freshness::validate_ledger_entries(root) &&
              !claim(root, 0) && !claim(root, 1) && claim(root, 0, 2);
    std::printf("concurrent_claims=%s\n", first_ok && second_ok ? "pass" : "FAIL");

    // A genuinely crash-truncated pending file must still poison new claims.
    const std::string pending = root + "/crashed.pending";
    int fd = ::open(pending.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    const bool created_pending = fd >= 0;
    if (fd >= 0 && ::close(fd) != 0) ok = false;
    ok = created_pending && !claim(root, 1, 3) && ok;
    (void)::unlink(pending.c_str());

    // Nonregular poison records must reject, not hang in open before fstat.
    const std::string fifo = root + "/nonregular.pending";
    ok = (::mkfifo(fifo.c_str(), 0600) == 0) && ok;
    const pid_t fifo_check = ::fork();
    if (fifo_check == 0) {
        ::alarm(2);
        ::_exit(claim(root, 1, 4) ? 1 : 0);
    }
    ok = fifo_check > 0 && succeeded(fifo_check) && ok;
    for (int fd_to_close : {ready[0], ready[1], release[0], release[1],
                            contender[0], contender[1]}) {
        (void)::close(fd_to_close);
    }
    std::filesystem::remove_all(root);
    std::puts(ok ? "correlation_freshness_control=pass"
                 : "correlation_freshness_control=FAIL");
    return ok ? 0 : 1;
}
