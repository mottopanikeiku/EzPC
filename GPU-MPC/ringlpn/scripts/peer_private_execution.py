#!/usr/bin/env python3
"""Fail-closed rootless-container backend for peer-private Ring-LPN runs.

The authenticated two-host coordinator is scripts/run_two_host_authenticated.sh.
This file deliberately does not open sockets, invoke ssh, or copy between hosts.
It only supplies party isolation and the post-exit checker boundary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, NoReturn, Sequence

SCHEMA = "ringlpn-peer-private-v1"
PRIVATE_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
PARTY_MOUNT = "/run/ringlpn/private"
LEDGER_MOUNT = "/run/ringlpn/ledger"
CHECK_P0_MOUNT = "/run/ringlpn/checker/party0"
CHECK_P1_MOUNT = "/run/ringlpn/checker/party1"
CHECK_OUT_MOUNT = "/run/ringlpn/checker/output"


def fail(message: str) -> NoReturn:
    print(f"peer-private: {message}", file=sys.stderr)
    raise SystemExit(2)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")



def parse_utc(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        fail(f"{label} is not an ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label} is not an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        fail(f"{label} must be UTC")
    return parsed

def run(argv: Sequence[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            check=check,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError:
        fail(f"required executable is unavailable: {argv[0]}")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"command failed closed ({exc.returncode}): {argv[0]}{': ' + detail if detail else ''}")


def require_absolute(path: str, label: str) -> pathlib.Path:
    value = pathlib.Path(path)
    if not value.is_absolute():
        fail(f"{label} must be absolute")
    if value.is_symlink():
        fail(f"{label} must not be a symlink")
    return value


def require_separate(paths: Sequence[pathlib.Path]) -> None:
    resolved = [p.resolve(strict=False) for p in paths]
    if len(set(resolved)) != len(resolved):
        fail("private/checker roots must be distinct")
    for i, left in enumerate(resolved):
        for right in resolved[i + 1 :]:
            if left in right.parents or right in left.parents:
                fail("private/checker roots must not contain one another")


def require_manifest_outside(root: pathlib.Path, manifest: pathlib.Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_manifest = manifest.resolve(strict=False)
    if resolved_manifest == resolved_root or resolved_root in resolved_manifest.parents:
        fail("manifest must be outside every private/checker root")


def mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def private_evidence(root: pathlib.Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    root_device = root.lstat().st_dev
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        relative = "." if path == root else str(path.relative_to(root))
        if stat.S_ISLNK(info.st_mode):
            fail(f"symlink is forbidden in private root: {path}")
        if stat.S_IMODE(info.st_mode) & 0o077:
            fail(f"group/other access is forbidden in private root: {path}")
        if path != root and info.st_dev != root_device:
            fail(f"nested mount/device is forbidden in private root: {path}")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            fail(f"hard-linked file is forbidden in private root: {path}")
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            fail(f"unsupported private-root object: {path}")
        evidence.append(
            {
                "path": relative,
                "type": "directory" if stat.S_ISDIR(info.st_mode) else "file",
                "uid": info.st_uid,
                "gid": info.st_gid,
                "mode": mode_string(info.st_mode),
                "size": info.st_size if stat.S_ISREG(info.st_mode) else None,
            }
        )
    return evidence


def prepare_private_root(root: pathlib.Path, *, party: bool) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=PRIVATE_MODE)
    os.chmod(root, PRIVATE_MODE)
    if party:
        for name in ("input", "tmp", "output"):
            child = root / name
            child.mkdir(mode=PRIVATE_MODE, exist_ok=True)
            if child.is_symlink() or not child.is_dir():
                fail(f"private {name} path is not a real directory")
            os.chmod(child, PRIVATE_MODE)
    private_evidence(root)
def fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def channel_auth_cleanup_marker(root: pathlib.Path) -> pathlib.Path:
    return root.parent / f".{root.name}.channel-auth-cleanup"


def channel_auth_marker_document(
    root: pathlib.Path,
    expected: str,
    session_id: str,
    invocation_id: str,
    party: int,
    root_info: os.stat_result,
) -> dict[str, Any]:
    return {
        "schema": "ringlpn-channel-auth-cleanup-v1",
        "session_id": str(session_id),
        "invocation_id": invocation_id,
        "party": party,
        "private_root": str(root.resolve(strict=False)),
        "channel_auth_secret_sha256": expected,
        "owner_uid": os.geteuid(),
        "root_device": root_info.st_dev,
        "root_inode": root_info.st_ino,
    }


def channel_auth_marker_payload(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def validate_channel_auth_cleanup_marker(
    root: pathlib.Path,
    expected: str,
    session_id: str,
    invocation_id: str,
    party: int,
) -> tuple[pathlib.Path, dict[str, Any]]:
    marker = channel_auth_cleanup_marker(root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(marker, flags)
        try:
            info = os.fstat(fd)
            payload = bytearray()
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                payload.extend(chunk)
        finally:
            os.close(fd)
        document = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"channel authenticator cleanup marker is unavailable: {exc}")
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != PRIVATE_FILE_MODE
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or document.get("schema") != "ringlpn-channel-auth-cleanup-v1"
        or document.get("session_id") != str(session_id)
        or document.get("invocation_id") != invocation_id
        or document.get("party") != party
        or document.get("private_root") != str(root.resolve(strict=False))
        or document.get("channel_auth_secret_sha256") != expected
        or document.get("owner_uid") != os.geteuid()
        or not isinstance(document.get("root_device"), int)
        or not isinstance(document.get("root_inode"), int)
    ):
        fail("channel authenticator cleanup marker is not the exact owner-only authorization")
    if root.exists() or root.is_symlink():
        root_info = root.lstat()
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != PRIVATE_MODE
            or root_info.st_dev != document["root_device"]
            or root_info.st_ino != document["root_inode"]
        ):
            fail("private root no longer matches its cleanup-authorized inode")
    return marker, document


def remove_channel_auth_transaction(
    root: pathlib.Path,
    expected: str,
    session_id: str,
    invocation_id: str,
    party: int,
) -> None:
    marker, _ = validate_channel_auth_cleanup_marker(
        root, expected, session_id, invocation_id, party
    )
    if root.exists() or root.is_symlink():
        allowed = {"input", "tmp", "output", "channel-auth.key"}
        if {path.name for path in root.iterdir()} - allowed:
            fail("cleanup-authorized private root contains unexpected entries")
        private_evidence(root)
        shutil.rmtree(root)
        fsync_directory(root.parent)
    marker.unlink()
    fsync_directory(marker.parent)
    if root.exists() or root.is_symlink() or marker.exists() or marker.is_symlink():
        fail("channel authenticator transaction cleanup left private state behind")


def provision_channel_auth_command(args: argparse.Namespace) -> int:
    root = require_absolute(args.private_root, "private root")
    expected = require_sha256(args.channel_auth_secret_sha256, "channel authenticator digest")
    party = int(args.party)
    if (
        not args.session_id
        or not args.invocation_id
        or len(args.invocation_id) != 32
        or any(character not in "0123456789abcdef" for character in args.invocation_id)
    ):
        fail("channel authenticator cleanup binding is malformed")
    marker = channel_auth_cleanup_marker(root)
    if root.exists() or root.is_symlink():
        fail("channel authenticator private root must be fresh")
    if marker.exists() or marker.is_symlink():
        fail("channel authenticator cleanup marker must be fresh")
    try:
        parent_info = root.parent.lstat()
    except OSError as exc:
        fail(f"channel authenticator private-root parent is unavailable: {exc}")
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or root.parent.is_symlink()
        or parent_info.st_uid != os.geteuid()
        or stat.S_IMODE(parent_info.st_mode) & 0o077
    ):
        fail("channel authenticator private-root parent must be an owner-only real directory")
    secret = bytearray(sys.stdin.buffer.read(33))
    root_created = False
    root_identity: tuple[int, int] | None = None
    marker_created = False
    marker_identity: tuple[int, int] | None = None
    authorization_durable = False
    try:
        if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != expected:
            fail("channel authenticator input is not the expected 32-byte secret")

        # Reserve the exact inode while it is empty. No secret-bearing mutation
        # occurs until its durable adjacent cleanup authorization is complete.
        root.mkdir(mode=PRIVATE_MODE)
        root_created = True
        root_info = root.lstat()
        root_identity = (root_info.st_dev, root_info.st_ino)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != PRIVATE_MODE
            or root_info.st_uid != os.geteuid()
        ):
            raise OSError("reserved private root identity is invalid")
        fsync_directory(root.parent)

        marker_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        marker_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        marker_fd = os.open(marker, marker_flags, PRIVATE_FILE_MODE)
        marker_created = True
        marker_info = os.fstat(marker_fd)
        marker_identity = (marker_info.st_dev, marker_info.st_ino)
        try:
            payload = channel_auth_marker_payload(
                channel_auth_marker_document(
                    root,
                    expected,
                    args.session_id,
                    args.invocation_id,
                    party,
                    root_info,
                )
            )
            cursor = 0
            while cursor < len(payload):
                wrote = os.write(marker_fd, payload[cursor:])
                if wrote <= 0:
                    raise OSError("short write while creating cleanup authorization")
                cursor += wrote
            os.fchmod(marker_fd, PRIVATE_FILE_MODE)
            os.fsync(marker_fd)
        finally:
            os.close(marker_fd)
        fsync_directory(marker.parent)
        validate_channel_auth_cleanup_marker(
            root, expected, args.session_id, args.invocation_id, party
        )
        authorization_durable = True

        prepare_private_root(root, party=True)
        path = root / "channel-auth.key"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, PRIVATE_FILE_MODE)
        try:
            os.fchmod(fd, PRIVATE_FILE_MODE)
            cursor = 0
            while cursor < len(secret):
                wrote = os.write(fd, secret[cursor:])
                if wrote <= 0:
                    raise OSError("short write while provisioning channel authenticator")
                cursor += wrote
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(root)
        validate_channel_auth_file(root, expected)
    except BaseException:
        try:
            if authorization_durable:
                remove_channel_auth_transaction(
                    root, expected, args.session_id, args.invocation_id, party
                )
            else:
                if marker_created:
                    info = marker.lstat()
                    if marker_identity != (info.st_dev, info.st_ino):
                        fail("cleanup marker identity changed during provisioning rollback")
                    marker.unlink()
                    fsync_directory(marker.parent)
                if root_created:
                    info = root.lstat()
                    if (
                        root_identity != (info.st_dev, info.st_ino)
                        or not stat.S_ISDIR(info.st_mode)
                        or any(root.iterdir())
                    ):
                        fail("reserved private root changed before authorization rollback")
                    root.rmdir()
                    fsync_directory(root.parent)
        except BaseException as cleanup_exc:
            fail(f"channel authenticator provisioning rollback failed: {cleanup_exc}")
        raise
    finally:
        for index in range(len(secret)):
            secret[index] = 0
    return 0


def validate_channel_auth_file(root: pathlib.Path, expected: str) -> None:
    path = root / "channel-auth.key"
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != PRIVATE_FILE_MODE
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or info.st_size != 32
        or sha256_file(path) != expected
    ):
        fail("channel authenticator is not the provisioned owner-only regular file")


def abort_provisioned_channel_auth_command(args: argparse.Namespace) -> int:
    root = require_absolute(args.private_root, "private root")
    expected = require_sha256(args.channel_auth_secret_sha256, "channel authenticator digest")
    remove_channel_auth_transaction(
        root, expected, args.session_id, args.invocation_id, int(args.party)
    )
    return 0



def write_manifest(path: pathlib.Path, document: dict[str, Any]) -> None:
    if not path.is_absolute():
        fail("manifest path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True, mode=PRIVATE_MODE)
    os.chmod(path.parent, PRIVATE_MODE)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, PRIVATE_FILE_MODE)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def load_manifest(
    path: pathlib.Path,
    expected_party: int | None = None,
    *,
    require_sealed: bool = True,
) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            fail(f"manifest must be an owner-only regular file: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {path}: {exc}")
    if document.get("schema") != SCHEMA or document.get("phase") != "party-exited":
        fail(f"manifest is not a completed party record: {path}")
    if expected_party is not None and document.get("party") != expected_party:
        fail(f"manifest party does not match expected party {expected_party}: {path}")
    if document.get("return_code") != 0 or document.get("container", {}).get("running") is not False:
        fail(f"party did not exit successfully: {path}")
    if not document.get("ended_at"):
        fail(f"party completion timestamp is absent: {path}")
    if require_sealed and document.get("sealed") is not True:
        fail(f"party output has not completed the post-exit seal phase: {path}")
    return document


def podman_info() -> tuple[str, dict[str, Any]]:
    binary = shutil.which("podman")
    if not binary:
        fail("rootless Podman is required; no permissive host-process fallback exists")
    completed = run([binary, "info", "--format", "json"], capture=True)
    try:
        info = json.loads(completed.stdout)
    except json.JSONDecodeError:
        fail("Podman returned malformed capability information")
    security = info.get("host", {}).get("security", {})
    if not security.get("rootless", False):
        fail("Podman must run rootless")
    if security.get("userNamespaceEnabled") is False:
        fail("Podman reports user namespaces disabled")
    return binary, info


def inspect_container(podman: str, name: str) -> dict[str, Any]:
    completed = run([podman, "inspect", name], capture=True)
    try:
        values = json.loads(completed.stdout)
        if len(values) != 1:
            raise ValueError("unexpected inspect result count")
        return values[0]
    except (json.JSONDecodeError, ValueError) as exc:
        fail(f"cannot inspect isolation container {name}: {exc}")


def container_evidence(inspect: dict[str, Any]) -> dict[str, Any]:
    state = inspect.get("State", {})
    config = inspect.get("Config", {})
    host_config = inspect.get("HostConfig", {})
    mounts = []
    for mount in inspect.get("Mounts", []):
        mounts.append(
            {
                "destination": mount.get("Destination"),
                "rw": mount.get("RW"),
                "type": mount.get("Type"),
            }
        )
    return {
        "id": inspect.get("Id"),
        "name": inspect.get("Name"),
        "image": inspect.get("ImageName") or config.get("Image"),
        "created": inspect.get("Created"),
        "running": bool(state.get("Running", False)),
        "status": state.get("Status"),
        "pid": state.get("Pid"),
        "exit_code": state.get("ExitCode"),
        "user": config.get("User"),
        "read_only_rootfs": host_config.get("ReadonlyRootfs"),
        "network_mode": host_config.get("NetworkMode"),
        "pid_mode": host_config.get("PidMode"),
        "ipc_mode": host_config.get("IpcMode"),
        "cap_add": host_config.get("CapAdd") or [],
        "cap_drop": host_config.get("CapDrop") or [],
        "security_opt": host_config.get("SecurityOpt") or [],
        "uid_map": inspect.get("HostConfig", {}).get("IDMappings", {}).get("UidMap", []),
        "gid_map": inspect.get("HostConfig", {}).get("IDMappings", {}).get("GidMap", []),
        "mounts": mounts,
        "labels": config.get("Labels") or {},
        "tmpfs": host_config.get("Tmpfs") or {},
        "devices": host_config.get("Devices") or [],
    }


def ensure_gpu(gpu: str) -> None:
    if not gpu or any(character.isspace() for character in gpu) or "/" in gpu:
        fail("GPU CDI selector must be a nonempty index or UUID without whitespace")


def parse_mode_evidence(output: str, root: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    prefix = "RINGLPN_MODE\t"
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            fail("container returned malformed file-mode evidence")
        _, path, kind, uid, gid, mode, size = fields
        if not path.startswith(root):
            fail("container returned file-mode evidence outside its private mount")
        evidence.append(
            {
                "path": "." if path == root else path.removeprefix(root + "/"),
                "type": kind,
                "uid": int(uid),
                "gid": int(gid),
                "mode": mode.zfill(4),
                "size": int(size),
            }
        )
    if not evidence:
        fail("container produced no file-mode evidence")
    if any(int(entry["mode"], 8) & 0o077 for entry in evidence):
        fail("container reports group/other access on a private path")
    return evidence


def host_root_evidence(root: pathlib.Path) -> dict[str, Any]:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != PRIVATE_MODE:
        fail(f"private root is not an owner-only directory: {root}")
    return {"path": str(root), "uid": info.st_uid, "gid": info.st_gid, "mode": mode_string(info.st_mode)}



def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash committed artifact {path}: {exc}")
    return digest.hexdigest()

def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        fail(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def machine_identity_sha256() -> str:
    for candidate in (pathlib.Path("/etc/machine-id"), pathlib.Path("/var/lib/dbus/machine-id")):
        try:
            value = candidate.read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if len(value) == 32 and all(character in "0123456789abcdef" for character in value):
            return hashlib.sha256(
                b"ringlpn-stable-machine-identity-v1\0" + value.encode("ascii")
            ).hexdigest()
    fail("no valid stable OS machine identity is available")


def machine_identity_command(_: argparse.Namespace) -> int:
    print(machine_identity_sha256())
    return 0


def runtime_identity_command(args: argparse.Namespace) -> int:
    binary = require_absolute(args.binary, "in-image binary")
    expected_image_digest = args.image.rsplit("@sha256:", 1)
    if len(expected_image_digest) != 2 or len(expected_image_digest[1]) != 64:
        fail("runtime image must be an immutable sha256 digest reference")
    require_sha256(expected_image_digest[1].lower(), "runtime image digest")
    podman, _ = podman_info()
    inspected = run([podman, "image", "inspect", args.image], capture=True)
    try:
        values = json.loads(inspected.stdout)
        image_id = values[0]["Id"]
        repo_digests = values[0].get("RepoDigests") or []
    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        fail("cannot establish exact local runtime image identity")
    if args.image not in repo_digests:
        fail("local runtime image does not expose the authorized repository digest")
    measured = run(
        [
            podman,
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--entrypoint",
            "/usr/bin/sha256sum",
            args.image,
            str(binary),
        ],
        capture=True,
    ).stdout.split()
    if len(measured) != 2 or measured[1] != str(binary):
        fail("runtime binary identity output is malformed")
    binary_sha256 = require_sha256(measured[0], "runtime binary digest")
    print(
        json.dumps(
            {
                "image": args.image,
                "image_id": image_id,
                "binary_path": str(binary),
                "binary_sha256": binary_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def capability_preflight_command(args: argparse.Namespace) -> int:
    ensure_gpu(args.gpu)
    expected_binary = require_sha256(args.binary_sha256, "authorized binary digest")
    binary = require_absolute(args.binary, "in-image binary")
    podman, info = podman_info()
    if os.geteuid() == 0 or info.get("host", {}).get("security", {}).get("rootless") is not True:
        fail("capability preflight requires native rootless Podman")
    required_cpu_features = {"aes", "avx2", "pclmulqdq", "rdseed", "sse4_1"}
    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text(encoding="ascii")
    except OSError as exc:
        fail(f"cannot read host CPU capabilities: {exc}")
    processor_features = []
    for block in cpuinfo.split("\n\n"):
        fields = {
            key.strip(): value.strip()
            for line in block.splitlines() if ":" in line
            for key, value in (line.split(":", 1),)
        }
        if "processor" in fields:
            processor_features.append(set(fields.get("flags", "").split()))
    common_features = (
        set.intersection(*processor_features) if processor_features else set()
    )
    missing_cpu = sorted(required_cpu_features - common_features)
    if missing_cpu:
        fail(
            "required CPU features unavailable on every visible CPU: "
            + ", ".join(missing_cpu)
        )
    user_names = {str(os.getuid())}
    try:
        import pwd

        user_names.add(pwd.getpwuid(os.getuid()).pw_name)
    except (ImportError, KeyError):
        pass
    subordinate = {}
    for label, source in (
        ("subuid", pathlib.Path("/etc/subuid")),
        ("subgid", pathlib.Path("/etc/subgid")),
    ):
        try:
            entries = [
                line.split(":")
                for line in source.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
        except OSError as exc:
            fail(f"cannot read /etc/{label}: {exc}")
        matching = [
            parts
            for parts in entries
            if len(parts) == 3
            and parts[0] in user_names
            and parts[1].isdigit()
            and parts[2].isdigit()
            and int(parts[2]) > 0
        ]
        if not matching:
            fail(f"rootless Podman user has no usable /etc/{label} allocation")
        subordinate[label] = sum(int(parts[2]) for parts in matching)
    runtime = json.loads(
        subprocess.check_output(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "runtime-identity",
             "--image", args.image, "--binary", str(binary)],
            text=True,
        )
    )
    if runtime.get("binary_sha256") != expected_binary:
        fail("capability preflight measured an unauthorized in-image binary")
    probe = run(
        [
            podman,
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--userns=auto",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            "--entrypoint",
            "/bin/sh",
            args.image,
            "-c",
            (
                'test -r "$1" && '
                'test "$(sha256sum "$1" | cut -d" " -f1)" = "$2" && '
                "exec nvidia-smi --query-gpu=compute_cap,uuid,pci.bus_id "
                "--format=csv,noheader,nounits"
            ),
            "ringlpn-preflight",
            str(binary),
            expected_binary,
        ],
        capture=True,
    )
    probe_lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if len(probe_lines) != 1:
        fail("selected CDI device did not expose exactly one physical GPU")
    measured_gpu = [field.strip() for field in probe_lines[0].split(",")]
    if (
        len(measured_gpu) != 3
        or measured_gpu[0] != "8.9"
        or not measured_gpu[1].startswith("GPU-")
        or ":" not in measured_gpu[2]
        or "." not in measured_gpu[2]
    ):
        fail("selected GPU physical identity output is malformed or not sm_89")
    print(
        json.dumps(
            {
                "schema": "ringlpn-native-podman-preflight-v1",
                "machine_identity_sha256": machine_identity_sha256(),
                "rootless": True,
                "subuid_count": subordinate["subuid"],
                "subgid_count": subordinate["subgid"],
                "gpu": f"nvidia.com/gpu={args.gpu}",
                "gpu_uuid": measured_gpu[1],
                "gpu_pci_bus_id": measured_gpu[2].lower(),
                "compute_capability": measured_gpu[0],
                "required_cpu_features": sorted(required_cpu_features),
                "runtime": runtime,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def load_prepared_manifest(
    path: pathlib.Path,
    *,
    session_id: Any,
    p0_record: pathlib.Path,
    p1_record: pathlib.Path,
    p0_manifest: pathlib.Path,
    p1_manifest: pathlib.Path,
) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != PRIVATE_FILE_MODE:
            fail("PREPARED manifest must be an owner-only regular file")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read PREPARED manifest: {exc}")
    exact_keys = {
        "schema",
        "state",
        "session_id",
        "invocation_id",
        "channel",
        "base_port",
        "reversed_port",
        "public_parameters_sha256",
        "channel_auth_secret_sha256",
        "runtime_identity",
        "machine_identities",
        "p0_exit_code",
        "p1_exit_code",
        "p0_record",
        "p1_record",
        "p0_isolation_manifest",
        "p1_isolation_manifest",
        "prepared_at",
    }
    if set(document) != exact_keys:
        fail("PREPARED manifest does not have the exact required fields")
    if (
        document.get("schema") != "ringlpn-two-host-prepare-v1"
        or document.get("state") != "PREPARED"
        or type(document.get("session_id")) is not int
        or document["session_id"] <= 0
        or str(document["session_id"]) != str(session_id)
        or document.get("channel") != "authenticated-ssh"
        or type(document.get("base_port")) is not int
        or document["base_port"] < 1
        or document["base_port"] > 65534
        or type(document.get("reversed_port")) is not int
        or document["reversed_port"] != document["base_port"] + 1
        or type(document.get("p0_exit_code")) is not int
        or document["p0_exit_code"] != 0
        or type(document.get("p1_exit_code")) is not int
        or document["p1_exit_code"] != 0
    ):
        fail("PREPARED manifest boundary, session, port, or exit state is invalid")
    require_sha256(
        document.get("channel_auth_secret_sha256"),
        "PREPARED channel authenticator digest",
    )
    invocation_id = document.get("invocation_id")
    if not isinstance(invocation_id, str) or len(invocation_id) != 32 or any(
        character not in "0123456789abcdef" for character in invocation_id
    ):
        fail("PREPARED manifest invocation ID is malformed")
    runtime = document.get("runtime_identity")
    if not isinstance(runtime, dict) or set(runtime) != {
        "container_image",
        "container_binary",
        "container_binary_sha256",
    }:
        fail("PREPARED manifest runtime identity is incomplete")
    require_absolute(str(runtime["container_binary"]), "PREPARED container binary")
    require_sha256(runtime["container_binary_sha256"], "PREPARED container binary digest")
    machine_identities = document.get("machine_identities")
    if not isinstance(machine_identities, dict) or set(machine_identities) != {
        "party0_sha256",
        "party1_sha256",
    }:
        fail("PREPARED manifest machine identities are incomplete")
    machine0 = require_sha256(machine_identities["party0_sha256"], "party 0 machine identity")
    machine1 = require_sha256(machine_identities["party1_sha256"], "party 1 machine identity")
    if machine0 == machine1:
        fail("PREPARED manifest identifies both parties as the same machine")
    require_sha256(document.get("public_parameters_sha256"), "PREPARED public-parameter digest")
    expected = {
        "p0_record": ("party0/key_p0.fc", p0_record),
        "p1_record": ("party1/key_p1.fc", p1_record),
        "p0_isolation_manifest": ("party0/isolation-manifest.json", p0_manifest),
        "p1_isolation_manifest": ("party1/isolation-manifest.json", p1_manifest),
    }
    for key, (relative_path, expected_path) in expected.items():
        entry = document.get(key)
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            fail(f"PREPARED manifest lacks exact {key} fields")
        if entry.get("path") != relative_path:
            fail(f"PREPARED manifest {key} path is not canonical")
        if (path.parent / relative_path).resolve(strict=False) != expected_path.resolve(strict=False):
            fail(f"PREPARED manifest {key} path does not match checker input")
        require_sha256(entry.get("sha256"), f"PREPARED {key} digest")
        if sha256_file(expected_path) != entry["sha256"]:
            fail(f"PREPARED manifest {key} digest mismatch")
    parse_utc(document.get("prepared_at"), "PREPARED timestamp")
    return document

def verify_container_isolation(
    evidence: dict[str, Any],
    *,
    writable_mounts: set[str],
    readonly_mounts: set[str],
) -> None:
    if evidence.get("read_only_rootfs") is not True:
        fail("container root filesystem is not read-only")
    if evidence.get("pid_mode") == "host" or evidence.get("ipc_mode") == "host":
        fail("container unexpectedly shares a host process or IPC namespace")
    if evidence.get("cap_add"):
        fail("container has added Linux capabilities")
    dropped = {str(value).upper() for value in evidence.get("cap_drop", [])}
    if "ALL" not in dropped and "CAP_ALL" not in dropped:
        fail("container does not drop all Linux capabilities")
    security = {str(value).lower() for value in evidence.get("security_opt", [])}
    if not any("no-new-privileges" in value for value in security):
        fail("container does not enforce no-new-privileges")
    seen_writable: set[str] = set()
    seen_readonly: set[str] = set()
    for mount in evidence.get("mounts", []):
        destination = mount.get("destination")
        if destination in writable_mounts and mount.get("rw") is True:
            seen_writable.add(destination)
        elif destination in readonly_mounts and mount.get("rw") is False:
            seen_readonly.add(destination)
        elif destination in writable_mounts | readonly_mounts:
            fail(f"container mount has the wrong access mode: {destination}")
        elif destination and str(destination).startswith("/run/ringlpn/"):
            fail(f"unexpected Ring-LPN mount in container: {destination}")
    if seen_writable != writable_mounts or seen_readonly != readonly_mounts:
        fail("container is missing a required private mount")


def party_command(args: argparse.Namespace) -> int:
    if not args.command:
        fail("run-party requires a command after --")
    party = int(args.party)
    root = require_absolute(args.private_root, "private root")
    ledger_root = require_absolute(args.ledger_root, "persistent ledger root")
    manifest = require_absolute(args.manifest, "manifest")
    require_separate([root, ledger_root])
    require_manifest_outside(root, manifest)
    require_manifest_outside(ledger_root, manifest)
    try:
        ledger_info = ledger_root.lstat()
    except OSError as exc:
        fail(f"persistent ledger root is unavailable: {exc}")
    if (
        not stat.S_ISDIR(ledger_info.st_mode)
        or ledger_root.is_symlink()
        or stat.S_IMODE(ledger_info.st_mode) & 0o077
    ):
        fail("persistent ledger root must be an owner-only real directory")
    private_evidence(ledger_root)
    ensure_gpu(args.gpu)
    if args.uid < 1:
        fail("container UID must be non-root and positive")
    require_sha256(args.public_parameters_sha256, "public-parameter digest")
    require_sha256(args.container_binary_sha256, "container binary digest")
    require_sha256(args.machine_identity_sha256, "machine identity digest")
    channel_auth_secret_sha256 = require_sha256(
        args.channel_auth_secret_sha256, "channel authenticator digest"
    )
    if args.machine_identity_sha256 != machine_identity_sha256():
        fail("party machine identity does not match the authenticated coordinator binding")
    if len(args.invocation_id) != 32 or any(
        character not in "0123456789abcdef" for character in args.invocation_id
    ):
        fail("invocation ID must be 32 lowercase hexadecimal characters")
    container_binary = require_absolute(args.container_binary, "container binary")
    validate_channel_auth_cleanup_marker(
        root,
        channel_auth_secret_sha256,
        args.session_id,
        args.invocation_id,
        party,
    )
    prepare_private_root(root, party=True)
    validate_channel_auth_file(root, channel_auth_secret_sha256)
    podman, info = podman_info()
    name = args.container_name or f"ringlpn-{args.session_id}-party{party}"
    if not name.replace("-", "").replace("_", "").isalnum():
        fail("container name contains unsupported characters")

    wrapper = (
        "umask 077; "
        "test ! -e /run/ringlpn/peer-private || exit 125; "
        "test \"$(stat -c %a /run/ringlpn/private)\" = 700 || exit 125; "
        "test -f /run/ringlpn/private/channel-auth.key || exit 125; "
        "test ! -L /run/ringlpn/private/channel-auth.key || exit 125; "
        "test \"$(stat -c %a /run/ringlpn/private/channel-auth.key)\" = 600 || exit 125; "
        "test \"$(stat -c %s /run/ringlpn/private/channel-auth.key)\" = 32 || exit 125; "
        "test \"$(stat -c %a /run/ringlpn/ledger)\" = 700 || exit 125; "
        "set +e; \"$@\" > /run/ringlpn/private/output/process.log 2>&1; rc=$?; "
        "test ! -e /run/ringlpn/private/channel-auth.key || rc=125; "
        "printf '%s\\n' \"$rc\" > /run/ringlpn/private/output/return-code; "
        "chmod 600 /run/ringlpn/private/output/process.log "
        "/run/ringlpn/private/output/return-code; "
        "for p in /run/ringlpn/private /run/ringlpn/private/input "
        "/run/ringlpn/private/tmp /run/ringlpn/private/output "
        "/run/ringlpn/private/output/*; do "
        "[ -e \"$p\" ] || continue; "
        "stat -c 'RINGLPN_MODE\t%n\t%F\t%u\t%g\t%a\t%s' \"$p\" || exit 125; "
        "done; exit \"$rc\""
    )
    create = [
        podman,
        "create",
        "--name",
        name,
        "--label",
        f"io.ezpc.ringlpn.schema={SCHEMA}",
        "--label",
        f"io.ezpc.ringlpn.session={args.session_id}",
        "--label",
        f"io.ezpc.ringlpn.party={party}",
        "--userns=auto",
        "--user",
        f"{args.uid}:{args.uid}",
        "--read-only",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--pids-limit",
        str(args.pids_limit),
        "--network",
        args.network,
        "--log-driver=none",
        "--device",
        f"nvidia.com/gpu={args.gpu}",
        "--env",
        "CUDA_VISIBLE_DEVICES=0",
        "--env",
        f"RINGLPN_PARTY={party}",
        "--env",
        f"RINGLPN_PRIVATE_ROOT={PARTY_MOUNT}",
        "--env",
        f"RINGLPN_PRIVATE_INPUT_DIR={PARTY_MOUNT}/input",
        "--env",
        f"RINGLPN_PRIVATE_TMP_DIR={PARTY_MOUNT}/tmp",
        "--env",
        f"RINGLPN_PRIVATE_OUTPUT_DIR={PARTY_MOUNT}/output",
        "--mount",
        f"type=bind,src={root},dst={PARTY_MOUNT},rw=true,relabel=private,U=true",
        "--mount",
        f"type=bind,src={ledger_root},dst={LEDGER_MOUNT},rw=true,relabel=private,U=true",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,mode=700",
        args.image,
        "/bin/sh",
        "-c",
        wrapper,
        "ringlpn-party",
        *args.command,
    ]
    started = now()
    initial = {
        "schema": SCHEMA,
        "phase": "party-starting",
        "session_id": args.session_id,
        "party": party,
        "host": os.uname().nodename,
        "coordinator_uid": os.getuid(),
        "coordinator_gid": os.getgid(),
        "container_uid": args.uid,
        "gpu": {"requested_cdi_device": f"nvidia.com/gpu={args.gpu}", "cuda_visible_devices": "0"},
        "invocation_id": args.invocation_id,
        "public_parameters_sha256": args.public_parameters_sha256,
        "channel_auth_secret_sha256": channel_auth_secret_sha256,
        "channel_auth_protocol": "ringlpn-mutual-hmac-sha256-v1",
        "machine_identity_sha256": args.machine_identity_sha256,
        "runtime_identity": {
            "container_image": args.image,
            "container_binary": str(container_binary),
            "container_binary_sha256": args.container_binary_sha256,
        },
        "container_name": name,
        "private_root": str(root),
        "started_at": started,
        "volume_topology": {
            "private_bind_source": str(root),
            "private_bind_destination": PARTY_MOUNT,
            "private_access": "rw",
            "persistent_ledger_bind_source": str(ledger_root),
            "persistent_ledger_bind_destination": LEDGER_MOUNT,
            "persistent_ledger_access": "rw",
            "peer_private_mounted": False,
            "shared_rw_private_mounts": [],
        },
        "podman": {"version": info.get("version", {}), "rootless": True, "userns": "auto"},
        "mount_contract": {
            "own_private": "rw",
            "persistent_ledger": "rw",
            "peer_private": "absent",
            "rootfs": "ro",
        },
    }
    write_manifest(manifest, initial)
    run(create, capture=True)
    before = inspect_container(podman, name)
    before_evidence = container_evidence(before)
    verify_container_isolation(
        before_evidence,
        writable_mounts={PARTY_MOUNT, LEDGER_MOUNT},
        readonly_mounts=set(),
    )
    execution = run([podman, "start", "--attach", name], capture=True, check=False)
    rc = execution.returncode
    ended = now()
    after = inspect_container(podman, name)
    after_evidence = container_evidence(after)
    root_evidence = parse_mode_evidence(execution.stdout, PARTY_MOUNT)
    root_host = host_root_evidence(root)
    mapped_uids = [root_host["uid"]]
    if os.getuid() in mapped_uids:
        fail("private root remained owned by the coordinator UID; userns isolation failed")
    if after_evidence["running"]:
        fail("party container still runs after attached execution returned")
    document = {
        **initial,
        "phase": "party-exited",
        "ended_at": ended,
        "return_code": rc,
        "container_before_start": before_evidence,
        "container": after_evidence,
        "private_path_evidence": root_evidence,
        "mapped_host_uids": mapped_uids,
        "private_root_host_evidence": root_host,
        "sealed": False,
    }
    write_manifest(manifest, document)
    return rc


def seal_command(args: argparse.Namespace) -> int:
    party = int(args.party)
    root = require_absolute(args.private_root, "private root")
    manifest = require_absolute(args.manifest, "manifest")
    require_manifest_outside(root, manifest)
    document = load_manifest(manifest, party, require_sealed=False)
    if root.resolve(strict=False) != pathlib.Path(document.get("private_root", "")).resolve(strict=False):
        fail("seal root does not match the party execution manifest")
    podman, _ = podman_info()
    container_name = document.get("container", {}).get("name")
    if not container_name:
        fail("party manifest has no container identity")
    current = container_evidence(inspect_container(podman, container_name))
    if current["running"] or current["status"] not in ("exited", "stopped", "configured"):
        fail("party container is live; refusing checker handoff")
    root_host = host_root_evidence(root)
    if root_host.get("uid") not in document.get("mapped_host_uids", []):
        fail("party private root ownership changed before sealing")
    document.update(
        {
            "sealed": True,
            "sealed_at": now(),
            "container": current,
            "private_root_host_evidence": root_host,
        }
    )
    write_manifest(manifest, document)
    return 0


def labeled_containers(
    podman: str, expected_labels: dict[str, str]
) -> list[str]:
    command = [podman, "ps", "--all"]
    for key, value in expected_labels.items():
        command.extend(["--filter", f"label={key}={value}"])
    command.extend(["--format", "{{.Names}}"])
    completed = run(command, capture=True)
    return sorted({name for name in completed.stdout.splitlines() if name})


def require_derived_container_labels(
    podman: str, name: str, expected_labels: dict[str, str]
) -> None:
    if run([podman, "container", "exists", name], check=False).returncode != 0:
        return
    inspected = inspect_container(podman, name)
    labels = inspected.get("Config", {}).get("Labels", {})
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        fail(f"derived-name container has mismatched cleanup labels: {name}")


def stop_wait_remove_labeled_containers(
    podman: str, expected_labels: dict[str, str]
) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    for name in labeled_containers(podman, expected_labels):
        inspected = inspect_container(podman, name)
        labels = inspected.get("Config", {}).get("Labels", {})
        if any(labels.get(key) != value for key, value in expected_labels.items()):
            fail(f"refusing cleanup of container with mismatched labels: {name}")
        evidence = container_evidence(inspected)
        if evidence["running"]:
            run([podman, "kill", "--signal", "TERM", name], capture=True)
            run([podman, "wait", name], capture=True)
            inspected = inspect_container(podman, name)
            if container_evidence(inspected)["running"]:
                fail(f"labeled worker remained live after kill/wait: {name}")
        run([podman, "rm", name], capture=True)
        removed.append(evidence)
    survivors = labeled_containers(podman, expected_labels)
    if survivors:
        fail("labeled containers survived the second absence sweep: " + ", ".join(survivors))
    return removed


def abort_command(args: argparse.Namespace) -> int:
    party = int(args.party)
    root = require_absolute(args.private_root, "private root")
    ledger_root = require_absolute(args.ledger_root, "persistent ledger root")
    manifest = require_absolute(args.manifest, "manifest")
    expected_auth = require_sha256(
        args.channel_auth_secret_sha256, "channel authenticator digest"
    )
    require_manifest_outside(root, manifest)
    require_manifest_outside(ledger_root, manifest)
    require_separate([root, ledger_root])
    auth_transaction_present = (
        root.exists()
        or root.is_symlink()
        or channel_auth_cleanup_marker(root).exists()
        or channel_auth_cleanup_marker(root).is_symlink()
    )
    if auth_transaction_present:
        validate_channel_auth_cleanup_marker(
            root, expected_auth, args.session_id, args.invocation_id, party
        )

    existing: dict[str, Any] = {}
    if manifest.exists() and not manifest.is_symlink() and manifest.is_file():
        try:
            candidate = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            candidate = {}
        if (
            candidate.get("schema") == SCHEMA
            and str(candidate.get("session_id")) == str(args.session_id)
            and candidate.get("invocation_id") == args.invocation_id
            and candidate.get("party") == party
            and pathlib.Path(candidate.get("private_root", "")).resolve(strict=False)
            == root.resolve(strict=False)
        ):
            existing = candidate

    podman, _ = podman_info()
    expected_labels = {
        "io.ezpc.ringlpn.schema": SCHEMA,
        "io.ezpc.ringlpn.session": str(args.session_id),
        "io.ezpc.ringlpn.party": str(party),
    }
    require_derived_container_labels(
        podman, f"ringlpn-{args.session_id}-party{party}", expected_labels
    )
    removed_containers = stop_wait_remove_labeled_containers(podman, expected_labels)

    retained_log: str | None = None
    deleted_evidence: list[dict[str, Any]] = []
    if root.exists():
        run([podman, "unshare", "chown", "-R", "0:0", str(root)], capture=True)
        source_log = root / "output" / "process.log"
        if (
            source_log.exists()
            and not source_log.is_symlink()
            and source_log.is_file()
            and source_log.lstat().st_nlink == 1
        ):
            retained = manifest.parent / f"{args.session_id}.party{party}.abort.log"
            shutil.copyfile(source_log, retained)
            os.chmod(retained, PRIVATE_FILE_MODE)
            retained_log = str(retained)
    if auth_transaction_present:
        remove_channel_auth_transaction(
            root, expected_auth, args.session_id, args.invocation_id, party
        )

    if not ledger_root.exists():
        fail("persistent party ledger disappeared during abort")
    run([podman, "unshare", "chown", "-R", "0:0", str(ledger_root)], capture=True)
    harden_tree(ledger_root)
    ledger_info = ledger_root.lstat()
    if (
        not stat.S_ISDIR(ledger_info.st_mode)
        or ledger_root.is_symlink()
        or stat.S_IMODE(ledger_info.st_mode) & 0o077
        or ledger_info.st_uid != os.geteuid()
    ):
        fail("abort did not retain the persistent ledger owner-only")
    fsync_directory(ledger_root)

    document = {
        **existing,
        "schema": SCHEMA,
        "phase": "party-aborted",
        "session_id": args.session_id,
        "party": party,
        "private_root": str(root),
        "aborted_at": now(),
        "container_removed": bool(removed_containers),
        "removed_container": removed_containers[0] if removed_containers else None,
        "removed_containers": removed_containers,
        "records_staged": False,
        "records_deleted": True,
        "deleted_private_path_evidence": deleted_evidence,
        "retained_owner_only_log": retained_log,
        "retained_persistent_ledger": str(ledger_root),
        "cleanup_status": "PASS",
    }
    write_manifest(manifest, document)
    return 0


def harden_tree(root: pathlib.Path) -> list[dict[str, Any]]:
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            fail(f"symlink is forbidden in checker-stage tree: {path}")
        if stat.S_ISDIR(info.st_mode):
            os.chmod(path, PRIVATE_MODE)
        elif stat.S_ISREG(info.st_mode):
            os.chmod(path, PRIVATE_FILE_MODE)
        else:
            fail(f"unsupported object in checker-stage tree: {path}")
    return private_evidence(root)


def stage_command(args: argparse.Namespace) -> int:
    party = int(args.party)
    root = require_absolute(args.private_root, "private root")
    manifest = require_absolute(args.manifest, "manifest")
    peer_manifest = require_absolute(args.peer_manifest, "peer manifest")
    export_root = require_absolute(args.export_root, "checker export root")
    require_separate([root, export_root])
    require_manifest_outside(root, manifest)
    require_manifest_outside(root, peer_manifest)
    require_manifest_outside(export_root, manifest)
    require_manifest_outside(export_root, peer_manifest)
    own = load_manifest(manifest, party)
    peer = load_manifest(peer_manifest, 1 - party)
    p0, p1 = (own, peer) if party == 0 else (peer, own)
    verify_party_pair(p0, p1)
    if root.resolve(strict=False) != pathlib.Path(own.get("private_root", "")).resolve(strict=False):
        fail("stage root does not match the sealed party manifest")
    if export_root.exists():
        fail("checker export root must not already exist")
    podman, _ = podman_info()
    run([podman, "unshare", "chown", "-R", "0:0", str(root)], capture=True)
    source = root / "output"
    if not source.is_dir() or source.is_symlink():
        fail("sealed party output directory is unavailable")
    harden_tree(root)
    export_root.mkdir(mode=PRIVATE_MODE)
    shutil.copytree(source, export_root / "output", copy_function=shutil.copy2)
    export_evidence = harden_tree(export_root)
    own.update(
        {
            "staged_for_checker": True,
            "staged_at": now(),
            "checker_export_root": str(export_root),
            "checker_export_uid": os.getuid(),
            "checker_export_gid": os.getgid(),
            "checker_export_path_evidence": export_evidence,
            "both_parties_sealed_before_stage": True,
        }
    )
    write_manifest(manifest, own)
    return 0

def purge_party_command(args: argparse.Namespace) -> int:
    party = int(args.party)
    root = require_absolute(args.private_root, "private root")
    export_root = require_absolute(args.export_root, "checker export root")
    manifest = require_absolute(args.manifest, "manifest")
    require_separate([root, export_root])
    require_manifest_outside(root, manifest)
    require_manifest_outside(export_root, manifest)
    document = load_manifest(manifest, party)
    expected_auth = require_sha256(
        document.get("channel_auth_secret_sha256"),
        "manifest channel authenticator digest",
    )
    validate_channel_auth_cleanup_marker(
        root,
        expected_auth,
        str(document["session_id"]),
        str(document["invocation_id"]),
        party,
    )
    if document.get("staged_for_checker") is not True:
        fail("success purge requires a staged party manifest")
    if root.resolve(strict=False) != pathlib.Path(
        str(document.get("private_root", ""))
    ).resolve(strict=False):
        fail("success purge private root does not match the party manifest")
    if {path.name for path in root.iterdir()} - {
        "input",
        "tmp",
        "output",
        "channel-auth.key",
    }:
        fail("success purge private root contains unexpected entries")
    if export_root.resolve(strict=False) != pathlib.Path(
        str(document.get("checker_export_root", ""))
    ).resolve(strict=False):
        fail("success purge export root does not match the party manifest")
    podman, _ = podman_info()
    container_name = document.get("container", {}).get("name")
    if not isinstance(container_name, str) or not container_name:
        fail("success purge requires the stopped party container identity")
    inspected = inspect_container(podman, container_name)
    if container_evidence(inspected)["running"]:
        fail("refusing to purge records while the party container is live")
    labels = inspected.get("Config", {}).get("Labels", {})
    expected_labels = {
        "io.ezpc.ringlpn.schema": SCHEMA,
        "io.ezpc.ringlpn.session": str(document["session_id"]),
        "io.ezpc.ringlpn.party": str(party),
    }
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        fail("refusing to purge a container without exact party/session labels")
    deleted = []
    for identity, private_tree in (("private-root", root), ("export-root", export_root)):
        if not private_tree.is_dir() or private_tree.is_symlink():
            fail("success purge requires both bound private trees")
        run(
            [podman, "unshare", "chown", "-R", "0:0", str(private_tree)],
            capture=True,
        )
        harden_tree(private_tree)
        parent = private_tree.parent
        shutil.rmtree(private_tree)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if private_tree.exists():
            fail("success purge left a private tree behind")
        deleted.append(
            {"identity": identity, "path": str(private_tree), "status": "absent"}
        )
    remove_channel_auth_transaction(
        root,
        expected_auth,
        str(document["session_id"]),
        str(document["invocation_id"]),
        party,
    )
    channel_auth = root / "channel-auth.key"
    if channel_auth.exists() or channel_auth.is_symlink():
        fail("channel authentication secret survived private-root deletion")
    deleted.append(
        {
            "identity": "channel-auth-secret",
            "path": str(channel_auth),
            "status": "absent",
        }
    )
    run([podman, "rm", container_name], capture=True)
    if run([podman, "container", "exists", container_name], check=False).returncode == 0:
        fail("success purge left the labeled party container behind")
    deleted.append(
        {
            "identity": f"party{party}-container",
            "path": container_name,
            "status": "absent",
        }
    )
    for identity, optional_path in (
        ("party-manifest", getattr(args, "remove_manifest", None)),
        ("peer-manifest-copy", getattr(args, "peer_manifest", None)),
    ):
        if optional_path is None:
            continue
        target = require_absolute(optional_path, identity)
        for private_tree in (root, export_root):
            require_manifest_outside(private_tree, target)
        if target.is_symlink() or not target.is_file():
            fail(f"success purge requires the exact {identity} regular file")
        target.unlink()
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if target.exists():
            fail(f"success purge left {identity} behind")
        deleted.append({"identity": identity, "path": str(target), "status": "absent"})
    ledger_root = require_absolute(
        str(document.get("volume_topology", {}).get("persistent_ledger_bind_source", "")),
        "persistent ledger root",
    )
    require_separate([root, export_root, ledger_root])
    run([podman, "unshare", "chown", "-R", "0:0", str(ledger_root)], capture=True)
    harden_tree(ledger_root)
    directory_fd = os.open(ledger_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    ledger_info = ledger_root.lstat()
    if (
        not stat.S_ISDIR(ledger_info.st_mode)
        or ledger_root.is_symlink()
        or stat.S_IMODE(ledger_info.st_mode) & 0o077
        or ledger_info.st_uid != os.geteuid()
    ):
        fail("persistent ledger root was not retained owner-only")
    persistent_ledger = {
        "identity": f"party{party}-consume-once-ledger",
        "path": str(ledger_root),
        "status": "retained-owner-only-distinct-mount",
    }
    print(
        json.dumps(
            {
                "schema": "ringlpn-deletion-fragment-v1",
                "party": party,
                "machine_identity_sha256": machine_identity_sha256(),
                "deletions": deleted,
                "persistent_ledger": persistent_ledger,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def launch_command(args: argparse.Namespace) -> int:
    launcher = require_absolute(args.authenticated_launcher, "authenticated launcher")
    remote_executor = require_absolute(args.remote_executor, "remote executor")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        fail("authenticated launcher must be an executable regular file")
    if launcher.resolve() == pathlib.Path(__file__).resolve():
        fail("authenticated launcher must be external to the isolation backend")
    if not args.launcher_args:
        fail("launch-two-host requires authenticated launcher arguments after --")
    local_executor = str(pathlib.Path(__file__).resolve())
    completed = run(
        [
            str(launcher),
            "--local-executor",
            local_executor,
            "--remote-executor",
            str(remote_executor),
            *args.launcher_args,
        ],
        check=False,
    )
    return completed.returncode


def verify_party_pair(p0: dict[str, Any], p1: dict[str, Any]) -> None:
    if p0.get("session_id") != p1.get("session_id"):
        fail("party completion manifests have different session IDs")
    auth_digest = require_sha256(
        p0.get("channel_auth_secret_sha256"), "party channel authenticator digest"
    )
    if (
        p1.get("channel_auth_secret_sha256") != auth_digest
        or p0.get("channel_auth_protocol") != "ringlpn-mutual-hmac-sha256-v1"
        or p1.get("channel_auth_protocol") != "ringlpn-mutual-hmac-sha256-v1"
    ):
        fail("party completion manifests do not bind the same channel authenticator")
    identity0 = (p0.get("host"), tuple(p0.get("mapped_host_uids", [])), p0.get("container", {}).get("id"))
    identity1 = (p1.get("host"), tuple(p1.get("mapped_host_uids", [])), p1.get("container", {}).get("id"))
    if identity0 == identity1:
        fail("party OS/container identities are not distinct")
    if p0.get("host") == p1.get("host"):
        if set(p0.get("mapped_host_uids", [])) & set(p1.get("mapped_host_uids", [])):
            fail("same-host party user namespaces reuse a mapped UID")
        if p0.get("gpu", {}).get("requested_cdi_device") == p1.get("gpu", {}).get("requested_cdi_device"):
            fail("same-host parties are not pinned to distinct GPUs")
    for party in (p0, p1):
        if party.get("mount_contract") != {
            "own_private": "rw",
            "persistent_ledger": "rw",
            "peer_private": "absent",
            "rootfs": "ro",
        }:
            fail("party mount contract is incomplete")
        mounts = party.get("container_before_start", {}).get("mounts", [])
        private = [mount for mount in mounts if mount.get("destination") == PARTY_MOUNT]
        ledger = [mount for mount in mounts if mount.get("destination") == LEDGER_MOUNT]
        if len(private) != 1 or private[0].get("rw") is not True:
            fail("party did not receive exactly one private read-write mount")
        if len(ledger) != 1 or ledger[0].get("rw") is not True:
            fail("party did not receive exactly one persistent ledger mount")
        topology = party.get("volume_topology", {})
        if topology.get("private_bind_source") == topology.get(
            "persistent_ledger_bind_source"
        ):
            fail("party private and persistent ledger mounts alias")
        if any(mount.get("destination") == "/run/ringlpn/peer-private" for mount in mounts):
            fail("party container received a peer-private mount")

def verify_publication_bindings(
    args: argparse.Namespace,
    p0: dict[str, Any],
    p1: dict[str, Any],
    p0_path: pathlib.Path,
    p1_path: pathlib.Path,
    prepared_path: pathlib.Path,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    if not getattr(args, "publication", False):
        return {}
    required_names = (
        "expected_session_id",
        "expected_invocation_id",
        "expected_public_parameters_sha256",
        "expected_container_image",
        "expected_container_binary",
        "expected_container_binary_sha256",
        "expected_executor_sha256",
        "launcher_result",
    )
    if any(not isinstance(getattr(args, name, None), str) or not getattr(args, name) for name in required_names):
        fail("publication verification requires every expected session/runtime binding")
    machine0 = require_sha256(p0.get("machine_identity_sha256"), "party 0 machine identity")
    machine1 = require_sha256(p1.get("machine_identity_sha256"), "party 1 machine identity")
    if machine0 == machine1:
        fail("publication verification requires two distinct stable machine identities")
    expected_runtime = {
        "container_image": args.expected_container_image,
        "container_binary": str(require_absolute(args.expected_container_binary, "expected container binary")),
        "container_binary_sha256": require_sha256(
            args.expected_container_binary_sha256, "expected container binary digest"
        ),
    }
    expected = {
        "session_id": args.expected_session_id,
        "invocation_id": args.expected_invocation_id,
        "public_parameters_sha256": require_sha256(
            args.expected_public_parameters_sha256, "expected public-parameter digest"
        ),
    }
    for label, party in (("party 0", p0), ("party 1", p1)):
        if str(party.get("session_id")) != str(expected["session_id"]):
            fail(f"{label} manifest is not bound to the expected publication session")
        if party.get("invocation_id") != expected["invocation_id"]:
            fail(f"{label} manifest is not bound to the expected invocation")
        if party.get("public_parameters_sha256") != expected["public_parameters_sha256"]:
            fail(f"{label} manifest is not bound to the expected public parameters")
        if party.get("runtime_identity") != expected_runtime:
            fail(f"{label} manifest is not bound to the authorized runtime")
        if party.get("channel_auth_secret_sha256") != prepared.get(
            "channel_auth_secret_sha256"
        ):
            fail(f"{label} manifest is not bound to the prepared channel authenticator")
    if (
        str(prepared.get("session_id")) != str(expected["session_id"])
        or prepared.get("invocation_id") != expected["invocation_id"]
        or prepared.get("public_parameters_sha256") != expected["public_parameters_sha256"]
        or prepared.get("runtime_identity") != expected_runtime
        or prepared.get("machine_identities")
        != {"party0_sha256": machine0, "party1_sha256": machine1}
        or prepared.get("channel_auth_secret_sha256")
        != p0.get("channel_auth_secret_sha256")
    ):
        fail("PREPARED manifest is not bound to the expected publication launch")
    result_path = require_absolute(args.launcher_result, "launcher preparation")
    try:
        result_info = result_path.lstat()
        if not stat.S_ISREG(result_info.st_mode) or stat.S_IMODE(result_info.st_mode) != PRIVATE_FILE_MODE:
            fail("launcher preparation must be an owner-only regular file")
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read launcher preparation: {exc}")
    if (
        result.get("schema") != "ringlpn-authenticated-launch-preparation-v1"
        or result.get("status") != "PREPARED"
        or str(result.get("session_id")) != str(expected["session_id"])
        or result.get("invocation_id") != expected["invocation_id"]
        or result.get("public_parameters_sha256") != expected["public_parameters_sha256"]
        or result.get("runtime_identity") != expected_runtime
        or result.get("executor_sha256") != require_sha256(
            args.expected_executor_sha256, "expected executor digest"
        )
        or result.get("machine_identities")
        != {"party0_sha256": machine0, "party1_sha256": machine1}
        or result.get("checker_stage") != str(prepared_path.parent)
    ):
        fail("launcher preparation does not match the expected publication launch")
    expected_parties = {
        "party0": {"path": str(p0_path), "sha256": sha256_file(p0_path)},
        "party1": {"path": str(p1_path), "sha256": sha256_file(p1_path)},
    }
    if result.get("party_manifests") != expected_parties:
        fail("launcher preparation is not bound to the checker party manifests")
    source0 = result_path.parent / "party0-sealed.json"
    source1 = result_path.parent / "party1-sealed.json"
    source0_sha256 = sha256_file(source0)
    source1_sha256 = sha256_file(source1)
    if (
        source0_sha256 != expected_parties["party0"]["sha256"]
        or source1_sha256 != expected_parties["party1"]["sha256"]
    ):
        fail("launcher party manifests differ from their prepared checker copies")
    if result.get("launcher_party_manifests") != {
        "party0": {"path": str(source0), "sha256": source0_sha256},
        "party1": {"path": str(source1), "sha256": source1_sha256},
    }:
        fail("launcher preparation is not bound to its generated party manifests")
    if result.get("prepared_manifest") != {
        "path": str(prepared_path),
        "sha256": sha256_file(prepared_path),
    }:
        fail("launcher preparation is not bound to the exact PREPARED manifest")
    return result


def checker_command(args: argparse.Namespace) -> int:
    if not args.command:
        fail("run-checker requires a command after --")
    p0_root = require_absolute(args.p0_root, "party 0 checker-stage root")
    p1_root = require_absolute(args.p1_root, "party 1 checker-stage root")
    output_root = require_absolute(args.checker_root, "checker output root")
    manifest = require_absolute(args.manifest, "checker manifest")
    require_separate([p0_root, p1_root, output_root])
    for checker_stage_root in (p0_root, p1_root, output_root):
        require_manifest_outside(checker_stage_root, manifest)
    p0_manifest_path = require_absolute(args.p0_manifest, "party 0 manifest")
    p1_manifest_path = require_absolute(args.p1_manifest, "party 1 manifest")
    p0 = load_manifest(p0_manifest_path, 0)
    p1 = load_manifest(p1_manifest_path, 1)
    verify_party_pair(p0, p1)
    if p0.get("staged_for_checker") is not True or p1.get("staged_for_checker") is not True:
        fail("checker requires both authenticated post-exit stage records")
    prepared_path = require_absolute(args.prepared_manifest, "PREPARED manifest")
    for checker_stage_root in (p0_root, p1_root, output_root):
        require_manifest_outside(checker_stage_root, prepared_path)
    p0_record = p0_root / "key_p0.fc"
    p1_record = p1_root / "key_p1.fc"
    prepared = load_prepared_manifest(
        prepared_path,
        session_id=p0["session_id"],
        p0_record=p0_record,
        p1_record=p1_record,
        p0_manifest=p0_manifest_path,
        p1_manifest=p1_manifest_path,
    )
    verify_publication_bindings(
        args, p0, p1, p0_manifest_path, p1_manifest_path, prepared_path, prepared
    )
    if parse_utc(prepared["prepared_at"], "PREPARED timestamp") <= max(
        parse_utc(p0["sealed_at"], "party 0 seal timestamp"),
        parse_utc(p1["sealed_at"], "party 1 seal timestamp"),
    ):
        fail("PREPARED manifest predates a party seal")
    prepare_private_root(p0_root, party=False)
    prepare_private_root(p1_root, party=False)
    prepare_private_root(output_root, party=False)
    p0_stage_evidence = private_evidence(p0_root)
    p1_stage_evidence = private_evidence(p1_root)
    checker_output_before = private_evidence(output_root)
    ensure_gpu(args.gpu)
    if args.uid < 1:
        fail("checker UID must be non-root and positive")
    if args.uid in (p0.get("container_uid"), p1.get("container_uid")):
        fail("checker UID must differ from both party container UIDs")
    if args.publication:
        checker_device = f"nvidia.com/gpu={args.gpu}"
        party_devices = {
            p0.get("gpu", {}).get("requested_cdi_device"),
            p1.get("gpu", {}).get("requested_cdi_device"),
        }
        if checker_device in party_devices:
            fail("publication checker GPU must differ from both party GPUs")

    checker_started = now()
    if checker_started <= max(p0["ended_at"], p1["ended_at"]):
        fail("checker phase did not begin strictly after both party exit timestamps")
    if parse_utc(checker_started, "checker start timestamp") <= parse_utc(
        prepared["prepared_at"], "PREPARED timestamp"
    ):
        fail("checker phase did not begin strictly after durable preparation")
    podman, info = podman_info()
    name = args.container_name or f"ringlpn-{p0['session_id']}-checker"
    wrapper = (
        "umask 077; "
        "if (: > /run/ringlpn/checker/party0/.write-probe) 2>/dev/null; then "
        "rm -f /run/ringlpn/checker/party0/.write-probe; exit 125; fi; "
        "if (: > /run/ringlpn/checker/party1/.write-probe) 2>/dev/null; then "
        "rm -f /run/ringlpn/checker/party1/.write-probe; exit 125; fi; "
        "set +e; \"$@\" > /run/ringlpn/checker/output/process.log 2>&1; rc=$?; "
        "printf '%s\\n' \"$rc\" > /run/ringlpn/checker/output/return-code; "
        "chmod 600 /run/ringlpn/checker/output/process.log "
        "/run/ringlpn/checker/output/return-code; "
        "for p in /run/ringlpn/checker/party0 /run/ringlpn/checker/party0/* "
        "/run/ringlpn/checker/party1 /run/ringlpn/checker/party1/* "
        "/run/ringlpn/checker/output /run/ringlpn/checker/output/*; do "
        "[ -e \"$p\" ] || continue; "
        "stat -c 'RINGLPN_MODE\t%n\t%F\t%u\t%g\t%a\t%s' \"$p\" || exit 125; "
        "done; exit \"$rc\""
    )
    create = [
        podman,
        "create",
        "--name",
        name,
        "--label",
        f"io.ezpc.ringlpn.schema={SCHEMA}",
        "--label",
        f"io.ezpc.ringlpn.session={p0['session_id']}",
        "--label",
        "io.ezpc.ringlpn.role=checker",
        "--userns=auto",
        "--user",
        f"{args.uid}:{args.uid}",
        "--read-only",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--pids-limit",
        str(args.pids_limit),
        "--network=none",
        "--log-driver=none",
        "--device",
        f"nvidia.com/gpu={args.gpu}",
        "--env",
        "CUDA_VISIBLE_DEVICES=0",
        "--env",
        f"RINGLPN_CHECKER_P0_ROOT={CHECK_P0_MOUNT}",
        "--env",
        f"RINGLPN_CHECKER_P1_ROOT={CHECK_P1_MOUNT}",
        "--env",
        f"RINGLPN_CHECKER_OUTPUT_DIR={CHECK_OUT_MOUNT}",
        "--mount",
        f"type=bind,src={p0_root},dst={CHECK_P0_MOUNT},ro=true,relabel=private,U=true",
        "--mount",
        f"type=bind,src={p1_root},dst={CHECK_P1_MOUNT},ro=true,relabel=private,U=true",
        "--mount",
        f"type=bind,src={output_root},dst={CHECK_OUT_MOUNT},rw=true,relabel=private,U=true",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,mode=700",
        args.image,
        "/bin/sh",
        "-c",
        wrapper,
        "ringlpn-checker",
        *args.command,
    ]
    run(create, capture=True)
    before_evidence = container_evidence(inspect_container(podman, name))
    verify_container_isolation(
        before_evidence,
        writable_mounts={CHECK_OUT_MOUNT},
        readonly_mounts={CHECK_P0_MOUNT, CHECK_P1_MOUNT},
    )
    mounts = before_evidence["mounts"]
    for destination in (CHECK_P0_MOUNT, CHECK_P1_MOUNT):
        matching = [mount for mount in mounts if mount.get("destination") == destination]
        if len(matching) != 1 or matching[0].get("rw") is not False:
            fail("checker party inputs are not mounted exactly once and read-only")
    execution = run([podman, "start", "--attach", name], capture=True, check=False)
    rc = execution.returncode
    ended = now()
    checker_mode_evidence = parse_mode_evidence(execution.stdout, "/run/ringlpn/checker")
    after = container_evidence(inspect_container(podman, name))
    if after["running"]:
        fail("checker container still runs after attached execution returned")
    document = {
        "schema": SCHEMA,
        "phase": "checker-exited",
        "session_id": p0["session_id"],
        "invocation_id": prepared["invocation_id"],
        "public_parameters_sha256": prepared["public_parameters_sha256"],
        "runtime_identity": prepared["runtime_identity"],
        "machine_identities": prepared["machine_identities"],
        "publication_mode": bool(args.publication),
        "launcher_preparation": str(args.launcher_result) if args.publication else None,
        "launcher_preparation_sha256": (
            sha256_file(require_absolute(args.launcher_result, "launcher preparation"))
            if args.publication
            else None
        ),
        "started_at": checker_started,
        "ended_at": ended,
        "return_code": rc,
        "host": os.uname().nodename,
        "coordinator_uid": os.getuid(),
        "coordinator_gid": os.getgid(),
        "container_uid": args.uid,
        "gpu": {"requested_cdi_device": f"nvidia.com/gpu={args.gpu}", "cuda_visible_devices": "0"},
        "podman": {"version": info.get("version", {}), "rootless": True, "userns": "auto"},
        "party_completion": {"party0_ended_at": p0["ended_at"], "party1_ended_at": p1["ended_at"]},
        "party_manifests": {"party0": str(args.p0_manifest), "party1": str(args.p1_manifest)},
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": sha256_file(prepared_path),
        "checker_stage_roots": {"party0": str(p0_root), "party1": str(p1_root)},
        "container_before_start": before_evidence,
        "container": after,
        "party0_path_evidence_before_handoff": p0_stage_evidence,
        "party1_path_evidence_before_handoff": p1_stage_evidence,
        "checker_output_evidence_before_handoff": checker_output_before,
        "checker_container_mode_evidence": checker_mode_evidence,
        "volume_topology": {
            "party0_source": str(p0_root),
            "party0_destination": CHECK_P0_MOUNT,
            "party0_access": "ro",
            "party1_source": str(p1_root),
            "party1_destination": CHECK_P1_MOUNT,
            "party1_access": "ro",
            "checker_output_source": str(output_root),
            "checker_output_destination": CHECK_OUT_MOUNT,
            "checker_output_access": "rw",
        },
        "invariants": {
            "both_parties_exited_before_checker": True,
            "checker_had_no_live_access": True,
            "party_inputs_read_only": True,
            "digest_bound_prepared_records": True,
            "checker_network": "none",
        },
    }
    write_manifest(manifest, document)
    return rc


def verify_command(args: argparse.Namespace) -> int:
    p0_path = require_absolute(args.p0_manifest, "party 0 manifest")
    p1_path = require_absolute(args.p1_manifest, "party 1 manifest")
    p0 = load_manifest(p0_path, 0)
    p1 = load_manifest(p1_path, 1)
    verify_party_pair(p0, p1)
    checker_path = require_absolute(args.checker_manifest, "checker manifest")
    try:
        checker_info = checker_path.lstat()
        if not stat.S_ISREG(checker_info.st_mode) or stat.S_IMODE(checker_info.st_mode) & 0o077:
            fail("checker manifest must be an owner-only regular file")
        checker = json.loads(checker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read checker manifest: {exc}")
    if checker.get("schema") != SCHEMA or checker.get("phase") != "checker-exited":
        fail("checker manifest has the wrong schema or phase")
    if checker.get("session_id") != p0.get("session_id") or checker.get("return_code") != 0:
        fail("checker manifest does not record successful completion for this session")
    if checker.get("started_at", "") <= max(p0["ended_at"], p1["ended_at"]):
        fail("checker started before both parties exited")
    roots = checker.get("checker_stage_roots", {})
    p0_record = pathlib.Path(str(roots.get("party0", ""))) / "key_p0.fc"
    p1_record = pathlib.Path(str(roots.get("party1", ""))) / "key_p1.fc"
    prepared_path = require_absolute(args.prepared_manifest, "PREPARED manifest")
    prepared = load_prepared_manifest(
        prepared_path,
        session_id=p0["session_id"],
        p0_record=p0_record,
        p1_record=p1_record,
        p0_manifest=p0_path,
        p1_manifest=p1_path,
    )
    verify_publication_bindings(args, p0, p1, p0_path, p1_path, prepared_path, prepared)
    if args.publication:
        if checker.get("publication_mode") is not True:
            fail("checker manifest lacks the publication boundary marker")
        if (
            checker.get("invocation_id") != prepared["invocation_id"]
            or checker.get("public_parameters_sha256") != prepared["public_parameters_sha256"]
            or checker.get("runtime_identity") != prepared["runtime_identity"]
            or checker.get("machine_identities") != prepared["machine_identities"]
            or checker.get("launcher_preparation")
            != str(require_absolute(args.launcher_result, "launcher preparation"))
            or checker.get("launcher_preparation_sha256")
            != sha256_file(require_absolute(args.launcher_result, "launcher preparation"))
        ):
            fail("checker manifest is not bound to the exact publication preparation")
        if checker.get("container_uid") in (
            p0.get("container_uid"),
            p1.get("container_uid"),
        ):
            fail("publication checker manifest reuses a party container UID")
        checker_device = checker.get("gpu", {}).get("requested_cdi_device")
        if checker_device in {
            p0.get("gpu", {}).get("requested_cdi_device"),
            p1.get("gpu", {}).get("requested_cdi_device"),
        }:
            fail("publication checker manifest reuses a party GPU")
    if checker.get("prepared_manifest") != str(prepared_path) or checker.get(
        "prepared_manifest_sha256"
    ) != sha256_file(prepared_path):
        fail("checker manifest is not bound to the durable PREPARED manifest")
    if parse_utc(checker.get("started_at"), "checker start timestamp") <= parse_utc(
        prepared["prepared_at"], "PREPARED timestamp"
    ):
        fail("checker manifest predates durable preparation")
    invariants = checker.get("invariants", {})
    required = {
        "both_parties_exited_before_checker": True,
        "checker_had_no_live_access": True,
        "party_inputs_read_only": True,
        "digest_bound_prepared_records": True,
        "checker_network": "none",
    }
    if invariants != required:
        fail("checker invariant evidence is incomplete")
    print("peer-private: manifest isolation checks pass")
    return 0

def purge_checker_records_command(args: argparse.Namespace) -> int:
    p0_root = require_absolute(args.p0_root, "party 0 checker-stage root")
    p1_root = require_absolute(args.p1_root, "party 1 checker-stage root")
    checker_root = require_absolute(args.checker_root, "checker output root")
    retained_log = require_absolute(args.retained_log, "retained checker log")
    p0_path = require_absolute(args.p0_manifest, "party 0 manifest")
    p1_path = require_absolute(args.p1_manifest, "party 1 manifest")
    prepared_path = require_absolute(args.prepared_manifest, "PREPARED manifest")
    checker_path = require_absolute(args.checker_manifest, "checker manifest")
    require_separate([p0_root, p1_root, checker_root])
    for root in (p0_root, p1_root, checker_root):
        require_manifest_outside(root, retained_log)
    p0 = load_manifest(p0_path, 0)
    p1 = load_manifest(p1_path, 1)
    verify_party_pair(p0, p1)
    p0_record = p0_root / "key_p0.fc"
    p1_record = p1_root / "key_p1.fc"
    prepared = load_prepared_manifest(
        prepared_path,
        session_id=p0["session_id"],
        p0_record=p0_record,
        p1_record=p1_record,
        p0_manifest=p0_path,
        p1_manifest=p1_path,
    )
    try:
        checker = json.loads(checker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read checker manifest for finalization: {exc}")
    if (
        checker.get("schema") != SCHEMA
        or checker.get("phase") != "checker-exited"
        or checker.get("return_code") != 0
        or checker.get("session_id") != prepared["session_id"]
        or checker.get("prepared_manifest") != str(prepared_path)
        or checker.get("prepared_manifest_sha256") != sha256_file(prepared_path)
        or checker.get("volume_topology", {}).get("checker_output_source")
        != str(checker_root)
    ):
        fail("finalization requires the successful bound post-exit checker")
    if retained_log.exists():
        fail("retained checker log must be fresh")
    process_log = checker_root / "process.log"
    return_code = checker_root / "return-code"
    if (
        not checker_root.is_dir()
        or checker_root.is_symlink()
        or not process_log.is_file()
        or process_log.is_symlink()
        or return_code.read_text(encoding="utf-8").strip() != "0"
    ):
        fail("checker output root does not contain the exact successful raw outputs")
    if {entry.name for entry in checker_root.iterdir()} != {"process.log", "return-code"}:
        fail("checker output root contains an unexpected duplicate or private output")
    shutil.copyfile(process_log, retained_log)
    os.chmod(retained_log, PRIVATE_FILE_MODE)
    with retained_log.open("rb+") as stream:
        os.fsync(stream.fileno())
    directory_fd = os.open(retained_log.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    podman, _ = podman_info()
    checker_name = checker.get("container", {}).get("name")
    if not isinstance(checker_name, str) or not checker_name:
        fail("checker manifest lacks the stopped checker container identity")
    inspected = inspect_container(podman, checker_name)
    if container_evidence(inspected)["running"]:
        fail("refusing finalization while checker container is live")
    labels = inspected.get("Config", {}).get("Labels", {})
    if (
        labels.get("io.ezpc.ringlpn.schema") != SCHEMA
        or labels.get("io.ezpc.ringlpn.session") != str(prepared["session_id"])
        or labels.get("io.ezpc.ringlpn.role") != "checker"
    ):
        fail("refusing to remove checker container with mismatched labels")
    deletions = []
    for identity, root, record in (
        ("checker-stage-party0-record", p0_root, p0_record),
        ("checker-stage-party1-record", p1_root, p1_record),
    ):
        if not root.is_dir() or root.is_symlink():
            fail("checker-stage root is unavailable during finalization")
        run([podman, "unshare", "chown", "-R", "0:0", str(root)], capture=True)
        if not record.is_file() or record.is_symlink():
            fail("finalization requires the exact prepared regular record")
        record.unlink()
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if record.exists() or {entry.name for entry in root.iterdir()} != {
            "isolation-manifest.json"
        }:
            fail("checker-stage root retained a raw or duplicate output")
        deletions.append({"identity": identity, "path": str(record), "status": "absent"})
    shutil.rmtree(checker_root)
    directory_fd = os.open(checker_root.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if checker_root.exists():
        fail("raw checker output root survived finalization")
    deletions.append(
        {"identity": "checker-output-root", "path": str(checker_root), "status": "absent"}
    )
    run([podman, "rm", checker_name], capture=True)
    if run([podman, "container", "exists", checker_name], check=False).returncode == 0:
        fail("labeled checker container survived finalization")
    deletions.append(
        {"identity": "checker-container", "path": checker_name, "status": "absent"}
    )
    print(
        json.dumps(
            {
                "schema": "ringlpn-deletion-fragment-v1",
                "machine_identity_sha256": machine_identity_sha256(),

                "retained_log": {
                    "path": str(retained_log),
                    "sha256": sha256_file(retained_log),
                },
                "deletions": deletions,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0
def abort_checker_command(args: argparse.Namespace) -> int:
    checker_root = require_absolute(args.checker_root, "checker output root")
    checker_manifest = require_absolute(args.checker_manifest, "checker manifest")
    require_manifest_outside(checker_root, checker_manifest)
    podman, _ = podman_info()
    expected_labels = {
        "io.ezpc.ringlpn.schema": SCHEMA,
        "io.ezpc.ringlpn.session": str(args.session_id),
        "io.ezpc.ringlpn.role": "checker",
    }
    require_derived_container_labels(
        podman, f"ringlpn-{args.session_id}-checker", expected_labels
    )
    stop_wait_remove_labeled_containers(podman, expected_labels)
    if checker_root.exists() or checker_root.is_symlink():
        if not checker_root.is_dir() or checker_root.is_symlink():
            fail("checker abort root is not a real directory")
        run([podman, "unshare", "chown", "-R", "0:0", str(checker_root)], capture=True)
        harden_tree(checker_root)
        shutil.rmtree(checker_root)
        fsync_directory(checker_root.parent)
    if checker_root.exists() or checker_root.is_symlink():
        fail("checker cleanup left its exact raw-output root behind")
    survivors = labeled_containers(podman, expected_labels)
    if survivors:
        fail("checker cleanup second absence sweep failed: " + ", ".join(survivors))
    return 0


def verify_session_containers_absent_command(args: argparse.Namespace) -> int:
    podman, _ = podman_info()
    completed = run(
        [
            podman,
            "ps",
            "--all",
            "--filter",
            f"label=io.ezpc.ringlpn.schema={SCHEMA}",
            "--filter",
            f"label=io.ezpc.ringlpn.session={args.session_id}",
            "--format",
            "{{.Names}}",
        ],
        capture=True,
    )
    survivors = [name for name in completed.stdout.splitlines() if name]
    if survivors:
        fail("labeled session containers survived cleanup: " + ", ".join(survivors))
    return 0



def add_container_options(parser: argparse.ArgumentParser, *, default_uid: int) -> None:
    parser.add_argument("--image", required=True, help="immutable container image reference or digest")
    parser.add_argument("--gpu", required=True, help="NVIDIA CDI GPU index or UUID")
    parser.add_argument("--uid", type=int, default=default_uid, help="distinct numeric identity inside the user namespace")
    parser.add_argument("--pids-limit", type=int, default=4096)
    parser.add_argument("--container-name")
    parser.add_argument("--manifest", required=True)

def add_publication_binding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--expected-session-id")
    parser.add_argument("--expected-invocation-id")
    parser.add_argument("--expected-public-parameters-sha256")
    parser.add_argument("--expected-container-image")
    parser.add_argument("--expected-container-binary")
    parser.add_argument("--expected-container-binary-sha256")
    parser.add_argument("--expected-executor-sha256")
    parser.add_argument("--launcher-result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Peer-private backend for run_two_host_authenticated.sh. "
            "No unauthenticated/local-process fallback is provided."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    provision_auth = subparsers.add_parser(
        "provision-channel-auth",
        help="read one 32-byte channel authenticator from stdin into a fresh private root",
    )
    provision_auth.add_argument("--private-root", required=True)
    provision_auth.add_argument("--channel-auth-secret-sha256", required=True)
    provision_auth.add_argument("--party", required=True, choices=("0", "1"))
    provision_auth.add_argument("--session-id", required=True)
    provision_auth.add_argument("--invocation-id", required=True)
    abort_provisioned_auth = subparsers.add_parser(
        "abort-provisioned-channel-auth",
        help="remove a not-yet-started provisioned channel authenticator root",
    )
    abort_provisioned_auth.add_argument("--private-root", required=True)
    abort_provisioned_auth.add_argument("--channel-auth-secret-sha256", required=True)
    abort_provisioned_auth.add_argument("--party", required=True, choices=("0", "1"))
    abort_provisioned_auth.add_argument("--session-id", required=True)
    abort_provisioned_auth.add_argument("--invocation-id", required=True)


    party = subparsers.add_parser("run-party", help="run one live party in a private user namespace")
    party.add_argument("--party", required=True, choices=("0", "1"))
    party.add_argument("--session-id", required=True)
    party.add_argument("--private-root", required=True)
    party.add_argument("--ledger-root", required=True)
    party.add_argument("--invocation-id", required=True)
    party.add_argument("--public-parameters-sha256", required=True)
    party.add_argument("--machine-identity-sha256", required=True)
    party.add_argument("--container-binary", required=True)
    party.add_argument("--container-binary-sha256", required=True)
    party.add_argument("--channel-auth-secret-sha256", required=True)
    party.add_argument("--network", choices=("host", "slirp4netns", "pasta"), default="host")
    add_container_options(party, default_uid=10001)
    party.add_argument("command", nargs=argparse.REMAINDER)

    seal = subparsers.add_parser("seal-party", help="seal a successful stopped party for authenticated transfer")
    seal.add_argument("--party", required=True, choices=("0", "1"))
    seal.add_argument("--private-root", required=True)
    seal.add_argument("--manifest", required=True)

    abort = subparsers.add_parser("abort-party", help="terminate one labeled session container and delete private records")
    abort.add_argument("--party", required=True, choices=("0", "1"))
    abort.add_argument("--session-id", required=True)
    abort.add_argument("--invocation-id", required=True)
    abort.add_argument("--private-root", required=True)
    abort.add_argument("--ledger-root", required=True)
    abort.add_argument("--channel-auth-secret-sha256", required=True)
    abort.add_argument("--manifest", required=True)

    stage = subparsers.add_parser("stage-party", help="expose one sealed output only after both parties exited")
    stage.add_argument("--party", required=True, choices=("0", "1"))
    stage.add_argument("--private-root", required=True)
    stage.add_argument("--manifest", required=True)
    stage.add_argument("--peer-manifest", required=True)
    stage.add_argument("--export-root", required=True)

    purge_party = subparsers.add_parser(
        "purge-party", help="delete one successfully staged party's private trees"
    )
    purge_party.add_argument("--party", required=True, choices=("0", "1"))
    purge_party.add_argument("--private-root", required=True)
    purge_party.add_argument("--export-root", required=True)
    purge_party.add_argument("--manifest", required=True)
    purge_party.add_argument("--remove-manifest")
    purge_party.add_argument("--peer-manifest")

    checker = subparsers.add_parser("run-checker", help="start a networkless checker only from two exit manifests")
    checker.add_argument("--p0-root", required=True)
    checker.add_argument("--p1-root", required=True)
    checker.add_argument("--p0-manifest", required=True)
    checker.add_argument("--p1-manifest", required=True)
    checker.add_argument("--prepared-manifest", required=True)
    checker.add_argument("--checker-root", required=True)
    add_container_options(checker, default_uid=10003)
    add_publication_binding_options(checker)
    checker.add_argument("command", nargs=argparse.REMAINDER)

    verify = subparsers.add_parser("verify-manifest", help="check recorded isolation and phase-order evidence")
    verify.add_argument("--p0-manifest", required=True)
    verify.add_argument("--p1-manifest", required=True)
    verify.add_argument("--checker-manifest", required=True)
    verify.add_argument("--prepared-manifest", required=True)
    add_publication_binding_options(verify)

    purge_checker = subparsers.add_parser(
        "purge-checker-records",
        help="remove checker raw outputs, prepared records, and the checker container",
    )
    purge_checker.add_argument("--p0-root", required=True)
    purge_checker.add_argument("--p1-root", required=True)
    purge_checker.add_argument("--p0-manifest", required=True)
    purge_checker.add_argument("--p1-manifest", required=True)
    purge_checker.add_argument("--checker-manifest", required=True)
    purge_checker.add_argument("--prepared-manifest", required=True)
    purge_checker.add_argument("--checker-root", required=True)
    purge_checker.add_argument("--retained-log", required=True)

    abort_checker = subparsers.add_parser(
        "abort-checker", help="remove one labeled checker container and raw output root"
    )
    abort_checker.add_argument("--session-id", required=True)
    abort_checker.add_argument("--checker-manifest", required=True)
    abort_checker.add_argument("--checker-root", required=True)

    verify_absent = subparsers.add_parser(
        "verify-session-containers-absent",
        help="reject any labeled container retained for a finalized session",
    )
    verify_absent.add_argument("--session-id", required=True)

    launch = subparsers.add_parser("launch-two-host", help="invoke the required external authenticated launcher")
    launch.add_argument("--authenticated-launcher", required=True)
    launch.add_argument("--remote-executor", required=True)
    launch.add_argument("launcher_args", nargs=argparse.REMAINDER)

    machine_identity = subparsers.add_parser(
        "machine-identity", help="emit a one-way stable OS machine identity"
    )

    runtime_identity = subparsers.add_parser(
        "runtime-identity", help="measure an immutable image and in-image binary"
    )
    runtime_identity.add_argument("--image", required=True)
    runtime_identity.add_argument("--binary", required=True)

    capability_preflight = subparsers.add_parser(
        "capability-preflight",
        help="prove native rootless Podman, userns, immutable runtime, and GPU readiness",
    )
    capability_preflight.add_argument("--image", required=True)
    capability_preflight.add_argument("--binary", required=True)
    capability_preflight.add_argument("--binary-sha256", required=True)
    capability_preflight.add_argument("--gpu", required=True)

    args = parser.parse_args()
    if hasattr(args, "command") and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if hasattr(args, "launcher_args") and args.launcher_args and args.launcher_args[0] == "--":
        args.launcher_args = args.launcher_args[1:]
    return args


def main() -> int:
    args = parse_args()
    if args.operation == "provision-channel-auth":
        return provision_channel_auth_command(args)
    if args.operation == "abort-provisioned-channel-auth":
        return abort_provisioned_channel_auth_command(args)
    if args.operation == "run-party":
        return party_command(args)
    if args.operation == "seal-party":
        return seal_command(args)
    if args.operation == "abort-party":
        return abort_command(args)
    if args.operation == "stage-party":
        return stage_command(args)
    if args.operation == "purge-party":
        return purge_party_command(args)
    if args.operation == "run-checker":
        return checker_command(args)
    if args.operation == "verify-manifest":
        return verify_command(args)
    if args.operation == "purge-checker-records":
        return purge_checker_records_command(args)
    if args.operation == "abort-checker":
        return abort_checker_command(args)
    if args.operation == "verify-session-containers-absent":
        return verify_session_containers_absent_command(args)
    if args.operation == "machine-identity":
        return machine_identity_command(args)
    if args.operation == "runtime-identity":
        return runtime_identity_command(args)
    if args.operation == "capability-preflight":
        return capability_preflight_command(args)
    if args.operation == "launch-two-host":
        return launch_command(args)
    fail("unknown operation")


if __name__ == "__main__":
    raise SystemExit(main())
