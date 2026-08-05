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


def load_commit_manifest(
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
            fail("COMMITTED manifest must be an owner-only regular file")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read COMMITTED manifest: {exc}")
    exact_keys = {
        "schema",
        "state",
        "session_id",
        "channel",
        "base_port",
        "reversed_port",
        "public_parameters_sha256",
        "p0_exit_code",
        "p1_exit_code",
        "p0_record",
        "p1_record",
        "p0_isolation_manifest",
        "p1_isolation_manifest",
        "committed_at",
    }
    if set(document) != exact_keys:
        fail("COMMITTED manifest does not have the exact required fields")
    if (
        document.get("schema") != "ringlpn-two-host-commit-v1"
        or document.get("state") != "COMMITTED"
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
        fail("COMMITTED manifest boundary, session, port, or exit state is invalid")
    public_digest = document.get("public_parameters_sha256")
    if not isinstance(public_digest, str) or len(public_digest) != 64 or any(
        character not in "0123456789abcdef" for character in public_digest
    ):
        fail("COMMITTED manifest public-parameter digest is malformed")
    expected = {
        "p0_record": ("party0/key_p0.fc", p0_record),
        "p1_record": ("party1/key_p1.fc", p1_record),
        "p0_isolation_manifest": ("party0/isolation-manifest.json", p0_manifest),
        "p1_isolation_manifest": ("party1/isolation-manifest.json", p1_manifest),
    }
    for key, (relative_path, expected_path) in expected.items():
        entry = document.get(key)
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            fail(f"COMMITTED manifest lacks exact {key} fields")
        if entry.get("path") != relative_path:
            fail(f"COMMITTED manifest {key} path is not canonical")
        if (path.parent / relative_path).resolve(strict=False) != expected_path.resolve(strict=False):
            fail(f"COMMITTED manifest {key} path does not match checker input")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            fail(f"COMMITTED manifest {key} digest is malformed")
        if sha256_file(expected_path) != digest:
            fail(f"COMMITTED manifest {key} digest mismatch")
    committed_at = document.get("committed_at")
    if not isinstance(committed_at, str) or not committed_at:
        fail("COMMITTED manifest lacks a durable commit timestamp")
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
    manifest = require_absolute(args.manifest, "manifest")
    require_manifest_outside(root, manifest)
    ensure_gpu(args.gpu)
    if args.uid < 1:
        fail("container UID must be non-root and positive")
    prepare_private_root(root, party=True)
    podman, info = podman_info()
    name = args.container_name or f"ringlpn-{args.session_id}-party{party}"
    if not name.replace("-", "").replace("_", "").isalnum():
        fail("container name contains unsupported characters")

    wrapper = (
        "umask 077; "
        "test ! -e /run/ringlpn/peer-private || exit 125; "
        "test \"$(stat -c %a /run/ringlpn/private)\" = 700 || exit 125; "
        "set +e; \"$@\" > /run/ringlpn/private/output/process.log 2>&1; rc=$?; "
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
        "container_name": name,
        "private_root": str(root),
        "started_at": started,
        "volume_topology": {
            "private_bind_source": str(root),
            "private_bind_destination": PARTY_MOUNT,
            "private_access": "rw",
            "peer_private_mounted": False,
            "shared_rw_private_mounts": [],
        },
        "podman": {"version": info.get("version", {}), "rootless": True, "userns": "auto"},
        "mount_contract": {"own_private": "rw", "peer_private": "absent", "rootfs": "ro"},
    }
    write_manifest(manifest, initial)
    run(create, capture=True)
    before = inspect_container(podman, name)
    before_evidence = container_evidence(before)
    verify_container_isolation(before_evidence, writable_mounts={PARTY_MOUNT}, readonly_mounts=set())
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


def abort_command(args: argparse.Namespace) -> int:
    party = int(args.party)
    root = require_absolute(args.private_root, "private root")
    manifest = require_absolute(args.manifest, "manifest")
    if not manifest.exists():
        fail("abort requires the owner-only session manifest; refusing an unbound private root")
    require_manifest_outside(root, manifest)
    podman, _ = podman_info()
    name = f"ringlpn-{args.session_id}-party{party}"
    existing: dict[str, Any] = {}
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot validate abort manifest: {exc}")
        if (
            existing.get("schema") != SCHEMA
            or str(existing.get("session_id")) != str(args.session_id)
            or existing.get("party") != party
            or pathlib.Path(existing.get("private_root", "")).resolve(strict=False) != root.resolve(strict=False)
        ):
            fail("abort manifest does not match requested session, party, and private root")
        name = existing.get("container", {}).get("name") or existing.get("container_name") or name
    exists = run([podman, "container", "exists", name], check=False).returncode == 0
    removed_container: dict[str, Any] | None = None
    if exists:
        inspected = inspect_container(podman, name)
        labels = inspected.get("Config", {}).get("Labels", {})
        expected = {
            "io.ezpc.ringlpn.schema": SCHEMA,
            "io.ezpc.ringlpn.session": str(args.session_id),
            "io.ezpc.ringlpn.party": str(party),
        }
        if any(labels.get(key) != value for key, value in expected.items()):
            fail("refusing to abort a container without exact Ring-LPN session labels")
        removed_container = container_evidence(inspected)
        run([podman, "rm", "--force", name], capture=True)
    retained_log: str | None = None
    deleted_evidence: list[dict[str, Any]] = []
    if root.exists():
        run([podman, "unshare", "chown", "-R", "0:0", str(root)], capture=True)
        deleted_evidence = harden_tree(root)
        source_log = root / "output" / "process.log"
        if source_log.exists():
            if source_log.is_symlink() or not source_log.is_file():
                fail("refusing to retain a non-regular party process log")
            retained = manifest.parent / f"{args.session_id}.party{party}.abort.log"
            shutil.copyfile(source_log, retained)
            os.chmod(retained, PRIVATE_FILE_MODE)
            retained_log = str(retained)
        shutil.rmtree(root)
    document = {
        **existing,
        "schema": SCHEMA,
        "phase": "party-aborted",
        "session_id": args.session_id,
        "party": party,
        "private_root": str(root),
        "aborted_at": now(),
        "container_removed": exists,
        "removed_container": removed_container,
        "records_staged": False,
        "records_deleted": True,
        "deleted_private_path_evidence": deleted_evidence,
        "retained_owner_only_log": retained_log,
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
        if party.get("mount_contract") != {"own_private": "rw", "peer_private": "absent", "rootfs": "ro"}:
            fail("party mount contract is incomplete")
        mounts = party.get("container_before_start", {}).get("mounts", [])
        private = [mount for mount in mounts if mount.get("destination") == PARTY_MOUNT]
        if len(private) != 1 or private[0].get("rw") is not True:
            fail("party did not receive exactly one private read-write mount")
        if any(mount.get("destination") == "/run/ringlpn/peer-private" for mount in mounts):
            fail("party container received a peer-private mount")


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
    commit_path = require_absolute(args.commit_manifest, "COMMITTED manifest")
    for checker_stage_root in (p0_root, p1_root, output_root):
        require_manifest_outside(checker_stage_root, commit_path)
    p0_record = p0_root / "key_p0.fc"
    p1_record = p1_root / "key_p1.fc"
    commit = load_commit_manifest(
        commit_path,
        session_id=p0["session_id"],
        p0_record=p0_record,
        p1_record=p1_record,
        p0_manifest=p0_manifest_path,
        p1_manifest=p1_manifest_path,
    )
    if parse_utc(commit["committed_at"], "commit timestamp") <= max(
        parse_utc(p0["sealed_at"], "party 0 seal timestamp"),
        parse_utc(p1["sealed_at"], "party 1 seal timestamp"),
    ):
        fail("COMMITTED manifest predates a party seal")
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

    checker_started = now()
    if checker_started <= max(p0["ended_at"], p1["ended_at"]):
        fail("checker phase did not begin strictly after both party exit timestamps")
    if parse_utc(checker_started, "checker start timestamp") <= parse_utc(
        commit["committed_at"], "commit timestamp"
    ):
        fail("checker phase did not begin strictly after durable COMMITTED publication")
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
        "commit_manifest": str(commit_path),
        "commit_manifest_sha256": sha256_file(commit_path),
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
            "digest_bound_committed_records": True,
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
    commit_path = require_absolute(args.commit_manifest, "COMMITTED manifest")
    commit = load_commit_manifest(
        commit_path,
        session_id=p0["session_id"],
        p0_record=p0_record,
        p1_record=p1_record,
        p0_manifest=p0_path,
        p1_manifest=p1_path,
    )
    if checker.get("commit_manifest") != str(commit_path) or checker.get(
        "commit_manifest_sha256"
    ) != sha256_file(commit_path):
        fail("checker manifest is not bound to the durable COMMITTED manifest")
    if parse_utc(checker.get("started_at"), "checker start timestamp") <= parse_utc(
        commit["committed_at"], "commit timestamp"
    ):
        fail("checker manifest predates durable COMMITTED publication")
    invariants = checker.get("invariants", {})
    required = {
        "both_parties_exited_before_checker": True,
        "checker_had_no_live_access": True,
        "party_inputs_read_only": True,
        "digest_bound_committed_records": True,
        "checker_network": "none",
    }
    if invariants != required:
        fail("checker invariant evidence is incomplete")
    print("peer-private: manifest isolation checks pass")
    return 0


def add_container_options(parser: argparse.ArgumentParser, *, default_uid: int) -> None:
    parser.add_argument("--image", required=True, help="immutable container image reference or digest")
    parser.add_argument("--gpu", required=True, help="NVIDIA CDI GPU index or UUID")
    parser.add_argument("--uid", type=int, default=default_uid, help="distinct numeric identity inside the user namespace")
    parser.add_argument("--pids-limit", type=int, default=4096)
    parser.add_argument("--container-name")
    parser.add_argument("--manifest", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Peer-private backend for run_two_host_authenticated.sh. "
            "No unauthenticated/local-process fallback is provided."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    party = subparsers.add_parser("run-party", help="run one live party in a private user namespace")
    party.add_argument("--party", required=True, choices=("0", "1"))
    party.add_argument("--session-id", required=True)
    party.add_argument("--private-root", required=True)
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
    abort.add_argument("--private-root", required=True)
    abort.add_argument("--manifest", required=True)

    stage = subparsers.add_parser("stage-party", help="expose one sealed output only after both parties exited")
    stage.add_argument("--party", required=True, choices=("0", "1"))
    stage.add_argument("--private-root", required=True)
    stage.add_argument("--manifest", required=True)
    stage.add_argument("--peer-manifest", required=True)
    stage.add_argument("--export-root", required=True)

    checker = subparsers.add_parser("run-checker", help="start a networkless checker only from two exit manifests")
    checker.add_argument("--p0-root", required=True)
    checker.add_argument("--p1-root", required=True)
    checker.add_argument("--p0-manifest", required=True)
    checker.add_argument("--p1-manifest", required=True)
    checker.add_argument("--commit-manifest", required=True)
    checker.add_argument("--checker-root", required=True)
    add_container_options(checker, default_uid=10003)
    checker.add_argument("command", nargs=argparse.REMAINDER)

    verify = subparsers.add_parser("verify-manifest", help="check recorded isolation and phase-order evidence")
    verify.add_argument("--p0-manifest", required=True)
    verify.add_argument("--p1-manifest", required=True)
    verify.add_argument("--checker-manifest", required=True)
    verify.add_argument("--commit-manifest", required=True)

    launch = subparsers.add_parser("launch-two-host", help="invoke the required external authenticated launcher")
    launch.add_argument("--authenticated-launcher", required=True)
    launch.add_argument("--remote-executor", required=True)
    launch.add_argument("launcher_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if hasattr(args, "command") and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if hasattr(args, "launcher_args") and args.launcher_args and args.launcher_args[0] == "--":
        args.launcher_args = args.launcher_args[1:]
    return args


def main() -> int:
    args = parse_args()
    if args.operation == "run-party":
        return party_command(args)
    if args.operation == "seal-party":
        return seal_command(args)
    if args.operation == "abort-party":
        return abort_command(args)
    if args.operation == "stage-party":
        return stage_command(args)
    if args.operation == "run-checker":
        return checker_command(args)
    if args.operation == "verify-manifest":
        return verify_command(args)
    if args.operation == "launch-two-host":
        return launch_command(args)
    fail("unknown operation")


if __name__ == "__main__":
    raise SystemExit(main())
