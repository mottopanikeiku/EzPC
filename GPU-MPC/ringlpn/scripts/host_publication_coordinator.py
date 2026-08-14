#!/usr/bin/env python3
"""Host-only fail-closed coordinator for authenticated two-host publication."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any
from retained_public_evidence import scan_private_artifacts

GUIDANCE = (
    "two-host-publication must run on each host's native rootless Podman; "
    "invoke ./GPU-MPC/ringlpn/scripts/reproduce_publication.sh "
    "two-host-publication directly on the coordinator host (never via docker run)."
)
SCHEMA = "ringlpn-publication-environment/v4"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"[ringlpn-reproduce] FAIL: {message}")


def run(argv: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"command failed before publication: {argv[0]}: {error}")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode())


def load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"invalid {label}: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def atomic_json(path: pathlib.Path, value: dict[str, Any], mode: int = 0o600) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return payload


def parse_args(argv: list[str]) -> tuple[str, str, list[str], dict[str, str]]:
    checker_uid = ""
    checker_gpu = ""
    try:
        separator = argv.index("--")
    except ValueError:
        fail("host coordinator options must be separated from launcher options by --")
    prefix, launcher = argv[:separator], argv[separator + 1 :]
    if len(prefix) != 4:
        fail("required host options are --checker-container-uid N --checker-gpu CDI")
    for index in range(0, len(prefix), 2):
        if prefix[index] == "--checker-container-uid" and not checker_uid:
            checker_uid = prefix[index + 1]
        elif prefix[index] == "--checker-gpu" and not checker_gpu:
            checker_gpu = prefix[index + 1]
        else:
            fail(f"unknown or duplicate host coordinator option: {prefix[index]}")
    if not re.fullmatch(r"[1-9][0-9]*", checker_uid):
        fail("checker container UID must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", checker_gpu):
        fail("checker GPU selector is malformed")
    if not launcher or len(launcher) % 2:
        fail("authenticated launcher arguments must be key/value pairs")
    forbidden = {"local-executor", "container-image", "container-binary", "container-binary-sha256",
                 "checker-container-uid", "checker-gpu"}
    values: dict[str, str] = {}
    for index in range(0, len(launcher), 2):
        option, value = launcher[index:index + 2]
        if not option.startswith("--"):
            fail(f"launcher argument is not an option: {option}")
        name = option[2:]
        if name in forbidden:
            fail(f"host coordinator owns launcher option: {option}")
        if name in values:
            fail(f"duplicate launcher option: {option}")
        values[name] = value
    required = {
        "peer", "identity", "known-hosts", "remote-executor", "local-private-root",
        "remote-private-root", "local-party-ledger-root", "remote-party-ledger-root",
        "local-party-manifest", "remote-party-manifest", "remote-peer-manifest",
        "local-export-root", "remote-export-root", "checker-stage",
        "output-dir", "local-container-uid", "remote-container-uid", "local-gpu", "remote-gpu",
        "session-id", "invocation-id", "ledger-root", "base-port", "qbits", "bw", "rows",
        "inner", "cols", "ole-n", "ole-c", "ole-t", "noise",
    }
    missing = sorted(required - values.keys())
    if missing:
        fail("missing publication launcher options: " + ", ".join("--" + name for name in missing))
    allowed = required | {"timeout", "fault-injection"}
    unexpected = sorted(values.keys() - allowed)
    if unexpected:
        fail("unsupported publication launcher options: " + ", ".join("--" + name for name in unexpected))
    return checker_uid, checker_gpu, launcher, values




def verify_authorized_worktree(repo: pathlib.Path, commit: str) -> None:
    scope = "GPU-MPC/ringlpn"
    visibility = run(
        ["git", "-C", str(repo), "ls-files", "-v", "-z"],
        capture=True,
    ).stdout
    hidden = []
    for record in visibility.split("\0"):
        if not record:
            continue
        marker, separator, name = record.partition(" ")
        if not separator or marker == "S" or marker.islower():
            hidden.append(name or record)
    if hidden:
        fail(
            "tracked paths use assume-unchanged or skip-worktree: "
            + " | ".join(hidden[:20])
        )

    try:
        tree = subprocess.check_output([
            "git", "-C", str(repo), "ls-tree", "-rz", "--full-tree",
            commit, "--", scope,
        ])
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot enumerate authorized tagged worktree blobs: {error}")
    entries: list[tuple[str, str, str]] = []
    for record in tree.split(b"\0"):
        if not record:
            continue
        header, separator, encoded_path = record.partition(b"\t")
        fields = header.split()
        if not separator or len(fields) != 3:
            fail("authorized tagged tree contains a malformed entry")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        if kind == "blob":
            entries.append((
                mode, object_id,
                encoded_path.decode("utf-8", "surrogateescape"),
            ))

    executed = {
        f"{scope}/scripts/reproduce_publication.sh",
        f"{scope}/scripts/host_publication_coordinator.py",
        f"{scope}/scripts/run_two_host_authenticated.sh",
        f"{scope}/scripts/peer_private_execution.py",
    }
    modes = {name: mode for mode, _, name in entries}
    if any(modes.get(name) != "100755" for name in executed):
        fail("authorized tag does not contain every executable publication control")

    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for mode, object_id, name in entries:
            path = repo / name
            try:
                metadata = path.lstat()
                if mode == "120000":
                    if not stat.S_ISLNK(metadata.st_mode):
                        fail("tracked worktree type differs from authorized tag: " + name)
                    observed = os.fsencode(os.readlink(path))
                else:
                    if not stat.S_ISREG(metadata.st_mode):
                        fail("tracked worktree type differs from authorized tag: " + name)
                    executable = bool(metadata.st_mode & 0o111)
                    if executable != (mode == "100755"):
                        fail("tracked worktree mode differs from authorized tag: " + name)
                    observed = path.read_bytes()
            except OSError as error:
                fail(f"cannot read tracked worktree blob {name}: {error}")
            process.stdin.write((object_id + "\n").encode("ascii"))
            process.stdin.flush()
            response = process.stdout.readline().split()
            if len(response) != 3 or response[1] != b"blob":
                fail("cannot read authorized tagged blob: " + name)
            size = int(response[2])
            authorized = process.stdout.read(size)
            if process.stdout.read(1) != b"\n" or observed != authorized:
                fail("tracked worktree bytes differ from authorized tag: " + name)
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()


def validate_source(repo: pathlib.Path, manifest_path: pathlib.Path,
                    authorization_path: pathlib.Path, executor: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    status = run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], capture=True).stdout
    if status:
        fail("source clone is dirty or contains untracked files")
    scan_private_artifacts(repo, manifest_path, publication=True)
    manifest = load_object(manifest_path, "publication environment manifest")
    if manifest.get("schema") != SCHEMA or manifest.get("manifest_digest") != canonical_digest(manifest, "manifest_digest"):
        fail("publication environment schema or self-digest differs")
    release = manifest.get("source_release")
    if not isinstance(release, dict):
        fail("source release authorization is absent")
    if release.get("unavailable_reason") is not None:
        fail("source release is unavailable: " + str(release["unavailable_reason"]))
    required_evidence = manifest.get("required_tracked_evidence")
    if not isinstance(required_evidence, list) or not required_evidence:
        fail("required immutable tracked evidence bindings are absent")
    for binding in required_evidence:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            fail("required tracked evidence binding is malformed or mutable")
        relative = pathlib.PurePosixPath(binding["path"])
        if relative.is_absolute() or ".." in relative.parts:
            fail("required tracked evidence path escapes the source clone")
        path = repo / relative
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()) != binding["sha256"]:
            fail("required immutable tracked evidence differs: " + relative.as_posix())
    tag = release.get("required_annotated_tag")
    if not isinstance(tag, str) or not tag:
        fail("required annotated source tag is absent")
    if not authorization_path.is_absolute():
        fail("source authorization path must be absolute")
    try:
        metadata = authorization_path.lstat()
        resolved = authorization_path.resolve(strict=True)
    except OSError as error:
        fail(f"cannot inspect source authorization: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail("source authorization must be a regular non-symlink")
    try:
        resolved.relative_to(repo.resolve(strict=True))
    except ValueError:
        pass
    else:
        fail("source authorization must be outside the source clone")
    if not os.statvfs(resolved).f_flag & os.ST_RDONLY:
        fail("source authorization must be supplied from a read-only mount")
    authorization = load_object(resolved, "source authorization")
    expected_fields = {"schema", "tag", "tag_object_id", "commit", "authorization_digest"}
    if set(authorization) != expected_fields or authorization.get("schema") != release.get("external_authorization_schema") or authorization.get("tag") != tag:
        fail("source authorization contract differs")
    if authorization.get("authorization_digest") != canonical_digest(authorization, "authorization_digest"):
        fail("source authorization self-digest differs")
    tag_type = run(["git", "-C", str(repo), "cat-file", "-t", f"refs/tags/{tag}"], capture=True).stdout.strip()
    tag_object = run(["git", "-C", str(repo), "rev-parse", f"refs/tags/{tag}"], capture=True).stdout.strip()
    tag_commit = run(["git", "-C", str(repo), "rev-parse", f"refs/tags/{tag}^{{commit}}"], capture=True).stdout.strip()
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True).stdout.strip()
    if tag_type != "tag" or tag_object != authorization.get("tag_object_id") or tag_commit != authorization.get("commit") or head != tag_commit:
        fail("external source authorization does not bind clean tagged HEAD")
    verify_authorized_worktree(repo, tag_commit)
    platform = manifest.get("platform")
    if not isinstance(platform, dict) or platform.get("required_cpu_features") != [
        "aes", "avx2", "pclmulqdq", "rdseed", "sse4_1"
    ] or platform.get("apt_sources") != (
        "ubuntu_snapshot_only_non_ubuntu_sources_disabled_empty"
    ):
        fail("publication CPU feature or APT source authorization is absent or malformed")
    runtime = manifest.get("publication_runtime")
    expected_runtime_contract = {
        "orchestrator": "host-native-rootless-podman",
        "network": "authenticated-two-host-ssh",
        "runtime_manifest_schema": "ringlpn-publication-runtime/v3",
        "evidence_manifest_schema": "ringlpn-publication-evidence/v3",
        "deletion_receipt_schema": "ringlpn-two-host-deletion-receipt-v1",
        "final_commit_schema": "ringlpn-two-host-final-commit-v2",
    }
    if not isinstance(runtime, dict) or any(
        runtime.get(key) != value for key, value in expected_runtime_contract.items()
    ):
        fail("publication runtime does not authorize the host coordinator")
    if runtime.get("unavailable_reason") is not None:
        fail("publication runtime is unavailable: " + str(runtime["unavailable_reason"]))
    image, binary = runtime.get("container_image"), runtime.get("container_binary_path")
    binary_sha, executor_sha = runtime.get("container_binary_sha256"), runtime.get("executor_sha256")
    valid_sha = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9._/:+@-]+@sha256:[0-9a-f]{64}", image) or not isinstance(binary, str) or not binary.startswith("/") or not valid_sha(binary_sha) or not valid_sha(executor_sha):
        fail("publication runtime hash authorization is malformed")
    if sha256(executor.read_bytes()) != executor_sha:
        fail("canonical local executor differs from the authorized SHA-256")
    return manifest, authorization



def validate_local_smoke(
    evidence_root: pathlib.Path,
    repo: pathlib.Path,
    manifest_path: pathlib.Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not evidence_root.is_absolute():
        fail("RINGLPN_LOCAL_SMOKE_EVIDENCE must be absolute")
    try:
        metadata = evidence_root.lstat()
        resolved = evidence_root.resolve(strict=True)
    except OSError as error:
        fail(f"cannot inspect local-smoke evidence: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail("local-smoke evidence must be a real directory")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        fail("local-smoke evidence root must be owner-owned and owner-only")
    try:
        resolved.relative_to(repo.resolve(strict=True))
    except ValueError:
        pass
    else:
        fail("local-smoke evidence root must be outside the source clone")
    evidence_path = resolved / "evidence-manifest.json"
    evidence = load_object(evidence_path, "local-smoke evidence manifest")
    if evidence.get("schema") != "ringlpn-publication-evidence/v3" or evidence.get("mode") != "local-smoke" or evidence.get("status") != "pass" or evidence.get("manifest_digest") != canonical_digest(evidence, "manifest_digest"):
        fail("local-smoke evidence manifest is not a self-bound PASS")
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True).stdout.strip()
    if evidence.get("repository_revision") != head:
        fail("local-smoke evidence does not bind the publication HEAD")
    static_sha = sha256(manifest_path.read_bytes())
    if evidence.get("static_manifest_sha256") != static_sha:
        fail("local-smoke evidence does not bind the current environment manifest")
    bindings = evidence.get("files")
    if not isinstance(bindings, list):
        fail("local-smoke retained file bindings are absent")
    by_path = {
        item.get("path"): item for item in bindings
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    runtime_bindings = [
        item for item in bindings
        if isinstance(item, dict) and item.get("retention") == "final-runtime-manifest"
    ]
    if len(runtime_bindings) != 1:
        fail("local-smoke evidence must bind exactly one finalized runtime manifest")
    runtime_binding = runtime_bindings[0]
    runtime_path = resolved / runtime_binding["path"]
    runtime_payload = runtime_path.read_bytes()
    if runtime_binding.get("bytes") != len(runtime_payload) or runtime_binding.get("sha256") != sha256(runtime_payload):
        fail("local-smoke runtime byte/hash binding failed")
    runtime = load_object(runtime_path, "local-smoke runtime manifest")
    expected_external = {"evidence-manifest.json"}
    for binding in bindings:
        if not isinstance(binding, dict) or binding.get("retention") == "tracked":
            continue
        relative_value = binding.get("path")
        if not isinstance(relative_value, str):
            fail("local-smoke external binding path is malformed")
        relative = pathlib.PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            fail("local-smoke external binding path escapes its evidence root")
        path = resolved / relative
        metadata = path.lstat()
        payload = path.read_bytes()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or binding.get("bytes") != len(payload) or binding.get("sha256") != sha256(payload):
            fail("local-smoke retained artifact byte/hash binding failed: " + relative.as_posix())
        expected_external.add(relative.as_posix())
    observed_external = set()
    for path in resolved.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            fail("local-smoke evidence contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            observed_external.add(path.relative_to(resolved).as_posix())
        elif not stat.S_ISDIR(metadata.st_mode):
            fail("local-smoke evidence contains a non-regular artifact")
    if observed_external != expected_external:
        fail(
            "local-smoke retained inventory differs; missing="
            + "|".join(sorted(expected_external - observed_external))
            + "; unexpected="
            + "|".join(sorted(observed_external - expected_external))
        )
    expected_image_reference = manifest.get("platform", {}).get("reproduction_gate_image")
    expected_image_id = manifest.get("platform", {}).get("reproduction_gate_image_id")
    if not isinstance(expected_image_reference, str) or not isinstance(expected_image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id):
        fail("tracked final reproduction gate image reference/ID is unavailable")
    if runtime.get("schema") != "ringlpn-publication-runtime/v3" or runtime.get("mode") != "local-smoke" or runtime.get("status") != "pass" or runtime.get("canonical_host_launch") is not True or runtime.get("repository_revision") != head or runtime.get("static_manifest_sha256") != static_sha or runtime.get("reproduction_image_reference") != expected_image_reference or runtime.get("reproduction_image_id") != expected_image_id or runtime.get("manifest_digest") != canonical_digest(runtime, "manifest_digest"):
        fail("local-smoke runtime/source/final-image binding failed")
    finished = runtime.get("finished_utc")
    try:
        finished_time = datetime.datetime.fromisoformat(finished)
    except (TypeError, ValueError):
        fail("local-smoke completion timestamp is malformed")
    age = datetime.datetime.now(datetime.timezone.utc) - finished_time
    if age.total_seconds() < 0 or age > datetime.timedelta(hours=24):
        fail("local-smoke prerequisite is not fresh (maximum age 24 hours)")
    smoke_binding = by_path.get("paper-checkpoint-smoke.log")
    smoke_path = resolved / "paper-checkpoint-smoke.log"
    smoke_payload = smoke_path.read_bytes()
    if not isinstance(smoke_binding, dict) or smoke_binding.get("bytes") != len(smoke_payload) or smoke_binding.get("sha256") != sha256(smoke_payload) or b"[paper-smoke] ALL GATES PASS" not in smoke_payload:
        fail("fresh local-smoke transcript is absent, unbound, or not ALL GATES PASS")
    evidence_payload = evidence_path.read_bytes()
    return {
        "evidence_manifest": {
            "bytes": len(evidence_payload),
            "sha256": sha256(evidence_payload),
        },
        "runtime_manifest": {
            "bytes": len(runtime_payload),
            "sha256": sha256(runtime_payload),
        },
        "repository_revision": head,
        "reproduction_image_id": expected_image_id,
    }

def validate_host_capabilities(executor: pathlib.Path, runtime: dict[str, Any], local_gpu: str,
                               local_uid: str, checker_uid: str) -> str:
    required_cpu = {"aes", "avx2", "pclmulqdq", "rdseed", "sse4_1"}
    processor_features = []
    try:
        for block in pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n"):
            fields = {
                key.strip(): value.strip()
                for line in block.splitlines() if ":" in line
                for key, value in (line.split(":", 1),)
            }
            if "processor" in fields:
                processor_features.append(set(fields.get("flags", "").split()))
    except OSError as error:
        fail(f"cannot inspect CPU features: {error}")
    missing_cpu = sorted(
        required_cpu - set.intersection(*processor_features)
        if processor_features else required_cpu
    )
    if missing_cpu:
        fail("required CPU features unavailable on every visible CPU: " + ",".join(missing_cpu))
    info = json.loads(run(["podman", "info", "--format", "json"], capture=True).stdout)
    security = info.get("host", {}).get("security", {})
    if security.get("rootless") is not True:
        fail("native Podman is not rootless")
    if security.get("userNamespaceEnabled") is False:
        fail("native Podman reports user namespaces disabled")
    names = {run(["id", "-un"], capture=True).stdout.strip(), str(os.getuid())}
    needed = max(int(local_uid), int(checker_uid)) + 1
    for filename in (pathlib.Path("/etc/subuid"), pathlib.Path("/etc/subgid")):
        ranges = []
        try:
            lines = filename.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            fail(f"cannot read subordinate-ID allocation {filename}: {error}")
        for line in lines:
            fields = line.split(":")
            if len(fields) == 3 and fields[0] in names:
                try:
                    ranges.append((int(fields[1]), int(fields[2])))
                except ValueError:
                    fail(f"malformed subordinate-ID allocation in {filename}")
        if not ranges or sum(count for _, count in ranges) < needed:
            fail(f"{filename} has no sufficient subordinate-ID allocation")
    image, binary, expected_binary_sha = (
        runtime["container_image"], runtime["container_binary_path"], runtime["container_binary_sha256"]
    )
    identity = load_json_text(run([
        str(executor), "runtime-identity", "--image", image, "--binary", binary
    ], capture=True).stdout, "local runtime identity")
    if identity.get("image") != image or identity.get("binary_path") != binary or identity.get("binary_sha256") != expected_binary_sha:
        fail("local immutable image or in-image FC binary differs from authorization")
    image_id = identity.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        fail("measured local runtime image ID is malformed")
    gpu = run([
        "podman", "run", "--rm", "--network=none", "--read-only",
        "--security-opt=no-new-privileges", "--cap-drop=all",
        "--device", f"nvidia.com/gpu={local_gpu}", "--entrypoint", "/usr/bin/nvidia-smi",
        image, "--query-gpu=compute_cap", "--format=csv,noheader",
    ], capture=True).stdout.splitlines()
    if not gpu or any(cap.strip() != "8.9" for cap in gpu):
        fail("selected local publication GPU is unavailable or not sm_89")
    return image_id


def load_json_text(value: str, label: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        fail(f"invalid {label}: {error}")
    if not isinstance(document, dict):
        fail(f"{label} must be a JSON object")
    return document


def main() -> int:
    if pathlib.Path("/.dockerenv").is_file() or pathlib.Path("/run/.containerenv").is_file():
        fail(GUIDANCE)
    if os.geteuid() == 0:
        fail("publication coordinator must not run as root")
    checker_uid, checker_gpu, launcher_args, values = parse_args(sys.argv[1:])
    script_root = pathlib.Path(__file__).resolve().parent
    root = script_root.parent
    repo = root.parent.parent
    manifest_path = script_root / "publication_environment_manifest_2026_08_10.json"
    executor = script_root / "peer_private_execution.py"
    launcher = script_root / "run_two_host_authenticated.sh"
    for path, label in ((executor, "executor"), (launcher, "authenticated launcher")):
        if not path.is_file() or not os.access(path, os.X_OK):
            fail(f"canonical {label} is absent or not executable")
    authorization_value = os.environ.get("RINGLPN_SOURCE_AUTHORIZATION", "")
    evidence_value = os.environ.get("RINGLPN_EVIDENCE_DIR", "")
    ledger_value = os.environ.get("RINGLPN_LEDGER_DIR", "")
    local_smoke_value = os.environ.get("RINGLPN_LOCAL_SMOKE_EVIDENCE", "")
    if not authorization_value:
        fail("RINGLPN_SOURCE_AUTHORIZATION is required for publication")
    if not evidence_value or not ledger_value or not local_smoke_value:
        fail("RINGLPN_EVIDENCE_DIR, RINGLPN_LEDGER_DIR, and RINGLPN_LOCAL_SMOKE_EVIDENCE are required")
    evidence, ledger = pathlib.Path(evidence_value), pathlib.Path(ledger_value)
    if not evidence.is_absolute() or not ledger.is_absolute():
        fail("evidence and ledger roots must be absolute")
    if values["ledger-root"] != str(ledger):
        fail("publication --ledger-root must equal RINGLPN_LEDGER_DIR")
    output_dir = pathlib.Path(values["output-dir"])
    if not output_dir.is_absolute() or output_dir.parent != evidence:
        fail("publication output-dir must be a fresh direct child of RINGLPN_EVIDENCE_DIR")
    if output_dir.exists():
        fail("publication output-dir must be fresh")
    local_party_ledger = pathlib.Path(values["local-party-ledger-root"])
    if not local_party_ledger.is_absolute():
        fail("local party ledger root must be absolute")
    for path, label in (
        (evidence, "evidence"),
        (ledger, "coordinator ledger"),
        (local_party_ledger, "local party ledger"),
    ):
        try:
            metadata = path.lstat()
        except OSError as error:
            fail(f"cannot inspect {label} root: {error}")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            fail(f"{label} root must be an owner-only owner-owned non-symlink directory")
        try:
            path.resolve(strict=True).relative_to(repo.resolve(strict=True))
        except ValueError:
            pass
        else:
            fail(f"{label} root must be outside the clean source clone")
    evidence_resolved = evidence.resolve()
    ledger_resolved = ledger.resolve()
    local_party_ledger_resolved = local_party_ledger.resolve()
    try:
        local_smoke_resolved = pathlib.Path(local_smoke_value).resolve(strict=True)
    except OSError as error:
        fail(f"cannot inspect local-smoke prerequisite root: {error}")
    roots = (
        evidence_resolved, ledger_resolved, local_party_ledger_resolved,
        local_smoke_resolved,
    )
    if any(left == right or left in right.parents or right in left.parents
           for index, left in enumerate(roots) for right in roots[index + 1:]):
        fail("evidence, coordinator/party ledgers, and local-smoke roots must be pairwise distinct and non-nested")
    transient_paths = [
        pathlib.Path(values[name]).resolve()
        for name in ("local-private-root", "local-export-root", "checker-stage",
                     "output-dir")
    ]
    if any(local_party_ledger_resolved == path or
           local_party_ledger_resolved in path.parents or
           path in local_party_ledger_resolved.parents
           for path in transient_paths):
        fail("local party ledger must be non-nested with private/export/checker/output paths")
    if any(evidence.iterdir()):
        fail("external evidence root must be empty before publication")
    for path, label in (
        (evidence_resolved, "evidence"),
        (ledger_resolved, "coordinator ledger"),
        (local_party_ledger_resolved, "local party ledger"),
        (local_smoke_resolved, "local-smoke evidence"),
    ):
        run(["mountpoint", "-q", str(path)])
        options = run(["findmnt", "-n", "-o", "OPTIONS", "--target", str(path)], capture=True).stdout.strip().split(",")
        if label.endswith("ledger") and "rw" not in options:
            fail(f"{label} mount must be read-write")
    mount_sources = [
        run(["findmnt", "-n", "-o", "SOURCE", "--target", str(path)], capture=True).stdout.strip()
        for path in (evidence_resolved, ledger_resolved,
                     local_party_ledger_resolved, local_smoke_resolved)
    ]
    if not all(mount_sources) or len(set(mount_sources)) != len(mount_sources):
        fail("publication/local-smoke evidence and coordinator/party ledgers must use distinct mount sources")
    manifest, authorization = validate_source(repo, manifest_path, pathlib.Path(authorization_value), executor)
    local_smoke_binding = validate_local_smoke(
        local_smoke_resolved, repo, manifest_path, manifest
    )
    runtime = manifest["publication_runtime"]
    image_id = validate_host_capabilities(executor, runtime, values["local-gpu"], values["local-container-uid"], checker_uid)
    command = [
        str(launcher), "--local-executor", str(executor),
        "--container-image", runtime["container_image"],
        "--container-binary", runtime["container_binary_path"],
        "--container-binary-sha256", runtime["container_binary_sha256"],
        "--checker-container-uid", checker_uid, "--checker-gpu", checker_gpu,
        *launcher_args,
    ]
    verify_authorized_worktree(repo, authorization["commit"])
    started = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    run(command)
    verify_authorized_worktree(repo, authorization["commit"])
    if set(evidence.iterdir()) != {output_dir}:
        fail("publication evidence root contains an unapproved survivor outside output-dir")
    launcher_result = load_object(output_dir / "launcher-result.json", "launcher result")
    receipt_path = output_dir / "deletion-receipt.json"
    committed_path = output_dir / "checker-stage/COMMITTED.manifest"
    receipt = load_object(receipt_path, "deletion receipt")
    committed = load_object(committed_path, "final COMMITTED manifest")
    if launcher_result.get("schema") != "ringlpn-authenticated-launch-result-v2" or launcher_result.get("status") != "PASS" or launcher_result.get("executor_sha256") != runtime["executor_sha256"]:
        fail("launcher PASS/executor authorization binding failed")
    if receipt.get("schema") != "ringlpn-two-host-deletion-receipt-v1" or receipt.get("status") != "CLEANED":
        fail("digest-bound deletion receipt is not CLEANED")
    if committed.get("schema") != "ringlpn-two-host-final-commit-v2" or committed.get("state") != "COMMITTED":
        fail("final checker/finalizer COMMITTED contract failed")
    if run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], capture=True).stdout:
        fail("publication mutated the original clean source clone")
    scan_private_artifacts(repo, manifest_path, publication=True)
    runtime_path = pathlib.Path(os.environ.get(
        "RINGLPN_RUNTIME_MANIFEST", str(evidence / f"runtime-{values['invocation-id']}.json")
    ))
    if not runtime_path.is_absolute() or runtime_path.parent != evidence:
        fail("RINGLPN_RUNTIME_MANIFEST must be a direct child of the external evidence root")
    runtime_document = {
        "schema": "ringlpn-publication-runtime/v3",
        "classification": "internal/advisor",
        "mode": "two-host-publication",
        "status": "pass",
        "execution_boundary": "host-native-rootless-podman",
        "repository_revision": run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True).stdout.strip(),
        "source_release_tag": authorization["tag"],
        "source_authorization_sha256": sha256(pathlib.Path(authorization_value).read_bytes()),
        "static_manifest_sha256": sha256(manifest_path.read_bytes()),
        "executor_sha256": runtime["executor_sha256"],
        "runtime_image": runtime["container_image"],
        "measured_local_runtime_image_id": image_id,
        "in_image_fc_binary": runtime["container_binary_path"],
        "in_image_fc_binary_sha256": runtime["container_binary_sha256"],
        "observed_cpu_features": ["aes", "avx2", "pclmulqdq", "rdseed", "sse4_1"],
        "local_smoke_prerequisite": local_smoke_binding,
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "secrets_or_private_data_recorded": False,
    }
    runtime_document["manifest_digest"] = canonical_digest(runtime_document, "manifest_digest")
    runtime_payload = (json.dumps(runtime_document, indent=2, sort_keys=True) + "\n").encode()
    expected_artifacts = {
        "launcher-result.json", "launcher-prepared.json",
        "authenticated-boundary.manifest", "authenticated-boundary.csv",
        "party0-sealed.json", "party1-sealed.json",
        "checker-isolation.json", "checker.log",
        "checker-stage/PREPARED.manifest", "checker-stage/COMMITTED.manifest",
        "checker-stage/party0/isolation-manifest.json",
        "checker-stage/party1/isolation-manifest.json",
        "ssh-master.log", "local-executor.log", "remote-executor.log",
        "deletion-receipt.json",
    }
    observed_artifacts = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*") if path.is_file()
    }
    if observed_artifacts != expected_artifacts:
        fail(
            "final two-host public artifact inventory differs; missing="
            + "|".join(sorted(expected_artifacts - observed_artifacts))
            + "; unexpected="
            + "|".join(sorted(observed_artifacts - expected_artifacts))
        )
    artifact_bindings = []
    for relative in sorted(expected_artifacts):
        path = output_dir / relative
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            fail("retained two-host artifact must be owner-only regular non-symlink: " + relative)
        payload = path.read_bytes()
        artifact_bindings.append({
            "path": f"{output_dir.name}/{relative}",
            "bytes": len(payload),
            "sha256": sha256(payload),
        })
    evidence_document = {
        "schema": "ringlpn-publication-evidence/v3",
        "classification": "internal/advisor",
        "mode": "two-host-publication",
        "status": "pass",
        "repository_revision": runtime_document["repository_revision"],
        "runtime_manifest": {"path": runtime_path.name, "bytes": len(runtime_payload), "sha256": sha256(runtime_payload)},
        "local_smoke_prerequisite": local_smoke_binding,
        "files": artifact_bindings,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "secrets_or_private_data_recorded": False,
    }
    evidence_document["manifest_digest"] = canonical_digest(evidence_document, "manifest_digest")
    evidence_path = evidence / "evidence-manifest.json"
    if evidence_path.exists():
        fail("evidence manifest destination already exists")
    atomic_json(runtime_path, runtime_document, 0o400)
    try:
        atomic_json(evidence_path, evidence_document, 0o400)
    except BaseException:
        try:
            evidence_path.unlink()
        except FileNotFoundError:
            pass
        runtime_document["status"] = "failed"
        runtime_document["failed_or_completed_phase"] = "publication-evidence-commit"
        runtime_document["finished_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).replace(microsecond=0).isoformat()
        runtime_document["manifest_digest"] = canonical_digest(
            runtime_document, "manifest_digest"
        )
        atomic_json(runtime_path, runtime_document, 0o400)
        raise
    print("[ringlpn-reproduce] TWO-HOST PUBLICATION GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
