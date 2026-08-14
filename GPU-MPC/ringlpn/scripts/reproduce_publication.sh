#!/usr/bin/env bash
# Host coordinator for publication; pinned-Docker entry point for check/smoke/build.
# It never records secret bytes or private data; owner-private trust-file paths may be argv.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
STATIC_MANIFEST="$ROOT/scripts/publication_environment_manifest_2026_08_10.json"
RUNTIME_MANIFEST="${RINGLPN_RUNTIME_MANIFEST:-/tmp/ringlpn-reproduction-manifest.json}"
EVIDENCE_DIR="${RINGLPN_EVIDENCE_DIR:-}"
SOURCE_AUTHORIZATION="${RINGLPN_SOURCE_AUTHORIZATION:-}"
LEDGER_DIR="${RINGLPN_LEDGER_DIR:-}"
MODE="${1:-}"
STATUS="failed"
PHASE="initializing"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUNTIME_FINALIZED=0
export SOURCE_DATE_EPOCH=1786320000
shift || true

verify_authorized_worktree() {
  python3 - "$REPO" "$STATIC_MANIFEST" <<'PY'
import json, os, pathlib, stat, subprocess, sys

repo, manifest_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
scope = "GPU-MPC/ringlpn"
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot load source-release authorization: {error}")
release = manifest.get("source_release")
tag = release.get("required_annotated_tag") if isinstance(release, dict) else None
if not isinstance(tag, str) or not tag:
    raise SystemExit("required annotated source tag is absent")
tag_ref = f"refs/tags/{tag}"
try:
    tag_type = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "-t", tag_ref], text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{tag_ref}^{{commit}}"], text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError) as error:
    raise SystemExit(f"cannot resolve authorized tagged commit: {error}")
if tag_type != "tag" or head != commit:
    raise SystemExit("clean-clone HEAD is not the authorized annotated-tag commit")

visibility = subprocess.check_output(
    ["git", "-C", str(repo), "ls-files", "-v", "-z"],
    text=True,
)
hidden = []
for record in visibility.split("\0"):
    if not record:
        continue
    marker, separator, name = record.partition(" ")
    if not separator or marker == "S" or marker.islower():
        hidden.append(name or record)
if hidden:
    raise SystemExit(
        "tracked paths use assume-unchanged or skip-worktree: "
        + " | ".join(hidden[:20])
    )

tree = subprocess.check_output([
    "git", "-C", str(repo), "ls-tree", "-rz", "--full-tree",
    commit, "--", scope,
])
entries = []
for record in tree.split(b"\0"):
    if not record:
        continue
    header, separator, encoded_path = record.partition(b"\t")
    fields = header.split()
    if not separator or len(fields) != 3:
        raise SystemExit("authorized tagged tree contains a malformed entry")
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
    raise SystemExit(
        "authorized tag does not contain every executable publication control"
    )

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
                    raise SystemExit(
                        "tracked worktree type differs from authorized tag: " + name
                    )
                observed = os.fsencode(os.readlink(path))
            else:
                if not stat.S_ISREG(metadata.st_mode):
                    raise SystemExit(
                        "tracked worktree type differs from authorized tag: " + name
                    )
                executable = bool(metadata.st_mode & 0o111)
                if executable != (mode == "100755"):
                    raise SystemExit(
                        "tracked worktree mode differs from authorized tag: " + name
                    )
                observed = path.read_bytes()
        except OSError as error:
            raise SystemExit(f"cannot read tracked worktree blob {name}: {error}")
        process.stdin.write((object_id + "\n").encode("ascii"))
        process.stdin.flush()
        response = process.stdout.readline().split()
        if len(response) != 3 or response[1] != b"blob":
            raise SystemExit("cannot read authorized tagged blob: " + name)
        size = int(response[2])
        authorized = process.stdout.read(size)
        if process.stdout.read(1) != b"\n" or observed != authorized:
            raise SystemExit(
                "tracked worktree bytes differ from authorized tag: " + name
            )
finally:
    process.stdin.close()
    process.stdout.close()
    process.wait()
PY
}

HOST_COORDINATOR_GUIDANCE="two-host-publication must run on each host's native rootless Podman; invoke ./GPU-MPC/ringlpn/scripts/reproduce_publication.sh two-host-publication directly on the coordinator host (never via docker run)."
if [[ "$MODE" == "two-host-publication" &&
      ( -f /.dockerenv || -f /run/.containerenv ) ]]; then
  echo "[ringlpn-reproduce] FAIL: $HOST_COORDINATOR_GUIDANCE" >&2
  exit 2
fi
case "$MODE" in
  check|local-smoke|remote-build|two-host-publication) ;;
  *)
    echo "usage: $0 {check|local-smoke|remote-build|two-host-publication} [host coordinator options] [-- authenticated-launcher options]" >&2
    exit 2;;
esac

if [[ "$MODE" != "two-host-publication" &&
      ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
  command -v docker >/dev/null 2>&1 ||
    { echo "[ringlpn-reproduce] FAIL: native Docker is required for $MODE" >&2; exit 2; }
  verify_authorized_worktree
  REPRODUCTION_IMAGE="${RINGLPN_REPRODUCTION_IMAGE:-ringlpn-repro:2026-08-10}"
  REPRODUCTION_IMAGE_ID="$(
    docker image inspect --format '{{.Id}}' "$REPRODUCTION_IMAGE"
  )" || {
    echo "[ringlpn-reproduce] FAIL: cannot inspect final reproduction image" >&2
    exit 2
  }
  [[ "$REPRODUCTION_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "[ringlpn-reproduce] FAIL: Docker returned a malformed reproduction image ID" >&2
    exit 2
  }
  python3 - "$STATIC_MANIFEST" "$REPRODUCTION_IMAGE" \
    "$REPRODUCTION_IMAGE_ID" "$MODE" <<'PY'
import json, pathlib, re, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
platform = manifest.get("platform", {})
if platform.get("reproduction_gate_image") != sys.argv[2]:
    raise SystemExit("measured final Docker image reference is not tracked")
if sys.argv[4] != "check":
    authorized_id = platform.get("reproduction_gate_image_id")
    if not isinstance(authorized_id, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", authorized_id):
        raise SystemExit(
            "publication mode requires a non-null authorized final Docker image ID"
        )
    if authorized_id != sys.argv[3]:
        raise SystemExit(
            "measured final Docker image ID is not exactly tracked and authorized"
        )
PY
  docker_args=(run --rm --network=none --gpus all
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -v "$REPO:/work/EzPC:ro"
    -e RINGLPN_CANONICAL_CONTAINER_LAUNCH=1
    -e "RINGLPN_REPRODUCTION_IMAGE_REFERENCE=$REPRODUCTION_IMAGE"
    -e "RINGLPN_REPRODUCTION_IMAGE_ID=$REPRODUCTION_IMAGE_ID")
  for selector in P0_GPU P1_GPU CHECK_GPU FULL_GRAPH_P0_GPU FULL_GRAPH_P1_GPU \
      FULL_GRAPH_CHECK_GPU FULL_GRAPH_TRUSTED_GPU; do
    [[ -z "${!selector:-}" ]] ||
      docker_args+=(-e "$selector=${!selector}")
  done
  if [[ "$MODE" != "check" ]]; then
    [[ -n "$EVIDENCE_DIR" && "$EVIDENCE_DIR" == /* ]] || {
      echo "[ringlpn-reproduce] FAIL: RINGLPN_EVIDENCE_DIR must be an absolute host path" >&2
      exit 2
    }
    python3 - "$EVIDENCE_DIR" "$REPO" "$(id -u)" <<'PY'
import pathlib, stat, sys
evidence, repo, uid = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
metadata = evidence.lstat()
resolved = evidence.resolve(strict=True)
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or \
        metadata.st_uid != uid or stat.S_IMODE(metadata.st_mode) & 0o077:
    raise SystemExit("host evidence root must be an owner-only owner-owned real directory")
try:
    resolved.relative_to(repo.resolve(strict=True))
except ValueError:
    pass
else:
    raise SystemExit("host evidence root must be outside the source clone")
if any(evidence.iterdir()):
    raise SystemExit("host evidence root must be empty")
PY
    docker_args+=(
      -v "$EVIDENCE_DIR:/output"
      -e RINGLPN_EVIDENCE_DIR=/output
      -e RINGLPN_RUNTIME_MANIFEST=/output/runtime-local.json)
  fi
  verify_authorized_worktree
  exec docker "${docker_args[@]}" "$REPRODUCTION_IMAGE_ID" "$MODE"
fi

emit_runtime_manifest() {
  local rc=$?
  ((RUNTIME_FINALIZED == 0)) || return "$rc"
  RINGLPN_EMIT_RC="$rc" RINGLPN_EMIT_STATUS="$STATUS" RINGLPN_EMIT_PHASE="$PHASE" \
  RINGLPN_EMIT_STARTED="$STARTED" RINGLPN_EMIT_MODE="$MODE" \
  RINGLPN_EMIT_REPO="$REPO" RINGLPN_EMIT_STATIC="$STATIC_MANIFEST" \
  RINGLPN_EMIT_SOURCE_AUTH="$SOURCE_AUTHORIZATION" \
  python3 - "$RUNTIME_MANIFEST" <<'PY'
import datetime, hashlib, json, os, pathlib, platform, subprocess, sys, tempfile

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
repo = pathlib.Path(os.environ["RINGLPN_EMIT_REPO"])
static = pathlib.Path(os.environ["RINGLPN_EMIT_STATIC"])
try:
    static_document = json.loads(static.read_text(encoding="utf-8"))
except Exception:
    static_document = {}
source_release = static_document.get("source_release")
source_tag = (
    source_release.get("required_annotated_tag")
    if isinstance(source_release, dict) else None
)
authorization_value = os.environ.get("RINGLPN_EMIT_SOURCE_AUTH")
authorization_path = pathlib.Path(authorization_value) if authorization_value else None
ubuntu_sources = pathlib.Path("/etc/apt/sources.list.d/ubuntu.sources")
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
cpu_sets = []
for block in pathlib.Path("/proc/cpuinfo").read_text().split("\n\n"):
    fields = {
        key.strip(): value.strip()
        for line in block.splitlines() if ":" in line
        for key, value in (line.split(":", 1),)
    }
    if "processor" in fields:
        cpu_sets.append(set(fields.get("flags", "").split()))
observed_cpu_features = sorted(set.intersection(*cpu_sets)) if cpu_sets else []
data = {
  "schema": "ringlpn-publication-runtime/v3",
  "classification": "internal/advisor",
  "mode": os.environ["RINGLPN_EMIT_MODE"],
  "status": os.environ["RINGLPN_EMIT_STATUS"],
  "failed_or_completed_phase": os.environ["RINGLPN_EMIT_PHASE"],
  "exit_code": int(os.environ["RINGLPN_EMIT_RC"]),
  "started_utc": os.environ["RINGLPN_EMIT_STARTED"],
  "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
  "repository_revision": command("git", "-C", str(repo), "rev-parse", "HEAD"),
  "source_date_epoch": int(os.environ["SOURCE_DATE_EPOCH"]),
  "source_release_tag": source_tag,
  "source_release_tag_commit": (
      command("git", "-C", str(repo), "rev-parse", f"refs/tags/{source_tag}^{{commit}}")
      if isinstance(source_tag, str) else None
  ),
  "static_manifest_sha256": hashlib.sha256(static.read_bytes()).hexdigest() if static.is_file() else None,
  "container_base_digest": os.environ.get("RINGLPN_REPRO_CONTAINER_DIGEST"),
  "reproduction_image_id": os.environ.get("RINGLPN_REPRODUCTION_IMAGE_ID"),
  "reproduction_image_reference": os.environ.get("RINGLPN_REPRODUCTION_IMAGE_REFERENCE"),
  "execution_boundary": "pinned-docker-clean-clone-gate",
  "ubuntu_archive_snapshot": os.environ.get("RINGLPN_UBUNTU_SNAPSHOT"),
  "ubuntu_sources_sha256": (
      hashlib.sha256(ubuntu_sources.read_bytes()).hexdigest()
      if ubuntu_sources.is_file() else None
  ),
  "canonical_host_launch": os.environ.get("RINGLPN_CANONICAL_CONTAINER_LAUNCH") == "1",
  "source_authorization_sha256": (
      hashlib.sha256(authorization_path.read_bytes()).hexdigest()
      if authorization_path is not None and authorization_path.is_file() else None
  ),
  "cpu_features": observed_cpu_features,
  "host": {"kernel": platform.release(), "machine": platform.machine()},
  "observed": {
    "nvcc": command("nvcc", "--version"),
    "gcc": command("gcc", "-dumpfullversion"),
    "g++": command("g++", "-dumpfullversion"),
    "cmake": command("cmake", "--version"),
    "gpu_compute_capabilities": command("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader")
  },
  "secrets_or_private_data_recorded": False
}
canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
data["manifest_digest"] = hashlib.sha256(canonical).hexdigest()
payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{out.name}.", suffix=".tmp", dir=out.parent
)
try:
    with os.fdopen(descriptor, "wb") as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, out)
    directory_fd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
    raise
PY
  echo "[ringlpn-reproduce] runtime manifest: $RUNTIME_MANIFEST" >&2
  return "$rc"
}
trap emit_runtime_manifest EXIT

fail() { echo "[ringlpn-reproduce] FAIL: $*" >&2; exit 2; }
require_command() { command -v "$1" >/dev/null 2>&1 || fail "required command missing: $1"; }

if [[ "$MODE" == "two-host-publication" ]]; then
  exec "$ROOT/scripts/host_publication_coordinator.py" "$@"
fi

case "$MODE" in
  check|local-smoke|remote-build) ;;
  *) fail "usage: $0 {check|local-smoke|remote-build|two-host-publication} [host coordinator options] [-- authenticated-launcher options]" ;;
esac
(($# == 0)) || fail "$MODE accepts no arguments"

PHASE="container-boundary"
[[ -f /.dockerenv ]] || fail "run only in scripts/Dockerfile.reproduction; host execution is not publication evidence"
[[ "${RINGLPN_CANONICAL_CONTAINER_LAUNCH:-}" == 1 ]] ||
  fail "direct docker run is non-evidence; invoke ./GPU-MPC/ringlpn/scripts/reproduce_publication.sh $MODE on the host"
[[ ${EUID:-$(id -u)} -ne 0 ]] || fail "refusing root: bind-mounted outputs would be root-owned"
[[ "${RINGLPN_REPRO_CONTAINER_DIGEST:-}" == "sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade" ]] ||
  fail "container base digest is absent or not pinned"
[[ "${RINGLPN_UBUNTU_SNAPSHOT:-}" == "20260810T000000Z" ]] ||
  fail "Ubuntu archive snapshot is absent or not pinned"
[[ -f /etc/apt/sources.list.d/cuda.list &&
   ! -s /etc/apt/sources.list.d/cuda.list ]] ||
  fail "CUDA apt source must be a deterministic empty disabled file"
grep -Fq \
  "URIs: https://snapshot.ubuntu.com/ubuntu/${RINGLPN_UBUNTU_SNAPSHOT}/" \
  /etc/apt/sources.list.d/ubuntu.sources ||
  fail "Ubuntu apt source is not snapshot-only"
if grep -Eq \
    'archive\.ubuntu\.com|security\.ubuntu\.com|developer\.download\.nvidia\.com' \
    /etc/apt/sources.list.d/*; then
  fail "mutable apt source remains enabled"
fi
[[ -f "$STATIC_MANIFEST" ]] || fail "immutable environment manifest missing"
for tool in git python3 sha256sum dpkg-query nvcc gcc g++ cmake nvidia-smi realpath; do require_command "$tool"; done
python3 - /proc/cpuinfo <<'PY'
import pathlib, sys
required = {"aes", "sse4_1", "pclmulqdq", "avx2", "rdseed"}
processors = []
for block in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").split("\n\n"):
    fields = {
        key.strip(): value.strip()
        for line in block.splitlines() if ":" in line
        for key, value in (line.split(":", 1),)
    }
    if "processor" in fields:
        processors.append(set(fields.get("flags", "").split()))
missing = sorted(required - set.intersection(*processors)) if processors else sorted(required)
if missing:
    raise SystemExit("required CPU features unavailable on every visible CPU: " + ",".join(missing))
PY
INITIAL_STATIC_MANIFEST_SHA256="$(sha256sum "$STATIC_MANIFEST" | cut -d' ' -f1)"
if [[ "$MODE" != "remote-build" ]]; then
  require_command pdflatex
  require_command pdffonts
  require_command pdfinfo
fi
if [[ "$MODE" != "check" ]]; then
  [[ -n "${RINGLPN_REPRODUCTION_IMAGE_ID:-}" ]] ||
    fail "RINGLPN_REPRODUCTION_IMAGE_ID must bind the measured final reproduction image ID"
  [[ "$RINGLPN_REPRODUCTION_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] ||
    fail "RINGLPN_REPRODUCTION_IMAGE_ID must be a sha256 image ID distinct from the pinned base digest"
  [[ "$RINGLPN_REPRODUCTION_IMAGE_ID" != "${RINGLPN_REPRO_CONTAINER_DIGEST}" ]] ||
    fail "final reproduction image ID must be distinct from the pinned base-image digest"
fi
ubuntu_sources_sum="$(sha256sum /etc/apt/sources.list.d/ubuntu.sources)"
[[ "${ubuntu_sources_sum%% *}" == \
  "0c91001954b305aaf387872987edb46e54dd525b2ed8dc7bf1f172929402363c" ]] ||
  fail "Ubuntu apt source digest differs from the pinned image"

PHASE="clean-clone"
verify_authorized_worktree
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || fail "repository is dirty or contains untracked files"
python3 "$ROOT/scripts/retained_public_evidence.py" \
  --repo "$REPO" --manifest "$STATIC_MANIFEST"
python3 - "$REPO" "$STATIC_MANIFEST" "$MODE" <<'PY'
import hashlib, json, os, pathlib, stat, subprocess, sys
repo, manifest_path, mode = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
m = json.loads(manifest_path.read_text())
if m.get("schema") != "ringlpn-publication-environment/v4" or \
        m.get("date") != "2026-08-14":
    raise SystemExit("unexpected static publication manifest schema or date")
manifest_digest = m.get("manifest_digest")
unsigned_manifest = dict(m)
unsigned_manifest.pop("manifest_digest", None)
calculated_manifest_digest = hashlib.sha256(json.dumps(
    unsigned_manifest, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
if manifest_digest != calculated_manifest_digest:
    raise SystemExit("static publication manifest self-digest differs")
release = m.get("source_release")
tag = release.get("required_annotated_tag") if isinstance(release, dict) else None
authorization_schema = (
    release.get("external_authorization_schema")
    if isinstance(release, dict) else None
)
if not isinstance(tag, str) or not tag or \
        authorization_schema != "ringlpn-source-authorization/v1":
    raise SystemExit("static manifest has no valid source release contract")
expected_evidence_policy = (
    "Every retained tracked artifact is a canonical clone-relative regular "
    "non-symlink with an immutable SHA-256 pinned here. Publication execution "
    "never mutates tracked source results; invocation-scoped outputs are written "
    "only to an external owner-private evidence directory."
)
if m.get("required_tracked_evidence_policy") != expected_evidence_policy:
    raise SystemExit("static manifest has no valid retained-evidence integrity policy")
platform = m.get("platform")
expected_frontend = (
    "docker/dockerfile:1@sha256:"
    "87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89"
)
if not isinstance(platform, dict) or \
        platform.get("ubuntu_archive_snapshot") != "20260810T000000Z" or \
        platform.get("ubuntu_snapshot_service") != "https://snapshot.ubuntu.com/" or \
        platform.get("apt_sources") != \
            "ubuntu_snapshot_only_non_ubuntu_sources_disabled_empty" or \
        platform.get("ubuntu_sources_sha256") != \
            "0c91001954b305aaf387872987edb46e54dd525b2ed8dc7bf1f172929402363c" or \
        platform.get("dockerfile_frontend") != expected_frontend or \
        platform.get("required_cpu_features") != \
            ["aes", "avx2", "pclmulqdq", "rdseed", "sse4_1"]:
    raise SystemExit("unexpected pinned container-build environment")
if mode != "check" and (
        platform.get("reproduction_gate_image") !=
            os.environ.get("RINGLPN_REPRODUCTION_IMAGE_REFERENCE") or
        platform.get("reproduction_gate_image_id") !=
            os.environ.get("RINGLPN_REPRODUCTION_IMAGE_ID")
):
    raise SystemExit("measured final reproduction image reference/ID is not tracked/authorized")
build = m.get("build")
if not isinstance(build, dict) or \
        build.get("source_date_epoch") != int(os.environ["SOURCE_DATE_EPOCH"]):
    raise SystemExit("unexpected deterministic build epoch")
for path_field, digest_field in (
        ("publication_source", "publication_source_sha256"),
        ("publication_pdf", "publication_pdf_sha256")):
    relative = build.get(path_field)
    expected_digest = build.get(digest_field)
    if not isinstance(relative, str) or pathlib.PurePosixPath(relative).is_absolute() or \
            ".." in pathlib.PurePosixPath(relative).parts or \
            not isinstance(expected_digest, str) or len(expected_digest) != 64 or \
            any(character not in "0123456789abcdef" for character in expected_digest):
        raise SystemExit("invalid publication source/PDF build binding")
    artifact = repo / relative
    if not artifact.is_file() or \
            hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_digest:
        raise SystemExit("publication source/PDF digest differs: " + relative)
packages = m.get("container_packages")
if not isinstance(packages, dict) or not packages or \
        not all(isinstance(name, str) and isinstance(version, str)
                for name, version in packages.items()):
    raise SystemExit("invalid pinned container package set")
for name, expected_version in sorted(packages.items()):
    observed = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Version}", name],
        text=True,
    ).strip()
    if observed != expected_version:
        raise SystemExit(
            f"container package mismatch: {name}={observed}, "
            f"expected {expected_version}"
        )
optional = {
    x["path"]
    for x in m.get("data_inputs", {}).get("excluded_optional_snapshots", [])
}
expected = {
    x["path"]: x["revision"] for x in m["sources"]
    if x["path"] != "GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu"
    and x["path"] not in optional
}
roots = sorted(
    path for path in expected
    if not any(path.startswith(parent + "/") for parent in expected if parent != path)
)
out = subprocess.check_output(
    ["git", "-C", str(repo), "submodule", "status", "--recursive", "--", *roots],
    text=True,
)
seen = {}
for line in out.splitlines():
    if not line or line[0] != " ":
        raise SystemExit("missing, conflicted, or wrong-revision gate submodule: " + line)
    fields = line[1:].split()
    seen[fields[1]] = fields[0]
if seen != expected:
    raise SystemExit("gate-required recursive submodule set/revisions differ from immutable manifest")
for path in expected:
    status = subprocess.check_output(
        ["git", "-C", str(repo / path), "status", "--porcelain",
         "--untracked-files=all"], text=True,
    )
    if status:
        raise SystemExit("dirty gate-required submodule: " + path)
for source in m["sources"]:
    lf, want = source.get("license_file"), source.get("license_sha256")
    if lf and not (repo / lf).is_file():
        raise SystemExit("missing retained license: " + lf)
    if lf and want and hashlib.sha256((repo / lf).read_bytes()).hexdigest() != want:
        raise SystemExit("license checksum mismatch: " + lf)
required = m.get("required_tracked_evidence")
if not isinstance(required, list) or not required:
    raise SystemExit("invalid required tracked evidence bindings")
required_paths = []
seen_required_paths = set()
for binding in required:
    if not isinstance(binding, dict):
        raise SystemExit("invalid required tracked evidence binding")
    path = binding.get("path")
    if not isinstance(path, str):
        raise SystemExit("invalid required tracked evidence binding")
    if set(binding) != {"path", "sha256"}:
        raise SystemExit("tracked evidence bindings must be immutable path/SHA-256 pairs")
    expected_digest = binding["sha256"]
    if not isinstance(path, str) or not isinstance(expected_digest, str) or \
            len(expected_digest) != 64 or \
            any(character not in "0123456789abcdef"
                for character in expected_digest):
        raise SystemExit("invalid required tracked evidence binding")
    relative = pathlib.PurePosixPath(path)
    tracked_path = repo / relative
    try:
        resolved = tracked_path.resolve(strict=True)
        resolved.relative_to(repo.resolve(strict=True))
        metadata = tracked_path.lstat()
    except (OSError, ValueError) as error:
        raise SystemExit("invalid tracked evidence path: " + path) from error
    if relative.is_absolute() or ".." in relative.parts or \
            relative.as_posix() != path or path in seen_required_paths or \
            stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("invalid tracked evidence path: " + path)
    seen_required_paths.add(path)
    required_paths.append(path)
    subprocess.check_call(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", path],
        stdout=subprocess.DEVNULL,
    )
    observed_digest = hashlib.sha256(tracked_path.read_bytes()).hexdigest()
    if observed_digest != expected_digest:
        raise SystemExit("required tracked evidence digest differs: " + path)

index_matches = [pathlib.Path(x) for x in required_paths
                 if x.endswith("/resnet18_full_graph_checkpoint_2026_08_10/INDEX.json")]
manifest_matches = [pathlib.Path(x) for x in required_paths
                    if x.endswith("/resnet18_full_graph_checkpoint_2026_08_10/FULL_GRAPH.manifest")]
if len(index_matches) != 1 or len(manifest_matches) != 1 or \
        index_matches[0].parent != manifest_matches[0].parent:
    raise SystemExit("retained full-graph evidence paths are not pinned exactly once")
index_path, full_path = repo / index_matches[0], repo / manifest_matches[0]
if index_path.is_symlink() or full_path.is_symlink():
    raise SystemExit("retained full-graph evidence must not be symlinks")

def load_object(path, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label}: {error}")
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value

def self_digest(value, field):
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(payload, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

index = load_object(index_path, "retained full-graph index")
full = load_object(full_path, "retained full-graph manifest")
def contains_absolute_path(value):
    if isinstance(value, str):
        return value.startswith("/")
    if isinstance(value, dict):
        return any(contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_absolute_path(item) for item in value)
    return False

if contains_absolute_path(index) or contains_absolute_path(full):
    raise SystemExit("retained full-graph evidence contains an absolute path")
if index.get("schema") != "ringlpn-known-zero-full-resnet18-retained-evidence-v2" or \
        index.get("status") != "pass" or \
        index.get("index_digest") != self_digest(index, "index_digest"):
    raise SystemExit("retained full-graph index contract failed")
if full.get("schema") != "ringlpn-known-zero-full-resnet18-graph-v4" or \
        full.get("status") != "pass" or \
        full.get("manifest_digest") != self_digest(full, "manifest_digest"):
    raise SystemExit("retained full-graph manifest contract failed")
if index.get("full_graph_manifest_digest") != full.get("manifest_digest"):
    raise SystemExit("retained full-graph index/manifest digest mismatch")
linear_binding = full.get("linear_record_set")
metrics = full.get("metrics")
graph_contract = full.get("graph_contract")
controls = full.get("controls")
gpu_assignment = full.get("gpu_assignment")
if not all(isinstance(value, dict)
           for value in (linear_binding, metrics, graph_contract,
                         gpu_assignment)) or not isinstance(controls, list):
    raise SystemExit("retained full-graph measured-contract objects are invalid")
aggregate = linear_binding.get("aggregate")
metric_rows = [metrics.get(role)
               for role in ("adapter", "party0", "party1", "checker")]
expected_controls = [
    "forced_second_rename",
    "reused_invocation",
    "stale_output",
    "party_record_swap",
    "truncated_nonlinear_graph_record",
    "nonlinear_graph_payload_corruption",
    "digest_valid_trace_corruption",
]
gpu_roles = ("linear_p0", "linear_p1", "linear_check")
gpu_values = [gpu_assignment.get(role) for role in gpu_roles]
if not isinstance(aggregate, dict) or \
        not all(isinstance(row, dict) for row in metric_rows) or \
        aggregate.get("layers") != 21 or \
        linear_binding.get("mode") != "generated_in_run" or \
        any(row.get("status") != "pass" for row in metric_rows) or \
        graph_contract.get("status") != "pass" or \
        [row.get("control") for row in controls
         if isinstance(row, dict)] != expected_controls or \
        len(controls) != len(expected_controls) or \
        any(not isinstance(row, dict) or
            row.get("expected_rejection") != "pass" or
            row.get("no_partial_output") != "pass" or
            row.get("status") != "pass" for row in controls) or \
        not all(isinstance(value, int) and not isinstance(value, bool) and
                value >= 0 for value in gpu_values) or \
        len(set(gpu_values)) != len(gpu_values) or \
        gpu_assignment.get("graph_p0") != gpu_assignment.get("linear_p0") or \
        gpu_assignment.get("graph_p1") != gpu_assignment.get("linear_p1"):
    raise SystemExit("retained full-graph measured-contract fields failed")

expected_files = {
    "FULL_GRAPH.manifest", "adapter.csv", "party0.csv", "party1.csv",
    "checker.csv", "controls.csv", "adapter.log", "party0.log", "party1.log",
    "checker.log", "linear/LINEAR_RECORD_SET.manifest",
    "linear/private_inputs/binary_approval.json",
    "linear/private_inputs/linear_adapter_build_provenance.json",
    "private_inputs/graph_binary_approval.json",
    "private_inputs/graph_build_provenance.json",
    "private_inputs/linear_binary_approval.json",
    "private_inputs/linear_adapter_build_provenance.json",
}
files = index.get("files")
if not isinstance(files, dict) or set(files) != expected_files:
    raise SystemExit("retained full-graph file inventory differs")
expected_directories = {"linear", "linear/private_inputs", "private_inputs"}
observed_files = set()
observed_directories = set()
for retained_path in index_path.parent.rglob("*"):
    relative = retained_path.relative_to(index_path.parent).as_posix()
    metadata = retained_path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(
            "retained full-graph directory contains a symlink: " + relative
        )
    if stat.S_ISDIR(metadata.st_mode):
        observed_directories.add(relative)
        continue
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(
            "retained full-graph directory contains a non-regular artifact: "
            + relative
        )
    observed_files.add(relative)
if observed_directories != expected_directories or \
        observed_files != expected_files | {"INDEX.json"}:
    raise SystemExit("retained full-graph directory inventory differs")
for name, binding in files.items():
    relative = pathlib.PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or
            relative.as_posix() != name or not isinstance(binding, dict)):
        raise SystemExit("invalid retained full-graph file binding")
    path = index_path.parent / relative
    if path.is_symlink() or not path.is_file():
        raise SystemExit("missing or symlinked retained full-graph file: " + name)
    rel = path.relative_to(repo).as_posix()
    subprocess.check_call(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
        stdout=subprocess.DEVNULL,
    )
    data = path.read_bytes()
    if binding.get("bytes") != len(data) or \
            binding.get("sha256") != hashlib.sha256(data).hexdigest():
        raise SystemExit("retained full-graph file binding failed: " + name)
relocations = index.get("relocations")
if not isinstance(relocations, list) or len(relocations) != 1:
    raise SystemExit("retained full-graph relocation inventory differs")
relocation = relocations[0]
approval = load_object(
    index_path.parent / "private_inputs/graph_binary_approval.json",
    "retained graph binary approval",
)
provenance_binding = approval.get("provenance")
if (not isinstance(relocation, dict) or set(relocation) != {
        "declared_path", "document", "field", "retained_path", "sha256"} or
        not isinstance(provenance_binding, dict) or
        relocation.get("document") !=
        "private_inputs/graph_binary_approval.json" or
        relocation.get("field") != "provenance.path" or
        relocation.get("declared_path") != provenance_binding.get("path") or
        relocation.get("retained_path") !=
        "private_inputs/graph_build_provenance.json" or
        relocation.get("sha256") != files[
            "private_inputs/graph_build_provenance.json"
        ].get("sha256")):
    raise SystemExit("retained graph provenance relocation binding differs")
for field, retained_name in (
        ("binary_approval", "private_inputs/linear_binary_approval.json"),
        ("linear_adapter_build_provenance",
         "private_inputs/linear_adapter_build_provenance.json"),
        ("graph_binary_approval",
         "private_inputs/graph_binary_approval.json"),
        ("graph_build_provenance",
         "private_inputs/graph_build_provenance.json"),
        ("linear_record_set", "linear/LINEAR_RECORD_SET.manifest")):
    binding = full.get(field)
    if (not isinstance(binding, dict) or
            binding.get("path_scope") != "run-output-relative" or
            binding.get("path") != retained_name or
            binding.get("sha256") != files[retained_name].get("sha256")):
        raise SystemExit(
            "retained full-graph declared-path binding differs: " + field
        )
linear = load_object(
    index_path.parent / "linear/LINEAR_RECORD_SET.manifest",
    "retained linear record-set manifest",
)
for field, declared_name, retained_name, graph_copy in (
        ("binary_approval", "private_inputs/binary_approval.json",
         "linear/private_inputs/binary_approval.json",
         "private_inputs/linear_binary_approval.json"),
        ("linear_adapter_build_provenance",
         "private_inputs/linear_adapter_build_provenance.json",
         "linear/private_inputs/linear_adapter_build_provenance.json",
         "private_inputs/linear_adapter_build_provenance.json")):
    binding = linear.get(field)
    if (not isinstance(binding, dict) or
            binding.get("path") != declared_name or
            binding.get("sha256") != files[retained_name].get("sha256") or
            files[graph_copy].get("sha256") !=
            files[retained_name].get("sha256")):
        raise SystemExit(
            "retained linear provenance declared-path binding differs: " + field
        )
PY

if [[ "$MODE" != "check" && "${RINGLPN_EXECUTION_COPY:-0}" != 1 ]]; then
  PHASE="private-execution-workspace"
  require_command cp
  require_command mktemp
  execution_parent="$(mktemp -d /tmp/ringlpn-reproduction.XXXXXXXX)"
  chmod 700 "$execution_parent"
  execution_repo="$execution_parent/EzPC"
  mkdir -m 700 "$execution_repo"
  cp -a "$REPO/." "$execution_repo/"
  trap - EXIT
  set +e
  RINGLPN_EXECUTION_COPY=1 RINGLPN_ORIGINAL_REPO="$REPO" \
    "$execution_repo/GPU-MPC/ringlpn/scripts/reproduce_publication.sh" "$MODE"
  workspace_rc=$?
  set -e
  if [[ "$workspace_rc" -eq 0 ]]; then
    rm -rf "$execution_parent"
  else
    echo "[ringlpn-reproduce] failed execution workspace retained at $execution_parent" >&2
  fi
  [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] ||
    fail "container gate mutated the original clean source clone"
  python3 "$ROOT/scripts/retained_public_evidence.py" \
    --repo "$REPO" --manifest "$STATIC_MANIFEST"
  exit "$workspace_rc"
fi

PHASE="ownership"
python3 - "$ROOT" <<'PY'
import os, pathlib, sys
if os.geteuid() == 0:
    raise SystemExit("root execution is forbidden")
bad=[]
for rel in ("bin", "host_bin", "results"):
    base=pathlib.Path(sys.argv[1], rel)
    if not base.exists(): continue
    for root, dirs, files in os.walk(base):
        for name in dirs+files:
            p=pathlib.Path(root,name)
            try:
                if p.lstat().st_uid == 0: bad.append(str(p))
            except FileNotFoundError: pass
            if len(bad) >= 10: break
        if len(bad) >= 10: break
if bad: raise SystemExit("root-owned outputs: " + ", ".join(bad))
PY

dpkg-query -W -f='${Status}' libmpfr-dev 2>/dev/null | grep -qx 'install ok installed' || fail "libmpfr-dev is absent"
[[ "$(gcc -dumpfullversion)" == "13.3.0" ]] || fail "gcc must be 13.3.0"
[[ "$(g++ -dumpfullversion)" == "13.3.0" ]] || fail "g++ must be 13.3.0"
nvcc_version="$(nvcc --version)"
[[ "$nvcc_version" == *"release 12.6, V12.6.85"* ]] || fail "nvcc must be CUDA 12.6.85"
[[ "$(cmake --version | sed -n '1p')" == "cmake version 3.28.3" ]] || fail "cmake must be 3.28.3"
[[ "$(python3 --version)" == "Python 3.12.3" ]] || fail "python must be 3.12.3"
if [[ "$MODE" != remote-build ]]; then
  [[ "$(pdflatex --version)" == *"TeX Live 2023"* ]] || fail "pdflatex must be from TeX Live 2023"
fi
[[ "${CUDA_ARCH:-89}" == 89 && "${GPU_ARCH:-89}" == 89 ]] || fail "GPU_ARCH/CUDA_ARCH must both be 89"
mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ')
((${#caps[@]} > 0)) || fail "no visible NVIDIA GPU"
minimum_visible_gpus="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["platform"]["minimum_visible_gpus"])' \
    "$STATIC_MANIFEST"
)"
[[ "$minimum_visible_gpus" =~ ^[1-9][0-9]*$ ]] ||
  fail "invalid minimum_visible_gpus in static manifest"
if [[ "$MODE" != remote-build ]] &&
    ((${#caps[@]} < minimum_visible_gpus)); then
  fail "local/full-graph gates require at least $minimum_visible_gpus visible sm_89 GPUs"
fi
for cap in "${caps[@]}"; do [[ "$cap" == "8.9" ]] || fail "wrong visible GPU architecture: expected compute capability 8.9, got $cap"; done
mapfile -t drivers < <(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' ')
python3 - "${drivers[@]}" <<'PY'
import sys
def version(s): return tuple(int(x) for x in s.split("."))
minimum = version("560.35.03")
if not sys.argv[1:] or any(version(v) < minimum for v in sys.argv[1:]):
    raise SystemExit("NVIDIA driver must be at least 560.35.03")
PY

if [[ "$MODE" == check ]]; then
  verify_authorized_worktree
  STATUS="pass"; PHASE="preflight-complete"
  echo "[ringlpn-reproduce] PREFLIGHT PASS (no build or gate run)"
  exit 0
fi

PHASE="evidence-output"
[[ -n "$EVIDENCE_DIR" && "$EVIDENCE_DIR" == /* ]] || fail "RINGLPN_EVIDENCE_DIR must be an absolute, external mounted directory"
case "$(realpath -m "$EVIDENCE_DIR")" in "$REPO"|"$REPO"/*) fail "evidence directory must be outside the source clone";; esac
mkdir -p "$EVIDENCE_DIR"
[[ -w "$EVIDENCE_DIR" ]] || fail "evidence directory is not writable"
python3 - "$REPO" "$EVIDENCE_DIR" <<'PY'
import os, pathlib, stat, sys

repo = pathlib.Path(sys.argv[1]).resolve(strict=True)
evidence = pathlib.Path(sys.argv[2])
try:
    metadata = evidence.lstat()
    resolved = evidence.resolve(strict=True)
    resolved.relative_to(repo)
except ValueError:
    pass
except OSError as error:
    raise SystemExit(f"cannot inspect evidence directory: {error}") from error
else:
    raise SystemExit("evidence directory resolved inside the source clone")
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("evidence directory must be a real directory, not a symlink")
stale = sorted(entry.name for entry in evidence.iterdir())
if stale:
    raise SystemExit(
        "evidence directory must be empty; refusing stale destination: "
        + " | ".join(stale[:20])
    )
PY

if [[ "$MODE" == remote-build ]]; then
  PHASE="remote-binary-build"
  GPU_ARCH=89 CUDA_ARCH=89 "$ROOT/scripts/build_two_party_fc_preprocess.sh"
  (cd "$ROOT" && sha256sum bin/test_two_party_fc_preprocess) \
    > "$EVIDENCE_DIR/remote_fc_binary.sha256"
  STATUS="pass"; PHASE="remote-build-complete"
  echo "[ringlpn-reproduce] REMOTE BUILD PASS"
  exit 0
fi

PHASE="publication-pdf"
REPORT_DIR="$ROOT/results/reports"
TEX="dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex"
(
  cd "$REPORT_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX"
)
python3 - "$STATIC_MANIFEST" "$REPORT_DIR/${TEX%.tex}.log" \
  "$REPORT_DIR/${TEX%.tex}.pdf" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
log = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
diagnostic = re.compile(
    r"LaTeX(?: Font)? Warning|Package [^\n]+ Warning|"
    r"(?:Over|Under)full \\[hv]box|undefined references|multiply defined",
    re.IGNORECASE,
)
if diagnostic.search(log):
    raise SystemExit("publication PDF build emitted a LaTeX warning or bad box")
pdf = pathlib.Path(sys.argv[3])
expected = manifest["build"]["publication_pdf_sha256"]
if hashlib.sha256(pdf.read_bytes()).hexdigest() != expected:
    raise SystemExit("deterministic publication PDF digest differs")
pdf_metadata = subprocess.check_output(
    ["pdfinfo", str(pdf)], text=True
)
page_match = re.search(r"^Pages:\s+(\d+)\s*$", pdf_metadata, re.MULTILINE)
expected_pages = manifest["build"].get("publication_pages")
if page_match is None or not isinstance(expected_pages, int) or \
        int(page_match.group(1)) != expected_pages:
    raise SystemExit("publication PDF page count differs")
font_lines = subprocess.check_output(
    ["pdffonts", str(pdf)], text=True
).splitlines()
if len(font_lines) < 3:
    raise SystemExit("publication PDF font inventory is empty")
header = font_lines[0]
try:
    type_at = header.index("type")
    encoding_at = header.index("encoding")
    embedded_at = header.index("emb")
    subset_at = header.index("sub")
except ValueError as error:
    raise SystemExit("publication PDF font inventory schema differs") from error
bad_fonts = []
for line in font_lines[2:]:
    if not line.strip():
        continue
    font_type = line[type_at:encoding_at].strip()
    embedded = line[embedded_at:subset_at].strip()
    if font_type != "Type 1" or embedded != "yes":
        bad_fonts.append(line)
if bad_fonts:
    raise SystemExit(
        "publication PDF requires embedded Type 1 fonts only: " +
        " | ".join(bad_fonts)
    )
PY
rm -f "$REPORT_DIR/${TEX%.tex}.aux" "$REPORT_DIR/${TEX%.tex}.log" \
  "$REPORT_DIR/${TEX%.tex}.out"

PHASE="canonical-component-gates"
fresh_graph_summary="$EVIDENCE_DIR/fresh-full-graph"
[[ ! -e "$fresh_graph_summary" ]] ||
  fail "stale fresh full-graph destination: $fresh_graph_summary"
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 RUN_REGULAR_SMOKE=1 \
  RUN_FULL_GRAPH_SMOKE=1 FULL_GRAPH_SUMMARY_ROOT="$fresh_graph_summary" \
  FULL_GRAPH_P0_GPU="${FULL_GRAPH_P0_GPU:-${P0_GPU:-0}}" \
  FULL_GRAPH_P1_GPU="${FULL_GRAPH_P1_GPU:-${P1_GPU:-1}}" \
  FULL_GRAPH_CHECK_GPU="${FULL_GRAPH_CHECK_GPU:-${CHECK_GPU:-2}}" \
  FULL_GRAPH_TRUSTED_GPU="${FULL_GRAPH_TRUSTED_GPU:-${FULL_GRAPH_P0_GPU:-${P0_GPU:-0}}}" \
  GPU_ARCH=89 CUDA_ARCH=89 "$ROOT/scripts/run_paper_checkpoint_smoke.sh" \
  2>&1 | tee "$EVIDENCE_DIR/paper-checkpoint-smoke.log"
[[ -f "$fresh_graph_summary/INDEX.json" &&
   -f "$fresh_graph_summary/FULL_GRAPH.manifest" ]] ||
  fail "canonical gate did not retain fresh full-graph evidence"



PHASE="source-integrity-postflight"
verify_authorized_worktree
python3 - "$REPO" <<'PY'
import pathlib
import subprocess
import sys

repo = pathlib.Path(sys.argv[1])
raw = subprocess.check_output([
    "git", "-C", str(repo), "status", "--porcelain=v1", "-z",
    "--untracked-files=all", "--ignore-submodules=none",
])
unexpected = []
for record in raw.split(b"\0"):
    if not record:
        continue
    status = record[:2]
    if b"R" in status or b"C" in status:
        unexpected.append(record.decode("utf-8", "replace"))
        continue
    path = record[3:].decode("utf-8", "surrogateescape")
    allowed = (
        path.startswith("GPU-MPC/ringlpn/results/")
        and not path.startswith("GPU-MPC/ringlpn/results/reports/")
        and path != "GPU-MPC/ringlpn/results/README.md"
    )
    if not allowed:
        unexpected.append(f"{status.decode('ascii', 'replace')} {path}")
if unexpected:
    raise SystemExit(
        "gate mutated source, submodules, manuscript, or unapproved paths: "
        + " | ".join(unexpected[:20])
    )
PY
python3 "$ROOT/scripts/retained_public_evidence.py" \
  --repo "$REPO" --manifest "$STATIC_MANIFEST"

PHASE="postflight"
python3 - "$ROOT" <<'PY'
import os, pathlib, sys
bad=[]
for rel in ("bin", "host_bin", "results"):
    base=pathlib.Path(sys.argv[1], rel)
    if not base.exists(): continue
    for root, dirs, files in os.walk(base):
        for name in dirs+files:
            p=pathlib.Path(root,name)
            try:
                if p.lstat().st_uid == 0: bad.append(str(p))
            except FileNotFoundError: pass
if bad: raise SystemExit("root-owned outputs after gates: " + ", ".join(bad[:10]))
PY

PHASE="retain-explicit-public-evidence"
(cd "$REPO" && sha256sum \
  GPU-MPC/ringlpn/scripts/publication_environment_manifest_2026_08_10.json \
  "GPU-MPC/ringlpn/results/reports/${TEX%.tex}.pdf") \
  > "$EVIDENCE_DIR/publication_artifact.sha256"
STATUS="pass"
PHASE="local-smoke-runtime-finalized"
emit_runtime_manifest
STATUS="failed"
PHASE="local-smoke-evidence-commit"
python3 - "$REPO" "$STATIC_MANIFEST" "$INITIAL_STATIC_MANIFEST_SHA256" \
  "$EVIDENCE_DIR" "$MODE" "$RUNTIME_MANIFEST" <<'PY'
import datetime, hashlib, json, os, pathlib, stat, subprocess, sys, tempfile
repo = pathlib.Path(sys.argv[1])
static = pathlib.Path(sys.argv[2])
initial_static_digest = sys.argv[3]
evidence = pathlib.Path(sys.argv[4])
binding_repo = pathlib.Path(os.environ.get("RINGLPN_ORIGINAL_REPO", str(repo)))
mode = sys.argv[5]
runtime_manifest = pathlib.Path(sys.argv[6])
if runtime_manifest.parent.resolve() != evidence.resolve():
    raise SystemExit("runtime manifest must be a direct child of the evidence directory")
static_payload = static.read_bytes()
if hashlib.sha256(static_payload).hexdigest() != initial_static_digest:
    raise SystemExit("static publication manifest mutated after preflight")
m = json.loads(static_payload)
files = []
runtime_payload = runtime_manifest.read_bytes()
files.append({
    "path": runtime_manifest.name,
    "bytes": len(runtime_payload),
    "sha256": hashlib.sha256(runtime_payload).hexdigest(),
    "retention": "final-runtime-manifest",
})
for binding in m["required_tracked_evidence"]:
    rel, expected_digest = binding["path"], binding["sha256"]
    p = binding_repo / rel
    payload = p.read_bytes()
    observed_digest = hashlib.sha256(payload).hexdigest()
    if observed_digest != expected_digest:
        raise SystemExit("required immutable tracked evidence digest differs: " + rel)
    entry = {
        "path": rel,
        "bytes": len(payload),
        "sha256": observed_digest,
        "retention": "tracked",
        "integrity": "pinned",
    }
    files.append(entry)
for name in ("paper-checkpoint-smoke.log", "publication_artifact.sha256"):
    p = evidence / name
    metadata = p.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("external public artifact is absent or non-regular: " + name)
    payload = p.read_bytes()
    files.append({
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "retention": "external-public",
    })
expected_fresh_graph_files = {
    "INDEX.json", "FULL_GRAPH.manifest", "adapter.csv", "party0.csv",
    "party1.csv", "checker.csv", "controls.csv", "adapter.log", "party0.log",
    "party1.log", "checker.log", "linear/LINEAR_RECORD_SET.manifest",
    "linear/private_inputs/binary_approval.json",
    "linear/private_inputs/linear_adapter_build_provenance.json",
    "private_inputs/graph_binary_approval.json",
    "private_inputs/graph_build_provenance.json",
    "private_inputs/linear_binary_approval.json",
    "private_inputs/linear_adapter_build_provenance.json",
}
external_root = evidence / "fresh-full-graph"
if external_root.is_symlink() or not external_root.is_dir():
    raise SystemExit("fresh full-graph evidence root is absent or symlinked")
observed_fresh_graph_files = set()
expected_fresh_graph_directories = {
    "linear", "linear/private_inputs", "private_inputs",
}
observed_fresh_graph_directories = set()
for p in sorted(external_root.rglob("*")):
    relative = p.relative_to(external_root).as_posix()
    metadata = p.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(
            "fresh full-graph evidence contains a symlink: " + relative
        )
    if stat.S_ISDIR(metadata.st_mode):
        observed_fresh_graph_directories.add(relative)
        continue
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(
            "fresh full-graph evidence contains a non-regular artifact: "
            + relative
        )
    observed_fresh_graph_files.add(relative)
    payload = p.read_bytes()
    files.append({
        "path": str(p.relative_to(evidence)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "retention": "fresh-full-graph",
    })
if observed_fresh_graph_directories != expected_fresh_graph_directories:
    raise SystemExit("fresh full-graph retained directory inventory differs")
if observed_fresh_graph_files != expected_fresh_graph_files:
    unexpected = sorted(
        observed_fresh_graph_files - expected_fresh_graph_files
    )
    missing = sorted(expected_fresh_graph_files - observed_fresh_graph_files)
    raise SystemExit(
        "fresh full-graph retained evidence inventory differs; "
        "unapproved=" + "|".join(unexpected[:20])
        + "; missing=" + "|".join(missing[:20])
    )
release = m.get("source_release")
source_tag = release.get("required_annotated_tag") if isinstance(release, dict) else None
tag_commit = None
if isinstance(source_tag, str):
    resolved = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"refs/tags/{source_tag}^{{commit}}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if resolved.returncode == 0:
        tag_commit = resolved.stdout.strip()
out = {
  "schema": "ringlpn-publication-evidence/v3",
  "classification": "internal/advisor",
  "mode": mode,
  "status": "pass",
  "repository_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
  "static_manifest_sha256": initial_static_digest,
  "source_date_epoch": int(os.environ["SOURCE_DATE_EPOCH"]),
  "source_release_tag": source_tag,
  "source_release_tag_commit": tag_commit,
  "source_authorization": None,
  "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
  "files": files,
  "secrets_or_private_data_recorded": False
}
canonical = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
out["manifest_digest"] = hashlib.sha256(canonical).hexdigest()
payload = (json.dumps(out, indent=2, sort_keys=True) + "\n").encode()
descriptor, temporary_name = tempfile.mkstemp(
    prefix=".evidence-manifest.", suffix=".tmp", dir=evidence
)
target = None
try:
    with os.fdopen(descriptor, "wb") as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.chmod(temporary_name, 0o400)
    target = evidence / "evidence-manifest.json"
    try:
        os.link(temporary_name, target, follow_symlinks=False)
    except FileExistsError as error:
        raise SystemExit("evidence manifest destination already exists") from error
    os.unlink(temporary_name)
    directory_fd = os.open(evidence, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
    if target is not None:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    raise
PY
RUNTIME_FINALIZED=1
trap - EXIT
STATUS="pass"
PHASE="local-smoke-complete"
echo "[ringlpn-reproduce] LOCAL SMOKE PASS — NOT TWO-HOST PUBLICATION EVIDENCE"
