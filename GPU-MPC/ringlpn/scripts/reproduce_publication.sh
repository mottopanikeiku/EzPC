#!/usr/bin/env bash
# Clean-clone/container entry point for the dated internal Ring-LPN artifact.
# It never records argv or environment values: SSH material and private data stay external.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
STATIC_MANIFEST="$ROOT/scripts/publication_environment_manifest_2026_08_04.json"
RUNTIME_MANIFEST="${RINGLPN_RUNTIME_MANIFEST:-/tmp/ringlpn-reproduction-manifest.json}"
EVIDENCE_DIR="${RINGLPN_EVIDENCE_DIR:-}"
MODE="${1:-}"
STATUS="failed"
PHASE="initializing"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
shift || true

emit_runtime_manifest() {
  local rc=$?
  RINGLPN_EMIT_RC="$rc" RINGLPN_EMIT_STATUS="$STATUS" RINGLPN_EMIT_PHASE="$PHASE" \
  RINGLPN_EMIT_STARTED="$STARTED" RINGLPN_EMIT_MODE="$MODE" \
  RINGLPN_EMIT_REPO="$REPO" RINGLPN_EMIT_STATIC="$STATIC_MANIFEST" \
  python3 - "$RUNTIME_MANIFEST" <<'PY'
import datetime, hashlib, json, os, pathlib, platform, subprocess, sys

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
repo = pathlib.Path(os.environ["RINGLPN_EMIT_REPO"])
static = pathlib.Path(os.environ["RINGLPN_EMIT_STATIC"])
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
data = {
  "schema": "ringlpn-publication-runtime/v1",
  "classification": "internal/advisor",
  "mode": os.environ["RINGLPN_EMIT_MODE"],
  "status": os.environ["RINGLPN_EMIT_STATUS"],
  "failed_or_completed_phase": os.environ["RINGLPN_EMIT_PHASE"],
  "exit_code": int(os.environ["RINGLPN_EMIT_RC"]),
  "started_utc": os.environ["RINGLPN_EMIT_STARTED"],
  "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
  "repository_revision": command("git", "-C", str(repo), "rev-parse", "HEAD"),
  "static_manifest_sha256": hashlib.sha256(static.read_bytes()).hexdigest() if static.is_file() else None,
  "container_base_digest": os.environ.get("RINGLPN_REPRO_CONTAINER_DIGEST"),
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
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
PY
  echo "[ringlpn-reproduce] runtime manifest: $RUNTIME_MANIFEST" >&2
  return "$rc"
}
trap emit_runtime_manifest EXIT

fail() { echo "[ringlpn-reproduce] FAIL: $*" >&2; exit 2; }
require_command() { command -v "$1" >/dev/null 2>&1 || fail "required command missing: $1"; }

case "$MODE" in
  check|local-smoke|two-host-publication|remote-build) ;;
  *) fail "usage: $0 {check|local-smoke|two-host-publication} [isolation options] [-- authenticated-launcher options]" ;;
esac
if [[ "$MODE" != two-host-publication && $# -ne 0 ]]; then
  fail "$MODE accepts no arguments"
fi

PHASE="container-boundary"
[[ -f /.dockerenv ]] || fail "run only in scripts/Dockerfile.reproduction; host execution is not publication evidence"
[[ ${EUID:-$(id -u)} -ne 0 ]] || fail "refusing root: bind-mounted outputs would be root-owned"
[[ "${RINGLPN_REPRO_CONTAINER_DIGEST:-}" == "sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade" ]] ||
  fail "container base digest is absent or not pinned"
[[ -f "$STATIC_MANIFEST" ]] || fail "immutable environment manifest missing"
for tool in git python3 sha256sum dpkg-query nvcc gcc g++ cmake nvidia-smi; do require_command "$tool"; done
if [[ "$MODE" != "remote-build" ]]; then require_command pdflatex; fi

PHASE="clean-clone"
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || fail "repository is dirty or contains untracked files"
python3 - "$REPO" "$STATIC_MANIFEST" <<'PY'
import hashlib, json, pathlib, subprocess, sys
repo, manifest_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
m = json.loads(manifest_path.read_text())
expected = {x["path"]: x["revision"] for x in m["sources"] if x["path"] != "GPU-MPC/ringlpn/src/bench_ntt_cuda_cheddar.cu"}
out = subprocess.check_output(["git", "-C", str(repo), "submodule", "status", "--recursive"], text=True)
seen = {}
for line in out.splitlines():
    if not line or line[0] != " ":
        raise SystemExit("missing, conflicted, or wrong-revision submodule: " + line)
    fields = line[1:].split()
    seen[fields[1]] = fields[0]
if seen != expected:
    raise SystemExit("recursive submodule set/revisions differ from immutable manifest")
for path, revision in expected.items():
    p = repo / path
    status = subprocess.check_output(["git", "-C", str(p), "status", "--porcelain", "--untracked-files=all"], text=True)
    if status:
        raise SystemExit("dirty submodule: " + path)
for source in m["sources"]:
    lf, want = source.get("license_file"), source.get("license_sha256")
    if lf and not (repo / lf).is_file():
        raise SystemExit("missing retained license: " + lf)
    if lf and want and hashlib.sha256((repo / lf).read_bytes()).hexdigest() != want:
        raise SystemExit("license checksum mismatch: " + lf)
for path in m["required_tracked_evidence"]:
    subprocess.check_call(["git", "-C", str(repo), "ls-files", "--error-unmatch", path], stdout=subprocess.DEVNULL)
PY

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
  STATUS="pass"; PHASE="preflight-complete"
  echo "[ringlpn-reproduce] PREFLIGHT PASS (no build or gate run)"
  exit 0
fi

PHASE="evidence-output"
[[ -n "$EVIDENCE_DIR" && "$EVIDENCE_DIR" == /* ]] || fail "RINGLPN_EVIDENCE_DIR must be an absolute, external mounted directory"
case "$(realpath -m "$EVIDENCE_DIR")" in "$REPO"|"$REPO"/*) fail "evidence directory must be outside the source clone";; esac
mkdir -p "$EVIDENCE_DIR"
[[ -w "$EVIDENCE_DIR" ]] || fail "evidence directory is not writable"

if [[ "$MODE" == remote-build ]]; then
  PHASE="remote-binary-build"
  GPU_ARCH=89 CUDA_ARCH=89 "$ROOT/scripts/build_two_party_fc_preprocess.sh"
  sha256sum "$ROOT/bin/test_two_party_fc_preprocess" > "$EVIDENCE_DIR/remote_fc_binary.sha256"
  STATUS="pass"; PHASE="remote-build-complete"
  echo "[ringlpn-reproduce] REMOTE BUILD PASS"
  exit 0
fi
if [[ "$MODE" == two-host-publication ]]; then
  PHASE="two-host-arguments"
  p0_manifest=""; p1_manifest=""; checker_manifest=""; commit_manifest=""; auth_args=()
  while (($#)); do
    case "$1" in
      --p0-isolation-manifest) (($# >= 2)) || fail "missing p0 isolation manifest"; p0_manifest="$2"; shift 2;;
      --p1-isolation-manifest) (($# >= 2)) || fail "missing p1 isolation manifest"; p1_manifest="$2"; shift 2;;
      --checker-isolation-manifest) (($# >= 2)) || fail "missing checker isolation manifest"; checker_manifest="$2"; shift 2;;
      --commit-isolation-manifest) (($# >= 2)) || fail "missing committed isolation manifest"; commit_manifest="$2"; shift 2;;
      --) shift; auth_args=("$@"); break;;
      *) fail "unknown reproduction option before --: $1";;
    esac
  done
  [[ -n "$p0_manifest" && -n "$p1_manifest" && -n "$checker_manifest" && -n "$commit_manifest" ]] || fail "party, checker, and durable commit isolation manifests are required"
  ((${#auth_args[@]} > 0)) || fail "authenticated two-host launcher arguments are required after --"
  [[ -x "$ROOT/scripts/run_two_host_authenticated.sh" ]] || fail "authenticated two-host launcher missing"
  [[ -f "$ROOT/scripts/peer_private_execution.py" ]] || fail "peer isolation verifier missing"
fi

PHASE="publication-pdf"
REPORT_DIR="$ROOT/results/reports"
TEX="dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex"
(
  cd "$REPORT_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX"
  rm -f "${TEX%.tex}.aux" "${TEX%.tex}.log" "${TEX%.tex}.out"
)

PHASE="canonical-component-gates"
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 RUN_REGULAR_SMOKE=1 GPU_ARCH=89 CUDA_ARCH=89 \
  "$ROOT/scripts/run_paper_checkpoint_smoke.sh"

if [[ "$MODE" == local-smoke ]]; then
  PHASE="local-smoke-gates-complete"
else

  PHASE="composed-live-fc-gates"
  P0_GPU="${P0_GPU:-1}" P1_GPU="${P1_GPU:-3}" CHECK_GPU="${CHECK_GPU:-1}" \
    "$ROOT/scripts/run_two_party_fc_preprocess.sh"
  PHASE="classifier-model-scale-gate"
  P0_GPU="${P0_GPU:-1}" P1_GPU="${P1_GPU:-3}" CHECK_GPU="${CHECK_GPU:-1}" \
    MODELS=ResNet18 WORKLOAD=classifier WARMUPS=1 TRIALS=10 \
    "$ROOT/scripts/run_two_party_fc_model_scale.sh"
  PHASE="local-live-binary-build"
  GPU_ARCH=89 CUDA_ARCH=89 "$ROOT/scripts/build_two_party_fc_preprocess.sh"
  PHASE="authenticated-two-host-gate"
  "$ROOT/scripts/run_two_host_authenticated.sh" "${auth_args[@]}"
  PHASE="peer-isolation-gate"
  python3 "$ROOT/scripts/peer_private_execution.py" verify-manifest \
    --p0-manifest "$p0_manifest" --p1-manifest "$p1_manifest" \
    --checker-manifest "$checker_manifest" --commit-manifest "$commit_manifest"
  PHASE="two-host-gates-complete"
fi

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

PHASE="retain-ignored-evidence"
mapfile -t ignored < <(git -C "$REPO" ls-files --others --ignored --exclude-standard -- GPU-MPC/ringlpn/results)
retained=()
for rel in "${ignored[@]}"; do
  case "$rel" in
    *.key|*.testmeta|*.spfss|*.noise|*.convert|*.bin|*.fc|*/sqlite-db/*) continue;;
  esac
  dest="$EVIDENCE_DIR/ignored-evidence/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -p "$REPO/$rel" "$dest"
  cmp -s "$REPO/$rel" "$dest" || fail "ignored evidence was not retained byte-for-byte: $rel"
  retained+=("$rel")
done
printf '%s\n' "${retained[@]}" > "$EVIDENCE_DIR/ignored-evidence-files.txt"
sha256sum "$STATIC_MANIFEST" "$REPORT_DIR/${TEX%.tex}.pdf" > "$EVIDENCE_DIR/publication_artifact.sha256"
python3 - "$REPO" "$STATIC_MANIFEST" "$EVIDENCE_DIR" "$MODE" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
repo, static, evidence, mode = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
m = json.loads(static.read_text())
files = []
for rel in m["required_tracked_evidence"]:
    p = repo / rel
    files.append({"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "retention": "tracked"})
ignored_root = evidence / "ignored-evidence"
if ignored_root.is_dir():
    for p in sorted(x for x in ignored_root.rglob("*") if x.is_file()):
        files.append({"path": str(p.relative_to(ignored_root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "retention": "external-copy"})
out = {
  "schema": "ringlpn-publication-evidence/v1",
  "classification": "internal/advisor",
  "mode": mode,
  "repository_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
  "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
  "files": files,
  "secrets_or_private_data_recorded": False
}
tmp = evidence / "evidence-manifest.json.tmp"
tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
tmp.replace(evidence / "evidence-manifest.json")
PY
STATUS="pass"
if [[ "$MODE" == local-smoke ]]; then
  PHASE="local-smoke-complete"
  echo "[ringlpn-reproduce] LOCAL SMOKE PASS — NOT TWO-HOST PUBLICATION EVIDENCE"
else
  PHASE="two-host-publication-complete"
  echo "[ringlpn-reproduce] TWO-HOST PUBLICATION GATES PASS"
fi
