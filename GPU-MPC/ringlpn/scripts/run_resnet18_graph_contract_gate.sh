#!/usr/bin/env bash
# Build the host-only contract probe and compare it with the pinned Orca graph.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"
PYTHON="${PYTHON:-python3}"
OUTPUT="$ROOT/bin/test_resnet18_graph_contract"
TEMPORARY="${OUTPUT}.tmp.$$"
PROVENANCE="$ROOT/bin/resnet18_full_graph_build_provenance.json"
GRAPH_LIBRARY_BUILD="${GRAPH_LIBRARY_BUILD:-$ROOT/build/graph-libraries}"
CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ringlpn-graph-contract.XXXXXX")"
PUBLISHED=0
cleanup() {
  status=$?
  rm -f "$TEMPORARY"
  rm -rf -- "$CONTROL_DIR"
  if [[ "$status" -ne 0 && "$PUBLISHED" -eq 0 ]]; then
    rm -f "$OUTPUT"
    "$PYTHON" -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)' \
      "$ROOT/bin" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "[resnet18-graph-contract] missing C++ compiler: $CXX" >&2
  exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[resnet18-graph-contract] missing Python interpreter: $PYTHON" >&2
  exit 2
fi
mkdir -p "$ROOT/bin"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
CONTRACT_FLAGS=(
  -O2 -std=c++17 -Wall -Wextra -Werror
  "-ffile-prefix-map=$REPO_ROOT=/ringlpn/source"
  "-fdebug-prefix-map=$REPO_ROOT=/ringlpn/source"
  -frandom-seed=ringlpn-resnet18-graph-contract-20260810
  -Wl,--build-id=none
  -I "$ROOT/src"
)
"$CXX" "${CONTRACT_FLAGS[@]}" \
  "$ROOT/src/test_resnet18_graph_contract.cpp" \
  -o "$TEMPORARY"
chmod 755 "$TEMPORARY"

"$ROOT/scripts/check_resnet18_graph_contract.py" \
  --manifest "$ROOT/results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json" \
  --contract-bin "$TEMPORARY"

cat >"$CONTROL_DIR/mismatch.py" <<'PY'
#!/usr/bin/env python3
import json
import os
import subprocess

output = subprocess.check_output(
    [os.environ["RINGLPN_REAL_CONTRACT_PROBE"]], text=True,
)
for line in output.splitlines():
    row = json.loads(line)
    if row.get("section") == "remask" and row.get("index") == 2:
        row["source"] = "__mismatch__"
    print(json.dumps(row, separators=(",", ":")))
PY
chmod 500 "$CONTROL_DIR/mismatch.py"
if RINGLPN_REAL_CONTRACT_PROBE="$TEMPORARY" \
    "$ROOT/scripts/check_resnet18_graph_contract.py" \
      --manifest "$ROOT/results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json" \
      --contract-bin "$CONTROL_DIR/mismatch.py" \
      >"$CONTROL_DIR/mismatch.stdout" 2>"$CONTROL_DIR/mismatch.stderr"; then
  echo "[resnet18-graph-contract] topology mismatch control unexpectedly passed" >&2
  exit 1
fi
grep -q 'remask dependency contract row 2 differs' \
  "$CONTROL_DIR/mismatch.stderr"
ln -s "$TEMPORARY" "$CONTROL_DIR/probe-link"
if "$ROOT/scripts/check_resnet18_graph_contract.py" \
    --manifest "$ROOT/results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json" \
    --contract-bin "$CONTROL_DIR/probe-link" \
    >"$CONTROL_DIR/symlink.stdout" 2>"$CONTROL_DIR/symlink.stderr"; then
  echo "[resnet18-graph-contract] symlink control unexpectedly passed" >&2
  exit 1
fi
grep -q 'contains a symlink component' "$CONTROL_DIR/symlink.stderr"

VERIFIED_PROVENANCE_DIGEST="$(
  "$PYTHON" "$ROOT/scripts/graph_build_provenance.py" verify \
    --repo-root "$REPO_ROOT" \
    --cmake-build "$GRAPH_LIBRARY_BUILD" \
    --manifest "$PROVENANCE"
)"
RINGLPN_CONTRACT_TEMPORARY="$TEMPORARY" \
RINGLPN_GRAPH_PROVENANCE="$PROVENANCE" \
RINGLPN_VERIFIED_PROVENANCE_DIGEST="$VERIFIED_PROVENANCE_DIGEST" \
  "$PYTHON" - <<'PY'
import hashlib
import json
import os
import pathlib
import stat


def canonical(document):
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


temporary = pathlib.Path(os.environ["RINGLPN_CONTRACT_TEMPORARY"])
provenance = json.loads(
    pathlib.Path(os.environ["RINGLPN_GRAPH_PROVENANCE"]).read_bytes()
)
binding = provenance.get("artifacts", {}).get("test_resnet18_graph_contract")
metadata = temporary.lstat()
payload = temporary.read_bytes()
claimed_provenance_digest = provenance.get("provenance_digest")
unsigned_provenance = dict(provenance)
unsigned_provenance.pop("provenance_digest", None)
if (
    claimed_provenance_digest
    != os.environ["RINGLPN_VERIFIED_PROVENANCE_DIGEST"]
    or hashlib.sha256(canonical(unsigned_provenance)).hexdigest()
    != claimed_provenance_digest
):
    raise SystemExit("graph provenance changed after verification")
if (
    not isinstance(binding, dict)
    or not stat.S_ISREG(metadata.st_mode)
    or stat.S_ISLNK(metadata.st_mode)
    or binding.get("size") != len(payload)
    or binding.get("sha256") != hashlib.sha256(payload).hexdigest()
):
    raise SystemExit(
        "rebuilt contract probe differs from verified graph provenance"
    )
PY

rm -f "$OUTPUT"
"$PYTHON" -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)' \
  "$ROOT/bin"
ln -- "$TEMPORARY" "$OUTPUT"
"$PYTHON" -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)' \
  "$ROOT/bin"
rm -f "$TEMPORARY"
PUBLISHED=1
echo "[resnet18-graph-contract] CONTROLS PASS (topology mismatch and symlink rejected; provenance-bound probe published)"
