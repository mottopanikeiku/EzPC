#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
DEPS_ROOT="${RINGLPN_EMP_DEPS_ROOT:-$ROOT/.deps/emp-silent}"
PREFIX="${RINGLPN_EMP_PREFIX:-$DEPS_ROOT/install}"
BRIDGE="${RINGLPN_EMP_SILENT_BRIDGE:-$PREFIX/lib/libringlpn_emp_silent_bridge.so}"
MODE="${1:---controls-only}"
BASE_PORT="${RINGLPN_EMP_TEST_PORT:-29761}"

if [[ "$MODE" != "--controls-only" && "$MODE" != "--full" ]]; then
  echo "usage: $0 [--controls-only|--full]" >&2
  exit 2
fi
if [[ ! -f "$BRIDGE" ]]; then
  "$ROOT/scripts/build_emp_silent_bridge.sh"
fi
python3 - "$BRIDGE" "$ROOT/src/emp_silent_bridge_authorization.h" <<'PY'
import hashlib
import os
import pathlib
import re
import stat
import sys

bridge = pathlib.Path(sys.argv[1])
authorization = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
match = re.search(
    r'RINGLPN_EMP_SILENT_BRIDGE_SHA256\s*\\\s*"([0-9a-f]{64})"',
    authorization,
)
if match is None:
    raise SystemExit("invalid source EMP bridge authorization")
metadata = bridge.lstat()
if (not bridge.is_absolute() or stat.S_ISLNK(metadata.st_mode) or
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or
        metadata.st_mode & 0o222 or metadata.st_uid not in {0, os.geteuid()}):
    raise SystemExit(
        "EMP bridge must be an absolute, read-only, single-link regular file "
        "owned by the effective user or root"
    )
measured = hashlib.sha256(bridge.read_bytes()).hexdigest()
if measured != match.group(1):
    raise SystemExit("EMP bridge SHA-256 is not source-authorized")
PY
mkdir -p "$ROOT/host_bin"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra \
  -maes -msse4.1 -mpclmul -mavx2 -mrdseed \
  -I"$SCI_SRC" -I"$ROOT/src" \
  "$ROOT/src/test_emp_silent_loopback.cpp" \
  -o "$ROOT/host_bin/test_emp_silent_loopback" \
  -lcrypto -lssl -ldl -pthread


CONTROL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ringlpn-emp-bridge-controls.XXXXXX")"
cleanup_controls() {
  rm -rf -- "$CONTROL_ROOT"
}
trap cleanup_controls EXIT
MODIFIED="$CONTROL_ROOT/modified.so"
cp -- "$BRIDGE" "$MODIFIED"
chmod u+w "$MODIFIED"
python3 - "$MODIFIED" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.stat().st_size == 0:
    raise SystemExit("cannot modify empty bridge control")
with path.open("ab") as output:
    output.write(b"\0")
PY
chmod a-w "$MODIFIED"
if "$ROOT/host_bin/test_emp_silent_loopback" \
    "$MODIFIED" --controls-only "$BASE_PORT" >/dev/null 2>&1; then
  echo "one-byte-modified EMP bridge was accepted" >&2
  exit 1
fi
ln -s -- "$BRIDGE" "$CONTROL_ROOT/symlink.so"
if "$ROOT/host_bin/test_emp_silent_loopback" \
    "$CONTROL_ROOT/symlink.so" --controls-only "$BASE_PORT" >/dev/null 2>&1; then
  echo "symlink EMP bridge was accepted" >&2
  exit 1
fi

"$ROOT/host_bin/test_emp_silent_loopback" "$BRIDGE" "$MODE" "$BASE_PORT"
