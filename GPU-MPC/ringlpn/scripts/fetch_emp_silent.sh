#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${RINGLPN_EMP_DEPS_ROOT:-$ROOT/.deps/emp-silent}"
EMP_TOOL_REV="97f335927dd7d38caaf5e80d93fca70edddd5423"
EMP_OT_REV="2fca139ff1974c039422af545bd4681e8d55acc1"
EMP_TOOL_LICENSE_SHA256="28a151e380aff8a26ffb5e3367a02cb7fb6da8a45f77cfd29bfb2afaa3a9d1f9"
EMP_OT_LICENSE_SHA256="91c094b763419adeca1545e05f381a86746e059f618452b978707432981ffd81"

fetch_exact() {
  local name="$1" url="$2" revision="$3" license_sha="$4"
  local checkout="$DEPS_ROOT/src/$name"
  if [[ ! -d "$checkout/.git" ]]; then
    rm -rf "$checkout"
    mkdir -p "$checkout"
    git -C "$checkout" init -q
    git -C "$checkout" remote add origin "$url"
  fi
  git -C "$checkout" fetch -q --depth=1 origin "$revision"
  git -C "$checkout" checkout -q --detach FETCH_HEAD
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual" != "$revision" ]]; then
    echo "$name revision mismatch: expected $revision, got $actual" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]]; then
    echo "$name checkout has modified tracked files" >&2
    exit 1
  fi
  local actual_license
  actual_license="$(sha256sum "$checkout/LICENSE" | cut -d' ' -f1)"
  if [[ "$actual_license" != "$license_sha" ]]; then
    echo "$name LICENSE mismatch: expected Apache-2.0 file sha256 $license_sha" >&2
    exit 1
  fi
  if ! grep -q "Apache License" "$checkout/LICENSE"; then
    echo "$name LICENSE is not the expected Apache-2.0 text" >&2
    exit 1
  fi
  printf '%s %s license-sha256=%s\n' "$name" "$actual" "$actual_license"
}

mkdir -p "$DEPS_ROOT/src"
fetch_exact emp-tool https://github.com/emp-toolkit/emp-tool.git \
  "$EMP_TOOL_REV" "$EMP_TOOL_LICENSE_SHA256"
fetch_exact emp-ot https://github.com/emp-toolkit/emp-ot.git \
  "$EMP_OT_REV" "$EMP_OT_LICENSE_SHA256"
