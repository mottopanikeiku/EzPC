#!/usr/bin/env bash
# Build FC/Conv adapters twice and emit hash-bound provenance and approval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd -P)"
PRODUCER="$ROOT/scripts/linear_adapter_build_provenance.py"
PROVENANCE="${LINEAR_BUILD_PROVENANCE:-$ROOT/results/fc/linear_adapter_build_provenance_2026_08_10.json}"
APPROVAL="${LINEAR_BINARY_APPROVAL:-$ROOT/results/fc/linear_adapter_binary_approval_2026_08_07.json}"
CUDA_ARCH="${CUDA_ARCH:-${GPU_ARCH:-89}}"
NVCC="$(realpath -e -- "${NVCC:-/usr/local/cuda/bin/nvcc}")"
CXX="$(realpath -e -- "${CXX:-/usr/bin/g++}")"
OBJCOPY="$(realpath -e -- "${OBJCOPY:-/usr/bin/objcopy}")"
BUILD_PATH="${LINEAR_BUILD_PATH:-/usr/local/cuda/bin:/usr/bin:/bin}"

for tool in "$PRODUCER" "$NVCC" "$CXX" "$OBJCOPY"; do
  if [[ ! -f "$tool" || -L "$tool" ]]; then
    echo "required provenance/build tool must be a regular non-symlink: $tool" >&2
    exit 1
  fi
done

WORKDIR="$(mktemp -d /tmp/ringlpn-linear-provenance.XXXXXX)"
chmod 700 "$WORKDIR"
cleanup() {
  rm -rf -- "$WORKDIR"
}
trap cleanup EXIT
mkdir -m 700 "$WORKDIR/home"

run_adapter_build() {
  local round="$1"
  local kind="$2"
  local receipt_dir="$WORKDIR/build-$round"
  local round_tmp="$WORKDIR/tmp-$round"
  local prefix="$receipt_dir/$kind"
  mkdir -p -m 700 "$receipt_dir" "$round_tmp"
  if ! env -i \
    CUDA_ARCH="$CUDA_ARCH" \
    CXX="$CXX" \
    HOME="$WORKDIR/home" \
    LANG=C \
    LC_ALL=C \
    NVCC="$NVCC" \
    OBJCOPY="$OBJCOPY" \
    PATH="$BUILD_PATH" \
    RINGLPN_LINEAR_KIND="$kind" \
    RINGLPN_LINEAR_DEPFILE="$prefix.d" \
    RINGLPN_LINEAR_LINK_MAP="$prefix.map" \
    RINGLPN_LINEAR_COMMAND_FILE="$prefix.commands" \
    RINGLPN_LINEAR_ENVIRONMENT_FILE="$prefix.environment" \
    SOURCE_DATE_EPOCH=0 \
    TMPDIR="$round_tmp" \
    TZ=UTC \
    ZERO_AR_DATE=1 \
    "$ROOT/scripts/build_two_party_fc_preprocess.sh" \
      >"$prefix.stdout" 2>"$prefix.stderr"; then
    /usr/bin/cat -- "$prefix.stderr" >&2
    return 1
  fi
  cp -- "$ROOT/bin/test_two_party_${kind}_preprocess" "$prefix.elf"
  chmod 500 "$prefix.elf"
}

for round in 1 2; do
  run_adapter_build "$round" fc
  run_adapter_build "$round" conv
done

PROVENANCE_TEMP="$WORKDIR/provenance.json"
"$PRODUCER" generate \
  --repo-root "$REPO_ROOT" \
  --ringlpn-root "$ROOT" \
  --build-dir "$WORKDIR/build-1" \
  --build-dir "$WORKDIR/build-2" \
  --output "$PROVENANCE_TEMP"
mkdir -p "$(dirname "$PROVENANCE")" "$(dirname "$APPROVAL")"
cp -- "$PROVENANCE_TEMP" "$PROVENANCE"

APPROVAL_TEMP="$WORKDIR/approval.json"
"$PRODUCER" approve \
  --repo-root "$REPO_ROOT" \
  --ringlpn-root "$ROOT" \
  --provenance "$PROVENANCE" \
  --output "$APPROVAL_TEMP"
"$PRODUCER" verify-approval \
  --repo-root "$REPO_ROOT" \
  --ringlpn-root "$ROOT" \
  --approval "$APPROVAL_TEMP" >/dev/null
cp -- "$APPROVAL_TEMP" "$APPROVAL"
"$PRODUCER" verify-approval \
  --repo-root "$REPO_ROOT" \
  --ringlpn-root "$ROOT" \
  --approval "$APPROVAL" >/dev/null

echo "Built byte-identical FC/Conv adapters twice."
echo "Provenance: $PROVENANCE"
echo "Approval: $APPROVAL"
