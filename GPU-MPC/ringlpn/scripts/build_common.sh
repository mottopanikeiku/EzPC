#!/usr/bin/env bash
# Shared, side-effect-free helpers for Ring-LPN component build wrappers.

ringlpn_build_init() {
  RINGLPN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  RINGLPN_PROJECT_ROOT="$(cd "$RINGLPN_ROOT/.." && pwd)"
  RINGLPN_REPO_ROOT="$(cd "$RINGLPN_ROOT/../.." && pwd)"
  RINGLPN_SCI_SRC="$RINGLPN_REPO_ROOT/SCI/src"
  RINGLPN_OUT_DIR="$RINGLPN_ROOT/bin"
  RINGLPN_HOST_OUT_DIR="$RINGLPN_ROOT/host_bin"
  RINGLPN_BUILD_DIR="$RINGLPN_ROOT/build"
}

ringlpn_require_command() {
  local variable_name="$1"
  local default_command="$2"
  local installation_hint="$3"
  local requested="${!variable_name:-$default_command}"
  local resolved
  resolved="$(command -v "$requested" 2>/dev/null || true)"
  if [[ -z "$resolved" ]]; then
    printf 'missing build tool %s (%s). %s\n' "$requested" "$variable_name" "$installation_hint" >&2
    return 1
  fi
  printf -v "$variable_name" '%s' "$resolved"
}

ringlpn_require_source() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    printf 'missing %s: %s\n' "$description" "$path" >&2
    printf 'initialize the clean clone with: git submodule update --init --recursive\n' >&2
    return 1
  fi
}

ringlpn_require_linear_sources() {
  ringlpn_require_source "$RINGLPN_PROJECT_ROOT/ext/cutlass/include/cutlass/cutlass.h" 'CUTLASS submodule'
  ringlpn_require_source "$RINGLPN_PROJECT_ROOT/ext/sytorch/include/sytorch/tensor.h" 'Sytorch source tree'
  ringlpn_require_source "$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/cryptoTools/cryptoTools/Crypto/PRNG.h" 'cryptoTools source tree'
  ringlpn_require_source "$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/llama/include/llama/api.h" 'LLAMA source tree'
  ringlpn_require_source "$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/bitpack/include/bitpack/bitpack.h" 'bitpack source tree'
  ringlpn_require_source "$RINGLPN_SCI_SRC/OT/ot.h" 'SCI source tree'
}

ringlpn_set_cuda_include_flags() {
  local -n destination="$1"
  destination=(
    -I"$RINGLPN_PROJECT_ROOT"
    -I"$RINGLPN_PROJECT_ROOT/ext/cutlass/include"
    -I"$RINGLPN_PROJECT_ROOT/ext/cutlass/tools/util/include"
    -I"$RINGLPN_PROJECT_ROOT/ext/sytorch/include"
    -I"$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/cryptoTools"
    -I"$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/llama/include"
    -I"$RINGLPN_PROJECT_ROOT/ext/sytorch/ext/bitpack"
    -I"$RINGLPN_SCI_SRC"
    -I"$RINGLPN_ROOT/src"
  )
}

ringlpn_prepare_build_root() {
  if ! mkdir -p -- "$RINGLPN_BUILD_DIR"; then
    printf 'cannot create build output root: %s\n' "$RINGLPN_BUILD_DIR" >&2
    return 1
  fi
  if [[ ! -O "$RINGLPN_BUILD_DIR" ]]; then
    printf 'build output root is not owned by uid %s: %s\n' "$(id -u)" "$RINGLPN_BUILD_DIR" >&2
    printf 'remove root-owned output and rebuild as an unprivileged user\n' >&2
    return 1
  fi
}

ringlpn_acquire_build_lock() {
  local lock_dir="$1"
  if ! mkdir -- "$lock_dir"; then
    printf 'stale or concurrent build root: %s\n' "$lock_dir" >&2
    printf 'remove it only after confirming that no component build is active\n' >&2
    return 1
  fi
  RINGLPN_ACTIVE_BUILD_LOCK="$lock_dir"
}

ringlpn_release_build_lock() {
  if [[ -n "${RINGLPN_ACTIVE_BUILD_LOCK:-}" ]]; then
    rmdir -- "$RINGLPN_ACTIVE_BUILD_LOCK"
    RINGLPN_ACTIVE_BUILD_LOCK=
  fi
}
