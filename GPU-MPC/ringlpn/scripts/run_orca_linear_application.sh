#!/usr/bin/env bash
# Build and run the source-native terminal Ring-LPN Orca application gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/cuda/bin:${PATH}"
export GPU_ARCH="${GPU_ARCH:-89}"

if [[ "${ORCA_LINEAR_SKIP_BUILD:-0}" != "1" ]]; then
  "$ROOT/scripts/build_component.sh" orca-linear-application
fi

ARGS=(
  --root "$ROOT"
  --p0-gpu "${P0_GPU:-0}"
  --p1-gpu "${P1_GPU:-1}"
)
if [[ -n "${WORKDIR:-}" ]]; then
  ARGS+=(--work-root "$WORKDIR")
fi
exec "${PYTHON:-python3}" "$ROOT/scripts/run_orca_linear_application.py" \
  "${ARGS[@]}" "$@"
