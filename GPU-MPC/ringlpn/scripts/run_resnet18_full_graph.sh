#!/usr/bin/env bash
# Build and produce one exact known-zero full ResNet18 graph checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 ABS_OUTPUT_ROOT ABS_STATE_ROOT [ABS_SUMMARY_ROOT]" >&2
  exit 2
fi
OUTPUT_ROOT="$1"
STATE_ROOT="$2"
SUMMARY_ROOT="${3:-}"
for path in "$OUTPUT_ROOT" "$STATE_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "Output and state roots must be absolute." >&2
    exit 2
  fi
done
if [[ -n "$SUMMARY_ROOT" && "$SUMMARY_ROOT" != /* ]]; then
  echo "Summary root must be absolute." >&2
  exit 2
fi

GRAPH_PROVENANCE="$ROOT/bin/resnet18_full_graph_build_provenance.json"
if [[ -f "$GRAPH_PROVENANCE" ]]; then
  python3 "$ROOT/scripts/graph_build_provenance.py" verify \
    --repo-root "$ROOT/../.." \
    --cmake-build "$ROOT/build/graph-libraries" \
    --manifest "$GRAPH_PROVENANCE" >/dev/null
else
  PATH="/usr/local/cuda/bin:$PATH" "$ROOT/scripts/build_resnet18_full_graph.sh"
fi

command=(
  python3 "$ROOT/scripts/run_resnet18_full_graph.py"
  --output-root "$OUTPUT_ROOT"
  --state-root "$STATE_ROOT"
  --p0-gpu "${P0_GPU:-0}"
  --p1-gpu "${P1_GPU:-1}"
  --check-gpu "${CHECK_GPU:-2}"
  --trusted-gpu "${TRUSTED_GPU:-1}"
  --linear-base-port "${LINEAR_BASE_PORT:-28800}"
  --graph-base-port "${GRAPH_BASE_PORT:-29000}"
  --timeout-seconds "${TIMEOUT_SECONDS:-86400}"
)
if [[ -n "${LINEAR_LANES:-}" ]]; then
  IFS=',' read -r -a linear_lanes <<<"$LINEAR_LANES"
  for lane in "${linear_lanes[@]}"; do
    if [[ -z "$lane" ]]; then
      echo "LINEAR_LANES contains an empty descriptor." >&2
      exit 2
    fi
    command+=(--linear-lane "$lane")
  done
fi
if [[ -n "$SUMMARY_ROOT" ]]; then
  command+=(--summary-root "$SUMMARY_ROOT")
fi
if [[ -n "${RECORD_SET_ROOT:-}" ]]; then
  command+=(--record-set-root "$RECORD_SET_ROOT")
fi
"${command[@]}"
