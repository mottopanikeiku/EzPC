#!/usr/bin/env bash
# Build the canonical live two-process Ring-LPN -> Orca forward-Conv2D artifact.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RINGLPN_LINEAR_KIND=conv exec "$ROOT/build_two_party_fc_preprocess.sh"
