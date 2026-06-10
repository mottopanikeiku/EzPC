#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/host_bin/test_orca_zp_bridge"
CSV="$ROOT/results/orca_fc/orca_zp_bridge_constant_scalar.csv"
MD="$ROOT/results/orca_fc/orca_zp_bridge_constant_scalar.md"

if [[ ! -x "$BIN" ]]; then
  "$ROOT/scripts/build_orca_zp_bridge_test.sh"
fi

mkdir -p "$ROOT/results/orca_fc"

"$BIN" --csv-header --bw 16 --rows 2 --inner 2 --cols 2 \
  --value-bound 255 --trials 1000 --forced-wraps 128 --seed 1 > "$CSV"

"$BIN" --qbits 64 --bw 32 --rows 1 --inner 1 --cols 1 \
  --value-bound 4294967295 --trials 1000 --forced-wraps 128 --seed 1 >> "$CSV"

"$BIN" --qbits 128 --bw 32 --rows 2 --inner 2 --cols 2 \
  --value-bound 4294967295 --trials 1000 --forced-wraps 128 --seed 2 >> "$CSV"

python3 - "$CSV" "$MD" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))

headers = [
    "requested_qbits",
    "actual_qbits",
    "bw",
    "rows",
    "inner",
    "cols",
    "value_bound",
    "naive_share_failures",
    "corrected_share_failures",
    "no_modulus_wrap_bound",
    "constant_scalar_matmul_validation",
    "counterexample_found",
]

with md_path.open("w") as f:
    f.write("# Orca Zp-to-Z2k Bridge Smoke\n\n")
    f.write("This host-only smoke validates the carry-corrected share conversion needed when a `Z_p` OLE/Beaver share is exported toward Orca's `Z_{2^bw}` linear-layer ring.\n\n")
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---:" if h not in {"constant_scalar_matmul_validation"} else "---" for h in headers]) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    f.write("\n")
    f.write("Interpretation:\n\n")
    f.write("- The corrected conversion subtracts the hidden prime carry `m*p` from one output share before reducing to `Z_{2^bw}`.\n")
    f.write("- Constant-polynomial scalar packing is valid only under the explicit no-wrap bound `inner * value_bound^2 < modulus`.\n")
    f.write("- The q62 `bw=32` full-range row is intentionally not claimed; it records a counterexample showing why q62 is insufficient for unrestricted 32-bit scalar products.\n")
    f.write("- The q128 row uses `M = p0*p1` and validates the bounded full-32-bit scalar case under the same dealer/oracle carry correction.\n")
PY

echo "Wrote $CSV"
echo "Wrote $MD"
