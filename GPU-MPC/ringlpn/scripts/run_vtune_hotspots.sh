#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$BASE_DIR/bin/bench_ntt"

N=65536
QBITS=64
ITERS=200
WARMUP=20
RESULT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)
      N="$2"
      shift 2
      ;;
    --qbits)
      QBITS="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--n N] [--qbits QBITS] [--iters ITERS] [--warmup WARMUP] [--result-dir DIR]"
      exit 1
      ;;
  esac
done

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="$BASE_DIR/results/vtune_hotspots_${N}_${QBITS}"
fi

if [[ ! -x "$BIN" ]]; then
  echo "CPU benchmark missing. Run ./scripts/build_bench.sh first."
  exit 1
fi

if ! command -v vtune >/dev/null 2>&1; then
  echo "vtune not found. Install Intel VTune / oneAPI first."
  exit 1
fi

vtune -collect hotspots -result-dir "$RESULT_DIR" "$BIN" --n "$N" --qbits "$QBITS" --iters "$ITERS" --warmup "$WARMUP"
vtune -report summary -result-dir "$RESULT_DIR"
vtune -report hotspots -result-dir "$RESULT_DIR"
