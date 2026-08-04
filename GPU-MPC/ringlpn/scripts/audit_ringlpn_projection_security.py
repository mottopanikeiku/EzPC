#!/usr/bin/env python3
"""Emit mechanically valid finite-field projection-model diagnostics.

Estimator source: IACR EUROCRYPT 2024 artifact 2024/a1, ``lpn-estimator.py``
(accepted artifact, MIT): https://artifacts.iacr.org/eurocrypt/2024/a1/

This script deliberately does not turn projected model costs into a Ring-LPN
security claim. The sparse-factor projection has dependent noise and a fully
split quasi-cyclic structure for which this repository has no reduction to the
estimator's finite-field models. Calls outside the estimator's combinatorial
domain are omitted explicitly.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import sys

ESTIMATOR_SHA256 = "c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc"
PRIMES = (
    ("p0", 4611686018326724609),
    ("p1", 4611686018309947393),
)
CONFIGS = (
    ("smoke_c2_t8", 13, 2, 8),
    ("literature_c4_t16", 14, 4, 16),
    ("current_c4_t64", 13, 4, 64),
    ("alternative_c8_t8", 14, 8, 8),
)


def load_estimator(path: Path):
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ESTIMATOR_SHA256:
        raise SystemExit(
            f"estimator checksum mismatch: expected {ESTIMATOR_SHA256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("eurocrypt_2024_lpn_estimator", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load estimator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reduced_weight(total_w: int, c: int, degree: int, per_poly_t: int) -> float:
    return (
        total_w
        - c * degree
        + (c * (degree - 1) + total_w)
        * (1 - 1 / degree) ** (per_poly_t - 1)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimator",
        required=True,
        type=Path,
        help="path to the pinned EUROCRYPT 2024 artifact lpn-estimator.py",
    )
    args = parser.parse_args()
    estimator = load_estimator(args.estimator)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "config",
            "limb",
            "prime",
            "ring_logn",
            "c",
            "per_poly_t",
            "total_w",
            "projection_i",
            "factor_degree",
            "lpn_N",
            "lpn_k",
            "expected_reduced_weight",
            "estimator_t_floor",
            "exact_attack_bits",
            "regular_attack_bits",
        )
    )

    for limb, prime in PRIMES:
        for name, ring_logn, c, per_poly_t in CONFIGS:
            total_w = c * per_poly_t
            for projection_i in range(1, ring_logn + 1):
                degree = 1 << projection_i
                expected = (
                    float(total_w)
                    if projection_i == ring_logn
                    else reduced_weight(total_w, c, degree, per_poly_t)
                )
                estimator_t = max(1, math.floor(expected))
                lpn_n = c * degree
                lpn_k = (c - 1) * degree
                if (
                    expected <= 0
                    or estimator_t > lpn_n - lpn_k - 1
                ):
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    exact = estimator.analysisforq(lpn_n, lpn_k, estimator_t, prime)
                    regular = estimator.analysisforqregular(
                        lpn_n, lpn_k, estimator_t, prime
                    )
                writer.writerow(
                    (
                        name,
                        limb,
                        prime,
                        ring_logn,
                        c,
                        per_poly_t,
                        total_w,
                        projection_i,
                        degree,
                        lpn_n,
                        lpn_k,
                        f"{expected:.12f}",
                        estimator_t,
                        f"{float(exact):.12f}",
                        f"{float(regular):.12f}",
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
