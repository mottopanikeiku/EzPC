#!/usr/bin/env python3
"""Audit mechanically valid calls to the finite-field LPN estimator.

This script does NOT pin Ring-LPN parameters or establish a security level.
The deployed sparse-factor projection has dependent noise and a fully split
quasi-cyclic structure. No reduction in this repository justifies replacing it
with the estimator's exact-weight or regular-noise finite-field input models,
and the accepted estimator does not analyze quasi-cyclic/DOOM effects.

The output is therefore diagnostic only: for each candidate it reports the
minimum finite-field *model* cost among mechanically defined projected tuples,
over both deployed primes and both estimator noise models. A missing valid
result for either prime rejects the row. The process exits nonzero after
printing the diagnostics so automation cannot mistake a high model cost for a
parameter pin.

For an estimator call ``analysisforq(n, k, t, q)`` to be mechanically defined,
its internal binomial coefficients require ``0 <= t <= n-k-1``. This check is
performed before every call. Earlier versions omitted it and produced invalid
2026-07-29 values for projected tuples including ``(n,k,t)=(128,96,111)``.

Estimator source: IACR EUROCRYPT 2024 artifact 2024/a1 ``lpn-estimator.py``
(accepted artifact, MIT), pinned by SHA-256. Its output is evidence about that
model only, not about this Ring-LPN construction.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ringlpn_projection_security import (  # noqa: E402
    PRIMES,
    load_estimator,
    reduced_weight,
)


def finite_field_model_bits(
    estimator, ring_logn: int, c: int, per_poly_t: int, prime: int
):
    """Return (bits, degree, model) for the cheapest defined model tuple.

    The ``expected <= lpn_k`` filter retains only BCG+20's potentially useful
    sparse-factor projections. The stricter integer-weight bound enforces the
    accepted estimator's actual combinatorial domain. Neither filter proves
    that the projected dependent-noise distribution reduces to either model.
    """
    total_w = c * per_poly_t
    best = None
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
        max_estimator_t = lpn_n - lpn_k - 1
        if (
            expected <= 0
            or expected > lpn_k
            or estimator_t > max_estimator_t
        ):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            exact = float(estimator.analysisforq(lpn_n, lpn_k, estimator_t, prime))
            regular = float(
                estimator.analysisforqregular(lpn_n, lpn_k, estimator_t, prime)
            )
        for model, bits in (("exact", exact), ("regular", regular)):
            if not math.isfinite(bits):
                continue
            if best is None or bits < best[0]:
                best = (bits, degree, model)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", required=True, type=Path)
    parser.add_argument("--reference-bits", type=float, default=128.0)
    parser.add_argument(
        "--logn", type=int, nargs="+", default=[13, 14, 16, 18, 20, 22]
    )
    parser.add_argument("--c", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--t", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--dmpf-slots-per-function",
        type=int,
        default=3,
        help="setup slots a DMPF encoder consumes per packed multi-point function",
    )
    args = parser.parse_args()
    estimator = load_estimator(args.estimator)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "ring_logn",
            "n",
            "c",
            "per_poly_t",
            "total_w",
            "raw_points",
            "finite_field_model_bits",
            "worst_prime",
            "worst_degree",
            "worst_model",
            "model_bits_ge_reference",
            "point_dpf_setup_slots",
            "point_dpf_bootstrap_ok",
            "point_dpf_net_slots",
            "encoder_indep_setup_slots_placeholder",
            "encoder_indep_bootstrap_ok",
        )
    )

    emitted = 0
    for ring_logn in args.logn:
        n = 1 << ring_logn
        for c in args.c:
            for per_poly_t in args.t:
                per_prime = []
                for _limb, prime in PRIMES:
                    got = finite_field_model_bits(
                        estimator, ring_logn, c, per_poly_t, prime
                    )
                    if got is not None:
                        per_prime.append((got[0], prime, got[1], got[2]))
                if len(per_prime) != len(PRIMES):
                    continue
                worst = min(per_prime)
                bits, prime, degree, model = worst
                raw_points = c * c * per_poly_t * per_poly_t
                point_slots = 3 * raw_points
                point_net = n - point_slots
                # Placeholder only: a DMPF's real distributed-generation cost is
                # not measured anywhere yet. What is structural is that it scales
                # with the number of packed functions (c^2), not with t^2.
                dmpf_slots = args.dmpf_slots_per_function * c * c
                row = (
                    ring_logn,
                    n,
                    c,
                    per_poly_t,
                    c * per_poly_t,
                    raw_points,
                    f"{bits:.6f}",
                    prime,
                    degree,
                    model,
                    "yes" if bits >= args.reference_bits else "no",
                    point_slots,
                    "yes" if point_slots < n else "no",
                    point_net,
                    f"{100.0 * point_net / n:.3f}",
                    dmpf_slots,
                    "yes" if dmpf_slots < n else "no",
                )
                writer.writerow(row)
                emitted += 1

    print(
        f"# DIAGNOSTIC ONLY: emitted {emitted} mechanically defined finite-field "
        "model rows; no Ring-LPN security level or parameter set is pinned",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
