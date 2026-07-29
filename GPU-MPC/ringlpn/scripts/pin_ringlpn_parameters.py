#!/usr/bin/env python3
"""Pin splittable Ring-LPN parameters under the CONSERVATIVE reading.

Decision recorded 2026-07-29 by the project owner: where BCG+20's literal
sparse-factor projection rule and its Table 1 disagree, adopt the *minimum*
attack cost. Concretely, for a candidate ``(ring_logn, c, per_poly_t)`` this
script reports

    conservative_bits = min over every projection degree the estimator accepts,
                        and over both the exact and regular noise models,
                        of the EUROCRYPT 2024 estimator's attack cost.

That is strictly stronger than following either published reading: it assumes
the adversary picks the cheapest projection available to it, not the one whose
weight formula the paper highlights. A candidate passes only if its
conservative cost is at least the target bit level for BOTH deployed primes.

The script also reports the per-encoder bootstrap requirement, because a
parameter set that is secure but cannot pay for its own next-epoch key
generation is not usable:

    per-point DPF encoder:  C_setup = 3 * c^2 * t^2  scalar-OLE slots
    DMPF encoder:          C_setup = f * c^2         with f setup slots per
                                                     packed multi-point function

Estimator source: IACR EUROCRYPT 2024 artifact 2024/a1 ``lpn-estimator.py``
(accepted artifact, MIT), pinned by SHA-256.
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


def conservative_bits(estimator, ring_logn: int, c: int, per_poly_t: int, prime: int):
    """Return (bits, degree, model) for the cheapest *useful* projection.

    Validity uses BCG+20's own criterion: a sparse-factor projection helps the
    attacker only while the expected reduced weight still fits the reduced
    instance's dimension, ``w_i <= (c-1)*2^i``. Without that filter the sweep
    admits near-full-weight instances (for example reduced weight 124.6 in a
    dimension-128 instance at ``c=4,t=44``, degree 32), where the estimator is
    outside its modelling regime and returns implausibly low costs; that made the
    conservative minimum non-monotone in the noise weight. Non-finite estimator
    outputs are also rejected rather than silently compared.
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
        if expected <= 0 or expected > lpn_k or estimator_t >= lpn_n:
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
    parser.add_argument("--target-bits", type=float, default=128.0)
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
            "conservative_bits",
            "worst_prime",
            "worst_degree",
            "worst_model",
            "meets_target",
            "point_dpf_setup_slots",
            "point_dpf_bootstrap_ok",
            "point_dpf_net_slots",
            "point_dpf_net_fraction",
            "encoder_indep_setup_slots_placeholder",
            "encoder_indep_bootstrap_ok",
        )
    )

    passing = []
    for ring_logn in args.logn:
        n = 1 << ring_logn
        for c in args.c:
            for per_poly_t in args.t:
                worst = None
                for _limb, prime in PRIMES:
                    got = conservative_bits(estimator, ring_logn, c, per_poly_t, prime)
                    if got is None:
                        continue
                    if worst is None or got[0] < worst[0]:
                        worst = (got[0], prime, got[1], got[2])
                if worst is None:
                    continue
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
                    "yes" if bits >= args.target_bits else "no",
                    point_slots,
                    "yes" if point_slots < n else "no",
                    point_net,
                    f"{100.0 * point_net / n:.3f}",
                    dmpf_slots,
                    "yes" if dmpf_slots < n else "no",
                )
                writer.writerow(row)
                if bits >= args.target_bits:
                    passing.append(
                        (raw_points, ring_logn, c, per_poly_t, bits, point_slots < n)
                    )

    if not passing:
        print(
            f"# NO CANDIDATE reaches {args.target_bits} conservative bits in this grid",
            file=sys.stderr,
        )
        return 1
    passing.sort()
    raw_points, ring_logn, c, per_poly_t, bits, point_ok = passing[0]
    print(
        f"# conservative pin: n=2^{ring_logn}, c={c}, t={per_poly_t} "
        f"-> {bits:.3f} bits, {raw_points} raw points, "
        f"point-DPF bootstrap {'ok' if point_ok else 'IMPOSSIBLE'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
