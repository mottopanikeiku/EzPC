#!/usr/bin/env python3
"""Reproduce Wang et al.'s classical hybrid-RSD cost at direct candidates.

This is a formula calculator, not an attack implementation and not a Ring-LPN
security estimator.  It implements Theorem 1 of ePrint 2025/1284 (archived
2025-09-07 PDF pinned below), exhaustively optimizing every admissible integer
(f_bar, u_bar, g).  Costs are field operations and the paper's asymptotic space
expression; no operation-to-clock or concrete-byte conversion is made.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Iterable

DATE = "2026-08-04"
SCHEMA = "hybrid-rsd-asiacrypt2025-direct-v1"
STATUS = "INTERNAL_ADVISOR_DIAGNOSTIC_NO_SECURITY_PIN"
PAPER_TITLE = "A Hybrid Algorithm for the Regular Syndrome Decoding Problem"
PAPER_EPRINT = "2025/1284"
PAPER_DOI = "10.1007/978-981-95-5113-2_15"
PAPER_REVISION = "ePrint PDF captured 2025-09-07; landing page reports last of 2 revisions 2025-09-09"
PAPER_ARCHIVE_URL = (
    "https://web.archive.org/web/20250907044313id_/"
    "https://eprint.iacr.org/2025/1284.pdf"
)
PAPER_PDF_SHA256 = "a8d050905021bc737537d054ed33de643512d017a4d3d5d893167d844b6d494a"
FORMULA = "ePrint-2025-1284-Theorem-1-equations-9-13"
OMEGA = 2.8
PRIMES = (("p0", 4611686018326724609), ("p1", 4611686018309947393))
DEFAULT_CANDIDATES = (
    ("n20_c4_t16", 1 << 20, 4, 16),
    ("n20_c4_t32", 1 << 20, 4, 32),
    ("n20_c4_t64", 1 << 20, 4, 64),
    ("n20_c8_t8", 1 << 20, 8, 8),
    ("n20_c8_t16", 1 << 20, 8, 16),
)


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    name: str
    ring_n: int
    c: int
    t: int

    @property
    def length(self) -> int:
        return self.c * self.ring_n

    @property
    def dimension(self) -> int:
        return (self.c - 1) * self.ring_n

    @property
    def weight(self) -> int:
        return self.c * self.t

    @property
    def beta(self) -> int:
        return self.ring_n // self.t

    def validate(self) -> None:
        if not self.name or any(ch in self.name for ch in "\r\n,"):
            raise InputError("candidate name must be nonempty and CSV-safe")
        if self.ring_n <= 0 or self.c < 2 or self.t <= 0:
            raise InputError(f"{self.name}: require n>0, c>=2, t>0")
        if self.ring_n % self.t:
            raise InputError(f"{self.name}: direct regular shape requires t | n")
        if self.length != self.weight * self.beta:
            raise InputError(f"{self.name}: inconsistent RSD block shape")


@dataclass(frozen=True)
class Cost:
    log2_time: float
    log2_p_inv: float
    log2_t1: float
    log2_t2: float
    log2_t3: float
    log2_space_expression: float
    f_bar: int
    u_bar: int
    g: int
    n0: int
    w: int
    v: int
    m1: int
    z: int
    feasible_points_evaluated: int


def parse_candidate(text: str) -> Candidate:
    fields = text.split(",")
    if len(fields) == 3:
        name = "custom_" + "_".join(fields)
        numbers = fields
    elif len(fields) == 4:
        name, *numbers = fields
    else:
        raise argparse.ArgumentTypeError("candidate must be N,C,T or NAME,N,C,T")
    try:
        ring_n, c, t = map(int, numbers)
        candidate = Candidate(name, ring_n, c, t)
        candidate.validate()
        return candidate
    except (ValueError, InputError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def log2_or_negative_infinity(value: float | int) -> float:
    if value < 0:
        raise InputError("negative operation count")
    return -math.inf if value == 0 else math.log2(value)


def log2_add(left: float, right: float) -> float:
    if left < right:
        left, right = right, left
    if right == -math.inf:
        return left
    return left + math.log2(1.0 + 2.0 ** (right - left))


def feasibility_value(r: int, base_length: int, h: int, g: int, f_bar: int) -> int:
    """Twice equation (9)'s LHS minus twice its RHS.

    With r=beta-u_bar, v=(h-g)r-(N-K)-f_bar and
    2*m1=(h-g)r(r-1)-2*f_bar(r-1).  Feasibility is
    2*m1-v(v+1) >= 5*z.
    """
    blocks = h - g
    v = blocks * r - base_length - f_bar
    twice_m1 = blocks * r * (r - 1) - 2 * f_bar * (r - 1)
    z = (g + 1) * v + 2 * g + g * (g - 1) // 2
    return twice_m1 - v * (v + 1) - 5 * z


def feasible_r_interval(beta: int, base_length: int, h: int, g: int, f_bar: int):
    """Return every feasible r as one exact interval, or None.

    The feasibility polynomial is concave quadratic.  Integer binary searches
    around its exact integer maximizer avoid floating-point root rounding.
    """
    blocks = h - g
    if blocks <= 0:
        return None
    low = max(2, (base_length + f_bar + blocks - 1) // blocks)
    high = beta
    if low > high:
        return None

    y0 = feasibility_value(0, base_length, h, g, f_bar)
    y1 = feasibility_value(1, base_length, h, g, f_bar)
    y2 = feasibility_value(2, base_length, h, g, f_bar)
    quadratic = (y2 - 2 * y1 + y0) // 2
    linear = y1 - y0 - quadratic
    if quadratic >= 0:
        raise InputError("expected a concave feasibility polynomial")
    vertex_floor = (-linear) // (2 * quadratic)
    candidates = {low, high}
    for value in range(vertex_floor - 2, vertex_floor + 3):
        candidates.add(min(high, max(low, value)))
    peak = max(candidates, key=lambda r: feasibility_value(r, base_length, h, g, f_bar))
    if feasibility_value(peak, base_length, h, g, f_bar) < 0:
        return None

    if feasibility_value(low, base_length, h, g, f_bar) >= 0:
        left = low
    else:
        lo, hi = low, peak
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if feasibility_value(mid, base_length, h, g, f_bar) >= 0:
                hi = mid
            else:
                lo = mid
        left = hi

    if feasibility_value(high, base_length, h, g, f_bar) >= 0:
        right = high
    else:
        lo, hi = peak, high
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if feasibility_value(mid, base_length, h, g, f_bar) >= 0:
                lo = mid
            else:
                hi = mid
        right = lo
    return left, right


def evaluate(candidate: Candidate, g: int, f_bar: int, r: int) -> Cost:
    length = candidate.length
    dimension = candidate.dimension
    h = candidate.weight
    beta = candidate.beta
    blocks = h - g
    u_bar = beta - r
    n0 = dimension - f_bar - blocks * u_bar
    w = g * beta
    v = n0 - w
    m1 = blocks * r * (r - 1) // 2 - f_bar * (r - 1)
    z = (g + 1) * v + 2 * g + g * (g - 1) // 2
    if min(n0, v, m1, z) < 0:
        raise InputError("internal negative Theorem 1 parameter")
    if 2 * m1 - v * (v + 1) < 5 * z:
        raise InputError("internal violation of equation (9)")

    ell = (length - dimension) // beta
    remaining = length - dimension - ell * (r - 1)
    t1 = (length - dimension) * remaining * (n0 + remaining)
    t2_base = (n0 * n0 + 3 * n0) / 2
    log2_t2 = OMEGA * log2_or_negative_infinity(t2_base)
    substitution_columns = v + 2 * w + w * v + w * (w - 1) // 2
    t3_rows = 5 * z / 2
    log2_t3 = log2_add(
        OMEGA * log2_or_negative_infinity(t3_rows),
        log2_or_negative_infinity(t3_rows)
        + log2_or_negative_infinity(substitution_columns),
    )
    log2_t1 = log2_or_negative_infinity(t1)
    log2_inner = log2_add(
        log2_t1,
        log2_add(log2_t2, g * math.log2(beta) + log2_t3),
    )
    log2_p_inv = (
        f_bar * (math.log2(beta) - math.log2(r - 1))
        + (blocks - f_bar) * (math.log2(beta) - math.log2(r))
    )
    space_expression = (
        (length - dimension) * length
        + m1 * (n0 * n0 + 3 * n0) / 2
    )
    return Cost(
        log2_p_inv + log2_inner,
        log2_p_inv,
        log2_t1,
        log2_t2,
        log2_t3,
        math.log2(space_expression),
        f_bar,
        u_bar,
        g,
        n0,
        w,
        v,
        m1,
        z,
        0,
    )


def optimize(candidate: Candidate) -> Cost:
    candidate.validate()
    h = candidate.weight
    beta = candidate.beta
    best: Cost | None = None
    evaluated = 0
    for g in range(h + 1):
        # For g>0, Ttotal >= beta^g.  Once this lower bound reaches the
        # incumbent, every larger g is also proved unable to improve it.
        if best is not None and g > 0 and g * math.log2(beta) >= best.log2_time:
            break
        blocks = h - g
        for f_bar in range(blocks + 1):
            interval = feasible_r_interval(
                beta, candidate.length - candidate.dimension, h, g, f_bar
            )
            if interval is None:
                continue
            for r in range(interval[0], interval[1] + 1):
                evaluated += 1
                cost = evaluate(candidate, g, f_bar, r)
                if best is None or cost.log2_time < best.log2_time:
                    best = cost
    if best is None:
        raise InputError(f"{candidate.name}: no parameters satisfy equation (9)")
    return Cost(
        best.log2_time,
        best.log2_p_inv,
        best.log2_t1,
        best.log2_t2,
        best.log2_t3,
        best.log2_space_expression,
        best.f_bar,
        best.u_bar,
        best.g,
        best.n0,
        best.w,
        best.v,
        best.m1,
        best.z,
        evaluated,
    )


def self_test() -> None:
    # Tables 5 and 6 use the paper's stated truncation convention.  These two
    # rows exercise g>0 and reproduce both published optima and rounded costs.
    @dataclass(frozen=True)
    class Fixture:
        name: str
        length: int
        dimension: int
        weight: int
        beta: int

        def validate(self) -> None:
            if self.length != self.weight * self.beta:
                raise InputError("invalid self-test fixture")

    fixtures = (
        (Fixture("table6_n12", 3956, 1449, 172, 23), 132.60, (21, 7, 2)),
        (Fixture("table5_n14", 16224, 3322, 338, 48), 133.15, (148, 7, 2)),
    )
    for fixture, expected, parameters in fixtures:
        got = optimize(fixture)
        if round(got.log2_time, 2) != expected:
            raise AssertionError((fixture.name, got.log2_time, expected))
        if (got.f_bar, got.u_bar, got.g) != parameters:
            raise AssertionError((fixture.name, got, parameters))
    print("self-test: reproduced ePrint Tables 5/6 rows 132.60 and 133.15")


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def rows(candidates: Iterable[Candidate]):
    script_hash = script_sha256()
    for candidate in candidates:
        cost = optimize(candidate)
        for limb, prime in PRIMES:
            for orbit_treatment, orbit_size, orbit_assumption, adjustment in (
                ("none", 1, "none", 0.0),
                (
                    "sqrt_full_orbit_heuristic_sensitivity",
                    candidate.ring_n,
                    "non-paper heuristic; no end-to-end decoder proof",
                    math.log2(candidate.ring_n) / 2,
                ),
            ):
                yield {
                    "schema": SCHEMA,
                    "status": STATUS,
                    "generated_utc": DATE,
                    "script_sha256": script_hash,
                    "paper_title": PAPER_TITLE,
                    "paper_eprint": PAPER_EPRINT,
                    "paper_doi": PAPER_DOI,
                    "paper_revision": PAPER_REVISION,
                    "paper_archive_url": PAPER_ARCHIVE_URL,
                    "paper_pdf_sha256": PAPER_PDF_SHA256,
                    "formula_pin": FORMULA,
                    "author_artifact": "not located as of 2026-08-04",
                    "calculator_executable": "yes",
                    "attack_implementation_executable": "no",
                    "candidate": candidate.name,
                    "distribution_model": "direct regular-SD: h blocks, one nonzero per beta block; iid uniform F_p^* payloads",
                    "matrix_model": "Theorem 1 input H; full-rank Macaulay heuristic; no structured-code reduction",
                    "ring_n": candidate.ring_n,
                    "c": candidate.c,
                    "t_per_polynomial": candidate.t,
                    "N": candidate.length,
                    "K": candidate.dimension,
                    "h": candidate.weight,
                    "beta": candidate.beta,
                    "limb": limb,
                    "prime": prime,
                    "classical_quantum_scope": "classical F_p-operation formula only; paper gives no quantum cost",
                    "optimization": "exhaustive admissible integer (f_bar,u_bar,g), equation (9), omega=2.8",
                    "f_bar": cost.f_bar,
                    "u_bar": cost.u_bar,
                    "g": cost.g,
                    "n0": cost.n0,
                    "w": cost.w,
                    "v": cost.v,
                    "m1": cost.m1,
                    "z": cost.z,
                    "omega": OMEGA,
                    "feasible_points_evaluated": cost.feasible_points_evaluated,
                    "success_semantics": "per independent puncturing iteration; reported time uses expected 1/P iterations",
                    "log2_success_probability_per_iteration": f"{-cost.log2_p_inv:.12f}",
                    "log2_expected_iterations": f"{cost.log2_p_inv:.12f}",
                    "log2_T1_field_operations": f"{cost.log2_t1:.12f}",
                    "log2_T2_field_operations": f"{cost.log2_t2:.12f}",
                    "log2_T3_field_operations_per_enumeration": f"{cost.log2_t3:.12f}",
                    "log2_time_field_operations_baseline": f"{cost.log2_time:.12f}",
                    "orbit_treatment": orbit_treatment,
                    "orbit_size": orbit_size,
                    "orbit_assumption": orbit_assumption,
                    "log2_orbit_adjustment": f"{-adjustment:.12f}",
                    "log2_time_field_operations_reported": f"{cost.log2_time-adjustment:.12f}",
                    "memory_semantics": "log2 of expression inside Theorem 1 big-O, in F_p elements; not bytes or a concrete bound",
                    "log2_space_expression_field_elements": f"{cost.log2_space_expression:.12f}",
                    "data_semantics": "one H in F_p^((N-K)xN) and syndrome; formula does not price acquisition/materialization",
                    "claim_scope": "diagnostic direct-RSD formula instantiation; not a concrete Ring-LPN claim or parameter pin",
                }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_candidate)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.output is None and not args.candidate:
            return 0
    candidates = args.candidate or [Candidate(*values) for values in DEFAULT_CANDIDATES]
    output = args.output or (
        Path(__file__).resolve().parents[1]
        / "results/security/hybrid_regular_sd_asiacrypt2025_2026_08_04.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows(candidates))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
