#!/usr/bin/env python3
"""Exact regular Ring-LPN projection diagnostics (not a security pin).

For n=B*t and a two-power d|n, this implements the shared 2026-08-04 law:
  d<=B:       c independent groups of t uniform balls in d bins;
  d=B*k>=B:  c*k independent groups of t/k uniform balls in B bins.

The occupied-support count recurrence for one group is
  A[j+1,s] = s*A[j,s] + (D-s+1)*A[j,s-1].
For deployed uniform-nonzero coefficients in F_p, the exact projected nonzero
support recurrence is
  Z[j+1,z+1] += Z[j,z]*(D-z)*(p-1)
  Z[j+1,z-1] += Z[j,z]*z
  Z[j+1,z]   += Z[j,z]*z*(p-2).
The one-sparse reduction scalar is nonzero, so multiplying a coefficient by it
preserves the conditional uniform F_p^* law. Independent group count vectors
are convolved. All probabilities and 2^-lambda decisions use integers.

BCG+20 Sections 8.2 and 9.1 are printed side by side, not reconciled. Optional
checksum-pinned EUROCRYPT-2024 estimator calls are mechanically guarded and
made only at explicit D:W points or selected-degree logarithmic searches.
Estimator values are unproved model diagnostics. For structured DOOM, the
orbit size |G/H|=d is formal, while treating sqrt(d) as an end-to-end decoder
speedup is separately labelled heuristic_diagnostic_only. Neither is a
concrete Ring-LPN security claim or parameter pin.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Callable, Iterable, Sequence

DATE = "2026-08-04"
SCHEMA = "ringlpn-regular-projection-v1"
STATUS = "INTERNAL_ADVISOR_DIAGNOSTIC_NO_SECURITY_PIN"
ESTIMATOR_SHA256 = "c5771c88665415559b21cc1773dcdf3298ec60db2882f4fb3a8b3a833f2d34dc"
LAW = ("n=B*t;d<=B:c*occupancy(t,d);d=B*k>=B:c*k*occupancy(t/k,B);"
       "coefficients=iid-uniform-Fq-star")
LAW_SHA256 = hashlib.sha256(LAW.encode("ascii")).hexdigest()
PRIMES = (("p0", 4611686018326724609), ("p1", 4611686018309947393))
DEFAULT_CANDIDATES = (
    ("n20_c4_t16", 1 << 20, 4, 16),
    ("n20_c4_t32", 1 << 20, 4, 32),
    ("n20_c4_t64", 1 << 20, 4, 64),
    ("n20_c8_t8", 1 << 20, 8, 8),
    ("n20_c8_t16", 1 << 20, 8, 16),
)
DEFAULT_TAIL_BITS = (64, 128)


class InputError(ValueError):
    pass


class InvariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    name: str
    n: int
    c: int
    t: int


@dataclass(frozen=True)
class Law:
    candidate: Candidate
    d: int
    bucket: int
    regime: str
    bins: int
    balls: int
    groups: int
    occupied: tuple[int, ...]
    occupied_denominator: int

    @property
    def total_balls(self) -> int:
        return self.candidate.c * self.candidate.t


FIELDS = (
    "schema_version", "record_type", "report_date", "status",
    "analysis_sha256", "script_sha256", "law_sha256", "estimator_sha256",
    "candidate", "n", "c", "t", "bucket_width", "d", "projection_regime",
    "bins_per_group", "balls_per_group", "independent_groups", "total_balls",
    "dpf_tree_count", "polynomial_pairs", "public_a_coefficients",
    "public_a_identity_a0_unsent",
    "occupied_support_min", "occupied_support_max",
    "occupied_distribution_denominator", "occupied_distribution_counts",
    "occupied_distribution_sha256", "distribution_checksum_encoding",
    "occupied_expectation_numerator",
    "occupied_expectation_denominator", "occupied_expectation_decimal",
    "bcg_section_8_2_expectation_numerator",
    "bcg_section_8_2_expectation_denominator",
    "bcg_section_8_2_expectation_decimal",
    "bcg_section_9_1_expectation_numerator",
    "bcg_section_9_1_expectation_denominator",
    "bcg_section_9_1_expectation_decimal",
    "coefficient_limb", "coefficient_prime",
    "nonzero_support_min", "nonzero_support_max",
    "nonzero_distribution_denominator", "nonzero_distribution_counts",
    "nonzero_distribution_sha256", "nonzero_expectation_numerator",
    "nonzero_expectation_denominator", "nonzero_expectation_decimal",
    "expected_support_loss_to_cancellation_numerator",
    "expected_support_loss_to_cancellation_denominator",
    "expected_support_loss_to_cancellation_decimal",
    "any_cancellation_union_bound_numerator",
    "any_cancellation_union_bound_denominator",
    "tail_budget_bits", "support_threshold_w", "event",
    "bound_numerator", "bound_denominator", "bound_log2_approx",
    "bound_log2_floor", "bound_log2_ceiling",
    "bound_log2_exact_power_of_two",
    "guaranteed_rejection_entropy_bits", "tail_budget_pass",
    "acceptance_lower_bound_numerator", "acceptance_lower_bound_denominator",
    "estimator_domain_guard", "estimator_N", "estimator_k",
    "estimator_max_weight", "estimator_model", "estimator_weight",
    "estimator_attack_bits", "structured_doom_orbit_size",
    "structured_doom_heuristic_loss_bits", "structured_doom_status",
    "structured_doom_source", "structured_doom_adjusted_model_bits",
    "attack_target_bits", "attack_margin_bits", "required_model_bits",
    "required_weight_search", "required_weight_cost_basis", "diagnostic_label",
)


def power2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def parse_candidate(text: str) -> Candidate:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) == 3:
        name, nums = "n%s_c%s_t%s" % tuple(parts), parts
    elif len(parts) == 4:
        name, nums = parts[0], parts[1:]
    else:
        raise argparse.ArgumentTypeError("candidate must be N,C,T or NAME,N,C,T")
    if not name or any(ch in name for ch in "\r\n,"):
        raise argparse.ArgumentTypeError("candidate name must be nonempty and CSV-safe")
    try:
        n, c, t = map(int, nums)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate dimensions must be integers") from exc
    return Candidate(name, n, c, t)


def parse_point(text: str) -> tuple[int, int]:
    try:
        d, w = map(int, text.split(":"))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("estimator point must be integer D:W") from exc
    return d, w


def validate_candidate(x: Candidate, cap: int) -> None:
    if x.n <= 0 or x.c < 2 or x.t <= 0:
        raise InputError(f"{x.name}: require n>0,c>=2,t>0")
    if not power2(x.n):
        raise InputError(f"{x.name}: n must be a power of two")
    if not power2(x.t) or x.n % x.t:
        raise InputError(f"{x.name}: regular t must be a power of two dividing n")
    if x.c * x.t > cap:
        raise InputError(f"{x.name}: c*t exceeds exact-DP cap {cap}")


def validate_degree(x: Candidate, d: int) -> None:
    if not power2(d) or x.n % d:
        raise InputError(f"{x.name}: d={d} must be a two-power divisor of n")


def degrees(x: Candidate, chosen: Sequence[int]) -> tuple[int, ...]:
    out = tuple(sorted(set(chosen))) if chosen else tuple(
        1 << i for i in range(x.n.bit_length())
    )
    for d in out:
        validate_degree(x, d)
    return out


def convolution(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    out[i + j] += ai * bj
    return out


def convolution_power(base: Sequence[int], exponent: int) -> list[int]:
    if exponent <= 0:
        raise InputError("group count must be positive")
    out, factor, e = [1], list(base), exponent
    while e:
        if e & 1:
            out = convolution(out, factor)
        e >>= 1
        if e:
            factor = convolution(factor, factor)
    return out


def occupied_group(bins: int, balls: int) -> list[int]:
    if bins <= 0 or balls <= 0:
        raise InputError("occupancy dimensions must be positive")
    counts = [1]
    for _ in range(balls):
        nxt = [0] * (len(counts) + 1)
        for s, count in enumerate(counts):
            nxt[s] += s * count
            if s < bins:
                nxt[s + 1] += (bins - s) * count
        counts = nxt
    return counts


def nonzero_group(bins: int, balls: int, prime: int) -> list[int]:
    if bins <= 0 or balls <= 0 or prime <= 2:
        raise InputError("nonzero-support DP dimensions are invalid")
    counts = [1]
    for _ in range(balls):
        nxt = [0] * (len(counts) + 1)
        for z, count in enumerate(counts):
            if z:
                nxt[z - 1] += count * z
                nxt[z] += count * z * (prime - 2)
            if z < bins:
                nxt[z + 1] += count * (bins - z) * (prime - 1)
        counts = nxt
    return counts


def make_law(x: Candidate, d: int) -> Law:
    validate_degree(x, d)
    B = x.n // x.t
    if d <= B:
        regime, bins, balls, groups = "d_le_bucket", d, x.t, x.c
    else:
        if d % B:
            raise InputError(f"{x.name}: d is not B*k")
        k = d // B
        if x.t % k:
            raise InputError(f"{x.name}: t is not divisible by d/B")
        regime, bins, balls, groups = (
            "d_ge_bucket_disjoint_intervals", B, x.t // k, x.c * k
        )
    counts = convolution_power(occupied_group(bins, balls), groups)
    denominator = pow(bins, balls * groups)
    if sum(counts) != denominator:
        raise InvariantError("occupied counts do not sum to denominator")
    law = Law(x, d, B, regime, bins, balls, groups, tuple(counts), denominator)
    expected = expectation(counts, denominator)
    closed = groups * bins * (1 - Fraction((bins - 1) ** balls, bins ** balls))
    if expected != closed:
        raise InvariantError("occupied DP expectation disagrees with shared law")
    return law


def exact_nonzero(law: Law, prime: int) -> tuple[tuple[int, ...], int]:
    group = nonzero_group(law.bins, law.balls, prime)
    counts = tuple(convolution_power(group, law.groups))
    denominator = pow(law.bins * (prime - 1), law.balls * law.groups)
    if sum(counts) != denominator:
        raise InvariantError("nonzero counts do not sum to denominator")
    return counts, denominator


def expectation(counts: Sequence[int], denominator: int) -> Fraction:
    return Fraction(sum(i * x for i, x in enumerate(counts)), denominator)


def decimal(value: Fraction, places: int = 12) -> str:
    sign = "-" if value < 0 else ""
    n, d = abs(value.numerator), value.denominator
    whole, rem = divmod(n, d)
    scale = 10 ** places
    frac, rem = divmod(rem * scale, d)
    if 2 * rem > d or (2 * rem == d and frac & 1):
        frac += 1
        if frac == scale:
            whole, frac = whole + 1, 0
    return f"{sign}{whole}.{frac:0{places}d}"


def log2_interval(value: Fraction) -> tuple[str, str, str]:
    if value < 0:
        raise InvariantError("negative probability")
    if value == 0:
        return "-inf", "-inf", "yes"
    n, d = value.numerator, value.denominator
    e = n.bit_length() - d.bit_length()
    below_power = n < (d << e) if e >= 0 else (n << -e) < d
    if below_power:
        e -= 1
    exact = power2(n) and power2(d)
    return (str(e), str(e) if exact else str(e + 1), "yes" if exact else "no")


def log2_approx_without_underflow(value: Fraction) -> str:
    if value == 0:
        return "-inf"

    def integer_log2_approx(x: int) -> float:
        shift = max(0, x.bit_length() - 53)
        return math.log2(x >> shift) + shift

    return f"{integer_log2_approx(value.numerator) - integer_log2_approx(value.denominator):.12f}"


def entropy_bits(value: Fraction) -> str:
    if value == 0:
        return "inf"
    floor, ceiling, exact = log2_interval(value)
    exponent = int(floor if exact == "yes" else ceiling)
    return str(max(0, -exponent))


def budget_pass(value: Fraction, bits: int) -> bool:
    return value.numerator * (1 << bits) <= value.denominator


def lower_tail(counts: Sequence[int], denominator: int, threshold: int) -> Fraction:
    if threshold <= 0:
        return Fraction(0)
    return Fraction(sum(counts[:min(threshold, len(counts))]), denominator)


def lower_quantile(counts: Sequence[int], denominator: int, bits: int) -> tuple[int, Fraction]:
    """max W such that Pr[support < W] <= 2^-bits."""
    cumulative = 0
    best = (0, Fraction(0))
    for W in range(len(counts) + 1):
        if W:
            cumulative += counts[W - 1]
        if cumulative * (1 << bits) <= denominator:
            best = (W, Fraction(cumulative, denominator))
        else:
            break
    return best


def bcg82(x: Candidate, d: int) -> Fraction:
    return x.c * d * (1 - Fraction((d - 1) ** x.t, d ** x.t))


def bcg91(x: Candidate, d: int) -> Fraction:
    w = x.c * x.t
    return w - x.c * d + (x.c * (d - 1) + w) * Fraction(
        (d - 1) ** (x.t - 1), d ** (x.t - 1)
    )


def encode_integer(value: int) -> str:
    """Exact integer encoding unaffected by Python's decimal digit safety cap."""
    prefix = "-0x" if value < 0 else "0x"
    return prefix + format(abs(value), "x")


def counts_text(counts: Sequence[int]) -> str:
    support = [i for i, value in enumerate(counts) if value]
    return (
        f"raw-counts-omitted;nonzero_entries={len(support)};"
        f"support_min={min(support)};support_max={max(support)}"
    )


def hash_integer(hasher, value: int) -> None:
    magnitude = abs(value)
    payload = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
    hasher.update(b"-" if value < 0 else b"+")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def checksum_counts(kind: str, law: Law, counts: Sequence[int], denominator: int,
                    prime: int | None = None) -> str:
    metadata = {"encoding": "signed-length-prefixed-big-endian-v1",
        "kind": kind, "n": law.candidate.n, "c": law.candidate.c,
        "t": law.candidate.t, "d": law.d, "prime": prime}
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    hash_integer(hasher, denominator)
    hasher.update(len(counts).to_bytes(8, "big"))
    for value in counts:
        hash_integer(hasher, value)
    return hasher.hexdigest()


def base(law: Law, kind: str, analysis: str, script: str, estimator: str) -> dict:
    x = law.candidate
    return {"schema_version": SCHEMA, "record_type": kind, "report_date": DATE,
            "status": STATUS, "analysis_sha256": analysis, "script_sha256": script,
            "law_sha256": LAW_SHA256, "estimator_sha256": estimator,
            "candidate": x.name, "n": x.n, "c": x.c, "t": x.t,
            "bucket_width": law.bucket, "d": law.d, "projection_regime": law.regime,
            "bins_per_group": law.bins, "balls_per_group": law.balls,
            "independent_groups": law.groups, "total_balls": law.total_balls,
            "dpf_tree_count": (x.c * x.t) ** 2, "polynomial_pairs": x.c ** 2,
            "public_a_coefficients": (x.c - 1) * x.n,
            "public_a_identity_a0_unsent": "yes"}


def candidate_row(x: Candidate, analysis: str, script: str, estimator: str) -> dict:
    return {
        "schema_version": SCHEMA,
        "record_type": "candidate_diagnostic",
        "report_date": DATE,
        "status": STATUS,
        "analysis_sha256": analysis,
        "script_sha256": script,
        "law_sha256": LAW_SHA256,
        "estimator_sha256": estimator,
        "candidate": x.name,
        "n": x.n,
        "c": x.c,
        "t": x.t,
        "bucket_width": x.n // x.t,
        "total_balls": x.c * x.t,
        "dpf_tree_count": (x.c * x.t) ** 2,
        "polynomial_pairs": x.c ** 2,
        "public_a_coefficients": (x.c - 1) * x.n,
        "public_a_identity_a0_unsent": "yes",
        "diagnostic_label": (
            "study candidate and implementation cost counters only; "
            "public a=(1,a1,...,a[c-1]) has unsent identity a0; "
            "NOT A PARAMETER OR SECURITY PIN"
        ),
    }


def put_bound(row: dict, value: Fraction, bits: int) -> None:
    row["bound_numerator"], row["bound_denominator"] = (
        encode_integer(value.numerator), encode_integer(value.denominator)
    )
    row["bound_log2_approx"] = log2_approx_without_underflow(value)
    lo, hi, exact = log2_interval(value)
    row["bound_log2_floor"], row["bound_log2_ceiling"] = lo, hi
    row["bound_log2_exact_power_of_two"] = exact
    row["guaranteed_rejection_entropy_bits"] = entropy_bits(value)
    row["tail_budget_pass"] = "yes" if budget_pass(value, bits) else "no"
    accept = 1 - value
    row["acceptance_lower_bound_numerator"] = encode_integer(accept.numerator)
    row["acceptance_lower_bound_denominator"] = encode_integer(accept.denominator)


def union_any_cancellation(law: Law, prime: int) -> Fraction:
    # At occupied support K, at most min(K,M-K) bins have multiplicity >=2;
    # each final sum cancels with probability <=1/(p-1).
    weighted = sum(count * min(k, law.total_balls - k)
                   for k, count in enumerate(law.occupied))
    return min(Fraction(1), Fraction(weighted,
        law.occupied_denominator * (prime - 1)))


def load_estimator(path: Path):
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read estimator: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ESTIMATOR_SHA256:
        raise InputError(f"estimator checksum mismatch: expected {ESTIMATOR_SHA256}, got {actual}")
    spec = importlib.util.spec_from_file_location("pinned_eurocrypt_2024_lpn", path)
    if spec is None or spec.loader is None:
        raise InputError("cannot import estimator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "analysisforq", None)) or not callable(
            getattr(module, "analysisforqregular", None)):
        raise InputError("pinned estimator API is incomplete")
    return module


def estimator_domain(c: int, d: int, weight: int) -> tuple[int, int, int]:
    """Fail closed before artifact code: redundancy is d, so t'<=d-1."""
    if c < 2 or d < 2:
        raise InputError("estimator domain requires c>=2 and d>=2")
    N, k, maximum = c * d, (c - 1) * d, d - 1
    if not 1 <= weight <= maximum:
        raise InputError(
            f"estimator (N,k,t)=({N},{k},{weight}) violates "
            f"the pinned implementation domain 1<=t<=N-k-1=d-1={maximum}"
        )
    return N, k, maximum


class Estimator:
    def __init__(self, module):
        self.module = module
        self.cache: dict[tuple[int, int, int, int, str], float] = {}

    def bits(self, c: int, d: int, w: int, prime: int, model: str) -> float:
        N, k, _maximum = estimator_domain(c, d, w)
        key = N, k, w, prime, model
        if key not in self.cache:
            fn = self.module.analysisforq if model == "exact" else self.module.analysisforqregular
            with contextlib.redirect_stdout(io.StringIO()):
                value = float(fn(N, k, w, prime))
            if not math.isfinite(value) or value < 0:
                raise InvariantError(f"estimator returned invalid value for {key}")
            self.cache[key] = value
        return self.cache[key]


def doom_loss(d: int) -> float:
    """sqrt(|G/H|) sensitivity with the exact orbit size |G/H|=d."""
    if d <= 0:
        raise InputError("structured-DOOM orbit degree must be positive")
    return 0.5 * math.log2(d)


def find_crossing(cost: Callable[[int], float], maximum: int,
                  required: float) -> tuple[int | None, float | None]:
    if cost(maximum) < required:
        return None, None
    lo, hi = 1, maximum
    while lo < hi:
        mid = (lo + hi) // 2
        if cost(mid) >= required:
            hi = mid
        else:
            lo = mid + 1
    return lo, cost(lo - 1) if lo > 1 else None


def core_rows(law: Law, tails: Sequence[int], analysis: str, script: str,
              estimator_hash: str) -> tuple[list[dict], dict[str, tuple[tuple[int, ...], int]]]:
    rows: list[dict] = []
    nonzero: dict[str, tuple[tuple[int, ...], int]] = {}
    occupied_mean = expectation(law.occupied, law.occupied_denominator)
    b82, b91 = bcg82(law.candidate, law.d), bcg91(law.candidate, law.d)
    support = [i for i, x in enumerate(law.occupied) if x]
    row = base(law, "occupied_distribution_exact", analysis, script, estimator_hash)
    row.update({"occupied_support_min": min(support), "occupied_support_max": max(support),
        "occupied_distribution_denominator": encode_integer(law.occupied_denominator),
        "occupied_distribution_counts": counts_text(law.occupied),
        "occupied_distribution_sha256": checksum_counts("occupied", law, law.occupied,
                                                         law.occupied_denominator),
        "distribution_checksum_encoding": "signed-length-prefixed-big-endian-v1",
        "occupied_expectation_numerator": encode_integer(occupied_mean.numerator),
        "occupied_expectation_denominator": encode_integer(occupied_mean.denominator),
        "occupied_expectation_decimal": decimal(occupied_mean),
        "bcg_section_8_2_expectation_numerator": encode_integer(b82.numerator),
        "bcg_section_8_2_expectation_denominator": encode_integer(b82.denominator),
        "bcg_section_8_2_expectation_decimal": decimal(b82),
        "bcg_section_9_1_expectation_numerator": encode_integer(b91.numerator),
        "bcg_section_9_1_expectation_denominator": encode_integer(b91.denominator),
        "bcg_section_9_1_expectation_decimal": decimal(b91),
        "event": "exact occupied support K; not projected nonzero support W",
        "diagnostic_label": "exact shared-law occupied distribution"})
    rows.append(row)

    for limb, prime in PRIMES:
        counts, denominator = exact_nonzero(law, prime)
        nonzero[limb] = counts, denominator
        mean = expectation(counts, denominator)
        loss = occupied_mean - mean
        if mean > occupied_mean:
            raise InvariantError(
                "projected nonzero-support expectation exceeds occupied support"
            )
        support = [i for i, x in enumerate(counts) if x]
        row = base(law, "nonzero_distribution_exact", analysis, script, estimator_hash)
        row.update({"coefficient_limb": limb, "coefficient_prime": prime,
            "nonzero_support_min": min(support), "nonzero_support_max": max(support),
            "nonzero_distribution_denominator": encode_integer(denominator),
            "nonzero_distribution_counts": counts_text(counts),
            "nonzero_distribution_sha256": checksum_counts("nonzero", law, counts,
                                                            denominator, prime),
            "distribution_checksum_encoding": "signed-length-prefixed-big-endian-v1",
            "nonzero_expectation_numerator": encode_integer(mean.numerator),
            "nonzero_expectation_denominator": encode_integer(mean.denominator),
            "nonzero_expectation_decimal": decimal(mean),
            "expected_support_loss_to_cancellation_numerator": encode_integer(loss.numerator),
            "expected_support_loss_to_cancellation_denominator": encode_integer(loss.denominator),
            "expected_support_loss_to_cancellation_decimal": decimal(loss),
            "event": "exact projected nonzero support W including coefficient cancellation",
            "diagnostic_label": "exact deployed-prime nonzero-support Markov DP"})
        rows.append(row)
        cancellation = union_any_cancellation(law, prime)
        row = base(law, "any_cancellation_bound", analysis, script, estimator_hash)
        row.update({"coefficient_limb": limb, "coefficient_prime": prime,
            "any_cancellation_union_bound_numerator": encode_integer(cancellation.numerator),
            "any_cancellation_union_bound_denominator": encode_integer(cancellation.denominator),
            "event": "Pr[at least one occupied coefficient sum cancels]",
            "tail_budget_bits": 0,
            "diagnostic_label": "rigorous coefficient-cancellation union bound"})
        put_bound(row, cancellation, 0)
        rows.append(row)

    for bits in tails:
        W, probability = lower_quantile(law.occupied, law.occupied_denominator, bits)
        row = base(law, "occupied_lower_tail_exact", analysis, script, estimator_hash)
        row.update({"tail_budget_bits": bits, "support_threshold_w": W,
            "event": "Pr[K_occupied < W]",
            "estimator_N": law.candidate.c * law.d,
            "estimator_k": (law.candidate.c - 1) * law.d,
            "estimator_max_weight": law.d - 1,
            "estimator_domain_guard": "pass" if 1 <= W <= law.d - 1 else "reject",
            "diagnostic_label": "exact occupied lower tail; strict event support<W"})
        put_bound(row, probability, bits)
        rows.append(row)
        for limb, prime in PRIMES:
            counts, denominator = nonzero[limb]
            W, probability = lower_quantile(counts, denominator, bits)
            row = base(law, "nonzero_lower_tail_exact", analysis, script, estimator_hash)
            row.update({"coefficient_limb": limb, "coefficient_prime": prime,
                "tail_budget_bits": bits, "support_threshold_w": W,
                "event": "Pr[W_nonzero < W]",
                "estimator_N": law.candidate.c * law.d,
                "estimator_k": (law.candidate.c - 1) * law.d,
                "estimator_max_weight": law.d - 1,
                "estimator_domain_guard": "pass" if 1 <= W <= law.d - 1 else "reject",
                "diagnostic_label": "exact deployed-prime nonzero lower tail; no reduction"})
            put_bound(row, probability, bits)
            rows.append(row)
    return rows, nonzero


def estimator_rows(law: Law, weight: int, runner: Estimator, models: Sequence[str],
                   analysis: str, script: str) -> list[dict]:
    if not 1 <= weight <= law.d - 1:
        raise InputError(f"explicit estimator d={law.d},w={weight} is outside 1<=w<=d-1")
    out = []
    loss, orbit_size = doom_loss(law.d), law.d
    for limb, prime in PRIMES:
        for model in models:
            bits = runner.bits(law.candidate.c, law.d, weight, prime, model)
            row = base(law, "estimator_model_diagnostic", analysis, script, ESTIMATOR_SHA256)
            row.update({"coefficient_limb": limb, "coefficient_prime": prime,
                "estimator_domain_guard": "pass", "estimator_N": law.candidate.c * law.d,
                "estimator_k": (law.candidate.c - 1) * law.d,
                "estimator_max_weight": law.d - 1, "estimator_model": model,
                "estimator_weight": weight, "estimator_attack_bits": f"{bits:.12f}",
                "structured_doom_orbit_size": orbit_size,
                "structured_doom_heuristic_loss_bits": f"{loss:.12f}",
                "structured_doom_status": (
                    "orbit |G/H|=d formal; sqrt(d) decoder speedup heuristic_diagnostic_only"
                ),
                "structured_doom_source": (
                    "Liu-Wang-Yang-Yu, EUROCRYPT 2024, Section 2.2; "
                    "orbit specialized here to |G/H|=d"
                ),
                "structured_doom_adjusted_model_bits": f"{bits-loss:.12f}",
                "diagnostic_label": "UNPROVED finite-field model + separate structured-DOOM heuristic_diagnostic_only; NEVER A SECURITY PIN"})
            out.append(row)
    return out


def required_rows(law: Law, nz: dict[str, tuple[tuple[int, ...], int]],
                  runner: Estimator, models: Sequence[str], target: float,
                  margin: float, basis: str, tails: Sequence[int], analysis: str,
                  script: str) -> list[dict]:
    if law.d <= 1:
        raise InputError("required-weight estimator mode needs d>=2")
    out, required = [], target + margin
    loss, orbit_size = doom_loss(law.d), law.d
    for limb, prime in PRIMES:
        for model in models:
            def cost(w: int) -> float:
                raw = runner.bits(law.candidate.c, law.d, w, prime, model)
                return raw - loss if basis == "structured-doom" else raw
            weight, predecessor = find_crossing(cost, law.d - 1, required)
            for tail_bits in tails:
                row = base(law, "required_nonzero_weight_sensitivity", analysis, script,
                           ESTIMATOR_SHA256)
                row.update({"coefficient_limb": limb, "coefficient_prime": prime,
                    "tail_budget_bits": tail_bits, "estimator_domain_guard": "pass",
                    "estimator_N": law.candidate.c * law.d,
                    "estimator_k": (law.candidate.c - 1) * law.d,
                    "estimator_max_weight": law.d - 1, "estimator_model": model,
                    "structured_doom_orbit_size": orbit_size,
                    "structured_doom_heuristic_loss_bits": f"{loss:.12f}",
                    "structured_doom_status": (
                        "orbit |G/H|=d formal; sqrt(d) decoder speedup heuristic_diagnostic_only"
                    ),
                    "structured_doom_source": (
                        "Liu-Wang-Yang-Yu, EUROCRYPT 2024, Section 2.2; "
                        "orbit specialized here to |G/H|=d"
                    ),
                    "attack_target_bits": f"{target:.12f}",
                    "attack_margin_bits": f"{margin:.12f}",
                    "required_model_bits": f"{required:.12f}",
                    "required_weight_search": "cached logarithmic binary search; minimum only under unproved estimator-output monotonicity",
                    "required_weight_cost_basis": basis,
                    "diagnostic_label": "REJECT/ACCEPT MODEL SENSITIVITY ONLY; NEVER A SECURITY PIN"})
                if weight is None:
                    row.update({"support_threshold_w": "none",
                        "event": "no mechanically valid weight reaches required model bits",
                        "tail_budget_pass": "no"})
                else:
                    raw = runner.bits(law.candidate.c, law.d, weight, prime, model)
                    counts, denominator = nz[limb]
                    rejection = lower_tail(counts, denominator, weight)
                    predecessor_text = (
                        "none" if predecessor is None else f"{predecessor:.12f}"
                    )
                    row.update({"support_threshold_w": weight,
                        "event": "Pr[actual projected nonzero support < required model W]",
                        "estimator_weight": weight, "estimator_attack_bits": f"{raw:.12f}",
                        "structured_doom_adjusted_model_bits": f"{raw-loss:.12f}",
                        "required_weight_search":
                            "cached logarithmic binary search; predecessor_cost=" +
                            predecessor_text + "; global monotonicity unproved"})
                    put_bound(row, rejection, tail_bits)
                out.append(row)
    return out


def enumerate_tiny(x: Candidate, d: int) -> tuple[list[int], int]:
    B, terms = x.n // x.t, x.c * x.t
    out = [0] * (terms + 1)
    for offsets in itertools.product(range(B), repeat=terms):
        support, q = set(), 0
        for poly in range(x.c):
            for bucket in range(x.t):
                support.add((poly, (bucket * B + offsets[q]) % d))
                q += 1
        out[len(support)] += 1
    return out, B ** terms


def enumerate_tiny_nonzero(
    x: Candidate, d: int, prime: int
) -> tuple[list[int], int]:
    B, terms = x.n // x.t, x.c * x.t
    out = [0] * (terms + 1)
    for offsets in itertools.product(range(B), repeat=terms):
        for coefficients in itertools.product(range(1, prime), repeat=terms):
            sums: dict[tuple[int, int], int] = {}
            q = 0
            for poly in range(x.c):
                for bucket in range(x.t):
                    key = (poly, (bucket * B + offsets[q]) % d)
                    sums[key] = (sums.get(key, 0) + coefficients[q]) % prime
                    q += 1
            out[sum(value != 0 for value in sums.values())] += 1
    return out, (B * (prime - 1)) ** terms


def self_test() -> None:
    for x in (Candidate("tiny_a", 8, 2, 2), Candidate("tiny_b", 8, 2, 4)):
        validate_candidate(x, 64)
        for d in degrees(x, ()):
            law = make_law(x, d)
            brute, brute_den = enumerate_tiny(x, d)
            scale, remainder = divmod(brute_den, law.occupied_denominator)
            expected = list(law.occupied) + [0] * (len(brute) - len(law.occupied))
            if remainder or [v * scale for v in expected] != brute:
                raise InvariantError(f"tiny occupied enumeration mismatch at {x.name},d={d}")
            for prime in (3, 5, PRIMES[0][1]):
                counts, den = exact_nonzero(law, prime)
                if sum(counts) != den:
                    raise InvariantError("nonzero invariant mismatch")
                if prime == 3:
                    brute_nonzero, brute_nonzero_den = enumerate_tiny_nonzero(x, d, prime)
                    scale, remainder = divmod(brute_nonzero_den, den)
                    expected_nonzero = list(counts) + [
                        0
                    ] * (len(brute_nonzero) - len(counts))
                    if remainder or [
                        value * scale for value in expected_nonzero
                    ] != brute_nonzero:
                        raise InvariantError(
                            f"tiny nonzero enumeration mismatch at {x.name},d={d}"
                        )
            for bits in (1, 8):
                W, p = lower_quantile(law.occupied, law.occupied_denominator, bits)
                if p != lower_tail(law.occupied, law.occupied_denominator, W):
                    raise InvariantError("quantile convention mismatch")
    if estimator_domain(4, 64, 63) != (256, 192, 63):
        raise InvariantError("estimator d-1 acceptance guard mismatch")
    try:
        estimator_domain(4, 64, 64)
    except InputError:
        pass
    else:
        raise InvariantError("estimator d rejection guard mismatch")
    if doom_loss(1) != 0.0:
        raise InvariantError("structured-DOOM d=1 loss must be zero")
    if any(doom_loss(64) != 3.0 for _c in (2, 4, 8)):
        raise InvariantError("structured-DOOM orbit loss must be c-independent")
    print("self-test: tiny enumerations and invariants pass", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exact regular-projection CSV audit; no security pin")
    p.add_argument("--candidate", action="append", type=parse_candidate,
                   help="[NAME,]N,C,T (repeatable; default: five n=2^20 study candidates)")
    p.add_argument("--degree", action="append", type=int, default=[],
                   help="two-power d (repeatable; default every d|n)")
    p.add_argument("--tail-bits", action="append", type=int, default=[],
                   help="lower-tail budget (repeatable; default 64,128)")
    p.add_argument("--max-total-balls", type=int, default=4096)
    p.add_argument("--estimator", type=Path)
    p.add_argument("--estimator-weight", action="append", type=parse_point, default=[],
                   metavar="D:W", help="explicit selected estimator point only")
    p.add_argument("--find-required-weight", action="append", type=int, default=[],
                   metavar="D", help="selected degree for cached logarithmic model search")
    p.add_argument("--estimator-model", choices=("exact", "regular", "both"), default="both")
    p.add_argument("--attack-target-bits", type=float, default=128.0)
    p.add_argument("--attack-margin-bits", type=float, default=0.0)
    p.add_argument("--required-weight-cost-basis", choices=("raw", "structured-doom"),
                   default="structured-doom")
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true",
                   help="run fast internal tiny enumerations and exit")
    return p


def render(rows: Iterable[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.self_test:
            if (args.candidate or args.degree or args.estimator or
                    args.estimator_weight or args.find_required_weight):
                raise InputError(
                    "--self-test cannot be combined with candidate/projection/estimator inputs"
                )
            self_test()
            return 0
        if args.max_total_balls <= 0:
            raise InputError("max-total-balls must be positive")
        tails = tuple(sorted(set(args.tail_bits or DEFAULT_TAIL_BITS)))
        if any(x <= 0 or x > 1_000_000 for x in tails):
            raise InputError("tail-bits must be in [1,1000000]")
        if not math.isfinite(args.attack_target_bits) or args.attack_target_bits < 0:
            raise InputError("attack-target-bits must be finite and nonnegative")
        if not math.isfinite(args.attack_margin_bits) or args.attack_margin_bits < 0:
            raise InputError("attack-margin-bits must be finite and nonnegative")
        candidates = args.candidate or [Candidate(*x) for x in DEFAULT_CANDIDATES]
        if len({x.name for x in candidates}) != len(candidates):
            raise InputError("candidate names must be unique")
        for x in candidates:
            validate_candidate(x, args.max_total_balls)
        estimator_requested = bool(args.estimator_weight or args.find_required_weight)
        if estimator_requested != (args.estimator is not None):
            raise InputError("--estimator is required exactly when estimator diagnostics are requested")
        point_set = set(args.estimator_weight)
        if len(point_set) != len(args.estimator_weight):
            raise InputError("duplicate estimator D:W point")
        points: dict[int, list[int]] = {}
        for d, weight in sorted(point_set):
            points.setdefault(d, []).append(weight)
        searches = set(args.find_required_weight)
        if len(searches) != len(args.find_required_weight):
            raise InputError("duplicate required-weight degree")
        module = load_estimator(args.estimator) if args.estimator else None
        runner = Estimator(module) if module else None
        estimator_hash = ESTIMATOR_SHA256 if runner else "not-called"
        models = ("exact", "regular") if args.estimator_model == "both" else (args.estimator_model,)
        script_path = Path(__file__).resolve()
        script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
        degree_map = {x.name: degrees(x, args.degree) for x in candidates}
        emitted = {d for ds in degree_map.values() for d in ds}
        unused = (set(points) | searches) - emitted
        if unused:
            raise InputError("estimator degrees not emitted: " + ",".join(map(str, sorted(unused))))
        manifest = {"schema": SCHEMA, "script": script_hash, "law": LAW_SHA256,
            "estimator": estimator_hash, "candidates": [[x.name,x.n,x.c,x.t] for x in candidates],
            "degrees": degree_map, "tails": tails, "points": sorted(point_set),
            "searches": sorted(searches), "models": models,
            "target": args.attack_target_bits, "margin": args.attack_margin_bits,
            "basis": args.required_weight_cost_basis}
        analysis = hashlib.sha256(json.dumps(manifest, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        rows: list[dict] = []
        for x in candidates:
            rows.append(candidate_row(x, analysis, script_hash, estimator_hash))
            for d in degree_map[x.name]:
                law = make_law(x, d)
                added, nz = core_rows(
                    law, tails, analysis, script_hash, estimator_hash
                )
                rows.extend(added)
                if d in points:
                    if runner is None:
                        raise InvariantError("missing estimator runner")
                    for weight in points[d]:
                        rows.extend(estimator_rows(
                            law, weight, runner, models, analysis, script_hash
                        ))
                if d in searches:
                    if runner is None:
                        raise InvariantError("missing estimator runner")
                    rows.extend(required_rows(
                        law, nz, runner, models, args.attack_target_bits,
                        args.attack_margin_bits, args.required_weight_cost_basis,
                        tails, analysis, script_hash
                    ))
        payload = render(rows)
        output_hash = hashlib.sha256(payload).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(args.output)
        else:
            sys.stdout.buffer.write(payload)
        print(f"# {STATUS}; rows={len(rows)}; output_sha256={output_hash}; "
              f"analysis_sha256={analysis}; no security parameter is pinned", file=sys.stderr)
        return 0
    except (InputError, InvariantError, OSError) as exc:
        print(f"audit rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
