#!/usr/bin/env python3
"""Source-bound ideal-leaf arithmetic diagnostic; NEVER a security certification.

Counts are user-supplied, hypothetical conditional hybrid comparisons per prime,
not inferred from tree/evaluator counters. No live seed independence is assumed.
The output states the ideal-law assumptions and missing full-key proof boundary.
Only Python's standard library and CPU integer arithmetic are used.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


STATUS = "IDEAL_LEAF_DIAGNOSTIC_ONLY_P_KEY_OPEN_NO_SECURITY_CERTIFICATION"
SOURCE_PINS = {
    "src/two_party_dpf_protocol.h": "352c794fb6b3f9ab921723489684624d2433b5c9df535c73e94b4ef7922f0e3c",
    "src/gpu_spfss_zp.cuh": "e1c1b711571b661a9934ba9082b5f10d3bdd7ecf7d5ad52d26a1fa914f6aa4da",
    "src/spfss_host.cpp": "041cdb9dd2ba907187c95fbb521aecdbaece7fa9d85e6f0202684df948778b57",
    "src/two_party_dpf_gpu.cuh": "097998a9d4a12ab5fe6050bde22701418f27b2fc69a7e56ab80f929bd181369b",
    "src/ringlpn_ole_party.cuh": "9587b3b3896e2ff877116840a0b1bcffe880fb074bd5335096d255f9ce99044a",
    "src/two_party_linear_preprocess.cuh": "c2fcb41d6f72341d6392e4b8b1803528122eef431ff5e4c0eca96af01b125ca7",
}


def source_primes(root: Path) -> tuple[int, int]:
    source = ""
    for relative, expected in SOURCE_PINS.items():
        content = (root / relative).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"source pin mismatch: {relative}; review before repinning")
        if relative == "src/two_party_dpf_protocol.h":
            source = content.decode("utf-8")
    values = []
    for name in ("kPrime62", "kPrime62Crt2"):
        matches = re.findall(rf"constexpr Word {name} = ([0-9]+)ULL;", source)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one deployed {name} declaration")
        values.append(int(matches[0]))
    return values[0], values[1]


def law(half_bits: int, p: int) -> dict:
    """Closed form for the nonwrapping triangular excess, including r=0."""
    h = 1 << half_bits
    k, r = divmod(h, p)
    if p < 2 or 2 * r - 2 >= p:
        raise ValueError("closed form requires modulus >= 2 and 2*r-2 < p")
    epsilon = Fraction(r * r, h * h)
    if r == 0:
        delta, threshold, support = Fraction(0), 0, 0
    else:
        threshold = r * r // p
        positive_count = 2 * (r - threshold) - 1
        positive_sum = r * r - threshold * (threshold + 1)
        delta = Fraction(p * positive_sum - positive_count * r * r, p * h * h)
        support = 2 * r - 1
    return {
        "half_bits": half_bits, "p": p, "k": k, "r": r,
        "excess_support_size": support, "positive_threshold_floor": threshold,
        "epsilon": epsilon, "delta": delta, "shift": p // 2,
        "disjoint_shift_supports": support <= p // 2,
    }


def exact(value: Fraction) -> dict:
    # All decisions use Fraction; this log is a display-only approximation.
    return {
        "numerator": str(value.numerator), "denominator": str(value.denominator),
        "fraction": str(value),
        "log2_approx_not_security_bits": (
            math.log2(value.numerator) - math.log2(value.denominator) if value else None
        ),
    }


def check_reduced_domains() -> list[dict]:
    """Independent pair enumeration, not a simulation or a live-AES experiment."""
    results = []
    # Nonzero threshold, zero threshold, overlapping shifted supports, and r=0.
    for bits, p in ((8, 61), (8, 127), (5, 13), (4, 13), (4, 16)):
        h = 1 << bits
        counts = [0] * p
        for lo in range(h):
            for hi in range(h):
                counts[(lo % p + hi % p) % p] += 1
        row = law(bits, p)
        k, r = row["k"], row["r"]
        expected = [
            p * k * k + 2 * k * r + max(0, min(v + 1, 2 * r - 1 - v, r))
            for v in range(p)
        ]
        if counts != expected or sum(counts) != h * h:
            raise ValueError(f"reduced distribution mismatch: bits={bits}, p={p}")
        tv = sum((Fraction(abs(p * n - h * h), 2 * p * h * h) for n in counts), Fraction(0))
        if tv != row["delta"]:
            raise ValueError(f"reduced exact TV mismatch: bits={bits}, p={p}")
        for shift in range(p):
            gap = Fraction(sum(abs(counts[v] - counts[(v - shift) % p]) for v in range(p)), 2 * h * h)
            if gap > row["epsilon"] or gap > 2 * tv:
                raise ValueError("reduced arbitrary-shift bound failed")
        shift = row["shift"]
        shift_tv = Fraction(sum(abs(counts[v] - counts[(v - shift) % p]) for v in range(p)), 2 * h * h)
        event_gap = Fraction(sum(counts[v] - counts[(v - shift) % p] for v in range(row["excess_support_size"])), h * h)
        if row["disjoint_shift_supports"] and (shift_tv != row["epsilon"] or event_gap != row["epsilon"]):
            raise ValueError("reduced disjoint-shift equality failed")
        results.append({
            "half_bits": bits, "modulus": p, "enumerated_half_pairs": h * h,
            "single_law_tv": exact(tv), "half_modulus_shift_tv": exact(shift_tv),
            "interval_event_gap": exact(event_gap),
            "independent_fair_tag_joint_event_gap": exact(event_gap / 2),
            "all_shifts_checked": p, "status": "EXACT_ENUMERATION_MATCH",
        })
    return results


def nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def budget_bits(value: str) -> int:
    parsed = nonnegative(value)
    if parsed > 4096:
        raise argparse.ArgumentTypeError("budget bits must be <= 4096")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons-p0", type=nonnegative, required=True,
                        help="hypothetical justified comparisons at p0; not automatically trees")
    parser.add_argument("--comparisons-p1", type=nonnegative, required=True,
                        help="hypothetical justified comparisons at p1; use 0 for one-limb scope")
    parser.add_argument("--budget-bits", type=budget_bits, required=True,
                        help="diagnostic leaf-only target 2^-b, NOT a claimed security level")
    parser.add_argument("--check-reduced", action="store_true",
                        help="enumerate tiny domains exactly; no deployed-field enumeration")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        primes = source_primes(root)
        rows = [law(64, p) for p in primes]
        if not all(row["disjoint_shift_supports"] for row in rows):
            raise ValueError("deployed half-modulus shift no longer has disjoint excess supports")
        reduced = check_reduced_domains() if args.check_reduced else None
    except (OSError, ValueError) as error:
        parser.exit(2, f"leaf diagnostic rejected: {error}\n")
    counts = (args.comparisons_p0, args.comparisons_p1)
    target = Fraction(1, 1 << args.budget_bits)
    total_delta = sum((n * row["delta"] for n, row in zip(counts, rows)), Fraction(0))
    total_epsilon = sum((n * row["epsilon"] for n, row in zip(counts, rows)), Fraction(0))
    diagnostics = {}
    for name, bound in (
        ("mapped_to_uniform_conditional_hybrid", total_delta),
        ("two_payload_shift_conditional_hybrid", total_epsilon),
    ):
        capped = min(Fraction(1), bound)
        diagnostics[name] = {
            "uncapped_triangle_sum": exact(bound), "capped_upper_bound": exact(capped),
            "upper_bound_within_leaf_only_target": capped <= target,
        }
    output = {
        "schema": "ringlpn-ideal-leaf-loss-v1", "status": STATUS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": SOURCE_PINS,
        "comparison_semantics": (
            "Each count is a hypothetical conditional-law replacement/comparison in one "
            "specified hybrid, aggregated over its declared lifetime. Counts are NOT "
            "derived from trees, keys, leaves, parties, or execution counters. Two "
            "composition rows are alternative experiments, not additive charges."
        ),
        "assumptions_not_established_for_live_keys": [
            "independent uniform 64-bit halves at each idealized leaf",
            "appropriate conditional law given adversarial view at every replacement",
            "independent uniform tag for the joint-event witness only",
            "a reviewed mapping from runtime lifetime to comparison counts",
        ],
        "comparison_counts": {"p0": counts[0], "p1": counts[1]},
        "leaf_only_target": exact(target), "budget_bits_not_security_level": args.budget_bits,
        "primes": [
            {
                **{key: value for key, value in row.items() if key not in ("delta", "epsilon")},
                "label": f"p{index}", "single_law_tv_exact": exact(row["delta"]),
                "mixture_weight_epsilon": exact(row["epsilon"]),
                "disjoint_payload_shift_tv_exact": exact(row["epsilon"]),
                "tag_conditioned_interval_event_gap_exact": exact(row["epsilon"]),
                "independent_fair_tag_joint_event_gap_exact": exact(row["epsilon"] / 2),
                "arbitrary_payload_shift_tv_upper_bound": exact(row["epsilon"]),
            }
            for index, row in enumerate(rows)
        ],
        "finite_comparison_diagnostics": diagnostics,
        "reduced_domain_checks": reduced,
        "not_included": [
            "AES/DPF replacement or key-privacy reduction", "OT/OLE/Ring-LPN losses",
            "conversion and sampler losses", "live-seed or multi-key independence proof",
            "independent human review or complete lifetime advantage budget",
        ],
        "target_semantics": (
            "True certifies only an exact arithmetic inequality under the declared ideal "
            "conditional-law/count hypotheses. False means this upper-bound certificate "
            "misses the target, not an actual composed lower bound or empirical attack. "
            "Exit status does not encode target attainment. P-KEY remains open."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
