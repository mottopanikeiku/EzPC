#!/usr/bin/env python3
"""Source-bound ideal-leaf and conditional lifetime arithmetic; not certification.

Explicit workloads count distinct generated tree pairs, not evaluator calls.
The role-specific one-hidden-leaf lemma justifies their statistical charge ONLY
under the conditional hidden-seed/key-prefix and transcript hypotheses in report
section 9.6 onward. AES, distributed-view and Ring-LPN reductions remain open.
Legacy comparison knobs remain explicitly hypothetical. CPU/stdlib only.
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
    "src/two_party_spfss.h": "fbdb56f84b1da9db63dad8b3464217fb05d21729344ae9ef88054b0677428461",
    "src/gpu_aes_prg_host.h": "e2b1742f88cf3d22c6f7e8c482fbb54449761bf9aefe464294777b9cd240cde1",
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


def workload(value: str) -> dict:
    """Explicit invocation population, not a security tuple or runtime admission."""
    try:
        n, c, t, limbs, mode, cross, invocations = value.split(",")
        n, c, t, limbs, cross, invocations = map(
            int, (n, c, t, limbs, cross, invocations))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected N,C,T,LIMBS,regular|uniform,CROSS_TERMS,INVOCATIONS") from error
    if min(n, c, t, cross, invocations) <= 0 or limbs not in (1, 2):
        raise argparse.ArgumentTypeError("positive sizes/counts and 1 or 2 limbs required")
    if mode not in ("regular", "uniform"):
        raise argparse.ArgumentTypeError("noise must be regular or uniform")
    if n & (n - 1) or not (1 << 13) <= n <= (1 << 20) or t > n:
        raise argparse.ArgumentTypeError("source requires power-of-two n in [2^13,2^20], t<=n")
    if mode == "regular" and (t & (t - 1) or n % t):
        raise argparse.ArgumentTypeError("regular source requires power-of-two t dividing n")
    trees = (c * t) ** 2
    reserve = 3 * trees
    domain = 2 * (n // t if mode == "regular" else n)
    depth = domain.bit_length() - 1
    if reserve >= n or not 2 <= depth <= 20 or trees > (1 << 14) or trees * domain > (1 << 24):
        raise argparse.ArgumentTypeError("source bootstrap/depth/tree/frontier limit exceeded")
    capacity = n - reserve
    batches = (cross + capacity - 1) // capacity
    if cross >= (1 << 64) or batches > (1 << 31) - 1:
        raise argparse.ArgumentTypeError("source cross-term/batch integer limit exceeded")
    return {
        "n": n, "c": c, "t": t, "limbs": limbs, "noise": mode,
        "cross_terms_per_invocation": cross, "invocations": invocations,
        "domain": domain, "depth": depth, "trees_per_ring_instance": trees,
        "application_capacity_per_ring_instance": capacity,
        "bootstrap_reserved_per_ring_instance": reserve,
        "ring_batches_per_invocation": batches,
    }


def lifetime_inventory(workloads: list[dict], rows: list[dict]) -> dict:
    counts = [0, 0]
    expansions = [0, 0]
    cases = []
    for item in workloads:
        trees, depth, domain = (item[key] for key in
                                ("trees_per_ring_instance", "depth", "domain"))
        invocations, limbs, batches = (item[key] for key in
                                      ("invocations", "limbs", "ring_batches_per_invocation"))
        instances_per_limb = invocations * 2 * batches
        keys_per_limb = instances_per_limb * trees
        instances = instances_per_limb * limbs
        keys = keys_per_limb * limbs
        reserve = item["bootstrap_reserved_per_ring_instance"]
        for limb in range(limbs):
            counts[limb] += keys_per_limb
            expansions[limb] += keys_per_limb * depth
        cases.append({
            **item,
            "directions": 2,
            "ring_instances_all_limbs": instances,
            "tree_pairs_per_active_limb": keys_per_limb,
            "tree_pairs_all_limbs": keys,
            "party_key_halves_all_limbs": 2 * keys,
            "keygen_leaf_conversions_both_parties": 2 * keys * domain,
            "one_full_evaluation_leaf_conversions_both_parties": 2 * keys * domain,
            "keygen_node_expansions_both_parties": 2 * keys * (domain - 1),
            "keygen_aes_block_calls_both_parties": 8 * keys * (domain - 1),
            "conditional_hidden_leaf_sites_one_role_one_world": keys,
            "candidate_hidden_path_expansions_one_role_one_world": keys * depth,
            "phase_b_string_ot_128_both_ot_directions": 2 * keys * depth,
            "phase_c_scalar_products_total": 3 * keys,
            "phase_c_external_epoch_zero_products": invocations * limbs * reserve,
            "phase_c_prior_ring_products_consumed": (instances - invocations * limbs) * reserve,
            "bootstrap_ring_slots_reserved": instances * reserve,
            "bootstrap_final_tail_discarded": invocations * limbs * reserve,
            "application_slots_used": invocations * 2 * limbs * item["cross_terms_per_invocation"],
            "application_slots_discarded": invocations * 2 * limbs * (
                batches * item["application_capacity_per_ring_instance"]
                - item["cross_terms_per_invocation"]),
        })
    single = sum((n * row["delta"] for n, row in zip(counts, rows)), Fraction(0))
    shifted = sum((n * row["epsilon"] for n, row in zip(counts, rows)), Fraction(0))
    hidden = sum(expansions)
    roots = 2 * sum(counts)
    # Four distinct AES inputs per node: ideal PRP/PRF switching, NOT AES security.
    switching = Fraction(6 * hidden, 1 << 128)
    root_collision = Fraction(roots * (roots - 1), 2 * (1 << 128))
    return {
        "scope": "successful fresh linear invocations; one fixed corrupted role; not a complete view proof",
        "workloads": cases,
        "tree_pairs_by_limb": {"p0": counts[0], "p1": counts[1]},
        "candidate_hidden_path_expansions_by_limb": {"p0": expansions[0], "p1": expansions[1]},
        "conditional_map_terms": {
            "one_world_to_uniform_final_cw_uncapped": exact(single),
            "two_world_common_simulator_uncapped": exact(2 * single),
            "two_world_common_simulator_capped": exact(min(Fraction(1), 2 * single)),
            "fixed_alpha_payload_only_common_prefix_uncapped": exact(shifted),
        },
        "separate_ideal_primitive_terms_not_live_certificates": {
            "four_query_prp_prf_switching_one_world_uncapped": exact(switching),
            "iid_128_bit_root_repetition_union_bound_uncapped": exact(root_collision),
            "root_draws_both_parties": roots,
        },
        "complete_conditional_bound": (
            "min(1, sigma + sum_{w=0,1}(rng_w + transcript_w + aes_w + "
            "switch_w + bad_w + ring_w + ot_ole_w + conversion_w + sampler_w) "
            "+ 2*sum_l K_l*delta_l). Terms must describe disjoint transitions; "
            "bad includes root repeats only if not already charged. "
            "No unknown term is assigned zero. See report 9.9."
        ),
        "live_complete_bound": None,
        "missing_reduction": [
            "conditional hidden-path AES/DPF replacement given one full key prefix and auxiliary view",
            "joint distributed-transcript simulation (not merely final-key equality)",
            "adaptive Figure-2 bootstrap/application joint simulation and structured Ring-LPN advantage",
            "unknown seed exposure/correction-induced collision and namespace bad events",
        ],
        "workload_limit": (
            "Counts inventory the declared successful calls, not arbitrary failures/restarts, "
            "a graph inferred from a model name, native AES adversary queries, or all shape admission checks."
        ),
    }


def check_role_replacement() -> dict:
    """Exact finalCW experiment for both roles/tags, including tag-zero leakage."""
    bits, p = 4, 13
    h = 1 << bits
    row = law(bits, p)
    counts = [0] * p
    for lo in range(h):
        for hi in range(h):
            counts[(lo + hi) % p] += 1
    checked = 0
    for role in (0, 1):
        for tag in (0, 1):
            for own in range(p):
                for beta in range(p):
                    laws = []
                    for payload in (beta, (beta + row["shift"]) % p):
                        cw_counts = [0] * p
                        for hidden, weight in enumerate(counts):
                            cw = ((2 * tag - 1) * (payload - own + hidden)
                                  if role == 0 else
                                  (1 - 2 * tag) * (payload - hidden + own)) % p
                            cw_counts[cw] += weight
                        tv = sum((Fraction(abs(p * n - h * h), 2 * p * h * h)
                                  for n in cw_counts), Fraction(0))
                        if tv != row["delta"]:
                            raise ValueError("role-specific finalCW uniform replacement mismatch")
                        laws.append(cw_counts)
                    gap = Fraction(sum(abs(a - b) for a, b in zip(*laws)), 2 * h * h)
                    if gap != row["epsilon"]:
                        raise ValueError("role/tag payload comparison mismatch")
                    checked += 1
    # V is the hidden raw seed itself: its marginal is ideal but not hidden.
    # Real (V, Convert(V)) vs (V, U_p) has exact TV 1-1/p.
    joint_tv = Fraction(0)
    for seed in range(h * h):
        converted = (seed % h + seed // h) % p
        for value in range(p):
            joint_tv += abs(Fraction(int(value == converted), h * h)
                            - Fraction(1, h * h * p)) / 2
    if joint_tv != 1 - Fraction(1, p) or joint_tv <= row["delta"]:
        raise ValueError("conditional-law counterexample mismatch")
    return {
        "half_bits": bits, "modulus": p, "role_tag_own_payload_cases": checked,
        "full_final_cw_tv_each_case": exact(row["delta"]),
        "payload_shift_tv_each_case": exact(row["epsilon"]),
        "revealed_seed_joint_tv_counterexample": exact(joint_tv),
        "independently_resampling_repeated_uniform_answer_equality_gap": exact(1 - Fraction(1, p)),
        "status": "EXACT_ROLE_ENUMERATION_AND_CONDITIONAL_COUNTEREXAMPLE_MATCH",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons-p0", type=nonnegative,
                        help="legacy hypothetical comparisons; incompatible with --workload")
    parser.add_argument("--comparisons-p1", type=nonnegative,
                        help="legacy hypothetical comparisons; incompatible with --workload")
    parser.add_argument("--workload", type=workload, action="append", default=[],
                        metavar="N,C,T,LIMBS,MODE,CROSS,INVOCATIONS",
                        help="repeat for each explicit population of fresh linear invocations")
    parser.add_argument("--budget-bits", type=budget_bits, required=True,
                        help="diagnostic leaf-only target 2^-b, NOT a claimed security level")
    parser.add_argument("--check-reduced", action="store_true",
                        help="enumerate tiny domains exactly; no deployed-field enumeration")
    args = parser.parse_args()
    legacy = (args.comparisons_p0, args.comparisons_p1)
    if args.workload:
        if any(value is not None for value in legacy):
            parser.error("--workload and hypothetical --comparisons-p* are mutually exclusive")
    elif any(value is None for value in legacy):
        parser.error("supply --workload or both hypothetical --comparisons-p* counts")
    root = Path(__file__).resolve().parents[1]
    try:
        primes = source_primes(root)
        rows = [law(64, p) for p in primes]
        if not all(row["disjoint_shift_supports"] for row in rows):
            raise ValueError("deployed half-modulus shift no longer has disjoint excess supports")
        reduced = check_reduced_domains() if args.check_reduced else None
        role_checks = check_role_replacement() if args.check_reduced else None
        inventory = lifetime_inventory(args.workload, rows) if args.workload else None
    except (OSError, ValueError) as error:
        parser.exit(2, f"leaf diagnostic rejected: {error}\n")
    counts = (tuple(inventory["tree_pairs_by_limb"][f"p{i}"] for i in range(2))
              if inventory else legacy)
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
        "schema": "ringlpn-ideal-leaf-loss-v2", "status": STATUS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": SOURCE_PINS,
        "comparison_semantics": (
            "Workload counts charge one hidden on-path map per tree pair for one fixed "
            "corrupted role in one world, ONLY under report 9.6-9.9 hypotheses; the "
            "two-world generic map term is twice that sum. Legacy counts remain "
            "hypothetical. Neither mode establishes live-key conditional laws."
        ),
        "assumptions_not_established_for_live_keys": [
            "independent uniform 64-bit halves at each idealized leaf",
            "appropriate conditional law given adversarial view at every replacement",
            "independent uniform tag for the joint-event witness only",
            "distributed-view and adaptive bootstrap lifting of the conditional one-key lemma",
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
        "role_replacement_checks": role_checks,
        "source_bound_lifetime": inventory,
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
