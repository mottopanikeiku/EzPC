#!/usr/bin/env python3
"""CPU-only ideal-correlation conversion audit; not a cryptographic approval.

Exact mode compares complete visible views on fixed-local-correlation slices,
not marginal histograms.  Random mode completes simulated views with an honest
input used ONLY by the auditor, then replays both parties' source equations.
No third-party modules, native protocol execution, GPU access, or files written.
"""

import argparse
from collections import Counter
from itertools import product
import json
import random

P0 = 4611686018326724609
P1 = 4611686018309947393


def bits(word, width):
    return tuple((word >> j) & 1 for j in range(width))


def circuit(role, public, secret, carry_in, triples, openings, position):
    """Source ripple(), including its gate-free bit-zero transition."""
    carry = carry_in if role == 0 else 0
    sums, trace = [], []
    for j, x in enumerate(secret):
        p = (public >> j) & 1
        old = carry
        sums.append(x ^ old ^ (p if role == 0 else 0))
        if j == 0:
            carry = x & carry_in
            gate = ()
        else:
            a, b, c = triples[position]
            d, e = openings[position]
            sent = (x ^ a, old ^ b)
            received = (sent[0] ^ d, sent[1] ^ e)
            carry = c ^ (d & b) ^ (e & a) ^ ((d & e) if role == 0 else 0)
            gate = (position, x, old, a, b, c, d, e, sent, received, carry)
            position += 1
        if p:
            carry ^= x ^ old
        trace.append((j, x, old, sums[-1], gate, carry))
    return tuple(sums), carry, tuple(trace), position


def view(role, z, q, width, local, opened, openings, h, raw):
    ell = (2 * q - 1).bit_length()
    modulus, outmod = 1 << ell, 1 << width
    ebits, earith, triples, dbit, masks = local
    aggregate = sum(a << j for j, a in enumerate(earith)) % modulus
    sent_a = (z + aggregate) % modulus
    peer_a = (opened - sent_a) % modulus
    negated = tuple(x ^ (role == 0) for x in ebits)
    sums, _, first, pos = circuit(role, opened, negated, 1, triples, openings, 0)
    _, wrap, second, pos = circuit(role, (-q) % modulus, sums, 0, triples, openings, pos)
    assert pos == len(triples) == 2 * ell - 2
    sent_h = wrap ^ dbit
    peer_h = sent_h ^ h
    corrected = raw if h == 0 else ((1 - role) - raw) % outmod
    output = (z - q * corrected) % outmod
    # All selected OT outputs / sender inputs visible in the ideal-OT wrapper.
    eda_wrappers = tuple(
        (x, a, (x - a) % modulus, (1 - x - a) % modulus)
        if role == 0 else (x, a)
        for x, a in zip(ebits, earith)
    )
    triple_wrappers = tuple(
        (a, b, mask, mask ^ b, c ^ (a & b) ^ mask)
        for (a, b, c), mask in zip(triples, masks)
    )
    da_wrapper = ((dbit, raw, (dbit - raw) % outmod, (1 - dbit - raw) % outmod)
                  if role == 0 else (dbit, raw))
    return (role, z, q, width, local, aggregate, sent_a, peer_a, opened,
            first, second, sent_h, peer_h, h, raw, corrected, output,
            eda_wrappers, triple_wrappers, da_wrapper)


def sample_local(rng, ell):
    return (tuple(rng.randrange(2) for _ in range(ell)),
            tuple(rng.randrange(1 << ell) for _ in range(ell)),
            tuple(tuple(rng.randrange(2) for _ in range(3)) for _ in range(2 * ell - 2)),
            rng.randrange(2),
            tuple(rng.randrange(2) for _ in range(2 * ell - 2)))


def simulate(role, z, prescribed, q, width, local, opened, openings, h):
    outmod = 1 << width
    corrected = (pow(q, -1, outmod) * (z - prescribed)) % outmod
    raw = corrected if h == 0 else ((1 - role) - corrected) % outmod
    result = view(role, z, q, width, local, opened, openings, h, raw)
    assert result[16] == prescribed
    return result


def honest_completion(visible, honest_z, openings):
    """Unique hidden ideal correlations, plus hidden wrapper masks.

    The honest input is NOT an argument of simulate().  Completion is a coupling
    witness and recomputation oracle, not a simulator with extra information.
    """
    role, z, q, width, local = visible[:5]
    ell, outmod = (2 * q - 1).bit_length(), 1 << width
    modulus = 1 << ell
    ebits, earith, triples, dbit, masks = local
    global_r = (visible[8] - z - honest_z) % modulus
    rb = bits(global_r, ell)
    honest_ebits = tuple(x ^ y for x, y in zip(ebits, rb))
    honest_earith = tuple((x - a) % modulus for x, a in zip(rb, earith))
    # Reconstruct the honest wires gate-by-gate, deriving its triple from d,e.
    honest_triples, honest_masks = [], []
    honest_sums = []
    pos = 0
    for public, secret, cin, corrupt_trace in (
        (visible[8], tuple(x ^ (role == 1) for x in honest_ebits), 1, visible[9]),
        ((-q) % modulus, honest_sums, 0, visible[10]),
    ):
        carry = cin if role == 1 else 0
        current_sums = []
        for j, x in enumerate(secret):
            old = carry
            p = (public >> j) & 1
            current_sums.append(x ^ old ^ (p if role == 1 else 0))
            if j == 0:
                carry = x & cin
            else:
                a, b, c = triples[pos]
                d, e = openings[pos]
                corrupt_x, corrupt_old = corrupt_trace[j][1:3]
                ga, gb = d ^ x ^ corrupt_x, e ^ old ^ corrupt_old
                ha, hb, hc = ga ^ a, gb ^ b, (ga & gb) ^ c
                honest_triples.append((ha, hb, hc))
                # Corrupt receiver output = honest mask XOR corrupt choice * honest b.
                received = c ^ (a & b) ^ masks[pos]
                honest_masks.append(received ^ (a & hb))
                carry = hc ^ (d & hb) ^ (e & ha) ^ ((d & e) if role == 1 else 0)
                pos += 1
            if p:
                carry ^= x ^ old
        if cin:
            honest_sums.extend(current_sums)
    wrap = int(z + honest_z >= q)
    global_d = wrap ^ visible[13]
    honest_local = (honest_ebits, honest_earith, tuple(honest_triples),
                    dbit ^ global_d, tuple(honest_masks))
    honest_raw = (global_d - visible[14]) % outmod
    peer = view(1 - role, honest_z, q, width, honest_local, visible[8],
                openings, visible[13], honest_raw)
    assert visible[6] == peer[7] and visible[7] == peer[6]
    assert visible[11] == peer[12] and visible[12] == peer[11]
    for mine_trace, peer_trace in zip(visible[9:11], peer[9:11]):
        for mine_step, peer_step in zip(mine_trace, peer_trace):
            if mine_step[4]:
                assert mine_step[4][8] == peer_step[4][9]
                assert mine_step[4][9] == peer_step[4][8]
    assert visible[10][-1][-1] ^ peer[10][-1][-1] == wrap
    assert (visible[16] + peer[16]) % outmod == ((z + honest_z) % q) % outmod
    # Check both directional selected OT equations, not only triple sums.
    for mine, theirs in zip(visible[18], peer[18]):
        assert mine[4] == (theirs[2] ^ (mine[0] & theirs[1]))
        assert theirs[4] == (mine[2] ^ (theirs[0] & mine[1]))
    sender, receiver = (visible, peer) if role == 0 else (peer, visible)
    for s, r in zip(sender[17], receiver[17]):
        assert s[2 + r[0]] == r[1]
    assert sender[19][2 + receiver[19][0]] == receiver[19][1]
    return peer


def real_view(role, z, honest_z, q, width, local, global_r, global_ab, global_d, raw):
    """Forward real hybrid: sample global triple masks, not openings."""
    ell = (2 * q - 1).bit_length()
    opened = (z + honest_z + global_r) % (1 << ell)
    pos, openings = 0, []
    global_sums = []
    for public, secret, cin in (
        (opened, tuple(x ^ 1 for x in bits(global_r, ell)), 1),
        ((-q) % (1 << ell), global_sums, 0),
    ):
        carry, current = cin, []
        for j, x in enumerate(secret):
            old, p = carry, (public >> j) & 1
            current.append(x ^ old ^ p)
            if j:
                ga, gb = global_ab[pos]
                openings.append((x ^ ga, old ^ gb))
                pos += 1
            carry = (x & old) ^ ((x ^ old) if p else 0)
        if cin:
            global_sums.extend(current)
    assert carry == int(z + honest_z >= q)
    h = carry ^ global_d
    return view(role, z, q, width, local, opened, tuple(openings), h, raw)


def exact_audit():
    """Q=3, ell=3, bw=1: all inputs, outputs, roles on two local slices.

    Each slice fixes the retained local edaBit components, triples, final Boolean
    share and wrapper masks. Those are independent uniform in the unconditioned
    law.  Enumerate EVERY remaining real coin and compare full-view Counters.
    This is a conditional-slice identity, not full enumeration of every slice.
    """
    q, width, ell = 3, 1, 3
    gate_masks = tuple(product(tuple(product(range(2), repeat=2)), repeat=4))
    slices = [sample_local(random.Random(seed), ell) for seed in (0, 1)]
    comparisons = worlds = 0
    for local in slices:
        for role, z, honest_z in product(range(2), range(q), range(q)):
            real = [Counter(), Counter()]
            for global_r, global_ab, global_d, raw in product(range(8), gate_masks, range(2), range(2)):
                actual = real_view(role, z, honest_z, q, width, local,
                                   global_r, global_ab, global_d, raw)
                real[actual[16]][actual] += 1
                worlds += 1
            for prescribed in range(2):
                simulated = Counter()
                for opened, openings, h in product(range(8), gate_masks, range(2)):
                    sampled = simulate(role, z, prescribed, q, width, local, opened, openings, h)
                    simulated[sampled] += 1
                assert real[prescribed] == simulated, (role, z, honest_z, prescribed)
                assert len(simulated) == 4096 and set(simulated.values()) == {1}
                comparisons += 1
    return {"scope": "complete-view conditional slices, Q=3 ell=3 bw=1",
            "local_slices": len(slices), "distribution_equalities": comparisons,
            "real_worlds_enumerated": worlds, "views_per_condition": 4096}


def sampled_audit(samples, seed):
    rng = random.Random(seed)
    cases, branches, roles = 0, set(), set()
    for q, width in ((3, 1), (3, 3), (5, 3), (9, 4), (P0, 3), (P0, 32), (P0 * P1, 32)):
        ell = (2 * q - 1).bit_length()
        # Include S=0,Q-1,Q,2Q-2 and both individual canonical endpoints.
        inputs = ((0, 0), (0, q - 1), (q - 1, 0), (1, q - 1), (q - 1, q - 1))
        for k in range(max(samples, len(inputs))):
            z, honest_z = inputs[k] if k < len(inputs) else (rng.randrange(q), rng.randrange(q))
            for role, h in product(range(2), repeat=2):
                local = sample_local(rng, ell)
                openings = tuple((rng.randrange(2), rng.randrange(2)) for _ in range(2 * ell - 2))
                prescribed = (0, (1 << width) - 1)[k % 2] if k < len(inputs) else rng.randrange(1 << width)
                sampled = simulate(role, z, prescribed, q, width, local,
                                   rng.randrange(1 << ell), openings, h)
                honest_completion(sampled, honest_z, openings)
                cases += 1
                branches.add(h)
                roles.add(role)
    return {"scope": "two-party recomputation and ideal-OT wrapper coupling, sampled not proof",
            "cases": cases, "h_branches": sorted(branches), "roles": sorted(roles),
            "production_moduli": [str(P0), str(P0 * P1)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exact", "sampled", "all"), default="all")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    result = {"model": "independent ideal logical coins; no DRBG-state or concrete-OT simulation",
              "qualified_human_review": "required; P-CONV not approved"}
    if args.mode in ("exact", "all"):
        result["exact"] = exact_audit()
    if args.mode in ("sampled", "all"):
        result["sampled"] = sampled_audit(args.samples, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
