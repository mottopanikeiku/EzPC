#!/usr/bin/env python3
"""Pinned CRYPTO-2024 Regular-ISD attack diagnostics; never a security pin.

The executable formulas below are a faithful stdlib transcription of the code
cell headed "Concrete Formulas of Different Algorithms" in the accepted
artifact.  The generic BJMM row is deliberately fail-closed: the immutable
artifact delegates it to an unversioned external CryptographicEstimators/Sage
checkout, so the archive alone cannot reproduce that computation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Callable, Sequence
import zipfile

DATE = "2026-08-04"
SCHEMA = "crypto-2024-regular-isd-diagnostic-v1"
STATUS = "INTERNAL_ADVISOR_DIAGNOSTIC_NO_SECURITY_PIN"
ARTIFACT_URL = "https://artifacts.iacr.org/files/crypto/2024/crypto-2024-a1.zip"
ARTIFACT_PAGE = "https://artifacts.iacr.org/crypto/2024/a1/"
ARTIFACT_GIT_REVISION = "afe1e408f8a46aebc15293462480f478ff969923"
ARTIFACT_SHA256 = "04ae2586fccb10481efb861104176e4aaabb380c3cb9704b97ce3c4768a282cb"
NOTEBOOK_MEMBER = "Regular-ISD/Concrete numbers/estimates.ipynb"
NOTEBOOK_SHA256 = "cebb0861f1faa53be59eb4c11a2e38219612e1fc8d39e6cb3ce597e28717c9ec"
PICKLE_MEMBER = "Regular-ISD/Concrete numbers/data_concrete.pkl"
PICKLE_SHA256 = "b376f6555c30b4e237cc41e272a64d05b0c7bb89b3a473775038b376a958ad03"
PRIMES = (("p0", 4611686018326724609), ("p1", 4611686018309947393))
CANDIDATES = (
    ("n20_c4_t16", 1 << 20, 4, 16),
    ("n20_c4_t32", 1 << 20, 4, 32),
    ("n20_c4_t64", 1 << 20, 4, 64),
    ("n20_c8_t8", 1 << 20, 8, 8),
    ("n20_c8_t16", 1 << 20, 8, 16),
)
FIELDS = (
    "date", "schema", "status", "row_kind", "candidate", "distribution",
    "source_model", "applicability", "N", "k", "w", "block_size_b",
    "projection_degree", "coefficient_limb", "coefficient_prime", "algorithm",
    "time_raw_log2", "memory_raw_log2", "optimal_parameters", "orbit_size",
    "orbit_sqrt_sensitivity_bits", "time_optional_orbit_sensitivity_log2",
    "orbit_treatment", "domain_guard", "executable", "dependency_status",
    "success_semantics", "memory_semantics", "data_semantics", "assumptions",
    "warnings", "source_revision", "artifact_url", "artifact_page",
    "artifact_zip_sha256", "source_member", "source_member_sha256",
    "analysis_sha256", "script_sha256", "record_sha256",
)


class AuditError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _round(x: float) -> int:
    return int(round(x))


def _binomial(a: float, b: float) -> int:
    # Exact artifact behavior: math.comb(int(a), int(b)).
    return math.comb(int(a), int(b))


def _checked(name: str, fn: Callable[[], tuple[dict, list[float]]]) -> tuple[dict, list[float]]:
    try:
        params, cost = fn()
    except (ArithmeticError, ValueError, KeyError) as exc:
        raise AuditError(f"{name} artifact formula rejected input: {type(exc).__name__}") from exc
    if not isinstance(params, dict) or len(cost) != 2 or any(not math.isfinite(float(x)) for x in cost):
        raise AuditError(f"{name} artifact formula found no finite point")
    return params, cost


def validate_regular(n: int, k: int, w: int, b: int) -> None:
    if not all(isinstance(x, int) for x in (n, k, w, b)):
        raise AuditError("regular parameters must be integers")
    if not (0 < k < n and 0 < w < n and b > 1):
        raise AuditError("requires 0<k<n, 0<w<n, b>1")
    if n != w * b:
        raise AuditError("requires exact regular partition n=w*b")
    if k - w <= 0:
        raise AuditError("artifact added-check domain requires k-w>0")


def permutation_cost(n: int, k: int, w: int, b: int) -> tuple[dict, list[float]]:
    validate_regular(n, k, w, b)
    k_prime = k - w
    p_iter = (1 - k_prime / n) ** w
    if not 0 < p_iter <= 1:
        raise AuditError("permutation success probability is outside (0,1]")
    t_iter = (n - k_prime) ** 2 * n
    return {}, [math.log2(t_iter) - math.log2(p_iter), math.log2(n - k) + math.log2(n)]


def enumeration_cost(n: int, k: int, w: int, b: int) -> tuple[dict, list[float]]:
    validate_regular(n, k, w, b)
    k_prime = k - w
    minimum = [1e30, 0.0]
    best: dict = {}
    p_max = 30
    for p in range(0, p_max, 2):
        l1_approx = max(1, _binomial(_round(w / 2), p / 2) * (k_prime / w) ** (p / 2))
        ell0 = math.log2(l1_approx)
        ell_min = math.ceil(ell0 * 0.5)
        ell_max = min(_round(ell0 * 1.5), n - k_prime)
        for ell in range(ell_min, ell_max):
            v = (k_prime + ell) / w
            if w / 2 < p / 2 or v / b >= 1 or v <= 0:
                continue
            p_iter = (math.log2(_binomial(math.floor(w / 2), _round(p / 2)))
                      + math.log2(_binomial(math.ceil(w / 2), _round(p / 2)))
                      + math.log2(v / b) * p + math.log2(1 - v / b) * (w - p))
            L = math.log2(_binomial(_round(w / 2), p / 2)) + math.log2(v) * (p / 2)
            t_iter = math.log2(n) + max(math.log2(n - k_prime) * 2, 1 + L, L * 2 - ell)
            cost = t_iter - p_iter
            if cost < minimum[0]:
                minimum = [cost, max(L, math.log2(n - k)) + math.log2(n)]
                best = {"p": p, "ell": ell}
    return best, minimum


def representation_cost(n: int, k: int, w: int, b: int) -> tuple[dict, list[float]]:
    validate_regular(n, k, w, b)
    k_prime = k - w
    minimum = [1e30, 0.0]
    best: dict = {}
    p_max, eps_x_max, eps_y_max = 40, 32, 20
    for p in range(0, p_max, 8):
        for eps_x in range(0, eps_x_max, 4):
            p_x = p / 2 + eps_x
            for eps_y in range(0, eps_y_max):
                p_y = p_x / 2 + eps_y
                l1_seed = max(_binomial(_round(w / 2), p_y // 2) * k_prime ** (p_y // 2), 1)
                ell_approx = _round(2 * math.log2(l1_seed))
                ell_min, ell_max = math.ceil(ell_approx * 0.5), math.floor(ell_approx * 1.5)
                for ell in range(ell_min, ell_max):
                    v = (k_prime + ell) / w
                    if (p / 2 < p / 4 or w / 2 - p / 2 < eps_x / 2
                            or p_x / 2 < p_x / 2 or w / 2 - p_x / 2 < eps_y / 2):
                        continue
                    rx = (math.log2(_binomial(p // 2, p // 4))
                          + math.log2(_binomial(w // 2 - p // 2, eps_x / 2))
                          + math.log2(v) * (eps_x / 2)) * 2
                    ry = (math.log2(_binomial(p_x // 2, p_x // 4))
                          + math.log2(_binomial(w // 2 - p_x // 2, eps_y // 2))
                          + math.log2(v) * (eps_y / 2)) * 2
                    ell_x, ell_y = math.floor(rx), math.floor(ry)
                    if ell_y > ell_x or w / 2 < p / 2 or v == 0 or v >= b:
                        continue
                    p_iter = (math.log2(_binomial(math.floor(w / 2), _round(p / 2)))
                              + math.log2(_binomial(math.ceil(w / 2), _round(p / 2)))
                              + math.log2(v / b) * p + math.log2(1 - v / b) * (w - p))
                    L1 = math.log2(_binomial(_round(w / 2), p_y / 2)) + math.log2(v) * (p_y / 2)
                    if -p_iter + L1 > minimum[0]:
                        continue
                    ly1 = L1 * 2 - ell_y
                    ny = ly1 * 2 - (ell_x - ell_y)
                    lx1 = math.log2(_binomial(_round(w / 2), p_x // 2)) * 2 + math.log2(v) * p_x - ell_x
                    nx = lx1 * 2 - (ell - ell_x)
                    t_iter = math.log2(n) + max(math.log2(n - k_prime) * 2, 3 + L1,
                                                2 + ly1, 1 + ny, 1 + lx1, nx)
                    cost = t_iter - p_iter
                    if cost < minimum[0]:
                        minimum = [cost, max(L1, ly1, lx1) + math.log2(n)]
                        best = {"p": p, "eps_x": eps_x, "eps_y": eps_y,
                                "ell_x": ell_x, "ell_y": ell_y, "ell_min": ell_min,
                                "ell": ell, "ell_max": ell_max}
    return best, minimum


def representation_depth2_cost(n: int, k: int, w: int, b: int) -> tuple[dict, list[float]]:
    validate_regular(n, k, w, b)
    k_prime = k - w
    minimum = [1e30, 0.0]
    best: dict = {}
    p_max, eps_x_max = 40, 32
    for p in range(0, p_max, 4):
        for eps_x in range(0, eps_x_max, 2):
            p_x = p / 2 + eps_x
            l1_approx = _binomial(round(w / 2), p_x / 2) * k_prime ** (p_x / 2)
            if l1_approx <= 0 or w / 2 < p / 2:
                continue
            R = (_binomial(p / 2, p / 4) * _binomial(w / 2 - p / 2, eps_x / 2)) ** 2
            ell_approx = math.ceil(2 * math.log2(l1_approx))
            ell_min, ell_max = math.floor(ell_approx * 0.5), math.floor(ell_approx * 1.5)
            for ell in range(ell_min, ell_max):
                v = (k_prime + ell) / w
                if R <= 0:
                    break
                ell_x = math.floor(math.log2(R))
                L1 = math.log2(_binomial(round(w / 2), p_x / 2)) + math.log2(v) * (p_x / 2)
                lx1 = L1 * 2 - ell_x
                nx = lx1 * 2 - (ell - ell_x)
                if w / 2 < p / 2 or v >= b or v <= 0:
                    continue
                p_iter = (math.log2(_binomial(math.floor(w / 2), round(p / 2)))
                          + math.log2(_binomial(math.ceil(w / 2), round(p / 2)))
                          + math.log2(v / b) * p + math.log2(1 - v / b) * (w - p))
                t_iter = math.log2(n) + max(math.log2(n - k_prime) * 2,
                                            2 + L1, 1 + lx1, nx)
                cost = t_iter - p_iter
                if cost < minimum[0]:
                    minimum = [cost, max(L1, lx1) + math.log2(n)]
                    best = {"p": p, "eps_x": eps_x, "ell_x": ell_x,
                            "ell_min": ell_min, "ell": ell, "ell_max": ell_max}
    return best, minimum


def ccj_cost(n: int, k: int, w: int, b: int) -> tuple[dict, list[float]]:
    validate_regular(n, k, w, b)
    minimum = [1e30, 0.0]
    best: dict = {}
    k_tilde_approx = k - (1 - k / n) / (1 - w / n) * w
    l1_approx = math.log2(n / w) * (w * k_tilde_approx / (2 * n))
    ell_min = math.ceil(l1_approx * 0.75)
    ell_max = min(_round(l1_approx * 1.5), n - k)
    for ell in range(ell_min, ell_max):
        k_tilde = k - (1 - (ell + k) / n) / (1 - w / n) * w
        if ell < n - k_tilde:
            L = (n / w) ** (w * (k_tilde + ell) / (2 * n))
            num_coll = L ** 2 * 2 ** -ell
            t_iter = (n - k_tilde) ** 2 * n + n * (2 * L + num_coll)
            cost = math.log2(t_iter)
            if cost < minimum[0]:
                minimum = [cost, math.log2(L) + math.log2(n)]
                best = {"k_tilde": k_tilde, "ell": ell}
    return best, minimum


def ccj_linear_cost(n: int, k: int, w: int, b: int) -> tuple[dict, list[float]]:
    validate_regular(n, k, w, b)
    iterations = math.log2((n - w) / (n - k)) * (w * (1 - w / n))
    t_iter = (n - k) ** 2 * n
    return {}, [math.log2(t_iter) + iterations, math.log2(n - k) + math.log2(n)]


FORMULAS: tuple[tuple[str, Callable], ...] = (
    ("permutation", permutation_cost),
    ("enumeration", enumeration_cost),
    ("representation", representation_cost),
    ("representation_depth_2", representation_depth2_cost),
    ("CCJ", ccj_cost),
    ("CCJ_linearization", ccj_linear_cost),
)


def artifact_payloads(path: Path) -> tuple[bytes, bytes]:
    try:
        archive = path.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read accepted artifact: {exc}") from exc
    if _sha(archive) != ARTIFACT_SHA256:
        raise AuditError(f"artifact checksum mismatch: expected {ARTIFACT_SHA256}, got {_sha(archive)}")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            notebook, dumped = zf.read(NOTEBOOK_MEMBER), zf.read(PICKLE_MEMBER)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise AuditError(f"accepted artifact layout mismatch: {exc}") from exc
    if _sha(notebook) != NOTEBOOK_SHA256 or _sha(dumped) != PICKLE_SHA256:
        raise AuditError("accepted artifact member checksum mismatch")
    return notebook, dumped


def self_test(dumped: bytes) -> None:
    # Tiny closed forms defend success/memory conventions.
    p, pc = permutation_cost(8, 4, 2, 4)
    expected = math.log2(8 * 6 * 6) - math.log2((1 - 2 / 8) ** 2)
    if p or abs(pc[0] - expected) > 1e-12 or pc[1] != 5.0:
        raise AuditError("tiny permutation self-test failed")
    # Pinned known row: artifact stored values after its BCGIKRS-only sqrt(10000)
    # subtraction. This checks all transcribed routines and the immutable dump.
    try:
        known = pickle.loads(dumped)["BCGIKRS"][1]
    except Exception as exc:  # payload is accepted only after both checksums pass
        raise AuditError(f"cannot decode pinned known-row dump: {exc}") from exc
    n, k, w, _claimed = known["params"]
    b = n // w
    new_n, new_k = w * b, w * b - (n - k)
    loss = 0.5 * math.log2(n - k)
    names = {"permutation": "Perm", "enumeration": "Enum", "representation": "Rep",
             "representation_depth_2": "RepD2", "CCJ": "CCJ",
             "CCJ_linearization": "CCJ-lin"}
    for name, fn in FORMULAS:
        params, cost = _checked(name, lambda fn=fn: fn(new_n, new_k, w, b))
        stored = known["estimates"][names[name]]
        compared_time = cost[0] if name == "CCJ_linearization" else cost[0] - loss
        if _round(compared_time) != stored["time"] or _round(cost[1]) != stored["memory"]:
            raise AuditError(f"known artifact row mismatch for {name}")
        if "params" in stored and params != stored["params"]:
            raise AuditError(f"known artifact optimum mismatch for {name}")
    bjmm = known["estimates"]["BJMM"]
    if (bjmm["time"], bjmm["memory"], bjmm["params"]) != (88, 37, {"r": 10, "p": 2, "p1": 1, "l": 28}):
        raise AuditError("pinned generic-BJMM dump row mismatch")


def base(script_hash: str, analysis: str) -> dict[str, object]:
    return {field: "" for field in FIELDS} | {
        "date": DATE, "schema": SCHEMA, "status": STATUS,
        "source_revision": ARTIFACT_GIT_REVISION, "artifact_url": ARTIFACT_URL,
        "artifact_page": ARTIFACT_PAGE, "artifact_zip_sha256": ARTIFACT_SHA256,
        "source_member": NOTEBOOK_MEMBER, "source_member_sha256": NOTEBOOK_SHA256,
        "analysis_sha256": analysis, "script_sha256": script_hash,
    }


def finish(row: dict[str, object]) -> dict[str, object]:
    material = {k: row[k] for k in FIELDS if k != "record_sha256"}
    row["record_sha256"] = _sha(json.dumps(material, sort_keys=True, separators=(",", ":")).encode())
    return row


def direct_rows(script_hash: str, analysis: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate, degree, c, t in CANDIDATES:
        n, k, w, b = c * degree, (c - 1) * degree, c * t, degree // t
        validate_regular(n, k, w, b)
        orbit, loss = degree, 0.5 * math.log2(degree)
        for algorithm, fn in FORMULAS:
            failure = ""
            try:
                params, cost = _checked(algorithm, lambda fn=fn: fn(n, k, w, b))
            except AuditError as exc:
                params, cost, failure = {}, [math.nan, math.nan], str(exc)
            warning = "artifact has no q input; transfer from its source model to iid-uniform-F_p-star payloads is unproved"
            if failure:
                warning += "; retained exact artifact-formula failure: " + failure
            elif params.get("p") in (28, 32, 36) or params.get("eps_x") == 28 or params.get("eps_y") == 19:
                warning += "; optimum touches a finite artifact search boundary"
            row = base(script_hash, analysis)
            row.update({
                "row_kind": "direct_regular_sd_formula" if not failure else "direct_regular_sd_formula_incompatibility",
                "candidate": candidate,
                "distribution": "exact live direct: c polynomials; one uniform position/public bucket; iid uniform F_p* payloads",
                "source_model": "CRYPTO-2024 artifact regular-SD concrete formula; q is absent",
                "applicability": ("DIRECT_RSD_CANDIDATE_FIELD_TRANSFER_UNPROVED" if not failure
                                  else "IMMUTABLE_ARTIFACT_NUMERIC_INCOMPATIBILITY"),
                "N": n, "k": k, "w": w, "block_size_b": b,
                "coefficient_limb": "p0+p1 (q not an artifact input)",
                "coefficient_prime": "not_input", "algorithm": algorithm,
                "time_raw_log2": "" if failure else f"{cost[0]:.12f}",
                "memory_raw_log2": "" if failure else f"{cost[1]:.12f}",
                "optimal_parameters": json.dumps(params, sort_keys=True, separators=(",", ":")),
                "orbit_size": orbit, "orbit_sqrt_sensitivity_bits": f"{loss:.12f}",
                "time_optional_orbit_sensitivity_log2": "" if failure else f"{cost[0] - loss:.12f}",
                "orbit_treatment": ("no subtraction because raw formula failed; optional sensitivity retained separately"
                                    if failure else
                                    "raw preserved; separate optional -0.5*log2(n) sensitivity only; arbitrary-decoder speedup heuristic"),
                "domain_guard": ("reject: " + failure if failure else
                                 "pass: N=w*b, 0<k<N, k-w>0, finite optimum"),
                "executable": "no" if failure else "yes",
                "dependency_status": ("immutable formula numeric failure" if failure else
                                      "stdlib transcription; accepted zip and members verified"),
                "success_semantics": "artifact expected-work formula including its per-iteration success term where present",
                "memory_semantics": ("unavailable" if failure else
                                     "artifact log2 storage-cost proxy; unit not specified by notebook"),
                "data_semantics": "one public sample; orbit generated locally and not materialized",
                "assumptions": "artifact rank/list/independence and source field model; structured negacyclic matrix transfer unproved; q-ary payload transfer unproved",
                "warnings": warning,
            })
            rows.append(finish(row))
        for limb, prime in PRIMES:
            row = base(script_hash, analysis)
            row.update({
                "row_kind": "generic_bjmm_incompatibility", "candidate": candidate,
                "distribution": "exact live direct RSD with iid uniform F_p* payloads",
                "source_model": "artifact calls CryptographicEstimators binary SDEstimator(n,k,w,nsolutions=0)",
                "applicability": "INCOMPATIBLE_WITH_LIVE_QARY_MODEL", "N": n, "k": k, "w": w,
                "block_size_b": b, "coefficient_limb": limb, "coefficient_prime": prime,
                "algorithm": "generic_BJMM_depth_2", "orbit_size": orbit,
                "orbit_sqrt_sensitivity_bits": f"{loss:.12f}",
                "orbit_treatment": "no subtraction because no executable raw cost; optional orbit sensitivity remains heuristic",
                "domain_guard": "reject: no q parameter and external estimator revision absent",
                "executable": "no", "dependency_status": "IMMUTABLE_ARTIFACT_INCOMPATIBILITY: CryptographicEstimators/Sage dependency has no pinned revision/version",
                "success_semantics": "artifact passes nsolutions=0; exact dependency semantics unavailable from archive",
                "memory_semantics": "unavailable; known dump rows only, no candidate row",
                "data_semantics": "one public sample; q-specific live limb listed explicitly",
                "assumptions": "none silently imported",
                "warnings": "no BJMM number emitted; installing a current estimator would not reproduce the immutable artifact",
            })
            rows.append(finish(row))
    return rows


def projected_rows(script_hash: str, analysis: str, degree: int, weight: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if degree <= 1 or weight < 1 or weight > degree - 1:
        raise AuditError("projected fixed weight must satisfy degree>1 and 1<=weight<=degree-1")
    for candidate, n0, c, t in CANDIDATES:
        if n0 % degree:
            raise AuditError(f"{candidate}: projection degree must divide n")
        n, k, orbit, loss = c * degree, (c - 1) * degree, degree, 0.5 * math.log2(degree)
        for limb, prime in PRIMES:
            row = base(script_hash, analysis)
            row.update({
                "row_kind": "projected_fixed_weight_incompatibility", "candidate": candidate,
                "distribution": "conditioned exact occupancy/cancellation projection at fixed nonzero support W; not regular RSD",
                "source_model": "generic binary BJMM delegated by artifact; regular formulas require a regular block partition",
                "applicability": "VALID_FIXED_WEIGHT_DIAGNOSTIC_BUT_ARTIFACT_INCOMPATIBLE", "N": n, "k": k,
                "w": weight, "projection_degree": degree, "coefficient_limb": limb,
                "coefficient_prime": prime, "algorithm": "generic_BJMM_depth_2",
                "orbit_size": orbit, "orbit_sqrt_sensitivity_bits": f"{loss:.12f}",
                "orbit_treatment": "raw unavailable; separate optional -0.5*log2(d) sensitivity named but never subtracted",
                "domain_guard": f"fixed-weight combinatorial guard pass: 1<=W={weight}<=d-1={degree-1}; model/dependency guard reject",
                "executable": "no", "dependency_status": "IMMUTABLE_ARTIFACT_INCOMPATIBILITY: unpinned binary estimator and no q input",
                "success_semantics": "conditional fixed-weight event only; event probability must be composed separately from exact projection DP",
                "memory_semantics": "unavailable", "data_semantics": "one projected public sample; orbit size d",
                "assumptions": "random-code BJMM transfer to structured projected code is unproved",
                "warnings": "regular-ISD formulas intentionally not called: projected occupancy/cancellation is not RSD; no cost emitted",
            })
            rows.append(finish(row))
    return rows


def render(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pinned CRYPTO-2024 Regular-ISD diagnostics; no security pin")
    p.add_argument("--artifact", type=Path, required=True, help="downloaded immutable crypto-2024-a1.zip")
    p.add_argument("--output", type=Path)
    p.add_argument("--projection-degree", type=int, default=64)
    p.add_argument("--projection-weight", type=int, default=63)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _notebook, dumped = artifact_payloads(args.artifact)
        self_test(dumped)
        if args.self_test:
            print("self-test: pinned archive, tiny formulas, known Regular-ISD and stored BJMM row pass", file=sys.stderr)
            return 0
        script_hash = _sha(Path(__file__).resolve().read_bytes())
        manifest = {"schema": SCHEMA, "artifact": ARTIFACT_SHA256, "notebook": NOTEBOOK_SHA256,
                    "pickle": PICKLE_SHA256, "script": script_hash, "candidates": CANDIDATES,
                    "primes": PRIMES, "projection": [args.projection_degree, args.projection_weight],
                    "orbit": "raw plus separate heuristic 0.5*log2(n_or_d) sensitivity"}
        analysis = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        rows = direct_rows(script_hash, analysis)
        rows.extend(projected_rows(script_hash, analysis, args.projection_degree, args.projection_weight))
        payload = render(rows)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_name(args.output.name + ".tmp")
            temp.write_bytes(payload)
            temp.replace(args.output)
        else:
            sys.stdout.buffer.write(payload)
        print(f"# {STATUS}; rows={len(rows)}; output_sha256={_sha(payload)}; analysis_sha256={analysis}; no security parameter is pinned", file=sys.stderr)
        return 0
    except (AuditError, OSError) as exc:
        print(f"audit rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
