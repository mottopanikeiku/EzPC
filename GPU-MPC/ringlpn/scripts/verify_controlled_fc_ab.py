#!/usr/bin/env python3
"""Verify same-physical-GPU FC preprocessing/dealer comparison evidence."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from decimal import Decimal, InvalidOperation
from typing import NoReturn

SCHEMA = "ringlpn.controlled-fc-ab.v1"
AUDIT_FIELDS = (
    "schema_version",
    "publication_date",
    "model",
    "source_layer",
    "trial",
    "sample_role",
    "invocation_id",
    "party0_gpu",
    "party1_gpu",
    "critical_party",
    "party0_setup_included_us",
    "party1_setup_included_us",
    "preprocess_gpu",
    "checker_gpu",
    "same_physical_gpu",
    "pre_protocol_quiescent",
    "pre_checker_quiescent",
    "post_checker_quiescent",
    "comparison_order",
    "status",
)
TIMING_FIELDS = {
    0: ("p0_total_us", "p0_preflight_us", "p0_ot_setup_us"),
    1: ("p1_total_us", "p1_preflight_us", "p1_ot_setup_us"),
}
HEX128 = re.compile(r"[0-9a-f]{32}\Z")
GPU_MAP = re.compile(
    r"party0:(?P<p0>[0-9]+),party1:(?P<p1>[0-9]+),"
    r"checker:per-sample-critical-party\Z"
)


def fail(message: str) -> NoReturn:
    print(f"controlled-fc-ab: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        fail(f"cannot read {path}: {error}")
    if fields is None:
        fail(f"missing CSV header: {path}")
    if len(fields) != len(set(fields)):
        fail(f"duplicate CSV header fields: {path}")
    if any(None in row or any(value is None for value in row.values())
           for row in rows):
        fail(f"CSV row width differs from header: {path}")
    return fields, rows


def read_environment(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            fail(f"duplicate environment field: {key}")
        values[key] = value
    return values


def decimal(value: str, field: str, invocation: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation:
        fail(f"{invocation}: {field} is not decimal")
    if not result.is_finite() or result < 0:
        fail(f"{invocation}: {field} must be finite and nonnegative")
    return result


def setup_included(row: dict[str, str], party: int) -> Decimal:
    invocation = row.get("invocation_id", "<missing>")
    return sum(
        (decimal(row.get(field, ""), field, invocation)
         for field in TIMING_FIELDS[party]),
        Decimal(0),
    )


def parse_gpu(value: str, field: str, invocation: str) -> int:
    try:
        gpu = int(value)
    except ValueError:
        fail(f"{invocation}: {field} is not an integer GPU ordinal")
    if gpu < 0 or str(gpu) != value:
        fail(f"{invocation}: {field} is not a canonical GPU ordinal")
    return gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=pathlib.Path)
    parser.add_argument("--audit", required=True, type=pathlib.Path)
    parser.add_argument("--environment", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_fields, raw_rows = read_csv(args.raw.resolve(strict=True))
    audit_fields, audit_rows = read_csv(args.audit.resolve(strict=True))
    environment = read_environment(args.environment.resolve(strict=True))

    required_raw = {
        "model", "source_layer", "trial", "sample_role", "retained", "status",
        "publication_date", "invocation_id", *TIMING_FIELDS[0], *TIMING_FIELDS[1],
        "matched_dealer_keygen_us", "matched_dealer_keygen_contract",
    }
    if not required_raw.issubset(raw_fields):
        fail("raw result is missing controlled-comparison fields")
    if tuple(audit_fields) != AUDIT_FIELDS:
        fail("controlled A/B audit header differs from the v1 schema")
    if environment.get("controlled_same_gpu") != "1":
        fail("environment does not bind controlled_same_gpu=1")
    match = GPU_MAP.fullmatch(environment.get("process_gpu_map", ""))
    if match is None:
        fail("environment lacks the dynamic critical-party checker map")
    environment_gpus = (int(match["p0"]), int(match["p1"]))
    if environment_gpus[0] == environment_gpus[1]:
        fail("environment assigns both parties to the same GPU")
    if environment.get("comparison_order") != (
        "dealer baseline after both preprocessing parties exit"
    ):
        fail("environment comparison order is absent or changed")
    if environment.get("critical_party_rule") != (
        "max(total_us+preflight_us+ot_setup_us); ties select party0"
    ):
        fail("environment critical-party rule is absent or changed")
    if not environment.get("gpu_quiescence_contract"):
        fail("environment lacks the GPU-quiescence contract")

    expected_rows = [
        row for row in raw_rows
        if row.get("sample_role") in {"warmup", "measured"}
        and row.get("retained") == "yes"
    ]
    if not expected_rows:
        fail("raw result contains no retained warmup or measured rows")
    if not any(row.get("sample_role") == "measured" for row in expected_rows):
        fail("raw result contains no measured rows")
    if any(row.get("status") != "pass" for row in expected_rows):
        fail("a retained comparison row did not pass")

    raw_by_invocation: dict[str, dict[str, str]] = {}
    for row in expected_rows:
        invocation = row.get("invocation_id", "")
        if HEX128.fullmatch(invocation) is None:
            fail("raw result contains a malformed invocation_id")
        if invocation in raw_by_invocation:
            fail(f"duplicate raw invocation_id: {invocation}")
        raw_by_invocation[invocation] = row

    audit_by_invocation: dict[str, dict[str, str]] = {}
    for row in audit_rows:
        invocation = row.get("invocation_id", "")
        if invocation in audit_by_invocation:
            fail(f"duplicate audit invocation_id: {invocation}")
        audit_by_invocation[invocation] = row
    if set(audit_by_invocation) != set(raw_by_invocation):
        missing = sorted(set(raw_by_invocation) - set(audit_by_invocation))
        extra = sorted(set(audit_by_invocation) - set(raw_by_invocation))
        fail(f"audit/raw invocation mismatch: missing={missing}, extra={extra}")

    for invocation, raw in raw_by_invocation.items():
        audit = audit_by_invocation[invocation]
        if raw["matched_dealer_keygen_contract"] != "pass":
            fail(f"{invocation}: matched dealer keygen contract did not pass")
        if decimal(raw["matched_dealer_keygen_us"],
                   "matched_dealer_keygen_us", invocation) <= 0:
            fail(f"{invocation}: matched dealer keygen timing must be positive")
        party0 = setup_included(raw, 0)
        party1 = setup_included(raw, 1)
        critical = 0 if party0 >= party1 else 1
        p0_gpu = parse_gpu(audit["party0_gpu"], "party0_gpu", invocation)
        p1_gpu = parse_gpu(audit["party1_gpu"], "party1_gpu", invocation)
        selected_gpu = (p0_gpu, p1_gpu)[critical]
        expected_literals = {
            "schema_version": SCHEMA,
            "publication_date": raw["publication_date"],
            "model": raw["model"],
            "source_layer": raw["source_layer"],
            "trial": raw["trial"],
            "sample_role": raw["sample_role"],
            "critical_party": str(critical),
            "preprocess_gpu": str(selected_gpu),
            "checker_gpu": str(selected_gpu),
            "same_physical_gpu": "yes",
            "pre_protocol_quiescent": "yes",
            "pre_checker_quiescent": "yes",
            "post_checker_quiescent": "yes",
            "comparison_order": "protocol_then_dealer",
            "status": "pass",
        }
        for field, expected in expected_literals.items():
            if audit.get(field) != expected:
                fail(
                    f"{invocation}: {field}={audit.get(field)!r}, "
                    f"expected {expected!r}"
                )
        if (p0_gpu, p1_gpu) != environment_gpus:
            fail(f"{invocation}: audit GPU map differs from environment")
        if p0_gpu == p1_gpu:
            fail(f"{invocation}: both parties use the same GPU")
        if decimal(audit["party0_setup_included_us"],
                   "party0_setup_included_us", invocation) != party0:
            fail(f"{invocation}: party0 setup-included timing differs from raw")
        if decimal(audit["party1_setup_included_us"],
                   "party1_setup_included_us", invocation) != party1:
            fail(f"{invocation}: party1 setup-included timing differs from raw")

    measured = sum(row["sample_role"] == "measured" for row in expected_rows)
    warmups = sum(row["sample_role"] == "warmup" for row in expected_rows)
    print(
        f"controlled-fc-ab: PASS ({measured} measured, {warmups} warmup; "
        "critical-party dealer baseline on the same quiescent GPU)"
    )


if __name__ == "__main__":
    main()
