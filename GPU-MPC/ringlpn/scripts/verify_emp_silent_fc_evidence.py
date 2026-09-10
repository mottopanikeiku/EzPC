#!/usr/bin/env python3
"""Fail closed on the focused EMP SilentFerret FC correctness evidence contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import re
import sys
from typing import NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED_CASES = {
    "q64_regular_small": ("64", "16", "2", "2", "2", "regular", "1"),
    "q64_uniform_small": ("64", "16", "2", "2", "2", "uniform", "1"),
    "q128_regular_small": ("128", "32", "2", "2", "2", "regular", "1"),
    "q128_uniform_small": ("128", "32", "2", "2", "2", "uniform", "1"),
    "q64_regular_multibatch": ("64", "16", "8", "65", "16", "regular", "2"),
}
EXPECTED_CONTROLS = {
    "rogue_first_connector": ("reject_before_preflight_then_accept_genuine", "0", "0", "NA"),
    "replayed_authenticator": ("reject_replayed_nonce_tag_then_accept_genuine", "0", "0", "NA"),
    "opposite_direction_reflection": ("reject_direction_swap_then_accept_genuine", "0", "0", "NA"),
    "duplicate_id": ("duplicate_consume_once_reject", "2", "2", "NA"),
    "restart_retry": ("restart_cannot_rollback_ledger", "2", "2", "NA"),
    "tail_slot_reuse": ("unused_tail_is_discarded", "2", "2", "NA"),
    "invocation_collision": ("same_invocation_different_compatibility_sid_reject", "2", "2", "NA"),
    "ledger_truncation": ("malformed_append_only_entry_reject", "2", "2", "NA"),
    "ledger_payload_corruption": ("full_length_claim_digest_reject_before_network", "2", "2", "NA"),
    "preflight_mismatch": ("bilateral_reject_before_output", "2", "2", "NA"),
    "wrong_channel_secret": ("reject_before_preflight_ot_drbg_output", "2", "2", "NA"),
    "stale_output": ("bilateral_reject_without_overwrite", "2", "2", "NA"),
    "bootstrap_capacity": ("nonpositive_epoch_budget_reject", "2", "2", "NA"),
    "rename_failure": ("bilateral_cleanup_after_staging", "1", "1", "NA"),
    "corrupt_record": ("offline_digest_reject", "NA", "NA", "1"),
    "swapped_records": ("offline_party_header_reject", "NA", "NA", "1"),
}
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def fail(message: str) -> NoReturn:
    print(f"emp-silent-fc-evidence: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(pattern: str, path: pathlib.Path, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"cannot read {label} {path}: {error}")
    match = re.search(pattern, text)
    if match is None:
        fail(f"cannot extract {label} from {path}")
    return match.group(1)


def load_rows(path: pathlib.Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            if fields is None:
                fail(f"{label} has no header")
            if len(fields) != len(set(fields)):
                fail(f"{label} has duplicate header fields")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        fail(f"cannot parse {label} {path}: {error}")
    if not rows:
        fail(f"{label} has no rows")
    if any(None in row or any(value is None for value in row.values())
           for row in rows):
        fail(f"{label} row width differs from header")
    return fields, rows


def require_fields(fields: list[str], required: set[str], label: str) -> None:
    missing = sorted(required.difference(fields))
    if missing:
        fail(f"{label} is missing fields: {', '.join(missing)}")


def integer(row: dict[str, str], field: str, case: str, *, positive: bool = False) -> int:
    try:
        value = int(row[field])
    except (KeyError, ValueError):
        fail(f"{case}: {field} is not an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        fail(f"{case}: {field} is not {qualifier}")
    return value


def verify_peer_bytes(row: dict[str, str], stem: str, case: str) -> None:
    p0_sent = integer(row, f"p0_{stem}_bytes_sent", case)
    p1_sent = integer(row, f"p1_{stem}_bytes_sent", case)
    p0_received = integer(row, f"p0_{stem}_bytes_received", case)
    p1_received = integer(row, f"p1_{stem}_bytes_received", case)
    if p0_sent != p1_received or p1_sent != p0_received:
        fail(f"{case}: {stem} peer byte counters disagree")


def verify_live_rows(
    fields: list[str], rows: list[dict[str, str]], revision: str, bridge_digest: str
) -> None:
    required = {
        "case", "qbits", "bw", "rows", "inner", "cols", "noise", "ring_batches",
        "status", "matched_dealer_keygen_contract", "key_order", "unchanged_online",
        "invocation_id", "ledger_digest", "ring_application_slots", "ring_bootstrap_slots",
    }
    for party in ("p0", "p1"):
        required.update({
            f"{party}_ot_backend", f"{party}_ot_backend_revision",
            f"{party}_ot_backend_review_status", f"{party}_ot_backend_bridge_sha256",
            f"{party}_ot_inventory_straight_declared", f"{party}_ot_inventory_straight_consumed",
            f"{party}_ot_inventory_reversed_declared", f"{party}_ot_inventory_reversed_consumed",
            f"{party}_base_ots", f"{party}_base_ot_setup_bytes_sent",
            f"{party}_base_ot_setup_bytes_received", f"{party}_transport_bytes_include_base_ot",
            f"{party}_base_ot_setup_dependency_rounds", f"{party}_dpf_trees",
            f"{party}_dpf_breadth_evaluator_calls", f"{party}_dpf_root_to_leaf_evaluator_calls",
        })
    for stem in (
        "ot_correlation_straight", "ot_correlation_reversed", "ot_adjustment",
        "ot_ciphertext", "channel_auth_straight", "channel_auth_reversed",
    ):
        for party in ("p0", "p1"):
            required.add(f"{party}_{stem}_bytes_sent")
            required.add(f"{party}_{stem}_bytes_received")
    require_fields(fields, required, "live CSV")

    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        case = row["case"]
        if case in keyed:
            fail(f"duplicate live case {case}")
        keyed[case] = row
    if set(keyed) != set(EXPECTED_CASES):
        fail(
            "live case set differs: expected "
            f"{sorted(EXPECTED_CASES)}, got {sorted(keyed)}"
        )

    invocation_ids: set[str] = set()
    ledger_digests: set[str] = set()
    dimensions = ("qbits", "bw", "rows", "inner", "cols", "noise", "ring_batches")
    na_fields = (
        "base_ots", "base_ot_setup_bytes_sent", "base_ot_setup_bytes_received",
        "transport_bytes_include_base_ot", "base_ot_setup_dependency_rounds",
    )
    for case, expected in EXPECTED_CASES.items():
        row = keyed[case]
        actual = tuple(row[field] for field in dimensions)
        if actual != expected:
            fail(f"{case}: dimensions/noise differ: expected {expected}, got {actual}")
        for field in ("status", "matched_dealer_keygen_contract", "key_order", "unchanged_online"):
            if row[field] != "pass":
                fail(f"{case}: {field} is not pass")
        if row["ring_application_slots"] != "7424" or row["ring_bootstrap_slots"] != "768":
            fail(f"{case}: unexpected Ring-LPN slot partition")
        invocation_id = row["invocation_id"]
        ledger_digest = row["ledger_digest"]
        if not HEX32.fullmatch(invocation_id) or invocation_id in invocation_ids:
            fail(f"{case}: invocation_id is malformed or reused")
        if not HEX64.fullmatch(ledger_digest) or ledger_digest in ledger_digests:
            fail(f"{case}: ledger_digest is malformed or reused")
        invocation_ids.add(invocation_id)
        ledger_digests.add(ledger_digest)

        for party in ("p0", "p1"):
            if row[f"{party}_ot_backend"] != "emp-silent":
                fail(f"{case}: {party} did not select emp-silent")
            if row[f"{party}_ot_backend_revision"] != revision:
                fail(f"{case}: {party} backend revision differs from pinned EMP-OT")
            if row[f"{party}_ot_backend_review_status"] != "unreviewed-measured":
                fail(f"{case}: {party} backend review/measurement boundary differs")
            if row[f"{party}_ot_backend_bridge_sha256"] != bridge_digest:
                fail(f"{case}: {party} bridge digest differs from authorized bytes")
            for direction in ("straight", "reversed"):
                declared = integer(row, f"{party}_ot_inventory_{direction}_declared", case, positive=True)
                consumed = integer(row, f"{party}_ot_inventory_{direction}_consumed", case, positive=True)
                if declared != consumed:
                    fail(f"{case}: {party} {direction} inventory not consumed exactly")
            for suffix in na_fields:
                if row[f"{party}_{suffix}"] != "NA":
                    fail(f"{case}: {party}_{suffix} must remain NA for this backend boundary")
            trees = integer(row, f"{party}_dpf_trees", case, positive=True)
            breadth = integer(row, f"{party}_dpf_breadth_evaluator_calls", case, positive=True)
            roots = integer(row, f"{party}_dpf_root_to_leaf_evaluator_calls", case)
            if breadth * 64 != trees or roots != 0:
                fail(f"{case}: {party} did not use 64-tree breadth-first DPF batches")

        for direction in ("straight", "reversed"):
            declared0 = row[f"p0_ot_inventory_{direction}_declared"]
            declared1 = row[f"p1_ot_inventory_{direction}_declared"]
            if declared0 != declared1:
                fail(f"{case}: parties declared different {direction} OT inventories")
        for stem in (
            "ot_correlation_straight", "ot_correlation_reversed", "ot_adjustment",
            "ot_ciphertext",
        ):
            verify_peer_bytes(row, stem, case)
        auth_values = tuple(
            row[f"p{party}_channel_auth_{direction}_bytes_{flow}"]
            for direction in ("straight", "reversed")
            for party, flow in ((0, "sent"), (1, "sent"), (0, "received"), (1, "received"))
        )
        expected_auth = (
            ("416", "128", "480", "128", "128", "128", "128", "128")
            if case == "q64_regular_small"
            else ("128",) * 8
        )
        if auth_values != expected_auth:
            fail(f"{case}: authenticated-channel byte controls differ")


def verify_controls(fields: list[str], rows: list[dict[str, str]]) -> None:
    require_fields(
        fields, {"control", "expected", "p0_rc", "p1_rc", "checker_rc", "status"},
        "controls CSV",
    )
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["control"]
        if name in keyed:
            fail(f"duplicate control {name}")
        keyed[name] = row
    if set(keyed) != set(EXPECTED_CONTROLS):
        fail(
            "control set differs: expected "
            f"{sorted(EXPECTED_CONTROLS)}, got {sorted(keyed)}"
        )
    for name, expected in EXPECTED_CONTROLS.items():
        row = keyed[name]
        actual = (row["expected"], row["p0_rc"], row["p1_rc"], row["checker_rc"])
        if actual != expected or row["status"] != "pass":
            fail(f"{name}: expected {expected} and pass, got {actual} and {row['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    parser.add_argument("--controls", required=True, type=pathlib.Path)
    parser.add_argument("--bridge", required=True, type=pathlib.Path)
    parser.add_argument(
        "--authorization", type=pathlib.Path,
        default=ROOT / "src" / "emp_silent_bridge_authorization.h",
    )
    parser.add_argument(
        "--fetch-script", type=pathlib.Path,
        default=ROOT / "scripts" / "fetch_emp_silent.sh",
    )
    args = parser.parse_args()

    if not args.bridge.is_file():
        fail(f"bridge is not a regular file: {args.bridge}")
    bridge_digest = sha256(args.bridge)
    authorized_digest = extract(
        r'RINGLPN_EMP_SILENT_BRIDGE_SHA256\s*\\\s*"([0-9a-f]{64})"',
        args.authorization,
        "authorized bridge digest",
    )
    if bridge_digest != authorized_digest:
        fail("installed bridge bytes differ from compile-time authorization")
    revision = extract(r'(?m)^EMP_OT_REV="([0-9a-f]{40})"$', args.fetch_script, "EMP-OT revision")

    live_fields, live_rows = load_rows(args.csv, "live CSV")
    control_fields, control_rows = load_rows(args.controls, "controls CSV")
    verify_live_rows(live_fields, live_rows, revision, bridge_digest)
    verify_controls(control_fields, control_rows)
    print(
        "emp-silent-fc-evidence: PASS "
        f"({len(live_rows)} live cases, {len(control_rows)} controls; "
        "correctness and exact transport accounting only, not performance or security)"
    )


if __name__ == "__main__":
    main()
