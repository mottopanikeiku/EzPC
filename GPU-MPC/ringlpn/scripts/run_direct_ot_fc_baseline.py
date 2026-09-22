#!/usr/bin/env python3
"""Fresh same-function FC comparison, not security-matched or publication evidence.

Both producers use SCI/IKNP, real private randomness, two processes, both CRT
limbs (qbits=128), the same conversion and authenticated consume-once records.
An UNCHANGED reference executable checks records AND independent mask states.
The direct baseline has no Ring-LPN, DPF, public-polynomial or slot expansion.
Private auth, masks, records and raw logs are destroyed after every invocation.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import signal
import subprocess
import sys
import tempfile
import time

import run_fc_review_experiments as common

ROOT = Path(__file__).resolve().parents[1]
DIRECT_MARKER = "DIRECT_OT_FC_RESULT "
SOURCE_FILES = ("src/test_direct_ot_fc_preprocess.cu", "src/two_party_linear_preprocess.cuh",
                "src/two_party_ot.h", "src/secure_convert.cpp", "src/secure_convert.h",
                "scripts/build_component.sh", "scripts/build_two_party_fc_preprocess.sh",
                "scripts/build_common.sh", "scripts/run_fc_review_experiments.py",
                "scripts/run_direct_ot_fc_baseline.py")


def direct_metrics(path, entry, invocation):
    rows = [json.loads(line[len(DIRECT_MARKER):]) for line in path.read_text().splitlines()
            if line.startswith(DIRECT_MARKER)]
    common.require(len(rows) == 1, "direct_result_missing_or_ambiguous")
    row = rows[0]
    m, k, n = entry["shape"]
    oles = m * k * n * (4 if entry["qbits"] == 128 else 2)
    expected = {"schema_version": 1, "protocol": "direct-gilboa-ot-fc-v1", "status": "pass",
                "qbits": entry["qbits"], "bw": entry["bw"], "rows": m, "inner": k, "cols": n,
                "cross_terms": m * k * n, "scalar_oles": oles, "ole_ots": oles * 62,
                "conversions": m * n, "payload_bytes": (m * k + k * n + m * n) * 8,
                "base_ots": 256, "invocation_id": invocation}
    common.require(all(row.get(key) == value for key, value in expected.items()), "direct_work_accounting_failed")
    return row


def run_one(args, entry, pins, sources, sample):
    common.require(all(common.sha256(ROOT / path) == digest for path, digest in sources.items()),
                   "source_changed_after_plan")
    binaries = [args.direct if entry["side"] == "direct" else args.reference] * 2
    if entry["side"] == "mismatch":
        binaries = [args.direct, args.reference]
    common.require(all(common.sha256(binary) == pins[str(binary)] for binary in [*binaries, args.reference]),
                   "binary_changed_after_plan")
    active_gpus = args.gpus if entry["side"] == "ringlpn" else (args.gpus[2],)
    sample.update({"entry": entry, "status": "incomplete", "environment_pre": common.gpu_snapshot(active_gpus)})
    processes = []
    with tempfile.TemporaryDirectory(prefix="ringlpn-direct-ot-fc-", dir="/tmp") as temporary:
        private = Path(temporary)
        for name in ("party0", "party1", "ledger"):
            (private / name).mkdir(mode=0o700)
        secret = secrets.token_bytes(32)
        for party in (0, 1):
            (private / f"party{party}/auth").write_bytes(secret)
        del secret
        invocation, sid = secrets.token_hex(16), str(secrets.randbelow((1 << 63) - 1) + 1)
        m, k, n = entry["shape"]
        shared = ["--host", "127.0.0.1", "--port", str(args.base_port + 2 * entry["ordinal"]),
                  "--sid", sid, "--invocation-id", invocation, "--ledger", str(private / "ledger"),
                  "--layer-ordinal", "1", "--qbits", str(entry["qbits"]), "--bw", str(entry["bw"]),
                  "--rows", str(m), "--inner", str(k), "--cols", str(n),
                  "--ole-n", "8192", "--ole-c", "2", "--ole-t", "8", "--noise", "regular", "--csv-header"]
        commands = [[str(binaries[party]), "--party", str(party), *shared,
                     "--out-prefix", str(private / f"party{party}/key"),
                     "--state-record", str(private / f"party{party}/mask.state"),
                     "--channel-auth-file", str(private / f"party{party}/auth")] for party in (0, 1)]
        checker = [str(args.reference), "--check", "--csv-header", "--p0-record", str(private / "party0/key_p0.fc"),
                   "--p1-record", str(private / "party1/key_p1.fc"),
                   "--p0-state", str(private / "party0/mask.state"), "--p1-state", str(private / "party1/mask.state")]
        envs = [{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                for gpu in args.gpus]
        for party in (0, 1):
            if binaries[party] == args.direct or entry["side"] == "mismatch":
                envs[party]["CUDA_VISIBLE_DEVICES"] = ""
        wrapper = ["/usr/bin/timeout", "--kill-after=5", str(args.timeout)]
        try:
            with contextlib.ExitStack() as stack:
                logs = [stack.enter_context((private / name).open("wb")) for name in ("p0.log", "p1.log", "check.log")]
                begin = time.monotonic_ns()
                for party in (0, 1):
                    processes.append(subprocess.Popen(wrapper + commands[party], stdout=logs[party], stderr=subprocess.STDOUT,
                                                      env=envs[party], start_new_session=True))
                sample["producer_return_codes"] = [process.wait() for process in processes]
                sample["controller_launch_to_party_exit_us"] = (time.monotonic_ns() - begin) / 1000
                if entry["side"] == "mismatch":
                    common.require(all(code in (1, 2) for code in sample["producer_return_codes"]), "mixed_protocol_not_rejected")
                    rejection = "channel peer authentication failed before preflight"
                    common.require(rejection in (private / "p0.log").read_text(),
                                   "mixed_protocol_direct_rejection_reason_mismatch")
                    common.require("[two-party-linear] authenticated execution rejected" in
                                   (private / "p1.log").read_text(), "mixed_protocol_reference_rejection_mismatch")
                    claims = [(private / "ledger" / f"{invocation}.p{party}.claim").read_bytes()
                              for party in (0, 1)]
                    common.require(all(len(claim) == 132 and claim[:16] == b"RLPNFRESHLEDGER1" and
                                       claim[20:36] == bytes.fromhex(invocation) and
                                       hashlib.sha256(claim[:-32]).digest() == claim[-32:]
                                       for claim in claims), "mixed_protocol_claim_evidence_invalid")
                    common.require(claims[0][-32:] != claims[1][-32:], "protocol_identity_not_domain_separated")
                    sample["public_claim_digests"] = [claim[-32:].hex() for claim in claims]
                    sample["direct_rejection_stage"] = "channel_authentication_before_preflight"
                    sample["reference_failure_detail_exposed"] = False  # Public CLI deliberately catches all.
                    common.require(not list(private.glob("party*/*.fc")) and not list(private.glob("party*/*.state")),
                                   "mixed_protocol_published_output")
                    sample["status"] = "pass"
                    sample["control"] = "distinct_public_claims_and_direct_peer_authentication_rejection"
                    return sample
                common.require(sample["producer_return_codes"] == [0, 0], "producer_failed_or_timed_out")
                if entry["side"] == "direct":
                    sample["parties"] = [direct_metrics(private / f"p{party}.log", entry, invocation) for party in (0, 1)]
                    p0, p1 = sample["parties"]
                    common.require(p0["party"] == 0 and p1["party"] == 1 and p0["ledger_digest"] == p1["ledger_digest"],
                                   "direct_party_identity_mismatch")
                    common.require(all(row[direction + "_bytes_received"] is None
                                       for row in (p0, p1) for direction in ("straight", "reversed")),
                                   "unexpected_receive_measurement_claim")
                    sample["aggregate_wire_bytes_sent_including_authentication"] = sum(
                        row["straight_bytes_sent"] + row["reversed_bytes_sent"] + row["authentication_bytes_sent"]
                        for row in (p0, p1))
                else:
                    sample["parties"] = [common.parse_metrics(private / f"p{party}.log", common.PARTY_FIELDS) for party in (0, 1)]
                common.require(not (private / "party0/key_p1.fc").exists() and not (private / "party1/key_p0.fc").exists(),
                               "party_output_ownership_violation")
                common.require(not any((private / f"party{party}/auth").exists() for party in (0, 1)), "auth_not_consumed")
                common.require(common.sha256(args.reference) == pins[str(args.reference)], "checker_changed_after_plan")
                process = subprocess.Popen(wrapper + checker, stdout=logs[2], stderr=subprocess.STDOUT,
                                           env=envs[2], start_new_session=True)
                processes.append(process)
                sample["checker_return_code"] = process.wait()
                common.require(sample["checker_return_code"] == 0, "mask_state_or_stock_consumer_failed")
                sample["checker"] = common.parse_metrics(private / "check.log", common.CHECK_FIELDS)
                common.require(all(sample["checker"][field] == "pass" for field in
                                   ("status", "key_order", "online_contract", "matched_dealer_keygen_contract")), "checker_contract_failed")
                if entry["side"] == "ringlpn":
                    common.accounting(sample, common.planned(("CNN3-fc5", m, k, n)), invocation)
                if entry.get("replay_control"):
                    replay = []
                    for party in (0, 1):
                        # A different output path cannot reopen a consumed invocation.
                        command = [value.replace(f"party{party}/key", f"party{party}/retry-key")
                                   .replace(f"party{party}/mask.state", f"party{party}/retry.state") for value in commands[party]]
                        result = subprocess.run(wrapper + command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                env=envs[party], timeout=args.timeout + 10, check=False)
                        replay.append(result.returncode)
                    common.require(replay == [2, 2], "consumed_invocation_accepted")
                    common.require(not list(private.glob("party*/retry*")), "replay_published_output")
                    sample["replay_control_return_codes"] = replay
                sample["status"] = "pass"
                sample["environment_post"] = common.gpu_snapshot(active_gpus)
                return sample
        finally:
            common.terminate_groups(processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compare", "direct-only"), default="compare")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--base-port", type=int, default=32380)
    args = parser.parse_args()
    args.gpus = (1, 2, 3)  # Explicit physical device allowlist; never enumerate GPU0.
    os.umask(0o077)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, common.interrupted)
    common.require(1 <= args.trials <= 100 and 0 <= args.warmups <= 100 and args.timeout > 0, "invalid_run_size")
    args.direct, args.reference, args.output = (path.resolve() for path in (args.direct, args.reference, args.output))
    pins = {str(path): common.sha256(path) for path in (args.direct, args.reference)}
    sources = {path: common.sha256(ROOT / path) for path in SOURCE_FILES}
    order = [{"side": "direct", "role": "functional", "shape": shape, "qbits": qbits, "bw": bw,
              "replay_control": index == 0}
             for index, (shape, qbits, bw) in enumerate((((2, 2, 2), 64, 16), ((2, 2, 2), 128, 32),
                                                        ((2, 1025, 3), 128, 32)))]
    generator = random.Random(20260922)
    for role, count in (("warmup", args.warmups), ("measured", args.trials)):
        for trial in range(count):
            sides = ["direct", "ringlpn"] if args.mode == "compare" else ["direct"]
            generator.shuffle(sides)
            order.extend({"side": side, "role": role, "trial": trial, "shape": (100, 64, 10), "qbits": 128, "bw": 32}
                         for side in sides)
    order.append({"side": "mismatch", "role": "control", "shape": (2, 2, 2), "qbits": 128, "bw": 32})
    for ordinal, entry in enumerate(order):
        entry["ordinal"] = ordinal
    common.require(0 < args.base_port and args.base_port + 2 * len(order) < 65535, "port_range_invalid")
    plan = {"schema_version": 1, "scope": "local_same_function_not_security_matched_not_raw_diagonal",
            "mode": args.mode, "physical_gpus": args.gpus if args.mode == "compare" else (args.gpus[2],),
            "producer_devices": {"direct": "CPU only; CUDA hidden",
                                 "ringlpn": args.gpus[:2] if args.mode == "compare" else None,
                                 "mismatch_control": "CUDA hidden; authentication rejection required"},
            "checker_gpu": args.gpus[2],
            "binary_sha256": {"direct": pins[str(args.direct)], "reference": pins[str(args.reference)]},
            "timeout_sha256": common.sha256(Path("/usr/bin/timeout")), "source_sha256": sources, "order": order,
            "checker": "unchanged_reference_binary_with_independent_input_output_mask_states",
            "timing": "controller launch to both party exits; includes setup, excludes checker; no predictor refit",
            "security_bits": None, "publication_approval": False}
    plan_path = args.output.with_suffix(".plan.json")
    common.require(not args.output.exists() and not plan_path.exists(), "output_already_exists")
    common.exclusive_json(plan_path, plan)
    result = {"schema_version": 1, "status": "incomplete", "plan_sha256": common.sha256(plan_path), "samples": []}
    common.exclusive_json(args.output, result)
    try:
        with common.gpu_locks(args.gpus if args.mode == "compare" else (args.gpus[2],)):
            for entry in order:
                sample = {"entry": entry, "status": "incomplete"}
                result["samples"].append(sample)
                run_one(args, entry, pins, sources, sample)
                common.checkpoint(args.output, result)
                print(json.dumps({"ordinal": entry["ordinal"], "side": entry["side"], "role": entry["role"], "status": "pass"}), flush=True)
        result["measured"] = {side: common.describe([sample["controller_launch_to_party_exit_us"] for sample in result["samples"]
                                                    if sample["entry"]["role"] == "measured" and sample["entry"]["side"] == side])
                              for side in (("direct", "ringlpn") if args.mode == "compare" else ("direct",))}
        result["ringlpn_over_direct_median_wall_ratio"] = (
            result["measured"]["ringlpn"]["median"] / result["measured"]["direct"]["median"]
            if args.mode == "compare" else None)
        result["status"] = "pass"
        common.checkpoint(args.output, result)
        print(json.dumps({"status": "pass", "measured": result["measured"],
                          "scope": plan["scope"]}), flush=True)
        return 0
    except (common.ExperimentError, OSError, ValueError, subprocess.SubprocessError) as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        if isinstance(error, common.ExperimentError):
            result["error_code"] = str(error)
        common.checkpoint(args.output, result)
        print(json.dumps({"status": "failed", "error_type": result["error_type"],
                          "error_code": result.get("error_code")}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
