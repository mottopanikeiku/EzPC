#!/usr/bin/env python3
"""Fresh, local feasibility experiments; never imports historical measurements.

Only the Python standard library and the existing FC executable are needed.
The executable's stock --check mode is an offline correctness consumer, not a
matched dealerless baseline. All raw process logs and records stay private and
are deleted; retained CSV cells are strictly schema/value allowlisted.
"""

import argparse
import contextlib
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
HARNESS = Path(__file__).resolve()
TRAINING = (
    ("CNN3-fc5", 100, 64, 10), ("ModelB-fc5", 100, 100, 10),
    ("CNN2-fc5", 100, 128, 10), ("P-SecureML-fc3", 128, 128, 10),
    ("AlexNet-gemm17", 100, 256, 10), ("P-LeNet-fc5", 128, 500, 10),
    ("ModelB-fc4", 100, 256, 100), ("CNN2-fc4", 100, 256, 128),
)
HELDOUT = (("P-SecureML-fc2", 128, 128, 128),
           ("AlexNet-gemm13", 100, 256, 256))
FEATURES = ("intercept", "ring_ole_instances", "cross_terms", "size_c", "payload_words")
SEED = 20260911
BOOTSTRAPS = 10000
PARTY_FIELDS = (
    "party,qbits,bw,rows,inner,cols,ole_n,ole_c,ole_t,noise,ring_batches,"
    "ring_application_slots,ring_bootstrap_slots,ring_ole_instances,slots_used,"
    "dpf_trees,dpf_string_ots,dpf_bit_triples,dpf_scalar_oles,"
    "dpf_epoch_zero_scalar_oles,dpf_pcg_scalar_oles,dpf_pcg_oles_reserved,"
    "dpf_pcg_oles_discarded,dpf_pcg_opening_words_sent,dpf_logical_opened_bits,"
    "dpf_meaningful_share_bits,spfss_key_bytes,public_a_seed_words_sent,"
    "derandomization_words_sent,conversions,conversion_logical_opened_bits,"
    "conversion_meaningful_share_bits,protocol_bytes_sent,protocol_direction_switches,"
    "total_us,status,protocol_dependency_rounds,preflight_us,ot_setup_us,"
    "dpf_phase_a_us,dpf_phase_b_us,dpf_phase_c_us,spfss_grouping_us,"
    "public_polynomial_exchange_us,gpu_ringlpn_expansion_us,derandomization_openings_us,"
    "conversion_us,serialization_us,commit_us,peak_host_rss_bytes,peak_gpu_bytes,"
    "min_gpu_free_bytes,transport_straight_bytes_sent,transport_straight_bytes_received,"
    "transport_reversed_bytes_sent,transport_reversed_bytes_received,base_ots,"
    "base_ot_setup_bytes_sent,base_ot_setup_bytes_received,channel_auth_straight_bytes_sent,"
    "channel_auth_straight_bytes_received,channel_auth_reversed_bytes_sent,"
    "channel_auth_reversed_bytes_received,transport_bytes_include_base_ot,"
    "base_ot_setup_dependency_rounds,invocation_id,ledger_digest,ot_backend,"
    "ot_backend_revision,ot_correlation_straight_bytes_sent,ot_correlation_straight_bytes_received,"
    "ot_correlation_reversed_bytes_sent,ot_correlation_reversed_bytes_received,"
    "ot_adjustment_bytes_sent,ot_adjustment_bytes_received,ot_ciphertext_bytes_sent,"
    "ot_ciphertext_bytes_received,ot_inventory_straight_declared,ot_inventory_straight_consumed,"
    "ot_inventory_reversed_declared,ot_inventory_reversed_consumed,ot_backend_review_status,"
    "ring_application_slots_discarded,ot_backend_bridge_sha256,dpf_breadth_evaluator_calls,"
    "dpf_root_to_leaf_evaluator_calls"
).split(",")
CHECK_FIELDS = (
    "qbits,bw,rows,inner,cols,ring_batches,final_payload_bytes_per_party,"
    "matched_dealer_keygen_us,checker_two_share_online_us,matched_dealer_keygen_contract,"
    "key_order,online_contract,status,checker_us,peak_host_rss_bytes,peak_gpu_bytes,"
    "min_gpu_free_bytes,invocation_id,ledger_digest"
).split(",")
VARIABLE_FIELDS = {field for field in PARTY_FIELDS if field.endswith("_us")} | {
    "peak_host_rss_bytes", "peak_gpu_bytes", "min_gpu_free_bytes", "invocation_id", "ledger_digest"}
CONTRACT_FIELDS = [field for field in PARTY_FIELDS if field not in VARIABLE_FIELDS]
TIMER_CONTRACT = {
    "controller_launch_to_party_exit_us": "monotonic_ns immediately before first producer Popen through blocking observation of both producer/timeout exits; includes second launch and controller observation/scheduling overhead; excludes fixtures, hash checks, GPU queries, CSV parsing and checker",
    "Y_us": "max_party(total_us+preflight_us+ot_setup_us); excludes earlier PartyChannel construction, socket establishment, authentication, process launch and checker",
    "stages": "raw per-party *_us retained; max_party stage summaries are separate extrema, not additive critical-path components",
    "public_polynomial_exchange_us": "joint seed setup plus local domain-separated SHAKE XOF generation; not communication-only time",
}
SCOPE = (
    "One host, fixed q128/bw32/n8192/c2/t8 regular SCI/IKNP feasibility tuple; "
    "wall-stage capacity prediction, not GPU attribution, network rounds, security, "
    "full-model inference or generalization to another GPU. Advisory GPU locks and "
    "compute-process admission are not exclusive host leases, pinned clocks or immunity "
    "to shared-host interference. GPU0 is never selected or queried. New experiment; "
    "no claims from the deleted historical worker harness."
)


class ExperimentError(Exception):
    """Only fixed, nonsecret diagnostic codes belong in this exception."""


def require(condition, code):
    if not condition:
        raise ExperimentError(code)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def object_sha256(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_json(path, value):
    # O_EXCL also rejects existing symlinks. Never replaces earlier evidence.
    with path.open("xb") as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(path.parent)


def checkpoint(path, value):
    # Called only after this invocation has exclusively created the result file.
    fd, name = tempfile.mkstemp(prefix=".fc-review-result-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def planned(shape):
    label, m, k, n = shape
    x, c, w = m * k * n, m * n, m * k + k * n + m * n
    batches = (x + 7423) // 7424
    return {"label": label, "shape": [m, k, n], "ring_batches": batches,
            "ring_application_slots": 7424, "ring_bootstrap_slots": 768,
            "ring_ole_instances": 4 * batches, "cross_terms": x, "size_c": c,
            "payload_words": w, "final_payload_bytes_per_party": 8 * w,
            "features": [1, 4 * batches, x, c, w],
            "slot_utilization": x / (batches * 7424)}


def source_bindings(shapes):
    manifest = ROOT / "results/fc/orca_forward_linear_layer_manifest_2026_08_04.json"
    doc = json.loads(manifest.read_text())
    registry = {row["path"]: row["sha256"] for row in doc["source_registry"]}
    bindings = []
    for label, m, k, n in shapes:
        rows = [row for row in doc["layers"] if row["model"] + "-" + row["layer"] == label]
        require(len(rows) == 1, "source_label_missing_or_ambiguous")
        row = rows[0]
        require(row["operator"] == "fc" and [row[f] for f in ("rows", "inner", "cols")] == [m, k, n],
                "source_geometry_mismatch")
        anchors = {}
        for key in ("source_anchor", "batch_source_anchor"):
            match = re.fullmatch(r"([^:]+):(\d+)(?:-(\d+))?", row[key])
            require(match is not None, "source_anchor_invalid")
            relative, first, last = match.groups()
            source = (REPO / relative).resolve()
            require(source.is_relative_to(REPO) and relative in registry, "source_outside_registry")
            # Bind current source bytes, not an assertion that unrelated source
            # edits since the historical manifest never happened. The layer
            # constructor text/geometry and batch anchor must still validate.
            source_bytes = source.read_bytes()
            current_digest = hashlib.sha256(source_bytes).hexdigest()
            lines = source_bytes.decode().splitlines()
            start, stop = int(first), int(last or first)
            require(1 <= start <= stop <= len(lines), "source_anchor_range_invalid")
            text = "\n".join(lines[start - 1:stop])
            anchors[key] = {"anchor": row[key], "file_sha256": current_digest,
                            "historical_manifest_file_sha256": registry[relative],
                            "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest()}
            if key == "source_anchor":
                require(anchors[key]["text_sha256"] == row["source_text_sha256"], "source_text_mismatch")
                constructor = re.search(r"new\s+FC<T>\((\d+)\s*,\s*(\d+)\s*,", text)
                require(constructor is not None and tuple(map(int, constructor.groups())) == (k, n),
                        "source_constructor_mismatch")
            else:
                require(str(m) in text and row["batch"] == m, "source_batch_mismatch")
        bindings.append({"label": label, **anchors})
    return {"manifest_sha256": sha256(manifest), "layers": bindings}


def order_plan(mode, warmups, trials):
    order = []
    groups = [("worker", TRAINING[:1])] if mode == "worker" else [("training", TRAINING), ("heldout", HELDOUT)]
    for phase, shapes in groups:
        for role, count in (("warmup", warmups), ("measured", trials)):
            for trial in range(count):
                # Rotate shape order inside each trial, never collect a whole
                # training layer's distribution before moving to another layer.
                rotated = shapes[trial % len(shapes):] + shapes[:trial % len(shapes)]
                for shape in rotated:
                    sides = ("before", "after") if trial % 2 == 0 else ("after", "before")
                    if mode == "model":
                        sides = ("binary",)
                    for side in sides:
                        order.append({"ordinal": len(order), "phase": phase, "sample_role": role,
                                      "trial": trial, "side": side, "label": shape[0]})
    return order


def gpu_snapshot(gpus):
    snapshots = []
    fields = "index,name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.limit"
    for gpu in gpus:
        args = ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        result = subprocess.run(args, capture_output=True, timeout=15, check=False)
        require(result.returncode == 0, "gpu_occupancy_query_failed")
        require(not result.stdout.strip(), "selected_gpu_has_compute_process")
        args[2] = "--query-gpu=" + fields
        result = subprocess.run(args, capture_output=True, timeout=15, check=False)
        require(result.returncode == 0, "gpu_environment_query_failed")
        cells = next(csv.reader([result.stdout.decode().strip()]))
        require(len(cells) == len(fields.split(",")), "gpu_environment_schema_mismatch")
        # No UUIDs, hostnames, PIDs or process paths are retained.
        require(all(re.fullmatch(r"[A-Za-z0-9 ._()/-]+", cell) for cell in cells), "gpu_environment_invalid")
        snapshots.append(dict(zip(fields.split(","), (cell.strip() for cell in cells))))
    return snapshots


@contextlib.contextmanager
def gpu_locks(gpus):
    root = Path(os.environ.get("XDG_RUNTIME_DIR") or
                str(Path(os.environ.get("TMPDIR", "/tmp")) / f"ringlpn-gpu-locks-{os.getuid()}"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    require(not root.is_symlink() and info.st_uid == os.getuid() and info.st_mode & 0o077 == 0,
            "gpu_lock_root_not_owner_private")
    handles = []
    try:
        for gpu in sorted(gpus):
            fd = os.open(root / f"gpu-{gpu}.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handles.append(fd)
            require(os.fstat(fd).st_uid == os.getuid(), "gpu_lock_owner_mismatch")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ExperimentError("selected_gpu_lock_busy") from None
        yield
    finally:
        for fd in reversed(handles):
            os.close(fd)
        # Lock files are shared synchronization names, never harness fixtures.


def parse_metrics(path, fields):
    lines = path.read_text(errors="replace").splitlines()
    header = ",".join(fields)
    indices = [i for i, line in enumerate(lines) if line == header]
    require(len(indices) == 1 and indices[0] + 1 < len(lines), "csv_header_missing_or_ambiguous")
    values = next(csv.reader([lines[indices[0] + 1]]))
    require(len(values) == len(fields), "csv_width_mismatch")
    result = dict(zip(fields, values))
    enums = {"noise": {"regular"}, "status": {"pass", "FAIL"},
             "matched_dealer_keygen_contract": {"pass", "FAIL"}, "key_order": {"pass", "FAIL"},
             "online_contract": {"pass", "FAIL"}, "ot_backend": {"sci-iknp"},
             "ot_backend_revision": {"SCI-IKNP-IN-TREE"}, "ot_backend_review_status": {"existing-default"},
             "transport_bytes_include_base_ot": {"yes"}, "ot_backend_bridge_sha256": {"NA"}}
    for key, value in result.items():
        if key in enums:
            require(value in enums[key], "csv_categorical_value_invalid")
        elif key in {"invocation_id", "ledger_digest"}:
            require(re.fullmatch(r"[0-9a-f]{" + str(32 if key == "invocation_id" else 64) + "}", value) is not None,
                    "csv_identity_invalid")
        else:
            require(value == "NA" or re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value) is not None,
                    "csv_numeric_value_invalid")
            if value != "NA":
                require(math.isfinite(float(value)), "csv_numeric_value_nonfinite")
    return result


def accounting(sample, plan, invocation):
    p0, p1 = sample["parties"]
    check = sample["checker"]
    m, k, n = plan["shape"]
    r, x = plan["ring_ole_instances"], plan["cross_terms"]
    common = {"qbits": 128, "bw": 32, "rows": m, "inner": k, "cols": n,
              "ring_batches": plan["ring_batches"], "invocation_id": invocation}
    expected = {**common, "ole_n": 8192, "ole_c": 2, "ole_t": 8, "noise": "regular",
                "ring_application_slots": 7424, "ring_bootstrap_slots": 768,
                "ring_ole_instances": r, "slots_used": 4 * x, "dpf_trees": 256 * r,
                "dpf_string_ots": 22 * 256 * r, "dpf_bit_triples": 10 * 256 * r,
                "dpf_scalar_oles": 768 * r, "dpf_epoch_zero_scalar_oles": 1536,
                "dpf_pcg_scalar_oles": 768 * r - 1536, "dpf_pcg_oles_reserved": 768 * r,
                "dpf_pcg_oles_discarded": 1536, "dpf_pcg_opening_words_sent": 768 * r - 1536,
                "ring_application_slots_discarded": 7424 * r - 4 * x,
                "public_a_seed_words_sent": 4, "derandomization_words_sent": 4 * x,
                "conversions": m * n, "status": "pass", "base_ots": 256,
                "transport_bytes_include_base_ot": "yes", "ot_backend": "sci-iknp"}
    for party, row in enumerate((p0, p1)):
        require(all(row[key] == str(value) for key, value in {**expected, "party": party}.items()),
                "party_planned_accounting_mismatch")
        require(int(row["dpf_breadth_evaluator_calls"]) + int(row["dpf_root_to_leaf_evaluator_calls"]) > 0,
                "dpf_evaluator_accounting_missing")
        require(all(row[field] != "NA" for field in PARTY_FIELDS if field.endswith("_us")), "required_timer_missing")
        for direction in ("straight", "reversed"):
            require(row[f"channel_auth_{direction}_bytes_sent"] == (p1, p0)[party][f"channel_auth_{direction}_bytes_received"],
                    "channel_auth_accounting_mismatch")
        sent = sum(int(row[f"transport_{direction}_bytes_sent"]) for direction in ("straight", "reversed"))
        # SCI counters exclude the separately authenticated channel sockets.
        # Preflight sends its fixed 160-byte FC context plus one validity byte
        # before the post-OT protocol timer/byte baseline.
        accounted = int(row["protocol_bytes_sent"]) + int(row["base_ot_setup_bytes_sent"]) + 161
        require(sent == accounted, "transport_accounting_mismatch")
    require(p0["ledger_digest"] == p1["ledger_digest"] == check["ledger_digest"], "ledger_binding_mismatch")
    expected_check = {**common, "final_payload_bytes_per_party": plan["final_payload_bytes_per_party"],
                      "matched_dealer_keygen_contract": "pass", "key_order": "pass",
                      "online_contract": "pass", "status": "pass"}
    require(all(check[key] == str(value) for key, value in expected_check.items()), "checker_contract_mismatch")
    sample["Y_us"] = max(sum(float(row[field]) for field in ("total_us", "preflight_us", "ot_setup_us"))
                         for row in (p0, p1))
    require(sample["Y_us"] > 0, "nonpositive_primary_timer")
    sample["cross_terms_per_us"] = x / sample["Y_us"]
    sample["cross_terms_per_second"] = 1e6 * x / sample["Y_us"]
    sample["slot_utilization"] = plan["slot_utilization"]
    sample["accounting_status"] = "pass"


def terminate_groups(processes):
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in signals}
    try:
        for process in processes:
            # A reaped PID can be reused. Only signal groups whose wrapper
            # still has an unreaped PID owned by this Popen object.
            if process.returncode is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_sample(args, binary, pin, entry, plan, sample):
    processes = []
    sample.update({"status": "incomplete", "producer_return_codes": [None, None],
                   "checker_return_code": None, "parties": [None, None], "checker": None})
    # Ignore TMPDIR for private data: the fixed system temporary root must be
    # outside this checkout. Each sample gets new auth, ledgers and records.
    require(not Path("/tmp").resolve().is_relative_to(REPO), "private_tmp_inside_repository")
    with tempfile.TemporaryDirectory(prefix="ringlpn-fc-review-", dir="/tmp") as temporary:
        private = Path(temporary)
        os.chmod(private, 0o700)
        for name in ("party0", "party1", "ledger"):
            (private / name).mkdir(mode=0o700)
        secret = secrets.token_bytes(32)
        for party in (0, 1):
            (private / f"party{party}/auth").write_bytes(secret)
        del secret
        invocation, sid = secrets.token_hex(16), str(secrets.randbelow((1 << 63) - 1) + 1)
        sample.update({"invocation_id": invocation, "session_id": sid,
                       "binary_sha256": pin, "environment_pre": gpu_snapshot(args.gpus)})
        port = args.base_port + 2 * entry["ordinal"]
        m, k, n = plan["shape"]
        common = ["--host", "127.0.0.1", "--port", str(port), "--sid", sid,
                  "--invocation-id", invocation, "--ledger", str(private / "ledger"),
                  "--qbits", "128", "--bw", "32", "--rows", str(m), "--inner", str(k),
                  "--cols", str(n), "--ole-n", "8192", "--ole-c", "2", "--ole-t", "8",
                  "--noise", "regular", "--ot-backend", "sci-iknp", "--csv-header"]
        commands = [[str(binary), "--party", str(party), "--channel-auth-file", str(private / f"party{party}/auth"),
                     *common, "--out-prefix", str(private / f"party{party}/key")] for party in (0, 1)]
        checker = [str(binary), "--check", "--p0-record", str(private / "party0/key_p0.fc"),
                   "--p1-record", str(private / "party1/key_p1.fc"), "--csv-header"]
        sample["argv_templates"] = [[value.replace(str(private), "<PRIVATE_INVOCATION_ROOT>")
                                      if value != str(binary) else "<" + entry["side"].upper() + "_BIN>"
                                      for value in command] for command in [*commands, checker]]
        # The actual timeout wrapper is pinned separately in the plan.
        wrapper = [args.timeout_binary, "--kill-after=5", str(args.timeout)]
        envs = [{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                for gpu in args.gpus]
        try:
            require(sha256(binary) == pin and sha256(HARNESS) == args.harness_pin, "executable_changed_after_plan")
            with contextlib.ExitStack() as stack:
                logs = [stack.enter_context((private / name).open("wb")) for name in ("p0.log", "p1.log", "check.log")]
                start = time.monotonic_ns()
                for party in (0, 1):
                    processes.append(subprocess.Popen(wrapper + commands[party], stdout=logs[party], stderr=subprocess.STDOUT,
                                                      env=envs[party], start_new_session=True))
                for party, process in enumerate(processes):
                    sample["producer_return_codes"][party] = process.wait()
                sample["controller_launch_to_party_exit_us"] = (time.monotonic_ns() - start) / 1000
                # Do not poll or parse inside the primary timing boundary.
                for party in (0, 1):
                    try:
                        sample["parties"][party] = parse_metrics(private / f"p{party}.log", PARTY_FIELDS)
                    except ExperimentError as error:
                        sample.setdefault("parse_failures", []).append({"consumer": f"party{party}", "code": str(error)})
                require(sample["producer_return_codes"] == [0, 0], "producer_failed_or_timed_out")
                require(all(sample["parties"]), "producer_csv_invalid")
                require(not (private / "party0/key_p1.fc").exists() and not (private / "party1/key_p0.fc").exists(),
                        "party_output_ownership_violation")
                sample["environment_before_checker"] = gpu_snapshot(args.gpus)
                require(sha256(binary) == pin, "executable_changed_before_checker")
                check_start = time.monotonic_ns()
                process = subprocess.Popen(wrapper + checker, stdout=logs[2], stderr=subprocess.STDOUT,
                                           env=envs[2], start_new_session=True)
                processes.append(process)
                sample["checker_return_code"] = process.wait()
                sample["controller_checker_us"] = (time.monotonic_ns() - check_start) / 1000
                try:
                    sample["checker"] = parse_metrics(private / "check.log", CHECK_FIELDS)
                except ExperimentError as error:
                    sample.setdefault("parse_failures", []).append({"consumer": "checker", "code": str(error)})
                require(sample["checker_return_code"] == 0, "checker_failed_or_timed_out")
                require(sample["checker"] is not None, "checker_csv_invalid")
                accounting(sample, plan, invocation)
                sample["environment_post"] = gpu_snapshot(args.gpus)
                require(sha256(binary) == pin and sha256(HARNESS) == args.harness_pin, "executable_changed_during_sample")
                sample["status"] = "pass"
        finally:
            # Parent log descriptors may already be closed; children are killed
            # and reaped before TemporaryDirectory removes any private fixtures.
            terminate_groups(processes)
            sample["producer_return_codes"] = [processes[i].returncode if i < len(processes) else None for i in (0, 1)]
            if len(processes) == 3:
                sample["checker_return_code"] = processes[2].returncode
            # Preserve complete allowlisted rows even after interruption or a
            # peer failure; never copy unparsed process diagnostics.
            for consumer, filename, fields in (("party0", "p0.log", PARTY_FIELDS),
                                                ("party1", "p1.log", PARTY_FIELDS),
                                                ("checker", "check.log", CHECK_FIELDS)):
                try:
                    row = parse_metrics(private / filename, fields)
                    if consumer == "checker":
                        sample["checker"] = row
                    else:
                        sample["parties"][int(consumer[-1])] = row
                except (ExperimentError, OSError):
                    pass


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


class NNLS:
    """Five columns: enumerate faces, with reorthogonalized QR projections.

    No normal-equation inverse or ridge fallback. Columns have unit L2 norm.
    Rank-deficient faces are reported and skipped: their cone optimum has an
    independent face representation. Near-rank exclusion and KKT tolerances are
    explicit, frozen controls, not adjusted after fitting or seeing heldouts.
    """

    rank_tolerance = 1e-10
    coefficient_tolerance = 1e-10
    kkt_tolerance = 1e-7

    def __init__(self, rows):
        self.scales = [math.sqrt(math.fsum(row[j] ** 2 for row in rows)) for j in range(5)]
        self.rows = [[row[j] / self.scales[j] for j in range(5)] for row in rows]
        self.faces, self.singular_subsets = [], []
        columns = list(zip(*self.rows))
        for mask in range(1, 32):
            active = [j for j in range(5) if mask & (1 << j)]
            q, upper = [], [[0.0] * len(active) for _ in active]
            for j, index in enumerate(active):
                vector = list(columns[index])
                for _ in range(2):
                    for i, basis in enumerate(q):
                        projection = dot(basis, vector)
                        upper[i][j] += projection
                        vector = [v - projection * u for v, u in zip(vector, basis)]
                norm = math.sqrt(dot(vector, vector))
                if norm <= self.rank_tolerance:
                    break
                upper[j][j] = norm
                q.append([v / norm for v in vector])
            if len(q) != len(active):
                self.singular_subsets.append(active)
                continue
            # P = R^-1 Q^T, prepared once for all 10,001 fits.
            projection = [[0.0] * len(rows) for _ in active]
            for i in reversed(range(len(active))):
                for observation in range(len(rows)):
                    projection[i][observation] = (q[i][observation] - math.fsum(
                        upper[i][j] * projection[j][observation] for j in range(i + 1, len(active)))) / upper[i][i]
            self.faces.append((active, projection))

    def fit(self, y):
        require(len(y) == len(self.rows) and all(math.isfinite(v) for v in y), "nnls_response_invalid")
        best, best_sse = [0.0] * 5, dot(y, y)
        tolerance = self.coefficient_tolerance * max(1.0, max(map(abs, y)))
        for active, projection in self.faces:
            coefficients = [dot(row, y) for row in projection]
            if any(value < -tolerance for value in coefficients):
                continue
            candidate = [0.0] * 5
            for j, value in zip(active, coefficients):
                candidate[j] = max(0.0, value)
            residuals = [target - dot(row, candidate) for row, target in zip(self.rows, y)]
            sse = dot(residuals, residuals)
            if sse < best_sse:
                best, best_sse = candidate, sse
        residuals = [dot(row, best) - target for row, target in zip(self.rows, y)]
        tolerance = self.kkt_tolerance * max(1.0, math.sqrt(dot(y, y)))
        for j in range(5):
            gradient = math.fsum(row[j] * residual for row, residual in zip(self.rows, residuals))
            require(abs(gradient) <= tolerance if best[j] > 0 else gradient >= -tolerance, "nnls_kkt_failed")
        coefficients = [best[j] / self.scales[j] for j in range(5)]
        require(all(math.isfinite(v) for v in coefficients), "nnls_coefficients_nonfinite")
        return coefficients


def quantile(values, probability):
    """R-7 linear-interpolated empirical quantile (including bootstrap endpoints)."""
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (position - low) * (values[high] - values[low])


def describe(values):
    return {"n": len(values), "median": statistics.median(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def seal_training(samples, plans, plan_digest, binary_pin, harness_pin):
    raw = [sample for sample in samples if sample["phase"] == "training"]
    medians = [statistics.median(sample["Y_us"] for sample in raw
                                if sample["label"] == shape[0] and sample["sample_role"] == "measured")
               for shape in TRAINING]
    rows = [plans[shape[0]]["features"] for shape in TRAINING]
    heldout_rows = [plans[shape[0]]["features"] for shape in HELDOUT]
    fitter = NNLS(rows)
    coefficients = fitter.fit(medians)
    fitted = [dot(row, coefficients) for row in rows]
    residuals = [observed - predicted for observed, predicted in zip(medians, fitted)]
    mean_residual = statistics.mean(residuals)
    centered = [value - mean_residual for value in residuals]
    rng = random.Random(SEED)
    draws = [[] for _ in HELDOUT]
    for _ in range(BOOTSTRAPS):
        synthetic = [value + rng.choice(centered) for value in fitted]
        bootstrap_coefficients = fitter.fit(synthetic)
        for index, row in enumerate(heldout_rows):
            # Independent residual draw for each heldout and resample. No
            # clipping: negative lower bounds honestly expose model uncertainty.
            draws[index].append(dot(row, bootstrap_coefficients) + rng.choice(centered))
    return {"schema": "ringlpn.fc-review-training-seal.v1", "plan_sha256": plan_digest,
            "binary_sha256": binary_pin, "harness_sha256": harness_pin,
            "training_raw_sha256": object_sha256(raw),
            "training_sample_sha256": [{"ordinal": sample["ordinal"], "sha256": object_sha256(sample)} for sample in raw],
            "training_labels": [shape[0] for shape in TRAINING], "training_medians_us": medians,
            "feature_names": FEATURES, "coefficients": coefficients, "training_fitted_us": fitted,
            "training_centered_residuals_us": centered, "column_l2_scales": fitter.scales,
            "excluded_rank_deficient_or_near_singular_subsets": fitter.singular_subsets,
            "predictions": [{"label": shape[0], "predicted_median_us": dot(row, coefficients),
                             "pointwise_95_prediction_interval_us": [quantile(draw, 0.025), quantile(draw, 0.975)]}
                            for shape, row, draw in zip(HELDOUT, heldout_rows, draws)],
            "bootstrap": {"seed": SEED, "resamples": BOOTSTRAPS, "quantiles": "R-7",
                          "method": "centered residuals of eight training shape medians; refit NNLS; add independently drawn training residual at each heldout; pointwise not simultaneous; assumes exchangeable shape-median errors, does not estimate within-shape or heteroskedastic uncertainty"}}


def worker_summary(samples, trials):
    measured = [sample for sample in samples if sample["sample_role"] == "measured"]
    require(len(measured) == 2 * trials, "worker_sample_count_mismatch")
    # Compare all 68 established contract/accounting fields by party, both
    # within pairs and across repetitions, without demanding equal party traffic.
    for party in (0, 1):
        reference = {field: samples[0]["parties"][party][field] for field in CONTRACT_FIELDS}
        require(all({field: sample["parties"][party][field] for field in CONTRACT_FIELDS} == reference
                    for sample in samples), "worker_cross_version_accounting_mismatch")
    summary = {"contract_fields": CONTRACT_FIELDS, "fields_per_party": len(CONTRACT_FIELDS)}
    for side in ("before", "after"):
        rows = [sample for sample in measured if sample["side"] == side]
        summary[side] = {"controller_launch_to_party_exit_us": describe([s["controller_launch_to_party_exit_us"] for s in rows]),
                         "Y_us": describe([s["Y_us"] for s in rows]),
                         "max_party_dpf_phase_b_us": describe([max(float(p["dpf_phase_b_us"]) for p in s["parties"]) for s in rows])}
    ratios = []
    for trial in range(trials):
        pair = {sample["side"]: sample for sample in measured if sample["trial"] == trial}
        ratios.append(pair["before"]["controller_launch_to_party_exit_us"] / pair["after"]["controller_launch_to_party_exit_us"])
    summary["paired_before_over_after_controller_ratio"] = describe(ratios)
    summary["paired_ratios_in_trial_order"] = ratios
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("worker", "model"):
        command = commands.add_parser(name, help="counterbalanced before/after" if name == "worker" else "frozen training-only NNLS and unseen heldouts")
        if name == "worker":
            command.add_argument("--before-bin", type=Path, required=True, help="existing SCI FC executable before change")
            command.add_argument("--after-bin", type=Path, required=True, help="existing SCI FC executable after change")
        else:
            command.add_argument("--binary", type=Path, required=True, help="one existing SCI FC executable, pinned for all shapes")
        command.add_argument("--output", type=Path, required=True, help="new JSON result; new sibling STEM.plan.json and (model) STEM.seal.json; refuses reuse")
        command.add_argument("--p0-gpu", type=int, default=1, help="physical GPU index (default 1; GPU0 forbidden)")
        command.add_argument("--p1-gpu", type=int, default=2, help="physical GPU index (default 2; GPU0 forbidden)")
        command.add_argument("--check-gpu", type=int, default=3, help="distinct post-exit stock-checker GPU (default 3; GPU0 forbidden)")
        command.add_argument("--base-port", type=int, default=28080, help="first loopback port; reserves two consecutive ports per invocation")
        command.add_argument("--timeout", type=float, default=1800, help="seconds per child under GNU timeout, plus 5-second kill grace")
        command.add_argument("--warmups", type=int, default=1, help="warmups per shape/version, excluded from fitting (default 1)")
        command.add_argument("--trials", type=int, default=10, help="measured trials per shape/version (default 10)")
    return result


def interrupted(signum, _frame):
    raise ExperimentError("interrupted_signal_" + str(signum))


def main():
    args = parser().parse_args()
    os.umask(0o077)
    result, output_owned = None, False
    try:
        args.gpus = [args.p0_gpu, args.p1_gpu, args.check_gpu]
        require(all(gpu > 0 for gpu in args.gpus) and len(set(args.gpus)) == 3, "gpu0_or_duplicate_gpu_forbidden")
        require(math.isfinite(args.timeout) and args.timeout > 0 and args.warmups >= 1 and args.trials >= 1,
                "invalid_trial_or_timeout_configuration")
        require(not os.environ.get("RINGLPN_EMP_SILENT_BRIDGE"), "emp_bridge_environment_forbidden")
        require(not os.environ.get("LD_PRELOAD"), "preload_environment_forbidden")
        order = order_plan(args.command, args.warmups, args.trials)
        require(1 <= args.base_port and args.base_port + 2 * len(order) - 1 <= 65535, "port_range_invalid")
        output = args.output.absolute()
        require(output.suffix == ".json" and output.parent.is_dir(), "output_requires_existing_parent_and_json_suffix")
        plan_path, seal_path = output.with_suffix(".plan.json"), output.with_suffix(".seal.json")
        require(not any(path.exists() or path.is_symlink() for path in (output, plan_path, seal_path)), "evidence_path_already_exists")
        binaries = ({"before": args.before_bin.resolve(), "after": args.after_bin.resolve()}
                    if args.command == "worker" else {"binary": args.binary.resolve()})
        require(all(path.is_file() and os.access(path, os.X_OK) for path in binaries.values()), "binary_not_executable")
        pins = {side: sha256(path) for side, path in binaries.items()}
        args.harness_pin = sha256(HARNESS)
        args.timeout_binary = shutil.which("timeout")
        require(args.timeout_binary is not None, "gnu_timeout_required")
        shapes = TRAINING[:1] if args.command == "worker" else TRAINING + HELDOUT
        plans = {shape[0]: planned(shape) for shape in shapes}
        binding = source_bindings(shapes)
        result = {"schema": "ringlpn.fc-review-experiment.v1", "command": args.command,
                  "status": "incomplete", "scope": SCOPE, "samples": []}
        exclusive_json(output, result)
        output_owned = True
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        with gpu_locks(args.gpus):
            plan = {"schema": "ringlpn.fc-review-plan.v1", "command": args.command,
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "binary_sha256": pins, "harness_sha256": args.harness_pin,
                    "source_bindings": binding, "scope": SCOPE,
                    "environment": {"os": platform.system(), "kernel": platform.release(), "architecture": platform.machine(),
                                    "python": platform.python_version(), "cpu_logical_count": os.cpu_count(),
                                    "gpu_mapping": args.gpus, "gpus_initial": gpu_snapshot(args.gpus),
                                    "transport": "single-host IPv4 loopback; authenticated test channel",
                                    "cuda_device_order": "PCI_BUS_ID", "ld_library_path_sha256": hashlib.sha256(os.environ.get("LD_LIBRARY_PATH", "").encode()).hexdigest(),
                                    "host_pinning": "none; no affinity/power/clock changes", "gpu_locks": "existing advisory gpu-N.lock convention"},
                    "timeout": {"seconds": args.timeout, "kill_after_seconds": 5,
                                "wrapper_sha256": sha256(Path(args.timeout_binary)),
                                "argv_template": ["<GNU_TIMEOUT>", "--kill-after=5", str(args.timeout), "<BINARY>", "<ARGS>"]},
                    "warmups": args.warmups, "trials": args.trials, "invocations_planned": len(order),
                    "base_port": args.base_port, "port_rule": "base_port + 2*ordinal; adjacent port for reversed channel",
                    "tuple": {"qbits": 128, "bw": 32, "n": 8192, "c": 2, "t": 8, "noise": "regular", "ot_backend": "sci-iknp"},
                    "plans": plans, "ordering": order, "timers": TIMER_CONTRACT,
                    "csv_fields": {"party": PARTY_FIELDS, "checker": CHECK_FIELDS},
                    "worker_accounting_fields": CONTRACT_FIELDS,
                    "feature_names": FEATURES,
                    "feature_rules": "[1,R=4*ceil(MKN/(8192-3*2^2*8^2)),X=MKN,C=MN,W=MK+KN+MN]",
                    "fit": "NNLS on the eight measured training shape medians only; exhaustive active subsets; unit-L2 column scaling; twice-reorthogonalized QR and precomputed R^-1 Q^T; deterministic smallest-mask tie order; no regularization or retuning",
                    "fit_controls": {"rank_tolerance": NNLS.rank_tolerance, "normalized_coefficient_relative_tolerance": NNLS.coefficient_tolerance,
                                     "kkt_relative_tolerance": NNLS.kkt_tolerance, "singular_policy": "report excluded subsets; fail failed KKT rather than invent coefficients"},
                    "bootstrap": {"seed": SEED, "resamples": BOOTSTRAPS, "quantile": "R-7", "coverage": "pointwise 95%, not simultaneous",
                                  "rule": "center residuals of eight medians; resample to fitted training values; refit NNLS; add one independent training residual at each heldout; no clipping",
                                  "assumption": "exchangeable shape-median residuals; no heteroskedastic or within-shape uncertainty model"},
                    "failure_criterion": "any child/checker/CSV/accounting/environment failure; worker any cross-version contract mismatch; model either heldout abs(observed_median-predicted)/observed_median > 0.15 OR observed_median outside sealed pointwise interval; no exclusions, retries, pooling old runs or retuning",
                    "freeze_sequence": "exclusive fsynced plan before any producer; training including warmups; exclusive fsynced seal with sanitized training raw hashes, coefficients and intervals; only then heldout warmups and measurements"}
            exclusive_json(plan_path, plan)
            result["plan_sha256"] = sha256(plan_path)
            checkpoint(output, result)
            seal = None
            for entry in order:
                if entry["phase"] == "heldout" and seal is None:
                    seal = seal_training(result["samples"], plans, result["plan_sha256"], pins["binary"], args.harness_pin)
                    exclusive_json(seal_path, seal)
                    result["seal_sha256"] = sha256(seal_path)
                    result["training_seal"] = seal
                    checkpoint(output, result)
                require(sha256(plan_path) == result["plan_sha256"], "plan_changed_after_freeze")
                if seal is not None:
                    require(sha256(seal_path) == result["seal_sha256"], "seal_changed_after_freeze")
                sample = dict(entry)
                result["samples"].append(sample)
                try:
                    run_sample(args, binaries[entry["side"]], pins[entry["side"]], entry, plans[entry["label"]], sample)
                finally:
                    checkpoint(output, result)
            if args.command == "worker":
                result["summary"] = worker_summary(result["samples"], args.trials)
            else:
                evaluations = []
                for prediction in seal["predictions"]:
                    values = [sample["Y_us"] for sample in result["samples"] if sample["label"] == prediction["label"] and sample["sample_role"] == "measured"]
                    require(len(values) == args.trials, "heldout_sample_count_mismatch")
                    observed = statistics.median(values)
                    error = abs(observed - prediction["predicted_median_us"]) / observed
                    low, high = prediction["pointwise_95_prediction_interval_us"]
                    evaluations.append({**prediction, "observed_Y_us": describe(values), "absolute_relative_median_error": error,
                                        "inside_sealed_interval": low <= observed <= high,
                                        "status": "pass" if error <= 0.15 and low <= observed <= high else "FAIL"})
                result["heldout_evaluations"] = evaluations
                require(all(row["status"] == "pass" for row in evaluations), "heldout_prediction_rejected")
            result["status"] = "pass"
            checkpoint(output, result)
        print("FC review experiment: pass (sanitized JSON, plan" + (", seal)." if args.command == "model" else ")."))
        return 0
    except BaseException as error:
        # Never stringify arbitrary OS/subprocess exceptions: they can carry
        # private paths, argv or process output. Logs are never copied here.
        code = str(error) if isinstance(error, ExperimentError) else type(error).__name__
        if output_owned:
            result["status"], result["failure"] = "FAIL", code
            try:
                checkpoint(output, result)
            except BaseException:
                print("Failed to finalize result; last checkpoint remains incomplete.", file=sys.stderr)
        print("FC review experiment: FAIL (" + code + ").", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
