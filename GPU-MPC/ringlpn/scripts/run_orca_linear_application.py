#!/usr/bin/env python3
"""Run the source-native terminal Ring-LPN Orca application gate."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pathlib
import secrets
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

MODULUS = 1 << 32
PREFLIGHT_REJECTION = "bilateral linear preflight rejected"


class GateError(RuntimeError):
    pass


@dataclass
class PairResult:
    returncodes: tuple[int, int]
    logs: tuple[str, str]


@dataclass
class Fixture:
    kind: str
    model: str
    sid: int
    invocation: str
    records: tuple[pathlib.Path, pathlib.Path]
    states: tuple[pathlib.Path, pathlib.Path]
    record_digests: tuple[str, str]
    state_digests: tuple[str, str]
    input_shares: tuple[pathlib.Path, pathlib.Path]
    weight_shares: tuple[pathlib.Path, pathlib.Path]
    biases: tuple[pathlib.Path, pathlib.Path]
    shape_args: list[str]
    output_words: int
    expected: list[int]


def private_directory(path: pathlib.Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    metadata = path.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or
            metadata.st_uid != os.geteuid() or
            stat.S_IMODE(metadata.st_mode) != 0o700):
        raise GateError(f"unsafe private directory: {path}")


def write_private(path: pathlib.Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) |
        getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        cursor = 0
        while cursor < len(payload):
            count = os.write(descriptor, payload[cursor:])
            if count <= 0:
                raise GateError(f"short private write: {path}")
            cursor += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if stat.S_IMODE(path.lstat().st_mode) != 0o600:
        raise GateError(f"private file mode changed: {path}")


def write_words(path: pathlib.Path, words: list[int]) -> None:
    if any(value < 0 or value >= MODULUS for value in words):
        raise GateError("attempted to write a noncanonical ring word")
    write_private(path, b"".join(struct.pack("<Q", value) for value in words))


def read_words(path: pathlib.Path, count: int) -> list[int]:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or
            metadata.st_uid != os.geteuid() or metadata.st_nlink != 1 or
            stat.S_IMODE(metadata.st_mode) != 0o600 or
            metadata.st_size != count * 8):
        raise GateError(f"invalid output record: {path}")
    payload = path.read_bytes()
    words = [value[0] for value in struct.iter_unpack("<Q", payload)]
    if len(words) != count or any(value >= MODULUS for value in words):
        raise GateError(f"noncanonical output record: {path}")
    return words


def require_private_artifact(path: pathlib.Path) -> None:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or
            metadata.st_uid != os.geteuid() or metadata.st_nlink != 1 or
            stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size <= 0):
        raise GateError(f"invalid private fixture artifact: {path}")


def require_binary(path: pathlib.Path) -> None:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or
            metadata.st_uid != os.geteuid() or not os.access(path, os.X_OK)):
        raise GateError(f"missing or unsafe executable: {path}")


def split_shares(clear: list[int]) -> tuple[list[int], list[int]]:
    party_zero = [secrets.randbits(32) for _ in clear]
    party_one = [(value - share) % MODULUS
                 for value, share in zip(clear, party_zero, strict=True)]
    return party_zero, party_one


def kill_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def request_termination(signum: int, _frame: object) -> None:
    # Unwind subprocess and scratch finally blocks on termination as on errors.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise SystemExit(128 + signum)


class GateRunner:
    def __init__(self, root: pathlib.Path, work: pathlib.Path,
                 p0_gpu: str, p1_gpu: str, timeout: float,
                 fc_port: int, conv_port: int) -> None:
        self.root = root
        self.project = root.parent
        self.work = work
        self.p0_gpu = p0_gpu
        self.p1_gpu = p1_gpu
        self.timeout = timeout
        self.fc_port = fc_port
        self.conv_port = conv_port
        self.bin = root / "bin"
        self.application = self.bin / "orca_inference_ringlpn"
        self.helper = self.bin / "test_orca_linear_helpers"
        self.api = root / "build/linear-library/bin/test_linear_preprocess_api"
        self.fc_preprocess = self.bin / "test_two_party_fc_preprocess"
        self.conv_preprocess = self.bin / "test_two_party_conv_preprocess"
        for executable in (
                self.application, self.helper, self.api, self.fc_preprocess,
                self.conv_preprocess):
            require_binary(executable)
        self.logs = work / "logs"
        private_directory(self.logs)

    def _pair_environment(self, party: int) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = (
            self.p0_gpu if party == 0 else self.p1_gpu
        )
        return environment

    def run_pair(self, label: str, p0_command: list[str],
                 p1_command: list[str]) -> PairResult:
        log_paths = (self.logs / f"{label}-p0.log",
                     self.logs / f"{label}-p1.log")
        processes: list[subprocess.Popen[bytes] | None] = [None, None]
        handles = [open(log_paths[0], "xb"), open(log_paths[1], "xb")]
        try:
            processes[0] = subprocess.Popen(
                p0_command, cwd=self.project, env=self._pair_environment(0),
                stdout=handles[0], stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(0.15)
            processes[1] = subprocess.Popen(
                p1_command, cwd=self.project, env=self._pair_environment(1),
                stdout=handles[1], stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.timeout
            while any(process is not None and process.poll() is None
                      for process in processes):
                if time.monotonic() >= deadline:
                    for process in processes:
                        kill_process(process)
                    raise GateError(f"{label} timed out")
                time.sleep(0.05)
            returncodes = tuple(
                process.wait() if process is not None else -1
                for process in processes
            )
        finally:
            for process in processes:
                kill_process(process)
                if process is not None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            for handle in handles:
                handle.close()
        logs = tuple(path.read_text(encoding="utf-8", errors="replace")
                     for path in log_paths)
        return PairResult(returncodes, logs)

    def run_single(self, label: str, command: list[str]) -> str:
        log_path = self.logs / f"{label}.log"
        process: subprocess.Popen[bytes] | None = None
        with open(log_path, "xb") as output:
            try:
                process = subprocess.Popen(
                    command, cwd=self.project, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                returncode = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise GateError(f"{label} timed out") from error
            finally:
                kill_process(process)
                if process is not None:
                    process.wait(timeout=5)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if returncode != 0:
            raise GateError(f"{label} failed with status {returncode}")
        return log

    def run_helper(self) -> None:
        result = self.run_pair(
            "orca-linear-helpers",
            [str(self.helper), "0", "127.0.0.1"],
            [str(self.helper), "1", "127.0.0.1"],
        )
        if result.returncodes != (0, 0) or any(
                "orca-linear-helpers,pass" not in log for log in result.logs):
            raise GateError("Orca linear helper regression failed")

    def _channel_auth_pair(self, p0: pathlib.Path,
                           p1: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        secret = secrets.token_bytes(32)
        paths = (p0 / "channel-auth.key", p1 / "channel-auth.key")
        write_private(paths[0], secret)
        write_private(paths[1], secret)
        return paths

    def generate_fixture(self, kind: str) -> Fixture:
        case = self.work / f"fixture-{kind}"
        private_directory(case)
        party_dirs = (case / "p0", case / "p1")
        for directory in party_dirs:
            private_directory(directory)
        ledger = case / "ledger"
        private_directory(ledger)
        auth = self._channel_auth_pair(*party_dirs)
        sid = secrets.randbelow((1 << 63) - 1) + 1
        invocation = secrets.token_hex(16)
        layer_ordinal = "1"
        prefixes = (party_dirs[0] / "key", party_dirs[1] / "key")
        states = (party_dirs[0] / "state.bin", party_dirs[1] / "state.bin")
        port = self.fc_port if kind == "fc" else self.conv_port
        common = [
            "--sid", str(sid), "--invocation-id", invocation,
            "--ledger", str(ledger), "--qbits", "128", "--bw", "32",
            "--ole-n", "8192", "--ole-c", "2", "--ole-t", "8",
            "--noise", "regular", "--layer-ordinal", layer_ordinal,
        ]
        if kind == "fc":
            binary = self.fc_preprocess
            common += ["--rows", "2", "--inner", "3", "--cols", "2"]
            suffix = "fc"
            model = "RingLPN-FC"
            shape_args = ["--rows", "2", "--inner", "3", "--cols", "2"]
        elif kind == "conv2d":
            binary = self.conv_preprocess
            common += [
                "--n", "1", "--h", "4", "--w", "4", "--ci", "1",
                "--fh", "3", "--fw", "3", "--co", "2",
                "--padding", "1", "--stride", "1",
            ]
            suffix = "conv"
            model = "RingLPN-Conv2D"
            shape_args = [
                "--n", "1", "--h", "4", "--w", "4", "--ci", "1",
                "--fh", "3", "--fw", "3", "--co", "2",
                "--padding", "1", "--stride", "1",
            ]
        else:
            raise GateError(f"unknown fixture kind: {kind}")
        commands: list[list[str]] = []
        for party in (0, 1):
            command = [
                str(binary), "--party", str(party), "--port", str(port),
                "--out-prefix", str(prefixes[party]), "--state-record",
                str(states[party]), "--channel-auth-file", str(auth[party]),
            ]
            if party == 1:
                command += ["--host", "127.0.0.1"]
            command += common
            commands.append(command)
        result = self.run_pair(f"preprocess-{kind}", *commands)
        if result.returncodes != (0, 0):
            raise GateError(f"{kind} preprocessing failed: {result.returncodes}")
        records = (
            pathlib.Path(f"{prefixes[0]}_p0.{suffix}"),
            pathlib.Path(f"{prefixes[1]}_p1.{suffix}"),
        )
        for artifact in (*records, *states):
            require_private_artifact(artifact)
        record_digests = tuple(path.read_bytes()[-32:].hex()
                               for path in records)
        state_digests = tuple(path.read_bytes()[-32:].hex()
                              for path in states)

        if kind == "fc":
            clear_input = [3, 5, 7, 11, 13, 17]
            clear_weight = [19, 23, 29, 31, 37, 41]
            public_bias = [181, 191]
            expected = []
            for row in range(2):
                for col in range(2):
                    value = public_bias[col]
                    for index in range(3):
                        value += (clear_input[row * 3 + index] *
                                  clear_weight[index * 2 + col])
                    expected.append(value % MODULUS)
        else:
            clear_input = list(range(1, 17))
            clear_weight = [3 + index % 7 for index in range(18)]
            public_bias = [17, 29]
            expected = []
            for output_h in range(4):
                for output_w in range(4):
                    for output_channel in range(2):
                        value = public_bias[output_channel]
                        for filter_h in range(3):
                            input_h = output_h + filter_h - 1
                            if input_h < 0 or input_h >= 4:
                                continue
                            for filter_w in range(3):
                                input_w = output_w + filter_w - 1
                                if input_w < 0 or input_w >= 4:
                                    continue
                                input_index = input_h * 4 + input_w
                                filter_index = (
                                    (output_channel * 3 + filter_h) * 3 +
                                    filter_w
                                )
                                value += (clear_input[input_index] *
                                          clear_weight[filter_index])
                        expected.append(value % MODULUS)
        if not all(clear_input) or not all(clear_weight) or not all(public_bias):
            raise GateError("fixture clear values must all be nonzero")
        input_split = split_shares(clear_input)
        weight_split = split_shares(clear_weight)
        input_paths = (party_dirs[0] / "input-share.bin",
                       party_dirs[1] / "input-share.bin")
        weight_paths = (party_dirs[0] / "weight-share.bin",
                        party_dirs[1] / "weight-share.bin")
        bias_paths = (party_dirs[0] / "bias.bin", party_dirs[1] / "bias.bin")
        for party in (0, 1):
            write_words(input_paths[party], input_split[party])
            write_words(weight_paths[party], weight_split[party])
            write_words(bias_paths[party], public_bias)
        return Fixture(
            kind=kind, model=model, sid=sid, invocation=invocation,
            records=records, states=states,
            record_digests=record_digests,
            state_digests=state_digests,
            input_shares=input_paths, weight_shares=weight_paths,
            biases=bias_paths, shape_args=shape_args,
            output_words=len(expected), expected=expected,
        )

    def run_public_api(self, fc: Fixture, conv: Fixture) -> None:
        command = [
            str(self.api),
            "--fc-p0-record", str(fc.records[0]),
            "--fc-p0-state", str(fc.states[0]),
            "--fc-p1-record", str(fc.records[1]),
            "--fc-p1-state", str(fc.states[1]),
            "--conv-p0-record", str(conv.records[0]),
            "--conv-p0-state", str(conv.states[0]),
            "--conv-p1-record", str(conv.records[1]),
            "--conv-p1-state", str(conv.states[1]),
        ]
        log = self.run_single("linear-library-api", command)
        if "linear-library-api,ok" not in log:
            raise GateError("public linear material API contract failed")
        print("linear-library-api,ok", flush=True)

    def application_commands(
            self, runtime: Fixture, material: Fixture,
            outputs: tuple[pathlib.Path, pathlib.Path], *,
            shape_overrides: dict[str, str] | None = None,
            state_overrides: tuple[pathlib.Path, pathlib.Path] | None = None,
            state_digest_overrides: tuple[str, str] | None = None,
            input_overrides: tuple[pathlib.Path, pathlib.Path] | None = None,
            sid_overrides: tuple[int, int] | None = None,
    ) -> tuple[list[str], list[str]]:
        shapes = list(runtime.shape_args)
        for key, value in (shape_overrides or {}).items():
            try:
                position = shapes.index(key)
            except ValueError as error:
                raise GateError(f"unknown shape override: {key}") from error
            shapes[position + 1] = value
        states = state_overrides or material.states
        state_digests = state_digest_overrides or material.state_digests
        inputs = input_overrides or runtime.input_shares
        sids = sid_overrides or (material.sid, material.sid)
        commands: list[list[str]] = []
        for party in (0, 1):
            commands.append([
                str(self.application), runtime.model, "32", "0", "2",
                str(party), "127.0.0.1",
                "--record", str(material.records[party]),
                "--state", str(states[party]),
                "--input-share", str(inputs[party]),
                "--weight-share", str(runtime.weight_shares[party]),
                "--bias", str(runtime.biases[party]),
                "--output", str(outputs[party]),
                "--sid", str(sids[party]),
                "--invocation-id", material.invocation,
                "--record-digest", material.record_digests[party],
                "--state-digest", state_digests[party],
                *shapes,
            ])
        return commands[0], commands[1]

    def output_paths(self, label: str) -> tuple[pathlib.Path, pathlib.Path]:
        directory = self.work / f"application-{label}"
        private_directory(directory)
        party_dirs = (directory / "p0", directory / "p1")
        for party_dir in party_dirs:
            private_directory(party_dir)
        return party_dirs[0] / "output.bin", party_dirs[1] / "output.bin"

    def run_success(self, fixture: Fixture) -> None:
        outputs = self.output_paths(f"success-{fixture.kind}")
        result = self.run_pair(
            f"application-success-{fixture.kind}",
            *self.application_commands(fixture, fixture, outputs),
        )
        if result.returncodes != (0, 0):
            raise GateError(
                f"{fixture.kind} application failed: {result.returncodes}"
            )
        observed = tuple(read_words(path, fixture.output_words)
                         for path in outputs)
        if observed[0] != fixture.expected or observed[1] != fixture.expected:
            raise GateError(f"{fixture.kind} output differs from clear oracle")
        print(f"ringlpn-orca-linear-application,{fixture.kind},pass",
              flush=True)

    def require_rejection(
            self, label: str, commands: tuple[list[str], list[str]],
            outputs: tuple[pathlib.Path, pathlib.Path],
            retained: dict[pathlib.Path, bytes] | None = None) -> None:
        retained = retained or {}
        result = self.run_pair(f"control-{label}", *commands)
        if result.returncodes[0] == 0 or result.returncodes[1] == 0:
            raise GateError(
                f"{label} control unexpectedly succeeded: {result.returncodes}"
            )
        if any(PREFLIGHT_REJECTION not in log for log in result.logs):
            raise GateError(f"{label} did not reject bilaterally in preflight")
        for output in outputs:
            if output in retained:
                if not output.is_file() or output.read_bytes() != retained[output]:
                    raise GateError(f"{label} changed its retained control output")
            elif output.exists() or output.is_symlink():
                raise GateError(f"{label} published an output record")
        print(f"ringlpn-orca-linear-control,{label},pass", flush=True)

    def rewrite_state(
            self, source: pathlib.Path, destination: pathlib.Path,
            mutate: Callable[[bytearray], None], refresh_digest: bool) -> str:
        payload = bytearray(source.read_bytes())
        if len(payload) < 192:
            raise GateError("mask-state fixture is too short")
        mutate(payload)
        if refresh_digest:
            payload[-32:] = hashlib.sha256(payload[:-32]).digest()
        write_private(destination, bytes(payload))
        return bytes(payload[-32:]).hex()

    def run_controls(self, fc: Fixture, conv: Fixture) -> None:
        outputs = self.output_paths("control-wrong-kind")
        self.require_rejection(
            "wrong-kind",
            self.application_commands(conv, fc, outputs), outputs,
        )

        outputs = self.output_paths("control-wrong-shape")
        self.require_rejection(
            "wrong-shape",
            self.application_commands(
                conv, conv, outputs, shape_overrides={"--padding": "0"}),
            outputs,
        )

        ordinal_root = self.work / "control-wrong-ordinal-state"
        private_directory(ordinal_root)
        ordinal_states = (ordinal_root / "p0.state", ordinal_root / "p1.state")
        ordinal_digests = []
        for party in (0, 1):
            ordinal_digests.append(self.rewrite_state(
                fc.states[party], ordinal_states[party],
                lambda payload: payload.__setitem__(
                    slice(24, 32), struct.pack("<Q", 2)),
                True,
            ))
        outputs = self.output_paths("control-wrong-ordinal")
        self.require_rejection(
            "wrong-ordinal",
            self.application_commands(
                fc, fc, outputs, state_overrides=ordinal_states,
                state_digest_overrides=(ordinal_digests[0],
                                        ordinal_digests[1])),
            outputs,
        )

        outputs = self.output_paths("control-swapped-state")
        self.require_rejection(
            "swapped-state",
            self.application_commands(
                fc, fc, outputs,
                state_overrides=(fc.states[1], fc.states[0]),
                state_digest_overrides=(fc.state_digests[1],
                                        fc.state_digests[0])),
            outputs,
        )

        corrupt_root = self.work / "control-corrupt-state-files"
        private_directory(corrupt_root)
        corrupt_states = (corrupt_root / "p0.state", corrupt_root / "p1.state")
        for party in (0, 1):
            self.rewrite_state(
                fc.states[party], corrupt_states[party],
                lambda payload: payload.__setitem__(
                    len(payload) // 2, payload[len(payload) // 2] ^ 1),
                False,
            )
        outputs = self.output_paths("control-corrupt-state")
        self.require_rejection(
            "corrupt-state",
            self.application_commands(
                fc, fc, outputs, state_overrides=corrupt_states),
            outputs,
        )

        permissive_root = self.work / "control-permissive-share-files"
        private_directory(permissive_root)
        permissive = permissive_root / "p0-input.bin"
        write_private(permissive, fc.input_shares[0].read_bytes())
        os.chmod(permissive, 0o644)
        outputs = self.output_paths("control-permissive-share")
        self.require_rejection(
            "permissive-share-file",
            self.application_commands(
                fc, fc, outputs,
                input_overrides=(permissive, fc.input_shares[1])),
            outputs,
        )

        outputs = self.output_paths("control-duplicate-output")
        sentinel = b"preexisting-output-control"
        write_private(outputs[0], sentinel)
        self.require_rejection(
            "duplicate-output",
            self.application_commands(fc, fc, outputs), outputs,
            retained={outputs[0]: sentinel},
        )

        outputs = self.output_paths("control-one-sided-metadata")
        mismatched_sid = fc.sid + 1 if fc.sid < (1 << 64) - 1 else fc.sid - 1
        self.require_rejection(
            "one-sided-metadata-mismatch",
            self.application_commands(
                fc, fc, outputs, sid_overrides=(mismatched_sid, fc.sid)),
            outputs,
        )

    def run(self) -> None:
        self.run_helper()
        fc = self.generate_fixture("fc")
        conv = self.generate_fixture("conv2d")
        self.run_public_api(fc, conv)
        self.run_success(fc)
        self.run_success(conv)
        self.run_controls(fc, conv)
        print("ORCA LINEAR APPLICATION PASS", flush=True)


def parse_args() -> argparse.Namespace:
    script_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=script_root)
    parser.add_argument("--work-root", type=pathlib.Path)
    parser.add_argument("--p0-gpu", default=os.environ.get("P0_GPU", "0"))
    parser.add_argument("--p1-gpu", default=os.environ.get("P1_GPU", "1"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--fc-port", type=int, default=28620)
    parser.add_argument("--conv-port", type=int, default=28630)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (not args.p0_gpu.isdigit() or not args.p1_gpu.isdigit() or
            int(args.p0_gpu) == int(args.p1_gpu) or
            not math.isfinite(args.timeout) or args.timeout <= 0 or
            args.fc_port == args.conv_port or
            any(port < 20400 or port > 29761
                for port in (args.fc_port, args.conv_port))):
        raise GateError("invalid GPU, timeout, or preprocessing-port selection")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_termination)
    root = args.root.resolve(strict=True)
    if args.work_root is None:
        work = pathlib.Path(tempfile.mkdtemp(
            prefix="ringlpn-orca-linear-application-"
        ))
        os.chmod(work, 0o700)
        retain = False
    else:
        work = args.work_root.absolute()
        if work.exists() or work.is_symlink():
            raise GateError("caller-supplied work root must not already exist")
        private_directory(work)
        retain = True
    try:
        GateRunner(
            root, work, args.p0_gpu, args.p1_gpu, args.timeout,
            args.fc_port, args.conv_port,
        ).run()
    finally:
        if not retain:
            shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"orca-linear-application: {error}", file=sys.stderr)
        raise SystemExit(1)
