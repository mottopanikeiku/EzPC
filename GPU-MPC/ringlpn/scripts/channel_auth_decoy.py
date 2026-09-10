#!/usr/bin/env python3
"""Exercise pre-OT channel authentication with a loopback decoy."""

import argparse
import hashlib
import hmac
import os
import pathlib
import socket
import stat
import struct
import time


def read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("authenticated endpoint closed during decoy challenge")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def load_and_unlink_secret(path: pathlib.Path) -> bytearray:
    info = path.lstat()
    if (
        stat.S_IMODE(info.st_mode) != 0o600
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
    ):
        raise SystemExit("replay authenticator must be an owner-only regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        secret = bytearray(os.read(descriptor, 33))
    finally:
        os.close(descriptor)
    if len(secret) != 32:
        for index in range(len(secret)):
            secret[index] = 0
        raise SystemExit("replay authenticator must contain exactly 32 bytes")
    path.unlink()
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return secret


def make_tag(secret: bytearray, p0: bytes, p1: bytes, sender: int) -> bytes:
    message = b"RLPN-AUTH-TAG-V1" + p0 + p1 + bytes((sender, 1 - sender))
    return hmac.new(secret, message, hashlib.sha256).digest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--claim-file", required=True, type=pathlib.Path)
    parser.add_argument("--frame-direction", required=True, type=int, choices=(0, 1))
    parser.add_argument("--valid-replay-secret-file", type=pathlib.Path)
    args = parser.parse_args()
    invocation = bytes.fromhex(args.invocation_id)
    if len(invocation) != 16:
        raise SystemExit("invocation ID must encode 16 bytes")
    deadline = time.monotonic() + 4.0
    while not args.claim_file.is_file():
        if time.monotonic() >= deadline:
            raise SystemExit("party claim was not published before decoy deadline")
        time.sleep(0.005)
    claim = args.claim_file.read_bytes()
    if len(claim) < 32:
        raise SystemExit("party claim is truncated")
    claim_digest = claim[-32:]
    challenge = (
        b"RLPNAUTH"
        + struct.pack(">I", 1)
        + bytes((args.frame_direction, 1, 0, 0))
        + invocation
        + claim_digest
        + bytes(32)
    )
    tag = bytes(32)
    if args.valid_replay_secret_file is not None:
        secret = load_and_unlink_secret(args.valid_replay_secret_file)
        old_p0_challenge = (
            b"RLPNAUTH"
            + struct.pack(">I", 1)
            + bytes((args.frame_direction, 0, 1, 0))
            + invocation
            + claim_digest
            + bytes((0xA5,)) * 32
        )
        try:
            tag = make_tag(secret, old_p0_challenge, challenge, 1)
        finally:
            for index in range(len(secret)):
                secret[index] = 0
    if len(challenge) != 96:
        raise AssertionError("invalid challenge control size")
    while True:
        try:
            with socket.create_connection(("127.0.0.1", args.port), timeout=0.25) as sock:
                sock.sendall(challenge)
                try:
                    read_exact(sock, 96)
                    sock.sendall(tag)
                except (BrokenPipeError, ConnectionError, RuntimeError, socket.timeout):
                    pass
                return 0
        except (ConnectionRefusedError, socket.timeout):
            if time.monotonic() >= deadline:
                raise SystemExit("channel listener was unavailable before decoy deadline")
            time.sleep(0.005)


if __name__ == "__main__":
    raise SystemExit(main())
