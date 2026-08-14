#!/usr/bin/env python3
"""Validate the exact manifest/INDEX-authorized retained private-named provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import stat
import subprocess
from typing import Any, NoReturn

PRIVATE_SUFFIXES = frozenset({
    ".claim", ".conv", ".convert", ".fc", ".key", ".noise",
    ".record", ".spfss", ".state", ".truncate",
})
PRIVATE_DIRECTORY_NAMES = frozenset({
    "correlation-ledger", "ledger", "private_inputs", "two_party_gpu_keys",
    "two_party_keys", "two_party_outputs",
})
PRIVATE_DIRECTORY_PREFIXES = (
    "two_party_fc_work_", "two_party_conv_work_",
    "two_party_fc_model_scale_work_", "forward_linear_record_set_",
)
INDEX_SCHEMA = "ringlpn-known-zero-full-resnet18-retained-evidence-v2"
EXPECTED_INDEX_FILES = frozenset({
    "FULL_GRAPH.manifest",
    "adapter.csv",
    "adapter.log",
    "checker.csv",
    "checker.log",
    "controls.csv",
    "linear/LINEAR_RECORD_SET.manifest",
    "linear/private_inputs/binary_approval.json",
    "linear/private_inputs/linear_adapter_build_provenance.json",
    "party0.csv",
    "party0.log",
    "party1.csv",
    "party1.log",
    "private_inputs/graph_binary_approval.json",
    "private_inputs/graph_build_provenance.json",
    "private_inputs/linear_adapter_build_provenance.json",
    "private_inputs/linear_binary_approval.json",
})
EXPECTED_PRIVATE_INDEX_FILES = frozenset(
    path for path in EXPECTED_INDEX_FILES if "private_inputs" in pathlib.PurePosixPath(path).parts
)

INTERNAL_CHECKPOINT_BOUNDARY = {
    "classification": "internal/advisor-only",
    "external_circulation_blockers": [
        "manifest-bound host-identifying adapter provenance",
        "historical linear party/checker GPU assignment is not pairwise distinct",
    ],
    "release_replacement": (
        "fresh canonical three-GPU full-graph run with normalized provenance"
    ),
}


def fail(message: str) -> NoReturn:
    raise SystemExit("retained-public-evidence: " + message)


def canonical_digest(value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def load_regular_json(path: pathlib.Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {label} {path}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular non-symlink: {path}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object: {path}")
    return value, payload


def canonical_relative(value: object, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        fail(f"{label} is absent or not a string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        fail(f"{label} is not a canonical relative path: {value!r}")
    return path


def require_tracked(repo: pathlib.Path, relative: pathlib.PurePosixPath) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative.as_posix()],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if completed.returncode:
        fail("authorized retained provenance is not tracked: " + relative.as_posix())


def authorized_private_paths(
        repo: pathlib.Path, manifest_path: pathlib.Path,
        publication: bool,
) -> tuple[frozenset[pathlib.PurePosixPath], frozenset[pathlib.PurePosixPath]]:
    manifest, _ = load_regular_json(manifest_path, "publication manifest")
    if manifest.get("manifest_digest") != canonical_digest(manifest, "manifest_digest"):
        fail("publication manifest self-digest differs")
    bindings = manifest.get("required_tracked_evidence")
    if not isinstance(bindings, list):
        fail("publication manifest retained-evidence bindings are absent")

    boundary = manifest.get("retained_checkpoint_boundary")
    if boundary != INTERNAL_CHECKPOINT_BOUNDARY:
        fail("publication manifest retained-checkpoint boundary differs")
    if publication:
        fail(
            "retained full-graph checkpoint is internal/advisor-only; "
            + boundary["release_replacement"] + " is required before publication"
        )
    index_candidates: list[tuple[pathlib.PurePosixPath, str]] = []
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            fail("publication manifest retained-evidence binding is malformed")
        relative = canonical_relative(binding["path"], "retained-evidence path")
        digest = binding["sha256"]
        if relative.name == "INDEX.json":
            index_candidates.append((relative, digest))
    if len(index_candidates) != 1:
        fail("publication manifest must bind exactly one retained full-graph INDEX.json")

    index_relative, expected_index_sha = index_candidates[0]
    index_path = repo / index_relative
    index, index_payload = load_regular_json(index_path, "retained full-graph INDEX")
    if hashlib.sha256(index_payload).hexdigest() != expected_index_sha:
        fail("retained full-graph INDEX SHA-256 differs from static manifest")
    if (index.get("schema") != INDEX_SCHEMA or index.get("status") != "pass" or
            index.get("index_digest") != canonical_digest(index, "index_digest")):
        fail("retained full-graph INDEX schema/status/self-digest differs")
    files = index.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_INDEX_FILES:
        fail("retained full-graph INDEX file set differs from the exact public policy")

    index_root_relative = index_relative.parent
    index_root = repo / index_root_relative
    require_tracked(repo, index_relative)
    for name in sorted(EXPECTED_INDEX_FILES):
        canonical_name = canonical_relative(name, "retained INDEX file name")
        binding = files[name]
        if not isinstance(binding, dict) or set(binding) != {"bytes", "sha256"}:
            fail("retained INDEX file binding is malformed: " + name)
        path = index_root / canonical_name
        try:
            metadata = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            fail(f"cannot read retained INDEX file {name}: {error}")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            fail("retained INDEX file is not a regular non-symlink: " + name)
        if metadata.st_size != binding["bytes"] or hashlib.sha256(payload).hexdigest() != binding["sha256"]:
            fail("retained INDEX file byte/SHA-256 binding differs: " + name)
        require_tracked(repo, index_root_relative / canonical_name)

    full_graph, _ = load_regular_json(
        index_root / "FULL_GRAPH.manifest", "retained full-graph manifest",
    )
    assignment = full_graph.get("gpu_assignment")
    if not isinstance(assignment, dict):
        fail("retained full-graph GPU assignment is absent")
    try:
        linear_roles = {
            int(assignment[name])
            for name in ("linear_p0", "linear_p1", "linear_check")
        }
        graph_roles = {
            int(assignment[name])
            for name in ("graph_p0", "graph_p1", "linear_check")
        }
    except (KeyError, TypeError, ValueError) as error:
        fail(f"retained full-graph GPU assignment is malformed: {error}")
    if min(*linear_roles, *graph_roles) < 0:
        fail("retained full-graph GPU assignment contains a negative ordinal")
    legacy_role_reuse = len(linear_roles) != 3 or len(graph_roles) != 3

    linear, _ = load_regular_json(
        index_root / "linear/LINEAR_RECORD_SET.manifest",
        "retained linear record-set manifest",
    )
    lanes = linear.get("resource_schedule", {}).get("lanes")
    if not isinstance(lanes, list) or not lanes:
        fail("retained linear record-set resource lanes are absent")
    used_gpus: set[int] = set()
    for lane_number, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            fail(f"retained linear resource lane {lane_number} is malformed")
        try:
            roles = {
                int(lane[name]) for name in ("p0_gpu", "p1_gpu", "check_gpu")
            }
        except (KeyError, TypeError, ValueError) as error:
            fail(f"retained linear resource lane {lane_number} is malformed: {error}")
        if min(roles) < 0:
            fail(f"retained linear resource lane {lane_number} has a negative GPU ordinal")
        if len(roles) != 3:
            legacy_role_reuse = True
        if used_gpus.intersection(roles):
            legacy_role_reuse = True
        used_gpus.update(roles)
    if not legacy_role_reuse:
        fail("internal checkpoint boundary is stale: retained GPU assignments are release-valid")

    allowed_files = frozenset(
        index_root_relative / pathlib.PurePosixPath(name)
        for name in EXPECTED_PRIVATE_INDEX_FILES
    )
    allowed_directories = frozenset(path.parent for path in allowed_files)
    for directory in allowed_directories:
        actual = {
            child.relative_to(repo).as_posix()
            for child in (repo / directory).iterdir()
        }
        expected = {
            path.as_posix() for path in allowed_files if path.parent == directory
        }
        if actual != expected:
            fail("sanctioned private_inputs directory has extra/missing entries: " + directory.as_posix())
    return allowed_files, allowed_directories


def scan_private_artifacts(
        repo: pathlib.Path, manifest_path: pathlib.Path,
        publication: bool = False,
) -> None:
    repo = repo.resolve(strict=True)
    ringlpn = repo / "GPU-MPC/ringlpn"
    allowed_files, allowed_directories = authorized_private_paths(
        repo, manifest_path, publication,
    )
    survivors: list[str] = []
    for path in ringlpn.rglob("*"):
        relative = pathlib.PurePosixPath(path.relative_to(repo).as_posix())
        relative_to_ringlpn = path.relative_to(ringlpn)
        private_component = any(
            part in PRIVATE_DIRECTORY_NAMES or part.startswith(PRIVATE_DIRECTORY_PREFIXES)
            for part in relative_to_ringlpn.parts
        )
        private_suffix = path.suffix in PRIVATE_SUFFIXES
        if relative in allowed_files:
            continue
        if path.is_dir() and not path.is_symlink() and relative in allowed_directories:
            continue
        if path.is_symlink():
            if private_component or private_suffix:
                survivors.append(relative.as_posix())
        elif private_component or (path.is_file() and private_suffix):
            survivors.append(relative.as_posix())
        if len(survivors) >= 20:
            break
    if survivors:
        fail(
            "private key/state/ledger/scratch artifacts or private-named symlinks exist: "
            + " | ".join(survivors)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument(
        "--publication", action="store_true",
        help="reject internal/advisor-only retained evidence",
    )
    args = parser.parse_args()
    scan_private_artifacts(args.repo, args.manifest, args.publication)


if __name__ == "__main__":
    main()
