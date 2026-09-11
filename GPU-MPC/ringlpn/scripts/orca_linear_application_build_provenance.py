#!/usr/bin/env python3
"""Bind application provenance across compilation using producer locks and seals.

The build recipe holds the graph/linear archive producer locks through publication.
The seals detect ordinary concurrent source, recipe, tool, environment, and archive
changes; the canonical source path remains a live view, not an immutable snapshot.
This is not protection against a hostile same-user writer or adversarial ABA edits.
Private seals retain host paths/stat identities but never enter public provenance.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

import graph_build_provenance as provenance

SCHEMA = "ringlpn-orca-linear-application-build-provenance-v1"
ARTIFACTS = (
    "orca_inference_ringlpn",
    "test_orca_linear_helpers",
)
REQUIRED_DEPENDENCIES = frozenset({
    "GPU-MPC/backend/orca_base.h",
    "GPU-MPC/experiments/orca/orca_inference.cu",
    "GPU-MPC/ringlpn/src/linear_preprocess.h",
    "GPU-MPC/ringlpn/src/orca_terminal_linear_backend.cuh",
    "GPU-MPC/ringlpn/src/test_orca_linear_helpers.cu",
})
RECIPES = (
    "cmake/graph_libraries/CMakeLists.txt",
    "scripts/build_common.sh",
    "scripts/build_component.sh",
    "scripts/build_linear_library.sh",
    "scripts/build_orca_linear_application.sh",
    "scripts/graph_build_provenance.py",
    "scripts/orca_linear_application_build_provenance.py",
)
IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def checked_binding(path: pathlib.Path) -> dict[str, Any]:
    before = provenance.regular_nonsymlink(path, "sealed build input")
    binding = provenance.file_binding(path)
    after = provenance.regular_nonsymlink(path, "sealed build input")
    identity = [getattr(before, field) for field in IDENTITY_FIELDS]
    if identity != [getattr(after, field) for field in IDENTITY_FIELDS]:
        raise provenance.ProvenanceError(f"build input changed while sealing: {path}")
    return {"binding": binding, "identity": identity}


def read_seal(path: pathlib.Path) -> dict[str, Any]:
    _, payload = provenance.stable_file_binding(path, "private input seal", True)
    assert payload is not None
    return provenance.load_canonical_document(payload, "private input seal")


def check_watched(watched: dict[str, Any]) -> None:
    for raw_path, expected in watched.items():
        if checked_binding(pathlib.Path(raw_path)) != expected:
            raise provenance.ProvenanceError(f"build input changed across compilation: {raw_path}")


def prepare(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve(strict=True)
    watched = {}
    for relative in RECIPES:
        path = repo / "GPU-MPC/ringlpn" / relative
        provenance.tracked_file(path, repo)
        watched[str(path)] = checked_binding(path)
    provenance.write_atomic_noreplace(args.output, provenance.canonical(watched) + b"\n", 0o400)


def input_manifest(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result["artifacts"] = {
        name: {key: value for key, value in binding.items() if key not in {"sha256", "size"}}
        for name, binding in document["artifacts"].items()
    }
    return result


def environment() -> dict[str, str]:
    # Bash changes '_' to the last invoked command, not a compiler input.
    return {key: value for key, value in os.environ.items() if key != "_"}


def watched_inputs(document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve(strict=True)
    build = args.cmake_build.resolve(strict=True)
    bindings = {repo / relative: binding for relative, binding in document["dependencies"].items()}
    bindings.update({pathlib.Path(path): binding for path, binding in
                     document["environment_dependencies"]["dependencies"].items()})
    for section in ("archives", "tools"):
        for binding in document[section].values():
            bindings[provenance.expand_path(binding["path"], repo, build)] = binding
    watched = {}
    for path, expected in bindings.items():
        actual = checked_binding(path)
        if actual["binding"] != {key: expected[key] for key in ("sha256", "size")}:
            raise provenance.ProvenanceError(f"build input changed during collection: {path}")
        watched[str(path)] = actual
    # These raw recipes/command inputs are read by the collector itself. Depfiles
    # are deliberately regenerated after compilation; their parsed closures are
    # compared in the input manifest instead of pinning their changing mtimes.
    extra = [build / "compile_commands.json"]
    extra.extend(build / "CMakeFiles" / f"{target}.dir/link.txt"
                 for target, binding in document["archives"].items()
                 if binding.get("kind") != "linked-prebuilt")
    extra.extend(provenance.parse_assignment(value, "command")[1] for value in args.command)
    for path in extra:
        watched[str(path)] = checked_binding(path)
    return watched


def generate_checked(args: argparse.Namespace, *, capture: bool) -> None:
    recipes = read_seal(args.recipe_seal)
    check_watched(recipes)
    previous = None if capture else read_seal(args.input_seal)
    if previous is not None:
        check_watched(previous["watched"])
        if previous["environment"] != environment():
            raise provenance.ProvenanceError("build process environment changed across compilation")
    document = provenance.collect_manifest(args, bind_artifacts=not capture)
    expected_groups = {f"{kind}-{index}" for kind in ("application", "helper") for index in range(5)}
    if set(document["dependency_groups"]) != expected_groups:
        raise provenance.ProvenanceError("application build requires all ten translation-unit dependency groups")
    repo = args.repo_root.resolve(strict=True)
    if set(document["recipes"]) != {str(pathlib.Path(path).relative_to(repo)) for path in recipes}:
        raise provenance.ProvenanceError("application recipe inventory differs from startup seal")
    current = input_manifest(document)
    watched = watched_inputs(document, args)
    check_watched(recipes)
    if capture:
        seal = {"inputs": current, "watched": watched, "environment": environment()}
        provenance.write_atomic_noreplace(args.input_seal, provenance.canonical(seal) + b"\n", 0o400)
        return
    assert previous is not None
    if current != previous["inputs"]:
        changed = sorted(key for key in current if current[key] != previous["inputs"].get(key))
        raise provenance.ProvenanceError("build input manifest changed across compilation: " + ", ".join(changed))
    check_watched(previous["watched"])
    document["provenance_digest"] = provenance.self_digest(document, "provenance_digest")
    provenance.write_atomic_noreplace(args.output, provenance.canonical(document) + b"\n", 0o400)


def main() -> None:
    provenance.SCHEMA = SCHEMA
    provenance.BINARY_NAMES = ARTIFACTS
    provenance.REQUIRED_GRAPH_DEPENDENCIES = REQUIRED_DEPENDENCIES
    values = list(sys.argv[1:])
    operation = "generate"
    if values and values[0] in {"prepare", "capture", "generate", "verify"}:
        operation = values.pop(0)
    if operation == "prepare":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--repo-root", type=pathlib.Path, required=True)
        parser.add_argument("--output", type=pathlib.Path, required=True)
        prepare(parser.parse_args(values))
    elif operation == "verify":
        arguments = provenance.parser().parse_args([operation, *values])
        arguments.handler(arguments)
    else:
        seal_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        seal_parser.add_argument("--input-seal", type=pathlib.Path, required=True)
        seal_parser.add_argument("--recipe-seal", type=pathlib.Path, required=True)
        seals, remaining = seal_parser.parse_known_args(values)
        arguments = provenance.parser().parse_args(["generate", *remaining])
        arguments.input_seal = seals.input_seal
        arguments.recipe_seal = seals.recipe_seal
        generate_checked(arguments, capture=operation == "capture")


if __name__ == "__main__":
    try:
        main()
    except (provenance.ProvenanceError, OSError, ValueError) as error:
        provenance.fail(str(error))
