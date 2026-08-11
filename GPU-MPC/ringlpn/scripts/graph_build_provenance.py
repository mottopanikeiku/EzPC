#!/usr/bin/env python3
"""Generate, verify, and approve canonical full-graph build provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, NoReturn

SCHEMA = "ringlpn-full-graph-build-provenance-v2"
APPROVAL_SCHEMA = "ringlpn-full-graph-binary-approval-v3"
VALIDATION_RECEIPT_SCHEMA = "ringlpn-full-graph-validation-receipt-v1"
APPROVAL_SCOPE = "current_tree_full_graph_binaries_validated_for_known_zero_execution"
VALIDATION_CHECKS = frozenset({
    "contract_gate",
    "deterministic_build",
    "internal_source_review",
})
MINIMUM_APPROVAL_SOURCES = (
    "cmake/graph_libraries/CMakeLists.txt",
    "scripts/build_common.sh",
    "scripts/build_component.sh",
    "scripts/build_resnet18_full_graph.sh",
    "scripts/build_linear_library.sh",
    "scripts/check_resnet18_graph_contract.py",
    "scripts/graph_build_provenance.py",
    "scripts/run_resnet18_graph_contract_gate.sh",
    "scripts/run_resnet18_full_graph.py",
    "scripts/run_resnet18_full_graph.sh",
    "src/graph_mask_state.h",
    "src/linear_preprocess.h",
    "src/linear_preprocess_backend.cuh",
    "src/linear_preprocess_conv.cu",
    "src/private_file.h",
    "src/resnet18_graph_contract.h",
    "src/stock_nonlinear_full_record.h",
    "src/test_resnet18_graph_contract.cpp",
    "src/test_resnet18_full_graph.cu",
    "src/test_stock_nonlinear_full_keygen.cu",
)
BINARY_NAMES = (
    "test_resnet18_graph_contract",
    "test_resnet18_full_graph",
    "test_stock_nonlinear_full_keygen",
)

REQUIRED_GRAPH_DEPENDENCIES = frozenset({
    "GPU-MPC/ringlpn/src/emp_silent_adapter.h",
    "GPU-MPC/ringlpn/src/emp_silent_bridge.h",
    "GPU-MPC/ringlpn/src/emp_silent_bridge_authorization.h",
    "GPU-MPC/ringlpn/src/linear_preprocess.h",
    "GPU-MPC/ringlpn/src/private_file.h",
    "GPU-MPC/ringlpn/src/public_ring_vector_xof.h",
})


class ProvenanceError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise SystemExit(f"graph-build-provenance: {message}")


def canonical(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_file_binding(path: pathlib.Path, label: str = "bound file",
                        payload_required: bool = False,
                        ) -> tuple[dict[str, Any], bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProvenanceError(f"cannot open {label} {path}: {error}") from error
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if payload_required else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProvenanceError(
                f"{label} must be a regular non-symlink: {path}"
            )
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            total += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ProvenanceError(f"cannot read {label} {path}: {error}") from error
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field)
           for field in stable_fields) or total != before.st_size:
        raise ProvenanceError(f"{label} changed while reading: {path}")
    payload = b"".join(chunks) if chunks is not None else None
    return {"sha256": digest.hexdigest(), "size": total}, payload


def digest_file(path: pathlib.Path) -> str:
    return stable_file_binding(path)[0]["sha256"]


def require_digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ProvenanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def self_digest(document: dict[str, Any], field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    return digest_bytes(canonical(payload))

def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ProvenanceError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def load_canonical_document(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid {label}: {error}") from error
    if not isinstance(document, dict):
        raise ProvenanceError(f"{label} must be a JSON object")
    if payload != canonical(document) + b"\n":
        raise ProvenanceError(f"{label} is not canonical JSON")
    return document


def regular_nonsymlink(path: pathlib.Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ProvenanceError(f"cannot inspect {label} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProvenanceError(f"{label} must be a regular non-symlink: {path}")
    return metadata


def reject_symlink_components(path: pathlib.Path, root: pathlib.Path,
                              label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ProvenanceError(f"{label} is outside repository: {path}") from error
    current = root
    for component in relative.parts:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ProvenanceError(
                    f"{label} contains a symlink component: {path}"
                )
        except OSError as error:
            raise ProvenanceError(f"cannot inspect {label} {path}: {error}") from error


def tracked_file(path: pathlib.Path, repo: pathlib.Path) -> str:
    path = path.absolute()
    reject_symlink_components(path, repo, "dependency")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(f"cannot resolve dependency {path}: {error}") from error
    metadata = regular_nonsymlink(path, "dependency")
    del metadata
    try:
        worktree = pathlib.Path(subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()).resolve(strict=True)
        worktree.relative_to(repo)
        relative_worktree = path.relative_to(worktree).as_posix()
        subprocess.run(
            ["git", "-C", str(worktree), "ls-files", "--error-unmatch",
             "--", relative_worktree],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ProvenanceError(
            f"dependency is not tracked by this repository or a nested submodule: {path}"
        ) from error
    return path.relative_to(repo).as_posix()


def file_binding(path: pathlib.Path) -> dict[str, Any]:
    return stable_file_binding(path)[0]

def canonical_system_root(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as error:
        raise ProvenanceError(
            f"cannot inspect environment dependency root {label}: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ProvenanceError(
            f"environment dependency root must be a canonical directory: {label}"
        )
    return resolved


def classify_environment_dependency(
        path: pathlib.Path, roots: dict[str, pathlib.Path],
) -> tuple[str, str]:
    try:
        canonical_path = path.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"cannot resolve outside-repository dependency {path}: {error}"
        ) from error
    regular_nonsymlink(canonical_path, "environment dependency")
    matches: list[tuple[int, str]] = []
    for label, root in roots.items():
        try:
            canonical_path.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), label))
    if not matches:
        raise ProvenanceError(
            f"dependency is outside repository and approved system roots: {path}"
        )
    _, root_label = max(matches)
    return str(canonical_path), root_label


def parse_assignment(value: str, label: str) -> tuple[str, pathlib.Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in name):
        raise ProvenanceError(f"invalid {label} assignment: {value}")
    return name, pathlib.Path(raw_path).absolute()

def parse_reference(value: str, label: str) -> tuple[str, str]:
    name, separator, reference = value.partition("=")
    valid = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if (not separator or not name or not reference or
            any(character not in valid for character in name) or
            any(character not in valid for character in reference)):
        raise ProvenanceError(f"invalid {label} reference: {value}")
    return name, reference


def parse_depfile(path: pathlib.Path, cwd: pathlib.Path) -> list[pathlib.Path]:
    regular_nonsymlink(path, "compiler depfile")
    text = path.read_text(encoding="utf-8").replace("\\\n", "")
    escaped = False
    separator = -1
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            separator = index
            break
    if separator < 0:
        raise ProvenanceError(f"compiler depfile has no target separator: {path}")
    tokens: list[str] = []
    token: list[str] = []
    escaped = False
    for character in text[separator + 1:]:
        if escaped:
            token.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(character)
    if escaped:
        raise ProvenanceError(f"compiler depfile ends in an escape: {path}")
    if token:
        tokens.append("".join(token))
    if not tokens:
        raise ProvenanceError(f"compiler depfile has no dependencies: {path}")
    return [
        (candidate if candidate.is_absolute() else cwd / candidate).absolute()
        for candidate in map(pathlib.Path, tokens)
    ]


def canonical_path_argument(value: str, repo: pathlib.Path,
                            build: pathlib.Path) -> str:
    replacements = sorted(
        (
            (str(repo), "${REPO}"),
            (str(build), "${BUILD}"),
            (str(repo).lstrip("/"), "${REPO}"),
            (str(build).lstrip("/"), "${BUILD}"),
        ),
        key=lambda item: len(item[0]), reverse=True,
    )
    for original, replacement in replacements:
        value = value.replace(original, replacement)
    return value


def resolve_tool(value: str, cwd: pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if candidate.parent != pathlib.Path(".") or candidate.is_absolute():
        resolved = candidate if candidate.is_absolute() else cwd / candidate
    else:
        found = shutil_which(value)
        if found is None:
            raise ProvenanceError(f"build executable is unavailable: {value}")
        resolved = pathlib.Path(found)
    resolved = resolved.resolve(strict=True)
    regular_nonsymlink(resolved, "resolved build executable")
    return resolved


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", os.defpath).split(os.pathsep):
        candidate = pathlib.Path(directory or ".") / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def command_record(label: str, cwd: pathlib.Path, argv: list[str],
                   repo: pathlib.Path, build: pathlib.Path,
                   tools: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not argv:
        raise ProvenanceError(f"empty build command: {label}")
    executable = resolve_tool(argv[0], cwd)
    tool_path = canonical_path_argument(str(executable), repo, build)
    tool_binding = {"path": tool_path, **file_binding(executable)}
    tool_id = digest_bytes(canonical(tool_binding))
    existing = tools.setdefault(tool_id, tool_binding)
    if existing != tool_binding:
        raise ProvenanceError(f"build tool identity collision: {label}")
    return {
        "arguments": [canonical_path_argument(argument, repo, build)
                      for argument in argv[1:]],
        "cwd": canonical_path_argument(str(cwd.absolute()), repo, build),
        "label": label,
        "tool": tool_id,
    }


def read_command_file(path: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    regular_nonsymlink(path, "command record")
    fields = path.read_bytes().split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    try:
        decoded = [field.decode("utf-8") for field in fields]
    except UnicodeError as error:
        raise ProvenanceError(f"command record is not UTF-8: {path}") from error
    if len(decoded) < 2:
        raise ProvenanceError(f"command record is incomplete: {path}")
    return pathlib.Path(decoded[0]).absolute(), decoded[1:]


def dependency_command(argv: list[str], depfile: pathlib.Path) -> list[str]:
    result: list[str] = []
    skip = False
    dependency_options = {"-MD", "-MMD", "-MP", "-MG"}
    dependency_values = {"-MF", "-MT", "-MQ"}
    index = 0
    while index < len(argv):
        argument = argv[index]
        if skip:
            skip = False
        elif argument == "-o":
            skip = True
        elif argument in dependency_options:
            pass
        elif argument in dependency_values:
            skip = True
        elif argument != "-c":
            result.append(argument)
        index += 1
    result.extend(["-MM", "-MT", "provenance", "-MF", str(depfile)])
    return result


def load_compile_commands(path: pathlib.Path) -> list[dict[str, Any]]:
    regular_nonsymlink(path, "CMake compile database")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid CMake compile database: {error}") from error
    if not isinstance(document, list):
        raise ProvenanceError("CMake compile database must be an array")
    return document


def compile_argv(entry: dict[str, Any]) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        return list(arguments)
    command = entry.get("command")
    if isinstance(command, str):
        return shlex.split(command)
    raise ProvenanceError("CMake compile entry has no valid command")


def target_from_output(entry: dict[str, Any]) -> str:
    output = entry.get("output")
    if not isinstance(output, str):
        argv = compile_argv(entry)
        try:
            output = argv[argv.index("-o") + 1]
        except (ValueError, IndexError) as error:
            raise ProvenanceError("CMake compile entry has no output") from error
    marker = "CMakeFiles/"
    if marker not in output or ".dir/" not in output:
        raise ProvenanceError(f"cannot determine CMake target from output: {output}")
    return output.split(marker, 1)[1].split(".dir/", 1)[0]


def read_link_commands(build: pathlib.Path, target: str) -> list[list[str]]:
    path = build / "CMakeFiles" / f"{target}.dir" / "link.txt"
    regular_nonsymlink(path, "CMake archive command")
    commands = [
        shlex.split(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not commands or any(not command for command in commands):
        raise ProvenanceError(f"CMake archive command is empty: {path}")
    return commands


def generate(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve(strict=True)
    ringlpn = args.ringlpn_root.resolve(strict=True)
    build = args.cmake_build.resolve(strict=True)
    compile_entries = load_compile_commands(build / "compile_commands.json")
    ringlpn.relative_to(repo)
    system_roots: dict[str, pathlib.Path] = {}
    for assignment in args.system_root:
        label, root_path = parse_assignment(assignment, "system root")
        if label in system_roots:
            raise ProvenanceError(f"duplicate system root label: {label}")
        system_roots[label] = canonical_system_root(root_path, label)
        try:
            system_roots[label].relative_to(repo)
        except ValueError:
            pass
        else:
            raise ProvenanceError(
                f"system root must be outside repository: {label}"
            )
    if not system_roots:
        raise ProvenanceError("at least one canonical system root is required")
    tools: dict[str, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}
    environment_inputs: dict[str, dict[str, Any]] = {}

    def bind_dependencies(paths: list[pathlib.Path]) -> list[str]:
        result: set[str] = set()
        for dependency in paths:
            dependency = dependency.absolute()
            try:
                dependency.relative_to(repo)
            except ValueError:
                canonical_path, root_label = classify_environment_dependency(
                    dependency, system_roots
                )
                binding = {
                    "root": root_label,
                    **file_binding(pathlib.Path(canonical_path)),
                }
                previous = environment_inputs.setdefault(canonical_path, binding)
                if previous != binding:
                    raise ProvenanceError(
                        f"environment dependency classification changed: {canonical_path}"
                    )
                result.add(f"@system:{canonical_path}")
                continue
            relative = tracked_file(dependency, repo)
            binding = file_binding(dependency)
            previous = inputs.setdefault(relative, binding)
            if previous != binding:
                raise ProvenanceError(f"dependency changed while hashing: {relative}")
            result.add(relative)
        return sorted(result)

    with tempfile.NamedTemporaryFile(
            prefix="ringlpn-unapproved-dependency-", dir="/tmp") as control:
        try:
            bind_dependencies([pathlib.Path(control.name)])
        except ProvenanceError:
            pass
        else:
            raise ProvenanceError(
                "outside-repository dependency negative control unexpectedly passed"
            )

    recipes = bind_dependencies([pathlib.Path(path).absolute()
                                 for path in args.recipe])
    external_dependencies: dict[str, list[str]] = {}
    for assignment in args.depfile:
        label, depfile = parse_assignment(assignment, "depfile")
        if label in external_dependencies:
            raise ProvenanceError(f"duplicate depfile label: {label}")
        external_dependencies[label] = bind_dependencies(
            parse_depfile(depfile, ringlpn)
        )
    for assignment in args.command:
        label, command_file = parse_assignment(assignment, "command")
        cwd, argv = read_command_file(command_file)
        commands.append(command_record(label, cwd, argv, repo, build, tools))
    archive_assignments: dict[str, pathlib.Path] = {}
    for value in args.archive:
        name, archive_path = parse_assignment(value, "archive")
        if name in archive_assignments:
            raise ProvenanceError(f"duplicate archive assignment: {name}")
        archive_assignments[name] = archive_path
    linked_archive_assignments: dict[str, pathlib.Path] = {}
    for value in args.linked_archive:
        name, archive_path = parse_assignment(value, "linked archive")
        if name in linked_archive_assignments:
            raise ProvenanceError(f"duplicate linked archive assignment: {name}")
        linked_archive_assignments[name] = archive_path
    duplicate_archives = set(archive_assignments).intersection(
        linked_archive_assignments
    )
    if duplicate_archives:
        raise ProvenanceError(
            "archive target has multiple origins: " +
            ", ".join(sorted(duplicate_archives))
        )
    archive_sources: dict[str, list[str]] = {name: [] for name in archive_assignments}
    archive_dependencies: dict[str, set[str]] = {name: set() for name in archive_assignments}
    archive_compile_labels: dict[str, list[str]] = {name: [] for name in archive_assignments}
    dep_root = build / "provenance-depfiles"
    dep_root.mkdir(mode=0o700, exist_ok=True)
    for index, entry in enumerate(compile_entries):
        if not isinstance(entry, dict):
            raise ProvenanceError("invalid CMake compile entry")
        target = target_from_output(entry)
        if target not in archive_assignments:
            continue
        cwd_value = entry.get("directory")
        source_value = entry.get("file")
        if not isinstance(cwd_value, str) or not isinstance(source_value, str):
            raise ProvenanceError("CMake compile entry lacks directory or source")
        cwd = pathlib.Path(cwd_value).absolute()
        argv = compile_argv(entry)
        source = pathlib.Path(source_value)
        if not source.is_absolute():
            source = cwd / source
        source_relative = tracked_file(source.absolute(), repo)
        archive_sources[target].append(source_relative)
        label = f"archive-{target}-compile-{len(archive_compile_labels[target]):03d}"
        archive_compile_labels[target].append(label)
        commands.append(command_record(label, cwd, argv, repo, build, tools))
        depfile = dep_root / f"{target}-{index:04d}.d"
        dependency_argv = dependency_command(argv, depfile)
        try:
            subprocess.run(dependency_argv, cwd=cwd, check=True,
                           stdout=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ProvenanceError(
                f"compiler dependency generation failed for {source_relative}: {error}"
            ) from error
        archive_dependencies[target].update(bind_dependencies(
            parse_depfile(depfile, cwd)
        ))

    archives: dict[str, dict[str, Any]] = {}
    for target in sorted(archive_assignments):
        sources = sorted(archive_sources[target])
        if not sources:
            raise ProvenanceError(f"archive target has no compile entries: {target}")
        if len(sources) != len(set(sources)):
            raise ProvenanceError(f"archive target repeats a source: {target}")
        link_labels: list[str] = []
        for index, argv in enumerate(read_link_commands(build, target)):
            link_label = f"archive-{target}-link-{index:03d}"
            link_labels.append(link_label)
            commands.append(command_record(
                link_label, build, argv, repo, build, tools
            ))
        archive_path = archive_assignments[target]
        archives[target] = {
            **file_binding(archive_path),
            "compile_commands": archive_compile_labels[target],
            "dependencies": sorted(archive_dependencies[target]),
            "link_commands": link_labels,
            "path": canonical_path_argument(str(archive_path), repo, build),
            "sources": sources,
        }
    for target, archive_path in sorted(linked_archive_assignments.items()):
        archives[target] = {
            **file_binding(archive_path),
            "kind": "linked-prebuilt",
            "path": canonical_path_argument(str(archive_path), repo, build),
            "recipe_dependencies": recipes,
        }

    artifact_paths: dict[str, pathlib.Path] = {}
    for assignment in args.artifact:
        name, path = parse_assignment(assignment, "artifact")
        if name in artifact_paths:
            raise ProvenanceError(f"duplicate artifact assignment: {name}")
        artifact_paths[name] = path
    if set(artifact_paths) != set(BINARY_NAMES):
        raise ProvenanceError("artifact inventory must contain the three graph binaries")

    def reference_map(values: list[str], label: str) -> dict[str, list[str]]:
        mappings: dict[str, set[str]] = {name: set() for name in artifact_paths}
        for value in values:
            name, reference = parse_reference(value, label)
            if name not in mappings:
                raise ProvenanceError(
                    f"{label} names unknown artifact: {name}"
                )
            if reference in mappings[name]:
                raise ProvenanceError(
                    f"duplicate {label} reference: {value}"
                )
            mappings[name].add(reference)
        return {name: sorted(references)
                for name, references in mappings.items()}

    artifact_commands = reference_map(
        args.artifact_command, "artifact command"
    )
    artifact_dependency_groups = reference_map(
        args.artifact_dependency_group, "artifact dependency group"
    )
    artifact_archives = reference_map(
        args.artifact_archive, "artifact archive"
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in sorted(artifact_paths.items()):
        if not artifact_commands[name] or not artifact_dependency_groups[name]:
            raise ProvenanceError(
                f"artifact has no command or dependency-group closure: {name}"
            )
        artifacts[name] = {
            **file_binding(path),
            "archives": artifact_archives[name],
            "commands": artifact_commands[name],
            "dependency_groups": artifact_dependency_groups[name],
            "path": canonical_path_argument(str(path), repo, build),
        }
    labels = [command["label"] for command in commands]
    if len(labels) != len(set(labels)):
        raise ProvenanceError("duplicate build command label")
    command_closure = {
        reference
        for references in artifact_commands.values()
        for reference in references
    }
    dependency_group_closure = {
        reference
        for references in artifact_dependency_groups.values()
        for reference in references
    }
    archive_closure = {
        reference
        for references in artifact_archives.values()
        for reference in references
    }
    for binding in archives.values():
        command_closure.update(binding.get("compile_commands", []))
        command_closure.update(binding.get("link_commands", []))
    if command_closure != set(labels):
        raise ProvenanceError("artifact command closure differs from inventory")
    if dependency_group_closure != set(external_dependencies):
        raise ProvenanceError(
            "artifact dependency-group closure differs from inventory"
        )
    if archive_closure != set(archives):
        raise ProvenanceError("artifact archive closure differs from inventory")
    dependency_closure = set(recipes)
    for references in external_dependencies.values():
        dependency_closure.update(references)
    for binding in archives.values():
        dependency_closure.update(binding.get("dependencies", []))
        dependency_closure.update(binding.get("sources", []))
    environment_references = {
        f"@system:{path}" for path in environment_inputs
    }
    if dependency_closure != set(inputs).union(environment_references):
        raise ProvenanceError(
            "build dependency inventory is not source/dependency closed"
        )
    missing_dependencies = REQUIRED_GRAPH_DEPENDENCIES.difference(inputs)
    if missing_dependencies:
        raise ProvenanceError(
            "compiler depfiles omit required graph dependencies: " +
            ", ".join(sorted(missing_dependencies))
        )
    referenced_system_roots = {
        binding["root"] for binding in environment_inputs.values()
    }
    document: dict[str, Any] = {
        "archives": archives,
        "artifacts": artifacts,
        "commands": sorted(commands, key=lambda command: command["label"]),
        "dependencies": inputs,
        "dependency_groups": external_dependencies,
        "environment_dependencies": {
            "dependencies": environment_inputs,
            "roots": {
                label: str(root)
                for label, root in sorted(system_roots.items())
                if label in referenced_system_roots
            },
        },
        "recipes": recipes,
        "root": "${REPO}",
        "schema": SCHEMA,
        "tools": tools,
    }
    document["provenance_digest"] = self_digest(document, "provenance_digest")
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_noreplace(output, canonical(document) + b"\n", 0o400)


def write_atomic_noreplace(path: pathlib.Path, payload: bytes, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        destination = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = -1
        with destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as error:
        raise ProvenanceError(f"refusing existing output: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def expand_path(value: str, repo: pathlib.Path, build: pathlib.Path) -> pathlib.Path:
    if not isinstance(value, str):
        raise ProvenanceError("manifest path must be a string")
    expanded = value.replace("${REPO}", str(repo)).replace("${BUILD}", str(build))
    if "${" in expanded:
        raise ProvenanceError(f"unsupported manifest path token: {value}")
    return pathlib.Path(expanded).absolute()


def verify_manifest(path: pathlib.Path, repo: pathlib.Path,
                    build: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    _, payload_value = stable_file_binding(
        path, "build provenance manifest", payload_required=True
    )
    assert payload_value is not None
    payload = payload_value
    document = load_canonical_document(payload, "build provenance manifest")
    expected_sections = {
        "archives", "artifacts", "commands", "dependencies",
        "dependency_groups", "environment_dependencies",
        "provenance_digest", "recipes", "root", "schema", "tools",
    }
    if set(document) != expected_sections or document.get("schema") != SCHEMA:
        raise ProvenanceError("unsupported or inexact build provenance schema")
    if document.get("root") != "${REPO}":
        raise ProvenanceError("build provenance has an invalid repository root")
    claimed = require_digest(document.get("provenance_digest"), "provenance_digest")
    if self_digest(document, "provenance_digest") != claimed:
        raise ProvenanceError("invalid build provenance self-digest")

    def expected_binding(binding: Any, keys: set[str],
                         label: str) -> dict[str, Any]:
        if not isinstance(binding, dict) or set(binding) != keys:
            raise ProvenanceError(f"invalid {label} binding")
        require_digest(binding.get("sha256"), f"{label}.sha256")
        if (not isinstance(binding.get("size"), int) or
                isinstance(binding.get("size"), bool) or binding["size"] < 0):
            raise ProvenanceError(f"{label}.size must be a nonnegative integer")
        return {"sha256": binding["sha256"], "size": binding["size"]}

    def portable_file(value: Any, label: str) -> pathlib.Path:
        if not isinstance(value, str):
            raise ProvenanceError(f"{label} path must be a string")
        expanded = expand_path(value, repo, build)
        if canonical_path_argument(str(expanded), repo, build) != value:
            raise ProvenanceError(f"{label} path is not canonical: {value}")
        return expanded

    dependencies = document["dependencies"]
    if not isinstance(dependencies, dict) or not dependencies:
        raise ProvenanceError("build provenance dependency inventory is empty")
    missing_dependencies = REQUIRED_GRAPH_DEPENDENCIES.difference(dependencies)
    if missing_dependencies:
        raise ProvenanceError(
            "build provenance omits required graph dependencies: " +
            ", ".join(sorted(missing_dependencies))
        )
    for relative, binding in dependencies.items():
        pure = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
        if (pure is None or pure.is_absolute() or not pure.parts or
                ".." in pure.parts or pure.as_posix() != relative):
            raise ProvenanceError("build provenance has a noncanonical dependency")
        path_value = repo / pure
        expected = expected_binding(
            binding, {"sha256", "size"}, f"dependency {relative}"
        )
        if tracked_file(path_value, repo) != relative or \
                file_binding(path_value) != expected:
            raise ProvenanceError(f"build provenance dependency differs: {relative}")

    environment = document["environment_dependencies"]
    if not isinstance(environment, dict) or set(environment) != {
            "dependencies", "roots"}:
        raise ProvenanceError("invalid environment dependency inventory")
    root_bindings = environment["roots"]
    environment_dependencies = environment["dependencies"]
    if not isinstance(root_bindings, dict) or not root_bindings or \
            not isinstance(environment_dependencies, dict) or \
            not environment_dependencies:
        raise ProvenanceError("invalid environment dependency roots or paths")
    verified_roots: dict[str, pathlib.Path] = {}
    for label, root_value in root_bindings.items():
        if not isinstance(label, str) or not isinstance(root_value, str):
            raise ProvenanceError("invalid environment dependency root binding")
        root = canonical_system_root(pathlib.Path(root_value), label)
        if str(root) != root_value:
            raise ProvenanceError(
                f"environment dependency root is not canonical: {label}"
            )
        try:
            root.relative_to(repo)
        except ValueError:
            pass
        else:
            raise ProvenanceError(
                f"environment dependency root is inside repository: {label}"
            )
        verified_roots[label] = root
    environment_references: set[str] = set()
    for dependency_path, binding in environment_dependencies.items():
        expected = expected_binding(
            binding, {"root", "sha256", "size"},
            f"environment dependency {dependency_path}",
        )
        if not isinstance(dependency_path, str) or \
                not isinstance(binding["root"], str):
            raise ProvenanceError("invalid environment dependency path binding")
        canonical_path, root_label = classify_environment_dependency(
            pathlib.Path(dependency_path), verified_roots
        )
        if canonical_path != dependency_path or root_label != binding["root"]:
            raise ProvenanceError(
                f"environment dependency classification differs: {dependency_path}"
            )
        if file_binding(pathlib.Path(canonical_path)) != expected:
            raise ProvenanceError(
                f"environment dependency differs: {dependency_path}"
            )
        environment_references.add(f"@system:{dependency_path}")
    if {binding["root"] for binding in environment_dependencies.values()} != \
            set(root_bindings):
        raise ProvenanceError(
            "environment dependency root inventory contains unreferenced entries"
        )
    dependency_references = set(dependencies).union(environment_references)

    tools = document["tools"]
    if not isinstance(tools, dict) or not tools:
        raise ProvenanceError("build provenance tool inventory is empty")
    for tool_id, binding in tools.items():
        expected = expected_binding(
            binding, {"path", "sha256", "size"}, f"build tool {tool_id}"
        )
        require_digest(tool_id, "build tool identity")
        tool_path = portable_file(binding["path"], f"build tool {tool_id}")
        if file_binding(tool_path) != expected:
            raise ProvenanceError(f"build tool differs: {binding['path']}")
        if digest_bytes(canonical(binding)) != tool_id:
            raise ProvenanceError(f"invalid build tool identity: {binding['path']}")

    commands = document["commands"]
    if not isinstance(commands, list) or not commands:
        raise ProvenanceError("build provenance command inventory is empty")
    command_labels: set[str] = set()
    referenced_tools: set[str] = set()
    for command in commands:
        if not isinstance(command, dict) or set(command) != {
                "arguments", "cwd", "label", "tool"}:
            raise ProvenanceError("invalid build command record")
        label = command["label"]
        arguments = command["arguments"]
        if (not isinstance(label, str) or not label or label in command_labels or
                not isinstance(arguments, list) or
                not all(isinstance(argument, str) for argument in arguments)):
            raise ProvenanceError("invalid or duplicate build command label")
        portable_file(command["cwd"], f"command {label} cwd")
        tool_id = command["tool"]
        if not isinstance(tool_id, str) or tool_id not in tools:
            raise ProvenanceError(f"command references unknown tool: {label}")
        command_labels.add(label)
        referenced_tools.add(tool_id)
    if commands != sorted(commands, key=lambda command: command["label"]):
        raise ProvenanceError("build commands are not canonically ordered")
    if referenced_tools != set(tools):
        raise ProvenanceError("build tool inventory contains unreferenced entries")

    dependency_groups = document["dependency_groups"]
    if not isinstance(dependency_groups, dict) or not dependency_groups:
        raise ProvenanceError("dependency-group inventory is empty")
    referenced_dependencies: set[str] = set()
    for label, references in dependency_groups.items():
        if (not isinstance(label, str) or not label or
                not isinstance(references, list) or not references or
                not all(isinstance(reference, str) for reference in references) or
                references != sorted(set(references)) or
                not all(reference in dependency_references
                        for reference in references)):
            raise ProvenanceError(f"invalid dependency group: {label}")
        referenced_dependencies.update(references)

    recipes = document["recipes"]
    if (not isinstance(recipes, list) or not recipes or
            not all(isinstance(reference, str) for reference in recipes) or
            recipes != sorted(set(recipes)) or
            not all(reference in dependencies for reference in recipes)):
        raise ProvenanceError("invalid recipe dependency references")
    referenced_dependencies.update(recipes)

    archives = document["archives"]
    if not isinstance(archives, dict) or not archives:
        raise ProvenanceError("build provenance archive inventory is empty")
    referenced_commands: set[str] = set()
    for name, binding in archives.items():
        if not isinstance(binding, dict):
            raise ProvenanceError(f"invalid archive binding: {name}")
        if binding.get("kind") == "linked-prebuilt":
            expected = expected_binding(
                binding,
                {"kind", "path", "recipe_dependencies", "sha256", "size"},
                f"archive {name}",
            )
            recipe_references = binding["recipe_dependencies"]
            if recipe_references != recipes:
                raise ProvenanceError(
                    f"linked archive recipe closure differs: {name}"
                )
        else:
            expected = expected_binding(
                binding,
                {"compile_commands", "dependencies", "link_commands", "path",
                 "sha256", "size", "sources"},
                f"archive {name}",
            )
            for field in ("compile_commands", "link_commands"):
                references = binding[field]
                if (not isinstance(references, list) or not references or
                        not all(isinstance(reference, str)
                                for reference in references) or
                        references != sorted(set(references)) or
                        not all(reference in command_labels
                                for reference in references)):
                    raise ProvenanceError(
                        f"archive {name} has invalid {field} references"
                    )
                referenced_commands.update(references)
            for field in ("dependencies", "sources"):
                references = binding[field]
                if (not isinstance(references, list) or not references or
                        not all(isinstance(reference, str)
                                for reference in references) or
                        references != sorted(set(references)) or
                        not all(reference in dependency_references
                                for reference in references)):
                    raise ProvenanceError(
                        f"archive {name} has invalid {field} references"
                    )
                referenced_dependencies.update(references)
        archive_path = portable_file(binding["path"], f"archive {name}")
        if file_binding(archive_path) != expected:
            raise ProvenanceError(f"build provenance archive differs: {name}")

    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(BINARY_NAMES):
        raise ProvenanceError("build provenance artifact inventory differs")
    referenced_groups: set[str] = set()
    referenced_archives: set[str] = set()
    for name, binding in artifacts.items():
        expected = expected_binding(
            binding,
            {"archives", "commands", "dependency_groups", "path",
             "sha256", "size"},
            f"artifact {name}",
        )
        for field, inventory, target in (
            ("commands", command_labels, referenced_commands),
            ("dependency_groups", set(dependency_groups), referenced_groups),
            ("archives", set(archives), referenced_archives),
        ):
            references = binding[field]
            if (not isinstance(references, list) or
                    (field != "archives" and not references) or
                    not all(isinstance(reference, str)
                            for reference in references) or
                    references != sorted(set(references)) or
                    not all(reference in inventory for reference in references)):
                raise ProvenanceError(
                    f"artifact {name} has invalid {field} closure"
                )
            target.update(references)
        artifact_path = portable_file(binding["path"], f"artifact {name}")
        if file_binding(artifact_path) != expected:
            raise ProvenanceError(f"build provenance artifact differs: {name}")
    if referenced_commands != command_labels:
        raise ProvenanceError("build command inventory is not artifact-closed")
    if referenced_groups != set(dependency_groups):
        raise ProvenanceError("dependency groups are not artifact-closed")
    if referenced_archives != set(archives):
        raise ProvenanceError("archive inventory is not artifact-closed")
    if referenced_dependencies != dependency_references:
        raise ProvenanceError("dependency inventory contains unreferenced entries")
    return payload, document


def verify(args: argparse.Namespace) -> None:
    _, document = verify_manifest(
        args.manifest.absolute(), args.repo_root.resolve(strict=True),
        args.cmake_build.resolve(strict=True),
    )
    print(document["provenance_digest"])


def approve(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve(strict=True)
    ringlpn = args.ringlpn_root.resolve(strict=True)
    build = args.cmake_build.resolve(strict=True)
    manifest = args.manifest.absolute()
    payload, provenance = verify_manifest(manifest, repo, build)
    sources = set(MINIMUM_APPROVAL_SOURCES)
    sources.update(args.source)
    source_bindings: dict[str, dict[str, Any]] = {}
    for relative in sorted(sources):
        pure = pathlib.PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise ProvenanceError(f"invalid approval source path: {relative}")
        candidate = ringlpn / pure
        tracked_file(candidate, repo)
        source_bindings[relative] = file_binding(candidate)

    binaries: dict[str, dict[str, Any]] = {}
    for name in BINARY_NAMES:
        candidate = ringlpn / "bin" / name
        expected_path = canonical_path_argument(str(candidate), repo, build)
        artifact = provenance["artifacts"][name]
        if artifact["path"] != expected_path:
            raise ProvenanceError(
                f"provenance artifact is not canonical bin/{name}"
            )
        actual = file_binding(candidate)
        expected = {
            "sha256": artifact["sha256"],
            "size": artifact["size"],
        }
        if actual != expected:
            raise ProvenanceError(
                f"approved binary differs from provenance artifact: {name}"
            )
        binaries[name] = {"path": f"bin/{name}", **actual}

    receipt_paths: dict[str, pathlib.Path] = {}
    for assignment in args.validation_receipt:
        check, receipt_path = parse_assignment(
            assignment, "validation receipt"
        )
        if check in receipt_paths:
            raise ProvenanceError(f"duplicate validation receipt: {check}")
        receipt_paths[check] = receipt_path
    if set(receipt_paths) != VALIDATION_CHECKS:
        raise ProvenanceError(
            "approval requires contract-gate, deterministic-build, and "
            "internal-source-review receipts"
        )
    artifact_digests = {
        name: binding["sha256"]
        for name, binding in sorted(provenance["artifacts"].items())
    }
    validation: dict[str, dict[str, Any]] = {}
    for check, receipt_path in sorted(receipt_paths.items()):
        binding, receipt_payload_value = stable_file_binding(
            receipt_path, f"{check} validation receipt", payload_required=True
        )
        assert receipt_payload_value is not None
        receipt_document = load_canonical_document(
            receipt_payload_value, f"{check} validation receipt"
        )
        if (set(receipt_document) != {
                "artifacts", "check", "provenance_digest", "schema", "status"} or
                receipt_document.get("schema") != VALIDATION_RECEIPT_SCHEMA or
                receipt_document.get("check") != check or
                receipt_document.get("status") != "pass" or
                receipt_document.get("provenance_digest") !=
                provenance["provenance_digest"] or
                receipt_document.get("artifacts") != artifact_digests):
            raise ProvenanceError(
                f"{check} validation receipt does not bind this provenance"
            )
        if check == "internal_source_review":
            tracked_file(receipt_path, repo)
        try:
            receipt_relative = receipt_path.relative_to(ringlpn).as_posix()
        except ValueError as error:
            raise ProvenanceError(
                f"{check} validation receipt must be under Ring-LPN"
            ) from error
        pure_receipt = pathlib.PurePosixPath(receipt_relative)
        if (pure_receipt.is_absolute() or ".." in pure_receipt.parts or
                pure_receipt.as_posix() != receipt_relative):
            raise ProvenanceError(
                f"{check} validation receipt path is not canonical"
            )
        validation[check] = {
            "evidence": receipt_document,
            "receipt": {"path": receipt_relative, **binding},
            "status": "pass",
        }
    try:
        manifest_relative = manifest.relative_to(ringlpn).as_posix()
    except ValueError as error:
        raise ProvenanceError("approval provenance manifest must be under Ring-LPN") from error
    pure_manifest = pathlib.PurePosixPath(manifest_relative)
    if (pure_manifest.is_absolute() or ".." in pure_manifest.parts or
            pure_manifest.as_posix() != manifest_relative):
        raise ProvenanceError(
            "approval provenance manifest path is not canonical"
        )
    document: dict[str, Any] = {
        "binaries": binaries,
        "provenance": {
            "path": manifest_relative,
            "provenance_digest": provenance["provenance_digest"],
            "sha256": digest_bytes(payload),
            "size": len(payload),
        },
        "schema": APPROVAL_SCHEMA,
        "scope": APPROVAL_SCOPE,
        "sources": source_bindings,
        "validation": validation,
    }
    document["approval_digest"] = self_digest(document, "approval_digest")
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_noreplace(output, canonical(document) + b"\n", 0o400)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    subparsers = top.add_subparsers(dest="operation", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    generate_parser.add_argument("--ringlpn-root", type=pathlib.Path, required=True)
    generate_parser.add_argument("--cmake-build", type=pathlib.Path, required=True)
    generate_parser.add_argument("--output", type=pathlib.Path, required=True)
    generate_parser.add_argument("--depfile", action="append", default=[])
    generate_parser.add_argument("--command", action="append", default=[])
    generate_parser.add_argument("--archive", action="append", default=[])
    generate_parser.add_argument("--linked-archive", action="append", default=[])
    generate_parser.add_argument("--artifact", action="append", default=[])
    generate_parser.add_argument(
        "--artifact-command", action="append", default=[]
    )
    generate_parser.add_argument(
        "--artifact-dependency-group", action="append", default=[]
    )
    generate_parser.add_argument(
        "--artifact-archive", action="append", default=[]
    )
    generate_parser.add_argument("--recipe", action="append", default=[])
    generate_parser.add_argument("--system-root", action="append", default=[])
    generate_parser.set_defaults(handler=generate)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    verify_parser.add_argument("--cmake-build", type=pathlib.Path, required=True)
    verify_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    verify_parser.set_defaults(handler=verify)
    approval_parser = subparsers.add_parser("approve")
    approval_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    approval_parser.add_argument("--ringlpn-root", type=pathlib.Path, required=True)
    approval_parser.add_argument("--cmake-build", type=pathlib.Path, required=True)
    approval_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    approval_parser.add_argument("--output", type=pathlib.Path, required=True)
    approval_parser.add_argument("--source", action="append", default=[])
    approval_parser.add_argument(
        "--validation-receipt", action="append", default=[]
    )
    approval_parser.set_defaults(handler=approve)
    return top


def main() -> None:
    try:
        args = parser().parse_args()
        args.handler(args)
    except ProvenanceError as error:
        fail(str(error))


if __name__ == "__main__":
    main()
