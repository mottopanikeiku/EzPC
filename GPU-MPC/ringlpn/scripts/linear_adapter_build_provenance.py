#!/usr/bin/env python3
"""Generate and verify byte-bound FC/Conv adapter build provenance and approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import shutil
import sys
from typing import Any, NoReturn

PROVENANCE_SCHEMA = "ringlpn-linear-adapter-build-provenance-v1"
APPROVAL_SCHEMA = "ringlpn-linear-adapter-binary-approval-v2"
APPROVAL_SCOPE = "reproducible_linear_adapters_for_internal_advisor_execution"
CLASSIFICATION = "internal/advisor"
BINARY_NAMES = (
    "test_two_party_conv_preprocess",
    "test_two_party_fc_preprocess",
)
KINDS = ("fc", "conv")
CANONICAL_BUILD_SOURCE = pathlib.Path(
    "/tmp/ringlpn-linear-adapter-reproducible-build/source"
)
RECIPE_PATHS = (
    "GPU-MPC/ringlpn/scripts/build_common.sh",
    "GPU-MPC/ringlpn/scripts/build_component.sh",
    "GPU-MPC/ringlpn/scripts/build_reproducible_linear_adapters.sh",
    "GPU-MPC/ringlpn/scripts/build_two_party_fc_preprocess.sh",
    "GPU-MPC/ringlpn/scripts/build_two_party_conv_preprocess.sh",
    "GPU-MPC/ringlpn/scripts/linear_adapter_build_provenance.py",
)


REQUIRED_TOOLS = frozenset({
    "cuda_cicc", "cuda_fatbinary", "cuda_nvlink", "cuda_ptxas",
    "host_assembler", "host_cc1plus", "host_collect2", "host_cxx",
    "host_linker", "nvcc", "objcopy",
})
class ProvenanceError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise SystemExit(f"linear-adapter-build-provenance: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def self_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return digest_bytes(canonical(unsigned))


def require_digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ProvenanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def regular_nonsymlink(path: pathlib.Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ProvenanceError(f"cannot inspect {label} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProvenanceError(f"{label} must be a regular non-symlink: {path}")
    return metadata


def file_binding(path: pathlib.Path) -> dict[str, Any]:
    metadata = regular_nonsymlink(path, "bound file")
    return {"sha256": digest_file(path), "size": metadata.st_size}


def reject_symlink_components(path: pathlib.Path, root: pathlib.Path,
                              label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ProvenanceError(f"{label} is outside repository: {path}") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ProvenanceError(f"cannot inspect {label} {path}: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ProvenanceError(f"{label} contains a symlink component: {path}")


def tracked_relative(path: pathlib.Path, repo: pathlib.Path) -> str:
    resolved = path.resolve(strict=True)
    reject_symlink_components(resolved, repo, "source dependency")
    regular_nonsymlink(resolved, "source dependency")
    try:
        worktree = pathlib.Path(subprocess.check_output(
            ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()).resolve(strict=True)
        worktree.relative_to(repo)
        relative_worktree = resolved.relative_to(worktree).as_posix()
        subprocess.run(
            ["git", "-C", str(worktree), "ls-files", "--error-unmatch", "--",
             relative_worktree],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ProvenanceError(
            f"source dependency is not tracked by this repository or a nested submodule: {resolved}"
        ) from error
    return resolved.relative_to(repo).as_posix()


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


def read_commands(path: pathlib.Path) -> list[dict[str, Any]]:
    regular_nonsymlink(path, "command receipt")
    chunks = path.read_bytes().split(b"\0\0")
    commands: list[dict[str, Any]] = []
    for chunk in chunks:
        fields = chunk.rstrip(b"\0").split(b"\0")
        if len(fields) < 2 or any(not field for field in fields):
            raise ProvenanceError(f"invalid NUL-delimited command receipt: {path}")
        try:
            decoded = [field.decode("utf-8") for field in fields]
        except UnicodeError as error:
            raise ProvenanceError(f"non-UTF-8 command receipt: {path}") from error
        commands.append({"argv": decoded[1:], "cwd": decoded[0]})
    if len(commands) != 2:
        raise ProvenanceError(f"command receipt must contain compile and objcopy commands: {path}")
    return commands


def read_environment(path: pathlib.Path) -> dict[str, str]:
    regular_nonsymlink(path, "environment receipt")
    result: dict[str, str] = {}
    for field in path.read_bytes().rstrip(b"\0").split(b"\0"):
        try:
            text = field.decode("utf-8")
        except UnicodeError as error:
            raise ProvenanceError(f"non-UTF-8 build environment: {path}") from error
        name, separator, value = text.partition("=")
        if not separator or not name or name in result:
            raise ProvenanceError(f"invalid build environment entry: {path}")
        result[name] = value
    if not result:
        raise ProvenanceError(f"empty build environment receipt: {path}")
    return dict(sorted(result.items()))


def merge_binding(bindings: dict[str, dict[str, Any]], key: str,
                  binding: dict[str, Any], label: str) -> None:
    previous = bindings.setdefault(key, binding)
    if previous != binding:
        raise ProvenanceError(f"{label} changed while receipt was generated: {key}")


def resolve_dependency(path: pathlib.Path, repo: pathlib.Path) -> pathlib.Path:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(CANONICAL_BUILD_SOURCE)
    except ValueError:
        try:
            return absolute.resolve(strict=True)
        except OSError as error:
            raise ProvenanceError(
                f"cannot resolve build dependency {path}: {error}"
            ) from error
    normalized_text = os.path.normpath(relative.as_posix())
    normalized = pathlib.PurePosixPath(normalized_text)
    if (not normalized.parts or normalized.is_absolute() or
            normalized_text == ".." or normalized_text.startswith("../") or
            "." in normalized.parts or ".." in normalized.parts):
        raise ProvenanceError(
            f"canonical-build dependency escapes the fixed source prefix: {path}"
        )
    try:
        canonical_metadata = CANONICAL_BUILD_SOURCE.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ProvenanceError(
            f"cannot inspect canonical build source prefix: {error}"
        ) from error
    else:
        if (not stat.S_ISLNK(canonical_metadata.st_mode) or
                CANONICAL_BUILD_SOURCE.resolve(strict=True) != repo):
            raise ProvenanceError(
                "canonical build source prefix exists with an unexpected target"
            )
    candidate = repo.joinpath(*normalized.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo)
    except (OSError, ValueError) as error:
        raise ProvenanceError(
            f"canonical-build dependency does not map inside repository: {path}"
        ) from error
    return resolved

def classify_input(path: pathlib.Path, repo: pathlib.Path,
                   tracked: dict[str, dict[str, Any]],
                   external: dict[str, dict[str, Any]], kind: str) -> None:
    resolved = resolve_dependency(path, repo)
    regular_nonsymlink(resolved, "build dependency")
    try:
        resolved.relative_to(repo)
    except ValueError:
        binding = file_binding(resolved)
        binding["kinds"] = [kind]
        key = str(resolved)
        previous = external.get(key)
        if previous is None:
            external[key] = binding
        elif previous["sha256"] != binding["sha256"] or previous["size"] != binding["size"]:
            raise ProvenanceError(f"external dependency changed while hashing: {resolved}")
        elif kind not in previous["kinds"]:
            previous["kinds"].append(kind)
            previous["kinds"].sort()
    else:
        relative = tracked_relative(resolved, repo)
        merge_binding(tracked, relative, file_binding(resolved), "tracked dependency")


def map_inputs(path: pathlib.Path, cwd: pathlib.Path) -> list[pathlib.Path]:
    regular_nonsymlink(path, "link map")
    candidates: set[pathlib.Path] = set()
    for raw in re.findall(r"(?:^|[\s(])(/[^\s()]+|[^\s()]+\.(?:a|so(?:\.[0-9.]+)?|o))(?:$|[\s)])",
                          path.read_text(encoding="utf-8", errors="strict"), re.MULTILINE):
        token = raw.rstrip(":,")
        candidate = pathlib.Path(token)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.lstat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            candidates.add(resolved)
    return sorted(candidates)


def tool_version(path: pathlib.Path) -> str | None:
    attempts = ([str(path), "--version"], [str(path), "-V"])
    for command in attempts:
        try:
            completed = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, timeout=15)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        text = completed.stdout.strip()
        if text:
            return text
    return None


def host_subtool(host_cxx: pathlib.Path, name: str) -> pathlib.Path:
    try:
        output = subprocess.check_output(
            [str(host_cxx), f"-print-prog-name={name}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(
            f"cannot resolve host compiler subtool {name}"
        ) from error
    if not output:
        raise ProvenanceError(f"host compiler did not resolve subtool {name}")
    candidate = pathlib.Path(output)
    if candidate.parent == pathlib.Path("."):
        found = shutil.which(output)
        if found is None:
            raise ProvenanceError(f"host compiler subtool is unavailable: {name}")
        candidate = pathlib.Path(found)
    return candidate

def build_receipt(build_dir: pathlib.Path, build_id: str, repo: pathlib.Path,
                  recipe_bindings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tracked = dict(recipe_bindings)
    external: dict[str, dict[str, Any]] = {}
    tools: dict[str, dict[str, Any]] = {}
    adapters: dict[str, Any] = {}
    environments: dict[str, dict[str, str]] = {}
    for kind in KINDS:
        commands = read_commands(build_dir / f"{kind}.commands")
        cwd = pathlib.Path(commands[0]["cwd"])
        environment = read_environment(build_dir / f"{kind}.environment")
        environments[kind] = environment
        depfiles = sorted(build_dir.glob(f"{kind}.d.*"))
        if len(depfiles) != 6:
            raise ProvenanceError(
                f"expected six compiler depfiles for {kind}, found {len(depfiles)}"
            )
        for depfile in depfiles:
            for dependency in parse_depfile(depfile, cwd):
                classify_input(
                    dependency, repo, tracked, external, "compiler-header"
                )
        for dependency in map_inputs(build_dir / f"{kind}.map", cwd):
            classify_input(dependency, repo, tracked, external, "link-input")
        compile_argv = commands[0]["argv"]
        nvcc = pathlib.Path(compile_argv[0]).resolve(strict=True)
        objcopy = pathlib.Path(commands[1]["argv"][0]).resolve(strict=True)
        host_values = [argument.split("=", 1)[1] for argument in compile_argv
                       if argument.startswith("-ccbin=")]
        if len(host_values) != 1:
            raise ProvenanceError(f"compile command lacks one -ccbin binding: {kind}")
        host_cxx = pathlib.Path(host_values[0]).resolve(strict=True)
        cuda_root = nvcc.parent.parent
        tool_candidates = {
            "cuda_cicc": cuda_root / "nvvm/bin/cicc",
            "cuda_fatbinary": nvcc.parent / "fatbinary",
            "cuda_nvlink": nvcc.parent / "nvlink",
            "cuda_ptxas": nvcc.parent / "ptxas",
            "host_assembler": host_subtool(host_cxx, "as"),
            "host_cc1plus": host_subtool(host_cxx, "cc1plus"),
            "host_collect2": host_subtool(host_cxx, "collect2"),
            "host_cxx": host_cxx,
            "host_linker": host_subtool(host_cxx, "ld"),
            "nvcc": nvcc,
            "objcopy": objcopy,
        }
        classify_input(
            cuda_root / "nvvm/libdevice/libdevice.10.bc",
            repo, tracked, external, "compiler-runtime",
        )
        for label, tool in tool_candidates.items():
            resolved_tool = tool.resolve(strict=True)
            binding = file_binding(resolved_tool)
            binding.update({"path": str(resolved_tool), "version": tool_version(resolved_tool)})
            previous = tools.setdefault(label, binding)
            if previous != binding:
                raise ProvenanceError(f"tool binding differs between adapters: {label}")
            classify_input(resolved_tool, repo, tracked, external, "build-tool")
        binary_name = f"test_two_party_{kind}_preprocess"
        binary_path = build_dir / f"{kind}.elf"
        adapters[binary_name] = {
            "commands": commands,
            "output": file_binding(binary_path),
        }
    receipt: dict[str, Any] = {
        "build_id": build_id,
        "tracked_inputs": dict(sorted(tracked.items())),
        "external_inputs": dict(sorted(external.items())),
        "tools": dict(sorted(tools.items())),
        "environment": environments,
        "adapters": dict(sorted(adapters.items())),
    }
    receipt["receipt_digest"] = self_digest(receipt, "receipt_digest")
    return receipt


def load_json(path: pathlib.Path, label: str) -> tuple[bytes, dict[str, Any]]:
    regular_nonsymlink(path, label)
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid {label}: {error}") from error
    if not isinstance(document, dict):
        raise ProvenanceError(f"{label} must be a JSON object")
    return payload, document


def validate_receipt(receipt: Any, repo: pathlib.Path,
                     verify_environment: bool) -> None:
    if not isinstance(receipt, dict) or set(receipt) != {
            "adapters", "build_id", "environment", "external_inputs",
            "receipt_digest", "tools", "tracked_inputs"}:
        raise ProvenanceError("invalid build receipt shape")
    claimed = require_digest(receipt.get("receipt_digest"), "receipt digest")
    if self_digest(receipt, "receipt_digest") != claimed:
        raise ProvenanceError("invalid build receipt self-digest")
    if receipt.get("build_id") not in ("independent-build-1", "independent-build-2"):
        raise ProvenanceError("invalid independent build id")
    tracked = receipt.get("tracked_inputs")
    if not isinstance(tracked, dict) or not set(RECIPE_PATHS).issubset(tracked):
        raise ProvenanceError("build receipt lacks required tracked recipes")
    for relative, binding in tracked.items():
        if not isinstance(relative, str):
            raise ProvenanceError("tracked input path must be a string")
        pure = pathlib.PurePosixPath(relative)
        if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
            raise ProvenanceError(f"noncanonical tracked input path: {relative}")
        candidate = repo.joinpath(*pure.parts)
        if tracked_relative(candidate, repo) != relative or file_binding(candidate) != binding:
            raise ProvenanceError(f"tracked input binding differs: {relative}")
    external = receipt.get("external_inputs")
    if not isinstance(external, dict) or not external:
        raise ProvenanceError("build receipt lacks external system dependencies")
    for raw_path, binding in external.items():
        path = pathlib.Path(raw_path)
        if not path.is_absolute() or str(path.resolve(strict=True)) != raw_path:
            raise ProvenanceError(f"noncanonical external dependency path: {raw_path}")
        if (not isinstance(binding, dict) or set(binding) != {"kinds", "sha256", "size"} or
                not isinstance(binding["kinds"], list) or not binding["kinds"]):
            raise ProvenanceError(f"invalid external dependency binding: {raw_path}")
        if file_binding(path) != {"sha256": binding["sha256"], "size": binding["size"]}:
            raise ProvenanceError(f"external dependency binding differs: {raw_path}")
    tools = receipt.get("tools")
    if not isinstance(tools, dict) or not REQUIRED_TOOLS.issubset(tools):
        raise ProvenanceError("build receipt lacks compiler/tool bindings")
    for label, binding in tools.items():
        if (not isinstance(binding, dict) or
                set(binding) != {"path", "sha256", "size", "version"} or
                (binding["version"] is not None and
                 not isinstance(binding["version"], str))):
            raise ProvenanceError(f"invalid tool binding: {label}")
        path = pathlib.Path(binding["path"])
        if file_binding(path) != {"sha256": binding["sha256"], "size": binding["size"]}:
            raise ProvenanceError(f"build tool binding differs: {label}")
    adapters = receipt.get("adapters")
    if not isinstance(adapters, dict) or set(adapters) != set(BINARY_NAMES):
        raise ProvenanceError("build receipt lacks exact FC/Conv adapters")
    for name, adapter in adapters.items():
        if (not isinstance(adapter, dict) or set(adapter) != {"commands", "output"} or
                not isinstance(adapter["commands"], list) or len(adapter["commands"]) != 2):
            raise ProvenanceError(f"invalid adapter command receipt: {name}")
        output = adapter["output"]
        if not isinstance(output, dict) or set(output) != {"sha256", "size"}:
            raise ProvenanceError(f"invalid adapter output binding: {name}")
        require_digest(output.get("sha256"), f"{name} output")
        if not isinstance(output.get("size"), int) or output["size"] <= 0:
            raise ProvenanceError(f"invalid adapter output size: {name}")
    environments = receipt.get("environment")
    if not isinstance(environments, dict) or set(environments) != set(KINDS):
        raise ProvenanceError("build receipt lacks exact FC/Conv environments")
    for kind, environment in environments.items():
        if not isinstance(environment, dict) or not environment:
            raise ProvenanceError(f"invalid build environment: {kind}")
        required = {"CUDA_ARCH", "CXX", "HOME", "LANG", "LC_ALL", "NVCC",
                    "OBJCOPY", "PATH", "SOURCE_DATE_EPOCH", "TZ", "ZERO_AR_DATE"}
        if not required.issubset(environment):
            raise ProvenanceError(f"build environment omits deterministic inputs: {kind}")
        if verify_environment and environment.get("LC_ALL") != "C":
            raise ProvenanceError(f"uncontrolled locale in build environment: {kind}")


def verify_provenance_document(document: dict[str, Any], repo: pathlib.Path,
                               ringlpn: pathlib.Path,
                               verify_final_binaries: bool) -> None:
    if (document.get("schema") != PROVENANCE_SCHEMA or
            document.get("classification") != CLASSIFICATION or
            set(document) != {"builds", "byte_comparison", "classification",
                              "provenance_digest", "schema"}):
        raise ProvenanceError("unsupported linear adapter provenance schema/scope")
    claimed = require_digest(document.get("provenance_digest"), "provenance digest")
    if self_digest(document, "provenance_digest") != claimed:
        raise ProvenanceError("invalid provenance self-digest")
    builds = document.get("builds")
    if not isinstance(builds, list) or len(builds) != 2:
        raise ProvenanceError("provenance requires exactly two independent build receipts")
    for receipt in builds:
        validate_receipt(receipt, repo, True)
    if [receipt["build_id"] for receipt in builds] != [
            "independent-build-1", "independent-build-2"]:
        raise ProvenanceError("independent build receipts are missing or reordered")
    comparison = document.get("byte_comparison")
    if (not isinstance(comparison, dict) or
            comparison.get("status") != "identical" or
            set(comparison) != {"binaries", "status"} or
            not isinstance(comparison.get("binaries"), dict) or
            set(comparison["binaries"]) != set(BINARY_NAMES)):
        raise ProvenanceError("provenance lacks a successful FC/Conv byte comparison")
    for name in BINARY_NAMES:
        first = builds[0]["adapters"][name]["output"]
        second = builds[1]["adapters"][name]["output"]
        if first != second or comparison["binaries"].get(name) != first:
            raise ProvenanceError(f"independent build outputs differ: {name}")
        if verify_final_binaries and file_binding(ringlpn / "bin" / name) != first:
            raise ProvenanceError(f"installed adapter differs from reproducible builds: {name}")


def generate(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve(strict=True)
    ringlpn = args.ringlpn_root.resolve(strict=True)
    ringlpn.relative_to(repo)
    if len(args.build_dir) != 2:
        raise ProvenanceError("generate requires exactly two --build-dir receipts")
    recipe_bindings = {
        relative: file_binding(repo / pathlib.PurePosixPath(relative))
        for relative in RECIPE_PATHS
    }
    for relative in recipe_bindings:
        tracked_relative(repo / pathlib.PurePosixPath(relative), repo)
    builds = [
        build_receipt(path.resolve(strict=True), f"independent-build-{index}",
                      repo, recipe_bindings)
        for index, path in enumerate(args.build_dir, 1)
    ]
    comparison: dict[str, Any] = {"binaries": {}, "status": "identical"}
    for name in BINARY_NAMES:
        first = builds[0]["adapters"][name]["output"]
        second = builds[1]["adapters"][name]["output"]
        if first != second:
            raise ProvenanceError(f"independent adapter builds are not byte-identical: {name}")
        comparison["binaries"][name] = first
        if file_binding(ringlpn / "bin" / name) != first:
            raise ProvenanceError(f"installed binary is not the second reproduced ELF: {name}")
    document: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "classification": CLASSIFICATION,
        "builds": builds,
        "byte_comparison": comparison,
    }
    document["provenance_digest"] = self_digest(document, "provenance_digest")
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(document, sort_keys=True, indent=2,
                                  ensure_ascii=True).encode("ascii") + b"\n")


def approve(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve(strict=True)
    ringlpn = args.ringlpn_root.resolve(strict=True)
    provenance_path = args.provenance.absolute()
    payload, provenance = load_json(provenance_path, "build provenance")
    verify_provenance_document(provenance, repo, ringlpn, True)
    try:
        relative = provenance_path.relative_to(ringlpn).as_posix()
    except ValueError as error:
        raise ProvenanceError("build provenance must be retained under Ring-LPN") from error
    receipts = [receipt["receipt_digest"] for receipt in provenance["builds"]]
    binaries = provenance["byte_comparison"]["binaries"]
    document: dict[str, Any] = {
        "schema": APPROVAL_SCHEMA,
        "scope": APPROVAL_SCOPE,
        "classification": CLASSIFICATION,
        "binaries": binaries,
        "build_provenance": {
            "path": relative,
            "provenance_digest": provenance["provenance_digest"],
            "receipt_digests": receipts,
            "sha256": digest_bytes(payload),
            "size": len(payload),
        },
        "validation": {
            "deterministic_build": {
                "byte_comparison": "identical",
                "receipt_digests": receipts,
                "status": "pass",
            },
            "internal_source_review": {"receipt": None, "status": "unavailable"},
            "advisor_review": {"receipt": None, "status": "unavailable"},
        },
    }
    document["approval_digest"] = self_digest(document, "approval_digest")
    args.output.absolute().write_bytes(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii") + b"\n"
    )


def validate_approval_document(document: dict[str, Any], payload: bytes,
                               repo: pathlib.Path, ringlpn: pathlib.Path) -> None:
    del payload
    if (document.get("schema") != APPROVAL_SCHEMA or
            document.get("scope") != APPROVAL_SCOPE or
            document.get("classification") != CLASSIFICATION or
            set(document) != {"approval_digest", "binaries", "build_provenance",
                              "classification", "schema", "scope", "validation"}):
        raise ProvenanceError("unsupported linear adapter approval schema/scope")
    claimed = require_digest(document.get("approval_digest"), "approval digest")
    if self_digest(document, "approval_digest") != claimed:
        raise ProvenanceError("invalid approval self-digest")
    validation = document.get("validation")
    if not isinstance(validation, dict) or set(validation) != {
            "advisor_review", "deterministic_build", "internal_source_review"}:
        raise ProvenanceError("invalid approval validation claims")
    unavailable = {"receipt": None, "status": "unavailable"}
    if (validation["internal_source_review"] != unavailable or
            validation["advisor_review"] != unavailable):
        raise ProvenanceError("approval asserts unavailable human review")
    provenance_binding = document.get("build_provenance")
    if not isinstance(provenance_binding, dict) or set(provenance_binding) != {
            "path", "provenance_digest", "receipt_digests", "sha256", "size"}:
        raise ProvenanceError("approval lacks exact build provenance binding")
    pure = pathlib.PurePosixPath(provenance_binding["path"])
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        raise ProvenanceError("approval build provenance path is noncanonical")
    provenance_path = ringlpn.joinpath(*pure.parts)
    provenance_payload, provenance = load_json(provenance_path, "approved build provenance")
    if (file_binding(provenance_path) != {
            "sha256": provenance_binding["sha256"], "size": provenance_binding["size"]} or
            digest_bytes(provenance_payload) != provenance_binding["sha256"]):
        raise ProvenanceError("approved build provenance bytes differ")
    verify_provenance_document(provenance, repo, ringlpn, True)
    receipts = [receipt["receipt_digest"] for receipt in provenance["builds"]]
    deterministic = validation["deterministic_build"]
    if deterministic != {"byte_comparison": "identical",
                          "receipt_digests": receipts, "status": "pass"}:
        raise ProvenanceError("automated build status is not bound to both receipts")
    if (provenance_binding["provenance_digest"] != provenance["provenance_digest"] or
            provenance_binding["receipt_digests"] != receipts or
            document.get("binaries") != provenance["byte_comparison"]["binaries"]):
        raise ProvenanceError("approval differs from its build receipts")


def verify_approval(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve(strict=True)
    ringlpn = args.ringlpn_root.resolve(strict=True)
    payload, document = load_json(args.approval.absolute(), "linear adapter approval")
    validate_approval_document(document, payload, repo, ringlpn)
    print(document["approval_digest"])


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="operation", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    generate_parser.add_argument("--ringlpn-root", type=pathlib.Path, required=True)
    generate_parser.add_argument("--build-dir", action="append", type=pathlib.Path, default=[])
    generate_parser.add_argument("--output", type=pathlib.Path, required=True)
    generate_parser.set_defaults(handler=generate)
    approve_parser = commands.add_parser("approve")
    approve_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    approve_parser.add_argument("--ringlpn-root", type=pathlib.Path, required=True)
    approve_parser.add_argument("--provenance", type=pathlib.Path, required=True)
    approve_parser.add_argument("--output", type=pathlib.Path, required=True)
    approve_parser.set_defaults(handler=approve)
    verify_parser = commands.add_parser("verify-approval")
    verify_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    verify_parser.add_argument("--ringlpn-root", type=pathlib.Path, required=True)
    verify_parser.add_argument("--approval", type=pathlib.Path, required=True)
    verify_parser.set_defaults(handler=verify_approval)
    return top


def main() -> None:
    args = parser().parse_args()
    try:
        args.handler(args)
    except ProvenanceError as error:
        fail(str(error))


if __name__ == "__main__":
    main()
