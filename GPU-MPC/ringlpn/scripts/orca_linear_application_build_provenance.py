#!/usr/bin/env python3
"""Generate or verify Orca linear provenance with the graph machinery."""

from __future__ import annotations

import sys

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


def main() -> None:
    provenance.SCHEMA = SCHEMA
    provenance.BINARY_NAMES = ARTIFACTS
    provenance.REQUIRED_GRAPH_DEPENDENCIES = REQUIRED_DEPENDENCIES
    values = list(sys.argv[1:])
    operation = "generate"
    if values and values[0] in {"generate", "verify"}:
        operation = values.pop(0)
    arguments = provenance.parser().parse_args([operation, *values])
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
