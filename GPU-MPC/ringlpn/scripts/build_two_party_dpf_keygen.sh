#!/usr/bin/env bash
# Builds the two-PROCESS distributed DPF keygen (real IKNP OT over TCP) and its
# TEST-ONLY offline validator.
#
# The OT stack is this repository's SCI code, used unmodified and header-only:
#   SCI/src/OT/split-iknp.h, SCI/src/OT/np.h, SCI/src/utils/*
# Only OpenSSL is linked. SEAL, GMP and libOTe are not required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCI_SRC="$(cd "$ROOT/../../SCI/src" && pwd)"
mkdir -p "$ROOT/host_bin"

# Instruction-set flags are what the unmodified SCI headers need: AES-NI,
# SSE4.1, PCLMUL, AVX2 and RDSEED.
COMMON=(-std=c++17 -O2 -Wall -Wextra -maes -msse4.1 -mpclmul -mavx2 -mrdseed
        -I "$SCI_SRC" -I "$ROOT/src")

g++ "${COMMON[@]}" \
  "$ROOT/src/test_two_party_dpf_keygen.cpp" \
  "$ROOT/src/spfss_host.cpp" \
  -o "$ROOT/host_bin/test_two_party_dpf_keygen" \
  -lcrypto -lssl -pthread

g++ "${COMMON[@]}" \
  "$ROOT/src/test_two_party_dpf_validate.cpp" \
  "$ROOT/src/spfss_host.cpp" \
  -o "$ROOT/host_bin/test_two_party_dpf_validate" \
  -lcrypto -lssl -pthread

echo "Built $ROOT/host_bin/test_two_party_dpf_keygen"
echo "Built $ROOT/host_bin/test_two_party_dpf_validate"
