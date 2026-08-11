#!/usr/bin/env bash
# Pinned-SSH, peer-private coordinator for one two-host Ring-LPN FC execution.
# Party 0 is local; party 1 reaches both local SCI listeners through two SSH
# remote forwards. Existing loopback launchers remain local-only evidence.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE=2026-08-10

usage() {
  cat >&2 <<'USAGE'
Usage: run_two_host_authenticated.sh \
  --peer USER@HOST --identity ABS --known-hosts ABS \
  --local-executor ABS --remote-executor ABS \
  --container-image NAME@sha256:HEX --container-binary ABS --container-binary-sha256 64hex \
  --local-private-root ABS --remote-private-root ABS \
  --local-party-ledger-root ABS --remote-party-ledger-root ABS \
  --local-party-manifest ABS --remote-party-manifest ABS \
  --remote-peer-manifest ABS --local-export-root ABS --remote-export-root ABS \
  --checker-stage ABS --output-dir ABS \
  --local-container-uid N --remote-container-uid N --checker-container-uid N \
  --local-gpu CDI --remote-gpu CDI --checker-gpu CDI \
  --session-id N [--invocation-id 32hex] --ledger-root ABS --base-port N \
  --qbits 64|128 --bw N --rows N --inner N --cols N \
  --ole-n N --ole-c N --ole-t N --noise regular|uniform [--timeout N] \
  [--fault-injection none|after-stage|prepare-rename|after-checker|cleanup-local-party|cleanup-remote-party|cleanup-checker|deletion-receipt|final-commit]

Run this coordinator on party 0's host. There is no SSH-config, ssh-agent,
password, host-key bypass, raw-WAN, local-process, or unisolated fallback.
Every path is absolute; all private/export/checker/output roots must be fresh.
USAGE
  exit 2
}

fail() { echo "[two-host-auth] $*" >&2; exit 2; }

peer= identity= known_hosts= local_executor= remote_executor=
container_image= container_binary= container_binary_sha256=
local_private_root= remote_private_root=
local_party_ledger_root= remote_party_ledger_root=
local_party_manifest= remote_party_manifest= remote_peer_manifest=
local_export_root= remote_export_root= checker_stage= output_dir=
local_container_uid= remote_container_uid= checker_container_uid=
local_gpu= remote_gpu= checker_gpu=
session_id= invocation_id= ledger_root= base_port= qbits= bw= rows= inner= cols=
ole_n= ole_c= ole_t= noise= timeout_seconds=1800 fault_injection=none

while (( $# )); do
  (( $# >= 2 )) || usage
  key="$1"; value="$2"; shift 2
  case "$key" in
    --peer) peer="$value" ;;
    --identity) identity="$value" ;;
    --known-hosts) known_hosts="$value" ;;
    --local-executor) local_executor="$value" ;;
    --remote-executor) remote_executor="$value" ;;
    --container-image) container_image="$value" ;;
    --container-binary) container_binary="$value" ;;
    --container-binary-sha256) container_binary_sha256="$value" ;;
    --local-private-root) local_private_root="$value" ;;
    --remote-private-root) remote_private_root="$value" ;;
    --local-party-ledger-root) local_party_ledger_root="$value" ;;
    --remote-party-ledger-root) remote_party_ledger_root="$value" ;;
    --local-party-manifest) local_party_manifest="$value" ;;
    --remote-party-manifest) remote_party_manifest="$value" ;;
    --remote-peer-manifest) remote_peer_manifest="$value" ;;
    --local-export-root) local_export_root="$value" ;;
    --remote-export-root) remote_export_root="$value" ;;
    --checker-stage) checker_stage="$value" ;;
    --output-dir) output_dir="$value" ;;
    --local-container-uid) local_container_uid="$value" ;;
    --remote-container-uid) remote_container_uid="$value" ;;
    --checker-container-uid) checker_container_uid="$value" ;;
    --local-gpu) local_gpu="$value" ;;
    --remote-gpu) remote_gpu="$value" ;;
    --checker-gpu) checker_gpu="$value" ;;
    --session-id) session_id="$value" ;;
    --invocation-id) invocation_id="$value" ;;
    --ledger-root) ledger_root="$value" ;;
    --base-port) base_port="$value" ;;
    --qbits) qbits="$value" ;;
    --bw) bw="$value" ;;
    --rows) rows="$value" ;;
    --inner) inner="$value" ;;
    --cols) cols="$value" ;;
    --ole-n) ole_n="$value" ;;
    --ole-c) ole_c="$value" ;;
    --ole-t) ole_t="$value" ;;
    --noise) noise="$value" ;;
    --timeout) timeout_seconds="$value" ;;
    --fault-injection) fault_injection="$value" ;;
    *) usage ;;
  esac
done

required=(peer identity known_hosts local_executor remote_executor container_image
  container_binary container_binary_sha256 local_private_root remote_private_root
  local_party_ledger_root remote_party_ledger_root local_party_manifest
  remote_party_manifest remote_peer_manifest local_export_root remote_export_root
  checker_stage output_dir local_container_uid remote_container_uid
  checker_container_uid local_gpu remote_gpu checker_gpu session_id ledger_root
  base_port qbits bw rows inner cols ole_n ole_c ole_t noise)
for name in "${required[@]}"; do
  [[ -n "${!name}" ]] || fail "missing required --${name//_/-}"
done
[[ -n "$invocation_id" ]] || invocation_id="$(openssl rand -hex 16)"
[[ "$invocation_id" =~ ^[0-9a-f]{32}$ ]] ||
  fail "invocation ID must be 32 lowercase hexadecimal characters"
[[ "$ledger_root" == /* ]] || fail "ledger root must be absolute"

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
for name in local_container_uid remote_container_uid checker_container_uid session_id \
            base_port qbits bw rows inner cols ole_n ole_c ole_t timeout_seconds; do
  is_uint "${!name}" || fail "--${name//_/-} must be an unsigned integer"
done
(( session_id > 0 )) || fail "session ID must be nonzero"
(( base_port >= 1 && base_port <= 65534 )) || fail "base port must leave base+1 valid"
[[ "$fault_injection" == none || "$fault_injection" == after-stage ||
   "$fault_injection" == prepare-rename || "$fault_injection" == after-checker ||
   "$fault_injection" == cleanup-local-party ||
   "$fault_injection" == cleanup-remote-party ||
   "$fault_injection" == cleanup-checker ||
   "$fault_injection" == deletion-receipt ||
   "$fault_injection" == final-commit ]] ||
  fail "unsupported deterministic fault-injection point"
(( local_container_uid > 0 && remote_container_uid > 0 && checker_container_uid > 0 )) ||
  fail "party/checker container UIDs must be non-root"
[[ "$qbits" == 64 || "$qbits" == 128 ]] || fail "qbits must be 64 or 128 (limb count, not security)"
(( bw > 2 && bw <= 32 && rows > 0 && inner > 0 && cols > 0 && ole_n > 0 && ole_c > 0 && ole_t > 0 && timeout_seconds > 0 )) || fail "invalid public dimensions or timeout"
[[ "$noise" == regular || "$noise" == uniform ]] || fail "noise must be regular or uniform"
[[ "$container_image" =~ ^[A-Za-z0-9._/:@+-]+@sha256:[0-9a-fA-F]{64}$ ]] ||
  fail "container image must be a shell-safe sha256 digest reference"
[[ "$container_binary_sha256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "container binary digest must be 64 lowercase hexadecimal characters"
[[ "$local_gpu" =~ ^[A-Za-z0-9_.:-]+$ &&
   "$remote_gpu" =~ ^[A-Za-z0-9_.:-]+$ &&
   "$checker_gpu" =~ ^[A-Za-z0-9_.:-]+$ ]] ||
  fail "GPU CDI selectors contain unsupported characters"
[[ "$checker_container_uid" != "$local_container_uid" &&
   "$checker_container_uid" != "$remote_container_uid" ]] ||
  fail "checker container UID must differ from both party UIDs"
[[ "$peer" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] ||
  fail "peer must be a hostname/IPv4 alias with optional user"
peer_host="${peer##*@}"

safe_abs() {
  [[ "$1" == /* && "$1" =~ ^/[A-Za-z0-9._/+:@-]+$ &&
     "$1" != *"/../"* && "$1" != */.. && "$1" != *"/./"* &&
     "$1" != *"//"* ]]
}
local_paths=(identity known_hosts local_executor container_binary local_private_root
  local_party_ledger_root local_party_manifest local_export_root checker_stage
  output_dir ledger_root)
remote_paths=(remote_executor remote_private_root remote_party_ledger_root
  remote_party_manifest remote_peer_manifest remote_export_root)
for name in "${local_paths[@]}" "${remote_paths[@]}"; do
  safe_abs "${!name}" || fail "--${name//_/-} must be a normalized absolute path without shell metacharacters"
done
[[ -d "$ledger_root" && ! -L "$ledger_root" &&
   -r "$ledger_root" && -w "$ledger_root" && -x "$ledger_root" ]] ||
  fail "ledger root must be a pre-existing writable non-symlink directory"
[[ "$(stat -c %u "$ledger_root")" == "${EUID:-$(id -u)}" ]] ||
  fail "ledger root must be owned by the coordinator user"
ledger_mode="$(stat -c %a "$ledger_root")"
(( (8#$ledger_mode & 077) == 0 )) ||
  fail "ledger root must not be group/other accessible"
[[ -d "$local_party_ledger_root" && ! -L "$local_party_ledger_root" &&
   -r "$local_party_ledger_root" && -w "$local_party_ledger_root" &&
   -x "$local_party_ledger_root" ]] ||
  fail "local party ledger root must be a pre-existing writable non-symlink directory"
[[ "$(stat -c %u "$local_party_ledger_root")" == "${EUID:-$(id -u)}" ]] ||
  fail "local party ledger root must be owned by the coordinator user"
local_party_ledger_mode="$(stat -c %a "$local_party_ledger_root")"
(( (8#$local_party_ledger_mode & 077) == 0 )) ||
  fail "local party ledger root must not be group/other accessible"
mountpoint -q "$local_party_ledger_root" ||
  fail "local party ledger root must be its own persistent mount"

contains_path() {
  local outer="${1%/}" inner="${2%/}"
  [[ "$inner" == "$outer" || "$inner" == "$outer/"* ]]
}
require_separate() {
  if contains_path "$1" "$2" || contains_path "$2" "$1"; then
    fail "$3 paths must be distinct and non-nested"
  fi
}
require_separate "$local_private_root" "$local_export_root" "local private/export"
require_separate "$local_private_root" "$checker_stage" "local private/checker"
require_separate "$local_private_root" "$output_dir" "local private/evidence"
require_separate "$local_export_root" "$checker_stage" "local export/checker"
require_separate "$ledger_root" "$local_private_root" "ledger/local private"
require_separate "$ledger_root" "$local_export_root" "ledger/local export"
require_separate "$ledger_root" "$checker_stage" "ledger/checker"
require_separate "$ledger_root" "$output_dir" "ledger/evidence"
require_separate "$local_party_ledger_root" "$ledger_root" "party/coordinator ledger"
require_separate "$local_party_ledger_root" "$local_private_root" "party ledger/private"
require_separate "$local_party_ledger_root" "$local_export_root" "party ledger/export"
require_separate "$local_party_ledger_root" "$checker_stage" "party ledger/checker"
require_separate "$local_party_ledger_root" "$output_dir" "party ledger/evidence"
require_separate "$remote_party_ledger_root" "$remote_private_root" "remote party ledger/private"
require_separate "$remote_party_ledger_root" "$remote_export_root" "remote party ledger/export"
require_separate "$remote_party_ledger_root" "${remote_party_manifest%/*}" "remote party ledger/manifest"
require_separate "$remote_party_ledger_root" "${remote_peer_manifest%/*}" "remote party ledger/peer-manifest"
require_separate "$remote_private_root" "$remote_export_root" "remote private/export"
require_separate "$remote_private_root" "${remote_party_manifest%/*}" "remote private/manifest"
require_separate "$remote_private_root" "${remote_peer_manifest%/*}" "remote private/peer-manifest"
require_separate "$remote_party_manifest" "$remote_peer_manifest" "remote manifest"
[[ "$local_party_manifest" == "$output_dir/party0-sealed.json" ]] ||
  fail "local party manifest must be OUTPUT_DIR/party0-sealed.json"
[[ "$checker_stage" == "$output_dir/checker-stage" ]] ||
  fail "checker stage must be OUTPUT_DIR/checker-stage"
[[ ! -e "$output_dir" && ! -e "$local_private_root" && ! -e "$local_export_root" && ! -e "$checker_stage" ]] || fail "local roots/output must be fresh"
[[ -x "$local_executor" ]] || fail "local executor is not executable"

control_dir="$(mktemp -d /tmp/ringlpn-ssh.XXXXXXXX)"
snapshot_bootstrap_cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  rm -rf -- "$control_dir"
  exit "$rc"
}
trap snapshot_bootstrap_cleanup EXIT INT TERM HUP
python3 - "$identity" "$known_hosts" "$control_dir" <<'PY'
import os, stat, sys

destination = sys.argv[3]
for source, name in ((sys.argv[1], "identity"), (sys.argv[2], "known-hosts")):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise SystemExit(f"cannot open {name} trust file without following links: {exc}")
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o077
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise SystemExit(
                f"{name} trust file must be owner-only, single-link, owner-owned, "
                "non-writable, and regular"
            )
        target = os.path.join(destination, name)
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        target_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        target_fd = os.open(target, target_flags, 0o400)
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:
                        raise OSError("short snapshot write")
                    view = view[written:]
            os.fchmod(target_fd, 0o400)
            os.fsync(target_fd)
            snapshot = os.fstat(target_fd)
            if (
                not stat.S_ISREG(snapshot.st_mode)
                or stat.S_IMODE(snapshot.st_mode) != 0o400
                or snapshot.st_uid != os.geteuid()
                or snapshot.st_nlink != 1
                or snapshot.st_size != before.st_size
            ):
                raise SystemExit(f"{name} descriptor snapshot validation failed")
        finally:
            os.close(target_fd)
        after = os.fstat(source_fd)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise SystemExit(f"{name} trust file changed during descriptor snapshot")
    finally:
        os.close(source_fd)
directory_fd = os.open(destination, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
identity="$control_dir/identity"
known_hosts="$control_dir/known-hosts"
ssh-keygen -F "$peer_host" -f "$known_hosts" >/dev/null ||
  fail "known-hosts snapshot has no pinned entry for $peer_host"

ssh_common=(-F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$known_hosts"
  -o GlobalKnownHostsFile=/dev/null -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no -o HostbasedAuthentication=no
  -o GSSAPIAuthentication=no -o ExitOnForwardFailure=yes
  -o PermitLocalCommand=no -o RequestTTY=no -o ControlMaster=no
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -i "$identity")
ssh "${ssh_common[@]}" "$peer" test -x "$remote_executor" ||
  fail "remote executor is not executable"
ssh "${ssh_common[@]}" "$peer" test -d "$remote_party_ledger_root" ||
  fail "remote party ledger root must be a pre-existing directory"
ssh "${ssh_common[@]}" "$peer" test ! -L "$remote_party_ledger_root" ||
  fail "remote party ledger root must not be a symlink"
ssh "${ssh_common[@]}" "$peer" test -r "$remote_party_ledger_root" ||
  fail "remote party ledger root must be readable"
ssh "${ssh_common[@]}" "$peer" test -w "$remote_party_ledger_root" ||
  fail "remote party ledger root must be writable"
ssh "${ssh_common[@]}" "$peer" test -x "$remote_party_ledger_root" ||
  fail "remote party ledger root must be searchable"
remote_party_ledger_uid="$(
  ssh "${ssh_common[@]}" "$peer" stat -c %u "$remote_party_ledger_root"
)"
remote_peer_uid="$(ssh "${ssh_common[@]}" "$peer" id -u)"
[[ "$remote_party_ledger_uid" == "$remote_peer_uid" ]] ||
  fail "remote party ledger root must be owned by the peer coordinator user"
remote_party_ledger_mode="$(
  ssh "${ssh_common[@]}" "$peer" stat -c %a "$remote_party_ledger_root"
)"
(( (8#$remote_party_ledger_mode & 077) == 0 )) ||
  fail "remote party ledger root must not be group/other accessible"
ssh "${ssh_common[@]}" "$peer" mountpoint -q "$remote_party_ledger_root" ||
  fail "remote party ledger root must be its own persistent mount"
local_executor_sha256="$(sha256sum "$local_executor" | cut -d' ' -f1)"
remote_executor_sha256="$(
  ssh "${ssh_common[@]}" "$peer" sha256sum "$remote_executor" | cut -d' ' -f1
)"
[[ "$local_executor_sha256" =~ ^[0-9a-f]{64}$ &&
   "$remote_executor_sha256" == "$local_executor_sha256" ]] ||
  fail "remote executor differs from the clean coordinator source"
local_machine_identity="$("$local_executor" machine-identity)"
remote_machine_identity="$(
  ssh "${ssh_common[@]}" "$peer" "$remote_executor" machine-identity
)"
[[ "$local_machine_identity" =~ ^[0-9a-f]{64}$ &&
   "$remote_machine_identity" =~ ^[0-9a-f]{64}$ ]] ||
  fail "stable machine identity output is malformed"
[[ "$local_machine_identity" != "$remote_machine_identity" ]] ||
  fail "authenticated two-host execution requires distinct stable machine identities"

validate_runtime_identity() {
  python3 - "$1" "$container_image" "$container_binary" "$container_binary_sha256" <<'PY'
import json, sys
document = json.loads(sys.argv[1])
expected = {
    "image": sys.argv[2],
    "binary_path": sys.argv[3],
    "binary_sha256": sys.argv[4],
}
if any(document.get(key) != value for key, value in expected.items()):
    raise SystemExit("measured runtime identity does not match the authorized launcher inputs")
image_id = document.get("image_id")
if not isinstance(image_id, str) or not image_id:
    raise SystemExit("measured runtime image ID is missing")
print(image_id)
PY
}
local_runtime_json="$(
  "$local_executor" runtime-identity --image "$container_image" --binary "$container_binary"
)"
remote_runtime_json="$(
  ssh "${ssh_common[@]}" "$peer" "$remote_executor" runtime-identity \
    --image "$container_image" --binary "$container_binary"
)"
local_image_id="$(validate_runtime_identity "$local_runtime_json")" ||
  fail "local runtime identity is unauthorized"
remote_image_id="$(validate_runtime_identity "$remote_runtime_json")" ||
  fail "remote runtime identity is unauthorized"
[[ "$local_image_id" == "$remote_image_id" ]] ||
  fail "local and remote immutable image IDs differ"
local_preflight_json="$(
  "$local_executor" capability-preflight --image "$container_image" \
    --binary "$container_binary" --binary-sha256 "$container_binary_sha256" \
    --gpu "$local_gpu"
)" || fail "local native-rootless-Podman capability preflight failed"
checker_preflight_json="$(
  "$local_executor" capability-preflight --image "$container_image" \
    --binary "$container_binary" --binary-sha256 "$container_binary_sha256" \
    --gpu "$checker_gpu"
)" || fail "checker native-rootless-Podman capability preflight failed"
remote_preflight_json="$(
  ssh "${ssh_common[@]}" "$peer" "$remote_executor" capability-preflight \
    --image "$container_image" --binary "$container_binary" \
    --binary-sha256 "$container_binary_sha256" --gpu "$remote_gpu"
)" || fail "remote native-rootless-Podman capability preflight failed"
gpu_identity_fields="$(
python3 - "$local_preflight_json" "$remote_preflight_json" \
  "$checker_preflight_json" "$local_machine_identity" \
  "$remote_machine_identity" "$local_gpu" "$remote_gpu" "$checker_gpu" <<'PY'
import json, sys
required_cpu = ["aes", "avx2", "pclmulqdq", "rdseed", "sse4_1"]
documents = {}
for label, raw, identity, gpu in (
    ("local", sys.argv[1], sys.argv[4], sys.argv[6]),
    ("remote", sys.argv[2], sys.argv[5], sys.argv[7]),
    ("checker", sys.argv[3], sys.argv[4], sys.argv[8]),
):
    document = json.loads(raw)
    uuid = document.get("gpu_uuid")
    pci = document.get("gpu_pci_bus_id")
    if (
        document.get("schema") != "ringlpn-native-podman-preflight-v1"
        or document.get("status") != "PASS"
        or document.get("rootless") is not True
        or document.get("machine_identity_sha256") != identity
        or document.get("compute_capability") != "8.9"
        or document.get("required_cpu_features") != required_cpu
        or document.get("gpu") != f"nvidia.com/gpu={gpu}"
        or not isinstance(uuid, str)
        or not uuid.startswith("GPU-")
        or not isinstance(pci, str)
        or ":" not in pci
        or "." not in pci
        or not isinstance(document.get("subuid_count"), int)
        or document["subuid_count"] <= 0
        or not isinstance(document.get("subgid_count"), int)
        or document["subgid_count"] <= 0
    ):
        raise SystemExit(f"{label} capability preflight evidence is malformed")
    documents[label] = (identity, uuid, pci)
uuid_identities = {(identity, uuid) for identity, uuid, _ in documents.values()}
pci_identities = {(identity, pci) for identity, _, pci in documents.values()}
if len(uuid_identities) != 3 or len(pci_identities) != 3:
    raise SystemExit(
        "party/checker physical GPUs alias despite distinct CDI selector strings"
    )
print(
    "\t".join(
        value
        for label in ("local", "remote", "checker")
        for value in documents[label][1:]
    )
)
PY
)" || fail "capability preflight identity validation failed"
IFS=$'\t' read -r local_gpu_uuid local_gpu_pci remote_gpu_uuid remote_gpu_pci \
  checker_gpu_uuid checker_gpu_pci <<< "$gpu_identity_fields"

# Numeric SID remains a compatibility handle. The high-entropy invocation ID
# is the actual global correlation namespace bound into preparation/final commit.
parameters_digest="$(printf '%s\0' "$session_id" "$invocation_id" "$base_port" \
  "$qbits" "$bw" "$rows" "$inner" "$cols" "$ole_n" "$ole_c" "$ole_t" \
  "$noise" external-loopback-tunnel | sha256sum | cut -d' ' -f1)"

# Both locks are consume-before-release under the same owner-only persistent
# ledger mount and survive every success/failure.
session_parent="$ledger_root/sessions"
invocation_parent="$ledger_root/invocations"
ensure_private_ledger_dir() {
  local path="$1" label="$2" mode
  if [[ ! -e "$path" ]]; then
    mkdir -m 700 "$path" || fail "cannot create $label ledger namespace"
  fi
  [[ -d "$path" && ! -L "$path" &&
     "$(stat -c %u "$path")" == "${EUID:-$(id -u)}" ]] ||
    fail "$label ledger namespace must be an owner directory"
  mode="$(stat -c %a "$path")"
  (( (8#$mode & 077) == 0 )) ||
    fail "$label ledger namespace must not be group/other accessible"
}
ensure_private_ledger_dir "$session_parent" "session"
ensure_private_ledger_dir "$invocation_parent" "invocation"
session_lock="$session_parent/$session_id"
mkdir -m 700 "$session_lock" 2>/dev/null ||
  fail "session ID was already used on this coordinator"
invocation_lock="$invocation_parent/$invocation_id"
mkdir -m 700 "$invocation_lock" 2>/dev/null ||
  fail "invocation ID was already consumed on this coordinator"
claim_tmp="$invocation_lock/claim.tmp"
claim_file="$invocation_lock/claim"
printf 'version=1\ninvocation_id=%s\nsession_id=%s\npublic_parameters_sha256=%s\n' \
  "$invocation_id" "$session_id" "$parameters_digest" > "$claim_tmp"
chmod 600 "$claim_tmp"
python3 - "$claim_tmp" "$invocation_lock" "$invocation_parent" \
  "$session_lock" "$session_parent" "$ledger_root" <<'PY'
import os, sys
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
PY
mv "$claim_tmp" "$claim_file"
python3 - "$claim_file" "$invocation_lock" "$invocation_parent" \
  "$session_lock" "$session_parent" "$ledger_root" <<'PY'
import os, sys
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
PY
coordinator_ledger_digest="$(sha256sum "$claim_file" | cut -d' ' -f1)"
mkdir -m 700 "$output_dir"
printf '%s\n' "$output_dir" > "$session_lock/evidence-path"
chmod 600 "$session_lock/evidence-path"
python3 - "$session_lock/evidence-path" "$session_lock" \
  "$session_parent" "$ledger_root" <<'PY'
import os, sys
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
PY

channel_auth_secret="$control_dir/channel-auth.secret"
openssl rand 32 > "$channel_auth_secret"
chmod 600 "$channel_auth_secret"
[[ ! -L "$channel_auth_secret" &&
   "$(stat -c %a "$channel_auth_secret")" == 600 &&
   "$(stat -c %s "$channel_auth_secret")" == 32 ]] ||
  fail "failed to create owner-only channel authenticator"
channel_auth_secret_sha256="$(sha256sum "$channel_auth_secret" | cut -d' ' -f1)"
control_socket="$control_dir/control"
remote_manifest_copy="$output_dir/party1-sealed.json"
deployment_manifest="$output_dir/authenticated-boundary.manifest"
metrics="$output_dir/authenticated-boundary.csv"
launcher_preparation="$output_dir/launcher-prepared.json"
launcher_result="$output_dir/launcher-result.json"
checker_manifest="$output_dir/checker-isolation.json"
checker_output_root="$output_dir/checker-output"
checker_log="$output_dir/checker.log"
deletion_receipt="$output_dir/deletion-receipt.json"
prepared_manifest="$checker_stage/PREPARED.manifest"
committed_manifest="$checker_stage/COMMITTED.manifest"
master_pid= remote_pid= local_pid=
master_ready=0 remote_started=0 local_started=0 transaction_committed=0
local_auth_provisioned=0 remote_auth_provisioned=0
local_auth_attempted=0 remote_auth_attempted=0
p0_rc=NA p1_rc=NA p0_digest=NA p1_digest=NA stage_tmp=
success_private_purged=no deletion_receipt_digest=NA
failure_cleanup_status=NOT_RUN
local_auth_marker="${local_private_root%/*}/.${local_private_root##*/}.channel-auth-cleanup"
remote_auth_marker="${remote_private_root%/*}/.${remote_private_root##*/}.channel-auth-cleanup"

control_opts=("${ssh_common[@]}" -S "$control_socket" -o ControlMaster=no)

write_evidence() {
  local status="$1" known_hash identity_public_hash
  known_hash="$(sha256sum "$known_hosts" | cut -d' ' -f1)"
  identity_public_hash="$(ssh-keygen -y -f "$identity" | sha256sum | cut -d' ' -f1)"
  cat > "$deployment_manifest.tmp" <<EOF
schema=ringlpn-authenticated-two-host-v1
date=$DATE
status=$status
classification=internal-advisor
security_claim=none
channel=authenticated-ssh-plus-mutual-hmac-sha256-v1
source_channel_label=external-loopback-tunnel
loopback_mode=process-authenticated-before-public-preflight
boundary=OpenSSH-carries-both-loopback-streams-and-per-run-mutual-HMAC-authenticates-each-party-process
trusted_endpoints=coordinator-and-peer-kernels-sshd-rootless-podman-and-pinned-container-image
peer=$peer
known_hosts_sha256=$known_hash
ssh_identity_public_sha256=$identity_public_hash
container_image=$container_image
local_container_image_id=$local_image_id
remote_container_image_id=$remote_image_id
container_binary=$container_binary
container_binary_sha256=$container_binary_sha256
local_executor_sha256=$local_executor_sha256
remote_executor_sha256=$remote_executor_sha256
local_machine_identity_sha256=$local_machine_identity
remote_machine_identity_sha256=$remote_machine_identity
local_party_gpu_uuid=$local_gpu_uuid
local_party_gpu_pci_bus_id=$local_gpu_pci
remote_party_gpu_uuid=$remote_gpu_uuid
remote_party_gpu_pci_bus_id=$remote_gpu_pci
checker_gpu_uuid=$checker_gpu_uuid
checker_gpu_pci_bus_id=$checker_gpu_pci
session_id=$session_id
invocation_id=$invocation_id
coordinator_ledger_digest=$coordinator_ledger_digest
local_party_ledger_root=$local_party_ledger_root
remote_party_ledger_root=$remote_party_ledger_root
straight_stream=peer-127.0.0.1:$base_port-to-coordinator-127.0.0.1:$base_port
reversed_stream=peer-127.0.0.1:$((base_port + 1))-to-coordinator-127.0.0.1:$((base_port + 1))
public_parameters_sha256=$parameters_digest
qbits=$qbits
bw=$bw
rows=$rows
inner=$inner
cols=$cols
ole_n=$ole_n
ole_c=$ole_c
ole_t=$ole_t
noise=$noise
fault_injection=$fault_injection
p0_isolation_manifest=$checker_stage/party0/isolation-manifest.json
p1_isolation_manifest=$checker_stage/party1/isolation-manifest.json
checker_manifest=$checker_manifest
checker_stage=$checker_stage
prepared_manifest=$prepared_manifest
prepared_schema=ringlpn-two-host-prepare-v1
commit_manifest=$committed_manifest
commit_schema=ringlpn-two-host-final-commit-v2
deletion_receipt=$deletion_receipt
deletion_receipt_sha256=$deletion_receipt_digest
p0_record_sha256=$p0_digest
p1_record_sha256=$p1_digest
p0_rc=$p0_rc
p1_rc=$p1_rc
success_private_and_export_roots_purged=$success_private_purged
failure_cleanup_status=$failure_cleanup_status
EOF
  mv "$deployment_manifest.tmp" "$deployment_manifest"
  printf '%s\n' 'date,session_id,invocation_id,coordinator_ledger_digest,channel,security_boundary,straight_port,reversed_port,public_parameters_sha256,p0_record_sha256,p1_record_sha256,p0_rc,p1_rc,status' > "$metrics.tmp"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$DATE" "$session_id" "$invocation_id" "$coordinator_ledger_digest" authenticated-ssh-plus-mutual-hmac-sha256-v1 ssh-pinned-two-stream-plus-mutual-hmac-sha256-v1 "$base_port" "$((base_port + 1))" "$parameters_digest" "$p0_digest" "$p1_digest" "$p0_rc" "$p1_rc" "$status" >> "$metrics.tmp"
  mv "$metrics.tmp" "$metrics"
}

remote_call() {
  ssh "${control_opts[@]}" "$peer" "$@"
}
direct_remote_call() {
  ssh "${ssh_common[@]}" "$peer" "$@"
}


abort_parties() {
  local cleanup_failed=0 remote_aborted=0 marker_or_root=0
  local post_ledger_uid= post_ledger_mode=
  set +e

  # Stop and reap launcher workers before asking Podman to remove their exact
  # labeled containers or any bind-mounted private root.
  for worker in local remote; do
    if [[ "$worker" == local ]]; then pid="$local_pid"; else pid="$remote_pid"; fi
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -TERM "$pid" >/dev/null 2>&1 || cleanup_failed=1
    fi
    wait "$pid" >/dev/null 2>&1 || true
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "[two-host-auth] $worker launcher worker survived kill/wait" >&2
      cleanup_failed=1
    fi
  done
  local_pid=
  remote_pid=

  if (( local_started )); then
    "$local_executor" abort-party --party 0 --session-id "$session_id" \
      --invocation-id "$invocation_id" \
      --private-root "$local_private_root" --ledger-root "$local_party_ledger_root" \
      --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
      --manifest "$local_party_manifest" >/dev/null 2>&1 || cleanup_failed=1
  elif (( local_auth_attempted )); then
    if [[ -e "$local_private_root" || -L "$local_private_root" ||
          -e "$local_auth_marker" || -L "$local_auth_marker" ]]; then
      "$local_executor" abort-provisioned-channel-auth \
        --private-root "$local_private_root" \
        --party 0 --session-id "$session_id" --invocation-id "$invocation_id" \
        --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
        >/dev/null 2>&1 || cleanup_failed=1
    fi
  fi

  marker_or_root=0
  if (( remote_auth_attempted )); then
    if (( master_ready )); then
      remote_call test ! -e "$remote_private_root" &&
        remote_call test ! -L "$remote_private_root" &&
        remote_call test ! -e "$remote_auth_marker" &&
        remote_call test ! -L "$remote_auth_marker" || marker_or_root=1
    else
      direct_remote_call test ! -e "$remote_private_root" &&
        direct_remote_call test ! -L "$remote_private_root" &&
        direct_remote_call test ! -e "$remote_auth_marker" &&
        direct_remote_call test ! -L "$remote_auth_marker" || marker_or_root=1
    fi
  fi
  if (( remote_started )); then
    if (( master_ready )) &&
       remote_call "$remote_executor" abort-party --party 1 \
         --invocation-id "$invocation_id" \
         --session-id "$session_id" --private-root "$remote_private_root" \
         --ledger-root "$remote_party_ledger_root" \
         --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
         --manifest "$remote_party_manifest" >/dev/null 2>&1; then
      remote_aborted=1
      scp "${ssh_common[@]}" -o "ControlPath=$control_socket" -p \
        "$peer:$remote_party_manifest" "$output_dir/party1-aborted.json" \
        >/dev/null 2>&1 || cleanup_failed=1
    elif direct_remote_call "$remote_executor" abort-party --party 1 \
           --invocation-id "$invocation_id" \
           --session-id "$session_id" --private-root "$remote_private_root" \
           --ledger-root "$remote_party_ledger_root" \
           --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
           --manifest "$remote_party_manifest" >/dev/null 2>&1; then
      remote_aborted=1
      scp "${ssh_common[@]}" -p "$peer:$remote_party_manifest" \
        "$output_dir/party1-aborted.json" >/dev/null 2>&1 || cleanup_failed=1
    fi
    if (( ! remote_aborted )); then
      echo "[two-host-auth] remote abort could not be acknowledged" >&2
      cleanup_failed=1
    fi
  elif (( marker_or_root )); then
    if (( master_ready )); then
      remote_call "$remote_executor" abort-provisioned-channel-auth \
        --private-root "$remote_private_root" \
        --party 1 --session-id "$session_id" --invocation-id "$invocation_id" \
        --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
        >/dev/null 2>&1 || cleanup_failed=1
    else
      direct_remote_call "$remote_executor" abort-provisioned-channel-auth \
        --private-root "$remote_private_root" \
        --party 1 --session-id "$session_id" --invocation-id "$invocation_id" \
        --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
        >/dev/null 2>&1 || cleanup_failed=1
    fi
  fi

  "$local_executor" abort-checker --session-id "$session_id" \
    --checker-manifest "$checker_manifest" \
    --checker-root "$checker_output_root" >/dev/null 2>&1 || cleanup_failed=1
  "$local_executor" verify-session-containers-absent \
    --session-id "$session_id" >/dev/null 2>&1 || cleanup_failed=1
  if (( master_ready )); then
    remote_call "$remote_executor" verify-session-containers-absent \
      --session-id "$session_id" >/dev/null 2>&1 || cleanup_failed=1
  else
    direct_remote_call "$remote_executor" verify-session-containers-absent \
      --session-id "$session_id" >/dev/null 2>&1 || cleanup_failed=1
  fi

  [[ ! -e "$local_private_root" && ! -L "$local_private_root" &&
     ! -e "$local_private_root/channel-auth.key" &&
     ! -L "$local_private_root/channel-auth.key" &&
     ! -e "$local_auth_marker" && ! -L "$local_auth_marker" ]] ||
    cleanup_failed=1
  if (( master_ready )); then
    remote_call test ! -e "$remote_private_root" >/dev/null 2>&1 &&
      remote_call test ! -L "$remote_private_root" >/dev/null 2>&1 &&
      remote_call test ! -e "$remote_private_root/channel-auth.key" >/dev/null 2>&1 &&
      remote_call test ! -L "$remote_private_root/channel-auth.key" >/dev/null 2>&1 &&
      remote_call test ! -e "$remote_auth_marker" >/dev/null 2>&1 &&
      remote_call test ! -L "$remote_auth_marker" >/dev/null 2>&1 ||
      cleanup_failed=1
  else
    direct_remote_call test ! -e "$remote_private_root" >/dev/null 2>&1 &&
      direct_remote_call test ! -L "$remote_private_root" >/dev/null 2>&1 &&
      direct_remote_call test ! -e "$remote_private_root/channel-auth.key" >/dev/null 2>&1 &&
      direct_remote_call test ! -L "$remote_private_root/channel-auth.key" >/dev/null 2>&1 &&
      direct_remote_call test ! -e "$remote_auth_marker" >/dev/null 2>&1 &&
      direct_remote_call test ! -L "$remote_auth_marker" >/dev/null 2>&1 ||
      cleanup_failed=1
  fi
  [[ -d "$local_party_ledger_root" && ! -L "$local_party_ledger_root" ]] ||
    cleanup_failed=1
  [[ "$(stat -c %u "$local_party_ledger_root" 2>/dev/null)" == "${EUID:-$(id -u)}" ]] ||
    cleanup_failed=1
  post_ledger_mode="$(stat -c %a "$local_party_ledger_root" 2>/dev/null)" ||
    cleanup_failed=1
  [[ "$post_ledger_mode" =~ ^[0-7]+$ ]] &&
    (( (8#$post_ledger_mode & 077) == 0 )) || cleanup_failed=1
  mountpoint -q "$local_party_ledger_root" || cleanup_failed=1
  if (( master_ready )); then
    remote_call test -d "$remote_party_ledger_root" >/dev/null 2>&1 &&
      remote_call test ! -L "$remote_party_ledger_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  else
    direct_remote_call test -d "$remote_party_ledger_root" >/dev/null 2>&1 &&
      direct_remote_call test ! -L "$remote_party_ledger_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  fi
  if (( master_ready )); then
    post_ledger_uid="$(remote_call stat -c %u "$remote_party_ledger_root" 2>/dev/null)" ||
      cleanup_failed=1
    post_ledger_mode="$(remote_call stat -c %a "$remote_party_ledger_root" 2>/dev/null)" ||
      cleanup_failed=1
    remote_call mountpoint -q "$remote_party_ledger_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  else
    post_ledger_uid="$(direct_remote_call stat -c %u "$remote_party_ledger_root" 2>/dev/null)" ||
      cleanup_failed=1
    post_ledger_mode="$(direct_remote_call stat -c %a "$remote_party_ledger_root" 2>/dev/null)" ||
      cleanup_failed=1
    direct_remote_call mountpoint -q "$remote_party_ledger_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  fi
  [[ "$post_ledger_uid" == "$remote_peer_uid" ]] || cleanup_failed=1
  [[ "$post_ledger_mode" =~ ^[0-7]+$ ]] &&
    (( (8#$post_ledger_mode & 077) == 0 )) || cleanup_failed=1

  rm -rf -- "$local_export_root" "$checker_stage" 2>/dev/null ||
    cleanup_failed=1
  if [[ -n "$stage_tmp" ]]; then
    rm -rf -- "$stage_tmp" 2>/dev/null || cleanup_failed=1
  fi
  if (( master_ready )); then
    remote_call rm -rf -- "$remote_export_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  else
    direct_remote_call rm -rf -- "$remote_export_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  fi
  if (( master_ready )); then
    remote_call test ! -e "$remote_export_root" >/dev/null 2>&1 &&
      remote_call test ! -L "$remote_export_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  else
    direct_remote_call test ! -e "$remote_export_root" >/dev/null 2>&1 &&
      direct_remote_call test ! -L "$remote_export_root" >/dev/null 2>&1 ||
      cleanup_failed=1
  fi
  [[ ! -e "$local_export_root" && ! -L "$local_export_root" &&
     ! -e "$checker_stage" && ! -L "$checker_stage" &&
     ! -e "$checker_output_root" && ! -L "$checker_output_root" ]] ||
    cleanup_failed=1

  if (( cleanup_failed )); then
    failure_cleanup_status=FAIL
    echo "[two-host-auth] explicit failure cleanup did not reach verified absence" >&2
    set -e
    return 1
  fi
  failure_cleanup_status=PASS
  set -e
  return 0
}

cleanup() {
  local rc=$? cleanup_failed=0
  trap - EXIT INT TERM HUP
  rm -f -- "$channel_auth_secret" || cleanup_failed=1
  if [[ -e "$channel_auth_secret" || -L "$channel_auth_secret" ]]; then
    echo "[two-host-auth] coordinator channel-auth staging secret survived failure cleanup" >&2
    cleanup_failed=1
  fi
  if (( ! transaction_committed )); then
    if ! abort_parties; then cleanup_failed=1; fi
    rm -f -- "$launcher_result" "$committed_manifest" || cleanup_failed=1
    (( cleanup_failed == 0 )) || failure_cleanup_status=FAIL
    write_evidence FAIL || cleanup_failed=1
  fi
  if (( master_ready )); then
    ssh "${control_opts[@]}" "$peer" -O exit >/dev/null 2>&1 || true
  fi
  [[ -z "$master_pid" ]] || wait "$master_pid" >/dev/null 2>&1 || true
  rm -rf -- "$control_dir" || cleanup_failed=1
  if (( cleanup_failed )); then
    echo "[two-host-auth] cleanup failed closed" >&2
    rc=2
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

write_evidence STARTING
ssh "${ssh_common[@]}" -M -S "$control_socket" -o ControlMaster=yes \
  -o ControlPersist=no -o GatewayPorts=no -N \
  -R "127.0.0.1:$base_port:127.0.0.1:$base_port" \
  -R "127.0.0.1:$((base_port + 1)):127.0.0.1:$((base_port + 1))" \
  "$peer" >"$output_dir/ssh-master.log" 2>&1 &
master_pid=$!
for _ in $(seq 1 100); do
  kill -0 "$master_pid" 2>/dev/null || fail "authenticated SSH master exited before readiness"
  if [[ -S "$control_socket" ]] && ssh "${control_opts[@]}" "$peer" true >/dev/null 2>&1; then
    master_ready=1
    break
  fi
  sleep 0.1
done
(( master_ready )) || fail "authenticated SSH tunnel did not become ready"

remote_call test -x "$remote_executor" || fail "remote executor is not executable"
for path in "$remote_private_root" "$remote_party_manifest" "$remote_peer_manifest" "$remote_export_root"; do
  remote_call test ! -e "$path" || fail "remote deployment path is not fresh: $path"
done
local_auth_attempted=1
"$local_executor" provision-channel-auth \
  --party 0 --session-id "$session_id" --invocation-id "$invocation_id" \
  --private-root "$local_private_root" \
  --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
  < "$channel_auth_secret"
local_auth_provisioned=1
remote_auth_attempted=1
remote_call "$remote_executor" provision-channel-auth \
  --party 1 --session-id "$session_id" --invocation-id "$invocation_id" \
  --private-root "$remote_private_root" \
  --channel-auth-secret-sha256 "$channel_auth_secret_sha256" \
  < "$channel_auth_secret"
remote_auth_provisioned=1
rm -f -- "$channel_auth_secret"
[[ ! -e "$channel_auth_secret" ]] ||
  fail "coordinator channel-auth staging secret survived provisioning"
sync -f "$control_dir"


public_args=(--host 127.0.0.1 --port "$base_port" --sid "$session_id"
  --invocation-id "$invocation_id"
  --ledger /run/ringlpn/ledger/correlation-ledger
  --channel-auth-file /run/ringlpn/private/channel-auth.key
  --channel external-loopback-tunnel --qbits "$qbits" --bw "$bw"
  --rows "$rows" --inner "$inner" --cols "$cols" --ole-n "$ole_n"
  --ole-c "$ole_c" --ole-t "$ole_t" --noise "$noise")
party_prefix=/run/ringlpn/private/output/key
local_command=("$local_executor" run-party --party 0 --session-id "$session_id"
  --invocation-id "$invocation_id" --public-parameters-sha256 "$parameters_digest"
  --machine-identity-sha256 "$local_machine_identity"
  --container-binary "$container_binary" --container-binary-sha256 "$container_binary_sha256"
  --channel-auth-secret-sha256 "$channel_auth_secret_sha256"
  --private-root "$local_private_root" --ledger-root "$local_party_ledger_root" \
  --gpu "$local_gpu"
  --uid "$local_container_uid" --image "$container_image"
  --manifest "$local_party_manifest" -- /usr/bin/timeout --signal=TERM
  --kill-after=10 "$timeout_seconds" "$container_binary" --party 0
  "${public_args[@]}" --out-prefix "$party_prefix")
remote_command=("$remote_executor" run-party --party 1 --session-id "$session_id"
  --invocation-id "$invocation_id" --public-parameters-sha256 "$parameters_digest"
  --machine-identity-sha256 "$remote_machine_identity"
  --container-binary "$container_binary" --container-binary-sha256 "$container_binary_sha256"
  --channel-auth-secret-sha256 "$channel_auth_secret_sha256"
  --private-root "$remote_private_root" --ledger-root "$remote_party_ledger_root" \
  --gpu "$remote_gpu"
  --uid "$remote_container_uid" --image "$container_image"
  --manifest "$remote_party_manifest" -- /usr/bin/timeout --signal=TERM
  --kill-after=10 "$timeout_seconds" "$container_binary" --party 1
  "${public_args[@]}" --out-prefix "$party_prefix")

local_started=1
"${local_command[@]}" >"$output_dir/local-executor.log" 2>&1 &
local_pid=$!
remote_started=1
remote_call "${remote_command[@]}" >"$output_dir/remote-executor.log" 2>&1 &
remote_pid=$!

set +e
wait -n -p first_finished "$master_pid" "$local_pid" "$remote_pid"
first_rc=$?
set -e
[[ "$first_finished" != "$master_pid" ]] || fail "authenticated tunnel died while parties were live"
if [[ "$first_finished" == "$local_pid" ]]; then p0_rc=$first_rc; other_pid=$remote_pid; else p1_rc=$first_rc; other_pid=$local_pid; fi
(( first_rc == 0 )) || fail "first party exit was nonzero"
set +e
wait -n -p second_finished "$master_pid" "$other_pid"
second_rc=$?
set -e
[[ "$second_finished" != "$master_pid" ]] || fail "authenticated tunnel died before bilateral exit"
if [[ "$second_finished" == "$local_pid" ]]; then p0_rc=$second_rc; else p1_rc=$second_rc; fi
(( second_rc == 0 )) || fail "second party exit was nonzero"

# No peer record is read or transferred before both party PIDs exit zero.
"$local_executor" seal-party --party 0 --private-root "$local_private_root" \
  --manifest "$local_party_manifest"
remote_call "$remote_executor" seal-party --party 1 \
  --private-root "$remote_private_root" --manifest "$remote_party_manifest"
scp "${ssh_common[@]}" -o "ControlPath=$control_socket" -p \
  "$peer:$remote_party_manifest" "$remote_manifest_copy"
scp "${ssh_common[@]}" -o "ControlPath=$control_socket" -p \
  "$local_party_manifest" "$peer:$remote_peer_manifest"

"$local_executor" stage-party --party 0 --private-root "$local_private_root" \
  --manifest "$local_party_manifest" --peer-manifest "$remote_manifest_copy" \
  --export-root "$local_export_root"
remote_call "$remote_executor" stage-party --party 1 \
  --private-root "$remote_private_root" --manifest "$remote_party_manifest" \
  --peer-manifest "$remote_peer_manifest" --export-root "$remote_export_root"
scp "${ssh_common[@]}" -o "ControlPath=$control_socket" -p \
  "$peer:$remote_party_manifest" "$remote_manifest_copy"
[[ "$fault_injection" != after-stage ]] ||
  fail "deterministic fault after bilateral staging"

stage_tmp="${checker_stage}.tmp.$$"
[[ ! -e "$stage_tmp" ]] || fail "checker staging temporary already exists"
mkdir -m 700 "$stage_tmp"
mkdir -m 700 "$stage_tmp/party0" "$stage_tmp/party1"
install -m 600 "$local_export_root/output/key_p0.fc" "$stage_tmp/party0/key_p0.fc"
scp "${ssh_common[@]}" -o "ControlPath=$control_socket" -p \
  "$peer:$remote_export_root/output/key_p1.fc" "$stage_tmp/party1/key_p1.fc"
chmod 600 "$stage_tmp/party1/key_p1.fc"
install -m 600 "$local_party_manifest" "$stage_tmp/party0/isolation-manifest.json"
install -m 600 "$remote_manifest_copy" "$stage_tmp/party1/isolation-manifest.json"
p0_digest="$(sha256sum "$stage_tmp/party0/key_p0.fc" | cut -d' ' -f1)"
p1_digest="$(sha256sum "$stage_tmp/party1/key_p1.fc" | cut -d' ' -f1)"
remote_p1_digest="$(remote_call sha256sum "$remote_export_root/output/key_p1.fc" | cut -d' ' -f1)"
[[ "$p1_digest" == "$remote_p1_digest" ]] || fail "authenticated party 1 transfer digest mismatch"
local_p0_digest="$(sha256sum "$local_export_root/output/key_p0.fc" | cut -d' ' -f1)"
[[ "$p0_digest" == "$local_p0_digest" ]] || fail "party 0 checker-stage digest mismatch"
p0_manifest_digest="$(sha256sum "$stage_tmp/party0/isolation-manifest.json" | cut -d' ' -f1)"
p1_manifest_digest="$(sha256sum "$stage_tmp/party1/isolation-manifest.json" | cut -d' ' -f1)"
prepared_at="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"))')"
prepare_tmp="$stage_tmp/.PREPARED.manifest.tmp"
python3 - "$prepare_tmp" "$session_id" "$base_port" "$parameters_digest" \
  "$p0_digest" "$p1_digest" "$p0_manifest_digest" "$p1_manifest_digest" \
  "$prepared_at" "$invocation_id" "$container_image" "$container_binary" \
  "$container_binary_sha256" "$local_machine_identity" "$remote_machine_identity" \
  "$channel_auth_secret_sha256" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
document = {
    "schema": "ringlpn-two-host-prepare-v1",
    "state": "PREPARED",
    "session_id": int(sys.argv[2]),
    "channel": "authenticated-ssh",
    "base_port": int(sys.argv[3]),
    "reversed_port": int(sys.argv[3]) + 1,
    "public_parameters_sha256": sys.argv[4],
    "channel_auth_secret_sha256": sys.argv[16],
    "invocation_id": sys.argv[10],
    "runtime_identity": {
        "container_image": sys.argv[11],
        "container_binary": sys.argv[12],
        "container_binary_sha256": sys.argv[13],
    },
    "machine_identities": {
        "party0_sha256": sys.argv[14],
        "party1_sha256": sys.argv[15],
    },
    "p0_exit_code": 0,
    "p1_exit_code": 0,
    "p0_record": {"path": "party0/key_p0.fc", "sha256": sys.argv[5]},
    "p1_record": {"path": "party1/key_p1.fc", "sha256": sys.argv[6]},
    "p0_isolation_manifest": {
        "path": "party0/isolation-manifest.json", "sha256": sys.argv[7]
    },
    "p1_isolation_manifest": {
        "path": "party1/isolation-manifest.json", "sha256": sys.argv[8]
    },
    "prepared_at": sys.argv[9],
}
with path.open("x", encoding="utf-8") as stream:
    os.chmod(path, 0o600)
    json.dump(document, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
write_evidence PREPARED
[[ "$fault_injection" != prepare-rename ]] ||
  fail "deterministic fault before PREPARED manifest rename"
mv "$prepare_tmp" "$stage_tmp/PREPARED.manifest"
for durable in "$stage_tmp/PREPARED.manifest" \
  "$stage_tmp/party0/key_p0.fc" "$stage_tmp/party1/key_p1.fc" \
  "$stage_tmp/party0/isolation-manifest.json" \
  "$stage_tmp/party1/isolation-manifest.json" "$stage_tmp"; do
  sync -f "$durable"
done
mv "$stage_tmp" "$checker_stage"
stage_tmp=
sync -f "$checker_stage"
sync -f "${checker_stage%/*}"

python3 - "$launcher_preparation" "$session_id" "$invocation_id" \
  "$parameters_digest" "$container_image" "$container_binary" \
  "$container_binary_sha256" "$local_machine_identity" \
  "$remote_machine_identity" "$checker_stage/party0/isolation-manifest.json" \
  "$checker_stage/party1/isolation-manifest.json" "$checker_stage" \
  "$prepared_manifest" "$local_party_manifest" "$remote_manifest_copy" \
  "$local_executor_sha256" <<'PY'
import hashlib, json, os, pathlib, sys
def bound(raw):
    path = pathlib.Path(raw)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path = pathlib.Path(sys.argv[1])
document = {
    "schema": "ringlpn-authenticated-launch-preparation-v1",
    "status": "PREPARED",
    "session_id": int(sys.argv[2]),
    "invocation_id": sys.argv[3],
    "public_parameters_sha256": sys.argv[4],
    "runtime_identity": {
        "container_image": sys.argv[5],
        "container_binary": sys.argv[6],
        "container_binary_sha256": sys.argv[7],
    },
    "machine_identities": {
        "party0_sha256": sys.argv[8],
        "party1_sha256": sys.argv[9],
    },
    "party_manifests": {"party0": bound(sys.argv[10]), "party1": bound(sys.argv[11])},
    "checker_stage": sys.argv[12],
    "prepared_manifest": bound(sys.argv[13]),
    "launcher_party_manifests": {
        "party0": bound(sys.argv[14]), "party1": bound(sys.argv[15])
    },
    "executor_sha256": sys.argv[16],
}
with path.open("x", encoding="utf-8") as stream:
    os.chmod(path, 0o600)
    json.dump(document, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
directory_fd = os.open(path.parent, os.O_RDONLY)
os.fsync(directory_fd)
os.close(directory_fd)
PY

publication_bindings=(--publication --expected-session-id "$session_id"
  --expected-invocation-id "$invocation_id"
  --expected-public-parameters-sha256 "$parameters_digest"
  --expected-container-image "$container_image"
  --expected-container-binary "$container_binary"
  --expected-container-binary-sha256 "$container_binary_sha256"
  --expected-executor-sha256 "$local_executor_sha256"
  --launcher-result "$launcher_preparation")
"$local_executor" run-checker \
  --p0-root "$checker_stage/party0" --p1-root "$checker_stage/party1" \
  --p0-manifest "$checker_stage/party0/isolation-manifest.json" \
  --p1-manifest "$checker_stage/party1/isolation-manifest.json" \
  --prepared-manifest "$prepared_manifest" \
  --checker-root "$checker_output_root" \
  --image "$container_image" --gpu "$checker_gpu" \
  --uid "$checker_container_uid" --manifest "$checker_manifest" \
  "${publication_bindings[@]}" -- "$container_binary" --check \
  --p0-record /run/ringlpn/checker/party0/key_p0.fc \
  --p1-record /run/ringlpn/checker/party1/key_p1.fc
"$local_executor" verify-manifest \
  --p0-manifest "$checker_stage/party0/isolation-manifest.json" \
  --p1-manifest "$checker_stage/party1/isolation-manifest.json" \
  --checker-manifest "$checker_manifest" \
  --prepared-manifest "$prepared_manifest" "${publication_bindings[@]}"
[[ "$fault_injection" != after-checker ]] ||
  fail "deterministic fault after bound checker success"

[[ "$fault_injection" != cleanup-local-party ]] ||
  fail "deterministic local party cleanup failure"
local_deletion_json="$(
  "$local_executor" purge-party --party 0 \
    --private-root "$local_private_root" --export-root "$local_export_root" \
    --manifest "$local_party_manifest"
)" || fail "local party success cleanup failed"
[[ ! -e "$local_private_root" && ! -e "$local_export_root" &&
   ! -e "$local_private_root/channel-auth.key" ]] ||
  fail "local success cleanup left private, export, or channel-auth records behind"

[[ "$fault_injection" != cleanup-remote-party ]] ||
  fail "deterministic remote party cleanup failure"
remote_deletion_json="$(
  remote_call "$remote_executor" purge-party --party 1 \
    --private-root "$remote_private_root" --export-root "$remote_export_root" \
    --manifest "$remote_party_manifest" --remove-manifest "$remote_party_manifest" \
    --peer-manifest "$remote_peer_manifest"
)" || fail "remote party success cleanup failed"
remote_call test ! -e "$remote_private_root"
remote_call test ! -e "$remote_export_root"
remote_call test ! -e "$remote_party_manifest"
remote_call test ! -e "$remote_peer_manifest"
remote_call test ! -e "$remote_private_root/channel-auth.key"
[[ -d "$local_party_ledger_root" && ! -L "$local_party_ledger_root" &&
   "$(stat -c %u "$local_party_ledger_root")" == "${EUID:-$(id -u)}" ]] ||
  fail "local persistent party ledger was not retained owner-only"
local_party_ledger_mode="$(stat -c %a "$local_party_ledger_root")"
(( (8#$local_party_ledger_mode & 077) == 0 )) ||
  fail "local persistent party ledger became group/other accessible"
remote_call test -d "$remote_party_ledger_root"
remote_call test ! -L "$remote_party_ledger_root"
[[ "$(remote_call stat -c %u "$remote_party_ledger_root")" == \
   "$(remote_call id -u)" ]] ||
  fail "remote persistent party ledger was not retained owner-only"
remote_party_ledger_mode="$(remote_call stat -c %a "$remote_party_ledger_root")"
(( (8#$remote_party_ledger_mode & 077) == 0 )) ||
  fail "remote persistent party ledger became group/other accessible"

[[ "$fault_injection" != cleanup-checker ]] ||
  fail "deterministic checker cleanup failure"
checker_deletion_json="$(
  "$local_executor" purge-checker-records \
    --p0-root "$checker_stage/party0" --p1-root "$checker_stage/party1" \
    --p0-manifest "$checker_stage/party0/isolation-manifest.json" \
    --p1-manifest "$checker_stage/party1/isolation-manifest.json" \
    --checker-manifest "$checker_manifest" \
    --prepared-manifest "$prepared_manifest" \
    --checker-root "$checker_output_root" --retained-log "$checker_log"
)" || fail "checker success cleanup failed"
[[ ! -e "$checker_stage/party0/key_p0.fc" &&
   ! -e "$checker_stage/party1/key_p1.fc" &&
   ! -e "$checker_output_root" ]] ||
  fail "checker cleanup left raw records or duplicate outputs behind"
"$local_executor" verify-session-containers-absent --session-id "$session_id"
remote_call "$remote_executor" verify-session-containers-absent \
  --session-id "$session_id"

[[ "$fault_injection" != deletion-receipt ]] ||
  fail "deterministic deletion receipt failure"
finalized_at="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"))')"
python3 - "$deletion_receipt.tmp" "$session_id" "$invocation_id" \
  "$parameters_digest" "$local_deletion_json" "$remote_deletion_json" \
  "$checker_deletion_json" "$prepared_manifest" "$checker_manifest" \
  "$launcher_preparation" "$coordinator_ledger_digest" "$ledger_root" \
  "$channel_auth_secret" "$finalized_at" <<'PY'
import hashlib, json, os, pathlib, sys
def bound(raw):
    path = pathlib.Path(raw)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
fragments = [json.loads(value) for value in sys.argv[5:8]]
if any(fragment.get("schema") != "ringlpn-deletion-fragment-v1" for fragment in fragments):
    raise SystemExit("cleanup helper emitted a malformed deletion fragment")
deletions = []
for host_role, fragment in zip(("party0-host", "party1-host", "checker-host"), fragments):
    for entry in fragment.get("deletions", []):
        if set(entry) != {"identity", "path", "status"} or entry["status"] != "absent":
            raise SystemExit("cleanup fragment does not prove exact absence")
        deletions.append({"host_role": host_role, **entry})
auth_deletions = [
    entry for entry in deletions
    if entry["identity"] == "channel-auth-secret"
]
if len(auth_deletions) != 2:
    raise SystemExit("cleanup receipt lacks bilateral channel-auth secret deletion")
if pathlib.Path(sys.argv[13]).exists():
    raise SystemExit("coordinator channel-auth staging secret survived cleanup")
deletions.append(
    {
        "host_role": "coordinator",
        "identity": "channel-auth-staging-secret",
        "path": sys.argv[13],
        "status": "absent",
    }
)
expected_deletions = {
    ("party0-host", "private-root"),
    ("party0-host", "export-root"),
    ("party0-host", "channel-auth-secret"),
    ("party0-host", "party0-container"),
    ("party1-host", "private-root"),
    ("party1-host", "export-root"),
    ("party1-host", "channel-auth-secret"),
    ("party1-host", "party1-container"),
    ("party1-host", "party-manifest"),
    ("party1-host", "peer-manifest-copy"),
    ("checker-host", "checker-stage-party0-record"),
    ("checker-host", "checker-stage-party1-record"),
    ("checker-host", "checker-output-root"),
    ("checker-host", "checker-container"),
    ("coordinator", "channel-auth-staging-secret"),
}
actual_deletions = {
    (entry["host_role"], entry["identity"]) for entry in deletions
}
if actual_deletions != expected_deletions:
    raise SystemExit("cleanup fragments do not cover the exact required deletion set")
persistent_ledgers = []
for host_role, fragment in zip(("party0-host", "party1-host"), fragments[:2]):
    entry = fragment.get("persistent_ledger")
    if (
        not isinstance(entry, dict)
        or set(entry) != {"identity", "path", "status"}
        or entry["status"] != "retained-owner-only-distinct-mount"
    ):
        raise SystemExit("party cleanup fragment lacks retained persistent ledger evidence")
    persistent_ledgers.append({"host_role": host_role, **entry})
persistent_ledgers.append(
    {
        "host_role": "coordinator",
        "identity": "coordinator-consume-once-ledger",
        "path": sys.argv[12],
        "status": "retained-owner-only-distinct-mount",
        "claim_sha256": sys.argv[11],
    }
)
document = {
    "schema": "ringlpn-two-host-deletion-receipt-v1",
    "status": "CLEANED",
    "session_id": int(sys.argv[2]),
    "invocation_id": sys.argv[3],
    "public_parameters_sha256": sys.argv[4],
    "machine_identities": {
        "party0_sha256": fragments[0]["machine_identity_sha256"],
        "party1_sha256": fragments[1]["machine_identity_sha256"],
        "checker_sha256": fragments[2]["machine_identity_sha256"],
    },
    "deletions": deletions,
    "retained_checker_log": fragments[2]["retained_log"],
    "prepared_manifest": bound(sys.argv[8]),
    "checker_manifest": bound(sys.argv[9]),
    "launcher_preparation": bound(sys.argv[10]),
    "persistent_ledgers": persistent_ledgers,
    "finalized_at": sys.argv[14],
}
path = pathlib.Path(sys.argv[1])
with path.open("x", encoding="utf-8") as stream:
    os.chmod(path, 0o600)
    json.dump(document, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
directory_fd = os.open(path.parent, os.O_RDONLY)
os.fsync(directory_fd)
os.close(directory_fd)
PY
mv "$deletion_receipt.tmp" "$deletion_receipt"
sync -f "$deletion_receipt"
sync -f "$output_dir"
deletion_receipt_digest="$(sha256sum "$deletion_receipt" | cut -d' ' -f1)"

[[ "$fault_injection" != final-commit ]] ||
  fail "deterministic final commit failure"
python3 - "$committed_manifest.tmp" "$session_id" "$invocation_id" \
  "$parameters_digest" "$prepared_manifest" "$checker_manifest" \
  "$deletion_receipt" "$finalized_at" <<'PY'
import hashlib, json, os, pathlib, sys
def bound(raw):
    path = pathlib.Path(raw)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path = pathlib.Path(sys.argv[1])
document = {
    "schema": "ringlpn-two-host-final-commit-v2",
    "state": "COMMITTED",
    "session_id": int(sys.argv[2]),
    "invocation_id": sys.argv[3],
    "public_parameters_sha256": sys.argv[4],
    "prepared_manifest": bound(sys.argv[5]),
    "checker_manifest": bound(sys.argv[6]),
    "deletion_receipt": bound(sys.argv[7]),
    "finalized_at": sys.argv[8],
}
with path.open("x", encoding="utf-8") as stream:
    os.chmod(path, 0o600)
    json.dump(document, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
mv "$committed_manifest.tmp" "$committed_manifest"
sync -f "$committed_manifest"
sync -f "$checker_stage"
sync -f "${checker_stage%/*}"
python3 - "$committed_manifest" "$deletion_receipt" "$session_id" \
  "$invocation_id" "$parameters_digest" "$prepared_manifest" \
  "$checker_manifest" <<'PY'
import hashlib, json, pathlib, sys
commit_path = pathlib.Path(sys.argv[1])
receipt_path = pathlib.Path(sys.argv[2])
commit_bytes = commit_path.read_bytes()
receipt_bytes = receipt_path.read_bytes()
prepared_path = pathlib.Path(sys.argv[6])
checker_path = pathlib.Path(sys.argv[7])
prepared_bytes = prepared_path.read_bytes()
checker_bytes = checker_path.read_bytes()
commit = json.loads(commit_bytes)
receipt = json.loads(receipt_bytes)
if set(commit) != {
    "schema", "state", "session_id", "invocation_id",
    "public_parameters_sha256", "prepared_manifest", "checker_manifest",
    "deletion_receipt", "finalized_at",
}:
    raise SystemExit("final COMMITTED manifest has unexpected fields")
if (
    commit.get("schema") != "ringlpn-two-host-final-commit-v2"
    or commit.get("state") != "COMMITTED"
    or commit.get("session_id") != int(sys.argv[3])
    or commit.get("invocation_id") != sys.argv[4]
    or commit.get("public_parameters_sha256") != sys.argv[5]
    or commit.get("prepared_manifest") != {
        "path": str(prepared_path),
        "sha256": hashlib.sha256(prepared_bytes).hexdigest(),
    }
    or commit.get("checker_manifest") != {
        "path": str(checker_path),
        "sha256": hashlib.sha256(checker_bytes).hexdigest(),
    }
    or commit.get("deletion_receipt") != {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
):
    raise SystemExit("final COMMITTED manifest acknowledgement failed")
if (
    receipt.get("schema") != "ringlpn-two-host-deletion-receipt-v1"
    or receipt.get("status") != "CLEANED"
    or receipt.get("session_id") != int(sys.argv[3])
    or receipt.get("invocation_id") != sys.argv[4]
    or any(entry.get("status") != "absent" for entry in receipt.get("deletions", []))
    or len(receipt.get("persistent_ledgers", [])) != 3
    or any(
        entry.get("status") != "retained-owner-only-distinct-mount"
        for entry in receipt.get("persistent_ledgers", [])
    )
):
    raise SystemExit("digest-bound deletion receipt acknowledgement failed")
PY
success_private_purged=yes
write_evidence PASS
sync -f "$deployment_manifest"
sync -f "$metrics"

completed_at="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"))')"
python3 - "$launcher_result.tmp" "$session_id" "$invocation_id" "$parameters_digest" \
  "$container_image" "$container_binary" "$container_binary_sha256" \
  "$local_machine_identity" "$remote_machine_identity" \
  "$committed_manifest" "$deletion_receipt" "$prepared_manifest" \
  "$checker_manifest" "$deployment_manifest" "$metrics" "$checker_log" \
  "$completed_at" "$local_executor_sha256" <<'PY'
import hashlib, json, os, pathlib, sys
def bound(raw):
    path = pathlib.Path(raw)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path = pathlib.Path(sys.argv[1])
document = {
    "schema": "ringlpn-authenticated-launch-result-v2",
    "status": "PASS",
    "session_id": int(sys.argv[2]),
    "invocation_id": sys.argv[3],
    "public_parameters_sha256": sys.argv[4],
    "runtime_identity": {
        "container_image": sys.argv[5],
        "container_binary": sys.argv[6],
        "container_binary_sha256": sys.argv[7],
    },
    "machine_identities": {
        "party0_sha256": sys.argv[8], "party1_sha256": sys.argv[9],
    },
    "success_cleanup": {
        "party_containers_absent": True,
        "checker_container_absent": True,
        "local_private_and_export_roots_absent": True,
        "remote_private_and_export_roots_absent": True,
        "checker_raw_records_and_duplicate_outputs_absent": True,
        "persistent_consume_once_ledgers_retained_owner_only": True,
    },
    "committed_manifest": bound(sys.argv[10]),
    "deletion_receipt": bound(sys.argv[11]),
    "prepared_manifest": bound(sys.argv[12]),
    "checker_manifest": bound(sys.argv[13]),
    "authenticated_boundary": {
        "manifest": bound(sys.argv[14]), "metrics": bound(sys.argv[15]),
    },
    "checker_log": bound(sys.argv[16]),
    "completed_at": sys.argv[17],
    "executor_sha256": sys.argv[18],
}
with path.open("x", encoding="utf-8") as stream:
    os.chmod(path, 0o600)
    json.dump(document, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
mv "$launcher_result.tmp" "$launcher_result"
sync -f "$launcher_result"
sync -f "$output_dir"
python3 - "$output_dir" "$local_party_manifest" <<'PY'
import pathlib, stat, sys
root = pathlib.Path(sys.argv[1])
local_manifest = pathlib.Path(sys.argv[2])
expected_files = {
    str(local_manifest.relative_to(root)),
    "party1-sealed.json",
    "authenticated-boundary.manifest",
    "authenticated-boundary.csv",
    "launcher-prepared.json",
    "launcher-result.json",
    "checker-isolation.json",
    "checker.log",
    "deletion-receipt.json",
    "ssh-master.log",
    "local-executor.log",
    "remote-executor.log",
    "checker-stage/PREPARED.manifest",
    "checker-stage/COMMITTED.manifest",
    "checker-stage/party0/isolation-manifest.json",
    "checker-stage/party1/isolation-manifest.json",
}
expected_directories = {
    "checker-stage",
    "checker-stage/party0",
    "checker-stage/party1",
}
actual_files = set()
actual_directories = set()
for path in root.rglob("*"):
    relative = str(path.relative_to(root))
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"retained evidence contains a symlink: {relative}")
    if stat.S_ISREG(info.st_mode):
        actual_files.add(relative)
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit(f"retained evidence file is not owner-only: {relative}")
    elif stat.S_ISDIR(info.st_mode):
        actual_directories.add(relative)
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise SystemExit(f"retained evidence directory is not owner-only: {relative}")
    else:
        raise SystemExit(f"retained evidence contains an unsupported object: {relative}")
if actual_files != expected_files or actual_directories != expected_directories:
    raise SystemExit(
        "retained evidence differs from the public allowlist; "
        f"extra_files={sorted(actual_files - expected_files)} "
        f"missing_files={sorted(expected_files - actual_files)} "
        f"extra_dirs={sorted(actual_directories - expected_directories)} "
        f"missing_dirs={sorted(expected_directories - actual_directories)}"
    )
PY
transaction_committed=1
ssh "${control_opts[@]}" "$peer" -O exit >/dev/null 2>&1 || true
master_ready=0
wait "$master_pid" >/dev/null 2>&1 || true
master_pid=
trap - EXIT INT TERM HUP
rm -rf -- "$control_dir"
echo "[two-host-auth] authenticated two-host execution finalized"
echo "[two-host-auth] manifest: $deployment_manifest"
echo "[two-host-auth] deletion receipt: $deletion_receipt"
echo "[two-host-auth] committed manifest: $committed_manifest"
echo "[two-host-auth] launcher result: $launcher_result"
