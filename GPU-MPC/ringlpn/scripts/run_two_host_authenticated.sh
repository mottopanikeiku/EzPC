#!/usr/bin/env bash
# Pinned-SSH, peer-private coordinator for one two-host Ring-LPN FC execution.
# Party 0 is local; party 1 reaches both local SCI listeners through two SSH
# remote forwards. Existing loopback launchers remain local-only evidence.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE=2026-08-04

usage() {
  cat >&2 <<'USAGE'
Usage: run_two_host_authenticated.sh \
  --peer USER@HOST --identity ABS --known-hosts ABS \
  --local-executor ABS --remote-executor ABS \
  --container-image NAME@sha256:HEX --container-binary ABS \
  --local-private-root ABS --remote-private-root ABS \
  --local-party-manifest ABS --remote-party-manifest ABS \
  --remote-peer-manifest ABS --local-export-root ABS --remote-export-root ABS \
  --checker-stage ABS --output-dir ABS \
  --local-container-uid N --remote-container-uid N \
  --local-gpu CDI --remote-gpu CDI \
  --session-id N [--invocation-id 32hex] [--ledger-root ABS] --base-port N \
  --qbits 64|128 --bw N --rows N --inner N --cols N \
  --ole-n N --ole-c N --ole-t N --noise regular|uniform [--timeout N] \
  [--fault-injection none|after-stage|commit-rename|post-commit-ack]

Run this coordinator on party 0's host. There is no SSH-config, ssh-agent,
password, host-key bypass, raw-WAN, local-process, or unisolated fallback.
Every path is absolute; all private/export/checker/output roots must be fresh.
USAGE
  exit 2
}

fail() { echo "[two-host-auth] $*" >&2; exit 2; }

peer= identity= known_hosts= local_executor= remote_executor=
container_image= container_binary=
local_private_root= remote_private_root=
local_party_manifest= remote_party_manifest= remote_peer_manifest=
local_export_root= remote_export_root= checker_stage= output_dir=
local_container_uid= remote_container_uid= local_gpu= remote_gpu=
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
    --local-private-root) local_private_root="$value" ;;
    --remote-private-root) remote_private_root="$value" ;;
    --local-party-manifest) local_party_manifest="$value" ;;
    --remote-party-manifest) remote_party_manifest="$value" ;;
    --remote-peer-manifest) remote_peer_manifest="$value" ;;
    --local-export-root) local_export_root="$value" ;;
    --remote-export-root) remote_export_root="$value" ;;
    --checker-stage) checker_stage="$value" ;;
    --output-dir) output_dir="$value" ;;
    --local-container-uid) local_container_uid="$value" ;;
    --remote-container-uid) remote_container_uid="$value" ;;
    --local-gpu) local_gpu="$value" ;;
    --remote-gpu) remote_gpu="$value" ;;
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
  container_binary local_private_root remote_private_root local_party_manifest
  remote_party_manifest remote_peer_manifest local_export_root remote_export_root
  checker_stage output_dir local_container_uid remote_container_uid local_gpu
  remote_gpu session_id base_port qbits bw rows inner cols ole_n ole_c ole_t noise)
for name in "${required[@]}"; do
  [[ -n "${!name}" ]] || fail "missing required --${name//_/-}"
done
[[ -n "$invocation_id" ]] || invocation_id="$(openssl rand -hex 16)"
ledger_root="${ledger_root:-$ROOT/results/deployment/correlation-ledger/coordinator-locks}"
[[ "$invocation_id" =~ ^[0-9a-f]{32}$ ]] ||
  fail "invocation ID must be 32 lowercase hexadecimal characters"
[[ "$ledger_root" == /* ]] || fail "ledger root must be absolute"

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
for name in local_container_uid remote_container_uid session_id base_port qbits bw \
            rows inner cols ole_n ole_c ole_t timeout_seconds; do
  is_uint "${!name}" || fail "--${name//_/-} must be an unsigned integer"
done
(( session_id > 0 )) || fail "session ID must be nonzero"
(( base_port >= 1 && base_port <= 65534 )) || fail "base port must leave base+1 valid"
[[ "$fault_injection" == none || "$fault_injection" == after-stage ||
   "$fault_injection" == commit-rename ||
   "$fault_injection" == post-commit-ack ]] ||
  fail "unsupported deterministic fault-injection point"
(( local_container_uid > 0 && remote_container_uid > 0 )) || fail "container UIDs must be non-root"
[[ "$qbits" == 64 || "$qbits" == 128 ]] || fail "qbits must be 64 or 128 (limb count, not security)"
(( bw > 2 && bw <= 32 && rows > 0 && inner > 0 && cols > 0 && ole_n > 0 && ole_c > 0 && ole_t > 0 && timeout_seconds > 0 )) || fail "invalid public dimensions or timeout"
[[ "$noise" == regular || "$noise" == uniform ]] || fail "noise must be regular or uniform"
[[ "$container_image" =~ ^[A-Za-z0-9._/:@+-]+@sha256:[0-9a-fA-F]{64}$ ]] ||
  fail "container image must be a shell-safe sha256 digest reference"
[[ "$local_gpu" =~ ^[A-Za-z0-9_.:-]+$ &&
   "$remote_gpu" =~ ^[A-Za-z0-9_.:-]+$ ]] ||
  fail "GPU CDI selectors contain unsupported characters"
[[ "$peer" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] ||
  fail "peer must be a hostname/IPv4 alias with optional user"
peer_host="${peer##*@}"

safe_abs() {
  [[ "$1" == /* && "$1" =~ ^/[A-Za-z0-9._/+:@-]+$ &&
     "$1" != *"/../"* && "$1" != */.. && "$1" != *"/./"* &&
     "$1" != *"//"* ]]
}
local_paths=(identity known_hosts local_executor container_binary local_private_root
  local_party_manifest local_export_root checker_stage output_dir ledger_root)
remote_paths=(remote_executor remote_private_root remote_party_manifest
  remote_peer_manifest remote_export_root)
for name in "${local_paths[@]}" "${remote_paths[@]}"; do
  safe_abs "${!name}" || fail "--${name//_/-} must be a normalized absolute path without shell metacharacters"
done

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
require_separate "$remote_private_root" "$remote_export_root" "remote private/export"
require_separate "$remote_private_root" "${remote_party_manifest%/*}" "remote private/manifest"
require_separate "$remote_private_root" "${remote_peer_manifest%/*}" "remote private/peer-manifest"
[[ "${local_party_manifest%/*}" == "$output_dir" ]] || fail "local party manifest must be directly under output-dir"
[[ ! -e "$output_dir" && ! -e "$local_private_root" && ! -e "$local_export_root" && ! -e "$checker_stage" ]] || fail "local roots/output must be fresh"
[[ -x "$local_executor" ]] || fail "local executor is not executable"
[[ -f "$identity" && -r "$identity" && -f "$known_hosts" && -r "$known_hosts" ]] || fail "identity/known-hosts files are unavailable"
identity_mode="$(stat -c %a "$identity")"
(( (8#$identity_mode & 077) == 0 )) || fail "SSH identity must not be group/other accessible"
ssh-keygen -F "$peer_host" -f "$known_hosts" >/dev/null || fail "known-hosts has no pinned entry for $peer_host"

# Numeric SID remains the compatibility handle and COMMITTED ABI. The
# high-entropy invocation ID is the actual global correlation namespace.
parameters_digest="$(printf '%s\0' "$session_id" "$invocation_id" "$base_port" \
  "$qbits" "$bw" "$rows" "$inner" "$cols" "$ole_n" "$ole_c" "$ole_t" \
  "$noise" external-loopback-tunnel | sha256sum | cut -d' ' -f1)"

# Both locks are consume-before-release and survive every success/failure.
lock_parent="$ROOT/results/deployment/session-locks"
mkdir -p "$lock_parent" "$ledger_root"
chmod 700 "$lock_parent" "$ledger_root"
session_lock="$lock_parent/$session_id"
mkdir "$session_lock" 2>/dev/null ||
  fail "session ID was already used on this coordinator"
invocation_lock="$ledger_root/$invocation_id"
mkdir "$invocation_lock" 2>/dev/null ||
  fail "invocation ID was already consumed on this coordinator"
claim_tmp="$invocation_lock/claim.tmp"
claim_file="$invocation_lock/claim"
printf 'version=1\ninvocation_id=%s\nsession_id=%s\npublic_parameters_sha256=%s\n' \
  "$invocation_id" "$session_id" "$parameters_digest" > "$claim_tmp"
chmod 600 "$claim_tmp"
python3 - "$claim_tmp" "$invocation_lock" "$ledger_root" <<'PY'
import os, sys
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
PY
mv "$claim_tmp" "$claim_file"
python3 - "$claim_file" "$invocation_lock" "$ledger_root" <<'PY'
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

control_dir="$(mktemp -d /tmp/ringlpn-ssh.XXXXXXXX)"
control_socket="$control_dir/control"
remote_manifest_copy="$output_dir/party1-sealed.json"
deployment_manifest="$output_dir/authenticated-boundary.manifest"
metrics="$output_dir/authenticated-boundary.csv"
master_pid= remote_pid= local_pid=
master_ready=0 remote_started=0 local_started=0 transaction_committed=0
p0_rc=NA p1_rc=NA p0_digest=NA p1_digest=NA stage_tmp=

ssh_common=(-F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$known_hosts"
  -o GlobalKnownHostsFile=/dev/null -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no -o HostbasedAuthentication=no
  -o GSSAPIAuthentication=no -o ExitOnForwardFailure=yes
  -o PermitLocalCommand=no -o RequestTTY=no -o ControlMaster=no
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -i "$identity")
control_opts=(-F /dev/null -S "$control_socket" -o ControlMaster=no)

write_evidence() {
  local status="$1" known_hash identity_hash
  known_hash="$(sha256sum "$known_hosts" | cut -d' ' -f1)"
  identity_hash="$(sha256sum "$identity" | cut -d' ' -f1)"
  cat > "$deployment_manifest.tmp" <<EOF
schema=ringlpn-authenticated-two-host-v1
date=$DATE
status=$status
classification=internal-advisor
security_claim=none
channel=authenticated-ssh
source_channel_label=external-loopback-tunnel
loopback_mode=local-only-not-authenticated-deployment
boundary=OpenSSH-authenticated-and-encrypted-remote-forwards-cover-both-SCI-NetIO-streams-end-to-end
trusted_endpoints=coordinator-and-peer-kernels-sshd-rootless-podman-and-pinned-container-image
peer=$peer
known_hosts_sha256=$known_hash
identity_file_sha256=$identity_hash
container_image=$container_image
container_binary=$container_binary
session_id=$session_id
invocation_id=$invocation_id
coordinator_ledger_digest=$coordinator_ledger_digest
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
p0_isolation_manifest=$local_party_manifest
p1_isolation_manifest=$remote_manifest_copy
checker_stage=$checker_stage
commit_manifest=$checker_stage/COMMITTED.manifest
commit_schema=ringlpn-two-host-commit-v1
p0_record_sha256=$p0_digest
p1_record_sha256=$p1_digest
p0_rc=$p0_rc
p1_rc=$p1_rc
EOF
  mv "$deployment_manifest.tmp" "$deployment_manifest"
  printf '%s\n' 'date,session_id,invocation_id,coordinator_ledger_digest,channel,security_boundary,straight_port,reversed_port,public_parameters_sha256,p0_record_sha256,p1_record_sha256,p0_rc,p1_rc,status' > "$metrics.tmp"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$DATE" "$session_id" "$invocation_id" "$coordinator_ledger_digest" authenticated-ssh ssh-pinned-two-stream-loopback-forward "$base_port" "$((base_port + 1))" "$parameters_digest" "$p0_digest" "$p1_digest" "$p0_rc" "$p1_rc" "$status" >> "$metrics.tmp"
  mv "$metrics.tmp" "$metrics"
}

remote_call() {
  ssh "${control_opts[@]}" "$peer" "$@"
}
direct_remote_call() {
  ssh "${ssh_common[@]}" "$peer" "$@"
}


abort_parties() {
  set +e
  if (( local_started )); then
    "$local_executor" abort-party --party 0 --session-id "$session_id" \
      --private-root "$local_private_root" --manifest "$local_party_manifest" >/dev/null 2>&1
  fi
  if (( remote_started )); then
    remote_aborted=0
    if (( master_ready )) &&
       remote_call "$remote_executor" abort-party --party 1 \
         --session-id "$session_id" --private-root "$remote_private_root" \
         --manifest "$remote_party_manifest" >/dev/null 2>&1; then
      remote_aborted=1
      scp -F /dev/null -o "ControlPath=$control_socket" -p \
        "$peer:$remote_party_manifest" "$output_dir/party1-aborted.json" \
        >/dev/null 2>&1
    elif direct_remote_call "$remote_executor" abort-party --party 1 \
           --session-id "$session_id" --private-root "$remote_private_root" \
           --manifest "$remote_party_manifest" >/dev/null 2>&1; then
      remote_aborted=1
      scp "${ssh_common[@]}" -p "$peer:$remote_party_manifest" \
        "$output_dir/party1-aborted.json" >/dev/null 2>&1
    fi
    (( remote_aborted )) ||
      echo "[two-host-auth] remote abort could not be acknowledged" >&2
  fi
  [[ -z "$local_pid" ]] || kill "$local_pid" >/dev/null 2>&1
  [[ -z "$remote_pid" ]] || kill "$remote_pid" >/dev/null 2>&1
  rm -rf -- "$local_export_root" "$checker_stage" "${checker_stage}.tmp."* 2>/dev/null
  if (( master_ready )); then
    remote_call rm -rf -- "$remote_export_root" >/dev/null 2>&1 ||
      direct_remote_call rm -rf -- "$remote_export_root" >/dev/null 2>&1
  else
    direct_remote_call rm -rf -- "$remote_export_root" >/dev/null 2>&1
  fi
  set -e
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if (( ! transaction_committed )); then
    abort_parties
    write_evidence FAIL || true
  fi
  if (( master_ready )); then
    ssh "${control_opts[@]}" "$peer" -O exit >/dev/null 2>&1 || true
  fi
  [[ -z "$master_pid" ]] || wait "$master_pid" >/dev/null 2>&1 || true
  rm -rf -- "$control_dir"
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

public_args=(--host 127.0.0.1 --port "$base_port" --sid "$session_id"
  --invocation-id "$invocation_id"
  --ledger /run/ringlpn/private/correlation-ledger
  --channel external-loopback-tunnel --qbits "$qbits" --bw "$bw"
  --rows "$rows" --inner "$inner" --cols "$cols" --ole-n "$ole_n"
  --ole-c "$ole_c" --ole-t "$ole_t" --noise "$noise")
party_prefix=/run/ringlpn/private/output/key
local_command=("$local_executor" run-party --party 0 --session-id "$session_id"
  --private-root "$local_private_root" --gpu "$local_gpu"
  --uid "$local_container_uid" --image "$container_image"
  --manifest "$local_party_manifest" -- /usr/bin/timeout --signal=TERM
  --kill-after=10 "$timeout_seconds" "$container_binary" --party 0
  "${public_args[@]}" --out-prefix "$party_prefix")
remote_command=("$remote_executor" run-party --party 1 --session-id "$session_id"
  --private-root "$remote_private_root" --gpu "$remote_gpu"
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
scp -F /dev/null -o "ControlPath=$control_socket" -p \
  "$peer:$remote_party_manifest" "$remote_manifest_copy"
scp -F /dev/null -o "ControlPath=$control_socket" -p \
  "$local_party_manifest" "$peer:$remote_peer_manifest"

"$local_executor" stage-party --party 0 --private-root "$local_private_root" \
  --manifest "$local_party_manifest" --peer-manifest "$remote_manifest_copy" \
  --export-root "$local_export_root"
remote_call "$remote_executor" stage-party --party 1 \
  --private-root "$remote_private_root" --manifest "$remote_party_manifest" \
  --peer-manifest "$remote_peer_manifest" --export-root "$remote_export_root"
scp -F /dev/null -o "ControlPath=$control_socket" -p \
  "$peer:$remote_party_manifest" "$remote_manifest_copy"
[[ "$fault_injection" != after-stage ]] ||
  fail "deterministic fault after bilateral staging"

stage_tmp="${checker_stage}.tmp.$$"
[[ ! -e "$stage_tmp" ]] || fail "checker staging temporary already exists"
mkdir -m 700 "$stage_tmp"
mkdir -m 700 "$stage_tmp/party0" "$stage_tmp/party1"
install -m 600 "$local_export_root/output/key_p0.fc" "$stage_tmp/party0/key_p0.fc"
scp -F /dev/null -o "ControlPath=$control_socket" -p \
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
committed_at="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"))')"
commit_tmp="$stage_tmp/.COMMITTED.manifest.tmp"
python3 -c '
import json, os, sys
path = sys.argv[1]
document = {
    "schema": "ringlpn-two-host-commit-v1",
    "state": "COMMITTED",
    "session_id": int(sys.argv[2]),
    "channel": "authenticated-ssh",
    "base_port": int(sys.argv[3]),
    "reversed_port": int(sys.argv[3]) + 1,
    "public_parameters_sha256": sys.argv[4],
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
    "committed_at": sys.argv[9],
}
stream = open(path, "x", encoding="utf-8")
os.chmod(path, 0o600)
json.dump(document, stream, indent=2, sort_keys=True)
stream.write("\n")
stream.flush()
os.fsync(stream.fileno())
stream.close()
' "$commit_tmp" "$session_id" "$base_port" "$parameters_digest" \
  "$p0_digest" "$p1_digest" "$p0_manifest_digest" "$p1_manifest_digest" \
  "$committed_at"
write_evidence PREPARED
[[ "$fault_injection" != commit-rename ]] ||
  fail "deterministic fault before COMMITTED manifest rename"
mv "$commit_tmp" "$stage_tmp/COMMITTED.manifest"
sync -f "$stage_tmp/COMMITTED.manifest"
sync -f "$stage_tmp"
mv "$stage_tmp" "$checker_stage"
stage_tmp=
sync -f "$checker_stage"
sync -f "${checker_stage%/*}"
[[ "$fault_injection" != post-commit-ack ]] ||
  fail "deterministic fault before COMMITTED manifest acknowledgement"
python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    document = json.load(stream)
if (document.get("schema") != "ringlpn-two-host-commit-v1" or
        document.get("state") != "COMMITTED" or
        document.get("session_id") != int(sys.argv[2]) or
        document.get("p0_record", {}).get("sha256") != sys.argv[3] or
        document.get("p1_record", {}).get("sha256") != sys.argv[4]):
    raise SystemExit("durable COMMITTED manifest acknowledgement failed")
' "$checker_stage/COMMITTED.manifest" "$session_id" "$p0_digest" "$p1_digest"
transaction_committed=1
write_evidence PASS
ssh "${control_opts[@]}" "$peer" -O exit >/dev/null 2>&1 || true
master_ready=0
wait "$master_pid" >/dev/null 2>&1 || true
master_pid=
trap - EXIT INT TERM HUP
rm -rf -- "$control_dir"
echo "[two-host-auth] authenticated two-host execution sealed and staged"
echo "[two-host-auth] manifest: $deployment_manifest"
echo "[two-host-auth] metrics: $metrics"
echo "[two-host-auth] checker stage: $checker_stage"
