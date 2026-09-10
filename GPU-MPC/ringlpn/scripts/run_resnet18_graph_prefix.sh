#!/usr/bin/env bash
# Default fail-closed launcher for the exact known-zero ResNet18 graph prefix.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results/graph"
MANIFEST="$ROOT/results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json"
EXPECTED_MANIFEST_SHA="685501ddba417d597607403dbd9fe6c2954081d6edca161f6cef40c9e4be9427"
BASE_PORT="${BASE_PORT:-}"
P0_GPU="${P0_GPU:-1}"
P1_GPU="${P1_GPU:-3}"
CHECK_GPU="${CHECK_GPU:-2}"
WORKDIR="${WORKDIR:-$(mktemp -d /tmp/ringlpn-graph-prefix.XXXXXX)}"
LEDGER="$WORKDIR/ledger"
P0_PID=""
P1_PID=""
CHANNEL_AUTH_FILES=()

cleanup() {
  local rc=$?
  if [[ -n "$P0_PID" ]]; then kill "$P0_PID" 2>/dev/null || true; fi
  if [[ -n "$P1_PID" ]]; then kill "$P1_PID" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  local auth_file
  for auth_file in "${CHANNEL_AUTH_FILES[@]}"; do
    rm -f -- "$auth_file"
  done
  if [[ "${KEEP_WORKDIR:-0}" != "1" ]]; then
    rm -rf -- "$WORKDIR"
  else
    echo "[graph-prefix] retained caller-requested private workdir: $WORKDIR" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$P0_GPU" == "$P1_GPU" ]]; then
  echo "[graph-prefix] party GPUs must be distinct" >&2
  exit 2
fi
mkdir -p "$RESULTS" "$WORKDIR/p0" "$WORKDIR/p1" "$LEDGER"
chmod 700 "$WORKDIR" "$WORKDIR/p0" "$WORKDIR/p1" "$LEDGER"
if [[ -z "$BASE_PORT" ]]; then
  BASE_PORT="$(python3 - <<'PY'
import random, socket
for _ in range(1000):
    base = random.randrange(50000, 64000, 8)
    sockets = []
    try:
        for port in range(base, base + 17):
            sock = socket.socket()
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
        print(base)
        break
    except OSError:
        pass
    finally:
        for sock in sockets:
            sock.close()
else:
    raise SystemExit("no free graph-prefix port range")
PY
)"
fi

actual_manifest_sha="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
if [[ "$actual_manifest_sha" != "$EXPECTED_MANIFEST_SHA" ]]; then
  echo "[graph-prefix] source-bound manifest digest mismatch" >&2
  exit 2
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  PATH="/usr/local/cuda/bin:$PATH" "$ROOT/scripts/build_two_party_conv_preprocess.sh"
  PATH="/usr/local/cuda/bin:$PATH" "$ROOT/scripts/build_resnet18_graph_prefix.sh"
fi

new_invocation() {
  python3 -c 'import secrets; print(secrets.token_hex(16))'
}
PAIR_P0_RC=1
PAIR_P1_RC=1
wait_pair_fail_fast() {
  local finished=""
  local first_rc=1
  set +e
  wait -n -p finished "$P0_PID" "$P1_PID"
  first_rc=$?
  if [[ $first_rc -ne 0 ]]; then
    if [[ "$finished" == "$P0_PID" ]]; then
      kill "$P1_PID" 2>/dev/null || true
    else
      kill "$P0_PID" 2>/dev/null || true
    fi
  fi
  if [[ "$finished" == "$P0_PID" ]]; then
    PAIR_P0_RC=$first_rc
    wait "$P1_PID"; PAIR_P1_RC=$?
  else
    PAIR_P1_RC=$first_rc
    wait "$P0_PID"; PAIR_P0_RC=$?
  fi
  set -e
  P0_PID=""
  P1_PID=""
}

run_preprocess_pair() {
  local name=$1
  local port=$2
  local invocation=$3
  local ordinal=$4
  shift 4
  local sid
  sid="$(python3 -c 'import secrets; print(secrets.randbelow((1 << 63) - 1) + 1)')"
  local common=(
    --host 127.0.0.1 --port "$port" --sid "$sid"
    --invocation-id "$invocation"
    --ledger "$LEDGER" --out-prefix "$WORKDIR/$name/key"
    --layer-ordinal "$ordinal" --qbits 128 --bw 32
    --ole-n 262144 --ole-c 2 --ole-t 8 --noise regular "$@"
  )
  mkdir -p "$WORKDIR/$name"
  chmod 700 "$WORKDIR/$name"
  mkdir -m 700 "$WORKDIR/$name/party0-private" "$WORKDIR/$name/party1-private"
  local p0_auth="$WORKDIR/$name/party0-private/channel-auth.key"
  local p1_auth="$WORKDIR/$name/party1-private/channel-auth.key"
  CHANNEL_AUTH_FILES+=("$p0_auth" "$p1_auth")
  openssl rand 32 > "$p0_auth"
  cp -- "$p0_auth" "$p1_auth"
  chmod 600 "$p0_auth" "$p1_auth"
  CUDA_VISIBLE_DEVICES="$P0_GPU" "$ROOT/bin/test_two_party_conv_preprocess" \
    --party 0 "${common[@]}" \
    --channel-auth-file "$p0_auth" \
    --state-record "$WORKDIR/p0/$name.state" --csv-header \
    >"$WORKDIR/$name.p0.csv" 2>"$WORKDIR/$name.p0.log" &
  P0_PID=$!
  CUDA_VISIBLE_DEVICES="$P1_GPU" "$ROOT/bin/test_two_party_conv_preprocess" \
    --party 1 "${common[@]}" \
    --channel-auth-file "$p1_auth" \
    --state-record "$WORKDIR/p1/$name.state" --csv-header \
    >"$WORKDIR/$name.p1.csv" 2>"$WORKDIR/$name.p1.log" &
  P1_PID=$!
  wait_pair_fail_fast
  local p0_rc=$PAIR_P0_RC
  local p1_rc=$PAIR_P1_RC
  if [[ $p0_rc -ne 0 || $p1_rc -ne 0 ]]; then
    cat "$WORKDIR/$name.p0.log" "$WORKDIR/$name.p1.log" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" \
    "$ROOT/bin/test_two_party_conv_preprocess" --check \
    --p0-record "$WORKDIR/$name/key_p0.conv" \
    --p1-record "$WORKDIR/$name/key_p1.conv" \
    --p0-state "$WORKDIR/p0/$name.state" \
    --p1-state "$WORKDIR/p1/$name.state" --csv-header \
    >"$WORKDIR/$name.check.csv"
}

conv0_invocation="$(new_invocation)"
conv3_invocation="$(new_invocation)"
graph_invocation="$(new_invocation)"

run_preprocess_pair conv0 "$BASE_PORT" "$conv0_invocation" 1 \
  --n 1 --h 224 --w 224 --ci 3 --fh 7 --fw 7 --co 64 \
  --padding 3 --stride 2
run_preprocess_pair conv3 "$((BASE_PORT + 2))" "$conv3_invocation" 2 \
  --n 1 --h 56 --w 56 --ci 64 --fh 3 --fw 3 --co 64 \
  --padding 1 --stride 1

record_digest() {
  python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); f=p.open("rb"); f.seek(-32, 2); b=f.read(32); assert len(b)==32; print(b.hex())' "$1"
}

conv0_p0_digest="$(record_digest "$WORKDIR/conv0/key_p0.conv")"
conv0_p1_digest="$(record_digest "$WORKDIR/conv0/key_p1.conv")"
conv3_p0_digest="$(record_digest "$WORKDIR/conv3/key_p0.conv")"
conv3_p1_digest="$(record_digest "$WORKDIR/conv3/key_p1.conv")"
NONLINEAR_EVIDENCE="$WORKDIR/resnet18_graph_prefix_stock_keygen_2026_08_09.csv"
stock_keygen_common=(
  --gpu 0 --invocation-id "$graph_invocation"
  --manifest-digest "$actual_manifest_sha"
  --p0-conv0-state "$WORKDIR/p0/conv0.state"
  --p1-conv0-state "$WORKDIR/p1/conv0.state"
  --p0-conv3-state "$WORKDIR/p0/conv3.state"
  --p1-conv3-state "$WORKDIR/p1/conv3.state"
)
CUDA_VISIBLE_DEVICES="$CHECK_GPU" \
  "$ROOT/bin/test_stock_nonlinear_prefix_keygen" \
  "${stock_keygen_common[@]}" \
  --p0-output "$WORKDIR/p0/stock-nonlinear.record" \
  --p1-output "$WORKDIR/p1/stock-nonlinear.record" --csv-header \
  >"$NONLINEAR_EVIDENCE"

mkdir -p "$WORKDIR/control-stale-nonlinear"
printf DO_NOT_OVERWRITE \
  >"$WORKDIR/control-stale-nonlinear/p0.record"
set +e
CUDA_VISIBLE_DEVICES="$CHECK_GPU" \
  "$ROOT/bin/test_stock_nonlinear_prefix_keygen" \
  "${stock_keygen_common[@]}" \
  --p0-output "$WORKDIR/control-stale-nonlinear/p0.record" \
  --p1-output "$WORKDIR/control-stale-nonlinear/p1.record" \
  >"$WORKDIR/control-stale-nonlinear.log" 2>&1
stale_nonlinear_rc=$?
set -e
if [[ $stale_nonlinear_rc -ne 2 ||
      "$(<"$WORKDIR/control-stale-nonlinear/p0.record")" != DO_NOT_OVERWRITE ||
      -e "$WORKDIR/control-stale-nonlinear/p0.record.tmp" ||
      -e "$WORKDIR/control-stale-nonlinear/p1.record" ||
      -e "$WORKDIR/control-stale-nonlinear/p1.record.tmp" ]]; then
  echo "[graph-prefix] stale nonlinear output control failed" >&2
  exit 1
fi

mkdir -p "$WORKDIR/control-nonlinear-publication/p1.record"
printf BLOCK_RENAME \
  >"$WORKDIR/control-nonlinear-publication/p1.record/sentinel"
CUDA_VISIBLE_DEVICES="$CHECK_GPU" \
  "$ROOT/bin/test_stock_nonlinear_prefix_keygen" --publication-control \
  --p0-output "$WORKDIR/control-nonlinear-publication/p0.record" \
  --p1-output "$WORKDIR/control-nonlinear-publication/p1.record" \
  --csv-header >"$WORKDIR/control-nonlinear-publication.csv"
if [[ -e "$WORKDIR/control-nonlinear-publication/p0.record" ||
      -e "$WORKDIR/control-nonlinear-publication/p0.record.tmp" ||
      -e "$WORKDIR/control-nonlinear-publication/p1.record.tmp" ||
      ! -f "$WORKDIR/control-nonlinear-publication/p1.record/sentinel" ]]; then
  echo "[graph-prefix] nonlinear partial-publication control failed" >&2
  exit 1
fi
nonlinear_p0_digest="$(record_digest "$WORKDIR/p0/stock-nonlinear.record")"
nonlinear_p1_digest="$(record_digest "$WORKDIR/p1/stock-nonlinear.record")"

common_graph=(
  --host 127.0.0.1 --port "$((BASE_PORT + 4))"
  --ledger "$LEDGER" --invocation-id "$graph_invocation"
  --manifest-digest "$actual_manifest_sha"
  --conv0-p0-digest "$conv0_p0_digest"
  --conv0-p1-digest "$conv0_p1_digest"
  --conv3-p0-digest "$conv3_p0_digest"
  --conv3-p1-digest "$conv3_p1_digest"
  --nonlinear-p0-digest "$nonlinear_p0_digest"
  --nonlinear-p1-digest "$nonlinear_p1_digest"
)

CUDA_VISIBLE_DEVICES="$P0_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
  --party 0 "${common_graph[@]}" \
  --conv0-record "$WORKDIR/conv0/key_p0.conv" \
  --conv0-state "$WORKDIR/p0/conv0.state" \
  --conv3-record "$WORKDIR/conv3/key_p0.conv" \
  --conv3-state "$WORKDIR/p0/conv3.state" \
  --nonlinear-record "$WORKDIR/p0/stock-nonlinear.record" \
  --output "$WORKDIR/p0/graph-prefix.run" --csv-header \
  >"$WORKDIR/graph.p0.csv" 2>"$WORKDIR/graph.p0.log" &
P0_PID=$!
CUDA_VISIBLE_DEVICES="$P1_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
  --party 1 "${common_graph[@]}" \
  --conv0-record "$WORKDIR/conv0/key_p1.conv" \
  --conv0-state "$WORKDIR/p1/conv0.state" \
  --conv3-record "$WORKDIR/conv3/key_p1.conv" \
  --conv3-state "$WORKDIR/p1/conv3.state" \
  --nonlinear-record "$WORKDIR/p1/stock-nonlinear.record" \
  --output "$WORKDIR/p1/graph-prefix.run" --csv-header \
  >"$WORKDIR/graph.p1.csv" 2>"$WORKDIR/graph.p1.log" &
P1_PID=$!
wait_pair_fail_fast
p0_rc=$PAIR_P0_RC
p1_rc=$PAIR_P1_RC
if [[ $p0_rc -ne 0 || $p1_rc -ne 0 ]]; then
  cat "$WORKDIR/graph.p0.log" "$WORKDIR/graph.p1.log" >&2
  exit 1
fi

GRAPH_EVIDENCE="$WORKDIR/resnet18_graph_prefix_2026_08_09.csv"
CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
  --check --manifest-digest "$actual_manifest_sha" \
  --p0-output "$WORKDIR/p0/graph-prefix.run" \
  --p1-output "$WORKDIR/p1/graph-prefix.run" \
  --p0-conv0-state "$WORKDIR/p0/conv0.state" \
  --p1-conv0-state "$WORKDIR/p1/conv0.state" \
  --p0-conv3-state "$WORKDIR/p0/conv3.state" \
  --p1-conv3-state "$WORKDIR/p1/conv3.state" \
  --p0-nonlinear-record "$WORKDIR/p0/stock-nonlinear.record" \
  --p1-nonlinear-record "$WORKDIR/p1/stock-nonlinear.record" --csv-header \
  >"$GRAPH_EVIDENCE"

run_preflight_rejection() {
  local name=$1
  local port=$2
  local invocation=$3
  local p0_conv0=$4
  local p1_conv0=$5
  local p0_conv3=$6
  local p1_conv3=$7
  local p0_nonlinear=$8
  local p1_nonlinear=$9
  local p0_output="$WORKDIR/$name-p0.run"
  local p1_output="$WORKDIR/$name-p1.run"
  local control_common=(
    --host 127.0.0.1 --port "$port" --ledger "$LEDGER"
    --invocation-id "$invocation" --manifest-digest "$actual_manifest_sha"
    --conv0-p0-digest "$conv0_p0_digest"
    --conv0-p1-digest "$conv0_p1_digest"
    --conv3-p0-digest "$conv3_p0_digest"
    --conv3-p1-digest "$conv3_p1_digest"
    --nonlinear-p0-digest "$nonlinear_p0_digest"
    --nonlinear-p1-digest "$nonlinear_p1_digest"
  )
  CUDA_VISIBLE_DEVICES="$P0_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
    --party 0 "${control_common[@]}" \
    --conv0-record "$p0_conv0" --conv0-state "$WORKDIR/p0/conv0.state" \
    --conv3-record "$p0_conv3" --conv3-state "$WORKDIR/p0/conv3.state" \
    --nonlinear-record "$p0_nonlinear" \
    --output "$p0_output" >"$WORKDIR/$name.p0.log" 2>&1 &
  P0_PID=$!
  CUDA_VISIBLE_DEVICES="$P1_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
    --party 1 "${control_common[@]}" \
    --conv0-record "$p1_conv0" --conv0-state "$WORKDIR/p1/conv0.state" \
    --conv3-record "$p1_conv3" --conv3-state "$WORKDIR/p1/conv3.state" \
    --nonlinear-record "$p1_nonlinear" \
    --output "$p1_output" >"$WORKDIR/$name.p1.log" 2>&1 &
  P1_PID=$!
  set +e
  wait "$P0_PID"; local p0_control_rc=$?
  wait "$P1_PID"; local p1_control_rc=$?
  set -e
  P0_PID=""
  P1_PID=""
  if [[ $p0_control_rc -ne 2 || $p1_control_rc -ne 2 ||
        -e "$p0_output" || -e "$p1_output" ]]; then
    echo "[graph-prefix] $name preflight control failed" >&2
    return 1
  fi
}

cp --reflink=auto "$WORKDIR/conv3/key_p1.conv" \
  "$WORKDIR/conv3/key_p1.corrupt.conv"
python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); f=p.open("r+b"); f.seek(1024); b=f.read(1); assert len(b)==1; f.seek(1024); f.write(bytes([b[0]^1])); f.flush(); f.close()' \
  "$WORKDIR/conv3/key_p1.corrupt.conv"
run_preflight_rejection corrupt-peer "$((BASE_PORT + 8))" \
  "$graph_invocation" \
  "$WORKDIR/conv0/key_p0.conv" "$WORKDIR/conv0/key_p1.conv" \
  "$WORKDIR/conv3/key_p0.conv" "$WORKDIR/conv3/key_p1.corrupt.conv" \
  "$WORKDIR/p0/stock-nonlinear.record" \
  "$WORKDIR/p1/stock-nonlinear.record"
run_preflight_rejection swapped-order "$((BASE_PORT + 10))" \
  "$graph_invocation" \
  "$WORKDIR/conv3/key_p0.conv" "$WORKDIR/conv3/key_p1.conv" \
  "$WORKDIR/conv0/key_p0.conv" "$WORKDIR/conv0/key_p1.conv" \
  "$WORKDIR/p0/stock-nonlinear.record" \
  "$WORKDIR/p1/stock-nonlinear.record"
cp --reflink=auto "$WORKDIR/p1/stock-nonlinear.record" \
  "$WORKDIR/p1/stock-nonlinear.corrupt.record"
python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); f=p.open("r+b"); f.seek(4096); b=f.read(1); assert len(b)==1; f.seek(4096); f.write(bytes([b[0]^1])); f.flush(); f.close()' \
  "$WORKDIR/p1/stock-nonlinear.corrupt.record"
run_preflight_rejection corrupt-nonlinear "$((BASE_PORT + 12))" \
  "$graph_invocation" \
  "$WORKDIR/conv0/key_p0.conv" "$WORKDIR/conv0/key_p1.conv" \
  "$WORKDIR/conv3/key_p0.conv" "$WORKDIR/conv3/key_p1.conv" \
  "$WORKDIR/p0/stock-nonlinear.record" \
  "$WORKDIR/p1/stock-nonlinear.corrupt.record"

# The consume-once claim must reject replay before OT/GPU work.
mkdir -p "$WORKDIR/replay-p0" "$WORKDIR/replay-p1"
replay_common=("${common_graph[@]}")
replay_common[3]="$((BASE_PORT + 14))"
CUDA_VISIBLE_DEVICES="$P0_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
  --party 0 "${replay_common[@]}" \
  --conv0-record "$WORKDIR/conv0/key_p0.conv" \
  --conv0-state "$WORKDIR/p0/conv0.state" \
  --conv3-record "$WORKDIR/conv3/key_p0.conv" \
  --conv3-state "$WORKDIR/p0/conv3.state" \
  --nonlinear-record "$WORKDIR/p0/stock-nonlinear.record" \
  --output "$WORKDIR/replay-p0/graph-prefix.run" \
  >"$WORKDIR/replay.p0.log" 2>&1 &
P0_PID=$!
CUDA_VISIBLE_DEVICES="$P1_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
  --party 1 "${replay_common[@]}" \
  --conv0-record "$WORKDIR/conv0/key_p1.conv" \
  --conv0-state "$WORKDIR/p1/conv0.state" \
  --conv3-record "$WORKDIR/conv3/key_p1.conv" \
  --conv3-state "$WORKDIR/p1/conv3.state" \
  --nonlinear-record "$WORKDIR/p1/stock-nonlinear.record" \
  --output "$WORKDIR/replay-p1/graph-prefix.run" \
  >"$WORKDIR/replay.p1.log" 2>&1 &
P1_PID=$!
set +e
wait "$P0_PID"; replay_p0_rc=$?
wait "$P1_PID"; replay_p1_rc=$?
set -e
P0_PID=""; P1_PID=""
if [[ $replay_p0_rc -eq 0 || $replay_p1_rc -eq 0 || \
      -e "$WORKDIR/replay-p0/graph-prefix.run" || \
      -e "$WORKDIR/replay-p1/graph-prefix.run" ]]; then
  echo "[graph-prefix] consume-once replay control failed" >&2
  exit 1
fi

# Any byte corruption must be rejected by the post-exit record checker.
cp --reflink=auto "$WORKDIR/p1/graph-prefix.run" \
  "$WORKDIR/p1/graph-prefix.corrupt.run"
python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); f=p.open("r+b"); f.seek(4096); b=f.read(1); assert len(b)==1; f.seek(4096); f.write(bytes([b[0]^1])); f.flush(); f.close()' \
  "$WORKDIR/p1/graph-prefix.corrupt.run"
set +e
CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$ROOT/bin/test_resnet18_graph_prefix" \
  --check --manifest-digest "$actual_manifest_sha" \
  --p0-output "$WORKDIR/p0/graph-prefix.run" \
  --p1-output "$WORKDIR/p1/graph-prefix.corrupt.run" \
  --p0-conv0-state "$WORKDIR/p0/conv0.state" \
  --p1-conv0-state "$WORKDIR/p1/conv0.state" \
  --p0-conv3-state "$WORKDIR/p0/conv3.state" \
  --p1-conv3-state "$WORKDIR/p1/conv3.state" \
  --p0-nonlinear-record "$WORKDIR/p0/stock-nonlinear.record" \
  --p1-nonlinear-record "$WORKDIR/p1/stock-nonlinear.record" \
  >"$WORKDIR/corrupt.log" 2>&1
corrupt_rc=$?
set -e
if [[ $corrupt_rc -ne 1 ]]; then
  echo "[graph-prefix] corrupted-record control failed (rc=$corrupt_rc)" >&2
  exit 1
fi

PARTY_METRICS="$WORKDIR/resnet18_graph_prefix_party_metrics_2026_08_09.csv"
{
  sed -n '1,2p' "$WORKDIR/graph.p0.csv"
  sed -n '2p' "$WORKDIR/graph.p1.csv"
} >"$PARTY_METRICS"
PREPROCESS_METRICS="$WORKDIR/resnet18_graph_prefix_linear_metrics_2026_08_09.csv"
python3 - "$PREPROCESS_METRICS" \
  "$WORKDIR/conv0.p0.csv" "$WORKDIR/conv0.p1.csv" \
  "$WORKDIR/conv0.check.csv" \
  "$WORKDIR/conv3.p0.csv" "$WORKDIR/conv3.p1.csv" \
  "$WORKDIR/conv3.check.csv" <<'PY'
import csv
import pathlib
import sys

def row(path):
    rows = list(csv.DictReader(pathlib.Path(path).open()))
    if len(rows) != 1:
        raise SystemExit(f"expected one row in {path}")
    return rows[0]

out_path = pathlib.Path(sys.argv[1])
inputs = sys.argv[2:]
fields = [
    "layer", "party0_total_us", "party1_total_us", "critical_path_us",
    "ring_batches", "ring_ole_instances", "dpf_trees",
    "party0_protocol_bytes_sent", "party1_protocol_bytes_sent",
    "final_payload_bytes_per_party", "checker_two_share_online_us",
    "linear_record_checker", "status",
]
with out_path.open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for layer, offset in (("conv0", 0), ("conv3", 3)):
        p0, p1, check = map(row, inputs[offset:offset + 3])
        passed = (
            p0["status"] == p1["status"] == check["status"] == "pass"
            and check["unchanged_online_contract"] == "pass"
        )
        writer.writerow({
            "layer": layer,
            "party0_total_us": p0["total_us"],
            "party1_total_us": p1["total_us"],
            "critical_path_us": max(float(p0["total_us"]), float(p1["total_us"])),
            "ring_batches": p0["ring_batches"],
            "ring_ole_instances": p0["ring_ole_instances"],
            "dpf_trees": p0["dpf_trees"],
            "party0_protocol_bytes_sent": p0["protocol_bytes_sent"],
            "party1_protocol_bytes_sent": p1["protocol_bytes_sent"],
            "final_payload_bytes_per_party": check["final_payload_bytes_per_party"],
            "checker_two_share_online_us": check["checker_two_share_online_us"],
            "linear_record_checker": check["unchanged_online_contract"],
            "status": "pass" if passed else "FAIL",
        })
        if not passed:
            raise SystemExit(f"{layer} evidence did not pass")
PY
CONTROLS="$WORKDIR/resnet18_graph_prefix_controls_2026_08_09.csv"
{
  echo "control,expected,observed,status"
  echo "consume_once_replay,reject,rejected,pass"
  echo "corrupt_peer_input,reject_before_gpu_or_output,rejected,pass"
  echo "swapped_layer_order,reject_before_gpu_or_output,rejected,pass"
  echo "corrupt_nonlinear_key_record,reject_before_gpu_or_output,rejected,pass"
  echo "stale_nonlinear_output,reject_without_overwrite,rejected,pass"
  echo "forced_nonlinear_second_rename,bilateral_rollback,both_outputs_absent,pass"
  echo "corrupt_output_record,reject_at_post_exit_checker,rejected,pass"
  echo "residual_branch_control,not_applicable_no_branch_in_prefix,not_applicable,pass"
} >"$CONTROLS"

HASHES="$WORKDIR/resnet18_graph_prefix_artifact_hashes_2026_08_09.txt"
hash_artifact() {
  local path=$1
  local label=$2
  printf '%s  %s\n' "$(sha256sum "$path" | cut -d' ' -f1)" "$label"
}
{
  hash_artifact "$ROOT/bin/test_two_party_conv_preprocess" \
    "bin/test_two_party_conv_preprocess"
  hash_artifact "$ROOT/bin/test_stock_nonlinear_prefix_keygen" \
    "bin/test_stock_nonlinear_prefix_keygen"
  hash_artifact "$ROOT/bin/test_resnet18_graph_prefix" \
    "bin/test_resnet18_graph_prefix"
  hash_artifact "$ROOT/src/test_resnet18_graph_prefix.cu" \
    "src/test_resnet18_graph_prefix.cu"
  hash_artifact "$ROOT/src/linear_preprocess.h" \
    "src/linear_preprocess.h"
  hash_artifact "$ROOT/src/linear_preprocess_backend.cuh" \
    "src/linear_preprocess_backend.cuh"
  hash_artifact "$ROOT/src/linear_preprocess_conv.cu" \
    "src/linear_preprocess_conv.cu"
  hash_artifact "$ROOT/src/test_stock_nonlinear_prefix_keygen.cu" \
    "src/test_stock_nonlinear_prefix_keygen.cu"
  hash_artifact "$ROOT/src/stock_nonlinear_prefix_record.h" \
    "src/stock_nonlinear_prefix_record.h"
  hash_artifact "$ROOT/src/graph_mask_state.h" "src/graph_mask_state.h"
  hash_artifact "$ROOT/scripts/run_resnet18_graph_prefix.sh" \
    "scripts/run_resnet18_graph_prefix.sh"
  hash_artifact "$MANIFEST" \
    "results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json"
  hash_artifact "$WORKDIR/conv0/key_p0.conv" "private:conv0_p0_record"
  hash_artifact "$WORKDIR/conv0/key_p1.conv" "private:conv0_p1_record"
  hash_artifact "$WORKDIR/conv3/key_p0.conv" "private:conv3_p0_record"
  hash_artifact "$WORKDIR/conv3/key_p1.conv" "private:conv3_p1_record"
  hash_artifact "$WORKDIR/p0/stock-nonlinear.record" \
    "private:p0_stock_nonlinear_record"
  hash_artifact "$WORKDIR/p1/stock-nonlinear.record" \
    "private:p1_stock_nonlinear_record"
  hash_artifact "$WORKDIR/p0/graph-prefix.run" "private:p0_graph_run"
  hash_artifact "$WORKDIR/p1/graph-prefix.run" "private:p1_graph_run"
  hash_artifact "$GRAPH_EVIDENCE" \
    "results/graph/resnet18_graph_prefix_2026_08_09.csv"
  hash_artifact "$PARTY_METRICS" \
    "results/graph/resnet18_graph_prefix_party_metrics_2026_08_09.csv"
  hash_artifact "$CONTROLS" \
    "results/graph/resnet18_graph_prefix_controls_2026_08_09.csv"
  hash_artifact "$PREPROCESS_METRICS" \
    "results/graph/resnet18_graph_prefix_linear_metrics_2026_08_09.csv"
  hash_artifact "$NONLINEAR_EVIDENCE" \
    "results/graph/resnet18_graph_prefix_stock_keygen_2026_08_09.csv"
} >"$HASHES"

METADATA="$WORKDIR/resnet18_graph_prefix_metadata_2026_08_09.json"
python3 - "$METADATA" <<PY
import json, pathlib
out = pathlib.Path(__import__('sys').argv[1])
data = {
  "schema": "ringlpn-resnet18-graph-prefix-v2",
  "publication_date": "2026-08-09",
  "source_manifest": "results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json",
  "source_manifest_sha256": "$actual_manifest_sha",
  "source_bound_nodes": [
    {"ordinal": 0, "node": "conv0", "shape": "1x224x224x3,7x7x3x64,stride2,pad3", "execution": "unchanged gpuConv2DBeaver"},
    {"ordinal": 1, "node": "StochasticTR(10)", "shape": "1x112x112x64", "execution": "live secure_stochastic_truncate_batch"},
    {"ordinal": 2, "node": "maxpool1", "shape": "3x3,stride2,pad1", "execution": "unchanged stock dcf::gpuMaxPool"},
    {"ordinal": 3, "node": "relu1", "shape": "1x56x56x64", "execution": "unchanged stock dcf::gpuReluExtend"},
    {"ordinal": 4, "node": "conv3", "shape": "1x56x56x64,3x3x64x64,stride1,pad1", "execution": "unchanged gpuConv2DBeaver"}
  ],
  "ring_modulus_bits": 128,
  "bitwidth": 32,
  "ring_lpn_parameters": {"n": 262144, "c": 2, "t": 8, "noise": "regular"},
  "truncate_shift": 10,
  "ot_backend": "sci-iknp",
  "transport": "single-host IPv4 loopback for Ring-LPN, truncation, and stock nonlinear online traffic",
  "known_clear_input": "all-zero",
  "known_clear_weights": "all-zero",
  "stock_nonlinear_key_execution": True,
  "dealerless_nonlinear_preprocessing": False,
  "nonlinear_key_source": "test-only trusted stock adapter reading both parties' source-bound mask states",
  "nonlinear_key_bytes_per_party": {
    "maxpool": 503968064,
    "relu": 68289576,
    "total": 572257640
  },
  "linear_metrics": "results/graph/resnet18_graph_prefix_linear_metrics_2026_08_09.csv",
  "stock_keygen_metrics": "results/graph/resnet18_graph_prefix_stock_keygen_2026_08_09.csv",
  "party_stage_metrics": "results/graph/resnet18_graph_prefix_party_metrics_2026_08_09.csv",
  "scope": "known-zero functionality/composition and unchanged-consumer control only; trusted nonlinear key source; not dealerless nonlinear preprocessing, private/trained model execution, full ResNet18 inference, accuracy, deployment, or an end-to-end security claim",
  "controls": {
    "consume_once_replay": "rejected",
    "corrupt_peer_input": "rejected before GPU/output",
    "swapped_layer_order": "rejected before GPU/output",
    "corrupted_nonlinear_record": "rejected before GPU/output",
    "stale_nonlinear_output": "rejected without overwrite",
    "forced_nonlinear_second_rename": "bilateral rollback; both outputs absent",
    "corrupted_run_record": "rejected by post-exit checker",
    "residual_branch": "not applicable; prefix has no branch"
  },
  "invocations": {"conv0": "$conv0_invocation", "conv3": "$conv3_invocation", "graph_and_stock_nonlinear": "$graph_invocation"},
  "record_digests": {
    "conv0_p0": "$conv0_p0_digest", "conv0_p1": "$conv0_p1_digest",
    "conv3_p0": "$conv3_p0_digest", "conv3_p1": "$conv3_p1_digest",
    "nonlinear_p0": "$nonlinear_p0_digest",
    "nonlinear_p1": "$nonlinear_p1_digest"
  }
}
out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
PY
publish_result() {
  local source=$1
  local name=$2
  local temporary="$RESULTS/.$name.$graph_invocation.tmp"
  cp "$source" "$temporary"
  chmod 444 "$temporary"
  mv -f "$temporary" "$RESULTS/$name"
}
publish_result "$GRAPH_EVIDENCE" "resnet18_graph_prefix_2026_08_09.csv"
publish_result "$PARTY_METRICS" \
  "resnet18_graph_prefix_party_metrics_2026_08_09.csv"
publish_result "$PREPROCESS_METRICS" \
  "resnet18_graph_prefix_linear_metrics_2026_08_09.csv"
publish_result "$NONLINEAR_EVIDENCE" \
  "resnet18_graph_prefix_stock_keygen_2026_08_09.csv"
publish_result "$CONTROLS" \
  "resnet18_graph_prefix_controls_2026_08_09.csv"
publish_result "$HASHES" \
  "resnet18_graph_prefix_artifact_hashes_2026_08_09.txt"
publish_result "$METADATA" \
  "resnet18_graph_prefix_metadata_2026_08_09.json"

echo "[graph-prefix] evidence: $RESULTS/resnet18_graph_prefix_2026_08_09.csv"
echo "[graph-prefix] metadata: $RESULTS/resnet18_graph_prefix_metadata_2026_08_09.json"
echo "[graph-prefix] exact known-zero prefix PASS"
