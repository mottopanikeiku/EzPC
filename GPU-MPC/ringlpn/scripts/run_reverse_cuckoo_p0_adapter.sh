#!/usr/bin/env bash
# Reproduces the closest distributed Reverse Cuckoo baseline without vendoring libOTe.
set -euo pipefail

LIBOTE_URL=https://github.com/osu-crypto/libOTe.git
LIBOTE_COMMIT=edb5d32822eabf2dda9f6844d85d0ce2e402cdd5
LIBOTE_LICENSE_SHA256=39a218ef068824bd03e653b675f4cc8880a155632370aa0cab0419b7010fadcd
CRYPTOTOOLS_COMMIT=0cf6986873e2b83966d5110398dca99172d63c20
CRYPTOTOOLS_LICENSE_SHA256=9c6a7e1292d3bcd93fcac9d1c8346a73a6c9dcecf1892210196fc62132a8f811
PAPER_URL=https://github.com/ladnir/dmpf.git
PAPER_COMMIT=b55bcc4696d10e57bdea8c282a851fdd4fad0c2b
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RINGLPN_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
WORK_DIR=${REVERSE_CUCKOO_WORK_DIR:-${TMPDIR:-/tmp}/reverse-cuckoo-p0-${LIBOTE_COMMIT}}
REPORT=${1:-$RINGLPN_DIR/results/reports/reverse_cuckoo_p0_baseline_2026_08_04.json}
LIBOTE_DIR=$WORK_DIR/libOTe
PAPER_DIR=$WORK_DIR/dmpf-paper
PATCH=$SCRIPT_DIR/libote_reverse_cuckoo_p0.patch
ADAPTER=$SCRIPT_DIR/libote_reverse_cuckoo_p0_adapter.cpp

mkdir -p -- "$WORK_DIR" "$(dirname -- "$REPORT")"

fail_closed() {
    local blocker=$1
    BLOCKER=$blocker REPORT_PATH=$REPORT python3 -c 'import json, os
path = os.environ["REPORT_PATH"]
data = {
  "schema_version": "reverse-cuckoo-p0-baseline-1",
  "date": "2026-08-04",
  "status": "unsupported",
  "blocker": os.environ["BLOCKER"],
  "metrics_emitted": False,
}
with open(path + ".tmp", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
os.replace(path + ".tmp", path)'
    printf '%s\n' "unsupported: $blocker" >&2
    exit 2
}

for command in git cmake python3 sha256sum; do
    command -v "$command" >/dev/null 2>&1 || fail_closed "required build command is unavailable: $command"
done

if [[ ! -d "$LIBOTE_DIR/.git" ]]; then
    git clone --branch dmpf --single-branch "$LIBOTE_URL" "$LIBOTE_DIR" \
        || fail_closed "could not clone libOTe dmpf branch from $LIBOTE_URL"
fi
git -C "$LIBOTE_DIR" checkout --detach "$LIBOTE_COMMIT" \
    || fail_closed "libOTe pin $LIBOTE_COMMIT is unavailable"
[[ $(git -C "$LIBOTE_DIR" rev-parse HEAD) == "$LIBOTE_COMMIT" ]] \
    || fail_closed "libOTe checkout did not resolve to $LIBOTE_COMMIT"
[[ $(sha256sum "$LIBOTE_DIR/LICENSE" | cut -d' ' -f1) == "$LIBOTE_LICENSE_SHA256" ]] \
    || fail_closed "libOTe LICENSE differs from the audited MIT license at $LIBOTE_COMMIT"
git -C "$LIBOTE_DIR" submodule update --init --recursive \
    || fail_closed "could not initialize libOTe's pinned cryptoTools submodule"
[[ $(git -C "$LIBOTE_DIR/cryptoTools" rev-parse HEAD) == "$CRYPTOTOOLS_COMMIT" ]] \
    || fail_closed "libOTe cryptoTools submodule did not resolve to audited pin $CRYPTOTOOLS_COMMIT"
[[ $(sha256sum "$LIBOTE_DIR/cryptoTools/LICENSE" | cut -d' ' -f1) == "$CRYPTOTOOLS_LICENSE_SHA256" ]] \
    || fail_closed "cryptoTools LICENSE differs from its audited MIT license at $CRYPTOTOOLS_COMMIT"

if [[ ! -d "$PAPER_DIR/.git" ]]; then
    git clone "$PAPER_URL" "$PAPER_DIR" \
        || fail_closed "could not clone paper source from $PAPER_URL"
fi
git -C "$PAPER_DIR" checkout --detach "$PAPER_COMMIT" \
    || fail_closed "paper source pin $PAPER_COMMIT is unavailable"
[[ $(git -C "$PAPER_DIR" rev-parse HEAD) == "$PAPER_COMMIT" ]] \
    || fail_closed "paper source checkout did not resolve to $PAPER_COMMIT"
for license_name in LICENSE LICENSE.md LICENSE.txt COPYING COPYING.md COPYING.txt; do
    [[ ! -e "$PAPER_DIR/$license_name" ]] \
        || fail_closed "paper source unexpectedly has $license_name; re-audit its separate license before running"
done

# Re-establish a clean pinned source tree. The only modifications that follow are this audited patch
# and a copied adapter translation unit; neither downloaded tree is inside the project repository.
git -C "$LIBOTE_DIR" reset --hard "$LIBOTE_COMMIT" >/dev/null
git -C "$LIBOTE_DIR" clean -ffd >/dev/null
if ! git -C "$LIBOTE_DIR" apply --check --ignore-space-change "$PATCH"; then
    fail_closed "source-level blocker: adapter patch does not apply to libOTe $LIBOTE_COMMIT (RingLpnTriple caller-factor/capture API or frontend CMake context changed)"
fi
git -C "$LIBOTE_DIR" apply --ignore-space-change "$PATCH"
install -m 0644 "$ADAPTER" "$LIBOTE_DIR/frontend/libote_reverse_cuckoo_p0_adapter.cpp"

cmake -S "$LIBOTE_DIR" -B "$LIBOTE_DIR/out/build/linux" \
    -DENABLE_ALL_OT=ON \
    -DENABLE_BOOST=ON -DFETCH_BOOST=ON \
    -DENABLE_SODIUM=ON -DFETCH_SODIUM=ON \
    -DSUDO_FETCH=OFF -DFETCH_AUTO=ON \
    -DPARALLEL_FETCH="${BUILD_JOBS:-$(nproc)}" \
    -DCMAKE_BUILD_TYPE=Release \
    || fail_closed "isolated libOTe dependency/configuration step failed at $LIBOTE_COMMIT; no metrics were produced"

cmake --build "$LIBOTE_DIR/out/build/linux" --target reverse_cuckoo_p0_adapter \
    --parallel "${BUILD_JOBS:-$(nproc)}" \
    || fail_closed "source-level blocker: exact p0 adapter did not compile against libOTe $LIBOTE_COMMIT; no metrics were produced"

BINARY=$LIBOTE_DIR/out/build/linux/frontend/reverse_cuckoo_p0_adapter
[[ -x "$BINARY" ]] \
    || fail_closed "built adapter executable is missing at the pinned build path: $BINARY"

"$BINARY" "$REPORT" \
    || fail_closed "isolated adapter rejected the run; inspect its source-level blocker above (no successful metrics retained)"
python3 -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    report = json.load(f)
if report.get("status") != "complete":
    raise SystemExit("adapter report did not complete")
if report["parameters"] != {"n": 1048576, "c": 4, "t": 16, "p0": "4611686018326724609", "coefficient_bits": 62}:
    raise SystemExit("adapter report parameters do not match the pinned diagnostic")
if report["layout"]["name"] != "libote_native_16_folded_raw":
    raise SystemExit("adapter report did not label the native folded layout")
if not report["controls"]["collision_accumulating_reference"] or not report["controls"]["corruption_rejected"]:
    raise SystemExit("adapter differential controls did not pass")' "$REPORT" \
    || fail_closed "completed adapter output failed the dated report contract"
printf '%s\n' "$REPORT"
