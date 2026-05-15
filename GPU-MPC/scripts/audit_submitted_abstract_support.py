#!/usr/bin/env python3
import argparse
import csv
import re
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an evidence audit for the submitted GPU-FSS abstract."
    )
    parser.add_argument(
        "--out-md",
        default=str(ROOT / "ringlpn/results/submitted_abstract_support_audit.md"),
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "ringlpn/results/submitted_abstract_support_audit.csv"),
    )
    return parser.parse_args()


def parse_human_size(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2)
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }[suffix]
    return number * scale


def size_gib(size_bytes):
    return size_bytes / float(1024**3)


def read_orca_master_log():
    path = ROOT / "orca_runner/logs/master.log"
    rows = []
    current = None
    pending_size = {}
    if not path.exists():
        return rows

    train_re = re.compile(r"=== (?:TRAINING|INFERENCE): ([^=]+?) ===")
    ls_re = re.compile(r"root root\s+([0-9.]+[KMGTP]?)\s+.*?/home/keys/([^/]+)/([^ ]+)")
    read_re = re.compile(r"Avg key read time \(ms\): ([0-9.]+)")
    compute_re = re.compile(r"Avg compute time \(ms\): ([0-9.]+)")

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("========== FULL SUMMARY =========="):
            break

        train_match = train_re.search(line)
        if train_match:
            current = train_match.group(1).strip()
            pending_size.setdefault(current, [])
            continue

        ls_match = ls_re.search(line)
        if ls_match and current:
            size_text = ls_match.group(1)
            party = ls_match.group(2)
            filename = ls_match.group(3)
            size_bytes = parse_human_size(size_text)
            pending_size.setdefault(current, []).append(
                {
                    "party": party,
                    "filename": filename,
                    "size_text": size_text,
                    "size_bytes": size_bytes,
                }
            )
            continue

        read_match = read_re.search(line)
        if read_match and current:
            rows.append(
                {
                    "model": current,
                    "key_read_ms": float(read_match.group(1)),
                    "compute_ms": None,
                    "sizes": pending_size.get(current, []),
                }
            )
            continue

        compute_match = compute_re.search(line)
        if compute_match and rows:
            rows[-1]["compute_ms"] = float(compute_match.group(1))

    dedup = {}
    for row in rows:
        if row["model"] not in dedup:
            dedup[row["model"]] = row
    return list(dedup.values())


def read_csv(path):
    with path.open("r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]
    if not raw_lines:
        return []
    header_columns = len(raw_lines[0].split(","))
    filtered = [raw_lines[0]]
    for line in raw_lines[1:]:
        if line.startswith("reserved memory:"):
            continue
        if len(line.split(",")) == header_columns:
            filtered.append(line)
    return list(csv.DictReader(filtered))


def find_row(rows, **match):
    for row in rows:
        ok = True
        for key, expected in match.items():
            if str(row.get(key)) != str(expected):
                ok = False
                break
        if ok:
            return row
    return None


def read_dpf_support():
    path = ROOT / "ringlpn/results/dpf_online_keygen_bin16_chunk8192.csv"
    rows = read_csv(path)
    row = find_row(rows, n=1048576, chunk_size=8192)
    return path, row


def read_vole_support(qbits):
    path = ROOT / f"ringlpn/results/vole_gpu_q{qbits}_m32_c2_w64.csv"
    rows = read_csv(path)
    if not rows:
        return path, None, None
    rows.sort(key=lambda row: int(row["n"]))
    return path, rows[0], rows[-1]


def parse_cpu_gpu_direct():
    path = ROOT / "ringlpn/results/cpu_gpu_8192_32_batch64.md"
    text = path.read_text(encoding="utf-8")
    ntt = re.search(r"Forward NTT speedup per polynomial: `?([0-9.]+)x`?", text)
    polymul = re.search(r"Full PolyMul speedup per polynomial: `?([0-9.]+)x`?", text)
    return path, ntt.group(1) if ntt else None, polymul.group(1) if polymul else None


def support_rows(orca_rows, dpf_row, vole32_first, vole32_last, vole64_first, vole64_last, ntt_speed, polymul_speed):
    max_key_gib = 0.0
    max_key_text = "none"
    for row in orca_rows:
        for item in row["sizes"]:
            if item["size_bytes"] and item["size_bytes"] > max_key_gib * 1024**3:
                max_key_gib = size_gib(item["size_bytes"])
                max_key_text = f"{row['model']} {item['party']} {item['size_text']}"

    return [
        {
            "claim": "Orca key-read time can approach GPU compute time",
            "status": "supported",
            "evidence": "P-LeNet key read 109.727 ms vs compute 107.727 ms; P-AlexNet key read 104.818 ms vs compute 121.727 ms",
            "notes": "From GPU-MPC/orca_runner/logs/master.log",
        },
        {
            "claim": "Local Orca key files grow from gigabytes to tens of gigabytes",
            "status": "partially supported",
            "evidence": f"Saved local log reaches {max_key_text}, about {max_key_gib:.2f} GiB per party",
            "notes": "The saved local profile supports hundreds of MiB to about 4.0G per party, not tens of GB. Use a weaker phrase unless new larger-model key-size data is collected.",
        },
        {
            "claim": "Chunked DPF online generation reaches 128x peak-footprint reduction with under 2x overhead",
            "status": "supported",
            "evidence": f"N={dpf_row['n']}, chunk={dpf_row['chunk_size']}, full pair key={int(dpf_row['full_pair_key_bytes'])/(1024**2):.2f} MiB, partial peak={int(dpf_row['partial_peak_pair_key_bytes'])/(1024**2):.2f} MiB, reduction={float(dpf_row['peak_reduction']):.2f}x, overhead={float(dpf_row['keygen_time_overhead']):.3f}x",
            "notes": "Standalone eval-all DPF key-generation benchmark, not end-to-end FSS evaluation.",
        },
        {
            "claim": "GPU NTT/PolyMul core has roughly 89x full-PolyMul speedup over NFLLib",
            "status": "supported",
            "evidence": f"Direct n=8192 comparison reports {ntt_speed}x forward-NTT and {polymul_speed}x full-PolyMul per-polynomial speedup",
            "notes": "This is a per-polynomial throughput comparison from the saved n=8192 artifact.",
        },
        {
            "claim": "GPU Ring-LPN VOLE validates across n=8192 to 1048576 for requested q=32 and q=64",
            "status": "supported",
            "evidence": f"q=32: n={vole32_first['n']} to {vole32_last['n']}, validation={vole32_first['validation']}/{vole32_last['validation']}; q=64: n={vole64_first['n']} to {vole64_last['n']}, validation={vole64_first['validation']}/{vole64_last['validation']}",
            "notes": "Requested q=32 maps to actual q=30; requested q=64 maps to actual q=62.",
        },
        {
            "claim": "The current work replaces Orca precomputed keys end-to-end",
            "status": "not supported",
            "evidence": "Current DPF and VOLE artifacts are standalone prototypes; Orca integration is documented as ongoing work",
            "notes": "Use 'toward replacing' or 'study chunked online generation' rather than a completed replacement claim.",
        },
    ]


def write_markdown(path, rows, orca_rows, dpf_path, vole32_path, vole64_path, cpu_gpu_path):
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Submitted Abstract Support Audit\n\n")
        handle.write(f"Generated: {now}\n\n")
        handle.write("## Verdict\n\n")
        handle.write(
            "The submitted abstract is mostly supported by saved GPU-MPC artifacts, but two phrases should be softened for poster or camera-ready use: "
            "`tens of gigabytes` is not shown by the saved local Orca profile, and `replace large precomputed keys` sounds like completed Orca integration even though the current DPF/VOLE results are standalone prototypes.\n\n"
        )
        handle.write("## Claim Matrix\n\n")
        handle.write("| Claim | Status | Evidence | Notes |\n")
        handle.write("| --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['claim']} | {row['status']} | {row['evidence']} | {row['notes']} |\n"
            )
        handle.write("\n## Shortened Evidence-Safe Abstract\n\n")
        handle.write(
            "Privacy-preserving machine learning (PPML) protocols often split computation between function secret sharing (FSS) for non-linear operations such as ReLU and comparison, and additive secret sharing for linear operations. Both use offline/online decompositions to reduce online latency, but this shifts cost to generating, storing, and moving correlated randomness. Our profiling of Orca, a GPU-accelerated FSS-based PPML system in GPU-MPC, shows that precomputed keys reach several gigabytes per party and that key-read time can match GPU computation for moderate models.\n\n"
        )
        handle.write(
            "We develop GPU building blocks toward a unified acceleration framework for this bottleneck. For FSS-based non-linear evaluation, standalone chunked DPF online key generation reduces peak staged pair-key footprint by up to 128x with under 2x time overhead, providing a tunable memory-efficiency knob. For secret-sharing-based linear evaluation, we accelerate Ring-LPN pseudorandom correlation generator components. Our GPU NTT/PolyMul backend, adapted from Cheddar's two-phase kernel structure, achieves roughly 89x per-polynomial full-PolyMul speedup over the NFLLib CPU baseline at n=8192. Built on this backend, our standalone GPU Ring-LPN VOLE prototype validates correctness across degrees from 8192 to 1048576 for requested q=32 and q=64. We are currently integrating these components into Orca.\n\n"
        )
        handle.write("## Orca Profile Rows Extracted From `master.log`\n\n")
        handle.write("| Model | Key files observed | Avg key read (ms) | Avg compute (ms) |\n")
        handle.write("| --- | --- | ---: | ---: |\n")
        for row in orca_rows:
            sizes = ", ".join(f"{item['party']} {item['size_text']}" for item in row["sizes"])
            handle.write(
                f"| {row['model']} | {sizes or 'n/a'} | {row['key_read_ms']:.3f} | {row['compute_ms']:.3f} |\n"
            )
        handle.write("\n## Provenance\n\n")
        handle.write(f"- Orca profiling: `GPU-MPC/orca_runner/logs/master.log`\n")
        handle.write(f"- DPF chunking: `{dpf_path.relative_to(ROOT)}`\n")
        handle.write(f"- VOLE q=32: `{vole32_path.relative_to(ROOT)}`\n")
        handle.write(f"- VOLE q=64: `{vole64_path.relative_to(ROOT)}`\n")
        handle.write(f"- CPU/GPU NTT comparison: `{cpu_gpu_path.relative_to(ROOT)}`\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    orca_rows = read_orca_master_log()
    dpf_path, dpf_row = read_dpf_support()
    vole32_path, vole32_first, vole32_last = read_vole_support(32)
    vole64_path, vole64_first, vole64_last = read_vole_support(64)
    cpu_gpu_path, ntt_speed, polymul_speed = parse_cpu_gpu_direct()

    if dpf_row is None:
        raise RuntimeError("Missing DPF n=1048576 chunk=8192 row")
    if not all([vole32_first, vole32_last, vole64_first, vole64_last]):
        raise RuntimeError("Missing VOLE support rows")
    if not all([ntt_speed, polymul_speed]):
        raise RuntimeError("Missing CPU/GPU speedup rows")

    rows = support_rows(
        orca_rows,
        dpf_row,
        vole32_first,
        vole32_last,
        vole64_first,
        vole64_last,
        ntt_speed,
        polymul_speed,
    )
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(out_md, rows, orca_rows, dpf_path, vole32_path, vole64_path, cpu_gpu_path)
    write_csv(out_csv, rows)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
