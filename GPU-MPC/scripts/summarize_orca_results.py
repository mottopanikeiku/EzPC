#!/usr/bin/env python3
import argparse
import csv
import os
import re
from datetime import datetime

LOG_PATTERNS = [
    re.compile(r"Avg key read time \(ms\):\s*(?P<key_read>[0-9.]+)", re.I),
    re.compile(r"Avg compute time \(ms\):\s*(?P<compute>[0-9.]+)", re.I),
    re.compile(r"Total time taken \(ms\):\s*(?P<total>[0-9.]+)", re.I),
    re.compile(r"Total bytes communicated:\s*(?P<comm>[0-9.]+)", re.I),
    re.compile(r"Comm \(B\)=\s*(?P<comm_b>[0-9.]+)", re.I),
    re.compile(r"Average time taken \(microseconds\)=\s*(?P<avg_us>[0-9.]+)", re.I),
]

MODEL_FROM_LOG = re.compile(r"^(?P<model>.+?)_(?P<role>dealer|eval|inf)_(?P<party>p0|p1|dealer)\.log$")

NSYS_API_RE = re.compile(r"CUDA API Summary", re.I)
NSYS_GPU_RE = re.compile(r"CUDA GPU Kernel Summary", re.I)


def parse_log(path):
    data = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pat in LOG_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                data.update({k: v for k, v in m.groupdict().items() if v is not None})
    return data


def read_nsys_stats(report_path):
    if not os.path.exists(report_path):
        return {}
    api_time = None
    gpu_time = None
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Total" in line and "Time (us)" in line:
                # Next lines contain totals; parsing nsys text is fragile, so keep raw.
                continue
    # Placeholder: return the path for manual review.
    return {"nsys_report": os.path.basename(report_path)}


def infer_role(filename):
    m = MODEL_FROM_LOG.match(os.path.basename(filename))
    if not m:
        return None
    return m.groupdict()


def float_or_none(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def compute_ratio(key_read, compute):
    if key_read is None or compute is None or compute == 0:
        return None
    return key_read / compute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = []
    for fname in sorted(os.listdir(args.log_dir)):
        info = infer_role(fname)
        if not info:
            continue
        path = os.path.join(args.log_dir, fname)
        data = parse_log(path)
        key_read = float_or_none(data.get("key_read"))
        compute = float_or_none(data.get("compute"))
        total = float_or_none(data.get("total"))
        comm = float_or_none(data.get("comm"))
        if comm is None:
            comm = float_or_none(data.get("comm_b"))
        avg_us = float_or_none(data.get("avg_us"))
        if total is None and avg_us is not None:
            total = avg_us / 1000.0
        ratio = compute_ratio(key_read, compute)

        rows.append({
            "model": info["model"],
            "role": info["role"],
            "party": info["party"],
            "key_read_ms": key_read,
            "compute_ms": compute,
            "total_ms": total,
            "comm_bytes": comm,
            "read_compute_ratio": ratio,
            "log_file": fname,
        })

    # Write CSV
    fieldnames = [
        "model",
        "role",
        "party",
        "key_read_ms",
        "compute_ms",
        "total_ms",
        "comm_bytes",
        "read_compute_ratio",
        "log_file",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Write Markdown summary
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"# ORCA Profiling Summary\n\nGenerated: {now}\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Model | Role | Party | Key Read (ms) | Compute (ms) | Total (ms) | Comm (B) | Read/Compute | Log |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                "| {model} | {role} | {party} | {key_read_ms} | {compute_ms} | {total_ms} | {comm_bytes} | {read_compute_ratio} | {log_file} |\n".format(**r)
            )
        f.write("\n")

        f.write("## Notes\n\n")
        f.write("- This summary parses standard ORCA log lines. If your logs differ, update patterns in summarize_orca_results.py.\n")
        f.write("- nsys summary extraction is left as a manual step; add parsing if you need numeric kernel/API breakdowns.\n")


if __name__ == "__main__":
    main()
