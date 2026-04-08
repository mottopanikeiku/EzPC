#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from io import StringIO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def main():
    args = parse_args()
    rows = []
    with open(args.csv, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]

    if not raw_lines:
        raise RuntimeError("CSV file is empty")

    header = raw_lines[0]
    header_columns = len(header.split(","))
    filtered_lines = [header]
    for line in raw_lines[1:]:
        if line.startswith("reserved memory:"):
            continue
        if len(line.split(",")) != header_columns:
            continue
        filtered_lines.append(line)

    reader = csv.DictReader(StringIO("\n".join(filtered_lines) + "\n"))
    for row in reader:
        n = int(row["n"])
        full_pair_key_bytes = int(row["full_pair_key_bytes"])
        partial_peak_pair_key_bytes = int(row["partial_peak_pair_key_bytes"])
        partial_total_pair_key_bytes = int(row["partial_total_pair_key_bytes"])
        rows.append(
            {
                "device": row["device"],
                "input_mode": row["input_mode"],
                "bin": int(row["bin"]),
                "n": n,
                "chunk_size": int(row["chunk_size"]),
                "iters": int(row["iters"]),
                "validation": row["validation"],
                "full_pair_key_bytes": full_pair_key_bytes,
                "partial_peak_pair_key_bytes": partial_peak_pair_key_bytes,
                "partial_total_pair_key_bytes": partial_total_pair_key_bytes,
                "peak_reduction": to_float(row["peak_reduction"]),
                "total_bytes_multiplier": to_float(row["total_bytes_multiplier"]),
                "full_pair_keygen_mean_us": to_float(row["full_pair_keygen_mean_us"]),
                "partial_pair_keygen_mean_us": to_float(row["partial_pair_keygen_mean_us"]),
                "keygen_time_overhead": to_float(row["keygen_time_overhead"]),
                "full_pair_key_mib": full_pair_key_bytes / (1024.0 * 1024.0),
                "partial_peak_pair_key_mib": partial_peak_pair_key_bytes / (1024.0 * 1024.0),
                "correct": int(row["correct"]),
            }
        )

    rows.sort(key=lambda row: row["n"])
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    bin_values = sorted({row["bin"] for row in rows})
    chunk_values = sorted({row["chunk_size"] for row in rows})
    bin_label = ", ".join(str(value) for value in bin_values) if bin_values else "?"
    chunk_label = ", ".join(str(value) for value in chunk_values) if chunk_values else "?"

    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write(f"# DPF Online Key Generation Sweep (bin={bin_label})\n\n")
        handle.write(f"Generated: {now}\n\n")
        handle.write("## Results\n\n")
        handle.write(
            "| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for row in rows:
            handle.write(
                "| {n} | {bin} | {chunk_size} | {validation} | {iters} | {full_pair_key_mib:.2f} | {partial_peak_pair_key_mib:.2f} | {peak_reduction:.2f}x | {total_bytes_multiplier:.3f}x | {full_pair_keygen_mean_us:.3f} | {partial_pair_keygen_mean_us:.3f} | {keygen_time_overhead:.3f}x |\n".format(
                    **row
                )
            )
        handle.write("\n")
        handle.write("## Notes\n\n")
        handle.write(
            f"- This sweep measures standalone DPF online key generation with eval-all keys for bin {bin_label} and chunk size {chunk_label}.\n"
        )
        handle.write(
            "- Full pair key is the total key material generated at once for both parties. Partial peak pair key is the maximum per-chunk key material when keys are generated only for the current chunk.\n"
        )
        handle.write(
            "- Peak reduction quantifies the reduction in peak key footprint from partial online key generation. Total bytes multiplier captures the total key material generated across all chunks relative to the one-shot offline baseline.\n"
        )
        handle.write(
            "- Full pair keygen mean measures one-shot generation for both parties. Partial pipeline mean measures generating all chunks for both parties.\n"
        )
        handle.write(
            "- Validation checks key serialization layout and parsed key metadata for both full and chunked modes. This sweep is a key-generation systems benchmark, not an end-to-end FSS evaluation benchmark.\n"
        )


if __name__ == "__main__":
    main()