#!/usr/bin/env python3
import argparse
import csv


def as_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    lines = [
        "# Ring-LPN OLE Linear-Layer Beaver Artifact",
        "",
        "Configuration: ring-polynomial matrix multiplication over the single 62-bit prime. Each ring product uses two Figure 2 OLE instances to form Beaver shares.",
        "",
        "| rows | inner | cols | n | c | t | validation | OLE instances | key bytes MiB | keygen us | linear expand mean us | linear expand std us |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        key_mib = as_float(row["spfss_pair_key_bytes"]) / (1024.0 * 1024.0)
        lines.append(
            f"| {row['rows']} | {row['inner']} | {row['cols']} | {row['n']} | "
            f"{row['c']} | {row['t']} | {row['validation']} | {row['ole_instances']} | "
            f"{key_mib:.2f} | {as_float(row['spfss_keygen_us']):,.3f} | "
            f"{as_float(row['linear_expand_mean_us']):,.3f} | "
            f"{as_float(row['linear_expand_std_us']):,.3f} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- This is the two-OLE-to-Beaver conversion applied to a linear layer whose entries are Ring-LPN polynomials.",
            "- It validates Beaver correctness for matrix multiplication over `Z_p[X]/(X^N+1)`.",
            "- It is not yet Orca FC integration: scalar packing and `Z_p -> Z_{2^bw}` share conversion remain follow-up work.",
        ]
    )

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
