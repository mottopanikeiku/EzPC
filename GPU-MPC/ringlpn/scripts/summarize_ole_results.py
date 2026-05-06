#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def fmt_us(value: str) -> str:
    try:
        return f"{float(value):,.3f}"
    except ValueError:
        return value


def fmt_mib(value: str) -> str:
    try:
        return f"{float(value) / (1024.0 * 1024.0):,.2f}"
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines: list[str] = []
    lines.append("# GPU Figure 2 OLE over Ring-LPN/SPFSS")
    lines.append("")
    lines.append("Configuration: single 62-bit prime, uniform sparse noise, SPFSS domain `[0, 2N)`, folded into `Z_p[X]/(X^N+1)`.")
    lines.append("")
    lines.append("| n | c | t | validation | host validation | key bytes MiB | keygen us | OLE expand mean us | OLE expand std us |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {n} | {c} | {t} | {validation} | {host_validation} | {key_mib} | {keygen} | {mean} | {std} |".format(
                n=row["n"],
                c=row["c"],
                t=row["t"],
                validation=row["validation"],
                host_validation=row["host_validation"],
                key_mib=fmt_mib(row["spfss_pair_key_bytes"]),
                keygen=fmt_us(row["spfss_keygen_us"]),
                mean=fmt_us(row["ole_expand_mean_us"]),
                std=fmt_us(row["ole_expand_std_us"]),
            )
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- `requested_qbits=64` maps to the promoted single 62-bit prime.")
    lines.append("- This artifact stops at OLE: it validates `z_0 + z_1 == x_0 * x_1`; Beaver triple conversion and Orca FC integration are follow-up work.")
    lines.append("- Uniform noise is intentionally the first-pass configuration; regular-noise and CRT lifts are separate follow-ups.")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
