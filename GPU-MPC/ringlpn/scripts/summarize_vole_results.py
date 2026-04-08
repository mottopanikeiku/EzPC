#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime


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
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            logn = int(row["logn"])
            requested_qbits = int(row["requested_qbits"])
            actual_qbits = int(row["actual_qbits"])
            outputs = int(row["m"])
            lanes = int(row["c"])
            noise_weight = int(row["noise_weight"])
            iters = int(row["iters"])
            x_mean = to_float(row["x_mean_us"])
            y_mean = to_float(row["y_mean_us"])
            z_mean = to_float(row["z_mean_us"])
            expand_mean = to_float(row["expand_mean_us"])
            expand_std = to_float(row["expand_std_us"])

            per_output_expand = None
            outputs_per_s = None
            pair_polymuls_per_s = None
            if expand_mean and expand_mean > 0:
                per_output_expand = expand_mean / outputs
                outputs_per_s = outputs * 1e6 / expand_mean
                pair_polymuls_per_s = (3 * outputs * lanes) * 1e6 / expand_mean

            rows.append(
                {
                    "device": row["device"],
                    "input_mode": row["input_mode"],
                    "n": n,
                    "logn": logn,
                    "requested_qbits": requested_qbits,
                    "actual_qbits": actual_qbits,
                    "outputs": outputs,
                    "lanes": lanes,
                    "noise_weight": noise_weight,
                    "iters": iters,
                    "validation": row["validation"],
                    "x_mean": x_mean,
                    "y_mean": y_mean,
                    "z_mean": z_mean,
                    "expand_mean": expand_mean,
                    "expand_std": expand_std,
                    "per_output_expand": per_output_expand,
                    "outputs_per_s": outputs_per_s,
                    "pair_polymuls_per_s": pair_polymuls_per_s,
                    "correct": int(row["correct"]),
                }
            )

    rows.sort(key=lambda row: (row["requested_qbits"], row["n"]))
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    requested_values = sorted({row["requested_qbits"] for row in rows})
    actual_values = sorted({row["actual_qbits"] for row in rows})
    input_modes = sorted({row["input_mode"] for row in rows})
    outputs_values = sorted({row["outputs"] for row in rows})
    lanes_values = sorted({row["lanes"] for row in rows})
    noise_values = sorted({row["noise_weight"] for row in rows})

    requested_label = ", ".join(str(value) for value in requested_values) if requested_values else "?"
    actual_label = ", ".join(str(value) for value in actual_values) if actual_values else "?"
    input_label = ", ".join(input_modes) if input_modes else "?"
    outputs_label = ", ".join(str(value) for value in outputs_values) if outputs_values else "?"
    lanes_label = ", ".join(str(value) for value in lanes_values) if lanes_values else "?"
    noise_label = ", ".join(str(value) for value in noise_values) if noise_values else "?"

    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write(f"# Ring-LPN VOLE GPU Sweep (Requested q={requested_label})\n\n")
        handle.write(f"Generated: {now}\n\n")
        handle.write("## Results\n\n")
        handle.write(
            "| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for row in rows:
            handle.write(
                "| {n} | {logn} | {requested_qbits} | {actual_qbits} | {outputs} | {lanes} | {noise_weight} | {validation} | {iters} | {x_mean:.3f} | {y_mean:.3f} | {z_mean:.3f} | {expand_mean:.3f} | {per_output_expand:.3f} | {outputs_per_s:.2f} | {pair_polymuls_per_s:.2f} |\n".format(
                    **row
                )
            )
        handle.write("\n")
        handle.write("## Notes\n\n")
        handle.write(
            f"- This sweep covers requested qbits {requested_label} and realizes them with actual qbits {actual_label} on the promoted single-prime GPU path.\n"
        )
        handle.write(
            f"- Input mode for this benchmark is {input_label}; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.\n"
        )
        handle.write(
            f"- These runs use m in {{{outputs_label}}}, c in {{{lanes_label}}}, and noise weight in {{{noise_label}}}.\n"
        )
        handle.write(
            "- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.\n"
        )
        handle.write(
            "- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.\n"
        )
        handle.write(
            "- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.\n"
        )
        handle.write(
            "- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.\n"
        )


if __name__ == "__main__":
    main()