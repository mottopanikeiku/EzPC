#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--unsupported-csv")
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def main():
    args = parse_args()
    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r["n"])
            logn = int(r["logn"])
            requested_qbits = int(r["requested_qbits"])
            actual_qbits = int(r["actual_qbits"])
            batch_size = int(r["batch_size"])
            iters = int(r["iters"])
            validation = r["validation"]
            ntt_mean = to_float(r["ntt_mean_us"])
            ntt_std = to_float(r["ntt_std_us"])
            intt_mean = to_float(r["intt_mean_us"])
            intt_std = to_float(r["intt_std_us"])
            mul_mean = to_float(r["poly_mul_mean_us"])
            mul_std = to_float(r["poly_mul_std_us"])
            correct = int(r["correct"])
            per_poly_mul = None
            polys_per_s = None
            coeff_gb_s = None
            if mul_mean and mul_mean > 0:
                per_poly_mul = mul_mean / batch_size
                polys_per_s = batch_size * 1e6 / mul_mean
                bytes_per_op = batch_size * n * 4 * 2
                coeff_gb_s = (bytes_per_op / 1e9) / (mul_mean * 1e-6)

            rows.append({
                "device": r["device"],
                "n": n,
                "logn": logn,
                "requested_qbits": requested_qbits,
                "actual_qbits": actual_qbits,
                "batch_size": batch_size,
                "iters": iters,
                "validation": validation,
                "ntt_mean": ntt_mean,
                "ntt_std": ntt_std,
                "intt_mean": intt_mean,
                "intt_std": intt_std,
                "mul_mean": mul_mean,
                "mul_std": mul_std,
                "per_poly_mul": per_poly_mul,
                "polys_per_s": polys_per_s,
                "coeff_gb_s": coeff_gb_s,
                "correct": correct,
            })

    rows.sort(key=lambda row: row["n"])

    unsupported_rows = []
    if args.unsupported_csv:
        with open(args.unsupported_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                unsupported_rows.append(r)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Ring-LPN GPU NTT Sweep (Requested q=32)\n\n")
        f.write(f"Generated: {now}\n\n")
        f.write("## Results\n\n")
        f.write("| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                "| {n} | {logn} | {requested_qbits} | {actual_qbits} | {batch_size} | {validation} | {iters} | {ntt_mean:.3f} | {intt_mean:.3f} | {mul_mean:.3f} | {per_poly_mul:.3f} | {polys_per_s:.2f} | {coeff_gb_s:.2f} |\n".format(
                    **r
                )
            )
        f.write("\n")

        if unsupported_rows:
            f.write("## Unsupported Requested Points\n\n")
            f.write("| n | log2(n) | q req | status | reason |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for r in unsupported_rows:
                f.write(
                    "| {n} | {logn} | {requested_qbits} | {status} | {reason} |\n".format(
                        **r
                    )
                )
            f.write("\n")

        f.write("## Notes\n\n")
        f.write("- This CUDA path currently targets requested qbits=32 and realizes it with a single 30-bit prime, so q actual is 30 in every supported run.\n")
        f.write("- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.\n")
        f.write("- Est. coeff GB/s uses bytes_per_op = batch_size * n * 4 * 2 as a rough traffic proxy, not a hardware counter.\n")
        f.write("- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.\n")
        f.write("- The selected 30-bit prime supports n up to 2^20, so this sweep intentionally extends past the CPU NFLLib uint32_t cutoff.\n")


if __name__ == "__main__":
    main()