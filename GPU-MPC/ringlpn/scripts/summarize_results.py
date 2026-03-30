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
            limb_bits = int(r["limb_bits"])
            limb_bytes = int(r["limb_bytes"])
            limb_type = r["limb_type"]
            iters = int(r["iters"])
            validation = r["validation"]
            ntt_mean = to_float(r["ntt_mean_us"])
            ntt_std = to_float(r["ntt_std_us"])
            intt_mean = to_float(r["intt_mean_us"])
            intt_std = to_float(r["intt_std_us"])
            mul_mean = to_float(r["poly_mul_mean_us"])
            mul_std = to_float(r["poly_mul_std_us"])
            pointwise_est = None
            if ntt_mean is not None and intt_mean is not None and mul_mean is not None:
                pointwise_est = mul_mean - (2.0 * ntt_mean) - intt_mean

            ops_per_s = None
            gb_per_s = None
            if mul_mean and mul_mean > 0:
                ops_per_s = 1e6 / mul_mean
                nb_moduli = max(1, actual_qbits // limb_bits)
                bytes_per_op = n * limb_bytes * nb_moduli * 2
                gb_per_s = (bytes_per_op / 1e9) / (mul_mean * 1e-6)

            rows.append({
                "n": n,
                "logn": logn,
                "requested_qbits": requested_qbits,
                "actual_qbits": actual_qbits,
                "limb_bits": limb_bits,
                "limb_bytes": limb_bytes,
                "limb_type": limb_type,
                "iters": iters,
                "validation": validation,
                "ntt_mean": ntt_mean,
                "ntt_std": ntt_std,
                "intt_mean": intt_mean,
                "intt_std": intt_std,
                "mul_mean": mul_mean,
                "mul_std": mul_std,
                "pointwise_est": pointwise_est,
                "ops_per_s": ops_per_s,
                "gb_per_s": gb_per_s,
            })

    rows.sort(key=lambda row: (row["n"], row["requested_qbits"]))

    unsupported_rows = []
    if args.unsupported_csv:
        with open(args.unsupported_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                unsupported_rows.append(r)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Ring-LPN CPU Baseline (NFLLib)\n\n")
        f.write(f"Generated: {now}\n\n")
        f.write("## Results\n\n")
        f.write("| n | log2(n) | q req | q actual | limb | validate | iters | NTT mean (us) | INTT mean (us) | Pointwise est. (us) | Full PolyMul mean (us) | Ops/s | Est. coeff GB/s |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                "| {n} | {logn} | {requested_qbits} | {actual_qbits} | {limb_type}/{limb_bits} | {validation} | {iters} | {ntt_mean:.3f} | {intt_mean:.3f} | {pointwise_est:.3f} | {mul_mean:.3f} | {ops_per_s:.2f} | {gb_per_s:.2f} |\n".format(
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
        f.write("- NFLLib only supports aggregated modulus sizes that are multiples of 30 bits in uint32_t mode or 62 bits in uint64_t mode.\n")
        f.write("- Requested qbits 32, 64, and 128 therefore resolve to actual qbits 30, 62, and 124 when the requested point is feasible.\n")
        f.write("- Requested qbits=32 is only feasible up to n=32768 because NFLLib uint32_t mode caps the degree there.\n")
        f.write("- Est. coeff GB/s uses bytes_per_op = n * limb_bytes * nb_moduli * 2 and is a rough throughput proxy, not a hardware counter.\n")
        f.write("- Full PolyMul is measured directly in the benchmark as NTT(a) + NTT(b) + pointwise multiply + INTT.\n")
        f.write("- Pointwise est. is back-computed as Full PolyMul - 2*NTT - INTT and is shown only as a sanity check.\n")
        f.write("- Ops/s = 1e6 / Full PolyMul_mean_us.\n")


if __name__ == "__main__":
    main()
