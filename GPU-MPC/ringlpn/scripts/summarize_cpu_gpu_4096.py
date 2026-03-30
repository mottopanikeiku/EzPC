#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-csv", required=True)
    p.add_argument("--gpu-csv", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def load_one(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def as_float(row, key):
    return float(row[key])


def main():
    args = parse_args()
    cpu = load_one(args.cpu_csv)
    gpu = load_one(args.gpu_csv)

    n = cpu.get("n", "?")
    qbits = cpu.get("qbits", "?")

    cpu_ntt = as_float(cpu, "ntt_mean_us")
    cpu_poly = as_float(cpu, "poly_mul_mean_us")
    gpu_ntt = as_float(gpu, "ntt_mean_us")
    gpu_poly = as_float(gpu, "poly_mul_mean_us")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"# CPU vs GPU NTT Baseline (n={n}, q={qbits})\n\n")
        f.write(f"Generated: {now}\n\n")
        f.write("## Comparison\n\n")
        f.write("| Impl | NTT mean (us) | NTT std (us) | INTT mean (us) | INTT std (us) | Full PolyMul mean (us) | Full PolyMul std (us) | Correct |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(
            f"| CPU (NFLLib) | {cpu['ntt_mean_us']} | {cpu['ntt_std_us']} | {cpu['intt_mean_us']} | {cpu['intt_std_us']} | {cpu['poly_mul_mean_us']} | {cpu['poly_mul_std_us']} | n/a |\n"
        )
        f.write(
            f"| GPU (CUDA) | {gpu['ntt_mean_us']} | {gpu['ntt_std_us']} | {gpu['intt_mean_us']} | {gpu['intt_std_us']} | {gpu['poly_mul_mean_us']} | {gpu['poly_mul_std_us']} | {gpu['correct']} |\n"
        )
        f.write("\n")
        f.write("## Speedups\n\n")
        f.write(f"- Forward NTT speedup: {cpu_ntt / gpu_ntt:.2f}x\n")
        f.write(f"- Full PolyMul speedup: {cpu_poly / gpu_poly:.2f}x\n")


if __name__ == "__main__":
    main()
