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
    cpu_requested_qbits = cpu.get("requested_qbits", cpu.get("qbits", "?"))
    cpu_actual_qbits = cpu.get("actual_qbits", cpu_requested_qbits)
    gpu_requested_qbits = gpu.get("requested_qbits", gpu.get("qbits", "?"))
    gpu_actual_qbits = gpu.get("actual_qbits", gpu_requested_qbits)
    gpu_batch = int(gpu.get("batch_size", "1"))

    cpu_ntt = as_float(cpu, "ntt_mean_us")
    cpu_poly = as_float(cpu, "poly_mul_mean_us")
    gpu_ntt = as_float(gpu, "ntt_mean_us")
    gpu_poly = as_float(gpu, "poly_mul_mean_us")
    gpu_ntt_per_poly = gpu_ntt / gpu_batch
    gpu_poly_per_poly = gpu_poly / gpu_batch

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(
            f"# CPU vs GPU NTT Comparison (n={n}, q req CPU/GPU={cpu_requested_qbits}/{gpu_requested_qbits})\n\n"
        )
        f.write(f"Generated: {now}\n\n")
        f.write("## Comparison\n\n")
        f.write("| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(
            f"| CPU (NFLLib) | {cpu_actual_qbits} | 1 | {cpu.get('validation', 'n/a')} | {cpu['ntt_mean_us']} | {cpu['intt_mean_us']} | {cpu['poly_mul_mean_us']} | {cpu['poly_mul_mean_us']} | n/a |\n"
        )
        f.write(
            f"| GPU (CUDA) | {gpu_actual_qbits} | {gpu_batch} | {gpu.get('validation', 'n/a')} | {gpu['ntt_mean_us']} | {gpu['intt_mean_us']} | {gpu['poly_mul_mean_us']} | {gpu_poly_per_poly:.3f} | {gpu['correct']} |\n"
        )
        f.write("\n")
        f.write("## Speedups\n\n")
        f.write(f"- Forward NTT speedup per polynomial: {cpu_ntt / gpu_ntt_per_poly:.2f}x\n")
        f.write(f"- Full PolyMul speedup per polynomial: {cpu_poly / gpu_poly_per_poly:.2f}x\n")


if __name__ == "__main__":
    main()
