#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "ringlpn" / "results"
CSV_PATH = RESULTS_DIR / f"dpf_online_keygen_bin{os.getenv('BIN_BITS', '16')}_chunk{os.getenv('CHUNK_SIZE', '8192')}.csv"
MD_PATH = RESULTS_DIR / f"dpf_online_keygen_bin{os.getenv('BIN_BITS', '16')}_chunk{os.getenv('CHUNK_SIZE', '8192')}.md"


def choose_schedule(n: int) -> tuple[int, int]:
    if n <= 32768:
        return 100, 10
    if n <= 131072:
        return 50, 5
    if n <= 262144:
        return 20, 3
    if n <= 524288:
        return 10, 2
    return 3, 1


def detect_gpu_arch() -> str | None:
    gpu_arch = os.getenv("GPU_ARCH")
    if gpu_arch:
        return gpu_arch
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first_line = completed.stdout.strip().splitlines()[0].strip()
    return first_line.replace(".", "") if first_line else None


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(cmd, cwd=BASE_DIR, check=True, capture_output=True, text=True, env=env)
    return completed.stdout.strip()


def extract_csv_lines(raw_output: str) -> list[str]:
    lines: list[str] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line or line.startswith("reserved memory:"):
            continue
        if "," in line:
            lines.append(line)
    if not lines:
        raise RuntimeError("Benchmark output did not contain a CSV line")
    return lines


def main() -> None:
    bin_bits = int(os.getenv("BIN_BITS", "16"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "8192"))
    n_values = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    run_env = os.environ.copy()
    gpu_arch = detect_gpu_arch()
    if gpu_arch:
        run_env["GPU_ARCH"] = gpu_arch

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    make_cmd = ["make"]
    if gpu_arch:
        make_cmd.append(f"GPU_ARCH={gpu_arch}")
    make_cmd.append("dpf_online_keygen")
    run_command(make_cmd, env=run_env)

    lines: list[str] = []
    for index, n in enumerate(n_values):
        iters, warmup = choose_schedule(n)
        cmd = [
            str(BASE_DIR / "tests" / "fss" / "dpf_online_keygen"),
            "--bin",
            str(bin_bits),
            "--n",
            str(n),
            "--chunk-size",
            str(chunk_size),
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
        ]
        if index == 0:
            cmd.append("--csv-header")
        lines.extend(extract_csv_lines(run_command(cmd, env=run_env)))

    CSV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_command([
        "python3",
        str(BASE_DIR / "scripts" / "summarize_dpf_online_keygen.py"),
        "--csv",
        str(CSV_PATH),
        "--out-md",
        str(MD_PATH),
    ], env=run_env)
    print(f"Wrote {CSV_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()