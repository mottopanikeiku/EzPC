#!/usr/bin/env python3
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    lines = [
        "# Orca FC Ring-LPN v1 Demo",
        "",
        "Configuration: forward-only small FC suite with bounded q62 constant-polynomial masks, `poly_n=8192`, `c=2`, `t=8`, regular-noise label, `tf=None`, and zero bias.",
        "",
        "| seed | second seed | shape | bw | bound | key bytes / party | baseline bytes / party | carry conversion | replay | second seed | baseline | baseline matches | validation |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]

    for row in rows:
        shape = f"{row['rows']}x{row['inner']}x{row['cols']}"
        lines.append(
            f"| {row['seed']} | {row['second_seed']} | {shape} | "
            f"{row['bw']} | {row.get('no_prime_wrap_bound', '1')} | "
            f"{row['key_bytes_per_party']} | {row.get('baseline_key_bytes_per_party', '0')} | "
            f"{row['corrected_carry_conversion']} | "
            f"{row['deterministic_replay']} | {row['second_seed_validation']} | "
            f"{row.get('baseline_online_contract', 'skipped')} | "
            f"{row.get('baseline_matches_ringlpn', '-1')} | "
            f"{row['validation']} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- The demo writes raw party buffers in `FCLayer::readForwardKey` order: `A`, `B`, `C_masked`.",
            "- `C_masked` is formed by converting q62 Beaver-product shares to `Z_{2^bw}` with the carry-corrected bridge and then adding an output mask in the ring.",
            "- The online phase calls the existing `gpuMatmulBeaver` implementation unchanged and reconstructs `clear FC output + output mask`.",
            "- The baseline column is generated with Orca's `gpuKeygenMatmul` using the same masks and deterministic P0/P1 random-share stream, then run through the same online path.",
            "- This is a v1 correctness demo, not q128/CRT, high-density packing, or secure distributed q62-to-ring conversion.",
        ]
    )

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
