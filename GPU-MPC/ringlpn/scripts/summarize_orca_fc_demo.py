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
        "Configuration: forward-only `2x2 * 2x2` FC layer, `bw=16`, bounded q62 constant-polynomial masks, `poly_n=8192`, `c=2`, `t=8`, regular-noise label, `tf=None`, zero bias.",
        "",
        "| seed | second seed | shape | key bytes / party | carry conversion | replay | second seed | distinct second seed | online contract | validation |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        shape = f"{row['rows']}x{row['inner']}x{row['cols']}"
        lines.append(
            f"| {row['seed']} | {row['second_seed']} | {shape} | "
            f"{row['key_bytes_per_party']} | {row['corrected_carry_conversion']} | "
            f"{row['deterministic_replay']} | {row['second_seed_validation']} | "
            f"{row['second_seed_distinct']} | {row['online_contract']} | "
            f"{row['validation']} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- The demo writes raw party buffers in `FCLayer::readForwardKey` order: `A`, `B`, `C_masked`.",
            "- `C_masked` is formed by converting q62 Beaver-product shares to `Z_{2^16}` with the carry-corrected bridge and then adding an output mask in the ring.",
            "- The online phase calls the existing `gpuMatmulBeaver` implementation unchanged and reconstructs `clear FC output + output mask`.",
            "- This is a v1 correctness demo, not q128/CRT, high-density packing, or secure distributed q62-to-ring conversion.",
        ]
    )

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
