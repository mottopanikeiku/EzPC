#!/usr/bin/env python3
"""Replay immutable FC matrix data and regress independent-layer statistics."""

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "fc"
SOURCE = RESULTS / "two_party_fc_model_scale_cnn2_cnn3_2026_08_14.csv"
HISTORICAL_INPUTS = RESULTS / "model_scale_inputs_2026_08_14"
LAYER_MANIFEST = HISTORICAL_INPUTS / "orca_forward_linear_layer_manifest_2026_08_04.json"
WORKLOAD_MANIFEST = HISTORICAL_INPUTS / "orca_model_scale_workload_manifest_2026_08_04.json"
ENVIRONMENT = RESULTS / "two_party_fc_model_scale_cnn2_cnn3_environment_2026_08_14.txt"
RAW_SCHEMA = RESULTS / "two_party_fc_model_scale_result_schemas_2026_08_04.json"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay(source, destination):
    # The runner's private execution_plan.json was intentionally not retained.
    # Reconstruct its public selection metadata; the aggregator independently
    # checks every identity/count against the immutable manifests and raw rows.
    destination.mkdir(parents=True, exist_ok=False)
    raw = read_rows(source)
    first = raw[0]
    layers = json.loads(LAYER_MANIFEST.read_text(encoding="utf-8"))["layers"]
    metadata = {name: first[name] for name in (
        "schema_version", "publication_date", "manifest_sha256",
        "workload_manifest_sha256", "workload")}
    metadata["models"] = []
    for model in sorted({row["model"] for row in raw}):
        selected = [row for row in layers if row["model"] == model]
        metadata["models"].append({
            "model": model,
            "model_order": selected[0]["model_order"],
            "expected_executable_layers": sum(
                row["operator"] == "fc" and
                (first["workload"] != "classifier" or row["is_classifier"])
                for row in selected),
            "unsupported_convolution_layers": sum(
                row["operator"] == "conv2d" for row in selected),
            "unsupported_truncation_layers": sum(
                row["truncation_status"] != "supported" for row in selected),
        })
    plan = destination / "execution_plan.json"
    plan.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "aggregate_two_party_fc_model_scale.py"),
        "--source-csv", str(source), "--plan-metadata", str(plan),
        "--layer-manifest", str(LAYER_MANIFEST),
        "--workload-manifest", str(WORKLOAD_MANIFEST),
        "--aggregate-csv", str(destination / "observations.csv"),
        "--statistics-csv", str(destination / "summary.csv"),
        "--output-schema-json", str(destination / "output_schema.json"),
        "--trials", "10", "--binary", "test_two_party_fc_preprocess",
        "--environment", str(ENVIRONMENT), "--result-schema", str(RAW_SCHEMA),
    ], check=True)
    return read_rows(destination / "summary.csv")


class IndependentLayerStatisticsTest(unittest.TestCase):
    def test_reversing_one_layers_labels_preserves_estimates_intervals_and_ratios(self):
        with tempfile.TemporaryDirectory(prefix="ringlpn-fc-statistics-") as directory:
            work = pathlib.Path(directory)
            original = replay(SOURCE, work / "original")
            raw = read_rows(SOURCE)
            for row in raw:
                if (row["model"], row["source_layer"], row["sample_role"]) == (
                        "CNN2", "fc5", "measured"):
                    row["trial"] = str(11 - int(row["trial"]))
            relabeled = work / "relabeled.csv"
            with relabeled.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(raw[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(raw)
            permuted = replay(relabeled, work / "permuted")
            # Raw-byte provenance is intentionally different. Every statistical
            # result, including each layer's paired ratio, must be identical.
            for population in (original, permuted):
                for row in population:
                    row.pop("raw_trials_sha256")
            self.assertEqual(original, permuted)
            by_key = {
                (row["model"], row["scope"], row["source_layer"], row["metric"]): row
                for row in original
            }
            total = by_key[("CNN2", "model", "", "critical_path_setup_included_us")]
            layer_sum = sum((Decimal(by_key[(
                "CNN2", "layer", layer, "critical_path_setup_included_us")]["mean"])
                for layer in ("fc4", "fc5")), Decimal(0))
            self.assertEqual(Decimal(total["estimate"]), layer_sum)
            self.assertLess(Decimal(total["mean_ci95_low"]), layer_sum)
            self.assertGreater(Decimal(total["mean_ci95_high"]), layer_sum)
            for field in ("n", "median", "sample_stdev", "q1", "q3", "iqr", "min", "max"):
                self.assertEqual(total[field], "NA")
            ratio = by_key[("CNN2", "model", "",
                "setup_included_preprocess_over_matched_dealer_ratio_of_summed_layer_means")]
            dealer = by_key[("CNN2", "model", "", "matched_dealer_keygen_us")]
            self.assertAlmostEqual(
                Decimal(ratio["estimate"]), layer_sum / Decimal(dealer["estimate"]), places=20)
            self.assertLess(Decimal(ratio["mean_ci95_low"]), Decimal(ratio["mean_ci95_high"]))


    def test_model_ratio_bootstrap_preserves_within_layer_pairing(self):
        with tempfile.TemporaryDirectory(prefix="ringlpn-fc-pairing-") as directory:
            work = pathlib.Path(directory)
            raw = read_rows(SOURCE)
            for row in raw:
                if row["sample_role"] not in {"warmup", "measured"}:
                    continue
                # A perfectly proportional paired protocol/dealer population
                # has no ratio uncertainty, despite variable wall-time samples.
                for party in (0, 1):
                    row[f"p{party}_total_us"] = str(Decimal(row["matched_dealer_keygen_us"]) * 7)
                    row[f"p{party}_preflight_us"] = "0"
                    row[f"p{party}_ot_setup_us"] = "0"
            paired = work / "paired.csv"
            with paired.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(raw[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(raw)
            summary = replay(paired, work / "result")
            ratios = {
                row["model"]: row for row in summary if row["scope"] == "model"
                and row["metric"] ==
                "setup_included_preprocess_over_matched_dealer_ratio_of_summed_layer_means"
            }
            self.assertEqual(set(ratios), {"CNN2", "CNN3"})
            for row in ratios.values():
                for field in ("estimate", "mean_ci95_low", "mean_ci95_high"):
                    self.assertEqual(Decimal(row[field]), Decimal(7))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=pathlib.Path,
                        help="replay the unchanged retained matrix into a NEW directory instead of running regressions")
    arguments, remaining = parser.parse_known_args()
    if arguments.replay_dir:
        if remaining:
            parser.error("unrecognized replay arguments: " + " ".join(remaining))
        replay(SOURCE, arguments.replay_dir)
        print(arguments.replay_dir / "summary.csv")
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
