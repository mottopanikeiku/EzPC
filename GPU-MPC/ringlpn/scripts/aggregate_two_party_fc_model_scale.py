#!/usr/bin/env python3
"""Strict, identity-bound aggregation for repeated FC model-scale measurements."""

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import tempfile
from decimal import Decimal, InvalidOperation, getcontext

getcontext().prec = 50

AGGREGATE_SCHEMA_VERSION = "ringlpn.two-party-fc-model-scale.aggregate.v3"
STATISTICS_SCHEMA_VERSION = "ringlpn.two-party-fc-model-scale.statistics.v2"
INTERVAL_METHOD = "two-sided Student t confidence interval for the arithmetic mean; confidence=0.95; df=n-1"
QUARTILE_METHOD = "R-7 inclusive linear interpolation at probabilities 0.25 and 0.75"
MISSING = {"", "NA"}

INTEGER_TOTALS = [
    "p0_ring_oles", "p1_ring_oles", "p0_dpf_trees", "p1_dpf_trees",
    "p0_public_a_seed_words", "p1_public_a_seed_words", "p0_protocol_bytes",
    "p1_protocol_bytes", "p0_record_bytes", "p1_record_bytes",
    "final_payload_bytes_per_party",
]
DECIMAL_TOTALS = [
    "p0_total_us", "p1_total_us", "matched_dealer_keygen_us",
    "checker_two_share_online_us", "stock_gpuKeygenMatmul_two_party_sequential_us",
    "unchanged_gpuMatmulBeaver_two_share_sequential_us",
]
PARTY_SUM_METRICS = [
    "protocol_dependency_rounds", "preflight_us", "ot_setup_us", "dpf_phase_a_us",
    "dpf_phase_b_us", "dpf_phase_c_us", "spfss_grouping_us",
    "public_polynomial_exchange_us", "gpu_ringlpn_expansion_us",
    "derandomization_openings_us", "conversion_us", "serialization_us", "commit_us",
    "transport_straight_bytes_sent", "transport_straight_bytes_received",
    "transport_reversed_bytes_sent", "transport_reversed_bytes_received", "base_ots",
    "base_ot_setup_bytes_sent", "base_ot_setup_bytes_received",
]
PARTY_MAX_METRICS = ["peak_host_rss_bytes", "peak_gpu_bytes"]
PARTY_MIN_METRICS = ["min_gpu_free_bytes"]
AVAILABILITY_FIELDS = [
    "p0_transport_bytes_include_base_ot", "p1_transport_bytes_include_base_ot",
    "p0_base_ot_setup_dependency_rounds", "p1_base_ot_setup_dependency_rounds",
]
CHECKER_FIELDS = [
    "checker_us_total", "checker_peak_host_rss_bytes_max",
    "checker_peak_gpu_bytes_max", "checker_min_gpu_free_bytes_min",
]
CRITICAL_METRICS = [
    "protocol_dependency_rounds", "preflight_us", "ot_setup_us", "dpf_phase_a_us",
    "dpf_phase_b_us", "dpf_phase_c_us", "spfss_grouping_us",
    "public_polynomial_exchange_us", "gpu_ringlpn_expansion_us",
    "derandomization_openings_us", "conversion_us", "serialization_us", "commit_us",
]
IDENTITY_FIELDS = [
    "source_schema_version", "publication_date", "manifest_sha256",
    "workload_manifest_sha256", "raw_trials_sha256", "binary_sha256",
    "environment_sha256", "result_schema_sha256",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_file_sha256(environment_path, expected_path):
    expected_name = pathlib.Path(expected_path).name
    matches = []
    with open(environment_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("  ", 1)
            if (len(parts) == 2 and len(parts[0]) == 64
                    and all(character in "0123456789abcdef" for character in parts[0])
                    and pathlib.Path(parts[1]).name == expected_name):
                matches.append(parts[0])
    if len(matches) != 1:
        raise SystemExit(
            f"environment must contain exactly one SHA-256 entry for {expected_name}; found {len(matches)}")
    return matches[0]


def parse_decimal(value, context):
    if value in MISSING:
        raise SystemExit(f"missing required numeric field {context}")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise SystemExit(f"invalid decimal in {context}: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise SystemExit(f"invalid nonnegative decimal in {context}: {value!r}")
    return parsed


def parse_nonnegative_integer(value, context):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"invalid integer in {context}: {value!r}") from exc
    if parsed < 0 or str(parsed) != value:
        raise SystemExit(f"non-canonical nonnegative integer in {context}: {value!r}")
    return parsed


def atomic_write_csv(path, fieldnames, rows):
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def incomplete_beta_fraction(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, 401):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 3e-14:
            return result
    raise SystemExit("Student t interval calculation did not converge")


def regularized_incomplete_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * incomplete_beta_fraction(a, b, x) / a
    return 1.0 - factor * incomplete_beta_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value, degrees_of_freedom):
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if value >= 0.0 else tail


def student_t_critical_975(degrees_of_freedom):
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    low, high = 0.0, 1.0
    while student_t_cdf(high, degrees_of_freedom) < 0.975:
        high *= 2.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < 0.975:
            low = midpoint
        else:
            high = midpoint
    return Decimal(str((low + high) / 2.0))


def quantile_r7(sorted_values, probability):
    if not sorted_values:
        raise ValueError("quantile requires a sample")
    position = Decimal(len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if fraction == 0:
        return sorted_values[lower]
    return sorted_values[lower] + fraction * (sorted_values[lower + 1] - sorted_values[lower])


def decimal_text(value):
    if value is None:
        return ""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def summarize(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return {name: "" for name in (
            "mean", "sample_stdev", "median", "q1", "q3", "iqr",
            "mean_ci95_low", "mean_ci95_high", "min", "max")}
    mean = sum(ordered, Decimal(0)) / Decimal(count)
    median = quantile_r7(ordered, Decimal("0.5"))
    q1 = quantile_r7(ordered, Decimal("0.25"))
    q3 = quantile_r7(ordered, Decimal("0.75"))
    stdev = None
    ci_low = None
    ci_high = None
    if count >= 2:
        variance = sum(((value - mean) ** 2 for value in ordered), Decimal(0)) / Decimal(count - 1)
        stdev = variance.sqrt()
        margin = student_t_critical_975(count - 1) * stdev / Decimal(count).sqrt()
        ci_low, ci_high = mean - margin, mean + margin
    return {
        "mean": decimal_text(mean), "sample_stdev": decimal_text(stdev),
        "median": decimal_text(median), "q1": decimal_text(q1),
        "q3": decimal_text(q3), "iqr": decimal_text(q3 - q1),
        "mean_ci95_low": decimal_text(ci_low), "mean_ci95_high": decimal_text(ci_high),
        "min": decimal_text(ordered[0]), "max": decimal_text(ordered[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--plan-metadata", required=True)
    parser.add_argument("--layer-manifest", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--statistics-csv", required=True)
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--require-current-binary", action="store_true")
    parser.add_argument("--result-schema", required=True)
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be positive")

    with open(args.plan_metadata, encoding="utf-8") as handle:
        metadata = json.load(handle)
    required_metadata = {
        "schema_version", "publication_date", "manifest_sha256",
        "workload_manifest_sha256", "workload", "models",
    }
    if not isinstance(metadata, dict) or not required_metadata.issubset(metadata):
        raise SystemExit("plan metadata is incomplete")
    if not isinstance(metadata["models"], list) or not metadata["models"]:
        raise SystemExit("plan metadata has no models")
    actual_layer_manifest_sha256 = sha256(args.layer_manifest)
    actual_workload_manifest_sha256 = sha256(args.workload_manifest)
    if metadata["manifest_sha256"] != actual_layer_manifest_sha256:
        raise SystemExit("plan metadata does not bind the supplied layer manifest")
    if metadata["workload_manifest_sha256"] != actual_workload_manifest_sha256:
        raise SystemExit("plan metadata does not bind the supplied workload manifest")
    with open(args.layer_manifest, encoding="utf-8") as handle:
        layer_manifest = json.load(handle)
    if not isinstance(layer_manifest, dict) or not isinstance(
            layer_manifest.get("layers"), list):
        raise SystemExit("layer manifest has no layers list")
    recorded_binary_sha256 = recorded_file_sha256(args.environment, args.binary)
    if args.require_current_binary and sha256(args.binary) != recorded_binary_sha256:
        raise SystemExit("current binary differs from the binary recorded in the environment")
    recorded_result_schema_sha256 = recorded_file_sha256(
        args.environment, args.result_schema)
    actual_result_schema_sha256 = sha256(args.result_schema)
    if actual_result_schema_sha256 != recorded_result_schema_sha256:
        raise SystemExit("current result schema differs from the schema recorded in the environment")
    actual_environment_sha256 = sha256(args.environment)
    with open(args.result_schema, encoding="utf-8") as handle:
        result_schema = json.load(handle)
    schema_columns = result_schema.get("per_layer", {}).get("columns")
    if not isinstance(schema_columns, list) or any(
            not isinstance(column, str) or not column for column in schema_columns):
        raise SystemExit("result schema has no valid per_layer.columns list")
    if len(schema_columns) != len(set(schema_columns)):
        raise SystemExit("result schema has duplicate per-layer columns")

    with open(args.source_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise SystemExit("source CSV has a missing or duplicate header")
        if reader.fieldnames != schema_columns:
            raise SystemExit("source CSV header does not exactly match result schema")
        rows = list(reader)
    required_columns = {
        "schema_version", "publication_date", "manifest_sha256", "workload_manifest_sha256",
        "model", "source_layer", "trial", "sample_role", "operator", "workload",
        "retained", "status", *INTEGER_TOTALS, *DECIMAL_TOTALS,
        *(f"p{party}_{name}" for name in PARTY_SUM_METRICS + PARTY_MAX_METRICS + PARTY_MIN_METRICS
          for party in (0, 1)),
        "p0_transport_bytes_include_base_ot", "p1_transport_bytes_include_base_ot",
        "checker_us", "checker_peak_host_rss_bytes", "checker_peak_gpu_bytes",
        "checker_min_gpu_free_bytes",
    }
    missing_columns = sorted(required_columns - set(reader.fieldnames))
    if missing_columns:
        raise SystemExit("source CSV lacks required columns: " + ",".join(missing_columns))
    if not rows:
        raise SystemExit("source CSV has no rows")
    for row_number, row in enumerate(rows, 2):
        if None in row:
            raise SystemExit(f"source CSV row {row_number} has extra columns")
        expected_identity = {
            "schema_version": metadata["schema_version"],
            "publication_date": metadata["publication_date"],
            "manifest_sha256": metadata["manifest_sha256"],
            "workload_manifest_sha256": metadata["workload_manifest_sha256"],
            "workload": metadata["workload"],
            "binary_sha256": recorded_binary_sha256,
            "environment_sha256": actual_environment_sha256,
            "result_schema_sha256": actual_result_schema_sha256,
        }
        for field, expected in expected_identity.items():
            if row[field] != str(expected):
                raise SystemExit(f"mixed identity in source CSV row {row_number}: {field}")

    model_metadata = {}
    for entry in metadata["models"]:
        required = {
            "model", "model_order", "expected_executable_layers",
            "unsupported_convolution_layers", "unsupported_truncation_layers",
        }
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise SystemExit("model plan metadata is incomplete")
        model = entry["model"]
        if model in model_metadata:
            raise SystemExit(f"duplicate model metadata: {model}")
        expected = parse_nonnegative_integer(str(entry["expected_executable_layers"]), f"{model}.expected_executable_layers")
        if expected < 1:
            raise SystemExit(f"model {model} has no executable layers")
        model_metadata[model] = entry
    selected_models = set(model_metadata)
    expected_layers = {}
    for entry in layer_manifest["layers"]:
        if not isinstance(entry, dict) or entry.get("model") not in selected_models:
            continue
        key = (entry.get("model"), entry.get("layer"))
        if None in key or key in expected_layers:
            raise SystemExit("selected layer manifest has a missing or duplicate layer identity")
        expected_layers[key] = entry
    if not expected_layers:
        raise SystemExit("layer manifest has no rows for selected models")
    for model, model_meta in model_metadata.items():
        manifest_rows = [
            entry for (entry_model, _), entry in expected_layers.items()
            if entry_model == model
        ]
        model_orders = {int(entry["model_order"]) for entry in manifest_rows}
        if len(model_orders) != 1:
            raise SystemExit(f"layer manifest has mixed model order for {model}")
        recomputed = {
            "model_order": model_orders.pop(),
            "expected_executable_layers": sum(
                entry["operator"] == "fc"
                and (metadata["workload"] != "classifier"
                     or entry["is_classifier"])
                for entry in manifest_rows),
            "unsupported_convolution_layers": sum(
                entry["operator"] == "conv2d" for entry in manifest_rows),
            "unsupported_truncation_layers": sum(
                entry["truncation_status"] != "supported"
                for entry in manifest_rows),
        }
        for field, expected in recomputed.items():
            if int(model_meta[field]) != expected:
                raise SystemExit(
                    f"plan metadata differs from layer manifest for {model}: {field}")
    seen_coverage = set()

    seen_groups = set()
    for row_number, row in enumerate(rows, 2):
        if row["model"] not in model_metadata:
            raise SystemExit(f"unexpected model in source CSV row {row_number}")
        if not row["source_layer"]:
            raise SystemExit(f"missing source layer in source CSV row {row_number}")
        manifest_key = (row["model"], row["source_layer"])
        expected_layer = expected_layers.get(manifest_key)
        if expected_layer is None:
            raise SystemExit(f"raw row {row_number} is absent from the selected layer manifest")
        static_identity = {
            "model": expected_layer["model"],
            "model_order": expected_layer["model_order"],
            "layer": "classifier" if expected_layer["is_classifier"] else expected_layer["layer"],
            "source_layer": expected_layer["layer"],
            "linear_order": expected_layer["linear_order"],
            "forward_order": expected_layer["forward_order"],
            "operator": expected_layer["operator"],
            "source_anchor": expected_layer["source_anchor"],
            "source_text_sha256": expected_layer["source_text_sha256"],
            "batch_source_anchor": expected_layer["batch_source_anchor"],
            "batch": expected_layer["batch"],
            "rows": expected_layer["rows"],
            "inner": expected_layer["inner"],
            "cols": expected_layer["cols"],
            "bw": expected_layer["bw"],
            "qbits": expected_layer["qbits"],
            "noise": expected_layer["noise"],
            "ole_n": expected_layer["ole_n"],
            "ole_c": expected_layer["ole_c"],
            "ole_t": expected_layer["ole_t"],
            "layout": expected_layer["layout"],
            "support_status": expected_layer["ringlpn_status"],
            "truncation_status": expected_layer["truncation_status"],
            "gap": expected_layer["gap"],
        }
        for field, expected in static_identity.items():
            if row[field] != str(expected):
                raise SystemExit(
                    f"raw row {row_number} differs from layer manifest: {field}")
        role = row["sample_role"]
        if role == "coverage":
            if row["trial"] != "-1" or row["retained"] != "no":
                raise SystemExit(f"malformed coverage row in source CSV row {row_number}")
            expected_status = (
                "unsupported" if expected_layer["operator"] == "conv2d"
                else "not_selected")
            if row["status"] != expected_status:
                raise SystemExit(
                    f"malformed coverage status in source CSV row {row_number}")
            expected_coverage = (
                expected_layer["operator"] != "fc"
                or (metadata["workload"] == "classifier"
                    and not expected_layer["is_classifier"]))
            if not expected_coverage or manifest_key in seen_coverage:
                raise SystemExit(f"unexpected or duplicate coverage row {row_number}")
            seen_coverage.add(manifest_key)
            continue
        expected_execution = (
            expected_layer["operator"] == "fc"
            and (metadata["workload"] != "classifier"
                 or expected_layer["is_classifier"]))
        if not expected_execution:
            raise SystemExit(f"raw row {row_number} executes an unselected layer")
        if role not in {"warmup", "measured"} or row["operator"] != "fc":
            raise SystemExit(f"unexpected executable row kind in source CSV row {row_number}")
        try:
            trial = int(row["trial"])
        except ValueError as exc:
            raise SystemExit(f"invalid trial in source CSV row {row_number}") from exc
        expected_role = "warmup" if trial == 0 else "measured"
        if trial < 0 or trial > args.trials or role != expected_role:
            raise SystemExit(f"invalid trial/sample_role pair in source CSV row {row_number}")
        if row["retained"] != "yes" or row["status"] != "pass":
            raise SystemExit(f"incomplete failed trial in source CSV row {row_number}")
        key = (row["model"], trial, row["source_layer"])
        if key in seen_groups:
            raise SystemExit(f"duplicate retained layer in source CSV row {row_number}")
        seen_groups.add(key)

    expected_coverage = {
        key for key, entry in expected_layers.items()
        if entry["operator"] != "fc"
        or (metadata["workload"] == "classifier" and not entry["is_classifier"])
    }
    if seen_coverage != expected_coverage:
        raise SystemExit("raw CSV coverage rows do not exactly match the selected layer plan")
    identity = {
        "source_schema_version": str(metadata["schema_version"]),
        "publication_date": str(metadata["publication_date"]),
        "manifest_sha256": actual_layer_manifest_sha256,
        "workload_manifest_sha256": actual_workload_manifest_sha256,
        "raw_trials_sha256": sha256(args.source_csv),
        "binary_sha256": recorded_binary_sha256,
        "environment_sha256": actual_environment_sha256,
        "result_schema_sha256": actual_result_schema_sha256,
    }
    party_sum_fields = [f"p{party}_{name}_total" for name in PARTY_SUM_METRICS for party in (0, 1)]
    party_max_fields = [f"p{party}_{name}_max" for name in PARTY_MAX_METRICS for party in (0, 1)]
    party_min_fields = [f"p{party}_{name}_min" for name in PARTY_MIN_METRICS for party in (0, 1)]
    critical_fields = [f"critical_path_{name}_total" for name in CRITICAL_METRICS]
    aggregate_fields = [
        "schema_version", *IDENTITY_FIELDS, "model", "model_order", "trial", "sample_role",
        "workload", "expected_executable_layers", "retained_layer_rows", "failed_layer_rows",
        "unsupported_convolution_layers", "unsupported_truncation_layers",
        *[name + "_total" for name in INTEGER_TOTALS + DECIMAL_TOTALS],
        *party_sum_fields, *party_max_fields, *party_min_fields, *AVAILABILITY_FIELDS,
        *CHECKER_FIELDS, *critical_fields, "critical_path_preprocess_us_total",
        "critical_path_setup_included_us_total",
        "execution_status", "full_model_status", "workload_status", "status",
    ]
    aggregates = []
    for model, model_meta in sorted(model_metadata.items(), key=lambda item: int(item[1]["model_order"])):
        expected = int(model_meta["expected_executable_layers"])
        for trial in range(args.trials + 1):
            role = "warmup" if trial == 0 else "measured"
            retained = [
                row for row in rows
                if row["model"] == model and row["trial"] == str(trial)
                and row["sample_role"] == role and row["operator"] == "fc"
            ]
            if len(retained) != expected:
                raise SystemExit(f"incomplete manifest for {model} trial {trial}: expected {expected}, found {len(retained)}")
            result = {
                "schema_version": AGGREGATE_SCHEMA_VERSION, **identity,
                "model": model, "model_order": model_meta["model_order"], "trial": trial,
                "sample_role": role, "workload": metadata["workload"],
                "expected_executable_layers": expected, "retained_layer_rows": len(retained),
                "failed_layer_rows": 0,
                "unsupported_convolution_layers": model_meta["unsupported_convolution_layers"],
                "unsupported_truncation_layers": model_meta["unsupported_truncation_layers"],
            }
            for name in INTEGER_TOTALS:
                result[name + "_total"] = sum(
                    parse_nonnegative_integer(row[name], f"{model}/{trial}/{row['source_layer']}/{name}")
                    for row in retained)
            for name in DECIMAL_TOTALS:
                result[name + "_total"] = sum(
                    (parse_decimal(row[name], f"{model}/{trial}/{row['source_layer']}/{name}") for row in retained),
                    Decimal(0))

            def complete_numeric(column, operation):
                values = [row[column] for row in retained]
                available = [value not in MISSING for value in values]
                if any(available) and not all(available):
                    raise SystemExit(f"partially available field {column} for {model} trial {trial}")
                if not any(available):
                    return "NA"
                parsed = [parse_decimal(value, f"{model}/{trial}/{column}") for value in values]
                return operation(parsed)

            for name in PARTY_SUM_METRICS:
                for party in (0, 1):
                    column = f"p{party}_{name}"
                    result[column + "_total"] = complete_numeric(column, lambda values: sum(values, Decimal(0)))
            for name in PARTY_MAX_METRICS:
                for party in (0, 1):
                    column = f"p{party}_{name}"
                    result[column + "_max"] = complete_numeric(column, max)
            for name in PARTY_MIN_METRICS:
                for party in (0, 1):
                    column = f"p{party}_{name}"
                    result[column + "_min"] = complete_numeric(column, min)
            for party in (0, 1):
                column = f"p{party}_transport_bytes_include_base_ot"
                values = [row[column] for row in retained]
                if all(value == "yes" for value in values):
                    result[column] = "yes"
                elif all(value in MISSING for value in values):
                    result[column] = "NA"
                else:
                    raise SystemExit(f"mixed transport availability field {column} for {model} trial {trial}")
                result[f"p{party}_base_ot_setup_dependency_rounds"] = "NA"
            result["checker_us_total"] = complete_numeric("checker_us", lambda values: sum(values, Decimal(0)))
            result["checker_peak_host_rss_bytes_max"] = complete_numeric("checker_peak_host_rss_bytes", max)
            result["checker_peak_gpu_bytes_max"] = complete_numeric("checker_peak_gpu_bytes", max)
            result["checker_min_gpu_free_bytes_min"] = complete_numeric("checker_min_gpu_free_bytes", min)
            for name in CRITICAL_METRICS:
                pairs = [(row[f"p0_{name}"], row[f"p1_{name}"]) for row in retained]
                if any((left in MISSING) != (right in MISSING) for left, right in pairs):
                    raise SystemExit(f"one-sided critical-path field {name} for {model} trial {trial}")
                availability = [left not in MISSING for left, _ in pairs]
                if any(availability) and not all(availability):
                    raise SystemExit(f"partially available critical-path field {name} for {model} trial {trial}")
                if not any(availability):
                    result[f"critical_path_{name}_total"] = "NA"
                else:
                    result[f"critical_path_{name}_total"] = sum((max(
                        parse_decimal(left, f"{model}/{trial}/p0_{name}"),
                        parse_decimal(right, f"{model}/{trial}/p1_{name}"))
                        for left, right in pairs), Decimal(0))
            result["critical_path_preprocess_us_total"] = sum((max(
                parse_decimal(row["p0_total_us"], f"{model}/{trial}/p0_total_us"),
                parse_decimal(row["p1_total_us"], f"{model}/{trial}/p1_total_us"))
                for row in retained), Decimal(0))
            result["critical_path_setup_included_us_total"] = sum((max(
                parse_decimal(row["p0_total_us"], f"{model}/{trial}/p0_total_us")
                + parse_decimal(row["p0_preflight_us"], f"{model}/{trial}/p0_preflight_us")
                + parse_decimal(row["p0_ot_setup_us"], f"{model}/{trial}/p0_ot_setup_us"),
                parse_decimal(row["p1_total_us"], f"{model}/{trial}/p1_total_us")
                + parse_decimal(row["p1_preflight_us"], f"{model}/{trial}/p1_preflight_us")
                + parse_decimal(row["p1_ot_setup_us"], f"{model}/{trial}/p1_ot_setup_us"))
                for row in retained), Decimal(0))
            full_ok = (int(model_meta["unsupported_convolution_layers"]) == 0
                       and int(model_meta["unsupported_truncation_layers"]) == 0)
            workload_ok = metadata["workload"] != "full-model" or full_ok
            result["execution_status"] = "pass"
            result["full_model_status"] = "pass" if full_ok else "FAIL_UNSUPPORTED"
            result["workload_status"] = "pass" if workload_ok else "FAIL"
            result["status"] = result["workload_status"]
            aggregates.append(result)

    measured_by_model = {
        model: [row for row in aggregates if row["model"] == model and row["sample_role"] == "measured"]
        for model in model_metadata
    }

    def required_values(measured, field):
        return [parse_decimal(str(row[field]), f"aggregate/{row['model']}/{row['trial']}/{field}") for row in measured]

    def optional_values(measured, field):
        values = [row[field] for row in measured]
        available = [value not in MISSING for value in values]
        if any(available) and not all(available):
            raise SystemExit(f"partially available measured metric {field}")
        return [] if not any(available) else [parse_decimal(str(value), f"statistics/{field}") for value in values]

    def model_metrics(measured):
        metrics = {
            "party0_preprocess_us": required_values(measured, "p0_total_us_total"),
            "party1_preprocess_us": required_values(measured, "p1_total_us_total"),
            "critical_path_preprocess_us": required_values(measured, "critical_path_preprocess_us_total"),
            "public_a_seed_words_total": [left + right for left, right in zip(
                required_values(measured, "p0_public_a_seed_words_total"),
                required_values(measured, "p1_public_a_seed_words_total"))],
            "application_bytes_total": [left + right for left, right in zip(
                required_values(measured, "p0_protocol_bytes_total"),
                required_values(measured, "p1_protocol_bytes_total"))],
            "matched_dealer_keygen_us": required_values(measured, "matched_dealer_keygen_us_total"),
            "checker_two_share_online_us": required_values(measured, "checker_two_share_online_us_total"),
        }
        dealer = metrics["matched_dealer_keygen_us"]
        if any(value == 0 for value in dealer):
            raise SystemExit("matched dealer keygen time is zero; ratio is undefined")
        metrics["preprocess_over_matched_dealer_ratio"] = [
            preprocess / baseline for preprocess, baseline in zip(metrics["critical_path_preprocess_us"], dealer)
        ]
        metrics["critical_path_setup_included_us"] = required_values(
            measured, "critical_path_setup_included_us_total")
        metrics["setup_included_preprocess_over_matched_dealer_ratio"] = [
            preprocess / baseline for preprocess, baseline in zip(
                metrics["critical_path_setup_included_us"], dealer)
        ]
        metrics["protocol_dependency_rounds"] = optional_values(measured, "critical_path_protocol_dependency_rounds_total")
        for stage in (
            "preflight_us", "ot_setup_us", "dpf_phase_a_us", "dpf_phase_b_us",
            "dpf_phase_c_us", "spfss_grouping_us", "public_polynomial_exchange_us",
            "gpu_ringlpn_expansion_us", "derandomization_openings_us", "conversion_us",
            "serialization_us", "commit_us",
        ):
            metrics[f"critical_path_{stage}"] = optional_values(measured, f"critical_path_{stage}_total")
        for output_name, left, right, operation in (
            ("peak_host_rss_bytes", "p0_peak_host_rss_bytes_max", "p1_peak_host_rss_bytes_max", max),
            ("peak_gpu_bytes", "p0_peak_gpu_bytes_max", "p1_peak_gpu_bytes_max", max),
            ("minimum_observed_gpu_free_bytes", "p0_min_gpu_free_bytes_min", "p1_min_gpu_free_bytes_min", min),
        ):
            left_values, right_values = optional_values(measured, left), optional_values(measured, right)
            if bool(left_values) != bool(right_values):
                raise SystemExit(f"one-sided availability for measured metric {output_name}")
            metrics[output_name] = [operation(a, b) for a, b in zip(left_values, right_values)]
        transport_fields = (
            "p0_transport_straight_bytes_sent_total", "p0_transport_reversed_bytes_sent_total",
            "p1_transport_straight_bytes_sent_total", "p1_transport_reversed_bytes_sent_total",
        )
        transport_columns = [optional_values(measured, field) for field in transport_fields]
        if any(transport_columns) and not all(transport_columns):
            raise SystemExit("partial transport metric availability")
        metrics["transport_bytes_total_including_base_ot"] = [sum(values, Decimal(0)) for values in zip(*transport_columns)] if all(transport_columns) else []
        base_left = optional_values(measured, "p0_base_ot_setup_bytes_sent_total")
        base_right = optional_values(measured, "p1_base_ot_setup_bytes_sent_total")
        if bool(base_left) != bool(base_right):
            raise SystemExit("one-sided base-OT metric availability")
        metrics["base_ot_setup_bytes_total"] = [a + b for a, b in zip(base_left, base_right)]
        metrics["checker_us"] = optional_values(measured, "checker_us_total")
        metrics["checker_peak_host_rss_bytes"] = optional_values(measured, "checker_peak_host_rss_bytes_max")
        metrics["checker_peak_gpu_bytes"] = optional_values(measured, "checker_peak_gpu_bytes_max")
        metrics["checker_min_gpu_free_bytes"] = optional_values(measured, "checker_min_gpu_free_bytes_min")
        return metrics

    units = {
        "application_bytes_total": "bytes", "public_a_seed_words_total": "uint64_words",
        "preprocess_over_matched_dealer_ratio": "ratio", "protocol_dependency_rounds": "rounds",
        "setup_included_preprocess_over_matched_dealer_ratio": "ratio",
        "peak_host_rss_bytes": "bytes", "peak_gpu_bytes": "bytes",
        "minimum_observed_gpu_free_bytes": "bytes",
        "transport_bytes_total_including_base_ot": "bytes", "base_ot_setup_bytes_total": "bytes",
        "checker_peak_host_rss_bytes": "bytes", "checker_peak_gpu_bytes": "bytes",
        "checker_min_gpu_free_bytes": "bytes",
    }
    statistics_fields = [
        "schema_version", *IDENTITY_FIELDS, "model", "workload", "population",
        "metric", "n", "mean", "sample_stdev", "median", "q1", "q3", "iqr",
        "mean_ci95_low", "mean_ci95_high", "mean_ci95_method", "quartile_method",
        "min", "max", "unit",
    ]
    statistics_rows = []
    for model, model_meta in sorted(model_metadata.items(), key=lambda item: int(item[1]["model_order"])):
        measured = measured_by_model[model]
        if len(measured) != args.trials:
            raise SystemExit(f"incomplete measured aggregate population for {model}")
        for name, values in model_metrics(measured).items():
            if values and len(values) != args.trials:
                raise SystemExit(f"incomplete measured statistic {model}/{name}")
            statistics_rows.append({
                "schema_version": STATISTICS_SCHEMA_VERSION, **identity,
                "model": model, "workload": metadata["workload"],
                "population": "passing measured aggregate rows only; warmups excluded",
                "metric": name, "n": len(values), **summarize(values),
                "mean_ci95_method": INTERVAL_METHOD if len(values) >= 2 else "undefined for n<2",
                "quartile_method": QUARTILE_METHOD,
                "unit": units.get(name, "us"),
            })

    atomic_write_csv(args.aggregate_csv, aggregate_fields, aggregates)
    atomic_write_csv(args.statistics_csv, statistics_fields, statistics_rows)


if __name__ == "__main__":
    main()
