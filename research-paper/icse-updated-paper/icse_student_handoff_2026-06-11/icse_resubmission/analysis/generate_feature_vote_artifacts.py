#!/usr/bin/env python3

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd
import psycopg
from scipy.stats import mannwhitneyu


FEATURES = [
    ("has_flop_division", "Has FLOP Division"),
    ("has_common_float_subexpr", "Has Float Subexprs"),
    ("has_special_math_functions", "Has Special Math"),
    ("has_loop_invariant_flops", "Has Loop-Invariant FLOPs"),
    ("calls_device_function", "Calls Helper Function"),
    ("has_rng_input_data", "Has RNG Inputs"),
    ("has_branching", "Has Branching"),
    ("uses_preprocessor_defines", "Uses Preproc Defines"),
    ("has_data_dependent_branching", "Has Data-Dep Branching"),
    ("reads_input_values_from_file", "Uses Input Data from File"),
    ("has_constant_propagatable_gridsz", "Deterministic Grid Size"),
    ("has_constant_propagatable_blocksz", "Deterministic Block Size"),
]
FEATURE_NAMES = [feature_name for feature_name, _ in FEATURES]
FEATURE_LABELS = dict(FEATURES)

THREAD_PATTERN = re.compile(
    r"_(?P<gpu>A100|3080|H100|A10)_(?P<safe_model>.+)_(?P<config>withsass|nosass|sass_imix|sass_noimix|nosass_imix|nosass_noimix)_trial(?P<trial>\d+)(?:_DRYRUN(?:\d+)?)?$"
)
PROMPT_LABELS = {
    (False, False): "Source-Only",
    (True, False): "Source+SASS",
}
MODEL_LABELS = {
    "anthropic/claude-4.6-opus": "Opus 4.6",
    "anthropic/claude-opus-4.6": "Opus 4.6",
    "openai/gpt-5.4": "GPT 5.4",
    "openai/gpt-oss-120b": "GPT OSS",
}


def _normalize_value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode()
    return value


def _load_tail_checkpoints(db_uri: str) -> dict[str, dict]:
    with psycopg.connect(db_uri) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint FROM checkpoints"
            )
            rows = cur.fetchall()

    checkpoints_by_thread: dict[str, list[dict]] = defaultdict(list)
    for thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint in rows:
        normalized_thread_id = _normalize_value(thread_id)
        checkpoints_by_thread[normalized_thread_id].append(
            {
                "thread_id": normalized_thread_id,
                "checkpoint_ns": _normalize_value(checkpoint_ns),
                "checkpoint_id": _normalize_value(checkpoint_id),
                "parent_checkpoint_id": _normalize_value(parent_checkpoint_id),
                "checkpoint": checkpoint,
            }
        )

    tails: dict[str, dict] = {}
    for thread_id, thread_checkpoints in checkpoints_by_thread.items():
        children_by_parent: dict[str | None, list[dict]] = defaultdict(list)
        roots: list[dict] = []
        for checkpoint in thread_checkpoints:
            parent_checkpoint_id = checkpoint["parent_checkpoint_id"]
            if parent_checkpoint_id is None:
                roots.append(checkpoint)
            else:
                children_by_parent[parent_checkpoint_id].append(checkpoint)

        if len(roots) != 1:
            continue

        current = roots[0]
        visited_ids: set[str] = set()
        while True:
            checkpoint_id = current["checkpoint_id"]
            if checkpoint_id in visited_ids:
                break
            visited_ids.add(checkpoint_id)

            children = children_by_parent.get(checkpoint_id, [])
            if not children:
                tails[thread_id] = current
                break
            if len(children) != 1:
                break
            current = children[0]

    return tails


def _load_blob_map(db_uri: str, channels: list[str]) -> dict[tuple[str, str, str, str], object]:
    with psycopg.connect(db_uri) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, checkpoint_ns, channel, version, blob FROM checkpoint_blobs WHERE channel = ANY(%s)",
                (channels,),
            )
            rows = cur.fetchall()

    return {
        (
            _normalize_value(thread_id),
            _normalize_value(checkpoint_ns),
            _normalize_value(channel),
            _normalize_value(version),
        ): msgpack.unpackb(bytes(blob), raw=False, strict_map_key=False)
        for thread_id, checkpoint_ns, channel, version, blob in rows
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _abs_pct_error(predicted_value: float, expected_value: float) -> float:
    if expected_value == 0.0:
        return 0.0 if predicted_value == 0.0 else float("inf")
    return abs(predicted_value - expected_value) / abs(expected_value) * 100.0


def _cliffs_delta(present_values: np.ndarray, absent_values: np.ndarray) -> float:
    if present_values.size == 0 or absent_values.size == 0:
        return float("nan")

    sorted_absent = np.sort(absent_values)
    win_counts = np.searchsorted(sorted_absent, present_values, side="left")
    loss_counts = absent_values.size - np.searchsorted(sorted_absent, present_values, side="right")
    return float((win_counts.sum() - loss_counts.sum()) / (present_values.size * absent_values.size))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    ranked_order = np.argsort(p_values)
    ranked_values = p_values[ranked_order]
    adjusted_ranked = np.empty_like(ranked_values)
    running_min = 1.0
    total = len(ranked_values)
    for index in range(total - 1, -1, -1):
        rank = index + 1
        running_min = min(running_min, ranked_values[index] * total / rank)
        adjusted_ranked[index] = running_min

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[ranked_order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def _normalized_model_label(thread_id: str) -> tuple[str, bool, bool] | None:
    match = THREAD_PATTERN.search(thread_id)
    if match is None:
        return None

    config = match.group("config")
    if config == "withsass":
        use_sass, use_imix = True, True
    elif config == "nosass":
        use_sass, use_imix = False, False
    else:
        use_sass = config.startswith("sass_")
        use_imix = config.endswith("_imix")

    safe_model_name = re.sub(r"-\d{8}$", "", match.group("safe_model")).replace("_", "/")
    return MODEL_LABELS.get(safe_model_name, safe_model_name), use_sass, use_imix


def load_kernel_feature_votes(code_features_db_uri: str) -> pd.DataFrame:
    tails = _load_tail_checkpoints(code_features_db_uri)

    records: list[dict] = []
    for thread_id, tail in tails.items():
        channel_values = tail["checkpoint"].get("channel_values", {})
        if "total_tokens" not in channel_values:
            continue

        record = {
            "thread_id": thread_id,
            "program_name": channel_values.get("program_name"),
            "kernel_mangled_name": channel_values.get("kernel_mangled_name"),
        }
        for feature_name in FEATURE_NAMES:
            record[feature_name] = channel_values.get(f"predicted_{feature_name}")
        records.append(record)

    vote_df = pd.DataFrame(records)
    if vote_df.empty:
        return vote_df

    aggregate_spec = {
        feature_name: (
            lambda series: series.dropna().astype(bool).mean() > 0.5 if series.dropna().shape[0] else pd.NA
        )
        for feature_name in FEATURE_NAMES
    }
    aggregated = (
        vote_df.groupby(["program_name", "kernel_mangled_name"], dropna=False)
        .agg(aggregate_spec)
        .reset_index()
    )
    return aggregated


def load_sample_error_rows(gpuflops_db_uri: str) -> pd.DataFrame:
    tails = _load_tail_checkpoints(gpuflops_db_uri)
    metrics_diff_blobs = _load_blob_map(gpuflops_db_uri, ["metrics_diff"])

    records: list[dict] = []
    for thread_id, tail in tails.items():
        channel_values = tail["checkpoint"].get("channel_values", {})
        if "total_tokens" not in channel_values:
            continue

        normalized_model = _normalized_model_label(thread_id)
        if normalized_model is None:
            continue

        model_label, use_sass, use_imix = normalized_model
        if use_imix:
            continue

        channel_versions = tail["checkpoint"].get("channel_versions", {})
        metrics_diff = metrics_diff_blobs[
            (
                thread_id,
                tail["checkpoint_ns"],
                "metrics_diff",
                _normalize_value(channel_versions["metrics_diff"]),
            )
        ]

        expected_read_bytes = float(channel_values["expected_read_bytes"])
        expected_write_bytes = float(channel_values["expected_write_bytes"])
        predicted_read_bytes = expected_read_bytes + float(metrics_diff["read_bytes"])
        predicted_write_bytes = expected_write_bytes + float(metrics_diff["write_bytes"])
        expected_total_bytes = expected_read_bytes + expected_write_bytes
        predicted_total_bytes = predicted_read_bytes + predicted_write_bytes

        for precision in ("fp16", "fp32", "fp64"):
            expected_flops = float(channel_values[f"expected_{precision}"])
            predicted_flops = expected_flops + float(metrics_diff[precision])
            expected_ai = _safe_divide(expected_flops, expected_total_bytes)
            predicted_ai = _safe_divide(predicted_flops, predicted_total_bytes)
            abs_error = (
                _abs_pct_error(predicted_ai, expected_ai)
                if math.isfinite(expected_ai) and math.isfinite(predicted_ai)
                else float("nan")
            )

            records.append(
                {
                    "program_name": channel_values.get("program_name"),
                    "kernel_mangled_name": channel_values.get("kernel_mangled_name"),
                    "model_label": model_label,
                    "prompt_type": PROMPT_LABELS[(use_sass, use_imix)],
                    "precision": precision,
                    "abs_ai_pct_error": abs_error,
                }
            )

    return pd.DataFrame(records)


def build_feature_support_tables(
    kernel_feature_df: pd.DataFrame,
    sample_error_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_df = sample_error_df.merge(
        kernel_feature_df,
        on=["program_name", "kernel_mangled_name"],
        how="left",
    )
    feature_long_df = merged_df.melt(
        id_vars=[
            "program_name",
            "kernel_mangled_name",
            "model_label",
            "prompt_type",
            "precision",
            "abs_ai_pct_error",
        ],
        value_vars=FEATURE_NAMES,
        var_name="feature_name",
        value_name="feature_present",
    )
    feature_long_df = feature_long_df[feature_long_df["feature_present"].notna()].copy()
    feature_long_df["feature_present"] = feature_long_df["feature_present"].astype(bool)

    model_prompt_records: list[dict] = []
    for (model_label, prompt_type, feature_name), group_df in feature_long_df.groupby(
        ["model_label", "prompt_type", "feature_name"],
        dropna=False,
    ):
        present_errors = (
            pd.to_numeric(group_df.loc[group_df["feature_present"], "abs_ai_pct_error"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        absent_errors = (
            pd.to_numeric(group_df.loc[~group_df["feature_present"], "abs_ai_pct_error"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        if len(present_errors) < 5 or len(absent_errors) < 5:
            continue

        p_value = mannwhitneyu(
            present_errors,
            absent_errors,
            alternative="two-sided",
            method="asymptotic",
        ).pvalue
        model_prompt_records.append(
            {
                "model_label": model_label,
                "prompt_type": prompt_type,
                "feature_name": feature_name,
                "feature_label": FEATURE_LABELS[feature_name],
                "n_present": int(len(present_errors)),
                "n_absent": int(len(absent_errors)),
                "association_score": _cliffs_delta(present_errors, absent_errors),
                "median_error_delta": float(np.median(present_errors) - np.median(absent_errors)),
                "p_value": float(p_value),
            }
        )

    model_prompt_df = pd.DataFrame(model_prompt_records)
    if model_prompt_df.empty:
        return model_prompt_df, pd.DataFrame()

    model_prompt_df["p_adj_bh"] = _benjamini_hochberg(model_prompt_df["p_value"].to_numpy())
    model_prompt_df["feature_name"] = pd.Categorical(
        model_prompt_df["feature_name"],
        categories=FEATURE_NAMES,
        ordered=True,
    )
    model_prompt_df = model_prompt_df.sort_values(
        ["model_label", "prompt_type", "feature_name"]
    ).reset_index(drop=True)

    total_kernel_count = int(len(kernel_feature_df))
    summary_records: list[dict] = []
    for feature_name, feature_label in FEATURES:
        feature_rows = model_prompt_df[model_prompt_df["feature_name"] == feature_name].copy()
        present_kernel_count = int(kernel_feature_df[feature_name].fillna(False).astype(bool).sum())
        summary_records.append(
            {
                "feature_name": feature_name,
                "feature_label": feature_label,
                "kernels_present": present_kernel_count,
                "kernel_pct": present_kernel_count / total_kernel_count * 100.0 if total_kernel_count else float("nan"),
                "positive_cells": int((feature_rows["association_score"] > 0.0).sum()),
                "positive_sig_cells": int(
                    (
                        (feature_rows["association_score"] > 0.0)
                        & (feature_rows["p_adj_bh"] < 0.05)
                    ).sum()
                ),
                "median_delta": float(feature_rows["association_score"].median()),
                "min_p_adj_bh": float(feature_rows["p_adj_bh"].min()),
                "max_p_adj_bh": float(feature_rows["p_adj_bh"].max()),
            }
        )

    summary_df = pd.DataFrame(summary_records).sort_values(
        ["positive_sig_cells", "median_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return model_prompt_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the appendix feature-vote support summaries from restored gpuflops_db and code_features_db instances."
    )
    parser.add_argument("--gpuflops-db-uri", required=True, help="PostgreSQL URI for gpuflops_db")
    parser.add_argument("--code-features-db-uri", required=True, help="PostgreSQL URI for code_features_db")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "generated"),
        help="Directory where CSV outputs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_feature_df = load_kernel_feature_votes(args.code_features_db_uri)
    sample_error_df = load_sample_error_rows(args.gpuflops_db_uri)
    model_prompt_df, summary_df = build_feature_support_tables(kernel_feature_df, sample_error_df)

    model_prompt_df.to_csv(output_dir / "feature_vote_model_prompt_stats.csv", index=False)
    summary_df.to_csv(output_dir / "feature_vote_feature_summary.csv", index=False)


if __name__ == "__main__":
    main()
