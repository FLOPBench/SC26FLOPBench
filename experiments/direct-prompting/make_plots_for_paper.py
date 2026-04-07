import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import importlib.util

import matplotlib
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)


def _load_module(module_name: str, relative_path: str):
	module_path = os.path.join(WORKSPACE_ROOT, relative_path)
	module_spec = importlib.util.spec_from_file_location(module_name, module_path)
	if module_spec is None or module_spec.loader is None:
		raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
	module = importlib.util.module_from_spec(module_spec)
	module_spec.loader.exec_module(module)
	return module


visualize_results = _load_module(
	"visualize_results",
	os.path.join("experiments", "direct-prompting", "visualize_results.py"),
)


AI_PRECISIONS = ["fp16", "fp32", "fp64"]
AI_LABELS = {
	"fp16": "FP16 RAI",
	"fp32": "FP32 RAI",
	"fp64": "FP64 RAI",
}
PRECISION_DISPLAY_LABELS = {
	"fp16": "FP16",
	"fp32": "FP32",
	"fp64": "FP64",
}
BOUND_LABELS = ["TP", "FP", "FN", "TN"]
SASS_PANEL_ORDER = [False, True]
SASS_PANEL_LABELS = {
	False: "Source-Only",
	True: "Source+SASS",
}
EXPECTED_RAI_DISTRIBUTION_COLUMNS = [
	"zero_rai_n",
	"nonzero_bandwidth_bound_n",
	"nonzero_compute_bound_n",
]
EXPECTED_RAI_DISTRIBUTION_LABELS = {
	"zero_rai_n": "0 RAI",
	"nonzero_bandwidth_bound_n": "Bandwidth-Bound (BB)",
	"nonzero_compute_bound_n": "Compute-Bound (CB)",
}
EXPECTED_RAI_DISTRIBUTION_COLORS = {
	"zero_rai_n": "#4c78a8",
	"nonzero_bandwidth_bound_n": "#f58518",
	"nonzero_compute_bound_n": "#54a24b",
}
RUNTIME_DISTRIBUTION_COLUMNS = ["cuda_n", "omp_n"]
RUNTIME_DISTRIBUTION_LABELS = {
	"cuda_n": "CUDA",
	"omp_n": "OpenMP",
}
RUNTIME_DISTRIBUTION_COLORS = {
	"cuda_n": "#4c78a8",
	"omp_n": "#e45756",
}
TOKEN_COLUMN_LABELS = {
	"input_tokens": "Input Tokens",
	"output_tokens": "Output Tokens",
}
POSITIVE_CLASS = "compute-bound"
NEGATIVE_CLASS = "bandwidth-bound"
BOUND_OUTCOME_LABELS = {
	"TP": "Expected Compute\nPredicted Compute",
	"FP": "Expected Bandwidth\nPredicted Compute",
	"FN": "Expected Compute\nPredicted Bandwidth",
	"TN": "Expected Bandwidth\nPredicted Bandwidth",
}
BOUND_CLASS_ORDER = [NEGATIVE_CLASS, POSITIVE_CLASS]
BOUND_CLASS_DISPLAY = {
	NEGATIVE_CLASS: "BB",
	POSITIVE_CLASS: "CB",
}
DEFAULT_OUTPUT_DIR = os.path.join(
	WORKSPACE_ROOT,
	"experiments",
	"direct-prompting",
	"paper-figure-output",
)
FIGURE_SIZE_SCALE = 0.78
PAPER_LEGEND_RIGHT_MARGIN = 0.88
PAPER_LEGEND_X = 0.89
RAI_X_TICKS = [-1e6, -1e4, -1e2, -1e0, -1e-2, 0.0, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
RAI_X_TICK_LABELS = [
	r"$-10^{6}$",
	r"$-10^{4}$",
	r"$-10^{2}$",
	r"$-10^{0}$",
	r"$-10^{-2}$",
	"0",
	r"$10^{-2}$",
	r"$10^{0}$",
	r"$10^{2}$",
	r"$10^{4}$",
	r"$10^{6}$",
	r"$10^{8}$",
	r"$10^{10}$",
]

GPU_ROOFLINE_TABLE = {
	"3080": {
		"display_name": "RTX 3080 (PCIe)",
		"memory_bandwidth_gb_per_s": 760.0,
		"peak_tflops": {
			"fp16": 30.55,
			"fp32": 30.55,
			"fp64": 0.477,
		},
	},
	"A10": {
		"display_name": "A10 (PCIe)",
		"memory_bandwidth_gb_per_s": 600.0,
		"peak_tflops": {
			"fp16": 15.62,
			"fp32": 15.62,
			"fp64": 0.244,
		},
	},
	"A100": {
		"display_name": "A100 (SXM4)",
		"memory_bandwidth_gb_per_s": 1555.0,
		"peak_tflops": {
			"fp16": 77.97,
			"fp32": 19.49,
			"fp64": 9.75,
		},
	},
	"H100": {
		"display_name": "H100 (SXM5)",
		"memory_bandwidth_gb_per_s": 3360.0,
		"peak_tflops": {
			"fp16": 133.82,
			"fp32": 66.91,
			"fp64": 33.45,
		},
	},
}
RUNTIME_DISPLAY_LABELS = {
	"cuda": "CUDA",
	"omp": "OpenMP",
}


def _bounded_plot_height(
	item_count: int,
	*,
	min_height: float,
	per_item: float,
	padding: float,
	max_height: float,
) -> float:
	return min(max_height, max(min_height, per_item * max(item_count, 1) + padding))


def _scaled_figsize(width: float, height: float) -> tuple[float, float]:
	return (width * FIGURE_SIZE_SCALE, height * FIGURE_SIZE_SCALE)


def _set_rai_symlog_ticks(axis: plt.Axes) -> None:
	axis.set_xlim(RAI_X_TICKS[0], RAI_X_TICKS[-1])
	axis.xaxis.set_major_locator(mticker.FixedLocator(RAI_X_TICKS))
	axis.xaxis.set_major_formatter(mticker.FixedFormatter(RAI_X_TICK_LABELS))
	axis.xaxis.set_minor_locator(mticker.NullLocator())


def _safe_divide(numerator: Any, denominator: Any) -> float:
	numerator_value = pd.to_numeric(numerator, errors="coerce")
	denominator_value = pd.to_numeric(denominator, errors="coerce")
	if pd.isna(numerator_value) or pd.isna(denominator_value):
		return float("nan")
	if float(denominator_value) == 0.0:
		return float("nan")
	return float(numerator_value) / float(denominator_value)


def _balance_point(peak_tflops: float, bandwidth_gb_per_s: float) -> float:
	return float(peak_tflops * 1000.0 / bandwidth_gb_per_s)


def _symlog_linthresh(values: pd.Series) -> float:
	numeric_values = pd.to_numeric(values, errors="coerce")
	finite_values = numeric_values[np.isfinite(numeric_values.to_numpy())]
	if finite_values.empty:
		return 1.0
	positive_magnitudes = finite_values.abs()
	positive_magnitudes = positive_magnitudes[positive_magnitudes > 0.0]
	if positive_magnitudes.empty:
		return 1.0
	percentile_value = float(np.percentile(positive_magnitudes, 10.0))
	if percentile_value <= 0.0 or not math.isfinite(percentile_value):
		return 1.0
	return percentile_value


def _classify_bound(ai_value: Any, balance_point: Any) -> str:
	ai_numeric = pd.to_numeric(ai_value, errors="coerce")
	balance_numeric = pd.to_numeric(balance_point, errors="coerce")
	if pd.isna(ai_numeric) or pd.isna(balance_numeric):
		return NEGATIVE_CLASS
	return NEGATIVE_CLASS if float(ai_numeric) <= float(balance_numeric) else POSITIVE_CLASS


def _is_nonzero_expected_ai(value: Any) -> bool:
	ai_numeric = pd.to_numeric(value, errors="coerce")
	return not pd.isna(ai_numeric) and math.isfinite(float(ai_numeric)) and float(ai_numeric) > 0.0


def _print_roofline_specs() -> None:
	print("GPU roofline specs used for paper plots:")
	for gpu_name, spec in GPU_ROOFLINE_TABLE.items():
		bandwidth = spec["memory_bandwidth_gb_per_s"]
		print(f"- {gpu_name}: {spec['display_name']}")
		print(f"  DRAM bandwidth: {bandwidth:.2f} GB/s")
		for precision in AI_PRECISIONS:
			peak_tflops = spec["peak_tflops"][precision]
			balance = _balance_point(peak_tflops, bandwidth)
			print(
				f"  {precision.upper()} peak: {peak_tflops:.3f} TFLOP/s | "
				f"balance point: {balance:.2f} FLOP/byte"
			)


def _load_samples_dataframe(db_uri: str, include_dry_run: bool) -> pd.DataFrame:
	parser = visualize_results.CheckpointDBParser(db_uri)
	attempt_tracker = visualize_results.QueryAttemptTracker(db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoint_result = parser.fetch_tail_checkpoints_by_thread(
			checkpoints=checkpoints,
			tolerate_errors=True,
		)
		tail_checkpoints = tail_checkpoint_result["tails"]
		invalid_threads = tail_checkpoint_result["invalid_threads"]
		for checkpoint in tail_checkpoints.values():
			channel_values = checkpoint["checkpoint"]["channel_values"]
			if "total_tokens" in channel_values:
				parser.hydrate_checkpoint_channels(
					checkpoint,
					["prediction", "metrics_diff", "metrics_pct_diff", "metrics_explanations"],
				)
		attempts = attempt_tracker.fetch_all_attempts()
	finally:
		parser.close()
		attempt_tracker.close()

	visualize_results._print_invalid_thread_warnings(invalid_threads)
	stored_thread_ids = visualize_results._stored_thread_ids(checkpoints, attempts)
	if not stored_thread_ids:
		raise RuntimeError("No checkpoint or query-attempt records were found in the database.")

	if not include_dry_run:
		non_dry_run_thread_ids = {
			thread_id for thread_id in stored_thread_ids if not visualize_results._is_dry_run_thread(thread_id)
		}
		if not non_dry_run_thread_ids:
			raise RuntimeError(
				"The database currently contains only dry-run thread IDs. "
				"Re-run with --includeDryRun or populate the database with non-dry experiment runs."
			)

	return visualize_results._database_dataframe(tail_checkpoints, attempts, include_dry_run)


def _enrich_completed_dataframe(samples_df: pd.DataFrame) -> pd.DataFrame:
	completed_df = samples_df[samples_df["status"] == "completed"].copy()
	if completed_df.empty:
		return completed_df

	completed_df["use_sass"] = completed_df["use_sass"].fillna(False).astype(bool)
	completed_df["use_imix"] = completed_df["use_imix"].fillna(False).astype(bool)

	for metric_name in [
		"expected_fp16",
		"expected_fp32",
		"expected_fp64",
		"expected_read_bytes",
		"expected_write_bytes",
		"metrics_diff_fp16",
		"metrics_diff_fp32",
		"metrics_diff_fp64",
		"metrics_diff_read_bytes",
		"metrics_diff_write_bytes",
	]:
		completed_df[metric_name] = pd.to_numeric(completed_df[metric_name], errors="coerce")

	completed_df["predicted_fp16"] = completed_df["expected_fp16"] + completed_df["metrics_diff_fp16"]
	completed_df["predicted_fp32"] = completed_df["expected_fp32"] + completed_df["metrics_diff_fp32"]
	completed_df["predicted_fp64"] = completed_df["expected_fp64"] + completed_df["metrics_diff_fp64"]
	completed_df["predicted_read_bytes"] = completed_df["expected_read_bytes"] + completed_df["metrics_diff_read_bytes"]
	completed_df["predicted_write_bytes"] = completed_df["expected_write_bytes"] + completed_df["metrics_diff_write_bytes"]
	completed_df["expected_total_bytes"] = completed_df["expected_read_bytes"] + completed_df["expected_write_bytes"]
	completed_df["predicted_total_bytes"] = completed_df["predicted_read_bytes"] + completed_df["predicted_write_bytes"]

	for precision in AI_PRECISIONS:
		completed_df[f"balance_point_{precision}"] = completed_df["gpu"].map(
			lambda gpu_name: _balance_point(
				GPU_ROOFLINE_TABLE[gpu_name]["peak_tflops"][precision],
				GPU_ROOFLINE_TABLE[gpu_name]["memory_bandwidth_gb_per_s"],
			)
			if gpu_name in GPU_ROOFLINE_TABLE
			else float("nan")
		)
		completed_df[f"expected_ai_{precision}"] = completed_df.apply(
			lambda row: _safe_divide(row[f"expected_{precision}"], row["expected_total_bytes"]),
			axis=1,
		)
		completed_df[f"predicted_ai_{precision}"] = completed_df.apply(
			lambda row: _safe_divide(row[f"predicted_{precision}"], row["predicted_total_bytes"]),
			axis=1,
		)
		completed_df[f"ai_diff_{precision}"] = (
			completed_df[f"predicted_ai_{precision}"] - completed_df[f"expected_ai_{precision}"]
		)
		completed_df[f"expected_bound_{precision}"] = completed_df.apply(
			lambda row: _classify_bound(row[f"expected_ai_{precision}"], row[f"balance_point_{precision}"]),
			axis=1,
		)
		completed_df[f"predicted_bound_{precision}"] = completed_df.apply(
			lambda row: _classify_bound(row[f"predicted_ai_{precision}"], row[f"balance_point_{precision}"]),
			axis=1,
		)

	return completed_df


def _paper_subset(completed_df: pd.DataFrame) -> pd.DataFrame:
	subset = completed_df[(completed_df["use_imix"] == False) & (completed_df["use_sass"].isin(SASS_PANEL_ORDER))].copy()
	if subset.empty:
		return subset
	return subset.sort_values(["model_name", "use_sass", "thread_id"]).reset_index(drop=True)


def _print_bound_class_distribution(plot_df: pd.DataFrame) -> None:
	print("\nExpected bound-class distribution for nonzero expected RAI cases used in paper figures:")
	for precision in AI_PRECISIONS:
		expected_ai_column = f"expected_ai_{precision}"
		expected_bound_column = f"expected_bound_{precision}"
		precision_mask = plot_df[expected_ai_column].apply(_is_nonzero_expected_ai)
		precision_df = plot_df.loc[precision_mask, ["gpu", expected_bound_column]].copy()
		print(f"- {precision.upper()}")
		if precision_df.empty:
			print("  No nonzero expected-RAI samples")
			continue
		overall_counts = precision_df[expected_bound_column].value_counts().to_dict()
		print(f"  Overall: {overall_counts}")
		gpu_counts = (
			precision_df.groupby(["gpu", expected_bound_column]).size().unstack(fill_value=0)
		)
		for gpu_name in sorted(gpu_counts.index.tolist()):
			bandwidth_count = int(gpu_counts.loc[gpu_name].get(NEGATIVE_CLASS, 0))
			compute_count = int(gpu_counts.loc[gpu_name].get(POSITIVE_CLASS, 0))
			total = bandwidth_count + compute_count
			if total <= 0:
				continue
			print(
				f"  {gpu_name}: bandwidth-bound={bandwidth_count} ({bandwidth_count / total * 100.0:.1f}%), "
				f"compute-bound={compute_count} ({compute_count / total * 100.0:.1f}%), total={total}"
			)


def _expected_rai_distribution_category(ai_value: Any, balance_point: Any) -> str:
	ai_numeric = pd.to_numeric(ai_value, errors="coerce")
	balance_numeric = pd.to_numeric(balance_point, errors="coerce")
	if pd.isna(ai_numeric) or not math.isfinite(float(ai_numeric)):
		return "nan_rai_n"
	if float(ai_numeric) == 0.0:
		return "zero_rai_n"
	if pd.isna(balance_numeric) or not math.isfinite(float(balance_numeric)):
		return "nan_rai_n"
	if float(ai_numeric) <= float(balance_numeric):
		return "nonzero_bandwidth_bound_n"
	return "nonzero_compute_bound_n"


def _summarize_expected_rai_distribution(plot_df: pd.DataFrame) -> pd.DataFrame:
	base_columns = ["program_name", "kernel_mangled_name", "gpu"]
	result_columns = [
		"gpu",
		"precision",
		*EXPECTED_RAI_DISTRIBUTION_COLUMNS,
		"nonzero_rai_n",
		"total_kernels",
		"count_string",
	]
	if plot_df.empty:
		return pd.DataFrame(columns=result_columns)

	value_columns = []
	for precision in AI_PRECISIONS:
		value_columns.extend([f"expected_ai_{precision}", f"balance_point_{precision}"])

	unique_samples_df = (
		plot_df[base_columns + value_columns]
		.drop_duplicates(subset=base_columns)
		.reset_index(drop=True)
	)

	rows: List[Dict[str, Any]] = []
	for _, row in unique_samples_df.iterrows():
		for precision in AI_PRECISIONS:
			rows.append(
				{
					"gpu": row["gpu"],
					"precision": PRECISION_DISPLAY_LABELS[precision],
					"rai_category": _expected_rai_distribution_category(
						row[f"expected_ai_{precision}"],
						row[f"balance_point_{precision}"],
					),
				}
			)

	distribution_long_df = pd.DataFrame(rows, columns=["gpu", "precision", "rai_category"])
	if distribution_long_df.empty:
		return pd.DataFrame(columns=result_columns)

	distribution_df = (
		distribution_long_df.groupby(["gpu", "precision", "rai_category"], dropna=False)
		.size()
		.unstack(fill_value=0)
		.reset_index()
	)
	for column in EXPECTED_RAI_DISTRIBUTION_COLUMNS:
		if column not in distribution_df.columns:
			distribution_df[column] = 0
	if "nan_rai_n" in distribution_df.columns:
		nan_count = int(pd.to_numeric(distribution_df["nan_rai_n"], errors="coerce").fillna(0).sum())
		if nan_count > 0:
			raise RuntimeError(
				"Figure 6 encountered expected-RAI NaN samples, but the NaN bucket is intentionally omitted. "
				"Inspect expected_total_bytes for zero-denominator cases before plotting."
			)

	distribution_df["nonzero_rai_n"] = (
		distribution_df["nonzero_bandwidth_bound_n"] + distribution_df["nonzero_compute_bound_n"]
	)
	distribution_df["total_kernels"] = distribution_df[
		EXPECTED_RAI_DISTRIBUTION_COLUMNS
	].sum(axis=1)
	distribution_df["count_string"] = distribution_df.apply(
		lambda row: (
			f"({int(row['zero_rai_n'])}|{int(row['nonzero_bandwidth_bound_n'])}|"
			f"{int(row['nonzero_compute_bound_n'])})"
		),
		axis=1,
	)

	gpu_order = {gpu_name: index for index, gpu_name in enumerate(GPU_ROOFLINE_TABLE.keys())}
	precision_order = {
		PRECISION_DISPLAY_LABELS[precision]: index for index, precision in enumerate(AI_PRECISIONS)
	}
	distribution_df["gpu_sort"] = distribution_df["gpu"].map(
		lambda gpu_name: gpu_order.get(gpu_name, len(gpu_order))
	)
	distribution_df["precision_sort"] = distribution_df["precision"].map(
		lambda precision_label: precision_order.get(precision_label, len(precision_order))
	)
	distribution_df = distribution_df.sort_values(
		["gpu_sort", "gpu", "precision_sort", "precision"],
		kind="stable",
	).reset_index(drop=True)
	return distribution_df[result_columns]


def _summarize_runtime_distribution(plot_df: pd.DataFrame) -> pd.DataFrame:
	result_columns = [
		"gpu",
		"precision",
		*RUNTIME_DISTRIBUTION_COLUMNS,
		"total_kernels",
		"count_string",
	]
	if plot_df.empty:
		return pd.DataFrame(columns=result_columns)

	unique_samples_df = (
		plot_df[["program_name", "kernel_mangled_name", "runtime"]]
		.dropna(subset=["program_name", "kernel_mangled_name", "runtime"])
		.drop_duplicates()
		.reset_index(drop=True)
	)
	if unique_samples_df.empty:
		return pd.DataFrame(columns=result_columns)

	runtime_df = (
		unique_samples_df.groupby(["runtime"], dropna=False)
		.size()
		.to_dict()
	)
	runtime_row = {
		"gpu": "All GPUs",
		"precision": "Runtime",
		"cuda_n": int(runtime_df.get("cuda", 0)),
		"omp_n": int(runtime_df.get("omp", 0)),
	}
	runtime_row["total_kernels"] = runtime_row["cuda_n"] + runtime_row["omp_n"]
	runtime_row["count_string"] = f"({runtime_row['cuda_n']}|{runtime_row['omp_n']})"
	return pd.DataFrame([runtime_row], columns=result_columns)


def _summarize_gpu_kernel_sample_coverage(plot_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
	identity_columns = ["program_name", "kernel_mangled_name", "gpu"]
	result = {
		"per_gpu": pd.DataFrame(columns=["gpu", "kernel_identity_n"]),
		"overlap_distribution": pd.DataFrame(columns=["gpu_overlap_n", "kernel_identity_n"]),
		"unique_per_gpu": pd.DataFrame(columns=["gpu", "unique_kernel_identity_n"]),
		"totals": pd.DataFrame(columns=["metric", "value"]),
	}
	if plot_df.empty:
		return result

	unique_samples_df = plot_df[identity_columns].dropna().drop_duplicates().reset_index(drop=True)
	if unique_samples_df.empty:
		return result

	gpu_order = {gpu_name: index for index, gpu_name in enumerate(GPU_ROOFLINE_TABLE.keys())}
	available_gpus = sorted(unique_samples_df["gpu"].unique().tolist(), key=lambda gpu: gpu_order.get(gpu, len(gpu_order)))
	identity_membership_df = (
		unique_samples_df.groupby(["program_name", "kernel_mangled_name"], dropna=False)["gpu"]
		.agg(lambda values: tuple(sorted(set(values), key=lambda gpu: gpu_order.get(gpu, len(gpu_order)))))
		.reset_index(name="gpu_membership")
	)
	identity_membership_df["gpu_overlap_n"] = identity_membership_df["gpu_membership"].map(len)

	per_gpu_df = (
		unique_samples_df.groupby("gpu", dropna=False)
		.size()
		.reset_index(name="kernel_identity_n")
	)
	per_gpu_df["gpu_sort"] = per_gpu_df["gpu"].map(lambda gpu: gpu_order.get(gpu, len(gpu_order)))
	per_gpu_df = per_gpu_df.sort_values(["gpu_sort", "gpu"]).drop(columns="gpu_sort").reset_index(drop=True)

	overlap_distribution_df = (
		identity_membership_df.groupby("gpu_overlap_n", dropna=False)
		.size()
		.reset_index(name="kernel_identity_n")
		.sort_values("gpu_overlap_n")
		.reset_index(drop=True)
	)

	unique_per_gpu_rows: List[Dict[str, Any]] = []
	for gpu_name in available_gpus:
		unique_count = int(
			identity_membership_df[identity_membership_df["gpu_membership"] == (gpu_name,)].shape[0]
		)
		unique_per_gpu_rows.append({
			"gpu": gpu_name,
			"unique_kernel_identity_n": unique_count,
		})
	unique_per_gpu_df = pd.DataFrame(unique_per_gpu_rows)

	all_gpu_overlap_n = int(
		identity_membership_df[identity_membership_df["gpu_overlap_n"] == len(available_gpus)].shape[0]
	)
	totals_df = pd.DataFrame(
		[
			{"metric": "distinct_gpu_n", "value": len(available_gpus)},
			{"metric": "union_kernel_identity_n", "value": int(identity_membership_df.shape[0])},
			{"metric": "all_gpu_overlap_kernel_identity_n", "value": all_gpu_overlap_n},
		]
	)

	return {
		"per_gpu": per_gpu_df,
		"overlap_distribution": overlap_distribution_df,
		"unique_per_gpu": unique_per_gpu_df,
		"totals": totals_df,
	}


def _print_gpu_kernel_sample_coverage_summary(plot_df: pd.DataFrame, title_suffix: str) -> None:
	summary = _summarize_gpu_kernel_sample_coverage(plot_df)
	print(f"\nLLM query sample coverage summary ({title_suffix}):")
	_print_summary_table("Distinct (program_name, kernel_mangled_name) per GPU", summary["per_gpu"])
	_print_summary_table("Overlap distribution by GPU coverage count", summary["overlap_distribution"])
	_print_summary_table("GPU-unique (program_name, kernel_mangled_name) counts", summary["unique_per_gpu"])
	_print_summary_table("Coverage totals", summary["totals"])


def _prepare_ai_long_df(completed_df: pd.DataFrame) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	for _, row in completed_df.iterrows():
		for precision in AI_PRECISIONS:
			expected_ai = pd.to_numeric(row[f"expected_ai_{precision}"], errors="coerce")
			if not _is_nonzero_expected_ai(expected_ai):
				continue
			ai_diff = pd.to_numeric(row[f"ai_diff_{precision}"], errors="coerce")
			if pd.isna(ai_diff) or not math.isfinite(float(ai_diff)):
				continue
			rows.append(
				{
					"model_name": row["model_name"],
					"gpu": row["gpu"],
					"runtime": row["runtime"],
					"use_sass": bool(row["use_sass"]),
					"precision": AI_LABELS[precision],
					"ai_diff": float(ai_diff),
				}
			)
	return pd.DataFrame(rows, columns=["model_name", "gpu", "runtime", "use_sass", "precision", "ai_diff"])


def _prepare_token_long_df(completed_df: pd.DataFrame) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	for _, row in completed_df.iterrows():
		for token_column, token_label in TOKEN_COLUMN_LABELS.items():
			token_value = pd.to_numeric(row.get(token_column), errors="coerce")
			if pd.isna(token_value) or not math.isfinite(float(token_value)):
				continue
			rows.append(
				{
					"model_name": row["model_name"],
					"use_sass": bool(row["use_sass"]),
					"token_type": token_label,
					"token_count": float(token_value),
				}
			)
	return pd.DataFrame(rows, columns=["model_name", "use_sass", "token_type", "token_count"])


def _token_axis_limits(token_long_df: pd.DataFrame) -> Dict[str, tuple[float, float]]:
	axis_limits: Dict[str, tuple[float, float]] = {}
	for token_label in TOKEN_COLUMN_LABELS.values():
		token_values = pd.to_numeric(
			token_long_df.loc[token_long_df["token_type"] == token_label, "token_count"],
			errors="coerce",
		).dropna()
		if token_values.empty:
			axis_limits[token_label] = (0.0, 1.0)
			continue
		max_value = float(token_values.max())
		padding = max(1.0, max_value * 0.02)
		axis_limits[token_label] = (0.0, max_value + padding)
	return axis_limits


def _ordered_columns(df: pd.DataFrame, preferred_columns: List[str]) -> List[str]:
	ordered = [column for column in preferred_columns if column in df.columns]
	remaining = [column for column in df.columns if column not in ordered]
	return ordered + remaining


def _summarize_ai_error(ai_long_df: pd.DataFrame, group_fields: List[str]) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	if ai_long_df.empty:
		return pd.DataFrame(
			columns=_ordered_columns(
				pd.DataFrame(columns=[]),
				group_fields
				+ [
					"use_sass",
					"sass_setting",
					"precision",
					"n",
					"median_ai_diff",
					"q1_ai_diff",
					"q3_ai_diff",
					"p10_ai_diff",
					"p90_ai_diff",
					"mean_ai_diff",
					"median_abs_ai_diff",
					"mean_abs_ai_diff",
				],
			),
		)

	summary_fields = group_fields + ["use_sass", "precision"]
	for group_key, group_df in ai_long_df.groupby(summary_fields, dropna=False):
		key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
		row: Dict[str, Any] = {}
		for field_name, field_value in zip(summary_fields, key_tuple):
			row[field_name] = field_value

		ai_diff = pd.to_numeric(group_df["ai_diff"], errors="coerce").dropna()
		if ai_diff.empty:
			continue
		abs_ai_diff = ai_diff.abs()
		row["sass_setting"] = SASS_PANEL_LABELS[bool(row["use_sass"])]
		row["n"] = int(ai_diff.shape[0])
		row["median_ai_diff"] = float(ai_diff.median())
		row["q1_ai_diff"] = float(ai_diff.quantile(0.25))
		row["q3_ai_diff"] = float(ai_diff.quantile(0.75))
		row["p10_ai_diff"] = float(ai_diff.quantile(0.10))
		row["p90_ai_diff"] = float(ai_diff.quantile(0.90))
		row["mean_ai_diff"] = float(ai_diff.mean())
		row["median_abs_ai_diff"] = float(abs_ai_diff.median())
		row["mean_abs_ai_diff"] = float(abs_ai_diff.mean())
		rows.append(row)

	result = pd.DataFrame(rows)
	if result.empty:
		return result
	result = result.sort_values(summary_fields).reset_index(drop=True)
	preferred_columns = _ordered_columns(
		result,
		group_fields
		+ [
			"use_sass",
			"sass_setting",
			"precision",
			"n",
			"median_ai_diff",
			"q1_ai_diff",
			"q3_ai_diff",
			"p10_ai_diff",
			"p90_ai_diff",
			"mean_ai_diff",
			"median_abs_ai_diff",
			"mean_abs_ai_diff",
		],
	)
	return result[preferred_columns]


def _summarize_bound_metrics(plot_df: pd.DataFrame, group_fields: List[str]) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	if plot_df.empty:
		return pd.DataFrame()

	summary_fields = group_fields + ["use_sass"]
	for group_key, group_df in plot_df.groupby(summary_fields, dropna=False):
		key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
		base_row: Dict[str, Any] = {}
		for field_name, field_value in zip(summary_fields, key_tuple):
			base_row[field_name] = field_value
		base_row["sass_setting"] = SASS_PANEL_LABELS[bool(base_row["use_sass"])]

		for precision in AI_PRECISIONS:
			expected_ai_column = f"expected_ai_{precision}"
			expected_column = f"expected_bound_{precision}"
			predicted_column = f"predicted_bound_{precision}"
			precision_df = group_df[
				[expected_ai_column, expected_column, predicted_column]
			].copy()
			precision_df = precision_df[
				precision_df[expected_ai_column].apply(_is_nonzero_expected_ai)
			].dropna(subset=[expected_column, predicted_column])
			if precision_df.empty:
				continue

			expected_series = precision_df[expected_column]
			predicted_series = precision_df[predicted_column]
			total_count = int(precision_df.shape[0])
			compute_count = int((expected_series == POSITIVE_CLASS).sum())
			bandwidth_count = int((expected_series == NEGATIVE_CLASS).sum())
			tp = int(((expected_series == POSITIVE_CLASS) & (predicted_series == POSITIVE_CLASS)).sum())
			tn = int(((expected_series == NEGATIVE_CLASS) & (predicted_series == NEGATIVE_CLASS)).sum())
			fp = int(((expected_series == NEGATIVE_CLASS) & (predicted_series == POSITIVE_CLASS)).sum())
			fn = int(((expected_series == POSITIVE_CLASS) & (predicted_series == NEGATIVE_CLASS)).sum())
			recall_compute = float(tp / compute_count) if compute_count > 0 else float("nan")
			recall_bandwidth = float(tn / bandwidth_count) if bandwidth_count > 0 else float("nan")
			accuracy = float((tp + tn) / total_count) if total_count > 0 else float("nan")
			balanced_accuracy = float((recall_compute + recall_bandwidth) / 2.0)

			row = dict(base_row)
			row["precision"] = AI_LABELS[precision]
			row["n"] = total_count
			row["compute_bound_n"] = compute_count
			row["bandwidth_bound_n"] = bandwidth_count
			row["tp"] = tp
			row["tn"] = tn
			row["fp"] = fp
			row["fn"] = fn
			row["accuracy"] = accuracy
			row["balanced_accuracy"] = balanced_accuracy
			row["recall_compute"] = recall_compute
			row["recall_bandwidth"] = recall_bandwidth
			rows.append(row)

	result = pd.DataFrame(rows)
	if result.empty:
		return result
	result = result.sort_values(summary_fields + ["precision"]).reset_index(drop=True)
	preferred_columns = _ordered_columns(
		result,
		group_fields
		+ [
			"use_sass",
			"sass_setting",
			"precision",
			"n",
			"compute_bound_n",
			"bandwidth_bound_n",
			"accuracy",
			"balanced_accuracy",
			"recall_compute",
			"recall_bandwidth",
			"tp",
			"tn",
			"fp",
			"fn",
		],
	)
	return result[preferred_columns]


def _write_summary_csv(df: pd.DataFrame, output_path: Path) -> None:
	if df.empty:
		pd.DataFrame().to_csv(output_path, index=False)
		return
	df.to_csv(output_path, index=False)


def _format_boxplot_summary_table(
	ai_long_df: pd.DataFrame,
	*,
	group_field: str,
	group_label: str,
	table_label: str,
) -> str:
	if ai_long_df.empty:
		return f"{table_label}\n(no plotted rows)"

	summary_df = ai_long_df[[group_field, "use_sass", "precision", "ai_diff"]].copy()
	summary_df["ai_diff"] = pd.to_numeric(summary_df["ai_diff"], errors="coerce")
	summary_df = summary_df[summary_df["ai_diff"].notna()].copy()
	if summary_df.empty:
		return f"{table_label}\n(no plotted rows)"

	group_order = sorted(summary_df[group_field].dropna().unique().tolist())
	prompt_order = [SASS_PANEL_LABELS[use_sass] for use_sass in SASS_PANEL_ORDER]
	precision_order = [AI_LABELS[precision] for precision in AI_PRECISIONS]

	summary_df["prompt_type"] = summary_df["use_sass"].map(lambda use_sass: SASS_PANEL_LABELS[bool(use_sass)])
	grouped = (
		summary_df.groupby([group_field, "prompt_type", "precision"], dropna=False)["ai_diff"]
		.agg(
			n="count",
			q1=lambda values: values.quantile(0.25),
			median="median",
			q3=lambda values: values.quantile(0.75),
		)
		.reset_index()
	)
	grouped["group_sort"] = grouped[group_field].map({name: index for index, name in enumerate(group_order)})
	grouped["prompt_sort"] = grouped["prompt_type"].map({name: index for index, name in enumerate(prompt_order)})
	grouped["precision_sort"] = grouped["precision"].map({name: index for index, name in enumerate(precision_order)})
	grouped["group_sort"] = grouped["group_sort"].fillna(len(group_order)).astype(int)
	grouped["prompt_sort"] = grouped["prompt_sort"].fillna(len(prompt_order)).astype(int)
	grouped["precision_sort"] = grouped["precision_sort"].fillna(len(precision_order)).astype(int)
	grouped = grouped.sort_values(
		by=["group_sort", "prompt_sort", "precision_sort", group_field, "prompt_type", "precision"],
		kind="stable",
	)

	rows = []
	for row in grouped.itertuples(index=False):
		rows.append(
			{
				group_label: str(getattr(row, group_field)),
				"Prompt Type": str(row.prompt_type),
				"Precision": str(row.precision),
				"N": str(int(row.n)),
				"Q1": f"{float(row.q1):.6g}",
				"Median": f"{float(row.median):.6g}",
				"Q3": f"{float(row.q3):.6g}",
			}
		)

	headers = [group_label, "Prompt Type", "Precision", "N", "Q1", "Median", "Q3"]
	widths = {
		header: max(len(header), *(len(row[header]) for row in rows))
		for header in headers
	}

	lines = [table_label]
	lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
	lines.append("-+-".join("-" * widths[header] for header in headers))
	for row in rows:
		lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))
	return "\n".join(lines)


def _print_summary_table(title: str, df: pd.DataFrame) -> None:
	print(f"\n{title}")
	if df.empty:
		print("  No rows")
		return
	print(df.to_string(index=False))


def _write_paper_summary_tables(
	plot_df: pd.DataFrame,
	output_dir: Path,
	expected_rai_distribution_df: pd.DataFrame | None = None,
) -> None:
	ai_long_df = _prepare_ai_long_df(plot_df)
	ai_by_model = _summarize_ai_error(ai_long_df, ["model_name"])
	ai_by_gpu = _summarize_ai_error(ai_long_df, ["gpu"])
	ai_by_runtime = _summarize_ai_error(ai_long_df, ["runtime"])
	bound_by_model = _summarize_bound_metrics(plot_df, ["model_name"])
	bound_by_gpu = _summarize_bound_metrics(plot_df, ["gpu"])
	bound_by_runtime = _summarize_bound_metrics(plot_df, ["runtime"])
	if expected_rai_distribution_df is None:
		expected_rai_distribution_df = _summarize_expected_rai_distribution(plot_df)

	summary_paths = {
		"RAI error summary by model": output_dir / "table_rq1_ai_error_by_model.csv",
		"RAI error summary by GPU": output_dir / "table_rq1_ai_error_by_gpu.csv",
		"RAI error summary by runtime": output_dir / "table_rq1_ai_error_by_runtime.csv",
		"Bound-class summary by model": output_dir / "table_rq1_bound_metrics_by_model.csv",
		"Bound-class summary by GPU": output_dir / "table_rq1_bound_metrics_by_gpu.csv",
		"Bound-class summary by runtime": output_dir / "table_rq1_bound_metrics_by_runtime.csv",
		"Expected RAI distribution by GPU / precision": output_dir / "table_figure6_expected_rai_distribution_by_gpu_precision.csv",
	}

	_write_summary_csv(ai_by_model, summary_paths["RAI error summary by model"])
	_write_summary_csv(ai_by_gpu, summary_paths["RAI error summary by GPU"])
	_write_summary_csv(ai_by_runtime, summary_paths["RAI error summary by runtime"])
	_write_summary_csv(bound_by_model, summary_paths["Bound-class summary by model"])
	_write_summary_csv(bound_by_gpu, summary_paths["Bound-class summary by GPU"])
	_write_summary_csv(bound_by_runtime, summary_paths["Bound-class summary by runtime"])
	_write_summary_csv(
		expected_rai_distribution_df,
		summary_paths["Expected RAI distribution by GPU / precision"],
	)

	figure1_summary = _format_boxplot_summary_table(
		ai_long_df,
		group_field="model_name",
		group_label="Model",
		table_label="figure1_rai_difference_summary",
	)
	figure3_summary = _format_boxplot_summary_table(
		ai_long_df,
		group_field="gpu",
		group_label="GPU",
		table_label="figure3_rai_difference_summary",
	)
	figure4_summary = _format_boxplot_summary_table(
		ai_long_df,
		group_field="runtime",
		group_label="Runtime",
		table_label="figure4_rai_difference_summary",
	)

	print(figure1_summary)
	print(figure3_summary)
	print(figure4_summary)

	_print_summary_table("RAI error summary by model / SASS / precision:", ai_by_model)
	_print_summary_table("RAI error summary by GPU / SASS / precision:", ai_by_gpu)
	_print_summary_table("RAI error summary by runtime / SASS / precision:", ai_by_runtime)
	_print_summary_table("Bound-class summary by model / SASS / precision:", bound_by_model)
	_print_summary_table("Bound-class summary by GPU / SASS / precision:", bound_by_gpu)
	_print_summary_table("Bound-class summary by runtime / SASS / precision:", bound_by_runtime)
	_print_summary_table(
		"Expected RAI distribution by GPU / precision (count string format: 0|BB|CB):",
		expected_rai_distribution_df,
	)

	print("\nPaper summary tables written to:")
	for summary_name, summary_path in summary_paths.items():
		print(f"- {summary_name}: {summary_path}")


def _confusion_heatmap_payload(plot_df: pd.DataFrame, model_name: str, use_sass: bool) -> tuple[pd.DataFrame, np.ndarray]:
	model_subset = plot_df[(plot_df["model_name"] == model_name) & (plot_df["use_sass"] == use_sass)]
	row_labels = [BOUND_CLASS_DISPLAY[label] for label in BOUND_CLASS_ORDER]
	column_labels = [BOUND_CLASS_DISPLAY[label] for label in BOUND_CLASS_ORDER]
	per_precision_matrices: Dict[str, pd.DataFrame] = {}

	for precision in AI_PRECISIONS:
		expected_column = f"expected_bound_{precision}"
		predicted_column = f"predicted_bound_{precision}"
		expected_ai_column = f"expected_ai_{precision}"
		precision_subset = model_subset[
			[expected_ai_column, expected_column, predicted_column]
		].copy()
		precision_subset = precision_subset[
			precision_subset[expected_ai_column].apply(_is_nonzero_expected_ai)
		].dropna(subset=[expected_column, predicted_column])
		counts_df = pd.DataFrame(
			0.0,
			index=BOUND_CLASS_ORDER,
			columns=BOUND_CLASS_ORDER,
		)
		for _, sample in precision_subset.iterrows():
			expected_bound = sample[expected_column]
			predicted_bound = sample[predicted_column]
			if expected_bound in counts_df.index and predicted_bound in counts_df.columns:
				counts_df.loc[expected_bound, predicted_bound] += 1.0
		for expected_bound in BOUND_CLASS_ORDER:
			row_total = float(counts_df.loc[expected_bound].sum())
			if row_total > 0.0:
				counts_df.loc[expected_bound] = counts_df.loc[expected_bound] / row_total * 100.0
		per_precision_matrices[precision] = counts_df

	mean_matrix = pd.DataFrame(0.0, index=BOUND_CLASS_ORDER, columns=BOUND_CLASS_ORDER)
	valid_counts = pd.DataFrame(0.0, index=BOUND_CLASS_ORDER, columns=BOUND_CLASS_ORDER)
	for precision in AI_PRECISIONS:
		precision_matrix = per_precision_matrices[precision]
		for expected_bound in BOUND_CLASS_ORDER:
			row_total = float(precision_matrix.loc[expected_bound].sum())
			if row_total > 0.0:
				mean_matrix.loc[expected_bound] = mean_matrix.loc[expected_bound] + precision_matrix.loc[expected_bound]
				valid_counts.loc[expected_bound] = valid_counts.loc[expected_bound] + 1.0
	with np.errstate(invalid="ignore", divide="ignore"):
		mean_matrix = mean_matrix.divide(valid_counts.where(valid_counts > 0.0))
	mean_matrix = mean_matrix.fillna(0.0)

	annotation = np.empty((len(BOUND_CLASS_ORDER), len(BOUND_CLASS_ORDER)), dtype=object)
	for row_index, expected_bound in enumerate(BOUND_CLASS_ORDER):
		for col_index, predicted_bound in enumerate(BOUND_CLASS_ORDER):
			annotation[row_index, col_index] = "\n".join(
				[
					f"{precision.upper()}: {per_precision_matrices[precision].loc[expected_bound, predicted_bound]:.1f}%"
					for precision in AI_PRECISIONS
				]
			)

	mean_matrix.index = row_labels
	mean_matrix.columns = column_labels
	return mean_matrix, annotation


def _save_ai_difference_boxplots(
	plot_df: pd.DataFrame,
	output_path: Path,
	*,
	group_field: str,
	group_label: str,
	title: str,
) -> None:
	ai_long_df = _prepare_ai_long_df(plot_df)
	plot_group_field = group_field
	group_order = sorted(plot_df[group_field].dropna().unique().tolist()) if not plot_df.empty else []
	if group_field == "runtime" and not ai_long_df.empty:
		plot_group_field = "runtime_display"
		ai_long_df[plot_group_field] = ai_long_df[group_field].map(
			lambda runtime_name: RUNTIME_DISPLAY_LABELS.get(str(runtime_name), str(runtime_name))
		)
		group_order = [
			RUNTIME_DISPLAY_LABELS[runtime_name]
			for runtime_name in ["cuda", "omp"]
			if runtime_name in set(ai_long_df[group_field].dropna().tolist())
		]
	sns.set_theme(style="whitegrid")
	fig_height = _bounded_plot_height(len(group_order), min_height=8.0, per_item=0.6, padding=2.0, max_height=12.0)
	fig, axes = plt.subplots(2, 1, figsize=_scaled_figsize(10.0, fig_height), sharex=True, sharey=True)
	precision_order = [AI_LABELS[precision] for precision in AI_PRECISIONS]
	linthresh = _symlog_linthresh(ai_long_df["ai_diff"]) if not ai_long_df.empty else 1.0
	legend_handles: List[Any] = []
	legend_labels: List[str] = []

	for axis, use_sass in zip(axes, SASS_PANEL_ORDER):
		subset = ai_long_df[ai_long_df["use_sass"] == use_sass].copy()
		if subset.empty:
			axis.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=axis.transAxes)
		else:
			sns.boxplot(
				data=subset,
				x="ai_diff",
				y=plot_group_field,
				hue="precision",
				order=group_order,
				hue_order=precision_order,
				orient="h",
				ax=axis,
			)
		axis.axvline(-1.0, color="red", linestyle="--", linewidth=1.5)
		axis.axvline(0.0, color="green", linestyle="--", linewidth=1.5)
		axis.axvline(1.0, color="red", linestyle="--", linewidth=1.5)
		axis.set_xscale("symlog", linthresh=linthresh)
		_set_rai_symlog_ticks(axis)
		axis.set_title(SASS_PANEL_LABELS[use_sass])
		axis.set_xlabel("Predicted RAI - Expected RAI (symmetric log scale)")
		axis.set_ylabel(group_label)
		axis.tick_params(axis="x", labelsize=8)
		legend = axis.get_legend()
		if legend is not None:
			if not legend_handles:
				handle_by_label = {}
				handles, labels = axis.get_legend_handles_labels()
				for handle, label in zip(handles, labels):
					if label in precision_order and label not in handle_by_label:
						handle_by_label[label] = handle
				legend_labels = [label for label in precision_order if label in handle_by_label]
				legend_handles = [handle_by_label[label] for label in legend_labels]
			legend.remove()

	if legend_handles:
		fig.legend(
			legend_handles,
			legend_labels,
			title="Precision",
			loc="center left",
			bbox_to_anchor=(PAPER_LEGEND_X, 0.5),
		)

	fig.tight_layout(rect=(0, 0, PAPER_LEGEND_RIGHT_MARGIN, 0.97))
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_figure1_ai_boxplots(plot_df: pd.DataFrame, output_path: Path) -> None:
	_save_ai_difference_boxplots(
		plot_df,
		output_path,
		group_field="model_name",
		group_label="Model Name",
		title="Roofline Arithmetic Intensity Difference by Model",
	)


def _save_figure3_ai_boxplots_by_gpu(plot_df: pd.DataFrame, output_path: Path) -> None:
	_save_ai_difference_boxplots(
		plot_df,
		output_path,
		group_field="gpu",
		group_label="GPU",
		title="Roofline Arithmetic Intensity Difference by GPU",
	)


def _save_figure4_ai_boxplots_by_runtime(plot_df: pd.DataFrame, output_path: Path) -> None:
	_save_ai_difference_boxplots(
		plot_df,
		output_path,
		group_field="runtime",
		group_label="Runtime",
		title="Roofline Arithmetic Intensity Difference by Runtime",
	)


def _save_figure5_token_count_histograms(plot_df: pd.DataFrame, output_path: Path) -> None:
	token_long_df = _prepare_token_long_df(plot_df)
	model_order = sorted(plot_df["model_name"].dropna().unique().tolist()) if not plot_df.empty else []
	token_type_order = [TOKEN_COLUMN_LABELS["input_tokens"], TOKEN_COLUMN_LABELS["output_tokens"]]
	token_axis_limits = _token_axis_limits(token_long_df)
	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(2, 2, figsize=_scaled_figsize(16.0, 10.0), sharey="col")
	palette = sns.color_palette("tab10", n_colors=max(len(model_order), 1))
	legend_handles = [
		Line2D([0], [0], color=color, lw=2.0, label=model_name)
		for color, model_name in zip(palette, model_order)
	]
	scientific_formatter = mticker.FuncFormatter(lambda value, _: f"{value:.1e}")

	for row_index, use_sass in enumerate(SASS_PANEL_ORDER):
		for col_index, token_type in enumerate(token_type_order):
			axis = axes[row_index][col_index]
			subset = token_long_df[
				(token_long_df["use_sass"] == use_sass) & (token_long_df["token_type"] == token_type)
			].copy()
			if subset.empty:
				axis.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=axis.transAxes)
				axis.set_axis_off()
				continue

			for color, model_name in zip(palette, model_order):
				model_subset = subset[subset["model_name"] == model_name]
				if model_subset.empty:
					continue
				sns.histplot(
					data=model_subset,
					x="token_count",
					bins=1000,
					stat="count",
					element="step",
					fill=False,
					common_bins=True,
					ax=axis,
					label=model_name,
					color=color,
				)

			axis.set_title(f"{SASS_PANEL_LABELS[use_sass]} | {token_type}")
			axis.set_xlabel(token_type)
			axis.set_ylabel("Count")
			axis.set_xlim(token_axis_limits[token_type])
			axis.xaxis.set_major_formatter(scientific_formatter)
			axis.tick_params(axis="x", labelsize=8)
			axis.xaxis.offsetText.set_visible(False)
			legend = axis.get_legend()
			if legend is not None:
				legend.remove()

	if legend_handles:
		fig.legend(
			legend_handles,
			[handle.get_label() for handle in legend_handles],
			title="Model Name",
			loc="center left",
			bbox_to_anchor=(PAPER_LEGEND_X, 0.5),
			ncol=1,
		)

	fig.suptitle("Token Count Distributions by Evidence Mode and Token Type", y=0.98)
	fig.tight_layout(rect=(0, 0, PAPER_LEGEND_RIGHT_MARGIN, 0.95))
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_figure2_bound_heatmaps(plot_df: pd.DataFrame, output_path: Path) -> None:
	model_order = sorted(plot_df["model_name"].dropna().unique().tolist())
	row_count = max(len(model_order), 1)
	sns.set_theme(style="whitegrid")
	fig_height = max(8.5, 3.2 * row_count)
	fig, axes = plt.subplots(row_count, 2, figsize=_scaled_figsize(12.5, fig_height), squeeze=False)
	cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
	vmin = 0.0
	vmax = 100.0

	if not model_order:
		for axis in axes.flatten():
			axis.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=axis.transAxes)
			axis.set_axis_off()
	else:
		for row_index, model_name in enumerate(model_order):
			for col_index, use_sass in enumerate(SASS_PANEL_ORDER):
				axis = axes[row_index][col_index]
				matrix_df, annotation = _confusion_heatmap_payload(plot_df, model_name, use_sass)
				sns.heatmap(
					matrix_df,
					ax=axis,
					annot=annotation,
					fmt="",
					cmap="crest",
					vmin=vmin,
					vmax=vmax,
					cbar=(row_index == 0 and col_index == 0),
					cbar_ax=cbar_ax if row_index == 0 and col_index == 0 else None,
					linewidths=0.5,
					linecolor="white",
				)
				axis.set_title(f"{model_name} | {SASS_PANEL_LABELS[use_sass]}", pad=8)
				axis.set_xlabel("Predicted Bound Class", labelpad=4)
				axis.set_ylabel("Expected Bound Class")

	if hasattr(cbar_ax, "collections") and cbar_ax.collections:
		cbar_ax.set_ylabel("Mean within-true-class prediction rate across FP16/FP32/FP64 (%)", rotation=90, labelpad=12)

	fig.subplots_adjust(left=0.09, right=0.9, top=0.92, bottom=0.08, hspace=0.5, wspace=0.28)
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_figure6_expected_rai_distribution(
	expected_rai_distribution_df: pd.DataFrame,
	output_path: Path,
	runtime_distribution_df: pd.DataFrame | None = None,
) -> None:
	sns.set_theme(style="whitegrid")
	combined_row_count = expected_rai_distribution_df.shape[0]
	if runtime_distribution_df is not None:
		combined_row_count += runtime_distribution_df.shape[0]
	fig_height = _bounded_plot_height(
		combined_row_count,
		min_height=6.5,
		per_item=0.55,
		padding=2.0,
		max_height=12.5,
	)
	fig, axis = plt.subplots(figsize=_scaled_figsize(11.0, fig_height))

	if expected_rai_distribution_df.empty and (runtime_distribution_df is None or runtime_distribution_df.empty):
		axis.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=axis.transAxes)
		axis.set_axis_off()
	else:
		runtime_distribution_df = runtime_distribution_df if runtime_distribution_df is not None else pd.DataFrame()
		plot_rows: List[Dict[str, Any]] = []
		intra_group_spacing = 0.82
		inter_group_gap = 0.55
		runtime_group_gap = 0.8
		bar_height = 0.64
		expected_lookup = {
			(gpu_name, precision): row
			for _, row in expected_rai_distribution_df.iterrows()
			for gpu_name, precision in [(row["gpu"], row["precision"])]
		}
		gpu_order = list(GPU_ROOFLINE_TABLE.keys())
		for gpu_name in gpu_order:
			for precision in [PRECISION_DISPLAY_LABELS[item] for item in AI_PRECISIONS]:
				row = expected_lookup.get((gpu_name, precision))
				if row is None:
					continue
				plot_rows.append(
					{
						"row_kind": "expected",
						"gpu": gpu_name,
						"precision": precision,
						"gpu_precision_label": f"({gpu_name}, {precision})",
						**row.to_dict(),
					}
				)
		if not runtime_distribution_df.empty:
			runtime_row = runtime_distribution_df.iloc[-1].to_dict()
			plot_rows.append(
				{
					"row_kind": "runtime",
					"gpu": runtime_row.get("gpu", "All GPUs"),
					"precision": "Runtime",
					"gpu_precision_label": "Runtime",
					**runtime_row,
				}
			)
		plot_df = pd.DataFrame(plot_rows).reset_index(drop=True)
		y_positions: List[float] = []
		current_y = 0.0
		previous_group: str | None = None
		for _, row in plot_df.iterrows():
			group_key = str(row["gpu"]) if row["row_kind"] == "expected" else "Runtime"
			if previous_group is not None:
				gap = runtime_group_gap if group_key == "Runtime" or previous_group == "Runtime" else inter_group_gap
				current_y += intra_group_spacing if group_key == previous_group else intra_group_spacing + gap
			y_positions.append(current_y)
			previous_group = group_key
		plot_df["y_position"] = y_positions
		added_legend_labels: set[str] = set()
		for row_index, row in plot_df.iterrows():
			left_offset = 0.0
			if row["row_kind"] == "expected":
				for category in EXPECTED_RAI_DISTRIBUTION_COLUMNS:
					count = float(pd.to_numeric(row.get(category), errors="coerce") or 0.0)
					label = EXPECTED_RAI_DISTRIBUTION_LABELS[category]
					axis.barh(
						float(row["y_position"]),
						count,
						left=left_offset,
						height=bar_height,
						label=label if label not in added_legend_labels else None,
						color=EXPECTED_RAI_DISTRIBUTION_COLORS[category],
					)
					added_legend_labels.add(label)
					left_offset += count
			else:
				for runtime_column in RUNTIME_DISTRIBUTION_COLUMNS:
					count = float(pd.to_numeric(row.get(runtime_column), errors="coerce") or 0.0)
					axis.barh(
						float(row["y_position"]),
						count,
						left=left_offset,
						height=bar_height,
						color=RUNTIME_DISTRIBUTION_COLORS[runtime_column],
					)
					if count > 0.0:
						axis.text(
							left_offset + count / 2.0,
							float(row["y_position"]),
							RUNTIME_DISTRIBUTION_LABELS[runtime_column],
							ha="center",
							va="center",
							fontsize=8,
							color="white",
							fontweight="bold",
						)
					left_offset += count

		max_total = float(plot_df["total_kernels"].max()) if not plot_df.empty else 1.0
		label_padding = max(1.0, max_total * 0.02)
		for row_index, row in plot_df.iterrows():
			axis.text(
				float(row["total_kernels"]) + label_padding,
				float(row["y_position"]),
				str(row["count_string"]),
				va="center",
				ha="left",
				fontsize=8,
			)

		axis.set_yticks(plot_df["y_position"].tolist(), plot_df["gpu_precision_label"])
		axis.invert_yaxis()
		axis.set_xlabel("Kernel Count")
		axis.set_ylabel("GPU / FLOP Precision")
		axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
		axis.grid(axis="x", color="#d9d9d9", linewidth=0.8)
		axis.grid(axis="y", visible=False)
		axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
		axis.set_xlim(0.0, max_total + label_padding + max(3.0, max_total * 0.15))

	fig.tight_layout(rect=(0, 0.1, 1, 0.98))
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def build_paper_plots(db_uri: str, output_dir: Path, include_dry_run: bool, only_shared_samples: bool) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	samples_df = _load_samples_dataframe(db_uri, include_dry_run)
	if samples_df.empty:
		raise RuntimeError("No matching checkpoint or failed-attempt records were found in the database.")

	completed_df = _enrich_completed_dataframe(samples_df)
	paper_candidate_df = _paper_subset(completed_df)
	_print_gpu_kernel_sample_coverage_summary(paper_candidate_df, "pre-shared-filter")
	filtered_samples_df = samples_df
	if only_shared_samples:
		filtered_samples_df = visualize_results._filter_only_shared_samples(samples_df, include_imix=False)

	completed_df = _enrich_completed_dataframe(filtered_samples_df)
	plot_df = _paper_subset(completed_df)
	if only_shared_samples:
		_print_gpu_kernel_sample_coverage_summary(plot_df, "shared across all GPUs/models/prompt-types via completed-or-failed presence")
	if plot_df.empty:
		raise RuntimeError(
			"No completed no-IMIX samples were found for the requested noSASS-noIMIX and wSASS-noIMIX plots."
		)
	_print_bound_class_distribution(plot_df)
	expected_rai_distribution_df = _summarize_expected_rai_distribution(plot_df)
	runtime_distribution_df = _summarize_runtime_distribution(plot_df)

	figure1_path = output_dir / "figure1_ai_difference_boxplots.png"
	figure2_path = output_dir / "figure2_ai_bound_confusion_heatmaps.png"
	figure3_path = output_dir / "figure3_ai_difference_boxplots_by_gpu.png"
	figure4_path = output_dir / "figure4_ai_difference_boxplots_by_runtime.png"
	figure5_path = output_dir / "figure5_token_count_histograms.png"
	figure6_path = output_dir / "figure6_expected_rai_distribution_by_gpu_precision.png"

	_save_figure1_ai_boxplots(plot_df, figure1_path)
	_save_figure2_bound_heatmaps(plot_df, figure2_path)
	_save_figure3_ai_boxplots_by_gpu(plot_df, figure3_path)
	_save_figure4_ai_boxplots_by_runtime(plot_df, figure4_path)
	_save_figure5_token_count_histograms(plot_df, figure5_path)
	_save_figure6_expected_rai_distribution(expected_rai_distribution_df, figure6_path, runtime_distribution_df)
	_write_paper_summary_tables(plot_df, output_dir, expected_rai_distribution_df)

	print("Paper plot artifacts written to:")
	print(f"- {figure1_path}")
	print(f"- {figure2_path}")
	print(f"- {figure3_path}")
	print(f"- {figure4_path}")
	print(f"- {figure5_path}")
	print(f"- {figure6_path}")


def build_arg_parser() -> argparse.ArgumentParser:
	arg_parser = argparse.ArgumentParser(description="Generate paper plots from direct-prompting PostgreSQL checkpoints")
	arg_parser.add_argument(
		"--dbUri",
		type=str,
		default=None,
		help="Explicit PostgreSQL database URI. Defaults to the local gpuflops_db database.",
	)
	arg_parser.add_argument(
		"--outputDir",
		type=str,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where the paper figures will be written.",
	)
	arg_parser.add_argument(
		"--includeDryRun",
		action="store_true",
		help="Include dry-run thread IDs in the paper plots. By default they are excluded.",
	)
	arg_parser.add_argument(
		"--onlySharedSamples",
		action="store_true",
		help="Keep only benchmark/kernel identities that have at least one stored row (completed or failed) for every GPU, every model name, and both plotted prompt types: Source-Only and Source+SASS.",
	)
	return arg_parser


def main() -> None:
	arg_parser = build_arg_parser()
	args = arg_parser.parse_args()

	_print_roofline_specs()
	visualize_results.ensure_postgres_running()
	db_uri = args.dbUri or visualize_results.setup_default_database()
	build_paper_plots(db_uri, Path(args.outputDir), args.includeDryRun, args.onlySharedSamples)


if __name__ == "__main__":
	main()
