import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import importlib.util

import matplotlib

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
	"fp16": "FP16 AI",
	"fp32": "FP32 AI",
	"fp64": "FP64 AI",
}
BOUND_LABELS = ["TP", "FP", "FN", "TN"]
SASS_PANEL_ORDER = [False, True]
SASS_PANEL_LABELS = {
	False: "No SASS / No IMIX",
	True: "SASS Only / No IMIX",
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
	NEGATIVE_CLASS: "Bandwidth-bound",
	POSITIVE_CLASS: "Compute-bound",
}
DEFAULT_OUTPUT_DIR = os.path.join(
	WORKSPACE_ROOT,
	"experiments",
	"direct-prompting",
	"paper-figure-output",
)

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


def _bounded_plot_height(
	item_count: int,
	*,
	min_height: float,
	per_item: float,
	padding: float,
	max_height: float,
) -> float:
	return min(max_height, max(min_height, per_item * max(item_count, 1) + padding))


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
	print("\nExpected bound-class distribution for nonzero expected AI cases used in paper figures:")
	for precision in AI_PRECISIONS:
		expected_ai_column = f"expected_ai_{precision}"
		expected_bound_column = f"expected_bound_{precision}"
		precision_mask = plot_df[expected_ai_column].apply(_is_nonzero_expected_ai)
		precision_df = plot_df.loc[precision_mask, ["gpu", expected_bound_column]].copy()
		print(f"- {precision.upper()}")
		if precision_df.empty:
			print("  No nonzero expected-AI samples")
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
	group_order = sorted(plot_df[group_field].dropna().unique().tolist()) if not plot_df.empty else []
	sns.set_theme(style="whitegrid")
	fig_height = _bounded_plot_height(len(group_order), min_height=8.0, per_item=0.6, padding=2.0, max_height=12.0)
	fig, axes = plt.subplots(2, 1, figsize=(12.0, fig_height), sharex=True, sharey=True)
	precision_order = [AI_LABELS[precision] for precision in AI_PRECISIONS]
	linthresh = _symlog_linthresh(ai_long_df["ai_diff"]) if not ai_long_df.empty else 1.0

	for axis, use_sass in zip(axes, SASS_PANEL_ORDER):
		subset = ai_long_df[ai_long_df["use_sass"] == use_sass].copy()
		if subset.empty:
			axis.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=axis.transAxes)
		else:
			sns.boxplot(
				data=subset,
				x="ai_diff",
				y=group_field,
				hue="precision",
				order=group_order,
				hue_order=precision_order,
				orient="h",
				ax=axis,
			)
		axis.axvline(0.0, color="green", linestyle="--", linewidth=1.5)
		axis.set_xscale("symlog", linthresh=linthresh)
		axis.set_title(SASS_PANEL_LABELS[use_sass])
		axis.set_xlabel("Predicted AI - Expected AI (symmetric log scale)")
		axis.set_ylabel(group_label)
		axis.tick_params(axis="x", labelsize=8)
		legend = axis.get_legend()
		if legend is not None and use_sass is False:
			legend.set_title("Precision")
		elif legend is not None:
			legend.remove()

	fig.suptitle(title)
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def _save_figure1_ai_boxplots(plot_df: pd.DataFrame, output_path: Path) -> None:
	_save_ai_difference_boxplots(
		plot_df,
		output_path,
		group_field="model_name",
		group_label="Model Name",
		title="Arithmetic Intensity Difference by Model",
	)


def _save_figure3_ai_boxplots_by_gpu(plot_df: pd.DataFrame, output_path: Path) -> None:
	_save_ai_difference_boxplots(
		plot_df,
		output_path,
		group_field="gpu",
		group_label="GPU",
		title="Arithmetic Intensity Difference by GPU",
	)


def _save_figure4_ai_boxplots_by_runtime(plot_df: pd.DataFrame, output_path: Path) -> None:
	_save_ai_difference_boxplots(
		plot_df,
		output_path,
		group_field="runtime",
		group_label="Runtime",
		title="Arithmetic Intensity Difference by Runtime",
	)


def _save_figure2_bound_heatmaps(plot_df: pd.DataFrame, output_path: Path) -> None:
	model_order = sorted(plot_df["model_name"].dropna().unique().tolist())
	row_count = max(len(model_order), 1)
	sns.set_theme(style="whitegrid")
	fig_height = max(8.5, 3.2 * row_count)
	fig, axes = plt.subplots(row_count, 2, figsize=(12.5, fig_height), squeeze=False)
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
				axis.set_title(f"{model_name} | {SASS_PANEL_LABELS[use_sass]}")
				axis.set_xlabel("Predicted Bound Class")
				axis.set_ylabel("Expected Bound Class")

	if hasattr(cbar_ax, "collections") and cbar_ax.collections:
		cbar_ax.set_ylabel("Mean within-true-class prediction rate across FP16/FP32/FP64 (%)", rotation=90, labelpad=12)

	fig.suptitle("AI Bound Classification by Expected vs Predicted Class")
	fig.subplots_adjust(left=0.09, right=0.9, top=0.92, bottom=0.08, hspace=0.35, wspace=0.28)
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def build_paper_plots(db_uri: str, output_dir: Path, include_dry_run: bool) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	samples_df = _load_samples_dataframe(db_uri, include_dry_run)
	if samples_df.empty:
		raise RuntimeError("No matching checkpoint or failed-attempt records were found in the database.")

	completed_df = _enrich_completed_dataframe(samples_df)
	plot_df = _paper_subset(completed_df)
	if plot_df.empty:
		raise RuntimeError(
			"No completed no-IMIX samples were found for the requested noSASS-noIMIX and wSASS-noIMIX plots."
		)
	_print_bound_class_distribution(plot_df)

	figure1_path = output_dir / "figure1_ai_difference_boxplots.png"
	figure2_path = output_dir / "figure2_ai_bound_confusion_heatmaps.png"
	figure3_path = output_dir / "figure3_ai_difference_boxplots_by_gpu.png"
	figure4_path = output_dir / "figure4_ai_difference_boxplots_by_runtime.png"

	_save_figure1_ai_boxplots(plot_df, figure1_path)
	_save_figure2_bound_heatmaps(plot_df, figure2_path)
	_save_figure3_ai_boxplots_by_gpu(plot_df, figure3_path)
	_save_figure4_ai_boxplots_by_runtime(plot_df, figure4_path)

	print("Paper plot artifacts written to:")
	print(f"- {figure1_path}")
	print(f"- {figure2_path}")
	print(f"- {figure3_path}")
	print(f"- {figure4_path}")


def main() -> None:
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
	args = arg_parser.parse_args()

	_print_roofline_specs()
	visualize_results.ensure_postgres_running()
	db_uri = args.dbUri or visualize_results.setup_default_database()
	build_paper_plots(db_uri, Path(args.outputDir), args.includeDryRun)


if __name__ == "__main__":
	main()
