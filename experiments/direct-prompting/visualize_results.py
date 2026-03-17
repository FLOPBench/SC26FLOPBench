import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import importlib.util


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)


db_manager_path = os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "db_manager.py")
db_manager_spec = importlib.util.spec_from_file_location("db_manager", db_manager_path)
db_manager_mod = importlib.util.module_from_spec(db_manager_spec)
if db_manager_spec and db_manager_spec.loader:
	db_manager_spec.loader.exec_module(db_manager_mod)

CheckpointDBParser = db_manager_mod.CheckpointDBParser
QueryAttemptTracker = db_manager_mod.QueryAttemptTracker
setup_default_database = db_manager_mod.setup_default_database
ensure_postgres_running = db_manager_mod.ensure_postgres_running


THREAD_PATTERN = re.compile(
	r"_(?P<gpu>A100|3080|H100|A10)_(?P<safe_model>.+)_(?P<sass>withsass|nosass)_trial(?P<trial>\d+)(?:_DRYRUN(?:\d+)?)?$"
)

METRIC_LABELS = {
	"fp16": "FP16 FLOPs",
	"fp32": "FP32 FLOPs",
	"fp64": "FP64 FLOPs",
	"read_bytes": "Read Bytes",
	"write_bytes": "Write Bytes",
}

COMPLETED_RECORD_COLUMNS = [
	"thread_id",
	"status",
	"program_name",
	"kernel_mangled_name",
	"kernel_demangled_name",
	"model_name",
	"safe_model_name",
	"use_sass",
	"gpu",
	"trial",
	"query_time",
	"cost_usd",
	"input_tokens",
	"output_tokens",
	"total_tokens",
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
	"sample_mean_pct_diff",
	"metrics_pct_diff_fp16",
	"metrics_pct_diff_fp32",
	"metrics_pct_diff_fp64",
	"metrics_pct_diff_read_bytes",
	"metrics_pct_diff_write_bytes",
]

FAILED_RECORD_COLUMNS = [
	"thread_id",
	"status",
	"program_name",
	"kernel_mangled_name",
	"kernel_demangled_name",
	"model_name",
	"safe_model_name",
	"use_sass",
	"gpu",
	"trial",
	"query_time",
	"cost_usd",
	"input_tokens",
	"output_tokens",
	"total_tokens",
	"sample_mean_pct_diff",
	"last_status",
	"failed_attempts",
	"last_error",
]


def _safe_filename(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "value"


def _display_model_name(safe_model_name: str) -> str:
	return safe_model_name.replace("_", "/")


def _is_dry_run_thread(thread_id: str) -> bool:
	return "_DRYRUN" in thread_id or thread_id.endswith("_DRYRUN")


def _stored_thread_ids(checkpoints: List[Dict[str, Any]], attempts: Dict[str, Dict[str, Any]]) -> set[str]:
	checkpoint_thread_ids = {checkpoint["thread_id"] for checkpoint in checkpoints}
	attempt_thread_ids = set(attempts.keys())
	return checkpoint_thread_ids | attempt_thread_ids


def _thread_metadata(thread_id: str) -> Dict[str, Any]:
	match = THREAD_PATTERN.search(thread_id)
	if not match:
		return {
			"gpu": None,
			"safe_model_name": "unknown-model",
			"model_name": "unknown-model",
			"use_sass": None,
			"trial": None,
		}

	safe_model_name = match.group("safe_model")
	return {
		"gpu": match.group("gpu"),
		"safe_model_name": safe_model_name,
		"model_name": _display_model_name(safe_model_name),
		"use_sass": match.group("sass") == "withsass",
		"trial": int(match.group("trial")),
	}


def _require_mapping_keys(mapping: Dict[str, Any], required_keys: List[str], context: str) -> None:
	missing_keys = [key for key in required_keys if key not in mapping]
	if missing_keys:
		available_keys = sorted(mapping.keys())
		raise KeyError(
			f"{context} is missing required keys {missing_keys}. Available keys: {available_keys}"
		)


def _tail_checkpoint_by_thread(checkpoints: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
	checkpoints_by_thread: Dict[str, List[Dict[str, Any]]] = {}
	for checkpoint in checkpoints:
		checkpoints_by_thread.setdefault(checkpoint["thread_id"], []).append(checkpoint)

	tails: Dict[str, Dict[str, Any]] = {}
	for thread_id, thread_checkpoints in checkpoints_by_thread.items():
		checkpoints_by_id = {
			checkpoint["checkpoint_id"]: checkpoint for checkpoint in thread_checkpoints
		}
		children_by_parent: Dict[Any, List[Dict[str, Any]]] = {}
		for checkpoint in thread_checkpoints:
			parent_checkpoint_id = checkpoint["parent_checkpoint_id"]
			children_by_parent.setdefault(parent_checkpoint_id, []).append(checkpoint)

		roots = children_by_parent[None] if None in children_by_parent else []
		if len(roots) != 1:
			raise ValueError(
				f"Thread {thread_id} expected exactly one root checkpoint, found {len(roots)}"
			)

		current = roots[0]
		visited_checkpoint_ids = set()
		while True:
			checkpoint_id = current["checkpoint_id"]
			if checkpoint_id in visited_checkpoint_ids:
				raise ValueError(f"Cycle detected in checkpoint chain for thread {thread_id}")
			visited_checkpoint_ids.add(checkpoint_id)

			if checkpoint_id not in checkpoints_by_id:
				raise KeyError(f"Checkpoint {checkpoint_id} missing from thread index for {thread_id}")

			children = children_by_parent[checkpoint_id] if checkpoint_id in children_by_parent else []
			if not children:
				tails[thread_id] = current
				break
			if len(children) != 1:
				raise ValueError(
					f"Thread {thread_id} expected a linear checkpoint chain, found {len(children)} children for checkpoint {checkpoint_id}"
				)
			current = children[0]

		if len(visited_checkpoint_ids) != len(thread_checkpoints):
			unvisited = len(thread_checkpoints) - len(visited_checkpoint_ids)
			raise ValueError(
				f"Thread {thread_id} has {unvisited} checkpoint entries disconnected from the root chain"
			)

	return tails


def _completed_checkpoint_by_thread(checkpoints: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
	completed: Dict[str, Dict[str, Any]] = {}
	for thread_id, checkpoint in _tail_checkpoint_by_thread(checkpoints).items():
		state = checkpoint["checkpoint"]["channel_values"]
		if "total_tokens" in state:
			completed[thread_id] = checkpoint
	return completed


def _sample_mean_pct_diff(metrics_pct_diff: Dict[str, Any]) -> float:
	values: List[float] = []
	for metric_name in METRIC_LABELS:
		value = metrics_pct_diff[metric_name]
		if value is None:
			continue
		try:
			numeric_value = float(value)
		except (TypeError, ValueError):
			continue
		if math.isfinite(numeric_value):
			values.append(numeric_value)

	if not values:
		return float("nan")
	return float(np.mean(values))


def _extract_completed_records(
	completed_checkpoints: Dict[str, Dict[str, Any]],
	include_dry_run: bool,
) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	for thread_id, checkpoint in completed_checkpoints.items():
		if not include_dry_run and _is_dry_run_thread(thread_id):
			continue

		state = checkpoint["checkpoint"]["channel_values"]
		_require_mapping_keys(
			state,
			[
				"metrics_diff",
				"metrics_pct_diff",
				"program_name",
				"kernel_mangled_name",
				"kernel_demangled_name",
				"llm_model_name",
				"query_time",
				"cost_usd",
				"input_tokens",
				"output_tokens",
				"total_tokens",
				"expected_fp16",
				"expected_fp32",
				"expected_fp64",
				"expected_read_bytes",
				"expected_write_bytes",
			],
			f"Completed checkpoint tail for thread {thread_id}",
		)
		metadata = _thread_metadata(thread_id)
		metrics_diff = state["metrics_diff"]
		metrics_pct_diff = state["metrics_pct_diff"]

		record: Dict[str, Any] = {
			"thread_id": thread_id,
			"status": "completed",
			"program_name": state["program_name"],
			"kernel_mangled_name": state["kernel_mangled_name"],
			"kernel_demangled_name": state["kernel_demangled_name"],
			"model_name": state["llm_model_name"] or metadata["model_name"],
			"safe_model_name": metadata["safe_model_name"],
			"use_sass": metadata["use_sass"],
			"gpu": metadata["gpu"],
			"trial": metadata["trial"],
			"query_time": pd.to_numeric(state["query_time"], errors="coerce"),
			"cost_usd": pd.to_numeric(state["cost_usd"], errors="coerce"),
			"input_tokens": pd.to_numeric(state["input_tokens"], errors="coerce"),
			"output_tokens": pd.to_numeric(state["output_tokens"], errors="coerce"),
			"total_tokens": pd.to_numeric(state["total_tokens"], errors="coerce"),
			"expected_fp16": pd.to_numeric(state["expected_fp16"], errors="coerce"),
			"expected_fp32": pd.to_numeric(state["expected_fp32"], errors="coerce"),
			"expected_fp64": pd.to_numeric(state["expected_fp64"], errors="coerce"),
			"expected_read_bytes": pd.to_numeric(state["expected_read_bytes"], errors="coerce"),
			"expected_write_bytes": pd.to_numeric(state["expected_write_bytes"], errors="coerce"),
			"sample_mean_pct_diff": _sample_mean_pct_diff(metrics_pct_diff),
		}

		for metric_key in METRIC_LABELS:
			record[f"metrics_diff_{metric_key}"] = pd.to_numeric(metrics_diff[metric_key], errors="coerce")
			record[f"metrics_pct_diff_{metric_key}"] = pd.to_numeric(metrics_pct_diff[metric_key], errors="coerce")

		records.append(record)

	return records


def _extract_failed_records(
	attempts: Dict[str, Dict[str, Any]],
	latest_checkpoints: Dict[str, Dict[str, Any]],
	completed_thread_ids: set[str],
	include_dry_run: bool,
) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	for thread_id, attempt in attempts.items():
		if thread_id in completed_thread_ids:
			continue
		if not include_dry_run and _is_dry_run_thread(thread_id):
			continue

		failed_attempts = attempt["failed_attempts"] or 0
		last_status = attempt["last_status"]
		if failed_attempts <= 0 and last_status != "failed":
			continue

		metadata = _thread_metadata(thread_id)
		partial_state = latest_checkpoints[thread_id]["checkpoint"]["channel_values"]
		_require_mapping_keys(
			partial_state,
			[
				"program_name",
				"kernel_mangled_name",
				"kernel_demangled_name",
			],
			f"Latest checkpoint tail for failed thread {thread_id}",
		)
		records.append(
			{
				"thread_id": thread_id,
				"status": "failed",
				"program_name": partial_state["program_name"],
				"kernel_mangled_name": partial_state["kernel_mangled_name"],
				"kernel_demangled_name": partial_state["kernel_demangled_name"],
				"model_name": partial_state["llm_model_name"] if "llm_model_name" in partial_state else metadata["model_name"],
				"safe_model_name": metadata["safe_model_name"],
				"use_sass": metadata["use_sass"],
				"gpu": metadata["gpu"],
				"trial": metadata["trial"],
				"query_time": np.nan,
				"cost_usd": np.nan,
				"input_tokens": np.nan,
				"output_tokens": np.nan,
				"total_tokens": np.nan,
				"sample_mean_pct_diff": np.nan,
				"last_status": last_status,
				"failed_attempts": failed_attempts,
				"last_error": attempt["last_error"],
			}
		)

	return records


def _completed_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
	return pd.DataFrame(records, columns=COMPLETED_RECORD_COLUMNS)


def _failed_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
	return pd.DataFrame(records, columns=FAILED_RECORD_COLUMNS)


def _database_dataframe(checkpoints: List[Dict[str, Any]], attempts: Dict[str, Dict[str, Any]], include_dry_run: bool) -> pd.DataFrame:
	latest_checkpoints = _tail_checkpoint_by_thread(checkpoints)
	completed_checkpoints = _completed_checkpoint_by_thread(checkpoints)

	completed_records = _extract_completed_records(completed_checkpoints, include_dry_run)
	failed_records = _extract_failed_records(attempts, latest_checkpoints, set(completed_checkpoints), include_dry_run)

	completed_df = _completed_dataframe(completed_records)
	failed_df = _failed_dataframe(failed_records)
	full_df = pd.concat([completed_df, failed_df], ignore_index=True, sort=False)

	if full_df.empty:
		return full_df

	if "model_name" in full_df.columns:
		full_df["model_name"] = full_df["model_name"].fillna("unknown-model")
	if "use_sass" in full_df.columns:
		full_df["use_sass"] = full_df["use_sass"].fillna(False)

	return full_df


def _prepare_metric_long_df(completed_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	columns = ["model_name", "use_sass", "metric", "value"]
	if completed_df.empty:
		return pd.DataFrame(columns=columns)

	for _, row in completed_df.iterrows():
		for metric_key, metric_label in METRIC_LABELS.items():
			value = row[f"{prefix}_{metric_key}"]
			if pd.isna(value):
				continue
			rows.append(
				{
					"model_name": row["model_name"],
					"use_sass": row["use_sass"],
					"metric": metric_label,
					"value": float(value),
				}
			)
	return pd.DataFrame(rows, columns=columns)


def _save_stacked_sample_count_plot(samples_df: pd.DataFrame, output_path: Path) -> None:
	plot_df = samples_df.copy()
	plot_df["use_sass_label"] = np.where(plot_df["use_sass"], "With SASS", "Without SASS")
	plot_df["status_segment"] = plot_df.apply(
		lambda row: f"{row['status'].title()} | {row['use_sass_label']}",
		axis=1,
	)

	segment_order = [
		"Completed | Without SASS",
		"Completed | With SASS",
		"Failed | Without SASS",
		"Failed | With SASS",
	]
	counts = (
		plot_df.groupby(["model_name", "status_segment"]).size().unstack(fill_value=0).reindex(columns=segment_order, fill_value=0)
	)
	model_names = counts.index.tolist()

	sns.set_theme(style="whitegrid")
	fig, ax = plt.subplots(figsize=(max(10, 1.8 * max(len(model_names), 1)), 7))
	colors = sns.color_palette("Set2", n_colors=len(segment_order))
	bottoms = np.zeros(len(model_names))

	for segment, color in zip(segment_order, colors):
		values = counts[segment].to_numpy()
		bars = ax.bar(model_names, values, bottom=bottoms, label=segment, color=color, edgecolor="white")
		for bar, value, bottom in zip(bars, values, bottoms):
			if value <= 0:
				continue
			ax.text(
				bar.get_x() + bar.get_width() / 2.0,
				bottom + value / 2.0,
				f"{int(value)}",
				ha="center",
				va="center",
				fontsize=8,
				color="black",
			)
		bottoms = bottoms + values

	for model_index, total in enumerate(bottoms):
		ax.text(model_index, total + 0.5, f"{int(total)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

	ax.set_title("Database Sample Counts by Model and SASS Configuration")
	ax.set_xlabel("Model Name")
	ax.set_ylabel("Sample Count")
	ax.legend(title="Sample Type", bbox_to_anchor=(1.02, 1), loc="upper left")
	ax.tick_params(axis="x", rotation=30)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_histogram_by_sass(
	completed_df: pd.DataFrame,
	value_column: str,
	title: str,
	x_label: str,
	output_path: Path,
) -> None:
	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

	if completed_df.empty or value_column not in completed_df.columns or "use_sass" not in completed_df.columns:
		for ax, use_sass in zip(axes, [False, True]):
			label = "With SASS" if use_sass else "Without SASS"
			ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
			ax.set_title(label)
			ax.set_xlabel(x_label)
			ax.set_ylabel("Sample Count")
		fig.suptitle(title)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		return

	for ax, use_sass in zip(axes, [False, True]):
		subset = completed_df[completed_df["use_sass"] == use_sass].copy()
		numeric_values = pd.to_numeric(subset[value_column], errors="coerce")
		subset = subset[numeric_values.notna()]
		numeric_values = numeric_values.loc[subset.index]
		subset = subset[np.isfinite(numeric_values.to_numpy())]
		subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
		label = "With SASS" if use_sass else "Without SASS"
		if subset.empty:
			ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
		else:
			sns.histplot(
				data=subset,
				x=value_column,
				hue="model_name",
				bins=30,
				multiple="layer",
				element="step",
				fill=False,
				common_norm=False,
				ax=ax,
			)
		ax.set_title(label)
		ax.set_xlabel(x_label)
		ax.set_ylabel("Sample Count")

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_metric_hist_grid(metric_df: pd.DataFrame, title: str, x_label: str, output_path: Path) -> None:
	if metric_df.empty or "model_name" not in metric_df.columns:
		sns.set_theme(style="whitegrid")
		fig, axes = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
		for ax in axes.flatten():
			ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
			ax.set_axis_off()
		fig.suptitle(title)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		return

	model_names = sorted(metric_df["model_name"].dropna().unique().tolist())
	row_count = max(len(model_names), 1)

	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(row_count, 2, figsize=(16, max(4, 4 * row_count)), squeeze=False)

	if not model_names:
		for ax in axes.flatten():
			ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
			ax.set_axis_off()
	else:
		for row_index, model_name in enumerate(model_names):
			for col_index, use_sass in enumerate([False, True]):
				ax = axes[row_index][col_index]
				subset = metric_df[(metric_df["model_name"] == model_name) & (metric_df["use_sass"] == use_sass)].copy()
				numeric_values = pd.to_numeric(subset["value"], errors="coerce")
				subset = subset[numeric_values.notna()]
				numeric_values = numeric_values.loc[subset.index]
				subset = subset[np.isfinite(numeric_values.to_numpy())]
				subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
				if subset.empty:
					ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
				else:
					sns.histplot(
						data=subset,
						x="value",
						hue="metric",
						bins=30,
						multiple="layer",
						element="step",
						fill=False,
						common_norm=False,
						ax=ax,
					)
				sass_label = "With SASS" if use_sass else "Without SASS"
				ax.set_title(f"{model_name} | {sass_label}")
				ax.set_xlabel(x_label)
				ax.set_ylabel("Metric Count")

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_table_figure(dataframe: pd.DataFrame, title: str, output_path: Path) -> None:
	display_df = dataframe.fillna("").copy()
	fig_width = max(12, 1.4 * len(display_df.columns))
	fig_height = max(2.5, 0.42 * len(display_df) + 1.6)

	fig, ax = plt.subplots(figsize=(fig_width, fig_height))
	ax.axis("off")
	table = ax.table(
		cellText=display_df.values,
		colLabels=display_df.columns,
		cellLoc="center",
		loc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(8)
	table.scale(1.0, 1.2)
	ax.set_title(title, pad=12)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _latex_escape(value: str) -> str:
	replacements = {
		"\\": r"\textbackslash{}",
		"&": r"\&",
		"%": r"\%",
		"$": r"\$",
		"#": r"\#",
		"_": r"\_",
		"{": r"\{",
		"}": r"\}",
		"~": r"\textasciitilde{}",
		"^": r"\textasciicircum{}",
	}
	escaped = []
	for character in value:
		escaped.append(replacements.get(character, character))
	return "".join(escaped)


def _latex_cell(value: Any) -> str:
	if pd.isna(value):
		return ""
	if isinstance(value, (float, np.floating)):
		if np.isinf(value):
			return r"$\infty$" if value > 0 else r"$-\infty$"
		return f"{value}"
	return _latex_escape(str(value))


def _write_booktabs_latex_table(
	dataframe: pd.DataFrame,
	output_path: Path,
	caption: str,
	label: str,
) -> None:
	column_count = len(dataframe.columns)
	column_alignment = "l" + "c" * max(column_count - 1, 0)
	header_row = " & ".join(_latex_escape(str(column_name)) for column_name in dataframe.columns) + r" \\"
	body_rows = [
		" & ".join(_latex_cell(value) for value in row) + r" \\"
		for row in dataframe.itertuples(index=False, name=None)
	]
	latex_lines = [
		r"\begin{table}[htbp]",
		r"\centering",
		f"\\caption{{{_latex_escape(caption)}}}",
		f"\\label{{{_latex_escape(label)}}}",
		f"\\begin{{tabular}}{{{column_alignment}}}",
		r"\toprule",
		header_row,
		r"\midrule",
	]
	latex_lines.extend(body_rows)
	latex_lines.extend([
		r"\bottomrule",
		r"\end{tabular}",
		r"\end{table}",
		"",
	])
	latex_table = "\n".join(latex_lines)
	output_path.write_text(latex_table, encoding="utf-8")


def _table1_summary(completed_df: pd.DataFrame, failed_df: pd.DataFrame) -> pd.DataFrame:
	if completed_df.empty and failed_df.empty:
		return pd.DataFrame(
			columns=[
				"model_name",
				"sass_configuration",
				"mean_percent_diff",
				"median_percent_diff",
				"completed_samples",
				"failed_samples",
			]
		)

	if completed_df.empty or "model_name" not in completed_df.columns or "use_sass" not in completed_df.columns:
		completed_summary = pd.DataFrame(columns=["model_name", "use_sass", "mean_percent_diff", "median_percent_diff", "completed_samples"])
	else:
		completed_summary = (
			completed_df.groupby(["model_name", "use_sass"], dropna=False)
			.agg(
				mean_percent_diff=("sample_mean_pct_diff", "mean"),
				median_percent_diff=("sample_mean_pct_diff", "median"),
				completed_samples=("thread_id", "count"),
			)
			.reset_index()
		)

	if failed_df.empty or "model_name" not in failed_df.columns or "use_sass" not in failed_df.columns:
		failed_summary = pd.DataFrame(columns=["model_name", "use_sass", "failed_samples"])
	else:
		failed_summary = (
			failed_df.groupby(["model_name", "use_sass"], dropna=False)
			.agg(failed_samples=("thread_id", "count"))
			.reset_index()
		)

	summary = completed_summary.merge(failed_summary, on=["model_name", "use_sass"], how="outer")
	summary["completed_samples"] = summary["completed_samples"].fillna(0).astype(int)
	summary["failed_samples"] = summary["failed_samples"].fillna(0).astype(int)
	summary["use_sass"] = summary["use_sass"].map({True: "With SASS", False: "Without SASS"}).fillna("Unknown")
	summary = summary.rename(columns={"use_sass": "sass_configuration"})
	summary["mean_percent_diff"] = summary["mean_percent_diff"].round(4)
	summary["median_percent_diff"] = summary["median_percent_diff"].round(4)
	return summary.sort_values(["model_name", "sass_configuration"]).reset_index(drop=True)


def _table2_best_worst(completed_df: pd.DataFrame) -> pd.DataFrame:
	if completed_df.empty:
		return pd.DataFrame(
			columns=[
				"model_name",
				"sass_configuration",
				"rank_group",
				"rank",
				"program_name",
				"kernel_mangled_name",
				"mean_percent_diff",
				"query_time",
				"cost_usd",
			]
		)

	rows: List[Dict[str, Any]] = []
	for (model_name, use_sass), group in completed_df.groupby(["model_name", "use_sass"], dropna=False):
		sortable = group.dropna(subset=["sample_mean_pct_diff"]).sort_values("sample_mean_pct_diff", ascending=True)
		best = sortable.head(5)
		worst = sortable.tail(5).sort_values("sample_mean_pct_diff", ascending=False)

		for rank, (_, sample) in enumerate(best.iterrows(), start=1):
			rows.append(
				{
					"model_name": model_name,
					"sass_configuration": "With SASS" if use_sass else "Without SASS",
					"rank_group": "best",
					"rank": rank,
					"program_name": sample["program_name"],
					"kernel_mangled_name": sample["kernel_mangled_name"],
					"mean_percent_diff": round(float(sample["sample_mean_pct_diff"]), 4),
					"query_time": round(float(sample["query_time"]), 4) if pd.notna(sample["query_time"]) else np.nan,
					"cost_usd": round(float(sample["cost_usd"]), 8) if pd.notna(sample["cost_usd"]) else np.nan,
				}
			)

		for rank, (_, sample) in enumerate(worst.iterrows(), start=1):
			rows.append(
				{
					"model_name": model_name,
					"sass_configuration": "With SASS" if use_sass else "Without SASS",
					"rank_group": "worst",
					"rank": rank,
					"program_name": sample["program_name"],
					"kernel_mangled_name": sample["kernel_mangled_name"],
					"mean_percent_diff": round(float(sample["sample_mean_pct_diff"]), 4),
					"query_time": round(float(sample["query_time"]), 4) if pd.notna(sample["query_time"]) else np.nan,
					"cost_usd": round(float(sample["cost_usd"]), 8) if pd.notna(sample["cost_usd"]) else np.nan,
				}
			)

	return pd.DataFrame(rows)


def _save_table2_group_figures(table2_df: pd.DataFrame, output_dir: Path) -> None:
	if table2_df.empty:
		return

	grouped_dir = output_dir / "table2_by_model"
	grouped_dir.mkdir(parents=True, exist_ok=True)
	for (model_name, sass_configuration), group in table2_df.groupby(["model_name", "sass_configuration"]):
		safe_name = _safe_filename(f"{model_name}_{sass_configuration}")
		_save_table_figure(
			group.reset_index(drop=True),
			f"Best and Worst Predictions: {model_name} | {sass_configuration}",
			grouped_dir / f"{safe_name}.png",
		)


def _write_suggestions(output_path: Path) -> None:
	suggestions = [
		"Scatter plot of cost_usd versus sample_mean_pct_diff to show the accuracy-cost frontier for each model and SASS setting.",
		"ECDF plots for query_time and cost_usd to compare tail latency and tail cost behavior across models.",
		"Heatmap of median percent error by program_name and model_name to find benchmarks that are consistently easy or hard.",
		"Boxplots of sample_mean_pct_diff grouped by GPU target to show whether some architectures are systematically harder to predict.",
		"Stacked bar chart of failure rate by model_name and use_sass to separate accuracy improvements from reliability regressions.",
		"Pairplot or correlation heatmap for token counts, query_time, cost_usd, and sample_mean_pct_diff to expose tradeoffs.",
	]
	output_path.write_text("\n".join(f"- {item}" for item in suggestions) + "\n", encoding="utf-8")


def build_visualizations(db_uri: str, output_dir: Path, include_dry_run: bool) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	parser = CheckpointDBParser(db_uri)
	attempt_tracker = QueryAttemptTracker(db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoints = _tail_checkpoint_by_thread(checkpoints)
		for checkpoint in tail_checkpoints.values():
			channel_values = checkpoint["checkpoint"]["channel_values"]
			if "total_tokens" in channel_values:
				parser.hydrate_checkpoint_channels(
					checkpoint,
					["metrics_diff", "metrics_pct_diff", "metrics_explanations"],
				)
		attempts = attempt_tracker.fetch_all_attempts()
	finally:
		parser.close()
		attempt_tracker.close()

	stored_thread_ids = _stored_thread_ids(checkpoints, attempts)
	if not stored_thread_ids:
		raise RuntimeError("No checkpoint or query-attempt records were found in the database.")

	if not include_dry_run:
		non_dry_run_thread_ids = {
			thread_id for thread_id in stored_thread_ids if not _is_dry_run_thread(thread_id)
		}
		if not non_dry_run_thread_ids:
			raise RuntimeError(
				"The database currently contains only dry-run thread IDs. "
				"Re-run with --includeDryRun or populate the database with non-dry experiment runs."
			)

	samples_df = _database_dataframe(checkpoints, attempts, include_dry_run)

	if samples_df.empty:
		raise RuntimeError("No matching checkpoint or failed-attempt records were found in the database.")

	print("\nParsed database dataframe head:")
	print(samples_df.head(10).to_string(index=False))
	print("\nParsed database dataframe columns:")
	for column_name in samples_df.columns:
		print(column_name)

	completed_df = samples_df[samples_df["status"] == "completed"].copy()
	failed_df = samples_df[samples_df["status"] == "failed"].copy()

	if not completed_df.empty:
		completed_df["use_sass"] = completed_df["use_sass"].astype(bool)
	if not failed_df.empty:
		failed_df["use_sass"] = failed_df["use_sass"].astype(bool)
	samples_df["use_sass"] = samples_df["use_sass"].fillna(False).astype(bool)

	plot1_path = output_dir / "plot1_sample_counts_by_model.png"
	_save_stacked_sample_count_plot(samples_df, plot1_path)

	plot2_path = output_dir / "plot2_query_time_distribution.png"
	_save_histogram_by_sass(
		completed_df,
		"query_time",
		"Query Time Distribution by Model and SASS Configuration",
		"Query Time (seconds)",
		plot2_path,
	)

	plot3_path = output_dir / "plot3_cost_distribution.png"
	_save_histogram_by_sass(
		completed_df,
		"cost_usd",
		"Query Cost Distribution by Model and SASS Configuration",
		"Cost (USD)",
		plot3_path,
	)

	metric_diff_df = _prepare_metric_long_df(completed_df, "metrics_diff")
	metric_pct_diff_df = _prepare_metric_long_df(completed_df, "metrics_pct_diff")

	plot4a_path = output_dir / "plot4a_metrics_diff_distribution.png"
	_save_metric_hist_grid(
		metric_diff_df,
		"Metric Difference Distribution by Model and SASS Configuration",
		"Predicted - Expected",
		plot4a_path,
	)

	plot4b_path = output_dir / "plot4b_metrics_pct_diff_distribution.png"
	_save_metric_hist_grid(
		metric_pct_diff_df,
		"Metric Percent Difference Distribution by Model and SASS Configuration",
		"Absolute Percent Difference",
		plot4b_path,
	)

	table1_df = _table1_summary(completed_df, failed_df)
	table1_tex = output_dir / "table1_model_percent_diff_summary.tex"
	table1_png = output_dir / "table1_model_percent_diff_summary.png"
	_write_booktabs_latex_table(
		table1_df,
		table1_tex,
		"Mean and Median Percent Difference by Model and SASS",
		"tab:model_percent_diff_summary",
	)
	_save_table_figure(table1_df, "Table 1: Mean and Median Percent Difference by Model and SASS", table1_png)

	table2_df = _table2_best_worst(completed_df)
	table2_csv = output_dir / "table2_best_and_worst_predictions.csv"
	table2_df.to_csv(table2_csv, index=False)
	_save_table2_group_figures(table2_df, output_dir)

	suggestions_path = output_dir / "other_visualizations.md"
	_write_suggestions(suggestions_path)

	print("Visualization artifacts written to:")
	for path in [
		plot1_path,
		plot2_path,
		plot3_path,
		plot4a_path,
		plot4b_path,
		table1_tex,
		table1_png,
		table2_csv,
		suggestions_path,
	]:
		print(f"- {path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize direct prompting results from PostgreSQL checkpoints")
	parser.add_argument("--dbUri", type=str, default=None, help="Explicit PostgreSQL database URI. Defaults to the local gpuflops_db database.")
	parser.add_argument(
		"--outputDir",
		type=str,
		default=os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "visualization-output"),
		help="Directory where plots and tables will be written.",
	)
	parser.add_argument(
		"--includeDryRun",
		action="store_true",
		help="Include dry-run thread IDs in the visualizations. By default they are excluded.",
	)
	args = parser.parse_args()

	ensure_postgres_running()
	db_uri = args.dbUri or setup_default_database()
	build_visualizations(db_uri, Path(args.outputDir), args.includeDryRun)


if __name__ == "__main__":
	main()
