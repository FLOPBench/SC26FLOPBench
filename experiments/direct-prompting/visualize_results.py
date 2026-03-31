import argparse
import ast
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
import matplotlib.transforms as mtransforms

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
	r"_(?P<gpu>A100|3080|H100|A10)_(?P<safe_model>.+)_(?P<config>withsass|nosass|sass_imix|sass_noimix|nosass_imix|nosass_noimix)_trial(?P<trial>\d+)(?:_DRYRUN(?:\d+)?)?$"
)
MODEL_DATE_SUFFIX_PATTERN = re.compile(r"-\d{8}$")
DEFAULT_EVIDENCE_CONFIGURATION = "No SASS / No IMIX"
PLOT_EVIDENCE_CONFIGURATION_LABELS = {
	(False, False): "Source-Only",
	(True, False): "Source+SASS",
	(False, True): "Source+IMIX",
	(True, True): "Source+SASS+IMIX",
}

METRIC_LABELS = {
	"fp16": "FP16 FLOPs",
	"fp32": "FP32 FLOPs",
	"fp64": "FP64 FLOPs",
	"read_bytes": "Read Bytes",
	"write_bytes": "Write Bytes",
	"block_size": "Block Size",
	"grid_size": "Grid Size",
}

COMPLETED_RECORD_COLUMNS = [
	"thread_id",
	"status",
	"program_name",
	"runtime",
	"kernel_mangled_name",
	"kernel_demangled_name",
	"model_name",
	"safe_model_name",
	"use_sass",
	"use_imix",
	"evidence_configuration",
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
	"expected_block_size",
	"expected_grid_size",
	"predicted_block_size",
	"predicted_grid_size",
	"metrics_diff_fp16",
	"metrics_diff_fp32",
	"metrics_diff_fp64",
	"metrics_diff_read_bytes",
	"metrics_diff_write_bytes",
	"metrics_diff_block_size",
	"metrics_diff_grid_size",
	"sample_mean_pct_diff",
	"metrics_pct_diff_fp16",
	"metrics_pct_diff_fp32",
	"metrics_pct_diff_fp64",
	"metrics_pct_diff_read_bytes",
	"metrics_pct_diff_write_bytes",
	"metrics_pct_diff_block_size",
	"metrics_pct_diff_grid_size",
]

FAILED_RECORD_COLUMNS = [
	"thread_id",
	"status",
	"program_name",
	"runtime",
	"kernel_mangled_name",
	"kernel_demangled_name",
	"model_name",
	"safe_model_name",
	"use_sass",
	"use_imix",
	"evidence_configuration",
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


def _bounded_plot_height(
	item_count: int,
	*,
	min_height: float,
	per_item: float,
	padding: float,
	max_height: float,
) -> float:
	return min(max_height, max(min_height, per_item * max(item_count, 1) + padding))


def _normalize_model_name(value: Optional[str]) -> str:
	if not value:
		return "unknown-model"
	return MODEL_DATE_SUFFIX_PATTERN.sub("", value)


def _display_model_name(safe_model_name: str) -> str:
	return _normalize_model_name(safe_model_name.replace("_", "/"))


def _runtime_from_program_name(program_name: Optional[str]) -> str:
	if not program_name:
		return "unknown"
	if program_name.endswith("-cuda"):
		return "cuda"
	if program_name.endswith("-omp"):
		return "omp"
	return "unknown"


def _is_dry_run_thread(thread_id: str) -> bool:
	return "_DRYRUN" in thread_id or thread_id.endswith("_DRYRUN")


def _parse_dim_triplet(value: Any) -> Optional[List[int]]:
	if isinstance(value, (list, tuple)) and len(value) == 3:
		try:
			return [int(component) for component in value]
		except (TypeError, ValueError):
			return None

	if isinstance(value, str):
		try:
			parsed = ast.literal_eval(value)
		except (SyntaxError, ValueError):
			return None

		if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
			try:
				return [int(component) for component in parsed]
			except (TypeError, ValueError):
				return None

	return None


def _dim_total(value: Any) -> float:
	triplet = _parse_dim_triplet(value)
	if triplet is None:
		return float("nan")
	return float(math.prod(triplet))


def _percent_diff(expected_value: float, predicted_value: float) -> float:
	if not np.isfinite(expected_value) or not np.isfinite(predicted_value):
		return float("nan")
	if expected_value == 0:
		return 0.0 if predicted_value == 0 else float("inf")
	return float(abs(predicted_value - expected_value) / expected_value * 100.0)


def _dimension_metrics(state: Dict[str, Any]) -> Dict[str, float]:
	prediction = state["prediction"] if "prediction" in state and isinstance(state["prediction"], dict) else {}
	expected_block_size = _dim_total(state.get("expected_block_size"))
	expected_grid_size = _dim_total(state.get("expected_grid_size"))
	predicted_block_size = _dim_total(prediction.get("blockSz"))
	predicted_grid_size = _dim_total(prediction.get("gridSz"))
	block_diff = predicted_block_size - expected_block_size if np.isfinite(expected_block_size) and np.isfinite(predicted_block_size) else float("nan")
	grid_diff = predicted_grid_size - expected_grid_size if np.isfinite(expected_grid_size) and np.isfinite(predicted_grid_size) else float("nan")
	return {
		"expected_block_size": expected_block_size,
		"expected_grid_size": expected_grid_size,
		"predicted_block_size": predicted_block_size,
		"predicted_grid_size": predicted_grid_size,
		"metrics_diff_block_size": block_diff,
		"metrics_diff_grid_size": grid_diff,
		"metrics_pct_diff_block_size": _percent_diff(expected_block_size, predicted_block_size),
		"metrics_pct_diff_grid_size": _percent_diff(expected_grid_size, predicted_grid_size),
	}


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
			"use_imix": None,
			"evidence_configuration": "Unknown",
			"trial": None,
		}

	safe_model_name = match.group("safe_model")
	config = match.group("config")
	if config == "withsass":
		use_sass = True
		use_imix = True
	elif config == "nosass":
		use_sass = False
		use_imix = False
	else:
		use_sass = config.startswith("sass_")
		use_imix = config.endswith("_imix")

	if use_sass and use_imix:
		evidence_configuration = "SASS + IMIX"
	elif use_sass:
		evidence_configuration = "SASS Only"
	elif use_imix:
		evidence_configuration = "IMIX Only"
	else:
		evidence_configuration = DEFAULT_EVIDENCE_CONFIGURATION

	return {
		"gpu": match.group("gpu"),
		"safe_model_name": _normalize_model_name(safe_model_name),
		"model_name": _display_model_name(safe_model_name),
		"use_sass": use_sass,
		"use_imix": use_imix,
		"evidence_configuration": evidence_configuration,
		"trial": int(match.group("trial")),
	}


def _require_mapping_keys(mapping: Dict[str, Any], required_keys: List[str], context: str) -> None:
	missing_keys = [key for key in required_keys if key not in mapping]
	if missing_keys:
		available_keys = sorted(mapping.keys())
		raise KeyError(
			f"{context} is missing required keys {missing_keys}. Available keys: {available_keys}"
		)


def _completed_checkpoint_by_thread(checkpoints: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
	completed: Dict[str, Dict[str, Any]] = {}
	for thread_id, checkpoint in checkpoints.items():
		state = checkpoint["checkpoint"]["channel_values"]
		if "total_tokens" in state:
			completed[thread_id] = checkpoint
	return completed


def _print_invalid_thread_warnings(invalid_threads: List[Dict[str, Any]]) -> None:
	if not invalid_threads:
		return

	reason_counts: Dict[str, int] = {}
	for item in invalid_threads:
		reason_counts[item["kind"]] = reason_counts.get(item["kind"], 0) + 1

	print("\nWarning: skipped malformed checkpoint histories during visualization:", file=sys.stderr)
	for reason, count in sorted(reason_counts.items()):
		print(f"- {reason}: {count}", file=sys.stderr)

	for item in invalid_threads[:5]:
		print(f"- {item['thread_id']}: {item['message']}", file=sys.stderr)

	remaining = len(invalid_threads) - min(len(invalid_threads), 5)
	if remaining > 0:
		print(f"- ... {remaining} more malformed thread(s) skipped", file=sys.stderr)


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
		metrics_diff = dict(state["metrics_diff"])
		metrics_pct_diff = dict(state["metrics_pct_diff"])
		dimension_metrics = _dimension_metrics(state)
		metrics_diff["block_size"] = dimension_metrics["metrics_diff_block_size"]
		metrics_diff["grid_size"] = dimension_metrics["metrics_diff_grid_size"]
		metrics_pct_diff["block_size"] = dimension_metrics["metrics_pct_diff_block_size"]
		metrics_pct_diff["grid_size"] = dimension_metrics["metrics_pct_diff_grid_size"]
		runtime = _runtime_from_program_name(state["program_name"])

		record: Dict[str, Any] = {
			"thread_id": thread_id,
			"status": "completed",
			"program_name": state["program_name"],
			"runtime": runtime,
			"kernel_mangled_name": state["kernel_mangled_name"],
			"kernel_demangled_name": state["kernel_demangled_name"],
			"model_name": _normalize_model_name(state["llm_model_name"] or metadata["model_name"]),
			"safe_model_name": metadata["safe_model_name"],
			"use_sass": metadata["use_sass"],
				"use_imix": metadata["use_imix"],
				"evidence_configuration": metadata["evidence_configuration"],
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
			"expected_block_size": pd.to_numeric(dimension_metrics["expected_block_size"], errors="coerce"),
			"expected_grid_size": pd.to_numeric(dimension_metrics["expected_grid_size"], errors="coerce"),
			"predicted_block_size": pd.to_numeric(dimension_metrics["predicted_block_size"], errors="coerce"),
			"predicted_grid_size": pd.to_numeric(dimension_metrics["predicted_grid_size"], errors="coerce"),
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
		try:
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
		except KeyError as error:
			print(f"Warning: skipping failed thread {thread_id}: {error}", file=sys.stderr)
			continue
		records.append(
			{
				"thread_id": thread_id,
				"status": "failed",
				"program_name": partial_state["program_name"],
				"runtime": _runtime_from_program_name(partial_state["program_name"]),
				"kernel_mangled_name": partial_state["kernel_mangled_name"],
				"kernel_demangled_name": partial_state["kernel_demangled_name"],
				"model_name": _normalize_model_name(partial_state["llm_model_name"] if "llm_model_name" in partial_state else metadata["model_name"]),
				"safe_model_name": metadata["safe_model_name"],
				"use_sass": metadata["use_sass"],
				"use_imix": metadata["use_imix"],
				"evidence_configuration": metadata["evidence_configuration"],
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


def _database_dataframe(latest_checkpoints: Dict[str, Dict[str, Any]], attempts: Dict[str, Dict[str, Any]], include_dry_run: bool) -> pd.DataFrame:
	completed_checkpoints = _completed_checkpoint_by_thread(latest_checkpoints)

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
	if "use_imix" in full_df.columns:
		full_df["use_imix"] = full_df["use_imix"].fillna(False)
	if "evidence_configuration" in full_df.columns:
		full_df["evidence_configuration"] = full_df["evidence_configuration"].fillna(DEFAULT_EVIDENCE_CONFIGURATION)
	if "runtime" in full_df.columns:
		full_df["runtime"] = full_df["runtime"].fillna("unknown")

	return full_df


def _plot_evidence_configuration_label(use_sass: Any, use_imix: Any) -> str:
	return PLOT_EVIDENCE_CONFIGURATION_LABELS[(bool(use_sass), bool(use_imix))]


def _plot_evidence_configuration_order(include_imix: bool) -> List[str]:
	order = [
		PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)],
		PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, False)],
	]
	if include_imix:
		order.extend(
			[
				PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, True)],
				PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, True)],
			]
		)
	return order


def _prepare_plot_dataframe(dataframe: pd.DataFrame, include_imix: bool) -> pd.DataFrame:
	if dataframe.empty:
		return dataframe.copy()

	plot_df = dataframe.copy()
	if "use_sass" in plot_df.columns:
		plot_df["use_sass"] = plot_df["use_sass"].fillna(False).astype(bool)
	if "use_imix" in plot_df.columns:
		plot_df["use_imix"] = plot_df["use_imix"].fillna(False).astype(bool)
	if not include_imix and "use_imix" in plot_df.columns:
		plot_df = plot_df[plot_df["use_imix"] == False].copy()
	if "use_sass" in plot_df.columns and "use_imix" in plot_df.columns:
		plot_df["evidence_configuration"] = plot_df.apply(
			lambda row: _plot_evidence_configuration_label(row["use_sass"], row["use_imix"]),
			axis=1,
		)
	return plot_df


def _prepare_metric_long_df(completed_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	columns = ["runtime", "model_name", "use_sass", "use_imix", "evidence_configuration", "metric", "value"]
	if completed_df.empty:
		return pd.DataFrame(columns=columns)

	for _, row in completed_df.iterrows():
		for metric_key, metric_label in METRIC_LABELS.items():
			value = row[f"{prefix}_{metric_key}"]
			if pd.isna(value):
				continue
			rows.append(
				{
					"runtime": row["runtime"],
					"model_name": row["model_name"],
					"use_sass": row["use_sass"],
					"use_imix": row["use_imix"],
					"evidence_configuration": row["evidence_configuration"],
					"metric": metric_label,
					"value": float(value),
				}
			)
	return pd.DataFrame(rows, columns=columns)


def _prepare_signed_metric_pct_long_df(completed_df: pd.DataFrame) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	columns = ["runtime", "model_name", "use_sass", "use_imix", "evidence_configuration", "metric", "value"]
	if completed_df.empty:
		return pd.DataFrame(columns=columns)

	expected_column_by_metric = {
		"fp16": "expected_fp16",
		"fp32": "expected_fp32",
		"fp64": "expected_fp64",
		"read_bytes": "expected_read_bytes",
		"write_bytes": "expected_write_bytes",
		"block_size": "expected_block_size",
		"grid_size": "expected_grid_size",
	}

	for _, row in completed_df.iterrows():
		for metric_key, metric_label in METRIC_LABELS.items():
			diff_value = pd.to_numeric(row[f"metrics_diff_{metric_key}"], errors="coerce")
			expected_value = pd.to_numeric(row[expected_column_by_metric[metric_key]], errors="coerce")
			if pd.isna(diff_value) or pd.isna(expected_value) or not np.isfinite(diff_value) or not np.isfinite(expected_value):
				continue
			if expected_value == 0:
				continue
			rows.append(
				{
					"runtime": row["runtime"],
					"model_name": row["model_name"],
					"use_sass": row["use_sass"],
					"use_imix": row["use_imix"],
					"evidence_configuration": row["evidence_configuration"],
					"metric": metric_label,
					"value": float(diff_value / expected_value * 100.0),
				}
			)
	return pd.DataFrame(rows, columns=columns)


def _models_with_completed_runs(samples_df: pd.DataFrame) -> set[str]:
	if samples_df.empty or "model_name" not in samples_df.columns or "status" not in samples_df.columns:
		return set()
	completed_models = samples_df.loc[samples_df["status"] == "completed", "model_name"].dropna().unique().tolist()
	return set(completed_models)


def _filter_plot_models(dataframe: pd.DataFrame, allowed_models: set[str]) -> pd.DataFrame:
	if dataframe.empty or not allowed_models or "model_name" not in dataframe.columns:
		return dataframe.iloc[0:0].copy() if "model_name" in dataframe.columns else dataframe.copy()
	return dataframe[dataframe["model_name"].isin(allowed_models)].copy()


def _sorted_model_names(dataframe: pd.DataFrame) -> List[str]:
	if dataframe.empty or "model_name" not in dataframe.columns:
		return []
	return sorted(dataframe["model_name"].dropna().unique().tolist())


def _annotate_boxplot_group_sums(
	ax: plt.Axes,
	plot_df: pd.DataFrame,
	value_column: str,
	model_order: List[str],
	hue_order: List[str],
) -> None:
	if plot_df.empty:
		return

	trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
	if len(hue_order) == 1:
		offsets = [0.0]
	else:
		step = 0.6 / max(len(hue_order) - 1, 1)
		offsets = [(-0.3 + step * index) for index in range(len(hue_order))]
	offset_by_hue = {label: offsets[index] for index, label in enumerate(hue_order)}
	palette = ["#4f4f4f", "#1f1f1f", "#5b5b5b", "#000000"]
	annotation_colors = {
		label: palette[index % len(palette)] for index, label in enumerate(hue_order)
	}
	group_sums = (
		plot_df.groupby(["model_name", "evidence_configuration"], dropna=False)[value_column].sum().to_dict()
	)

	for model_index, model_name in enumerate(model_order):
		for hue_label in hue_order:
			total_cost = group_sums.get((model_name, hue_label))
			if total_cost is None or not math.isfinite(float(total_cost)):
				continue
			ax.text(
				1.02,
				model_index + offset_by_hue[hue_label],
				f"${float(total_cost):.4f}",
				transform=trans,
				ha="left",
				va="center",
				fontsize=8,
				color=annotation_colors[hue_label],
			)


def _save_stacked_sample_count_plot(samples_df: pd.DataFrame, output_path: Path, evidence_order: List[str]) -> None:
	plot_df = samples_df.copy()
	plot_df["status_segment"] = plot_df.apply(
		lambda row: f"{row['status'].title()} | {row['evidence_configuration']}",
		axis=1,
	)

	segment_order = []
	for evidence_configuration in reversed(evidence_order):
		segment_order.append(f"Failed | {evidence_configuration}")
		segment_order.append(f"Completed | {evidence_configuration}")
	segment_colors = {
		"Completed | Source+SASS+IMIX": "#1b5e20",
		"Failed | Source+SASS+IMIX": "#b71c1c",
		"Completed | Source+SASS": "#388e3c",
		"Failed | Source+SASS": "#d32f2f",
		"Completed | Source+IMIX": "#66bb6a",
		"Failed | Source+IMIX": "#ef5350",
		"Completed | Source-Only": "#a5d6a7",
		"Failed | Source-Only": "#ffcdd2",
	}
	legend_order = []
	for evidence_configuration in evidence_order:
		legend_order.append(f"Completed | {evidence_configuration}")
		legend_order.append(f"Failed | {evidence_configuration}")
	counts = (
		plot_df.groupby(["model_name", "status_segment"]).size().unstack(fill_value=0).reindex(columns=segment_order, fill_value=0)
	)
	model_names = _sorted_model_names(plot_df)
	counts = counts.reindex(model_names, fill_value=0)

	sns.set_theme(style="whitegrid")
	fig_height = _bounded_plot_height(len(model_names), min_height=6.5, per_item=0.45, padding=1.5, max_height=9.5)
	fig, ax = plt.subplots(figsize=(10.5, fig_height))
	lefts = np.zeros(len(model_names))

	for segment in segment_order:
		values = counts[segment].to_numpy()
		bars = ax.barh(model_names, values, left=lefts, label=segment, color=segment_colors[segment], edgecolor="white")
		for bar, value, left in zip(bars, values, lefts):
			if value <= 0:
				continue
			ax.text(
				left + value / 2.0,
				bar.get_y() + bar.get_height() / 2.0,
				f"{int(value)}",
				ha="center",
				va="center",
				fontsize=8,
				color="black",
			)
		lefts = lefts + values

	for model_index, total in enumerate(lefts):
		ax.text(total + 0.5, model_index, f"{int(total)}", ha="left", va="center", fontsize=10, fontweight="bold")

	ax.set_title("Database Sample Counts by Model and Evidence Configuration")
	ax.set_xlabel("Sample Count")
	ax.set_ylabel("Model Name")
	handles, labels = ax.get_legend_handles_labels()
	handle_by_label = dict(zip(labels, handles))
	ax.legend(
		[handle_by_label[label] for label in legend_order if label in handle_by_label],
		[label for label in legend_order if label in handle_by_label],
		title="Sample Type",
		bbox_to_anchor=(1.02, 1),
		loc="upper left",
	)
	ax.tick_params(axis="y", pad=12)
	for label in ax.get_yticklabels():
		label.set_rotation(0)
		label.set_horizontalalignment("right")
	ax.margins(x=0.05)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _save_histogram_by_sass(
	completed_df: pd.DataFrame,
	value_column: str,
	title: str,
	x_label: str,
	output_path: Path,
	evidence_order: List[str],
	annotate_group_sums: bool = False,
) -> None:
	sns.set_theme(style="whitegrid")
	model_count = completed_df["model_name"].nunique() if "model_name" in completed_df.columns else 0
	fig_height = _bounded_plot_height(model_count, min_height=6.0, per_item=0.45, padding=1.25, max_height=9.0)
	fig, ax = plt.subplots(figsize=(10.5, fig_height))
	hue_order = evidence_order

	if completed_df.empty or value_column not in completed_df.columns or "evidence_configuration" not in completed_df.columns or "model_name" not in completed_df.columns:
		ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel(x_label)
		ax.set_ylabel("Model Name")
		ax.set_title(title)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		return

	plot_df = completed_df.copy()
	plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
	plot_df = plot_df[plot_df[value_column].notna()]
	plot_df = plot_df[np.isfinite(plot_df[value_column].to_numpy())]
	model_order = _sorted_model_names(plot_df)

	if plot_df.empty:
		ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
	else:
		sns.boxplot(
			data=plot_df,
			x=value_column,
			y="model_name",
				hue="evidence_configuration",
			order=model_order,
			hue_order=hue_order,
			orient="h",
			ax=ax,
		)
		if annotate_group_sums:
			_annotate_boxplot_group_sums(ax, plot_df, value_column, model_order, hue_order)

	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel("Model Name")
	legend = ax.get_legend()
	if legend is not None:
		legend.set_title("Evidence Configuration")
	if annotate_group_sums:
		fig.tight_layout(rect=(0, 0, 0.9, 1))
	else:
		fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _apply_shared_metric_x_limits(
	axes: np.ndarray,
	*,
	x_left_limit: Optional[float] = None,
) -> None:
	x_limits: List[tuple[float, float]] = []
	for ax in axes.flatten():
		if not ax.has_data():
			continue
		left, right = ax.get_xlim()
		if not np.isfinite(left) or not np.isfinite(right):
			continue
		x_limits.append((float(left), float(right)))

	if not x_limits:
		return

	shared_left = min(left for left, _ in x_limits)
	shared_right = max(right for _, right in x_limits)
	if x_left_limit is not None:
		shared_left = float(x_left_limit)
	if shared_left == shared_right:
		shared_right = shared_left + 1.0

	for ax in axes.flatten():
		if not ax.has_data():
			continue
		ax.set_xlim(shared_left, shared_right)


def _save_metric_hist_grid(
	metric_df: pd.DataFrame,
	title: str,
	x_label: str,
	output_path: Path,
	evidence_order: List[str],
	x_left_limit: Optional[float] = None,
) -> None:
	if metric_df.empty or "model_name" not in metric_df.columns:
		sns.set_theme(style="whitegrid")
		fig, axes = plt.subplots(3, len(evidence_order), figsize=(22, 10), squeeze=False)
		for ax in axes.flatten():
			ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
			ax.set_axis_off()
		fig.suptitle(title)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		plt.close(fig)
		return

	model_names = _sorted_model_names(metric_df)
	model_count = max(len(model_names), 1)
	metric_order = list(METRIC_LABELS.values())
	runtime_rows = [
		("cuda", "CUDA"),
		("omp", "OpenMP"),
		("combined", "CUDA + OpenMP"),
	]

	sns.set_theme(style="whitegrid")
	row_height = _bounded_plot_height(model_count, min_height=3.5, per_item=0.4, padding=0.75, max_height=5.5)
	fig_height = min(16.5, max(10.5, row_height * len(runtime_rows)))
	fig_width = max(18.0, 5.0 * len(evidence_order))
	fig, axes = plt.subplots(3, len(evidence_order), figsize=(fig_width, fig_height), squeeze=False, sharey="row")
	legend_handles: List[Any] = []
	legend_labels: List[str] = []

	if not model_names:
		for ax in axes.flatten():
			ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
			ax.set_axis_off()
	else:
		for row_index, (runtime_key, runtime_label) in enumerate(runtime_rows):
			for col_index, evidence_configuration in enumerate(evidence_order):
				ax = axes[row_index][col_index]
				subset = metric_df[metric_df["evidence_configuration"] == evidence_configuration].copy()
				if runtime_key != "combined":
					subset = subset[subset["runtime"] == runtime_key]
				numeric_values = pd.to_numeric(subset["value"], errors="coerce")
				subset = subset[numeric_values.notna()]
				numeric_values = numeric_values.loc[subset.index]
				subset = subset[np.isfinite(numeric_values.to_numpy())]
				subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
				if subset.empty:
					ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes)
				else:
					sns.boxplot(
						data=subset,
						x="value",
						y="model_name",
						hue="metric",
						order=model_names,
						hue_order=metric_order,
						orient="h",
						ax=ax,
					)
					launch_subset = subset[subset["metric"].isin(["Block Size", "Grid Size"])]
					if not launch_subset.empty:
						sns.stripplot(
							data=launch_subset,
							x="value",
							y="model_name",
							hue="metric",
							order=model_names,
							hue_order=metric_order,
							orient="h",
							dodge=True,
							jitter=False,
							size=2.4,
							alpha=0.55,
							linewidth=0,
							ax=ax,
						)
				if row_index == 0:
					ax.set_title(evidence_configuration)
				ax.set_xscale("symlog", linthresh=1.0)
				ax.set_xlabel(x_label)
				ax.set_ylabel(f"{runtime_label}\nModel Name" if col_index == 0 else "")
				ax.tick_params(axis="x", labelsize=8)
				legend = ax.get_legend()
				if legend is not None:
					handles, labels = ax.get_legend_handles_labels()
					if not legend_handles:
						handle_by_label = {}
						for handle, label in zip(handles, labels):
							if label in metric_order and label not in handle_by_label:
								handle_by_label[label] = handle
						legend_labels = [label for label in metric_order if label in handle_by_label]
						legend_handles = [handle_by_label[label] for label in legend_labels]
					legend.remove()

	_apply_shared_metric_x_limits(axes, x_left_limit=x_left_limit)

	if legend_handles:
		fig.legend(
			legend_handles,
			legend_labels,
			title="Metric",
			loc="upper left",
			bbox_to_anchor=(0.87, 0.97),
		)

	fig.suptitle(title)
	fig.tight_layout(rect=(0, 0, 0.86, 0.98))
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
				"evidence_configuration",
				"mean_percent_diff",
				"median_percent_diff",
				"completed_samples",
				"failed_samples",
			]
		)

	if completed_df.empty or "model_name" not in completed_df.columns or "evidence_configuration" not in completed_df.columns:
		completed_summary = pd.DataFrame(columns=["model_name", "evidence_configuration", "mean_percent_diff", "median_percent_diff", "completed_samples"])
	else:
		completed_summary = (
			completed_df.groupby(["model_name", "evidence_configuration"], dropna=False)
			.agg(
				mean_percent_diff=("sample_mean_pct_diff", "mean"),
				median_percent_diff=("sample_mean_pct_diff", "median"),
				completed_samples=("thread_id", "count"),
			)
			.reset_index()
		)

	if failed_df.empty or "model_name" not in failed_df.columns or "evidence_configuration" not in failed_df.columns:
		failed_summary = pd.DataFrame(columns=["model_name", "evidence_configuration", "failed_samples"])
	else:
		failed_summary = (
			failed_df.groupby(["model_name", "evidence_configuration"], dropna=False)
			.agg(failed_samples=("thread_id", "count"))
			.reset_index()
		)

	summary = completed_summary.merge(failed_summary, on=["model_name", "evidence_configuration"], how="outer")
	summary["completed_samples"] = summary["completed_samples"].fillna(0).astype(int)
	summary["failed_samples"] = summary["failed_samples"].fillna(0).astype(int)
	summary["mean_percent_diff"] = summary["mean_percent_diff"].round(4)
	summary["median_percent_diff"] = summary["median_percent_diff"].round(4)
	return summary.sort_values(["model_name", "evidence_configuration"]).reset_index(drop=True)


def _table2_best_worst(completed_df: pd.DataFrame) -> pd.DataFrame:
	if completed_df.empty:
		return pd.DataFrame(
			columns=[
				"model_name",
				"evidence_configuration",
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
	for (model_name, evidence_configuration), group in completed_df.groupby(["model_name", "evidence_configuration"], dropna=False):
		sortable = group.dropna(subset=["sample_mean_pct_diff"]).sort_values("sample_mean_pct_diff", ascending=True)
		best = sortable.head(5)
		worst = sortable.tail(5).sort_values("sample_mean_pct_diff", ascending=False)

		for rank, (_, sample) in enumerate(best.iterrows(), start=1):
			rows.append(
				{
					"model_name": model_name,
					"evidence_configuration": evidence_configuration,
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
					"evidence_configuration": evidence_configuration,
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
	for (model_name, evidence_configuration), group in table2_df.groupby(["model_name", "evidence_configuration"]):
		safe_name = _safe_filename(f"{model_name}_{evidence_configuration}")
		_save_table_figure(
			group.reset_index(drop=True),
			f"Best and Worst Predictions: {model_name} | {evidence_configuration}",
			grouped_dir / f"{safe_name}.png",
		)


def _write_suggestions(output_path: Path) -> None:
	suggestions = [
		"Scatter plot of cost_usd versus sample_mean_pct_diff to show the accuracy-cost frontier for each model and evidence configuration.",
		"ECDF plots for query_time and cost_usd to compare tail latency and tail cost behavior across models.",
		"Heatmap of median percent error by program_name and model_name to find benchmarks that are consistently easy or hard.",
		"Boxplots of sample_mean_pct_diff grouped by GPU target to show whether some architectures are systematically harder to predict.",
		"Stacked bar chart of failure rate by model_name and evidence_configuration to separate accuracy improvements from reliability regressions.",
		"Pairplot or correlation heatmap for token counts, query_time, cost_usd, and sample_mean_pct_diff to expose tradeoffs.",
	]
	output_path.write_text("\n".join(f"- {item}" for item in suggestions) + "\n", encoding="utf-8")


def build_visualizations(db_uri: str, output_dir: Path, include_dry_run: bool, include_imix: bool) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	parser = CheckpointDBParser(db_uri)
	attempt_tracker = QueryAttemptTracker(db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoint_result = parser.fetch_tail_checkpoints_by_thread(checkpoints=checkpoints, tolerate_errors=True)
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

	_print_invalid_thread_warnings(invalid_threads)

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

	samples_df = _database_dataframe(tail_checkpoints, attempts, include_dry_run)

	if samples_df.empty:
		raise RuntimeError("No matching checkpoint or failed-attempt records were found in the database.")

	# print("\nParsed database dataframe head:")
	# print(samples_df.head(10).to_string(index=False))
	# print("\nParsed database dataframe columns:")
	# for column_name in samples_df.columns:
	# 	print(column_name)

	completed_df = samples_df[samples_df["status"] == "completed"].copy()
	failed_df = samples_df[samples_df["status"] == "failed"].copy()

	if not completed_df.empty:
		completed_df["use_sass"] = completed_df["use_sass"].astype(bool)
		completed_df["use_imix"] = completed_df["use_imix"].astype(bool)
	if not failed_df.empty:
		failed_df["use_sass"] = failed_df["use_sass"].astype(bool)
		failed_df["use_imix"] = failed_df["use_imix"].astype(bool)
	samples_df["use_sass"] = samples_df["use_sass"].fillna(False).astype(bool)
	samples_df["use_imix"] = samples_df["use_imix"].fillna(False).astype(bool)
	samples_df["evidence_configuration"] = samples_df["evidence_configuration"].fillna(DEFAULT_EVIDENCE_CONFIGURATION)
	plot_evidence_order = _plot_evidence_configuration_order(include_imix)
	plot_samples_df = _prepare_plot_dataframe(samples_df, include_imix)
	plot_completed_df = _prepare_plot_dataframe(completed_df, include_imix)
	plot_models = _models_with_completed_runs(plot_samples_df)
	plot_samples_df = _filter_plot_models(plot_samples_df, plot_models)
	plot_completed_df = _filter_plot_models(plot_completed_df, plot_models)

	plot1_path = output_dir / "plot1_sample_counts_by_model.png"
	_save_stacked_sample_count_plot(plot_samples_df, plot1_path, plot_evidence_order)

	plot2_path = output_dir / "plot2_query_time_distribution.png"
	_save_histogram_by_sass(
		plot_completed_df,
		"query_time",
		"Query Time Distribution by Model and Evidence Configuration",
		"Query Time (seconds)",
		plot2_path,
		plot_evidence_order,
	)

	plot3_path = output_dir / "plot3_cost_distribution.png"
	_save_histogram_by_sass(
		plot_completed_df,
		"cost_usd",
		"Query Cost Distribution by Model and Evidence Configuration",
		"Cost (USD)",
		plot3_path,
		plot_evidence_order,
		annotate_group_sums=True,
	)

	metric_diff_df = _prepare_metric_long_df(plot_completed_df, "metrics_diff")
	metric_pct_diff_df = _prepare_signed_metric_pct_long_df(plot_completed_df)

	plot4a_path = output_dir / "plot4a_metrics_diff_distribution.png"
	_save_metric_hist_grid(
		metric_diff_df,
		"Metric Difference Distribution by Model and Evidence Configuration",
		"Predicted - Expected",
		plot4a_path,
		plot_evidence_order,
	)

	plot4b_path = output_dir / "plot4b_metrics_pct_diff_distribution.png"
	_save_metric_hist_grid(
		metric_pct_diff_df,
		"Metric Percent Difference Distribution by Model and Evidence Configuration",
		"Percent Difference",
		plot4b_path,
		plot_evidence_order,
		x_left_limit=-200.0,
	)

	table1_df = _table1_summary(completed_df, failed_df)
	table1_tex = output_dir / "table1_model_percent_diff_summary.tex"
	table1_png = output_dir / "table1_model_percent_diff_summary.png"
	_write_booktabs_latex_table(
		table1_df,
		table1_tex,
		"Mean and Median Percent Difference by Model and Evidence Configuration",
		"tab:model_percent_diff_summary",
	)
	_save_table_figure(table1_df, "Table 1: Mean and Median Percent Difference by Model and Evidence Configuration", table1_png)

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
	parser.add_argument(
		"--includeIMIX",
		action="store_true",
		help="Include Source+IMIX and Source+SASS+IMIX categories in the plots. By default, plots show only Source-Only and Source+SASS.",
	)
	args = parser.parse_args()

	ensure_postgres_running()
	db_uri = args.dbUri or setup_default_database()
	build_visualizations(db_uri, Path(args.outputDir), args.includeDryRun, args.includeIMIX)


if __name__ == "__main__":
	main()
