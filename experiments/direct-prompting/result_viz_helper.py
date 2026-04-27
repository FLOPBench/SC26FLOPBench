import ast
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
FIGURE_SIZE_SCALE = 0.78
PROMPT_TYPE_LABEL = "Prompt Type"
BOTTOM_LEGEND_Y = -0.2
BOTTOM_LEGEND_RECT = 0.18
MODEL_DISPLAY_NAME_MAP = {
	"anthropic/claude-4.6-opus": "Opus 4.6",
	"openai/gpt-5.4": "GPT 5.4",
	"openai/gpt-oss-120b": "GPT OSS",
}
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


def _display_plot_model_name(value: Optional[str]) -> str:
	normalized_value = _normalize_model_name(value)
	return MODEL_DISPLAY_NAME_MAP.get(normalized_value.casefold(), normalized_value)


def _display_model_name(safe_model_name: str) -> str:
	return _display_plot_model_name(safe_model_name.replace("_", "/"))


def _scaled_figsize(width: float, height: float) -> tuple[float, float]:
	return (width * FIGURE_SIZE_SCALE, height * FIGURE_SIZE_SCALE)


def _legend_ncols(item_count: int, max_columns: int = 4) -> int:
	return max(1, min(item_count, max_columns))


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
			"model_name": _display_plot_model_name(state["llm_model_name"] or metadata["model_name"]),
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
				"model_name": _display_plot_model_name(partial_state["llm_model_name"] if "llm_model_name" in partial_state else metadata["model_name"]),
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


def _shared_sample_identity_columns() -> List[str]:
	return ["program_name", "kernel_mangled_name"]


def _shared_sample_gpu_column() -> str:
	return "gpu"


def _shared_sample_prompt_columns() -> List[str]:
	return ["use_sass", "use_imix"]


def _required_shared_prompt_configurations(include_imix: bool) -> List[tuple[bool, bool]]:
	required_configurations = [(False, False), (True, False)]
	if include_imix:
		required_configurations.extend([(False, True), (True, True)])
	return required_configurations


def _shared_sample_keys(dataframe: pd.DataFrame, include_imix: bool = False) -> pd.DataFrame:
	if dataframe.empty:
		return pd.DataFrame(columns=_shared_sample_identity_columns())

	required_columns = _shared_sample_identity_columns() + [_shared_sample_gpu_column()] + _shared_sample_prompt_columns() + ["model_name", "status"]
	missing_columns = [column for column in required_columns if column not in dataframe.columns]
	if missing_columns:
		raise KeyError(f"Shared-sample filtering requires columns {missing_columns}")

	filtered_df = dataframe.copy()
	filtered_df["use_sass"] = filtered_df["use_sass"].fillna(False).astype(bool)
	filtered_df["use_imix"] = filtered_df["use_imix"].fillna(False).astype(bool)
	filtered_df = filtered_df[filtered_df["model_name"].notna()].copy()
	if filtered_df.empty:
		return pd.DataFrame(columns=_shared_sample_identity_columns())

	required_prompt_configurations = set(_required_shared_prompt_configurations(include_imix))
	filtered_df = filtered_df[
		filtered_df.apply(lambda row: (row["use_sass"], row["use_imix"]) in required_prompt_configurations, axis=1)
	].copy()
	if filtered_df.empty:
		return pd.DataFrame(columns=_shared_sample_identity_columns())

	all_models = set(filtered_df["model_name"].dropna().unique().tolist())
	if not all_models:
		return pd.DataFrame(columns=_shared_sample_identity_columns())
	all_gpus = set(filtered_df[_shared_sample_gpu_column()].dropna().unique().tolist())
	if not all_gpus:
		return pd.DataFrame(columns=_shared_sample_identity_columns())

	identity_rows = filtered_df[
		_shared_sample_identity_columns() + [_shared_sample_gpu_column(), "model_name"] + _shared_sample_prompt_columns()
	].drop_duplicates()
	shared_sample_counts = (
		identity_rows.groupby(_shared_sample_identity_columns(), dropna=False)
		.size()
		.reset_index(name="model_count")
	)
	required_combination_count = len(all_models) * len(required_prompt_configurations) * len(all_gpus)
	shared_sample_counts = shared_sample_counts[
		shared_sample_counts["model_count"] == required_combination_count
	].copy()
	if shared_sample_counts.empty:
		return pd.DataFrame(columns=_shared_sample_identity_columns())

	return shared_sample_counts[_shared_sample_identity_columns()].drop_duplicates()


def _filter_only_shared_samples(dataframe: pd.DataFrame, include_imix: bool = False) -> pd.DataFrame:
	if dataframe.empty:
		return dataframe.copy()

	filtered_df = dataframe.copy()
	filtered_df["use_sass"] = filtered_df["use_sass"].fillna(False).astype(bool)
	filtered_df["use_imix"] = filtered_df["use_imix"].fillna(False).astype(bool)
	filtered_df = filtered_df[filtered_df["model_name"].notna()].copy()
	if filtered_df.empty:
		return filtered_df

	required_prompt_configurations = set(_required_shared_prompt_configurations(include_imix))
	filtered_df = filtered_df[
		filtered_df.apply(lambda row: (row["use_sass"], row["use_imix"]) in required_prompt_configurations, axis=1)
	].copy()
	if filtered_df.empty:
		return filtered_df

	shared_sample_keys = _shared_sample_keys(dataframe, include_imix=include_imix)
	if shared_sample_keys.empty:
		return filtered_df.iloc[0:0].copy()
	return filtered_df.merge(shared_sample_keys, on=_shared_sample_identity_columns(), how="inner")


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
