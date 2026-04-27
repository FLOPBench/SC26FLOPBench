import argparse
import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)


AI_PRECISIONS = ["fp16", "fp32", "fp64"]
DEFAULT_GPUFLOPS_DB = "gpuflops_db"
DEFAULT_CODE_FEATURES_DB = "code_features_db"
JOIN_KEY_COLUMNS = ["program_name", "kernel_mangled_name"]
FEATURE_TRIAL_PATTERN = re.compile(r"_trial(?P<trial>\d+)(?:_DRYRUN(?:\d+)?)?$")


def _load_module(module_name: str, relative_path: str):
	module_path = os.path.join(WORKSPACE_ROOT, relative_path)
	module_spec = importlib.util.spec_from_file_location(module_name, module_path)
	if module_spec is None or module_spec.loader is None:
		raise RuntimeError(f"Unable to load module {module_name} from {module_path}")

	module = importlib.util.module_from_spec(module_spec)
	module_spec.loader.exec_module(module)
	return module


direct_db_manager = _load_module(
	"error_analysis_direct_db_manager",
	os.path.join("experiments", "direct-prompting", "db_manager.py"),
)
feature_db_manager = _load_module(
	"error_analysis_feature_db_manager",
	os.path.join("experiments", "feature-voting", "db_manager.py"),
)
visualize_results = _load_module(
		"error_analysis_result_viz_helper",
		os.path.join("experiments", "direct-prompting", "result_viz_helper.py"),
)
run_voting_queries = _load_module(
	"error_analysis_run_voting_queries",
	os.path.join("experiments", "feature-voting", "run_voting_queries.py"),
)


FEATURE_FIELDS = list(run_voting_queries.FEATURE_FIELDS)
FEATURE_FLAG_COLUMNS = [fallback_key for _, fallback_key in FEATURE_FIELDS]


def _default_db_uri(db_name: str, manager_module: Any) -> str:
	manager_module.ensure_postgres_running()
	return manager_module.setup_default_database(db_name=db_name)


def default_gpuflops_db_uri() -> str:
	return _default_db_uri(DEFAULT_GPUFLOPS_DB, direct_db_manager)


def default_code_features_db_uri() -> str:
	return _default_db_uri(DEFAULT_CODE_FEATURES_DB, feature_db_manager)


def _safe_divide(numerator: Any, denominator: Any) -> float:
	numerator_value = pd.to_numeric(numerator, errors="coerce")
	denominator_value = pd.to_numeric(denominator, errors="coerce")
	if pd.isna(numerator_value) or pd.isna(denominator_value):
		return float("nan")
	if float(denominator_value) == 0.0:
		return float("nan")
	return float(numerator_value) / float(denominator_value)


def _abs_pct_error(predicted_value: Any, expected_value: Any) -> float:
	predicted_numeric = pd.to_numeric(predicted_value, errors="coerce")
	expected_numeric = pd.to_numeric(expected_value, errors="coerce")
	if pd.isna(predicted_numeric) or pd.isna(expected_numeric):
		return float("nan")

	predicted_float = float(predicted_numeric)
	expected_float = float(expected_numeric)
	if expected_float == 0.0:
		return 0.0 if predicted_float == 0.0 else float("inf")
	return abs(predicted_float - expected_float) / abs(expected_float) * 100.0


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


def _parse_trial(thread_id: str) -> Optional[int]:
	match = FEATURE_TRIAL_PATTERN.search(thread_id)
	if match is None:
		return None
	return int(match.group("trial"))


def _nonempty_first(values: Iterable[Any]) -> Any:
	for value in values:
		if pd.notna(value) and value not in ("", [], {}, ()):
			return value
	return pd.NA


def _load_tail_checkpoints(
	parser_class: Any,
	db_uri: str,
	hydrate_channels: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
	parser = parser_class(db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoint_result = parser.fetch_tail_checkpoints_by_thread(
			checkpoints=checkpoints,
			tolerate_errors=True,
		)
		tail_checkpoints = tail_checkpoint_result["tails"]
		invalid_threads = tail_checkpoint_result["invalid_threads"]
		for checkpoint in tail_checkpoints.values():
			channel_values = checkpoint["checkpoint"].get("channel_values", {})
			if "total_tokens" in channel_values:
				parser.hydrate_checkpoint_channels(checkpoint, list(hydrate_channels))
	finally:
		parser.close()
	return tail_checkpoints, invalid_threads, checkpoints


def load_gpuflops_samples_dataframe(
	db_uri: Optional[str] = None,
	*,
	include_dry_run: bool = False,
) -> pd.DataFrame:
	resolved_db_uri = db_uri or default_gpuflops_db_uri()
	parser = visualize_results.CheckpointDBParser(resolved_db_uri)
	attempt_tracker = visualize_results.QueryAttemptTracker(resolved_db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoint_result = parser.fetch_tail_checkpoints_by_thread(
			checkpoints=checkpoints,
			tolerate_errors=True,
		)
		tail_checkpoints = tail_checkpoint_result["tails"]
		invalid_threads = tail_checkpoint_result["invalid_threads"]
		for checkpoint in tail_checkpoints.values():
			channel_values = checkpoint["checkpoint"].get("channel_values", {})
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
	dataframe = visualize_results._database_dataframe(tail_checkpoints, attempts, include_dry_run)
	if dataframe.empty:
		return dataframe

	dataframe["runtime"] = dataframe["runtime"].fillna(
		dataframe["program_name"].map(_runtime_from_program_name)
	)
	return dataframe


def enrich_gpuflops_with_ai_metrics(samples_df: pd.DataFrame) -> pd.DataFrame:
	completed_df = samples_df[samples_df["status"] == "completed"].copy()
	if completed_df.empty:
		return completed_df

	numeric_columns = [
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
	]
	for column_name in numeric_columns:
		completed_df[column_name] = pd.to_numeric(completed_df[column_name], errors="coerce")

	completed_df["predicted_fp16"] = completed_df["expected_fp16"] + completed_df["metrics_diff_fp16"]
	completed_df["predicted_fp32"] = completed_df["expected_fp32"] + completed_df["metrics_diff_fp32"]
	completed_df["predicted_fp64"] = completed_df["expected_fp64"] + completed_df["metrics_diff_fp64"]
	completed_df["predicted_read_bytes"] = (
		completed_df["expected_read_bytes"] + completed_df["metrics_diff_read_bytes"]
	)
	completed_df["predicted_write_bytes"] = (
		completed_df["expected_write_bytes"] + completed_df["metrics_diff_write_bytes"]
	)
	completed_df["expected_total_bytes"] = (
		completed_df["expected_read_bytes"] + completed_df["expected_write_bytes"]
	)
	completed_df["predicted_total_bytes"] = (
		completed_df["predicted_read_bytes"] + completed_df["predicted_write_bytes"]
	)

	for precision in AI_PRECISIONS:
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
		completed_df[f"abs_ai_pct_error_{precision}"] = completed_df.apply(
			lambda row: _abs_pct_error(
				row[f"predicted_ai_{precision}"],
				row[f"expected_ai_{precision}"],
			),
			axis=1,
		)

	return completed_df


def build_sample_ai_error_long_dataframe(completed_ai_df: pd.DataFrame) -> pd.DataFrame:
	if completed_ai_df.empty:
		return pd.DataFrame()

	shared_columns = [
		"thread_id",
		"program_name",
		"kernel_mangled_name",
		"kernel_demangled_name",
		"runtime",
		"gpu",
		"model_name",
		"safe_model_name",
		"trial",
		"use_sass",
		"use_imix",
		"evidence_configuration",
	]
	frames: list[pd.DataFrame] = []
	for precision in AI_PRECISIONS:
		precision_df = completed_ai_df[
			shared_columns
			+ [
				f"expected_ai_{precision}",
				f"predicted_ai_{precision}",
				f"ai_diff_{precision}",
				f"abs_ai_pct_error_{precision}",
			]
		].copy()
		precision_df = precision_df.rename(
			columns={
				f"expected_ai_{precision}": "expected_ai",
				f"predicted_ai_{precision}": "predicted_ai",
				f"ai_diff_{precision}": "ai_diff",
				f"abs_ai_pct_error_{precision}": "abs_ai_pct_error",
			}
		)
		precision_df["precision"] = precision
		frames.append(precision_df)

	ai_error_df = pd.concat(frames, ignore_index=True)
	ai_error_df["abs_ai_pct_error"] = pd.to_numeric(ai_error_df["abs_ai_pct_error"], errors="coerce")
	return ai_error_df


def summarize_kernel_ai_errors(
	sample_ai_error_df: pd.DataFrame,
	*,
	selected_metric: str = "max_abs_ai_pct_error",
) -> pd.DataFrame:
	if sample_ai_error_df.empty:
		return pd.DataFrame()

	group_columns = JOIN_KEY_COLUMNS + ["kernel_demangled_name", "runtime", "precision"]
	records: list[dict[str, Any]] = []

	for group_key, group_df in sample_ai_error_df.groupby(group_columns, dropna=False):
		group_df = group_df.copy()
		numeric_errors = pd.to_numeric(group_df["abs_ai_pct_error"], errors="coerce")
		finite_error_mask = np.isfinite(numeric_errors.to_numpy())
		finite_errors = numeric_errors[finite_error_mask]

		record = {
			"program_name": group_key[0],
			"kernel_mangled_name": group_key[1],
			"kernel_demangled_name": group_key[2],
			"runtime": group_key[3],
			"precision": group_key[4],
			"sample_count": int(len(group_df)),
			"finite_error_sample_count": int(finite_errors.shape[0]),
			"max_abs_ai_pct_error": float(finite_errors.max()) if not finite_errors.empty else float("nan"),
			"median_abs_ai_pct_error": float(finite_errors.median()) if not finite_errors.empty else float("nan"),
			"mean_abs_ai_pct_error": float(finite_errors.mean()) if not finite_errors.empty else float("nan"),
		}

		if not finite_errors.empty:
			worst_row = group_df.loc[finite_errors.idxmax()]
			record.update(
				{
					"worst_thread_id": worst_row.get("thread_id"),
					"worst_model_name": worst_row.get("model_name"),
					"worst_gpu": worst_row.get("gpu"),
					"worst_trial": worst_row.get("trial"),
				}
			)
		else:
			record.update(
				{
					"worst_thread_id": pd.NA,
					"worst_model_name": pd.NA,
					"worst_gpu": pd.NA,
					"worst_trial": pd.NA,
				}
			)

		records.append(record)

	summary_df = pd.DataFrame(records)
	if summary_df.empty:
		return summary_df

	if selected_metric not in summary_df.columns:
		raise KeyError(f"Unknown selected metric '{selected_metric}'")
	summary_df["selected_abs_ai_pct_error"] = summary_df[selected_metric]
	return summary_df.sort_values(
		["precision", "selected_abs_ai_pct_error"],
		ascending=[True, False],
		na_position="last",
	).reset_index(drop=True)


def load_code_feature_vote_dataframe(
	db_uri: Optional[str] = None,
	*,
	include_dry_run: bool = False,
) -> pd.DataFrame:
	resolved_db_uri = db_uri or default_code_features_db_uri()
	parser = feature_db_manager.CheckpointDBParser(resolved_db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoint_result = parser.fetch_tail_checkpoints_by_thread(
			checkpoints=checkpoints,
			tolerate_errors=True,
		)
		tail_checkpoints = tail_checkpoint_result["tails"]
		invalid_threads = tail_checkpoint_result["invalid_threads"]
		for checkpoint in tail_checkpoints.values():
			channel_values = checkpoint["checkpoint"].get("channel_values", {})
			if "total_tokens" in channel_values:
				parser.hydrate_checkpoint_channels(checkpoint, ["prediction"])
	finally:
		parser.close()

	if invalid_threads:
		print(f"Warning: skipped {len(invalid_threads)} invalid code-features checkpoint threads.", file=sys.stderr)

	records: list[dict[str, Any]] = []
	for thread_id, checkpoint in tail_checkpoints.items():
		if not include_dry_run and _is_dry_run_thread(thread_id):
			continue

		channel_values = checkpoint["checkpoint"].get("channel_values", {})
		if "total_tokens" not in channel_values:
			continue

		record: dict[str, Any] = {
			"thread_id": thread_id,
			"program_name": channel_values.get("program_name"),
			"runtime": _runtime_from_program_name(channel_values.get("program_name")),
			"kernel_mangled_name": channel_values.get("kernel_mangled_name"),
			"kernel_demangled_name": channel_values.get("kernel_demangled_name"),
			"model_name": channel_values.get("llm_model_name"),
			"trial": _parse_trial(thread_id),
			"query_time": pd.to_numeric(channel_values.get("query_time"), errors="coerce"),
			"cost_usd": pd.to_numeric(channel_values.get("cost_usd"), errors="coerce"),
			"total_tokens": pd.to_numeric(channel_values.get("total_tokens"), errors="coerce"),
		}

		for predicted_key, fallback_key in FEATURE_FIELDS:
			record[fallback_key] = run_voting_queries._extract_feature_vote(
				channel_values,
				predicted_key,
				fallback_key,
			)
		records.append(record)

	dataframe = pd.DataFrame(records)
	if dataframe.empty:
		return dataframe

	for feature_column in FEATURE_FLAG_COLUMNS:
		dataframe[feature_column] = dataframe[feature_column].astype("boolean")
	return dataframe


def aggregate_feature_votes(vote_df: pd.DataFrame) -> pd.DataFrame:
	if vote_df.empty:
		return pd.DataFrame()

	records: list[dict[str, Any]] = []
	group_columns = JOIN_KEY_COLUMNS
	for group_key, group_df in vote_df.groupby(group_columns, dropna=False):
		record: dict[str, Any] = {
			"program_name": group_key[0],
			"kernel_mangled_name": group_key[1],
			"kernel_demangled_name": _nonempty_first(group_df["kernel_demangled_name"].tolist()),
			"runtime": _nonempty_first(group_df["runtime"].tolist()),
			"vote_record_count": int(len(group_df)),
			"model_count": int(group_df["model_name"].dropna().nunique()),
		}
		for feature_column in FEATURE_FLAG_COLUMNS:
			feature_votes = group_df[feature_column]
			total_votes = int(feature_votes.notna().sum())
			yes_votes = int(feature_votes.fillna(False).astype(bool).sum())
			record[f"yes_votes_{feature_column}"] = yes_votes
			record[f"total_votes_{feature_column}"] = total_votes
			if total_votes == 0:
				record[feature_column] = pd.NA
				record[f"vote_fraction_{feature_column}"] = float("nan")
			else:
				vote_fraction = yes_votes / total_votes
				record[feature_column] = vote_fraction > 0.5
				record[f"vote_fraction_{feature_column}"] = vote_fraction
		records.append(record)

	dataframe = pd.DataFrame(records)
	for feature_column in FEATURE_FLAG_COLUMNS:
		dataframe[feature_column] = dataframe[feature_column].astype("boolean")
	return dataframe.sort_values(JOIN_KEY_COLUMNS).reset_index(drop=True)


def merge_sample_errors_with_feature_flags(
	sample_ai_error_df: pd.DataFrame,
	kernel_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
	return _merge_with_features(sample_ai_error_df, kernel_feature_df)


def merge_kernel_errors_with_feature_flags(
	kernel_error_df: pd.DataFrame,
	kernel_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
	return _merge_with_features(kernel_error_df, kernel_feature_df)


def _merge_with_features(
	error_df: pd.DataFrame,
	kernel_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
	if error_df.empty or kernel_feature_df.empty:
		merged_df = error_df.merge(kernel_feature_df, on=JOIN_KEY_COLUMNS, how="left", suffixes=("", "_feature"))
		diagnostics = {
			"error_rows": int(len(error_df)),
			"feature_rows": int(len(kernel_feature_df)),
			"matched_error_rows": 0,
			"unmatched_error_rows": int(len(error_df)),
			"matched_kernel_count": 0,
			"unmatched_feature_rows": int(len(kernel_feature_df)),
		}
		return merged_df, diagnostics

	merged_df = error_df.merge(
		kernel_feature_df,
		on=JOIN_KEY_COLUMNS,
		how="left",
		suffixes=("", "_feature"),
		indicator=True,
	)
	matched_mask = merged_df["_merge"] == "both"
	matched_error_rows = int(matched_mask.sum())
	matched_keys = set(
		merged_df.loc[matched_mask, JOIN_KEY_COLUMNS].itertuples(index=False, name=None)
	)
	feature_keys = set(kernel_feature_df[JOIN_KEY_COLUMNS].itertuples(index=False, name=None))
	error_keys = set(error_df[JOIN_KEY_COLUMNS].itertuples(index=False, name=None))
	diagnostics = {
		"error_rows": int(len(error_df)),
		"feature_rows": int(len(kernel_feature_df)),
		"matched_error_rows": matched_error_rows,
		"unmatched_error_rows": int((~matched_mask).sum()),
		"matched_kernel_count": len(matched_keys),
		"unmatched_feature_rows": len(feature_keys - error_keys),
	}
	merged_df = merged_df.drop(columns=["_merge"])
	return merged_df, diagnostics


def load_analysis_frames(
	gpuflops_db_uri: Optional[str] = None,
	code_features_db_uri: Optional[str] = None,
	*,
	include_dry_run: bool = False,
	kernel_error_metric: str = "max_abs_ai_pct_error",
) -> dict[str, Any]:
	gpuflops_samples_df = load_gpuflops_samples_dataframe(
		db_uri=gpuflops_db_uri,
		include_dry_run=include_dry_run,
	)
	gpuflops_completed_ai_df = enrich_gpuflops_with_ai_metrics(gpuflops_samples_df)
	sample_ai_error_df = build_sample_ai_error_long_dataframe(gpuflops_completed_ai_df)
	kernel_ai_error_df = summarize_kernel_ai_errors(
		sample_ai_error_df,
		selected_metric=kernel_error_metric,
	)

	feature_vote_df = load_code_feature_vote_dataframe(
		db_uri=code_features_db_uri,
		include_dry_run=include_dry_run,
	)
	kernel_feature_df = aggregate_feature_votes(feature_vote_df)

	sample_with_features_df, sample_merge_diagnostics = merge_sample_errors_with_feature_flags(
		sample_ai_error_df,
		kernel_feature_df,
	)
	kernel_with_features_df, kernel_merge_diagnostics = merge_kernel_errors_with_feature_flags(
		kernel_ai_error_df,
		kernel_feature_df,
	)

	return {
		"gpuflops_samples_df": gpuflops_samples_df,
		"gpuflops_completed_ai_df": gpuflops_completed_ai_df,
		"sample_ai_error_df": sample_ai_error_df,
		"kernel_ai_error_df": kernel_ai_error_df,
		"feature_vote_df": feature_vote_df,
		"kernel_feature_df": kernel_feature_df,
		"sample_with_features_df": sample_with_features_df,
		"kernel_with_features_df": kernel_with_features_df,
		"sample_merge_diagnostics": sample_merge_diagnostics,
		"kernel_merge_diagnostics": kernel_merge_diagnostics,
	}


def _print_dataframe_summary(name: str, dataframe: pd.DataFrame) -> None:
	print(f"\n{name}: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
	if dataframe.empty:
		return
	preview_columns = list(dataframe.columns[: min(10, len(dataframe.columns))])
	print(f"Columns: {preview_columns}")


def _write_csv_exports(output_dir: Path, frames: dict[str, Any]) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	for frame_name, frame_value in frames.items():
		if not isinstance(frame_value, pd.DataFrame):
			continue
		frame_value.to_csv(output_dir / f"{frame_name}.csv", index=False)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Load gpuflops_db and code_features_db into pandas dataframes for error analysis.",
	)
	parser.add_argument("--gpuflopsDbUri", default=None, help="Override URI for gpuflops_db")
	parser.add_argument("--codeFeaturesDbUri", default=None, help="Override URI for code_features_db")
	parser.add_argument(
		"--includeDryRun",
		action="store_true",
		help="Include dry-run thread IDs when present.",
	)
	parser.add_argument(
		"--kernelErrorMetric",
		default="max_abs_ai_pct_error",
		choices=["max_abs_ai_pct_error", "median_abs_ai_pct_error", "mean_abs_ai_pct_error"],
		help="Kernel-level AI error summary to use as the selected analysis metric.",
	)
	parser.add_argument(
		"--exportDir",
		default=None,
		help="Optional directory to export intermediate dataframes as CSV files.",
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	frames = load_analysis_frames(
		gpuflops_db_uri=args.gpuflopsDbUri,
		code_features_db_uri=args.codeFeaturesDbUri,
		include_dry_run=args.includeDryRun,
		kernel_error_metric=args.kernelErrorMetric,
	)

	for frame_name in [
		"gpuflops_samples_df",
		"gpuflops_completed_ai_df",
		"sample_ai_error_df",
		"feature_vote_df",
		"kernel_feature_df",
		"kernel_with_features_df",
	]:
		_print_dataframe_summary(frame_name, frames[frame_name])

	print("\nKernel merge diagnostics:")
	for key, value in frames["kernel_merge_diagnostics"].items():
		print(f"- {key}: {value}")

	if args.exportDir:
		_write_csv_exports(Path(args.exportDir), frames)
		print(f"\nExported CSV snapshots to {args.exportDir}")


if __name__ == "__main__":
	main()
