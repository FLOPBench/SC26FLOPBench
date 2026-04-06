import argparse
import re
import sys
from typing import Any, Dict, List

import pandas as pd

from db_manager import CheckpointDBParser, QueryAttemptTracker, ensure_postgres_running, setup_default_database


THREAD_PATTERN = re.compile(
	r"_(?P<gpu>A100|3080|H100|A10)_(?P<safe_model>.+)_(?P<config>withsass|nosass|sass_imix|sass_noimix|nosass_imix|nosass_noimix)_trial(?P<trial>\d+)(?:_DRYRUN(?:\d+)?)?$"
)
MODEL_DATE_SUFFIX_PATTERN = re.compile(r"-\d{8}$")
PLOT_PROMPT_TYPES = ["Source", "Source+SASS"]


def _normalize_model_name(value: str | None) -> str:
	if not value:
		return "unknown-model"
	return MODEL_DATE_SUFFIX_PATTERN.sub("", value)


def _model_name_from_safe(safe_model_name: str | None) -> str:
	if not safe_model_name:
		return "unknown-model"
	return _normalize_model_name(safe_model_name.replace("_", "/"))


def _prompt_type(use_sass: bool | None, use_imix: bool | None) -> str | None:
	if use_imix:
		return None
	return "Source+SASS" if bool(use_sass) else "Source"


def _is_dry_run_thread(thread_id: str) -> bool:
	return "_DRYRUN" in thread_id or thread_id.endswith("_DRYRUN")


def _thread_metadata(thread_id: str) -> Dict[str, Any]:
	match = THREAD_PATTERN.search(thread_id)
	if not match:
		return {
			"gpu": None,
			"model_name": "unknown-model",
			"use_sass": None,
			"use_imix": None,
			"prompt_type": None,
			"trial": None,
		}

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

	return {
		"gpu": match.group("gpu"),
		"model_name": _model_name_from_safe(match.group("safe_model")),
		"use_sass": use_sass,
		"use_imix": use_imix,
		"prompt_type": _prompt_type(use_sass, use_imix),
		"trial": int(match.group("trial")),
	}


def _print_invalid_thread_warnings(invalid_threads: List[Dict[str, Any]]) -> None:
	if not invalid_threads:
		return

	reason_counts: Dict[str, int] = {}
	for item in invalid_threads:
		reason_counts[item["kind"]] = reason_counts.get(item["kind"], 0) + 1

	print("\nWarning: skipped malformed checkpoint histories while tabularizing:", file=sys.stderr)
	for reason, count in sorted(reason_counts.items()):
		print(f"- {reason}: {count}", file=sys.stderr)

	for item in invalid_threads[:5]:
		print(f"- {item['thread_id']}: {item['message']}", file=sys.stderr)


def _completed_checkpoint_by_thread(checkpoints: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
	completed: Dict[str, Dict[str, Any]] = {}
	for thread_id, checkpoint in checkpoints.items():
		state = checkpoint["checkpoint"]["channel_values"]
		if "total_tokens" in state:
			completed[thread_id] = checkpoint
	return completed


def _extract_completed_records(
	completed_checkpoints: Dict[str, Dict[str, Any]],
	include_dry_run: bool,
) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	for thread_id, checkpoint in completed_checkpoints.items():
		if not include_dry_run and _is_dry_run_thread(thread_id):
			continue

		state = checkpoint["checkpoint"]["channel_values"]
		if "program_name" not in state or "kernel_mangled_name" not in state:
			continue

		metadata = _thread_metadata(thread_id)
		records.append(
			{
				"thread_id": thread_id,
				"status": "completed",
				"program_name": state["program_name"],
				"kernel_mangled_name": state["kernel_mangled_name"],
				"gpu": metadata["gpu"],
				"model_name": _normalize_model_name(state.get("llm_model_name") or metadata["model_name"]),
				"prompt_type": metadata["prompt_type"],
				"query_time": pd.to_numeric(state.get("query_time"), errors="coerce"),
				"cost_usd": pd.to_numeric(state.get("cost_usd"), errors="coerce"),
			}
		)

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

		failed_attempts = attempt.get("failed_attempts") or 0
		last_status = attempt.get("last_status")
		if failed_attempts <= 0 and last_status != "failed":
			continue

		checkpoint_record = latest_checkpoints.get(thread_id)
		if checkpoint_record is None:
			continue

		partial_state = checkpoint_record["checkpoint"]["channel_values"]
		if "program_name" not in partial_state or "kernel_mangled_name" not in partial_state:
			print(f"Warning: skipping failed thread {thread_id}: missing program/kernel metadata", file=sys.stderr)
			continue

		metadata = _thread_metadata(thread_id)
		records.append(
			{
				"thread_id": thread_id,
				"status": "failed",
				"program_name": partial_state["program_name"],
				"kernel_mangled_name": partial_state["kernel_mangled_name"],
				"gpu": metadata["gpu"],
				"model_name": _normalize_model_name(partial_state.get("llm_model_name") or metadata["model_name"]),
				"prompt_type": metadata["prompt_type"],
				"query_time": float("nan"),
				"cost_usd": float("nan"),
			}
		)

	return records


def _load_samples_dataframe(db_uri: str, include_dry_run: bool) -> pd.DataFrame:
	parser = CheckpointDBParser(db_uri)
	attempt_tracker = QueryAttemptTracker(db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_result = parser.fetch_tail_checkpoints_by_thread(checkpoints=checkpoints, tolerate_errors=True)
		tail_checkpoints = tail_result["tails"]
		invalid_threads = tail_result["invalid_threads"]
		attempts = attempt_tracker.fetch_all_attempts()
	finally:
		parser.close()
		attempt_tracker.close()

	_print_invalid_thread_warnings(invalid_threads)
	completed_checkpoints = _completed_checkpoint_by_thread(tail_checkpoints)
	completed_records = _extract_completed_records(completed_checkpoints, include_dry_run)
	failed_records = _extract_failed_records(attempts, tail_checkpoints, set(completed_checkpoints), include_dry_run)
	dataframe = pd.DataFrame(completed_records + failed_records)
	if dataframe.empty:
		return pd.DataFrame(
			columns=[
				"thread_id",
				"status",
				"program_name",
				"kernel_mangled_name",
				"gpu",
				"model_name",
				"prompt_type",
				"query_time",
				"cost_usd",
			]
		)

	dataframe["query_time"] = pd.to_numeric(dataframe["query_time"], errors="coerce")
	dataframe["cost_usd"] = pd.to_numeric(dataframe["cost_usd"], errors="coerce")
	return dataframe


def _selected_prompt_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
	if dataframe.empty:
		return dataframe.copy()
	return dataframe[
		dataframe["prompt_type"].isin(PLOT_PROMPT_TYPES)
		& dataframe["gpu"].notna()
		& dataframe["model_name"].notna()
		& dataframe["program_name"].notna()
		& dataframe["kernel_mangled_name"].notna()
	].copy()


def _identity_combo_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
	if dataframe.empty:
		return pd.DataFrame(columns=["program_name", "kernel_mangled_name", "gpu", "model_name", "prompt_type"])
	return dataframe[
		["program_name", "kernel_mangled_name", "gpu", "model_name", "prompt_type"]
	].drop_duplicates()


def _filter_only_shared_samples(dataframe: pd.DataFrame) -> pd.DataFrame:
	selected_df = _selected_prompt_dataframe(dataframe)
	if selected_df.empty:
		return selected_df

	all_gpus = sorted(selected_df["gpu"].dropna().unique().tolist())
	all_models = sorted(selected_df["model_name"].dropna().unique().tolist())
	required_combo_count = len(all_gpus) * len(all_models) * len(PLOT_PROMPT_TYPES)
	if required_combo_count == 0:
		return selected_df.iloc[0:0].copy()

	identity_combo_df = _identity_combo_dataframe(selected_df)
	shared_identity_df = (
		identity_combo_df.groupby(["program_name", "kernel_mangled_name"], dropna=False)
		.size()
		.reset_index(name="combo_count")
	)
	shared_identity_df = shared_identity_df[
		shared_identity_df["combo_count"] == required_combo_count
	].copy()
	if shared_identity_df.empty:
		return selected_df.iloc[0:0].copy()

	shared_keys = shared_identity_df[["program_name", "kernel_mangled_name"]].drop_duplicates()
	return selected_df.merge(shared_keys, on=["program_name", "kernel_mangled_name"], how="inner")


def _overlap_distribution(identity_combo_df: pd.DataFrame, combo_total: int) -> tuple[pd.DataFrame, int, int]:
	if identity_combo_df.empty:
		return pd.DataFrame(columns=["combo_overlap_n", "kernel_identity_n"]), 0, 0

	distribution_df = (
		identity_combo_df.groupby(["program_name", "kernel_mangled_name"], dropna=False)
		.size()
		.reset_index(name="combo_overlap_n")
		.groupby("combo_overlap_n", dropna=False)
		.size()
		.reset_index(name="kernel_identity_n")
		.sort_values("combo_overlap_n")
		.reset_index(drop=True)
	)
	union_n = int(identity_combo_df[["program_name", "kernel_mangled_name"]].drop_duplicates().shape[0])
	shared_n = 0
	if combo_total > 0:
		shared_n = int(
			identity_combo_df.groupby(["program_name", "kernel_mangled_name"], dropna=False)
			.size()
			.reset_index(name="combo_overlap_n")
			.query("combo_overlap_n == @combo_total")
			.shape[0]
		)
	return distribution_df, union_n, shared_n


def _summarize_gpu_prompt_stage(dataframe: pd.DataFrame) -> pd.DataFrame:
	selected_df = _selected_prompt_dataframe(dataframe)
	columns = [
		"gpu",
		"prompt_type",
		"completed_query_n",
		"failed_query_n",
		"total_query_n",
		"distinct_kernel_identity_n",
		"completed_cost_usd_sum",
		"completed_query_time_sum",
	]
	if selected_df.empty:
		return pd.DataFrame(columns=columns)

	grouped_rows: List[Dict[str, Any]] = []
	for (gpu_name, prompt_type), group_df in selected_df.groupby(["gpu", "prompt_type"], dropna=False):
		completed_mask = group_df["status"] == "completed"
		grouped_rows.append(
			{
				"gpu": gpu_name,
				"prompt_type": prompt_type,
				"completed_query_n": int(completed_mask.sum()),
				"failed_query_n": int((group_df["status"] == "failed").sum()),
				"total_query_n": int(group_df.shape[0]),
				"distinct_kernel_identity_n": int(
					group_df[["program_name", "kernel_mangled_name"]].drop_duplicates().shape[0]
				),
				"completed_cost_usd_sum": float(group_df.loc[completed_mask, "cost_usd"].sum()),
				"completed_query_time_sum": float(group_df.loc[completed_mask, "query_time"].sum()),
			}
		)

	summary_df = pd.DataFrame(grouped_rows, columns=columns)
	gpu_order = {gpu_name: index for index, gpu_name in enumerate(["3080", "A10", "A100", "H100"])}
	prompt_order = {prompt_type: index for index, prompt_type in enumerate(PLOT_PROMPT_TYPES)}
	summary_df["gpu_sort"] = summary_df["gpu"].map(lambda value: gpu_order.get(value, len(gpu_order)))
	summary_df["prompt_sort"] = summary_df["prompt_type"].map(lambda value: prompt_order.get(value, len(prompt_order)))
	summary_df = summary_df.sort_values(["gpu_sort", "gpu", "prompt_sort", "prompt_type"]).drop(
		columns=["gpu_sort", "prompt_sort"]
	)
	return summary_df.reset_index(drop=True)


def _summarize_combo_overlap(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
	selected_df = _selected_prompt_dataframe(dataframe)
	combo_columns = ["gpu", "model_name", "prompt_type"]
	identity_columns = ["program_name", "kernel_mangled_name"]
	result = {
		"per_combo": pd.DataFrame(
			columns=[
				*combo_columns,
				"completed_query_n",
				"failed_query_n",
				"total_query_n",
				"completed_kernel_identity_n",
				"failed_kernel_identity_n",
				"total_kernel_identity_n",
			]
		),
		"overlap_distribution": pd.DataFrame(columns=["combo_overlap_n", "kernel_identity_n"]),
		"totals": pd.DataFrame(columns=["metric", "value"]),
	}
	if selected_df.empty:
		return result

	per_combo_rows: List[Dict[str, Any]] = []
	for combo_values, combo_df in selected_df.groupby(combo_columns, dropna=False):
		gpu_name, model_name, prompt_type = combo_values
		completed_subset = combo_df[combo_df["status"] == "completed"]
		failed_subset = combo_df[combo_df["status"] == "failed"]
		per_combo_rows.append(
			{
				"gpu": gpu_name,
				"model_name": model_name,
				"prompt_type": prompt_type,
				"completed_query_n": int(completed_subset.shape[0]),
				"failed_query_n": int(failed_subset.shape[0]),
				"total_query_n": int(combo_df.shape[0]),
				"completed_kernel_identity_n": int(
					completed_subset[identity_columns].drop_duplicates().shape[0]
				),
				"failed_kernel_identity_n": int(
					failed_subset[identity_columns].drop_duplicates().shape[0]
				),
				"total_kernel_identity_n": int(
					combo_df[identity_columns].drop_duplicates().shape[0]
				),
			}
		)

	per_combo_df = pd.DataFrame(per_combo_rows)
	combo_total = int(selected_df[combo_columns].drop_duplicates().shape[0])
	total_identity_combo_df = _identity_combo_dataframe(selected_df)
	completed_identity_combo_df = _identity_combo_dataframe(selected_df[selected_df["status"] == "completed"])
	failed_identity_combo_df = _identity_combo_dataframe(selected_df[selected_df["status"] == "failed"])
	overlap_distribution_df, total_union_n, total_shared_n = _overlap_distribution(total_identity_combo_df, combo_total)
	_, completed_union_n, completed_shared_n = _overlap_distribution(completed_identity_combo_df, combo_total)
	_, failed_union_n, failed_shared_n = _overlap_distribution(failed_identity_combo_df, combo_total)

	totals_df = pd.DataFrame(
		[
			{"metric": "selected_combo_n", "value": combo_total},
			{"metric": "total_union_kernel_identity_n", "value": total_union_n},
			{"metric": "total_shared_across_all_combos_n", "value": total_shared_n},
			{"metric": "completed_union_kernel_identity_n", "value": completed_union_n},
			{"metric": "completed_shared_across_all_combos_n", "value": completed_shared_n},
			{"metric": "failed_union_kernel_identity_n", "value": failed_union_n},
			{"metric": "failed_shared_across_all_combos_n", "value": failed_shared_n},
		]
	)

	gpu_order = {gpu_name: index for index, gpu_name in enumerate(["3080", "A10", "A100", "H100"])}
	prompt_order = {prompt_type: index for index, prompt_type in enumerate(PLOT_PROMPT_TYPES)}
	per_combo_df["gpu_sort"] = per_combo_df["gpu"].map(lambda value: gpu_order.get(value, len(gpu_order)))
	per_combo_df["prompt_sort"] = per_combo_df["prompt_type"].map(lambda value: prompt_order.get(value, len(prompt_order)))
	per_combo_df = per_combo_df.sort_values(
		["gpu_sort", "gpu", "model_name", "prompt_sort", "prompt_type"]
	).drop(columns=["gpu_sort", "prompt_sort"]).reset_index(drop=True)

	result["per_combo"] = per_combo_df
	result["overlap_distribution"] = overlap_distribution_df
	result["totals"] = totals_df
	return result


def _print_table(title: str, dataframe: pd.DataFrame) -> None:
	print(f"\n{title}")
	if dataframe.empty:
		print("  No rows")
		return
	print(dataframe.to_string(index=False))


def _print_stage_report(stage_name: str, dataframe: pd.DataFrame) -> None:
	_print_table(
		f"{stage_name}: Sample counts by GPU and prompt type",
		_summarize_gpu_prompt_stage(dataframe),
	)
	overlap_summary = _summarize_combo_overlap(dataframe)
	_print_table(
		f"{stage_name}: Identity coverage by GPU, model, and prompt type",
		overlap_summary["per_combo"],
	)
	_print_table(
		f"{stage_name}: (program_name, kernel_mangled_name) overlap distribution across GPU/model/prompt combinations (total = completed + failed)",
		overlap_summary["overlap_distribution"],
	)
	_print_table(
		f"{stage_name}: Identity overlap totals",
		overlap_summary["totals"],
	)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Tabularize gpuflops_db query outcomes before and after strict shared-sample filtering."
	)
	parser.add_argument(
		"--dbUri",
		type=str,
		default=None,
		help="Explicit PostgreSQL database URI. Defaults to the local gpuflops_db database.",
	)
	parser.add_argument(
		"--includeDryRun",
		action="store_true",
		help="Include dry-run thread IDs in the summary tables. By default they are excluded.",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	ensure_postgres_running()
	db_uri = args.dbUri or setup_default_database()
	samples_df = _load_samples_dataframe(db_uri, include_dry_run=args.includeDryRun)
	if samples_df.empty:
		raise RuntimeError("No matching checkpoint or failed-attempt records were found in gpuflops_db.")

	pre_filter_df = _selected_prompt_dataframe(samples_df)
	post_filter_df = _filter_only_shared_samples(samples_df)

	print("Tabularized gpuflops_db results for Source and Source+SASS prompt types.")
	_print_stage_report("Pre-filter", pre_filter_df)
	_print_stage_report("Post-filter (--onlySharedSamples semantics)", post_filter_df)


if __name__ == "__main__":
	main()