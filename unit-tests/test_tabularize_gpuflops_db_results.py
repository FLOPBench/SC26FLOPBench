import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "experiments" / "direct-prompting"


def _load_module(module_name: str, file_name: str):
	sys.path.insert(0, str(MODULE_DIR))
	try:
		spec = importlib.util.spec_from_file_location(module_name, MODULE_DIR / file_name)
		module = importlib.util.module_from_spec(spec)
		assert spec is not None and spec.loader is not None
		spec.loader.exec_module(module)
		return module
	finally:
		sys.path.pop(0)


tabularize_results = _load_module("tabularize_gpuflops_db_results", "tabularize_gpuflops_db_results.py")


def _sample_row(**overrides):
	row = {
		"thread_id": "thread-0",
		"status": "completed",
		"program_name": "demo-cuda",
		"kernel_mangled_name": "_Z4demov",
		"gpu": "A100",
		"model_name": "openai/gpt-5.4",
		"prompt_type": "Source",
		"query_time": 1.0,
		"cost_usd": 0.1,
	}
	row.update(overrides)
	return row


def test_filter_only_shared_samples_requires_all_gpu_model_prompt_combinations():
	dataframe = pd.DataFrame(
		[
			_sample_row(thread_id="keep-a100-gpt-source", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source"),
			_sample_row(thread_id="keep-a100-gpt-sass", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source+SASS"),
			_sample_row(thread_id="keep-a100-oss-source", gpu="A100", model_name="openai/gpt-oss-120b", prompt_type="Source"),
			_sample_row(thread_id="keep-a100-oss-sass", gpu="A100", model_name="openai/gpt-oss-120b", prompt_type="Source+SASS"),
			_sample_row(thread_id="keep-h100-gpt-source", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source"),
			_sample_row(thread_id="keep-h100-gpt-sass", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source+SASS"),
			_sample_row(thread_id="keep-h100-oss-source", gpu="H100", model_name="openai/gpt-oss-120b", prompt_type="Source"),
			_sample_row(thread_id="keep-h100-oss-sass", gpu="H100", model_name="openai/gpt-oss-120b", prompt_type="Source+SASS"),
			_sample_row(thread_id="keep-failed-a100-gpt-source", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="keep-failed-a100-gpt-sass", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source+SASS", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="keep-failed-h100-gpt-source", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="keep-failed-h100-gpt-sass", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source+SASS", kernel_mangled_name="_Z11failkeepv", status="failed", query_time=float("nan"), cost_usd=float("nan")),
			_sample_row(thread_id="keep-failed-a100-oss-source", gpu="A100", model_name="openai/gpt-oss-120b", prompt_type="Source", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="keep-failed-a100-oss-sass", gpu="A100", model_name="openai/gpt-oss-120b", prompt_type="Source+SASS", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="keep-failed-h100-oss-source", gpu="H100", model_name="openai/gpt-oss-120b", prompt_type="Source", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="keep-failed-h100-oss-sass", gpu="H100", model_name="openai/gpt-oss-120b", prompt_type="Source+SASS", kernel_mangled_name="_Z11failkeepv"),
			_sample_row(thread_id="drop-a100-gpt-source", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z8dropmev"),
			_sample_row(thread_id="drop-a100-gpt-sass", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source+SASS", kernel_mangled_name="_Z8dropmev"),
			_sample_row(thread_id="drop-h100-gpt-source", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z8dropmev"),
		]
	)

	filtered = tabularize_results._filter_only_shared_samples(dataframe)

	assert set(filtered["thread_id"].tolist()) == {
		"keep-a100-gpt-source",
		"keep-a100-gpt-sass",
		"keep-a100-oss-source",
		"keep-a100-oss-sass",
		"keep-h100-gpt-source",
		"keep-h100-gpt-sass",
		"keep-h100-oss-source",
		"keep-h100-oss-sass",
		"keep-failed-a100-gpt-source",
		"keep-failed-a100-gpt-sass",
		"keep-failed-h100-gpt-source",
		"keep-failed-h100-gpt-sass",
		"keep-failed-a100-oss-source",
		"keep-failed-a100-oss-sass",
		"keep-failed-h100-oss-source",
		"keep-failed-h100-oss-sass",
	}
	assert set(filtered["kernel_mangled_name"].tolist()) == {"_Z4demov", "_Z11failkeepv"}
	assert int((filtered["status"] == "failed").sum()) == 1


def test_summarize_combo_overlap_reports_shared_identity_totals():
	dataframe = pd.DataFrame(
		[
			_sample_row(thread_id="shared-a100-gpt-source", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source"),
			_sample_row(thread_id="shared-a100-gpt-sass", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source+SASS"),
			_sample_row(thread_id="shared-h100-gpt-source", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source"),
			_sample_row(thread_id="shared-h100-gpt-sass", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source+SASS"),
			_sample_row(thread_id="shared-failed-a100-gpt-source", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z10sharedfailv"),
			_sample_row(thread_id="shared-failed-a100-gpt-sass", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source+SASS", kernel_mangled_name="_Z10sharedfailv"),
			_sample_row(thread_id="shared-failed-h100-gpt-source", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z10sharedfailv"),
			_sample_row(thread_id="shared-failed-h100-gpt-sass", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source+SASS", kernel_mangled_name="_Z10sharedfailv", status="failed", query_time=float("nan"), cost_usd=float("nan")),
			_sample_row(thread_id="partial-a100-gpt-source", gpu="A100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z7partialv"),
			_sample_row(thread_id="partial-h100-gpt-source", gpu="H100", model_name="openai/gpt-5.4", prompt_type="Source", kernel_mangled_name="_Z7partialv"),
		]
	)

	summary = tabularize_results._summarize_combo_overlap(dataframe)
	totals = dict(zip(summary["totals"]["metric"], summary["totals"]["value"]))
	distribution = dict(zip(summary["overlap_distribution"]["combo_overlap_n"], summary["overlap_distribution"]["kernel_identity_n"]))
	per_combo = summary["per_combo"].set_index(["gpu", "model_name", "prompt_type"])

	assert totals["selected_combo_n"] == 4
	assert totals["total_union_kernel_identity_n"] == 3
	assert totals["total_shared_across_all_combos_n"] == 2
	assert totals["completed_union_kernel_identity_n"] == 3
	assert totals["completed_shared_across_all_combos_n"] == 1
	assert totals["failed_union_kernel_identity_n"] == 1
	assert totals["failed_shared_across_all_combos_n"] == 0
	assert distribution[2] == 1
	assert distribution[4] == 2
	assert per_combo.loc[("H100", "openai/gpt-5.4", "Source+SASS"), "completed_query_n"] == 1
	assert per_combo.loc[("H100", "openai/gpt-5.4", "Source+SASS"), "failed_query_n"] == 1
	assert per_combo.loc[("H100", "openai/gpt-5.4", "Source+SASS"), "total_query_n"] == 2
	assert per_combo.loc[("H100", "openai/gpt-5.4", "Source+SASS"), "completed_kernel_identity_n"] == 1
	assert per_combo.loc[("H100", "openai/gpt-5.4", "Source+SASS"), "failed_kernel_identity_n"] == 1
	assert per_combo.loc[("H100", "openai/gpt-5.4", "Source+SASS"), "total_kernel_identity_n"] == 2