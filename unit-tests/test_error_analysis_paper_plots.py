import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ERROR_ANALYSIS_DIR = REPO_ROOT / "experiments" / "error-analysis"

if str(ERROR_ANALYSIS_DIR) not in sys.path:
	sys.path.insert(0, str(ERROR_ANALYSIS_DIR))


def _load_module(module_name: str, file_path: Path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	assert spec is not None and spec.loader is not None
	spec.loader.exec_module(module)
	return module


paper_plots = _load_module(
	"error_analysis_make_plots_for_paper",
	ERROR_ANALYSIS_DIR / "make_plots_for_paper.py",
)


def _sample_rows() -> pd.DataFrame:
	rows = []
	for error_value, has_branching, has_special_math in [
		(80.0, True, True),
		(70.0, True, False),
		(60.0, True, True),
		(15.0, False, False),
		(10.0, False, False),
		(5.0, False, False),
	]:
		row = {
			"program_name": "demo-cuda",
			"kernel_mangled_name": "_Zdemo",
			"kernel_demangled_name": "demo()",
			"runtime": "cuda",
			"precision": "fp32",
			"gpu": "A100",
			"model_name": "gpt-5.4",
			"safe_model_name": "gpt-5.4",
			"evidence_configuration": "Source Only",
			"abs_ai_pct_error": error_value,
		}
		for feature_name in paper_plots.db_reader.FEATURE_FLAG_COLUMNS:
			row[feature_name] = False
		row["has_branching"] = has_branching
		row["has_special_math_functions"] = has_special_math
		rows.append(row)
	return pd.DataFrame(rows)


def test_cliffs_delta_extremes():
	assert paper_plots._cliffs_delta(np.array([10.0, 12.0]), np.array([1.0, 2.0])) == 1.0
	assert paper_plots._cliffs_delta(np.array([1.0, 2.0]), np.array([10.0, 12.0])) == -1.0


def test_build_association_dataframe_computes_feature_effects():
	clean_df = paper_plots._clean_sample_dataframe(_sample_rows())
	feature_long_df = paper_plots._feature_presence_long_frame(clean_df)
	association_df = paper_plots._build_association_dataframe(
		feature_long_df,
		min_present=2,
		min_absent=2,
	)

	branching_row = association_df[association_df["feature_name"] == "has_branching"].iloc[0]
	assert branching_row["n_present"] == 3
	assert branching_row["n_absent"] == 3
	assert bool(branching_row["is_valid"])
	assert branching_row["association_score"] > 0.9
	assert branching_row["median_error_delta"] == 60.0

	math_row = association_df[association_df["feature_name"] == "has_special_math_functions"].iloc[0]
	assert math_row["n_present"] == 2
	assert math_row["n_absent"] == 4
	assert bool(math_row["is_valid"])
	assert math_row["association_score"] > 0.5

	invalid_feature_row = association_df[
		association_df["feature_name"] == "calls_device_function"
	].iloc[0]
	assert invalid_feature_row["n_present"] == 0
	assert invalid_feature_row["n_absent"] == 6
	assert not bool(invalid_feature_row["is_valid"])
	assert math.isnan(invalid_feature_row["association_score"])


def test_clean_sample_dataframe_uses_prompt_type_and_drops_imix_rows():
	sample_df = pd.DataFrame(
		[
			{
				"precision": "fp32",
				"runtime": "cuda",
				"gpu": "A100",
				"model_name": "openai/gpt-5.4",
				"safe_model_name": "openai_gpt-5.4",
				"abs_ai_pct_error": 12.0,
				"use_sass": False,
				"use_imix": False,
				"evidence_configuration": "Source-Only",
			},
			{
				"precision": "fp32",
				"runtime": "cuda",
				"gpu": "A100",
				"model_name": "openai/gpt-oss-120b",
				"safe_model_name": "openai_gpt-oss-120b",
				"abs_ai_pct_error": 15.0,
				"use_sass": True,
				"use_imix": False,
				"evidence_configuration": "Source+SASS",
			},
			{
				"precision": "fp32",
				"runtime": "cuda",
				"gpu": "A100",
				"model_name": "anthropic/claude-4.6-opus",
				"safe_model_name": "anthropic_claude-4.6-opus",
				"abs_ai_pct_error": 18.0,
				"use_sass": False,
				"use_imix": True,
				"evidence_configuration": "Source+IMIX",
			},
		]
	)

	clean_df = paper_plots._clean_sample_dataframe(sample_df)
	assert clean_df.shape[0] == 2
	assert clean_df["prompt_type"].tolist() == ["Source-Only", "Source+SASS"]
	assert clean_df["model_label"].tolist() == ["GPT 5.4", "GPT OSS"]


def test_format_model_label_normalizes_claude_opus_variant():
	assert paper_plots._format_model_label(
		"anthropic/claude-opus-4.6",
		"anthropic_claude-opus-4.6",
	) == "Opus 4.6"


def test_filter_only_shared_samples_keeps_model_matched_rows():
	sample_df = pd.DataFrame(
		[
			{
				"program_name": "demo-cuda",
				"kernel_mangled_name": "_Zshared",
				"gpu": "A100",
				"use_sass": False,
				"use_imix": False,
				"model_name": "GPT 5.4",
				"abs_ai_pct_error": 10.0,
			},
			{
				"program_name": "demo-cuda",
				"kernel_mangled_name": "_Zshared",
				"gpu": "A100",
				"use_sass": False,
				"use_imix": False,
				"model_name": "GPT OSS",
				"abs_ai_pct_error": 11.0,
			},
			{
				"program_name": "demo-cuda",
				"kernel_mangled_name": "_Zunshared",
				"gpu": "A100",
				"use_sass": False,
				"use_imix": False,
				"model_name": "GPT 5.4",
				"abs_ai_pct_error": 12.0,
			},
		]
	)

	filtered_df = paper_plots._filter_only_shared_samples(sample_df)
	assert filtered_df.shape[0] == 2
	assert filtered_df["kernel_mangled_name"].unique().tolist() == ["_Zshared"]


def test_build_association_dataframe_supports_collapsed_variants():
	clean_df = paper_plots._clean_sample_dataframe(_sample_rows())
	feature_long_df = paper_plots._feature_presence_long_frame(clean_df)

	collapsed_model_df = paper_plots._build_association_dataframe(
		feature_long_df,
		min_present=2,
		min_absent=2,
		collapse_model=True,
		summary_mode="collapsed_model",
	)
	assert set(collapsed_model_df["model_label"].unique().tolist()) == {paper_plots.ALL_MODELS_LABEL}
	assert set(collapsed_model_df["column_key"].unique().tolist()) == {"A100"}

	collapsed_precision_df = paper_plots._build_association_dataframe(
		feature_long_df,
		min_present=2,
		min_absent=2,
		collapse_precision=True,
		summary_mode="collapsed_precision",
	)
	assert set(collapsed_precision_df["precision"].unique().tolist()) == {paper_plots.ALL_PRECISIONS_LABEL}