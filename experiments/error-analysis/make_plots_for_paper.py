import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import db_reader


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)
DEFAULT_OUTPUT_DIR = os.path.join(
	WORKSPACE_ROOT,
	"experiments",
	"error-analysis",
	"paper-figure-output",
)
RUNTIME_ORDER = ["cuda", "omp"]
GPU_ORDER = ["3080", "A10", "A100", "H100"]
DEFAULT_MIN_PRESENT = 5
DEFAULT_MIN_ABSENT = 5
PROMPT_TYPE_LABEL = "Prompt Type"
ALL_MODELS_LABEL = "All Models"
ALL_RUNTIMES_LABEL = "All Runtimes"
ALL_PRECISIONS_LABEL = "All Precisions"
ALL_PROMPT_TYPES_LABEL = "All Prompt Types"
RUNTIME_DISPLAY_LABELS = {
	"cuda": "CUDA",
	"omp": "OpenMP",
}

SUMMARY_VARIANTS = [
	("full", "", False, False),
	("collapsed_model", "Collapsed Over Models", True, False),
	("collapsed_precision", "Collapsed Over Precisions", False, True),
	("collapsed_model_precision", "Collapsed Over Models And Precisions", True, True),
]

FEATURE_LABEL_OVERRIDES = {
	"has_branching": "Has Branching",
	"has_data_dependent_branching": "Has Data-Dep Branching",
	"has_flop_division": "Has FLOP Division",
	"uses_preprocessor_defines": "Uses Preproc Defines",
	"has_common_float_subexpr": "Has Float Subexprs",
	"has_loop_invariant_flops": "Has Loop-Invariant FLOPs",
	"has_special_math_functions": "Has Special Math",
	"calls_device_function": "Calls Helper Function",
	"has_rng_input_data": "Has RNG Inputs",
	"reads_input_values_from_file": "Uses Input Data from File",
	"has_constant_propagatable_gridsz": "Deterministic Grid Size",
	"has_constant_propagatable_blocksz": "Deterministic Block Size",
}


def _load_module(module_name: str, relative_path: str):
	module_path = os.path.join(WORKSPACE_ROOT, relative_path)
	module_spec = importlib.util.spec_from_file_location(module_name, module_path)
	if module_spec is None or module_spec.loader is None:
		raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
	module = importlib.util.module_from_spec(module_spec)
	module_spec.loader.exec_module(module)
	return module


direct_visualize_results = _load_module(
	"error_analysis_result_viz_helper",
	os.path.join("experiments", "direct-prompting", "result_viz_helper.py"),
)


def _safe_filename(value: str) -> str:
	return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("_") or "value"


def _format_feature_label(feature_name: str) -> str:
	return FEATURE_LABEL_OVERRIDES.get(feature_name, feature_name.replace("_", " ").strip().title())


def _format_model_label(model_name: Any, safe_model_name: Any) -> str:
	for candidate, is_safe_name in [(safe_model_name, True), (model_name, False)]:
		if isinstance(candidate, str) and candidate.strip():
			normalized_candidate = candidate.strip()
			if is_safe_name:
				normalized_candidate = normalized_candidate.replace("_", "/")
			if "/" not in normalized_candidate:
				if normalized_candidate.startswith("gpt-"):
					normalized_candidate = f"openai/{normalized_candidate}"
				elif normalized_candidate.startswith("claude"):
					normalized_candidate = f"anthropic/{normalized_candidate}"
			normalized_candidate = normalized_candidate.replace("anthropic/claude-opus-4.6", "anthropic/claude-4.6-opus")
			return direct_visualize_results._display_plot_model_name(normalized_candidate)
	return "Unknown Model"


def _normalize_prompt_type(use_sass: Any, use_imix: Any, fallback_value: Any) -> str:
	if pd.notna(use_sass) or pd.notna(use_imix):
		return direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(bool(use_sass), bool(use_imix))]
	if isinstance(fallback_value, str) and fallback_value.strip():
		normalized = fallback_value.strip().casefold()
		if "imix" in normalized:
			return fallback_value.strip()
		if "source+sass" in normalized or "withsass" in normalized:
			return direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, False)]
		if "source-only" in normalized or "no sass" in normalized or "nosass" in normalized:
			return direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)]
	return direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)]


def _set_plot_theme() -> None:
	sns.set_theme(
		style="whitegrid",
		context="paper",
		rc={
			"axes.spines.top": False,
			"axes.spines.right": False,
			"axes.facecolor": "#f8f7f2",
			"figure.dpi": 160,
			"savefig.dpi": 300,
			"font.family": "DejaVu Serif",
		},
	)


def _clean_sample_dataframe(sample_with_features_df: pd.DataFrame) -> pd.DataFrame:
	if sample_with_features_df.empty:
		return sample_with_features_df.copy()

	plot_df = sample_with_features_df.copy()
	plot_df["abs_ai_pct_error"] = pd.to_numeric(plot_df["abs_ai_pct_error"], errors="coerce")
	plot_df = plot_df[np.isfinite(plot_df["abs_ai_pct_error"].to_numpy())].copy()
	plot_df = plot_df[plot_df["precision"].isin(db_reader.AI_PRECISIONS)].copy()
	plot_df["runtime"] = plot_df["runtime"].fillna("unknown")
	plot_df["gpu"] = plot_df["gpu"].fillna("Unknown GPU")
	plot_df["use_sass"] = plot_df.get("use_sass", False)
	plot_df["use_imix"] = plot_df.get("use_imix", False)
	plot_df["use_sass"] = plot_df["use_sass"].fillna(False).astype(bool)
	plot_df["use_imix"] = plot_df["use_imix"].fillna(False).astype(bool)
	plot_df["prompt_type"] = plot_df.apply(
		lambda row: _normalize_prompt_type(row.get("use_sass"), row.get("use_imix"), row.get("evidence_configuration")),
		axis=1,
	)
	plot_df = plot_df[plot_df["use_imix"] == False].copy()
	plot_df = plot_df[
		plot_df["prompt_type"].isin(
			[
				direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)],
				direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, False)],
			]
		)
	].copy()
	plot_df["model_label"] = plot_df.apply(
		lambda row: _format_model_label(row.get("model_name"), row.get("safe_model_name")),
		axis=1,
	)
	return plot_df


def _filter_only_shared_samples(sample_with_features_df: pd.DataFrame) -> pd.DataFrame:
	if sample_with_features_df.empty:
		return sample_with_features_df.copy()

	shared_df = sample_with_features_df.copy()
	if "status" not in shared_df.columns:
		shared_df["status"] = "completed"
	filtered_df = direct_visualize_results._filter_only_shared_samples(shared_df)
	return filtered_df.drop(columns=["status"], errors="ignore")


def _feature_presence_long_frame(sample_with_features_df: pd.DataFrame) -> pd.DataFrame:
	if sample_with_features_df.empty:
		return pd.DataFrame()

	base_columns = [
		"program_name",
		"kernel_mangled_name",
		"kernel_demangled_name",
		"runtime",
		"precision",
		"gpu",
		"model_name",
		"safe_model_name",
		"model_label",
		"prompt_type",
		"abs_ai_pct_error",
	]
	available_base_columns = [column for column in base_columns if column in sample_with_features_df.columns]
	plot_df = sample_with_features_df.melt(
		id_vars=available_base_columns,
		value_vars=db_reader.FEATURE_FLAG_COLUMNS,
		var_name="feature_name",
		value_name="feature_present",
	)
	plot_df = plot_df[plot_df["feature_present"].notna()].copy()
	plot_df["feature_present"] = plot_df["feature_present"].astype(bool)
	plot_df["feature_label"] = plot_df["feature_name"].map(_format_feature_label)
	return plot_df


def _cliffs_delta(present_values: np.ndarray, absent_values: np.ndarray) -> float:
	if present_values.size == 0 or absent_values.size == 0:
		return float("nan")

	sorted_absent = np.sort(absent_values)
	win_counts = np.searchsorted(sorted_absent, present_values, side="left")
	loss_counts = absent_values.size - np.searchsorted(sorted_absent, present_values, side="right")
	return float((win_counts.sum() - loss_counts.sum()) / (present_values.size * absent_values.size))


def _build_association_dataframe(
	feature_long_df: pd.DataFrame,
	*,
	min_present: int,
	min_absent: int,
	collapse_model: bool = False,
	collapse_precision: bool = False,
	summary_mode: str = "full",
) -> pd.DataFrame:
	if feature_long_df.empty:
		return pd.DataFrame()

	group_columns = ["prompt_type"]
	if not collapse_precision:
		group_columns.append("precision")
	group_columns.extend(["runtime", "gpu"])
	if not collapse_model:
		group_columns.append("model_label")
	group_columns.extend(["feature_name", "feature_label"])
	records: list[dict[str, Any]] = []
	for group_key, group_df in feature_long_df.groupby(group_columns, dropna=False):
		group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
		present_errors = pd.to_numeric(
			group_df.loc[group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		present_errors = present_errors[np.isfinite(present_errors.to_numpy())]
		absent_errors = pd.to_numeric(
			group_df.loc[~group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		absent_errors = absent_errors[np.isfinite(absent_errors.to_numpy())]

		n_present = int(present_errors.shape[0])
		n_absent = int(absent_errors.shape[0])
		is_valid = n_present >= min_present and n_absent >= min_absent
		median_present = float(present_errors.median()) if n_present else float("nan")
		median_absent = float(absent_errors.median()) if n_absent else float("nan")

		records.append(
			{
				"summary_mode": summary_mode,
				"prompt_type": group_values["prompt_type"],
				"precision": group_values.get("precision", ALL_PRECISIONS_LABEL),
				"runtime": group_values["runtime"],
				"gpu": group_values["gpu"],
				"model_label": group_values.get("model_label", ALL_MODELS_LABEL),
				"feature_name": group_values["feature_name"],
				"feature_label": group_values["feature_label"],
				"n_total": int(len(group_df)),
				"n_present": n_present,
				"n_absent": n_absent,
				"median_error_present": median_present,
				"median_error_absent": median_absent,
				"median_error_delta": median_present - median_absent,
				"association_score": _cliffs_delta(present_errors.to_numpy(), absent_errors.to_numpy()) if is_valid else float("nan"),
				"is_valid": bool(is_valid),
			}
		)

	association_df = pd.DataFrame(records)
	if association_df.empty:
		return association_df

	association_df["collapsed_model"] = bool(collapse_model)
	association_df["collapsed_precision"] = bool(collapse_precision)
	association_df["model_tick_label"] = association_df["model_label"]
	association_df["gpu_model_label"] = association_df.apply(
		lambda row: f"{row['gpu']}::{row['model_label']}",
		axis=1,
	)
	association_df["column_key"] = np.where(
		association_df["collapsed_model"],
		association_df["gpu"],
		association_df["gpu_model_label"],
	)
	association_df["column_tick_label"] = np.where(
		association_df["collapsed_model"],
		association_df["gpu"],
		association_df["model_label"],
	)
	association_df["direction_label"] = np.where(
		association_df["association_score"] > 0.0,
		"Higher Error When Present",
		"Lower Error When Present",
	)
	association_df.loc[association_df["association_score"].isna(), "direction_label"] = "Insufficient Support"
	return association_df.sort_values(group_columns).reset_index(drop=True)


def _build_runtime_feature_summary_dataframe(
	feature_long_df: pd.DataFrame,
	*,
	min_present: int,
	min_absent: int,
) -> pd.DataFrame:
	if feature_long_df.empty:
		return pd.DataFrame()

	group_columns = ["prompt_type", "runtime", "feature_name", "feature_label"]
	records: list[dict[str, Any]] = []
	for group_key, group_df in feature_long_df.groupby(group_columns, dropna=False):
		group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
		present_errors = pd.to_numeric(
			group_df.loc[group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		present_errors = present_errors[np.isfinite(present_errors.to_numpy())]
		absent_errors = pd.to_numeric(
			group_df.loc[~group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		absent_errors = absent_errors[np.isfinite(absent_errors.to_numpy())]

		n_present = int(present_errors.shape[0])
		n_absent = int(absent_errors.shape[0])
		is_valid = n_present >= min_present and n_absent >= min_absent
		median_present = float(present_errors.median()) if n_present else float("nan")
		median_absent = float(absent_errors.median()) if n_absent else float("nan")

		records.append(
			{
				"summary_mode": "collapsed_runtime_feature",
				"prompt_type": group_values["prompt_type"],
				"precision": ALL_PRECISIONS_LABEL,
				"runtime": group_values["runtime"],
				"gpu": "All GPUs",
				"model_label": ALL_MODELS_LABEL,
				"feature_name": group_values["feature_name"],
				"feature_label": group_values["feature_label"],
				"n_total": int(len(group_df)),
				"n_present": n_present,
				"n_absent": n_absent,
				"median_error_present": median_present,
				"median_error_absent": median_absent,
				"median_error_delta": median_present - median_absent,
				"association_score": _cliffs_delta(present_errors.to_numpy(), absent_errors.to_numpy()) if is_valid else float("nan"),
				"is_valid": bool(is_valid),
			}
		)

	runtime_df = pd.DataFrame(records)
	if runtime_df.empty:
		return runtime_df

	runtime_df["collapsed_model"] = True
	runtime_df["collapsed_precision"] = True
	runtime_df["model_tick_label"] = runtime_df["model_label"]
	runtime_df["gpu_model_label"] = runtime_df["runtime"]
	runtime_df["column_key"] = runtime_df["feature_label"]
	runtime_df["column_tick_label"] = runtime_df["feature_label"]
	runtime_df["direction_label"] = np.where(
		runtime_df["association_score"] > 0.0,
		"Higher Error When Present",
		"Lower Error When Present",
	)
	runtime_df.loc[runtime_df["association_score"].isna(), "direction_label"] = "Insufficient Support"
	return runtime_df.sort_values(group_columns).reset_index(drop=True)


def _feature_order(association_df: pd.DataFrame) -> list[str]:
	if association_df.empty:
		return []
	order_df = association_df.copy()
	order_df["score_magnitude"] = order_df["association_score"].abs()
	return list(
		order_df.groupby("feature_label")["score_magnitude"]
		.median()
		.sort_values(ascending=False, na_position="last")
		.index
	)


def _runtime_feature_order(runtime_summary_df: pd.DataFrame) -> list[str]:
	if runtime_summary_df.empty:
		return []
	order_df = runtime_summary_df.copy()
	feature_scores = (
		order_df.groupby("feature_label", as_index=False)
		.agg(
			median_association_score=("association_score", "median"),
			median_abs_association_score=("association_score", lambda series: float(pd.Series(series).abs().median())),
		)
		.sort_values(
			["median_association_score", "median_abs_association_score", "feature_label"],
			ascending=[False, False, True],
			na_position="last",
			kind="stable",
		)
	)
	return feature_scores["feature_label"].tolist()


def _ordered_gpu_values(gpu_values: pd.Series) -> list[str]:
	available_gpus = [gpu_name for gpu_name in gpu_values.dropna().unique().tolist() if isinstance(gpu_name, str) and gpu_name.strip()]
	preferred_gpu_order = [gpu_name for gpu_name in GPU_ORDER if gpu_name in available_gpus]
	remaining_gpus = sorted(gpu_name for gpu_name in available_gpus if gpu_name not in preferred_gpu_order)
	return preferred_gpu_order + remaining_gpus


def _ordered_model_values(model_values: pd.Series) -> list[str]:
	return sorted(
		model_name
		for model_name in model_values.dropna().unique().tolist()
		if isinstance(model_name, str) and model_name.strip()
	)


def _column_order(association_df: pd.DataFrame) -> list[str]:
	if association_df.empty:
		return []
	if bool(association_df["collapsed_model"].iloc[0]):
		return sorted(association_df["gpu"].dropna().unique().tolist())
	unique_pairs = association_df[["gpu", "model_label", "column_key"]].drop_duplicates().copy()
	return [row["column_key"] for _, row in unique_pairs.sort_values(["gpu", "model_label"], kind="stable").iterrows()]


def _gpu_group_centers(column_order: list[str]) -> list[tuple[float, str]]:
	if not column_order:
		return []
	centers: list[tuple[float, str]] = []
	start_index = 0
	current_gpu = column_order[0].split("::", maxsplit=1)[0]
	for column_index, column_name in enumerate(column_order[1:], start=1):
		gpu_name = column_name.split("::", maxsplit=1)[0]
		if gpu_name != current_gpu:
			centers.append((((start_index + (column_index - 1)) / 2.0) + 0.5, current_gpu))
			start_index = column_index
			current_gpu = gpu_name
	centers.append((((start_index + (len(column_order) - 1)) / 2.0) + 0.5, current_gpu))
	return centers


def _heatmap_annotation(score_df: pd.DataFrame) -> pd.DataFrame:
	annotation_df = pd.DataFrame("", index=score_df.index, columns=score_df.columns, dtype=object)
	for row_index in score_df.index:
		for column_name in score_df.columns:
			value = score_df.loc[row_index, column_name]
			annotation_df.loc[row_index, column_name] = "" if pd.isna(value) else f"{float(value):.2f}"
	return annotation_df


def _save_runtime_feature_summary_heatmap(
	runtime_summary_df: pd.DataFrame,
	output_dir: Path,
	*,
	feature_order: list[str],
) -> None:
	if runtime_summary_df.empty:
		return

	prompt_type_order = [
		direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)],
		direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, False)],
	]
	available_prompt_types = [
		prompt_type for prompt_type in prompt_type_order if prompt_type in runtime_summary_df["prompt_type"].unique()
	]
	if not available_prompt_types:
		return

	fig_width = max(9.4, 0.64 * len(feature_order) + 1.8)
	fig_height = max(6.4, 2.9 * len(available_prompt_types) + 1.0)
	fig, axes = plt.subplots(len(available_prompt_types), 1, figsize=(fig_width, fig_height), squeeze=False, sharex=True)

	first_heatmap = None
	for row_index, prompt_type in enumerate(available_prompt_types):
		ax = axes[row_index][0]
		panel_df = runtime_summary_df[runtime_summary_df["prompt_type"] == prompt_type].copy()
		score_df = panel_df.pivot(index="runtime", columns="feature_label", values="association_score")
		score_df = score_df.reindex(index=RUNTIME_ORDER, columns=feature_order)
		score_df.index = [RUNTIME_DISPLAY_LABELS.get(runtime_name, runtime_name.title()) for runtime_name in score_df.index]
		annotation_df = _heatmap_annotation(score_df)

		heatmap = sns.heatmap(
			score_df,
			mask=score_df.isna(),
			annot=annotation_df,
			fmt="",
			cmap="RdBu_r",
			vmin=-1.0,
			vmax=1.0,
			center=0.0,
			linewidths=0.5,
			linecolor="#ffffff",
			cbar=False,
			ax=ax,
			annot_kws={"fontsize": 11},
		)
		if first_heatmap is None:
			first_heatmap = heatmap.collections[0]

		ax.set_ylabel(prompt_type, fontsize=15, labelpad=8)
		ax.tick_params(axis="y", labelrotation=0, labelsize=13)
		ax.tick_params(axis="x", labelrotation=35, labelsize=13)
		if row_index == len(available_prompt_types) - 1:
			ax.set_xlabel("Feature", fontsize=16)
		else:
			ax.set_xlabel("")
			ax.tick_params(axis="x", labelbottom=False)
		for tick_label in ax.get_xticklabels():
			tick_label.set_ha("right")
			tick_label.set_rotation_mode("anchor")

	colorbar_axis = fig.add_axes([0.90, 0.16, 0.024, 0.68])
	colorbar = fig.colorbar(first_heatmap, cax=colorbar_axis)
	colorbar.set_label("Cliff's Delta", fontsize=15)
	colorbar.ax.tick_params(labelsize=12)

	fig.subplots_adjust(left=0.20, right=0.87, bottom=0.24, top=0.97, hspace=0.38)
	fig.savefig(output_dir / "runtime_feature_association_heatmap.png", bbox_inches="tight")
	plt.close(fig)


def _build_gpu_feature_summary_dataframe(
	feature_long_df: pd.DataFrame,
	*,
	min_present: int,
	min_absent: int,
) -> pd.DataFrame:
	if feature_long_df.empty:
		return pd.DataFrame()

	group_columns = ["gpu", "runtime", "feature_name", "feature_label"]
	records: list[dict[str, Any]] = []
	for group_key, group_df in feature_long_df.groupby(group_columns, dropna=False):
		group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
		present_errors = pd.to_numeric(
			group_df.loc[group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		present_errors = present_errors[np.isfinite(present_errors.to_numpy())]
		absent_errors = pd.to_numeric(
			group_df.loc[~group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		absent_errors = absent_errors[np.isfinite(absent_errors.to_numpy())]

		n_present = int(present_errors.shape[0])
		n_absent = int(absent_errors.shape[0])
		is_valid = n_present >= min_present and n_absent >= min_absent
		median_present = float(present_errors.median()) if n_present else float("nan")
		median_absent = float(absent_errors.median()) if n_absent else float("nan")

		records.append(
			{
				"summary_mode": "collapsed_gpu_feature",
				"prompt_type": ALL_PROMPT_TYPES_LABEL,
				"precision": ALL_PRECISIONS_LABEL,
				"runtime": group_values["runtime"],
				"gpu": group_values["gpu"],
				"model_label": ALL_MODELS_LABEL,
				"feature_name": group_values["feature_name"],
				"feature_label": group_values["feature_label"],
				"n_total": int(len(group_df)),
				"n_present": n_present,
				"n_absent": n_absent,
				"median_error_present": median_present,
				"median_error_absent": median_absent,
				"median_error_delta": median_present - median_absent,
				"association_score": _cliffs_delta(present_errors.to_numpy(), absent_errors.to_numpy()) if is_valid else float("nan"),
				"is_valid": bool(is_valid),
			}
		)

	gpu_df = pd.DataFrame(records)
	if gpu_df.empty:
		return gpu_df

	gpu_df["collapsed_model"] = True
	gpu_df["collapsed_precision"] = True
	gpu_df["model_tick_label"] = gpu_df["model_label"]
	gpu_df["gpu_model_label"] = gpu_df["gpu"]
	gpu_df["column_key"] = gpu_df["feature_label"]
	gpu_df["column_tick_label"] = gpu_df["feature_label"]
	gpu_df["direction_label"] = np.where(
		gpu_df["association_score"] > 0.0,
		"Higher Error When Present",
		"Lower Error When Present",
	)
	gpu_df.loc[gpu_df["association_score"].isna(), "direction_label"] = "Insufficient Support"
	return gpu_df.sort_values(group_columns).reset_index(drop=True)


def _save_gpu_feature_summary_heatmap(
	gpu_summary_df: pd.DataFrame,
	output_dir: Path,
	*,
	feature_order: list[str],
) -> None:
	if gpu_summary_df.empty:
		return

	available_gpus = _ordered_gpu_values(gpu_summary_df["gpu"])
	if not available_gpus:
		return

	fig_width = max(9.4, 0.64 * len(feature_order) + 1.8)
	fig_height = max(7.6, 2.45 * len(available_gpus) + 1.0)
	fig, axes = plt.subplots(len(available_gpus), 1, figsize=(fig_width, fig_height), squeeze=False, sharex=True)

	first_heatmap = None
	for row_index, gpu_name in enumerate(available_gpus):
		ax = axes[row_index][0]
		panel_df = gpu_summary_df[gpu_summary_df["gpu"] == gpu_name].copy()
		score_df = panel_df.pivot(index="runtime", columns="feature_label", values="association_score")
		score_df = score_df.reindex(index=RUNTIME_ORDER, columns=feature_order)
		score_df.index = [RUNTIME_DISPLAY_LABELS.get(runtime_name, runtime_name.title()) for runtime_name in score_df.index]
		annotation_df = _heatmap_annotation(score_df)

		heatmap = sns.heatmap(
			score_df,
			mask=score_df.isna(),
			annot=annotation_df,
			fmt="",
			cmap="RdBu_r",
			vmin=-1.0,
			vmax=1.0,
			center=0.0,
			linewidths=0.5,
			linecolor="#ffffff",
			cbar=False,
			ax=ax,
			annot_kws={"fontsize": 11},
		)
		if first_heatmap is None:
			first_heatmap = heatmap.collections[0]

		ax.set_ylabel(f"{gpu_name}", fontsize=15, labelpad=8)
		ax.tick_params(axis="y", labelrotation=0, labelsize=13)
		ax.tick_params(axis="x", labelrotation=35, labelsize=13)
		if row_index == len(available_gpus) - 1:
			ax.set_xlabel("Feature", fontsize=16)
		else:
			ax.set_xlabel("")
			ax.tick_params(axis="x", labelbottom=False)
		for tick_label in ax.get_xticklabels():
			tick_label.set_ha("right")
			tick_label.set_rotation_mode("anchor")

	colorbar_axis = fig.add_axes([0.90, 0.14, 0.024, 0.72])
	colorbar = fig.colorbar(first_heatmap, cax=colorbar_axis)
	colorbar.set_label("Cliff's Delta", fontsize=15)
	colorbar.ax.tick_params(labelsize=12)

	fig.subplots_adjust(left=0.20, right=0.87, bottom=0.20, top=0.98, hspace=0.36)
	fig.savefig(output_dir / "gpu_feature_association_heatmap.png", bbox_inches="tight")
	plt.close(fig)


def _build_model_feature_summary_dataframe(
	feature_long_df: pd.DataFrame,
	*,
	min_present: int,
	min_absent: int,
) -> pd.DataFrame:
	if feature_long_df.empty:
		return pd.DataFrame()

	group_columns = ["model_label", "runtime", "feature_name", "feature_label"]
	records: list[dict[str, Any]] = []
	for group_key, group_df in feature_long_df.groupby(group_columns, dropna=False):
		group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
		present_errors = pd.to_numeric(
			group_df.loc[group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		present_errors = present_errors[np.isfinite(present_errors.to_numpy())]
		absent_errors = pd.to_numeric(
			group_df.loc[~group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		absent_errors = absent_errors[np.isfinite(absent_errors.to_numpy())]

		n_present = int(present_errors.shape[0])
		n_absent = int(absent_errors.shape[0])
		is_valid = n_present >= min_present and n_absent >= min_absent
		median_present = float(present_errors.median()) if n_present else float("nan")
		median_absent = float(absent_errors.median()) if n_absent else float("nan")

		records.append(
			{
				"summary_mode": "collapsed_model_feature",
				"prompt_type": ALL_PROMPT_TYPES_LABEL,
				"precision": ALL_PRECISIONS_LABEL,
				"runtime": group_values["runtime"],
				"gpu": "All GPUs",
				"model_label": group_values["model_label"],
				"feature_name": group_values["feature_name"],
				"feature_label": group_values["feature_label"],
				"n_total": int(len(group_df)),
				"n_present": n_present,
				"n_absent": n_absent,
				"median_error_present": median_present,
				"median_error_absent": median_absent,
				"median_error_delta": median_present - median_absent,
				"association_score": _cliffs_delta(present_errors.to_numpy(), absent_errors.to_numpy()) if is_valid else float("nan"),
				"is_valid": bool(is_valid),
			}
		)

	model_df = pd.DataFrame(records)
	if model_df.empty:
		return model_df

	model_df["collapsed_model"] = True
	model_df["collapsed_precision"] = True
	model_df["model_tick_label"] = model_df["model_label"]
	model_df["gpu_model_label"] = model_df["model_label"]
	model_df["column_key"] = model_df["feature_label"]
	model_df["column_tick_label"] = model_df["feature_label"]
	model_df["direction_label"] = np.where(
		model_df["association_score"] > 0.0,
		"Higher Error When Present",
		"Lower Error When Present",
	)
	model_df.loc[model_df["association_score"].isna(), "direction_label"] = "Insufficient Support"
	return model_df.sort_values(group_columns).reset_index(drop=True)


def _build_model_prompt_type_feature_summary_dataframe(
	feature_long_df: pd.DataFrame,
	*,
	min_present: int,
	min_absent: int,
) -> pd.DataFrame:
	if feature_long_df.empty:
		return pd.DataFrame()

	group_columns = ["model_label", "prompt_type", "feature_name", "feature_label"]
	records: list[dict[str, Any]] = []
	for group_key, group_df in feature_long_df.groupby(group_columns, dropna=False):
		group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
		present_errors = pd.to_numeric(
			group_df.loc[group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		present_errors = present_errors[np.isfinite(present_errors.to_numpy())]
		absent_errors = pd.to_numeric(
			group_df.loc[~group_df["feature_present"], "abs_ai_pct_error"],
			errors="coerce",
		)
		absent_errors = absent_errors[np.isfinite(absent_errors.to_numpy())]

		n_present = int(present_errors.shape[0])
		n_absent = int(absent_errors.shape[0])
		is_valid = n_present >= min_present and n_absent >= min_absent
		median_present = float(present_errors.median()) if n_present else float("nan")
		median_absent = float(absent_errors.median()) if n_absent else float("nan")

		records.append(
			{
				"summary_mode": "collapsed_runtime_model_prompt_type_feature",
				"prompt_type": group_values["prompt_type"],
				"precision": ALL_PRECISIONS_LABEL,
				"runtime": ALL_RUNTIMES_LABEL,
				"gpu": "All GPUs",
				"model_label": group_values["model_label"],
				"feature_name": group_values["feature_name"],
				"feature_label": group_values["feature_label"],
				"n_total": int(len(group_df)),
				"n_present": n_present,
				"n_absent": n_absent,
				"median_error_present": median_present,
				"median_error_absent": median_absent,
				"median_error_delta": median_present - median_absent,
				"association_score": _cliffs_delta(present_errors.to_numpy(), absent_errors.to_numpy()) if is_valid else float("nan"),
				"is_valid": bool(is_valid),
			}
		)

	model_prompt_type_df = pd.DataFrame(records)
	if model_prompt_type_df.empty:
		return model_prompt_type_df

	model_prompt_type_df["collapsed_model"] = False
	model_prompt_type_df["collapsed_precision"] = True
	model_prompt_type_df["model_tick_label"] = model_prompt_type_df["model_label"]
	model_prompt_type_df["gpu_model_label"] = model_prompt_type_df["model_label"]
	model_prompt_type_df["column_key"] = model_prompt_type_df["feature_label"]
	model_prompt_type_df["column_tick_label"] = model_prompt_type_df["feature_label"]
	model_prompt_type_df["direction_label"] = np.where(
		model_prompt_type_df["association_score"] > 0.0,
		"Higher Error When Present",
		"Lower Error When Present",
	)
	model_prompt_type_df.loc[model_prompt_type_df["association_score"].isna(), "direction_label"] = "Insufficient Support"
	return model_prompt_type_df.sort_values(group_columns).reset_index(drop=True)


def _save_model_feature_summary_heatmap(
	model_summary_df: pd.DataFrame,
	output_dir: Path,
	*,
	feature_order: list[str],
) -> None:
	if model_summary_df.empty:
		return

	available_models = _ordered_model_values(model_summary_df["model_label"])
	if not available_models:
		return

	fig_width = max(9.4, 0.64 * len(feature_order) + 1.8)
	fig_height = max(7.6, 2.45 * len(available_models) + 1.0)
	fig, axes = plt.subplots(len(available_models), 1, figsize=(fig_width, fig_height), squeeze=False, sharex=True)

	first_heatmap = None
	for row_index, model_name in enumerate(available_models):
		ax = axes[row_index][0]
		panel_df = model_summary_df[model_summary_df["model_label"] == model_name].copy()
		score_df = panel_df.pivot(index="runtime", columns="feature_label", values="association_score")
		score_df = score_df.reindex(index=RUNTIME_ORDER, columns=feature_order)
		score_df.index = [RUNTIME_DISPLAY_LABELS.get(runtime_name, runtime_name.title()) for runtime_name in score_df.index]
		annotation_df = _heatmap_annotation(score_df)

		heatmap = sns.heatmap(
			score_df,
			mask=score_df.isna(),
			annot=annotation_df,
			fmt="",
			cmap="RdBu_r",
			vmin=-1.0,
			vmax=1.0,
			center=0.0,
			linewidths=0.5,
			linecolor="#ffffff",
			cbar=False,
			ax=ax,
			annot_kws={"fontsize": 11},
		)
		if first_heatmap is None:
			first_heatmap = heatmap.collections[0]

		ax.set_ylabel(model_name, fontsize=15, labelpad=8)
		ax.tick_params(axis="y", labelrotation=0, labelsize=13)
		ax.tick_params(axis="x", labelrotation=35, labelsize=13)
		if row_index == len(available_models) - 1:
			ax.set_xlabel("Feature", fontsize=16)
		else:
			ax.set_xlabel("")
			ax.tick_params(axis="x", labelbottom=False)
		for tick_label in ax.get_xticklabels():
			tick_label.set_ha("right")
			tick_label.set_rotation_mode("anchor")

	colorbar_axis = fig.add_axes([0.90, 0.14, 0.024, 0.72])
	colorbar = fig.colorbar(first_heatmap, cax=colorbar_axis)
	colorbar.set_label("Cliff's Delta", fontsize=15)
	colorbar.ax.tick_params(labelsize=12)

	fig.subplots_adjust(left=0.23, right=0.87, bottom=0.20, top=0.98, hspace=0.36)
	fig.savefig(output_dir / "model_feature_association_heatmap.png", bbox_inches="tight")
	plt.close(fig)


def _save_model_prompt_type_feature_summary_heatmap(
	model_prompt_type_summary_df: pd.DataFrame,
	output_dir: Path,
	*,
	feature_order: list[str],
) -> None:
	if model_prompt_type_summary_df.empty:
		return

	prompt_type_order = [
		direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)],
		direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, False)],
	]
	available_models = _ordered_model_values(model_prompt_type_summary_df["model_label"])
	if not available_models:
		return

	fig_width = max(9.4, 0.64 * len(feature_order) + 1.8)
	fig_height = max(7.6, 2.45 * len(available_models) + 1.0)
	fig, axes = plt.subplots(len(available_models), 1, figsize=(fig_width, fig_height), squeeze=False, sharex=True)

	first_heatmap = None
	for row_index, model_name in enumerate(available_models):
		ax = axes[row_index][0]
		panel_df = model_prompt_type_summary_df[model_prompt_type_summary_df["model_label"] == model_name].copy()
		score_df = panel_df.pivot(index="prompt_type", columns="feature_label", values="association_score")
		score_df = score_df.reindex(index=prompt_type_order, columns=feature_order)
		annotation_df = _heatmap_annotation(score_df)

		heatmap = sns.heatmap(
			score_df,
			mask=score_df.isna(),
			annot=annotation_df,
			fmt="",
			cmap="RdBu_r",
			vmin=-1.0,
			vmax=1.0,
			center=0.0,
			linewidths=0.5,
			linecolor="#ffffff",
			cbar=False,
			ax=ax,
			annot_kws={"fontsize": 11},
		)
		if first_heatmap is None:
			first_heatmap = heatmap.collections[0]

		ax.set_ylabel(model_name, fontsize=15, labelpad=8)
		ax.tick_params(axis="y", labelrotation=0, labelsize=13)
		ax.tick_params(axis="x", labelrotation=35, labelsize=13)
		if row_index == len(available_models) - 1:
			ax.set_xlabel("Feature", fontsize=16)
		else:
			ax.set_xlabel("")
			ax.tick_params(axis="x", labelbottom=False)
		for tick_label in ax.get_xticklabels():
			tick_label.set_ha("right")
			tick_label.set_rotation_mode("anchor")

	colorbar_axis = fig.add_axes([0.90, 0.14, 0.024, 0.72])
	colorbar = fig.colorbar(first_heatmap, cax=colorbar_axis)
	colorbar.set_label("Cliff's Delta", fontsize=15)
	colorbar.ax.tick_params(labelsize=12)

	fig.subplots_adjust(left=0.23, right=0.87, bottom=0.20, top=0.98, hspace=0.36)
	fig.savefig(output_dir / "figure1_model_feature_association_heatmap.png", bbox_inches="tight")
	plt.close(fig)


def _save_association_heatmap_grid(
	association_df: pd.DataFrame,
	prompt_type: str,
	output_dir: Path,
	*,
	feature_order: list[str],
	column_order: list[str],
	figure_suffix: str,
	figure_descriptor: str,
) -> None:
	evidence_df = association_df[association_df["prompt_type"] == prompt_type].copy()
	if evidence_df.empty:
		return

	precisions = [precision for precision in db_reader.AI_PRECISIONS if precision in evidence_df["precision"].unique()]
	if not precisions and ALL_PRECISIONS_LABEL in evidence_df["precision"].unique():
		precisions = [ALL_PRECISIONS_LABEL]
	runtimes = [runtime for runtime in RUNTIME_ORDER if runtime in evidence_df["runtime"].unique()]
	if not precisions or not runtimes:
		return

	collapsed_model = bool(evidence_df["collapsed_model"].iloc[0])

	fig_width = max(12.0, 1.1 * len(column_order) + 3.5)
	fig_height = max(8.5, 0.5 * len(feature_order) * len(precisions) + 2.0)
	fig, axes = plt.subplots(len(precisions), len(runtimes), figsize=(fig_width, fig_height), squeeze=False)

	first_heatmap = None
	for row_index, precision in enumerate(precisions):
		for column_index, runtime in enumerate(runtimes):
			ax = axes[row_index][column_index]
			panel_df = evidence_df[
				(evidence_df["precision"] == precision)
				& (evidence_df["runtime"] == runtime)
			].copy()
			ax.set_facecolor("#ece8dc")
			if panel_df.empty:
				ax.axis("off")
				ax.set_title(f"{precision.upper()} | {runtime.upper()}\nNo Data")
				continue

			score_df = panel_df.pivot(index="feature_label", columns="column_key", values="association_score")
			score_df = score_df.reindex(index=feature_order, columns=column_order)
			annotation_df = _heatmap_annotation(score_df)
			heatmap = sns.heatmap(
				score_df,
				mask=score_df.isna(),
				annot=annotation_df,
				fmt="",
				cmap="RdBu_r",
				vmin=-1.0,
				vmax=1.0,
				center=0.0,
				linewidths=0.5,
				linecolor="#ffffff",
				cbar=False,
				ax=ax,
				annot_kws={"fontsize": 7},
			)
			if first_heatmap is None:
				first_heatmap = heatmap.collections[0]

			if not collapsed_model:
				for separator_index in range(1, len(column_order)):
					previous_gpu = column_order[separator_index - 1].split("::", maxsplit=1)[0]
					current_gpu = column_order[separator_index].split("::", maxsplit=1)[0]
					if previous_gpu != current_gpu:
						ax.axvline(separator_index, color="#444444", linewidth=1.25)

			ax.set_title(f"{precision.upper()} | {runtime.upper()}")
			ax.set_xlabel("GPU" if collapsed_model else "GPU / Model")
			ax.set_ylabel("Feature" if column_index == 0 else "")
			if collapsed_model:
				ax.set_xticklabels(column_order)
			else:
				ax.set_xticklabels([column_name.split("::", maxsplit=1)[1] for column_name in column_order])
			ax.tick_params(axis="x", labelrotation=35, labelsize=8)
			ax.tick_params(axis="y", labelsize=8)
			for tick_label in ax.get_xticklabels():
				tick_label.set_ha("right")
				tick_label.set_rotation_mode("anchor")

			if not collapsed_model:
				top_axis = ax.twiny()
				top_axis.set_xlim(ax.get_xlim())
				gpu_centers = _gpu_group_centers(column_order)
				top_axis.set_xticks([center for center, _ in gpu_centers])
				top_axis.set_xticklabels([gpu_name for _, gpu_name in gpu_centers], fontsize=9)
				top_axis.tick_params(axis="x", labelrotation=0, length=0, pad=4)
				top_axis.spines["top"].set_visible(False)
				top_axis.spines["bottom"].set_visible(False)
				top_axis.spines["left"].set_visible(False)
				top_axis.spines["right"].set_visible(False)

	if first_heatmap is not None:
		colorbar_axis = fig.add_axes([0.945, 0.14, 0.015, 0.70])
		colorbar = fig.colorbar(first_heatmap, cax=colorbar_axis)
		colorbar.set_label("Cliff's Delta: Positive Means Higher AI Error When Feature Is Present")

	fig.suptitle(
		f"Feature Association With AI Misprediction\n{PROMPT_TYPE_LABEL}: {prompt_type}{figure_descriptor}",
		fontsize=14,
		y=0.995,
	)
	fig.subplots_adjust(left=0.20, right=0.92, bottom=0.12, top=0.90, wspace=0.28, hspace=0.34)
	figure_path = output_dir / f"{_safe_filename(prompt_type)}{figure_suffix}_feature_association_heatmaps.png"
	fig.savefig(figure_path, bbox_inches="tight")
	plt.close(fig)


def _save_precision_summary_bars(association_df: pd.DataFrame, output_dir: Path) -> None:
	valid_df = association_df[association_df["is_valid"]].copy()
	if valid_df.empty:
		return

	summary_df = (
		valid_df.groupby(["precision", "feature_label"], as_index=False)
		.agg(
			median_abs_association=("association_score", lambda series: float(pd.Series(series).abs().median())),
			positive_cell_fraction=("association_score", lambda series: float((pd.Series(series) > 0.0).mean() * 100.0)),
			valid_cell_count=("association_score", "size"),
		)
	)
	if summary_df.empty:
		return

	fig, axes = plt.subplots(1, len(db_reader.AI_PRECISIONS), figsize=(15, 7), squeeze=False, sharey=True)
	feature_order = _feature_order(valid_df)
	for axis_index, precision in enumerate(db_reader.AI_PRECISIONS):
		ax = axes[0][axis_index]
		panel_df = summary_df[summary_df["precision"] == precision].copy()
		if panel_df.empty:
			ax.axis("off")
			continue
		panel_df = panel_df.sort_values("median_abs_association", ascending=True)
		sns.barplot(
			data=panel_df,
			x="median_abs_association",
			y="feature_label",
			order=feature_order,
			orient="h",
			color="#3b6f91",
			ax=ax,
		)
		ax.set_xlim(0.0, 1.0)
		ax.set_title(precision.upper())
		ax.set_xlabel("Median |Cliff's Delta|")
		ax.set_ylabel("Feature" if axis_index == 0 else "")
		for patch, cell_count in zip(ax.patches, panel_df.set_index("feature_label").reindex(feature_order)["valid_cell_count"].fillna(0).astype(int)):
			ax.text(patch.get_width() + 0.02, patch.get_y() + patch.get_height() / 2.0, f"n={cell_count}", va="center", fontsize=7)

	fig.suptitle("Consistency Ranking Across GPU / Model / Runtime Cells", fontsize=14, y=0.98)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
	fig.savefig(output_dir / "feature_association_summary.png", bbox_inches="tight")
	plt.close(fig)


def generate_paper_plots(
	sample_with_features_df: pd.DataFrame,
	output_dir: Path,
	*,
	min_present: int,
	min_absent: int,
	only_shared_samples: bool,
) -> pd.DataFrame:
	output_dir.mkdir(parents=True, exist_ok=True)
	_set_plot_theme()

	filtered_sample_df = _filter_only_shared_samples(sample_with_features_df) if only_shared_samples else sample_with_features_df.copy()
	clean_df = _clean_sample_dataframe(filtered_sample_df)
	feature_long_df = _feature_presence_long_frame(clean_df)
	association_frames = [
		_build_association_dataframe(
			feature_long_df,
			min_present=min_present,
			min_absent=min_absent,
			collapse_model=collapse_model,
			collapse_precision=collapse_precision,
			summary_mode=summary_mode,
		)
		for summary_mode, _, collapse_model, collapse_precision in SUMMARY_VARIANTS
	]
	association_df = pd.concat([frame for frame in association_frames if not frame.empty], ignore_index=True)
	if association_df.empty:
		print("No sample-level feature associations could be computed.")
		return association_df

	for summary_mode, descriptor, _, _ in SUMMARY_VARIANTS:
		mode_df = association_df[association_df["summary_mode"] == summary_mode].copy()
		if mode_df.empty:
			continue
		feature_order = _feature_order(mode_df)
		column_order = _column_order(mode_df)
		figure_suffix = "" if summary_mode == "full" else f"_{summary_mode}"
		figure_descriptor = "" if not descriptor else f"\n{descriptor}"
		for prompt_type in [
			direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(False, False)],
			direct_visualize_results.PLOT_EVIDENCE_CONFIGURATION_LABELS[(True, False)],
		]:
			_save_association_heatmap_grid(
				mode_df,
				prompt_type,
				output_dir,
				feature_order=feature_order,
				column_order=column_order,
				figure_suffix=figure_suffix,
				figure_descriptor=figure_descriptor,
			)
	runtime_summary_df = _build_runtime_feature_summary_dataframe(
		feature_long_df,
		min_present=min_present,
		min_absent=min_absent,
	)
	if not runtime_summary_df.empty:
		_save_runtime_feature_summary_heatmap(
			runtime_summary_df,
			output_dir,
			feature_order=_runtime_feature_order(runtime_summary_df),
		)
	gpu_summary_df = _build_gpu_feature_summary_dataframe(
		feature_long_df,
		min_present=min_present,
		min_absent=min_absent,
	)
	if not gpu_summary_df.empty:
		_save_gpu_feature_summary_heatmap(
			gpu_summary_df,
			output_dir,
			feature_order=_runtime_feature_order(gpu_summary_df),
		)
	model_summary_df = _build_model_feature_summary_dataframe(
		feature_long_df,
		min_present=min_present,
		min_absent=min_absent,
	)
	if not model_summary_df.empty:
		_save_model_feature_summary_heatmap(
			model_summary_df,
			output_dir,
			feature_order=_runtime_feature_order(model_summary_df),
		)
	model_prompt_type_summary_df = _build_model_prompt_type_feature_summary_dataframe(
		feature_long_df,
		min_present=min_present,
		min_absent=min_absent,
	)
	if not model_prompt_type_summary_df.empty:
		_save_model_prompt_type_feature_summary_heatmap(
			model_prompt_type_summary_df,
			output_dir,
			feature_order=_runtime_feature_order(model_prompt_type_summary_df),
		)
	_save_precision_summary_bars(
		association_df[association_df["summary_mode"] == "full"].copy(),
		output_dir,
	)
	combined_association_df = pd.concat(
		[association_df, runtime_summary_df, gpu_summary_df, model_summary_df, model_prompt_type_summary_df],
		ignore_index=True,
	)
	combined_association_df.to_csv(output_dir / "feature_error_associations.csv", index=False)
	return combined_association_df


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate paper plots correlating AI error with code-feature votes.",
	)
	parser.add_argument("--gpuflopsDbUri", default=None, help="Override URI for gpuflops_db")
	parser.add_argument("--codeFeaturesDbUri", default=None, help="Override URI for code_features_db")
	parser.add_argument(
		"--includeDryRun",
		action="store_true",
		help="Include dry-run thread IDs when present.",
	)
	parser.add_argument(
		"--onlySharedSamples",
		action="store_true",
		help="Keep only kernel samples that have at least one completed row for every model name, matched by program name, kernel name, GPU, and prompt type.",
	)
	parser.add_argument(
		"--kernelErrorMetric",
		default="max_abs_ai_pct_error",
		choices=["max_abs_ai_pct_error", "median_abs_ai_pct_error", "mean_abs_ai_pct_error"],
		help="Retained for compatibility with the shared loader; sample-level plots ignore this kernel summary setting.",
	)
	parser.add_argument(
		"--minPresent",
		type=int,
		default=DEFAULT_MIN_PRESENT,
		help="Minimum number of feature-present samples required before plotting an association cell.",
	)
	parser.add_argument(
		"--minAbsent",
		type=int,
		default=DEFAULT_MIN_ABSENT,
		help="Minimum number of feature-absent samples required before plotting an association cell.",
	)
	parser.add_argument(
		"--outputDir",
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where the generated plots will be written.",
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	frames = db_reader.load_analysis_frames(
		gpuflops_db_uri=args.gpuflopsDbUri,
		code_features_db_uri=args.codeFeaturesDbUri,
		include_dry_run=args.includeDryRun,
		kernel_error_metric=args.kernelErrorMetric,
	)
	sample_with_features_df = frames["sample_with_features_df"]
	print("Sample merge diagnostics:")
	for key, value in frames["sample_merge_diagnostics"].items():
		print(f"- {key}: {value}")

	association_df = generate_paper_plots(
		sample_with_features_df,
		Path(args.outputDir),
		min_present=args.minPresent,
		min_absent=args.minAbsent,
		only_shared_samples=args.onlySharedSamples,
	)
	if not association_df.empty:
		valid_cell_count = int(association_df["is_valid"].sum())
		print(f"\nComputed {len(association_df)} association cells ({valid_cell_count} valid after support masking).")
	print(f"\nSaved plots to {args.outputDir}")


if __name__ == "__main__":
	main()
