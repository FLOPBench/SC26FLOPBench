#!/usr/bin/env python3

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SITE_DATA_CANDIDATES = [
    REPO_ROOT / "research-paper" / "icse-updated-paper" / "icse_student_handoff_2026-06-11" / "original_inputs" / "_website_mirror" / "site-data.json",
]
DATASET_CANDIDATES = [
    REPO_ROOT / "dataset-creation" / "gpuFLOPBench.json",
]
TABLES_DIR = ROOT / "tables"
GENERATED_DIR = ROOT / "generated"

CORE_COLUMNS = ["program_name", "runtime", "kernel_mangled_name", "gpu"]
MODELS = ["GPT 5.4", "GPT OSS", "Opus 4.6"]
PROMPTS = ["Source-Only", "Source+SASS"]
PRECISIONS = ["fp16", "fp32", "fp64"]
SHARED_REQUIREMENTS = [
    ("GPT 5.4", "Source-Only"),
    ("GPT 5.4", "Source+SASS"),
    ("GPT OSS", "Source-Only"),
    ("GPT OSS", "Source+SASS"),
    ("Opus 4.6", "Source-Only"),
    ("Opus 4.6", "Source+SASS"),
]
GPU_ARCH = {
    "3080": "sm_86",
    "A10": "sm_86",
    "A100": "sm_80",
    "H100": "sm_90",
}
SOURCE_HEURISTIC_LABEL = "Source lexical"
CONTEXT_MEDIAN_LABEL = "Context median CV"
SOURCE_LEARNED_LABEL = "Source learned RF"
SOURCE_EXTRA_TREES_LABEL = "Source learned ET"
SASS_HEURISTIC_LABEL = "SASS mnemonic"
SASS_LEARNED_LABEL = "SASS learned RF"
SASS_EXTRA_TREES_LABEL = "SASS learned ET"
BOOTSTRAP_SAMPLES = 2000
LEARNED_BASELINE_SEEDS = [0, 1, 2, 3, 4]
BUDGET_FRACTIONS = [0.05, 0.10, 0.20, 0.30]


def load_prediction_rows() -> pd.DataFrame:
    site_data_path = next((path for path in SITE_DATA_CANDIDATES if path.exists()), None)
    if site_data_path is None:
        raise FileNotFoundError(f"Could not find site-data.json in: {SITE_DATA_CANDIDATES}")
    site_data = json.loads(site_data_path.read_text())
    rows = pd.DataFrame(site_data["llmIndex"]["predictionRows"])
    numeric_columns = [
        "expected_ai",
        "predicted_ai",
        "balance_point",
        "ai_abs_percent_error",
        "query_time",
        "cost_usd",
    ]
    for column_name in numeric_columns:
        rows[column_name] = pd.to_numeric(rows[column_name], errors="coerce")
    return rows


def load_dataset() -> dict:
    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError(f"Could not find gpuFLOPBench.json in: {DATASET_CANDIDATES}")
    return json.loads(dataset_path.read_text())


def shared_subset(rows: pd.DataFrame) -> pd.DataFrame:
    availability = rows.groupby(CORE_COLUMNS + ["model_name", "prompt_type"]).size().reset_index(name="n")
    availability_pivot = availability.pivot_table(
        index=CORE_COLUMNS,
        columns=["model_name", "prompt_type"],
        values="n",
        fill_value=0,
    )
    shared_keys = availability_pivot[(availability_pivot[SHARED_REQUIREMENTS] > 0).all(axis=1)].index.to_frame(index=False)
    return rows.merge(shared_keys, on=CORE_COLUMNS, how="inner")


def _task_a_balanced_accuracy(expected_ai: pd.Series, predicted_ai: pd.Series) -> float:
    true_zero = expected_ai.eq(0.0)
    predicted_zero = predicted_ai.eq(0.0)
    zero_recall = float((predicted_zero[true_zero] == True).mean())
    nonzero_recall = float((predicted_zero[~true_zero] == False).mean())
    return (zero_recall + nonzero_recall) / 2.0


def _nonzero_bound_labels(
    expected_ai: pd.Series,
    predicted_ai: pd.Series,
    balance_point: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    nonzero_rows = expected_ai.gt(0.0)
    truth = np.where(expected_ai[nonzero_rows] >= balance_point[nonzero_rows], "cb", "bb")
    pred = np.where(
        predicted_ai[nonzero_rows].fillna(0.0) <= 0.0,
        "bb",
        np.where(predicted_ai[nonzero_rows] >= balance_point[nonzero_rows], "cb", "bb"),
    )
    return truth, pred


def _task_c_balanced_accuracy(expected_ai: pd.Series, predicted_ai: pd.Series, balance_point: pd.Series) -> float:
    truth, pred = _nonzero_bound_labels(expected_ai, predicted_ai, balance_point)
    bb_recall = float((pred[truth == "bb"] == "bb").mean())
    cb_recall = float((pred[truth == "cb"] == "cb").mean())
    return (bb_recall + cb_recall) / 2.0


def _task_d_balanced_accuracy(expected_ai: pd.Series, predicted_ai: pd.Series, balance_point: pd.Series) -> float:
    truth = np.where(expected_ai.eq(0.0), "zero", np.where(expected_ai >= balance_point, "cb", "bb"))
    pred = np.where(
        predicted_ai.fillna(0.0) <= 0.0,
        "zero",
        np.where(predicted_ai >= balance_point, "cb", "bb"),
    )
    recalls = []
    for label in ["zero", "bb", "cb"]:
        recalls.append(float((pred[truth == label] == label).mean()))
    return sum(recalls) / 3.0


def _nonzero_median_ape(expected_ai: pd.Series, predicted_ai: pd.Series) -> float:
    nonzero_rows = expected_ai.gt(0.0)
    ape = (
        (predicted_ai[nonzero_rows] - expected_ai[nonzero_rows]).abs()
        / expected_ai[nonzero_rows]
        * 100.0
    )
    return float(ape.median())


def _strip_comments(source_text: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", " ", source_text, flags=re.MULTILINE | re.DOTALL)


def build_source_heuristics(dataset: dict) -> pd.DataFrame:
    rows = []
    fp32_math = re.compile(r"\b(?:sinf|cosf|tanf|expf|logf|sqrtf|rsqrtf|powf)\b")
    fp64_math = re.compile(r"\b(?:sin|cos|tan|exp|log|sqrt|rsqrt|pow)\b")
    arithmetic_keywords = re.compile(r"\b(?:fma|mad|mul|add|sub|div)\b")

    for program_name, program_data in dataset.items():
        kernel_sources = {kernel_name: [] for kernel_name in program_data["kernels"].keys()}
        for source_path, kernel_names in program_data["source_to_kernels"].items():
            source_text = program_data["sources"].get(source_path, "") or ""
            for kernel_name in kernel_names:
                kernel_sources.setdefault(kernel_name, []).append(source_text)

        for kernel_name, source_texts in kernel_sources.items():
            merged_text = _strip_comments("\n".join(source_texts))
            approx_memory_refs = merged_text.count("[") + merged_text.count("->")
            approx_arithmetic = (
                sum(merged_text.count(token) for token in ["+", "-", "*", "/"])
                + len(arithmetic_keywords.findall(merged_text))
            )
            approx_ai = approx_arithmetic / (4.0 * max(approx_memory_refs, 1))
            has_fp16 = bool(re.search(r"\b(?:__half2?|half2?|half)\b", merged_text))
            has_fp32 = bool(re.search(r"\bfloat\b", merged_text) or fp32_math.search(merged_text))
            has_fp64 = bool(re.search(r"\bdouble\b", merged_text) or fp64_math.search(merged_text))
            rows.append(
                {
                    "program_name": program_name,
                    "kernel_mangled_name": kernel_name,
                    "predicted_ai_fp16": approx_ai if has_fp16 else 0.0,
                    "predicted_ai_fp32": approx_ai if has_fp32 else 0.0,
                    "predicted_ai_fp64": approx_ai if has_fp64 else 0.0,
                }
            )

    return pd.DataFrame(rows)


def build_sass_heuristics(dataset: dict) -> pd.DataFrame:
    rows = []
    for program_name, program_data in dataset.items():
        for kernel_name, kernel_data in program_data["kernels"].items():
            for gpu_name, arch_name in GPU_ARCH.items():
                imix = (kernel_data.get("imix") or {}).get(arch_name, {})
                global_memory_ops = (
                    imix.get("LDG", 0)
                    + imix.get("STG", 0)
                    + imix.get("ATOMG", 0)
                    + imix.get("ATOM", 0)
                )
                denominator = max(global_memory_ops, 1)
                fp16_ops = (
                    2 * imix.get("HFMA2", 0)
                    + 2 * imix.get("HADD2", 0)
                    + 2 * imix.get("HMUL2", 0)
                    + 16 * imix.get("HMMA", 0)
                )
                fp32_ops = (
                    2 * imix.get("FFMA", 0)
                    + imix.get("FADD", 0)
                    + imix.get("FMUL", 0)
                    + imix.get("MUFU", 0)
                    + imix.get("FSEL", 0)
                    + imix.get("FMNMX", 0)
                )
                fp64_ops = 2 * imix.get("DFMA", 0) + imix.get("DADD", 0) + imix.get("DMUL", 0)
                rows.append(
                    {
                        "program_name": program_name,
                        "kernel_mangled_name": kernel_name,
                        "gpu": gpu_name,
                        "predicted_ai_fp16": fp16_ops / (2.0 * denominator),
                        "predicted_ai_fp32": fp32_ops / (4.0 * denominator),
                        "predicted_ai_fp64": fp64_ops / (8.0 * denominator),
                    }
                )
    return pd.DataFrame(rows)


def build_source_feature_rows(dataset: dict) -> pd.DataFrame:
    rows = []
    comment_re = re.compile(r"//.*?$|/\*.*?\*/", flags=re.MULTILINE | re.DOTALL)
    math_pattern = re.compile(r"\b(?:sin|cos|tan|exp|log|sqrt|pow|rsqrt|fma|fmaf|erf|mufu)f?\b")
    kernel_index_pattern = re.compile(r"blockidx|threadidx|get_global_id|omp_get", flags=re.IGNORECASE)

    for program_name, program_data in dataset.items():
        kernel_sources = {kernel_name: [] for kernel_name in program_data["kernels"].keys()}
        for source_path, kernel_names in program_data["source_to_kernels"].items():
            source_text = program_data["sources"].get(source_path, "") or ""
            for kernel_name in kernel_names:
                kernel_sources.setdefault(kernel_name, []).append(source_text)

        for kernel_name, source_texts in kernel_sources.items():
            merged_text = comment_re.sub(" ", "\n".join(source_texts))
            lower_text = merged_text.lower()
            rows.append(
                {
                    "program_name": program_name,
                    "kernel_mangled_name": kernel_name,
                    "src_len": len(merged_text),
                    "src_lines": merged_text.count("\n") + 1,
                    "src_arith_ops": sum(merged_text.count(token) for token in ["+", "-", "*", "/", "%"]),
                    "src_add": merged_text.count("+"),
                    "src_mul": merged_text.count("*"),
                    "src_div": merged_text.count("/"),
                    "src_mod": merged_text.count("%"),
                    "src_mem_refs": merged_text.count("[") + merged_text.count("->"),
                    "src_loops": len(re.findall(r"\b(for|while|do)\b", merged_text)),
                    "src_ifs": len(re.findall(r"\bif\b", merged_text)),
                    "src_switch": len(re.findall(r"\bswitch\b", merged_text)),
                    "src_atomic": lower_text.count("atomic"),
                    "src_shared": lower_text.count("__shared__") + lower_text.count("shared"),
                    "src_sync": lower_text.count("__syncthreads") + lower_text.count("barrier") + lower_text.count("sync"),
                    "src_float": len(re.findall(r"\bfloat\b", merged_text)),
                    "src_double": len(re.findall(r"\bdouble\b", merged_text)),
                    "src_half": len(re.findall(r"__half|\bhalf\b", merged_text)),
                    "src_math": len(math_pattern.findall(lower_text)),
                    "src_const": len(re.findall(r"\bconst\b", merged_text)),
                    "src_template": len(re.findall(r"\btemplate\b", merged_text)),
                    "src_idx": len(kernel_index_pattern.findall(lower_text)),
                }
            )
    return pd.DataFrame(rows)


def build_sass_feature_rows(dataset: dict) -> pd.DataFrame:
    rows = []
    for program_name, program_data in dataset.items():
        for kernel_name, kernel_data in program_data["kernels"].items():
            for gpu_name, arch_name in GPU_ARCH.items():
                imix = (kernel_data.get("imix") or {}).get(arch_name, {})

                def instruction_count(*mnemonics: str) -> int:
                    return sum(imix.get(mnemonic, 0) for mnemonic in mnemonics)

                rows.append(
                    {
                        "program_name": program_name,
                        "kernel_mangled_name": kernel_name,
                        "gpu": gpu_name,
                        "sass_total_inst": sum(imix.values()),
                        "sass_mem_global": instruction_count("LDG", "STG", "ATOMG", "ATOM"),
                        "sass_mem_shared": instruction_count("LDS", "STS"),
                        "sass_mem_local": instruction_count("LDL", "STL", "LD", "ST", "LDC", "ULDC"),
                        "sass_branch": instruction_count("BRA", "BSSY", "BSYNC", "CALL", "RET", "EXIT", "BREAK", "BRX", "BRXU"),
                        "sass_integer": instruction_count(
                            "IMAD",
                            "IADD3",
                            "ISETP",
                            "LOP3",
                            "LEA",
                            "SHF",
                            "IMNMX",
                            "UIADD3",
                            "UIMAD",
                            "VIADD",
                            "VIMNMX",
                            "VIADDMNMX",
                            "IABS",
                            "POPC",
                            "FLO",
                            "SGXT",
                        ),
                        "sass_fp16": instruction_count("HFMA2", "HADD2", "HMUL2", "HMMA", "HSETP2", "HMNMX2"),
                        "sass_fp32": instruction_count(
                            "FFMA",
                            "FADD",
                            "FMUL",
                            "MUFU",
                            "FSEL",
                            "FMNMX",
                            "FSETP",
                            "FRND",
                            "FCHK",
                            "FSET",
                            "F2F",
                            "F2I",
                            "I2F",
                        ),
                        "sass_fp64": instruction_count("DFMA", "DADD", "DMUL", "DSETP"),
                        "sass_special": instruction_count("MUFU", "SHFL", "PRMT", "VOTE", "VOTEU", "MATCH", "WARPSYNC"),
                        "sass_atomic": instruction_count("ATOMG", "ATOM", "ATOMS"),
                        "sass_barrier": instruction_count("BAR", "MEMBAR", "ERRBAR", "CGAERRBAR"),
                        "sass_predicate": instruction_count("ISETP", "FSETP", "DSETP", "UISETP", "PLOP3", "P2R", "R2P"),
                    }
                )
    return pd.DataFrame(rows)


def deterministic_baseline_predictions(shared_precision_rows: pd.DataFrame, dataset: dict) -> pd.DataFrame:
    source_heuristics = build_source_heuristics(dataset)
    sass_heuristics = build_sass_heuristics(dataset)

    source_eval = shared_precision_rows.merge(
        source_heuristics,
        on=["program_name", "kernel_mangled_name"],
        how="left",
    )
    sass_eval = shared_precision_rows.merge(
        sass_heuristics,
        on=["program_name", "kernel_mangled_name", "gpu"],
        how="left",
    )

    prediction_rows = []
    for baseline_name, prompt_type, frame in [
        (SOURCE_HEURISTIC_LABEL, "Source-Only", source_eval),
        (SASS_HEURISTIC_LABEL, "Source+SASS", sass_eval),
    ]:
        for precision in PRECISIONS:
            precision_rows = frame[frame["precision"] == precision].copy()
            predicted_column = f"predicted_ai_{precision}"
            precision_rows["predicted_ai"] = precision_rows[predicted_column]
            precision_rows["baseline_name"] = baseline_name
            precision_rows["baseline_family"] = "deterministic"
            precision_rows["prompt_type"] = prompt_type
            prediction_rows.append(
                precision_rows[
                    CORE_COLUMNS
                    + [
                        "precision",
                        "expected_ai",
                        "balance_point",
                        "predicted_ai",
                        "baseline_name",
                        "baseline_family",
                        "prompt_type",
                    ]
                ]
            )
    return pd.concat(prediction_rows, ignore_index=True)


def context_median_predictions(shared_precision_rows: pd.DataFrame) -> pd.DataFrame:
    prediction_rows = []
    group_kfold = GroupKFold(n_splits=5)
    for precision in PRECISIONS:
        precision_rows = shared_precision_rows[shared_precision_rows["precision"] == precision].copy().reset_index(drop=True)
        groups = precision_rows["program_name"].copy()
        predicted_ai = np.zeros(len(precision_rows))
        for train_index, test_index in group_kfold.split(precision_rows, groups=groups):
            train_rows = precision_rows.iloc[train_index].copy()
            test_rows = precision_rows.iloc[test_index].copy()
            runtime_gpu_median = (
                train_rows.groupby(["runtime", "gpu"])["expected_ai"].median().to_dict()
            )
            runtime_median = train_rows.groupby("runtime")["expected_ai"].median().to_dict()
            gpu_median = train_rows.groupby("gpu")["expected_ai"].median().to_dict()
            global_median = float(train_rows["expected_ai"].median())
            fold_predictions = []
            for _, test_row in test_rows.iterrows():
                fold_predictions.append(
                    runtime_gpu_median.get(
                        (test_row["runtime"], test_row["gpu"]),
                        runtime_median.get(
                            test_row["runtime"],
                            gpu_median.get(test_row["gpu"], global_median),
                        ),
                    )
                )
            predicted_ai[test_index] = fold_predictions
        precision_rows["predicted_ai"] = predicted_ai
        precision_rows["baseline_name"] = CONTEXT_MEDIAN_LABEL
        precision_rows["baseline_family"] = "deterministic"
        precision_rows["prompt_type"] = "Source-Only"
        prediction_rows.append(
            precision_rows[
                CORE_COLUMNS
                + [
                    "precision",
                    "expected_ai",
                    "balance_point",
                    "predicted_ai",
                    "baseline_name",
                    "baseline_family",
                    "prompt_type",
                ]
            ]
        )
    return pd.concat(prediction_rows, ignore_index=True)


def _make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0))]),
                numeric_columns,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )


def _make_tree_estimator(kind: str, family: str, random_state: int):
    if family == "rf" and kind == "classifier":
        return RandomForestClassifier(
            n_estimators=120,
            random_state=random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    if family == "rf" and kind == "regressor":
        return RandomForestRegressor(
            n_estimators=150,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    if family == "et" and kind == "classifier":
        return ExtraTreesClassifier(
            n_estimators=160,
            random_state=random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    if family == "et" and kind == "regressor":
        return ExtraTreesRegressor(
            n_estimators=200,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported family/kind combination: {family}/{kind}")


def _predict_positive_class_probability(classifier: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    probabilities = classifier.predict_proba(X_test)
    class_list = classifier.named_steps["model"].classes_
    if len(class_list) == 1:
        return np.full(len(X_test), 1.0 if class_list[0] == 1 else 0.0)
    positive_index = int(np.where(class_list == 1)[0][0])
    return probabilities[:, positive_index]


def _two_stage_grouped_tree_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    family: str,
    random_state: int,
) -> pd.DataFrame:
    categorical_columns = ["runtime", "gpu"]
    group_kfold = GroupKFold(n_splits=5)
    predictions = []

    for precision in PRECISIONS:
        precision_rows = frame[frame["precision"] == precision].copy().reset_index(drop=True)
        X = precision_rows[feature_columns + categorical_columns].copy()
        y = precision_rows["expected_ai"].copy()
        groups = precision_rows["program_name"].copy()
        predicted_ai = np.zeros(len(precision_rows))

        for train_index, test_index in group_kfold.split(X, groups=groups):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]

            zero_classifier = Pipeline(
                [
                    ("preprocessor", _make_preprocessor(feature_columns, categorical_columns)),
                    ("model", _make_tree_estimator("classifier", family, random_state)),
                ]
            )
            zero_classifier.fit(X_train, y_train.eq(0.0).astype(int))
            zero_probabilities = _predict_positive_class_probability(zero_classifier, X_test)
            test_zero_mask = zero_probabilities >= 0.5

            nonzero_train = y_train.gt(0.0)
            if int(nonzero_train.sum()) == 0:
                fold_predictions = np.zeros(len(X_test))
            else:
                ai_regressor = Pipeline(
                    [
                        ("preprocessor", _make_preprocessor(feature_columns, categorical_columns)),
                        ("model", _make_tree_estimator("regressor", family, random_state)),
                    ]
                )
                ai_regressor.fit(X_train.loc[nonzero_train], np.log1p(y_train[nonzero_train]))
                fold_predictions = np.expm1(ai_regressor.predict(X_test))
                fold_predictions = np.clip(fold_predictions, 0.0, None)
            fold_predictions[test_zero_mask] = 0.0
            predicted_ai[test_index] = fold_predictions

        precision_rows["predicted_ai"] = predicted_ai
        predictions.append(
            precision_rows[
                CORE_COLUMNS
                + [
                    "precision",
                    "expected_ai",
                    "balance_point",
                    "predicted_ai",
                ]
            ]
        )

    return pd.concat(predictions, ignore_index=True)


def _metrics_from_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for precision in PRECISIONS:
        precision_rows = frame[frame["precision"] == precision].copy()
        metrics[f"{precision}_task_a"] = _task_a_balanced_accuracy(
            precision_rows["expected_ai"],
            precision_rows["predicted_ai"],
        )
        metrics[f"{precision}_task_c"] = _task_c_balanced_accuracy(
            precision_rows["expected_ai"],
            precision_rows["predicted_ai"],
            precision_rows["balance_point"],
        )
        metrics[f"{precision}_task_d"] = _task_d_balanced_accuracy(
            precision_rows["expected_ai"],
            precision_rows["predicted_ai"],
            precision_rows["balance_point"],
        )
        metrics[f"{precision}_medape"] = _nonzero_median_ape(
            precision_rows["expected_ai"],
            precision_rows["predicted_ai"],
        )
    return metrics


def summarize_baseline_metrics(prediction_rows: pd.DataFrame) -> pd.DataFrame:
    results = []
    ordered_columns = ["baseline_name", "baseline_family", "prompt_type"]
    for key, subset in prediction_rows.groupby(ordered_columns, sort=False):
        baseline_name, baseline_family, prompt_type = key
        row = {
            "baseline_name": baseline_name,
            "baseline_family": baseline_family,
            "prompt_type": prompt_type,
        }
        row.update(_metrics_from_prediction_frame(subset))
        results.append(row)
    return pd.DataFrame(results)


def learned_baseline_predictions(
    shared_precision_rows: pd.DataFrame,
    dataset: dict,
    families: list[tuple[str, str, str]],
    random_state: int,
) -> pd.DataFrame:
    source_features = build_source_feature_rows(dataset)
    sass_features = build_sass_feature_rows(dataset)

    source_eval = shared_precision_rows.merge(
        source_features,
        on=["program_name", "kernel_mangled_name"],
        how="left",
    )
    sass_eval = source_eval.merge(
        sass_features,
        on=["program_name", "kernel_mangled_name", "gpu"],
        how="left",
    )

    source_feature_columns = [column for column in source_eval.columns if column.startswith("src_")]
    sass_feature_columns = [column for column in sass_eval.columns if column.startswith("src_") or column.startswith("sass_")]

    prediction_rows = []
    for family, source_label, sass_label in families:
        source_predictions = _two_stage_grouped_tree_predictions(
            source_eval,
            source_feature_columns,
            family,
            random_state,
        )
        source_predictions["baseline_name"] = source_label
        source_predictions["baseline_family"] = "learned"
        source_predictions["prompt_type"] = "Source-Only"
        prediction_rows.append(source_predictions)

        sass_predictions = _two_stage_grouped_tree_predictions(
            sass_eval,
            sass_feature_columns,
            family,
            random_state,
        )
        sass_predictions["baseline_name"] = sass_label
        sass_predictions["baseline_family"] = "learned"
        sass_predictions["prompt_type"] = "Source+SASS"
        prediction_rows.append(sass_predictions)

    return pd.concat(prediction_rows, ignore_index=True)


def learned_seed_sensitivity(
    shared_precision_rows: pd.DataFrame,
    dataset: dict,
    random_states: list[int],
) -> pd.DataFrame:
    prediction_rows = []
    for random_state in random_states:
        seed_predictions = learned_baseline_predictions(
            shared_precision_rows,
            dataset,
            families=[
                ("rf", SOURCE_LEARNED_LABEL, SASS_LEARNED_LABEL),
                ("et", SOURCE_EXTRA_TREES_LABEL, SASS_EXTRA_TREES_LABEL),
            ],
            random_state=random_state,
        )
        seed_predictions["random_state"] = random_state
        prediction_rows.append(seed_predictions)
    seed_frame = pd.concat(prediction_rows, ignore_index=True)

    results = []
    for key, subset in seed_frame.groupby(["baseline_name", "baseline_family", "prompt_type", "random_state"], sort=False):
        baseline_name, baseline_family, prompt_type, random_state = key
        row = {
            "baseline_name": baseline_name,
            "baseline_family": baseline_family,
            "prompt_type": prompt_type,
            "random_state": random_state,
        }
        row.update(_metrics_from_prediction_frame(subset))
        results.append(row)
    return pd.DataFrame(results)


def summarize_seed_sensitivity(seed_rows: pd.DataFrame) -> pd.DataFrame:
    results = []
    for key, subset in seed_rows.groupby(["baseline_name", "baseline_family", "prompt_type"], sort=False):
        baseline_name, baseline_family, prompt_type = key
        row = {
            "baseline_name": baseline_name,
            "baseline_family": baseline_family,
            "prompt_type": prompt_type,
        }
        for precision in PRECISIONS:
            row[f"{precision}_task_c_mean"] = float(subset[f"{precision}_task_c"].mean())
            row[f"{precision}_task_c_std"] = float(subset[f"{precision}_task_c"].std(ddof=0))
            row[f"{precision}_medape_mean"] = float(subset[f"{precision}_medape"].mean())
            row[f"{precision}_medape_std"] = float(subset[f"{precision}_medape"].std(ddof=0))
        results.append(row)
    return pd.DataFrame(results)


def shared_class_balance(shared_precision_rows: pd.DataFrame) -> pd.DataFrame:
    rows = shared_precision_rows.copy()
    rows["truth_label"] = np.where(
        rows["expected_ai"].eq(0.0),
        "zero",
        np.where(rows["expected_ai"] >= rows["balance_point"], "cb", "bb"),
    )
    results = []
    for precision in PRECISIONS:
        subset = rows[rows["precision"] == precision].copy()
        counts = subset["truth_label"].value_counts()
        total_rows = int(len(subset))
        zero_rows = int(counts.get("zero", 0))
        bb_rows = int(counts.get("bb", 0))
        cb_rows = int(counts.get("cb", 0))
        results.append(
            {
                "precision": precision,
                "total_rows": total_rows,
                "zero_rows": zero_rows,
                "bb_rows": bb_rows,
                "cb_rows": cb_rows,
                "nonzero_rows": bb_rows + cb_rows,
                "zero_fraction": float(zero_rows) / total_rows if total_rows else float("nan"),
                "cb_fraction_among_nonzero": (
                    float(cb_rows) / (bb_rows + cb_rows) if (bb_rows + cb_rows) else float("nan")
                ),
            }
        )
    return pd.DataFrame(results)


def _precision_recall_f1(
    truth_positive: pd.Series,
    predicted_positive: pd.Series,
) -> dict[str, float]:
    truth = truth_positive.astype(bool)
    pred = predicted_positive.astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = float(tp) / (tp + fp) if (tp + fp) else float("nan")
    recall = float(tp) / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
        else float("nan")
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_bound_prf(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result_rows = []
    for group_key, group_frame in rows.groupby(group_columns, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        result = {column_name: value for column_name, value in zip(group_columns, group_key)}
        for precision in PRECISIONS:
            subset = group_frame[(group_frame["precision"] == precision) & (group_frame["expected_ai"] > 0.0)].copy()
            truth_positive = subset["expected_ai"] >= subset["balance_point"]
            predicted_positive = subset["predicted_ai"].gt(0.0) & subset["predicted_ai"].ge(subset["balance_point"])
            stats = _precision_recall_f1(truth_positive, predicted_positive)
            result[f"{precision}_support_nonzero"] = int(len(subset))
            result[f"{precision}_support_cb"] = int(truth_positive.sum())
            result[f"{precision}_tp"] = stats["tp"]
            result[f"{precision}_fp"] = stats["fp"]
            result[f"{precision}_fn"] = stats["fn"]
            result[f"{precision}_precision"] = stats["precision"]
            result[f"{precision}_recall"] = stats["recall"]
            result[f"{precision}_f1"] = stats["f1"]
        result_rows.append(result)
    return pd.DataFrame(result_rows)


def task_confusion_counts(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result_rows = []
    for group_key, group_frame in rows.groupby(group_columns, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_identity = {column_name: value for column_name, value in zip(group_columns, group_key)}
        for precision in PRECISIONS:
            subset = group_frame[group_frame["precision"] == precision].copy()

            task_a_truth = np.where(subset["expected_ai"].eq(0.0), "zero", "nonzero")
            task_a_pred = np.where(subset["predicted_ai"].eq(0.0), "zero", "nonzero")
            task_a_counts = (
                pd.DataFrame({"truth_label": task_a_truth, "pred_label": task_a_pred})
                .value_counts()
                .reset_index(name="count")
            )
            for _, row in task_a_counts.iterrows():
                result_rows.append(
                    {
                        **group_identity,
                        "precision": precision,
                        "task": "task_a_zero_nonzero",
                        "truth_label": row["truth_label"],
                        "pred_label": row["pred_label"],
                        "count": int(row["count"]),
                    }
                )

            nonzero_subset = subset[subset["expected_ai"] > 0.0].copy()
            task_c_truth, task_c_pred = _nonzero_bound_labels(
                nonzero_subset["expected_ai"],
                nonzero_subset["predicted_ai"],
                nonzero_subset["balance_point"],
            )
            task_c_counts = (
                pd.DataFrame({"truth_label": task_c_truth, "pred_label": task_c_pred})
                .value_counts()
                .reset_index(name="count")
            )
            for _, row in task_c_counts.iterrows():
                result_rows.append(
                    {
                        **group_identity,
                        "precision": precision,
                        "task": "task_c_nonzero_bound",
                        "truth_label": row["truth_label"],
                        "pred_label": row["pred_label"],
                        "count": int(row["count"]),
                    }
                )

            task_d_truth = np.where(
                subset["expected_ai"].eq(0.0),
                "zero",
                np.where(subset["expected_ai"] >= subset["balance_point"], "cb", "bb"),
            )
            task_d_pred = np.where(
                subset["predicted_ai"].fillna(0.0) <= 0.0,
                "zero",
                np.where(subset["predicted_ai"] >= subset["balance_point"], "cb", "bb"),
            )
            task_d_counts = (
                pd.DataFrame({"truth_label": task_d_truth, "pred_label": task_d_pred})
                .value_counts()
                .reset_index(name="count")
            )
            for _, row in task_d_counts.iterrows():
                result_rows.append(
                    {
                        **group_identity,
                        "precision": precision,
                        "task": "task_d_three_way",
                        "truth_label": row["truth_label"],
                        "pred_label": row["pred_label"],
                        "count": int(row["count"]),
                    }
                )
    return pd.DataFrame(result_rows)


def _task_c_correctness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    nonzero_rows = frame[frame["expected_ai"] > 0.0].copy()
    truth, pred = _nonzero_bound_labels(
        nonzero_rows["expected_ai"],
        nonzero_rows["predicted_ai"],
        nonzero_rows["balance_point"],
    )
    result = nonzero_rows[CORE_COLUMNS + ["precision", "expected_ai", "balance_point", "predicted_ai"]].copy()
    result["truth_label"] = truth
    result["pred_label"] = pred
    result["is_correct"] = result["truth_label"] == result["pred_label"]
    return result


def best_llm_vs_static_task_c_comparison(
    shared_rows: pd.DataFrame,
    static_prediction_rows: pd.DataFrame,
) -> pd.DataFrame:
    llm_summary_rows = []
    for (model_name, prompt_type), subset in shared_rows.groupby(["model_name", "prompt_type"], sort=False):
        for precision in PRECISIONS:
            precision_rows = subset[subset["precision"] == precision].copy()
            llm_summary_rows.append(
                {
                    "name": model_name,
                    "group_type": "llm",
                    "prompt_type": prompt_type,
                    "precision": precision,
                    "task_c": _task_c_balanced_accuracy(
                        precision_rows["expected_ai"],
                        precision_rows["predicted_ai"],
                        precision_rows["balance_point"],
                    ),
                    "medape": _nonzero_median_ape(
                        precision_rows["expected_ai"],
                        precision_rows["predicted_ai"],
                    ),
                }
            )
    llm_summary = pd.DataFrame(llm_summary_rows)

    static_summary_rows = []
    for (baseline_name, prompt_type), subset in static_prediction_rows.groupby(["baseline_name", "prompt_type"], sort=False):
        for precision in PRECISIONS:
            precision_rows = subset[subset["precision"] == precision].copy()
            static_summary_rows.append(
                {
                    "name": baseline_name,
                    "group_type": "static",
                    "prompt_type": prompt_type,
                    "precision": precision,
                    "task_c": _task_c_balanced_accuracy(
                        precision_rows["expected_ai"],
                        precision_rows["predicted_ai"],
                        precision_rows["balance_point"],
                    ),
                    "medape": _nonzero_median_ape(
                        precision_rows["expected_ai"],
                        precision_rows["predicted_ai"],
                    ),
                }
            )
    static_summary = pd.DataFrame(static_summary_rows)

    result_rows = []
    for prompt_type in PROMPTS:
        for precision in PRECISIONS:
            best_llm = (
                llm_summary[
                    (llm_summary["prompt_type"] == prompt_type)
                    & (llm_summary["precision"] == precision)
                ]
                .sort_values(["task_c", "name"], ascending=[False, True])
                .iloc[0]
            )
            best_static = (
                static_summary[
                    (static_summary["prompt_type"] == prompt_type)
                    & (static_summary["precision"] == precision)
                ]
                .sort_values(["task_c", "name"], ascending=[False, True])
                .iloc[0]
            )

            llm_rows = shared_rows[
                (shared_rows["model_name"] == best_llm["name"])
                & (shared_rows["prompt_type"] == prompt_type)
                & (shared_rows["precision"] == precision)
            ].copy()
            llm_correctness = _task_c_correctness_frame(llm_rows)[
                CORE_COLUMNS + ["precision", "is_correct"]
            ].rename(columns={"is_correct": "llm_correct"})

            static_rows = static_prediction_rows[
                (static_prediction_rows["baseline_name"] == best_static["name"])
                & (static_prediction_rows["prompt_type"] == prompt_type)
                & (static_prediction_rows["precision"] == precision)
            ].copy()
            static_correctness = _task_c_correctness_frame(static_rows)[
                CORE_COLUMNS + ["precision", "is_correct"]
            ].rename(columns={"is_correct": "static_correct"})

            paired = llm_correctness.merge(
                static_correctness,
                on=CORE_COLUMNS + ["precision"],
                how="inner",
            )
            llm_wins = int((paired["llm_correct"] & ~paired["static_correct"]).sum())
            static_wins = int((paired["static_correct"] & ~paired["llm_correct"]).sum())
            discordant = llm_wins + static_wins
            pvalue = (
                float(binomtest(llm_wins, discordant, 0.5).pvalue)
                if discordant
                else float("nan")
            )

            result_rows.append(
                {
                    "prompt_type": prompt_type,
                    "precision": precision,
                    "best_llm": best_llm["name"],
                    "best_llm_task_c": float(best_llm["task_c"]),
                    "best_llm_medape": float(best_llm["medape"]),
                    "best_static": best_static["name"],
                    "best_static_task_c": float(best_static["task_c"]),
                    "best_static_medape": float(best_static["medape"]),
                    "llm_minus_static_task_c": float(best_llm["task_c"] - best_static["task_c"]),
                    "llm_task_c_discordant_wins": llm_wins,
                    "static_task_c_discordant_wins": static_wins,
                    "task_c_exact_pvalue": pvalue,
                }
            )
    return pd.DataFrame(result_rows)


def budgeted_compute_recall(
    rows: pd.DataFrame,
    group_columns: list[str],
    budget_fractions: list[float],
) -> pd.DataFrame:
    rows = rows.copy()
    rows["predicted_score"] = rows["predicted_ai"] / rows["balance_point"]
    rows["predicted_score"] = rows["predicted_score"].replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    rows["is_compute_bound"] = rows["expected_ai"].gt(0.0) & rows["expected_ai"].ge(rows["balance_point"])

    result_rows = []
    for group_key, group_frame in rows.groupby(group_columns, sort=False):
        result = {}
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for column_name, value in zip(group_columns, group_key):
            result[column_name] = value
        for precision in PRECISIONS:
            subset = group_frame[group_frame["precision"] == precision].copy()
            subset = subset.sort_values("predicted_score", ascending=False)
            total_compute_bound = int(subset["is_compute_bound"].sum())
            result[f"{precision}_compute_bound_total"] = total_compute_bound
            for budget_fraction in budget_fractions:
                budget_label = int(round(budget_fraction * 100))
                budget_rows = max(1, int(math.ceil(len(subset) * budget_fraction)))
                selected = subset.head(budget_rows)
                result[f"{precision}_budget_rows_at_{budget_label}"] = budget_rows
                result[f"{precision}_recall_at_{budget_label}"] = (
                    float(selected["is_compute_bound"].sum()) / total_compute_bound
                    if total_compute_bound
                    else float("nan")
                )
                result[f"{precision}_precision_at_{budget_label}"] = float(selected["is_compute_bound"].mean())
                base_rate = float(subset["is_compute_bound"].mean())
                result[f"{precision}_precision_lift_at_{budget_label}"] = (
                    result[f"{precision}_precision_at_{budget_label}"] / base_rate if base_rate else float("nan")
                )
        result_rows.append(result)
    return pd.DataFrame(result_rows)


def _bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        sample = values[rng.integers(0, len(values), size=len(values))]
        medians[index] = np.median(sample)
    ci_low, ci_high = np.percentile(medians, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def paired_source_vs_sass_stats(shared_rows: pd.DataFrame) -> pd.DataFrame:
    nonzero_rows = shared_rows[shared_rows["expected_ai"] > 0.0].copy()
    results = []
    seed_counter = 0
    for model_name in MODELS:
        for precision in PRECISIONS:
            subset = nonzero_rows[
                (nonzero_rows["model_name"] == model_name)
                & (nonzero_rows["precision"] == precision)
            ].copy()
            source_rows = subset[subset["prompt_type"] == "Source-Only"][
                CORE_COLUMNS + ["ai_abs_percent_error", "expected_ai", "predicted_ai", "balance_point"]
            ].rename(columns={"ai_abs_percent_error": "source_medape_row", "predicted_ai": "source_predicted_ai"})
            sass_rows = subset[subset["prompt_type"] == "Source+SASS"][
                CORE_COLUMNS + ["ai_abs_percent_error", "expected_ai", "predicted_ai", "balance_point"]
            ].rename(columns={"ai_abs_percent_error": "sass_medape_row", "predicted_ai": "sass_predicted_ai"})
            paired = source_rows.merge(
                sass_rows,
                on=CORE_COLUMNS + ["expected_ai", "balance_point"],
                how="inner",
            ).dropna(subset=["source_medape_row", "sass_medape_row"])

            paired_deltas = (paired["source_medape_row"] - paired["sass_medape_row"]).to_numpy()
            improved_count = int((paired_deltas > 0.0).sum())
            worsened_count = int((paired_deltas < 0.0).sum())
            tied_count = int((paired_deltas == 0.0).sum())
            non_tie_total = improved_count + worsened_count
            sign_p = (
                float(binomtest(improved_count, non_tie_total, 0.5, alternative="greater").pvalue)
                if non_tie_total
                else float("nan")
            )
            median_delta = float(np.median(paired_deltas)) if len(paired_deltas) else float("nan")
            ci_low, ci_high = _bootstrap_median_ci(paired_deltas, seed=seed_counter)
            seed_counter += 1

            truth_labels = np.where(paired["expected_ai"] >= paired["balance_point"], "cb", "bb")
            source_labels = np.where(
                paired["source_predicted_ai"] <= 0.0,
                "bb",
                np.where(paired["source_predicted_ai"] >= paired["balance_point"], "cb", "bb"),
            )
            sass_labels = np.where(
                paired["sass_predicted_ai"] <= 0.0,
                "bb",
                np.where(paired["sass_predicted_ai"] >= paired["balance_point"], "cb", "bb"),
            )
            source_correct = source_labels == truth_labels
            sass_correct = sass_labels == truth_labels
            source_to_sass_wins = int((~source_correct & sass_correct).sum())
            sass_to_source_wins = int((source_correct & ~sass_correct).sum())
            mcnemar_p = (
                float(
                    binomtest(
                        source_to_sass_wins,
                        source_to_sass_wins + sass_to_source_wins,
                        0.5,
                        alternative="greater",
                    ).pvalue
                )
                if (source_to_sass_wins + sass_to_source_wins)
                else float("nan")
            )

            source_medape = float(
                subset[subset["prompt_type"] == "Source-Only"]["ai_abs_percent_error"].median()
            )
            sass_medape = float(
                subset[subset["prompt_type"] == "Source+SASS"]["ai_abs_percent_error"].median()
            )
            results.append(
                {
                    "model_name": model_name,
                    "precision": precision,
                    "source_medape": source_medape,
                    "sass_medape": sass_medape,
                    "paired_rows": int(len(paired)),
                    "median_delta": median_delta,
                    "median_delta_ci_low": ci_low,
                    "median_delta_ci_high": ci_high,
                    "improved_count": improved_count,
                    "worsened_count": worsened_count,
                    "tied_count": tied_count,
                    "improved_non_tie_fraction": (
                        float(improved_count) / non_tie_total if non_tie_total else float("nan")
                    ),
                    "sign_test_pvalue": sign_p,
                    "source_to_sass_task_c_wins": source_to_sass_wins,
                    "sass_to_source_task_c_wins": sass_to_source_wins,
                    "mcnemar_pvalue": mcnemar_p,
                }
            )
    return pd.DataFrame(results)


def _format_decimal(value: float, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{decimals}f}"


def _format_one_decimal(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.1f}"


def _format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 1e-3:
        return "<0.001"
    return f"{value:.3f}"


def _tex_model_name(model_name: str) -> str:
    return {
        "GPT 5.4": r"\gptfivefour",
        "GPT OSS": r"\gptoss",
        "Opus 4.6": r"\opus",
    }[model_name]


def _tex_prompt_name(prompt_type: str) -> str:
    return {
        "Source-Only": r"\sourceonly",
        "Source+SASS": r"\sourcesass",
    }[prompt_type]


def write_budget_table(budget_rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Compute-bound recall when profiling only the top 10\% of samples ranked by predicted \rai relative to the GPU balance point on the 732-sample shared subset. Each precision budgets 74 of 732 samples. Random ranking would recover 10\% by construction.}",
        r"  \label{tab:budget-recall-at-10}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Model & Evidence & FP16 R@10\% & FP32 R@10\% & FP64 R@10\% \\",
        r"    \midrule",
    ]
    for _, row in budget_rows.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    _tex_model_name(row["model_name"]),
                    _tex_prompt_name(row["prompt_type"]),
                    _format_decimal(row["fp16_recall_at_10"]),
                    _format_decimal(row["fp32_recall_at_10"]),
                    _format_decimal(row["fp64_recall_at_10"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    (TABLES_DIR / "budget_recall_at_10.tex").write_text("\n".join(lines) + "\n")


def write_static_baselines_table(baseline_rows: pd.DataFrame) -> None:
    sort_order = {
        CONTEXT_MEDIAN_LABEL: 0,
        SOURCE_HEURISTIC_LABEL: 1,
        SASS_HEURISTIC_LABEL: 2,
        SOURCE_LEARNED_LABEL: 3,
        SOURCE_EXTRA_TREES_LABEL: 4,
        SASS_LEARNED_LABEL: 5,
        SASS_EXTRA_TREES_LABEL: 6,
    }
    baseline_rows = baseline_rows.sort_values(
        by="baseline_name",
        key=lambda column: column.map(sort_order),
    ).reset_index(drop=True)
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Static reference baselines on the 732-sample shared subset. Context median is a grouped cross-validated runtime$\times$GPU baseline; lexical and mnemonic are deterministic heuristics; the learned baselines are five-fold grouped tree baselines grouped by program. For each precision, the slash-separated triple reports Task~A zero/nonzero balanced accuracy, Task~C nonzero \bandwidthbound/\computebound balanced accuracy, and Task~D three-way balanced accuracy. ``MedAPE'' is the median nonzero absolute percent error.}",
        r"  \label{tab:static-baselines}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Baseline & FP16 A/C/D & FP32 A/C/D & FP64 A/C/D & FP16 MedAPE & FP32 MedAPE & FP64 MedAPE \\",
        r"    \midrule",
    ]
    for _, row in baseline_rows.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    row["baseline_name"],
                    " / ".join(
                        [
                            _format_decimal(row["fp16_task_a"]),
                            _format_decimal(row["fp16_task_c"]),
                            _format_decimal(row["fp16_task_d"]),
                        ]
                    ),
                    " / ".join(
                        [
                            _format_decimal(row["fp32_task_a"]),
                            _format_decimal(row["fp32_task_c"]),
                            _format_decimal(row["fp32_task_d"]),
                        ]
                    ),
                    " / ".join(
                        [
                            _format_decimal(row["fp64_task_a"]),
                            _format_decimal(row["fp64_task_c"]),
                            _format_decimal(row["fp64_task_d"]),
                        ]
                    ),
                    _format_one_decimal(row["fp16_medape"]),
                    _format_one_decimal(row["fp32_medape"]),
                    _format_one_decimal(row["fp64_medape"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    (TABLES_DIR / "deterministic_baselines.tex").write_text("\n".join(lines) + "\n")


def select_budget_recall_rows(
    llm_budget_rows: pd.DataFrame,
    static_budget_rows: pd.DataFrame,
) -> pd.DataFrame:
    llm_eval = llm_budget_rows.copy()
    static_eval = static_budget_rows.copy()
    llm_eval["mean_recall"] = llm_eval[[f"{precision}_recall_at_{int(budget * 100)}" for precision in PRECISIONS for budget in BUDGET_FRACTIONS]].mean(axis=1)
    static_eval["mean_recall"] = static_eval[[f"{precision}_recall_at_{int(budget * 100)}" for precision in PRECISIONS for budget in BUDGET_FRACTIONS]].mean(axis=1)

    selected_rows = []
    for prompt_type in PROMPTS:
        best_llm = (
            llm_eval[llm_eval["prompt_type"] == prompt_type]
            .sort_values(["mean_recall", "model_name"], ascending=[False, True])
            .iloc[0]
        )
        selected_rows.append(
            {
                "row_label": f"{prompt_type} best LLM ({best_llm['model_name']})",
                "row_type": "llm",
                "prompt_type": prompt_type,
                **best_llm.to_dict(),
            }
        )
        best_static = (
            static_eval[static_eval["prompt_type"] == prompt_type]
            .sort_values(["mean_recall", "baseline_name"], ascending=[False, True])
            .iloc[0]
        )
        selected_rows.append(
            {
                "row_label": f"{prompt_type} best static ({best_static['baseline_name']})",
                "row_type": "static",
                "prompt_type": prompt_type,
                **best_static.to_dict(),
            }
        )
    return pd.DataFrame(selected_rows)


def write_budget_sweep_table(selected_rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Budget-sweep compute-bound recall for the strongest source-only and source+SASS methods in each family, selected by mean recall across the 5/10/20/30\% sweep on the shared subset. This shows whether the ranking behind Table~\ref{tab:budget-recall-at-10} is stable beyond the single 10\% operating point.}",
        r"  \label{tab:budget-recall-sweep}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Precision & Method & R@5\% & R@10\% & R@20\% & R@30\% \\",
        r"    \midrule",
    ]
    for precision in PRECISIONS:
        precision_label = precision.upper()
        for _, row in selected_rows.iterrows():
            method_label = row["row_label"].replace("Source-Only", r"\sourceonly").replace("Source+SASS", r"\sourcesass")
            lines.append(
                "    "
                + " & ".join(
                    [
                        precision_label,
                        method_label,
                        _format_decimal(row[f"{precision}_recall_at_5"]),
                        _format_decimal(row[f"{precision}_recall_at_10"]),
                        _format_decimal(row[f"{precision}_recall_at_20"]),
                        _format_decimal(row[f"{precision}_recall_at_30"]),
                    ]
                )
                + r" \\"
            )
        if precision != PRECISIONS[-1]:
            lines.append(r"    \midrule")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    (TABLES_DIR / "budget_recall_sweep.tex").write_text("\n".join(lines) + "\n")


def write_seed_sensitivity_table(seed_summary_rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Seed sensitivity for the learned static baselines across five grouped-CV seeds. Each entry reports mean$\pm$std over seeds for Task~C balanced accuracy and nonzero MedAPE. The low spread shows that the static-baseline comparisons are not driven by an unusually favorable random seed.}",
        r"  \label{tab:seed-sensitivity}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Baseline & FP16 C / MedAPE & FP32 C / MedAPE & FP64 C / MedAPE & FP16 std(C) & FP32 std(C) & FP64 std(C) \\",
        r"    \midrule",
    ]
    sort_order = {
        SOURCE_LEARNED_LABEL: 0,
        SOURCE_EXTRA_TREES_LABEL: 1,
        SASS_LEARNED_LABEL: 2,
        SASS_EXTRA_TREES_LABEL: 3,
    }
    seed_summary_rows = seed_summary_rows.sort_values(
        by="baseline_name",
        key=lambda column: column.map(sort_order),
    ).reset_index(drop=True)
    for _, row in seed_summary_rows.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    row["baseline_name"],
                    f"{_format_decimal(row['fp16_task_c_mean'])} / {_format_one_decimal(row['fp16_medape_mean'])}",
                    f"{_format_decimal(row['fp32_task_c_mean'])} / {_format_one_decimal(row['fp32_medape_mean'])}",
                    f"{_format_decimal(row['fp64_task_c_mean'])} / {_format_one_decimal(row['fp64_medape_mean'])}",
                    _format_decimal(row["fp16_task_c_std"]),
                    _format_decimal(row["fp32_task_c_std"]),
                    _format_decimal(row["fp64_task_c_std"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    (TABLES_DIR / "learned_seed_sensitivity.tex").write_text("\n".join(lines) + "\n")


def write_shared_class_balance_table(class_balance_rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Ground-truth class balance on the 732-sample shared subset used for all paired cross-model comparisons. The imbalance is precision-dependent: FP16/FP32 have more nonzero samples than FP64, but all three precisions remain zero-heavy, which is why Tasks~A/C/D are reported separately.}",
        r"  \label{tab:shared-class-balance}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Precision & Zero & Nonzero BB & Nonzero CB & Zero share \\",
        r"    \midrule",
    ]
    for _, row in class_balance_rows.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    row["precision"].upper(),
                    str(int(row["zero_rows"])),
                    str(int(row["bb_rows"])),
                    str(int(row["cb_rows"])),
                    _format_decimal(row["zero_fraction"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    (TABLES_DIR / "shared_class_balance.tex").write_text("\n".join(lines) + "\n")


def write_compute_bound_prf_table(llm_prf_rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Nonzero compute-bound precision/recall/F1 on the 732-sample shared subset. These metrics complement Task~C balanced accuracy by exposing whether a method reaches high recall by over-predicting the compute-bound side.}",
        r"  \label{tab:compute-bound-prf}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Model & Evidence & FP16 P/R/F1 & FP32 P/R/F1 & FP64 P/R/F1 \\",
        r"    \midrule",
    ]
    model_order = {model_name: index for index, model_name in enumerate(MODELS)}
    prompt_order = {"Source-Only": 0, "Source+SASS": 1}
    llm_prf_rows = llm_prf_rows.sort_values(
        by=["model_name", "prompt_type"],
        key=lambda column: column.map(model_order if column.name == "model_name" else prompt_order),
    ).reset_index(drop=True)
    for _, row in llm_prf_rows.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    _tex_model_name(row["model_name"]),
                    _tex_prompt_name(row["prompt_type"]),
                    f"{_format_decimal(row['fp16_precision'])} / {_format_decimal(row['fp16_recall'])} / {_format_decimal(row['fp16_f1'])}",
                    f"{_format_decimal(row['fp32_precision'])} / {_format_decimal(row['fp32_recall'])} / {_format_decimal(row['fp32_f1'])}",
                    f"{_format_decimal(row['fp64_precision'])} / {_format_decimal(row['fp64_recall'])} / {_format_decimal(row['fp64_f1'])}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    (TABLES_DIR / "compute_bound_prf.tex").write_text("\n".join(lines) + "\n")


def write_llm_vs_static_task_c_table(comparison_rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Best-off-the-shelf LLM versus best static reference for Task~C nonzero \bandwidthbound/\computebound triage on the shared subset. Each cell reports balanced accuracy and nonzero MedAPE for the strongest method in that evidence family, selected separately within each prompt/precision regime. The discordant-pair columns report exact paired wins for Task~C correctness on the same samples.}",
        r"  \label{tab:llm-vs-static-task-c}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lllrllr}",
        r"    \toprule",
        r"    Evidence & Precision & Best LLM & Task~C / MedAPE & Best static & Task~C / MedAPE & Discordant wins LLM/static ($p$) \\",
        r"    \midrule",
    ]
    prompt_order = {"Source-Only": 0, "Source+SASS": 1}
    precision_order = {precision: index for index, precision in enumerate(PRECISIONS)}
    comparison_rows = comparison_rows.sort_values(
        by=["prompt_type", "precision"],
        key=lambda column: column.map(prompt_order if column.name == "prompt_type" else precision_order),
    ).reset_index(drop=True)
    for _, row in comparison_rows.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    _tex_prompt_name(row["prompt_type"]),
                    row["precision"].upper(),
                    _tex_model_name(row["best_llm"]),
                    f"{_format_decimal(row['best_llm_task_c'])} / {_format_one_decimal(row['best_llm_medape'])}",
                    row["best_static"],
                    f"{_format_decimal(row['best_static_task_c'])} / {_format_one_decimal(row['best_static_medape'])}",
                    f"{int(row['llm_task_c_discordant_wins'])}/{int(row['static_task_c_discordant_wins'])} ({_format_pvalue(row['task_c_exact_pvalue'])})",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    (TABLES_DIR / "llm_vs_static_task_c.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    dataset = load_dataset()
    prediction_rows = load_prediction_rows()
    shared = shared_subset(prediction_rows)
    shared_precision_rows = shared.drop_duplicates(CORE_COLUMNS + ["precision"]).copy()

    deterministic_predictions = deterministic_baseline_predictions(shared_precision_rows, dataset)
    context_predictions = context_median_predictions(shared_precision_rows)
    deterministic_rows = summarize_baseline_metrics(
        pd.concat([context_predictions, deterministic_predictions], ignore_index=True)
    )
    learned_predictions = learned_baseline_predictions(
        shared_precision_rows,
        dataset,
        families=[
            ("rf", SOURCE_LEARNED_LABEL, SASS_LEARNED_LABEL),
            ("et", SOURCE_EXTRA_TREES_LABEL, SASS_EXTRA_TREES_LABEL),
        ],
        random_state=0,
    )
    learned_rows = summarize_baseline_metrics(learned_predictions)
    static_baseline_rows = pd.concat([deterministic_rows, learned_rows], ignore_index=True)
    llm_budget_rows = budgeted_compute_recall(shared, ["model_name", "prompt_type"], BUDGET_FRACTIONS)
    static_budget_rows = budgeted_compute_recall(
        pd.concat([context_predictions, deterministic_predictions, learned_predictions], ignore_index=True),
        ["baseline_name", "prompt_type"],
        BUDGET_FRACTIONS,
    )
    selected_budget_rows = select_budget_recall_rows(llm_budget_rows, static_budget_rows)
    seed_rows = learned_seed_sensitivity(shared_precision_rows, dataset, LEARNED_BASELINE_SEEDS)
    seed_summary_rows = summarize_seed_sensitivity(seed_rows)
    paired_rows = paired_source_vs_sass_stats(shared)
    class_balance_rows = shared_class_balance(shared_precision_rows)
    llm_prf_rows = compute_bound_prf(shared, ["model_name", "prompt_type"])
    static_prf_rows = compute_bound_prf(
        pd.concat([context_predictions, deterministic_predictions, learned_predictions], ignore_index=True),
        ["baseline_name", "prompt_type"],
    )
    llm_confusion_rows = task_confusion_counts(shared, ["model_name", "prompt_type"])
    static_confusion_rows = task_confusion_counts(
        pd.concat([context_predictions, deterministic_predictions, learned_predictions], ignore_index=True),
        ["baseline_name", "prompt_type"],
    )
    llm_vs_static_rows = best_llm_vs_static_task_c_comparison(
        shared,
        pd.concat([context_predictions, deterministic_predictions, learned_predictions], ignore_index=True),
    )

    deterministic_rows.to_csv(GENERATED_DIR / "deterministic_baselines.csv", index=False)
    learned_rows.to_csv(GENERATED_DIR / "learned_baselines.csv", index=False)
    static_baseline_rows.to_csv(GENERATED_DIR / "static_baselines.csv", index=False)
    pd.concat([context_predictions, deterministic_predictions, learned_predictions], ignore_index=True).to_csv(
        GENERATED_DIR / "static_prediction_rows.csv",
        index=False,
    )
    llm_budget_rows.to_csv(GENERATED_DIR / "budget_recall_at_10.csv", index=False)
    static_budget_rows.to_csv(GENERATED_DIR / "static_budget_recall_sweep.csv", index=False)
    selected_budget_rows.to_csv(GENERATED_DIR / "budget_recall_sweep_selected.csv", index=False)
    seed_rows.to_csv(GENERATED_DIR / "learned_baseline_seed_sensitivity.csv", index=False)
    seed_summary_rows.to_csv(GENERATED_DIR / "learned_baseline_seed_summary.csv", index=False)
    paired_rows.to_csv(GENERATED_DIR / "paired_source_vs_sass_stats.csv", index=False)
    class_balance_rows.to_csv(GENERATED_DIR / "shared_class_balance.csv", index=False)
    llm_prf_rows.to_csv(GENERATED_DIR / "llm_compute_bound_prf.csv", index=False)
    static_prf_rows.to_csv(GENERATED_DIR / "static_compute_bound_prf.csv", index=False)
    llm_confusion_rows.to_csv(GENERATED_DIR / "llm_task_confusion_counts.csv", index=False)
    static_confusion_rows.to_csv(GENERATED_DIR / "static_task_confusion_counts.csv", index=False)
    llm_vs_static_rows.to_csv(GENERATED_DIR / "llm_vs_static_task_c.csv", index=False)

    write_static_baselines_table(static_baseline_rows)
    write_budget_table(llm_budget_rows)
    write_budget_sweep_table(selected_budget_rows)
    write_seed_sensitivity_table(seed_summary_rows)
    write_shared_class_balance_table(class_balance_rows)
    write_compute_bound_prf_table(llm_prf_rows)
    write_llm_vs_static_task_c_table(llm_vs_static_rows)

    print("Wrote:")
    for path in [
        GENERATED_DIR / "deterministic_baselines.csv",
        GENERATED_DIR / "learned_baselines.csv",
        GENERATED_DIR / "static_baselines.csv",
        GENERATED_DIR / "static_prediction_rows.csv",
        GENERATED_DIR / "budget_recall_at_10.csv",
        GENERATED_DIR / "static_budget_recall_sweep.csv",
        GENERATED_DIR / "budget_recall_sweep_selected.csv",
        GENERATED_DIR / "learned_baseline_seed_sensitivity.csv",
        GENERATED_DIR / "learned_baseline_seed_summary.csv",
        GENERATED_DIR / "paired_source_vs_sass_stats.csv",
        GENERATED_DIR / "shared_class_balance.csv",
        GENERATED_DIR / "llm_compute_bound_prf.csv",
        GENERATED_DIR / "static_compute_bound_prf.csv",
        GENERATED_DIR / "llm_task_confusion_counts.csv",
        GENERATED_DIR / "static_task_confusion_counts.csv",
        GENERATED_DIR / "llm_vs_static_task_c.csv",
        TABLES_DIR / "deterministic_baselines.tex",
        TABLES_DIR / "budget_recall_at_10.tex",
        TABLES_DIR / "budget_recall_sweep.tex",
        TABLES_DIR / "learned_seed_sensitivity.tex",
        TABLES_DIR / "shared_class_balance.tex",
        TABLES_DIR / "compute_bound_prf.tex",
        TABLES_DIR / "llm_vs_static_task_c.tex",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
