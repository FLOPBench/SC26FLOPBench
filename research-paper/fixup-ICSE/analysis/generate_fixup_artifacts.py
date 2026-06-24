#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SITE_DATA_CANDIDATES = [
    REPO_ROOT / "research-paper" / "icse-updated-paper" / "icse_student_handoff_2026-06-11" / "original_inputs" / "_website_mirror" / "site-data.json",
]
GENERATED_DIR = ROOT / "generated"
TABLES_DIR = ROOT / "tables"

CORE_COLUMNS = ["program_name", "runtime", "kernel_mangled_name", "gpu"]
SHARED_REQUIREMENTS = [
    ("GPT 5.4", "Source-Only"),
    ("GPT 5.4", "Source+SASS"),
    ("GPT OSS", "Source-Only"),
    ("GPT OSS", "Source+SASS"),
    ("Opus 4.6", "Source-Only"),
    ("Opus 4.6", "Source+SASS"),
]
MODEL_MACROS = {
    "GPT 5.4": r"\gptfivefour",
    "GPT OSS": r"\gptoss",
    "Opus 4.6": r"\opus",
}
PROMPT_MACROS = {
    "Source-Only": r"\sourceonly",
    "Source+SASS": r"\sourcesass",
}
PRECISION_LABELS = {"fp16": "FP16", "fp32": "FP32", "fp64": "FP64"}


def load_rows() -> pd.DataFrame:
    site_data_path = next((path for path in SITE_DATA_CANDIDATES if path.exists()), None)
    if site_data_path is None:
        raise FileNotFoundError(f"Could not find site-data.json in: {SITE_DATA_CANDIDATES}")
    rows = pd.DataFrame(json.loads(site_data_path.read_text())["llmIndex"]["predictionRows"])
    for column in [
        "expected_ai",
        "predicted_ai",
        "ai_abs_percent_error",
        "balance_point",
        "cost_usd",
        "query_time",
    ]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def shared_subset(rows: pd.DataFrame) -> pd.DataFrame:
    availability = rows.groupby(CORE_COLUMNS + ["model_name", "prompt_type"]).size().reset_index(name="n")
    pivot = availability.pivot_table(
        index=CORE_COLUMNS,
        columns=["model_name", "prompt_type"],
        values="n",
        fill_value=0,
    )
    shared_keys = pivot[(pivot[SHARED_REQUIREMENTS] > 0).all(axis=1)].index.to_frame(index=False)
    return rows.merge(shared_keys, on=CORE_COLUMNS, how="inner")


def generate_threshold_success(shared: pd.DataFrame) -> None:
    nonzero = shared[shared["expected_ai"] > 0].copy()
    summary_rows = []
    for model in ["GPT 5.4", "GPT OSS", "Opus 4.6"]:
        for prompt in ["Source-Only", "Source+SASS"]:
            for precision in ["fp16", "fp32", "fp64"]:
                subset = nonzero[
                    (nonzero["model_name"] == model)
                    & (nonzero["prompt_type"] == prompt)
                    & (nonzero["precision"] == precision)
                ]
                summary_rows.append(
                    {
                        "model": model,
                        "evidence": prompt,
                        "precision": precision,
                        "nonzero_rows": len(subset),
                        "within_10": 100.0 * (subset["ai_abs_percent_error"] <= 10.0).mean(),
                        "within_25": 100.0 * (subset["ai_abs_percent_error"] <= 25.0).mean(),
                        "within_50": 100.0 * (subset["ai_abs_percent_error"] <= 50.0).mean(),
                        "medape": subset["ai_abs_percent_error"].median(),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(GENERATED_DIR / "rai_threshold_success.csv", index=False)

    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    MODEL_MACROS[row["model"]],
                    PROMPT_MACROS[row["evidence"]],
                    PRECISION_LABELS[row["precision"]],
                    f"{int(row['nonzero_rows'])}",
                    f"{row['within_10']:.1f}",
                    f"{row['within_25']:.1f}",
                    f"{row['within_50']:.1f}",
                    f"{row['medape']:.1f}",
                ]
            )
            + r" \\"
        )

    table = r"""\begin{table*}[t]
  \centering
  \caption{Nonzero \rai prediction success rates on the 732-sample shared subset.
  A row is counted as accurate at threshold $x$ when its absolute percent error is at most $x$.
  The 25\% column is the main success-envelope threshold; 10\% and 50\% show stricter and looser views of the same distribution.}
  \label{tab:rai-threshold-success}
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lllrrrrr}
    \toprule
    Model & Evidence & Precision & Nonzero samples & $\le$10\% & $\le$25\% & $\le$50\% & MedAPE \\
    \midrule
"""
    table += "\n".join(rows)
    table += r"""
    \bottomrule
  \end{tabular}
\end{table*}
"""
    (TABLES_DIR / "rai_threshold_success.tex").write_text(table)


def write_static_tables() -> None:
    metric_guide = r"""\begin{table}[t]
  \centering
  \caption{How to read the evaluation metrics.}
  \label{tab:metric-guide}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{p{0.25\columnwidth}p{0.65\columnwidth}}
    \toprule
    Metric & Reader interpretation \\
    \midrule
    $\le$25\% APE & Fraction of nonzero samples where numeric \rai is close enough to treat as an accurate static estimate. \\
    MedAPE & Typical nonzero numeric error; useful for seeing heavy-tailed failures, not the whole story. \\
    BalAcc & Macro recall for imbalanced triage tasks, so zero or bandwidth-heavy classes cannot dominate the score. \\
    \bottomrule
  \end{tabular}
\end{table}
"""
    (TABLES_DIR / "metric_guide.tex").write_text(metric_guide)

    baseline_inputs = r"""\begin{table}[t]
  \centering
  \caption{Static reference baselines and their inputs.}
  \label{tab:baseline-inputs}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{p{0.28\columnwidth}p{0.62\columnwidth}}
    \toprule
    Baseline & Inputs and training protocol \\
    \midrule
    Context median & Runtime and GPU only; grouped cross-validation predicts the training-fold median \rai with runtime/GPU fallbacks. \\
    Source lexical & Deterministic source-token counts from \texttt{gpuFLOPBench.json}: arithmetic operators, memory references, types, and math calls; no training. \\
    SASS mnemonic & Deterministic architecture-specific instruction-mix counts from \texttt{gpuFLOPBench.json}: floating-point and global-memory mnemonics; no dynamic counts. \\
    Learned RF/ET & Random forest or extra-trees Stage~1 zero classifier plus Stage~2 $\log(1+\rai)$ regressor; GroupKFold by program; \sourceonly uses source lexical counts plus runtime/GPU, and \sourcesass adds static SASS instruction counts. \\
    \bottomrule
  \end{tabular}
\end{table}
"""
    (TABLES_DIR / "baseline_inputs.tex").write_text(baseline_inputs)

    regime_map = r"""\begin{table*}[t]
  \centering
  \caption{Empirical success envelope for off-the-shelf \rai prediction in this benchmark.}
  \label{tab:regime-map}
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{p{0.15\textwidth}p{0.23\textwidth}p{0.28\textwidth}p{0.24\textwidth}}
    \toprule
    Regime & Evidence pattern & What works & Main caution \\
    \midrule
    Works well & \sourcesass FP16/FP32 with lowered arithmetic visible & \opus reaches 47.0\%/53.0\% of nonzero samples within 25\% error and 0.984/0.967 BB/CB BalAcc. & Requires compilation to SASS for the target architecture. \\
    Works well & \sourceonly FP32 with source-visible loop and memory structure & \opus is within 25\% error on 36.0\% of nonzero samples and reaches 0.940 BB/CB BalAcc. & DRAM-visible traffic can still break source-level reasoning. \\
    Mixed & \sourceonly FP64 and \sourcesass FP64 & Many individual samples are accurate; static SASS baselines can match or beat LLM triage in aggregate. & Treat LLM output as one signal in a static-analysis cascade. \\
    Poor & \sourceonly FP16 & Nearly all nonzero samples miss the 25\% threshold and BB/CB triage stays at the trivial baseline. & Use SASS or cheap static filters before trusting numeric FP16 predictions. \\
    Poor & Context-overflow or very long prompts & \gptoss has lower completion coverage and weaker SASS use. & Coverage must be reported, and incomplete samples should not be hidden in aggregate metrics. \\
    \bottomrule
  \end{tabular}
\end{table*}
"""
    (TABLES_DIR / "regime_map.tex").write_text(regime_map)


def main() -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    rows = load_rows()
    shared = shared_subset(rows)
    generate_threshold_success(shared)
    write_static_tables()


if __name__ == "__main__":
    main()
