#!/usr/bin/env python3
"""Analyze repeat-trials response variability and emit the paper's Repeated-Query Robustness artifacts.

Reads the dedicated ``gpuflops_repeat_db`` (read-only), joins the curated panel manifest, and characterizes
how much each model's predictions vary across repeated queries of the SAME (kernel, GPU, evidence) input.

Metrics (per cell = program x kernel x GPU x model x evidence x precision, across its trials):
  * pred_rai_cv         -- coefficient of variation (std/mean) of predicted RAI  [primary numeric consistency]
  * pred_rai_logstd     -- std of log1p(predicted RAI)                            [scale-robust companion]
  * bbcb_consistent     -- did the bandwidth/compute-bound call agree on every trial?  [decision consistency]
  * nz_consistent       -- did zero/nonzero agree on every trial?                  [zero-detection consistency]
  * mean_ape / std_ape  -- per-trial absolute percent error: accuracy + its run-to-run spread

Outputs: a main-text LaTeX table (overwrites the placeholder), a seaborn appendix figure, and per-cell /
per-group CSVs. Works on whatever data is present (partial runs included). All plotting uses seaborn.
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DB_URI = "postgresql://postgres:postgres@localhost:5432/gpuflops_repeat_db"
DEFAULT_MANIFEST = HERE / "repeat_trials_manifest.csv"
PAPER_TABLE = REPO_ROOT / "research-paper" / "fixup-ICSE" / "tables" / "repeated_query_variance.tex"
PAPER_FIG_DIR = REPO_ROOT / "research-paper" / "fixup-ICSE" / "figures"
PRECISIONS = ["fp16", "fp32", "fp64"]
MODEL_ORDER = ["Opus 4.6", "GPT 5.4", "GPT OSS"]
EVIDENCE_ORDER = ["source-only", "source+SASS"]

# The repeat DB stores llm_model_name inconsistently: some models land as friendly labels ("GPT OSS")
# while others keep a raw, date-versioned id ("gpt-5.4-2026-03-05"). Normalize to the display labels in
# MODEL_ORDER so grouping, sorting, and the paper table line up. Date suffixes (-YYYY-MM-DD or -YYYYMMDD)
# are stripped before lookup.
_MODEL_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$")
_MODEL_LABEL_MAP = {
    "anthropic/claude-opus-4.6": "Opus 4.6",
    "claude-opus-4.6": "Opus 4.6",
    "opus 4.6": "Opus 4.6",
    "openai/gpt-5.4": "GPT 5.4",
    "gpt-5.4": "GPT 5.4",
    "gpt 5.4": "GPT 5.4",
    "openai/gpt-oss-120b": "GPT OSS",
    "gpt-oss-120b": "GPT OSS",
    "gpt oss": "GPT OSS",
}


def _normalize_model_name(value):
    if not isinstance(value, str):
        return value
    base = _MODEL_DATE_SUFFIX.sub("", value).strip()
    return _MODEL_LABEL_MAP.get(base.casefold(), base)


def _mark_invalid_predictions(completed, long_df):
    """Flag and null out predicted AIs built from negative sentinels.

    A model emits ``-1`` ("cannot determine") for a FLOP or DRAM-byte field it declines to estimate. Those
    are abstentions, not numeric predictions, yet they flow through AI = FLOP / bytes and produce impossible
    intensities -- negative when one side is negative (e.g. 4608 / -2 = -2304) AND, more sneakily, *positive*
    when both sides are negative (e.g. -1 / -2 = 0.5). So invalidity is decided at the field level (precision
    FLOP < 0 or predicted_total_bytes < 0), not by the sign of the AI. Invalid trials get predicted_ai /
    abs_ai_pct_error nulled (so they don't pollute the variability stats) and an ``invalid_pred`` flag (so we
    can report an abstention rate). Zero-byte cases are already NaN via db_reader's _safe_divide."""
    total_bytes = pd.to_numeric(completed["predicted_total_bytes"], errors="coerce")
    records = []
    for precision in PRECISIONS:
        flop = pd.to_numeric(completed[f"predicted_{precision}"], errors="coerce")
        invalid = ((flop < 0) | (total_bytes < 0)).fillna(False)
        records.append(pd.DataFrame({
            "thread_id": completed["thread_id"].values,
            "precision": precision,
            "invalid_pred": invalid.values,
        }))
    inv = pd.concat(records, ignore_index=True)
    out = long_df.merge(inv, on=["thread_id", "precision"], how="left")
    out["invalid_pred"] = out["invalid_pred"].fillna(False)
    out.loc[out["invalid_pred"], ["predicted_ai", "abs_ai_pct_error"]] = np.nan
    return out


def _load_db_reader():
    path = REPO_ROOT / "experiments" / "error-analysis" / "db_reader.py"
    spec = importlib.util.spec_from_file_location("db_reader", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_frames(db_uri):
    """Return (completed_per_query_df, per_trial_long_df). The completed frame has one row per query
    (with cost_usd); the long frame explodes it to one row per precision."""
    dbr = _load_db_reader()
    samples = dbr.load_gpuflops_samples_dataframe(db_uri=db_uri)
    completed = dbr.enrich_gpuflops_with_ai_metrics(samples)
    long_df = dbr.build_sample_ai_error_long_dataframe(completed)
    completed["model_name"] = completed["model_name"].map(_normalize_model_name)
    long_df["model_name"] = long_df["model_name"].map(_normalize_model_name)
    long_df = _mark_invalid_predictions(completed, long_df)
    return completed, long_df


def _load_sample_true_costs(db_uri):
    """Return the sample_true_costs table (thread_id, model, true_cost_usd, cost_updated) or None if absent."""
    try:
        import psycopg
        with psycopg.connect(db_uri) as conn, conn.cursor() as cur:
            cur.execute("SELECT thread_id, model, true_cost_usd, cost_updated FROM sample_true_costs")
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return None


def cost_summary(completed, db_uri):
    """Per-model experiment cost. Prefers the authoritative OpenRouter cost from sample_true_costs
    (true_cost_usd where cost_updated), falling back to the stored per-query estimate for samples not yet
    backfilled. If sample_true_costs is absent, reports the stored estimate (and says so)."""
    c = completed.copy()
    c["cost_usd"] = pd.to_numeric(c.get("cost_usd"), errors="coerce")
    est_by_thread = dict(zip(c["thread_id"], c["cost_usd"]))

    stc = _load_sample_true_costs(db_uri)
    if stc is None or stc.empty:
        per_model = (c.groupby("model_name")
                     .agg(threads=("cost_usd", "size"), total_cost_usd=("cost_usd", "sum"))
                     .reset_index().sort_values("total_cost_usd", ascending=False))
        per_model["updated"] = 0
        return per_model, "estimate"

    stc["true_cost_usd"] = pd.to_numeric(stc["true_cost_usd"], errors="coerce")
    stc["cost_updated"] = stc["cost_updated"].astype(bool)
    stc["effective_cost"] = np.where(
        stc["cost_updated"], stc["true_cost_usd"], stc["thread_id"].map(est_by_thread))
    per_model = (stc.groupby("model")
                 .agg(threads=("thread_id", "size"),
                      updated=("cost_updated", "sum"),
                      total_cost_usd=("effective_cost", "sum"))
                 .reset_index().rename(columns={"model": "model_name"})
                 .sort_values("total_cost_usd", ascending=False))
    return per_model, "true+estimate"


def _cv(values):
    """Coefficient of variation of predicted RAI across trials. All-zero predictions are degenerately
    consistent (cv=0); needs >=2 trials and a nonzero mean otherwise."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 2:
        return np.nan
    if (v == 0).all():
        return 0.0
    mean = v.mean()
    if mean == 0:
        return np.nan
    return float(v.std(ddof=1) / abs(mean))


def build_cell_table(long_df, manifest_path, min_trials):
    long_df = long_df[long_df["use_imix"] == False].copy()  # noqa: E712 -- pandas mask
    long_df["evidence"] = np.where(long_df["use_sass"].astype(bool), "source+SASS", "source-only")
    long_df["predicted_ai"] = pd.to_numeric(long_df["predicted_ai"], errors="coerce")
    long_df["abs_ai_pct_error"] = pd.to_numeric(long_df["abs_ai_pct_error"], errors="coerce")

    man = pd.read_csv(manifest_path)[
        ["program_name", "kernel_mangled_name", "gpu", "precision",
         "balance_point", "bound_class", "proximity_band", "hard"]
    ]
    # Inner join: restrict to the curated, stratified panel cells (nonzero-GT, eligible).
    d = long_df.merge(man, on=["program_name", "kernel_mangled_name", "gpu", "precision"], how="inner")
    d["hard_easy"] = d["hard"].map({True: "hard", False: "easy"})
    # Class / zero verdicts only make sense for valid numeric predictions; invalid (sentinel) trials carry a
    # NaN predicted_ai and must stay NaN here rather than defaulting to "BB"/"zero".
    d["pred_class"] = np.where(d["predicted_ai"] > d["balance_point"], "CB", "BB")
    d.loc[d["predicted_ai"].isna(), "pred_class"] = np.nan
    d["pred_nonzero"] = np.where(d["predicted_ai"].isna(), np.nan, d["predicted_ai"] > 0)

    cell_keys = ["program_name", "kernel_mangled_name", "gpu", "model_name", "evidence", "precision"]
    rows = []
    for key, sub in d.groupby(cell_keys, dropna=False):
        n = int(sub["trial"].nunique())
        n_invalid = int(sub["invalid_pred"].sum())
        pred_valid = pd.to_numeric(sub["predicted_ai"], errors="coerce").dropna()
        n_valid = int(len(pred_valid))
        pred_class = sub["pred_class"].dropna()
        pred_nonzero = sub["pred_nonzero"].dropna()
        ape = sub["abs_ai_pct_error"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            **dict(zip(cell_keys, key)),
            "hard_easy": sub["hard_easy"].iloc[0],
            "bound_class": sub["bound_class"].iloc[0],
            "proximity_band": sub["proximity_band"].iloc[0],
            "n_trials": n,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "invalid_rate": float(sub["invalid_pred"].mean()) if len(sub) else np.nan,
            "pred_rai_cv": _cv(pred_valid),
            "pred_rai_logstd": float(np.log1p(pred_valid).std(ddof=1)) if n_valid >= 2 else np.nan,
            "bbcb_consistent": (int(pred_class.nunique() == 1) if len(pred_class) >= min_trials else np.nan),
            "nz_consistent": (int(pred_nonzero.nunique() == 1) if len(pred_nonzero) >= min_trials else np.nan),
            "mean_ape": float(ape.mean()) if len(ape) else np.nan,
            "std_ape": float(ape.std(ddof=1)) if len(ape) >= 2 else np.nan,
        })
    return pd.DataFrame(rows)


def build_group_table(cells):
    """Aggregate cells by (model, evidence, precision); also by hard/easy in a separate frame."""
    def agg(g):
        total_trials = float(g["n_trials"].sum())
        return pd.Series({
            "n_cells": int(len(g)),
            "median_cv": float(np.nanmedian(g["pred_rai_cv"])) if g["pred_rai_cv"].notna().any() else np.nan,
            "bbcb_agreement": float(np.nanmean(g["bbcb_consistent"])) if g["bbcb_consistent"].notna().any() else np.nan,
            "nz_agreement": float(np.nanmean(g["nz_consistent"])) if g["nz_consistent"].notna().any() else np.nan,
            "median_mean_ape": float(np.nanmedian(g["mean_ape"])) if g["mean_ape"].notna().any() else np.nan,
            "invalid_rate": float(g["n_invalid"].sum() / total_trials) if total_trials else np.nan,
        })
    by_group = cells.groupby(["model_name", "evidence", "precision"], dropna=False).apply(
        agg, include_groups=False).reset_index()
    by_group_he = cells.groupby(["model_name", "evidence", "precision", "hard_easy"], dropna=False).apply(
        agg, include_groups=False).reset_index()
    return by_group, by_group_he


def _fmt(x, pct=False):
    if pd.isna(x):
        return "--"
    return f"{x*100:.0f}\\%" if pct else f"{x:.2f}"


def write_latex_table(by_group, out_path):
    """Main-text table: per (model, evidence) -> per-precision median CV and BB/CB agreement."""
    g = by_group.set_index(["model_name", "evidence", "precision"])
    models = [m for m in MODEL_ORDER if m in by_group["model_name"].unique()]
    models += [m for m in by_group["model_name"].unique() if m not in models]
    lines = [
        r"\begin{table*}[t]", r"  \centering",
        r"  \caption{Repeated-query response variability on the repeat-trials panel. "
        r"``CV'' is the median coefficient of variation of predicted \rai across repeats (lower is more "
        r"consistent); ``Agr'' is the fraction of cells whose \bandwidthbound/\computebound call is unanimous "
        r"across repeats. Computed at the main-paper decode settings.}",
        r"  \label{tab:repeated-query-variance}", r"  \scriptsize", r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llrrrrrr}", r"    \toprule",
        r"    & & \multicolumn{2}{c}{FP16} & \multicolumn{2}{c}{FP32} & \multicolumn{2}{c}{FP64} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"    Model & Evidence & CV & Agr & CV & Agr & CV & Agr \\", r"    \midrule",
    ]
    for model in models:
        for ev in EVIDENCE_ORDER:
            cv_agr = []
            any_data = False
            for p in PRECISIONS:
                if (model, ev, p) in g.index:
                    row = g.loc[(model, ev, p)]
                    cv_agr += [_fmt(row["median_cv"]), _fmt(row["bbcb_agreement"], pct=True)]
                    any_data = True
                else:
                    cv_agr += ["--", "--"]
            if any_data:
                lines.append(f"    \\texttt{{{model}}} & {ev} & " + " & ".join(cv_agr) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def make_figures(cells, out_dir, paper_fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    model_order = [m for m in MODEL_ORDER if m in cells["model_name"].unique()]

    # Figure 1: per-cell CV of predicted RAI, faceted evidence x precision, split by hard/easy.
    cv_df = cells.dropna(subset=["pred_rai_cv"])
    paths = []
    if not cv_df.empty:
        grid = sns.catplot(
            data=cv_df, kind="box", row="evidence", col="precision",
            x="model_name", y="pred_rai_cv", hue="hard_easy",
            order=model_order, col_order=PRECISIONS, row_order=EVIDENCE_ORDER,
            hue_order=["easy", "hard"], height=3.2, aspect=1.1, fliersize=2,
        )
        grid.set_axis_labels("model", "CV of predicted RAI across repeats")
        grid.set_titles(row_template="{row_name}", col_template="{col_name}")
        for ax in grid.axes.flat:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(20)
        grid.figure.suptitle("Repeat-trials: predicted-RAI variability (CV) by model, evidence, precision",
                             y=1.02)
        p1 = out_dir / "repeat_trials_variability.png"
        grid.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(grid.figure); paths.append(p1)

    # Figure 2: BB/CB agreement rate (companion), same faceting.
    agr = (cells.dropna(subset=["bbcb_consistent"])
           .groupby(["model_name", "evidence", "precision", "hard_easy"], dropna=False)["bbcb_consistent"]
           .mean().reset_index())
    if not agr.empty:
        grid2 = sns.catplot(
            data=agr, kind="bar", row="evidence", col="precision",
            x="model_name", y="bbcb_consistent", hue="hard_easy",
            order=model_order, col_order=PRECISIONS, row_order=EVIDENCE_ORDER,
            hue_order=["easy", "hard"], height=3.2, aspect=1.1,
        )
        grid2.set_axis_labels("model", "BB/CB agreement across repeats")
        grid2.set(ylim=(0, 1))
        grid2.set_titles(row_template="{row_name}", col_template="{col_name}")
        for ax in grid2.axes.flat:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(20)
        p2 = out_dir / "repeat_trials_agreement.png"
        grid2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(grid2.figure); paths.append(p2)

    # Copy the primary variability figure into the paper's figures dir if it exists (skip self-copy).
    if paths and paper_fig_dir.exists():
        import shutil
        dest = (paper_fig_dir / "repeat_trials_variability.png").resolve()
        if dest != paths[0].resolve():
            shutil.copy(paths[0], dest)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Analyze repeat-trials response variability.")
    ap.add_argument("--dbUri", default=DEFAULT_DB_URI)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--outDir", default=str(HERE))
    ap.add_argument("--paperTable", default=str(PAPER_TABLE))
    ap.add_argument("--paperFigDir", default=str(PAPER_FIG_DIR))
    ap.add_argument("--minTrials", type=int, default=2, help="min completed trials for a consistency verdict")
    ap.add_argument("--noPaperTable", action="store_true", help="don't overwrite the paper table placeholder")
    args = ap.parse_args()

    out_dir = Path(args.outDir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading from", args.dbUri, "...")
    completed, long_df = load_frames(args.dbUri)
    if long_df.empty:
        print("No completed samples found in the repeat DB yet.", file=sys.stderr); sys.exit(1)

    cells = build_cell_table(long_df, args.manifest, args.minTrials)
    if cells.empty:
        print("No panel cells matched the manifest (no curated-cell data yet).", file=sys.stderr); sys.exit(1)
    by_group, by_group_he = build_group_table(cells)

    cells.to_csv(out_dir / "per_cell_variability.csv", index=False)
    by_group.to_csv(out_dir / "per_group_variability.csv", index=False)
    by_group_he.to_csv(out_dir / "per_group_variability_by_feature.csv", index=False)

    if not args.noPaperTable:
        write_latex_table(by_group, Path(args.paperTable))

    figs = make_figures(cells, out_dir, Path(args.paperFigDir))

    # Console summary
    print(f"\ncells analyzed: {len(cells)}  (>= {args.minTrials} trials for consistency verdicts)")
    print("models present:", sorted(cells["model_name"].unique()))
    print("trials per cell: min/median/max =",
          int(cells["n_trials"].min()), int(cells["n_trials"].median()), int(cells["n_trials"].max()))
    total_trials = int(cells["n_trials"].sum())
    total_invalid = int(cells["n_invalid"].sum())
    print(f"invalid (sentinel) trials: {total_invalid}/{total_trials} "
          f"({(100.0 * total_invalid / total_trials if total_trials else 0):.1f}%) "
          "-- nulled out of CV/APE/BB-CB, reported as invalid_rate")
    print("\nper (model, evidence, precision):")
    show = by_group.copy()
    for c in ["median_cv", "bbcb_agreement", "nz_agreement", "median_mean_ape", "invalid_rate"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    # Experiment cost per model (prefers authoritative OpenRouter cost from sample_true_costs).
    per_model, cost_source = cost_summary(completed, args.dbUri)
    per_model.to_csv(out_dir / "experiment_cost_by_model.csv", index=False)
    if cost_source == "true+estimate":
        print("\nEXPERIMENT COST per model (true OpenRouter cost where backfilled, else stored estimate):")
        for _, r in per_model.iterrows():
            print(f"  {r['model_name']:10}  {int(r['updated'])}/{int(r['threads'])} samples true-costed   "
                  f"${r['total_cost_usd']:.4f}")
        print(f"  {'TOTAL':10}  ${per_model['total_cost_usd'].sum():.4f}")
        print("  (run run_repeat_trials.py with an API key to backfill any remaining samples into "
              "sample_true_costs)")
    else:
        print("\nEXPERIMENT COST per model (stored per-query ESTIMATES; sample_true_costs not populated yet):")
        for _, r in per_model.iterrows():
            print(f"  {r['model_name']:10}  {int(r['threads']):4d} queries   ${r['total_cost_usd']:.4f}")
        print(f"  {'TOTAL':10}  ${per_model['total_cost_usd'].sum():.4f}")
        print("  (estimates are unreliable, esp. GPT-OSS ~50% low; run run_repeat_trials.py to backfill true costs)")

    print("\nWROTE:")
    print(f"  {out_dir/'per_cell_variability.csv'}  ({len(cells)} cells)")
    print(f"  {out_dir/'per_group_variability.csv'} ; {out_dir/'per_group_variability_by_feature.csv'}")
    if not args.noPaperTable:
        print(f"  {args.paperTable}  (paper table)")
    for p in figs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
