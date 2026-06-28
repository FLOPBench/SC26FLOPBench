#!/usr/bin/env python3
"""Select a small, stratified, auditable panel of kernels for the repeat-trials experiment.

The repeat-trials experiment resamples the LLMs (no code variants) on a handful of kernels to
characterize response variability, directly answering the SC reviewers who asked for repeated-query
consistency evidence.

This script picks ~24 kernels from the paper's *evaluated* set so the variability result augments the
reported numbers, then emits:
  * repeat_trials_subset.json   -- a pruned copy of dataset-creation/gpuFLOPBench.json (selected kernels
                                   only) that the repeat-trials runner consumes.
  * repeat_trials_manifest.csv  -- one row per selected (kernel, GPU, precision) documenting why it was
                                   picked (RAI, balance point, class, proximity band, feature flags, ...).
  * repeat_trials_rooflines.png -- a faceted (GPU x precision) roofline scatter for human double-checking.
  * a console summary + coverage report + a grounded per-model cost estimate for the planned runs
    (from true OpenRouter per-query costs in request_metadata; see --costCsv / --trials).

Ground-truth RAI / balance points come from the paper artifact
research-paper/fixup-ICSE/generated/static_prediction_rows.csv (which also gates eligibility to the 243
evaluated kernels). Per-kernel code-feature flags (for the hard/easy split) come from a CSV produced from
code_features_db (see --featureCsv); the raw FLOP/time used for the roofline plot come from the dataset JSON.

The selection is fully deterministic (no randomness): ties are broken by program/kernel name, so a no-arg
run reproduces the same panel every time (provided kernel_feature_flags.csv is present).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- Roofline specs: identical to experiments/direct-prompting/make_plots_for_paper.py GPU_ROOFLINE_TABLE ---
GPU_ROOFLINE_TABLE = {
    "3080": {"memory_bandwidth_gb_per_s": 760.0, "peak_tflops": {"fp16": 30.55, "fp32": 30.55, "fp64": 0.477}},
    "A10":  {"memory_bandwidth_gb_per_s": 600.0, "peak_tflops": {"fp16": 15.62, "fp32": 15.62, "fp64": 0.244}},
    "A100": {"memory_bandwidth_gb_per_s": 1555.0, "peak_tflops": {"fp16": 77.97, "fp32": 19.49, "fp64": 9.75}},
    "H100": {"memory_bandwidth_gb_per_s": 3360.0, "peak_tflops": {"fp16": 133.82, "fp32": 66.91, "fp64": 33.45}},
}
GPUS = ["3080", "A10", "A100", "H100"]
PRECISIONS = ["fp16", "fp32", "fp64"]
FLOP_KEY = {"fp16": "HP_FLOP", "fp32": "SP_FLOP", "fp64": "DP_FLOP"}
# Code features that the error analysis flagged as error-prone; presence => "hard" kernel.
HARD_FEATURES = ["has_flop_division", "has_special_math_functions",
                 "has_common_float_subexpr", "has_loop_invariant_flops"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "dataset-creation" / "gpuFLOPBench.json"
DEFAULT_EVAL_CSV = REPO_ROOT / "research-paper" / "fixup-ICSE" / "generated" / "static_prediction_rows.csv"
DEFAULT_FEATURE_CSV = REPO_ROOT / "experiments" / "repeat-trials" / "kernel_feature_flags.csv"
DEFAULT_COST_CSV = REPO_ROOT / "experiments" / "repeat-trials" / "query_costs.csv"
DEFAULT_OUTDIR = REPO_ROOT / "experiments" / "repeat-trials"

# Models the repeat-trials runner will query, mapped to the display names used in gpuflops_db / query_costs.csv.
RUN_MODEL_DISPLAY = {
    "anthropic/claude-opus-4.6": "Opus 4.6",
    "openai/gpt-5.4": "GPT 5.4",
    "openai/gpt-oss-120b": "GPT OSS",
}
DEFAULT_RUN_MODELS = list(RUN_MODEL_DISPLAY)
EVIDENCE = [("source-only", False), ("source+SASS", True)]  # the two no-IMIX prompt settings


def balance_point(gpu, precision):
    spec = GPU_ROOFLINE_TABLE[gpu]
    return spec["peak_tflops"][precision] * 1000.0 / spec["memory_bandwidth_gb_per_s"]


def build_eligibility_table(eval_csv, min_dist_pct, near_max_pct):
    """Per (program, kernel, gpu, precision) eligible nonzero cell with class + proximity band."""
    df = pd.read_csv(eval_csv)
    cols = ["program_name", "runtime", "kernel_mangled_name", "gpu", "precision",
            "expected_ai", "balance_point"]
    df = df[cols].drop_duplicates()
    df = df[(df["expected_ai"] > 0) & (df["balance_point"] > 0)].copy()
    df["rai"] = df["expected_ai"].astype(float)
    df["bp"] = df["balance_point"].astype(float)
    # cross-check the artifact balance points against our roofline table
    df["bp_table"] = df.apply(lambda r: balance_point(r["gpu"], r["precision"]), axis=1)
    max_bp_drift = (df["bp"] - df["bp_table"]).abs().max()
    df["dist"] = df["rai"] / df["bp"] - 1.0
    df = df[df["dist"].abs() >= min_dist_pct].copy()
    df["bound_class"] = df["rai"].gt(df["bp"]).map({True: "CB", False: "BB"})
    df["band"] = df["dist"].abs().le(near_max_pct).map({True: "near", False: "far"})
    return df, float(max_bp_drift)


def load_feature_flags(feature_csv):
    """Return dict (program_name, kernel_mangled_name) -> {'hard': bool, flags...} or None if unavailable."""
    path = Path(feature_csv)
    if not path.exists():
        return None
    fdf = pd.read_csv(path)
    if "program_name" not in fdf.columns or "kernel_mangled_name" not in fdf.columns:
        raise ValueError(f"{feature_csv} must contain program_name and kernel_mangled_name columns")
    present_hard = [c for c in HARD_FEATURES if c in fdf.columns]
    if not present_hard:
        raise ValueError(f"{feature_csv} has none of the hard-feature columns {HARD_FEATURES}")
    flags = {}
    for _, row in fdf.iterrows():
        key = (row["program_name"], row["kernel_mangled_name"])
        hard = any(bool(row.get(c, False)) for c in present_hard)
        rec = {"hard": hard}
        for c in HARD_FEATURES:
            if c in fdf.columns:
                rec[c] = bool(row.get(c, False))
        flags[key] = rec
    return flags


def build_kernel_records(elig_df, feature_flags):
    """Aggregate eligible cells to per-kernel records used by the selector."""
    records = {}
    for (prog, kern), g in elig_df.groupby(["program_name", "kernel_mangled_name"], sort=True):
        strata = set(zip(g["precision"], g["bound_class"], g["band"]))
        nonzero_precisions = set(g["precision"])
        feat = feature_flags.get((prog, kern)) if feature_flags else None
        records[(prog, kern)] = {
            "program_name": prog,
            "kernel_mangled_name": kern,
            "runtime": g["runtime"].iloc[0],
            "strata": strata,
            "gpus": set(g["gpu"]),
            "nonzero_precisions": nonzero_precisions,
            "mixed": len(nonzero_precisions) >= 2,
            "hard": (feat["hard"] if feat is not None else None),
            "has_features": feat is not None,
            "n_cells": int(len(g)),
        }
    return records


def _feat_of(r):
    return "hard" if r["hard"] is True else ("easy" if r["hard"] is False else None)


def select_panel(records, num_kernels, require_features, near_target, per_cell_target, max_cells_per_kernel,
                 max_kernels):
    """Greedy selection that BALANCES the panel at the (precision x class x feature) level so a hard-vs-easy
    variability comparison is possible within each precision/class, while also balancing CUDA vs OpenMP and
    the near band, then fills toward overall 50/50 hard/easy. A final phase grows the panel from num_kernels
    up to max_kernels to even out the CUDA/OpenMP runtime split. Mixed kernels are de-prioritized by cell
    count (and an optional --maxCellsPerKernel soft cap) so a few high-cell kernels cannot dominate.
    Deterministic: ties break by program/kernel name."""
    pool = sorted([k for k in records if (records[k]["has_features"] or not require_features)],
                  key=lambda k: (records[k]["program_name"], records[k]["kernel_mangled_name"]))

    # Per-kernel (precision, class, feature) strata and near (precision, class) sets.
    kpcf, knear = {}, {}
    for k in pool:
        r = records[k]; f = _feat_of(r)
        kpcf[k] = {(p, c, f) for (p, c, _) in r["strata"]} if f is not None else set()
        knear[k] = {(p, c) for (p, c, b) in r["strata"] if b == "near"}

    pcf_all = sorted(set().union(*kpcf.values())) if kpcf else []
    avail = {s: sum(1 for k in pool if s in kpcf[k]) for s in pcf_all}
    targets = {s: min(per_cell_target, avail[s]) for s in pcf_all}

    selected, sel = [], set()
    pcf_cov = defaultdict(int)
    near_cov = defaultdict(int)

    def over_cap(k):
        return 1 if (max_cells_per_kernel and records[k]["n_cells"] > max_cells_per_kernel) else 0

    def add(k):
        selected.append(k); sel.add(k)
        for s in kpcf[k]:
            pcf_cov[s] += 1
        for nc in knear[k]:
            near_cov[nc] += 1

    def need(k):
        return sum(1 for s in kpcf[k] if pcf_cov[s] < targets[s])

    def near_gain(k):
        return sum(1 for nc in knear[k] if near_cov[nc] < near_target)

    def rt_counts():
        c = {"cuda": 0, "omp": 0}
        for k in selected:
            c[records[k]["runtime"]] = c.get(records[k]["runtime"], 0) + 1
        return c

    def runtime_gain(k):
        c = rt_counts(); rt = records[k]["runtime"]
        other = "omp" if rt == "cuda" else "cuda"
        return 1 if c.get(rt, 0) <= c.get(other, 0) else 0  # prefer the under/equal-represented runtime

    def balance_gain(k):
        hc = sum(1 for x in selected if records[x]["hard"] is True)
        ec = len(selected) - hc
        if records[k]["hard"] is True:
            return 1 if hc <= ec else 0
        if records[k]["hard"] is False:
            return 1 if ec <= hc else 0
        return -1  # unknown-feature kernels deprioritized

    # Phase 1: meet per-(precision, class, feature) targets; among equals prefer the rarer runtime.
    while len(selected) < num_kernels:
        rem = [k for k in pool if k not in sel and need(k) > 0]
        if not rem:
            break
        rem.sort(key=lambda k: (over_cap(k), -need(k), -runtime_gain(k), records[k]["n_cells"],
                                -near_gain(k), records[k]["program_name"], records[k]["kernel_mangled_name"]))
        add(rem[0])

    # Phase 1b: boost the scarce near band; among equals prefer the rarer runtime.
    while len(selected) < num_kernels:
        rem = [k for k in pool if k not in sel and near_gain(k) > 0]
        if not rem:
            break
        rem.sort(key=lambda k: (over_cap(k), -near_gain(k), -runtime_gain(k), records[k]["n_cells"],
                                records[k]["program_name"], records[k]["kernel_mangled_name"]))
        add(rem[0])

    # Phase 2: fill to num_kernels, jointly improving hard/easy and runtime balance, preferring lean kernels.
    while len(selected) < num_kernels:
        rem = [k for k in pool if k not in sel]
        if not rem:
            break
        rem.sort(key=lambda k: (over_cap(k), -(balance_gain(k) + runtime_gain(k)), records[k]["n_cells"],
                                records[k]["program_name"], records[k]["kernel_mangled_name"]))
        add(rem[0])

    # Phase 3: grow from num_kernels up to max_kernels to even out the CUDA/OpenMP split.
    while len(selected) < max_kernels:
        c = rt_counts()
        if abs(c.get("cuda", 0) - c.get("omp", 0)) <= 1:
            break  # already as balanced as a one-kernel step allows
        under = "omp" if c.get("omp", 0) < c.get("cuda", 0) else "cuda"
        rem = [k for k in pool if k not in sel and records[k]["runtime"] == under]
        if not rem:
            break  # no more kernels of the under-represented runtime
        rem.sort(key=lambda k: (over_cap(k), -balance_gain(k), records[k]["n_cells"],
                                records[k]["program_name"], records[k]["kernel_mangled_name"]))
        add(rem[0])

    all_strata = sorted({s for r in records.values() for s in r["strata"]})
    covered = set().union(*[records[k]["strata"] for k in selected]) if selected else set()
    unachievable = sorted(set((p, c, b) for p in PRECISIONS for c in ("BB", "CB")
                              for b in ("near", "far")) - set(all_strata))
    pcf_report = {"targets": targets, "covered": dict(pcf_cov),
                  "unmet": {s: (pcf_cov[s], targets[s]) for s in targets if pcf_cov[s] < targets[s]},
                  "avail": avail}
    return selected, all_strata, covered, unachievable, pcf_report


def prune_dataset(dataset, selected_keys):
    """Return a pruned copy of the dataset JSON keeping only selected kernels (mirrors the dataset builder)."""
    by_program = defaultdict(set)
    for prog, kern in selected_keys:
        by_program[prog].add(kern)
    out = {}
    for prog, keep_kernels in by_program.items():
        data = dataset[prog]
        kept_kernels = {k: v for k, v in data["kernels"].items() if k in keep_kernels}
        kept_s2k = {}
        for src, knames in data.get("source_to_kernels", {}).items():
            filtered = [k for k in knames if k in keep_kernels]
            if filtered:
                kept_s2k[src] = filtered
        out[prog] = {
            "exeArgs": data.get("exeArgs", ""),
            "source_to_kernels": kept_s2k,
            "kernels": kept_kernels,
            "compile_commands": data.get("compile_commands", {}),
            "sources": data.get("sources", {}),
        }
    return out


def build_manifest(selected_keys, elig_df, records, dataset):
    rows = []
    sel = set(selected_keys)
    sub = elig_df[elig_df.apply(lambda r: (r["program_name"], r["kernel_mangled_name"]) in sel, axis=1)]
    for _, r in sub.iterrows():
        key = (r["program_name"], r["kernel_mangled_name"])
        rec = records[key]
        demangled = dataset.get(r["program_name"], {}).get("kernels", {}).get(
            r["kernel_mangled_name"], {}).get("demangledName", "")
        rows.append({
            "program_name": r["program_name"],
            "kernel_mangled_name": r["kernel_mangled_name"],
            "kernel_demangled_name": demangled,
            "runtime": r["runtime"],
            "gpu": r["gpu"],
            "precision": r["precision"],
            "expected_rai": round(float(r["rai"]), 6),
            "balance_point": round(float(r["bp"]), 6),
            "dist_pct": round(float(r["dist"]) * 100.0, 2),
            "bound_class": r["bound_class"],
            "proximity_band": r["band"],
            "hard": rec["hard"],
            "mixed": rec["mixed"],
            "stratum": f"{r['precision']}|{r['bound_class']}|{r['band']}",
        })
    mdf = pd.DataFrame(rows).sort_values(
        ["precision", "bound_class", "proximity_band", "program_name", "gpu"]).reset_index(drop=True)
    return mdf


def achieved_tflops(metric, precision):
    flop = metric.get(FLOP_KEY[precision], 0)
    xt_ns = metric.get("xtime_ns", 0)
    if not flop or not xt_ns:
        return None
    return flop / (xt_ns * 1e-9) / 1e12


def make_roofline_plot(manifest_df, dataset, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(len(GPUS), len(PRECISIONS), figsize=(16, 18), squeeze=False)
    class_color = {"BB": "#f58518", "CB": "#54a24b"}
    hard_marker = {True: "X", False: "o", None: "s"}

    for gi, gpu in enumerate(GPUS):
        for pj, prec in enumerate(PRECISIONS):
            ax = axes[gi][pj]
            spec = GPU_ROOFLINE_TABLE[gpu]
            peak = spec["peak_tflops"][prec]
            bw_tb = spec["memory_bandwidth_gb_per_s"] / 1000.0  # TB/s so bw*AI gives TFLOP/s
            bp = balance_point(gpu, prec)
            ai = np.logspace(-3, 3, 200)
            roof = np.minimum(peak, bw_tb * ai)
            ax.plot(ai, roof, color="0.4", lw=1.2, zorder=1)
            ax.axvline(bp, ls="--", color="0.6", lw=0.8, zorder=1)

            cell = manifest_df[(manifest_df["gpu"] == gpu) & (manifest_df["precision"] == prec)]
            for _, r in cell.iterrows():
                metric = dataset[r["program_name"]]["kernels"][r["kernel_mangled_name"]]["metrics"].get(gpu, {})
                perf = achieved_tflops(metric, prec)
                if perf is None or perf <= 0:
                    continue
                ax.scatter([r["expected_rai"]], [perf], s=70, zorder=3,
                           color=class_color.get(r["bound_class"], "gray"),
                           marker=hard_marker.get(r["hard"], "s"),
                           edgecolors="black", linewidths=0.5)
                label = f"{r['program_name'].replace('-cuda','').replace('-omp','')}::{r['kernel_demangled_name'][:18] or r['kernel_mangled_name'][:18]}"
                ax.annotate(label, (r["expected_rai"], perf), fontsize=5,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{gpu} / {prec} (BP={bp:.2f})", fontsize=9)
            if gi == len(GPUS) - 1:
                ax.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=8)
            if pj == 0:
                ax.set_ylabel(f"{gpu}\nAchieved TFLOP/s", fontsize=8)
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=class_color["BB"], markersize=8, label="Bandwidth-bound"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=class_color["CB"], markersize=8, label="Compute-bound"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markersize=8, label="hard kernel"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="easy kernel"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Repeat-trials panel: selected kernels on per-(GPU, precision) rooflines", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def load_cost_table(cost_csv):
    """Return (per_key, median) from observed query costs, or None if the CSV is absent.
    per_key:  (program, kernel, gpu, model_display, use_sass) -> observed cost_usd
    median:   (model_display, use_sass) -> median cost_usd (fallback for missing keys)."""
    path = Path(cost_csv)
    if not path.exists():
        return None
    cdf = pd.read_csv(path)
    cdf["use_sass"] = cdf["use_sass"].astype(bool)
    cdf["cost_usd"] = pd.to_numeric(cdf["cost_usd"], errors="coerce")
    cdf = cdf[cdf["cost_usd"].notna()]
    per_key = {(r.program_name, r.kernel_mangled_name, r.gpu, r.model_name, bool(r.use_sass)): float(r.cost_usd)
               for r in cdf.itertuples(index=False)}
    median = {(mn, bool(us)): float(v)
              for (mn, us), v in cdf.groupby(["model_name", "use_sass"])["cost_usd"].median().items()}
    return per_key, median


def estimate_costs(selected_keys, model_ids, trials, cost_table):
    """Grounded cost: sum the observed cost of each selected (kernel, GPU, model, evidence) from the original
    run (median fallback for any missing key), times the planned trial count. Queries all 4 GPUs per kernel."""
    per_key, median = cost_table
    rows = []
    grand = 0.0
    for mid in model_ids:
        disp = RUN_MODEL_DISPLAY.get(mid, mid)
        ev_costs = {}
        model_total = 0.0
        for ev_name, use_sass in EVIDENCE:
            per_trial = 0.0; obs = 0; fb = 0
            for (prog, kern) in selected_keys:
                for gpu in GPUS:
                    k = (prog, kern, gpu, disp, use_sass)
                    if k in per_key:
                        per_trial += per_key[k]; obs += 1
                    else:
                        per_trial += median.get((disp, use_sass), 0.0); fb += 1
            ev_total = per_trial * trials
            ev_costs[ev_name] = (ev_total, obs, fb)
            model_total += ev_total
        rows.append((disp, ev_costs, model_total))
        grand += model_total
    return rows, grand


def print_cost_estimate(selected_keys, model_ids, trials, cost_table):
    n_kernels = len(selected_keys); n_kg = n_kernels * len(GPUS)
    print("\n" + "=" * 78)
    print("COST ESTIMATE (grounded in true OpenRouter per-query costs from request_metadata):")
    if cost_table is None:
        print("  (skipped: query_costs.csv not found; see README Step 0b to regenerate it from request_metadata)")
        return
    rows, grand = estimate_costs(selected_keys, model_ids, trials, cost_table)
    print(f"  basis: {n_kernels} kernels x {len(GPUS)} GPUs = {n_kg} kernel-GPU queries "
          f"per (model,evidence) per trial; trials={trials}")
    print(f"  {'model':14} {'source-only':>13} {'source+SASS':>13} {'model total':>13}")
    for disp, ev, total in rows:
        print(f"  {disp:14} {'$'+format(ev['source-only'][0],',.2f'):>13} "
              f"{'$'+format(ev['source+SASS'][0],',.2f'):>13} {'$'+format(total,',.2f'):>13}")
    print(f"  {'TOTAL':14} {'':>13} {'':>13} {'$'+format(grand,',.2f'):>13}")
    tot_obs = sum(ev[e][1] for _, ev, _ in rows for e in ("source-only", "source+SASS"))
    tot_fb = sum(ev[e][2] for _, ev, _ in rows for e in ("source-only", "source+SASS"))
    n_calls = n_kg * len(model_ids) * len(EVIDENCE) * trials
    print(f"  total LLM calls: {n_kg} x {len(model_ids)} models x {len(EVIDENCE)} evidence x {trials} trials "
          f"= {n_calls:,}")
    print(f"  cost basis: {tot_obs} observed / {tot_fb} median-fallback per-key costs"
          + ("  (fallback = original GPT-OSS incompletions etc.)" if tot_fb else ""))


def print_summary(selected, records, all_strata, covered, unachievable, manifest_df, max_bp_drift, pcf_report):
    print("\n" + "=" * 78)
    print("BALANCE POINTS (peak_tflops*1000/bandwidth), per (GPU, precision):")
    for gpu in GPUS:
        bps = "  ".join(f"{p}={balance_point(gpu, p):.2f}" for p in PRECISIONS)
        print(f"  {gpu:5} {bps}")
    print(f"  (max drift vs artifact balance_point column: {max_bp_drift:.4f})")

    print("\n" + "=" * 78)
    print(f"SELECTED KERNELS ({len(selected)}):")
    print(f"{'program':28} {'kernel':40} {'rt':4} {'hard':5} {'mixed':5}")
    for key in sorted(selected, key=lambda k: records[k]["program_name"]):
        r = records[key]
        print(f"{r['program_name'][:27]:28} {r['kernel_mangled_name'][:39]:40} "
              f"{r['runtime'][:3]:4} {str(r['hard']):5} {str(r['mixed']):5}")

    hard = sum(1 for k in selected if records[k]["hard"] is True)
    easy = sum(1 for k in selected if records[k]["hard"] is False)
    unknown = sum(1 for k in selected if records[k]["hard"] is None)
    mixed = sum(1 for k in selected if records[k]["mixed"])
    rt = {"cuda": 0, "omp": 0}
    for k in selected:
        rt[records[k]["runtime"]] = rt.get(records[k]["runtime"], 0) + 1
    print("\n" + "=" * 78)
    print("COVERAGE REPORT:")
    print(f"  kernels: {len(selected)}   hard/easy/unknown: {hard}/{easy}/{unknown}   "
          f"mixed-precision: {mixed}")
    print(f"  runtime (kernel-level): cuda={rt.get('cuda',0)}  omp={rt.get('omp',0)}")
    print(f"  strata covered: {len(covered)}/{len(all_strata)} achievable "
          f"(of 12 theoretical)")
    if unachievable:
        print(f"  UNACHIEVABLE strata (no eligible kernel in evaluated set): "
              f"{', '.join('|'.join(s) for s in unachievable)}")

    print("\n  KERNELS per (precision|class|feature) stratum  (covered / target [available]):")
    for p in PRECISIONS:
        for c in ("BB", "CB"):
            cells = []
            for f in ("easy", "hard"):
                s = (p, c, f)
                cov = pcf_report["covered"].get(s, 0)
                tgt = pcf_report["targets"].get(s, 0)
                av = pcf_report["avail"].get(s, 0)
                cells.append(f"{f}={cov}/{tgt}[{av}]")
            print(f"    {p}|{c}: " + "  ".join(cells))
    if pcf_report["unmet"]:
        print("  UNMET targets: " + ", ".join(
            f"{'|'.join(s)} ({cov}/{tgt})" for s, (cov, tgt) in pcf_report["unmet"].items()))

    print("\n  manifest cells per stratum (precision|class|band):")
    counts = manifest_df.groupby(["precision", "bound_class", "proximity_band"]).size()
    for (p, c, b), n in counts.items():
        print(f"    {p}|{c}|{b}: {n}")
    print("\n  cells by GPU: " + ", ".join(f"{g}:{int((manifest_df['gpu']==g).sum())}" for g in GPUS))
    print("  cells by runtime: " + ", ".join(
        f"{rt}:{int((manifest_df['runtime']==rt).sum())}" for rt in sorted(manifest_df['runtime'].unique())))

    # Feature x class balance (cell-level = scatter points). Kernel-level hard/easy is the 12/12 above.
    df = manifest_df.copy()
    df["feat"] = df["hard"].map({True: "hard", False: "easy"}).fillna("unknown")

    def _xtab(group_col, group_values):
        header = f"{group_col:6} {'BB.easy':>8} {'BB.hard':>8} {'CB.easy':>8} {'CB.hard':>8}  {'total':>6}"
        print("    " + header)
        for gv in group_values:
            sub = df[df[group_col] == gv]
            cells = {(c, f): int(((sub['bound_class'] == c) & (sub['feat'] == f)).sum())
                     for c in ("BB", "CB") for f in ("easy", "hard")}
            print(f"    {str(gv):6} {cells[('BB','easy')]:>8} {cells[('BB','hard')]:>8} "
                  f"{cells[('CB','easy')]:>8} {cells[('CB','hard')]:>8}  {len(sub):>6}")

    print("\n  FEATURE x CLASS balance (cells), by precision:")
    _xtab("precision", PRECISIONS)
    print("\n  FEATURE x CLASS balance (cells), by GPU:")
    _xtab("gpu", GPUS)
    cell_hard = int((df["feat"] == "hard").sum())
    cell_easy = int((df["feat"] == "easy").sum())
    print(f"\n  cell-level easy/hard: {cell_easy}/{cell_hard}  "
          f"(kernel-level is {easy}/{hard}; hard kernels span more precisions/GPUs => more cells)")
    # warn where a hard-vs-easy comparison will be impossible
    gaps = []
    for p in PRECISIONS:
        for c in ("BB", "CB"):
            for f in ("easy", "hard"):
                n = int(((df["precision"] == p) & (df["bound_class"] == c) & (df["feat"] == f)).sum())
                if n == 0:
                    gaps.append(f"{p}|{c}|{f}")
    if gaps:
        print("  WARNING: empty feature/class cells (no hard-vs-easy comparison there): " + ", ".join(gaps))


def main():
    ap = argparse.ArgumentParser(description="Select the repeat-trials kernel panel.")
    ap.add_argument("--datasetJson", default=str(DEFAULT_DATASET))
    ap.add_argument("--evalCsv", default=str(DEFAULT_EVAL_CSV))
    ap.add_argument("--featureCsv", default=str(DEFAULT_FEATURE_CSV))
    ap.add_argument("--outDir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--numKernels", type=int, default=24,
                    help="base/minimum panel size before the CUDA/OpenMP balancing growth phase")
    ap.add_argument("--maxKernels", type=int, default=30,
                    help="hard cap; the panel may grow past --numKernels (up to this) to balance runtimes")
    ap.add_argument("--minDistPct", type=float, default=0.05, help="min |RAI/BP-1| for an unambiguous label")
    ap.add_argument("--nearMaxPct", type=float, default=0.50, help="upper |RAI/BP-1| of the near band")
    ap.add_argument("--nearTarget", type=int, default=3,
                    help="desired #kernels per near-band stratum (capped by availability)")
    ap.add_argument("--perCellTarget", type=int, default=2,
                    help="target #kernels per (precision,class,feature) stratum (capped by availability); "
                         "this is what balances hard vs easy within each precision/class")
    ap.add_argument("--maxCellsPerKernel", type=int, default=6,
                    help="soft cap on eligible cells per kernel; over-cap kernels are de-prioritized so a "
                         "few high-cell mixed kernels cannot dominate (0/None disables)")
    ap.add_argument("--noFeatures", action="store_true",
                    help="degraded mode: select without the hard/easy split (for testing only)")
    ap.add_argument("--trials", type=int, default=4,
                    help="planned repeat trials per query; used only for the cost estimate (default 4 ~= $153)")
    ap.add_argument("--costCsv", default=str(DEFAULT_COST_CSV),
                    help="observed per-(kernel,GPU,model,evidence) query costs for the cost estimate")
    ap.add_argument("--models", default=",".join(DEFAULT_RUN_MODELS),
                    help="models the runner will query (OpenRouter ids), for the cost estimate")
    args = ap.parse_args()

    out_dir = Path(args.outDir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Building eligibility table from evaluated-set GT ...")
    elig_df, max_bp_drift = build_eligibility_table(args.evalCsv, args.minDistPct, args.nearMaxPct)
    print(f"  eligible nonzero cells: {len(elig_df)} across "
          f"{elig_df[['program_name','kernel_mangled_name']].drop_duplicates().shape[0]} kernels")

    feature_flags = None if args.noFeatures else load_feature_flags(args.featureCsv)
    if feature_flags is None and not args.noFeatures:
        print(f"\nERROR: feature flags not found at {args.featureCsv}.\n"
              f"  Produce it from code_features_db (see experiments/repeat-trials/README), or\n"
              f"  re-run with --noFeatures for a degraded (no hard/easy split) test panel.", file=sys.stderr)
        sys.exit(2)
    if feature_flags is not None:
        print(f"  loaded feature flags for {len(feature_flags)} kernels")

    records = build_kernel_records(elig_df, feature_flags)
    selected, all_strata, covered, unachievable, pcf_report = select_panel(
        records, args.numKernels, require_features=not args.noFeatures,
        near_target=args.nearTarget, per_cell_target=args.perCellTarget,
        max_cells_per_kernel=(args.maxCellsPerKernel or None), max_kernels=args.maxKernels)
    selected_keys = set(selected)

    print("Loading dataset JSON ...")
    with open(args.datasetJson) as f:
        dataset = json.load(f)

    subset = prune_dataset(dataset, selected_keys)
    subset_path = out_dir / "repeat_trials_subset.json"
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)

    manifest_df = build_manifest(selected_keys, elig_df, records, dataset)
    manifest_path = out_dir / "repeat_trials_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    roofline_png = out_dir / "repeat_trials_rooflines.png"
    make_roofline_plot(manifest_df, dataset, roofline_png)

    print_summary(selected, records, all_strata, covered, unachievable, manifest_df, max_bp_drift, pcf_report)
    cost_table = load_cost_table(args.costCsv)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    print_cost_estimate(selected_keys, model_ids, args.trials, cost_table)
    print("\nWROTE:")
    print(f"  {subset_path}  ({len(subset)} programs, {len(selected)} kernels)")
    print(f"  {manifest_path}  ({len(manifest_df)} (kernel,GPU,precision) rows)")
    print(f"  {roofline_png}")


if __name__ == "__main__":
    main()
