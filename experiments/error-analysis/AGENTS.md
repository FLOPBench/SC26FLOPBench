# Error Analysis

## Purpose

This directory contains the analysis code used to connect kernel code features to arithmetic intensity (AI) prediction error from the direct-prompting experiments.

The current paper-facing workflow is centered on sample-level analysis rather than kernel-level summaries. Each sample keeps its GPU, runtime, precision, model, and prompt-type identity, then joins those AI error measurements with kernel feature flags from feature voting. That lets the plots answer questions of the form:

- Which features are associated with higher AI error?
- Does that association change by GPU, runtime, precision, or LLM?
- Which features are consistently difficult across many analysis cells?

## Files

### [db_reader.py](/gpuFLOPBench-updated/experiments/error-analysis/db_reader.py)

Loads and merges the analysis inputs.

- Reads direct-prompting results from `gpuflops_db`
- Reads feature-voting outputs from `code_features_db`
- Produces `sample_with_features_df` for sample-level plotting
- Produces merge diagnostics so unmatched kernels are visible

Key join identity:

- `program_name`
- `kernel_mangled_name`

### [make_plots_for_paper.py](/gpuFLOPBench-updated/experiments/error-analysis/make_plots_for_paper.py)

Builds the paper figures that summarize feature/error associations.

Current behavior:

- Operates on `sample_with_features_df`
- Restricts prompt types to `Source-Only` and `Source+SASS`
- Drops IMIX-backed rows from this analysis
- Optionally keeps only shared samples across all compared models with `--onlySharedSamples`
- Computes present-vs-absent feature association scores using Cliff's delta
- Renders several heatmap families:
	- Full grids by prompt type, precision, runtime, GPU, and model
	- Heatmaps collapsed over model
	- Heatmaps collapsed over precision
	- Heatmaps collapsed over both model and precision
	- A fully collapsed runtime summary with rows `CUDA` and `OpenMP`
- Writes a CSV containing the plotted association cells

## How To Run

From the repository root:

```bash
cd experiments/error-analysis
python make_plots_for_paper.py --onlySharedSamples
```

Useful options:

```bash
python make_plots_for_paper.py \
	--onlySharedSamples \
	--minPresent 5 \
	--minAbsent 5 \
	--outputDir ./paper-figure-output
```

Database overrides:

```bash
python make_plots_for_paper.py \
	--gpuflopsDbUri postgresql://... \
	--codeFeaturesDbUri postgresql://...
```

### Important CLI Flags

- `--onlySharedSamples`: keeps only kernel samples that exist across all model names for a matched program, kernel, GPU, and prompt type. Use this when the comparison across models should be fairness-preserving.
- `--minPresent`: minimum number of feature-present samples required before a cell is considered valid.
- `--minAbsent`: minimum number of feature-absent samples required before a cell is considered valid.
- `--outputDir`: destination for all generated figures and CSV exports.

## Outputs

The default output directory is [paper-figure-output](/gpuFLOPBench-updated/experiments/error-analysis/paper-figure-output).

Important outputs include:

- `Source-Only_feature_association_heatmaps.png`
- `Source_SASS_feature_association_heatmaps.png`
- collapsed summary heatmap variants
- `runtime_feature_association_heatmap.png`
- `feature_association_summary.png`
- `feature_error_associations.csv`

## Statistical Method: Cliff's Delta

### What It Measures

For each analysis cell, the plotting script splits samples into two groups:

- kernels where a given feature is present
- kernels where that feature is absent

It then computes Cliff's delta between the two AI-error distributions.

Interpretation:

- Positive values: feature-present samples tend to have higher AI error than feature-absent samples
- Negative values: feature-present samples tend to have lower AI error than feature-absent samples
- Values near zero: little directional separation between the two groups
- Values near `1` or `-1`: strong separation

Conceptually, Cliff's delta asks how often a randomly chosen feature-present sample has larger error than a randomly chosen feature-absent sample, minus the reverse case.

### Why It Fits This Task

This analysis is about directional association between a binary kernel feature flag and a continuous error quantity. The metric needs to work reliably when:

- the error distributions are skewed
- the sample counts differ sharply between feature-present and feature-absent groups
- outliers exist
- we care more about effect size than hypothesis-test p-values

Cliff's delta fits that well because it is:

- nonparametric
- robust to monotonic rescaling and distribution shape
- easy to interpret as a directional effect size
- directly aligned with the question: does feature presence tend to push AI error up or down?

### Why It Was Used Instead Of Other Nonparametric Methods

Cliff's delta was chosen over common alternatives because the goal here is ranking feature/error association strength, not performing significance testing alone.

Compared with Mann-Whitney U or Wilcoxon rank-sum:

- those tests primarily answer whether two groups differ, not how strongly feature presence shifts error
- their p-values depend heavily on sample size, which is a poor fit for heatmaps meant to compare many cells side by side
- Cliff's delta can be derived from the same rank-comparison idea but yields an interpretable signed effect size directly

Compared with Spearman or Kendall correlation:

- those are more natural for ordered or continuous predictors on both axes
- here the predictor is binary feature presence, so a present-vs-absent effect-size comparison is the more direct framing
- Cliff's delta preserves that binary-group comparison explicitly

Compared with point-biserial correlation:

- point-biserial is simple, but it is tied more closely to mean shifts and is less robust when the error distributions are heavy-tailed or strongly non-normal
- AI percentage error can be skewed and outlier-prone, which makes a rank-based effect size more stable

### Practical Interpretation In These Plots

When reading the heatmaps:

- larger positive values indicate features that are more strongly associated with higher AI error when present
- more negative values indicate features whose presence is associated with lower AI error
- masked cells indicate there were not enough present or absent samples to support a stable comparison under the configured thresholds

The fully collapsed runtime heatmap orders features from left to right by overall signed association, so the left side emphasizes features most associated with higher AI error and the right side emphasizes those associated with lower AI error.
