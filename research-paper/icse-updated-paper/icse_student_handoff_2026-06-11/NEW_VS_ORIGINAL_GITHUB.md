# What Is New Vs. The Original GitHub Repository

## Original Repository Anchor

- Repository: `https://github.com/FLOPBench/SC26FLOPBench.git`
- Local mirror used during the rewrite: `/home/galoren/roofline_ICSE/SC26FLOPBench`
- Checked commit: `3bad4dc7e81e62f98f3a1bec44968c13874d2c32`

## Important Scope Note

The original GitHub repository was not rewritten in place.

Instead, the ICSE rewrite was developed as a separate clean submission project:

- `icse_resubmission/`

That means the easiest way to see the new work is not by looking for edits inside the original repository clone. The new manuscript, new analysis scripts, new tables, and the review-response/planning materials live in the separate ICSE project tree.

## Major New Work Added In This Rewrite

## 1. New ICSE Submission Project

- `icse_resubmission/main.tex`
- `icse_resubmission/sections/`
- `icse_resubmission/refs.bib`
- `icse_resubmission/main.pdf`

This is a full anonymous ACM double-column ICSE submission project, not a patch over the older SC-style source.

## 2. Full Manuscript Repositioning

The paper was rewritten to present the work as:

- execution-free early performance triage for GPU software engineering
- useful to developers and AI coding agents under scarce profiling access
- explicitly not a profiler replacement

This affects:

- abstract
- introduction
- related work
- task/methodology
- study design
- results
- discussion
- threats
- conclusion

## 3. New Artifact-Backed Analyses

Added new local analysis scripts:

- `icse_resubmission/analysis/generate_additional_artifacts.py`
- `icse_resubmission/analysis/generate_feature_vote_artifacts.py`

These scripts generated new manuscript evidence directly from the locally available artifacts, including:

- deterministic baselines
- lightweight learned baselines
- seed-sensitivity summaries
- fixed-budget profiling analysis
- budget sweep tables
- paired source-vs-SASS significance summaries
- confusion-count exports
- feature-vote prevalence and BH-corrected appendix support summaries

## 4. New Generated Outputs

Key generated outputs are in:

- `icse_resubmission/generated/`
- `icse_resubmission/tables/`

Important examples:

- `deterministic_baselines.*`
- `learned_baselines.*`
- `learned_baseline_seed_summary.csv`
- `budget_recall_sweep_selected.csv`
- `paired_source_vs_sass_stats.csv`
- `llm_vs_static_task_c.*`
- `feature_vote_feature_summary.csv`
- `feature_vote_model_prompt_stats.csv`

## 5. New Review/Planning/Validation Files

- `icse_resubmission/review_action_trace.md`
- `icse_resubmission/self_review_rounds.md`
- `icse_resubmission/pdf_visual_check.md`
- `icse_resubmission/CHANGELOG_ICSE_REWRITE.md`
- `icse_resubmission/FINAL_REPORT.md`

These are the “why we changed this / what is still left” documents the student should use first.

## 6. New Main-Text Comparison Story

Compared with the earlier manuscript framing, the current paper now explicitly includes:

- clearer task split: zero/nonzero, nonzero regression, nonzero BB/CB triage, and three-way triage
- stronger static reference baselines
- best-LLM-vs-best-static regime comparison in the main text
- deployment-routing summary for tool builders
- fixed-budget profiling utility framing
- stronger threats and claim discipline

## What Was Not Bundled From The Original Repository

The full original repository clone is about `11G`, so it is not included inside this handoff zip.

That is intentional.

The bundle instead includes:

- the original repo identity and commit hash
- the original source snapshot
- the public website mirror data
- the full new ICSE project tree

If the student needs the full original repository locally, they should clone the remote at the commit listed above.
