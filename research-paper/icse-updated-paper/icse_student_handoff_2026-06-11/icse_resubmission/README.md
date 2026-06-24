# ICSE Resubmission Project

This directory contains a clean ACM `acmart` rewrite of the paper as an ICSE submission project.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` reports a missing TeX package, install it with `tlmgr` in user mode against the TeX Live 2021 archive already configured in this workspace.

## Project Layout

- `main.tex`: top-level ACM/ICSE manuscript
- `analysis/generate_additional_artifacts.py`: recomputes the static-baseline tables, class-balance and compute-bound precision/recall tables, budget-sweep tables, learned-baseline seed-sensitivity tables, paired source-vs-SASS statistics, and raw confusion-count CSVs from the local benchmark and website mirrors
- `analysis/generate_feature_vote_artifacts.py`: recomputes the appendix feature-prevalence and BH-corrected support summaries from restored `gpuflops_db` and `code_features_db` PostgreSQL instances
- `sections/`: section files
- `figures/`: copied or redrawn manuscript figures based on existing results
- `tables/`: reusable LaTeX tables
- `generated/`: CSV exports from the manuscript-side analysis script
  - includes the static baseline summaries, shared-subset class counts, compute-bound precision/recall summaries, raw confusion counts, budget sweeps, selected budget-sweep rows, learned-baseline seed-sensitivity summaries, and appendix feature-vote support summaries used in the final rewrite
- `refs.bib`: bibliography carried forward from the original source
- `review_action_trace.md`: mapping from review issues to manuscript changes
- `self_review_rounds.md`: internal review log, including the mandatory three rounds and the final polish pass
- `pdf_visual_check.md`: PDF inspection log, including the required three passes and later validation passes
- `FINAL_REPORT.md`: final delivery summary and validation checklist

## Anonymous Review Notes

This manuscript is kept in anonymous review mode.
The current workspace contains public mirrors of the benchmark website and repository:

- public website mirror source: `/home/galoren/roofline_ICSE/_website_mirror/`
- public repository mirror: `/home/galoren/roofline_ICSE/SC26FLOPBench/`

The manuscript intentionally omits those public URLs.
For a de-anonymized camera-ready version, replace the generic benchmark-release wording with the public website and repository links, then restore acknowledgments and author metadata.

## Evidence Sources Used in the Rewrite

- original SC-era manuscript source: `/home/galoren/roofline_ICSE/_paper_src/`
- review document text extraction: `/home/galoren/roofline_ICSE/_review_tmp.md`
- benchmark website export: `/home/galoren/roofline_ICSE/_website_mirror/site-data.json`
- benchmark repository: `/home/galoren/roofline_ICSE/SC26FLOPBench/`

## Optional Database-Backed Analysis

The paper build does not require PostgreSQL.
The optional feature-vote support script expects restored local databases and writes CSVs only:

```bash
python analysis/generate_feature_vote_artifacts.py \
  --gpuflops-db-uri postgresql://postgres@localhost:55432/gpuflops_db \
  --code-features-db-uri postgresql://postgres@localhost:55432/code_features_db
```

That script is only for regenerating the exploratory appendix support summaries in `generated/feature_vote_*.csv`.
