# Final Report

## Deliverables

1. Final PDF: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
2. Final LaTeX project: `/home/galoren/roofline_ICSE/icse_resubmission/`
3. Build command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## Major Changes

- Rebuilt the manuscript as a clean anonymous ACM/ICSE submission project.
- Corrected a layout bug in the final template wiring: `\maketitle` had been omitted, which left the earlier PDF in single-column review layout despite the `sigconf` class option. The final PDF now renders in ACM double-column conference format.
- Repositioned the paper around execution-free performance reasoning for AI-assisted GPU software engineering instead of profiler-style HPC framing.
- Rewrote the abstract, introduction, related work, methodology, results, discussion, threats, and conclusion to emphasize software artifacts, early triage, and disciplined claims.
- Added a shared-subset aggregate results table, explicit task decomposition, stronger practical implications, and a more explicit limitations section.
- Added a fixed-budget profiling-queue analysis that shows how many true compute-bound rows each configuration can surface under a 10% profiling budget.
- Added a grouped runtime$\times$GPU context-median baseline, deterministic source-lexical and SASS-mnemonic heuristics, lightweight grouped random-forest/extra-trees baselines, a 5/10/20/30% budget sweep, shared-subset class-balance tables, compute-bound precision/recall summaries, raw confusion-count exports, and learned-baseline seed-sensitivity summaries generated directly from the local released artifacts.
- Added paired source-vs-SASS significance checks plus bootstrap confidence intervals, and documented the committed model identifiers plus the released decode defaults used by the direct-prompting pipeline.
- Replaced the old main-text failure-analysis RQ with a stronger best-LLM-versus-best-static regime comparison, promoted that regime table into the main results section, added a compact deployment-routing matrix for tool builders, moved the exploratory feature heatmap to the appendix, added illustrative released-row case studies, and added a BH-corrected feature-prevalence sanity check recovered from the restored PostgreSQL dumps.
- Added `analysis/generate_additional_artifacts.py` and the generated CSV/table outputs so the new evidence is reproducible from the workspace mirrors.
- Added `analysis/generate_feature_vote_artifacts.py` plus the generated appendix feature-vote support CSVs.
- Preserved the original empirical basis and figures where useful, while redrawing the argument around faithful released-artifact evidence.
- Cleaned the cited bibliography entries so the ACM reference list no longer emits the earlier HeCBench `n.d.` artifact or empty publisher/address warnings for the cited IEEE proceedings.

## Review-Document Issues Addressed

- Addressed the ICSE-framing problem by centering the developer/agent workflow and scarce profiling-budget motivation.
- Addressed the overclaim problem by narrowing the paper to execution-free early triage rather than profiler replacement or general runtime prediction.
- Addressed the task-mixing problem by separating zero/nonzero detection, nonzero regression, nonzero BB/CB triage, and three-way triage.
- Addressed the hardware-free imprecision by distinguishing source-only from source+SASS and using “execution-free at prediction time.”
- Addressed the ground-truth precision issue by listing the exact Nsight counters and the first-invocation profiling setup.
- Addressed the unvalidated-feature issue partially by treating the feature analysis as exploratory only.
- Addressed the remaining feature-prevalence/multiple-comparison complaint as far as the released dumps allow by restoring the PostgreSQL artifacts and exporting kernel-prevalence plus BH-corrected per-cell support summaries for the appendix heatmap.
- Addressed the baseline criticism by adding trivial baselines, a grouped runtime$\times$GPU context baseline, deterministic source-only/SASS heuristics, and lightweight learned static baselines from the released artifact.
- Addressed the paired-evidence criticism more directly by adding exact paired tests and paired bootstrap intervals for the source-versus-SASS deltas.
- Addressed the raw-count/class-conditional-metric criticism by adding an appendix shared-subset class-balance table, an appendix compute-bound precision/recall/F1 table, and generated Task A/C/D confusion-count CSVs.
- Addressed the “why should a developer care?” problem more directly by adding a fixed-budget profiling simulation tied to scarce profiling queues, extending it to a 5/10/20/30% budget sweep against the strongest static references, and adding a compact deployment-routing summary for tool builders in the discussion.
- Addressed part of the reproducibility complaint by documenting the committed model identifiers, single-trial configuration, and released decode defaults used by the querying pipeline.
- Addressed the “make the comparison story easier to see” problem by promoting the best-LLM-versus-best-static regime table into the main results section and cleaning the cited bibliography noise that would otherwise distract reviewers.

## Issues Not Fully Addressed by This Rewrite

- No remaining gap is blocked specifically by missing GPU hardware.
- The unresolved issues instead require new work outside the faithful-rewrite scope:
  - repeated LLM-query robustness runs,
  - human validation of the feature-labeling pipeline,
  - stronger compiler-grade or externally trained baselines beyond the lightweight grouped-CV references added here.
- The local machine is an RTX 3090, not one of the paper's original 3080/A10/A100/H100 rows, so I did not merge new hardware-dependent measurements into the paper's core claims.

## Self-Review Summary

- Round 1 improved ICSE fit, contribution framing, and reviewer-facing terminology.
- Round 2 tightened evidence presentation, made the task split explicit, and audited claims against the released artifact.
- Round 3 cleaned template/readability issues, added accessibility metadata, and removed residual revision-memo phrasing.
- Round 4 revalidated the revised static-baseline story, reran the artifact-generation checks, and synchronized the submission package with the final manuscript state.
- Round 5 closed the remaining artifact-only comparison gaps with additional static baselines, budget sweeps, seed-sensitivity checks, and explicit direct-prompting configuration details.
- Round 6 added the remaining feasible class-balance and compute-bound precision/recall evidence and re-audited the still-blocked feature-voting requests.
- Round 7 restored the PostgreSQL dumps, reproduced the appendix feature map numerically, and added kernel-prevalence plus BH-corrected support summaries without promoting the exploratory analysis back into the main contribution path.

Detailed log: `self_review_rounds.md`

## PDF Visual Inspection Summary

- Pass 1 found one reviewer-visible wording issue (`no-IMIX`) and no layout breakage.
- Pass 2 confirmed that figures, tables, captions, and page flow were visually sound after the wording fix.
- Pass 3 confirmed final build stability and no visible PDF defects.
- A final post-review format-correction pass fixed the missing-`\maketitle` bug and revalidated the resulting ACM double-column PDF.
- Pass 4 revalidated the manuscript after adding the deterministic-baseline appendix table and the fixed-budget profiling table.
- Pass 5 revalidated the manuscript after adding the lightweight learned baseline and paired-uncertainty revisions.
- Pass 6 revalidated the manuscript after adding the context baseline, extra-trees baseline, budget-sweep appendix table, and seed-sensitivity appendix table.
- Pass 7 confirmed no layout regressions after adding the committed model identifiers and decode defaults.
- Pass 8 confirmed that the added class-balance and compute-bound precision/recall appendix tables fit cleanly and that the main-text references to them resolve correctly.
- Pass 9 revalidated the manuscript after the regime-comparison rewrite, the appendix-only feature-map demotion, the illustrative released-row table, and the expanded Figure 7 support caption.
- Pass 10 revalidated the manuscript after promoting the main comparison table, adding the deployment-routing matrix, and cleaning the cited bibliography metadata.
- Pass 11 applied a final submission-polish pass that tightened first-page flow and added a small TeX stretch allowance while preserving the 14-page layout.

Detailed log: `pdf_visual_check.md`

## Remaining Submission Risks

- The paper is now much better aligned with ICSE expectations, but it still does not include repeated-query robustness data.
- The new static references are still lightweight compared with compiler-grade analyzers or externally trained predictors.
- The feature/error heatmap remains exploratory because the feature labels are not human-validated.
- The final page has ordinary reference-page whitespace; it is not a layout error, but it reflects the current paper length rather than a perfectly filled final page.

## Final Validation Checklist

- [x] New ICSE project created.
- [x] Correct ICSE-style ACM review template used.
- [x] Paper compiles without fatal errors.
- [x] Bibliography compiles.
- [x] No unresolved citations.
- [x] No unresolved references.
- [x] No LaTeX TODOs remain.
- [x] No anonymous-review violations unless explicitly documented.
- [x] All figures/tables fit.
- [x] Three self-review rounds completed.
- [x] Three visual PDF passes completed.
- [x] Review DOCX action items traced.
- [x] Claims are faithful to existing data.
- [x] No fabricated experiments/results.
- [x] Final PDF produced.
- [x] Final source project ready.
