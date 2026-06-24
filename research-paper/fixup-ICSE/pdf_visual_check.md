# PDF Visual Check Log

## Pass 1

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass1/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass1/contact_sheet.png`
- Issues found:
  - No overlaps, cut-off figures, or unresolved references.
  - One reviewer-visible wording issue remained on page 6: the study-design section still used the internal term `no-IMIX`.
- Fixes made:
  - Rewrote the study-design sentence to describe the two prompt configurations directly as source-only and source+SASS.
- Another pass needed: Yes.

## Pass 2

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass2/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass2/contact_sheet.png`
- Issues found:
  - No remaining layout breakage.
  - Figures, tables, captions, and appendix plots were legible in the ACM review layout.
  - No missing citations, no cut-off floats, and no orphaned section headings were visible.
- Fixes made:
  - No content/layout fixes required after this pass.
- Another pass needed: Yes, one more confirmation pass was required by the workflow.

## Pass 3

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass3/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass3/contact_sheet.png`
- Issues found:
  - None beyond the known non-fatal underfull vbox warnings in the log; no visible artifact was associated with them.
  - The manuscript remained stable after the final rebuild.
- Fixes made:
  - None.
- Another pass needed: No.

## Post-Review Format Correction

- Trigger: a manual re-check found that the earlier PDF was still rendering as a single-column review manuscript instead of the intended ACM `sigconf` conference layout.
- Root cause: `main.tex` was missing `\maketitle`, so `acmart` never executed the `sigconf` two-column topmatter path.
- Fix applied: inserted `\maketitle` and rebuilt the manuscript; then cleaned the most visible two-column overflow in the profiler-counter paragraph and rerendered the updated PDF.
- Verification:
  - rebuilt PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
  - rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/format_fix_updated/pages/`
  - contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/format_fix_updated/contact_sheet.png`
- Outcome:
  - The final PDF now uses ACM double-column conference layout throughout.
  - Title, abstract, figures, tables, references, and appendix pages were visually rechecked after the fix.

## Pass 4

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass4_budget_baselines/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass4_budget_baselines/contact_sheet.png`
- Issues found:
  - No overlaps, missing references, or cut-off floats after adding the fixed-budget profiling table and deterministic-baseline appendix table.
  - The appendix baseline table is smaller than the main-text tables, but it remains readable in the rendered PDF and does not overflow its page.
  - The log still contains non-fatal overfull/underfull warnings, but the inspected pages did not show corresponding visible defects.
- Fixes made:
  - No additional layout edits were required after visual inspection of the updated pages.
- Another pass needed: No.

## Pass 5

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass5_learned_baseline/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass5_learned_baseline/contact_sheet.png`
- Issues found:
  - No overlaps, cut-off figures, missing references, or unreadable captions after adding the lightweight learned baseline and paired-uncertainty revisions.
  - The appendix static-baseline table remains dense, but it is still legible in the rendered ACM layout and does not overflow its page.
  - The appendix plots and reference pages remain stable after the added table content.
- Fixes made:
  - No further manuscript-layout changes were required after the pass-5 inspection.
- Another pass needed: No.

## Pass 6

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass6_comparison_closure/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass6_comparison_closure/contact_sheet.png`
- Issues found:
  - No cut-off floats or missing references after adding the context-median baseline, extra-trees baseline, budget-sweep appendix table, and learned-baseline seed-sensitivity table.
  - The appendix static-baseline table remains dense but readable; the new budget-sweep and seed-sensitivity tables fit cleanly on pages 12 and 13.
  - Page 13 contains ordinary reference-page whitespace below the seed-sensitivity table, but no overlap or template defect.
- Fixes made:
  - No layout changes were required after the comparison-closure inspection.
- Another pass needed: Yes, one more spot check was needed after the final reproducibility sentence was added.

## Pass 7

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass7_model_settings/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass7_model_settings/contact_sheet.png`
- Issues found:
  - No new layout regressions after adding the exact committed model identifiers and decode defaults to the study-design section.
  - Page count remained stable at 13 and the appendix tables remained legible.
- Fixes made:
  - None.
- Another pass needed: No.

## Pass 8

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass8_class_balance_prf/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass8_class_balance_prf/contact_sheet.png`
- Issues found:
  - No unresolved references or cut-off floats after adding the appendix class-balance table and the compute-bound precision/recall table.
  - Page 12 is dense because it now carries the GPU-stratified plot plus two appendix tables, but the rendered PDF remains legible and there is no overlap.
  - Page 14 contains ordinary whitespace below the final seed-sensitivity table, but no layout defect or template violation is visible.
- Fixes made:
  - No further layout edits were required after this pass.
- Another pass needed: No.

## Pass 9

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass9_regime_rewrite/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass9_regime_rewrite/contact_sheet.png`
- Issues found:
  - No overlap, cut-off text, or missing references after the regime-comparison rewrite, illustrative-row appendix table, and feature-map demotion.
  - Page 13 is dense because it now carries the static-baseline, budget-sweep, seed-sensitivity, and illustrative-row tables above the references, but the rendered ACM PDF remains legible.
  - Page 14 remains visually stable after expanding the Figure 7 caption with the feature-prevalence and BH-corrected support summary.
- Fixes made:
  - No further layout edits were required after this pass.
- Another pass needed: No.

## Pass 10

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass10_accept_push/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass10_accept_push/contact_sheet.png`
- Issues found:
  - No overlap, cut-off text, or unresolved references after promoting the best-LLM-versus-best-static table into the main results section and adding the one-column deployment-routing summary in the discussion.
  - The promoted Table 3 remains legible at the top of page 8, and the deployment matrix fits cleanly in one column on page 9 without forcing a page-count increase.
  - Bibliography warnings from cited entries were eliminated; the references now render without the earlier `n.d.` HeCBench artifact.
- Fixes made:
  - Cleaned the cited BibTeX entries that were still missing publisher/address metadata and removed the stale HeCBench repository citation from the benchmark-corpus paragraph.
- Another pass needed: No.

## Pass 11

- Compile command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF path: `/home/galoren/roofline_ICSE/icse_resubmission/main.pdf`
- Rendered image path: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass11_polish/pages/`
- Contact sheet: `/home/galoren/roofline_ICSE/icse_resubmission/visual_checks/pass11_polish/contact_sheet.png`
- Issues found:
  - The PDF was already structurally sound, but a few narrow-column paragraphs on the title page and in the methods/results discussion still produced reviewer-visible stretching pressure in the ACM layout.
  - A more aggressive text rewrite reduced those warnings but pushed the package to 15 pages, which would be worse for submission readiness.
- Fixes made:
  - Kept the page budget at 14 pages, retained the earlier content edits that improved first-page flow, and added a small `\emergencystretch=1em` adjustment in `main.tex` so TeX can absorb the remaining narrow-column pressure more gracefully.
  - Rebuilt and visually rechecked the updated PDF to confirm no new float, caption, or page-break regressions.
- Another pass needed: No.
