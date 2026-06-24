# ICSE Student Handoff

This bundle is the handoff package for the ICSE rewrite of the paper originally titled:

`Off-the-Shelf LLMs are Static Roofline Triage Tools for GPU Kernels`

The goal of this package is to let the student review everything that was done, understand what changed relative to the original GitHub project, rebuild the current ICSE submission, and continue the remaining work needed to move the paper from its current borderline-accept state into a safer accept range.

## What Is Included

- `icse_resubmission/`
  - the full current ICSE LaTeX project
  - the final ACM double-column PDF
  - generated tables/CSVs
  - analysis scripts added during the rewrite
  - review/planning/validation notes
- `original_inputs/`
  - the original Overleaf/source snapshot
  - the original source zip
  - the review DOCX
  - the extracted review text
  - the mirrored public website data used during the rewrite
- `NEW_VS_ORIGINAL_GITHUB.md`
  - what was newly produced outside the original GitHub repository
- `CURRENT_ICSE_JUDGMENT.md`
  - the current reviewer-style assessment
- `NEXT_STEPS_TO_ACCEPT.md`
  - the remaining high-value work to push the paper upward

## Suggested Reading Order

1. `CURRENT_ICSE_JUDGMENT.md`
2. `NEXT_STEPS_TO_ACCEPT.md`
3. `NEW_VS_ORIGINAL_GITHUB.md`
4. `icse_resubmission/FINAL_REPORT.md`
5. `icse_resubmission/review_action_trace.md`
6. `icse_resubmission/main.pdf`
7. `icse_resubmission/` source files and `analysis/` scripts

## Main Paths

- Current final PDF: `icse_resubmission/main.pdf`
- Current LaTeX source: `icse_resubmission/`
- Main build command:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## Note On The Original SC-Era PDF

The exact original PDF file named `SC26_LLMs4Roofline_Submission (5)(3).pdf` was not present in the workspace at handoff time.

What is included instead:

- the original LaTeX source tree in `original_inputs/_paper_src/`
- the original source zip in `original_inputs/roofline_overleaf.zip`

A local rebuild of the original source was attempted and got past the first missing-package issue, but it still fails in this environment because of an `xcolor` option clash in the older source stack. The source snapshot is therefore the authoritative “before” material included in this bundle.
