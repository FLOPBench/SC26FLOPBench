# Original Inputs

This folder contains the original materials that fed the ICSE rewrite.

## Included

- `_paper_src/`
  - original SC-era LaTeX source snapshot found in the workspace
- `roofline_overleaf.zip`
  - original source zip found in the workspace
- `ICSE_review_LLMs_Roofline.docx`
  - the review/decision-preparation document used as the rewrite checklist
- `_review_tmp.md`
  - extracted text version used during review-action tracing
- `_website_mirror/`
  - local mirror of the public benchmark website data used during the rewrite

## Original PDF Status

The exact original submission PDF file was not present in the workspace under the expected filename.

A local rebuild of `_paper_src/` was attempted. It progressed after installing the missing `algorithms` TeX package, but still fails in this environment because of an `xcolor` option clash in the older source stack. For that reason, the source snapshot and zip are the preserved original-paper materials included here.
