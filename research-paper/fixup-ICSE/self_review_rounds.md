# Self-Review Rounds

## Round 1: Contribution and ICSE Fit Review

### Focus

- Is the software-engineering problem explicit in the first page?
- Is the paper clearly positioned for ICSE rather than only for HPC/performance venues?
- Are the contributions concrete and aligned with the evidence?

### Issues found

- A few sentences still sounded like an internal revision memo rather than a finished submission.
- The contribution wording was still slightly defensive in places.
- The study-design section still used internal artifact jargon (`no-IMIX`) that would be opaque to reviewers.

### Fixes made

- Rewrote the contribution framing in the introduction to state the software-engineering contribution directly.
- Removed meta references to the rewrite, the previous version, and the review itself.
- Replaced internal prompt-configuration jargon with plain source-only/source+SASS wording.

### Outcome

- The paper now opens as an ICSE paper about execution-free performance reasoning for AI-assisted GPU development.

## Round 2: Evidence, Validity, and No-Fabrication Review

### Focus

- Does every reported number match the released artifact?
- Are task boundaries and class-imbalance effects handled explicitly?
- Are unsupported claims or hidden assumptions removed?

### Issues found

- The study design defined four tasks, but the results section mainly foregrounded only regression and nonzero BB/CB triage.
- The conservative treatment of non-finite predictions in zero-aware tasks was not yet explicit.
- The profiler-counter paragraph was accurate but visually brittle and harder to audit.

### Fixes made

- Added explicit zero/nonzero and three-way triage results to RQ1.
- Documented the conservative handling of non-finite predictions in Task A and Task D.
- Reworked the ground-truth paragraph to list the exact Nsight Compute counters with line-break-safe formatting.
- Kept the baseline additions limited to the baselines that could be computed faithfully from the released artifact.

### Outcome

- The paper now matches the released evidence more closely and states its validity constraints more precisely.

## Round 3: Writing, Compactness, and Reviewer Readability Review

### Focus

- Is the prose compact and consistent?
- Are section transitions and captions carrying the argument clearly?
- Does the project satisfy the ACM review-template requirements cleanly?

### Issues found

- The draft was missing ACM CCS metadata and image descriptions.
- A few phrases still leaned on internal context instead of reviewer-facing language.
- Appendix float placement still produced a minor float-placement warning.

### Fixes made

- Added CCS metadata and `\Description{...}` text for all figures.
- Standardized reviewer-facing language across introduction, results, discussion, and threats.
- Adjusted appendix float placement and reran the manuscript through the build/visual loop.

### Outcome

- The final source is compact, template-compliant, and reviewer-readable without relying on hidden revision context.

## Round 4: Final Resubmission Polish Review

### Focus

- Does the paper still read coherently after the added static-baseline and paired-uncertainty evidence?
- Are the appendix pages and delivery documents synchronized with the final manuscript state?

### Issues found

- The manuscript itself was consistent, but the package-level reports still reflected the pre-learned-baseline state.
- The revised appendix static-baseline table needed one more visual-legibility confirmation after the learned rows were added.

### Fixes made

- Rechecked the manuscript end to end against the regenerated baseline and paired-statistics outputs.
- Added a final PDF visual pass focused on the revised results and appendix pages.
- Updated the review trace, changelog, README, visual-check log, and final delivery report to match the final manuscript state.

### Outcome

- The paper and the submission package are now synchronized and polished for resubmission.

## Round 5: Comparison-Closure and Reproducibility Review

### Focus

- Does the paper now compare the LLM results against enough credible static alternatives?
- Are the remaining reproducibility complaints narrowed to items that truly require new external runs or human work?

### Issues found

- The baseline story was still too narrow: one learned family was not enough, and the paper still lacked a simple runtime$\times$GPU context baseline.
- The fixed-budget utility analysis still relied on a single 10% operating point and did not show where cheap static rankers already beat one-shot prompting.
- The study design still omitted the exact committed model identifiers and decode defaults that were recoverable from the released repository.

### Fixes made

- Added the grouped runtime$\times$GPU context-median baseline, a second lightweight tree family, and five-seed sensitivity summaries for the learned baselines.
- Added a 5/10/20/30% budget sweep against the strongest LLM and static methods in each evidence family and rewrote the results/discussion around the narrower, more honest comparison story.
- Documented the committed model identifiers, single-trial setting, and released OpenRouter decode defaults in the study-design section.

### Outcome

- The comparison question is now much better closed from the existing artifacts: the paper states clearly where static baselines already win, where frontier LLMs still add value, and which remaining reviewer requests truly need new closed-model runs or human annotation.

## Round 6: Remaining-Feasible-Work Review

### Focus

- After the baseline and utility additions, what reviewer-facing evidence was still feasible from the released artifacts alone?
- Could any remaining feature-analysis requests be completed cleanly in this environment without fabricating new infrastructure or partial results?

### Issues found

- The paper still lacked exact shared-subset class counts and class-conditional compute-bound precision/recall summaries, which made it harder for a reviewer to judge how much of the queueing signal came from class imbalance versus true compute-bound identification.
- The remaining feature-analysis requests (raw prevalence counts and multiple-comparison control) depended on reconstructing the PostgreSQL-backed feature-voting dump, but this environment does not currently provide `pg_restore`/`psql`, and Docker access is unavailable.

### Fixes made

- Added an appendix class-balance table, an appendix compute-bound precision/recall/F1 table for all six LLM configurations, and raw Task A/C/D confusion-count CSV exports for both the LLM rows and the static baselines.
- Rechecked the feature-analysis path, documented why the remaining feature-voting requests are not cleanly reproducible in the current toolchain, and kept the manuscript’s feature discussion explicitly descriptive.

### Outcome

- The remaining feasible comparison and utility evidence from the released prediction rows is now in the paper or artifact package.
- At the end of this round, the unresolved items were narrowed to repeated-query robustness, human validation, and the still-unreconstructed PostgreSQL-backed feature-voting requests.

## Round 7: Feature-Vote Reconstruction Review

### Focus

- Was the last still-feasible appendix request actually recoverable from the released PostgreSQL dumps?
- If yes, could it be added without promoting the exploratory feature analysis back into the paper's main contribution path?

### Issues found

- The package docs still claimed that raw feature-prevalence and multiple-comparison support could not be reconstructed locally.
- The appendix heatmap was already safely demoted, but it still lacked a reviewer-visible sanity check tying the repeated positive cells to actual kernel prevalence and adjusted significance.

### Fixes made

- Bootstrapped a private user-space PostgreSQL toolchain, restored both `gpuflops_db.dump` and `code_features_db.dump`, and reproduced the appendix heatmap numerically from the raw sample rows.
- Added `analysis/generate_feature_vote_artifacts.py` plus generated CSVs for kernel-prevalence counts and BH-corrected per-cell support.
- Added an appendix paragraph and discussion/threats updates so the feature analysis stays exploratory, but no longer depends only on a visual heatmap.

### Outcome

- The last artifact-only appendix request is now closed.
- The remaining risks are no longer about missing local analysis; they are repeated-query robustness and human validation of the feature labels.
