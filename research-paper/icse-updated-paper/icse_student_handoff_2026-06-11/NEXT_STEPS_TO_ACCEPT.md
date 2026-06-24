# Next Steps To Move Toward Accept

This is the prioritized remaining work after the artifact-backed rewrite.

## Highest-Value Remaining Work

## 1. Add Repeated-Query Robustness For The Closed Models

Status:

- not done
- requires API access and budget, not more local GPU work

Why it matters:

- this is the biggest remaining methods objection
- it would make the source-vs-SASS gains and the model ordering much easier to trust

Recommended minimum:

- run `3` to `5` repeated queries on a matched subset
- keep the subset balanced across runtime, GPU, precision, and prompt setting
- report variance and whether the key claims remain stable

## 2. Add A Stronger External Or Compiler-Grade Static Baseline

Status:

- not done
- requires new engineering beyond the current artifact-only rewrite

Why it matters:

- the current static comparisons are credible, but still lightweight
- a stronger baseline would improve reviewer confidence that the paper is not comparing only against weak alternatives

Recommended target:

- one serious execution-free baseline family beyond the current lexical/mnemonic/tree references

## 3. Human-Validate The Feature Labels

Status:

- not done
- requires human annotation

Why it matters:

- the paper already treats the feature analysis as exploratory
- human validation would make that section much safer and potentially promotable again

Recommended target:

- validate a representative subset
- report agreement or disagreement explicitly
- keep the section scoped unless the validation is strong

## 4. Optional Workflow Validation

Status:

- not required for a resubmission
- would help if feasible

Why it matters:

- the current practical-value evidence is an offline queueing simulation
- a simple developer/agent workflow study would strengthen the software-engineering contribution further

## What Can Be Reused Immediately

- The current ICSE manuscript structure is already strong enough to keep.
- The static-baseline, budget-sweep, and paired-statistics pipeline should be reused exactly unless new experiments require extension.
- The claim discipline should be preserved. Do not broaden the claims while adding experiments.

## Practical Integration Checklist For The Student

1. Read `CURRENT_ICSE_JUDGMENT.md`, `NEW_VS_ORIGINAL_GITHUB.md`, and `icse_resubmission/FINAL_REPORT.md`.
2. Review `icse_resubmission/review_action_trace.md` before touching the manuscript.
3. Rebuild the current paper first and confirm it matches `icse_resubmission/main.pdf`.
4. Decide which new experiments are realistically possible before editing the narrative.
5. If new experiments are added, rerun the manuscript-side analysis scripts and update every dependent table and claim.
6. Re-run a full PDF inspection loop after any content change.

## What Not To Undo

- Do not revert the ICSE framing back toward an HPC-only paper.
- Do not remove the stronger threats/limitations language.
- Do not overclaim profiler replacement, runtime prediction, or general deployment utility.
- Do not hide where static baselines already win. That honesty is currently helping the paper.
