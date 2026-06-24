# Reframing Plan: Turning the Existing Data into an Acceptance-Quality Paper

## Context

The current `fixup-ICSE` draft is framed as *"can off-the-shelf LLMs predict RAI?"* The honest answer in
the data is "mostly no" (best config is within 25% on only ~7–17% of samples), so the paper reads as
positive spin on a negative result — the same trap that sank the SC26 submission. We are **not** running a
large new experimental campaign. The goal is to reframe around the parts of the existing data that are
genuinely strong, defang the cost objection with bookkeeping we already have, and add only a few cheap
targeted experiments.

Note on SASS (settled): cross-compilation (`nvcc -arch=sm_90` + `cuobjdump -sass`) produces target SASS on
any machine **without** the target GPU. So `source+SASS` is legitimately hardware-free, and the FP16
source→SASS result is a real no-hardware finding, not a motivation violation. The only residual SASS-related
concern is pure cost, which is handled in §4.

## The new thesis

Change what the paper is *about* so the strong numbers become the headline:

> **Source code alone is insufficient for an LLM to reason about a GPU kernel's Roofline regime, because the
> decisive arithmetic is often compiler-synthesized. Cross-compiled SASS — obtainable without target
> hardware — recovers it, lifting off-the-shelf LLM triage from chance to near-perfect for half precision.
> We characterize when each evidence tier is necessary, and the cost/accuracy tiering this implies for
> AI-assisted GPU tooling.**

Three pillars, all supported by existing data:

1. **Classification, not regression, is the deliverable.** Roofline-regime (bandwidth- vs compute-bound)
   triage is the actionable developer question and is where the numbers are strong:
   - source-only FP32: **0.940 BalAcc (Opus)**, 0.889 (GPT-5.4), 0.875 (GPT-OSS)
   - source+SASS FP16/FP32: **0.984 / 0.967 (Opus)**, 0.870 FP16 (GPT-5.4)
   Numeric RAI becomes a *diagnostic* that explains why classification succeeds or fails — not a claim we
   defend on a 25% threshold.

2. **"What does post-compilation evidence add?" is the scientific question.** The cleanest result is FP16
   source→SASS: **0.500 → 0.984**, with a mechanistic explanation (mixed-precision FLOPs are compiler-injected
   `HFMA2`, invisible in source, recovered by disassembly). Failure regimes (source-only FP16) become
   findings — "compilation is necessary, not optional, for this class of reasoning" — rather than weaknesses.

3. **The benchmark is the durable artifact.** 95 programs / 254 kernels / 1,016 profiled samples × 4 GPUs,
   with source + build context + runtime args + SASS + profiler ground truth + model outputs. A
   benchmark/empirical paper is allowed to report that the obvious method partially fails.

## Section-by-section restructure

| Section | Becomes |
| --- | --- |
| Title/Abstract | Lead with the evidence-tier finding and classification framing. Headline numbers = BalAcc, not within-25%. |
| Introduction | SE problem = agents need a cheap pre-profiling Roofline-regime signal. RQ shift to: (RQ1) can LLMs classify BB/CB from source? (RQ2) what does cross-compiled SASS add, and why? (RQ3) cost/accuracy tiering vs static baselines and vs profiling. |
| Related work | Keep; sharpen the delta vs Bolet'25 (off-the-shelf + the source-vs-SASS evidence question, not just BB/CB). |
| Task/Method | Keep decomposition (it's how RAI is derived) but state classification as the primary output, numeric RAI as diagnostic. Keep first-invocation/DRAM scope. |
| Results | Restructure around: (1) classification accuracy by tier, (2) the source→SASS mechanism with the HFMA2 case, (3) cost/accuracy tiering, (4) where static baselines win. Demote the within-25% success-envelope table to support. |
| Discussion | Cost/accuracy tiering for agent tooling; the cheap-model source-only triage story. Keep cascade but downgrade to "hypothesis," not validated system. |
| Threats | Keep; add the contamination-probe result (§5) once run. |
| Conclusion | Explicitly restate + answer the (now-fewer) RQs (closes SC R2's ask). |

## What gets demoted or cut

- **Demote** the numeric within-25%/MedAPE success-envelope framing from headline to a diagnostic table.
- **Downgrade** the deployment-matrix/cascade from "recommendation" to "hypothesis"; do not claim it as a
  validated system (it is never built end-to-end, and regime routing needs post-hoc knowledge).
- **Keep but reposition** the static/learned baselines as the "is the LLM worth it?" comparison, honestly
  noting where the free SASS-mnemonic / learned baselines match or beat the LLM (FP64 triage).

## The cost story (from existing data — no new runs)

This is the rebuttal to the repeated SC "why not just rent a GPU" objection, and we already have the numbers:

- The cost objection only bites for Opus+SASS (\$0.1066/query).
- **GPT-OSS does source-only FP32 triage at 0.875 BalAcc for \$0.0006/query** → ~**\$0.44 to triage the whole
  732-sample shared subset**, unambiguously cheaper than profiling it 3× on four GPUs.
- Message = a **cost/accuracy tier**: cheap models give useful source-only BB/CB triage at ~\$0.001–0.02 per
  kernel (orders of magnitude below a profiling run); frontier+SASS is the premium tier that additionally
  cracks FP16.
- **Action:** add an explicit comparison subsection: cloud \$/GPU-hour × measured profiling wall-clock for the
  corpus vs. total API spend per tier. Pure bookkeeping from existing logs.

## Deferred experiments (do NOT run now — write with existing data, leave manuscript placeholders)

**Decision:** This pass writes the paper entirely from the data we already have. The variance and
contamination experiments are **deferred**. Do not run them now; instead **stub them in the manuscript** so
results can be dropped in later:
- A "Robustness to repeated queries" subsection + placeholder table (CI columns marked TBD), with a one-line
  note that variance characterization is pending.
- A "Reasoning vs. memorization probe" subsection describing the planned cosmetic/semantic perturbation
  design (per the sampling protocol below), marked as planned future evidence.
- Until filled, keep single-query nondeterminism and contamination as explicit, honest threats.

The ranked list below is the reference for when we *do* run them later (ranked by ROI):

1. **Repeated-query variance — on the cheap model.** Run GPT-OSS 5–10× on a stratified ~50-kernel subset to
   put CIs on the classification numbers; kills the R2/R3 nondeterminism complaint. ~single-digit dollars.
   Add a *small* Opus repeat (~20–30 samples × 5) only to bound frontier-tier variance. Do **not** re-run
   Opus broadly.
2. **Contamination / reasoning probe — highest scientific ROI.** ~30–50 kernels with (a) semantics-changing
   perturbations (alter loop bounds, array sizes, literal constants) and (b) cosmetic ones (rename
   identifiers, reorder functions). Predictions tracking semantic changes but invariant to cosmetic ones =
   evidence of reasoning over recall; defends the "source-visible FP32 reasoning" claim against the
   "memorized HeCBench" attack. Mostly cheap-model queries + a few Opus checks. ~\$20–50.
3. **One runnable prior-art baseline** (R2's ask) — pick a single static/analytical predictor we can actually
   run, or argue concretely why each cited one is inapplicable to per-kernel DRAM-level RAI. Higher effort;
   only if a reviewer-credible option exists.

**Do not spend on:** broad Opus re-runs across all 1,016 samples, more GPUs, AMD/SYCL, or another frontier
model generation.

## Sampling protocol for the ~50-kernel additional runs

Treat the variance and contamination studies as **two overlapping panels with different selection logic**.
Selecting only low-error kernels (the tempting default) biases the variance study and gives a one-sided
contamination test. The per-kernel predictions in `generated/static_prediction_rows.csv` / `site-data.json`
support a scripted stratified draw.

### Variance panel (~30–40 kernels) — representative, NOT low-error
Goal: honest CIs on the headline classification numbers. Low-error-only selection underestimates variance
and is an obvious selection bias. Stratify across:
- **Precision:** FP16/FP32/FP64 all present (FP16 nonzero is concentrated on A100/H100 — include those).
- **Error bucket** (from existing per-kernel APE): ~equal thirds of near-exact (<10%), moderate (10–50%), tail (>50%).
- **Balance-point proximity:** over-sample kernels whose predicted RAI sits *near* the GPU balance point —
  these flip BB/CB run-to-run and dominate the variance on BalAcc.
- **Class:** force in the rare compute-bound cases (only 34 FP16 / 50 FP32 / 53 FP64 CB in the shared subset).
- **Runtime:** CUDA and OpenMP; include some zero-GT cases (does it flip zero↔nonzero across runs?).
- **Model:** repeats primarily on GPT-OSS (~$0.0006/query); a small Opus subset to bound the frontier tier.

### Contamination / reasoning panel (~20–30 kernels, may overlap)
Goal: separate reasoning from memorization of a public benchmark. Here the low-error instinct applies —
include the near-exact / "famous" kernels (boxfilter, asmooth, burger, accuracy, plus canonical algorithms
like matmul/stencil/FFT/reduction) because those are where memorization is the competing explanation. Tag
every kernel **famous vs. obscure** (the split is itself a contamination signal) and include a few
**high-error** kernels as controls. Use two perturbation types with opposite expectations:
- **Cosmetic (invariance test):** rename kernel/variables, reorder functions, reformat. Reasoning ⇒
  prediction ~unchanged; surface-string memorization ⇒ it degrades.
- **Semantic (tracking test) — decisive:** change loop trip counts / array dims / grid size / problem-size
  CLI args by a *known factor*. Reasoning ⇒ prediction **tracks the new ground truth** in direction and
  rough magnitude; memorization ⇒ it stays **anchored to the original**. Report correlation between
  predicted-Δ and true-Δ, not just absolute error.

Cost caveat: the semantic test needs the perturbed variant's true RAI. Either re-profile perturbed variants
on the local **RTX 3090** (single GPU is fine — this probes the model, not cross-GPU generality), or derive
expected RAI analytically for source-legible kernels (e.g., doubling trip count ≈ doubles FLOPs at ~constant
bytes).

## Prior-art baseline: structured non-applicability argument (in lieu of running one)

R2 asked for comparison to prior art. Arguing non-applicability is viable **only** as a per-category
argument, and it hinges on the Category-2 linchpin. Verified targets of the cited works:

| Cited work | What it predicts | Why not directly comparable |
| --- | --- | --- |
| hong'09, baghsorkhi'10 | execution time (analytical, CUDA) | Different target; repurposing = our reimplementation, a strawman. |
| alavani'18 | execution time from PTX | Different target; PTX pipeline, old archs. |
| guerreiro'19 | exec time/power/energy under DVFS (PTX+RNN) | Different target; Maxwell–Turing; no retargetable artifact. |
| braun'21 | execution time + power (RF) | Different target; **but the learned-RF class = our RF/ET baseline.** |
| CUDAsap'23 | basic-block execution frequencies (static CFG) | Outputs exec statistics, not memory bytes / RAI. |
| tran'25 | latency for Triton kernels (XGBoost+PTX) | Different target; **learned class = our tree baseline.** |
| LLMPerf'24 | OpenCL kernel cost (fine-tuned LLM, MAPE 24–46%) | Fine-tuned; different target; off-the-shelf is orthogonal. |
| Omniwise'25 | mem BW, cache hit, GFLOPs, **arithmetic intensity** (fine-tuned LLM, AMD MI250/MI300X) | Closest neighbor; **confirmed: AMD-only, not NVIDIA-retargetable** — handle head-on (below). |
| Bolet'25 (own) | BB/CB classification of parallel code | Precursor this work generalizes. |

Argument structure:
1. **Category 1 (target mismatch, 8/10):** none predict per-precision DRAM-level RAI; comparison requires
   re-purposing them to an unvalidated target, so any result reflects our reimplementation, not the
   published method. Reinforce with: PTX/fitted-feature inputs outside our pipeline, old-arch-only with no
   retargetable Ampere/Hopper artifact, CUDA/OpenCL-only (can't touch the OpenMP-offload half of the corpus).
2. **Category 2 — the linchpin:** braun'21 / tran'25 / guerreiro'19 are learned tabular/sequence models over
   compiler-IR features — exactly the class our grouped-CV **RF/ET baselines already instantiate**, retargeted
   to DRAM-level RAI. State explicitly that we compared to the learned-predictor class via a faithful
   reimplementation appropriate to the target, rather than mis-running an artifact built for a different one.
   This is what makes non-applicability credible rather than evasive.
3. **Category 3 — fine-tuned LLMs; do NOT skip:** LLMPerf'24 and especially **Omniwise'25** (predicts AI,
   execution-free, >90% within 10%, AMD MI250/MI300X). Confront it: it is **fine-tuned** (our question is the
   zero-training/off-the-shelf regime — orthogonal), targets a different ISA with no NVIDIA-retargetable
   release, and likely uses a different AI definition than first-invocation DRAM-level per-precision RAI.
   Frame as the complementary zero-training counterpart; cite its 90%@10% as motivation. **Confirmed:
   Omniwise is AMD-only (MI250/MI300X) and not NVIDIA-retargetable**, so it cannot be run on this corpus —
   that single fact carries the argument and should be stated plainly in related work.
4. **Category 4 — own precursor:** Bolet'25 generalized from coarse BB/CB to quantitative decomposed RAI +
   the source-vs-SASS evidence question; frame as continuity, keep third-person for anonymity, optionally add
   a qualitative BB/CB comparison if setups align.

## Venue

- Primary: **ICSE / FSE** empirical or AI-for-SE track ("what evidence do off-the-shelf LLMs need to reason
  about GPU performance, and the cost/accuracy tiering for agent tooling").
- Fallbacks: **MSR** (natural fit for the benchmark/dataset+mining angle), or **EMSE / TOSEM** (journals
  tolerant of characterization/negative results).

## Execution checklist

### Now — existing data only
- [ ] Rewrite abstract + intro around the classification/evidence thesis; new RQ list.
- [ ] Restructure results: classification-first; HFMA2 source→SASS mechanism as centerpiece.
- [ ] Demote within-25%/MedAPE to diagnostic; downgrade cascade to hypothesis.
- [ ] Add cost-comparison subsection (bookkeeping from existing logs).
- [ ] Write the per-category prior-art non-applicability argument (Omniwise confirmed AMD-only, not NVIDIA-retargetable).
- [ ] Restate + answer RQs in conclusion.
- [ ] Add placeholder subsections + stub tables for the two deferred studies (see below).

### Later — deferred runs (leave placeholders now; do not run this pass)
- [ ] Run cheap variance experiment on the stratified (representative) panel; fill CIs into the placeholder table.
- [ ] Run contamination/reasoning probe (cosmetic + semantic perturbations) on the famous/low-error panel; fill in results + tighten threats.
