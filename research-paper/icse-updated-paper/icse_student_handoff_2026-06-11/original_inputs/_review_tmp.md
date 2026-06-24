**ICSE Review Draft**

*Paper: "Off-the-Shelf LLMs are Static Roofline Triage Tools for GPU
Kernels"*

Recommended verdict, scores, detailed justification, and revision
roadmap

<table>
<tbody>
<tr class="odd">
<td><p><strong>Bottom line for the reviewer</strong></p>
<p>Recommended verdict: Weak Reject.</p>
<p>Reason: the paper has a promising and timely idea, a nontrivial benchmark, and readable presentation, but the current evidence is not yet strong enough for ICSE because the evaluation lacks essential baselines, uncertainty/replicability analysis, and a sufficiently validated link between the measured metric and actionable software-engineering utility.</p>
<p>This is not a Strong Reject: the topic is real, the dataset/evaluation scale is meaningful, and the paper could become acceptable after substantial revision. It is also not a Weak Accept: several central claims require additional experiments or reframing, not merely camera-ready edits.</p></td>
</tr>
</tbody>
</table>

# 1\. Recommended ICSE-style ratings

Use the exact field names in the review system if they differ; the
values below are intended to map to the usual ICSE-style categories.

| **Field**                   | **Recommended rating**                                      | **Rationale**                                                                                                                                                                                                               |
| --------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overall recommendation      | Weak Reject                                                 | Promising idea and sizable evaluation, but major methodological and evidentiary gaps remain.                                                                                                                                |
| Numerical score if required | \-1 on a -3..+3 scale; or 2/5 on a 1..5 acceptability scale | Equivalent to reject with encouragement to resubmit after substantial revision.                                                                                                                                             |
| Reviewer confidence         | 4/5                                                         | I can assess the performance-engineering/LLM-evaluation aspects with reasonable confidence; exact system form may differ.                                                                                                   |
| Soundness                   | 2/5                                                         | The core experimental setup is plausible, but the absence of baselines, repetition, uncertainty estimates, and validated feature labels weakens the claims.                                                                 |
| Significance / impact       | 3/5                                                         | The problem is important for LLM-assisted performance engineering, but current evidence stops at offline prediction rather than developer or optimizer utility.                                                             |
| Novelty                     | 3/5                                                         | Quantitative RAI prediction with off-the-shelf LLMs is a useful angle, but prior work on static GPU prediction and LLM performance prediction makes novelty incremental unless the benchmark and analysis are strengthened. |
| Presentation                | 3/5                                                         | Generally readable and well organized, with useful figures, but several definitions, metrics, and claims require sharper wording.                                                                                           |
| Reproducibility / artifact  | 2/5 now; potentially 4/5 with artifact                      | The paper promises a benchmark, but the review needs exact scripts, raw prompts/responses, Nsight metrics, model settings, versions, and released data to verify the results.                                               |
| Appropriateness for ICSE    | Borderline-to-relevant                                      | The work is relevant if framed as AI-assisted performance engineering for developers; currently it reads more like an HPC measurement paper than a software-engineering paper.                                              |

# 2\. Copy-ready overall review

The text in this section is written in a style suitable for pasting into
an author-visible review field.

## Summary of the paper

The paper studies whether off-the-shelf large language models can
predict a Roofline-relevant signal, Roofline Arithmetic Intensity (RAI),
for GPU kernels without task-specific training. Instead of asking the
model to output RAI directly, the authors ask it to predict launch
dimensions, FP16/FP32/FP64 operation counts, and DRAM bytes
read/written, and then derive arithmetic intensity. The evaluation uses
254 CUDA and OpenMP kernels from HeCBench on four NVIDIA GPUs (RTX 3080,
A10, A100, H100), compares source-only prompts to source+SASS prompts,
and reports numerical RAI error, Roofline-class agreement, cost/latency,
and an error-feature analysis. The main claims are that stronger LLMs
can provide useful early Roofline triage, that SASS helps especially for
FP16 and bound classification, and that a cheaper model remains
competitive in the source-only setting.

## Overall assessment

I recommend Weak Reject. The paper has a timely and potentially useful
idea: using LLMs as early static triage assistants for GPU performance
engineering, before full profiling is available. The paper is also well
motivated, has a sizable multi-GPU evaluation, and makes a reasonable
effort to analyze where the models fail. However, I do not think the
current version provides enough evidence for acceptance at ICSE. The
central problem is that the evaluation demonstrates that LLMs sometimes
correlate with profiled arithmetic intensity, but it does not yet
establish that this is the right tool, that the results are robust, or
that the method improves a software-engineering workflow relative to
simpler static baselines. Several claims also depend on single
stochastic model queries, an LLM-generated error-feature labeling
pipeline, and a dataset dominated in places by zero-RAI cases. These
issues are substantial and cannot be resolved by minor edits alone.

## Strengths

  - The problem is meaningful. GPU optimization workflows often depend
    on scarce profiling access, and an early triage signal for deciding
    whether a kernel is likely memory-bound or compute-bound would be
    practically valuable.

  - The paper asks a narrower and more interpretable question than
    general performance prediction: can models estimate the components
    of arithmetic intensity? This decomposition into launch
    configuration, FLOPs, and DRAM traffic is useful and makes failure
    analysis possible.

  - The evaluation is nontrivial in scale: 254 kernels, two programming
    models, four GPU architectures, and multiple prompt settings/models.
    This is much stronger than a purely anecdotal prompt-engineering
    paper.

  - The paper is mostly well written. The introduction clearly explains
    the hardware-access bottleneck, the figures communicate the pipeline
    effectively, and the related-work section acknowledges both static
    GPU-performance modeling and LLM-based performance-prediction work.

  - The comparison between source-only and source+SASS prompts is
    interesting. It is plausible that SASS exposes compiler-realized
    behavior that source-only prompts miss, and this distinction is
    important for future tools.

  - The promised benchmark could be a valuable contribution if the
    artifact is complete, anonymized, reproducible, and includes raw
    prompts/responses, profiling data, scripts, and metadata.

## Major weaknesses

  - **Missing non-LLM and simple baselines.** The main missing
    comparison is against simple static analysis baselines. Since the
    source+SASS setting supplies assembly, a rule-based SASS instruction
    counter, a source/AST-level operation counter, a regex/heuristic
    counter, a majority-class classifier, and simple per-runtime/per-GPU
    baselines are essential. Without these, it is unclear whether the
    LLM is doing something uniquely useful or merely approximating what
    a much cheaper deterministic analyzer could do more reliably.

  - **Robustness of LLM predictions is not established.** The paper
    states that expensive models are queried only once per kernel/GPU
    setting. If temperature, model versioning, tool-calling behavior, or
    nondeterminism are not controlled and repeated, the results may not
    be stable. A single query per item is particularly problematic when
    differences between source-only and source+SASS are used to support
    a central claim.

  - **The “useful triage” claim is stronger than what is measured.** The
    experiments measure RAI numerical error and class agreement, but do
    not evaluate whether developers or optimizer agents would actually
    make better decisions. For an ICSE paper, the software-engineering
    impact should be operationalized: for example, whether the method
    correctly prioritizes kernels for memory vs compute optimization,
    reduces wasted profiling time, or improves an LLM optimization loop
    under a fixed hardware budget.

  - **The definition and handling of RAI need to be tightened.** The
    paper computes separate FP16, FP32, and FP64 RAIs and then
    classifies each against precision-specific balance points. This can
    be reasonable, but it is not the standard way many readers think
    about operational intensity for mixed-precision kernels. The paper
    needs to explain how mixed precision is handled, why
    compiler-injected FP16 instructions should count as useful Roofline
    work, and how these per-precision classifications translate into
    actionable optimization advice.

  - **Zero-RAI cases risk distorting the conclusions.** The paper notes
    that most kernels have zero RAI. This raises two concerns: first, a
    large part of the dataset may not be representative of the intended
    floating-point Roofline task; second, classification performance may
    be inflated or dominated by a trivial zero/nonzero distinction. The
    paper partially excludes zero cases from percent-error plots, but
    the analysis needs a clearer separation between zero detection,
    nonzero RAI regression, and bound classification among nonzero
    kernels.

  - **The error-feature analysis is not sufficiently validated.** The
    paper uses a voting ensemble of inexpensive LLMs to label
    source-code features that are then correlated with LLM prediction
    errors. This is potentially circular and introduces another
    unvalidated model-based measurement layer. The feature labels need
    human validation, inter-rater agreement, raw prevalence counts,
    confidence intervals for Cliff’s Delta, and correction for multiple
    comparisons.

  - **The source+SASS setting complicates the hardware-free narrative.**
    The paper motivates the work as useful before hardware access, but
    source+SASS relies on target-specific compilation artifacts. That
    may still be hardware-free if compilation for the target
    architecture is possible without the device, but the paper must be
    precise about what resources are required and what is unavailable.
    It should distinguish “no runtime/profiling access” from “no
    compiler/toolchain/access to target architecture.”

  - **The profiling ground truth and memory-traffic assumptions require
    more precision.** The paper should list exact Nsight Compute
    metrics, units, conversion formulas, counter reliability concerns,
    replay effects, compiler/driver versions, and how first invocation
    is isolated. DRAM writes and cache write-back behavior are
    especially important because the prompt example assumes writes may
    remain in L2 and therefore not count as DRAM writes; this needs to
    be reconciled with the actual profiler metrics used as ground truth.

  - **ICSE framing is currently underdeveloped.** The paper is relevant
    to software engineering through LLM-assisted performance
    engineering, but it currently reads closer to an
    HPC/performance-modeling measurement paper. For ICSE, the paper
    should more directly articulate the developer task, the tool
    workflow, the cost/benefit tradeoff, and the concrete way this
    changes software engineering practice.

## Detailed comments by criterion

**Originality:** Moderate. The paper is not simply another LLM
code-generation paper; quantitative RAI prediction is a useful and
interpretable target. However, the novelty is reduced by substantial
prior work on static GPU performance prediction, learned GPU performance
prediction, and LLM-based GPU metric prediction. The paper needs to
sharpen what is uniquely learned here beyond “LLMs can be prompted for
another metric.”

**Technical soundness:** Currently borderline. The pipeline is
plausible, but the lack of static baselines, repetitions, uncertainty
estimates, and validated labels makes the evidence insufficient. The
strongest statements about SASS, model superiority, and architecture
effects should be treated as hypotheses unless supported by paired
statistical tests and robustness checks.

**Evaluation adequacy:** Not yet sufficient. The number of kernel-GPU
datapoints is good, but the evaluation design does not yet answer the
most important questions: does the method beat obvious baselines, does
it remain stable across LLM runs and model versions, does it help triage
decisions, and how often would it lead a developer to choose the wrong
optimization strategy?

**Reproducibility:** Potentially strong but not demonstrated in the
paper. A benchmark URL is promising, but review requires enough detail
to know whether the results can be reproduced: exact prompt templates,
raw outputs, parsing scripts, API settings, model dates, compiler
versions, driver versions, Nsight metrics, filtering logic, and complete
kernel metadata.

**Presentation:** Generally clear. The narrative is coherent and the
figures are useful. However, the results section would benefit from
compact aggregate tables with medians, interquartile ranges, balanced
accuracy, confidence intervals, and paired deltas. Several grammar
issues also recur, such as “it’s” where “its” is required, “divison,”
and minor singular/plural errors.

**Relevance to ICSE:** Relevant but needs reframing. The paper should
make the software-engineering contribution explicit: this is about
developer support, CI/IDE integration, agentic code optimization, and
efficient allocation of scarce profiling resources. A purely
hardware-performance framing may be seen as more appropriate for
SC/PACT/CGO unless the SE contribution is emphasized and evaluated.

## Questions for the authors

  - What deterministic static baselines does the LLM beat in the
    source-only and source+SASS settings?

  - What temperature and decoding settings were used? Are predictions
    stable if the same prompt is run multiple times?

  - How many kernels are zero-RAI for each runtime/GPU/precision, and
    what is performance on nonzero-only bound classification?

  - Which exact Nsight Compute metrics define FP16/FP32/FP64 operations
    and DRAM bytes read/written? How are replay and counter noise
    handled?

  - For mixed-precision kernels, what is the intended operational
    meaning of separate FP16, FP32, and FP64 RAI values?

  - Does source+SASS require only compilation for a target architecture,
    or did the workflow require access to each target GPU at any point
    before profiling?

  - How accurate are the LLM-generated feature labels used in the error
    analysis? Was there any human validation or inter-rater agreement?

  - What developer decision would change based on a predicted RAI, and
    how often would the predicted decision match the profiler-based
    decision?

## Why the verdict is Weak Reject rather than stronger or weaker

Why not Strong Reject: the paper addresses a real bottleneck, is
technically plausible, contains a reasonably large empirical study, and
could produce a useful benchmark. The weaknesses are serious but not
fatal to the research direction.

Why not Weak Accept: the missing evidence is central. A camera-ready
revision could improve wording and add clarifications, but it cannot
easily add the necessary baselines, robustness analysis, validated
labels, and developer-utility evaluation unless those results already
exist.

Why not Accept: ICSE acceptance should require convincing evidence that
the proposed technique is sound, useful, and better than appropriate
alternatives. The current paper does not yet meet that bar.

# 3\. Optional private note to the PC / meta-reviewer

<table>
<tbody>
<tr class="odd">
<td><p><strong>Private calibration note</strong></p>
<p>This paper is promising but premature for ICSE. My main concern is not presentation; it is evidentiary. The paper demonstrates an interesting correlation between LLM predictions and profiled RAI, but without baselines and robustness checks it is impossible to tell whether the proposed LLM-based method is a useful software-engineering contribution or simply an expensive substitute for deterministic static analysis. I would encourage resubmission after major revision, especially if the authors can release a strong artifact and evaluate actual triage utility.</p></td>
</tr>
</tbody>
</table>

# 4\. Required actions to make the paper fully acceptable

The following roadmap is deliberately detailed. The first tier contains
changes I would consider essential for acceptance at a top-tier ICSE
venue; the second tier contains changes that would substantially improve
the paper but may be less central.

## Tier 1: Essential changes

**Add strong baselines.**

  - Add a source-only static baseline that counts operations from the
    source/AST/LLVM IR or uses a simple code-feature model.

  - Add a SASS-only or source+SASS deterministic baseline that counts
    floating-point opcodes and memory instructions with
    architecture-specific rules. Even if imperfect for caches and helper
    functions, it is the most natural baseline for the source+SASS
    setting.

  - Add trivial and heuristic baselines: always-zero RAI, majority bound
    class, median RAI by runtime/GPU/precision, and simple
    launch-dimension propagation.

  - If feasible, add a lightweight learned baseline such as random
    forest/XGBoost over static source/SASS features. This helps separate
    “LLMs are useful” from “any statistical predictor works.”

  - Report where the LLM actually beats these baselines and where it
    does not. If the LLM mainly helps source-only cases but not SASS
    cases, state that precisely.

**Repeat enough LLM queries to quantify stochastic variability.**

  - State temperature, top-p, seed availability, tool-calling schema,
    and whether the models are deterministic.

  - Run at least 3-5 repetitions on the full dataset if cost allows;
    otherwise run repetitions on a stratified subset covering
    zero/nonzero RAI, FP16/FP32/FP64, runtime, GPU, and hard features.

  - Report median and variance across repetitions. A central result
    should not depend on one prompt draw.

  - If model version pinning is impossible, report model snapshot
    names/dates and discuss reproducibility limitations for closed APIs.

**Separate the evaluation into the right tasks.**

  - Task A: zero vs nonzero RAI detection.

  - Task B: numerical RAI regression on nonzero cases only.

  - Task C: bandwidth-bound vs compute-bound classification among
    nonzero cases.

  - Task D: actionable triage decision quality, i.e., whether the
    predicted class would lead to the same optimization focus as the
    profiler-derived class.

  - Avoid mixing these tasks in a way that lets a large zero-RAI
    majority mask poor nonzero regression performance.

**Use more appropriate statistics and uncertainty reporting.**

  - For numerical error, add median absolute percentage error, median
    log error, symmetric MAPE or log-ratio error, and
    per-precision/per-GPU confidence intervals.

  - For classification, report balanced accuracy, macro F1, confusion
    matrices with raw counts, and precision/recall for compute-bound
    cases.

  - For source-only vs source+SASS, use paired tests or bootstrap
    confidence intervals on per-kernel deltas.

  - For feature/error correlations, report confidence intervals for
    Cliff’s Delta, feature prevalences, and multiple-comparison
    correction.

**Validate the feature-labeling pipeline.**

  - Do not rely solely on LLM votes to label error-prone code features.

  - Hand-label at least a statistically meaningful sample, report
    agreement, and calibrate LLM-vote precision/recall against human
    labels.

  - Publish the feature definitions and labeling prompts. Clarify
    whether labels refer to source-visible patterns or actually executed
    paths.

  - Show examples of false-positive and false-negative feature labels;
    otherwise the heatmap may overstate the reliability of the causal
    interpretation.

**Tighten the definition of RAI and ground truth.**

  - List exact Nsight Compute counter names for each FLOP precision and
    for DRAM reads/writes.

  - Explain how counters are converted to bytes and FLOPs, how FMAs are
    counted, and whether helper/library instructions are included.

  - Define precisely how FP16, FP32, and FP64 RAI are interpreted for
    mixed-precision kernels.

  - Explain the treatment of compiler-injected FP16 instructions: are
    these semantically useful computation, implementation artifact, or
    both?

  - Clarify whether DRAM writebacks after kernel completion are counted,
    whether L2 write-back effects are visible in the chosen counters,
    and how this affects the denominator of RAI.

**Make the “hardware-free” claim precise.**

  - Replace broad “without hardware access” language with “without
    runtime profiling on the target GPU,” if that is the actual claim.

  - Explain what is still required: source code, build system, target
    architecture, compiler, cuobjdump/nvdisasm, GPU specs, and possibly
    target-specific libraries.

  - Discuss scenarios where SASS is unavailable or compilation for the
    target architecture is impossible, and whether source-only results
    remain useful enough.

**Demonstrate software-engineering utility.**

  - Add an evaluation that connects predicted RAI to decisions
    developers or LLM agents would make.

  - For example: given a fixed profiling budget, use predicted RAI to
    prioritize kernels, then measure how often the selected kernels
    match profiler-derived priorities.

  - Alternatively, integrate the predictor into a simple optimization
    agent and show whether it reduces wasted profiling iterations or
    improves time-to-good-variant compared with random or heuristic
    triage.

  - Even a simulation based on existing profiler data would strengthen
    the ICSE fit substantially.

**Strengthen the artifact.**

  - Release all kernels, exact command lines, compile logs, SASS
    extraction scripts, profiler outputs, processed CSVs, prompt
    templates, raw model responses, parsing code, and notebooks/scripts
    that generate every figure and table.

  - Document hardware/software versions: GPU SKU, driver,
    CUDA/NVHPC/Clang versions, Nsight Compute version, OS, clocks, power
    settings, and model API versions/dates.

  - Provide a one-command reproduction path for at least the data
    processing and plotting stages, even if re-running all profiling
    requires the GPUs.

## Tier 2: Strong improvements

**Refine the claims.**

  - Avoid saying that LLMs “can estimate RAI well enough” without
    specifying in which regimes, for which metrics, and relative to
    which baselines.

  - State negative results more prominently: source-only FP16 is often
    poor; GPT-OSS may not use SASS effectively; smaller-cache GPUs are
    harder; some features systematically degrade performance.

  - Frame the result as “conditional early triage under known
    limitations,” not as a replacement for profiling.

**Improve result presentation.**

  - Add a compact aggregate table with median absolute error, IQR,
    balanced accuracy, and raw counts by model/prompt/precision.

  - Make Figure 6 easier to interpret by reporting key summary
    statistics in the caption or a table.

  - For Table III, explain denominators and raw sample counts for every
    cell, especially where some precision/runtime combinations have few
    nonzero samples.

  - For Figure 7, include raw counts in addition to percentages so
    readers can detect class imbalance effects.

**Strengthen related work positioning.**

  - More explicitly contrast against static instruction-counting tools,
    compiler analysis, analytical Roofline/traffic models, and prior GPU
    static performance predictors.

  - Clarify how this differs from LLMPerf, Omniwise, and coarse Roofline
    classification work beyond the absence of fine-tuning.

  - Explain why zero-training prompting is the right design point for
    developers, rather than fine-tuning or task-specific static
    analysis.

**Improve ICSE framing and title/abstract.**

  - Consider a title that emphasizes developer support or
    profiling-budget triage, not only GPU kernels.

  - In the abstract and introduction, state the developer workflow and
    decision problem explicitly.

  - In the conclusion, translate findings into concrete guidance for
    tool builders and practitioners.

**Polish writing and terminology.**

  - Replace “it’s” with “its” where possessive is intended.

  - Fix typos such as “divison,” “much account,” “per-gpu,” and
    inconsistent capitalization of Source-Only/source-only/source+SASS.

  - Define all acronyms at first use and keep “RAI,” “Arithmetic
    Intensity,” and “Roofline Arithmetic Intensity” consistent.

  - Use “CUDA/OpenMP runtimes” carefully; OpenMP target offload is a
    programming model/runtime, while CUDA is also a programming model
    and platform.

# 5\. Field-by-field text for the review system

Below are concise versions of the review that can be pasted into
separate form fields if the review system asks for them.

## Brief summary

<table>
<tbody>
<tr class="odd">
<td><p><strong>Brief summary</strong></p>
<p>This paper asks whether off-the-shelf LLMs can estimate Roofline Arithmetic Intensity for GPU kernels before profiling. The authors prompt models to predict launch dimensions, FP16/FP32/FP64 operation counts, and DRAM bytes read/written, then derive RAI and compare predictions against Nsight Compute profiling for 254 CUDA/OpenMP kernels across RTX 3080, A10, A100, and H100 GPUs. The paper compares source-only prompts with source+SASS prompts and studies error patterns, cost, and response time.</p></td>
</tr>
</tbody>
</table>

## Main strengths

<table>
<tbody>
<tr class="odd">
<td><p><strong>Main strengths</strong></p>
<p>The paper studies a practical bottleneck in LLM-assisted performance engineering, uses an interpretable metric, evaluates on a sizable multi-GPU benchmark, and distinguishes source-only from compiler-artifact-aided prediction. The paper is readable and the benchmark could be valuable if fully released.</p></td>
</tr>
</tbody>
</table>

## Main weaknesses

<table>
<tbody>
<tr class="odd">
<td><p><strong>Main weaknesses</strong></p>
<p>The evaluation lacks essential deterministic/static baselines, uses only one query per expensive model setting, does not provide enough uncertainty or statistical testing, and relies on unvalidated LLM-generated feature labels for the error analysis. The paper also needs a clearer definition of RAI for mixed precision and zero-RAI cases, more precise profiling-counter details, and a stronger demonstration that the predictions improve an actual developer or optimizer workflow.</p></td>
</tr>
</tbody>
</table>

## Overall recommendation

<table>
<tbody>
<tr class="odd">
<td><p><strong>Overall recommendation</strong></p>
<p>Weak Reject. I find the direction promising, but the current evidence is not yet sufficient for ICSE acceptance. The paper should be encouraged toward resubmission after adding baselines, robustness analysis, validated labels, clearer metric definitions, and a stronger software-engineering utility evaluation.</p></td>
</tr>
</tbody>
</table>

## Confidence

<table>
<tbody>
<tr class="odd">
<td><p><strong>Confidence</strong></p>
<p>4/5. I am reasonably confident in the assessment of the paper’s performance-engineering and LLM-evaluation methodology, although I would still welcome additional details from the authors about exact profiler metrics, model settings, and artifact completeness.</p></td>
</tr>
</tbody>
</table>

# 6\. What could change the judgment during rebuttal?

The rebuttal could improve confidence but would probably not be enough
to move the paper to acceptance unless the authors already have
substantial missing evidence. The most important rebuttal items would
be:

  - Show results against at least one meaningful static baseline and one
    trivial baseline.

  - Provide evidence that predictions are stable across repeated queries
    or that deterministic settings were used.

  - Give raw counts and denominators for zero/nonzero, BB/CB, runtime,
    GPU, and precision groups.

  - Clarify exact Nsight metrics and how FLOPs/bytes/RAI are computed.

  - Explain the artifact status and whether all prompts, outputs,
    scripts, and processed data are available.

  - Make a concrete argument for ICSE utility: how would a developer or
    optimization agent use this and how often would the predicted choice
    be correct?

If the rebuttal can only clarify wording but cannot add baselines or
robustness evidence, I would keep the recommendation as Weak Reject.

# Appendix: acceptance checklist for a revised version

  - \[ \] Includes deterministic source-only and SASS-based baselines.

  - \[ \] Reports raw counts, denominators, confidence intervals, and
    paired source vs SASS deltas.

  - \[ \] Separates zero detection, nonzero regression, bound
    classification, and triage utility.

  - \[ \] Quantifies LLM run-to-run variability or pins deterministic
    settings.

  - \[ \] Validates LLM-generated code-feature labels against human
    annotations.

  - \[ \] Defines exact profiler counters, FLOP conventions, byte
    conventions, and mixed-precision RAI semantics.

  - \[ \] Reframes hardware-free as no runtime profiling, if applicable.

  - \[ \] Demonstrates practical developer/tool value under a
    profiling-budget scenario.

  - \[ \] Provides complete artifact with scripts, raw data, prompts,
    model outputs, versions, and reproduction instructions.

  - \[ \] Rewrites the abstract/conclusion so the claims match the
    evidence and limitations.
