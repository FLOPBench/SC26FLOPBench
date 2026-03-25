# Direct Prompting LLM Experiments

This directory contains LangGraph pipelines designed to automatically query large language models for target source files and compilation artifacts to predict FLOP counts and DRAM accesses.

## Overview

The direct-prompting pipeline works natively against the `dataset-creation/gpuFLOPBench.json` definitions. For every single GPU kernel invocation detected, the pipeline translates the compiled source paths, compiler arguments, SASS extractions, and hardware profiles into an XML structure evaluated using an LLM.

- `prompts.py`: Defines the strictly validated `Pydantic` targets models and dynamically populates `<command_line_input_args>` and XML extraction models into LLM payloads.
- `graph.py`: Contains the actual `StateGraph` definition combining queries, token processing, Pydantic Structured Returns, and integration natively executing with the `PostgresSaver`.
- `db_manager.py`: Sets up a local `gpuflops_db` PostgreSQL instance, manages PostgreSQL startup, provides checkpoint parsing, tracks failed attempts, and supports dump/restore workflows.
- `run_queries.py`: Top-level iterator parsing datasets into structured LLM runs mapped via unique `thread_ids`. It handles resume behavior, retry limits, timeouts, dry runs, and database lifecycle controls.
- `visualize_results.py`: Reads stored checkpoints and retry metadata from PostgreSQL, builds a pandas dataframe of completed and failed samples, and writes plots and summary tables.
- `make_plots_for_paper.py`: Builds a paper-oriented dataframe from the same PostgreSQL checkpoints, derives arithmetic intensity and roofline bound classifications, filters to the no-IMIX paper subset, and writes the focused Figure 1 through Figure 4 artifacts used for the paper.
- `print_highlevel_flopbench_stats_for_paper.py`: Reads `dataset-creation/gpuFLOPBench.json` and prints high-level CUDA versus OpenMP code and kernel counts for paper tables or narrative summaries.
- `runExperiments.sh`: Convenience shell wrapper that launches several representative `run_queries.py` configurations for the currently used Anthropic and OpenAI models across different evidence configurations.

## Database Layout

The direct-prompting workflow uses a local PostgreSQL database named `gpuflops_db` by default.

Two tables matter for the current runner and visualizer:

- `checkpoints`: Managed by LangGraph `PostgresSaver`. This is the primary results table. Each row stores a `thread_id`, checkpoint identifiers, and a serialized graph state payload in the `checkpoint` column. The visualizer reads `checkpoint.channel_values` from this table to recover values such as:
	- program and kernel names
	- expected FLOP and DRAM metrics
	- predicted metrics
	- `metrics_diff` and recomputed `metrics_pct_diff`
	- token counts, query time, model/provider metadata, and estimated cost
	- `metrics_explanations` strings emitted by the validator node
- `query_attempts`: Managed by `QueryAttemptTracker` in `db_manager.py`. This table tracks operational status per `thread_id`, including total attempts, failed attempts, last status, last error, and update time. The visualizer uses this table to include failed runs alongside completed runs.

In other words, visualization is not reading a pre-aggregated reporting table. It reconstructs the reporting dataframe from raw LangGraph checkpoints in `checkpoints` plus retry/failure metadata in `query_attempts`.

## Thread IDs And Resume Behavior

Each query is keyed by a deterministic `thread_id` built from:

- program name
- mangled kernel name
- GPU target
- model name
- evidence configuration: `nosass_noimix`, `sass_noimix`, `nosass_imix`, or `sass_imix`
- trial number

Dry runs append a `_DRYRUN` suffix so they do not collide with normal experiment runs.

Legacy runs may still use the older two-state suffixes `withsass` and `nosass`. The visualizer treats those as backward-compatible aliases for `sass_imix` and `nosass_noimix`, respectively.

Normal runs inspect the database before execution and skip:

- queries that already reached a completed checkpoint
- queries whose failed-attempt count has already reached `--maxFailedAttempts`

`--singleDryRun` intentionally ignores prior completion state so the API path can be revalidated on demand.

## Quickstart / Dry Run

To verify the installation pathing, environment, OpenRouter API keys, and Graph validations, it is strongly recommended you do a single explicit Dry Run first. This maps the script directly to exclusively evaluate `adam-cuda` on the `H100` architecture.

```bash
# 1. Export your required OpenRouter/OpenAI authentication tokens
export OPENROUTER_API_KEY=sk-or-your_api_key_here

# 2. Run the single dry run (skips already-completed logs natively)
python3 experiments/direct-prompting/run_queries.py --singleDryRun --verbose

# Note: Using --singleDryRun forces the script to bypass any existing "completed" states natively to re-invoke the API.
```

## Running Full Execution

To trigger the complete dataset traversal natively over all execution kernels:

```bash
export OPENROUTER_API_KEY=your_api_key
python3 experiments/direct-prompting/run_queries.py
```

By default it uses `openai/gpt-5.1-codex-mini`. You can adjust it via `--modelName`.

```bash
# Run on standard paths tracking repeating 3 inference trails explicitly via --trials
python3 experiments/direct-prompting/run_queries.py --modelName openai/gpt-4o-mini --trials 3 --verbose

# Source-only prompt
python3 experiments/direct-prompting/run_queries.py

# Include SASS only
python3 experiments/direct-prompting/run_queries.py --useSASS

# Include IMIX only
python3 experiments/direct-prompting/run_queries.py --useIMIX

# Include both SASS and IMIX
python3 experiments/direct-prompting/run_queries.py --useSASS --useIMIX
```

## run_queries.py Arguments

- `--modelName`: OpenRouter model identifier. This value is also encoded into each `thread_id` so results from different models do not collide.
- `--trials`: Determines how many repeating loop tests are natively sent per specific environment architecture mapped.
- `--maxQueries`: Caps how many runnable queries will execute in the current invocation after completed and skipped queries are filtered out.
- `--maxTimeout`: Maximum wall-clock seconds allowed for a single query before the runner interrupts it.
- `--maxSpend`: Caps the total new USD spend allowed during the current invocation. Already-completed runs already stored in PostgreSQL do not count toward this budget.
- `--queryBatchSize`: Number of runnable queries to execute in parallel per batch.
- `--maxFailedAttempts`: If a query reaches this many failed attempts, future runs skip it automatically until the database is reset or edited.
- `--singleDryRun`: Restricts execution to a deterministic `adam-cuda` on `H100` sample. This mode always re-runs the selected query even if an older completed checkpoint exists.
- `--verbose`: Prints the full prompt, then prints the final metric summary, token usage, model metadata, query time, and estimated cost after each run.
- `--printPrompts`: Prints the full system and human prompts for each query before the API call.
- `--useSASS`: Includes optional SASS data in the prompt. This changes the `thread_id` namespace so SASS-only, IMIX-only, both, and neither stay separate.
- `--useIMIX`: Includes optional static IMIX data in the prompt. This is independent of `--useSASS`.
- `--importDBDumpFile`: Restores the database from a PostgreSQL custom-format dump before query execution begins.
- `--deleteDBFreshStart`: Drops the working database before execution and starts from a clean database state.
- `--dumpDBOnFinish`: Writes a PostgreSQL custom-format dump to `gpuflops_db.dump` after a successful run.
- `--exportDBOnly`: Skips query execution and only exports the current `gpuflops_db` contents after any requested wipe or restore operations.

## Database Lifecycle Behavior

`run_queries.py` calls `ensure_postgres_running()` before doing any work. That helper inspects the local PostgreSQL cluster and starts or restarts it if needed.

Database behavior at startup is:

- if `--deleteDBFreshStart` is set, drop the database first
- if `--importDBDumpFile` is set, restore that dump into a fresh `gpuflops_db`
- otherwise, ensure `gpuflops_db` exists and connect to it

Database behavior during the run is:

- LangGraph writes graph state snapshots into `checkpoints`
- the runner writes operational status into `query_attempts`
- completed queries are recognized by the presence of `total_tokens` in checkpoint `channel_values`

Database behavior at the end is:

- if `--dumpDBOnFinish` is set and the run completed successfully, create `gpuflops_db.dump`

## What The Graph Persists

The graph stores plain serializable data in checkpoint state. Important persisted fields include:

- expected FLOP counts and DRAM bytes
- predicted FLOP counts and DRAM bytes
- `metrics_diff`
- `metrics_pct_diff`
- `metrics_explanations`
- token counts and query time
- model name, provider, response id, and response metadata
- estimated `cost_usd`

This is why later analysis can be performed directly from PostgreSQL without rerunning the model.

## visualize_results.py

`visualize_results.py` connects to the same PostgreSQL database, reads both `checkpoints` and `query_attempts`, and reconstructs one combined dataframe containing:

- completed samples recovered from the latest completed checkpoint per `thread_id`
- failed samples recovered from `query_attempts`, with partial metadata filled from the latest available checkpoint when present
- model name, evidence configuration, GPU target, trial number, timing, token, and cost metadata
- metric deltas and recomputed percent differences

The script derives model, evidence configuration, GPU, and trial by parsing the `thread_id` naming convention.

### Usage

```bash
python3 experiments/direct-prompting/visualize_results.py
python3 experiments/direct-prompting/visualize_results.py --includeDryRun
python3 experiments/direct-prompting/visualize_results.py --outputDir experiments/direct-prompting/my-plots
```

### Visualization Arguments

- `--dbUri`: Explicit PostgreSQL URI. If omitted, the script connects to the default local `gpuflops_db`.
- `--outputDir`: Output directory for all plots, tables, and markdown artifacts.
- `--includeDryRun`: Includes `_DRYRUN` thread IDs in the analysis. By default those are excluded.

### Plots And Tables Produced

The script currently writes these artifacts:

- `plot1_sample_counts_by_model.png`: Stacked bar chart of sample counts by model and evidence configuration, including both completed and failed samples.
- `plot2_query_time_distribution.png`: Query-time boxplots grouped by evidence configuration, with model overlays.
- `plot3_cost_distribution.png`: Cost boxplots grouped by evidence configuration, with model overlays.
- `plot4a_metrics_diff_distribution.png`: Distribution grid of raw prediction-minus-expected metric differences, grouped by model and evidence configuration.
- `plot4b_metrics_pct_diff_distribution.png`: Distribution grid of percent differences, grouped by model and evidence configuration.
- `table1_model_percent_diff_summary.tex` and `.png`: Mean and median percent-difference summary by model and evidence configuration, plus completed and failed sample counts.
- `table2_best_and_worst_predictions.csv`: Top and bottom samples per model and evidence configuration based on mean percent difference.
- `table2_by_model/*.png`: Per-model rendered tables for the best and worst predictions.
- `other_visualizations.md`: Additional suggested analyses not yet automated.

The script also prints the head of the parsed dataframe and the dataframe column list to the console for debugging.

## make_plots_for_paper.py

`make_plots_for_paper.py` connects to the same PostgreSQL database and reuses the checkpoint reconstruction logic from `visualize_results.py`, but then derives additional paper-specific metrics and filtered views.

The script currently:

- reconstructs completed direct-prompting samples from `checkpoints` and `query_attempts`
- derives predicted FLOP and DRAM counts from expected values plus stored `metrics_diff`
- computes expected and predicted arithmetic intensity for FP16, FP32, and FP64
- computes GPU-specific roofline balance points from hardcoded paper hardware specs for 3080, A10, A100, and H100
- classifies each precision-specific sample as `bandwidth-bound` or `compute-bound`
- excludes precision-specific cases where expected arithmetic intensity is zero
- filters the paper plots to the no-IMIX comparisons: `No SASS / No IMIX` and `SASS Only / No IMIX`
- prints expected bound-class distributions by GPU and precision for the filtered nonzero-AI subset

### Usage

```bash
python3 experiments/direct-prompting/make_plots_for_paper.py
python3 experiments/direct-prompting/make_plots_for_paper.py --includeDryRun
python3 experiments/direct-prompting/make_plots_for_paper.py --outputDir experiments/direct-prompting/paper-figure-output
```

### Paper Plot Arguments

- `--dbUri`: Explicit PostgreSQL URI. If omitted, the script connects to the default local `gpuflops_db`.
- `--outputDir`: Output directory for the paper figures.
- `--includeDryRun`: Includes `_DRYRUN` thread IDs in the paper plots. By default those are excluded.

### Paper Figures Produced

The script currently writes these artifacts:

- `figure1_ai_difference_boxplots.png`: Two stacked boxplot panels showing `Predicted AI - Expected AI` by model for FP16, FP32, and FP64 on a symmetric log x-axis, split between `No SASS / No IMIX` and `SASS Only / No IMIX`.
- `figure2_ai_bound_confusion_heatmaps.png`: A 2x2 subplot grid over model and SASS setting showing expected-versus-predicted bound-class confusion heatmaps. Each cell is row-normalized by the true bound class, and annotations report the FP16, FP32, and FP64 percentages separately.
- `figure3_ai_difference_boxplots_by_gpu.png`: Two stacked boxplot panels showing `Predicted AI - Expected AI` by GPU target for FP16, FP32, and FP64, again split between `No SASS / No IMIX` and `SASS Only / No IMIX`.
- `figure4_ai_difference_boxplots_by_runtime.png`: Two stacked boxplot panels showing `Predicted AI - Expected AI` by runtime (`cuda` versus `omp`) for FP16, FP32, and FP64, again split between `No SASS / No IMIX` and `SASS Only / No IMIX`.

The paper figures are written by default to `experiments/direct-prompting/paper-figure-output/`.

## print_highlevel_flopbench_stats_for_paper.py

`print_highlevel_flopbench_stats_for_paper.py` is a lightweight dataset-summary helper. It reads `dataset-creation/gpuFLOPBench.json` and prints:

- number of CUDA codes
- number of OpenMP codes
- number of CUDA kernels
- number of OpenMP kernels
- total code count
- total kernel count

### Usage

```bash
python3 experiments/direct-prompting/print_highlevel_flopbench_stats_for_paper.py
python3 experiments/direct-prompting/print_highlevel_flopbench_stats_for_paper.py --dataset dataset-creation/gpuFLOPBench.json
```

## runExperiments.sh

`runExperiments.sh` is a convenience wrapper that launches a small batch of representative `run_queries.py` configurations. At the time of writing it covers:

- `anthropic/claude-opus-4.6` without SASS
- `anthropic/claude-opus-4.6` with SASS
- `openai/gpt-5.4` without SASS or IMIX
- `openai/gpt-5.4` with SASS only
- `openai/gpt-5.4` with IMIX only
- `openai/gpt-5.4` with both SASS and IMIX

Each invocation currently uses `--trials 1`, `--queryBatchSize 4`, `--maxSpend 50.0`, `--verbose`, and `--printPrompts`.

**PostgreSQL**: Ensure `postgresql` service is running safely locally at port `5432` natively. The agent uses password mapping: `postgres:postgres@localhost`.
