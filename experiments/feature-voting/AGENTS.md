# Feature Voting Experiment

## Purpose

This directory contains the code-feature voting experiment.

The goal is to ask one or more LLMs to inspect benchmark host code plus target kernel source code and return a structured boolean checklist of code features for the first execution path of the target kernel.

This experiment is for data collection only.

It does not:

- compute vote aggregates
- decide a final consensus label
- use GPU-specific profiling information
- use SASS or IMIX evidence
- use compiler command metadata

Each model response is stored independently in PostgreSQL so a later script can read the collected votes and perform aggregation or analysis.

## Current Design

The current feature-voting pipeline is intentionally kernel-level and GPU-agnostic.

One query is generated per:

- program
- target kernel
- model
- trial

The same kernel query is reused across all GPUs because this experiment is focused on static source-level code features rather than hardware-specific execution behavior.

### Inputs Used

Each query includes only:

- `program_name`
- `kernel_mangled_name`
- `kernel_demangled_name`
- `exe_args`
- `source_code_files`

### Inputs Not Used

The following inputs were intentionally removed from this experiment:

- `gpu_roofline_specs`
- `compile_commands`
- `sass_dict`
- `imix_dict`

### Structured Output

The LLM returns a `CodeFeatureFlags` object from [prompts.py](/gpuFLOPBench-updated/experiments/feature-voting/prompts.py) with boolean fields such as:

- `has_branching`
- `has_data_dependent_branching`
- `has_flop_division`
- `uses_preprocessor_defines`
- `has_common_float_subexpr`
- `has_loop_invariant_flops`
- `has_special_math_functions`
- `calls_device_function`
- `has_rng_input_data`
- `reads_input_values_from_file`
- `has_constant_propagatable_gridsz`
- `has_constant_propagatable_blocksz`

The graph also stores top-level `predicted_*` copies of these fields in the checkpoint tail so they can be queried without unpacking the nested `prediction` object.

### Persistence Model

Results are stored in PostgreSQL database `code_features_db` using:

- LangGraph checkpoints for per-query state
- `query_attempts` table for retry and failure tracking

The experiment supports resuming partial runs. Completed queries are skipped automatically unless the database is wiped or dry-run mode is used.

## File Roles

- [run_voting_queries.py](/gpuFLOPBench-updated/experiments/feature-voting/run_voting_queries.py): CLI entry point, dataset traversal, batching, retries, and progress reporting.
- [graph.py](/gpuFLOPBench-updated/experiments/feature-voting/graph.py): LangGraph query pipeline, structured output call, and checkpoint metadata extraction.
- [prompts.py](/gpuFLOPBench-updated/experiments/feature-voting/prompts.py): system prompt, human prompt generation, and `CodeFeatureFlags` schema.
- [db_manager.py](/gpuFLOPBench-updated/experiments/feature-voting/db_manager.py): PostgreSQL startup, database create/wipe/dump/restore helpers, and checkpoint inspection utilities.

## Running The Experiment

Run all commands from this directory unless noted otherwise:

```bash
cd /gpuFLOPBench-updated/experiments/feature-voting
```

### Basic Run

```bash
python run_voting_queries.py --modelNames "openai/gpt-5.1-codex-mini"
```

### Multiple Models

Use a comma-separated list with no extra quoting inside the value:

```bash
python run_voting_queries.py \
	--modelNames "openai/gpt-5.1-codex-mini,anthropic/claude-sonnet-4,openai/gpt-4.1"
```

### Dry Run

This runs only the first kernel query and is the safest way to verify prompt formatting, API access, and checkpoint writes.

```bash
python run_voting_queries.py \
	--singleDryRun \
	--modelNames "openai/gpt-5.1-codex-mini" \
	--verbose \
	--printPrompts
```

### Batching, Spend Limits, And Timeouts

```bash
python run_voting_queries.py \
	--modelNames "openai/gpt-5.1-codex-mini,anthropic/claude-sonnet-4" \
	--trials 2 \
	--queryBatchSize 4 \
	--maxTimeout 300 \
	--maxSpend 20.0 \
	--maxQueries 100
```

### Fresh Start

This drops `code_features_db` and starts collection from scratch.

```bash
python run_voting_queries.py \
	--deleteDBFreshStart \
	--modelNames "openai/gpt-5.1-codex-mini"
```

### Export Database Only

This writes a PostgreSQL custom dump to `code_features_db.dump` without running any queries.

```bash
python run_voting_queries.py --exportDBOnly
```

### Restore From Dump

```bash
python run_voting_queries.py \
	--importDBDumpFile /path/to/code_features_db.dump \
	--modelNames "openai/gpt-5.1-codex-mini"
```

## Important CLI Flags

- `--modelNames`: comma-separated OpenRouter model list.
- `--trials`: repeat count per model and kernel.
- `--queryBatchSize`: number of parallel queries per batch.
- `--maxTimeout`: per-query timeout in seconds.
- `--maxSpend`: stop after this much spend during the current script run.
- `--maxQueries`: cap how many runnable queries this invocation will execute.
- `--maxFailedAttempts`: skip queries that have failed this many times.
- `--singleDryRun`: run only the first kernel query.
- `--verbose`: print completed results.
- `--printPrompts`: print the system and human prompts before the query is sent.
- `--deleteDBFreshStart`: wipe `code_features_db` before running.
- `--importDBDumpFile`: restore a dump before running.
- `--dumpDBOnFinish`: export a dump after a successful run.
- `--exportDBOnly`: export the current database and exit.

## Result Shape

Each completed checkpoint tail contains:

- query identity fields such as program and kernel names
- nested `prediction` object with the raw `CodeFeatureFlags`
- top-level `predicted_*` boolean fields
- token usage
- cost estimate
- model/provider metadata
- query time

`run_voting_queries.py` prints the `predicted_*` fields in human-readable label form during verbose or dry-run output.

## Operational Notes

- `--singleDryRun` ignores prior completion state and always runs the first query.
- Non-dry runs prompt for confirmation before executing runnable queries.
- Existing historical checkpoint rows from older versions of this experiment may use older thread-id formats. They can coexist with the current kernel-level format in the same database.
- This directory currently has no aggregation script. Any consensus logic should be implemented separately so raw per-model votes remain preserved.

## Guidance For Future Changes

If modifying this experiment, keep these constraints in mind:

- Preserve the kernel-level, GPU-agnostic query design unless there is a clear reason to reintroduce hardware-specific inputs.
- Keep `CodeFeatureFlags` synchronized between [prompts.py](/gpuFLOPBench-updated/experiments/feature-voting/prompts.py), [graph.py](/gpuFLOPBench-updated/experiments/feature-voting/graph.py), and [run_voting_queries.py](/gpuFLOPBench-updated/experiments/feature-voting/run_voting_queries.py).
- If a new boolean flag is added, update:
	- the prompt checklist
	- the Pydantic schema
	- the `predicted_*` fields in `GraphState`
	- the `validator_node`
	- the human-readable printing list in `print_run_result`
- Avoid mixing vote collection with vote aggregation in the same script.
