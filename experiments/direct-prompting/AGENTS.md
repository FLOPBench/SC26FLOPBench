# Direct Prompting LLM Experiments

This directory contains LangGraph pipelines designed to automatically query large language models for target source files and compilation artifacts to predict FLOP counts and DRAM accesses.

## Overview

The direct-prompting pipeline works natively against the `dataset-creation/gpuFLOPBench.json` definitions. For every single GPU kernel invocation detected, the pipeline translates the compiled source paths, compiler arguments, SASS extractions, and hardware profiles into an XML structure evaluated using an LLM.

- `prompts.py`: Defines the strictly validated `Pydantic` targets models and dynamically populates `<command_line_input_args>` and XML extraction models into LLM payloads.
- `graph.py`: Contains the actual `StateGraph` definition combining queries, token processing, Pydantic Structured Returns, and integration natively executing with the `PostgresSaver`.
- `db_manager.py`: Sets up a local `gpuflops_db` PostgreSQL instance enabling Langgraph Checkpointer persistence safely for large iteration loads natively out of the box.
- `run_queries.py`: Top-level iterator parsing datasets into structured LLM runs mapped via unique `thread_ids`. Enables safe, deterministic repetitions!

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

By default it uses `openai/gpt-5.1-codex-mini`. You can adjust it via `--model-name` flags natively.

```bash
# Run on standard paths tracking repeating 3 inference trails explicitly via --trials
python3 experiments/direct-prompting/run_queries.py --model-name openai/gpt-4o-mini --trials 3 --verbose
```

### Argument Details:

- `--model-name`: Pass the explicit target model identifier dynamically injected into the Graph configuration layer correctly. Defaults to standard OpenAI OpenRouter endpoints.
- `--trials`: Determines how many repeating loop tests are natively sent per specific environment architecture mapped.
- `--verbose`: Append execution outputs continuously matching the console logs!
- `--singleDryRun`: Restricts full execution paths statically mapping to a single deterministic target cleanly to ensure outputs and structured predictions don't break JSON endpoints. Note that `singleDryRun` mode always forces execution to evaluate API endpoints, effectively ignoring previously completed queries in the database to ensure we can verify the pipeline successfully connects and functions.

**PostgreSQL**: Ensure `postgresql` service is running safely locally at port `5432` natively. The agent uses password mapping: `postgres:postgres@localhost`.
