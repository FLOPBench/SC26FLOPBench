# Direct Prompting Experiment

This directory contains pipelines designed to query Large Language Models (via LangGraph) to predict GPU FLOP operations and DRAM utilization for kernels described in the `gpuFLOPBench` dataset. These predictions are then verified against profiling data (collected via Nsight Compute) directly within the graph's workflow to establish absolute and percentage differences.

## File Structure

- **`prompts.py`**: Defines the prompt logic, string generator templates, and the Pydantic schema used to securely structure the LLM output into predictable mathematical counts directly tied to expected performance profiles.
- **`graph.py`**: Constructs the state graph executed by LangGraph. It establishes the sequential operations mapping inputs and validating outputs, whilst storing expected and true evaluation metrics across precision ranges (FP16, FP32, FP64) alongside read/write capabilities dynamically during LLM calls.
- **`run_queries.py`**: Drives the execution logic. Opens the compiled `dataset-creation/gpuFLOPBench.json` entries and dispatches LLM queries specifically matched against program names, unique demangled kernels, and targeted platform GPUs. Stores resulting inferences via Postgres DB checkpointer and intelligently resumes incomplete jobs utilizing deterministic target IDs.
- **`db_parser.py`**: Includes the `CheckpointDBParser` wrapper. It queries the PostgreSQL checkpointer natively directly leveraging DB cursor objects, establishing metrics representing API costs, token ingestion throughput, and the total execution scale elapsed.

## Workflow

1. A dataset containing program hierarchies and metric mapping (`dataset-creation/gpuFLOPBench.json`) acts as execution fodder containing kernels bound against multiple simulated hardware instances (3080, A100, H100).
2. The `run_queries.py` iterator attempts to compile execution pipelines for every combination while querying `db_parser.py` implicitly to remove previously queried permutations based explicitly on the active `thread_id`.
3. Valid candidates initialize their context mapping attributes globally in LangGraph, and are executed. The `query_node` uses OpenRouter-configured LLMs to interpret sources and instructions, predicting precision outputs sequentially utilizing formatting guided efficiently via `Pydantic`.
4. Outputs transfer instantly to the `validator_node` which extracts cost metrics from inference parameters and computes the algebraic offset against initially provided baseline data in the dataset.
