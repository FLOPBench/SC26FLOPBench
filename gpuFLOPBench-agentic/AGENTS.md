# AGENTS

## 1) Your role
- Document the code-search tools and unit-test conventions so future contributors can quickly understand what is available and how to extend it.
- Treat `langchain-tools/code-search-tools` as the single source of truth for CUDA tooling and `unit-tests/` as the verification batch for those tools plus some ancillary checks; anybody touching either area should update this file as part of new work.
- Whenever you describe a tool or a test, mention both the intent (what it is supposed to do) and the concrete module/path so readers can immediately find the implementation.
- The repository layout is deliberate: `gpuFLOPBench/` holds the `*-cuda` benchmarks under `src/`, `langchain-tools/` exposes LangChain agents (including `code-search-tools`) and shared helpers, `langchain-tools/treesitter-tools/` houses the `cst_utils.py` helpers that agents import via short shell-launched Python scripts for CUDA/OpenMP parsing, `unit-tests/` houses the regression suite plus the extracted-kernel-solution data, and higher-level helpers like `agents/` and `helper-scripts/` sit alongside `run_tests.sh` for project-wide operations.

## 1.1) Agent guidance
- Agents live in `agents/` and should be documented whenever new tooling or workflows depend on them.
- Each agent must clearly state which LangChain tools it consumes (list the concrete tool entrypoints such as `code_search_tools.cuda_file_tree.cuda_file_tree`) and how it wires them into its middleware/checkpointer. Mention any helper modules it depends on (e.g., `agents/llm_models.py`), and note when it runs Python scripts that import `langchain-tools/treesitter-tools/cst_utils.py` so contributors know the script-level dependency graph rather than expecting new LangChain tool wrappers.
- When adding a new agent, explain how callers should construct the LLM(s) (e.g., via `agents/llm_models.build_configurable_llm` or similar factory helpers) so that consumers know the expected credentials/overrides. Include notes about checkpointing (e.g., SQLite saver) if relevant.
- Document the expected runtime behavior: what prompts it requires, which state schema it uses, and how middleware (logging, tool/model call limits, error handling) is configured.
- The backwards slicing agent exposes `default_backwards_slicing_system_prompt` in `agents/backwards_slicing_agent.py` to describe the script-driven workflow. When you document or evolve this agent, explain how that prompt enumerates the available `treesitter_tools.cst_utils` helpers, stresses the `/tmp`-only write policy, and reminds the agent to use the built-in filesystem middleware plus the `execute` tool for running Python scripts.
- You are most likely executing in a Docker container, therefore you'll need to escalate shell access to run shell commands.

## 2) Adding new code search tools
- The tools live under `langchain-tools/code-search-tools`. Each module loads the shared `utils.py` helpers, defines a `langchain.tools.tool` decorated function, and exposes a single entrypoint that takes a `cuda_name` (via `CudaSubdirArgs`) so it cannot escape `gpuFLOPBench/src`.
- Current tools:
  - `cuda_file_tree` (`cuda-file-tree.py`): Builds a sorted, indented tree of the requested `*-cuda` directory so callers know the layout before inspecting files.
  - `cuda_global_functions` (`cuda-global-functions.py`): Scans CUDA/CPP/HEADER files for `__global__` definitions and emits kernel names with file/line coordinates.
  - `cuda_compile_commands` (`cuda-compile-commands.py`): Reads `gpuFLOPBench/cuda-profiling/compile_commands.json`, filters for the benchmark, and reports compiler/arguments/output pairs for every source mentioned.
- `cuda_main_files` (`cuda-main-files.py`): Searches for free-function `main()` definitions in the benchmark tree so integration tests can later confirm which files act as entry points.
- `extract_kernel_source_definition` (`extract-kernel-source-definition.py`): Replays the `__global__` definitions (including template headers and qualifiers) for a particular kernel name, letting consumers compare against known source snapshots.
- `function_definition_lister` (`function-definition-lister.py`): Enumerates every declaration/definition that `tree_sitter` sees inside the benchmark's C/C++/CUDA sources (relying on `gpuFLOPBench/utils/ts_helper.py` for parser setup) and emits a newline-delimited list containing the template signature (if any), CUDA qualifiers such as `__global__`/`__device__`, the return type, and the fully qualified name annotated with `(decl)` or `(defnt)` so callers can cross-check what methods exist and where they live. Supply the optional `file_name` (relative to the benchmark root) to focus on a single source file; omit it to scan every supported source file together.
- `include_tree_extractor` (`include-tree-extractor.py`): Builds the `#include` dependency tree for a single file inside the requested benchmark, annotating missing headers with `(DNE)` and stopping recursion when existing includes would otherwise loop back into an ancestor. Callers pass `cuda_name` plus the relative `file_name` to see which headers each translation unit actually includes.
- To add a new tool:
  1. Follow the established pattern: import the shared helpers (`utils.py`), define any additional argument model(s), and wrap the functionality in a `@tool` function that validates the `cuda_name`.
  2. Update `_CODE_SEARCH_TOOL_SPECS` in `unit-tests/test_backwards_slicing_agent.py` only when the backwards-slicing agent switches back to loading those LangChain helpers; the current workflow ignores `langchain-tools/code-search-tools` and relies instead on Python scripts that import `treesitter_tools.cst_utils`.
  3. Expand `unit-tests/check_code_search_tools.py` if you expect the new tool to be verified per benchmark (e.g., add assertions or new helper metadata). This file already reloads each tool module directly, so add your assertions near the other tests and reuse the per-benchmark metadata / solution directories as needed.
  4. Document the new tool here (or in another README) so downstream testers and agents know why it exists and how to invoke it.

## 3) Adding new unit tests
- `unit-tests/check_code_search_tools.py` is the canonical regression suite for the CUDA tooling. It imports each `*-tree_and_kernel_names.py` helper from `unit-tests/extracted-kernel-solutions/<cuda-name>-solutions/`, reads `EXPECTED_TREE`, `EXPECTED_KERNELS`, and `EXPECTED_MAIN_FILES`, and then exercises:
  1. `cuda_file_tree`, asserting the tree string matches `EXPECTED_TREE`.
 2. `cuda_global_functions`, validating the list of kernel metadata matches `EXPECTED_KERNELS`.
 3. `cuda_compile_commands`, confirming the expected source files appear in `compile_commands.json` entries and that each compile command carries the right includes/outputs.
 4. `extract_kernel_source_definition`, comparing extracted sources against the scripts in the same solution directory (each `<cuda-name>---<kernel>.py` must define a `solution` list of canonicalized kernel bodies).
 5. `cuda_main_files`, matching the helper’s `EXPECTED_MAIN_FILES` list.
- The solution directories (`unit-tests/extracted-kernel-solutions/<cuda-name>-solutions/`) must contain one metadata module plus one kernel solution per extracted kernel; metadata modules serve as the glue between tool expectations and the extracted solution snippets.
- When `function_definition_lister` needs verification, the metadata helpers should expose per-file `EXPECTED_FUNCTION_DEFINITIONS` and `EXPECTED_FUNCTION_DECLARATIONS` dictionaries so the unit tests can assert that each file returns the right `(defnt)`/`(decl)` lines instead of relying on a single aggregated string.
- When adding a new benchmark/test case:
  1. Create the `<cuda-name>-tree_and_kernel_names.py` metadata (string tree, list of kernel dictionaries, main file list).
 2. Add `<cuda-name>---<kernel>.py` files that expose their `solution` list so `_normalize_kernel_source` can compare against the extractor output.
 3. Register the new benchmark in `_EXPECTED_*` maps inside `check_code_search_tools.py`.
 4. If the benchmark ought to run through the backwards-slicing agent, ensure `test_backwards_slicing_agent.py` can see any new tools it depends on.
- Additional unit tests:
- `unit-tests/test_backwards_slicing_agent.py` runs the backwards-slicing agent without the CUDA-specific LangChain tools, uses the default prompt helper to describe the `treesitter_tools.cst_utils` helpers and the `/tmp`-only write policy, and instructs the model to write a Python script that analyzes `gpuFLOPBench/src/lulesh-cuda/lulesh.cu` and `gpuFLOPBench/src/lulesh-omp/lulesh.cc`. The test still deletes `test_backwards_slicing_with_llm_checkpoint.sqlite`, boots the agent via `agents.backwards_slicing_agent.make_backwards_slicing_agent`, and asserts the invocation returns at least one message.
  - `unit-tests/test_cuda_kernel_uniqueness.py` exercises `extract_kernel_source_definition` directly to assert every kernel of interest has a specific number of definitions (useful for catching duplicates or missing extractions).
  - `unit-tests/check_openrouter_api.py` ensures the OpenRouter LLM in `agents/llm_models.py` is reachable and returns a sensible reply; it requires `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in the environment.
- Run the suite via `python -m pytest -vv -s ./unit-tests/check_code_search_tools.py` (and the other pytest targets) whenever you touch the tools or their data to guarantee there are no regressions.
- The codes that we perform unit tests on are:
  - `lulesh-cuda`
  - `tsne-cuda`
  - `all-pairs-distance-cuda`
  - `addBiasResidualLayerNorm-cuda`
  - `multimaterial-cuda`
  - `atomicReduction-cuda`
  - `gmm-cuda`
  - `particlefilter-cuda`
  - `ert-cuda`
  - `bmf-cuda`
  - `miniFE-cuda`

## 4) Agent unit tests
- `unit-tests/test_backwards_slicing_agent.py` is the exemplar. It builds the backwards slicing agent via `agents.backwards_slicing_agent.make_backwards_slicing_agent` with no extra CUDA-specific tools, deletes `test_backwards_slicing_with_llm_checkpoint.sqlite` before creating the checkpointer, and runs a single invocation using the prompt helper plus the `/tmp`-only write policy. The test confirms the agent returns at least one message after being instructed to write and execute a Python script that uses `treesitter_tools.cst_utils` to analyze `gpuFLOPBench/src/lulesh-cuda/lulesh.cu` and `gpuFLOPBench/src/lulesh-omp/lulesh.cc`.
- Add similar tests whenever a new agent is introduced: load only the tools the agent actually needs, create the required prompts/state, and exercise at least one tool call if the agent is expected to perform work. Keep the SQLite checkpoint driver or another durable saver around, and delete or reset its backing file before each run so the artifact does not keep growing between test runs.


## 5) Project Files Description
- `gpuFLOPBench/`
  - `src/`: hosts the benchmark sources; each `cuda_name` under here takes the form `*-cuda` and corresponds to one CUDA program we train and test.
- `agents/`
  - `backwards_slicing_agent.py`: LangChain agent wiring that orchestrates the script-driven slicing pipeline, exposes `default_backwards_slicing_system_prompt`/`make_backwards_slicing_agent`, and relies on middleware (logging, call limits, checkpointer) plus the builtin filesystem/execute tools rather than new LangChain helpers.
  - `llm_models.py`: LLM factory helpers (OpenRouter/OpenAI) that agents call to build configurable models with the expected credentials/overrides.
- `helper-scripts/`
  - `find_duplicate_kernels.py`: standalone helper used to spot duplicate kernel definitions across the extracted solution set.
  - `sqlite_reader.py`: inspects `langgraph` SQLite checkpoint files (e.g., `unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite`) so humans can print every saved message/write when debugging the backwards slicing agent run; accompany any changes with `unit-tests/test_sqlite_reader.py`.
  - `unit-tests/test_sqlite_reader.py`: exercises the helper against `unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite`, prints the first few recovered messages plus any pending writes (run it with `python -m pytest -s unit-tests/test_sqlite_reader.py` for live output), and deliberately lacks defensive guards so schema mismatches surface immediately.
- `langchain-tools/`
  - `code-search-tools/`: LangChain tool collection (`cuda-file-tree.py`, `cuda-global-functions.py`, `cuda-compile-commands.py`, `cuda-main-files.py`, `extract-kernel-source-definition.py`, `function-definition-lister.py`, and shared `utils.py`) that operate over the `gpuFLOPBench/src` tree and drive the regression suite.
  - `treesitter-tools/`: helper library centered on `cst_utils.py` plus composition modules (`caches.py`, `traversal.py`, `openmp.py`, etc.) that agents invoke by running short Python scripts in the sandboxed shell rather than defining new LangChain tool wrappers.
- `unit-tests/`
  - `check_code_search_tools.py`: regression harness that loads the tool modules plus metadata helpers under `extracted-kernel-solutions/` to validate tree listings, kernels, compile commands, extracted sources, and function metadata per benchmark.
  - `extracted-kernel-solutions/`: per-benchmark metadata (`<cuda-name>-tree_and_kernel_names.py`) and per-kernel solution snippets (`<cuda-name>---*.py`) that canonicalize the expected output of the tools.
  - `test_backwards_slicing_agent.py`: agent-level smoke test that runs the backwards slicing agent with the default prompt helper, insists on a `/tmp`-only write policy, and ensures the invocation completes after the model writes and executes a Python script that imports `treesitter_tools.cst_utils`.
  - `test_cuda_kernel_uniqueness.py`: extra guardrail that exercises `extract_kernel_source_definition` to ensure every tracked kernel has the right number of definitions.
  - `test_sqlite_reader.py`: ensures the new `helper-scripts/sqlite_reader.py` helper can deserialize `unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite`, list thread IDs, and print stored messages/pending writes for debugging agent runs.
- Root-level helpers
  - `run_tests.sh`: convenience wrapper for running the full regression suite.
  - `Dockerfile`, `README.md`, `AGENTS.md`: project-level documentation and packaging entrypoints.
