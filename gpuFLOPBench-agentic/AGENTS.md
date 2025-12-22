# AGENTS

## 1) Your role
- Document the code-search tools and unit-test conventions so future contributors can quickly understand what is available and how to extend it.
- Treat `mcp-servers/code-search-tools` as the single source of truth for CUDA tooling and `unit-tests/` as the verification batch for those tools plus some ancillary checks; anybody touching either area should update this file as part of new work.
- Whenever you describe a tool or a test, mention both the intent (what it is supposed to do) and the concrete module/path so readers can immediately find the implementation.
- The repository layout is deliberate: `gpuFLOPBench/` holds the `*-cuda` benchmarks under `src/`, `mcp-servers/` exposes LangChain agents (including `code-search-tools`) and shared helpers, `unit-tests/` houses the regression suite plus the extracted-kernel-solution data, and higher-level helpers like `agents/` and `helper-scripts/` sit alongside `run_tests.sh` for project-wide operations.

## 1.1) Agent guidance
- Agents live in `agents/` and should be documented whenever new tooling or workflows depend on them.
- Each agent must clearly state which LangChain tools it consumes (list the concrete tool entrypoints such as `code_search_tools.cuda_file_tree.cuda_file_tree`) and how it wires them into its middleware/checkpointer. Mention any helper modules it depends on (e.g., `agents/llm_models.py`).
- When adding a new agent, explain how callers should construct the LLM(s) (e.g., via `agents/llm_models.build_configurable_llm` or similar factory helpers) so that consumers know the expected credentials/overrides. Include notes about checkpointing (e.g., SQLite saver) if relevant.
- Document the expected runtime behavior: what prompts it requires, which state schema it uses, and how middleware (logging, tool/model call limits, error handling) is configured.

## 2) Adding new code search tools
- The tools live under `mcp-servers/code-search-tools`. Each module loads the shared `utils.py` helpers, defines a `langchain.tools.tool` decorated function, and exposes a single entrypoint that takes a `cuda_name` (via `CudaSubdirArgs`) so it cannot escape `gpuFLOPBench/src`.
- Current tools:
  - `cuda_file_tree` (`cuda-file-tree.py`): Builds a sorted, indented tree of the requested `*-cuda` directory so callers know the layout before inspecting files.
  - `cuda_global_functions` (`cuda-global-functions.py`): Scans CUDA/CPP/HEADER files for `__global__` definitions and emits kernel names with file/line coordinates.
  - `cuda_compile_commands` (`cuda-compile-commands.py`): Reads `gpuFLOPBench/cuda-profiling/compile_commands.json`, filters for the benchmark, and reports compiler/arguments/output pairs for every source mentioned.
  - `cuda_main_files` (`cuda-main-files.py`): Searches for free-function `main()` definitions in the benchmark tree so integration tests can later confirm which files act as entry points.
  - `extract_kernel_source_definition` (`extract-kernel-source-definition.py`): Replays the `__global__` definitions (including template headers and qualifiers) for a particular kernel name, letting consumers compare against known source snapshots.
- To add a new tool:
  1. Follow the established pattern: import the shared helpers (`utils.py`), define any additional argument model(s), and wrap the functionality in a `@tool` function that validates the `cuda_name`.
  2. Update `_CODE_SEARCH_TOOL_SPECS` in `unit-tests/test_backwards_slicing_agent.py` if the backwards-slicing agent should load the new tool so the agent still wires up to all available helpers.
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
- When adding a new benchmark/test case:
  1. Create the `<cuda-name>-tree_and_kernel_names.py` metadata (string tree, list of kernel dictionaries, main file list).
 2. Add `<cuda-name>---<kernel>.py` files that expose their `solution` list so `_normalize_kernel_source` can compare against the extractor output.
 3. Register the new benchmark in `_EXPECTED_*` maps inside `check_code_search_tools.py`.
 4. If the benchmark ought to run through the backwards-slicing agent, ensure `test_backwards_slicing_agent.py` can see any new tools it depends on.
- Additional unit tests:
  - `unit-tests/test_backwards_slicing_agent.py` loads all LangChain tools via `_CODE_SEARCH_TOOL_SPECS`, spins up the backwards-slicing agent with placeholder prompts, and confirms the agent runs at least once (persisting state in `test_backwards_slicing_with_llm_checkpoint.sqlite`).
  - `unit-tests/test_cuda_kernel_uniqueness.py` exercises `extract_kernel_source_definition` directly to assert every kernel of interest has a specific number of definitions (useful for catching duplicates or missing extractions).
  - `unit-tests/check_openrouter_api.py` ensures the OpenRouter LLM in `agents/llm_models.py` is reachable and returns a sensible reply; it requires `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in the environment.
- Run the suite via `python -m pytest -vv -s ./unit-tests/check_code_search_tools.py` (and the other pytest targets) whenever you touch the tools or their data to guarantee there are no regressions.

## 4) Agent unit tests
- `unit-tests/test_backwards_slicing_agent.py` is the exemplar. It loads every tool listed in `_CODE_SEARCH_TOOL_SPECS`, builds an LLM by calling the helpers in `agents/llm_models.py`, deletes `test_backwards_slicing_with_llm_checkpoint.sqlite` before creating the checkpointer (to prevent unbounded growth), and instantiates `agents.backwards_slicing_agent.make_backwards_slicing_agent` (including middleware/checkpointer). The test runs a single invocation with placeholder prompts and asserts a response message, ensuring the agent wiring and tool access stay healthy.
- Add similar tests whenever a new agent is introduced: load only the tools the agent actually needs, create the required prompts/state, and exercise at least one tool call if the agent is expected to perform work. Keep the SQLite checkpoint driver or another durable saver around, and delete or reset its backing file before each run so the artifact does not keep growing between test runs.
