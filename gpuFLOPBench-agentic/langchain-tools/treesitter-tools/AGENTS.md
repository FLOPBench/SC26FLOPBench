# cst_utils.py

## Overview
A small helper library for tree-sitter-based CST analysis of C++/CUDA/OpenMP code.
Designed so agents can write short Python scripts that parse files, locate criteria
(CUDA launches / OpenMP regions), enumerate callsites, and compute conservative
def-use summaries, without clang/LLVM.

---

## Imports / Globals
- Standard libs: os, re, json, pathlib, functools, dataclasses, typing
- Tree-sitter libs:
  - tree_sitter (Parser, Language)
  - tree_sitter_cpp (if used) / or built-in C++ language binding
  - tree_sitter_cuda (CUDA language binding)
- Global caches:
  - _TREE_CACHE: {(path, mtime_ns, lang): Tree}
  - _TEXT_CACHE: {(path, mtime_ns): str}
  - _LANG_CACHE: {lang: Language}
- Keyword/type filter sets:
  - CPP_KEYWORDS
  - CPP_BUILTIN_TYPES
  - COMMON_MACROS (optional)
- Node type constants (best-effort, depends on grammar):
  - FUNCTION_NODE_TYPES
  - CLASS_NODE_TYPES
  - CALL_NODE_TYPES
  - ASSIGN_NODE_TYPES
  - UPDATE_NODE_TYPES

---

## Data Models (dataclasses)
### Span
- file: str
- start_byte: int
- end_byte: int
- start_point: (row:int, col:int)  # 0-based
- end_point: (row:int, col:int)

### NodeRef
- type: str
- start_byte: int
- end_byte: int
- start_point: (row, col)
- end_point: (row, col)

### Callsite
- kind: "direct" | "method" | "virtual_may" | "funcptr_may" | "cuda_launch"
- span: Span
- callee_text: str
- receiver_text: Optional[str]
- template_args_text: Optional[str]
- arg_span: Optional[Span]
- launch_cfg_span: Optional[Span]

### OmpRegion
- pragma_span: Span
- pragma_text: str
- associated_stmt_span: Span
- kind_text: str
- clauses: List[str]

---

## Text / Path Utilities
### read_text(path) -> str
- reads file, utf-8 with fallback
- caches by (path, mtime_ns)

### normpath(path) -> str
- normalize separators, resolve '..' and '.' (no filesystem resolution required)

### relpath(path, project_root) -> str
- stable relative path (used by SymbolIds/reporting)

### byte_offset_for_line_col(text, line0, col0) -> int
- deterministic mapping from line/col to byte offset

### span_text(text, span) -> str
- returns text[span.start_byte:span.end_byte] with byte-safe handling

---

## Tree-sitter Language + Parsing
### pick_language_for_file(path, default="cpp") -> "cpp" | "cuda"
- .cu/.cuh => cuda else cpp (customizable)

### get_language(lang) -> Language
- loads the requested Language from bindings
- caches in _LANG_CACHE

### parse_file(path, lang, use_cache=True) -> Tree
- reads text
- parses using Parser(Language)
- caches by (path, mtime_ns, lang)

### node_to_ref(node) -> NodeRef
### ref_to_span(path, ref) -> Span

---

## Generic CST Traversal Helpers
### iter_descendants(node) -> Iterator[Node]
### iter_nodes_of_type(tree_or_node, types: Set[str]) -> Iterator[NodeRef]
### node_contains_byte(ref_or_node, byte_offset) -> bool

### find_smallest_enclosing_node(tree, byte_offset, types: Set[str]) -> Optional[NodeRef]
- used by find_enclosing_function and similar

---

## Criterion Location
### find_enclosing_function(tree, byte_offset) -> Optional[NodeRef]
- returns nearest function_definition (or method definition)
- fallback to function_declarator wrapper if needed

### find_cuda_launches_on_line(tree, text, line1) -> List[NodeRef]
- returns CST nodes representing kernel launches that intersect the line
- deterministic order by start_byte

### pick_best_node_at_line(nodes, prefer_smallest=True) -> Optional[NodeRef]
- deterministic: smallest span, then earliest start_byte

### extract_cuda_launch_parts(path, text, launch_ref) -> dict
Returns:
- full_launch_span: Span
- kernel_name_text: str (best-effort)
- kernel_name_span: Optional[Span]
- launch_cfg_span: Optional[Span]  # <<<...>>>
- arg_span: Optional[Span]         # (...), kernel args
- template_args_text/span (if present)

---

## OpenMP Pragmas (Lexical + CST association)
### find_omp_pragmas_near_line(text, line1, window=2) -> List[Span]
- returns spans for lines containing '#pragma omp' near line

### parse_omp_pragma(pragma_line: str) -> (kind_text: str, clauses: List[str])
- normalize directive kind
- extract raw clauses conservatively

### find_statement_after_byte(tree, byte_offset) -> Optional[NodeRef]
- locate next statement/for_statement/compound_statement after pragma byte offset

### build_omp_region(path, text, tree, pragma_span) -> Optional[OmpRegion]
- attaches pragma to next statement span
- populates kind_text + clauses

### find_cuda_launches_on_line(file_path, line, language=None) -> list[dict]
- exposes the CUDA launch helper as a reusable script function, returning JSON-friendly spans and extracted kernel metadata (name, cfg, args, template text).

### build_omp_region(file_path, line, window=2, language=None) -> list[dict]
- exposes the OpenMP region helper as a reusable script function, returning pragma span/kind/clauses plus the associated statement span so agents can invoke it directly from shell-driven Python snippets.

---

## `langchain-tools/treesitter-tools/cst_utils.py` overview
- **Goal:** expose shared Tree-sitter helpers, deterministic span models, and reusable functions so agents can analyze CUDA/OpenMP criteria via short Python scripts without duplicating the built-in filesystem middleware (`ls`, `read_file`, etc.) already provided by `deepagents`.
- **Caches:** `_TEXT_CACHE`, `_TREE_CACHE`, `_LANG_CACHE`, and `_PARSER_CACHE` keep per-file `mtime`/size data so repeated reads/parses stay fast.
- **Exports:** `Span`, `NodeRef`, `Callsite`, `OmpRegion`, traversal helpers, and the helper functions (`find_cuda_launches_on_line`, `build_omp_region`) that scripts can call directly.
- **Future helpers (stubs already present in `cst_utils.py`):**
  * Identifier utilities: `is_identifier_token`, `normalize_ws_punct`, `collect_identifiers_in_span`, `collect_identifiers_in_node`.
  * Symbol metadata / signature builders: `function_signature_text`, `extract_template_param_list`, `build_symbol_id`.
  * Callsite/def-use helpers: `collect_callsites_in_function`, `summarize_def_use`, `find_assignments_in_function`.
  * Include handling: `collect_includes`, `resolve_include_to_path`, `is_project_path`.
  * Serialization helpers: `callsite_to_json`, `ompreg_to_json`, `noderef_to_json`.
  These stubs currently raise `NotImplementedError` and serve as placeholders until their behavior is fully defined, but they ensure the module's API is stable for whoever depends on them next.

## Supporting modules in `langchain-tools/treesitter-tools/`
- `datatypes.py` – defines the immutable `Span`, `NodeRef`, `Callsite`, and `OmpRegion` record types so every helper shares consistent representations.
- `caches.py` – configures the per-file caches (`_TEXT_CACHE`, `_TREE_CACHE`, `_LANG_CACHE`, `_PARSER_CACHE`), path normalization helpers, UTF-8-aware text reading, `parse_file`, and `span_from_bytes` formatting.
- `traversal.py` – tree traversal utilities (`node_to_ref`, `ref_to_span`, `iter_descendants`, `iter_nodes_of_type`, `find_enclosing_function`) plus `find_cuda_launches_on_line` as the shared detector of kernel launch nodes.
- `openmp.py` – OpenMP helpers for locating pragmas (`find_omp_pragmas_near_line`), parsing clause lists (`parse_omp_pragma`), finding the subsequent statement (`find_statement_after_byte`), and building `OmpRegion` objects.
- `identifiers.py` – keyword/builtin/type sets (`CPP_KEYWORDS`, `CPP_BUILTIN_TYPES`, `COMMON_MACROS`) and placeholder functions for identifier normalization, symbol signatures, and `build_symbol_id`.
- `callsite.py` – placeholder entrypoints for callsite collection, def/use summarization, and assignment inference that will be fleshed out when the dataflow pipeline is implemented.
- `includes.py` – stub helpers for lexing `#include` directives, resolving them against include paths, and tagging project vs system headers.
- `serialization.py` – stub serializers (`callsite_to_json`, `ompreg_to_json`, `noderef_to_json`) so downstream agents can emit JSON-safe representations without clang/LLVM.

---

## Identifiers and Normalization
### is_identifier_token(s: str) -> bool
### normalize_ws_punct(s: str) -> str
- whitespace collapse + punctuation tightening for stable IDs

### collect_identifiers_in_span(text, tree, span) -> Set[str]
- collects identifier nodes overlapping span
- filters keywords and builtin types

### collect_identifiers_in_node(text, node_ref) -> Set[str]
- helper used by def-use and region capture

---

## Function Signature / SymbolId Helpers
### function_signature_text(text, func_ref) -> str
- best-effort extraction of signature text
- includes preceding template_parameter_list if present

### extract_template_param_list(text, func_ref) -> Optional[str]
- returns normalized 'template<...>' string if present

### build_symbol_id(file, func_ref, signature_text, project_root=None, cuda_attr=None) -> str
- stable SymbolId format including template params and location suffix

---

## Callsite Extraction
### collect_callsites_in_function(path, text, func_ref) -> List[Callsite]
Detect:
- direct call_expression: foo(...)
- method calls: obj.foo(...), ptr->foo(...)
- virtual_may: conservative method call labeling
- funcptr_may: unresolved callee identifiers that are called
- cuda_launch: kernel<<<...>>>(...)
Extract:
- callee_text (base)
- receiver_text (if method)
- template_args_text (if present)
- arg_span / launch_cfg_span
Sorted deterministically by start_byte.

---

## Def-Use Summarization (Conservative)
### summarize_def_use(path, text, func_ref, global_names: Optional[Set[str]] = None) -> dict[str, Set[str]]
Returns sets:
- reads
- writes
- addr_taken
- escapes
- globals_read (optional)
- globals_written (optional)
Rules:
- assignment/compound_assignment/update => writes
- identifiers in RHS/conditions/args => reads
- &x => addr_taken
- escape heuristics: returned/passed/stored patterns (syntax-based)

### find_assignments_in_function(path, text, func_ref, name) -> Set[str]
- finds RHS identifier candidates assigned to 'name'
- used for function pointer target inference

---

## Includes (Materialization Support)
### collect_includes(text) -> List[(raw_include: str, span: Span)]
- parses #include lines (lexical)
- returns in file order

### resolve_include_to_path(includer_file, raw_include, include_paths, project_root) -> Optional[str]
- resolve "..." using includer dir + include_paths
- if resolved within project_root return path else None

### is_project_path(path, project_root) -> bool

---

## Serialization Helpers
### span_to_json(span) -> dict
### callsite_to_json(callsite) -> dict
### ompreg_to_json(region) -> dict
### noderef_to_json(ref) -> dict

---

## Testing Hooks (optional)
### _debug_dump_tree(tree, text, out_path)
### _self_test_parse(path)

## Agent integration & regression coverage
- `agents/backwards_slicing_agent.default_backwards_slicing_system_prompt` is the canonical way agents learn about `treesitter_tools.cst_utils`: it enumerates the helpers, reiterates the `/tmp`-only write restriction, and reminds callers to use the built-in DeepAgents filesystem/`execute` tools when invoking analysis scripts.
- `unit-tests/test_cst_utils.py` directly imports `treesitter_tools.cst_utils` to verify CUDA launch detection, CUDA/OpenMP region helpers, and real-world coverage on `gpuFLOPBench/src/lulesh-cuda/lulesh.cu` and `gpuFLOPBench/src/lulesh-omp/lulesh.cc`. Keep this test in sync with any helper API changes and rerun it via `python -m pytest unit-tests/test_cst_utils.py`.
- `unit-tests/test_backwards_slicing_agent.py` now instantiates the backwards slicing agent with no LangChain code-search tools, passes the default system prompt, and instructs the model to craft a Python script (written to `/tmp`) that imports `treesitter_tools.cst_utils` to extract `__global__`/`__device__` definitions and OpenMP regions from the lulesh sources. Use this test to guard the prompt wording and the overall script-driven workflow when the agent changes.
- `helper-scripts/sqlite_reader.py` exposes helpers to list LangGraph thread IDs, deserialize checkpoints, and dump the stored `messages` channel plus any pending `writes` from `unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite`; it intentionally omits defensive guards so schema drift surfaces as a failure.
- `unit-tests/test_sqlite_reader.py` exercises the helper, prints the earliest recovered messages along with pending write entries, and runs without `capfd` so `python -m pytest -s unit-tests/test_sqlite_reader.py` streams the output directly to the console when debugging the backwards slicing agent.
