# Summary: Deep Agents Backward Slicing for C++/CUDA/OpenMP using Tree-sitter (No clang/LLVM)

## 1. Objective

Build a static-conservative backward slice for a C++ program containing CUDA and/or OpenMP.
The slice is defined by a *criterion* at a specific file+line:
- CUDA: a specific kernel launch expression `kernel<<<...>>>(...)` on that line
- OpenMP: a specific `#pragma omp ...` region on that line

The output is:
- `slice.hpp`: a single header containing all required project headers (inlined) and required declarations/definitions, while preserving external/system includes as `#include <...>` or `#include "..."` for non-project headers.
- `slice.cpp`: a single source containing all required function definitions to compile and execute the original workload up to and through the criterion.
- `slice_report.json`: a structured report capturing included symbols/files and the reason/confidence for inclusion.

Constraints:
- Must be **static conservative** (valid for a wide range of runtime inputs)
- Must preserve computation semantics (“same workload” as original)
- The slicing agent **must not execute** the program
- For CUDA slicing, the slice must include:
  - the host function containing the launch
  - the kernel definition of the launched `__global__` function
  - all transitively reachable `__device__` functions called by the kernel (device closure)
- For OpenMP slicing, the slice must include:
  - the enclosing host function containing the OpenMP region
  - the OpenMP region’s structured block/loop
  - conservative dependencies that influence variables used within that region, including `target` map/to/from semantics

## 2. Key Design Choice: Tree-sitter CST + Minimal Tool Surface

Instead of relying on clang/LLVM for AST/semantic analysis, the approach relies on:
- tree-sitter + tree-sitter-cuda to parse each source/header into a Concrete Syntax Tree (CST)
- subagents that write and execute short Python scripts via a shell tool
- a shared helper library (`cst_utils.py`) to keep scripts short, consistent, and deterministic

This reduces “tool sprawl”: rather than exposing many specialized tools, expose only:
- filesystem operations
- shell execution (Python scripts)
- optional compile_db helper (or parse in scripts)

All structural analysis (callsites, def-use, criterion spans) is implemented by scripts using `cst_utils`.

## 3. Overall Architecture

A parent Deep Agent orchestrates a pipeline of subagents. Each subagent:
- receives structured state inputs
- generates/executes Python scripts using tree-sitter via shell
- returns structured JSON outputs
- updates a shared `AgentState`

Primary subagents:
1. BuildContextAgent
2. IndexAgent
3. CriterionAgent
4. SummaryAgent Pool (batched summaries)
5. CallGraphAgent
6. DataFlowAgent (fixed-point slicer)
7. MaterializerAgent

The pipeline is summary-driven:
- SummaryAgent computes per-function summaries (callsites + def-use)
- CallGraphAgent builds conservative call graph edges from summaries + symbol index
- DataFlowAgent performs deterministic fixed-point closure on data dependencies + call dependencies
This is stronger than a naive “call tree only” approach and supports global/shared state changes.

## 4. Data Structures / State

The state contains:
- BuildContext: compile_commands-derived include paths/defines and header→representative TU mapping
- SymbolIndex: symbol table of defs/decls, inheritance graph, CUDA attrs, file→symbols mapping
- SliceCriterion:
  - CUDA: launch span + enclosing host function + kernel candidates + seed identifiers
  - OpenMP: pragma span + associated statement span + seed identifiers + clause strings
- FunctionSummaries: per function, callsites + def-use (reads/writes/escapes/addr_taken)
- CallGraph: edges annotated as must/may and direct/virtual/funcptr/cuda/omp
- SliceSet: selected functions/types/files/headers + `why` reasons and confidence
- Artifacts: paths to slice.hpp/slice.cpp/slice_report.json

Symbol identity is stable and deterministic:
- includes template parameter lists on templated definitions (`template<...>`)
- signature text normalized (whitespace/punctuation tightened)
- suffixed with file+line location to avoid collisions without semantic resolution

## 5. Subagent Roles in Detail

### 5.1 BuildContextAgent
Inputs:
- compile_commands.json
- project root directory

Responsibilities:
- parse compilation database entries into a normalized TUCommand structure
- extract include paths (`-I`, `-isystem`), defines (`-D`), language flags, CUDA/OpenMP flags
- compute header → representative TU candidates:
  - approximate include relationships by scanning `#include` lines and resolving using TU include paths
  - rank candidate TUs by proximity and inclusion likelihood
Outputs:
- BuildContext, including header→TUs mapping

### 5.2 IndexAgent
Inputs:
- project file list
- tree-sitter parsing via scripts

Responsibilities:
- build a syntax-based symbol index:
  - function/method definitions + spans
  - classes/structs + method membership
  - namespace nesting (best-effort)
  - inheritance graph (Base → Derived)
  - CUDA attributes on function definitions (`__global__`, `__device__`, etc.)
Outputs:
- SymbolIndex used to resolve call targets and compute override sets conservatively

### 5.3 CriterionAgent
Inputs:
- criterion kind + (file, line)
- SymbolIndex

Responsibilities:
- produce an unambiguous machine slice seed:
  - CUDA:
    - locate the exact kernel launch node at that line using CST
    - extract kernel name token text, launch config span, arg span, full launch span
    - find enclosing host function by walking up CST
    - seed identifiers from launch config and kernel argument expressions
    - resolve kernel symbol candidates using the index (best-effort)
    - mark kernel + enclosing host function as must-include
    - mark device closure required
  - OpenMP:
    - locate `#pragma omp` line near target line (lexical scan)
    - attach to next structured statement node (loop/body) using CST
    - parse directive kind and clause list conservatively
    - seed identifiers from the associated statement/body span
    - mark enclosing function as must-include
Outputs:
- SliceCriterion object with spans + seeds

### 5.4 SummaryAgent Pool
Inputs:
- a batch list of SymbolIds to summarize
- ability to extract function definitions by span or by index lookup

Responsibilities:
- for each function:
  - enumerate callsites inside the function using CST:
    - direct calls, method calls, CUDA launches
    - conservatively label likely virtual calls
    - detect unresolved call expressions and label as funcptr_may
    - capture template argument spellings at callsites when present
  - compute conservative def-use summary:
    - reads/writes via assignment/update/compound assignment nodes
    - addr_taken via unary `&` patterns
    - escapes via returns/passing address to unresolved calls/storing patterns (syntax-only)
Outputs:
- FunctionSummary objects, sorted and deterministic

### 5.5 CallGraphAgent
Inputs:
- FunctionSummaries
- SymbolIndex
- SliceCriterion

Responsibilities:
- build a conservative call graph:
  - direct calls resolved by name to candidate symbol defs
  - method calls resolved by class membership when possible
  - virtual dispatch:
    - include all overrides across derived classes found in index
    - represent edges as `may` edges with reasons
  - function pointer calls:
    - attempt to find assignments to the function pointer in scope via CST queries
    - if unresolved, fallback to broader candidate sets (low confidence)
  - CUDA:
    - host enclosing function → kernel entry edge
  - OpenMP:
    - represent region as special edge or node; region’s internal calls already appear in summaries
Outputs:
- CallGraph with typed edges and confidence annotations

### 5.6 DataFlowAgent (Deterministic Fixed-Point Slicer)
Inputs:
- SliceCriterion
- CallGraph
- FunctionSummaries
- SymbolIndex

Core principle:
- Solve for a conservative closure of:
  - IncludedFunctions
  - NeededRegions (coarse memory/value “regions” derived from identifiers)
by repeatedly propagating dependencies until no change (fixed point).

Definitions:
- Regions are coarse keys (e.g., `var:x`, `obj:o`, `arr:a`, `deref:p`, `global:::g`)
- Determinism enforced by sorted iteration and fixed caps/depths.

Algorithm outline:
1. Initialize IncludedFunctions with criterion must-include symbols.
2. Initialize NeededRegions with criterion seed identifiers:
   - CUDA: tokens used in launch config and kernel arguments
   - OMP: tokens used in associated region statement/body (+ map clause vars)
3. Ensure summaries exist for all IncludedFunctions (spawn SummaryAgent batches as needed).
4. Call closure expansion:
   - include callees from call graph (direct/may), using conservative rules and deterministic ordering
5. Data dependency propagation (repeat until stable):
   - if a function writes a NeededRegion, then everything it reads becomes NeededRegion
   - if a function reads a NeededRegion, include conservative producers:
     - prefer callers/reverse-neighborhood first
     - then broader producers found by scanning summaries for writers
   - unresolved callsites cause conservative expansion via escapes/addr_taken
6. CUDA device closure:
   - BFS from kernel entry including all device-callable functions reachable by call edges
7. Emit final SliceSet (functions/files/headers/types) and reasons.

Outputs:
- SliceSet with `why` map for explainability and confidence tags.

### 5.7 MaterializerAgent
Inputs:
- SliceSet
- SymbolIndex (for spans)
- file content and includes (via cst_utils)

Responsibilities:
- construct `slice.hpp`:
  - inline project headers (resolved under project_root)
  - keep external/system headers as `#include <...>` / `#include "..."` unchanged
  - conservatively prune unused declarations/definitions only when confidence is high
  - preserve macros/templates broadly unless proven unused
- construct `slice.cpp`:
  - include required function definitions and related globals
  - avoid ODR collisions when merging multiple TUs:
    - simplest strategy: wrap each original TU chunk in a unique namespace block
    - preserve `static` internal linkage behavior as best as possible
  - keep CUDA/OpenMP syntax verbatim (no rewriting)
- emit `slice_report.json`:
  - included symbols/files and reasons
  - unresolved edges and confidence warnings
Outputs:
- slice.hpp, slice.cpp, slice_report.json

## 6. Why This Works Without clang/LLVM

Tree-sitter provides robust syntactic structure:
- reliable span location for kernel launches and associated OpenMP regions
- robust callsite enumeration and statement classification
- deterministic def-use extraction for conservative summarization

Semantic gaps remain (overload resolution, preprocessing, alias analysis), so the design compensates with:
- conservative over-approximation of call targets (virtual/function pointer)
- conservative region-based dataflow closure
- cautious pruning: remove only what is proven unused with high confidence

## 7. Determinism and Safety

Determinism:
- all sets processed in sorted SymbolId order
- callsites sorted by byte offset
- fixed depth limits and fixed fallback rules

Safety (static-conservative):
- when uncertain, include more
- device closure always included for kernels
- OpenMP capture/mapping conservatively included

## 8. Outputs and Deliverables

- `slice.hpp`: inlined project headers + required declarations/definitions; external includes preserved
- `slice.cpp`: required definitions merged with ODR-safe strategy
- `slice_report.json`: dependency graph explanation and confidence diagnostics



## TODO
1. **Document `backwards_slicing_agent.py` in terms of the new `treesitter` helper graph** – expand the agent notes to describe how `agents/backwards_slicing_agent.py` now orchestrates builds of context, criteria, summaries, and materialization using the `langchain-tools/treesitter-tools/cst_utils.py` helpers (via `treesitter_tools.cst_utils` imports), what prompts/system instructions drive the subagents, how `agents/llm_models.build_configurable_llm` is configured, and how shared `AgentState` sections (BuildContext, SymbolIndex, SliceCriterion, etc.) are seeded so downstream contributors know how the middleware (TodoList/Filesystem/SubAgent/Summarization) and SQLite checkpointer are arranged around these helper calls.
2. **Implement the helper details referenced by `cst_utils` stubs** – deliver the actual logic for `langchain-tools/treesitter-tools/identifiers.py`, `callsite.py`, `includes.py`, and `serialization.py`, keeping the LangChain-facing tool names (`find_cuda_launches_on_line`, `build_omp_region`) in `cst_utils.py` and ensuring each helper exposes the documented APIs (identifier normalization, callsite collection, include resolution, JSON serialization). Keep `datatypes.py`, `caches.py`, `traversal.py`, and `openmp.py` as supporting modules, and export the deterministic helpers (e.g., `collect_identifiers_in_span`, `collect_callsites_in_function`, `resolve_include_to_path`, `callsite_to_json`) so downstream scripts can continue to rely on their stable signatures.
3. **Define how the newly refactored `cst_utils` helpers are exercised and verified** – document which tests (such as `unit-tests/test_cst_utils.py`, the backwards slicing agent smoke test, and any future dataflow/materializer suites) should import `treesitter_tools.cst_utils` (or the helper aliases) to confirm outputs, describe how the unit tests expect `.func` wrappers to remain stable, and describe the handbook for rerunning `python -m pytest unit-tests/test_cst_utils.py` and other suites (`test_backwards_slicing_agent.py`) whenever helper logic changes so determinism is preserved.
