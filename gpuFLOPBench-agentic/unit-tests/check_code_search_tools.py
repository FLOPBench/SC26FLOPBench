from __future__ import annotations

import functools
import importlib.util
import re
import runpy
import sys
from pathlib import Path
from typing import Any

# Each CUDA benchmark adds a small helper module under
# `unit-tests/extracted-kernel-solutions/<cuda-name>-solutions/`
# named `<cuda-name>-tree_and_kernel_names.py`. Those helpers export
# `EXPECTED_TREE`, `EXPECTED_KERNELS`, and `EXPECTED_MAIN_FILES` so the main test
# file can keep the hard-coded metadata with the extraction solutions themselves.
# The helper modules are imported via `_load_expected_tree_and_kernel_names`,
# which loads the tree/kernel expectations alongside the per-kernel
# solution scripts (`<cuda-name>---*.py`).
#
# To add tests for a new CUDA benchmark:
#   * create `<cuda-name>-tree_and_kernel_names.py` with `EXPECTED_TREE`
#     (string), `EXPECTED_KERNELS` (list of dicts), and `EXPECTED_MAIN_FILES`
#     (list of strings) inside the
#     appropriate `extracted-kernel-solutions/<cuda-name>-solutions/`.
#   * add the usual `<cuda-name>---<kernel>.py` solution files under the
#     same directory to capture each kernel's source snippet.
#   * import the metadata in this module by calling
#     `_load_expected_tree_and_kernel_names("<cuda-name>")` so the shared
#     constants can be reused by the assertions.
#   * add a new `test_<cuda-name>_cuda_tools()` that follows the established
#     pattern: load the four LangChain tools (`cuda_file_tree`,
#     `cuda_global_functions`, `cuda_compile_commands`,
#     `extract_kernel_source_definition`), assert the tree and kernel lists,
#     verify compile commands, and compare extracted sources against the
#     solutions directory.
#   * For reference, see the existing test functions below, along with their
#     corresponding files in the extracted-kernel-solutions subdirectory.
#
# To test whether these unit tests pass, you can use the following command:
#   python -m pytest -vv -s ./unit-tests/check_code_search_tools.py

_SOLUTION_ROOT = Path(__file__).resolve().parent / "extracted-kernel-solutions"
_TOOL_DIR = Path(__file__).resolve().parents[1] / "langchain-tools" / "code-search-tools"
_GPU_SRC_ROOT = Path(__file__).resolve().parents[1] / "gpuFLOPBench" / "src"


@functools.lru_cache(maxsize=None)
def _load_solution_metadata_module(cuda_name: str) -> Any:
    solution_dir = _SOLUTION_ROOT / f"{cuda_name}-solutions"
    metadata_path = solution_dir / f"{cuda_name}-tree_and_kernel_names.py"
    spec = importlib.util.spec_from_file_location(
        f"{cuda_name.replace('-', '_')}_tree_and_kernel_names",
        metadata_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import metadata for {cuda_name} from {metadata_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_expected_tree_and_kernel_names(cuda_name: str) -> tuple[str, list[dict[str, Any]]]:
    module = _load_solution_metadata_module(cuda_name)
    solution_dir = _SOLUTION_ROOT / f"{cuda_name}-solutions"
    metadata_path = solution_dir / f"{cuda_name}-tree_and_kernel_names.py"
    tree = getattr(module, "EXPECTED_TREE", None)
    kernels = getattr(module, "EXPECTED_KERNELS", None)
    if not isinstance(tree, str):
        raise AssertionError(f"{metadata_path} must define EXPECTED_TREE")
    if not isinstance(kernels, list):
        raise AssertionError(f"{metadata_path} must define EXPECTED_KERNELS")
    return tree, kernels


def _load_expected_main_files(cuda_name: str) -> list[str]:
    module = _load_solution_metadata_module(cuda_name)
    solution_dir = _SOLUTION_ROOT / f"{cuda_name}-solutions"
    metadata_path = solution_dir / f"{cuda_name}-tree_and_kernel_names.py"
    main_files = getattr(module, "EXPECTED_MAIN_FILES", None)
    if not isinstance(main_files, list):
        raise AssertionError(f"{metadata_path} must define EXPECTED_MAIN_FILES")
    if not all(isinstance(item, str) for item in main_files):
        raise AssertionError(f"{metadata_path} EXPECTED_MAIN_FILES must be a list of strings")
    return main_files


def _load_expected_include_trees(cuda_name: str) -> dict[str, str]:
    module = _load_solution_metadata_module(cuda_name)
    solution_dir = _SOLUTION_ROOT / f"{cuda_name}-solutions"
    metadata_path = solution_dir / f"{cuda_name}-tree_and_kernel_names.py"
    include_trees = getattr(module, "EXPECTED_INCLUDE_TREES", None)
    if not isinstance(include_trees, dict):
        raise AssertionError(f"{metadata_path} must define EXPECTED_INCLUDE_TREES")
    normalized: dict[str, str] = {}
    for key, value in include_trees.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise AssertionError(f"{metadata_path} EXPECTED_INCLUDE_TREES must map strings to strings")
        normalized[key] = value.rstrip("\n")
    return normalized


def _load_expected_include_tree(cuda_name: str, file_name: str) -> str:
    include_trees = _load_expected_include_trees(cuda_name)
    tree = include_trees.get(file_name)
    if tree is None:
        raise AssertionError(f"EXPECTED_INCLUDE_TREES in {cuda_name} does not contain {file_name!r}")
    return tree


def _load_expected_function_entries(cuda_name: str, attribute: str) -> dict[str, str]:
    module = _load_solution_metadata_module(cuda_name)
    solution_dir = _SOLUTION_ROOT / f"{cuda_name}-solutions"
    metadata_path = solution_dir / f"{cuda_name}-tree_and_kernel_names.py"
    entries = getattr(module, attribute, None)
    if not isinstance(entries, dict):
        raise AssertionError(f"{metadata_path} must define {attribute} as a dict[str, str]")
    normalized: dict[str, str] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise AssertionError(f"{metadata_path} {attribute} must map strings to strings")
        normalized[key] = value
    return normalized


def _resolve_cuda_source_file(cuda_name: str, file_name: str) -> Path:
    cuda_root = (_GPU_SRC_ROOT / cuda_name).resolve()
    if not cuda_root.exists() or not cuda_root.is_dir():
        raise AssertionError(f"{cuda_root} is not a valid CUDA benchmark directory")
    target_path = (cuda_root / file_name).resolve()
    try:
        target_path.relative_to(cuda_root)
    except ValueError:
        raise AssertionError(f"{file_name!r} escapes {cuda_root}")
    if not target_path.exists() or not target_path.is_file():
        raise AssertionError(f"{target_path} does not exist")
    return target_path


def _resolve_cuda_directory(cuda_name: str) -> Path:
    cuda_root = (_GPU_SRC_ROOT / cuda_name).resolve()
    if not cuda_root.exists() or not cuda_root.is_dir():
        raise AssertionError(f"{cuda_root} is not a valid CUDA benchmark directory")
    try:
        cuda_root.relative_to(_GPU_SRC_ROOT)
    except ValueError:
        raise AssertionError(f"{cuda_root} escapes {_GPU_SRC_ROOT}")
    return cuda_root


def _load_expected_function_definitions(cuda_name: str) -> dict[str, str]:
    return _load_expected_function_entries(cuda_name, "EXPECTED_FUNCTION_DEFINITIONS")


def _load_expected_function_declarations(cuda_name: str) -> dict[str, str]:
    return _load_expected_function_entries(cuda_name, "EXPECTED_FUNCTION_DECLARATIONS")


_EXPECTED_LULESH_TREE, _EXPECTED_LULESH_KERNELS = _load_expected_tree_and_kernel_names("lulesh-cuda")
_EXPECTED_LULESH_MAIN_FILES = _load_expected_main_files("lulesh-cuda")

_EXPECTED_TSNE_TREE, _EXPECTED_TSNE_KERNELS = _load_expected_tree_and_kernel_names("tsne-cuda")
_EXPECTED_TSNE_MAIN_FILES = _load_expected_main_files("tsne-cuda")

_EXPECTED_ALL_PAIRS_TREE, _EXPECTED_ALL_PAIRS_KERNELS = _load_expected_tree_and_kernel_names("all-pairs-distance-cuda")
_EXPECTED_ALL_PAIRS_MAIN_FILES = _load_expected_main_files("all-pairs-distance-cuda")

_EXPECTED_ADD_BIAS_TREE, _EXPECTED_ADD_BIAS_KERNELS = _load_expected_tree_and_kernel_names("addBiasResidualLayerNorm-cuda")
_EXPECTED_ADD_BIAS_MAIN_FILES = _load_expected_main_files("addBiasResidualLayerNorm-cuda")

_EXPECTED_MULTIMATERIAL_TREE, _EXPECTED_MULTIMATERIAL_KERNELS = _load_expected_tree_and_kernel_names("multimaterial-cuda")
_EXPECTED_MULTIMATERIAL_MAIN_FILES = _load_expected_main_files("multimaterial-cuda")

_EXPECTED_ATOMIC_REDUCTION_TREE, _EXPECTED_ATOMIC_REDUCTION_KERNELS = _load_expected_tree_and_kernel_names("atomicReduction-cuda")
_EXPECTED_ATOMIC_REDUCTION_MAIN_FILES = _load_expected_main_files("atomicReduction-cuda")

_EXPECTED_GMM_TREE, _EXPECTED_GMM_KERNELS = _load_expected_tree_and_kernel_names("gmm-cuda")
_EXPECTED_GMM_MAIN_FILES = _load_expected_main_files("gmm-cuda")

_EXPECTED_PARTICLEFILTER_TREE, _EXPECTED_PARTICLEFILTER_KERNELS = _load_expected_tree_and_kernel_names("particlefilter-cuda")
_EXPECTED_PARTICLEFILTER_MAIN_FILES = _load_expected_main_files("particlefilter-cuda")

_EXPECTED_ERT_TREE, _EXPECTED_ERT_KERNELS = _load_expected_tree_and_kernel_names("ert-cuda")
_EXPECTED_ERT_MAIN_FILES = _load_expected_main_files("ert-cuda")

_EXPECTED_BMF_TREE, _EXPECTED_BMF_KERNELS = _load_expected_tree_and_kernel_names("bmf-cuda")
_EXPECTED_BMF_MAIN_FILES = _load_expected_main_files("bmf-cuda")

_EXPECTED_MINIFE_TREE, _EXPECTED_MINIFE_KERNELS = _load_expected_tree_and_kernel_names("miniFE-cuda")
_EXPECTED_MINIFE_MAIN_FILES = _load_expected_main_files("miniFE-cuda")

_EXPECTED_KERNELS_BY_CUDA = {
    "lulesh-cuda": _EXPECTED_LULESH_KERNELS,
    "tsne-cuda": _EXPECTED_TSNE_KERNELS,
    "all-pairs-distance-cuda": _EXPECTED_ALL_PAIRS_KERNELS,
    "addBiasResidualLayerNorm-cuda": _EXPECTED_ADD_BIAS_KERNELS,
    "multimaterial-cuda": _EXPECTED_MULTIMATERIAL_KERNELS,
    "atomicReduction-cuda": _EXPECTED_ATOMIC_REDUCTION_KERNELS,
    "gmm-cuda": _EXPECTED_GMM_KERNELS,
    "particlefilter-cuda": _EXPECTED_PARTICLEFILTER_KERNELS,
    "ert-cuda": _EXPECTED_ERT_KERNELS,
    "bmf-cuda": _EXPECTED_BMF_KERNELS,
    "miniFE-cuda": _EXPECTED_MINIFE_KERNELS,
}

_EXPECTED_MAIN_FILES_BY_CUDA = {
    "lulesh-cuda": _EXPECTED_LULESH_MAIN_FILES,
    "tsne-cuda": _EXPECTED_TSNE_MAIN_FILES,
    "all-pairs-distance-cuda": _EXPECTED_ALL_PAIRS_MAIN_FILES,
    "addBiasResidualLayerNorm-cuda": _EXPECTED_ADD_BIAS_MAIN_FILES,
    "multimaterial-cuda": _EXPECTED_MULTIMATERIAL_MAIN_FILES,
    "atomicReduction-cuda": _EXPECTED_ATOMIC_REDUCTION_MAIN_FILES,
    "gmm-cuda": _EXPECTED_GMM_MAIN_FILES,
    "particlefilter-cuda": _EXPECTED_PARTICLEFILTER_MAIN_FILES,
    "ert-cuda": _EXPECTED_ERT_MAIN_FILES,
    "bmf-cuda": _EXPECTED_BMF_MAIN_FILES,
    "miniFE-cuda": _EXPECTED_MINIFE_MAIN_FILES,
}

_KERNEL_NAME_RE = re.compile(r"__global__\s+void\s+([A-Za-z0-9_:]+)")


def _normalize_kernel_source(source: str) -> str:
    normalized = source.strip().expandtabs(4)
    while True:
        if normalized.startswith("/*"):
            end = normalized.find("*/", 2)
            if end == -1:
                return normalized
            normalized = normalized[end + 2 :].strip()
            continue
        if normalized.startswith("//"):
            newline = normalized.find("\n", 2)
            if newline == -1:
                return ""
            normalized = normalized[newline + 1 :].strip()
            continue
        break
    return normalized


def _extract_kernel_name(source: str) -> str:
    match = _KERNEL_NAME_RE.search(source)
    if match is None:
        raise AssertionError(f"could not find __global__ definition in {source!r}")
    return match.group(1).split("::")[-1]


def _count_trailing_brace_lines(lines: list[str]) -> int:
    count = 0
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "}":
            count += 1
            continue
        break
    return count


def _align_trailing_brace_lines(expected: str, extracted: str) -> tuple[str, str]:
    expected_lines = expected.splitlines()
    extracted_lines = extracted.splitlines()
    expected_trailing = _count_trailing_brace_lines(expected_lines)
    extracted_trailing = _count_trailing_brace_lines(extracted_lines)
    diff = expected_trailing - extracted_trailing
    if diff > 0:
        expected_lines = expected_lines[: len(expected_lines) - diff]
    elif diff < 0:
        extracted_lines = extracted_lines[: len(extracted_lines) + diff]
    return "\n".join(expected_lines), "\n".join(extracted_lines)


def _assert_source_lists_equal(expected_sources: list[str], extracted_sources: list[str]) -> None:
    def _canonicalize(source: str) -> str:
        lines = source.splitlines()
        while lines and lines[-1].strip() == "}":
            lines.pop()
        return "\n".join(lines).strip()

    expected_counter = {}
    for source in expected_sources:
        canon = _canonicalize(source)
        expected_counter[canon] = expected_counter.get(canon, 0) + 1

    extracted_counter = {}
    for source in extracted_sources:
        canon = _canonicalize(source)
        extracted_counter[canon] = extracted_counter.get(canon, 0) + 1

    assert expected_counter == extracted_counter


def _function_lines_by_kind(output: str, kind: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    kind_suffix = f"({kind})"
    filtered = [line for line in lines if line.endswith(kind_suffix)]
    return "\n".join(filtered)


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _load_kernel_solutions(cuda_name: str) -> dict[str, list[str]]:
    solution_dir = _SOLUTION_ROOT / f"{cuda_name}-solutions"
    solutions: dict[str, list[str]] = {}
    for path in sorted(solution_dir.glob(f"{cuda_name}---*.py")):
        namespace = runpy.run_path(path)
        solution_list = namespace["solution"]
        if not isinstance(solution_list, list):
            raise AssertionError(f"{path} did not define a list called solution")
        normalized_sources = [_normalize_kernel_source(item) for item in solution_list]
        if not normalized_sources:
            raise AssertionError(f"{path} defined an empty solution list")
        canonical_kernel = None
        for source in normalized_sources:
            try:
                canonical_kernel = _extract_kernel_name(source)
                break
            except AssertionError:
                continue
        if canonical_kernel is None:
            stem = path.stem
            prefix = f"{cuda_name}---"
            if stem.startswith(prefix):
                canonical_kernel = stem[len(prefix) :]
            elif "---" in stem:
                canonical_kernel = stem.split("---", 1)[1]
            else:
                raise AssertionError(f"{path} does not contain a __global__ definition")
        solutions[canonical_kernel] = normalized_sources
    return solutions


@functools.lru_cache(maxsize=None)
def _load_tool_module(filename: str, module_name: str) -> Any:
    """Load a LangChain tool module that was moved into the code-search-tools directory."""
    tool_path = _TOOL_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module at {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_tools() -> tuple[Any, Any, Any, Any]:
    """Load the LangChain tools defined in the split tool modules."""
    file_tree_module = _load_tool_module("cuda-file-tree.py", "code_search_tools.cuda_file_tree")
    global_functions_module = _load_tool_module(
        "cuda-global-functions.py",
        "code_search_tools.cuda_global_functions",
    )
    compile_commands_module = _load_tool_module(
        "cuda-compile-commands.py",
        "code_search_tools.cuda_compile_commands",
    )
    source_definition_module = _load_tool_module(
        "extract-kernel-source-definition.py",
        "code_search_tools.extract_kernel_source_definition",
    )
    return (
        file_tree_module.cuda_file_tree,
        global_functions_module.cuda_global_functions,
        compile_commands_module.cuda_compile_commands,
        source_definition_module.extract_kernel_source_definition,
    )


def _load_main_files_tool() -> Any:
    module = _load_tool_module("cuda-main-files.py", "code_search_tools.cuda_main_files")
    return module.cuda_main_files


def _load_include_tree_tool() -> Any:
    module = _load_tool_module(
        "include-tree-extractor.py",
        "code_search_tools.include_tree_extractor",
    )
    return module.include_tree_extractor


def _load_function_definitions_tool() -> Any:
    module = _load_tool_module(
        "function-definition-lister.py",
        "code_search_tools.function_definition_lister",
    )
    return module.function_definition_lister


def _assert_compile_entries(result: dict[str, Any], cuda_name: str, expected_files: set[str]) -> None:
    commands = result["commands"]
    assert {entry["file"] for entry in commands} == expected_files
    assert all(entry["output"].endswith(".o") for entry in commands)
    assert all(entry["compiler"].endswith("clang++") for entry in commands)
    # Each command should include the top-level benchmark source path in its include list.
    assert all(
        any(arg.startswith("-I") and f"src/{cuda_name}" in arg for arg in entry["arguments"])
        for entry in commands
    )


def _assert_function_definition_listings(cuda_name: str) -> None:
    function_definitions_tool = _load_function_definitions_tool()
    expected_function_definitions = _load_expected_function_definitions(cuda_name)
    expected_function_declarations = _load_expected_function_declarations(cuda_name)
    all_files = sorted(set(expected_function_definitions) | set(expected_function_declarations))
    total_actual_definitions = 0
    total_expected_definitions = 0
    total_actual_declarations = 0
    total_expected_declarations = 0
    for file_name in all_files:
        file_path = _resolve_cuda_source_file(cuda_name, file_name)
        function_list = function_definitions_tool.run({"file_path": str(file_path)})
        actual_def = _function_lines_by_kind(function_list, "defnt")
        expected_def = expected_function_definitions.get(file_name, "")
        assert actual_def == expected_def
        assert len(actual_def.splitlines()) == len(expected_def.splitlines())

        actual_decl = _function_lines_by_kind(function_list, "decl")
        expected_decl = expected_function_declarations.get(file_name, "")
        assert actual_decl == expected_decl
        assert len(actual_decl.splitlines()) == len(expected_decl.splitlines())

        total_actual_definitions += _count_nonempty_lines(actual_def)
        total_expected_definitions += _count_nonempty_lines(expected_def)
        total_actual_declarations += _count_nonempty_lines(actual_decl)
        total_expected_declarations += _count_nonempty_lines(expected_decl)

    assert total_actual_definitions == total_expected_definitions
    assert total_actual_declarations == total_expected_declarations


def test_lulesh_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    cuda_dir = _resolve_cuda_directory("lulesh-cuda")
    assert file_list_tree_tool.run({"dir_path": str(cuda_dir)}) == _EXPECTED_LULESH_TREE
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(cuda_dir)}) == _EXPECTED_LULESH_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(cuda_dir)})
    _assert_compile_entries(
        compile_result,
        "lulesh-cuda",
        {"lulesh.cu", "lulesh-init.cu", "lulesh-util.cu", "lulesh-viz.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("lulesh-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_LULESH_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(cuda_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)

_assert_function_definition_listings("lulesh-cuda")


def test_include_tree_extractor_lulesh():
    include_tree_tool = _load_include_tree_tool()
    expected_tree = _load_expected_include_tree("lulesh-cuda", "lulesh.cu")
    cuda_dir = _resolve_cuda_directory("lulesh-cuda")
    result = include_tree_tool.run({"file_path": str(cuda_dir / "lulesh.cu")})
    assert result == expected_tree


def test_lulesh_main_files_tool():
    main_files_tool = _load_main_files_tool()
    cuda_dir = _resolve_cuda_directory("lulesh-cuda")
    assert main_files_tool.run({"dir_path": str(cuda_dir)}) == _EXPECTED_LULESH_MAIN_FILES


def test_include_tree_extractor_main_files():
    include_tree_tool = _load_include_tree_tool()
    for cuda_name, main_files in _EXPECTED_MAIN_FILES_BY_CUDA.items():
        expected_trees = _load_expected_include_trees(cuda_name)
        cuda_dir = _resolve_cuda_directory(cuda_name)
        for file_name in main_files:
            expected_tree = expected_trees.get(file_name)
            if expected_tree is None:
                raise AssertionError(
                    f"{cuda_name}-tree_and_kernel_names.py must list {file_name!r} in EXPECTED_INCLUDE_TREES"
                )
            result = include_tree_tool.run({"file_path": str(cuda_dir / file_name)})
            assert result == expected_tree


def test_all_cuda_main_files_tool():
    main_files_tool = _load_main_files_tool()
    for cuda_name, expected_main_files in _EXPECTED_MAIN_FILES_BY_CUDA.items():
        cuda_dir = _resolve_cuda_directory(cuda_name)
        result = main_files_tool.run({"dir_path": str(cuda_dir)})
        assert result == expected_main_files


def test_tsne_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    tsne_dir = _resolve_cuda_directory("tsne-cuda")
    assert file_list_tree_tool.run({"dir_path": str(tsne_dir)}) == _EXPECTED_TSNE_TREE
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(tsne_dir)}) == _EXPECTED_TSNE_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(tsne_dir)})
    _assert_compile_entries(
        compile_result,
        "tsne-cuda",
        {
            "apply_forces.cu",
            "attr_forces.cu",
            "cuda_utils.cu",
            "debug_utils.cu",
            "distance_utils.cu",
            "fit_tsne.cu",
            "main.cu",
            "matrix_broadcast_utils.cu",
            "math_utils.cu",
            "nbodyfft.cu",
            "perplexity_search.cu",
            "rep_forces.cu",
        },
    )
    main_entry = next(entry for entry in compile_result["commands"] if entry["file"] == "main.cu")
    assert any(arg.startswith("-I") and "src/tsne-cuda" in arg and "data" not in arg for arg in main_entry["arguments"])
    assert any(arg.startswith("-I") and "src/tsne-cuda/data" in arg for arg in main_entry["arguments"])
    kernel_solutions_map = _load_kernel_solutions("tsne-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_TSNE_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(tsne_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_tsne_function_definition_listings():
    _assert_function_definition_listings("tsne-cuda")


def test_all_pairs_distance_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    alldir = _resolve_cuda_directory("all-pairs-distance-cuda")
    assert file_list_tree_tool.run({"dir_path": str(alldir)}) == _EXPECTED_ALL_PAIRS_TREE
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(alldir)}) == _EXPECTED_ALL_PAIRS_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"dir_path": str(alldir)}),
        "all-pairs-distance-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("all-pairs-distance-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ALL_PAIRS_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(alldir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_all_pairs_distance_function_definition_listings():
    _assert_function_definition_listings("all-pairs-distance-cuda")


def test_additional_cuda_function_definition_listings():
    for cuda_name in (
        "addBiasResidualLayerNorm-cuda",
        "multimaterial-cuda",
        "atomicReduction-cuda",
        "gmm-cuda",
        "particlefilter-cuda",
        "ert-cuda",
        "bmf-cuda",
        "miniFE-cuda",
    ):
        _assert_function_definition_listings(cuda_name)


def test_addBiasResidualLayerNorm_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    add_bias_dir = _resolve_cuda_directory("addBiasResidualLayerNorm-cuda")
    assert file_list_tree_tool.run({"dir_path": str(add_bias_dir)}) == _EXPECTED_ADD_BIAS_TREE
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(add_bias_dir)}) == _EXPECTED_ADD_BIAS_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"dir_path": str(add_bias_dir)}),
        "addBiasResidualLayerNorm-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("addBiasResidualLayerNorm-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ADD_BIAS_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(add_bias_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_multimaterial_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("multimaterial-cuda"))}) == _EXPECTED_MULTIMATERIAL_TREE
    multimaterial_dir = _resolve_cuda_directory("multimaterial-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(multimaterial_dir)}) == _EXPECTED_MULTIMATERIAL_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"dir_path": str(multimaterial_dir)}),
        "multimaterial-cuda",
        {"compact.cu", "full_matrix.cu", "multimat.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("multimaterial-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_MULTIMATERIAL_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(multimaterial_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_atomic_reduction_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("atomicReduction-cuda"))}) == _EXPECTED_ATOMIC_REDUCTION_TREE
    atomic_dir = _resolve_cuda_directory("atomicReduction-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(atomic_dir)}) == _EXPECTED_ATOMIC_REDUCTION_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"dir_path": str(atomic_dir)}),
        "atomicReduction-cuda",
        {"reduction.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("atomicReduction-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ATOMIC_REDUCTION_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(atomic_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_gmm_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("gmm-cuda"))}) == _EXPECTED_GMM_TREE
    gmm_dir = _resolve_cuda_directory("gmm-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(gmm_dir)}) == _EXPECTED_GMM_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(gmm_dir)})
    _assert_compile_entries(
        compile_result,
        "gmm-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("gmm-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_GMM_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(gmm_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_particlefilter_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("particlefilter-cuda"))}) == _EXPECTED_PARTICLEFILTER_TREE
    particle_dir = _resolve_cuda_directory("particlefilter-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(particle_dir)}) == _EXPECTED_PARTICLEFILTER_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(particle_dir)})
    _assert_compile_entries(
        compile_result,
        "particlefilter-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("particlefilter-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_PARTICLEFILTER_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(particle_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_ert_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("ert-cuda"))}) == _EXPECTED_ERT_TREE
    ert_dir = _resolve_cuda_directory("ert-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(ert_dir)}) == _EXPECTED_ERT_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(ert_dir)})
    _assert_compile_entries(
        compile_result,
        "ert-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("ert-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ERT_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(ert_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_bmf_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("bmf-cuda"))}) == _EXPECTED_BMF_TREE
    bmf_dir = _resolve_cuda_directory("bmf-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(bmf_dir)}) == _EXPECTED_BMF_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(bmf_dir)})
    _assert_compile_entries(
        compile_result,
        "bmf-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("bmf-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_BMF_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(bmf_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_minife_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"dir_path": str(_resolve_cuda_directory("miniFE-cuda"))}) == _EXPECTED_MINIFE_TREE
    minife_dir = _resolve_cuda_directory("miniFE-cuda")
    assert cuda_kernel_functions_identifier_tool.run({"dir_path": str(minife_dir)}) == _EXPECTED_MINIFE_KERNELS
    compile_result = compile_commands_extractor_tool.run({"dir_path": str(minife_dir)})
    _assert_compile_entries(
        compile_result,
        "miniFE-cuda",
        {
            "BoxPartition.cpp",
            "YAML_Doc.cpp",
            "YAML_Element.cpp",
            "main.cpp",
            "mytimer.cpp",
            "param_utils.cpp",
            "utils.cpp",
        },
    )
    kernel_solutions_map = _load_kernel_solutions("miniFE-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_MINIFE_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"file_path": str(minife_dir), "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)
