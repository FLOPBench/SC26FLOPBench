from __future__ import annotations

import functools
import importlib.util
import re
import runpy
from pathlib import Path
from typing import Any

# Each CUDA benchmark adds a small helper module under
# `unit-tests/extracted-kernel-solutions/<cuda-name>-solutions/`
# named `<cuda-name>-tree_and_kernel_names.py`. Those helpers export
# `EXPECTED_TREE` and `EXPECTED_KERNELS` so the main test file can
# keep the hard-coded metadata with the extraction solutions themselves.
# The helper modules are imported via `_load_expected_tree_and_kernel_names`,
# which loads the tree/kernel expectations alongside the per-kernel
# solution scripts (`<cuda-name>---*.py`).
#
# To add tests for a new CUDA benchmark:
#   * create `<cuda-name>-tree_and_kernel_names.py` with `EXPECTED_TREE`
#     (string) and `EXPECTED_KERNELS` (list of dicts) inside the
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


def _load_expected_tree_and_kernel_names(cuda_name: str) -> tuple[str, list[dict[str, Any]]]:
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
    tree = getattr(module, "EXPECTED_TREE", None)
    kernels = getattr(module, "EXPECTED_KERNELS", None)
    if not isinstance(tree, str):
        raise AssertionError(f"{metadata_path} must define EXPECTED_TREE")
    if not isinstance(kernels, list):
        raise AssertionError(f"{metadata_path} must define EXPECTED_KERNELS")
    return tree, kernels


_EXPECTED_LULESH_TREE, _EXPECTED_LULESH_KERNELS = _load_expected_tree_and_kernel_names("lulesh-cuda")
_EXPECTED_TSNE_TREE, _EXPECTED_TSNE_KERNELS = _load_expected_tree_and_kernel_names("tsne-cuda")
_EXPECTED_ALL_PAIRS_TREE, _EXPECTED_ALL_PAIRS_KERNELS = _load_expected_tree_and_kernel_names("all-pairs-distance-cuda")
_EXPECTED_ADD_BIAS_TREE, _EXPECTED_ADD_BIAS_KERNELS = _load_expected_tree_and_kernel_names("addBiasResidualLayerNorm-cuda")
_EXPECTED_MULTIMATERIAL_TREE, _EXPECTED_MULTIMATERIAL_KERNELS = _load_expected_tree_and_kernel_names("multimaterial-cuda")
_EXPECTED_ATOMIC_REDUCTION_TREE, _EXPECTED_ATOMIC_REDUCTION_KERNELS = _load_expected_tree_and_kernel_names("atomicReduction-cuda")
_EXPECTED_GMM_TREE, _EXPECTED_GMM_KERNELS = _load_expected_tree_and_kernel_names("gmm-cuda")
_EXPECTED_PARTICLEFILTER_TREE, _EXPECTED_PARTICLEFILTER_KERNELS = _load_expected_tree_and_kernel_names("particlefilter-cuda")
_EXPECTED_ERT_TREE, _EXPECTED_ERT_KERNELS = _load_expected_tree_and_kernel_names("ert-cuda")
_EXPECTED_BMF_TREE, _EXPECTED_BMF_KERNELS = _load_expected_tree_and_kernel_names("bmf-cuda")
_EXPECTED_MINIFE_TREE, _EXPECTED_MINIFE_KERNELS = _load_expected_tree_and_kernel_names("miniFE-cuda")

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


@functools.lru_cache(maxsize=1)
def _load_tools() -> tuple[Any, Any, Any, Any]:
    """Load the LangChain tools defined in `mcp-servers/code_search_tools.py`."""
    root = Path(__file__).resolve().parents[1]
    tool_path = root / "mcp-servers" / "code_search_tools.py"
    spec = importlib.util.spec_from_file_location("code_search_tools", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return (
        module.cuda_file_tree,
        module.cuda_global_functions,
        module.cuda_compile_commands,
        module.extract_kernel_source_definition,
    )


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


def test_lulesh_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "lulesh-cuda"}) == _EXPECTED_LULESH_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "lulesh-cuda"}) == _EXPECTED_LULESH_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "lulesh-cuda"})
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
            {"cuda_name": "lulesh-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_tsne_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "tsne-cuda"}) == _EXPECTED_TSNE_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "tsne-cuda"}) == _EXPECTED_TSNE_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "tsne-cuda"})
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
            {"cuda_name": "tsne-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_all_pairs_distance_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "all-pairs-distance-cuda"}) == _EXPECTED_ALL_PAIRS_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "all-pairs-distance-cuda"}) == _EXPECTED_ALL_PAIRS_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"cuda_name": "all-pairs-distance-cuda"}),
        "all-pairs-distance-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("all-pairs-distance-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ALL_PAIRS_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"cuda_name": "all-pairs-distance-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_addBiasResidualLayerNorm_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "addBiasResidualLayerNorm-cuda"}) == _EXPECTED_ADD_BIAS_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "addBiasResidualLayerNorm-cuda"}) == _EXPECTED_ADD_BIAS_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"cuda_name": "addBiasResidualLayerNorm-cuda"}),
        "addBiasResidualLayerNorm-cuda",
        {"main.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("addBiasResidualLayerNorm-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ADD_BIAS_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"cuda_name": "addBiasResidualLayerNorm-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_multimaterial_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "multimaterial-cuda"}) == _EXPECTED_MULTIMATERIAL_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "multimaterial-cuda"}) == _EXPECTED_MULTIMATERIAL_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"cuda_name": "multimaterial-cuda"}),
        "multimaterial-cuda",
        {"compact.cu", "full_matrix.cu", "multimat.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("multimaterial-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_MULTIMATERIAL_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"cuda_name": "multimaterial-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_atomic_reduction_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "atomicReduction-cuda"}) == _EXPECTED_ATOMIC_REDUCTION_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "atomicReduction-cuda"}) == _EXPECTED_ATOMIC_REDUCTION_KERNELS
    _assert_compile_entries(
        compile_commands_extractor_tool.run({"cuda_name": "atomicReduction-cuda"}),
        "atomicReduction-cuda",
        {"reduction.cu"},
    )
    kernel_solutions_map = _load_kernel_solutions("atomicReduction-cuda")
    expected_kernels = {entry["kernel"] for entry in _EXPECTED_ATOMIC_REDUCTION_KERNELS}
    assert set(kernel_solutions_map) == expected_kernels
    for kernel, expected_sources in kernel_solutions_map.items():
        extracted = source_extractor_tool.run(
            {"cuda_name": "atomicReduction-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_gmm_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "gmm-cuda"}) == _EXPECTED_GMM_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "gmm-cuda"}) == _EXPECTED_GMM_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "gmm-cuda"})
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
            {"cuda_name": "gmm-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_particlefilter_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "particlefilter-cuda"}) == _EXPECTED_PARTICLEFILTER_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "particlefilter-cuda"}) == _EXPECTED_PARTICLEFILTER_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "particlefilter-cuda"})
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
            {"cuda_name": "particlefilter-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_ert_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "ert-cuda"}) == _EXPECTED_ERT_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "ert-cuda"}) == _EXPECTED_ERT_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "ert-cuda"})
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
            {"cuda_name": "ert-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_bmf_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "bmf-cuda"}) == _EXPECTED_BMF_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "bmf-cuda"}) == _EXPECTED_BMF_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "bmf-cuda"})
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
            {"cuda_name": "bmf-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)


def test_minife_cuda_tools():
    file_list_tree_tool, cuda_kernel_functions_identifier_tool, compile_commands_extractor_tool, source_extractor_tool = _load_tools()
    assert file_list_tree_tool.run({"cuda_name": "miniFE-cuda"}) == _EXPECTED_MINIFE_TREE
    assert cuda_kernel_functions_identifier_tool.run({"cuda_name": "miniFE-cuda"}) == _EXPECTED_MINIFE_KERNELS
    compile_result = compile_commands_extractor_tool.run({"cuda_name": "miniFE-cuda"})
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
            {"cuda_name": "miniFE-cuda", "kernel_name": kernel}
        )
        extracted_sources = [_normalize_kernel_source(entry["source"]) for entry in extracted]
        _assert_source_lists_equal(expected_sources, extracted_sources)
