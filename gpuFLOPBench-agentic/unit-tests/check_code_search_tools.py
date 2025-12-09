from __future__ import annotations

import functools
import importlib.util
import re
import runpy
from pathlib import Path
from typing import Any

_EXPECTED_LULESH_TREE = (
    "lulesh-cuda/\n"
    "  lulesh-init.cu\n"
    "  lulesh-util.cu\n"
    "  lulesh-viz.cu\n"
    "  lulesh.cu\n"
    "  lulesh.h\n"
    "  Makefile"
)

_EXPECTED_LULESH_KERNELS = [
    {"file": "lulesh.cu", "kernel": "fill_sig", "line": 686},
    {"file": "lulesh.cu", "kernel": "integrateStress", "line": 699},
    {"file": "lulesh.cu", "kernel": "acc_final_force", "line": 773},
    {"file": "lulesh.cu", "kernel": "hgc", "line": 804},
    {"file": "lulesh.cu", "kernel": "fb", "line": 891},
    {"file": "lulesh.cu", "kernel": "collect_final_force", "line": 1098},
    {"file": "lulesh.cu", "kernel": "accelerationForNode", "line": 1129},
    {
        "file": "lulesh.cu",
        "kernel": "applyAccelerationBoundaryConditionsForNodes",
        "line": 1147,
    },
    {"file": "lulesh.cu", "kernel": "calcVelocityForNodes", "line": 1167},
    {"file": "lulesh.cu", "kernel": "calcPositionForNodes", "line": 1197},
    {"file": "lulesh.cu", "kernel": "calcKinematicsForElems", "line": 1214},
    {"file": "lulesh.cu", "kernel": "calcStrainRates", "line": 1327},
    {
        "file": "lulesh.cu",
        "kernel": "calcMonotonicQGradientsForElems",
        "line": 1356,
    },
    {"file": "lulesh.cu", "kernel": "calcMonotonicQForElems", "line": 1514},
    {"file": "lulesh.cu", "kernel": "applyMaterialPropertiesForElems", "line": 1686},
]

_EXPECTED_TSNE_TREE = (
    "tsne-cuda/\n"
    "  data/\n"
    "    cifar10_faissed/\n"
    "      distances\n"
    "      indices\n"
    "    mnist_faissed/\n"
    "      distances\n"
    "      indices\n"
    "  apply_forces.cu\n"
    "  apply_forces.h\n"
    "  attr_forces.cu\n"
    "  attr_forces.h\n"
    "  common.h\n"
    "  cuda_utils.cu\n"
    "  cuda_utils.h\n"
    "  cxxopts.hpp\n"
    "  debug_utils.cu\n"
    "  debug_utils.h\n"
    "  distance_utils.cu\n"
    "  distance_utils.h\n"
    "  fit_tsne.cu\n"
    "  fit_tsne.h\n"
    "  main.cu\n"
    "  Makefile\n"
    "  math_utils.cu\n"
    "  math_utils.h\n"
    "  matrix_broadcast_utils.cu\n"
    "  matrix_broadcast_utils.h\n"
    "  nbodyfft.cu\n"
    "  nbodyfft.h\n"
    "  options.h\n"
    "  perplexity_search.cu\n"
    "  perplexity_search.h\n"
    "  rep_forces.cu\n"
    "  rep_forces.h\n"
    "  thrust_transform_functions.h"
)

_EXPECTED_TSNE_KERNELS = [
    {"file": "apply_forces.cu", "kernel": "IntegrationKernel", "line": 41},
    {"file": "attr_forces.cu", "kernel": "ComputePijxQijKernelV3", "line": 41},
    {"file": "attr_forces.cu", "kernel": "reduce_sum_kernel", "line": 71},
    {
        "file": "distance_utils.cu",
        "kernel": "PostprocessNeighborIndicesKernel",
        "line": 94,
    },
    {"file": "math_utils.cu", "kernel": "syv2k", "line": 62},
    {
        "file": "matrix_broadcast_utils.cu",
        "kernel": "BroadcastRowVector",
        "line": 46,
    },
    {
        "file": "matrix_broadcast_utils.cu",
        "kernel": "BroadcastColumnVector",
        "line": 65,
    },
    {"file": "nbodyfft.cu", "kernel": "copy_to_fft_input", "line": 50},
    {"file": "nbodyfft.cu", "kernel": "copy_from_fft_output", "line": 72},
    {"file": "nbodyfft.cu", "kernel": "compute_point_box_idx", "line": 94},
    {"file": "nbodyfft.cu", "kernel": "interpolate_device", "line": 128},
    {
        "file": "nbodyfft.cu",
        "kernel": "compute_interpolated_indices",
        "line": 159,
    },
    {"file": "nbodyfft.cu", "kernel": "compute_potential_indices", "line": 193},
    {"file": "nbodyfft.cu", "kernel": "compute_kernel_tilde", "line": 233},
    {
        "file": "nbodyfft.cu",
        "kernel": "compute_upper_and_lower_bounds",
        "line": 259,
    },
    {"file": "nbodyfft.cu", "kernel": "DFT2D1gpu", "line": 285},
    {"file": "nbodyfft.cu", "kernel": "DFT2D2gpu", "line": 307},
    {"file": "nbodyfft.cu", "kernel": "iDFT2D1gpu", "line": 331},
    {"file": "nbodyfft.cu", "kernel": "iDFT2D2gpu", "line": 359},
    {"file": "perplexity_search.cu", "kernel": "ComputePijKernel", "line": 40},
    {"file": "perplexity_search.cu", "kernel": "RowSumKernel", "line": 66},
    {"file": "perplexity_search.cu", "kernel": "NegEntropyKernel", "line": 86},
    {
        "file": "perplexity_search.cu",
        "kernel": "PerplexitySearchKernel",
        "line": 107,
    },
    {
        "file": "rep_forces.cu",
        "kernel": "compute_repulsive_forces_kernel",
        "line": 33,
    },
    {
        "file": "rep_forces.cu",
        "kernel": "compute_chargesQij_kernel",
        "line": 90,
    },
]


_EXPECTED_ALL_PAIRS_TREE = "all-pairs-distance-cuda/\n  main.cu\n  Makefile"
_EXPECTED_ALL_PAIRS_KERNELS = [
    {"file": "main.cu", "kernel": "k1", "line": 37},
    {"file": "main.cu", "kernel": "k2", "line": 66},
    {"file": "main.cu", "kernel": "k3", "line": 131},
]


_EXPECTED_ADD_BIAS_TREE = (
    "addBiasResidualLayerNorm-cuda/\n"
    "  kernels.h\n"
    "  main.cu\n"
    "  Makefile"
)
_EXPECTED_ADD_BIAS_KERNELS = [
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNormV2", "line": 202},
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNorm", "line": 275},
    {
        "file": "kernels.h",
        "kernel": "generalAddBiasResidualPostLayerNorm",
        "line": 327,
    },
]


_EXPECTED_MULTIMATERIAL_TREE = (
    "multimaterial-cuda/\n"
    "  compact.cu\n"
    "  full_matrix.cu\n"
    "  Makefile\n"
    "  multimat.cu\n"
    "  volfrac.dat.tgz"
)
_EXPECTED_MULTIMATERIAL_KERNELS = [
    {"file": "compact.cu", "kernel": "ccc_loop1", "line": 18},
    {"file": "compact.cu", "kernel": "ccc_loop1_2", "line": 60},
    {"file": "compact.cu", "kernel": "ccc_loop2", "line": 80},
    {"file": "compact.cu", "kernel": "ccc_loop2_2", "line": 128},
    {"file": "compact.cu", "kernel": "ccc_loop3", "line": 144},
]


_EXPECTED_ATOMIC_REDUCTION_TREE = (
    "atomicReduction-cuda/\n"
    "  kernels.h\n"
    "  Makefile\n"
    "  reduction.cu"
)
_EXPECTED_ATOMIC_REDUCTION_KERNELS = [
    {"file": "kernels.h", "kernel": "atomic_reduction", "line": 1},
    {"file": "kernels.h", "kernel": "atomic_reduction_v2", "line": 10},
    {"file": "kernels.h", "kernel": "atomic_reduction_v4", "line": 19},
    {"file": "kernels.h", "kernel": "atomic_reduction_v8", "line": 28},
    {"file": "kernels.h", "kernel": "atomic_reduction_v16", "line": 37},
]

_EXPECTED_KERNELS_BY_CUDA = {
    "lulesh-cuda": _EXPECTED_LULESH_KERNELS,
    "tsne-cuda": _EXPECTED_TSNE_KERNELS,
    "all-pairs-distance-cuda": _EXPECTED_ALL_PAIRS_KERNELS,
    "addBiasResidualLayerNorm-cuda": _EXPECTED_ADD_BIAS_KERNELS,
    "multimaterial-cuda": _EXPECTED_MULTIMATERIAL_KERNELS,
    "atomicReduction-cuda": _EXPECTED_ATOMIC_REDUCTION_KERNELS,
}

_SOLUTION_ROOT = Path(__file__).resolve().parent / "extracted-kernel-solutions"
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
    assert len(expected_sources) == len(extracted_sources)
    for expected, extracted in zip(expected_sources, extracted_sources):
        expected_trimmed, extracted_trimmed = _align_trailing_brace_lines(expected, extracted)
        assert extracted_trimmed == expected_trimmed


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
        canonical_kernel = _extract_kernel_name(normalized_sources[0])
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
