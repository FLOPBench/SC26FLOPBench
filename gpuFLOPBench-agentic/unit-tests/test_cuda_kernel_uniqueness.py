from __future__ import annotations

import functools
from pathlib import Path
import importlib.util

import pytest

from deepagents.backends import FilesystemBackend

_EXTRACT_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "langchain-tools"
    / "code-search-tools"
    / "extract-kernel-source-definition.py"
)
_GPU_SRC_ROOT = Path(__file__).resolve().parents[1] / "gpuFLOPBench" / "src"


@functools.lru_cache(maxsize=1)
def _extract_kernel_source_definition_tool(cuda_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "code_search_tools.extract_kernel_source_definition",
        _EXTRACT_TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load tool at {_EXTRACT_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = FilesystemBackend(root_dir=str(cuda_dir), virtual_mode=False)
    return module.make_extract_kernel_source_definition_tool(backend=backend)


EXPECTED_KERNEL_DEFINITIONS: dict[str, dict[str, int]] = {
    "bmf-cuda": {"initFactor": 2, "computeDistanceRowsShared": 2},
    "crossEntropy-cuda": {"loss_bwd": 2},
    "ert-cuda": {"block_stride": 2},
    "gamma-correction-cuda": {"gamma_correction": 2},
    "md5hash-cuda": {"md5hash_kernel": 2},
    "mf-sgd-cuda": {"init_rand_state": 2},
    "segsort-cuda": {
        "kern_block_sort": 2,
        "kern_block_merge": 2,
        "kern_copy": 2,
        "gen_copy": 2,
        "gen_bk256_wp2_tc1_r2_r2_orig": 2,
        "gen_bk128_wp2_tc2_r3_r4_orig": 2,
        "gen_bk128_wp2_tc4_r5_r8_orig": 2,
        "gen_bk128_wp4_tc4_r9_r16_strd": 2,
        "gen_bk128_wp8_tc4_r17_r32_strd": 2,
        "gen_bk128_wp16_tc4_r33_r64_strd": 2,
        "gen_bk256_wp8_tc16_r65_r128_strd": 2,
        "gen_bk256_wp32_tc8_r129_r256_strd": 2,
        "gen_bk128_tc4_r257_r512_orig": 2,
        "gen_bk256_tc4_r513_r1024_orig": 2,
        "gen_bk512_tc4_r1025_r2048_orig": 2,
    },
    "vmc-cuda": {"initran": 2, "initialize": 2, "propagate": 2},
    "warpsort-cuda": {"sortDevice": 2},
}


def _assert_kernel_counts(cuda_name: str, kernel_counts: dict[str, int]) -> None:
    cuda_dir = _GPU_SRC_ROOT / cuda_name
    extractor_tool = _extract_kernel_source_definition_tool(cuda_dir)
    for kernel_name, expected_count in kernel_counts.items():
        actual_defs = extractor_tool.run(
            {"file_path": str(cuda_dir), "kernel_name": kernel_name}
        )
        actual_count = len(actual_defs)
        if actual_count != expected_count:
            raise AssertionError(
                f"{cuda_name}/{kernel_name}: expected {expected_count} definitions "
                f"but only {actual_count} were extracted."
            )


@pytest.mark.parametrize(
    "cuda_name",
    sorted(EXPECTED_KERNEL_DEFINITIONS),
)
def test_cuda_kernel_uniqueness(cuda_name: str) -> None:
    """Ensure extract_kernel_source_definition returns every unique kernel body."""

    kernel_counts = EXPECTED_KERNEL_DEFINITIONS[cuda_name]
    _assert_kernel_counts(cuda_name, kernel_counts)
