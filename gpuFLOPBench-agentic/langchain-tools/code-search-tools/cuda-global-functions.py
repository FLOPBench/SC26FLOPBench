from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from typing import Iterator, List, Tuple

from langchain.tools import tool

_UTILS_MODULE_NAME = "code_search_tools.utils"


def _load_utils_module() -> object:
    module = sys.modules.get(_UTILS_MODULE_NAME)
    if module is None:
        spec = util.spec_from_file_location(
            _UTILS_MODULE_NAME,
            Path(__file__).resolve().with_name("utils.py"),
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load shared utils module")
        module = util.module_from_spec(spec)
        sys.modules[_UTILS_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module

_utils = _load_utils_module()
CudaSubdirArgs = _utils.CudaSubdirArgs
_resolve_cuda_dir = _utils._resolve_cuda_dir
_gather_cuda_files = _utils._gather_cuda_files
_iterate_cuda_kernel_definitions = _utils._iterate_cuda_kernel_definitions


class CudaGlobalFunctionsArgs(CudaSubdirArgs):
    """Arguments for listing __global__ CUDA functions inside a subdirectory."""


def _extract_cuda_global_definitions(text: str) -> Iterator[Tuple[str, str, int]]:
    for _, qualified, kernel, line, _ in _iterate_cuda_kernel_definitions(text):
        yield qualified, kernel, line


@tool(
    "cuda_global_functions",
    args_schema=CudaGlobalFunctionsArgs,
    description=(
        "List __global__ CUDA kernel definitions (name, file, line) under a specific *-cuda directory. "
        "Example: cuda_global_functions(cuda_name=\"lulesh-cuda\")."
    ),
)
def cuda_global_functions(cuda_name: str) -> List[dict[str, str | int]]:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    results: List[dict[str, str | int]] = []
    for source_file in _gather_cuda_files(cuda_dir):
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for qualified, kernel, line in _extract_cuda_global_definitions(text):
            results.append(
                {
                    "file": str(source_file.relative_to(cuda_dir)),
                    "kernel": kernel,
                    "line": line,
                }
            )
    results.sort(key=lambda entry: (entry["file"], entry["line"], entry["kernel"]))
    return results
