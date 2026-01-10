from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from typing import Iterator, List, Tuple

from pydantic import BaseModel, Field
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
_GPU_SRC_DIR = _utils.GPU_SRC_DIR
_gather_cuda_files = _utils._gather_cuda_files
_iterate_cuda_kernel_definitions = _utils._iterate_cuda_kernel_definitions


class DirectoryArgs(BaseModel):
    """Arguments for listing __global__ CUDA functions inside a specific directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute directory path or virtual FilesystemBackend path (e.g., `/lulesh-cuda`)"
            " where the CUDA source files live."
        ),
    )


def _extract_cuda_global_definitions(text: str) -> Iterator[Tuple[str, str, int]]:
    for _, qualified, kernel, line, _ in _iterate_cuda_kernel_definitions(text):
        yield qualified, kernel, line


def _resolve_directory(dir_path: str) -> Path:
    candidate = Path(dir_path)
    search_paths: list[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
        try:
            virtual_rel = candidate.relative_to("/")
        except ValueError:
            pass
        else:
            search_paths.append(_GPU_SRC_DIR / virtual_rel)
    else:
        search_paths.append(_GPU_SRC_DIR / candidate)

    for path in search_paths:
        if not path.exists() or not path.is_dir():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(_GPU_SRC_DIR)
        except ValueError:
            continue
        return resolved
    raise ValueError(f"{dir_path!r} does not point to a directory under {_GPU_SRC_DIR}")


@tool(
    "cuda_global_functions",
    args_schema=DirectoryArgs,
    description=(
        "List __global__ CUDA kernel definitions (name, file, line) inside the provided directory. "
        "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
    ),
)
def cuda_global_functions(dir_path: str) -> List[dict[str, str | int]]:
    cuda_dir = _resolve_directory(dir_path)
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
