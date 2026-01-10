from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import util
from pathlib import Path
import sys
from typing import List, Tuple

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime
from pydantic import BaseModel, Field

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
_gather_cuda_files = _utils._gather_cuda_files
_iterate_cuda_kernel_definitions = _utils._iterate_cuda_kernel_definitions


class DirectoryArgs(BaseModel):
    """Arguments for listing `__global__` CUDA functions in a directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute directory path or virtual FilesystemBackend path "
            "(e.g., `/lulesh-cuda`) where the CUDA source files live."
        ),
    )
    
TOOL_DESCRIPTION = (
    "List __global__ CUDA kernel definitions (name, file, line) inside the provided directory. "
    "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
)


def _extract_cuda_global_definitions(text: str) -> Iterator[Tuple[str, str, int]]:
    for _, qualified, kernel, line, _ in _iterate_cuda_kernel_definitions(text):
        yield qualified, kernel, line


def _collect_global_entries(directory: Path) -> List[dict[str, str | int]]:
    results: List[dict[str, str | int]] = []
    for source_file in _gather_cuda_files(directory):
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for _, kernel, line in _extract_cuda_global_definitions(text):
            results.append(
                {
                    "file": str(source_file.relative_to(directory)),
                    "kernel": kernel,
                    "line": line,
                }
            )
    results.sort(key=lambda entry: (entry["file"], entry["line"], entry["kernel"]))
    return results


def _resolve_backend_directory(dir_path: str, backend: BackendProtocol) -> Path:
    normalized = dir_path.rstrip("/")
    if not normalized:
        normalized = "/"
    base_path: Path | None = None
    cwd = getattr(backend, "cwd", None)
    if cwd is not None:
        base_path = Path(cwd)
    relative = normalized.lstrip("/")
    if normalized.startswith("/"):
        absolute_candidate = Path(normalized).resolve()
        if absolute_candidate.exists() and absolute_candidate.is_dir():
            return absolute_candidate
    if base_path is not None:
        if relative and relative == base_path.name:
            candidate = base_path.resolve()
        else:
            candidate = (base_path / relative).resolve()
    else:
        candidate = Path(normalized if normalized.startswith("/") else "/" + normalized).resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"{dir_path!r} is not a directory")
    return candidate




def make_cuda_global_functions_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    *,
    description: str | None = None,
) -> StructuredTool:
    """Build a cuda_global_functions tool that runs against the provided backend."""

    tool_description = description or TOOL_DESCRIPTION

    def _run(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> List[dict[str, str | int]]:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(dir_path)
        directory = _resolve_backend_directory(validated, resolved_backend)
        return _collect_global_entries(directory)

    async def _arun(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> List[dict[str, str | int]]:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(dir_path)
        directory = _resolve_backend_directory(validated, resolved_backend)
        return _collect_global_entries(directory)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="cuda_global_functions",
        description=tool_description,
        args_schema=DirectoryArgs,
    )
