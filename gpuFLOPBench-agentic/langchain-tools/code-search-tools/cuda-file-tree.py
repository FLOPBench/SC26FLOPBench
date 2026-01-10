from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath
from typing import List

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime
from pydantic import BaseModel, Field


class CudaTreeArgs(BaseModel):
    """Arguments for generating a file tree from a specific directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute directory path or virtual FilesystemBackend path (e.g., `/lulesh-cuda`) "
            "to render a CUDA file tree for."
        ),
    )


TOOL_DESCRIPTION = (
    "Generate an indented file tree for the provided directory. "
    "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
)


def _normalize_virtual_dir_path(dir_path: str) -> str:
    normalized = dir_path.rstrip("/")
    if not normalized:
        return "/"
    return normalized


def _entry_display_name(entry_path: str) -> str:
    stripped = entry_path.rstrip("/")
    if not stripped:
        return "/"
    return PurePosixPath(stripped).name


def _sorted_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(entry: dict[str, object]) -> tuple[bool, str]:
        path = entry.get("path", "")
        name = _entry_display_name(path)
        is_dir = entry.get("is_dir", path.endswith("/"))
        return (not is_dir, name.lower())

    return sorted(entries, key=sort_key)


def _tree_lines_from_backend(
    backend: BackendProtocol,
    dir_path: str,
    indent: str = "",
) -> List[str]:
    lines: List[str] = []
    entries = backend.ls_info(dir_path)
    for entry in _sorted_entries(entries):
        entry_path = entry.get("path")
        if not entry_path:
            continue
        is_dir = entry.get("is_dir", entry_path.endswith("/"))
        suffix = "/" if is_dir else ""
        lines.append(f"{indent}{_entry_display_name(entry_path)}{suffix}")
        if is_dir:
            child_path = _normalize_virtual_dir_path(entry_path)
            lines.extend(_tree_lines_from_backend(backend, child_path, indent + "  "))
    return lines


async def _tree_lines_from_backend_async(
    backend: BackendProtocol,
    dir_path: str,
    indent: str = "",
) -> List[str]:
    lines: List[str] = []
    entries = await backend.als_info(dir_path)
    for entry in _sorted_entries(entries):
        entry_path = entry.get("path")
        if not entry_path:
            continue
        is_dir = entry.get("is_dir", entry_path.endswith("/"))
        suffix = "/" if is_dir else ""
        lines.append(f"{indent}{_entry_display_name(entry_path)}{suffix}")
        if is_dir:
            child_path = _normalize_virtual_dir_path(entry_path)
            lines.extend(await _tree_lines_from_backend_async(backend, child_path, indent + "  "))
    return lines


def _build_backend_tree(backend: BackendProtocol, dir_path: str) -> str:
    normalized_root = _normalize_virtual_dir_path(dir_path)
    backend_root_name = getattr(getattr(backend, "cwd", None), "name", "") or "/"
    root_name = PurePosixPath(normalized_root).name or backend_root_name
    lines = [f"{root_name}/"] + _tree_lines_from_backend(backend, normalized_root, indent="  ")
    return "\n".join(lines)


async def _build_backend_tree_async(backend: BackendProtocol, dir_path: str) -> str:
    normalized_root = _normalize_virtual_dir_path(dir_path)
    backend_root_name = getattr(getattr(backend, "cwd", None), "name", "") or "/"
    root_name = PurePosixPath(normalized_root).name or backend_root_name
    child_lines = await _tree_lines_from_backend_async(backend, normalized_root, indent="  ")
    lines = [f"{root_name}/"] + child_lines
    return "\n".join(lines)


def make_cuda_file_tree_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    *,
    description: str | None = None,
) -> StructuredTool:
    """Build a cuda_file_tree tool that runs against the provided backend."""

    tool_description = description or TOOL_DESCRIPTION

    def _run(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> str:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(dir_path)
        return _build_backend_tree(resolved_backend, validated)

    async def _arun(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> str:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(dir_path)
        return await _build_backend_tree_async(resolved_backend, validated)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="cuda_file_tree",
        description=tool_description,
        args_schema=CudaTreeArgs,
    )
