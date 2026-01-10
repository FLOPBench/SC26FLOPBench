from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from typing import List

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


class CudaTreeArgs(BaseModel):
    """Arguments for generating a file tree from a specific directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute directory path or virtual FilesystemBackend path (e.g., `/lulesh-cuda`) "
            "to render a CUDA file tree for."
        ),
    )


def _tree_lines(root: Path, indent: str = "") -> List[str]:
    """Return a sorted list of tree lines for the provided directory."""
    lines: List[str] = []
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{indent}{entry.name}{suffix}")
        if entry.is_dir():
            lines.extend(_tree_lines(entry, indent + "  "))
    return lines


def _resolve_directory(dir_path: str) -> Path:
    candidate = Path(dir_path)
    search_paths: List[Path] = []
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
    "cuda_file_tree",
    args_schema=CudaTreeArgs,
    description=(
        "Generate an indented file tree for the provided directory. "
        "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
    ),
)
def cuda_file_tree(dir_path: str) -> str:
    cuda_dir = _resolve_directory(dir_path)
    lines = [f"{cuda_dir.name}/"] + _tree_lines(cuda_dir, indent="  ")
    return "\n".join(lines)
