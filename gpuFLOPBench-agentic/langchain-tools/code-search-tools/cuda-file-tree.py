from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from typing import List

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


class CudaTreeArgs(CudaSubdirArgs):
    """Arguments for generating a file tree of a CUDA subdirectory."""


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


@tool(
    "cuda_file_tree",
    args_schema=CudaTreeArgs,
    description=(
        "Generate an indented file tree for a specific *-cuda directory in gpuFLOPBench/src. "
        "Example: cuda_file_tree(cuda_name=\"lulesh-cuda\")."
    ),
)
def cuda_file_tree(cuda_name: str) -> str:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    lines = [f"{cuda_dir.name}/"] + _tree_lines(cuda_dir, indent="  ")
    return "\n".join(lines)
