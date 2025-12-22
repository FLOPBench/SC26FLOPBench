from __future__ import annotations

from importlib import util
from pathlib import Path
import re
import sys
from typing import List, Optional, Set

from pydantic import Field

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


_COMMENT_PATTERN = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_INCLUDE_PATTERN = re.compile(r'#\s*include\s*(?P<target>"[^"]+"|<[^>]+>)', re.MULTILINE)
_INDENT_UNIT = "  "


class IncludeTreeArgs(CudaSubdirArgs):
    """Arguments for walking the include tree of a specific source file."""

    file_name: str = Field(
        ...,
        description="Relative path (from the benchmark root) of the CUDA/C++ file to analyze.",
        min_length=1,
    )


def _strip_comments(contents: str) -> str:
    return _COMMENT_PATTERN.sub("", contents)


def _collect_includes(source_path: Path) -> List[str]:
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
    stripped = _strip_comments(source_text)
    return [match.group("target") for match in _INCLUDE_PATTERN.finditer(stripped)]


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_include_target(include_token: str, current_dir: Path, root_dir: Path) -> Optional[Path]:
    """Resolve include paths relative to the including file, then the benchmark root."""
    if not include_token:
        return None
    inner_path = include_token[1:-1]
    for base in (current_dir, root_dir):
        try:
            candidate = (base / inner_path).resolve()
        except (OSError, RuntimeError):
            continue
        if candidate.exists() and _is_within_root(candidate, root_dir):
            return candidate
    return None


def _describe_annotation(resolved: Optional[Path], ancestry: Set[Path]) -> str:
    if resolved is None:
        return " (DNE)"
    if resolved in ancestry:
        return " (loop detected)"
    return ""


def _build_include_lines(
    path: Path,
    root_dir: Path,
    ancestry: Set[Path],
    indent_level: int,
) -> List[str]:
    lines: List[str] = []
    for include_token in _collect_includes(path):
        resolved = _resolve_include_target(include_token, path.parent, root_dir)
        annotation = _describe_annotation(resolved, ancestry)
        line = f"{_INDENT_UNIT * indent_level}#include {include_token}{annotation}"
        lines.append(line)
        if annotation:
            # Either DNE or loop detected; avoid further recursion.
            continue
        if resolved is None or resolved in ancestry:
            continue
        lines.extend(
            _build_include_lines(
                resolved,
                root_dir,
                ancestry | {resolved},
                indent_level + 1,
            )
        )
    return lines


@tool(
    "include_tree_extractor",
    args_schema=IncludeTreeArgs,
    description=(
        "Walk the #include hierarchy for a specific CUDA/C++ file inside a *-cuda "
        "benchmark, annotating missing files (DNE) and stopping recursion when loops "
        "are detected. Example: include_tree_extractor(cuda_name=\"lulesh-cuda\", file_name=\"src/main.cu\")."
    ),
)
def include_tree_extractor(cuda_name: str, file_name: str) -> str:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    target = (cuda_dir / file_name).resolve()
    if not _is_within_root(target, cuda_dir):
        raise ValueError(f"{file_name!r} escapes the benchmark root")
    if not target.exists():
        raise ValueError(f"{file_name!r} does not exist under {cuda_dir}")
    if not target.is_file():
        raise ValueError(f"{file_name!r} is not a file")

    relative_target = target.relative_to(cuda_dir).as_posix()
    lines = [relative_target]
    lines.extend(_build_include_lines(target, cuda_dir, {target}, 1))
    return "\n".join(lines)
