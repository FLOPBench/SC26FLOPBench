from __future__ import annotations

from collections.abc import Callable
from importlib import util
from pathlib import Path
import re
import sys
from typing import List, Optional, Set

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime

from .descriptions import INCLUDE_TREE_DESCRIPTION, IncludeTreeArgs

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

_COMMENT_PATTERN = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_INCLUDE_PATTERN = re.compile(r'#\s*include\s*(?P<target>"[^"]+"|<[^>]+>)', re.MULTILINE)
_INDENT_UNIT = "  "


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


def _normalize_virtual_path(path: str) -> str:
    trimmed = path.rstrip("/")
    return trimmed or "/"


def _resolve_backend_source_file(file_path: str, backend: BackendProtocol) -> Path:
    normalized = _normalize_virtual_path(file_path)
    candidate = Path(normalized)
    if getattr(backend, "virtual_mode", False):
        cwd = getattr(backend, "cwd", None)
        if cwd is None:
            raise ValueError("Backend does not expose a root directory")
        relative = normalized.lstrip("/")
        if not relative:
            raise ValueError("Virtual path must reference a file")
        candidate = (cwd / relative).resolve()
        try:
            candidate.relative_to(cwd)
        except ValueError:
            raise ValueError(f"{file_path!r} escapes the backend root directory")
    else:
        if not candidate.is_absolute():
            base = getattr(backend, "cwd", None) or Path.cwd()
            candidate = (base / candidate).resolve()
        else:
            candidate = candidate.resolve()

    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"{file_path!r} is not a file")
    return candidate


def _determine_cuda_root(
    target: Path,
    backend_root: Path | None,
) -> Path:
    candidate = target.parent
    for ancestor in (candidate, *candidate.parents):
        if ancestor.name.endswith("-cuda"):
            return ancestor
    if backend_root is not None:
        try:
            if backend_root == candidate or backend_root in candidate.parents:
                return backend_root
        except Exception:
            pass
    return candidate


def _relative_target_path(target: Path, cuda_root: Path) -> str:
    try:
        return target.relative_to(cuda_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{target} is not under {cuda_root}") from exc


def make_include_tree_extractor_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    *,
    description: str | None = None,
) -> StructuredTool:
    tool_description = description or INCLUDE_TREE_DESCRIPTION

    def _run(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> str:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(file_path)
        target = _resolve_backend_source_file(validated, resolved_backend)
        backend_root = getattr(resolved_backend, "cwd", None)
        cuda_root = _determine_cuda_root(target, backend_root)
        relative_target = _relative_target_path(target, cuda_root)
        lines = [relative_target]
        lines.extend(_build_include_lines(target, cuda_root, {target}, 1))
        return "\n".join(lines)

    async def _arun(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> str:
        return _run(file_path, runtime)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="include_tree_extractor",
        description=tool_description,
        args_schema=IncludeTreeArgs,
    )
