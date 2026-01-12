from __future__ import annotations

from collections.abc import Callable
from importlib import util
from pathlib import Path
import sys
from typing import Callable, List, Optional, Tuple

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime

from .descriptions import EXTRACT_KERNEL_SOURCE_DESCRIPTION, KernelSourceArgs

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
_skip_whitespace = _utils._skip_whitespace
_skip_string = _utils._skip_string



def _find_matching_angle(text: str, idx: int) -> Optional[int]:
    if idx >= len(text) or text[idx] != "<":
        return None
    depth = 0
    i = idx
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return i
        elif ch in {"\"", "'"}:
            i = _skip_string(text, i)
            if i is None:
                return None
            continue
        elif ch == "/" and i + 1 < length:
            if text[i + 1] == "/":
                newline = text.find("\n", i + 2)
                i = newline if newline != -1 else length
                continue
            if text[i + 1] == "*":
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
        i += 1
    return None


def _include_template_prefix(text: str, start_idx: int) -> int:
    cursor = start_idx
    search_end = start_idx
    while True:
        template_pos = text.rfind("template", 0, search_end)
        if template_pos == -1:
            return cursor
        after_keyword = template_pos + len("template")
        after_keyword = _skip_whitespace(text, after_keyword)
        if after_keyword < len(text) and text[after_keyword] == "<":
            angle_end = _find_matching_angle(text, after_keyword)
            if angle_end is None:
                return cursor
            gap = text[angle_end + 1 : search_end]
            if gap.strip() == "":
                cursor = template_pos
                search_end = template_pos
                continue
        search_end = template_pos
    return cursor


def _find_matching_brace(text: str, idx: int) -> Optional[int]:
    if idx >= len(text) or text[idx] != "{":
        return None
    depth = 0
    i = idx
    length = len(text)
    preprocessor_stack: list[int] = []
    while i < length:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        elif ch in {"\"", "'"}:
            i = _skip_string(text, i)
            if i is None:
                return None
            continue
        elif ch == "/" and i + 1 < length:
            if text[i + 1] == "/":
                newline = text.find("\n", i + 2)
                i = newline if newline != -1 else length
                continue
            if text[i + 1] == "*":
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
        elif ch == "#":
            newline = text.find("\n", i + 1)
            newline = newline if newline != -1 else length
            directive = text[i + 1 : newline].strip()
            keyword = directive.split(None, 1)[0] if directive else ""
            if keyword in {"if", "ifdef", "ifndef"}:
                preprocessor_stack.append(depth)
            elif keyword in {"else", "elif"} and preprocessor_stack:
                depth = preprocessor_stack[-1]
            elif keyword == "endif" and preprocessor_stack:
                preprocessor_stack.pop()
            i = newline
            continue
        i += 1
    return None


def _find_kernel_source_definitions(
    text: str,
    kernel_name: str,
) -> list[Tuple[int, str, str, int, int]]:
    results: list[Tuple[int, str, str, int, int]] = []
    for match_pos, qualified, kernel, line, brace_start in _iterate_cuda_kernel_definitions(text):
        if kernel != kernel_name:
            continue
        brace_end = _find_matching_brace(text, brace_start)
        if brace_end is None:
            raise ValueError("Could not find matching brace for kernel definition")
        start_idx = _include_template_prefix(text, match_pos)
        results.append((start_idx, qualified, kernel, line, brace_end))
    return results


def _normalize_virtual_path(path: str) -> str:
    trimmed = path.rstrip("/")
    return trimmed or "/"


def _resolve_backend_target_path(path: str, backend: BackendProtocol) -> Path:
    normalized = _normalize_virtual_path(path)
    if getattr(backend, "virtual_mode", False):
        cwd = getattr(backend, "cwd", None)
        if cwd is None:
            raise ValueError("Backend does not expose a root directory")
        relative = normalized.lstrip("/")
        candidate = (cwd / relative).resolve() if relative else cwd.resolve()
        try:
            candidate.relative_to(cwd)
        except ValueError:
            raise ValueError(f"{path!r} escapes the backend root directory")
        return candidate
    candidate = Path(normalized)
    if not candidate.is_absolute():
        base = getattr(backend, "cwd", None) or Path.cwd()
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _collect_target_files(file_path: str, target: Path) -> list[Path]:
    if not target.exists():
        raise ValueError(f"{file_path!r} does not exist")
    if target.is_file():
        return [target]
    if target.is_dir():
        return list(_gather_cuda_files(target))
    raise ValueError(f"{file_path!r} is not a file or directory")


def _determine_cuda_root(target: Path, backend_root: Path | None) -> Path:
    candidate = target if target.is_dir() else target.parent
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


def _relative_to_cuda_root(source_file: Path, cuda_root: Path) -> str | None:
    try:
        return source_file.relative_to(cuda_root).as_posix()
    except ValueError:
        return None


def _extract_entries_from_files(
    search_files: list[Path],
    kernel_name: str,
    cuda_root: Path,
) -> list[dict[str, str | int]]:
    results: list[dict[str, str | int]] = []
    cuda_name = cuda_root.name
    for source_file in search_files:
        relative_file = _relative_to_cuda_root(source_file, cuda_root)
        if relative_file is None:
            continue
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = _find_kernel_source_definitions(text, kernel_name)
        if not matches:
            continue
        for start_idx, qualified, kernel, line, brace_end in matches:
            source = text[start_idx : brace_end + 1]
            results.append(
                {
                    "cuda_name": cuda_name,
                    "file": relative_file,
                    "kernel": kernel,
                    "qualified": qualified,
                    "line": line,
                    "source": source,
                }
            )
    results.sort(
        key=lambda entry: (
            entry["file"],
            entry["line"] if isinstance(entry["line"], int) else 0,
            entry["kernel"],
        )
    )
    return results


def make_extract_kernel_source_definition_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    *,
    description: str | None = None,
) -> StructuredTool:
    tool_description = description or EXTRACT_KERNEL_SOURCE_DESCRIPTION

    def _run(
        file_path: str,
        kernel_name: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> list[dict[str, str | int]]:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(file_path)
        target = _resolve_backend_target_path(validated, resolved_backend)
        search_files = _collect_target_files(validated, target)
        backend_root = getattr(resolved_backend, "cwd", None)
        cuda_root = _determine_cuda_root(target, backend_root)
        entries = _extract_entries_from_files(search_files, kernel_name, cuda_root)
        if not entries:
            raise ValueError(f"Kernel {kernel_name!r} was not found under {file_path!r}")
        return entries

    async def _arun(
        file_path: str,
        kernel_name: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> list[dict[str, str | int]]:
        return _run(file_path, kernel_name, runtime)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="extract_kernel_source_definition",
        description=tool_description,
        args_schema=KernelSourceArgs,
    )
