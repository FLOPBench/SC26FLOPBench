from __future__ import annotations

from collections.abc import Callable
from importlib import util
from pathlib import Path
import sys
from typing import List

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime

from .descriptions import CUDA_MAIN_FILES_DESCRIPTION, DirectoryArgs

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
_skip_whitespace = _utils._skip_whitespace
_skip_string = _utils._skip_string
_find_matching_paren = _utils._find_matching_paren
_find_first_special_char = _utils._find_first_special_char
_find_matching_brace = _utils._find_matching_brace
_qualifier_block_start_line = _utils._qualifier_block_start_line


def _find_main_definition_range(text: str, lines: list[str]) -> tuple[int, int] | None:
    """Return the (offset_line, line_count) for the first free-function main definition."""
    idx = 0
    length = len(text)
    normalized_lines = lines if lines else [""]
    while idx < length:
        ch = text[idx]
        if ch in {"\"", "'"}:
            new_idx = _skip_string(text, idx)
            if new_idx is None:
                return None
            idx = new_idx
            continue
        if ch == "/" and idx + 1 < length:
            if text[idx + 1] == "/":
                newline = text.find("\n", idx + 2)
                idx = newline if newline != -1 else length
                continue
            if text[idx + 1] == "*":
                end = text.find("*/", idx + 2)
                idx = end + 2 if end != -1 else length
                continue
        if text.startswith("main", idx):
            if idx > 0:
                prev = text[idx - 1]
                if prev.isalnum() or prev == "_" or prev == ":":
                    idx += 1
                    continue
            after_main = idx + len("main")
            if after_main < length and (
                text[after_main].isalnum()
                or text[after_main] == "_"
                or text[after_main] == ":"
            ):
                idx += 1
                continue
            after_main = _skip_whitespace(text, after_main)
            if after_main >= length or text[after_main] != "(":
                idx += 1
                continue
            close_paren = _find_matching_paren(text, after_main)
            if close_paren is None:
                idx += 1
                continue
            marker = _find_first_special_char(text, close_paren + 1)
            if marker is None or marker >= len(text) or text[marker] != "{":
                idx = close_paren + 1
                continue
            brace_end = _find_matching_brace(text, marker)
            if brace_end is None:
                return None
            line_number = text.count("\n", 0, idx) + 1
            line_idx = min(max(0, line_number - 1), len(normalized_lines) - 1)
            offset_idx = _qualifier_block_start_line(normalized_lines, line_idx)
            end_line_idx = text.count("\n", 0, brace_end)
            line_count = max(1, end_line_idx - offset_idx + 1)
            return offset_idx + 1, line_count
        idx += 1
    return None


def _gather_main_files(cuda_dir: Path) -> list[dict[str, str | int]]:
    files: list[dict[str, str | int]] = []
    for source_file in _gather_cuda_files(cuda_dir):
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        main_range = _find_main_definition_range(text, lines)
        if main_range is None:
            continue
        offset, count = main_range
        files.append(
            {
                "file": str(source_file.relative_to(cuda_dir)),
                "offset": offset,
                "lines": count,
            }
        )
    files.sort(key=lambda entry: entry["file"])
    return files


def _resolve_backend_directory(dir_path: str, backend: BackendProtocol) -> Path:
    normalized = (dir_path if dir_path.startswith("/") else "/" + dir_path).rstrip("/")
    if not normalized:
        normalized = "/"
    candidate = Path(normalized)
    if getattr(backend, "virtual_mode", False):
        cwd = getattr(backend, "cwd", None)
        if cwd is None:
            raise ValueError("Backend does not expose a drawable directory")
        relative = normalized.lstrip("/")
        candidate = (Path(cwd) / relative).resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"{dir_path!r} is not a directory")
    return candidate




def make_cuda_main_files_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    *,
    description: str | None = None,
) -> StructuredTool:
    tool_description = description or CUDA_MAIN_FILES_DESCRIPTION

    def _run(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> List[dict[str, str | int]]:
        resolved_backend = _get_backend(backend, runtime)
        validated = _validate_path(dir_path)
        directory = _resolve_backend_directory(validated, resolved_backend)
        main_files = _gather_main_files(directory)
        if not main_files:
            raise ValueError(f"No main() definitions were found under {dir_path!r}")
        return main_files

    async def _arun(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> List[dict[str, str | int]]:
        return _run(dir_path, runtime)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="cuda_main_files",
        description=tool_description,
        args_schema=DirectoryArgs,
    )
