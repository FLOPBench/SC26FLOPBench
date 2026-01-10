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
_gather_cuda_files = _utils._gather_cuda_files
_skip_whitespace = _utils._skip_whitespace
_skip_string = _utils._skip_string
_find_matching_paren = _utils._find_matching_paren
_find_first_special_char = _utils._find_first_special_char
_resolve_directory = _utils._resolve_directory
DirectoryArgs = _utils.DirectoryArgs


def _contains_main_definition(text: str) -> bool:
    """Return True if the provided text contains a free-function main definition."""
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch in {"\"", "'"}:
            new_idx = _skip_string(text, idx)
            if new_idx is None:
                return False
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
            if marker is not None and marker < len(text) and text[marker] == "{":
                return True
            idx = close_paren + 1
            continue
        idx += 1
    return False


def _gather_main_files(cuda_dir: Path) -> list[str]:
    files: list[str] = []
    for source_file in _gather_cuda_files(cuda_dir):
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _contains_main_definition(text):
            files.append(str(source_file.relative_to(cuda_dir)))
    files.sort()
    return files


@tool(
    "cuda_main_files",
    args_schema=DirectoryArgs,
    description=(
        "List source files under the provided directory that define a free-function main(). "
        "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda`)."
    ),
)
def cuda_main_files(dir_path: str) -> List[str]:
    cuda_dir = _resolve_directory(dir_path)
    main_files = _gather_main_files(cuda_dir)
    if not main_files:
        raise ValueError(f"No main() definitions were found under {dir_path!r}")
    return main_files
