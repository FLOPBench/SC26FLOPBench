from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from typing import List, Optional, Tuple

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
_skip_whitespace = _utils._skip_whitespace
_skip_string = _utils._skip_string
_resolve_directory = _utils._resolve_directory
_resolve_source_file = _utils._resolve_source_file


class KernelSourceArgs(BaseModel):
    """Arguments for fetching the source code of a specific CUDA kernel."""

    file_path: str = Field(
        ...,
        description=(
            "Absolute disk path or virtual FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu` or `/lulesh-cuda`) "
            "that identifies a file or directory under gpuFLOPBench/src."
        ),
    )
    kernel_name: str = Field(
        ...,
        description="Name of the __global__ CUDA kernel to extract.",
        min_length=1,
    )


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


@tool(
    "extract_kernel_source_definition",
    args_schema=KernelSourceArgs,
    description=(
        "Return the source code for a specific __global__ kernel within the provided file or directory. "
        "Pass an absolute disk path or a FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu`)."
    ),
)
def extract_kernel_source_definition(file_path: str, kernel_name: str) -> List[dict[str, str | int]]:
    results: List[dict[str, str | int]] = []
    try:
        search_files = [_resolve_source_file(file_path)]
    except ValueError as source_error:
        try:
            directory = _resolve_directory(file_path)
        except ValueError:
            raise ValueError(
                f"{file_path!r} is not a file or directory under {_GPU_SRC_DIR}"
            ) from source_error
        else:
            search_files = list(_gather_cuda_files(directory))

    for source_file in search_files:
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = _find_kernel_source_definitions(text, kernel_name)
        if not matches:
            continue
        relative_path = source_file.relative_to(_GPU_SRC_DIR)
        cuda_name = relative_path.parts[0]
        relative_file = str(relative_path.relative_to(cuda_name))
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
    if not results:
        raise ValueError(f"Kernel {kernel_name!r} was not found under {file_path!r}")
    results.sort(key=lambda entry: (entry["file"], entry["line"] if isinstance(entry["line"], int) else 0, entry["kernel"]))
    return results
