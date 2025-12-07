from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from langchain.tools import tool
from pydantic import BaseModel, Field

GPU_SRC_DIR = Path(__file__).resolve().parents[1] / "gpuFLOPBench" / "src"

GLOBAL_KEYWORD = "__global__"

CUDA_SOURCE_EXTENSIONS = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}

NAME_CAPTURE = re.compile(r"([\w:]+)\s*$")
TRAILING_COMMENT = re.compile(r"(?:/\*.*?\*/\s*|//[^\n\r]*\s*)$", re.DOTALL)
IDENTIFIER = re.compile(r"[_A-Za-z]\w*")


class CudaSubdirArgs(BaseModel):
    """Shared arguments for tools that operate on GPU benchmark subdirectories."""

    cuda_name: str = Field(
        ...,
        description="Name of the *-cuda directory inside gpuFLOPBench/src to inspect.",
        min_length=1,
    )


class CudaTreeArgs(CudaSubdirArgs):
    """Arguments for generating a file tree of a CUDA subdirectory."""


class CudaGlobalFunctionsArgs(CudaSubdirArgs):
    """Arguments for listing __global__ CUDA functions inside a subdirectory."""


def _resolve_cuda_dir(cuda_name: str) -> Path:
    """Guardrail the requested path so it stays within gpuFLOPBench/src."""
    candidate = (GPU_SRC_DIR / cuda_name).resolve()
    if not candidate.exists():
        raise ValueError(f"{cuda_name!r} does not exist under {GPU_SRC_DIR}")
    if not candidate.is_dir():
        raise ValueError(f"{cuda_name!r} is not a directory")
    if not cuda_name.endswith("-cuda"):
        raise ValueError(f"{cuda_name!r} is not a *-cuda benchmark directory")
    try:
        candidate.relative_to(GPU_SRC_DIR)
    except ValueError:
        raise ValueError("Requested path escapes the gpuFLOPBench/src root")
    return candidate


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


def _gather_cuda_files(root: Path) -> Iterable[Path]:
    """Yield CUDA-relevant source files below the directory."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in CUDA_SOURCE_EXTENSIONS:
            yield path


def _skip_whitespace(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _skip_prefixed_macro_calls(text: str, idx: int) -> int:
    """Consume identifier(...) blocks (like __launch_bounds__) after __global__."""
    while True:
        idx = _skip_whitespace(text, idx)
        match = IDENTIFIER.match(text[idx:])
        if not match:
            break
        name_end = idx + match.end()
        next_idx = _skip_whitespace(text, name_end)
        if next_idx < len(text) and text[next_idx] == "(":
            closing = _find_matching_paren(text, next_idx)
            if closing is None:
                break
            idx = closing + 1
            continue
        break
    return idx


def _skip_string(text: str, idx: int) -> Optional[int]:
    quote = text[idx]
    i = idx + 1
    length = len(text)
    while i < length:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return None


def _find_matching_paren(text: str, idx: int) -> Optional[int]:
    if idx >= len(text) or text[idx] != "(":
        return None
    depth = 0
    i = idx
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
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


def _strip_trailing_comments(segment: str) -> str:
    while True:
        stripped = TRAILING_COMMENT.sub("", segment)
        if stripped == segment:
            return stripped
        segment = stripped


def _find_first_special_char(text: str, idx: int) -> Optional[int]:
    i = idx
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "/" and i + 1 < length:
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
        if ch in {"\"", "'"}:
            i = _skip_string(text, i)
            if i is None:
                return None
            continue
        if ch in {";", "{"}:
            return i
        i += 1
    return None


def _match_kernel_definition(text: str, start_idx: int) -> Optional[Tuple[str, str]]:
    search_pos = start_idx
    while True:
        paren_pos = text.find("(", search_pos)
        if paren_pos == -1:
            return None
        prefix = text[start_idx:paren_pos]
        prefix = _strip_trailing_comments(prefix).strip()
        if not prefix or not re.search(r"\s", prefix):
            search_pos = paren_pos + 1
            continue
        name_match = NAME_CAPTURE.search(prefix)
        if not name_match:
            search_pos = paren_pos + 1
            continue
        close_paren = _find_matching_paren(text, paren_pos)
        if close_paren is None:
            return None
        marker = _find_first_special_char(text, close_paren + 1)
        if marker is None:
            return None
        if text[marker] == "{":
            qualified = name_match.group(1)
            kernel = qualified.split("::")[-1]
            return qualified, kernel
        return None


def _extract_cuda_global_definitions(text: str) -> Iterator[Tuple[str, str, int]]:
    pos = 0
    while True:
        match_pos = text.find(GLOBAL_KEYWORD, pos)
        if match_pos == -1:
            return
        signature_start = match_pos + len(GLOBAL_KEYWORD)
        signature_start = _skip_whitespace(text, signature_start)
        signature_start = _skip_prefixed_macro_calls(text, signature_start)
        signature_start = _skip_whitespace(text, signature_start)
        definition = _match_kernel_definition(text, signature_start)
        if definition:
            qualified, kernel = definition
            line_number = text.count("\n", 0, match_pos) + 1
            yield qualified, kernel, line_number
        pos = match_pos + len(GLOBAL_KEYWORD)


@tool(
    "cuda_file_tree",
    args_schema=CudaTreeArgs,
    description="Generate an indented file tree for a specific *-cuda directory in gpuFLOPBench/src.",
)
def cuda_file_tree(cuda_name: str) -> str:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    lines = [f"{cuda_dir.name}/"] + _tree_lines(cuda_dir, indent="  ")
    return "\n".join(lines)


@tool(
    "cuda_global_functions",
    args_schema=CudaGlobalFunctionsArgs,
    description="List __global__ CUDA kernel definitions (name, file, line) under a specific *-cuda directory.",
)
def cuda_global_functions(cuda_name: str) -> List[dict[str, str | int]]:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    results: List[dict[str, str | int]] = []
    for source_file in _gather_cuda_files(cuda_dir):
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for qualified, kernel, line in _extract_cuda_global_definitions(text):
            results.append(
                {
                    "file": str(source_file.relative_to(cuda_dir)),
                    "kernel": kernel,
                    "line": line,
                }
            )
    results.sort(key=lambda entry: (entry["file"], entry["line"], entry["kernel"]))
    return results
