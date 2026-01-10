from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_SRC_DIR = _REPO_ROOT / "gpuFLOPBench" / "src"

GLOBAL_KEYWORD = "__global__"
CUDA_SOURCE_EXTENSIONS = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
TRAILING_COMMENT = re.compile(r"(?:/\*.*?\*/\s*|//[^\n\r]*\s*)$", re.DOTALL)
IDENTIFIER = re.compile(r"[_A-Za-z]\w*")


class CudaSubdirArgs(BaseModel):
    """Shared arguments for tools that operate on GPU benchmark subdirectories."""

    cuda_name: str = Field(
        ...,
        description="Name of the *-cuda directory inside gpuFLOPBench/src to inspect.",
        min_length=1,
    )


class DirectoryArgs(BaseModel):
    """Shared arguments for tools rooted at a particular GPU source directory."""

    dir_path: str = Field(
        ...,
        description=(
            "Absolute disk path or virtual FilesystemBackend path (e.g., `/lulesh-cuda`) "
            "to a directory that lives under gpuFLOPBench/src."
        ),
    )


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


def _gather_cuda_files(root: Path) -> Iterable[Path]:
    """Yield CUDA-relevant source files below the directory."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in CUDA_SOURCE_EXTENSIONS:
            yield path


def _resolve_directory(dir_path: str) -> Path:
    candidate = Path(dir_path)
    search_paths: list[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
        try:
            virtual_rel = candidate.relative_to("/")
        except ValueError:
            pass
        else:
            search_paths.append(GPU_SRC_DIR / virtual_rel)
    else:
        search_paths.append(GPU_SRC_DIR / candidate)

    for path in search_paths:
        if not path.exists() or not path.is_dir():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(GPU_SRC_DIR)
        except ValueError:
            continue
        return resolved
    raise ValueError(f"{dir_path!r} does not point to a directory under {GPU_SRC_DIR}")


def _resolve_source_file(file_path: str) -> Path:
    candidate = Path(file_path)
    search_paths: list[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
        try:
            virtual_relative = candidate.relative_to("/")
        except ValueError:
            pass
        else:
            search_paths.append(GPU_SRC_DIR / virtual_relative)
    else:
        search_paths.append(GPU_SRC_DIR / candidate)

    for path in search_paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(GPU_SRC_DIR)
        except ValueError:
            continue
        return resolved
    raise ValueError(f"{file_path!r} does not point to a file under {GPU_SRC_DIR}")


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


def _extract_kernel_name_from_prefix(prefix: str) -> Optional[str]:
    """Return the namespace-qualified kernel name before any template arguments."""
    idx = len(prefix) - 1
    while idx >= 0 and prefix[idx].isspace():
        idx -= 1
    while idx >= 0 and prefix[idx] == ">":
        depth = 0
        while idx >= 0:
            ch = prefix[idx]
            if ch == ">":
                depth += 1
            elif ch == "<":
                depth -= 1
                idx -= 1
                break
            idx -= 1
        else:
            return None
        while idx >= 0 and prefix[idx].isspace():
            idx -= 1
    end = idx
    if end < 0:
        return None
    while idx >= 0 and (
        prefix[idx].isalnum() or prefix[idx] == "_" or prefix[idx] == ":"
    ):
        idx -= 1
    start = idx + 1
    name = prefix[start : end + 1]
    if not name or all(ch == ":" for ch in name):
        return None
    return name


def _match_kernel_definition(text: str, start_idx: int) -> Optional[Tuple[str, str, int]]:
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
        qualified = _extract_kernel_name_from_prefix(prefix)
        if not qualified:
            search_pos = paren_pos + 1
            continue
        close_paren = _find_matching_paren(text, paren_pos)
        if close_paren is None:
            return None
        marker = _find_first_special_char(text, close_paren + 1)
        if marker is None:
            return None
        if text[marker] == "{":
            kernel = qualified.split("::")[-1]
            return qualified, kernel, marker
        return None


def _iterate_cuda_kernel_definitions(text: str) -> Iterator[Tuple[int, str, str, int, int]]:
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
            qualified, kernel, brace_start = definition
            line_number = text.count("\n", 0, match_pos) + 1
            yield match_pos, qualified, kernel, line_number, brace_start
        pos = match_pos + len(GLOBAL_KEYWORD)
