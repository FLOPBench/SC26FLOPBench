from __future__ import annotations

import functools
import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from langchain.tools import tool
from pydantic import BaseModel, Field

GPU_SRC_DIR = Path(__file__).resolve().parents[1] / "gpuFLOPBench" / "src"

GLOBAL_KEYWORD = "__global__"

CUDA_SOURCE_EXTENSIONS = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}

NAME_CAPTURE = re.compile(r"([\w:]+)\s*$")
TRAILING_COMMENT = re.compile(r"(?:/\*.*?\*/\s*|//[^\n\r]*\s*)$", re.DOTALL)
IDENTIFIER = re.compile(r"[_A-Za-z]\w*")
#

COMPILE_COMMANDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "gpuFLOPBench"
    / "cuda-profiling"
    / "compile_commands.json"
)


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


class KernelSourceArgs(CudaSubdirArgs):
    """Arguments for fetching the source code of a specific CUDA kernel."""

    kernel_name: str = Field(
        ...,
        description="Name of the __global__ CUDA kernel to extract.",
        min_length=1,
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


def _extract_cuda_global_definitions(text: str) -> Iterator[Tuple[str, str, int]]:
    for _, qualified, kernel, line, _ in _iterate_cuda_kernel_definitions(text):
        yield qualified, kernel, line


def _find_kernel_source_definitions(text: str, kernel_name: str) -> list[Tuple[int, str, str, int, int]]:
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
            if after_main < length and (text[after_main].isalnum() or text[after_main] == "_" or text[after_main] == ":"):
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


def _gather_main_files(cuda_dir: Path) -> List[str]:
    """Return a sorted list of CUDA source files that define main()."""
    files: List[str] = []
    for source_file in _gather_cuda_files(cuda_dir):
        try:
            text = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _contains_main_definition(text):
            files.append(str(source_file.relative_to(cuda_dir)))
    files.sort()
    return files



@functools.lru_cache(maxsize=1)
def _load_compile_commands() -> list[dict[str, Any]]:
    if not COMPILE_COMMANDS_PATH.exists():
        raise FileNotFoundError(
            f"{COMPILE_COMMANDS_PATH} was not found. Have you generated the database?"
        )
    try:
        data = json.loads(COMPILE_COMMANDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("compile_commands.json contains invalid JSON") from exc
    if not isinstance(data, list):
        raise ValueError("compile_commands.json expected a list of entries")
    return data


def _gather_compile_entries(cuda_name: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in _load_compile_commands():
        file_path = Path(entry.get("file", ""))
        if cuda_name not in file_path.parts:
            continue
        command = entry.get("command")
        if not command:
            continue
        tokens = shlex.split(command)
        if not tokens:
            continue
        entries.append(
            {
                "file": file_path.name,
                "directory": entry.get("directory", ""),
                "output": entry.get("output"),
                "compiler": tokens[0],
                "arguments": tokens[1:],
                "command": command,
            }
        )
    entries.sort(key=lambda item: item["file"])
    return entries


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


@tool(
    "extract_kernel_source_definition",
    args_schema=KernelSourceArgs,
    description="Return the source code for a specific __global__ kernel within a *-cuda benchmark.",
)
def extract_kernel_source_definition(cuda_name: str, kernel_name: str) -> List[dict[str, str | int]]:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    results: List[dict[str, str | int]] = []
    for source_file in _gather_cuda_files(cuda_dir):
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
                    "file": str(source_file.relative_to(cuda_dir)),
                    "kernel": kernel,
                    "qualified": qualified,
                    "line": line,
                    "source": source,
                }
            )
    if not results:
        raise ValueError(f"Kernel {kernel_name!r} was not found under {cuda_name!r}")
    return results


@tool(
    "cuda_compile_commands",
    args_schema=CudaSubdirArgs,
    description="Return the compiler arguments listed in gpuFLOPBench/cuda-profiling/compile_commands.json for the requested *-cuda benchmark.",
)
def cuda_compile_commands(cuda_name: str) -> dict[str, Any]:
    _resolve_cuda_dir(cuda_name)
    entries = _gather_compile_entries(cuda_name)
    if not entries:
        raise ValueError(f"No compile commands were found for {cuda_name!r}")
    return {"cuda_name": cuda_name, "commands": entries}


@tool(
    "cuda_main_files",
    args_schema=CudaSubdirArgs,
    description="List source files under the requested *-cuda directory that define a free-function main().",
)
def cuda_main_files(cuda_name: str) -> List[str]:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    main_files = _gather_main_files(cuda_dir)
    if not main_files:
        raise ValueError(f"No main() definitions were found under {cuda_name!r}")
    return main_files


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
