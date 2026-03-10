from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import warnings

from tree_sitter import Language, Parser, Tree
import tree_sitter_cuda

try:
    import tree_sitter_cpp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tree_sitter_cpp = None  # type: ignore

from .datatypes import Span

_TEXT_CACHE: dict[str, tuple[str, int, int]] = {}
_TREE_CACHE: dict[tuple[str, int, str], "Tree"] = {}
_LANG_CACHE: dict[str, Language] = {}
_PARSER_CACHE: dict[str, Parser] = {}


def normpath(path: Union[str, Path]) -> str:
    normalized = os.path.normpath(str(path))
    return normalized


def relpath(path: Union[str, Path], project_root: Union[str, Path]) -> str:
    return os.path.relpath(normpath(path), start=normpath(project_root))


def read_text(path: Union[str, Path]) -> str:
    normalized = normpath(path)
    file_path = Path(normalized)
    stat = file_path.stat()
    mtime = stat.st_mtime_ns
    size = stat.st_size
    cached = _TEXT_CACHE.get(normalized)
    if cached and cached[1] == mtime and cached[2] == size:
        return cached[0]
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    _TEXT_CACHE[normalized] = (text, mtime, size)
    return text


def byte_offset_for_line_col(text: str, line0: int, col0: int) -> int:
    if line0 < 0 or col0 < 0:
        raise ValueError("line and column must be zero or positive")
    lines = text.splitlines(True)
    if line0 >= len(lines):
        return len(text.encode("utf-8"))
    prefix = "".join(lines[:line0])
    column_bytes = len(lines[line0][:col0].encode("utf-8"))
    return len(prefix.encode("utf-8")) + column_bytes


def span_text(text: str, span: Span) -> str:
    text_bytes = text.encode("utf-8")
    return text_bytes[span.start_byte:span.end_byte].decode("utf-8", errors="ignore")


def _point_for_byte(text: str, byte_index: int) -> Tuple[int, int]:
    encoded = text.encode("utf-8")
    capped_index = min(byte_index, len(encoded))
    line = encoded.count(b"\n", 0, capped_index)
    prev_newline = encoded.rfind(b"\n", 0, capped_index)
    segment_start = prev_newline + 1 if prev_newline >= 0 else 0
    column_bytes = encoded[segment_start:capped_index]
    column = len(column_bytes.decode("utf-8", errors="ignore"))
    return line, column


point_for_byte = _point_for_byte


def pick_language_for_file(path: Union[str, Path], default: str = "cpp") -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cu", ".cuh"}:
        return "cuda"
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".hpp", ".h"}:
        return "cpp"
    return default


def get_language(lang: str) -> Language:
    key = lang.lower()
    cached = _LANG_CACHE.get(key)
    if cached:
        return cached
    if key == "cuda":
        language = Language(tree_sitter_cuda.language())
    elif key == "cpp":
        lang_module = tree_sitter_cpp or tree_sitter_cuda
        if tree_sitter_cpp is None:
            warnings.warn(
                "tree_sitter_cpp is not installed; falling back to tree_sitter_cuda for C++ parsing.",
                stacklevel=2,
            )
        language = Language(lang_module.language())
    else:
        raise ValueError(f"unknown language '{lang}'")
    _LANG_CACHE[key] = language
    return language


def _get_parser_for_lang(lang: str) -> Parser:
    parser = _PARSER_CACHE.get(lang)
    if parser is None:
        parser = Parser(get_language(lang))
        _PARSER_CACHE[lang] = parser
    return parser


def parse_file(
    path: Union[str, Path],
    lang: Optional[str] = None,
    use_cache: bool = True,
) -> Tree:
    normalized = normpath(path)
    text = read_text(normalized)
    target_lang = lang or pick_language_for_file(normalized)
    stat = Path(normalized).stat()
    mtime = stat.st_mtime_ns
    cache_key = (normalized, mtime, target_lang)
    if use_cache and cache_key in _TREE_CACHE:
        return _TREE_CACHE[cache_key]
    parser = _get_parser_for_lang(target_lang)
    tree = parser.parse(text.encode("utf-8"))
    if use_cache:
        _TREE_CACHE[cache_key] = tree
    return tree


def span_from_bytes(path: Union[str, Path], text: str, start: int, end: int) -> Span:
    return Span(
        file=normpath(path),
        start_byte=start,
        end_byte=end,
        start_point=point_for_byte(text, start),
        end_point=point_for_byte(text, end),
    )
