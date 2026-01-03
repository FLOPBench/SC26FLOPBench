from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain.tools import tool
from pydantic import BaseModel, Field

from .callsite import (
    collect_callsites_in_function,
    find_assignments_in_function,
    summarize_def_use,
)
from .caches import (
    byte_offset_for_line_col,
    normpath,
    pick_language_for_file,
    read_text,
    relpath,
    span_from_bytes,
    span_text,
    parse_file,
)
from .datatypes import Callsite, NodeRef, OmpRegion, Span
from .identifiers import (
    COMMON_MACROS,
    CPP_BUILTIN_TYPES,
    CPP_KEYWORDS,
    build_symbol_id,
    collect_identifiers_in_node,
    collect_identifiers_in_span,
    extract_template_param_list,
    function_signature_text,
    is_identifier_token,
    normalize_ws_punct,
)
from .includes import collect_includes, is_project_path, resolve_include_to_path
from .openmp import (
    build_omp_region as _build_omp_region_helper,
    find_omp_pragmas_near_line,
    find_statement_after_byte,
    parse_omp_pragma,
)
collect_omp_region = _build_omp_region_helper
from .serialization import callsite_to_json, noderef_to_json, ompreg_to_json
from .traversal import (
    find_enclosing_function,
    find_smallest_enclosing_node,
    find_cuda_launches_on_line as _find_cuda_launches_on_line,
    iter_descendants,
    iter_nodes_of_type,
    node_contains_byte,
    node_to_ref,
    ref_to_span,
)
# expose helper alias for tests / other callers
collect_cuda_launches_on_line = _find_cuda_launches_on_line

__all__ = [
    "Span",
    "NodeRef",
    "Callsite",
    "OmpRegion",
    "read_text",
    "normpath",
    "relpath",
    "byte_offset_for_line_col",
    "span_text",
    "pick_language_for_file",
    "parse_file",
    "span_from_bytes",
    "node_to_ref",
    "ref_to_span",
    "iter_descendants",
    "iter_nodes_of_type",
    "node_contains_byte",
    "find_smallest_enclosing_node",
    "find_enclosing_function",
    "find_cuda_launches_on_line",
    "collect_cuda_launches_on_line",
    "extract_cuda_launch_parts",
    "find_omp_pragmas_near_line",
    "parse_omp_pragma",
    "find_statement_after_byte",
    "build_omp_region",
    "is_identifier_token",
    "normalize_ws_punct",
    "collect_identifiers_in_span",
    "collect_identifiers_in_node",
    "function_signature_text",
    "extract_template_param_list",
    "build_symbol_id",
    "collect_omp_region",
    "CPP_KEYWORDS",
    "CPP_BUILTIN_TYPES",
    "COMMON_MACROS",
    "collect_callsites_in_function",
    "summarize_def_use",
    "find_assignments_in_function",
    "collect_includes",
    "resolve_include_to_path",
    "is_project_path",
    "callsite_to_json",
    "ompreg_to_json",
    "noderef_to_json",
    "span_to_json",
    "CstLaunchArgs",
    "CstOmpArgs",
]


def span_to_json(span: Span, text: Optional[str] = None) -> dict:
    payload = {
        "file": span.file,
        "start_byte": span.start_byte,
        "end_byte": span.end_byte,
        "start_point": span.start_point,
        "end_point": span.end_point,
    }
    if text is not None:
        payload["text"] = span_text(text, span)
    return payload


def extract_cuda_launch_parts(
    path: Union[str, Path], text: str, launch_ref: NodeRef
) -> dict:
    span = ref_to_span(path, launch_ref, text)
    text_bytes = text.encode("utf-8")
    snippet = text_bytes[span.start_byte : span.end_byte]
    result: dict = {"full_launch_span": span}
    kernel_match = re.search(rb"([A-Za-z_][\w:]*)\s*(<[^>]+>)?\s*<<<", snippet)
    if kernel_match:
        name_bytes = kernel_match.group(1)
        result["kernel_name_text"] = name_bytes.decode("utf-8", errors="ignore")
        name_start = span.start_byte + kernel_match.start(1)
        name_end = span.start_byte + kernel_match.end(1)
        result["kernel_name_span"] = span_from_bytes(path, text, name_start, name_end)
        if kernel_match.group(2):
            result["template_args_text"] = kernel_match.group(2).decode("utf-8", errors="ignore")
    cfg_start = snippet.find(b"<<<")
    cfg_span = None
    if cfg_start >= 0:
        cfg_end = snippet.find(b">>>", cfg_start)
        if cfg_end >= 0:
            cfg_end += 3
            cfg_span = span_from_bytes(
                path, text, span.start_byte + cfg_start, span.start_byte + cfg_end
            )
    result["launch_cfg_span"] = cfg_span
    arg_span = None
    if cfg_span:
        after_cfg = cfg_span.end_byte
    else:
        after_cfg = span.start_byte
    arg_start = _find_char_within(buffer=text_bytes, char=b"(", start=after_cfg, end=span.end_byte)
    if arg_start is not None:
        close_pos = _find_matching(
            buffer=text_bytes, open_pos=arg_start, open_char=b"(", close_char=b")"
        )
        if close_pos:
            arg_span = span_from_bytes(path, text, arg_start, close_pos)
    result["arg_span"] = arg_span
    return result


def _find_char_within(*, buffer: bytes, char: bytes, start: int, end: int) -> Optional[int]:
    assert len(char) == 1
    if start >= len(buffer) or start >= end:
        return None
    position = buffer.find(char, start, end)
    if position == -1:
        return None
    return position


def _find_matching(
    *, buffer: bytes, open_pos: int, open_char: bytes, close_char: bytes
) -> Optional[int]:
    if open_pos < 0 or open_pos >= len(buffer):
        return None
    depth = 0
    for idx in range(open_pos, len(buffer)):
        current = buffer[idx : idx + 1]
        if current == open_char:
            depth += 1
        elif current == close_char:
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


class CstLaunchArgs(BaseModel):
    file_path: str = Field(..., description="Filesystem path to the CUDA source file.")
    line: int = Field(..., description="1-indexed line number where the launch occurs.")
    language: Optional[str] = Field(None, description="Override parser language (e.g., 'cuda').")


@tool(
    "find_cuda_launches_on_line",
    args_schema=CstLaunchArgs,
    description=(
        "Return kernel launch spans and metadata on the given line of a CUDA source file. "
        "Example: find_cuda_launches_on_line(file_path=\"gpuFLOPBench/src/lulesh-cuda/src/main.cu\", line=123)."
    ),
)
def find_cuda_launches_on_line(file_path: str, line: int, language: Optional[str] = None) -> list[dict]:
    normalized = normpath(file_path)
    text = read_text(normalized)
    tree = parse_file(normalized, lang=language)
    launches = _find_cuda_launches_on_line(tree, text, line)
    results: list[dict] = []
    for launch in launches:
        parts = extract_cuda_launch_parts(normalized, text, launch)
        entry: dict = {
            "kernel_name_text": parts.get("kernel_name_text"),
            "template_args_text": parts.get("template_args_text"),
        }
        span_keys = [
            "full_launch_span",
            "kernel_name_span",
            "launch_cfg_span",
            "arg_span",
        ]
        for key in span_keys:
            span_value = parts.get(key)
            entry[key] = span_to_json(span_value, text) if span_value else None
        results.append(entry)
    return results


class CstOmpArgs(BaseModel):
    file_path: str = Field(..., description="Filesystem path to the source file.")
    line: int = Field(..., description="1-indexed line near the OpenMP pragma.")
    window: int = Field(2, description="Number of lines before/after to search.")
    language: Optional[str] = Field(None, description="Override parser language (e.g., 'cpp').")


@tool(
    "build_omp_region",
    args_schema=CstOmpArgs,
    description=(
        "Return OpenMP pragma spans + associated statement spans for a line in a source file. "
        "Example: build_omp_region(file_path=\"gpuFLOPBench/src/miniFE-cuda/src/miniFE.cpp\", line=120)."
    ),
)
def build_omp_region(
    file_path: str, line: int, window: int = 2, language: Optional[str] = None
) -> list[Dict[str, Optional[dict]]]:
    normalized = normpath(file_path)
    text = read_text(normalized)
    tree = parse_file(normalized, lang=language)
    pragma_spans = find_omp_pragmas_near_line(normalized, text, line, window)
    results: list[Dict[str, Optional[dict]]] = []
    for span in pragma_spans:
        region = _build_omp_region_helper(normalized, text, tree, span)
        if region is None:
            continue
        entry = {
            "kind_text": region.kind_text,
            "clauses": region.clauses,
            "pragma_text": region.pragma_text,
            "pragma_span": span_to_json(region.pragma_span, text),
            "associated_stmt_span": span_to_json(region.associated_stmt_span, text),
        }
        results.append(entry)
    return results
