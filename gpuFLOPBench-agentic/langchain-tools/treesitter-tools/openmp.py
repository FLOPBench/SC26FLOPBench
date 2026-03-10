from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Union

from tree_sitter import Node, Tree

from .caches import byte_offset_for_line_col, span_text, span_from_bytes
from .datatypes import NodeRef, OmpRegion, Span
from .traversal import iter_nodes_of_type, node_to_ref, ref_to_span

OMPRAGMA_PATTERN = re.compile(r"#\s*pragma\s+omp\b.*", re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r"[A-Za-z_]+\([^)]*\)|[A-Za-z_]+")
STATEMENT_NODE_TYPES: Set[str] = {
    "expression_statement",
    "compound_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "if_statement",
    "switch_statement",
    "return_statement",
    "jump_statement",
    "declaration_statement",
}


def find_omp_pragmas_near_line(
    path: Union[str, Path], text: str, line1: int, window: int = 2
) -> List[Span]:
    if line1 < 1:
        return []
    lines = text.splitlines()
    start = max(1, line1 - window)
    end = min(len(lines), line1 + window)
    spans: List[Span] = []
    for idx in range(start, end + 1):
        line_text = lines[idx - 1]
        match = OMPRAGMA_PATTERN.search(line_text)
        if match:
            spans.append(_span_for_line(path, text, idx, match))
    return spans


def _span_for_line(
    path: Union[str, Path], text: str, line_idx: int, match: re.Match
) -> Span:
    start_byte = byte_offset_for_line_col(text, line_idx - 1, match.start())
    end_byte = byte_offset_for_line_col(text, line_idx - 1, match.end())
    return span_from_bytes(path, text, start_byte, end_byte)


def parse_omp_pragma(pragma_line: str) -> tuple[str, List[str]]:
    line = pragma_line.strip()
    match = re.match(r"#\s*pragma\s+omp\b(.*)", line, re.IGNORECASE)
    if not match:
        return "", []
    remainder = match.group(1).strip()
    if not remainder:
        return "", []
    parts = remainder.split(None, 1)
    kind = parts[0]
    clauses: List[str] = []
    if len(parts) > 1:
        clause_text = parts[1].strip()
        clauses = CLAUSE_PATTERN.findall(clause_text)
    return kind, clauses


def find_statement_after_byte(tree: Tree, byte_offset: int) -> Optional[NodeRef]:
    best: Node | None = None
    for node in iter_nodes_of_type(tree, STATEMENT_NODE_TYPES):
        if node.start_byte >= byte_offset:
            if best is None or node.start_byte < best.start_byte:
                best = node
    return node_to_ref(best) if best else None


def build_omp_region(
    path: Union[str, Path], text: str, tree: Tree, pragma_span: Span
) -> Optional[OmpRegion]:
    statement_ref = find_statement_after_byte(tree, pragma_span.end_byte)
    if statement_ref is None:
        return None
    stmt_span = ref_to_span(path, statement_ref, text)
    pragma_text = span_text(text, pragma_span)
    kind_text, clauses = parse_omp_pragma(pragma_text)
    return OmpRegion(
        pragma_span=pragma_span,
        pragma_text=pragma_text,
        associated_stmt_span=stmt_span,
        kind_text=kind_text,
        clauses=clauses,
    )
