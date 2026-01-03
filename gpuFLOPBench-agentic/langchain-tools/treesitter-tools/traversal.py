from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Optional, Set, Union

from tree_sitter import Node, Tree

from .caches import byte_offset_for_line_col, normpath, point_for_byte
from .datatypes import NodeRef, Span

FUNCTION_NODE_TYPES: Set[str] = {"function_definition", "function_declarator", "method_definition"}
CALL_NODE_TYPES: Set[str] = {"call_expression"}


def node_to_ref(node: Node) -> NodeRef:
    return NodeRef(
        type=node.type,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_point=node.start_point,
        end_point=node.end_point,
    )


def ref_to_span(path: Union[str, Path], ref: NodeRef, text: str) -> Span:
    return Span(
        file=normpath(path),
        start_byte=ref.start_byte,
        end_byte=ref.end_byte,
        start_point=point_for_byte(text, ref.start_byte),
        end_point=point_for_byte(text, ref.end_byte),
    )


def iter_descendants(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


def iter_nodes_of_type(tree_or_node: Union[Tree, Node], types: Set[str]) -> Iterator[Node]:
    root = tree_or_node.root_node if isinstance(tree_or_node, Tree) else tree_or_node
    for node in iter_descendants(root):
        if node.type in types:
            yield node


def node_contains_byte(node_or_ref: Union[Node, NodeRef], offset: int) -> bool:
    start = node_or_ref.start_byte if isinstance(node_or_ref, NodeRef) else node_or_ref.start_byte
    end = node_or_ref.end_byte if isinstance(node_or_ref, NodeRef) else node_or_ref.end_byte
    return start <= offset < end


def find_smallest_enclosing_node(
    tree: Tree, byte_offset: int, types: Set[str]
) -> Optional[NodeRef]:
    best: Optional[Node] = None
    for node in iter_nodes_of_type(tree, types):
        if node.start_byte <= byte_offset < node.end_byte:
            if best is None or (node.end_byte - node.start_byte) < (best.end_byte - best.start_byte):
                best = node
    return node_to_ref(best) if best else None


def find_enclosing_function(tree: Tree, byte_offset: int) -> Optional[NodeRef]:
    node = tree.root_node.named_descendant_for_byte_range(byte_offset, byte_offset)
    while node:
        if node.type in FUNCTION_NODE_TYPES:
            return node_to_ref(node)
        node = node.parent
    return None


def find_cuda_launches_on_line(tree: Tree, text: str, line1: int) -> List[NodeRef]:
    if line1 < 1:
        return []
    line_index = line1 - 1
    lines = text.splitlines()
    if line_index >= len(lines):
        return []
    pattern = re.compile(r"\b([A-Za-z_][\w:]*)\s*<<<")
    matches: list[NodeRef] = []
    line_text = lines[line_index]
    line_start = byte_offset_for_line_col(text, line_index, 0)
    for match in pattern.finditer(line_text):
        pre_match = line_text[: match.start()]
        match_offset = line_start + len(pre_match.encode("utf-8"))
        node_ref = find_smallest_enclosing_node(tree, match_offset, CALL_NODE_TYPES)
        if node_ref:
            matches.append(node_ref)
    return sorted(matches, key=lambda ref: ref.start_byte)
