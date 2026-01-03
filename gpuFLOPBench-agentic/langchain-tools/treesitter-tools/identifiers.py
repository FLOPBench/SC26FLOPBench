from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set, Union

from tree_sitter import Tree

from .datatypes import NodeRef, Span

CPP_KEYWORDS: Set[str] = {
    "alignas",
    "alignof",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "constexpr",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "false",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "namespace",
    "new",
    "nullptr",
    "operator",
    "private",
    "protected",
    "public",
    "register",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
}

CPP_BUILTIN_TYPES: Set[str] = {
    "int",
    "short",
    "long",
    "float",
    "double",
    "char",
    "void",
    "bool",
}

COMMON_MACROS: Set[str] = {"__syncthreads", "__global__", "__device__", "__host__", "__constant__"}


def is_identifier_token(s: str) -> bool:
    raise NotImplementedError("TODO: implement identifier token detection")


def normalize_ws_punct(s: str) -> str:
    raise NotImplementedError("TODO: implement whitespace/punctuation normalization")


def collect_identifiers_in_span(text: str, tree: Tree, span: Span) -> Set[str]:
    raise NotImplementedError("TODO: collect identifiers overlapping a span")


def collect_identifiers_in_node(text: str, node_ref: NodeRef) -> Set[str]:
    raise NotImplementedError("TODO: collect identifiers from a CST node")


def function_signature_text(text: str, func_ref: NodeRef) -> str:
    raise NotImplementedError("TODO: extract the function signature text")


def extract_template_param_list(text: str, func_ref: NodeRef) -> Optional[str]:
    raise NotImplementedError("TODO: extract template parameter list")


def build_symbol_id(
    file: Union[str, Path],
    func_ref: NodeRef,
    signature_text: str,
    project_root: Union[str, Path] | None = None,
    cuda_attr: Optional[str] = None,
) -> str:
    raise NotImplementedError("TODO: generate a stable symbol identifier")
