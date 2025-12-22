from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
import sys
from typing import Sequence

from pydantic import Field

from langchain.tools import tool

from tree_sitter import Language, Parser
import tree_sitter_cuda

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
PARSER = Parser(CUDA_LANGUAGE)

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
CudaSubdirArgs = _utils.CudaSubdirArgs
_resolve_cuda_dir = _utils._resolve_cuda_dir
_gather_cuda_files = _utils._gather_cuda_files


class FunctionDefinitionListerArgs(CudaSubdirArgs):
    """Arguments for listing declarations/definitions inside a specific benchmark source file."""

    file_name: str | None = Field(
        None,
        description="Relative path of the CUDA/C++ file (within the benchmark) to inspect.",
    )

_SCOPED_NODE_NAME_TYPES: dict[str, tuple[str, ...]] = {
    "class_specifier": ("type_identifier", "identifier"),
    "struct_specifier": ("type_identifier", "identifier"),
    "union_specifier": ("type_identifier", "identifier"),
    "namespace_definition": ("namespace_identifier", "identifier"),
}

_NAME_LEAF_TYPES: tuple[str, ...] = (
    "identifier",
    "field_identifier",
    "destructor_name",
    "operator_function_id",
    "operator_name",
)

_TEMPLATE_PARAMETER_NODE_TYPES: tuple[str, ...] = (
    "template_parameter_list",
    "type_parameter_list",
    "template_type_parameter_list",
)

_QUALIFIER_KEYWORDS: tuple[str, ...] = ("__global__", "__device__", "__host__", "inline")

@dataclass(frozen=True)
class FunctionEntry:
    name: str
    kind: str
    relative_file: str
    line: int
    return_type: str | None
    templates: tuple[str, ...]
    qualifiers: tuple[str, ...]
    signature: str | None


def _scope_name_for_node(node) -> str | None:
    expected_types = _SCOPED_NODE_NAME_TYPES.get(node.type)
    if not expected_types:
        return None
    for child in node.named_children:
        if child.type in expected_types:
            name = child.text.decode("utf8").strip()
            if name:
                return name
    return None


def _find_function_name(node) -> str | None:
    if node.type == "qualified_identifier":
        text = node.text.decode("utf8").strip()
        if text:
            return text
    for child in node.named_children:
        name = _find_function_name(child)
        if name:
            return name
    if node.type in _NAME_LEAF_TYPES:
        text = node.text.decode("utf8").strip()
        if text:
            return text
    return None


def _qualify_name(base_name: str, context: Sequence[str]) -> str:
    if not context:
        return base_name
    prefix = "::".join(context)
    if not prefix:
        return base_name
    if base_name.startswith(prefix + "::") or base_name == prefix:
        return base_name
    return f"{prefix}::{base_name}"


def _normalize_node_text(node) -> str:
    return " ".join(node.text.decode("utf8").split())


def _extract_template_signature(node) -> str | None:
    for child in node.named_children:
        if child.type in _TEMPLATE_PARAMETER_NODE_TYPES:
            params = _normalize_node_text(child)
            if params:
                return f"template {params}"
    return None


def _get_return_type(node) -> str | None:
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return None
    text = _normalize_node_text(type_node)
    if not text:
        return None
    return text


def _extract_function_signature(declarator) -> str | None:
    if declarator is None:
        return None
    text = _normalize_node_text(declarator)
    if not text:
        return None
    return text


def _find_function_declarator(node):
    if node.type == "function_declarator":
        return node
    for child in node.named_children:
        found = _find_function_declarator(child)
        if found:
            return found
    return None


def _extract_function_qualifiers(node) -> tuple[str, ...]:
    qualifiers: list[str] = []
    for child in node.children:
        text = child.text.decode("utf8").strip()
        if text in _QUALIFIER_KEYWORDS:
            qualifiers.append(text)
    return tuple(qualifiers)


def _format_function_line(entry: FunctionEntry) -> str:
    parts: list[str] = []
    if entry.templates:
        parts.append(" ".join(entry.templates))
    if entry.qualifiers:
        parts.append(" ".join(entry.qualifiers))
    if entry.return_type:
        parts.append(entry.return_type)
    if entry.signature:
        parts.append(entry.signature)
    else:
        parts.append(entry.name)
    parts.append(f"({entry.kind})")
    return " ".join(parts)


def _maybe_add_function(
    start_node,
    context: tuple[str, ...],
    entries: list[FunctionEntry],
    relative_file: str,
    kind: str,
    templates: tuple[str, ...],
    declarator_node=None,
) -> None:
    declarator = declarator_node or start_node.child_by_field_name("declarator")
    if declarator is None:
        return
    function_declarator = _find_function_declarator(declarator)
    if function_declarator is None:
        return
    base_name = _find_function_name(declarator)
    if not base_name:
        return
    qualified_name = _qualify_name(base_name, context)
    line = start_node.start_point[0] + 1
    return_type = _get_return_type(start_node)
    qualifiers = _extract_function_qualifiers(start_node)
    signature = _extract_function_signature(declarator)
    entries.append(
        FunctionEntry(
            qualified_name,
            kind,
            relative_file,
            line,
            return_type,
            templates,
            qualifiers,
            signature,
        )
    )


def _collect_functions(
    node,
    context: tuple[str, ...],
    template_stack: tuple[str, ...],
    entries: list[FunctionEntry],
    relative_file: str,
) -> None:
    if node.type == "template_declaration":
        template_sig = _extract_template_signature(node)
        new_templates = template_stack
        if template_sig:
            new_templates = template_stack + (template_sig,)
        for child in node.named_children:
            _collect_functions(child, context, new_templates, entries, relative_file)
        return

    scope_name = _scope_name_for_node(node)
    child_context = context
    if scope_name:
        child_context = context + (scope_name,)

    if node.type == "function_definition":
        _maybe_add_function(
            node,
            child_context,
            entries,
            relative_file,
            "defnt",
            template_stack,
        )
    elif node.type == "function_declaration":
        _maybe_add_function(
            node,
            child_context,
            entries,
            relative_file,
            "decl",
            template_stack,
        )
    elif node.type == "field_declaration":
        declarator = node.child_by_field_name("declarator")
        if declarator is not None and declarator.type == "function_declarator":
            _maybe_add_function(
                node,
                child_context,
                entries,
                relative_file,
                "decl",
                template_stack,
                declarator,
            )

    for child in node.named_children:
        _collect_functions(child, child_context, template_stack, entries, relative_file)


def _extract_function_entries(text: str, relative_file: str) -> list[FunctionEntry]:
    tree = PARSER.parse(text.encode("utf8"))
    entries: list[FunctionEntry] = []
    _collect_functions(tree.root_node, (), (), entries, relative_file)
    return entries


def _resolve_source_file(cuda_dir: Path, file_name: str) -> Path:
    requested_path = Path(file_name)
    if requested_path.is_absolute():
        raise ValueError("file_name must be relative to the benchmark directory")
    candidate = cuda_dir / requested_path
    if not candidate.exists():
        raise ValueError(f"{file_name!r} does not exist under {cuda_dir}")
    if not candidate.is_file():
        raise ValueError(f"{file_name!r} is not a file")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(cuda_dir)
    except ValueError:
        raise ValueError("file_name escapes the benchmark directory")
    return resolved


def _collect_entries_for_file(source_file: Path, cuda_dir: Path) -> list[FunctionEntry]:
    try:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    relative_path = str(source_file.relative_to(cuda_dir))
    return _extract_function_entries(text, relative_path)


@tool(
    "function_definition_lister",
    args_schema=FunctionDefinitionListerArgs,
    description=(
        "Return an alphabetically ordered list of every function declaration and definition "
        "found in a benchmark's CUDA/C++/header sources. Provide an optional file_name "
        "(relative to the benchmark directory) to inspect a single file; omit it to scan every "
        "supported source file together."
    ),
)
def function_definition_lister(cuda_name: str, file_name: str | None = None) -> str:
    cuda_dir = _resolve_cuda_dir(cuda_name)
    entries: list[FunctionEntry] = []
    if file_name:
        target_file = _resolve_source_file(cuda_dir, file_name)
        entries.extend(_collect_entries_for_file(target_file, cuda_dir))
    else:
        for source_file in sorted(_gather_cuda_files(cuda_dir)):
            entries.extend(_collect_entries_for_file(source_file, cuda_dir))
    entries.sort(key=lambda entry: (entry.relative_file, entry.line, entry.name))
    return "\n".join(_format_function_line(entry) for entry in entries)
