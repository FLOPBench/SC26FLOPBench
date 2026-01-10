from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime
from pydantic import BaseModel, Field

from tree_sitter import Language, Parser
import tree_sitter_cuda

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
PARSER = Parser(CUDA_LANGUAGE)


class FunctionDefinitionListerArgs(BaseModel):
    """Arguments for listing declarations/definitions inside a specific source file."""

    file_path: str = Field(
        ...,
        description=(
            "Absolute path to the CUDA/C++ file on disk, or the virtual path that "
            "the FilesystemBackend exposes (e.g., `/lulesh-cuda/lulesh.cu`)."
        ),
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

_QUALIFIER_KEYWORDS: tuple[str, ...] = ("__global__", "__device__", "__host__", "inline", "__shared__")

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


def _collect_entries_for_file(source_file: Path) -> list[FunctionEntry]:
    try:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    relative_path = source_file.name
    return _extract_function_entries(text, relative_path)


def _resolve_source_file(file_path: str) -> Path:
    path = Path(file_path)
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ValueError(f"{file_path!r} is not a valid path") from exc
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{file_path!r} is not a file")
    return resolved
TOOL_DESCRIPTION = (
    "Return every function declaration or definition found in the provided CUDA/C++/header file. "
    "Pass either an actual disk path or the virtual FilesystemBackend path (e.g., `/lulesh-cuda/lulesh.cu`)."
)


def _formatted_entries(entries: list[FunctionEntry]) -> str:
    entries.sort(key=lambda entry: (entry.relative_file, entry.line, entry.name))
    return "\n".join(_format_function_line(entry) for entry in entries)


def _entries_from_source_file(source_file: Path) -> str:
    entries = _collect_entries_for_file(source_file)
    return _formatted_entries(entries)


def _local_function_definition_lister(file_path: str) -> str:
    source_file = _resolve_source_file(file_path)
    return _entries_from_source_file(source_file)


def _backend_candidate_paths(dir_path: str, backend: BackendProtocol) -> list[str]:
    candidates: list[str] = [dir_path]
    backend_cwd = getattr(backend, "cwd", None)
    if backend_cwd is not None:
        cwd_path = Path(backend_cwd)
        normalized = dir_path.lstrip("/")
        if normalized:
            candidates.append(str(cwd_path / normalized))
            root_name = cwd_path.name
            prefix = f"{root_name}/"
            if normalized.startswith(prefix):
                remainder = normalized[len(prefix) :]
                candidates.append(str(cwd_path / remainder))
    return candidates


def _resolve_backend_file_path(dir_path: str, backend: BackendProtocol) -> Path:
    for candidate in _backend_candidate_paths(dir_path, backend):
        try:
            return _resolve_source_file(candidate)
        except ValueError:
            continue
    raise ValueError(f"{dir_path!r} could not be resolved via the backend")


def _backend_function_definition_lister(dir_path: str, backend: BackendProtocol) -> str:
    source_file = _resolve_backend_file_path(dir_path, backend)
    return _entries_from_source_file(source_file)


def make_function_definition_lister_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol] | None = None,
    *,
    description: str | None = None,
) -> StructuredTool:
    tool_description = description or TOOL_DESCRIPTION

    def _run(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> str:
        validated = _validate_path(file_path)
        if backend is None:
            return _local_function_definition_lister(validated)
        resolved_backend = _get_backend(backend, runtime)
        return _backend_function_definition_lister(validated, resolved_backend)

    async def _arun(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> str:
        validated = _validate_path(file_path)
        if backend is None:
            return _local_function_definition_lister(validated)
        resolved_backend = _get_backend(backend, runtime)
        return _backend_function_definition_lister(validated, resolved_backend)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="function_definition_lister",
        description=tool_description,
        args_schema=FunctionDefinitionListerArgs,
    )
