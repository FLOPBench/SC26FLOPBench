from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import util
from pathlib import Path
import re
import sys
from typing import Any, Sequence, Literal

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime

from .descriptions import (
    FunctionDefinitionListerArgs,
    FUNCTION_DEFINITION_LISTER_DESCRIPTION,
)

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
_qualifier_block_start_line = _utils._qualifier_block_start_line

from tree_sitter import Language, Parser
import tree_sitter_cuda

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
PARSER = Parser(CUDA_LANGUAGE)



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

_QUALIFIER_KEYWORDS: tuple[str, ...] = (
    "__global__",
    "__device__",
    "__host__",
    "inline",
    "__shared__",
    "static",
)

@dataclass(frozen=True)
class FunctionEntry:
    name: str
    kind: str
    relative_file: str
    offset: int
    lines: int
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


_POINTER_STAR_PATTERN = re.compile(r"(?<=[\w>\)])\s+\*")


def _normalize_pointer_spacing(text: str) -> str:
    stripped = text.strip()
    return _POINTER_STAR_PATTERN.sub("*", stripped)


def _extract_declarator_prefix(declarator, function_declarator) -> str:
    if declarator is None or function_declarator is None:
        return ""
    declarator_text = _normalize_node_text(declarator)
    function_text = _normalize_node_text(function_declarator)
    if not function_text:
        return declarator_text.strip()
    if declarator_text.endswith(function_text):
        prefix = declarator_text[: -len(function_text)].strip()
        return prefix
    return declarator_text.strip()


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


def _maybe_add_function(
    start_node,
    context: tuple[str, ...],
    entries: list[FunctionEntry],
    relative_file: str,
    kind: str,
    templates: tuple[str, ...],
    lines: Sequence[str],
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
    start_line_idx = start_node.start_point[0]
    offset_idx = _qualifier_block_start_line(lines, start_line_idx)
    end_line_idx = max(start_line_idx, start_node.end_point[0])
    line_count = max(1, end_line_idx - offset_idx + 1)
    return_type = _get_return_type(start_node)
    pointer_prefix = _extract_declarator_prefix(declarator, function_declarator)
    if pointer_prefix:
        pointer_prefix = _normalize_pointer_spacing(pointer_prefix)
        if return_type:
            return_type = f"{return_type} {pointer_prefix}"
        else:
            return_type = pointer_prefix
    if return_type:
        return_type = _normalize_pointer_spacing(return_type)
    qualifiers = _extract_function_qualifiers(start_node)
    signature = _extract_function_signature(function_declarator)
    entries.append(
        FunctionEntry(
            qualified_name,
            kind,
            relative_file,
            offset_idx + 1,
            line_count,
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
    lines: Sequence[str],
) -> None:
    if node.type == "template_declaration":
        template_sig = _extract_template_signature(node)
        new_templates = template_stack
        if template_sig:
            new_templates = template_stack + (template_sig,)
        for child in node.named_children:
            _collect_functions(child, context, new_templates, entries, relative_file, lines)
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
            lines,
        )
    elif node.type == "function_declaration":
        _maybe_add_function(
            node,
            child_context,
            entries,
            relative_file,
            "decl",
            template_stack,
            lines,
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
                lines,
                declarator,
            )

    for child in node.named_children:
        _collect_functions(child, child_context, template_stack, entries, relative_file, lines)


def _extract_function_entries(text: str, relative_file: str) -> list[FunctionEntry]:
    tree = PARSER.parse(text.encode("utf8"))
    entries: list[FunctionEntry] = []
    lines = text.splitlines()
    _collect_functions(tree.root_node, (), (), entries, relative_file, lines)
    return entries


_BENCHMARK_DIR_SUFFIXES = ("-cuda", "-omp")


def _find_benchmark_root(source_file: Path) -> Path:
    for candidate in (source_file, *source_file.parents):
        if any(candidate.name.endswith(suffix) for suffix in _BENCHMARK_DIR_SUFFIXES):
            return candidate
    suffix_list = " or ".join(f"*-{suffix.lstrip('-')}" for suffix in _BENCHMARK_DIR_SUFFIXES)
    raise ValueError(f"{source_file!r} is not located under a benchmark directory matching {suffix_list}")


def _collect_entries_for_file(source_file: Path) -> list[FunctionEntry]:
    try:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    benchmark_root = _find_benchmark_root(source_file)
    relative_path = source_file.relative_to(benchmark_root).as_posix()
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


def _structure_function_entries(entries: list[FunctionEntry]) -> list[dict[str, Any]]:
    grouped: dict[str, list[FunctionEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.relative_file, []).append(entry)
    result: list[dict[str, Any]] = []
    for file in sorted(grouped):
        functions = grouped[file]
        functions.sort(key=lambda entry: (entry.offset, entry.name))
        serialized: list[dict[str, Any]] = []
        for fn in functions:
            serialized.append(
                {
                    "name": fn.name,
                    "kind": fn.kind,
                    "offset": fn.offset,
                    "lines": fn.lines,
                    "return_type": fn.return_type,
                    "templates": list(fn.templates),
                    "qualifiers": list(fn.qualifiers),
                    "signature": fn.signature,
                }
            )
        result.append({"file": file, "functions": serialized})
    return result


def _filter_function_entries(
    entries: list[FunctionEntry],
    qualifiers_filter: list[str] | None,
    template_only: bool,
    defs_or_decls: Literal["defs", "decls"] | None,
) -> list[FunctionEntry]:
    if qualifiers_filter:
        required_qualifiers = tuple(qualifiers_filter)
    else:
        required_qualifiers = ()
    filtered: list[FunctionEntry] = []
    for entry in entries:
        if required_qualifiers:
            qualifier_set = set(entry.qualifiers)
            if any(qualifier not in qualifier_set for qualifier in required_qualifiers):
                continue
        if template_only and not entry.templates:
            continue
        if defs_or_decls == "defs" and entry.kind != "defnt":
            continue
        if defs_or_decls == "decls" and entry.kind != "decl":
            continue
        filtered.append(entry)
    return filtered


def _local_function_definition_lister(
    file_path: str,
    qualifiers_filter: list[str] | None,
    template_only: bool,
    defs_or_decls: Literal["defs", "decls"] | None,
) -> list[dict[str, Any]]:
    source_file = _resolve_source_file(file_path)
    entries = _collect_entries_for_file(source_file)
    filtered = _filter_function_entries(entries, qualifiers_filter, template_only, defs_or_decls)
    return _structure_function_entries(filtered)


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


def _backend_function_definition_lister(
    dir_path: str,
    backend: BackendProtocol,
    qualifiers_filter: list[str] | None,
    template_only: bool,
    defs_or_decls: Literal["defs", "decls"] | None,
) -> list[dict[str, Any]]:
    source_file = _resolve_backend_file_path(dir_path, backend)
    entries = _collect_entries_for_file(source_file)
    filtered = _filter_function_entries(entries, qualifiers_filter, template_only, defs_or_decls)
    return _structure_function_entries(filtered)


def make_function_definition_lister_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol] | None = None,
    *,
    description: str | None = None,
) -> StructuredTool:
    tool_description = description or FUNCTION_DEFINITION_LISTER_DESCRIPTION

    def _run(
        file_path: str,
        qualifiers_filter: list[str] | None = None,
        template_only: bool = False,
        defs_or_decls: Literal["defs", "decls"] | None = None,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> list[dict[str, Any]]:
        validated = _validate_path(file_path)
        if backend is None:
            return _local_function_definition_lister(validated, qualifiers_filter, template_only, defs_or_decls)
        resolved_backend = _get_backend(backend, runtime)
        return _backend_function_definition_lister(
            validated,
            resolved_backend,
            qualifiers_filter,
            template_only,
            defs_or_decls,
        )

    async def _arun(
        file_path: str,
        qualifiers_filter: list[str] | None = None,
        template_only: bool = False,
        defs_or_decls: Literal["defs", "decls"] | None = None,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> list[dict[str, Any]]:
        validated = _validate_path(file_path)
        if backend is None:
            return _local_function_definition_lister(validated, qualifiers_filter, template_only, defs_or_decls)
        resolved_backend = _get_backend(backend, runtime)
        return _backend_function_definition_lister(
            validated,
            resolved_backend,
            qualifiers_filter,
            template_only,
            defs_or_decls,
        )

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="function_definition_lister",
        description=tool_description,
        args_schema=FunctionDefinitionListerArgs,
    )
