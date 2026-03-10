"""Smoke test that bakes the FilesystemBackend into the CUDA code-search workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = REPO_ROOT / "langchain-tools" / "code-search-tools"
_SOLUTIONS_DIR = (
    REPO_ROOT
    / "unit-tests"
    / "extracted-kernel-solutions"
    / "lulesh-cuda-solutions"
)
_LULESH_ROOT = REPO_ROOT / "gpuFLOPBench" / "src" / "lulesh-cuda"


def _ensure_utils_module() -> None:
    module_name = "code_search_tools.utils"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, _TOOLS_DIR / "utils.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the code search helper utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _ensure_descriptions_module()


def _ensure_descriptions_module() -> None:
    module_name = "code_search_tools.descriptions"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "descriptions.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the code search descriptions module")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _load_function_definition_tool(backend: FilesystemBackend | None = None):
    _ensure_utils_module()
    module_name = "code_search_tools.function_definition_lister"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            _TOOLS_DIR / "function-definition-lister.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load the function-definition-lister tool")
        module = importlib.util.module_from_spec(spec)
        spec.submodule_search_locations = [str(_TOOLS_DIR)]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.make_function_definition_lister_tool(backend=backend)


def _load_file_tree_tool(backend: FilesystemBackend):
    _ensure_utils_module()
    module_name = "code_search_tools.cuda_file_tree"
    if module_name in sys.modules:
        return sys.modules[module_name].make_cuda_file_tree_tool(backend=backend)
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "cuda-file-tree.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the cuda-file-tree tool")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_cuda_file_tree_tool(backend=backend)


def _load_global_functions_tool(backend: FilesystemBackend):
    _ensure_utils_module()
    module_name = "code_search_tools.cuda_global_functions"
    if module_name in sys.modules:
        return sys.modules[module_name].make_cuda_global_functions_tool(backend=backend)
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "cuda-global-functions.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the cuda-global-functions tool")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_cuda_global_functions_tool(backend=backend)


def _load_main_files_tool(backend: FilesystemBackend):
    _ensure_utils_module()
    module_name = "code_search_tools.cuda_main_files"
    if module_name in sys.modules:
        return sys.modules[module_name].make_cuda_main_files_tool(backend=backend)
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "cuda-main-files.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the cuda-main-files tool")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_cuda_main_files_tool(backend=backend)


def _load_include_tree_tool(backend: FilesystemBackend):
    _ensure_utils_module()
    module_name = "code_search_tools.include_tree_extractor"
    if module_name in sys.modules:
        return sys.modules[module_name].make_include_tree_extractor_tool(backend=backend)
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "include-tree-extractor.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the include-tree-extractor tool")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_include_tree_extractor_tool(backend=backend)


def _load_kernel_source_tool(backend: FilesystemBackend):
    _ensure_utils_module()
    module_name = "code_search_tools.extract_kernel_source_definition"
    if module_name in sys.modules:
        return sys.modules[module_name].make_extract_kernel_source_definition_tool(backend=backend)
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "extract-kernel-source-definition.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the extract-kernel-source-definition tool")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_extract_kernel_source_definition_tool(backend=backend)


def _load_compile_commands_tool(backend: FilesystemBackend):
    _ensure_utils_module()
    module_name = "code_search_tools.cuda_compile_commands"
    if module_name in sys.modules:
        return sys.modules[module_name].make_cuda_compile_commands_tool(backend=backend)
    spec = importlib.util.spec_from_file_location(
        module_name,
        _TOOLS_DIR / "cuda-compile-commands.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the cuda-compile-commands tool")
    module = importlib.util.module_from_spec(spec)
    spec.submodule_search_locations = [str(_TOOLS_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_cuda_compile_commands_tool(backend=backend)


def _make_backend() -> FilesystemBackend:
    return FilesystemBackend(root_dir="/codex/gpuFLOPBench/src/lulesh-cuda/", virtual_mode=True)

def _assert_function_entry_ranges(entries: list[dict[str, Any]]) -> None:
    for file_entry in entries:
        for fn in file_entry.get("functions", []):
            offset = fn.get("offset")
            line_count = fn.get("lines")
            assert isinstance(offset, int) and offset >= 1
            assert isinstance(line_count, int) and line_count >= 1


def _load_expected_function_list_json() -> list[dict[str, Any]]:
    path = _SOLUTIONS_DIR / "function_definitions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AssertionError(f"could not read {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON in {path}") from exc


def _load_expected_templated_function_definitions_json() -> str:
    path = _SOLUTIONS_DIR / "templated_function_definitions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AssertionError(f"could not read {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON in {path}") from exc


def _assert_function_snippets(entries: list[dict[str, Any]], backend: FilesystemBackend) -> None:
    root = _LULESH_ROOT
    text_cache: dict[Path, list[str]] = {}
    for file_entry in entries:
        file_name = file_entry["file"]
        path = root / file_name
        if path not in text_cache:
            try:
                text_cache[path] = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                raise AssertionError(f"could not read {path}")
        lines = text_cache[path]
        for fn in file_entry.get("functions", []):
            offset = fn.get("offset", 1) - 1
            line_count = fn.get("lines", 1)
            assert offset >= 0 and line_count >= 1
            snippet_lines, snippet = _collect_snippet(lines, offset, line_count)
            function_name = fn.get("name")
            if function_name:
                simple_name = function_name.split("::")[-1]
                assert simple_name in snippet
            qualifiers = fn.get("qualifiers") or []
            for qualifier in qualifiers:
                assert qualifier in snippet
            signature = fn.get("signature")
            if signature:
                normalized_signature = "".join(signature.split())
                normalized_snippet = "".join(snippet.split())
                assert normalized_signature in normalized_snippet
            if fn.get("kind") == "defnt":
                assert _balanced_braces(snippet)


def _collect_snippet(lines: list[str], offset: int, line_count: int) -> tuple[list[str], str]:
    end_idx = min(offset + line_count, len(lines))
    snippet_lines = lines[offset:end_idx]
    return snippet_lines, "\n".join(snippet_lines)


def _balanced_braces(snippet: str) -> bool:
    stack: list[str] = []
    for ch in snippet:
        if ch == "{":
            stack.append(ch)
        elif ch == "}":
            if not stack:
                return False
            stack.pop()
    return not stack


def test_function_definition_lister_via_filesystem_backend() -> None:
    """Ensure the function-definition-lister works when the agent is sandboxed to "/"."""

    backend = _make_backend()
    root_entries = backend.ls_info("/")
    assert root_entries, "FilesystemBackend should expose benchmark files at /"
    assert any(entry["path"].endswith(".cu") for entry in root_entries)

    snippet = backend.read("/lulesh.cu", limit=1000)
    assert "__global__ void fill_sig" in snippet

    tool = _load_function_definition_tool(backend)
    file_paths = ("/lulesh.cu", "/lulesh.h")
    actual_entries: list[dict[str, Any]] = []
    for file_path in file_paths:
        result = tool.run({"file_path": file_path, "defs_or_decls": "defs"})
        assert result, f"Function definition lister returned no entries for {file_path}"
        actual_entries.extend(result)
    expected_entries = _load_expected_function_list_json()
    assert actual_entries == expected_entries
    _assert_function_entry_ranges(actual_entries)
    _assert_function_snippets(actual_entries, backend)

    tree_tool = _load_file_tree_tool(backend)
    tree_output = tree_tool.run({"dir_path": "/"})
    assert tree_output.startswith("/")
    assert "lulesh.cu" in tree_output


def test_function_definition_lister_templated_defs_via_filesystem_backend() -> None:
    backend = _make_backend()
    tool = _load_function_definition_tool(backend)
    templated_results = tool.run(
        {
            "file_path": "/lulesh.cu",
            "template_only": True,
            "defs_or_decls": "defs",
        }
    )
    assert templated_results, "Expect templated function definitions to be reported"
    assert all(
        fn.get("kind") == "defnt" for file_entry in templated_results for fn in file_entry.get("functions", [])
    )
    expected_entries = _load_expected_templated_function_definitions_json()
    assert templated_results == expected_entries


def test_cuda_global_functions_via_filesystem_backend() -> None:
    backend = _make_backend()
    root_entries = backend.ls_info("/")
    assert root_entries
    assert any(entry["path"].endswith(".cu") for entry in root_entries)

    global_fn_tool = _load_global_functions_tool(backend)
    results = global_fn_tool.run({"dir_path": "/"})
    assert results, "cuda_global_functions should return kernel metadata"
    assert all("file" in entry and "kernel" in entry and "line" in entry for entry in results)
    assert any(entry["kernel"] == "fill_sig" for entry in results)
    assert all(isinstance(entry.get("offset"), int) and entry["offset"] >= 1 for entry in results)
    assert all(isinstance(entry.get("lines"), int) and entry["lines"] >= 1 for entry in results)


def test_cuda_main_files_via_filesystem_backend() -> None:
    backend = _make_backend()
    main_files_tool = _load_main_files_tool(backend)
    result = main_files_tool.run({"dir_path": "/"})
    expected_main = "lulesh.cu"
    file_names = {entry["file"] for entry in result}
    assert expected_main in file_names
    assert all(isinstance(entry.get("offset"), int) and entry["offset"] >= 1 for entry in result)
    assert all(isinstance(entry.get("lines"), int) and entry["lines"] >= 1 for entry in result)


def test_extract_kernel_source_definition_via_filesystem_backend() -> None:
    backend = _make_backend()
    extractor_tool = _load_kernel_source_tool(backend)
    # verify both directory and file mode work via the backend
    dir_results = extractor_tool.run({"file_path": "/", "kernel_name": "fill_sig"})
    assert dir_results, "Extracted kernel data should be available when pointing at the directory"
    assert any(entry["kernel"] == "fill_sig" for entry in dir_results)
    file_results = extractor_tool.run({"file_path": "/lulesh.cu", "kernel_name": "fill_sig"})
    assert file_results
    assert file_results[0]["file"] == "lulesh.cu"
    assert "__global__ void fill_sig" in file_results[0]["source"]


def test_include_tree_extractor_via_filesystem_backend() -> None:
    backend = _make_backend()
    include_tool = _load_include_tree_tool(backend)
    output = include_tool.run({"file_path": "/lulesh.cu"})
    assert output.startswith("lulesh.cu")
    assert '#include "lulesh.h"' in output


def test_cuda_compile_commands_via_filesystem_backend() -> None:
    backend = _make_backend()
    compile_tool = _load_compile_commands_tool(backend)
    result = compile_tool.run({"dir_path": "/"})
    assert result.get("commands"), "Expected compile commands to exist for lulesh-cuda"
    assert any(entry["file"] == "lulesh.cu" for entry in result["commands"])
    assert result["dir_path"].endswith("/lulesh-cuda")
