"""Smoke test that bakes the FilesystemBackend into the CUDA code-search workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = REPO_ROOT / "langchain-tools" / "code-search-tools"


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


def test_function_definition_lister_via_filesystem_backend() -> None:
    """Ensure the function-definition-lister works when the agent is sandboxed to "/"."""

    backend = _make_backend()
    root_entries = backend.ls_info("/")
    assert root_entries, "FilesystemBackend should expose benchmark files at /"
    assert any(entry["path"].endswith(".cu") for entry in root_entries)

    snippet = backend.read("/lulesh.cu", limit=1000)
    assert "__global__ void fill_sig" in snippet

    tool = _load_function_definition_tool(backend)
    output = tool.run({"file_path": "/lulesh.cu"})
    assert output, "Function definition lister should return the parsed entries"
    assert "__global__ void fill_sig(" in output
    assert "__device__ static inline void CalcElemShapeFunctionDerivatives(" in output
    assert "__device__ static inline void CalcElemNodeNormals(" in output
    assert "__device__ static inline void SumElemStressesToNodeForces(" in output
    assert "template <typename T> T *Allocate(size_t size)" in output
    assert "(defnt)" in output

    tree_tool = _load_file_tree_tool(backend)
    tree_output = tree_tool.run({"dir_path": "/"})
    assert tree_output.startswith("lulesh-cuda/")
    assert "lulesh.cu" in tree_output


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


def test_cuda_main_files_via_filesystem_backend() -> None:
    backend = _make_backend()
    main_files_tool = _load_main_files_tool(backend)
    result = main_files_tool.run({"dir_path": "/"})
    expected_main = "lulesh.cu"
    assert expected_main in result


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
