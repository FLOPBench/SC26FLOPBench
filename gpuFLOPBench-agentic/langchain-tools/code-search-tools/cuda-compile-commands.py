from __future__ import annotations

from importlib import util
import functools
import json
import shlex
from pathlib import Path
import sys
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemState, _get_backend, _validate_path
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolRuntime
from pydantic import Field

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
_GPU_SRC_DIR = _utils.GPU_SRC_DIR
DirectoryArgs = _utils.DirectoryArgs

_REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_COMMANDS_PATH = (
    _REPO_ROOT / "gpuFLOPBench" / "cuda-profiling" / "compile_commands.json"
)


@functools.lru_cache(maxsize=1)
def _load_compile_commands() -> list[dict[str, Any]]:
    if not COMPILE_COMMANDS_PATH.exists():
        raise FileNotFoundError(
            f"{COMPILE_COMMANDS_PATH} was not found. Have you generated the database?"
        )
    try:
        data = json.loads(COMPILE_COMMANDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("compile_commands.json contains invalid JSON") from exc
    if not isinstance(data, list):
        raise ValueError("compile_commands.json expected a list of entries")
    return data


def _gather_compile_entries(cuda_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        relative_root = cuda_dir.relative_to(_GPU_SRC_DIR)
    except ValueError as exc:
        raise ValueError(f"{cuda_dir!r} is not under {_GPU_SRC_DIR}") from exc
    root_parts = relative_root.parts
    for entry in _load_compile_commands():
        file_path = Path(entry.get("file", ""))
        if not file_path.is_absolute():
            directory = Path(entry.get("directory", ""))
            file_path = (directory / file_path).resolve()
        if not _path_contains_prefix(file_path.parts, root_parts):
            continue
        command = entry.get("command")
        if not command:
            continue
        tokens = shlex.split(command)
        if not tokens:
            continue
        entries.append(
            {
                "file": file_path.name,
                "directory": entry.get("directory", ""),
                "output": entry.get("output"),
                "compiler": tokens[0],
                "arguments": tokens[1:],
                "command": command,
            }
        )
    entries.sort(key=lambda item: item["file"])
    return entries


def _path_contains_prefix(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    if not prefix:
        return True
    for i in range(len(parts) - len(prefix) + 1):
        if parts[i : i + len(prefix)] == prefix:
            return True
    return False


def _normalize_virtual_dir_path(dir_path: str) -> str:
    trimmed = dir_path.rstrip("/")
    return trimmed or "/"


def _resolve_backend_directory(dir_path: str, backend: BackendProtocol) -> Path:
    normalized = _normalize_virtual_dir_path(dir_path)
    if getattr(backend, "virtual_mode", False):
        cwd = getattr(backend, "cwd", None)
        if cwd is None:
            raise ValueError("Backend does not expose a root directory")
        relative = normalized.lstrip("/")
        candidate = (Path(cwd) / relative).resolve() if relative else cwd.resolve()
        try:
            candidate.relative_to(cwd)
        except ValueError:
            raise ValueError(f"{dir_path!r} escapes the backend root directory")
    else:
        candidate = Path(normalized)
        if not candidate.is_absolute():
            base = getattr(backend, "cwd", None) or Path.cwd()
            candidate = (base / candidate).resolve()
        else:
            candidate = candidate.resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"{dir_path!r} is not a directory")
    return candidate


def _local_compile_commands(dir_path: str) -> dict[str, Any]:
    cuda_dir = _resolve_directory(dir_path)
    entries = _gather_compile_entries(cuda_dir)
    if not entries:
        raise ValueError(f"No compile commands were found for {dir_path!r}")
    return {"dir_path": str(cuda_dir), "commands": entries}


TOOL_DESCRIPTION = (
    "Return the compiler arguments listed in gpuFLOPBench/cuda-profiling/compile_commands.json for the provided CUDA directory. "
    "Example: cuda_compile_commands(dir_path=\"/lulesh-cuda\")."
)


def make_cuda_compile_commands_tool(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    *,
    description: str | None = None,
) -> StructuredTool:
    tool_description = description or TOOL_DESCRIPTION

    def _run(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> dict[str, Any]:
        validated = _validate_path(dir_path)
        resolved_backend = _get_backend(backend, runtime)
        directory = _resolve_backend_directory(validated, resolved_backend)
        entries = _gather_compile_entries(directory)
        if not entries:
            raise ValueError(f"No compile commands were found for {dir_path!r}")
        return {"dir_path": str(directory), "commands": entries}

    async def _arun(
        dir_path: str,
        runtime: ToolRuntime[None, FilesystemState] | None = None,
    ) -> dict[str, Any]:
        return _run(dir_path, runtime)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="cuda_compile_commands",
        description=tool_description,
        args_schema=DirectoryArgs,
    )
