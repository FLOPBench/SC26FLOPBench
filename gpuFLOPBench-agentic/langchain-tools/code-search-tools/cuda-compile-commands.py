from __future__ import annotations

from importlib import util
import functools
import json
import shlex
from pathlib import Path
import sys
from typing import Any

from langchain.tools import tool

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
_resolve_directory = _utils._resolve_directory
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


def _path_contains_prefix(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    if not prefix:
        return True
    for i in range(len(parts) - len(prefix) + 1):
        if parts[i : i + len(prefix)] == prefix:
            return True
    return False


def _gather_compile_entries(cuda_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    relative_root = cuda_dir.relative_to(_GPU_SRC_DIR)
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


@tool(
    "cuda_compile_commands",
    args_schema=DirectoryArgs,
    description=(
        "Return the compiler arguments listed in gpuFLOPBench/cuda-profiling/compile_commands.json for the provided CUDA directory. "
        "Example: cuda_compile_commands(dir_path=\"/lulesh-cuda\")."
    ),
)
def cuda_compile_commands(dir_path: str) -> dict[str, Any]:
    cuda_dir = _resolve_directory(dir_path)
    entries = _gather_compile_entries(cuda_dir)
    if not entries:
        raise ValueError(f"No compile commands were found for {dir_path!r}")
    return {"dir_path": str(cuda_dir), "commands": entries}
