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
CudaSubdirArgs = _utils.CudaSubdirArgs
_resolve_cuda_dir = _utils._resolve_cuda_dir

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


def _gather_compile_entries(cuda_name: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in _load_compile_commands():
        file_path = Path(entry.get("file", ""))
        if cuda_name not in file_path.parts:
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
    args_schema=CudaSubdirArgs,
    description="Return the compiler arguments listed in gpuFLOPBench/cuda-profiling/compile_commands.json for the requested *-cuda benchmark.",
)
def cuda_compile_commands(cuda_name: str) -> dict[str, Any]:
    _resolve_cuda_dir(cuda_name)
    entries = _gather_compile_entries(cuda_name)
    if not entries:
        raise ValueError(f"No compile commands were found for {cuda_name!r}")
    return {"cuda_name": cuda_name, "commands": entries}
