"""Smoke tests for the default LangChain filesystem tools."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import _glob_tool_generator, _ls_tool_generator
from langgraph.prebuilt.tool_node import ToolRuntime


class _DummyStreamWriter:
    def write(self, chunk: str) -> None:
        del chunk


def _make_runtime() -> ToolRuntime:
    return ToolRuntime(
        state={},
        context={},
        config=SimpleNamespace(),
        stream_writer=_DummyStreamWriter(),
        tool_call_id="test_langchain_tools",
        store=None,
    )


def _make_backend_factory() -> FilesystemBackend:
    return FilesystemBackend(root_dir="/gpuFLOPBench-updated/HeCBench/src", virtual_mode=True)


def _parse_tool_output(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = []
    return parsed


def test_ls_root_returns_paths() -> None:
    """Ensure the ls tool returns actual paths when rooted at the source tree."""
    backend_factory = lambda runtime: _make_backend_factory()
    ls_tool = _ls_tool_generator(backend_factory)
    runtime = _make_runtime()

    raw = ls_tool.func(runtime, "/")
    paths = _parse_tool_output(raw)
    print("ls / ->", paths[:5])
    assert paths, "Expected ls / to list at least one entry"


def test_ls_cuda_subdir_returns_entries() -> None:
    backend_factory = lambda runtime: _make_backend_factory()
    ls_tool = _ls_tool_generator(backend_factory)
    runtime = _make_runtime()

    raw = ls_tool.func(runtime, "/lulesh-cuda")
    paths = _parse_tool_output(raw)
    print("ls /HeCBench/src ->", paths[:5])
    assert paths, "Expected ls /HeCBench/src to show files/directories"


@pytest.mark.parametrize("pattern", ["**/*.cu", "**/*.h"])
def test_glob_patterns_match_files(pattern: str) -> None:
    backend_factory = lambda runtime: _make_backend_factory()
    glob_tool = _glob_tool_generator(backend_factory)
    runtime = _make_runtime()

    raw = glob_tool.func(pattern, runtime, "/")
    paths = _parse_tool_output(raw)
    print(f"glob {pattern} ->", paths[:5])
    assert paths, f"Expected glob {pattern} to return at least one match"
