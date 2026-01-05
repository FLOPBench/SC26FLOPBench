from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from typing import Any

from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.backwards_slicing_agent import (
    default_backwards_slicing_system_prompt,
    make_backwards_slicing_agent,
)
from agents.llm_models import OpenRouterLLMSettings, build_openrouter_llm

from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
)


def delete_sqlite_db_if_exists(db_path: Path) -> None:
    """Delete the SQLite database file if it exists."""
    if db_path.exists():
        db_path.unlink()


SYSTEM_PROMPT = default_backwards_slicing_system_prompt()
INITIAL_PROMPT = HumanMessage(
    content=dedent(
        """\

        File to examine:
        - lulesh-cuda/*

        Target Kernel Name: `fill_sig<<<...>>>`

        As additional context, the full source files are located in the
        `/codex/gpuFLOPBench/src/lulesh-cuda` directory. The shell tool by default
        starts you in the `/codex/gpuFLOPBench/src` directory.

        """
    )
)

SQLITE_DB_PATH = Path(__file__).resolve().parent / "test_backwards_slicing_with_llm_checkpoint.sqlite"


def _message_to_str(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, default=_json_default)
    if isinstance(content, list):
        return json.dumps(content, default=_json_default)
    return str(content)


def _serialize_for_print(value: Any) -> str:
    if isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    try:
        return json.dumps(value, default=_json_default)
    except Exception:
        return repr(value)


def _json_default(obj: Any) -> str:
    return repr(obj)


_TOOL_DIR = Path(__file__).resolve().parents[1] / "langchain-tools" / "code-search-tools"


def _load_tool_module(filename: str, module_name: str) -> Any:
    tool_path = _TOOL_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {module_name} from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_code_search_tools() -> list[Any]:
    file_tree_module = _load_tool_module("cuda-file-tree.py", "code_search_tools.cuda_file_tree")
    global_functions_module = _load_tool_module(
        "cuda-global-functions.py", "code_search_tools.cuda_global_functions"
    )
    compile_commands_module = _load_tool_module(
        "cuda-compile-commands.py", "code_search_tools.cuda_compile_commands"
    )
    source_definition_module = _load_tool_module(
        "extract-kernel-source-definition.py", "code_search_tools.extract_kernel_source_definition"
    )
    main_files_module = _load_tool_module("cuda-main-files.py", "code_search_tools.cuda_main_files")
    include_tree_module = _load_tool_module(
        "include-tree-extractor.py", "code_search_tools.include_tree_extractor"
    )
    function_definitions_module = _load_tool_module(
        "function-definition-lister.py", "code_search_tools.function_definition_lister"
    )
    return [
        file_tree_module.cuda_file_tree,
        global_functions_module.cuda_global_functions,
        compile_commands_module.cuda_compile_commands,
        source_definition_module.extract_kernel_source_definition,
        main_files_module.cuda_main_files,
        include_tree_module.include_tree_extractor,
        function_definitions_module.function_definition_lister,
    ]


def test_backwards_slicing_agent_can_run():
    """The backwards-slicing agent should run with the script-oriented instructions."""

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        pytest.skip("OPENAI_API_KEY/OPENROUTER_API_KEY missing; skipping OpenRouter-backed test.")

    delete_sqlite_db_if_exists(SQLITE_DB_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    config = {"thread_id": "1"}
    checkpoint_query = {"configurable": {"thread_id": "1"}}
    result: dict[str, Any] | None = None

    try:
        openrouter_settings = OpenRouterLLMSettings(model_name="openai/gpt-5.1-codex-mini")
        llm = build_openrouter_llm(openrouter_settings)

        agent = make_backwards_slicing_agent(
            llm=llm,
            checkpointer=checkpointer,
            tools=_load_code_search_tools(),
            middleware=[
                ShellToolMiddleware(
                    workspace_root="/codex/gpuFLOPBench/src",
                    execution_policy=HostExecutionPolicy(),
                )
            ],  # middleware will be extended inside the helper
            system_prompt=SYSTEM_PROMPT,
        )

        config = {"thread_id": "1"}
        result = agent.invoke({"messages": [INITIAL_PROMPT]}, config=config)
        print("=== agent result messages ===")
        for idx, msg in enumerate(result.get("messages") or []):
            print(f"message[{idx}]:", _message_to_str(msg))
    except Exception as exc:  # pragma: no cover - allow output to show when failures happen
        print("backwards slicing agent invocation raised:", exc)
    finally:
        conn.close()

    # we still want to print the checkpointing results
    checkpoints = list(checkpointer.list(checkpoint_query, limit=5))
    print("=== recent checkpoints ===")
    for checkpoint in checkpoints:
        checkpoint_id = checkpoint.checkpoint["id"]
        ts = checkpoint.checkpoint["ts"]
        channel_values = checkpoint.checkpoint["channel_values"]
        print(f"- id={checkpoint_id} ts={ts}")
        print("  channel_values:", json.dumps(channel_values, indent=2, default=_json_default))
        if checkpoint.pending_writes:
            print("  pending writes:")
            for task_id, channel, value in checkpoint.pending_writes:
                print(f"    task={task_id} channel={channel} value={_serialize_for_print(value)}")

    assert isinstance(result, dict)
    assert result.get("messages"), "Agent run should always return at least one message"
