from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from textwrap import dedent

import pytest
import json
from typing import Any

from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.backwards_slicing_agent import (
    default_backwards_slicing_system_prompt,
    make_backwards_slicing_agent,
)
from agents.llm_models import OpenRouterLLMSettings, build_openrouter_llm


def delete_sqlite_db_if_exists(db_path: Path) -> None:
    """Delete the SQLite database file if it exists."""
    if db_path.exists():
        db_path.unlink()


SYSTEM_PROMPT = default_backwards_slicing_system_prompt()
INITIAL_PROMPT = HumanMessage(
    content=dedent(
        """\
        Your goal is to build a short Python analysis script that imports
        `treesitter_tools.cst_utils` and inspects the CUDA/OpenMP driver code.
        Select the repository files:
        - gpuFLOPBench/src/lulesh-cuda/lulesh.cu
        - gpuFLOPBench/src/lulesh-omp/lulesh.cc

        Steps:
        1. Use the built-in filesystem tools to create `/tmp/backwards_slice.py`.
        2. In the script, read both files with `treesitter_tools.cst_utils` and:
           - collect every `__global__`/`__device__` definition, recording the
             file path, line number, and at least the signature or snippet for each.
           - collect every OpenMP pragma region, including its directive, clauses,
             and associated statement span.
        3. Save the script to `/tmp/backwards_slice.py`, then execute it via the
           `execute` tool (`python /tmp/backwards_slice.py`).
        4. After the script runs, call the built-in `BackwardsSlicingState` tool
           with all required fields (`target_file`, `target_line`, `cuda_kernels`,
           `openmp_regions`) to store the structured results. Summaries can be
           textual (e.g., `"file:line -> kernel(__global__...)"`), and OpenMP
           regions should include pragma + clause text along with the span context.
        5. Return a final answer referencing the filled `BackwardsSlicingState`,
           mention the CLI command used, and confirm you did not modify the
           original source files.
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


def test_backwards_slicing_agent_can_run():
    """The backwards-slicing agent should run with the script-oriented instructions."""

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        pytest.skip("OPENAI_API_KEY/OPENROUTER_API_KEY missing; skipping OpenRouter-backed test.")

    delete_sqlite_db_if_exists(SQLITE_DB_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    config = {"thread_id": "1"}
    checkpoint_query = {"configurable": {"thread_id": "1"}}

    try:
        openrouter_settings = OpenRouterLLMSettings(model_name="openai/gpt-5.1-codex-mini")
        llm = build_openrouter_llm(openrouter_settings)

        agent = make_backwards_slicing_agent(
            llm=llm,
            checkpointer=checkpointer,
            tools=[],
            middleware=[],  # middleware will be extended inside the helper
            system_prompt=SYSTEM_PROMPT,
        )

        config = {"thread_id": "1"}
        result = agent.invoke({"messages": [INITIAL_PROMPT]}, config=config)
        print("=== agent result messages ===")
        for idx, msg in enumerate(result.get("messages") or []):
            print(f"message[{idx}]:", _message_to_str(msg))
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
    finally:
        conn.close()

    assert isinstance(result, dict)
    assert result.get("messages"), "Agent run should always return at least one message"
