from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.backwards_slicing_agent import make_backwards_slicing_agent
from agents.llm_models import OpenRouterLLMSettings, build_openrouter_llm

_CODE_SEARCH_TOOLS_DIR = (
    Path(__file__).resolve().parents[1] / "mcp-servers" / "code-search-tools"
)

_CODE_SEARCH_TOOL_SPECS = (
    ("cuda-file-tree.py", "code_search_tools.cuda_file_tree", "cuda_file_tree"),
    ("cuda-global-functions.py", "code_search_tools.cuda_global_functions", "cuda_global_functions"),
    ("cuda-compile-commands.py", "code_search_tools.cuda_compile_commands", "cuda_compile_commands"),
    ("cuda-main-files.py", "code_search_tools.cuda_main_files", "cuda_main_files"),
    (
        "extract-kernel-source-definition.py",
        "code_search_tools.extract_kernel_source_definition",
        "extract_kernel_source_definition",
    ),
)


def _load_code_search_tool(filename: str, module_name: str, attribute: str):
    """Load a LangChain tool object from the split code-search-tools modules."""

    path = _CODE_SEARCH_TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(
            f"{module_name} does not expose the expected tool {attribute}"
        ) from exc


def _load_all_code_search_tools():
    """Return all LangChain tool objects defined under code-search-tools."""

    return [
        _load_code_search_tool(filename, module_name, attribute)
        for filename, module_name, attribute in _CODE_SEARCH_TOOL_SPECS
    ]


# TODO: Replace these placeholders with realistic backwards slicing prompts later.
SYSTEM_PROMPT = SystemMessage(content="TODO: describe the system-level instructions for this agent.")
INITIAL_PROMPT = HumanMessage(content="TODO: define the initial question or target for the agent.")


def test_backwards_slicing_agent_can_run():
    """The backwards-slicing agent should build with the code-search tools and run once."""

    sqlite_db_path = Path(__file__).resolve().parent / "test_backwards_slicing_with_llm_checkpoint"
    conn = sqlite3.connect(sqlite_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    result: dict | None = None
    try:
        tools = _load_all_code_search_tools()
        openrouter_settings = OpenRouterLLMSettings(model_name="openai/gpt-5.1-codex-mini")
        llm = build_openrouter_llm(openrouter_settings)

        agent = make_backwards_slicing_agent(
            llm=llm,
            checkpointer=checkpointer,
            tools=tools,
            middleware=[],
            system_prompt=SYSTEM_PROMPT,
        )

        config = {"thread_id": "1"}

        result = agent.invoke({"messages": [INITIAL_PROMPT]}, config=config)
    finally:
        conn.close()

    assert isinstance(result, dict)
    assert result.get("messages"), "Agent run should always return at least one message"
