from __future__ import annotations

from textwrap import dedent

from deepagents import create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    wrap_tool_call,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import Field
from typing import Annotated, Any, Callable, List, TypedDict


class BackwardsSlicingState(TypedDict):
    target_file: Annotated[str, Field(description="Full repository path to the criterion file.")]
    target_line: Annotated[int, Field(description="One-based line number of the CUDA launch or OpenMP pragma.")]
    cuda_kernels: Annotated[
        List[str],
        Field(description="List of extracted `__global__` / `__device__` definitions, either text excerpts or symbol names."),
    ]
    openmp_regions: Annotated[
        List[str],
        Field(description="OpenMP region descriptions comprising directive, clauses, and enclosing statement snippet."),
    ]



def default_backwards_slicing_system_prompt() -> SystemMessage:
    """Construct the base system prompt that describes the script-based workflow."""
    template = dedent(
        """\
        You are the backwards slicing controller. All detailed analysis happens inside Python
        scripts that you create and execute via the sandboxed shell. Use the built-in deepagents
        filesystem tools (`read_file`, `ls`, `write_file`, `edit_file`, `glob`, `grep`, `execute`)
        for interacting with the repository; do not invent new LangChain tools for parsing.

        Every script you write should import `treesitter_tools.cst_utils` (path:
        langchain-tools/treesitter-tools/cst_utils.py) and may also reference the supporting
        helpers (`caches`, `traversal`, `openmp`, `callsite`, `includes`, `identifiers`,
        `serialization`). The key helpers you can rely on immediately are:
        `read_text`, `parse_file`, `find_cuda_launches_on_line`, `collect_cuda_launches_on_line`,
        `build_omp_region`, `collect_callsites_in_function`, `summarize_def_use`, and the various
        span helpers (`span_text`, `ref_to_span`, `span_from_bytes`).

        Never edit existing source files under `/codex` (including gpuFLOPBench, agents, or
        langchain-tools). Only create or modify files that you place under `/tmp` or other
        ephemeral directories that you create during the run. When you need to run your Python
        script, call `execute` with `python /tmp/your_script.py`.

        After your script gathers the CUDA kernel definitions and OpenMP regions, call the
        `BackwardsSlicingState` tool (the state schema tool wired into this agent) with the
        fields `target_file`, `target_line`, `cuda_kernels`, and `openmp_regions` so the
        structured result is recorded instead of writing custom output files.

        Model calls must stay within the configured limits and your actions should be deterministic:
        mention which files you read, which scripts you created, and what key results you produced.
        You will not receive any follow-up human feedback during this run, so only ask the
        agent to ask questions that it can resolve on its own without external clarification.

        In order to not clog up input context, sub-agents should be used for filesystem exploration.
        Once the desired file(s)/filepaths are found, the subagent should return the requested file(s)/filepath.
        """
    )
    return SystemMessage(content=template)


# We're going to use the langchain API for creating this agent
# They recently added an Agents API that makes it easy to create agents with tools
# we're also able to add middleware functions to adjust the agent state and monitor behavior

# the create_agent allows us to also specify a checkpointer so we can save the agent state between runs

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


# we're not going to use the Langchain MultiServerMCPClient
# because all it does is convert MCP server calls to Langchain tools
# it's easier for us to simply make/supply the custom tools directly.
def make_backwards_slicing_agent(
    llm,
    checkpointer,
    tools: list,
    middleware: list,
    system_prompt: SystemMessage | str,
    max_model_calls_limit: int = 10,
    max_tool_calls_limit: int = 10,
):
    """Create an agent that performs backwards slicing via shell-launched Python scripts."""

    llm_call_limit_middleware = ModelCallLimitMiddleware(
        thread_limit=max_model_calls_limit,
        run_limit=10,
        exit_behavior="end"
        )

    tool_call_limit_middleware = ToolCallLimitMiddleware(
        thread_limit=max_tool_calls_limit, 
        run_limit=10,
        exit_behavior="end"
        )

    extra_middlewares = [LoggingMiddleware(), 
                         handle_tool_errors, 
                         llm_call_limit_middleware,
                         tool_call_limit_middleware]

    prompt_content = system_prompt.content if isinstance(system_prompt, SystemMessage) else system_prompt
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        middleware=middleware + extra_middlewares,
        checkpointer=checkpointer,
        system_prompt=prompt_content,
        context_schema=BackwardsSlicingState,
    )
    return agent
