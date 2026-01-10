from __future__ import annotations

import json
from textwrap import dedent

from deepagents import create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    wrap_tool_call,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import Field
from typing import Annotated, Any, Callable, List, TypedDict


class BackwardsSlicingState(TypedDict):
    target_kernel_name: Annotated[
        str,
        Field(
            ...,
            description=(
                "The CUDA kernel the agent should stop slicing at; includes the "
                "launch syntax (e.g., `kernel_name<<<...>>>`)."
            ),
        ),
    ]
    workspace_root: Annotated[
        str,
        Field(
            default="/codex/gpuFLOPBench/src",
            description="Absolute path to the benchmark root where source files live.",
        ),
    ]
    script_path: Annotated[
        str,
        Field(
            default="/tmp/slice.cpp",
            description="Location of the output C++ backwards slice file the agent must write/execute.",
        ),
    ]
    main_file: Annotated[
        str,
        Field(
            description="Relative path (from workspace root) of the `main()` entry point we are analyzing.",
        ),
    ]
    kernel_def_file: Annotated[
        str,
        Field(
            default="/tmp/extracted_kernel.cu",
            description="Relative path to the file containing the definition of `target_kernel_name`.",
        ),
    ]
    filetree: Annotated[
        str,
        Field(
            default="",
            description="Indented tree string summarizing the files under the workspace for context.",
        ),
    ]



def default_backwards_slicing_system_prompt() -> SystemMessage:
    """Construct the base system prompt that describes the script-based workflow."""
    template = dedent(
        """\
        You are the backwards slicing agent controller. 
        Your utlimate task is to produce a `/tmp/slice.cpp` and `/tmp/slice.hpp`
        file of the requested source code.

        You are NOT allowed to modify any of the source code we examine.
        You can only write/edit files in the `/tmp/` directory.

        Delegate any filesystem exploration or large file parsing to sub-agents
        so as not to clog up your input context.
        This task is free of human feedback, so any questions you have will need 
        to be answered by you or a subagent of your choice.

        Your goal is to examine the requested source files and extract
        all the code leading up to the FIRST invocation of the target CUDA
        kernel that the user requests.
        Drop any of the code after the first kernel invocation, 
        but keep all of the code that influences the inputs and 
        execution to the target CUDA kernel.
        Keep the kernel definition, along with any other `__device__`
        or `__global__` kernels that it calls.
        You're write all the code out to one file: `/tmp/slice.cpp`, and
        any relevant header code to `/tmp/slice.h`.
        The final slice.cpp should be compilable, but you're not allowed
        to use the compiler to check that.
        The resulting execution of the target CUDA kernel should result in
        the same execution as the original unmodified code.
        A user will check your result later, so do not run or compile the 
        code you write.

        Requirements:
        1) You are NOT allowed to call the compiler, clang, LLVM, or llmv tools. 
        2) For system shell execution, at most, you can write Python scripts 
        using the treesitter-cuda library to help you parse the source code.
        3) Keep original code comments when slicing code
        4) If the target function is called in a loop, only take the first loop
        iteration that makes the first kernel invocation. 
        5) Your output should be a `/tmp/slice.cpp` file which contains all the 
        relevant code up to the first kernel's invocation (including the invocation
        and kernel source definition). The `/tmp/slice.h` should contain a sliced
        version of any of the headers in the project source, leaving out any unused
        code.
        6) Cleanup code to free memory should be kept so the code correctly executes.
        7) The sliced code should include the `main()` function and any code in between
        the `main` and the target CUDA kernel that influences it's inputs.
        8) If you're unsure whether to drop some code, it's better to keep it so that
        we get a program that compiles and runs later.
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
        if not state["messages"]:
            print("Model returned: <no messages>")
            return None

        last_message = state["messages"][-1]
        print(f"Model returned: {last_message.content}")
        if isinstance(last_message, ToolMessage):
            tool_call = _find_tool_call_by_id(state["messages"], last_message.tool_call_id)
            if tool_call:
                print(
                    "Tool call executed:",
                    json.dumps(tool_call, indent=2, default=str),
                )
            else:
                print(f"Tool call executed: <could not locate metadata for {last_message.tool_call_id}>")

            print(f"Tool return status: {last_message.status}")
            print(f"Tool return value: {last_message.content}")
            if last_message.artifact is not None:
                print("Tool artifact:", json.dumps(last_message.artifact, default=str))
        return None


def _find_tool_call_by_id(messages: list[Any], tool_call_id: str | None) -> dict[str, Any] | None:
    if not tool_call_id:
        return None
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call.get("id") == tool_call_id:
                    return tool_call
    return None


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        print('Tool calling error occured!', str(e))
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
    backend,
    tools: list,
    middleware: list,
    system_prompt: SystemMessage | str,
    max_model_calls_limit: int = 10,
    max_tool_calls_limit: int = 10,
):
    """Create an agent that performs backwards slicing via shell-launched Python scripts."""

    llm_call_limit_middleware = ModelCallLimitMiddleware(
        thread_limit=max_model_calls_limit,
        run_limit=max_model_calls_limit,
        exit_behavior="end"
        )

    tool_call_limit_middleware = ToolCallLimitMiddleware(
        thread_limit=max_tool_calls_limit, 
        run_limit=max_tool_calls_limit,
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
        backend=backend,
        middleware=middleware + extra_middlewares,
        checkpointer=checkpointer,
        system_prompt=prompt_content,
        context_schema=BackwardsSlicingState,
    )
    return agent
