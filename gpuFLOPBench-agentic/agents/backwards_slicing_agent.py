from __future__ import annotations

import json
import traceback
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

from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command


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

        General Backslicing Steps:
        1) Read the compilation commands to see what files are used to build the project.
        2) Identify the `main()` function in the main source file.
        3) Identify where the target CUDA kernel is invoked from `main()` or other functions.
        4) Trace the code from `main()` to find where the target CUDA kernel is invoked.
        5) Extract all relevant code, including function definitions, data structures, 
           and variable declarations that influence the kernel invocation.
        6) Write the extracted code to `/tmp/slice.cpp` and `/tmp/slice.h`.

        There are various code search and analysis tools at your disposal to help you with this task.
        You should use these tools BEFORE going and exploring the files manually using `ls` `glob` or `read_file`.
        The available tools are:
        - cuda_compile_commands: Returns the compilation commands used to build the source, and returns the compiler, arguments, and output path for each source so callers verify which source files are used to build the final executable.
        - cuda_file_tree: Builds a sorted, indented tree for a given directory so callers can survey the filesystem layout before inspecting files.
        - cuda_global_functions: Scans CUDA/C++/header sources for __global__ kernel definitions and reports each kernel name plus the file/line coordinates that define it, helping downstream logic locate kernels quickly.
        - cuda_main_files: Searches through the benchmark tree for free-function main() definitions so integration tests know which files serve as entry points for each CUDA project.
        - extract_kernel_source_definition: Replays the full __global__ declaration/definition for a named kernel (complete with templates and qualifiers) by pointing at the owning directory or source file, allowing comparisons against canonical snapshots stored in unit-tests/extracted-kernel-solutions.
        - function_definition_lister: Uses Tree-sitter to enumerate every declaration or definition in a single CUDA/C++/header file, emitting lines that include template signatures, CUDA qualifiers (__global__, __device__, etc.), return types, and (decl)/(defnt) annotations so agents can see the precise function metadata.
        - include_tree_extractor: Builds the #include dependency tree for one translation unit, annotating missing headers with (DNE) and stopping recursion when an include path would loop back to an ancestor, which helps agents trace header relationships without re-entering already-visited nodes.
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
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
):
    """Handle tool execution errors with custom messages."""
    print(f"\tExecuting tool: {request.tool_call['name']}")
    print(f"\tArguments: {request.tool_call['args']}")
    try:
        return handler(request)
    except Exception as e:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_id = tool_call.get("id", "<unknown>")
        tool_name = tool_call.get("name", "<unknown>")
        tool_arguments = tool_call.get("arguments")
        print("Tool calling error occurred!", str(e))
        print("  Tool call info:")
        print(f"    id: {tool_id}")
        print(f"    name: {tool_name}")
        if tool_arguments is not None:
            try:
                print("    arguments:", json.dumps(tool_arguments, indent=2, default=str))
            except Exception:  # pragma: no cover - best-effort logging
                print("    arguments (repr):", repr(tool_arguments))
        else:
            print("    arguments: <none>")
        print("  Traceback:")
        traceback.print_exc()
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=tool_id,
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
