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
        and kernel source definition). The `/tmp/slice.hpp` should contain a sliced
        version of any of the headers in the project source, leaving out any unused
        code.
        6) Cleanup code to free memory should be kept so the code correctly executes.
        7) The sliced code should include the `main()` function and any code in between
        the `main` and the target CUDA kernel that influences it's inputs.
        8) If you're unsure whether to drop some code, it's better to keep it so that
        we get a program that compiles and runs later.

        General Backslicing Steps:
        1) Call cuda_compile_commands(dir_path) for the benchmark directory to inspect the preprocessor arguments and sources that participate in the build.
        2) Use cuda_main_files(dir_path) to discover each free-function `main()` entry point and its offset so you know which translation unit to trace from.
        3) Run cuda_global_functions(dir_path) to list every __global__ kernel definition under that directory and identify the file/line where the requested kernel resides.
        4) Use the `write_todos` tool to plan out your slicing steps before executing them. 
        5) Trace the call path from `main()` toward the kernel by combining function_definition_lister(file_path, qualifiers_filter=['__global__','__device__'], defs_or_decls='defnt') results with include_tree_extractor(file_path) outputs, capturing the functions, includes, and qualifiers that influence the invocation.
        6) Invoke extract_kernel_source_definition(file_path, kernel_name) to pull the canonical kernel definition plus any dependent device functions before composing your slice.
        7) Write the extracted code to `/tmp/slice.cpp` and `/tmp/slice.hpp`.
        8) Perform sanity checks to ensure the slice contains the `main()` function, the target kernel is defined and invoked, and all relevant function definitions and declarations are present.

        There are various code search and analysis tools at your disposal to help you with this task.
        You should use these tools BEFORE going and exploring the files manually using `ls` `glob` or `read_file`.
        The available tools are:
        - cuda_compile_commands(dir_path): Returns the compile commands for the requested directory; each entry lists the compiler, preprocessor arguments, output, and raw command so the agent understands how that portion of the project is built.
        - cuda_file_tree(dir_path): Builds a sorted, indented tree for any directory argument, giving a quick overview of the layout before the agent dives into specific files.
        - cuda_global_functions(dir_path): Lists every __global__ CUDA kernel defined under the directory argument, including file, line, offset, and span, so the agent can quickly locate entry points.
        - cuda_main_files(dir_path): Reports the offset and length of every free-function main() found under the directory argument, ensuring the agent knows which translation units hold entry points.
        - extract_kernel_source_definition(file_path, kernel_name): Returns the complete declaration and definition (with templates/qualifiers) for the specified kernel name, searching either the provided file or all CUDA sources under the given path.
        - function_definition_lister(file_path, qualifiers_filter, template_only, defs_or_decls): Parses the single source file argument and lists declarations/definitions with structured metadata (name, qualifiers, templates, signature, offsets), allowing optional filtering by qualifiers, templated functions only, or defs vs. decls.
        - include_tree_extractor(file_path): Walks the #include graph starting from the provided source file, annotating missing targets or loops so the agent can trace header dependencies without repeatedly opening the same units.

        Instructions for `/tmp/slice.cpp` and `/tmp/slice.hpp`:
        - The `/tmp/slice.cpp` file should contain all the relevant code up to and including the first invocation of the target CUDA kernel, as well as the kernel's definition itself.
        - The `/tmp/slice.hpp` file should contain any necessary header code that supports the sliced `slice.cpp`, excluding any unused declarations or definitions.
        - Ensure that both files are syntactically correct and maintain the original code's comments and structure as much as possible.
        - Do not include any code beyond what is necessary for the execution leading up to and including the target kernel invocation
        - The slice.cpp SHOULD include the `main()` function and all relevant code paths leading to the kernel call.
        - The `main()` function should take in the same argv/argc parameters as the original code.
        - The resulting files should be ready for compilation and execution, although you are NOT allowed to compile or run them yourself.
        - Sanity Checks:
            1) Ensure that the `/tmp/slice.cpp` file contains a `main()` function.
            2) Verify that the target CUDA kernel is defined AND invoked in `/tmp/slice.cpp`.
            3) Use the function_definition_lister tool on `/tmp/slice.cpp` and `/tmp/slice.hpp` to confirm that all function definitions leading up to the kernel invocation are present and that ALL function declarations have a matching function definition.
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
        result = handler(request)
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
    else:
        print(f"\tTool {request.tool_call['name']} executed successfully.")
        return result


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
