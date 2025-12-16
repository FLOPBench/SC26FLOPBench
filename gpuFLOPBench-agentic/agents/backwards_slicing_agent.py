from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    wrap_tool_call,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.messages import ToolMessage, SystemMessage
from langgraph.runtime import Runtime
from typing import Any, Callable


@dataclass
class BackwardsSlicingState(TypedDict):
    target_name: str
    kernel_name: str


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
def make_backwards_slicing_agent(llm, 
                              checkpointer, 
                              tools: list,
                              middleware: list, 
                              system_prompt: SystemMessage,
                              max_model_calls_limit: int = 10,
                              max_tool_calls_limit: int = 10):
    """Create an agent that can perform backwards slicing of the requested code."""

    llm_call_limit_middleware = ModelCallLimitMiddleware(
        thread_limit=max_model_calls_limit,
        run_limit=5,
        exit_behavior="end"
        )

    tool_call_limit_middleware = ToolCallLimitMiddleware(
        thread_limit=max_tool_calls_limit, 
        run_limit=5,
        exit_behavior="end"
        )

    extra_middlewares = [LoggingMiddleware(), 
                         handle_tool_errors, 
                         llm_call_limit_middleware,
                         tool_call_limit_middleware]

    agent = create_agent(
        llm=llm,
        tools=tools,
        middleware=middleware + extra_middlewares,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        context_schema=BackwardsSlicingState,
        state_schema=BackwardsSlicingState,
    )
    return agent




