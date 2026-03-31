import os
import sys
import time
import json
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from typing_extensions import TypedDict

import importlib.util

# Ensure we can import from workspace
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)

# Import prompts dynamically because of the hyphen in directory name
prompts_path = os.path.join(WORKSPACE_ROOT, "experiments", "feature-voting", "prompts.py")
prompts_spec = importlib.util.spec_from_file_location("prompts", prompts_path)
prompts = importlib.util.module_from_spec(prompts_spec)
sys.modules["prompts"] = prompts
if prompts_spec and prompts_spec.loader:
    prompts_spec.loader.exec_module(prompts)

CodeFeatureFlags = prompts.CodeFeatureFlags
DirectPromptGenerator = prompts.DirectPromptGenerator

class GraphState(TypedDict):
    program_name: str
    kernel_mangled_name: str
    kernel_demangled_name: str
    source_code_files: Dict[str, str]
    exe_args: str

    raw_response: Optional[Dict[str, Any]]
    prediction: Optional[Dict[str, Any]]
    predicted_has_branching: Optional[bool]
    predicted_has_data_dependent_branching: Optional[bool]
    predicted_has_flop_division: Optional[bool]
    predicted_has_preprocessor_defines: Optional[bool]
    predicted_has_common_float_subexpr: Optional[bool]
    predicted_has_special_math_functions: Optional[bool]
    predicted_calls_device_function: Optional[bool]
    predicted_has_rng_input_data: Optional[bool]
    predicted_reads_input_values_from_file: Optional[bool]
    predicted_has_hardcoded_gridsz: Optional[bool]
    predicted_has_hardcoded_blocksz: Optional[bool]
    query_time: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    llm_model_name: Optional[str]
    llm_provider: Optional[str]
    llm_response_id: Optional[str]
    llm_response_metadata: Optional[Dict[str, Any]]


@lru_cache(maxsize=1)
def _openrouter_model_pricing() -> Dict[str, Dict[str, Decimal]]:
    with urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pricing_by_model: Dict[str, Dict[str, Decimal]] = {}
    for model_data in payload.get("data", []):
        raw_pricing = model_data.get("pricing") or {}
        parsed_pricing: Dict[str, Decimal] = {}
        for key in ("prompt", "completion", "input_cache_read"):
            value = raw_pricing.get(key)
            if value is None:
                continue
            try:
                parsed_pricing[key] = Decimal(str(value))
            except (InvalidOperation, TypeError, ValueError):
                continue

        if not parsed_pricing:
            continue

        model_id = model_data.get("id")
        canonical_slug = model_data.get("canonical_slug")
        if model_id:
            pricing_by_model[model_id] = parsed_pricing
        if canonical_slug:
            pricing_by_model[canonical_slug] = parsed_pricing

    return pricing_by_model


def _calculate_cost_usd(response_metadata: Dict[str, Any], usage: Dict[str, Any]) -> Optional[float]:
    model_name = response_metadata.get("model_name") or response_metadata.get("model")
    if not model_name:
        return None

    try:
        pricing = _openrouter_model_pricing().get(model_name)
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None

    if not pricing:
        return None

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cache_read_tokens = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)

    prompt_price = pricing.get("prompt", Decimal("0"))
    completion_price = pricing.get("completion", Decimal("0"))
    cache_read_price = pricing.get("input_cache_read", prompt_price)

    cost = (
        Decimal(uncached_input_tokens) * prompt_price
        + Decimal(output_tokens) * completion_price
        + Decimal(cache_read_tokens) * cache_read_price
    )
    return float(cost)


def _try_validate_prediction(candidate: Any) -> Optional[CodeFeatureFlags]:
    if candidate is None:
        return None

    try:
        if isinstance(candidate, CodeFeatureFlags):
            return candidate
        if isinstance(candidate, str):
            return CodeFeatureFlags.model_validate_json(candidate)
        if isinstance(candidate, dict):
            return CodeFeatureFlags.model_validate(candidate)
    except Exception:
        return None

    return None


def _parse_prediction_from_raw_response(raw_response: Any) -> Optional[CodeFeatureFlags]:
    if raw_response is None:
        return None

    tool_calls = getattr(raw_response, "tool_calls", None) or []
    for tool_call in tool_calls:
        parsed = _try_validate_prediction(tool_call.get("args"))
        if parsed is not None:
            return parsed

    additional_kwargs = getattr(raw_response, "additional_kwargs", {}) or {}
    for tool_call in additional_kwargs.get("tool_calls", []) or []:
        function_payload = tool_call.get("function", {}) or {}
        parsed = _try_validate_prediction(function_payload.get("arguments"))
        if parsed is not None:
            return parsed

    content = getattr(raw_response, "content", None)
    if isinstance(content, str):
        return _try_validate_prediction(content)

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                parsed = _try_validate_prediction(item.get("text"))
                if parsed is not None:
                    return parsed

    return None


def _extract_prediction_from_response(response: Dict[str, Any]) -> CodeFeatureFlags:
    parsed = _try_validate_prediction(response.get("parsed"))
    if parsed is not None:
        return parsed

    raw_response = response.get("raw")
    parsed = _parse_prediction_from_raw_response(raw_response)
    if parsed is not None:
        return parsed

    parsing_error = response.get("parsing_error")
    if parsing_error is not None:
        raise ValueError(f"Structured output parsing failed: {parsing_error}") from parsing_error

    raise ValueError("Structured output did not contain a parsed response or recoverable tool-call arguments.")

def query_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    generator = DirectPromptGenerator(
        program_name=state["program_name"],
        kernel_mangled_name=state["kernel_mangled_name"],
        kernel_demangled_name=state["kernel_demangled_name"],
        source_code_files=state["source_code_files"],
        exe_args=state["exe_args"],
    )

    system_prompt = generator.generate_system_prompt()
    human_prompt = generator.generate_prompt()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    if config["configurable"].get("print_prompts", False):
        thread_id = config["configurable"].get("thread_id", "<unknown-thread>")
        print("\n" + "=" * 70)
        print(f" QUERY PROMPT [{thread_id}] ")
        print("=" * 70)
        print("--- System Prompt ---")
        print(system_prompt)
        print("\n--- Human Prompt ---")
        print(human_prompt)
        print("=" * 70 + "\n")
    
    llm = config["configurable"]["llm"]

    llm_with_structure_raw = llm.with_structured_output(
        CodeFeatureFlags,
        method="function_calling",
        include_raw=True,
    )

    start_time = time.time()
    response = llm_with_structure_raw.invoke(messages)
    end_time = time.time()

    parsed = _extract_prediction_from_response(response)
    
    return {
        "prediction": parsed.model_dump(),
        "raw_response": response["raw"].model_dump(),
        "query_time": end_time - start_time
    }

def validator_node(state: GraphState) -> Dict[str, Any]:
    raw = state.get("raw_response", {})
    usage = raw.get("usage_metadata", {})
    response_metadata = raw.get("response_metadata", {})
    prediction = state.get("prediction") or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    cost_usd = _calculate_cost_usd(response_metadata, usage)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "predicted_has_branching": prediction.get("has_branching"),
        "predicted_has_data_dependent_branching": prediction.get("has_data_dependent_branching"),
        "predicted_has_flop_division": prediction.get("has_flop_division"),
        "predicted_has_preprocessor_defines": prediction.get("has_preprocessor_defines"),
        "predicted_has_common_float_subexpr": prediction.get("has_common_float_subexpr"),
        "predicted_has_special_math_functions": prediction.get("has_special_math_functions"),
        "predicted_calls_device_function": prediction.get("calls_device_function"),
        "predicted_has_rng_input_data": prediction.get("has_rng_input_data"),
        "predicted_reads_input_values_from_file": prediction.get("reads_input_values_from_file"),
        "predicted_has_hardcoded_gridsz": prediction.get("has_hardcoded_gridsz"),
        "predicted_has_hardcoded_blocksz": prediction.get("has_hardcoded_blocksz"),
        "llm_model_name": response_metadata.get("model_name") or response_metadata.get("model"),
        "llm_provider": response_metadata.get("model_provider"),
        "llm_response_id": response_metadata.get("id") or raw.get("id"),
        "llm_response_metadata": response_metadata,
    }

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("query", query_node)
    builder.add_node("validator", validator_node)

    builder.add_edge(START, "query")
    builder.add_edge("query", "validator")
    builder.add_edge("validator", END)

    return builder

def compile_graph_with_postgres(db_uri: str):
    pool = ConnectionPool(conninfo=db_uri)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)
