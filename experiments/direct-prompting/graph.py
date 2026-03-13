import os
import sys
import time
import json
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Dict, Any, Optional
from urllib.error import URLError
from urllib.request import urlopen

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

import importlib.util

# Ensure we can import from workspace
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)

# Import llm_models dynamically because of the hyphen in directory name
llm_models_path = os.path.join(WORKSPACE_ROOT, "gpuFLOPBench-agentic", "agents", "llm_models.py")
llm_models_spec = importlib.util.spec_from_file_location("llm_models", llm_models_path)
llm_models = importlib.util.module_from_spec(llm_models_spec)
sys.modules["llm_models"] = llm_models
if llm_models_spec and llm_models_spec.loader:
    llm_models_spec.loader.exec_module(llm_models)

build_openrouter_llm = llm_models.build_openrouter_llm
OpenRouterLLMSettings = llm_models.OpenRouterLLMSettings

# Import prompts dynamically because of the hyphen in directory name
prompts_path = os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "prompts.py")
prompts_spec = importlib.util.spec_from_file_location("prompts", prompts_path)
prompts = importlib.util.module_from_spec(prompts_spec)
sys.modules["prompts"] = prompts
if prompts_spec and prompts_spec.loader:
    prompts_spec.loader.exec_module(prompts)

DirectPromptGenerator = prompts.DirectPromptGenerator
SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
KernelMetricsPrediction = prompts.KernelMetricsPrediction

class GraphState(TypedDict):
    program_name: str
    kernel_mangled_name: str
    kernel_demangled_name: str
    source_code_files: Dict[str, str]
    gpu_roofline_specs: Dict[str, Any]
    compile_commands: list
    exe_args: str
    sass_dict: Optional[Dict[str, str]]
    imix_dict: Optional[Dict[str, str]]
    
    # Expected metrics
    expected_fp16: int
    expected_fp32: int
    expected_fp64: int
    expected_read_bytes: int
    expected_write_bytes: int
    expected_grid_size: str
    expected_block_size: str

    # Outputs
    raw_response: Optional[Dict[str, Any]]
    prediction: Optional[Dict[str, Any]]
    
    # Metadata
    query_time: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    llm_model_name: Optional[str]
    llm_provider: Optional[str]
    llm_response_id: Optional[str]
    llm_response_metadata: Optional[Dict[str, Any]]
    
    # Validation results
    metrics_diff: Optional[Dict[str, int]]
    metrics_pct_diff: Optional[Dict[str, float]]


from langchain_core.runnables import RunnableConfig


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

def query_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    
    # Initialize PromptGenerator
    generator = DirectPromptGenerator(
        program_name=state["program_name"],
        kernel_mangled_name=state["kernel_mangled_name"],
        kernel_demangled_name=state["kernel_demangled_name"],
        source_code_files=state["source_code_files"],
        gpu_roofline_specs=state["gpu_roofline_specs"],
        compile_commands=state["compile_commands"],
        exe_args=state["exe_args"],
        sass_dict=state.get("sass_dict"),
        imix_dict=state.get("imix_dict")
    )
    
    # Generate human prompt
    human_prompt = generator.generate_prompt()
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt)
    ]

    if config["configurable"].get("verbose", False):
        print("\n" + "=" * 70)
        print(" QUERY PROMPT ")
        print("=" * 70)
        print("--- System Prompt ---")
        print(SYSTEM_PROMPT)
        print("\n--- Human Prompt ---")
        print(human_prompt)
        print("=" * 70 + "\n")
    
    # Retrieve LLM from configurable context setup globally in compilation
    llm = config["configurable"]["llm"]
    
    llm_with_structure_raw = llm.with_structured_output(
        KernelMetricsPrediction,
        method="function_calling",
        include_raw=True,
    )

    start_time = time.time()
    response = llm_with_structure_raw.invoke(messages)
    end_time = time.time()

    parsing_error = response.get("parsing_error")
    if parsing_error is not None:
        raise parsing_error

    parsed = response.get("parsed")
    if parsed is None:
        raise ValueError("Structured output did not contain a parsed response.")
    
    return {
        "prediction": parsed.model_dump(),
        "raw_response": response["raw"].model_dump(),
        "query_time": end_time - start_time
    }

def validator_node(state: GraphState) -> Dict[str, Any]:
    raw = state.get("raw_response", {})
    usage = raw.get("usage_metadata", {})
    response_metadata = raw.get("response_metadata", {})

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    cost_usd = _calculate_cost_usd(response_metadata, usage)
    
    prediction = state.get("prediction")
    metrics_diff = {}
    metrics_pct_diff = {}
    
    if prediction:
        predicted = {
            "fp16": prediction.get("fp16_flop_count", 0),
            "fp32": prediction.get("fp32_flop_count", 0),
            "fp64": prediction.get("fp64_flop_count", 0),
            "read_bytes": prediction.get("dram_bytes_read_count", 0),
            "write_bytes": prediction.get("dram_bytes_written_count", 0),
        }
        
        expected = {
            "fp16": state.get("expected_fp16", 0),
            "fp32": state.get("expected_fp32", 0),
            "fp64": state.get("expected_fp64", 0),
            "read_bytes": state.get("expected_read_bytes", 0),
            "write_bytes": state.get("expected_write_bytes", 0)
        }
        
        for k in expected:
            diff = predicted[k] - expected[k]
            metrics_diff[k] = diff
            
            if expected[k] == 0:
                metrics_pct_diff[k] = 0.0 if predicted[k] == 0 else float('inf')
            else:
                metrics_pct_diff[k] = (abs(diff) / expected[k]) * 100.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "llm_model_name": response_metadata.get("model_name") or response_metadata.get("model"),
        "llm_provider": response_metadata.get("model_provider"),
        "llm_response_id": response_metadata.get("id") or raw.get("id"),
        "llm_response_metadata": response_metadata,
        "metrics_diff": metrics_diff,
        "metrics_pct_diff": metrics_pct_diff
    }

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("query", query_node)
    builder.add_node("validator", validator_node)
    
    builder.add_edge(START, "query")
    builder.add_edge("query", "validator")
    builder.add_edge("validator", END)
    
    return builder

from psycopg_pool import ConnectionPool

def compile_graph_with_postgres(db_uri: str):
    # E.g., db_uri = "postgresql://user:password@localhost:5432/mydb"
    pool = ConnectionPool(conninfo=db_uri)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    
    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)
