import os
import sys
import time
from typing import Dict, Any, Optional

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

    # Outputs
    raw_response: Optional[Dict[str, Any]]
    prediction: Optional[KernelMetricsPrediction]
    
    # Metadata
    query_time: Optional[float]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    
    # Validation results
    metrics_diff: Optional[Dict[str, int]]
    metrics_pct_diff: Optional[Dict[str, float]]


def query_node(state: GraphState, config: Dict[str, Any]) -> Dict[str, Any]:
    
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
    
    # Retrieve LLM from configurable context setup globally in compilation
    llm = config["configurable"]["llm"]

    # Invoke LLM (we use the non-structured output as well to get metadata, or we can use structured and then access metadata)
    # Wait, with_structured_output usually strips metadata unless we use include_raw=True.
    
    llm_with_structure_raw = llm.with_structured_output(KernelMetricsPrediction, include_raw=True)

    start_time = time.time()
    response = llm_with_structure_raw.invoke(messages)
    end_time = time.time()
    
    return {
        "prediction": response["parsed"],
        "raw_response": response["raw"].dict(),
        "query_time": end_time - start_time
    }

def validator_node(state: GraphState) -> Dict[str, Any]:
    raw = state.get("raw_response", {})
    usage = raw.get("usage_metadata", {})
    
    total_tokens = usage.get("total_tokens", 0)
    cost_usd = getattr(raw, "cost_usd", None) or 0.0
    
    prediction = state.get("prediction")
    metrics_diff = {}
    metrics_pct_diff = {}
    
    if prediction:
        # Pydantic models might be dicts when re-loaded from state
        is_dict = isinstance(prediction, dict)
        
        predicted = {
            "fp16": prediction.get("fp16_flop_count", 0) if is_dict else getattr(prediction, "fp16_flop_count", 0),
            "fp32": prediction.get("fp32_flop_count", 0) if is_dict else getattr(prediction, "fp32_flop_count", 0),
            "fp64": prediction.get("fp64_flop_count", 0) if is_dict else getattr(prediction, "fp64_flop_count", 0),
            "read_bytes": prediction.get("dram_bytes_read_count", 0) if is_dict else getattr(prediction, "dram_bytes_read_count", 0),
            "write_bytes": prediction.get("dram_bytes_written_count", 0) if is_dict else getattr(prediction, "dram_bytes_written_count", 0),
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
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
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
