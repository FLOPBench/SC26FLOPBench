import json
import os
import sys
import argparse
from tqdm import tqdm
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

import importlib.util

# Setup workspace roots and dynamic imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)

# Import llm_models dynamically because of the hyphen in directory name
llm_models_path = os.path.join(WORKSPACE_ROOT, "gpuFLOPBench-agentic", "agents", "llm_models.py")
llm_models_spec = importlib.util.spec_from_file_location("llm_models", llm_models_path)
llm_models = importlib.util.module_from_spec(llm_models_spec)
if llm_models_spec and llm_models_spec.loader:
    llm_models_spec.loader.exec_module(llm_models)

build_openrouter_llm = llm_models.build_openrouter_llm
OpenRouterLLMSettings = llm_models.OpenRouterLLMSettings

# Import graph
graph_path = os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "graph.py")
graph_spec = importlib.util.spec_from_file_location("graph", graph_path)
graph_mod = importlib.util.module_from_spec(graph_spec)
if graph_spec and graph_spec.loader:
    graph_spec.loader.exec_module(graph_mod)
build_graph = graph_mod.build_graph

# Import db manager for checking runs
db_manager_path = os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "db_manager.py")
db_manager_spec = importlib.util.spec_from_file_location("db_manager", db_manager_path)
db_manager_mod = importlib.util.module_from_spec(db_manager_spec)
if db_manager_spec and db_manager_spec.loader:
    db_manager_spec.loader.exec_module(db_manager_mod)
CheckpointDBParser = db_manager_mod.CheckpointDBParser
setup_default_database = db_manager_mod.setup_default_database

def print_run_result(state: dict):
    # Support both raw dictionary states (from app.invoke) and DB checkpoint parsings
    if "checkpoint" in state:
        state = state.get("checkpoint", {}).get("channel_values", {})
    
    print("\n" + "="*70)
    print(" RUN COMPLETE ")
    print("="*70)
    print(f"Program: {state.get('program_name')}")
    print(f"Kernel Mangled: {state.get('kernel_mangled_name')}")
    print(f"Kernel Demangled: {state.get('kernel_demangled_name')}")
    print(f"GPU Target: {state.get('gpu_roofline_specs', {}).get('gpu_target')}")
    print(f"Architecture: {state.get('gpu_roofline_specs', {}).get('arch')}")
    
    predicted = state.get("prediction", {})
    if not isinstance(predicted, dict):
        predicted = vars(predicted) if hasattr(predicted, '__dict__') else {}
    diff = state.get("metrics_pct_diff", {})
    
    def get_diff(name_hints):
        for k, v in diff.items():
            if any(hint.lower() in k.lower() for hint in name_hints):
                return f"{v:.2f}%" if v is not None else "N/A"
        return "N/A"
        
    metrics_data = [
        ("FP64", state.get('expected_fp64'), predicted.get('fp64_flop_count'), get_diff(["fp64", "dp"])),
        ("FP32", state.get('expected_fp32'), predicted.get('fp32_flop_count'), get_diff(["fp32", "sp"])),
        ("FP16", state.get('expected_fp16'), predicted.get('fp16_flop_count'), get_diff(["fp16", "hp"])),
        ("Bytes Read", state.get('expected_read_bytes'), predicted.get('dram_bytes_read_count'), get_diff(["read"])),
        ("Bytes Written", state.get('expected_write_bytes'), predicted.get('dram_bytes_written_count'), get_diff(["write", "written"]))
    ]
    
    print("\n--- Metric Results ---")
    print(f"{'Metric':<15} | {'Expected':<15} | {'Predicted':<15} | {'% Diff':>10}")
    print("-" * 65)
    for name, exp, pred, pct in metrics_data:
        exp_str = str(exp) if exp is not None else "N/A"
        pred_str = str(pred) if pred is not None else "N/A"
        print(f"{name:<15} | {exp_str:<15} | {pred_str:<15} | {pct:>10}")
        
    print(f"\n--- Dimension Results ---")
    print(f"Block Size      | Expected: {str(state.get('expected_block_size')):<20} | Predicted: {str(predicted.get('blockSz'))}")
    print(f"Grid Size       | Expected: {str(state.get('expected_grid_size')):<20} | Predicted: {str(predicted.get('gridSz'))}")

    print("\n--- LLM Query Performance ---")
    print(f"Total Tokens: {state.get('total_tokens')}")
    print(f"Query Time (s): {state.get('query_time')}")
    cost = state.get("query_cost")
    if cost is not None:
        print(f"Estimated Cost ($): {cost}")
        
    print("="*70 + "\n")

def get_architecture(gpu: str) -> str:
    mapping = {
        "A100": "sm_80",
        "3080": "sm_86",
        "H100": "sm_90",
        "A10": "sm_86"
    }
    return mapping.get(gpu, "sm_80")

def load_dataset(path: str):
    with open(path, "r") as f:
        return json.load(f)

def run_queries(db_uri: str, dataset_path: str, model_name: str, trials: int, single_dry_run: bool = False, verbose: bool = False, use_sass: bool = False):
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    
    # Configure LLM dynamically
    settings = OpenRouterLLMSettings(model_name=model_name)
    configurable_llm = build_openrouter_llm(settings)
    
    # We will generate a list of task queries
    queries = []
    
    for program_name, prog_data in data.items():
        if single_dry_run and program_name != "adam-cuda":
            continue
            
        kernels = prog_data["kernels"]
        sources = prog_data["sources"]
        compile_commands = prog_data["compile_commands"]
        exe_args = prog_data["exeArgs"]
        
        for mangled_kernel, kernel_data in kernels.items():
            demangled_name = kernel_data["demangledName"]
            metrics = kernel_data["metrics"]
            
            for gpu_name, gpu_metrics in metrics.items():
                if single_dry_run and gpu_name != "H100":
                    continue
                    
                arch = get_architecture(gpu_name)
                sass_data = kernel_data["sass_code"][arch]
                imix_data = kernel_data["imix"][arch]
                gpu_compile_commands = compile_commands[gpu_name] if isinstance(compile_commands, dict) else compile_commands
                
                # Make sass/imix as dict with dummy key or just string parsing
                if use_sass:
                    sass_dict = {mangled_kernel: sass_data}
                    imix_dict = imix_data
                else:
                    sass_dict = None
                    imix_dict = None
                
                # We identify a task uniquely by program + kernel + gpu
                # Ensure no invalid characters in thread_id
                safe_prog = program_name.replace("/", "_").replace("\\", "_")
                safe_kernel = mangled_kernel.replace("/", "_").replace("\\", "_")
                safe_gpu = gpu_name.replace("/", "_").replace("\\", "_")
                base_thread_id = f"{safe_prog}_{safe_kernel}_{safe_gpu}"
                
                state_inputs = {
                    "program_name": program_name,
                    "kernel_mangled_name": mangled_kernel,
                    "kernel_demangled_name": demangled_name,
                    "source_code_files": sources,
                    "gpu_roofline_specs": {"gpu_target": gpu_name, "arch": arch},
                    "compile_commands": gpu_compile_commands,
                    "exe_args": exe_args,
                    "sass_dict": sass_dict,
                    "imix_dict": imix_dict,
                    "expected_fp16": gpu_metrics["HP_FLOP"],
                    "expected_fp32": gpu_metrics["SP_FLOP"],
                    "expected_fp64": gpu_metrics["DP_FLOP"],
                    "expected_read_bytes": gpu_metrics["bytesRead"],
                    "expected_write_bytes": gpu_metrics["bytesWritten"],
                    "expected_grid_size": kernel_data["gridSz"],
                    "expected_block_size": kernel_data["blockSz"],
                }
                
                for trial in range(trials):
                    target_thread_id = f"{base_thread_id}_trial{trial}"
                    # Append dry run identifier to thread_id so it doesn't pollute real runs
                    if single_dry_run:
                        target_thread_id += "_DRYRUN_V2"
                        
                    queries.append({
                        "thread_id": target_thread_id,
                        "state": state_inputs
                    })
                
                if single_dry_run:
                    break
            
            if single_dry_run:
                break
                
        if single_dry_run:
            break
                
    # Determine already completed queries by checking the DB
    print("Checking database for completed queries...")
    parser = CheckpointDBParser(db_uri)
    try:
        checkpoints = parser.fetch_all_checkpoints()
    except Exception as e:
        print(f"Warning: could not fetch checkpoints. Proceeding as if starting fresh. Err: {e}")
        checkpoints = []
    parser.close()
    
    completed_threads = set()
    completed_checkpoints = []
    for cp in checkpoints:
        # If Validator node executed, we consider it completed
        state_data = cp.get("checkpoint", {})
        if "channel_values" in state_data:
            if "total_tokens" in state_data["channel_values"]:
                completed_threads.add(cp["thread_id"])
                completed_checkpoints.append(cp)
                
    # If in dry-run mode, we always run it, ignoring past completion
    if single_dry_run:
        queries_to_run = queries
    else:
        queries_to_run = [q for q in queries if q["thread_id"] not in completed_threads]
    
    print(f"\n--- Query Execution Summary ---")
    print(f"Total defined queries:       {len(queries)}")
    print(f"Already completed queries:   {len(completed_threads)}")
    print(f"Queries remaining to execute:{len(queries_to_run)}")
    
    if len(queries_to_run) == 0:
        print("All queries have been successfully completed! Exiting.")
        if verbose:
            print("\nPrinting previously completed queries:")
            for cp in completed_checkpoints:
                if single_dry_run and "_DRYRUN" not in cp["thread_id"]:
                    continue
                print_run_result(cp)
        return
        
    if not single_dry_run:
        input("Press 'Enter' to continue and run the remaining queries...")
    
    # Configure and run LangGraph
    pool = ConnectionPool(conninfo=db_uri, kwargs={"autocommit": True})
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    
    graph_builder = build_graph()
    app = graph_builder.compile(checkpointer=checkpointer)
    
    # Run the remaining configurations
    for query in tqdm(queries_to_run, desc="Running Queries"):
        config = {
            "configurable": {
                "thread_id": query["thread_id"],
                "llm": configurable_llm,
                "verbose": verbose,
            }
        }
        try:
            # We execute the graph directly. State starts with user outputs.
            final_state = app.invoke(query["state"], config=config)
            
            if single_dry_run or verbose:
                print_run_result(final_state)
        except Exception as e:
            print(f"\nError running query {query['thread_id']}: {e}")
            raise  # Fail hard as requested so user can intervene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLOP prediction LLM queries")
    parser.add_argument("--model-name", type=str, default="openai/gpt-5.1-codex-mini", help="OpenRouter model identifier")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeat trials to run for each query")
    parser.add_argument("--singleDryRun", action="store_true", help="Perform a single dry run query of only the first kernel to verify LLM API functionality")
    parser.add_argument("--verbose", action="store_true", help="Print the results of each query after it finishes")
    parser.add_argument("--useSASS", action="store_true", help="Include optional SASS and IMIX in the query input")
    args = parser.parse_args()

    DB_URI = setup_default_database()
    DATASET_PATH = os.path.join(WORKSPACE_ROOT, "dataset-creation", "gpuFLOPBench.json")
    
    run_queries(DB_URI, DATASET_PATH, args.model_name, args.trials, args.singleDryRun, args.verbose, args.useSASS)
