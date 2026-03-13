import json
import os
import sys
from tqdm import tqdm
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

import importlib.util

# Setup workspace roots and dynamic imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)

# Import graph
graph_path = os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "graph.py")
graph_spec = importlib.util.spec_from_file_location("graph", graph_path)
graph_mod = importlib.util.module_from_spec(graph_spec)
if graph_spec and graph_spec.loader:
    graph_spec.loader.exec_module(graph_mod)
build_graph = graph_mod.build_graph

# Import db parser for checking runs
db_parser_path = os.path.join(WORKSPACE_ROOT, "experiments", "direct-prompting", "db_parser.py")
db_parser_spec = importlib.util.spec_from_file_location("db_parser", db_parser_path)
db_parser_mod = importlib.util.module_from_spec(db_parser_spec)
if db_parser_spec and db_parser_spec.loader:
    db_parser_spec.loader.exec_module(db_parser_mod)
CheckpointDBParser = db_parser_mod.CheckpointDBParser

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

def run_queries(db_uri: str, dataset_path: str):
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    
    # We will generate a list of task queries
    queries = []
    
    for program_name, prog_data in data.items():
        kernels = prog_data.get("kernels", {})
        sources = prog_data.get("sources", {})
        compile_commands = prog_data.get("compile_commands", [])
        exe_args = prog_data.get("exeArgs", "")
        
        for mangled_kernel, kernel_data in kernels.items():
            demangled_name = kernel_data.get("demangledName", mangled_kernel)
            metrics = kernel_data.get("metrics", {})
            
            for gpu_name, gpu_metrics in metrics.items():
                arch = get_architecture(gpu_name)
                sass_data = kernel_data.get("sass_code", {}).get(arch)
                imix_data = kernel_data.get("imix", {}).get(arch)
                
                # Make sass/imix as dict with dummy key or just string parsing
                sass_dict = {mangled_kernel: sass_data} if sass_data else None
                imix_dict = imix_data if isinstance(imix_data, dict) else None
                
                # We identify a task uniquely by program + kernel + gpu
                # Ensure no invalid characters in thread_id
                safe_prog = program_name.replace("/", "_").replace("\\", "_")
                safe_kernel = mangled_kernel.replace("/", "_").replace("\\", "_")
                safe_gpu = gpu_name.replace("/", "_").replace("\\", "_")
                target_thread_id = f"{safe_prog}_{safe_kernel}_{safe_gpu}"
                
                state_inputs = {
                    "program_name": program_name,
                    "kernel_mangled_name": mangled_kernel,
                    "kernel_demangled_name": demangled_name,
                    "source_code_files": sources,
                    "gpu_roofline_specs": {"gpu_target": gpu_name, "arch": arch},
                    "compile_commands": compile_commands,
                    "exe_args": exe_args,
                    "sass_dict": sass_dict,
                    "imix_dict": imix_dict,
                    "expected_fp16": gpu_metrics.get("HP_FLOP", 0),
                    "expected_fp32": gpu_metrics.get("SP_FLOP", 0),
                    "expected_fp64": gpu_metrics.get("DP_FLOP", 0),
                    "expected_read_bytes": gpu_metrics.get("bytesRead", 0),
                    "expected_write_bytes": gpu_metrics.get("bytesWritten", 0),
                }
                
                queries.append({
                    "thread_id": target_thread_id,
                    "state": state_inputs
                })
                
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
    for cp in checkpoints:
        # If Validator node executed, we consider it completed
        state_data = cp.get("checkpoint", {})
        if "channel_values" in state_data:
            if "total_tokens" in state_data["channel_values"]:
                completed_threads.add(cp["thread_id"])
                
    queries_to_run = [q for q in queries if q["thread_id"] not in completed_threads]
    
    print(f"\n--- Query Execution Summary ---")
    print(f"Total defined queries:       {len(queries)}")
    print(f"Already completed queries:   {len(completed_threads)}")
    print(f"Queries remaining to execute:{len(queries_to_run)}")
    
    if len(queries_to_run) == 0:
        print("All queries have been successfully completed! Exiting.")
        return
        
    input("Press 'Enter' to continue and run the remaining queries...")
    
    # Configure and run LangGraph
    pool = ConnectionPool(conninfo=db_uri)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    
    graph_builder = build_graph()
    app = graph_builder.compile(checkpointer=checkpointer)
    
    # Run the remaining configurations
    for query in tqdm(queries_to_run, desc="Running Queries"):
        config = {"configurable": {"thread_id": query["thread_id"]}}
        try:
            # We execute the graph directly. State starts with user outputs.
            app.invoke(query["state"], config=config)
        except Exception as e:
            print(f"\nError running query {query['thread_id']}: {e}")
            raise  # Fail hard as requested so user can intervene

if __name__ == "__main__":
    DB_URI = "postgresql://user:password@localhost:5432/mydb"  # Modify appropriately
    DATASET_PATH = os.path.join(WORKSPACE_ROOT, "dataset-creation", "gpuFLOPBench.json")
    
    run_queries(DB_URI, DATASET_PATH)
