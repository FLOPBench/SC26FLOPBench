import json
import os
import sys
import argparse
import signal
from contextlib import contextmanager
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
QueryAttemptTracker = db_manager_mod.QueryAttemptTracker
setup_default_database = db_manager_mod.setup_default_database
ensure_postgres_running = db_manager_mod.ensure_postgres_running
wipe_database = db_manager_mod.wipe_database
restore_database_from_dump = db_manager_mod.restore_database_from_dump
dump_database = db_manager_mod.dump_database


@contextmanager
def _query_timeout(timeout_seconds: int):
    if timeout_seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Query exceeded max timeout of {timeout_seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

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
    print(f"Model: {state.get('llm_model_name')}")
    print(f"Provider: {state.get('llm_provider')}")
    print(f"Response ID: {state.get('llm_response_id')}")
    print(f"Input Tokens: {state.get('input_tokens')}")
    print(f"Output Tokens: {state.get('output_tokens')}")
    print(f"Total Tokens: {state.get('total_tokens')}")
    print(f"Query Time (s): {state.get('query_time')}")
    cost = state.get("cost_usd")
    if cost is not None:
        print(f"Estimated Cost ($): {cost:.8f}")

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


def _sanitize_thread_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _sass_thread_part(use_sass: bool) -> str:
    return "withsass" if use_sass else "nosass"


def _format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0.00%)"
    return f"{count}/{total} ({(count / total) * 100.0:.2f}%)"


def _format_query_calculation(query_counts_by_gpu: dict[str, int], trials: int) -> list[str]:
    if not query_counts_by_gpu:
        return ["No queries defined."]

    lines = []
    total_terms = []
    for gpu_name, base_query_count in query_counts_by_gpu.items():
        gpu_total = base_query_count * trials
        total_terms.append(str(gpu_total))
        lines.append(
            f"{gpu_name:<25} {base_query_count} kernel/GPU pairs x {trials} trial(s) = {gpu_total} samples"
        )

    total_expression = " + ".join(total_terms)
    lines.append(f"{'Total formula':<25} {total_expression} = {sum(query_counts_by_gpu.values()) * trials}")
    return lines


def _extract_cost_usd_from_state(state: dict | None) -> float | None:
    if state is None:
        return None

    channel_values = state.get("checkpoint", {}).get("channel_values", state)
    cost_usd = channel_values.get("cost_usd")
    if cost_usd is None:
        return None

    return float(cost_usd)


def _extract_cost_usd_from_tail_checkpoint(parser: CheckpointDBParser, thread_id: str) -> float | None:
    checkpoint = parser.fetch_tail_checkpoint_for_thread(thread_id)
    if checkpoint is None:
        return None

    channel_values = checkpoint["checkpoint"]["channel_values"]
    cost_usd = channel_values.get("cost_usd")
    if cost_usd is None:
        return None

    return float(cost_usd)

def run_queries(db_uri: str, dataset_path: str, model_name: str, trials: int, single_dry_run: bool = False, verbose: bool = False, use_sass: bool = False, max_timeout: int = 240, max_queries: int | None = None, cli_config: dict | None = None, max_failed_attempts: int = 3, skip_completed_check: bool = False, max_spend: float | None = None):
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    
    # Configure LLM dynamically
    settings = OpenRouterLLMSettings(model_name=model_name)
    configurable_llm = build_openrouter_llm(settings)
    
    # We will generate a list of task queries
    queries = []
    query_counts_by_gpu = {}
    
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
                safe_prog = _sanitize_thread_part(program_name)
                safe_kernel = _sanitize_thread_part(mangled_kernel)
                safe_gpu = _sanitize_thread_part(gpu_name)
                safe_model = _sanitize_thread_part(model_name)
                safe_sass_config = _sass_thread_part(use_sass)
                base_thread_id = f"{safe_prog}_{safe_kernel}_{safe_gpu}_{safe_model}_{safe_sass_config}"
                query_counts_by_gpu[gpu_name] = query_counts_by_gpu.get(gpu_name, 0) + 1
                
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
                        target_thread_id += "_DRYRUN"
                        
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
    attempt_tracker = QueryAttemptTracker(db_uri)
    try:
        query_thread_ids = {query["thread_id"] for query in queries}
        if skip_completed_check:
            checkpoints = []
            db_stats = {
                "total_checkpoint_entries": 0,
                "completed_threads": 0,
                "runs_with_all_trials_completed": 0,
            }
        else:
            try:
                checkpoints = parser.fetch_all_checkpoints()
                db_stats = parser.calculate_database_run_statistics(trials, list(query_thread_ids))
            except Exception as e:
                print(f"Warning: could not fetch checkpoints. Proceeding as if starting fresh. Err: {e}")
                checkpoints = []
                db_stats = {
                    "total_checkpoint_entries": 0,
                    "completed_threads": 0,
                    "runs_with_all_trials_completed": 0,
                }
        completed_threads = set()
        checkpoint_threads = set()
        for cp in checkpoints:
            checkpoint_threads.add(cp["thread_id"])
            # If Validator node executed, we consider it completed
            state_data = cp.get("checkpoint", {})
            if "channel_values" in state_data:
                if "total_tokens" in state_data["channel_values"]:
                    completed_threads.add(cp["thread_id"])

        attempt_data = attempt_tracker.fetch_attempts(list(query_thread_ids))
        completed_for_run = completed_threads.intersection(query_thread_ids)
        failed_for_run = (checkpoint_threads.intersection(query_thread_ids) - completed_for_run)
        total_failed_runs_db = sum(
            1
            for info in attempt_data.values()
            if info.get("failed_attempts", 0) > 0 or info.get("last_status") == "failed"
        )

        skipped_thread_ids = {
            thread_id
            for thread_id, info in attempt_data.items()
            if thread_id not in completed_for_run and info.get("failed_attempts", 0) >= max_failed_attempts
        }

        # If in dry-run mode, we always run it, ignoring past completion
        if single_dry_run:
            remaining_queries = queries
        else:
            remaining_queries = [q for q in queries if q["thread_id"] not in completed_threads]

        remaining_queries = [q for q in remaining_queries if q["thread_id"] not in skipped_thread_ids]

        if max_queries is not None and max_queries >= 0:
            queries_to_run = remaining_queries[:max_queries]
        else:
            queries_to_run = remaining_queries

        total_queries = len(queries)
        completed_count = len(completed_for_run)
        failed_count = len(failed_for_run)
        skipped_count = len(skipped_thread_ids)
        remaining_count = total_queries - completed_count - skipped_count
        selected_count = len(queries_to_run)

        print(f"\n--- Query Execution Summary ---")
        print(f"Model Name:                 {model_name}")
        print(f"Trial Count:                {trials}")
        print(f"Max Timeout:                {max_timeout} seconds")
        print(f"Max Queries:                {max_queries if max_queries is not None else 'unlimited'}")
        print(f"Max Spend:                  ${max_spend:.8f}" if max_spend is not None else "Max Spend:                  unlimited")
        print(f"Max Failed Attempts:        {max_failed_attempts}")
        print(f"Single Dry Run Enabled:     {single_dry_run}")
        print(f"Verbose Output Enabled:     {verbose}")
        print(f"Use SASS Enabled:           {use_sass}")
        print(f"----------------------------------------------------------------")
        print(f"Database checkpoint entries: {db_stats['total_checkpoint_entries']}")
        print(f"Database completed threads:  {db_stats['completed_threads']}")
        print(f"DB runs with all trials done: {db_stats['runs_with_all_trials_completed']}")
        print(f"DB runs with failures:       {total_failed_runs_db}")
        print(f"----------------------------------------------------------------")
        print("Query count calculation:")
        for calculation_line in _format_query_calculation(query_counts_by_gpu, trials):
            print(f"  {calculation_line}")
        print(f"Total defined queries:       {total_queries}")
        print(f"Completed progress:          {_format_ratio(completed_count, total_queries)}")
        print(f"Remaining progress:          {_format_ratio(remaining_count, total_queries)}")
        print(f"Failed runs in database:     {failed_count}")
        print(f"Skipped after failures:      {skipped_count}")
        print(f"Queries selected this run:   {selected_count}")
        if max_queries is not None:
            print(f"Run cap (--maxQueries):      {max_queries}")
        print(f"Failure retry limit:         {max_failed_attempts}")
        print(f"----------------------------------------------------------------")

        if skipped_thread_ids:
            print("\n--- Skipped Queries ---")
            for thread_id in sorted(skipped_thread_ids):
                failed_attempts = attempt_data.get(thread_id, {}).get("failed_attempts", 0)
                print(f"Skipping {thread_id}: reached {failed_attempts} failed attempts.")
            print(f"----------------------------------------------------------------")

        if len(queries_to_run) == 0:
            if skipped_count > 0 and remaining_count == 0:
                print("No runnable queries remain. Some queries were skipped after repeated failures.")
            else:
                print("All queries have been successfully completed! Exiting.")
            return

        if not single_dry_run:
            input("Press 'Enter' to continue and run the remaining queries...")

        # Configure and run LangGraph
        pool = ConnectionPool(conninfo=db_uri, kwargs={"autocommit": True})
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        graph_builder = build_graph()
        app = graph_builder.compile(checkpointer=checkpointer)

        session_spend_usd = 0.0

        # Run the remaining configurations
        for query in tqdm(queries_to_run, desc="Running Queries"):
            attempt_tracker.mark_attempt_started(query["thread_id"])
            config = {
                "configurable": {
                    "thread_id": query["thread_id"],
                    "llm": configurable_llm,
                    "verbose": verbose,
                }
            }
            try:
                # We execute the graph directly. State starts with user outputs.
                with _query_timeout(max_timeout):
                    final_state = app.invoke(query["state"], config=config)

                attempt_tracker.mark_attempt_success(query["thread_id"])

                query_cost_usd = _extract_cost_usd_from_state(final_state)
                if query_cost_usd is None:
                    query_cost_usd = _extract_cost_usd_from_tail_checkpoint(parser, query["thread_id"])
                if query_cost_usd is not None:
                    session_spend_usd += query_cost_usd

                if single_dry_run or verbose:
                    print_run_result(final_state)

                if max_spend is not None and session_spend_usd >= max_spend:
                    print(
                        f"Max spend limit reached or surpassed: spent ${session_spend_usd:.8f} "
                        f"with limit ${max_spend:.8f}. Stopping execution."
                    )
                    return
            except Exception as e:
                attempt_tracker.mark_attempt_failure(query["thread_id"], str(e))

                query_cost_usd = _extract_cost_usd_from_tail_checkpoint(parser, query["thread_id"])
                if query_cost_usd is not None:
                    session_spend_usd += query_cost_usd

                print(f"\nError running query {query['thread_id']}: {e}")

                if max_spend is not None and session_spend_usd >= max_spend:
                    print(
                        f"Max spend limit reached or surpassed: spent ${session_spend_usd:.8f} "
                        f"with limit ${max_spend:.8f}. Stopping execution."
                    )
                    return

                raise  # Fail hard as requested so user can intervene
    finally:
        attempt_tracker.close()
        parser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLOP prediction LLM queries")
    parser.add_argument("--modelName", type=str, default="openai/gpt-5.1-codex-mini", help="OpenRouter model identifier")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeat trials to run for each query")
    parser.add_argument("--maxQueries", type=int, default=None, help="Maximum number of graph queries to execute during this script run")
    parser.add_argument("--maxTimeout", type=int, default=240, help="Maximum time in seconds allowed for each query before it is interrupted")
    parser.add_argument("--maxSpend", type=float, default=None, help="Maximum USD spend allowed for queries executed during this script run; already-completed database runs do not count toward this limit")
    parser.add_argument("--maxFailedAttempts", type=int, default=3, help="Maximum number of failed attempts allowed for a query before it is skipped")
    parser.add_argument("--importDBDumpFile", type=str, default=None, help="Restore the supplied PostgreSQL custom dump file into a freshly recreated gpuflops_db before execution begins")
    parser.add_argument("--deleteDBFreshStart", action="store_true", help="Drop gpuflops_db before execution and treat this run as a fresh start; if combined with --importDBDumpFile, the database is wiped first, then the dump is restored, and restored completed queries are still eligible for skipping")
    parser.add_argument("--dumpDBOnFinish", action="store_true", help="After a successful run, dump the final gpuflops_db contents to experiments/direct-prompting/gpuflops_db.dump")
    parser.add_argument("--exportDBOnly", action="store_true", help="Skip query execution and only export the current gpuflops_db state to experiments/direct-prompting/gpuflops_db.dump after any requested wipe/restore steps")
    parser.add_argument("--singleDryRun", action="store_true", help="Perform a single dry run query of only the first kernel to verify LLM API functionality")
    parser.add_argument("--verbose", action="store_true", help="Print the results of each query after it finishes")
    parser.add_argument("--useSASS", action="store_true", help="Include optional SASS and IMIX in the query input")
    args = parser.parse_args()

    ensure_postgres_running()
    default_dump_file = os.path.join(os.path.dirname(__file__), "gpuflops_db.dump")

    if args.deleteDBFreshStart:
        print("Deleting PostgreSQL database for a fresh start...")
        wipe_database()

    if args.importDBDumpFile:
        print(f"Restoring PostgreSQL database from dump: {args.importDBDumpFile}")
        DB_URI = restore_database_from_dump(args.importDBDumpFile)
    else:
        DB_URI = setup_default_database()

    if args.exportDBOnly:
        dump_path = dump_database(default_dump_file)
        print(f"Database dump written to: {dump_path}")
        sys.exit(0)

    DATASET_PATH = os.path.join(WORKSPACE_ROOT, "dataset-creation", "gpuFLOPBench.json")

    skip_completed_check = args.deleteDBFreshStart and not args.importDBDumpFile

    run_succeeded = False
    try:
        run_queries(DB_URI, DATASET_PATH, args.modelName, args.trials, args.singleDryRun, args.verbose, args.useSASS, args.maxTimeout, args.maxQueries, vars(args), args.maxFailedAttempts, skip_completed_check, args.maxSpend)
        run_succeeded = True
    finally:
        if args.dumpDBOnFinish and run_succeeded:
            dump_path = dump_database(default_dump_file)
            print(f"Database dump written to: {dump_path}")
