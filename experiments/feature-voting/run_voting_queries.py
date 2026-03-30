import json
import math
import os
import sys
import argparse
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
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
graph_path = os.path.join(WORKSPACE_ROOT, "experiments", "feature-voting", "graph.py")
graph_spec = importlib.util.spec_from_file_location("graph", graph_path)
graph_mod = importlib.util.module_from_spec(graph_spec)
if graph_spec and graph_spec.loader:
    graph_spec.loader.exec_module(graph_mod)
build_graph = graph_mod.build_graph

# Import db manager for checking runs
db_manager_path = os.path.join(WORKSPACE_ROOT, "experiments", "feature-voting", "db_manager.py")
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
    if "checkpoint" in state:
        state = state.get("checkpoint", {}).get("channel_values", {})

    def format_feature_label(name: str) -> str:
        if name.startswith("predicted_"):
            name = name[len("predicted_"):]
        return name.replace("_", " ").strip().title()

    feature_fields = [
        ("predicted_has_branching", "has_branching"),
        ("predicted_has_data_dependent_branching", "has_data_dependent_branching"),
        ("predicted_has_flop_division", "has_flop_division"),
        ("predicted_has_preprocessor_defines", "has_preprocessor_defines"),
        ("predicted_has_common_float_subexpr", "has_common_float_subexpr"),
        ("predicted_has_special_math_functions", "has_special_math_functions"),
        ("predicted_calls_device_function", "calls_device_function"),
        ("predicted_has_rng_input_data", "has_rng_input_data"),
        ("predicted_reads_input_values_from_file", "reads_input_values_from_file"),
        ("predicted_has_hardcoded_gridsz", "has_hardcoded_gridsz"),
        ("predicted_has_hardcoded_blocksz", "has_hardcoded_blocksz"),
    ]

    print("\n" + "="*70)
    print(" RUN COMPLETE ")
    print("="*70)
    print(f"Program: {state.get('program_name')}")
    print(f"Kernel Mangled: {state.get('kernel_mangled_name')}")
    print(f"Kernel Demangled: {state.get('kernel_demangled_name')}")

    predicted = state.get("prediction", {})
    if not isinstance(predicted, dict):
        predicted = vars(predicted) if hasattr(predicted, '__dict__') else {}

    display_features = []
    for predicted_key, fallback_key in feature_fields:
        if predicted_key in state:
            display_features.append((format_feature_label(predicted_key), state.get(predicted_key)))
        elif fallback_key in predicted:
            display_features.append((format_feature_label(fallback_key), predicted.get(fallback_key)))

    print("\n--- Code Feature Flags ---")
    for label, value in display_features:
        print(f"{label}: {value}")

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

def load_dataset(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _normalize_exe_args(exe_args) -> str:
    if isinstance(exe_args, float) and math.isnan(exe_args):
        return "<no arguments>"
    return exe_args


def _sanitize_thread_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0.00%)"
    return f"{count}/{total} ({(count / total) * 100.0:.2f}%)"


def _parse_model_names(model_names_arg: str) -> list[str]:
    model_names = [name.strip() for name in model_names_arg.split(",") if name.strip()]
    if not model_names:
        raise ValueError("--modelNames must contain at least one model identifier")
    return model_names


def _format_query_calculation(query_counts_by_program: dict[str, int], model_count: int, trials: int) -> list[str]:
    if not query_counts_by_program:
        return ["No queries defined."]

    lines = []
    total_terms = []
    for program_name, base_query_count in query_counts_by_program.items():
        program_total = base_query_count * model_count * trials
        total_terms.append(str(program_total))
        lines.append(
            f"{program_name:<25} {base_query_count} kernel(s) x {model_count} model(s) x {trials} trial(s) = {program_total} samples"
        )

    total_expression = " + ".join(total_terms)
    lines.append(f"{'Total formula':<25} {total_expression} = {sum(query_counts_by_program.values()) * model_count * trials}")
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


def _iter_query_batches(queries: list[dict], batch_size: int):
    for start_index in range(0, len(queries), batch_size):
        yield queries[start_index:start_index + batch_size]


def _execute_query_worker(
    db_uri: str,
    query: dict,
    print_prompts: bool,
    max_timeout: int,
) -> dict:
    thread_id = query["thread_id"]
    model_name = query["model_name"]
    parser = CheckpointDBParser(db_uri)
    attempt_tracker = QueryAttemptTracker(db_uri)
    pool = None

    try:
        settings = OpenRouterLLMSettings(model_name=model_name)
        configurable_llm = build_openrouter_llm(settings)

        pool = ConnectionPool(conninfo=db_uri, kwargs={"autocommit": True})
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        graph_builder = build_graph()
        app = graph_builder.compile(checkpointer=checkpointer)

        attempt_tracker.mark_attempt_started(thread_id)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "llm": configurable_llm,
                "print_prompts": print_prompts,
            }
        }

        with _query_timeout(max_timeout):
            final_state = app.invoke(query["state"], config=config)

        attempt_tracker.mark_attempt_success(thread_id)

        query_cost_usd = _extract_cost_usd_from_state(final_state)
        if query_cost_usd is None:
            query_cost_usd = _extract_cost_usd_from_tail_checkpoint(parser, thread_id)

        return {
            "thread_id": thread_id,
            "status": "completed",
            "final_state": final_state,
            "cost_usd": query_cost_usd,
            "error": None,
        }
    except Exception as e:
        attempt_tracker.mark_attempt_failure(thread_id, str(e))

        query_cost_usd = _extract_cost_usd_from_tail_checkpoint(parser, thread_id)

        return {
            "thread_id": thread_id,
            "status": "failed",
            "final_state": None,
            "cost_usd": query_cost_usd,
            "error": str(e),
        }
    finally:
        if pool is not None:
            pool.close()
        attempt_tracker.close()
        parser.close()

def run_queries(db_uri: str, dataset_path: str, model_names: list[str], trials: int, single_dry_run: bool = False, verbose: bool = False, print_prompts: bool = False, max_timeout: int = 240, max_queries: int | None = None, cli_config: dict | None = None, max_failed_attempts: int = 3, skip_completed_check: bool = False, max_spend: float | None = None, query_batch_size: int = 1):
    print("Loading dataset...")
    data = load_dataset(dataset_path)

    queries = []
    query_counts_by_program = {}

    for program_name, prog_data in data.items():
        if single_dry_run and program_name != "adam-cuda":
            continue

        kernels = prog_data["kernels"]
        sources = prog_data["sources"]
        exe_args = _normalize_exe_args(prog_data["exeArgs"])

        for mangled_kernel, kernel_data in kernels.items():
            demangled_name = kernel_data["demangledName"]

            safe_prog = _sanitize_thread_part(program_name)
            safe_kernel = _sanitize_thread_part(mangled_kernel)
            query_counts_by_program[program_name] = query_counts_by_program.get(program_name, 0) + 1

            state_inputs = {
                "program_name": program_name,
                "kernel_mangled_name": mangled_kernel,
                "kernel_demangled_name": demangled_name,
                "source_code_files": sources,
                "exe_args": exe_args,
            }

            for model_name in model_names:
                safe_model = _sanitize_thread_part(model_name)
                base_thread_id = f"{safe_prog}_{safe_kernel}_{safe_model}"
                for trial in range(trials):
                    target_thread_id = f"{base_thread_id}_trial{trial}"
                    if single_dry_run:
                        target_thread_id += "_DRYRUN"

                    queries.append({
                        "thread_id": target_thread_id,
                        "model_name": model_name,
                        "state": state_inputs,
                    })

            if single_dry_run:
                break

        if single_dry_run:
            break

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
        print(f"Model Names:                {', '.join(model_names)}")
        print(f"Model Count:                {len(model_names)}")
        print(f"Trial Count:                {trials}")
        print(f"Query Batch Size:           {query_batch_size}")
        print(f"Max Timeout:                {max_timeout} seconds")
        print(f"Max Queries:                {max_queries if max_queries is not None else 'unlimited'}")
        print(f"Max Spend:                  ${max_spend:.8f}" if max_spend is not None else "Max Spend:                  unlimited")
        print(f"Max Failed Attempts:        {max_failed_attempts}")
        print(f"Single Dry Run Enabled:     {single_dry_run}")
        print(f"Verbose Output Enabled:     {verbose}")
        print(f"Print Prompts Enabled:      {print_prompts}")
        print(f"----------------------------------------------------------------")
        print(f"Database checkpoint entries: {db_stats['total_checkpoint_entries']}")
        print(f"Database completed threads:  {db_stats['completed_threads']}")
        print(f"DB runs with all trials done: {db_stats['runs_with_all_trials_completed']}")
        print(f"DB runs with failures:       {total_failed_runs_db}")
        print(f"----------------------------------------------------------------")
        print("Query count calculation:")
        for calculation_line in _format_query_calculation(query_counts_by_program, len(model_names), trials):
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

        session_spend_usd = 0.0

        failed_queries: list[tuple[str, str]] = []
        worker_print_prompts = print_prompts

        with tqdm(total=total_queries, initial=completed_count, desc="Running Queries") as progress_bar:
            for batch_number, query_batch in enumerate(_iter_query_batches(queries_to_run, query_batch_size), start=1):
                if max_spend is not None and session_spend_usd >= max_spend:
                    print(
                        f"Max spend limit reached or surpassed: spent ${session_spend_usd:.8f} "
                        f"with limit ${max_spend:.8f}. Stopping execution before batch {batch_number}."
                    )
                    break

                with ProcessPoolExecutor(
                    max_workers=query_batch_size,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    future_to_thread_id = {
                        executor.submit(
                            _execute_query_worker,
                            db_uri,
                            query,
                            worker_print_prompts,
                            max_timeout,
                        ): query["thread_id"]
                        for query in query_batch
                    }

                    for future in as_completed(future_to_thread_id):
                        thread_id = future_to_thread_id[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            attempt_tracker.mark_attempt_failure(thread_id, str(e))
                            failed_queries.append((thread_id, str(e)))
                            print(f"\nError running query {thread_id}: {e}")
                            progress_bar.update(1)
                            continue

                        progress_bar.update(1)

                        query_cost_usd = result.get("cost_usd")
                        if query_cost_usd is not None:
                            session_spend_usd += query_cost_usd

                        if result["status"] == "completed":
                            if single_dry_run or verbose:
                                print_run_result(result["final_state"])
                        else:
                            error_message = result.get("error") or "Unknown error"
                            failed_queries.append((thread_id, error_message))
                            print(f"\nError running query {thread_id}: {error_message}")

                if max_spend is not None and session_spend_usd >= max_spend:
                    print(
                        f"Max spend limit reached or surpassed: spent ${session_spend_usd:.8f} "
                        f"with limit ${max_spend:.8f}. Stopping execution."
                    )
                    break

        if failed_queries:
            failed_thread_ids = ", ".join(thread_id for thread_id, _ in failed_queries[:10])
            if len(failed_queries) > 10:
                failed_thread_ids += ", ..."
            raise RuntimeError(
                f"{len(failed_queries)} queries failed during execution. "
                f"Failed thread IDs: {failed_thread_ids}"
            )
    finally:
        attempt_tracker.close()
        parser.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run code-feature voting LLM queries")
    parser.add_argument("--modelNames", type=str, default="openai/gpt-5.1-codex-mini", help="Comma-separated OpenRouter model identifiers")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeat trials to run for each query")
    parser.add_argument("--maxQueries", type=int, default=None, help="Maximum number of graph queries to execute during this script run")
    parser.add_argument("--maxTimeout", type=int, default=240, help="Maximum time in seconds allowed for each query before it is interrupted")
    parser.add_argument("--maxSpend", type=float, default=None, help="Maximum USD spend allowed for queries executed during this script run; already-completed database runs do not count toward this limit")
    parser.add_argument("--queryBatchSize", type=int, default=1, help="Number of queries to execute in parallel per batch")
    parser.add_argument("--maxFailedAttempts", type=int, default=3, help="Maximum number of failed attempts allowed for a query before it is skipped")
    parser.add_argument("--importDBDumpFile", type=str, default=None, help="Restore the supplied PostgreSQL custom dump file into a freshly recreated code_features_db before execution begins")
    parser.add_argument("--deleteDBFreshStart", action="store_true", help="Drop code_features_db before execution and treat this run as a fresh start; if combined with --importDBDumpFile, the database is wiped first, then the dump is restored, and restored completed queries are still eligible for skipping")
    parser.add_argument("--dumpDBOnFinish", action="store_true", help="After a successful run, dump the final code_features_db contents to experiments/feature-voting/code_features_db.dump")
    parser.add_argument("--exportDBOnly", action="store_true", help="Skip query execution and only export the current code_features_db state to experiments/feature-voting/code_features_db.dump after any requested wipe/restore steps")
    parser.add_argument("--singleDryRun", action="store_true", help="Perform a single dry run query of only the first kernel to verify LLM API functionality")
    parser.add_argument("--verbose", action="store_true", help="Print the results of each query after it finishes")
    parser.add_argument("--printPrompts", action="store_true", help="Print the full system and human prompts for each query")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    model_names = _parse_model_names(args.modelNames)

    ensure_postgres_running()
    default_dump_file = os.path.join(os.path.dirname(__file__), "code_features_db.dump")

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

    if args.queryBatchSize < 1:
        raise ValueError("--queryBatchSize must be at least 1")

    skip_completed_check = args.deleteDBFreshStart and not args.importDBDumpFile

    run_succeeded = False
    try:
        run_queries(DB_URI, DATASET_PATH, model_names, args.trials, args.singleDryRun, args.verbose, args.printPrompts, args.maxTimeout, args.maxQueries, vars(args), args.maxFailedAttempts, skip_completed_check, args.maxSpend, args.queryBatchSize)
        run_succeeded = True
    finally:
        if args.dumpDBOnFinish and run_succeeded:
            dump_path = dump_database(default_dump_file)
            print(f"Database dump written to: {dump_path}")
