import json
import math
import os
import sys
import argparse
import multiprocessing
import signal
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from tqdm import tqdm
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

import importlib.util

# Setup workspace roots and dynamic imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(WORKSPACE_ROOT)

# Import llm_models from experiments/
llm_models_path = os.path.join(WORKSPACE_ROOT, "experiments", "llm_models.py")
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
delete_thread_history = db_manager_mod.delete_thread_history


FEATURE_FIELDS = [
    ("predicted_has_branching", "has_branching"),
    ("predicted_has_data_dependent_branching", "has_data_dependent_branching"),
    ("predicted_has_flop_division", "has_flop_division"),
    ("predicted_uses_preprocessor_defines", "uses_preprocessor_defines"),
    ("predicted_has_common_float_subexpr", "has_common_float_subexpr"),
    ("predicted_has_loop_invariant_flops", "has_loop_invariant_flops"),
    ("predicted_has_special_math_functions", "has_special_math_functions"),
    ("predicted_calls_device_function", "calls_device_function"),
    ("predicted_has_rng_input_data", "has_rng_input_data"),
    ("predicted_reads_input_values_from_file", "reads_input_values_from_file"),
    ("predicted_has_constant_propagatable_gridsz", "has_constant_propagatable_gridsz"),
    ("predicted_has_constant_propagatable_blocksz", "has_constant_propagatable_blocksz"),
]


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
    state = _unwrap_channel_values(state)

    def format_feature_label(name: str) -> str:
        if name.startswith("predicted_"):
            name = name[len("predicted_"):]
        return name.replace("_", " ").strip().title()

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
    for predicted_key, fallback_key in FEATURE_FIELDS:
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


def _unwrap_channel_values(state: dict | None) -> dict:
    if state is None:
        return {}
    if "checkpoint" in state:
        return state.get("checkpoint", {}).get("channel_values", {})
    return state


def _format_feature_label(name: str) -> str:
    if name.startswith("predicted_"):
        name = name[len("predicted_"):]
    return name.replace("_", " ").strip().title()


def _extract_feature_vote(state: dict, predicted_key: str, fallback_key: str) -> bool | None:
    channel_values = _unwrap_channel_values(state)
    if predicted_key in channel_values:
        value = channel_values.get(predicted_key)
        return None if value is None else bool(value)

    prediction = channel_values.get("prediction", {})
    if not isinstance(prediction, dict):
        prediction = vars(prediction) if hasattr(prediction, "__dict__") else {}

    if fallback_key in prediction:
        value = prediction.get(fallback_key)
        return None if value is None else bool(value)

    return None


def _short_model_label(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _load_completed_states(parser: CheckpointDBParser, thread_ids: set[str]) -> dict[str, dict]:
    completed_states: dict[str, dict] = {}
    for thread_id in sorted(thread_ids):
        checkpoint = parser.fetch_tail_checkpoint_for_thread(thread_id)
        if checkpoint is None:
            continue

        channel_values = checkpoint.get("checkpoint", {}).get("channel_values", {})
        if "total_tokens" in channel_values:
            completed_states[thread_id] = channel_values

    return completed_states


def _print_kernel_vote_consensus(
    kernel_group: dict,
    completed_states: dict[str, dict],
    total_cost_usd: float | None = None,
) -> None:
    vote_queries = sorted(
        kernel_group["queries"],
        key=lambda query: (query["model_name"], query["trial_index"]),
    )

    vote_labels = [
        f"{_short_model_label(query['model_name'])} t{query['trial_index'] + 1}"
        for query in vote_queries
    ]

    headers = ["Feature", "Yes/Total", "% Yes", *vote_labels]
    rows: list[list[str]] = []

    for predicted_key, fallback_key in FEATURE_FIELDS:
        yes_votes = 0
        total_votes = 0
        vote_cells: list[str] = []

        for query in vote_queries:
            vote_value = _extract_feature_vote(completed_states.get(query["thread_id"], {}), predicted_key, fallback_key)
            if vote_value is True:
                yes_votes += 1
                total_votes += 1
                vote_cells.append("X")
            elif vote_value is False:
                total_votes += 1
                vote_cells.append("")
            else:
                vote_cells.append("?")

        percent_yes = (yes_votes / total_votes * 100.0) if total_votes else 0.0
        row = [
            _format_feature_label(fallback_key),
            f"{yes_votes}/{total_votes}",
            f"{percent_yes:.1f}%",
            *vote_cells,
        ]
        rows.append(row)

    column_widths = []
    for column_index, header in enumerate(headers):
        width = len(header)
        for row in rows:
            width = max(width, len(row[column_index]))
        column_widths.append(width)

    def _format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(column_widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in column_widths)

    print("\n" + "=" * 70)
    print(" KERNEL VOTING CONSENSUS ")
    print("=" * 70)
    print(f"Program: {kernel_group['program_name']}")
    print(f"Kernel Mangled: {kernel_group['kernel_mangled_name']}")
    print(f"Kernel Demangled: {kernel_group['kernel_demangled_name']}")
    print(_format_row(headers))
    print(separator)
    for row in rows:
        print(_format_row(row))
    if total_cost_usd is not None:
        print(f"Total Query Cost (USD): {total_cost_usd:.8f}")
    print("=" * 70 + "\n")


def _maybe_print_kernel_consensus(
    kernel_key: tuple[str, str],
    kernel_groups: dict[tuple[str, str], dict],
    completed_states: dict[str, dict],
    terminal_thread_ids: set[str],
    total_cost_usd: float | None = None,
) -> None:
    kernel_group = kernel_groups[kernel_key]
    if kernel_group["consensus_printed"]:
        return

    expected_thread_ids = kernel_group["expected_thread_ids"]
    if all(thread_id in terminal_thread_ids for thread_id in expected_thread_ids):
        _print_kernel_vote_consensus(kernel_group, completed_states, total_cost_usd=total_cost_usd)
        kernel_group["consensus_printed"] = True


def _ensure_checkpoint_schema(db_uri: str) -> None:
    pool = ConnectionPool(conninfo=db_uri, kwargs={"autocommit": True})
    try:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
    finally:
        pool.close()


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

    _ensure_checkpoint_schema(db_uri)

    queries = []
    query_counts_by_program = {}
    kernel_groups: dict[tuple[str, str], dict] = {}

    for program_name, prog_data in data.items():
        if single_dry_run and program_name != "adam-cuda":
            continue

        kernels = prog_data["kernels"]
        sources = prog_data["sources"]
        exe_args = _normalize_exe_args(prog_data["exeArgs"])

        for mangled_kernel, kernel_data in kernels.items():
            demangled_name = kernel_data["demangledName"]
            kernel_key = (program_name, mangled_kernel)

            safe_prog = _sanitize_thread_part(program_name)
            safe_kernel = _sanitize_thread_part(mangled_kernel)
            query_counts_by_program[program_name] = query_counts_by_program.get(program_name, 0) + 1

            kernel_groups[kernel_key] = {
                "program_name": program_name,
                "kernel_mangled_name": mangled_kernel,
                "kernel_demangled_name": demangled_name,
                "queries": [],
                "expected_thread_ids": set(),
                "consensus_printed": False,
            }

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

                    query_record = {
                        "thread_id": target_thread_id,
                        "model_name": model_name,
                        "trial_index": trial,
                        "kernel_key": kernel_key,
                        "state": state_inputs,
                    }
                    queries.append(query_record)
                    kernel_groups[kernel_key]["queries"].append(query_record)
                    kernel_groups[kernel_key]["expected_thread_ids"].add(target_thread_id)

            if single_dry_run:
                break

        if single_dry_run:
            break

    print("Checking database for completed queries...")
    parser = CheckpointDBParser(db_uri)
    attempt_tracker = QueryAttemptTracker(db_uri)
    try:
        query_thread_ids = {query["thread_id"] for query in queries}
        if single_dry_run:
            delete_thread_history(db_uri, sorted(query_thread_ids))

        ignore_db_state = single_dry_run or skip_completed_check
        if ignore_db_state:
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

        attempt_data = {} if single_dry_run else attempt_tracker.fetch_attempts(list(query_thread_ids))
        completed_for_run = completed_threads.intersection(query_thread_ids)
        failed_for_run = (checkpoint_threads.intersection(query_thread_ids) - completed_for_run)
        total_failed_runs_db = 0 if single_dry_run else sum(
            1
            for info in attempt_data.values()
            if info.get("failed_attempts", 0) > 0 or info.get("last_status") == "failed"
        )

        skipped_thread_ids = set() if single_dry_run else {
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

        active_kernel_keys = {query["kernel_key"] for query in queries_to_run}
        if single_dry_run:
            completed_states = {}
        else:
            preload_thread_ids = set()
            for kernel_key in active_kernel_keys:
                preload_thread_ids.update(kernel_groups[kernel_key]["expected_thread_ids"])
            completed_states = _load_completed_states(parser, completed_for_run.intersection(preload_thread_ids))
        terminal_thread_ids = set(completed_states)

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

        if single_dry_run:
            input("Press 'Enter' to continue and run the dry-run query...")
        else:
            input("Press 'Enter' to continue and run the remaining queries...")

        session_spend_usd = 0.0

        failed_queries: list[tuple[str, str]] = []
        worker_print_prompts = print_prompts
        next_query_index = 0
        future_to_query = {}
        spend_limit_reached = False

        with tqdm(total=total_queries, initial=completed_count, desc="Running Queries") as progress_bar:
            with ProcessPoolExecutor(
                max_workers=query_batch_size,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                def _fill_inflight_slots() -> None:
                    nonlocal next_query_index, spend_limit_reached

                    while next_query_index < len(queries_to_run) and len(future_to_query) < query_batch_size:
                        if max_spend is not None and session_spend_usd >= max_spend:
                            if not spend_limit_reached:
                                print(
                                    f"Max spend limit reached or surpassed: spent ${session_spend_usd:.8f} "
                                    f"with limit ${max_spend:.8f}. Stopping new submissions and waiting for in-flight queries to finish."
                                )
                                spend_limit_reached = True
                            return

                        query = queries_to_run[next_query_index]
                        future = executor.submit(
                            _execute_query_worker,
                            db_uri,
                            query,
                            worker_print_prompts,
                            max_timeout,
                        )
                        future_to_query[future] = query
                        next_query_index += 1

                _fill_inflight_slots()

                while future_to_query:
                    done_futures, _ = wait(tuple(future_to_query), return_when=FIRST_COMPLETED)

                    for future in done_futures:
                        query = future_to_query.pop(future)
                        thread_id = query["thread_id"]
                        try:
                            result = future.result()
                        except Exception as e:
                            attempt_tracker.mark_attempt_failure(thread_id, str(e))
                            failed_queries.append((thread_id, str(e)))
                            print(f"\nError running query {thread_id}: {e}")
                            progress_bar.update(1)
                            terminal_thread_ids.add(thread_id)
                            _maybe_print_kernel_consensus(
                                query["kernel_key"],
                                kernel_groups,
                                completed_states,
                                terminal_thread_ids,
                            )
                            _fill_inflight_slots()
                            continue

                        progress_bar.update(1)
                        terminal_thread_ids.add(thread_id)

                        query_cost_usd = result.get("cost_usd")
                        if query_cost_usd is not None:
                            session_spend_usd += query_cost_usd

                        if result["status"] == "completed":
                            completed_states[thread_id] = _unwrap_channel_values(result["final_state"])
                            if single_dry_run or verbose:
                                print_run_result(result["final_state"])
                            if not single_dry_run:
                                _maybe_print_kernel_consensus(
                                    query["kernel_key"],
                                    kernel_groups,
                                    completed_states,
                                    terminal_thread_ids,
                                )
                        else:
                            error_message = result.get("error") or "Unknown error"
                            failed_queries.append((thread_id, error_message))
                            print(f"\nError running query {thread_id}: {error_message}")
                            _maybe_print_kernel_consensus(
                                query["kernel_key"],
                                kernel_groups,
                                completed_states,
                                terminal_thread_ids,
                            )

                        _fill_inflight_slots()

        for kernel_key in sorted(active_kernel_keys):
            _maybe_print_kernel_consensus(
                kernel_key,
                kernel_groups,
                completed_states,
                terminal_thread_ids,
                total_cost_usd=session_spend_usd if single_dry_run else None,
            )

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
    parser.add_argument("--queryBatchSize", type=int, default=4, help="Number of queries to execute in parallel per batch")
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
