#!/usr/bin/env python3
"""Dedicated runner for the repeat-trials experiment.

This resamples the LLMs on the repeat-trials kernel subset (no code variants) to characterize response
variability. It deliberately does NOT modify the original direct-prompting pipeline and writes to its OWN
PostgreSQL database (`gpuflops_repeat_db`) and dump file so it cannot corrupt the main `gpuflops_db` results.

It reuses `run_queries()` and the DB lifecycle helpers from
`experiments/direct-prompting/run_queries.py` by import (that module's CLI is __main__-guarded, so importing
is side-effect free beyond loading the graph/model helpers). Decode settings are inherited from the same
`build_openrouter_llm` path, so they match the main paper (temp 0.2, top-p 0.1).

For each (model, evidence) pair it runs `--trials` repeats against the subset. Thread IDs already encode
(program, kernel, GPU, model, evidence, trial), so resume/skip works within the dedicated DB.

Example (full run; 24-kernel panel x 4 trials x 3 models x 2 evidence ~= $153 total):
    export OPENROUTER_API_KEY=...
    python experiments/repeat-trials/run_repeat_trials.py --trials 4 --queryBatchSize 4 \
        --maxSpend 100 --dumpDBOnFinish --yes

(Use --yes for non-interactive runs to auto-confirm run_queries' "Press Enter" prompt; omit it in an
interactive terminal if you want to review the plan and confirm manually.)

Smoke test (no API key needed; just creates the dedicated DB and verifies wiring):
    python experiments/repeat-trials/run_repeat_trials.py --setupOnly
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_QUERIES_PATH = REPO_ROOT / "experiments" / "direct-prompting" / "run_queries.py"
HERE = Path(__file__).resolve().parent
DEFAULT_SUBSET = HERE / "repeat_trials_subset.json"
DEFAULT_DUMP = HERE / "gpuflops_repeat_db.dump"
DEFAULT_DB_NAME = "gpuflops_repeat_db"
DEFAULT_MODELS = ["anthropic/claude-opus-4.6", "openai/gpt-5.4", "openai/gpt-oss-120b"]
EVIDENCE_SETTINGS = {  # name -> (use_sass, use_imix); main-paper comparison uses no IMIX
    "source-only": (False, False),
    "source+sass": (True, False),
}
# graph.py reads this env var; when set, it records parse-failed generation ids (still billed) so we can
# backfill their true cost. Must match graph.REPEAT_TRIALS_FAILED_GEN_DB_ENV.
FAILED_GEN_DB_ENV = "REPEAT_TRIALS_FAILED_GEN_DB_URI"


def _model_display_from_thread(thread_id):
    t = thread_id or ""
    if "gpt-oss" in t:
        return "GPT OSS"
    if "gpt-5.4" in t:
        return "GPT 5.4"
    if "opus" in t:
        return "Opus 4.6"
    return "<unknown>"


def _ensure_failed_gen_table(db_uri):
    """Pre-create the failed-generation side table single-threaded so the spawn workers only INSERT."""
    import psycopg
    with psycopg.connect(db_uri, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS failed_generation_ids ("
            " thread_id TEXT, generation_id TEXT, model TEXT, provider TEXT, response_metadata JSONB,"
            " recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (thread_id, generation_id))"
        )


def backfill_true_costs(db_uri, api_key, requests_per_second, request_timeout, max_retries):
    """Fetch authoritative OpenRouter costs for every recoverable generation id (completed runs + parse-failed
    runs captured in failed_generation_ids) and write them into sample_true_costs, one row per attempted
    thread with a cost_updated flag. Reuses fetch_openrouter_request_metadata for the OpenRouter calls."""
    import psycopg
    import requests
    from tqdm import tqdm
    import fetch_openrouter_request_metadata as fm  # direct-prompting is already on sys.path
    from db_manager import CheckpointDBParser

    # 1. generation ids: completed runs (from checkpoints) + parse-failed runs (from our side table)
    thread_gen, thread_model = {}, {}
    parser = CheckpointDBParser(db_uri)
    try:
        for rec in fm.collect_openrouter_generation_records(parser, include_dry_run=False):
            thread_gen[rec.thread_id] = rec.generation_id
            thread_model[rec.thread_id] = rec.llm_model_name
    finally:
        parser.close()

    with psycopg.connect(db_uri, autocommit=True) as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT thread_id, generation_id, model FROM failed_generation_ids")
            for tid, gid, model in cur.fetchall():
                thread_gen.setdefault(tid, gid)
                thread_model.setdefault(tid, model)
        except Exception:
            pass
        cur.execute("SELECT thread_id FROM query_attempts")
        all_threads = [row[0] for row in cur.fetchall()]

    # 2. fetch true cost per unique generation id
    gens = sorted({g for g in thread_gen.values() if g})
    cost_by_gen = {}
    if gens:
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        rate_limiter = fm.RateLimiter(requests_per_second)
        for gen in tqdm(gens, desc="OpenRouter true cost", unit="gen"):
            result = fm.fetch_generation_metadata(
                session, gen, rate_limiter,
                timeout_seconds=request_timeout, max_retries=max_retries, retry_base_delay_seconds=1.0,
            )
            if result.fetch_status == "success":
                data = (result.response_json or {}).get("data", {}) or {}
                if data.get("total_cost") is not None:
                    cost_by_gen[gen] = float(data["total_cost"])

    # 3. write one sample_true_costs row per attempted thread, with the cost_updated flag
    with psycopg.connect(db_uri, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sample_true_costs ("
            " thread_id TEXT PRIMARY KEY, generation_id TEXT, model TEXT, true_cost_usd DOUBLE PRECISION,"
            " cost_updated BOOLEAN NOT NULL DEFAULT FALSE, fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
        )
        for tid in all_threads:
            gid = thread_gen.get(tid)
            cost = cost_by_gen.get(gid) if gid else None
            model = thread_model.get(tid) or _model_display_from_thread(tid)
            cur.execute(
                "INSERT INTO sample_true_costs (thread_id, generation_id, model, true_cost_usd, cost_updated, fetched_at)"
                " VALUES (%s, %s, %s, %s, %s, NOW())"
                " ON CONFLICT (thread_id) DO UPDATE SET generation_id = EXCLUDED.generation_id,"
                " model = EXCLUDED.model, true_cost_usd = EXCLUDED.true_cost_usd,"
                " cost_updated = EXCLUDED.cost_updated, fetched_at = NOW()",
                (tid, gid, model, cost, cost is not None),
            )
        cur.execute(
            "SELECT model, COUNT(*), SUM(CASE WHEN cost_updated THEN 1 ELSE 0 END), COALESCE(SUM(true_cost_usd),0)"
            " FROM sample_true_costs GROUP BY model ORDER BY 4 DESC"
        )
        rows = cur.fetchall()

    print("\nTRUE COST (OpenRouter request metadata) -> sample_true_costs (cost_updated flag per sample):")
    grand = 0.0
    for model, n, updated, total in rows:
        grand += float(total or 0)
        print(f"  {str(model):12} {int(updated)}/{int(n)} samples updated   true ${float(total or 0):.4f}")
    print(f"  {'TOTAL':12} true ${grand:.4f}")
    unrecoverable = sum(1 for tid in all_threads if not thread_gen.get(tid))
    if unrecoverable:
        print(f"  ({unrecoverable} attempted samples had no recoverable generation id -> cost_updated=False; "
              f"these are pre-capture parse failures whose ids were discarded.)")


def _load_run_queries():
    # run_queries() runs its workers in a SPAWN process pool, so each fresh child re-imports the worker
    # function by its module name. We therefore import run_queries as a REAL top-level module (by putting
    # experiments/direct-prompting on sys.path) rather than via a synthetic importlib name -- spawn
    # propagates sys.path to children, so they can re-import "run_queries" and unpickle the worker.
    dp_dir = str(RUN_QUERIES_PATH.parent)
    if dp_dir not in sys.path:
        sys.path.insert(0, dp_dir)
    import run_queries as mod
    return mod


def _ensure_checkpoint_schema(db_uri):
    """Create the LangGraph checkpoint tables once, single-threaded, mirroring run_queries' worker setup,
    so the spawn-pool workers don't race to initialize the schema on a fresh DB."""
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver
    pool = ConnectionPool(conninfo=db_uri, kwargs={"autocommit": True})
    try:
        PostgresSaver(pool).setup()
    finally:
        pool.close()


def main():
    ap = argparse.ArgumentParser(description="Run the repeat-trials resampling experiment (dedicated DB).")
    ap.add_argument("--subset", default=str(DEFAULT_SUBSET), help="Path to repeat_trials_subset.json")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated OpenRouter model identifiers")
    ap.add_argument("--evidence", default="both", choices=["both", "source-only", "source+sass"])
    ap.add_argument("--trials", type=int, default=4,
                    help="Repeat trials per (kernel,GPU,model,evidence). Default 4 => ~$153 for the 24-kernel "
                         "panel across all 3 models x 2 evidence (see the selector's COST ESTIMATE).")
    ap.add_argument("--dbName", default=DEFAULT_DB_NAME, help="Dedicated PostgreSQL database name")
    ap.add_argument("--dumpFile", default=str(DEFAULT_DUMP))
    ap.add_argument("--maxSpend", type=float, default=None, help="USD cap PER (model,evidence) invocation")
    ap.add_argument("--queryBatchSize", type=int, default=4)
    ap.add_argument("--maxTimeout", type=int, default=240)
    ap.add_argument("--maxQueries", type=int, default=None)
    ap.add_argument("--maxFailedAttempts", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--printPrompts", action="store_true")
    ap.add_argument("--deleteDBFreshStart", action="store_true",
                    help="Drop the dedicated repeat DB before running")
    ap.add_argument("--importDBDumpFile", default=None,
                    help="Restore a dump into the dedicated repeat DB before running")
    ap.add_argument("--dumpDBOnFinish", action="store_true",
                    help="Dump the dedicated repeat DB to --dumpFile after a successful run")
    ap.add_argument("--setupOnly", action="store_true",
                    help="Create/verify the dedicated DB and print the run plan, then exit (no API calls)")
    ap.add_argument("--yes", action="store_true",
                    help="auto-confirm run_queries' interactive 'Press Enter' prompt; required for "
                         "non-interactive runs (e.g. piped stdin or Claude Code's '!' prefix)")
    ap.add_argument("--noFetchCosts", action="store_true",
                    help="skip the post-sampling OpenRouter true-cost backfill into sample_true_costs")
    ap.add_argument("--requestsPerSecond", type=float, default=2.0, help="OpenRouter metadata fetch rate cap")
    ap.add_argument("--requestTimeout", type=float, default=30.0, help="per metadata request timeout (s)")
    ap.add_argument("--maxRetries", type=int, default=4, help="retries for retryable metadata fetch failures")
    args = ap.parse_args()

    if args.dbName == "gpuflops_db":
        ap.error("refusing to use the main 'gpuflops_db'; pick a dedicated name to avoid corrupting results")

    subset_path = Path(args.subset)
    if not subset_path.exists():
        ap.error(f"subset not found: {subset_path}. Run select_repeat_trial_kernels.py first.")

    rq = _load_run_queries()

    rq.ensure_postgres_running()
    if args.deleteDBFreshStart:
        print(f"Dropping dedicated DB '{args.dbName}' for a fresh start ...")
        rq.wipe_database(db_name=args.dbName)
    if args.importDBDumpFile:
        print(f"Restoring '{args.dbName}' from dump: {args.importDBDumpFile}")
        db_uri = rq.restore_database_from_dump(args.importDBDumpFile, db_name=args.dbName)
    else:
        db_uri = rq.setup_default_database(db_name=args.dbName)

    # Initialize the LangGraph checkpoint schema ONCE (single-threaded). run_queries' workers each call
    # PostgresSaver.setup(); on a fresh dedicated DB the spawn workers would otherwise race to create the
    # checkpoint_migrations table and fail with a duplicate-key error. (The main gpuflops_db never hit this
    # because its tables already existed from the restored dump.)
    _ensure_checkpoint_schema(db_uri)
    # Pre-create the failed-generation side table and point graph.py at this DB so parse-failed (still-billed)
    # generation ids get recorded for later true-cost backfill. Spawn workers inherit this env var.
    _ensure_failed_gen_table(db_uri)
    os.environ[FAILED_GEN_DB_ENV] = db_uri

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    evidence_names = list(EVIDENCE_SETTINGS) if args.evidence == "both" else [args.evidence]

    print("\nRepeat-trials run plan:")
    print(f"  dedicated DB : {args.dbName}  (separate from the main gpuflops_db)")
    print(f"  subset       : {subset_path}")
    print(f"  models       : {models}")
    print(f"  evidence     : {evidence_names}")
    print(f"  trials       : {args.trials}")
    print(f"  configs      : {len(models) * len(evidence_names)} (model x evidence)")

    if args.setupOnly:
        print("\n--setupOnly: dedicated DB is ready and wiring verified. No queries executed.")
        return

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("\nERROR: no API key in the environment (need OPENROUTER_API_KEY).\n"
              "  Env vars do NOT persist across separate shell invocations, so set it in the SAME command:\n"
              "    export OPENROUTER_API_KEY=sk-or-...\n"
              "    python experiments/repeat-trials/run_repeat_trials.py --models openai/gpt-oss-120b "
              "--trials 1 --maxQueries 4 --yes --verbose", file=sys.stderr)
        sys.exit(2)

    if args.yes:
        import builtins
        builtins.input = lambda *a, **k: ""  # auto-confirm run_queries' interactive 'Press Enter' prompt

    # Run each (model, evidence) config independently. run_queries() raises if ANY of its queries fail
    # (e.g. GPT-OSS structured-output flakes), so we catch per-config and continue -- failures are recorded
    # in the DB and retried on the next run (resume). One flaky query must not abort the whole batch.
    config_status = []
    try:
        for model in models:
            for ev in evidence_names:
                use_sass, use_imix = EVIDENCE_SETTINGS[ev]
                print(f"\n=== {model} | {ev} | trials={args.trials} ===")
                cli_config = {
                    "modelName": model, "trials": args.trials, "useSASS": use_sass,
                    "useIMIX": use_imix, "queryBatchSize": args.queryBatchSize,
                    "maxSpend": args.maxSpend, "experiment": "repeat-trials",
                }
                try:
                    rq.run_queries(
                        db_uri, str(subset_path), model, args.trials,
                        False,                # single_dry_run
                        args.verbose, args.printPrompts, use_sass, use_imix,
                        args.maxTimeout, args.maxQueries, cli_config,
                        args.maxFailedAttempts, False,  # skip_completed_check
                        args.maxSpend, args.queryBatchSize,
                    )
                    config_status.append((model, ev, "ok"))
                except Exception as exc:  # noqa: BLE001 -- keep batching across configs
                    print(f"  WARNING: {model} | {ev} finished with failures: {exc}")
                    config_status.append((model, ev, f"had failures: {exc}"))
    finally:
        if args.dumpDBOnFinish:
            dump_path = rq.dump_database(str(Path(args.dumpFile)), db_name=args.dbName)
            print(f"\nDedicated DB dump written to: {dump_path}")

    # Backfill authoritative OpenRouter costs (completed + captured parse-failed gen ids) into the repeat DB.
    if not args.noFetchCosts:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\nSkipping true-cost backfill: no API key in environment.", file=sys.stderr)
        else:
            try:
                backfill_true_costs(db_uri, api_key, args.requestsPerSecond, args.requestTimeout, args.maxRetries)
            except Exception as exc:  # noqa: BLE001 -- backfill is best-effort, must not fail the run
                print(f"\nWARNING: true-cost backfill failed: {exc}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("Per-config summary:")
    for model, ev, status in config_status:
        print(f"  {model} | {ev}: {status}")
    n_failed = sum(1 for *_, s in config_status if s != "ok")
    if n_failed:
        print(f"\n{n_failed}/{len(config_status)} configs had query failures. Re-run the same command to "
              f"retry them (completed queries are skipped; failures retry up to --maxFailedAttempts). "
              f"Persistent GPT-OSS structured-output failures are expected and become 'incompletions'.")


if __name__ == "__main__":
    main()
