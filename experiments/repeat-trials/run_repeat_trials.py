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
        --maxSpend 100 --dumpDBOnFinish

Smoke test (no API key needed; just creates the dedicated DB and verifies wiring):
    python experiments/repeat-trials/run_repeat_trials.py --setupOnly
"""

import argparse
import importlib.util
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


def _load_run_queries():
    spec = importlib.util.spec_from_file_location("dp_run_queries", str(RUN_QUERIES_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    run_succeeded = False
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
                rq.run_queries(
                    db_uri, str(subset_path), model, args.trials,
                    False,                # single_dry_run
                    args.verbose, args.printPrompts, use_sass, use_imix,
                    args.maxTimeout, args.maxQueries, cli_config,
                    args.maxFailedAttempts, False,  # skip_completed_check
                    args.maxSpend, args.queryBatchSize,
                )
        run_succeeded = True
    finally:
        if args.dumpDBOnFinish and run_succeeded:
            dump_path = rq.dump_database(str(Path(args.dumpFile)), db_name=args.dbName)
            print(f"\nDedicated DB dump written to: {dump_path}")


if __name__ == "__main__":
    main()
