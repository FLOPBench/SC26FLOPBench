#!/usr/bin/env bash
#
# doRuns.sh — recreate the repeat-trials experiment.
#
# Runs the 3 paper models on the repeat-trials kernel panel (24 kernels x 4 GPUs x
# 2 evidence configs x 4 repeats), writing to the dedicated gpuflops_repeat_db.
# Provider split (for now):
#   - OpenRouter: gpt-oss-120B and Opus 4.6   (needs OPENROUTER_API_KEY)
#   - Azure:      GPT-5.4                       (needs AZURE_API_KEY; free, cost is estimated)
#
# Resume-safe: completed queries are skipped by thread_id, so you can re-run these
# commands to pick up failures (each retried with a fresh seed, up to --maxFailedAttempts).
#
# The commands below are meant to be copy-pasted one at a time. First export your keys:
#   export OPENROUTER_API_KEY=...
#   export AZURE_API_KEY=...
#
# WARNING: do NOT add any destructive flags (--deleteDBFreshStart / --importDBDumpFile).
# Those wipe gpuflops_repeat_db. See README.md.

# --- OpenRouter: gpt-oss-120B ---
python3 run_repeat_trials.py --models "openai/gpt-oss-120b" --trials 4 --queryBatchSize 8 --yes --maxFailedAttempts 10

# --- OpenRouter: Opus 4.6 ---
python3 run_repeat_trials.py --models "anthropic/claude-opus-4.6" --trials 4 --queryBatchSize 8 --yes --maxFailedAttempts 10

# --- Azure: GPT-5.4 (free; cost is estimated, so skip the OpenRouter cost backfill; dump the DB when done) ---
python3 run_repeat_trials.py --models "azure/gpt-5.4" --trials 4 --queryBatchSize 8 --yes --maxFailedAttempts 10 --noFetchCosts --dumpDBOnFinish
