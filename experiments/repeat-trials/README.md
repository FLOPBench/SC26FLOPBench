# Repeat-Trials Experiment

Resamples the LLMs (no code variants) on a small, stratified, **auditable** panel of kernels to characterize
**response variability** across repeated queries. This directly answers the SC reviewers who asked for
repeated-query consistency evidence.

It is self-contained: it does **not** modify the original `direct-prompting` pipeline and writes to its **own**
PostgreSQL database (`gpuflops_repeat_db`) so it cannot corrupt the main `gpuflops_db` results.

## Files

- `select_repeat_trial_kernels.py` — selects the ~24-kernel panel and emits the subset + manifest + plot.
- `run_repeat_trials.py` — dedicated runner (own DB) that resamples the panel.
- `kernel_feature_flags.csv` — per-kernel code-feature flags used for the hard/easy split (generated; see Step 0).
- `query_costs.csv` — true per-`(kernel,GPU,model,evidence)` OpenRouter query costs (from `request_metadata`,
  not the unreliable `gpuflops_db` cost field), used for the grounded cost estimate (generated; see Step 0b).
- `repeat_trials_subset.json` — pruned `gpuFLOPBench.json` (selected kernels only); input to the runner.
- `repeat_trials_manifest.csv` — one row per selected `(kernel, GPU, precision)` documenting every pick.
- `repeat_trials_rooflines.png` — faceted (GPU × precision) roofline scatter for human double-checking.

## Design (why it is reviewer-proof)

- Panel ≈ 24 kernels selected from the paper's **evaluated set** (`static_prediction_rows.csv`), so the
  variability result augments the reported numbers. Ground-truth RAI / balance points come straight from that
  artifact (and match `make_plots_for_paper.py`'s `GPU_ROOFLINE_TABLE`).
- **Each GPU has 3 balance points** (one per precision); everything keys on `(GPU, precision)`.
- Stratified over `precision × class(BB/CB) × proximity(near/far)` with a **≥5%** distance floor for an
  unambiguous label, deliberately boosting the scarce **near band** (where classification can flip).
- ~50/50 **hard/easy** split (hard = division / special-math / common-subexpr / loop-invariant FLOPs).
- One `(kernel,GPU)` query covers all precisions; the runner queries all 4 GPUs, getting BB/CB coverage from
  the cross-GPU balance-point shift.

## Step 0 — (re)generate `kernel_feature_flags.csv` from `code_features_db`

Only needed if the CSV is missing. Requires the PostgreSQL dump
`experiments/feature-voting/code_features_db.dump`. In the project Docker image PostgreSQL `16/main` is
preconfigured (`postgres/postgres`); restore and export with:

```bash
# restore the feature votes DB (uses the project db_manager defaults)
python experiments/feature-voting/run_voting_queries.py --importDBDumpFile experiments/feature-voting/code_features_db.dump --exportDBOnly
# export the per-kernel aggregate via the error-analysis reader
python - <<'PY'
import importlib.util
s=importlib.util.spec_from_file_location("db_reader","experiments/error-analysis/db_reader.py")
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
uri="postgresql://postgres:postgres@localhost:5432/code_features_db"
kf=m.aggregate_feature_votes(m.load_code_feature_vote_dataframe(db_uri=uri))
cols=["program_name","kernel_mangled_name"]+list(m.FEATURE_FLAG_COLUMNS)
kf[[c for c in cols if c in kf.columns]].to_csv("experiments/repeat-trials/kernel_feature_flags.csv",index=False)
print("wrote kernel_feature_flags.csv", len(kf))
PY
```

## Step 0b — (re)generate `query_costs.csv` from `request_metadata`

Only needed if the CSV is missing. The authoritative per-query cost is in the `request_metadata` DB (the
`gpuflops_db` cost field is unreliable, notably for GPT-OSS). We take the **true cost** from
`request_metadata` and the **parsed fields** (program/kernel/GPU/model/evidence) from `gpuflops_db`, joined on
`thread_id`. Restore both DBs, then export:

```bash
python experiments/direct-prompting/fetch_openrouter_request_metadata.py --importAndExit  # restores request_metadata
python experiments/direct-prompting/run_queries.py --importAndExit                        # restores gpuflops_db
python - <<'PY'
import importlib.util, pandas as pd, psycopg
s=importlib.util.spec_from_file_location("db_reader","experiments/error-analysis/db_reader.py")
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
fields=m.load_gpuflops_samples_dataframe(db_uri="postgresql://postgres:postgres@localhost:5432/gpuflops_db")
fields=fields[fields.status=="completed"][["thread_id","program_name","kernel_mangled_name","gpu","model_name","use_sass","use_imix"]]
con=psycopg.connect("postgresql://postgres:postgres@localhost:5432/request_metadata")
cost=pd.read_sql("""SELECT s.thread_id, m.total_cost FROM openrouter_generation_sources s
                    JOIN openrouter_generation_metadata m ON m.generation_id=s.generation_id
                    WHERE m.fetch_status='success' AND m.total_cost IS NOT NULL""", con); con.close()
cost=cost.groupby("thread_id",as_index=False)["total_cost"].median()
df=fields.merge(cost,on="thread_id").query("use_imix==False")
key=["program_name","kernel_mangled_name","gpu","model_name","use_sass"]
(df.groupby(key,as_index=False)["total_cost"].median().rename(columns={"total_cost":"cost_usd"})
   .to_csv("experiments/repeat-trials/query_costs.csv",index=False))
print("wrote query_costs.csv")
PY
```

## Step 1 — select the panel

```bash
python experiments/repeat-trials/select_repeat_trial_kernels.py
# knobs: --numKernels 24  --nearTarget 3  --minDistPct 0.05  --nearMaxPct 0.50
# degraded test without the hard/easy split: --noFeatures
```
Inspect the console summary, `repeat_trials_manifest.csv`, and `repeat_trials_rooflines.png` before running.

## Step 2 — run the repeat trials (dedicated DB; needs an API key)

```bash
export OPENROUTER_API_KEY=...
python experiments/repeat-trials/run_repeat_trials.py --trials 4 --queryBatchSize 4 \
    --maxSpend 100 --dumpDBOnFinish
# smoke test only (no API calls): --setupOnly
```
Runs the 3 paper models × {source-only, source+SASS} × `--trials` repeats. Decode settings are inherited from
the main pipeline (temp 0.2, top-p 0.1). Results land in `gpuflops_repeat_db` (dumped to
`gpuflops_repeat_db.dump`).

**Chosen plan:** 24 kernels × 4 GPUs × 2 evidence × 3 models × **4 trials** = 2,304 LLM calls, ≈ **$153**
(grounded estimate; Opus ≈ $116, GPT-5.4 ≈ $35, GPT-OSS ≈ $1.50). `--maxSpend` is a per-(model,evidence)
cap; the largest single arm (Opus+SASS) is ≈ $78, so 100 leaves headroom. The selector prints the live COST
ESTIMATE so you can re-check before spending.

## Step 3 — analysis (to be implemented)

A separate analysis script will read `gpuflops_repeat_db` and report, per `(precision, evidence, model)`:
within-cell variability (CV of predicted FLOPs/bytes and log-RAI) and the BB/CB **classification-flip rate**
across repeats, crossed with hard/easy — emitting a boxplot + LaTeX table for the paper's
"Repeated-Query Robustness" section.
