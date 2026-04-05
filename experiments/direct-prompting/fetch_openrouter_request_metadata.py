import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import psycopg
import requests
from tqdm import tqdm

from db_manager import CheckpointDBParser, ensure_postgres_running, setup_default_database


OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DEFAULT_PLOT_OUTPUT_DIR = os.path.join(
	WORKSPACE_ROOT,
	"experiments",
	"direct-prompting",
	"paper-figure-output",
	"request-metadata",
)
DEFAULT_MAX_HTTP_ERROR_REPEATS = 3


def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
	for key in keys:
		value = os.environ.get(key)
		if value:
			return value
	return default


def _is_dry_run_thread(thread_id: str) -> bool:
	return "_DRYRUN" in thread_id or thread_id.endswith("_DRYRUN")


def _load_module(module_name: str, relative_path: str):
	module_path = os.path.join(WORKSPACE_ROOT, relative_path)
	module_spec = importlib.util.spec_from_file_location(module_name, module_path)
	if module_spec is None or module_spec.loader is None:
		raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
	module = importlib.util.module_from_spec(module_spec)
	module_spec.loader.exec_module(module)
	return module


@dataclass(frozen=True)
class SourceGenerationRecord:
	generation_id: str
	thread_id: str
	llm_provider: Optional[str]
	llm_model_name: Optional[str]


@dataclass(frozen=True)
class FetchResult:
	generation_id: str
	fetch_status: str
	http_status: Optional[int]
	response_json: Optional[Dict[str, Any]]
	error_text: Optional[str]


class RateLimiter:
	def __init__(self, requests_per_second: float):
		if requests_per_second < 0:
			raise ValueError("requests_per_second must be >= 0")
		self.interval = 0.0 if requests_per_second == 0 else 1.0 / requests_per_second
		self.next_allowed_at = 0.0

	def wait(self) -> None:
		if self.interval <= 0:
			return
		now = time.monotonic()
		if self.next_allowed_at > now:
			time.sleep(self.next_allowed_at - now)
			now = time.monotonic()
		self.next_allowed_at = max(self.next_allowed_at, now) + self.interval


class RequestMetadataStore:
	def __init__(self, db_uri: str):
		self.db_uri = db_uri
		self.conn = psycopg.connect(self.db_uri, autocommit=True)
		self.setup()

	def setup(self) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				CREATE TABLE IF NOT EXISTS openrouter_generation_metadata (
					generation_id TEXT PRIMARY KEY,
					fetch_status TEXT NOT NULL,
					http_status INTEGER,
					response_json JSONB,
					error_text TEXT,
					upstream_id TEXT,
					request_id TEXT,
					provider_name TEXT,
					model TEXT,
					total_cost DOUBLE PRECISION,
					usage DOUBLE PRECISION,
					latency_ms DOUBLE PRECISION,
					generation_time_ms DOUBLE PRECISION,
					moderation_latency_ms DOUBLE PRECISION,
					tokens_prompt INTEGER,
					tokens_completion INTEGER,
					created_at TIMESTAMPTZ,
					fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
					updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
					fetch_attempts INTEGER NOT NULL DEFAULT 0,
					http_error_attempts INTEGER NOT NULL DEFAULT 0
				)
				"""
			)
			cur.execute(
				"ALTER TABLE openrouter_generation_metadata ADD COLUMN IF NOT EXISTS fetch_attempts INTEGER NOT NULL DEFAULT 0"
			)
			cur.execute(
				"ALTER TABLE openrouter_generation_metadata ADD COLUMN IF NOT EXISTS http_error_attempts INTEGER NOT NULL DEFAULT 0"
			)
			cur.execute(
				"""
				CREATE TABLE IF NOT EXISTS openrouter_generation_sources (
					generation_id TEXT NOT NULL REFERENCES openrouter_generation_metadata(generation_id) ON DELETE CASCADE,
					thread_id TEXT NOT NULL,
					llm_provider TEXT,
					llm_model_name TEXT,
					discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
					PRIMARY KEY (generation_id, thread_id)
				)
				"""
			)

	def fetch_existing_generation_ids(self, *, success_only: bool) -> set[str]:
		query = "SELECT generation_id FROM openrouter_generation_metadata"
		params: tuple[Any, ...] = ()
		if success_only:
			query += " WHERE fetch_status = %s"
			params = ("success",)
		with self.conn.cursor() as cur:
			cur.execute(query, params)
			return {row[0] for row in cur.fetchall()}

	def fetch_status_summary(self) -> Dict[str, int]:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				SELECT fetch_status, COUNT(*) AS row_count
				FROM openrouter_generation_metadata
				GROUP BY fetch_status
				"""
			)
			rows = cur.fetchall()

		summary = {str(fetch_status): int(row_count) for fetch_status, row_count in rows}
		summary["all_fetched_rows"] = int(sum(summary.values()))
		return summary

	def fetch_generation_retry_state(self) -> Dict[str, Dict[str, int | str | None]]:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				SELECT generation_id, fetch_status, http_status, fetch_attempts, http_error_attempts
				FROM openrouter_generation_metadata
				"""
			)
			rows = cur.fetchall()

		return {
			str(generation_id): {
				"fetch_status": fetch_status,
				"http_status": http_status,
				"fetch_attempts": int(fetch_attempts or 0),
				"http_error_attempts": int(http_error_attempts or 0),
			}
			for generation_id, fetch_status, http_status, fetch_attempts, http_error_attempts in rows
		}

	def fetch_success_dataframe(self):
		import pandas as pd

		query = """
			SELECT
				s.thread_id,
				m.generation_id,
				m.total_cost,
				m.latency_ms,
				m.generation_time_ms,
				m.moderation_latency_ms,
				m.request_id,
				m.provider_name,
				m.model,
				m.created_at,
				m.fetch_status,
				m.http_status
			FROM openrouter_generation_sources AS s
			INNER JOIN openrouter_generation_metadata AS m
				ON m.generation_id = s.generation_id
			WHERE m.fetch_status = 'success'
		"""
		with self.conn.cursor() as cur:
			cur.execute(query)
			rows = cur.fetchall()
			columns = [desc.name for desc in cur.description]
		return pd.DataFrame(rows, columns=columns)

	def upsert_generation_result(self, result: FetchResult) -> None:
		payload = result.response_json or {}
		data = payload.get("data") if isinstance(payload, dict) else None
		if not isinstance(data, dict):
			data = {}

		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO openrouter_generation_metadata (
					generation_id,
					fetch_status,
					http_status,
					response_json,
					error_text,
					upstream_id,
					request_id,
					provider_name,
					model,
					total_cost,
					usage,
					latency_ms,
					generation_time_ms,
					moderation_latency_ms,
					tokens_prompt,
					tokens_completion,
					created_at,
					fetched_at,
					updated_at,
					fetch_attempts,
					http_error_attempts
				)
				VALUES (
					%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), 1,
					CASE WHEN %s = 'http-error' THEN 1 ELSE 0 END
				)
				ON CONFLICT (generation_id) DO UPDATE
				SET fetch_status = EXCLUDED.fetch_status,
					http_status = EXCLUDED.http_status,
					response_json = EXCLUDED.response_json,
					error_text = EXCLUDED.error_text,
					upstream_id = EXCLUDED.upstream_id,
					request_id = EXCLUDED.request_id,
					provider_name = EXCLUDED.provider_name,
					model = EXCLUDED.model,
					total_cost = EXCLUDED.total_cost,
					usage = EXCLUDED.usage,
					latency_ms = EXCLUDED.latency_ms,
					generation_time_ms = EXCLUDED.generation_time_ms,
					moderation_latency_ms = EXCLUDED.moderation_latency_ms,
					tokens_prompt = EXCLUDED.tokens_prompt,
					tokens_completion = EXCLUDED.tokens_completion,
					created_at = EXCLUDED.created_at,
					fetched_at = EXCLUDED.fetched_at,
					updated_at = NOW(),
					fetch_attempts = openrouter_generation_metadata.fetch_attempts + 1,
					http_error_attempts = CASE
						WHEN EXCLUDED.fetch_status = 'http-error' THEN openrouter_generation_metadata.http_error_attempts + 1
						ELSE openrouter_generation_metadata.http_error_attempts
					END
				""",
				(
					result.generation_id,
					result.fetch_status,
					result.http_status,
					json.dumps(payload),
					result.error_text,
					data.get("upstream_id"),
					data.get("request_id"),
					data.get("provider_name"),
					data.get("model"),
					data.get("total_cost"),
					data.get("usage"),
					data.get("latency"),
					data.get("generation_time"),
					data.get("moderation_latency"),
					data.get("tokens_prompt"),
					data.get("tokens_completion"),
					data.get("created_at"),
					result.fetch_status,
				),
			)

	def upsert_generation_source(self, record: SourceGenerationRecord) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO openrouter_generation_sources (
					generation_id,
					thread_id,
					llm_provider,
					llm_model_name
				)
				VALUES (%s, %s, %s, %s)
				ON CONFLICT (generation_id, thread_id) DO UPDATE
				SET llm_provider = EXCLUDED.llm_provider,
					llm_model_name = EXCLUDED.llm_model_name
				""",
				(
					record.generation_id,
					record.thread_id,
					record.llm_provider,
					record.llm_model_name,
				),
			)

	def close(self) -> None:
		self.conn.close()


def _response_json_or_none(response: requests.Response) -> Optional[Dict[str, Any]]:
	try:
		payload = response.json()
	except ValueError:
		return None
	return payload if isinstance(payload, dict) else {"data": payload}


def _validate_generation_payload(payload: Optional[Dict[str, Any]], generation_id: str) -> Optional[str]:
	if payload is None:
		return "Response body was not valid JSON."

	data = payload.get("data")
	if not isinstance(data, dict):
		return "Response JSON did not contain an object-valued 'data' field."

	returned_generation_id = data.get("id")
	if not isinstance(returned_generation_id, str) or not returned_generation_id:
		return "Response JSON data did not contain a valid generation id."

	if returned_generation_id != generation_id:
		return (
			f"Response generation id mismatch: expected {generation_id}, "
			f"received {returned_generation_id}."
		)

	return None


def _retry_delay_seconds(response: Optional[requests.Response], attempt_index: int, base_delay: float) -> float:
	if response is not None:
		retry_after = response.headers.get("Retry-After")
		if retry_after:
			try:
				return max(float(retry_after), 0.0)
			except ValueError:
				pass
	return base_delay * (2 ** attempt_index)


def fetch_generation_metadata(
	session: requests.Session,
	generation_id: str,
	rate_limiter: RateLimiter,
	*,
	timeout_seconds: float,
	max_retries: int,
	retry_base_delay_seconds: float,
) -> FetchResult:
	last_error: Optional[str] = None
	for attempt_index in range(max_retries + 1):
		rate_limiter.wait()
		response: Optional[requests.Response] = None
		try:
			response = session.get(
				OPENROUTER_GENERATION_URL,
				params={"id": generation_id},
				timeout=timeout_seconds,
			)
		except requests.RequestException as exc:
			last_error = str(exc)
			if attempt_index >= max_retries:
				return FetchResult(
					generation_id=generation_id,
					fetch_status="network-error",
					http_status=None,
					response_json=None,
					error_text=last_error,
				)
			time.sleep(_retry_delay_seconds(None, attempt_index, retry_base_delay_seconds))
			continue

		payload = _response_json_or_none(response)
		if response.status_code == 200:
			payload_error = _validate_generation_payload(payload, generation_id)
			if payload_error is None:
				return FetchResult(
					generation_id=generation_id,
					fetch_status="success",
					http_status=response.status_code,
					response_json=payload,
					error_text=None,
				)

			last_error = payload_error
			if attempt_index < max_retries:
				time.sleep(_retry_delay_seconds(response, attempt_index, retry_base_delay_seconds))
				continue

			return FetchResult(
				generation_id=generation_id,
				fetch_status="invalid-payload",
				http_status=response.status_code,
				response_json=payload,
				error_text=payload_error,
			)

		last_error = None
		if payload is not None:
			last_error = json.dumps(payload, sort_keys=True)
		else:
			last_error = response.text.strip() or f"HTTP {response.status_code}"

		if response.status_code in RETRYABLE_STATUS_CODES and attempt_index < max_retries:
			time.sleep(_retry_delay_seconds(response, attempt_index, retry_base_delay_seconds))
			continue

		return FetchResult(
			generation_id=generation_id,
			fetch_status="http-error",
			http_status=response.status_code,
			response_json=payload,
			error_text=last_error,
		)

	return FetchResult(
		generation_id=generation_id,
		fetch_status="network-error",
		http_status=None,
		response_json=None,
		error_text=last_error or "Unknown fetch failure",
	)


def collect_openrouter_generation_records(
	parser: CheckpointDBParser,
	*,
	include_dry_run: bool,
) -> list[SourceGenerationRecord]:
	checkpoints = parser.fetch_all_checkpoints()
	tail_result = parser.fetch_tail_checkpoints_by_thread(checkpoints=checkpoints, tolerate_errors=True)
	records: list[SourceGenerationRecord] = []

	for thread_id, checkpoint in tail_result["tails"].items():
		if not include_dry_run and _is_dry_run_thread(thread_id):
			continue

		channel_values = checkpoint["checkpoint"].get("channel_values", {})
		if "total_tokens" not in channel_values:
			continue

		generation_id = channel_values.get("llm_response_id")
		llm_provider = channel_values.get("llm_provider")
		llm_model_name = channel_values.get("llm_model_name")

		if not generation_id:
			parser.hydrate_checkpoint_channels(checkpoint, ["raw_response"])
			raw_response = checkpoint["checkpoint"]["channel_values"].get("raw_response") or {}
			response_metadata = raw_response.get("response_metadata") or {}
			generation_id = response_metadata.get("id")
			llm_provider = llm_provider or response_metadata.get("model_provider")
			llm_model_name = llm_model_name or response_metadata.get("model_name") or response_metadata.get("model")

		if not generation_id:
			continue

		generation_id = str(generation_id)
		if not generation_id.startswith("gen-"):
			continue

		records.append(
			SourceGenerationRecord(
				generation_id=generation_id,
				thread_id=thread_id,
				llm_provider=llm_provider,
				llm_model_name=llm_model_name,
			)
		)

	records.sort(key=lambda record: (record.generation_id, record.thread_id))
	return records


def _dedupe_generation_records(records: Iterable[SourceGenerationRecord]) -> list[SourceGenerationRecord]:
	first_record_by_generation_id: Dict[str, SourceGenerationRecord] = {}
	for record in records:
		first_record_by_generation_id.setdefault(record.generation_id, record)
	return list(first_record_by_generation_id.values())


def _load_samples_dataframe(db_uri: str, include_dry_run: bool):
	visualize_results = _load_module(
		"visualize_results_for_request_metadata",
		os.path.join("experiments", "direct-prompting", "visualize_results.py"),
	)
	parser = visualize_results.CheckpointDBParser(db_uri)
	attempt_tracker = visualize_results.QueryAttemptTracker(db_uri)
	try:
		checkpoints = parser.fetch_all_checkpoints()
		tail_checkpoint_result = parser.fetch_tail_checkpoints_by_thread(
			checkpoints=checkpoints,
			tolerate_errors=True,
		)
		tail_checkpoints = tail_checkpoint_result["tails"]
		invalid_threads = tail_checkpoint_result["invalid_threads"]
		for checkpoint in tail_checkpoints.values():
			channel_values = checkpoint["checkpoint"]["channel_values"]
			channel_versions = checkpoint["checkpoint"].get("channel_versions", {})
			if "total_tokens" in channel_values:
				channels_to_hydrate = [
					channel
					for channel in ["prediction", "metrics_diff", "metrics_pct_diff", "metrics_explanations"]
					if channel in channel_values or channel in channel_versions
				]
				if channels_to_hydrate:
					parser.hydrate_checkpoint_channels(checkpoint, channels_to_hydrate)
		thread_generation_ids: Dict[str, Optional[str]] = {}
		for thread_id, checkpoint in tail_checkpoints.items():
			channel_values = checkpoint["checkpoint"]["channel_values"]
			channel_versions = checkpoint["checkpoint"].get("channel_versions", {})
			generation_id = channel_values.get("llm_response_id")
			if generation_id is None and "llm_response_id" in channel_versions:
				parser.hydrate_checkpoint_channels(checkpoint, ["llm_response_id"])
				generation_id = checkpoint["checkpoint"]["channel_values"].get("llm_response_id")
			if not generation_id:
				if "raw_response" in channel_values or "raw_response" in channel_versions:
					parser.hydrate_checkpoint_channels(checkpoint, ["raw_response"])
					raw_response = checkpoint["checkpoint"]["channel_values"].get("raw_response") or {}
					response_metadata = raw_response.get("response_metadata") or {}
					generation_id = response_metadata.get("id")
			if generation_id is not None:
				generation_id = str(generation_id)
				if not generation_id.startswith("gen-"):
					generation_id = None
			thread_generation_ids[thread_id] = generation_id
		attempts = attempt_tracker.fetch_all_attempts()
	finally:
		parser.close()
		attempt_tracker.close()

	visualize_results._print_invalid_thread_warnings(invalid_threads)
	stored_thread_ids = visualize_results._stored_thread_ids(checkpoints, attempts)
	if not stored_thread_ids:
		raise RuntimeError("No checkpoint or query-attempt records were found in the database.")

	if not include_dry_run:
		non_dry_run_thread_ids = {
			thread_id for thread_id in stored_thread_ids if not visualize_results._is_dry_run_thread(thread_id)
		}
		if not non_dry_run_thread_ids:
			raise RuntimeError(
				"The database currently contains only dry-run thread IDs. "
				"Re-run with --includeDryRun or populate the database with non-dry experiment runs."
			)

	samples_df = visualize_results._database_dataframe(tail_checkpoints, attempts, include_dry_run)
	if not samples_df.empty:
		samples_df["llm_response_id"] = samples_df["thread_id"].map(thread_generation_ids)
	return visualize_results, samples_df


def _build_request_metadata_plot_dataframe(
	*,
	source_db_uri: str,
	target_store: RequestMetadataStore,
	include_dry_run: bool,
	only_shared_samples: bool,
):
	import pandas as pd

	visualize_results, samples_df = _load_samples_dataframe(source_db_uri, include_dry_run)
	completed_df = samples_df[samples_df["status"] == "completed"].copy()
	if completed_df.empty:
		raise RuntimeError("No completed samples were found in the source database.")
	completed_df["llm_response_id"] = completed_df.get("llm_response_id")
	completed_df["llm_response_id"] = completed_df["llm_response_id"].astype(str)
	completed_df["llm_response_id"] = completed_df["llm_response_id"].where(
		completed_df["llm_response_id"].str.startswith("gen-"),
		None,
	)

	metadata_df = target_store.fetch_success_dataframe()
	if metadata_df.empty:
		raise RuntimeError("No successful OpenRouter metadata rows were found in request_metadata.")

	metadata_df = metadata_df.drop_duplicates(subset=["generation_id"], keep="last").copy()
	metadata_df["query_time"] = pd.to_numeric(metadata_df["generation_time_ms"], errors="coerce") / 1000.0
	metadata_df["latency_seconds"] = pd.to_numeric(metadata_df["latency_ms"], errors="coerce") / 1000.0
	missing_generation_time_mask = metadata_df["query_time"].isna()
	metadata_df.loc[missing_generation_time_mask, "query_time"] = metadata_df.loc[missing_generation_time_mask, "latency_seconds"]
	metadata_df["cost_usd"] = pd.to_numeric(metadata_df["total_cost"], errors="coerce")
	metadata_plot_df = metadata_df[
		["generation_id", "query_time", "cost_usd", "latency_seconds", "generation_time_ms", "latency_ms"]
	].rename(
		columns={
			"query_time": "metadata_query_time",
			"cost_usd": "metadata_cost_usd",
			"latency_seconds": "metadata_latency_seconds",
			"generation_time_ms": "metadata_generation_time_ms",
			"latency_ms": "metadata_latency_ms",
		}
	)

	completed_df = completed_df[completed_df["llm_response_id"].notna()].copy()
	if completed_df.empty:
		raise RuntimeError("No completed source samples exposed an OpenRouter generation id.")

	plot_df = completed_df.merge(
		metadata_plot_df,
		left_on="llm_response_id",
		right_on="generation_id",
		how="inner",
	)
	if plot_df.empty:
		raise RuntimeError("No completed source samples matched rows in request_metadata.")

	plot_df["query_time"] = plot_df["metadata_query_time"]
	plot_df["cost_usd"] = plot_df["metadata_cost_usd"]
	plot_df["latency_seconds"] = plot_df["metadata_latency_seconds"]
	plot_df["generation_time_ms"] = plot_df["metadata_generation_time_ms"]
	plot_df["latency_ms"] = plot_df["metadata_latency_ms"]

	plot_df["status"] = "completed"
	plot_df["use_sass"] = plot_df["use_sass"].fillna(False).astype(bool)
	plot_df["use_imix"] = plot_df["use_imix"].fillna(False).astype(bool)

	completed_generation_ids = set(completed_df["llm_response_id"].dropna().tolist())
	metadata_generation_ids = set(metadata_df["generation_id"].dropna().tolist())
	matched_thread_ids = set(plot_df["thread_id"].dropna().tolist())
	missing_generation_ids = sorted(completed_generation_ids - metadata_generation_ids)
	missing_metadata_df = completed_df[completed_df["llm_response_id"].isin(missing_generation_ids)].copy()
	missing_thread_ids = sorted(missing_metadata_df["thread_id"].dropna().tolist())
	missing_generation_ids = sorted(
		{
			generation_id
			for generation_id in missing_metadata_df["llm_response_id"].dropna().tolist()
			if isinstance(generation_id, str) and generation_id.startswith("gen-")
		}
	)

	if only_shared_samples:
		plot_df = visualize_results._filter_only_shared_samples(plot_df, include_imix=False)
		if plot_df.empty:
			raise RuntimeError("No joined samples remained after applying --onlySharedSamples.")

	return {
		"visualize_results": visualize_results,
		"plot_df": plot_df,
		"fallback_latency_rows": int(missing_generation_time_mask.sum()),
		"metadata_row_count": int(len(metadata_df)),
		"completed_row_count": int(len(completed_df)),
		"matched_row_count": int(len(matched_thread_ids)),
		"missing_thread_ids": missing_thread_ids,
		"missing_generation_ids": missing_generation_ids,
		"missing_metadata_count": int(len(missing_thread_ids)),
	}


def _format_metric_summary_table(
	plot_df,
	*,
	value_column: str,
	metric_label: str,
	evidence_order: list[str],
	model_order: list[str],
) -> str:
	import pandas as pd

	if plot_df.empty or value_column not in plot_df.columns:
		return f"{metric_label}\n(no plotted rows)"

	summary_df = plot_df[["model_name", "evidence_configuration", value_column]].copy()
	summary_df[value_column] = pd.to_numeric(summary_df[value_column], errors="coerce")
	summary_df = summary_df[summary_df[value_column].notna()].copy()
	if summary_df.empty:
		return f"{metric_label}\n(no plotted rows)"

	summary_df["model_sort"] = summary_df["model_name"].map(
		{name: index for index, name in enumerate(model_order)}
	)
	summary_df["evidence_sort"] = summary_df["evidence_configuration"].map(
		{name: index for index, name in enumerate(evidence_order)}
	)

	grouped = (
		summary_df.groupby(["model_name", "evidence_configuration"], dropna=False)[value_column]
		.agg(
			n="count",
			total_sum="sum",
			q1=lambda values: values.quantile(0.25),
			median="median",
			q3=lambda values: values.quantile(0.75),
		)
		.reset_index()
	)
	grouped["model_sort"] = grouped["model_name"].map({name: index for index, name in enumerate(model_order)})
	grouped["evidence_sort"] = grouped["evidence_configuration"].map(
		{name: index for index, name in enumerate(evidence_order)}
	)
	grouped["model_sort"] = grouped["model_sort"].fillna(len(model_order)).astype(int)
	grouped["evidence_sort"] = grouped["evidence_sort"].fillna(len(evidence_order)).astype(int)
	grouped = grouped.sort_values(
		by=["model_sort", "evidence_sort", "model_name", "evidence_configuration"],
		kind="stable",
	)

	rows = []
	for row in grouped.itertuples(index=False):
		rows.append(
			{
				"Model": str(row.model_name),
				"Prompt Type": str(row.evidence_configuration),
				"N": str(int(row.n)),
				"Cumulative Sum": f"{float(row.total_sum):.6g}",
				"Q1": f"{float(row.q1):.6g}",
				"Median": f"{float(row.median):.6g}",
				"Q3": f"{float(row.q3):.6g}",
			}
		)

	headers = ["Model", "Prompt Type", "N", "Cumulative Sum", "Q1", "Median", "Q3"]
	widths = {
		header: max(len(header), *(len(row[header]) for row in rows))
		for header in headers
	}

	lines = [metric_label]
	lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
	lines.append("-+-".join("-" * widths[header] for header in headers))
	for row in rows:
		lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))
	return "\n".join(lines)


def _annotate_request_metadata_group_sums(
	ax,
	plot_df,
	value_column: str,
	model_order: list[str],
	hue_order: list[str],
) -> None:
	import math
	import matplotlib.transforms as mtransforms

	if plot_df.empty:
		return

	trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
	if len(hue_order) == 1:
		offsets = [0.0]
	else:
		step = 0.6 / max(len(hue_order) - 1, 1)
		offsets = [(-0.3 + step * index) for index in range(len(hue_order))]
	offset_by_hue = {label: offsets[index] for index, label in enumerate(hue_order)}
	annotation_colors = {
		label: color
		for label, color in zip(hue_order, ["#424242", "#111111", "#616161", "#000000"], strict=False)
	}
	group_sums = (
		plot_df.groupby(["model_name", "evidence_configuration"], dropna=False)[value_column].sum().to_dict()
	)

	for model_index, model_name in enumerate(model_order):
		for hue_label in hue_order:
			total_value = group_sums.get((model_name, hue_label))
			if total_value is None or not math.isfinite(float(total_value)):
				continue
			ax.text(
				0.982,
				model_index + offset_by_hue[hue_label],
				f"${float(total_value):.4f}",
				transform=trans,
				ha="right",
				va="center",
				fontsize=9.5,
				color=annotation_colors.get(hue_label, "#111111"),
				bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.35},
			)


def _save_request_metadata_histogram(
	visualize_results,
	completed_df,
	value_column: str,
	title: str,
	x_label: str,
	output_path: Path,
	evidence_order: list[str],
	*,
	annotate_group_sums: bool = False,
) -> None:
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import seaborn as sns

	sns.set_theme(
		style="whitegrid",
		rc={
			"axes.titlesize": 15,
			"axes.labelsize": 13,
			"xtick.labelsize": 11,
			"ytick.labelsize": 12,
			"legend.fontsize": 11,
			"legend.title_fontsize": 12,
		},
	)
	model_count = completed_df["model_name"].nunique() if "model_name" in completed_df.columns else 0
	fig_height = visualize_results._bounded_plot_height(
		model_count,
		min_height=6.5,
		per_item=0.55,
		padding=1.3,
		max_height=9.5,
	)
	fig, ax = plt.subplots(figsize=visualize_results._scaled_figsize(8.6, fig_height))
	hue_order = evidence_order

	if completed_df.empty or value_column not in completed_df.columns or "evidence_configuration" not in completed_df.columns or "model_name" not in completed_df.columns:
		ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes, fontsize=13)
		ax.set_xlabel(x_label)
		ax.set_ylabel("Model Name")
		ax.set_title(title)
		fig.tight_layout()
		fig.savefig(output_path, dpi=220, bbox_inches="tight")
		plt.close(fig)
		return

	plot_df = completed_df.copy()
	plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
	plot_df = plot_df[plot_df[value_column].notna()]
	plot_df = plot_df[np.isfinite(plot_df[value_column].to_numpy())]
	model_order = visualize_results._sorted_model_names(plot_df)

	if plot_df.empty:
		ax.text(0.5, 0.5, "No completed samples", ha="center", va="center", transform=ax.transAxes, fontsize=13)
	else:
		sns.boxplot(
			data=plot_df,
			x=value_column,
			y="model_name",
			hue="evidence_configuration",
			order=model_order,
			hue_order=hue_order,
			orient="h",
			width=0.6,
			linewidth=1.35,
			ax=ax,
		)
		if annotate_group_sums:
			_annotate_request_metadata_group_sums(ax, plot_df, value_column, model_order, hue_order)

	ax.set_title(title, pad=10)
	ax.set_xlabel(x_label)
	ax.set_ylabel("Model Name")
	ax.tick_params(axis="y", pad=6)
	legend = ax.get_legend()
	if legend is not None:
		handles, labels = ax.get_legend_handles_labels()
		legend.remove()
		ax.legend(
			handles,
			labels,
			title=visualize_results.PROMPT_TYPE_LABEL,
			loc="upper center",
			bbox_to_anchor=(0.5, visualize_results.BOTTOM_LEGEND_Y),
			ncol=visualize_results._legend_ncols(len(labels)),
			frameon=False,
		)
	fig.tight_layout(rect=(0, visualize_results.BOTTOM_LEGEND_RECT, 1, 1))
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def make_plots_for_paper(
	*,
	source_db_uri: str,
	target_store: RequestMetadataStore,
	output_dir: Path,
	include_dry_run: bool,
	only_shared_samples: bool,
) -> None:
	plot_context = _build_request_metadata_plot_dataframe(
		source_db_uri=source_db_uri,
		target_store=target_store,
		include_dry_run=include_dry_run,
		only_shared_samples=only_shared_samples,
	)
	visualize_results = plot_context["visualize_results"]
	completed_df = plot_context["plot_df"]
	fallback_latency_rows = plot_context["fallback_latency_rows"]
	metadata_row_count = plot_context["metadata_row_count"]
	completed_row_count = plot_context["completed_row_count"]
	matched_row_count = plot_context["matched_row_count"]
	missing_thread_ids = plot_context["missing_thread_ids"]
	missing_generation_ids = plot_context["missing_generation_ids"]
	missing_metadata_count = plot_context["missing_metadata_count"]

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_evidence_order = visualize_results._plot_evidence_configuration_order(False)
	plot_completed_df = visualize_results._prepare_plot_dataframe(completed_df, include_imix=False)
	rows_after_prepare_plot_dataframe = len(plot_completed_df)
	plot_models = visualize_results._models_with_completed_runs(plot_completed_df)
	model_count_after_prepare = len(plot_models)
	plot_completed_df = visualize_results._filter_plot_models(plot_completed_df, plot_models)
	rows_after_model_filter = len(plot_completed_df)
	plot_model_order = visualize_results._sorted_model_names(plot_completed_df)

	evidence_counts: Dict[str, int] = {}
	if not plot_completed_df.empty and "evidence_configuration" in plot_completed_df.columns:
		evidence_counts = {
			str(label): int(count)
			for label, count in plot_completed_df["evidence_configuration"].value_counts(dropna=False).to_dict().items()
		}

	model_counts: Dict[str, int] = {}
	if not plot_completed_df.empty and "model_name" in plot_completed_df.columns:
		model_counts = {
			str(label): int(count)
			for label, count in plot_completed_df["model_name"].value_counts(dropna=False).to_dict().items()
		}

	plot2_path = output_dir / "plot2_query_time_distribution.png"
	_save_request_metadata_histogram(
		visualize_results,
		plot_completed_df,
		"query_time",
		"Query Time Distribution by Model and Prompt Type",
		"Query Time (seconds)",
		plot2_path,
		plot_evidence_order,
	)

	plot3_path = output_dir / "plot3_cost_distribution.png"
	_save_request_metadata_histogram(
		visualize_results,
		plot_completed_df,
		"cost_usd",
		"Query Cost Distribution by Model and Prompt Type",
		"Cost (USD)",
		plot3_path,
		plot_evidence_order,
		annotate_group_sums=True,
	)

	query_time_summary = _format_metric_summary_table(
		plot_completed_df,
		value_column="query_time",
		metric_label="query_time_summary_seconds",
		evidence_order=plot_evidence_order,
		model_order=plot_model_order,
	)
	cost_summary = _format_metric_summary_table(
		plot_completed_df,
		value_column="cost_usd",
		metric_label="cost_summary_usd",
		evidence_order=plot_evidence_order,
		model_order=plot_model_order,
	)

	print(f"completed_source_rows: {completed_row_count}")
	print(f"joined_completed_rows: {len(completed_df)}")
	print(f"matched_thread_ids_before_shared_sample_filter: {matched_row_count}")
	print(f"metadata_rows_considered: {metadata_row_count}")
	print(f"rows_using_latency_fallback: {fallback_latency_rows}")
	print(f"missing_metadata_rows: {missing_metadata_count}")
	print(f"rows_after_prepare_plot_dataframe: {rows_after_prepare_plot_dataframe}")
	print(f"model_count_after_prepare_plot_dataframe: {model_count_after_prepare}")
	print(f"rows_after_model_filter: {rows_after_model_filter}")
	if evidence_counts:
		print("plot_evidence_configuration_counts:")
		for label, count in sorted(evidence_counts.items()):
			print(f"- {label}: {count}")
	if model_counts:
		print("plot_model_counts:")
		for label, count in sorted(model_counts.items()):
			print(f"- {label}: {count}")
	if rows_after_model_filter == 0:
		print(
			"plot_warning: No completed samples survived the plotting filters. "
			"Check joined_completed_rows, evidence counts, model counts, and onlySharedSamples filtering."
		)
	if missing_thread_ids:
		print("missing_thread_id_examples:")
		for thread_id in missing_thread_ids[:10]:
			print(f"- {thread_id}")
	if missing_generation_ids:
		print("missing_generation_id_examples:")
		for generation_id in missing_generation_ids[:10]:
			print(f"- {generation_id}")
	print(query_time_summary)
	print(cost_summary)
	print("plot_artifacts:")
	print(f"- {plot2_path}")
	print(f"- {plot3_path}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Fetch OpenRouter request metadata for generation IDs stored in gpuflops_db and write results to request_metadata."
	)
	parser.add_argument("--sourceDbUri", type=str, default=None, help="Explicit source PostgreSQL URI. Defaults to gpuflops_db on localhost.")
	parser.add_argument("--sourceDbName", type=str, default="gpuflops_db", help="Source PostgreSQL database name when sourceDbUri is not supplied.")
	parser.add_argument("--targetDbUri", type=str, default=None, help="Explicit target PostgreSQL URI. Defaults to request_metadata on localhost.")
	parser.add_argument("--targetDbName", type=str, default="request_metadata", help="Target PostgreSQL database name when targetDbUri is not supplied.")
	parser.add_argument("--requestsPerSecond", type=float, default=1.0, help="Maximum number of OpenRouter requests per second. Use 0 to disable client-side pacing.")
	parser.add_argument("--requestTimeout", type=float, default=30.0, help="Per-request timeout in seconds.")
	parser.add_argument("--maxRetries", type=int, default=4, help="Maximum retry attempts for retryable HTTP/network failures.")
	parser.add_argument("--retryBaseDelay", type=float, default=1.0, help="Base delay in seconds for exponential retry backoff.")
	parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N unique generation IDs after filtering.")
	parser.add_argument("--includeDryRun", action="store_true", help="Include dry-run thread IDs from the source checkpoint database.")
	parser.add_argument("--overwrite", action="store_true", help="Re-fetch generation IDs that already have a successful row in the target database.")
	parser.add_argument("--openrouterApiKey", type=str, default=None, help="OpenRouter API key. Defaults to OPENROUTER_API_KEY or OPENAI_API_KEY.")
	parser.add_argument("--makePlotsForPaper", action="store_true", help="Join request_metadata with gpuflops_db and generate the query-time and cost plots used for paper figures.")
	parser.add_argument("--onlySharedSamples", action="store_true", help="Keep only joined benchmark/kernel/GPU tuples that have at least one completed row for every model name across both plotted prompt types: Source-Only and Source+SASS.")
	parser.add_argument("--plotOutputDir", type=str, default=DEFAULT_PLOT_OUTPUT_DIR, help="Directory where --makePlotsForPaper artifacts will be written.")
	parser.add_argument("--maxHttpErrorRepeats", type=int, default=DEFAULT_MAX_HTTP_ERROR_REPEATS, help="Maximum number of automatic repeat attempts for generation IDs whose latest stored status is http-error before they are treated as exhausted.")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	api_key = args.openrouterApiKey or _env_first("OPENROUTER_API_KEY", "OPENAI_API_KEY")

	ensure_postgres_running()
	source_db_uri = args.sourceDbUri or setup_default_database(db_name=args.sourceDbName)
	target_db_uri = args.targetDbUri or setup_default_database(db_name=args.targetDbName)

	source_parser = CheckpointDBParser(source_db_uri)
	target_store = RequestMetadataStore(target_db_uri)

	try:
		if args.makePlotsForPaper:
			make_plots_for_paper(
				source_db_uri=source_db_uri,
				target_store=target_store,
				output_dir=Path(args.plotOutputDir),
				include_dry_run=args.includeDryRun,
				only_shared_samples=args.onlySharedSamples,
			)
			return

		source_records = collect_openrouter_generation_records(source_parser, include_dry_run=args.includeDryRun)
		if not source_records:
			print("No OpenRouter generation IDs were found in the source database.")
			return

		unique_generation_records = _dedupe_generation_records(source_records)
		retry_state_by_generation_id = target_store.fetch_generation_retry_state()
		pending_generation_records: list[SourceGenerationRecord] = []
		already_success_count = 0
		exhausted_http_error_count = 0
		for record in unique_generation_records:
			retry_state = retry_state_by_generation_id.get(record.generation_id)
			if retry_state is None:
				pending_generation_records.append(record)
				continue

			fetch_status = retry_state.get("fetch_status")
			http_error_attempts = int(retry_state.get("http_error_attempts") or 0)

			if not args.overwrite and fetch_status == "success":
				already_success_count += 1
				continue

			if fetch_status == "http-error" and http_error_attempts >= args.maxHttpErrorRepeats:
				exhausted_http_error_count += 1
				continue

			pending_generation_records.append(record)

		already_captured_count = already_success_count + exhausted_http_error_count
		if args.limit is not None:
			pending_generation_records = pending_generation_records[: args.limit]

		print(f"source_threads_with_openrouter_ids: {len(source_records)}")
		print(f"unique_generation_ids: {len(unique_generation_records)}")
		print(f"pending_generation_ids: {len(pending_generation_records)}")
		print(f"already_success_generation_ids: {already_success_count}")
		print(f"exhausted_http_error_generation_ids: {exhausted_http_error_count}")
		print(f"already_captured_generation_ids: {already_captured_count}")
		print(f"resume_mode_skip_existing_successes: {not args.overwrite}")
		print(f"max_http_error_repeats: {args.maxHttpErrorRepeats}")

		source_records_by_generation_id: Dict[str, list[SourceGenerationRecord]] = {}
		for record in source_records:
			source_records_by_generation_id.setdefault(record.generation_id, []).append(record)

		invocation_summary = {
			"success": 0,
			"http-error": 0,
			"network-error": 0,
			"invalid-payload": 0,
			"skipped": already_success_count,
			"exhausted-http-error": exhausted_http_error_count,
		}

		if pending_generation_records:
			if not api_key:
				raise RuntimeError(
					"An OpenRouter API key is required via --openrouterApiKey or OPENROUTER_API_KEY when there are pending generation IDs to fetch."
				)
			rate_limiter = RateLimiter(args.requestsPerSecond)
			session = requests.Session()
			session.headers.update(
				{
					"Authorization": f"Bearer {api_key}",
					"Accept": "application/json",
				}
			)

			progress_total = len(unique_generation_records)
			if args.limit is not None:
				progress_total = already_captured_count + len(pending_generation_records)

			progress_bar = tqdm(
				pending_generation_records,
				total=progress_total,
				initial=already_captured_count,
				desc="OpenRouter metadata",
				unit="req",
				dynamic_ncols=True,
			)
			for generation_record in progress_bar:
				progress_bar.set_postfix_str(generation_record.generation_id, refresh=False)
				result = fetch_generation_metadata(
					session,
					generation_record.generation_id,
					rate_limiter,
					timeout_seconds=args.requestTimeout,
					max_retries=args.maxRetries,
					retry_base_delay_seconds=args.retryBaseDelay,
				)
				target_store.upsert_generation_result(result)
				for source_record in source_records_by_generation_id.get(generation_record.generation_id, []):
					target_store.upsert_generation_source(source_record)
				invocation_summary[result.fetch_status] = invocation_summary.get(result.fetch_status, 0) + 1
			progress_bar.close()

		database_summary = target_store.fetch_status_summary()

		print("summary_current_invocation:")
		for key in sorted(invocation_summary):
			print(f"- {key}: {invocation_summary[key]}")
		print("summary_database_total:")
		for key in sorted(database_summary):
			print(f"- {key}: {database_summary[key]}")
		print(f"target_database: {args.targetDbName if args.targetDbUri is None else target_db_uri}")
	finally:
		source_parser.close()
		target_store.close()


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		raise