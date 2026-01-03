"""Utilities for inspecting LangGraph/DeepAgents checkpoint SQLite files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Mapping

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

SERIALIZER = JsonPlusSerializer(allowed_json_modules=True)


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path).resolve())
    conn.row_factory = sqlite3.Row
    return conn


def _deserialize_checkpoint(row: sqlite3.Row) -> Mapping[str, Any]:
    return SERIALIZER.loads_typed((row["type"], row["checkpoint"]))


def _deserialize_metadata(raw: bytes | None) -> Mapping[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {"raw": raw.decode("utf-8", errors="ignore")}


def _serialize_message(message: Any) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        return message
    payload: dict[str, Any] = {"type": type(message).__name__}
    for attr in (
        "role",
        "name",
        "content",
        "additional_kwargs",
        "metadata",
        "tool_name",
        "tool_input",
        "tool_args",
        "tool_response",
    ):
        if hasattr(message, attr):
            value = getattr(message, attr)
            payload[attr] = value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
    return payload


def _trace_value(value: Any) -> str:
    try:
        return json.dumps(value, default=lambda obj: repr(obj), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def get_thread_ids_from_sqlite(full_path: Path | str, *, success_only: bool = False) -> list[str]:
    with _connect(full_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row[0] for row in cursor.fetchall()]

    if success_only:
        return [tid for tid in thread_ids if _thread_succeeded(full_path, tid)]
    return thread_ids


def _thread_succeeded(db_path: Path | str, thread_id: str) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT type, checkpoint FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id DESC LIMIT 1",
            (thread_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        checkpoint = SERIALIZER.loads_typed((row["type"], row["checkpoint"]))
        tasks = checkpoint.get("channel_values", {}).get("__pregel_tasks")
        return not bool(tasks)


def iter_checkpoints(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    limit: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        sql = "SELECT * FROM checkpoints"
        params: list[str] = []
        if thread_id:
            sql += " WHERE thread_id=?"
            params.append(thread_id)
        sql += " ORDER BY checkpoint_id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(str(limit))
        cursor.execute(sql, params)
        for row in cursor:
            yield {
                "thread_id": row["thread_id"],
                "checkpoint_id": row["checkpoint_id"],
                "checkpoint": _deserialize_checkpoint(row),
                "metadata": _deserialize_metadata(row["metadata"]),
            }


def iter_writes(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    checkpoint_id: str | None = None,
) -> Iterator[Mapping[str, Any]]:
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        sql = "SELECT * FROM writes"
        filters: list[str] = []
        params: list[str] = []
        if thread_id:
            filters.append("thread_id=?")
            params.append(thread_id)
        if checkpoint_id:
            filters.append("checkpoint_id=?")
            params.append(checkpoint_id)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY checkpoint_id, task_id, idx"
        cursor.execute(sql, params)
        for row in cursor:
            yield {
                "thread_id": row["thread_id"],
                "checkpoint_id": row["checkpoint_id"],
                "task_id": row["task_id"],
                "channel": row["channel"],
                "value": SERIALIZER.loads_typed((row["type"], row["value"])),
            }


def iter_checkpoint_messages(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    limit: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    for checkpoint in iter_checkpoints(db_path, thread_id=thread_id, limit=limit):
        messages = checkpoint["checkpoint"]["channel_values"].get("messages")
        if not messages:
            continue
        for idx, message in enumerate(messages):
            yield {
                "thread_id": checkpoint["thread_id"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "index": idx,
                "message": _serialize_message(message),
            }


def print_checkpoint_messages(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    limit: int | None = None,
) -> None:
    for entry in iter_checkpoint_messages(db_path, thread_id=thread_id, limit=limit):
        header = f"[{entry['thread_id']} {entry['checkpoint_id']}] message[{entry['index']}]"
        print(header)
        print(_trace_value(entry["message"]))


def print_pending_writes(db_path: Path | str, *, thread_id: str | None = None) -> None:
    for write in iter_writes(db_path, thread_id=thread_id):
        header = (
            f"[thread={write['thread_id']} checkpoint={write['checkpoint_id']} task={write['task_id']} channel={write['channel']}]"
        )
        print(header)
        print(f"  value={_trace_value(write['value'])}")


def summarize_database(db_path: Path | str, *, thread_id: str | None = None) -> None:
    print(f"Inspecting {Path(db_path).resolve()}")
    for checkpoint in iter_checkpoints(db_path, thread_id=thread_id, limit=5):
        cp = checkpoint["checkpoint"]
        ts = cp.get("ts")
        messages = cp["channel_values"].get("messages", [])
        print(f"checkpoint {checkpoint['checkpoint_id']} ts={ts} messages={len(messages)}")
    print("Pending writes:")
    print_pending_writes(db_path, thread_id=thread_id)
