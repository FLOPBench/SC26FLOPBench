"""Minimal helper for inspecting LangGraph/DeepAgents checkpoint SQLite files."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping

import json

from langchain.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage


from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

SERIALIZER = JsonPlusSerializer(allowed_json_modules=True)


"""
The sqlite save files contain two tables: `checkpoints` and `writes`.

The `checkpoints` table has the following columns: 
thread_id, 
checkpoint_ns, 
checkpoint_id,
parent_checkpoint_id, 
type, 
checkpoint,
metadata. 

The `writes` table has the following columns: 
thread_id, 
checkpoint_ns, 
checkpoint_id, 
task_id, 
idx, 
channel, 
type, 
value. 

The "checkpoint", "metadata", and "value" columns are all serialized with the JsonPlusSerializer, so they should be deserialized using it too.
"""


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path).resolve())
    conn.row_factory = sqlite3.Row
    return conn


def _deserialize_column(row: sqlite3.Row, col_name: str, dtype: Any = None) -> Mapping[str, Any]:
    if dtype is None:
        return SERIALIZER.loads_typed((row["type"], row[col_name]))
    else:
        return SERIALIZER.loads_typed((dtype, row[col_name]))


def _serialize_message(message: Any) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    serialized: dict[str, Any] = {"type": type(message).__name__}
    for attr in ("role", "name", "content", "additional_kwargs", "metadata", "tool_name", "tool_input"):
        if hasattr(message, attr):
            value = getattr(message, attr)
            serialized[attr] = value
    return serialized


def get_thread_ids_from_sqlite(full_path: Path | str) -> list[str]:
    with _connect(full_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cursor.fetchall()
    thread_ids = [row[0] for row in rows]
    return thread_ids



def iter_checkpoints_table(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    limit: int | None = None,
) -> list[Mapping[str, Any]]:
    query = "SELECT * FROM checkpoints"
    params: list[str | int] = []
    if thread_id is not None:
        query += " WHERE thread_id=?"
        params.append(thread_id)
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    checkpoints: list[Mapping[str, Any]] = []
    for row in rows:
        checkpoints.append(
            {
                "thread_id": row["thread_id"],
                "checkpoint_id": row["checkpoint_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "parent_checkpoint_id": row["parent_checkpoint_id"],
                "type": row["type"],
                "checkpoint": _deserialize_column(row, "checkpoint"),
                "metadata": _deserialize_column(row, "metadata", "json"),
            }
        )
    return checkpoints

"""
def iter_writes_table(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    checkpoint_id: int | None = None,
) -> list[Mapping[str, Any]]:
    query = "SELECT * FROM writes"
    conditions: list[str] = []
    params: list[Any] = []
    if thread_id is not None:
        conditions.append("thread_id=?")
        params.append(thread_id)
    if checkpoint_id is not None:
        conditions.append("checkpoint_id=?")
        params.append(checkpoint_id)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY checkpoint_id, task_id, idx"

    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    writes: list[Mapping[str, Any]] = []
    for row in rows:
        writes.append(
            {
                "thread_id": row["thread_id"],
                "checkpoint_id": row["checkpoint_id"],
                "task_id": row["task_id"],
                "idx": row["idx"],
                "channel": row["channel"],
                "value": _deserialize_column(row, "value"),
            }
        )
    return writes
"""


def print_checkpoint_messages(
    db_path: Path | str,
    *,
    thread_id: str | None = None,
    limit: int | None = None,
    target_idx: int | None = None,
) -> None:
    entries = iter_checkpoints_table(db_path, thread_id=thread_id, limit=limit)

    if not entries:
        return

    entry = entries[-1]
    header = f"[{entry['thread_id']} {entry['checkpoint_id']}] type[{entry['type']}]"
    print("=========== CHECKPOINT START ===========")
    print(header)

    metadata = entry.get("metadata")
    if metadata:
        print("  metadata:")
        print(metadata)
    checkpoint = entry.get("checkpoint")
    if checkpoint:
        print("  checkpoint:")
        print(checkpoint)

    messages = checkpoint['channel_values']['messages']
    for msg in messages:
        print(" ======= MSG BEGIN =======")
        print(f'msg type: [{msg.type}]')

        if type(msg) is AIMessage:
            msg_cost = msg.response_metadata['token_usage']['cost']
            has_tool_calls = len(msg.tool_calls)
            if has_tool_calls:
                print('has tool calls!')
            else:
                print(msg.content)
            print(f'cost: {msg_cost}')

        if type(msg) is ToolMessage:
            print('got a tool message response!')

        print(msg.content)
        print(" ======= MSG END =======")
            
    print("============ CHECKPOINT END ============\n")


def print_pending_writes(db_path: Path | str, *, thread_id: str | None = None) -> None:
    writes = iter_writes_table(db_path, thread_id=thread_id)
    for write in writes:
        header = (
            f"[thread={write['thread_id']} checkpoint={write['checkpoint_id']} task={write['task_id']} "
            f"channel={write['channel']}]"
        )
        print(header)
        print(f"  value: {write['value']}")
