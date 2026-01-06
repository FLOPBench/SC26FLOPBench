"""Minimal helper for inspecting LangGraph/DeepAgents checkpoint SQLite files."""

from __future__ import annotations

import pprint
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

import matplotlib.pyplot as plt
import pandas as pd

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


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return pprint.pformat(value, width=120)


def _print_section(
    label: str,
    value: Any,
    *,
    indent: int = 2,
    skip_empty: bool = True,
) -> None:
    if skip_empty:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        if isinstance(value, (list, dict)) and not value:
            return

    print(" " * indent + f"{label}:")
    rendered = _stringify_value(value)
    if not rendered:
        return
    for line in rendered.splitlines():
        print(" " * (indent + 2) + line)


def _print_tool_call(call: Mapping[str, Any], *, indent: int) -> None:
    name = call.get("name") or call.get("tool_name") or "<unnamed>"
    call_id = call.get("id") or call.get("tool_call_id")
    header = f"- {name}"
    if call_id:
        header += f" (id={call_id})"
    print(" " * indent + header)

    args = call.get("args")
    if args is not None:
        _print_section("args", args, indent=indent + 2, skip_empty=False)

    for key, value in call.items():
        if key in {"name", "args", "id", "tool_call_id"}:
            continue
        _print_section(key, value, indent=indent + 2)


def _print_ai_message(message: AIMessage, *, indent: int = 2) -> None:
    if getattr(message, "content", None):
        _print_section("content", message.content, indent=indent)

    response_metadata = dict(message.response_metadata or {})
    token_usage = response_metadata.pop("token_usage", None)
    if token_usage:
        _print_section("token_usage", token_usage, indent=indent)
    if response_metadata:
        _print_section("response_metadata", response_metadata, indent=indent)

    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        if hasattr(usage_metadata, "model_dump"):
            usage_metadata = usage_metadata.model_dump()
        _print_section("usage_metadata", usage_metadata, indent=indent)

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        print(" " * indent + "tool_calls:")
        for call in tool_calls:
            if isinstance(call, Mapping):
                _print_tool_call(call, indent=indent + 2)
            else:
                _print_section("call", call, indent=indent + 2, skip_empty=False)

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None) or []
    if invalid_tool_calls:
        print(" " * indent + "invalid_tool_calls:")
        for call in invalid_tool_calls:
            if isinstance(call, Mapping):
                _print_tool_call(call, indent=indent + 2)
            else:
                _print_section("call", call, indent=indent + 2, skip_empty=False)


def _print_tool_message(message: ToolMessage, *, indent: int = 2) -> None:
    name = message.name or "<unnamed>"
    header = f"tool name={name} (tool_call_id={message.tool_call_id})"
    if getattr(message, "status", None):
        header += f" status={message.status}"
    print(" " * indent + header)
    _print_section("content", message.content, indent=indent + 2)
    _print_section("artifact", getattr(message, "artifact", None), indent=indent + 2)
    _print_section("response_metadata", message.response_metadata, indent=indent + 2)
    _print_section("additional_kwargs", message.additional_kwargs, indent=indent + 2)


def _collect_channel_messages(channel_values: Mapping[str, Any] | None) -> list[Any]:
    if not isinstance(channel_values, Mapping):
        return []
    messages: list[Any] = []
    direct = channel_values.get("messages")
    if isinstance(direct, list):
        messages.extend(direct)
    for value in channel_values.values():
        if isinstance(value, Mapping):
            nested = value.get("messages")
            if isinstance(nested, list):
                messages.extend(nested)
    return messages


def _print_message_header(msg: Any, idx: int) -> str:
    if isinstance(msg, Mapping):
        type_name = msg.get("type") or msg.get("role") or "Mapping"
        msg_id = msg.get("id")
    else:
        type_name = type(msg).__name__
        msg_id = getattr(msg, "id", None)
    header = f"======= Message {idx} [{type_name}]"
    if msg_id:
        header += f" id={msg_id}"
    print(header)
    return header


def _print_message(msg: Any, idx: int) -> None:
    if idx:
        print()
    header = _print_message_header(msg, idx)
    indent = 2
    if isinstance(msg, AIMessage):
        _print_ai_message(msg, indent=indent)
    elif isinstance(msg, ToolMessage):
        _print_tool_message(msg, indent=indent)
    elif isinstance(msg, (HumanMessage, SystemMessage)):
        _print_section("content", msg.content, indent=indent, skip_empty=False)
        _print_section("additional_kwargs", msg.additional_kwargs, indent=indent)
        _print_section("response_metadata", msg.response_metadata, indent=indent)
    elif isinstance(msg, Mapping):
        _print_section("payload", msg, indent=indent, skip_empty=False)
    else:
        _print_section("payload", repr(msg), indent=indent, skip_empty=False)
    print("=" * len(header))


def _normalize_usage_metadata(usage_metadata: Any) -> Mapping[str, Any]:
    if usage_metadata is None:
        return {}
    if isinstance(usage_metadata, Mapping):
        return usage_metadata
    if hasattr(usage_metadata, "model_dump"):
        try:
            return usage_metadata.model_dump()
        except Exception:
            pass
    if hasattr(usage_metadata, "__dict__"):
        return dict(vars(usage_metadata))
    return {}


def _parse_int_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, )):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return None
    return None


def _parse_float_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_numeric_from_sources(
    sources: list[Any], keys: tuple[str, ...], parser: Callable[[Any], Any]
) -> Any:
    for source in sources:
        if not source:
            continue
        for key in keys:
            if isinstance(source, Mapping):
                val = source.get(key)
            else:
                val = getattr(source, key, None)
            if val is None:
                continue
            parsed = parser(val)
            if parsed is not None:
                return parsed
    return None


def _summarize_llm_usage(messages: list[Any]) -> tuple[int, int, float, list[dict[str, float | int]]]:
    total_input = 0
    total_output = 0
    total_cost = 0.0
    had_cost = False
    records: list[dict[str, float | int]] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        usage_metadata = getattr(msg, "usage_metadata", None)
        usage_data = _normalize_usage_metadata(usage_metadata)
        token_usage = (msg.response_metadata or {}).get("token_usage") or {}
        sources = [usage_data, token_usage]

        input_tokens = _extract_numeric_from_sources(
            sources, ("input_tokens", "prompt_tokens"), _parse_int_value
        )
        if input_tokens is not None:
            total_input += input_tokens

        output_tokens = _extract_numeric_from_sources(
            sources, ("output_tokens", "completion_tokens"), _parse_int_value
        )
        if output_tokens is not None:
            total_output += output_tokens

        cost_value = _extract_numeric_from_sources(sources, ("cost",), _parse_float_value)
        if cost_value is not None:
            total_cost += cost_value
            had_cost = True

        records.append(
            {
                "input_tokens": input_tokens if input_tokens is not None else 0,
                "output_tokens": output_tokens if output_tokens is not None else 0,
                "cost_usd": cost_value if cost_value is not None else 0.0,
            }
        )

    if not had_cost:
        total_cost = 0.0
    return total_input, total_output, total_cost, records


def _collect_tool_call_records(messages: list[Any]) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        for call in tool_calls:
            if isinstance(call, Mapping):
                name = call.get("name") or call.get("tool_name") or "<unnamed>"
                args = call.get("args")
                records.append((name, args))
            else:
                records.append((str(call), None))
    return records


def _plot_llm_usage(
    records: list[dict[str, float | int]],
    db_path: Path | str,
    *,
    plot_name: str = "test_backwards_slicing_agent_costs.png",
) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    if df.empty:
        return

    df["input_tokens"] = df["input_tokens"].fillna(0).astype(int)
    df["output_tokens"] = df["output_tokens"].fillna(0).astype(int)
    df["cost_usd"] = df["cost_usd"].fillna(0.0).astype(float)

    df.insert(0, "query_index", range(1, len(df) + 1))
    df["cumulative_cost_usd"] = df["cost_usd"].cumsum()

    fig, ax_tokens = plt.subplots(dpi=200)
    line_in, = ax_tokens.plot(df["query_index"], df["input_tokens"], marker="o", label="Input tokens")
    line_out, = ax_tokens.plot(df["query_index"], df["output_tokens"], marker="o", label="Output tokens")
    ax_tokens.set_xlabel("Query # (corrected order)")
    ax_tokens.set_ylabel("Tokens")

    ax_cost = ax_tokens.twinx()
    line_cost, = ax_cost.plot(
        df["query_index"],
        df["cumulative_cost_usd"],
        marker="o",
        color="green",
        label="Cumulative cost (USD)",
    )
    ax_cost.set_ylabel("Cumulative cost (USD)")

    handles = [line_in, line_out, line_cost]
    labels = [h.get_label() for h in handles]
    ax_tokens.legend(handles, labels, loc="best")

    plt.title("Tokens and cumulative cost per query (corrected order)")
    plt.tight_layout()

    db_path_obj = Path(db_path).resolve()
    plot_path = db_path_obj.with_name(plot_name)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved tokens/cost plot to {plot_path}")


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

    if target_idx is not None:
        if target_idx < 0 or target_idx >= len(entries):
            print(f"target_idx {target_idx} out of range (0-{len(entries) - 1})")
            return
        entry = entries[target_idx]
    else:
        entry = entries[-1]

    header = f"[{entry['thread_id']} {entry['checkpoint_id']}] type[{entry['type']}]"
    print("=========== CHECKPOINT START ===========")
    print(header)

    metadata = entry.get("metadata")
    _print_section("metadata", metadata, indent=2)

    checkpoint = entry.get("checkpoint")
    if not checkpoint:
        print("  (no checkpoint payload)")
        print("============ CHECKPOINT END ============\n")
        return

    checkpoint_extras = {k: v for k, v in checkpoint.items() if k != "channel_values"}
    if checkpoint_extras:
        _print_section("checkpoint", checkpoint_extras, indent=2)

    channel_values = checkpoint.get("channel_values")
    if isinstance(channel_values, Mapping):
        for key, value in channel_values.items():
            if key == "messages":
                continue
            if isinstance(value, Mapping) and "messages" in value:
                continue
            _print_section(f"channel_values.{key}", value, indent=2)

    messages = _collect_channel_messages(channel_values)
    if not messages:
        print("  (no channel messages)")
    else:
        print()
        for idx, msg in enumerate(messages):
            _print_message(msg, idx)


    tool_calls = _collect_tool_call_records(messages)
    if tool_calls:
        print()
        print("Tool call summary:")
        for idx, (name, args) in enumerate(tool_calls, 1):
            header = f"  {idx}. {name}"
            print(header)
            if args is not None:
                args_repr = _stringify_value(args)
                for line in args_repr.splitlines():
                    print("    " + line)
        counts = Counter(name for name, _ in tool_calls)
        total_tool_calls = sum(counts.values())
        if counts and total_tool_calls:
            print()
            print("Tool call counts:")
            for name, count in counts.most_common():
                pct = count / total_tool_calls * 100
                print(f"  {name}: {count} ({pct:.1f}%)")

    input_tokens, output_tokens, total_cost, usage_records = _summarize_llm_usage(messages)
    print()
    print("LLM usage summary:")
    print(f"  input tokens : {input_tokens}")
    print(f"  output tokens: {output_tokens}")
    print(f"  total cost   : ${total_cost:.6f}")

    print()
    _plot_llm_usage(usage_records, db_path)

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
