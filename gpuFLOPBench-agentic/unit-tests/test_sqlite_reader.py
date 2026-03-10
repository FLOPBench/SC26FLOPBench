from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_reader_module():
    path = Path("helper-scripts/sqlite_reader.py")
    spec = importlib.util.spec_from_file_location("sqlite_reader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_sqlite_reader_can_list_threads():
    reader = _load_reader_module()
    db_path = Path("unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite")
    assert db_path.exists()

    thread_ids = reader.get_thread_ids_from_sqlite(db_path)
    assert thread_ids, "Expected at least one thread_id recorded in the checkpoint DB"

    success_threads = reader.get_thread_ids_from_sqlite(db_path, success_only=True)
    assert isinstance(success_threads, list)
    assert all(thread in thread_ids for thread in success_threads)


def test_sqlite_reader_iterates_messages_and_prints():
    reader = _load_reader_module()
    db_path = Path("unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite")
    messages = list(reader.iter_checkpoint_messages(db_path, limit=3))
    assert messages, "Expected the checkpoint DB to contain messages"

    print("checkpoint messages sample:")
    for entry in messages:
        print(f"- checkpoint {entry['checkpoint_id']} message[{entry['index']}]: {entry['message'].get('content')}")

    reader.print_checkpoint_messages(db_path, limit=2)


def test_sqlite_reader_can_show_writes():
    reader = _load_reader_module()
    db_path = Path("unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite")
    writes = list(reader.iter_writes(db_path))
    print(f"pending writes count: {len(writes)}")
    reader.print_pending_writes(db_path)
    assert isinstance(writes, list)
