"""Command-line helper that reuses :mod:`helper-scripts.sqlite_reader` to print the stored messages."""

from __future__ import annotations

import argparse
from pathlib import Path

import sqlite_reader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print every message stored inside a LangGraph SqliteSaver checkpoint."
    )
    parser.add_argument(
        "sqlite_file",
        type=Path,
        help="Path to the sqlite checkpoint file (e.g., unit-tests/test_backwards_slicing_with_llm_checkpoint.sqlite).",
    )
    parser.add_argument(
        "--thread-id",
        "-t",
        type=str,
        default=None,
        help="Restrict output to a single LangGraph thread identifier.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of checkpoints to display per thread.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sqlite_reader.print_checkpoint_messages(
        args.sqlite_file,
        thread_id=args.thread_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
