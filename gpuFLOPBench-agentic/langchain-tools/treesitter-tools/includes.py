from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from .datatypes import Span


def collect_includes(text: str) -> List[Tuple[str, Span]]:
    raise NotImplementedError("TODO: lexically collect include directives")


def resolve_include_to_path(
    includer_file: Union[str, Path],
    raw_include: str,
    include_paths: Iterable[Union[str, Path]],
    project_root: Union[str, Path],
) -> Optional[str]:
    raise NotImplementedError("TODO: resolve include paths relative to include directories")


def is_project_path(path: Union[str, Path], project_root: Union[str, Path]) -> bool:
    raise NotImplementedError("TODO: determine whether a path lives inside the project root")
