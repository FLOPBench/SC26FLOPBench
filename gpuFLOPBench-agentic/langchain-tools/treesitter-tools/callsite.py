from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .datatypes import Callsite, NodeRef


def collect_callsites_in_function(path: Union[str, Path], text: str, func_ref: NodeRef) -> List[Callsite]:
    raise NotImplementedError("TODO: collect callsites within a function")


def summarize_def_use(
    path: Union[str, Path],
    text: str,
    func_ref: NodeRef,
    global_names: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    raise NotImplementedError("TODO: summarize reads/writes/escapes for a function")


def find_assignments_in_function(path: Union[str, Path], text: str, func_ref: NodeRef, name: str) -> Set[str]:
    raise NotImplementedError("TODO: find assignments to a named identifier within a function")
