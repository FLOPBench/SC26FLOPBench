from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Span:
    file: str
    start_byte: int
    end_byte: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]


@dataclass(frozen=True)
class NodeRef:
    type: str
    start_byte: int
    end_byte: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]


@dataclass(frozen=True)
class Callsite:
    kind: str
    span: Span
    callee_text: str
    receiver_text: Optional[str] = None
    template_args_text: Optional[str] = None
    arg_span: Optional[Span] = None
    launch_cfg_span: Optional[Span] = None


@dataclass(frozen=True)
class OmpRegion:
    pragma_span: Span
    pragma_text: str
    associated_stmt_span: Span
    kind_text: str
    clauses: List[str]
