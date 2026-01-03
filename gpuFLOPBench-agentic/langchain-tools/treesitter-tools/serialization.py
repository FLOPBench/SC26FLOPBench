from __future__ import annotations

from .datatypes import Callsite, NodeRef, OmpRegion


def callsite_to_json(callsite: Callsite) -> dict:
    raise NotImplementedError("TODO: serialize Callsite objects")


def ompreg_to_json(region: OmpRegion) -> dict:
    raise NotImplementedError("TODO: serialize OmpRegion objects")


def noderef_to_json(ref: NodeRef) -> dict:
    raise NotImplementedError("TODO: serialize NodeRef objects")
