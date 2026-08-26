"""Immutable graph storage and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import *
from .ids import *
from .provenance import Provenance

if TYPE_CHECKING:
    from .edge import Edge
    from .graph import Graph, GraphBuilder
    from .node import Node
    from .validation import GraphValidator

__all__ = [
    "Edge",
    "Graph",
    "GraphBuilder",
    "GraphId",
    "GraphValidator",
    "Node",
    "ParameterId",
    "Provenance",
    "StorageId",
    "EdgeId",
    "NodeId",
]

_LAZY = {
    "Edge": (".edge", "Edge"),
    "Node": (".node", "Node"),
    "Graph": (".graph", "Graph"),
    "GraphBuilder": (".graph", "GraphBuilder"),
    "GraphValidator": (".validation", "GraphValidator"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        import importlib

        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
