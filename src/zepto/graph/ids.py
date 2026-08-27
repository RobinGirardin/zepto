"""Stable graph, edge, node, and storage identities."""

from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class GraphId:
    """Stable identity for one structural graph."""

    value: UUID

    @classmethod
    def new(cls) -> "GraphId":
        """Generate a new graph identity."""
        return cls(uuid4())


@dataclass(frozen=True, slots=True)
class EdgeId:
    """Graph-owned identity for one data-flow edge."""

    graph_id: GraphId
    index: int


@dataclass(frozen=True, slots=True)
class ParameterId:
    """Graph-owned identity for one shared parameter declaration."""

    graph_id: GraphId
    index: int


@dataclass(frozen=True, slots=True)
class NodeId:
    """Graph-owned identity for one recorded operation node."""

    graph_id: GraphId
    index: int


@dataclass(frozen=True, slots=True)
class StorageId:
    """Graph-owned identity for future physical storage relationships."""

    graph_id: GraphId
    index: int
