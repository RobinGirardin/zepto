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
class DimensionScope:
    """Namespace that prevents symbolic dimensions from colliding."""

    value: UUID

    @classmethod
    def new(cls) -> "DimensionScope":
        """Generate a new symbolic-dimension namespace."""
        return cls(uuid4())


@dataclass(frozen=True, slots=True)
class TensorId:
    """Graph-owned identity for one semantic tensor value."""

    graph_id: GraphId
    index: int


@dataclass(frozen=True, slots=True)
class ParameterId:
    """Graph-owned identity for one shared parameter declaration."""

    graph_id: GraphId
    index: int


@dataclass(frozen=True, slots=True)
class OperationId:
    """Graph-owned identity for one structural operation."""

    graph_id: GraphId
    index: int


@dataclass(frozen=True, slots=True)
class StorageId:
    """Graph-owned identity for future physical storage relationships."""

    graph_id: GraphId
    index: int
