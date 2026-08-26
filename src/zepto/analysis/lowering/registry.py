"""Implementation registry and selection records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from zepto.graph.ids import NodeId
from zepto.graph.node import Node
from .errors import LoweringError

if TYPE_CHECKING:
    from zepto.graph.graph import Graph
    from zepto.semantic.operations.records import EstimationContext
    from ..lowered import LoweredNode
    from .context import InvocationContext


@dataclass(frozen=True, slots=True)
class ImplementationDescriptor:
    """Stable identity and selection metadata for one implementation."""

    id: str
    family: str
    priority: int = 0
    capabilities: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ImplementationSelection:
    """Record of which implementation was chosen for one graph node."""

    node_id: NodeId
    chosen: ImplementationDescriptor
    rejected: tuple[tuple[ImplementationDescriptor, str], ...]
    reason: str


class Implementation(Protocol):
    """Concrete execution strategy selected at lowering time."""

    @property
    def descriptor(self) -> ImplementationDescriptor: ...

    def compatible(
        self,
        node: Node,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        """Return None if compatible, else a rejection reason."""

    def lower(
        self,
        node: Node,
        graph: Graph,
        context: InvocationContext,
        edge_map: dict,
        *,
        estimation: EstimationContext,
        lowered_edges: dict,
    ) -> LoweredNode:
        """Produce one lowered node for a graph node."""


class LoweringRegistry:
    """Registry of implementations keyed by family and id."""

    def __init__(self) -> None:
        self._by_id: dict[str, Implementation] = {}
        self._by_family: dict[str, list[Implementation]] = {}

    def register(self, implementation: Implementation) -> None:
        descriptor = implementation.descriptor
        if descriptor.id in self._by_id:
            raise ValueError(f"Duplicate implementation id {descriptor.id!r}")
        self._by_id[descriptor.id] = implementation
        self._by_family.setdefault(descriptor.family, []).append(implementation)

    def get(self, impl_id: str) -> Implementation | None:
        return self._by_id.get(impl_id)

    def candidates(self, family: str) -> tuple[Implementation, ...]:
        return tuple(self._by_family.get(family, ()))


def select_implementation(
    node: Node,
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[Implementation, ImplementationSelection]:
    """Select an implementation for one graph node."""
    family = node.operation_family
    candidates = registry.candidates(family)

    if not candidates:
        raise LoweringError(f"No implementations registered for family {family!r}")

    rejected: list[tuple[ImplementationDescriptor, str]] = []

    if family in context.implementation_pins:
        pin = context.implementation_pins[family]
        impl = registry.get(pin)
        if impl is None:
            raise LoweringError(f"Pinned implementation {pin!r} not registered")
        reason = impl.compatible(node, graph, context)
        if reason is not None:
            raise LoweringError(f"Pinned {pin!r} incompatible: {reason}")
        return impl, ImplementationSelection(
            node.id, impl.descriptor, (), "pinned"
        )

    viable: list[Implementation] = []
    for impl in candidates:
        reason = impl.compatible(node, graph, context)
        if reason is None:
            viable.append(impl)
        else:
            rejected.append((impl.descriptor, reason))

    if not viable:
        raise LoweringError(
            f"No compatible implementation for {family!r}; rejected: {rejected}"
        )

    viable.sort(key=lambda item: (-item.descriptor.priority, item.descriptor.id))
    chosen = viable[0]

    return chosen, ImplementationSelection(
        node.id,
        chosen.descriptor,
        tuple(rejected),
        "highest_priority" if len(viable) > 1 else "only_candidate",
    )
