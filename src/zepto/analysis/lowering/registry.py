"""Implementation registry and selection records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from zepto.graph.ids import NodeId
from zepto.graph.node import Node
from .errors import LoweringError
from .region import PatternMatchRule, ProvenanceMatchRule, Region

if TYPE_CHECKING:
    from zepto.graph.graph import Graph
    from zepto.semantic.operations.records import EstimationContext
    from ..lowered import LoweredNode
    from .context import InvocationContext
    from .helpers import RegionEstimationContext


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


@dataclass(frozen=True, slots=True)
class RegionImplementationDescriptor:
    """Stable identity and discovery metadata for one region implementation."""

    id: str
    kind: str
    priority: int = 0
    capabilities: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    provenance_rule: ProvenanceMatchRule | None = None
    pattern_rule: PatternMatchRule | None = None


@dataclass(frozen=True, slots=True)
class RegionImplementationSelection:
    """Record of which region implementation was chosen."""

    region_id: str
    node_ids: tuple[NodeId, ...]
    chosen: RegionImplementationDescriptor
    rejected: tuple[tuple[RegionImplementationDescriptor, str], ...]
    reason: str


@runtime_checkable
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


@runtime_checkable
class RegionImplementation(Protocol):
    """Concrete execution strategy for one region."""

    @property
    def descriptor(self) -> RegionImplementationDescriptor: ...

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        """Return None if compatible, else a rejection reason."""

    def lower(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
        edge_map: dict,
        *,
        estimation: RegionEstimationContext,
        lowered_edges: dict,
    ) -> LoweredNode:
        """Produce one lowered node for a region."""


class LoweringRegistry:
    """Registry of implementations keyed by family, id, and region kind."""

    def __init__(self) -> None:
        self._by_id: dict[str, Implementation | RegionImplementation] = {}
        self._by_family: dict[str, list[Implementation]] = {}
        self._by_region_kind: dict[str, list[RegionImplementation]] = {}
        self._region_descriptors: dict[str, RegionImplementationDescriptor] = {}

    def register(self, implementation: Implementation) -> None:
        descriptor = implementation.descriptor
        if descriptor.id in self._by_id:
            raise ValueError(f"Duplicate implementation id {descriptor.id!r}")
        self._by_id[descriptor.id] = implementation
        self._by_family.setdefault(descriptor.family, []).append(implementation)

    def register_region(self, implementation: RegionImplementation) -> None:
        descriptor = implementation.descriptor
        if descriptor.id in self._by_id:
            raise ValueError(f"Duplicate implementation id {descriptor.id!r}")
        self._by_id[descriptor.id] = implementation
        self._by_region_kind.setdefault(descriptor.kind, []).append(implementation)
        self._region_descriptors[descriptor.kind] = descriptor

    def get(self, impl_id: str) -> Implementation | RegionImplementation | None:
        return self._by_id.get(impl_id)

    def candidates(self, family: str) -> tuple[Implementation, ...]:
        return tuple(self._by_family.get(family, ()))

    def region_candidates(self, kind: str) -> tuple[RegionImplementation, ...]:
        return tuple(self._by_region_kind.get(kind, ()))

    def registered_region_kinds(self) -> tuple[str, ...]:
        return tuple(self._region_descriptors.keys())

    def region_descriptor(self, kind: str) -> RegionImplementationDescriptor:
        return self._region_descriptors[kind]


def _filter_region_candidates(
    region: Region,
    candidates: tuple[RegionImplementation, ...],
    graph: Graph,
    context: InvocationContext,
) -> tuple[list[RegionImplementation], list[tuple[RegionImplementationDescriptor, str]]]:
    rejected: list[tuple[RegionImplementationDescriptor, str]] = []
    viable: list[RegionImplementation] = []
    for impl in candidates:
        if context.requested_capabilities:
            if not context.requested_capabilities <= impl.descriptor.capabilities:
                rejected.append(
                    (
                        impl.descriptor,
                        f"missing capabilities {context.requested_capabilities - impl.descriptor.capabilities}",
                    )
                )
                continue
        reason = impl.compatible(region, graph, context)
        if reason is None:
            viable.append(impl)
        else:
            rejected.append((impl.descriptor, reason))
    return viable, rejected


def region_has_compatible_implementation(
    region: Region,
    registry: LoweringRegistry,
    context: InvocationContext,
    *,
    graph: Graph | None = None,
) -> bool:
    """Return whether at least one region implementation can lower this region."""
    candidates = registry.region_candidates(region.kind)
    if not candidates:
        return False
    if graph is None:
        return True
    viable, _rejected = _filter_region_candidates(
        region, candidates, graph, context
    )
    return bool(viable)


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
        if impl is None or not isinstance(impl.descriptor, ImplementationDescriptor):
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


def select_region_implementation(
    region: Region,
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[RegionImplementation, RegionImplementationSelection]:
    """Select a region implementation for one region."""
    candidates = registry.region_candidates(region.kind)
    if not candidates:
        raise LoweringError(
            f"No region implementations registered for kind {region.kind!r}"
        )

    if region.kind in context.region_implementation_pins:
        pin = context.region_implementation_pins[region.kind]
        impl = registry.get(pin)
        if impl is None or not isinstance(
            impl.descriptor, RegionImplementationDescriptor
        ):
            raise LoweringError(f"Pinned region implementation {pin!r} not registered")
        reason = impl.compatible(region, graph, context)
        if reason is not None:
            raise LoweringError(f"Pinned {pin!r} incompatible: {reason}")
        return impl, RegionImplementationSelection(
            region.id,
            region.operation_ids,
            impl.descriptor,
            (),
            "region_pinned",
        )

    component_type = region.anchor.component_type
    if component_type and component_type in context.module_implementation_pins:
        pin = context.module_implementation_pins[component_type]
        impl = registry.get(pin)
        if impl is None or not isinstance(
            impl.descriptor, RegionImplementationDescriptor
        ):
            raise LoweringError(f"Pinned module implementation {pin!r} not registered")
        reason = impl.compatible(region, graph, context)
        if reason is not None:
            raise LoweringError(f"Pinned {pin!r} incompatible: {reason}")
        return impl, RegionImplementationSelection(
            region.id,
            region.operation_ids,
            impl.descriptor,
            (),
            "module_pinned",
        )

    viable, rejected = _filter_region_candidates(
        region, candidates, graph, context
    )
    if not viable:
        raise LoweringError(
            f"No compatible region implementation for {region.kind!r}; "
            f"rejected: {rejected}"
        )

    viable.sort(key=lambda item: (-item.descriptor.priority, item.descriptor.id))
    chosen = viable[0]
    return chosen, RegionImplementationSelection(
        region.id,
        region.operation_ids,
        chosen.descriptor,
        tuple(rejected),
        "highest_priority" if len(viable) > 1 else "only_candidate",
    )
