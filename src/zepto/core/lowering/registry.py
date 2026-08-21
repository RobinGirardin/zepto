"""Implementation registry and selection records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..ids import OperationId
from ..operation.structural import StructuralOperation
from .errors import LoweringError

if TYPE_CHECKING:
    from ..graph import StructuralGraph
    from ..operation.records import EstimationContext
    from ..lowered import LoweredOperation
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
    """Record of which implementation was chosen for one structural operation."""

    structural_operation_id: OperationId
    chosen: ImplementationDescriptor
    rejected: tuple[tuple[ImplementationDescriptor, str], ...]
    reason: str


class Implementation(Protocol):
    """Concrete execution strategy selected at lowering time."""

    @property
    def descriptor(self) -> ImplementationDescriptor: ...

    def compatible(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        """Return None if compatible, else a rejection reason."""

    def lower(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
        tensor_map: dict,
        *,
        estimation: EstimationContext,
        lowered_tensors: dict,
    ) -> LoweredOperation:
        """Produce one lowered operation for a structural node."""


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
    structural: StructuralOperation,
    graph: StructuralGraph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[Implementation, ImplementationSelection]:
    """Select an implementation for one structural operation."""
    family = structural.operation_family
    candidates = registry.candidates(family)

    if not candidates:
        raise LoweringError(f"No implementations registered for family {family!r}")

    rejected: list[tuple[ImplementationDescriptor, str]] = []

    if family in context.implementation_pins:
        pin = context.implementation_pins[family]
        impl = registry.get(pin)
        if impl is None:
            raise LoweringError(f"Pinned implementation {pin!r} not registered")
        reason = impl.compatible(structural, graph, context)
        if reason is not None:
            raise LoweringError(f"Pinned {pin!r} incompatible: {reason}")
        return impl, ImplementationSelection(
            structural.id, impl.descriptor, (), "pinned"
        )

    viable: list[Implementation] = []
    for impl in candidates:
        reason = impl.compatible(structural, graph, context)
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
        structural.id,
        chosen.descriptor,
        tuple(rejected),
        "highest_priority" if len(viable) > 1 else "only_candidate",
    )
