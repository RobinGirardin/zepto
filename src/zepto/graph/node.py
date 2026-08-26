"""Recorded operation nodes in the structural graph."""

from dataclasses import dataclass
from typing import Mapping

from .ids import EdgeId, NodeId, ParameterId
from zepto.semantic.ports import Port, PortLink
from .provenance import Provenance
from zepto.semantic.operations.base import Operation
from zepto.semantic.operations.records import OperationResult


@dataclass(frozen=True, slots=True)
class Node:
    """One recorded occurrence of a reusable operation declaration."""

    id: NodeId
    operation_family: str
    input_ports: tuple[Port, ...]
    parameter_ports: tuple[Port, ...]
    output_ports: tuple[Port, ...]
    input_edges: tuple[EdgeId, ...]
    parameter_ids: tuple[ParameterId, ...]
    output_edges: tuple[EdgeId, ...]
    provenance: Provenance
    declaration: Operation | None = None
    result: OperationResult | None = None
    saved_for_backward: tuple[PortLink, ...] = ()
    auxiliary_ports: tuple[Port, ...] = ()
    auxiliary_edges: Mapping[str, EdgeId] | None = None

    def __post_init__(self) -> None:
        """Validate the required graph-node operation family name."""
        if not self.operation_family:
            raise ValueError("Operation family cannot be empty")
