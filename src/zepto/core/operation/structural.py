"""Graph-bound occurrences of reusable operation declarations."""

from dataclasses import dataclass

from ..ids import OperationId, ParameterId, TensorId
from ..ports import PortRef, PortSpec
from ..provenance import Provenance
from .base import Operation
from .records import OperationResult


@dataclass(frozen=True, slots=True)
class StructuralOperation:
    """One graph occurrence of a reusable operation declaration."""

    id: OperationId
    operation_family: str
    input_ports: tuple[PortSpec, ...]
    output_ports: tuple[PortSpec, ...]
    input_tensors: tuple[TensorId, ...]
    output_tensors: tuple[TensorId, ...]
    parameter_ids: tuple[ParameterId, ...]
    provenance: Provenance
    declaration: Operation | None = None
    result: OperationResult | None = None
    saved_for_backward: tuple[PortRef, ...] = ()

    def __post_init__(self) -> None:
        """Validate the required graph-node operation family name."""
        if not self.operation_family:
            raise ValueError("Operation family cannot be empty")
