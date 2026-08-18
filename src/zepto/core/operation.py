from dataclasses import dataclass
from typing import Any

from .ids import OperationId, ParameterId, TensorId
from .ports import PortSpec
from .provenance import Provenance


@dataclass(frozen=True, slots=True)
class StructuralOperation:
    """Immutable semantic operation node in a structural graph."""

    id: OperationId
    operation_family: str
    input_ports: tuple[PortSpec, ...]
    output_ports: tuple[PortSpec, ...]
    input_tensors: tuple[TensorId, ...]
    output_tensors: tuple[TensorId, ...]
    parameter_ids: tuple[ParameterId, ...]
    provenance: Provenance
    declaration: Any | None = None

    def __post_init__(self) -> None:
        if not self.operation_family:
            raise ValueError("Operation family cannot be empty")
