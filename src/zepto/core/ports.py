"""Operation port declarations and concrete port references."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TYPE_CHECKING

from .ids import OperationId
from .metadata import ValueMetadata

if TYPE_CHECKING:
    from .operation import StructuralOperation


class ValueKind(StrEnum):
    """Semantic category carried by a structural operation port."""

    TENSOR = "tensor"
    STATE = "state"
    PARAMETER = "parameter"
    GRADIENT = "gradient"


@dataclass(frozen=True, slots=True)
class PortSpec:
    """Direction-neutral declaration of an operation port."""

    name: str
    value_kind: ValueKind = ValueKind.TENSOR
    metadata: ValueMetadata | None = None

    def __post_init__(self) -> None:
        """Reject port declarations without a name."""
        if not self.name:
            raise ValueError("Port name cannot be empty")


@dataclass(frozen=True, slots=True)
class PortRef:
    """Concrete reference to a named port on a structural operation."""

    operation_id: OperationId
    port_name: str
    role: Literal["input", "output", "auxiliary", "parameter"]

    def resolve(self, operation: "StructuralOperation") -> PortSpec:
        """Resolve this reference against its owning operation.

        Args:
            operation: Operation whose declared ports should be inspected.

        Returns:
            The matching port declaration.

        Raises:
            ValueError: If the operation belongs to another identity.
            KeyError: If the named port does not exist.
        """
        if operation.id != self.operation_id:
            raise ValueError("PortRef belongs to a different operation")
        if self.role == "input":
            ports = operation.input_ports
        elif self.role == "parameter":
            ports = operation.parameter_ports
        elif self.role == "output":
            ports = operation.output_ports
        else:
            ports = operation.auxiliary_ports
        for port in ports:
            if port.name == self.port_name:
                return port
        raise KeyError(f"Unknown {self.role} port {self.port_name!r}")
