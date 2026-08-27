"""Operation port declarations and concrete port references."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TYPE_CHECKING

from zepto.graph.ids import NodeId
from .metadata import PortContract

if TYPE_CHECKING:
    from zepto.graph.node import Node


class ValueKind(StrEnum):
    """Semantic category carried by a structural operation port."""

    TENSOR = "tensor"
    STATE = "state"
    PARAMETER = "parameter"
    GRADIENT = "gradient"


@dataclass(frozen=True, slots=True)
class Port:
    """Direction-neutral declaration of an operation port."""

    name: str
    value_kind: ValueKind = ValueKind.TENSOR
    contract: PortContract | None = None

    def __post_init__(self) -> None:
        """Reject port declarations without a name."""
        if not self.name:
            raise ValueError("Port name cannot be empty")


@dataclass(frozen=True, slots=True)
class PortLink:
    """Concrete reference to a named port on a structural operation."""

    node_id: NodeId
    port_name: str
    role: Literal["input", "output", "auxiliary", "parameter"]

    def resolve(self, node: "Node") -> Port:
        """Resolve this reference against its owning node.

        Args:
            node: Node whose declared ports should be inspected.

        Returns:
            The matching port declaration.

        Raises:
            ValueError: If the node belongs to another identity.
            KeyError: If the named port does not exist.
        """
        if node.id != self.node_id:
            raise ValueError("PortLink belongs to a different node")
        if self.role == "input":
            ports = node.input_ports
        elif self.role == "parameter":
            ports = node.parameter_ports
        elif self.role == "output":
            ports = node.output_ports
        else:
            ports = node.auxiliary_ports
        for port in ports:
            if port.name == self.port_name:
                return port
        raise KeyError(f"Unknown {self.role} port {self.port_name!r}")
