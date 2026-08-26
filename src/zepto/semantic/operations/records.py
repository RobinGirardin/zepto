"""Immutable records used by operation declarations and estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from zepto.compose.values import Tensor

if TYPE_CHECKING:
    from zepto.analysis.accounting import PrecisionPolicy
    from zepto.analysis.resolved import ResolvedValue


class OperationError(ValueError):
    """Raised when an operation declaration or result is invalid."""


IncompleteOperationError = OperationError


class ResourceEventKind(StrEnum):
    """Kinds of logical resource-lifetime events."""

    ALLOCATE = "allocate"
    RELEASE = "release"
    ALIAS = "alias"
    SAVE = "save"
    PERSIST = "persist"
    WORKSPACE = "workspace"


class Materialization(StrEnum):
    """Storage strategy associated with an alias declaration."""

    VIEW = "view"
    CONTIGUOUS_COPY = "contiguous_copy"


@dataclass(frozen=True, slots=True)
class AliasSpec:
    """Describe an output's storage relationship to an input port."""

    source_port: str
    materialization: Materialization = Materialization.VIEW


@dataclass(frozen=True, slots=True)
class BackwardSpec:
    """Describe backend-neutral backward and saved-state requirements.

    ``saved_for_backward`` declares the port names an operation is *allowed*
    to save. The invocation-specific subset is selected by the operation's
    ``saved_for_backward(inputs, parameters, outputs)`` method, never declared here.

    For every port name in ``gradient_outputs``, the produced gradient has the
    same shape as the forward tensor bound to that port. Broadcast operands
    imply sum-reduction FLOPs (``backward_flops``) and gradient auxiliary ports
    (``auxiliary_ports``).
    """

    supported: bool = False
    saved_for_backward: tuple[str, ...] = ()
    gradient_inputs: tuple[str, ...] = ()
    gradient_outputs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Logical storage semantics for one invocation.

    ``saved_for_backward`` records the invocation-specific selection made by
    the operation's ``saved_for_backward(inputs, outputs)`` method. It must
    be a subset of the names declared by ``BackwardSpec.saved_for_backward``;
    ``Operation.infer_result()`` composes the selection into the result and
    ``Operation.validate_result()`` rejects results that disagree with it.
    The names are resolved to graph-specific ``PortLink`` values when the
    enclosing node is created.

    Output and auxiliary tensor descriptions live on graph edges, not here.
    """

    aliases: tuple[AliasSpec | None, ...] = ()
    auxiliary_aliases: tuple[AliasSpec | None, ...] = ()
    saved_for_backward: tuple[str, ...] = ()
    active_auxiliary_ports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.active_auxiliary_ports) != len(set(self.active_auxiliary_ports)):
            raise OperationError("active_auxiliary_ports must be unique")


@dataclass(frozen=True, slots=True)
class InferenceBundle:
    """Transient inference products: tensors plus semantics-only result."""

    outputs: tuple[Tensor, ...]
    auxiliary_outputs: tuple[Tensor, ...]
    result: OperationResult


@dataclass(frozen=True, slots=True)
class EstimationContext:
    """Immutable context supplied to operation cost estimators."""

    phase: str = "forward"
    port_values: tuple[tuple[str, ResolvedValue], ...] = ()
    precision: PrecisionPolicy | None = None
    state: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        from zepto.analysis.resolved import ResolvedValue

        names = [name for name, _ in self.port_values]
        if len(names) != len(set(names)):
            raise OperationError("Estimation port names must be unique")
        if not all(
            isinstance(name, str) and name and isinstance(value, ResolvedValue)
            for name, value in self.port_values
        ):
            raise OperationError(
                "Estimation context requires named ResolvedValue entries"
            )

    def value_for(self, port_name: str) -> ResolvedValue | None:
        """Return the resolved accounting value for a named operation port."""
        for name, value in self.port_values:
            if name == port_name:
                return value
        return None

    def tensor_for(self, port_name: str) -> Tensor | None:
        """Return the structural tensor for a named operation port."""
        value = self.value_for(port_name)
        return None if value is None else value.tensor


@dataclass(frozen=True, slots=True)
class ResourceEvent:
    """One ordered logical resource-lifetime event."""

    kind: ResourceEventKind
    value: str
    phase: str = "forward"
    storage: str | None = None
