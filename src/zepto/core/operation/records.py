"""Immutable records used by operation declarations and estimation."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..accounting import PrecisionPolicy
from ..metadata import ValueMetadata


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
    """Resolved metadata and logical storage semantics for one invocation.

    ``saved_for_backward`` records the invocation-specific selection made by
    the operation's ``saved_for_backward(inputs, outputs)`` method. It must
    be a subset of the names declared by ``BackwardSpec.saved_for_backward``;
    ``Operation.infer_result()`` composes the selection into the result and
    ``Operation.validate_result()`` rejects results that disagree with it.
    The names are resolved to graph-specific ``PortRef`` values when the
    enclosing structural operation is created.
    """

    outputs: tuple[ValueMetadata, ...]
    auxiliary_outputs: tuple[ValueMetadata, ...] = ()
    aliases: tuple[AliasSpec | None, ...] = ()
    auxiliary_aliases: tuple[AliasSpec | None, ...] = ()
    saved_for_backward: tuple[str, ...] = ()
    active_auxiliary_ports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject alias declarations whose arity differs from the outputs."""
        if self.aliases and len(self.aliases) != len(self.outputs):
            raise OperationError("Alias declarations must match output arity")
        if self.auxiliary_aliases and len(self.auxiliary_aliases) != len(
            self.auxiliary_outputs
        ):
            raise OperationError(
                "Auxiliary alias declarations must match auxiliary arity"
            )
        if len(self.active_auxiliary_ports) != len(set(self.active_auxiliary_ports)):
            raise OperationError("active_auxiliary_ports must be unique")


@dataclass(frozen=True, slots=True)
class EstimationContext:
    """Immutable context supplied to operation cost estimators."""

    phase: str = "forward"
    port_metadata: tuple[tuple[str, ValueMetadata], ...] = ()
    precision: PrecisionPolicy | None = None
    state: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate or invalid port metadata entries."""
        names = [name for name, _ in self.port_metadata]
        if len(names) != len(set(names)):
            raise OperationError("Estimation metadata port names must be unique")
        if not all(
            isinstance(name, str) and name and isinstance(metadata, ValueMetadata)
            for name, metadata in self.port_metadata
        ):
            raise OperationError(
                "Estimation metadata requires named ValueMetadata entries"
            )

    def metadata_for(self, port_name: str) -> ValueMetadata | None:
        """Return resolved metadata for a named operation port."""
        for name, metadata in self.port_metadata:
            if name == port_name:
                return metadata
        return None


@dataclass(frozen=True, slots=True)
class ResourceEvent:
    """One ordered logical resource-lifetime event."""

    kind: ResourceEventKind
    value: str
    phase: str = "forward"
    storage: str | None = None
