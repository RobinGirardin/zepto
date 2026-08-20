"""Immutable records used by operation declarations and estimation."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..metadata import TensorMetadata


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
    GRADIENT = "gradient"


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
    ``saved_for_backward(inputs, outputs)`` method, never declared here.
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

    outputs: tuple[TensorMetadata, ...]
    aliases: tuple[AliasSpec | None, ...] = ()
    saved_for_backward: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject alias declarations whose arity differs from the outputs."""
        if self.aliases and len(self.aliases) != len(self.outputs):
            raise OperationError("Alias declarations must match output arity")


@dataclass(frozen=True, slots=True)
class EstimationContext:
    """Immutable context supplied to operation cost estimators."""

    phase: str = "forward"
    port_metadata: tuple[tuple[str, TensorMetadata], ...] = ()
    dtype: str = "unknown"
    layout: str = "unknown"
    state: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate or invalid port metadata entries."""
        names = [name for name, _ in self.port_metadata]
        if len(names) != len(set(names)):
            raise OperationError("Estimation metadata port names must be unique")
        if not all(
            isinstance(name, str) and name and isinstance(metadata, TensorMetadata)
            for name, metadata in self.port_metadata
        ):
            raise OperationError(
                "Estimation metadata requires named TensorMetadata entries"
            )

    def metadata_for(self, port_name: str) -> TensorMetadata | None:
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
