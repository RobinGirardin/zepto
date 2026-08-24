"""Structural graph parameter declarations."""

from dataclasses import dataclass, replace

from .ids import ParameterId
from .metadata import TensorRole, ValueMetadata


@dataclass(frozen=True, slots=True)
class Parameter:
    """Immutable shared parameter declaration in a structural graph."""

    id: ParameterId
    metadata: ValueMetadata
    trainable: bool = True


def bound_parameter_metadata(parameter: Parameter) -> ValueMetadata:
    """Return metadata for inference and estimation at a parameter port."""
    return replace(
        parameter.metadata,
        requires_grad=parameter.trainable,
        role=TensorRole.PARAMETER,
        persistent=True,
    )
