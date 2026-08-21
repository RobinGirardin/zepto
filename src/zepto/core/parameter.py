"""Structural graph parameter declarations."""

from dataclasses import dataclass

from .ids import ParameterId
from .metadata import TensorMetadata


@dataclass(frozen=True, slots=True)
class Parameter:
    """Immutable shared parameter declaration in a structural graph."""

    id: ParameterId
    metadata: TensorMetadata
    trainable: bool = True
