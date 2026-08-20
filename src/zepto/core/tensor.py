"""Structural graph tensor values and connectivity metadata."""

from dataclasses import dataclass

from .ids import StorageId, TensorId
from .metadata import TensorMetadata
from .ports import PortRef
from .provenance import Provenance


@dataclass(frozen=True, slots=True)
class Tensor:
    """Immutable graph value with connectivity and storage identity metadata."""

    id: TensorId
    metadata: TensorMetadata
    provenance: Provenance | None
    producer: PortRef | None
    consumers: tuple[PortRef, ...]
    storage_id: StorageId | None = None
