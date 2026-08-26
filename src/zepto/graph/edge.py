"""Immutable graph edges carrying frozen composition tensor values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zepto.compose.values import Tensor
from .ids import EdgeId, StorageId
from .provenance import Provenance

if TYPE_CHECKING:
    from zepto.semantic.ports import PortLink


@dataclass(frozen=True, slots=True)
class Edge:
    """Immutable graph connection with a frozen tensor description."""

    id: EdgeId
    tensor: Tensor
    provenance: Provenance | None
    producer: PortLink | None
    consumers: tuple[PortLink, ...]
    storage_id: StorageId | None = None
