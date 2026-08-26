"""Immutable lowered graph records."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .ids import OperationId, TensorId
from .lowering.context import InvocationContext
from .lowering.registry import (
    ImplementationSelection,
    RegionImplementationSelection,
)
from .metadata import ValueMetadata
from .operation.records import ResourceEvent


@dataclass(frozen=True, slots=True)
class LoweredTensor:
    """One tensor value in a lowered invocation graph."""

    id: str
    structural_tensor_id: TensorId | None
    metadata: ValueMetadata
    storage_id: str
    workspace: bool = False


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    """One lowered operation with frozen resource events and FLOP counts."""

    id: str
    structural_operation_ids: tuple[OperationId, ...]
    implementation: str
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    auxiliary_tensors: tuple[str, ...]
    resource_events: tuple[ResourceEvent, ...]
    forward_flops: int
    backward_flops: int
    module_path: tuple[str, ...] = ()
    component_type: str | None = None
    region_id: str | None = None
    auxiliary_metadata: Mapping[str, ValueMetadata] = MappingProxyType({})

    @property
    def structural_operation_id(self) -> OperationId:
        """Backward-compatible single-op accessor."""
        return self.structural_operation_ids[0]


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    """Immutable lowered graph for one invocation."""

    tensors: Mapping[str, LoweredTensor]
    operations: tuple[LoweredOperation, ...]
    context: InvocationContext
    tensor_map: Mapping[TensorId, str]
    operation_map: Mapping[OperationId, str]
    selections: tuple[ImplementationSelection, ...]
    fusion_map: Mapping[OperationId, str] = MappingProxyType({})
    region_map: Mapping[str, tuple[OperationId, ...]] = MappingProxyType({})
    region_selections: tuple[RegionImplementationSelection, ...] = ()
