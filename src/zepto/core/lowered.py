"""Immutable lowered graph records."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .ids import OperationId, TensorId
from .lowering.context import InvocationContext
from .lowering.registry import ImplementationSelection
from .metadata import TensorMetadata
from .operation.records import ResourceEvent


@dataclass(frozen=True, slots=True)
class LoweredTensor:
    """One tensor value in a lowered invocation graph."""

    id: str
    structural_tensor_id: TensorId | None
    metadata: TensorMetadata
    storage_id: str
    workspace: bool = False


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    """One lowered operation with frozen resource events and FLOP counts."""

    id: str
    structural_operation_id: OperationId
    implementation: str
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    auxiliary_tensors: tuple[str, ...]
    resource_events: tuple[ResourceEvent, ...]
    forward_flops: int
    backward_flops: int
    auxiliary_metadata: Mapping[str, TensorMetadata] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    """Immutable lowered graph for one invocation."""

    tensors: Mapping[str, LoweredTensor]
    operations: tuple[LoweredOperation, ...]
    context: InvocationContext
    tensor_map: Mapping[TensorId, str]
    operation_map: Mapping[OperationId, str]
    selections: tuple[ImplementationSelection, ...]
