"""Identity lowering implementations delegating to structural operations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...graph import StructuralGraph
from ...lowered import LoweredOperation
from ...operation.base import Operation
from ...operation.records import EstimationContext
from ...operation.structural import StructuralOperation
from ..context import InvocationContext
from ..helpers import (
    append_save_events,
    ensure_lowered_tensor,
    remap_events,
)
from ..registry import ImplementationDescriptor


@dataclass(frozen=True, slots=True)
class IdentityImplementation:
    """Wrap one structural Operation as a 1:1 lowered implementation."""

    operation: Operation
    descriptor: ImplementationDescriptor

    def compatible(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        del graph, context
        if structural.operation_family != self.descriptor.family:
            return "family mismatch"
        return None

    def lower(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
        tensor_map: dict,
        *,
        estimation: EstimationContext,
        lowered_tensors: dict,
    ) -> LoweredOperation:
        op = structural.declaration
        if op is None:
            op = self.operation
        result = structural.result
        if result is None:
            raise ValueError(f"Operation {structural.id} has no result")

        for tensor_id in structural.input_tensors:
            ensure_lowered_tensor(
                tensor_id, graph, context, lowered_tensors, tensor_map
            )
        for tensor_id in structural.output_tensors:
            ensure_lowered_tensor(
                tensor_id, graph, context, lowered_tensors, tensor_map
            )
        if structural.auxiliary_tensors is not None:
            for tensor_id in structural.auxiliary_tensors.values():
                ensure_lowered_tensor(
                    tensor_id, graph, context, lowered_tensors, tensor_map
                )

        events = op.resource_events(estimation, result)
        forward = op.forward_flops(estimation)
        backward = op.backward_flops(replace(estimation, phase="backward"))

        if not isinstance(forward, int) or forward < 0:
            raise ValueError(f"{op.family}: forward_flops must be non-negative int")
        if not isinstance(backward, int) or backward < 0:
            raise ValueError(f"{op.family}: backward_flops must be non-negative int")

        remapped = remap_events(events, structural, tensor_map)
        with_saves = append_save_events(remapped, structural, tensor_map)

        input_ids = tuple(tensor_map[tid] for tid in structural.input_tensors)
        output_ids = tuple(tensor_map[tid] for tid in structural.output_tensors)
        auxiliary_ids = ()
        if structural.auxiliary_tensors is not None:
            auxiliary_ids = tuple(
                tensor_map[tid] for tid in structural.auxiliary_tensors.values()
            )

        return LoweredOperation(
            id=f"op{structural.id.index}",
            structural_operation_ids=(structural.id,),
            implementation=self.descriptor.id,
            input_tensors=input_ids,
            output_tensors=output_ids,
            auxiliary_tensors=auxiliary_ids,
            resource_events=with_saves,
            forward_flops=forward,
            backward_flops=backward,
        )
