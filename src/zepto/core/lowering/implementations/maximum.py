"""Specialized maximum-family lowering implementations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from ...operation.helpers import numel
from ...graph import StructuralGraph
from ...lowered import LoweredOperation
from ...metadata import DType, TensorRole
from ...operation.records import EstimationContext, ResourceEvent, ResourceEventKind
from ...operation.structural import StructuralOperation
from ..context import InvocationContext
from ..helpers import (
    ensure_lowered_tensor,
    is_zero_operand,
    register_auxiliary_tensor,
    resolve_tensor_metadata,
)
from ..registry import ImplementationDescriptor


@dataclass(frozen=True, slots=True)
class ReLUMaskImplementation:
    """Lower maximum(left, 0) with a retained forward mask for backward."""

    descriptor: ImplementationDescriptor

    def compatible(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        if structural.operation_family != "maximum":
            return "not maximum"

        if len(structural.input_tensors) != 2:
            return "maximum expects two inputs"
        _left_id, right_id = structural.input_tensors
        right = graph.tensor(right_id)
        if not is_zero_operand(right, graph):
            return "right operand is not provably zero"
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
        left_id, _right_id = structural.input_tensors
        output_id = structural.output_tensors[0]

        for tensor_id in (left_id, output_id):
            ensure_lowered_tensor(
                tensor_id, graph, context, lowered_tensors, tensor_map
            )

        left_meta = estimation.metadata_for("left")
        output_meta = estimation.metadata_for("output")
        if left_meta is None or output_meta is None:
            raise ValueError("ReLU lowering requires left and output port metadata")

        n = numel(output_meta)
        lowered_out = tensor_map[output_id]
        lowered_mask = f"op{structural.id.index}:mask"

        mask_meta = resolve_tensor_metadata(
            replace(
                output_meta,
                semantic_type="relu_mask",
                dtype=DType.BOOL,
                requires_grad=False,
                persistent=False,
            ),
            context,
            role=TensorRole.AUXILIARY,
        )
        register_auxiliary_tensor(
            lowered_mask,
            mask_meta,
            storage_id=lowered_mask,
            lowered_tensors=lowered_tensors,
        )

        forward_flops = n
        backward_flops = n if left_meta.requires_grad else 0

        events: tuple[ResourceEvent, ...] = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_mask),
            ResourceEvent(ResourceEventKind.SAVE, lowered_mask),
        )
        if context.phase == "backward":
            events = (
                *events,
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    lowered_mask,
                    phase="backward",
                ),
            )

        return LoweredOperation(
            id=f"op{structural.id.index}",
            structural_operation_id=structural.id,
            implementation=self.descriptor.id,
            input_tensors=(tensor_map[left_id],),
            output_tensors=(lowered_out,),
            auxiliary_tensors=(lowered_mask,),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            auxiliary_metadata=MappingProxyType({lowered_mask: mask_meta}),
        )


RELU_MASK = ReLUMaskImplementation(
    descriptor=ImplementationDescriptor(
        id="maximum/relu-mask",
        family="maximum",
        priority=10,
        capabilities=frozenset({"relu", "mask_retention"}),
    ),
)
