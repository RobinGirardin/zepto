"""ReLU region lowering via pattern discovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from ....graph import StructuralGraph
from ....lowered import LoweredOperation
from ....metadata import DType, TensorRole
from ....operation.helpers import numel
from ....operation.records import ResourceEvent, ResourceEventKind
from ...context import InvocationContext
from ...helpers import (
    RegionEstimationContext,
    ensure_lowered_tensor,
    register_auxiliary_tensor,
    resolve_tensor_metadata,
)
from ...region import PatternConstraint, PatternMatchRule, StructuralRegion
from ...registry import RegionImplementationDescriptor


@dataclass(frozen=True, slots=True)
class ReLURegionImplementation:
    """Lower maximum(left, 0) as one fused region with mask retention."""

    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: StructuralRegion,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if len(region.operation_ids) != 1:
            return "ReLU region expects one operation"
        structural = graph.operation(region.operation_ids[0])
        if structural.operation_family != "maximum":
            return "not maximum"
        if len(structural.input_tensors) != 2:
            return "maximum expects two inputs"
        return None

    def lower(
        self,
        region: StructuralRegion,
        graph: StructuralGraph,
        context: InvocationContext,
        tensor_map: dict,
        *,
        estimation: RegionEstimationContext,
        lowered_tensors: dict,
    ) -> LoweredOperation:
        structural = graph.operation(region.operation_ids[0])
        left_id, _right_id = structural.input_tensors
        output_id = structural.output_tensors[0]

        for tensor_id in (left_id, output_id):
            ensure_lowered_tensor(
                tensor_id, graph, context, lowered_tensors, tensor_map
            )

        left_meta = estimation.input_metadata.get("input")
        output_meta = estimation.output_metadata.get("output")
        if left_meta is None:
            left_meta = resolve_tensor_metadata(
                graph.tensor(left_id).metadata, context
            )
        if output_meta is None:
            output_meta = resolve_tensor_metadata(
                graph.tensor(output_id).metadata, context
            )

        n = numel(output_meta)
        lowered_out = tensor_map[output_id]
        lowered_mask = f"region:{region.id}:mask"

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
            id=f"region:{region.id}",
            structural_operation_ids=region.operation_ids,
            implementation=self.descriptor.id,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
            input_tensors=(tensor_map[left_id],),
            output_tensors=(lowered_out,),
            auxiliary_tensors=(lowered_mask,),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            auxiliary_metadata=MappingProxyType({lowered_mask: mask_meta}),
        )


RELU_PATTERN = PatternMatchRule(
    id="pat-relu",
    kind="region/relu",
    priority=10,
    op_families=("maximum",),
    constraints=(PatternConstraint(kind="right_operand_is_zero"),),
)

RELU_REGION = ReLURegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/relu",
        kind="region/relu",
        priority=10,
        capabilities=frozenset({"relu", "mask_retention"}),
        pattern_rule=RELU_PATTERN,
    ),
)
