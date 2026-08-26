"""Fused linear region lowering via provenance discovery."""

from __future__ import annotations

from dataclasses import dataclass

from ....graph import StructuralGraph
from ....lowered import LoweredOperation
from ....operation.records import EstimationContext
from ....operation.structural import StructuralOperation
from ...context import InvocationContext
from ...helpers import (
    RegionEstimationContext,
    append_save_events,
    build_estimation_context,
    ensure_lowered_tensor,
    remap_events,
)
from ...region import ProvenanceMatchRule, StructuralRegion
from ...registry import RegionImplementationDescriptor


@dataclass(frozen=True, slots=True)
class FusedLinearRegionImplementation:
    """Lower one linear_matmul inside a Linear module envelope."""

    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: StructuralRegion,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if len(region.operation_ids) != 1:
            return "fused linear expects one operation"
        structural = graph.operation(region.operation_ids[0])
        if structural.operation_family != "linear_matmul":
            return "not linear_matmul"
        if region.anchor.component_type != "Linear":
            return "not a Linear module invocation"
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
        del estimation
        structural = graph.operation(region.operation_ids[0])
        return _lower_linear_matmul(
            structural,
            graph,
            context,
            tensor_map,
            lowered_tensors,
            region=region,
            implementation_id=self.descriptor.id,
        )


def _lower_linear_matmul(
    structural: StructuralOperation,
    graph: StructuralGraph,
    context: InvocationContext,
    tensor_map: dict,
    lowered_tensors: dict,
    *,
    region: StructuralRegion,
    implementation_id: str,
) -> LoweredOperation:
    op = structural.declaration
    if op is None:
        raise ValueError(f"Operation {structural.id} has no declaration")
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

    estimation = build_estimation_context(structural, graph, context)
    events = op.resource_events(estimation, result)
    forward = op.forward_flops(estimation)
    backward = op.backward_flops(
        EstimationContext(
            phase="backward",
            port_metadata=estimation.port_metadata,
            precision=estimation.precision,
            state=estimation.state,
        )
    )

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
        id=f"region:{region.id}",
        structural_operation_ids=region.operation_ids,
        implementation=implementation_id,
        module_path=region.anchor.module_path,
        component_type=region.anchor.component_type,
        region_id=region.id,
        input_tensors=input_ids,
        output_tensors=output_ids,
        auxiliary_tensors=auxiliary_ids,
        resource_events=with_saves,
        forward_flops=forward,
        backward_flops=backward,
    )


LINEAR_PROVENANCE = ProvenanceMatchRule(
    id="prov-linear",
    kind="region/linear",
    priority=5,
    component_type="Linear",
)

FUSED_LINEAR_REGION = FusedLinearRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/linear",
        kind="region/linear",
        priority=5,
        provenance_rule=LINEAR_PROVENANCE,
    ),
)
