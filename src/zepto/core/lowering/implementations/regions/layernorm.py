"""Fused layer-norm region lowering via hybrid discovery."""

from __future__ import annotations

from dataclasses import dataclass

from ....graph import StructuralGraph
from ....lowered import LoweredOperation
from ....operation.helpers import numel
from ....operation.records import ResourceEvent, ResourceEventKind
from ...context import InvocationContext
from ...helpers import RegionEstimationContext, ensure_lowered_tensor, register_auxiliary_tensor, resolve_tensor_metadata
from ...region import (
    PatternConstraint,
    PatternMatchRule,
    ProvenanceMatchRule,
    StructuralRegion,
)
from ...registry import RegionImplementationDescriptor


LAYERNORM_PATTERN = PatternMatchRule(
    id="pat-layernorm-decomposed",
    kind="region/layernorm",
    priority=5,
    op_families=(
        "reduce_sum",
        "subtract",
        "multiply",
        "add",
        "square_root",
        "divide",
    ),
    edge_constraints=(
        (0, 1, "output_to_second_input"),
        (1, 2, "output_to_input"),
        (2, 3, "output_to_input"),
        (3, 4, "output_to_input"),
        (4, 5, "output_to_second_input"),
        (1, 5, "output_to_first_input"),
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=2),
    ),
)

LAYERNORM_PROVENANCE = ProvenanceMatchRule(
    id="prov-layernorm",
    kind="region/layernorm",
    priority=5,
    component_type="LayerNorm",
)


@dataclass(frozen=True, slots=True)
class FusedLayerNormRegionImplementation:
    """Lower a decomposed layer-norm primitive chain as one fused region."""

    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: StructuralRegion,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if region.anchor.component_type != "LayerNorm":
            return "not a LayerNorm module invocation"
        if len(region.operation_ids) < 2:
            return "layer norm fusion expects multiple ops"
        families = tuple(
            graph.operation(op_id).operation_family
            for op_id in region.operation_ids
        )
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if families != expected.op_families:
            return "unexpected operation sequence"
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
        for tensor_id in region.boundary_inputs:
            ensure_lowered_tensor(
                tensor_id, graph, context, lowered_tensors, tensor_map
            )
        for tensor_id in region.boundary_outputs:
            ensure_lowered_tensor(
                tensor_id, graph, context, lowered_tensors, tensor_map
            )

        input_meta = estimation.input_metadata.get("input")
        output_meta = estimation.output_metadata.get("output")
        if input_meta is None and region.boundary_inputs:
            input_meta = resolve_tensor_metadata(
                graph.tensor(region.boundary_inputs[0]).metadata, context
            )
        if output_meta is None and region.boundary_outputs:
            output_meta = resolve_tensor_metadata(
                graph.tensor(region.boundary_outputs[0]).metadata, context
            )
        if input_meta is None or output_meta is None:
            raise ValueError("LayerNorm region requires boundary metadata")

        n = numel(input_meta)
        output_id = region.boundary_outputs[0]
        lowered_out = tensor_map[output_id]
        saved_mean = f"region:{region.id}:mean"
        saved_inv_std = f"region:{region.id}:inv_std"

        aux_meta = resolve_tensor_metadata(output_meta, context)
        register_auxiliary_tensor(
            saved_mean,
            aux_meta,
            storage_id=saved_mean,
            lowered_tensors=lowered_tensors,
        )
        register_auxiliary_tensor(
            saved_inv_std,
            aux_meta,
            storage_id=saved_inv_std,
            lowered_tensors=lowered_tensors,
        )

        forward_flops = 5 * n
        backward_flops = 5 * n if input_meta.requires_grad else 0

        events: tuple[ResourceEvent, ...] = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, saved_mean),
            ResourceEvent(ResourceEventKind.SAVE, saved_mean),
            ResourceEvent(ResourceEventKind.ALLOCATE, saved_inv_std),
            ResourceEvent(ResourceEventKind.SAVE, saved_inv_std),
        )

        return LoweredOperation(
            id=f"region:{region.id}",
            structural_operation_ids=region.operation_ids,
            implementation=self.descriptor.id,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
            input_tensors=tuple(
                tensor_map[tid] for tid in region.boundary_inputs
            ),
            output_tensors=(lowered_out,),
            auxiliary_tensors=(saved_mean, saved_inv_std),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
        )


FUSED_LAYERNORM_REGION = FusedLayerNormRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/layernorm",
        kind="region/layernorm",
        priority=8,
        provenance_rule=LAYERNORM_PROVENANCE,
        pattern_rule=LAYERNORM_PATTERN,
    ),
)
