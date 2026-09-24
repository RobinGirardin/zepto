"""Specialized maximum-family lowering implementations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.node import Node
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.helpers import includes_backward, numel
from zepto.semantic.operations.records import EstimationContext, ResourceEvent, ResourceEventKind
from ..context import InvocationContext
from ..helpers import (
    ensure_lowered_edge,
    is_zero_operand,
    register_auxiliary_edge,
)
from ..role import RoleContext
from ..registry import ImplementationDescriptor


@dataclass(frozen=True, slots=True)
class ReLUMaskImplementation:
    """Lower maximum(left, 0) with a retained forward mask for backward."""

    descriptor: ImplementationDescriptor

    def compatible(
        self,
        node: Node,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if node.operation_family != "maximum":
            return "not maximum"

        if len(node.input_edges) != 2:
            return "maximum expects two inputs"
        _left_id, right_id = node.input_edges
        right = graph.edge(right_id)
        if not is_zero_operand(right, graph):
            return "right operand is not provably zero"
        return None

    def lower(
        self,
        node: Node,
        graph: Graph,
        context: InvocationContext,
        edge_map: dict,
        *,
        estimation: EstimationContext,
        lowered_edges: dict,
    ) -> LoweredNode:
        left_id, _right_id = node.input_edges
        output_id = node.output_edges[0]
        left_port, _right_port = node.input_ports
        (output_port,) = node.output_ports

        ensure_lowered_edge(
            left_id,
            graph,
            context,
            lowered_edges,
            edge_map,
            node=node,
            port_name=left_port.name,
            port_direction="input",
        )
        ensure_lowered_edge(
            output_id,
            graph,
            context,
            lowered_edges,
            edge_map,
            node=node,
            port_name=output_port.name,
            port_direction="output",
        )

        left = estimation.tensor_for("left")
        output = estimation.tensor_for("output")
        if left is None or output is None:
            raise ValueError("ReLU lowering requires left and output port tensors")

        n = numel(output)
        lowered_out = edge_map[output_id]
        lowered_mask = f"op{node.id.index}:mask"

        mask_tensor = Tensor(
            shape=output.shape,
            semantic_type="relu_mask",
            dtype=DType.BOOL,
            requires_grad=False,
            persistent=False,
        )
        register_auxiliary_edge(
            lowered_mask,
            mask_tensor,
            role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
            context=context,
            storage_id=lowered_mask,
            lowered_edges=lowered_edges,
        )

        forward_flops = n
        backward_flops = n if left.requires_grad else 0

        events: tuple[ResourceEvent, ...] = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_mask),
            ResourceEvent(ResourceEventKind.SAVE, lowered_mask),
        )
        if includes_backward(context.phase):
            events = (
                *events,
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    lowered_mask,
                    phase="backward",
                ),
            )

        return LoweredNode(
            id=f"op{node.id.index}",
            node_id=node.id,
            implementation=self.descriptor.id,
            input_edges=(edge_map[left_id],),
            output_edges=(lowered_out,),
            auxiliary_edges=(lowered_mask,),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
        )


RELU_MASK = ReLUMaskImplementation(
    descriptor=ImplementationDescriptor(
        id="maximum/relu-mask",
        family="maximum",
        priority=10,
        capabilities=frozenset({"relu", "mask_retention"}),
    ),
)
