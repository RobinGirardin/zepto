"""Immutable lowered graph records."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from zepto.compose.values import Tensor
from zepto.graph.ids import EdgeId, NodeId
from zepto.semantic.metadata import TensorRole
from .lowering.context import InvocationContext
from .lowering.registry import ImplementationSelection, RegionImplementationSelection
from zepto.semantic.operations.records import ResourceEvent


@dataclass(frozen=True, slots=True)
class LoweredEdge:
    """One invocation-refined graph edge."""

    id: str
    edge_id: EdgeId | None
    tensor: Tensor
    role: TensorRole
    storage_id: str
    workspace: bool = False


@dataclass(frozen=True, slots=True)
class LoweredNode:
    """One invocation-refined graph node with FLOP and memory accounting."""

    id: str
    node_id: NodeId
    implementation: str
    input_edges: tuple[str, ...]
    output_edges: tuple[str, ...]
    auxiliary_edges: tuple[str, ...]
    resource_events: tuple[ResourceEvent, ...]
    forward_flops: int
    backward_flops: int
    module_path: tuple[str, ...] = ()
    component_type: str | None = None
    region_id: str | None = None
    node_ids: tuple[NodeId, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_ids:
            object.__setattr__(self, "node_ids", (self.node_id,))


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    """Immutable lowered graph for one invocation."""

    edges: Mapping[str, LoweredEdge]
    nodes: tuple[LoweredNode, ...]
    context: InvocationContext
    edge_map: Mapping[EdgeId, str]
    node_map: Mapping[NodeId, str]
    selections: tuple[ImplementationSelection, ...]
    fusion_map: Mapping[NodeId, str] = MappingProxyType({})
    region_map: Mapping[str, tuple[NodeId, ...]] = MappingProxyType({})
    region_selections: tuple[RegionImplementationSelection, ...] = ()
