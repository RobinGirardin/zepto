"""Immutable lowered graph records."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from zepto.compose.values import Tensor
from zepto.graph.ids import EdgeId, NodeId, ParameterId
from zepto.semantic.metadata import TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

if TYPE_CHECKING:
    from .lowering.context import InvocationContext
    from .lowering.registry import ImplementationSelection, RegionImplementationSelection


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
class StatePortEvent:
    """Cross-step state port mutation recorded during lowering."""

    port_name: str
    kind: ResourceEventKind
    bytes: int
    layer_index: int | None = None


@dataclass(frozen=True, slots=True)
class LoweredParameter:
    """One invocation-refined graph parameter."""

    id: str
    parameter_id: ParameterId
    tensor: Tensor
    role: TensorRole
    storage_id: str
    trainable: bool


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
    parameters: Mapping[str, LoweredParameter]
    parameter_map: Mapping[ParameterId, str]
    nodes: tuple[LoweredNode, ...]
    context: InvocationContext
    edge_map: Mapping[EdgeId, str]
    node_map: Mapping[NodeId, str]
    output_edge_ids: tuple[str, ...]
    selections: tuple[ImplementationSelection, ...]
    fusion_map: Mapping[NodeId, str] = MappingProxyType({})
    region_map: Mapping[str, tuple[NodeId, ...]] = MappingProxyType({})
    region_selections: tuple[RegionImplementationSelection, ...] = ()
    state_port_events: tuple[StatePortEvent, ...] = ()
