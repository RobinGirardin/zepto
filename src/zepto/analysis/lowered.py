"""Immutable lowered graph records."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from zepto.compose.values import Tensor
from zepto.graph.ids import EdgeId, NodeId
from zepto.semantic.metadata import TensorRole
from .lowering.context import InvocationContext
from .lowering.registry import ImplementationSelection
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


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    """Immutable lowered graph for one invocation."""

    edges: Mapping[str, LoweredEdge]
    nodes: tuple[LoweredNode, ...]
    context: InvocationContext
    edge_map: Mapping[EdgeId, str]
    node_map: Mapping[NodeId, str]
    selections: tuple[ImplementationSelection, ...]
