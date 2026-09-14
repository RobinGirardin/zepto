"""Shared helpers for the lowering pass."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping

from zepto.compose.values import Tensor
from zepto.graph.edge import Edge
from zepto.graph.graph import Graph
from zepto.graph.ids import EdgeId
from zepto.graph.node import Node
from zepto.semantic.metadata import TensorRole
from zepto.semantic.operations.records import EstimationContext, ResourceEvent, ResourceEventKind
from zepto.semantic.ports import ValueKind
from ..accounting import PrecisionPolicy
from ..lowered import StatePortEvent
from .context import InvocationContext
from .region import Region
from .role import RoleContext, resolve_for_accounting, resolve_lowered_edge

if TYPE_CHECKING:
    from zepto.analysis.horizon.state import KVCacheState


def edge_storage_id(edge: Edge) -> str:
    """Return a string storage identity for accounting."""
    if edge.storage_id is None:
        return f"anon:{edge.id.index}"
    return f"s{edge.storage_id.index}"


def _port_value_kind(
    node: Node | None,
    port_name: str | None,
    port_direction: str | None,
) -> ValueKind | None:
    if node is None or port_name is None or port_direction is None:
        return None
    if port_direction == "input":
        ports = node.input_ports
    elif port_direction == "parameter":
        ports = node.parameter_ports
    elif port_direction == "output":
        ports = node.output_ports
    elif port_direction == "auxiliary":
        ports = node.auxiliary_ports
    else:
        return None
    for port in ports:
        if port.name == port_name:
            return port.value_kind
    return None


def ensure_lowered_edge(
    edge_id: EdgeId,
    graph: Graph,
    context: InvocationContext,
    lowered_edges: dict[str, object],
    edge_map: dict[EdgeId, str],
    *,
    node: Node | None = None,
    port_name: str | None = None,
    port_direction: str | None = None,
) -> str:
    """Materialize one graph edge in the lowered graph if needed."""
    from ..lowered import LoweredEdge

    if edge_id in edge_map:
        return edge_map[edge_id]

    source = graph.edge(edge_id)
    lowered_id = f"t{edge_id.index}"
    role_ctx = RoleContext(
        graph=graph,
        edge_id=edge_id,
        port_name=port_name,
        port_direction=port_direction,  # type: ignore[arg-type]
        port_value_kind=_port_value_kind(node, port_name, port_direction),
    )
    tensor, role = resolve_lowered_edge(
        source.tensor, role_ctx=role_ctx, context=context
    )
    lowered_edges[lowered_id] = LoweredEdge(
        id=lowered_id,
        edge_id=edge_id,
        tensor=tensor,
        role=role,
        storage_id=edge_storage_id(source),
    )
    edge_map[edge_id] = lowered_id
    return lowered_id


def build_estimation_context(
    node: Node,
    graph: Graph,
    context: InvocationContext,
) -> EstimationContext:
    """Build per-node estimation values using the shared role pipeline."""
    port_values: list[tuple[str, object]] = []
    for port, bound_edge_id in zip(node.input_ports, node.input_edges, strict=True):
        role_ctx = RoleContext(
            graph=graph,
            edge_id=bound_edge_id,
            port_name=port.name,
            port_direction="input",
            port_value_kind=port.value_kind,
        )
        port_values.append(
            (
                port.name,
                resolve_for_accounting(
                    graph.edge(bound_edge_id).tensor,
                    role_ctx=role_ctx,
                    context=context,
                ),
            )
        )
    for port, parameter_id in zip(node.parameter_ports, node.parameter_ids, strict=True):
        param = graph.parameter(parameter_id)
        role_ctx = RoleContext(
            graph=graph,
            port_name=port.name,
            port_direction="parameter",
            port_value_kind=port.value_kind,
        )
        port_values.append(
            (
                port.name,
                resolve_for_accounting(
                    param.as_tensor(),
                    role_ctx=role_ctx,
                    context=context,
                ),
            )
        )
    for port, bound_edge_id in zip(node.output_ports, node.output_edges, strict=True):
        role_ctx = RoleContext(
            graph=graph,
            edge_id=bound_edge_id,
            port_name=port.name,
            port_direction="output",
            port_value_kind=port.value_kind,
        )
        port_values.append(
            (
                port.name,
                resolve_for_accounting(
                    graph.edge(bound_edge_id).tensor,
                    role_ctx=role_ctx,
                    context=context,
                ),
            )
        )
    if node.auxiliary_edges is not None:
        aux_by_name = {port.name: port for port in node.auxiliary_ports}
        for port_name, bound_edge_id in node.auxiliary_edges.items():
            port = aux_by_name[port_name]
            role_ctx = RoleContext(
                graph=graph,
                edge_id=bound_edge_id,
                port_name=port.name,
                port_direction="auxiliary",
                port_value_kind=port.value_kind,
            )
            port_values.append(
                (
                    port.name,
                    resolve_for_accounting(
                        graph.edge(bound_edge_id).tensor,
                        role_ctx=role_ctx,
                        context=context,
                    ),
                )
            )
    return EstimationContext(
        phase=context.phase,
        port_values=tuple(port_values),  # type: ignore[arg-type]
        precision=context.precision,
        state=context.state,
    )


def port_edge_id(
    node: Node,
    port_name: str,
    role: str,
) -> EdgeId:
    """Resolve a port name to its bound graph edge id."""
    if role == "input":
        ports = node.input_ports
        edge_ids = node.input_edges
    elif role == "output":
        ports = node.output_ports
        edge_ids = node.output_edges
    elif role == "auxiliary":
        if node.auxiliary_edges is None:
            raise KeyError(f"Unknown auxiliary port {port_name!r}")
        return node.auxiliary_edges[port_name]
    else:
        raise KeyError(f"Unknown port role {role!r}")
    for port, bound_edge_id in zip(ports, edge_ids, strict=True):
        if port.name == port_name:
            return bound_edge_id
    raise KeyError(f"Unknown {role} port {port_name!r}")


def remap_events(
    events: tuple[ResourceEvent, ...],
    node: Node,
    edge_map: dict[EdgeId, str],
) -> tuple[ResourceEvent, ...]:
    """Remap port-relative event targets to lowered edge ids."""
    port_targets: dict[str, str] = {}
    for index, port in enumerate(node.input_ports):
        port_targets[port.name] = edge_map[node.input_edges[index]]
    for index, port in enumerate(node.output_ports):
        lowered = edge_map[node.output_edges[index]]
        port_targets[port.name] = lowered
        port_targets[f"output:{index}"] = lowered
    if node.auxiliary_edges is not None:
        for port_name, bound_edge_id in node.auxiliary_edges.items():
            port_targets[port_name] = edge_map[bound_edge_id]

    def remap(value: str | None) -> str | None:
        if value is None:
            return None
        if value in port_targets:
            return port_targets[value]
        if value.startswith("output:"):
            index = int(value.split(":", 1)[1])
            return edge_map[node.output_edges[index]]
        return value

    return tuple(
        replace(event, value=remap(event.value), storage=remap(event.storage))
        for event in events
    )


def append_save_events(
    events: tuple[ResourceEvent, ...],
    node: Node,
    edge_map: dict[EdgeId, str],
) -> tuple[ResourceEvent, ...]:
    """Append SAVE events for saved-for-backward port refs."""
    save_events: list[ResourceEvent] = []
    for port_ref in node.saved_for_backward:
        if port_ref.role == "parameter":
            continue
        bound_edge_id = port_edge_id(node, port_ref.port_name, port_ref.role)
        save_events.append(
            ResourceEvent(ResourceEventKind.SAVE, edge_map[bound_edge_id])
        )
    return (*events, *save_events)


def register_auxiliary_edge(
    aux_id: str,
    tensor: Tensor,
    *,
    role_ctx: RoleContext,
    context: InvocationContext,
    storage_id: str | None = None,
    lowered_edges: dict[str, object],
) -> None:
    """Register a lowering-local auxiliary edge."""
    from ..lowered import LoweredEdge

    if aux_id in lowered_edges:
        return
    resolved_tensor, role = resolve_lowered_edge(
        tensor, role_ctx=role_ctx, context=context
    )
    lowered_edges[aux_id] = LoweredEdge(
        id=aux_id,
        edge_id=None,
        tensor=resolved_tensor,
        role=role,
        storage_id=storage_id or aux_id,
        workspace=False,
    )


def is_zero_operand(edge: Edge, graph: Graph) -> bool:
    """Return whether an edge is provably a constant-zero operand."""
    del graph
    return edge.tensor.semantic_type == "constant_zero"


@dataclass(frozen=True, slots=True)
class RegionEstimationContext:
    """Aggregated port tensors for one region."""

    region: Region
    phase: str
    input_tensors: Mapping[str, Tensor]
    output_tensors: Mapping[str, Tensor]
    parameter_tensors: Mapping[str, Tensor]
    precision: PrecisionPolicy | None
    state: tuple[tuple[str, object], ...]


def build_region_estimation_context(
    region: Region,
    graph: Graph,
    context: InvocationContext,
) -> RegionEstimationContext:
    """Resolve boundary tensor and parameter values for a region."""
    input_tensors: dict[str, Tensor] = {}
    for index, edge_id in enumerate(region.boundary_inputs):
        edge = graph.edge(edge_id)
        role_ctx = RoleContext(graph=graph, edge_id=edge_id, port_direction="input")
        tensor, _role = resolve_lowered_edge(
            edge.tensor, role_ctx=role_ctx, context=context
        )
        input_tensors["input" if index == 0 else f"input{index}"] = tensor

    output_tensors: dict[str, Tensor] = {}
    for index, edge_id in enumerate(region.boundary_outputs):
        edge = graph.edge(edge_id)
        role_ctx = RoleContext(graph=graph, edge_id=edge_id, port_direction="output")
        tensor, _role = resolve_lowered_edge(
            edge.tensor, role_ctx=role_ctx, context=context
        )
        output_tensors["output" if index == 0 else f"output{index}"] = tensor

    parameter_tensors: dict[str, Tensor] = {}
    for index, parameter_id in enumerate(region.parameter_ids):
        param = graph.parameter(parameter_id)
        role_ctx = RoleContext(
            port_direction="parameter",
            port_value_kind=ValueKind.PARAMETER,
        )
        tensor, _role = resolve_lowered_edge(
            param.as_tensor(), role_ctx=role_ctx, context=context
        )
        parameter_tensors[f"param{index}"] = tensor

    return RegionEstimationContext(
        region=region,
        phase=context.phase,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        parameter_tensors=parameter_tensors,
        precision=context.precision,
        state=context.state,
    )


@dataclass(frozen=True, slots=True)
class KVScenario:
    """Resolved KV cache geometry for state-aware GQA lowering."""

    num_kv_heads: int
    head_dim: int
    cache_seq_len: int
    dtype_itemsize: int
    is_decode: bool
    layer_index: int | None = None


def kv_cache_from_state(
    state: tuple[tuple[str, object], ...],
) -> KVCacheState | None:
    """Return the first KV cache snapshot in invocation state, if any."""
    from zepto.analysis.horizon.state import KVCacheState as KVState

    for _name, value in state:
        if isinstance(value, KVState):
            return value
    return None


def kv_template_from_state(
    state: tuple[tuple[str, object], ...],
) -> object | None:
    """Return KV template metadata injected by StatePortRegistry."""
    for name, value in state:
        if name == "kv_template":
            return value
    return None


def layer_index_from_module_path(module_path: tuple[str, ...]) -> int | None:
    """Extract decoder layer index from a module path such as layers.3.attn."""
    for index, part in enumerate(module_path):
        if part == "layers" and index + 1 < len(module_path):
            try:
                return int(module_path[index + 1])
            except ValueError:
                return None
    return None


def resolve_kv_scenario(
    *,
    context: InvocationContext,
    query_seq_len: int,
    num_heads: int,
    head_dim: int,
    module_path: tuple[str, ...],
) -> KVScenario | None:
    """Resolve KV geometry when horizon state is present on the invocation."""
    from zepto.analysis.horizon.state import KVCacheState as KVState

    kv_state = kv_cache_from_state(context.state)
    template = kv_template_from_state(context.state)

    if kv_state is None and template is None:
        return None

    if kv_state is not None:
        num_kv_heads = kv_state.num_kv_heads
        dtype_itemsize = kv_state.dtype.itemsize or 0
        cache_seq_len = kv_state.seq_len
        is_decode = query_seq_len == 1 and cache_seq_len > query_seq_len
    else:
        num_kv_heads = template.num_kv_heads  # type: ignore[union-attr]
        dtype_itemsize = template.dtype.itemsize or 0  # type: ignore[union-attr]
        cache_seq_len = query_seq_len
        is_decode = False

    return KVScenario(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_seq_len=cache_seq_len,
        dtype_itemsize=dtype_itemsize,
        is_decode=is_decode,
        layer_index=layer_index_from_module_path(module_path),
    )


def layer_kv_bytes(scenario: KVScenario, *, seq_len: int) -> int:
    """Bytes for one layer's K+V cache at the given sequence length."""
    return (
        2
        * scenario.num_kv_heads
        * seq_len
        * scenario.head_dim
        * scenario.dtype_itemsize
    )


def kv_state_port_events(
    *,
    region_id: str,
    scenario: KVScenario,
    seq_len: int,
) -> tuple[StatePortEvent, ResourceEvent, ResourceEvent, str]:
    """Build state-port record and resource events for one layer's KV cache."""
    aux_id = f"region:{region_id}:kv_cache"
    byte_count = layer_kv_bytes(scenario, seq_len=seq_len)
    layer = scenario.layer_index
    port_name = f"kv_cache:{layer}" if layer is not None else aux_id
    state_event = StatePortEvent(
        port_name=port_name,
        kind=ResourceEventKind.ALLOCATE,
        bytes=byte_count,
        layer_index=layer,
    )
    allocate = ResourceEvent(ResourceEventKind.ALLOCATE, aux_id)
    persist = ResourceEvent(ResourceEventKind.PERSIST, aux_id)
    return state_event, allocate, persist, aux_id


def register_kv_cache_aux(
    *,
    aux_id: str,
    scenario: KVScenario,
    seq_len: int,
    context: InvocationContext,
    lowered_edges: dict[str, object],
) -> None:
    """Register a lowering-local KV cache auxiliary edge."""
    shape = (scenario.num_kv_heads, seq_len, scenario.head_dim)
    register_auxiliary_edge(
        aux_id,
        Tensor(
            shape=shape,
            semantic_type="kv_cache",
            dtype=context.precision.default_dtype,
            requires_grad=False,
            persistent=True,
        ),
        role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
        context=context,
        storage_id=aux_id,
        lowered_edges=lowered_edges,
    )
