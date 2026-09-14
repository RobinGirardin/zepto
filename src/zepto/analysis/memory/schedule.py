"""Schedule-driven RELEASE augmentation for peak VRAM (Pass 2)."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredGraph, LoweredNode
from zepto.semantic.metadata import TensorRole
from zepto.semantic.operations.records import ResourceEventKind

from .types import TimelineEvent


@dataclass(frozen=True, slots=True)
class LastUseRecord:
    """Last schedule point where a tensor storage slot is still needed."""

    edge_id: str
    producer_node_index: int
    last_consumer_node_index: int


def build_last_use_map(lowered: LoweredGraph) -> dict[str, LastUseRecord]:
    """Build last-use records keyed by lowered edge id."""
    records: dict[str, LastUseRecord] = {}
    output_edges = set(lowered.output_edge_ids)

    edge_producer: dict[str, int] = {}
    edge_consumers: dict[str, list[int]] = {}
    saved_edges: dict[str, int] = {}

    for node_index, node in enumerate(lowered.nodes):
        for event in node.resource_events:
            if event.kind is ResourceEventKind.ALLOCATE:
                if event.value not in edge_producer:
                    edge_producer[event.value] = node_index
            elif event.kind is ResourceEventKind.SAVE:
                saved_edges[event.value] = max(
                    saved_edges.get(event.value, node_index), node_index
                )

        for edge_id in _node_referenced_edges(node):
            edge_consumers.setdefault(edge_id, []).append(node_index)

    for edge_id, lowered_edge in lowered.edges.items():
        if edge_id in output_edges:
            continue
        if _protected_by_output_alias(lowered, edge_id):
            continue
        if lowered_edge.role is TensorRole.INPUT and not lowered_edge.tensor.persistent:
            consumers = edge_consumers.get(edge_id, [])
            if len(consumers) <= 1:
                continue
            records[edge_id] = LastUseRecord(
                edge_id=edge_id,
                producer_node_index=-1,
                last_consumer_node_index=max(consumers),
            )
            continue

        if edge_id in saved_edges:
            records[edge_id] = LastUseRecord(
                edge_id=edge_id,
                producer_node_index=edge_producer.get(edge_id, -1),
                last_consumer_node_index=saved_edges[edge_id],
            )
            continue

        if edge_id in edge_producer:
            consumers = edge_consumers.get(edge_id, [])
            if consumers:
                records[edge_id] = LastUseRecord(
                    edge_id=edge_id,
                    producer_node_index=edge_producer[edge_id],
                    last_consumer_node_index=max(consumers),
                )

    for node_index, node in enumerate(lowered.nodes):
        for aux_edge_id in _gradient_auxiliary_edges(lowered, node):
            source_input = _grad_source_input_edge(lowered, node, aux_edge_id)
            if source_input is None:
                continue
            producer = edge_producer.get(source_input, -1)
            records[aux_edge_id] = LastUseRecord(
                edge_id=aux_edge_id,
                producer_node_index=producer,
                last_consumer_node_index=producer,
            )

    return records


def augment_timeline_with_scheduled_releases(
    timeline: list[TimelineEvent],
    last_use: dict[str, LastUseRecord],
    lowered: LoweredGraph,
) -> list[TimelineEvent]:
    """Insert schedule-driven RELEASE events after Pass 1 timeline construction."""
    if not last_use:
        return list(timeline)

    gradient_aux = _all_gradient_aux_edges(lowered)
    scheduled: list[TimelineEvent] = []
    seen: set[tuple[int, str]] = set()

    for edge_id, record in last_use.items():
        if edge_id in gradient_aux:
            release_node = record.producer_node_index
        else:
            release_node = record.last_consumer_node_index
        if release_node < 0:
            continue
        key = (release_node, edge_id)
        if key in seen:
            continue
        seen.add(key)
        scheduled.append(
            TimelineEvent(
                node_index=release_node,
                kind=ResourceEventKind.RELEASE,
                edge_id=edge_id,
                storage_id=_storage_for_edge(lowered, edge_id),
                delta_live=0,
                phase="scheduled",
                scheduled=True,
            )
        )

    merged = list(timeline)
    for event in scheduled:
        insert_at = _insert_index_after_node(merged, event.node_index)
        merged.insert(insert_at, event)
    return merged


def _node_referenced_edges(node: LoweredNode) -> tuple[str, ...]:
    return node.input_edges + node.output_edges + node.auxiliary_edges


def _gradient_auxiliary_edges(
    lowered: LoweredGraph, node: LoweredNode
) -> tuple[str, ...]:
    return tuple(
        edge_id
        for edge_id in node.auxiliary_edges
        if lowered.edges[edge_id].role is TensorRole.GRADIENT
    )


def _all_gradient_aux_edges(lowered: LoweredGraph) -> set[str]:
    result: set[str] = set()
    for node in lowered.nodes:
        result.update(_gradient_auxiliary_edges(lowered, node))
    return result


def _grad_source_input_edge(
    lowered: LoweredGraph,
    node: LoweredNode,
    aux_edge_id: str,
) -> str | None:
    gradient_aux = _gradient_auxiliary_edges(lowered, node)
    if aux_edge_id not in gradient_aux:
        return None
    input_index = gradient_aux.index(aux_edge_id)
    if input_index >= len(node.input_edges):
        return None
    return node.input_edges[input_index]


def _storage_for_edge(lowered: LoweredGraph, edge_id: str) -> str:
    if edge_id in lowered.edges:
        return lowered.edges[edge_id].storage_id
    return edge_id


def _protected_by_output_alias(lowered: LoweredGraph, edge_id: str) -> bool:
    """Return whether an edge feeds a graph output through a view alias."""
    for node in lowered.nodes:
        alias_events = [
            event
            for event in node.resource_events
            if event.kind is ResourceEventKind.ALIAS
        ]
        for event in alias_events:
            if event.value not in lowered.output_edge_ids:
                continue
            source = event.storage or (
                node.input_edges[0]
                if event.value in node.output_edges and node.input_edges
                else event.value
            )
            if source == edge_id:
                return True
    return False


def _insert_index_after_node(timeline: list[TimelineEvent], node_index: int) -> int:
    for index in range(len(timeline) - 1, -1, -1):
        if timeline[index].node_index <= node_index:
            return index + 1
    return len(timeline)
