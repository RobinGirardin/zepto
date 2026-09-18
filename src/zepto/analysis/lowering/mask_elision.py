"""Post-pass: elide materialized mask VRAM when flash-class GQA regions fuse."""

from __future__ import annotations

from dataclasses import replace

from zepto.graph.graph import Graph
from zepto.graph.ids import NodeId
from zepto.semantic.operations.records import ResourceEventKind

from .implementations.regions.gqa.shared import mask_producer_for_region
from .region import Region

_FLASH_CLASS_IMPLEMENTATIONS = frozenset(
    {
        "region/gqa/flash2",
        "region/gqa/flash3",
        "region/gqa/sdpa-flash",
        "region/gqa/sdpa-mem-efficient",
        "region/gqa-sink/flash2",
        "region/gqa-sink/flash3",
    }
)

_ELIDE_KINDS = frozenset({ResourceEventKind.ALLOCATE, ResourceEventKind.PERSIST})


def _strip_mask_storage_events(events: tuple) -> tuple:
    return tuple(ev for ev in events if ev.kind not in _ELIDE_KINDS)


def apply_mask_elision(
    graph: Graph,
    regions: tuple[Region, ...],
    *,
    nodes: list,
    node_map: dict[NodeId, str],
) -> None:
    """Remove mask ALLOCATE/PERSIST when a flash-class GQA region absorbed the add op."""
    lowered_by_id = {node.id: node for node in nodes}
    region_by_id = {region.id: region for region in regions}
    elided_lowered_ids: set[str] = set()

    for lowered in nodes:
        if lowered.implementation not in _FLASH_CLASS_IMPLEMENTATIONS:
            continue
        if lowered.region_id is None:
            continue
        region = region_by_id.get(lowered.region_id)
        if region is None:
            continue
        mask_producer_id = mask_producer_for_region(region, graph)
        if mask_producer_id is None:
            continue
        mask_lowered_id = node_map.get(mask_producer_id)
        if mask_lowered_id is None or mask_lowered_id in elided_lowered_ids:
            continue
        mask_node = lowered_by_id.get(mask_lowered_id)
        if mask_node is None:
            continue
        updated = replace(
            mask_node,
            resource_events=_strip_mask_storage_events(mask_node.resource_events),
        )
        lowered_by_id[mask_lowered_id] = updated
        elided_lowered_ids.add(mask_lowered_id)

    for index, node in enumerate(nodes):
        if node.id in elided_lowered_ids:
            nodes[index] = lowered_by_id[node.id]
