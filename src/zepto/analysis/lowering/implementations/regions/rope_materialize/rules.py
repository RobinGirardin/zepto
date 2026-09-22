"""Discovery rules for RoPEMaterialize regions."""

from __future__ import annotations

from ....region import ProvenanceMatchRule

ROPE_MATERIALIZE_PROVENANCE = ProvenanceMatchRule(
    id="prov-rope-materialize",
    kind="region/rope_materialize",
    priority=5,
    component_type="RoPEMaterialize",
    require_contiguous_in_graph_order=True,
)
