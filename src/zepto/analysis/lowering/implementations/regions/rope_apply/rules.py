"""Discovery rules for RoPEApply regions."""

from __future__ import annotations

from ....region import ProvenanceMatchRule

ROPE_APPLY_PROVENANCE = ProvenanceMatchRule(
    id="prov-rope-apply",
    kind="region/rope_apply",
    priority=5,
    component_type="RoPEApply",
    require_contiguous_in_graph_order=True,
)
