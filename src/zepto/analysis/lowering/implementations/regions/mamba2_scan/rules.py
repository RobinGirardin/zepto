"""Discovery rules for fused Mamba-2 selective SSM scan regions."""

from __future__ import annotations

from ....region import ProvenanceMatchRule

MAMBA2_SCAN_PROVENANCE = ProvenanceMatchRule(
    id="prov-mamba2-scan",
    kind="region/mamba2_scan",
    priority=10,
    component_type="SelectiveSSMScan",
    require_contiguous_in_graph_order=True,
)
