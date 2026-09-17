"""Discovery rules for fused gated delta scan regions."""

from __future__ import annotations

from ....region import ProvenanceMatchRule

GATED_DELTA_SCAN_PROVENANCE = ProvenanceMatchRule(
    id="prov-gated-delta-scan",
    kind="region/gated_delta_scan",
    priority=10,
    component_type="GatedDeltaScan",
    require_contiguous_in_graph_order=True,
)

GATED_DELTA_SCAN_STEP_CORE: tuple[str, ...] = (
    "exp",
    "multiply",
    "matmul",
    "subtract",
    "multiply",
    "matmul",
    "add",
    "matmul",
)
