"""Discovery rules for depthwise causal conv1d regions."""

from __future__ import annotations

from ....region import ProvenanceMatchRule

DEPTHWISE_CAUSAL_CONV1D_PROVENANCE = ProvenanceMatchRule(
    id="prov-depthwise-causal-conv1d",
    kind="region/depthwise_causal_conv1d",
    priority=10,
    component_type="DepthwiseCausalConv1d",
    require_contiguous_in_graph_order=True,
)
