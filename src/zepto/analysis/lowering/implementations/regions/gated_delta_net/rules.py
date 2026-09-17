"""Discovery rules for fused GatedDeltaNet block regions.

Subsumption (step5 §8.1): when ``region/gated_delta_net`` wins overlap resolution,
do not also bill ``region/linear``, ``region/depthwise_causal_conv1d/*``,
``region/l2_normalize/*``, ``region/gated_delta_scan/*``, or
``region/gated_rms_norm/*`` on overlapping ops.
"""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

GATED_DELTA_NET_PROVENANCE = ProvenanceMatchRule(
    id="prov-gated-delta-net",
    kind="region/gated_delta_net",
    priority=10,
    component_type="GatedDeltaNet",
    module_path_suffix=(),
    module_path_envelope=True,
    require_contiguous_in_graph_order=False,
)

GATED_DELTA_NET_PATTERN = PatternMatchRule(
    id="pat-gated-delta-net-decomposed",
    kind="region/gated_delta_net",
    priority=5,
    op_families=(
        "linear_matmul",
        "depthwise_causal_conv1d",
        "l2_normalize",
        "sigmoid",
        "softplus",
        "gated_delta_scan",
        "gated_rms_norm",
        "repeat_kv",
        "reshape",
        "split",
        "concat",
        "transpose",
    ),
)
