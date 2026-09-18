"""Discovery rules for fused Mamba-2 mixer block regions.

Subsumption (step5 §8.1): when ``region/mamba2_mixer`` wins overlap resolution,
do not also bill ``region/linear`` (in_proj / o_proj on the same mixer instance),
``region/depthwise_causal_conv1d/*``, ``region/mamba2_scan/*``, or
``region/gated_grouped_rms_norm/*`` on overlapping ops.
"""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

MAMBA2_MIXER_PROVENANCE = ProvenanceMatchRule(
    id="prov-mamba2-mixer",
    kind="region/mamba2_mixer",
    priority=10,
    component_type="Mamba2Mixer",
    module_path_suffix=(),
    module_path_envelope=True,
    require_contiguous_in_graph_order=False,
)

MAMBA2_MIXER_PATTERN = PatternMatchRule(
    id="pat-mamba2-mixer-decomposed",
    kind="region/mamba2_mixer",
    priority=5,
    op_families=(
        "linear_matmul",
        "split",
        "transpose",
        "depthwise_causal_conv1d",
        "reshape",
        "selective_ssm_scan",
        "gated_grouped_rms_norm",
    ),
)
