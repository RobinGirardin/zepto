"""Discovery rules for fused GQA with sink softmax (GPT-OSS boundary C)."""

from __future__ import annotations

from ....region import PatternMatchRule

GQA_SINK_PATTERN = PatternMatchRule(
    id="pat-gqa-sink-flash-core",
    kind="region/gqa-sink",
    priority=10,
    op_families=(
        "repeat_kv",
        "repeat_kv",
        "transpose",
        "matmul",
        "add",
        "attention_softmax_with_sink",
        "matmul",
    ),
)

GQA_SINK_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (
    GQA_SINK_PATTERN.op_families,
)
