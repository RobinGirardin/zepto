"""Discovery rules for fused GQA prefill attention (boundary C)."""

from __future__ import annotations

from ....region import PatternMatchRule

# Attention core inside GroupedQueryAttention: RepeatKV×2 → QKᵀ → masked softmax → PV.
GQA_PATTERN = PatternMatchRule(
    id="pat-gqa-flash-core",
    kind="region/gqa",
    priority=10,
    op_families=(
        "repeat_kv",
        "repeat_kv",
        "transpose",
        "matmul",
        "add",
        "multiply",
        "exp",
        "reduce_sum",
        "divide",
        "matmul",
    ),
)

GQA_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (GQA_PATTERN.op_families,)
