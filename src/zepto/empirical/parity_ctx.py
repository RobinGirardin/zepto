"""Zepto InvocationContext parity with HF eager twin."""

from __future__ import annotations

from zepto.analysis import PrecisionPolicy, reference_invocation
from zepto.semantic.metadata import DType


def build_invocation_context(
    precision: str,
    *,
    cuda_capability: tuple[int, int],
):
    base = dict(
        optim_prec=4,
        attention_backend="eager",
        hardware="cuda",
        compute_capability=cuda_capability,
        requested_capabilities=frozenset({"fused", "sdpa", "gqa"}),
    )
    if precision == "fp32":
        return reference_invocation(phase="forward", default_dtype=DType.FP32, **base)
    if precision == "fp16":
        return reference_invocation(
            phase="forward",
            default_dtype=DType.FP16,
            precision=PrecisionPolicy.from_byte_sizes(param_bytes=2, grad_bytes=2),
            **base,
        )
    raise ValueError(f"unknown precision: {precision!r}")
