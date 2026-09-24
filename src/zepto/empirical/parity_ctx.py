"""Zepto InvocationContext parity with HF eager twin."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import PrecisionPolicy, reference_invocation
from zepto.semantic.metadata import DType

# ``requested ⊆ impl.capabilities``. ``{"fused"}`` lets RMSNorm / xIELU /
# Softmax / linear-CE reference leaves win. ``sdpa``+``gqa`` is a GQA
# profile that those ATen leaves cannot cover. Attention stays eager
# identity (no Flash / SDPA). Pins lock ``*/reference`` so the study
# names eager ATen, not Liger/CUDA variants.
EAGER_ATEN_REGION_PINS = MappingProxyType(
    {
        "region/rmsnorm": "region/rmsnorm/reference",
        "region/xielu": "region/xielu/reference",
        "region/softmax": "region/softmax/reference",
        "region/linear_ce": "region/linear_ce/reference",
    }
)


def build_invocation_context(
    precision: str,
    *,
    cuda_capability: tuple[int, int],
):
    # Vanilla AdamW on a uniform-dtype twin: moments follow param_prec.
    optim_prec = 2 if precision == "fp16" else 4
    base = dict(
        optim_prec=optim_prec,
        attention_backend="eager",
        hardware="cuda",
        compute_capability=cuda_capability,
        requested_capabilities=frozenset({"fused"}),
        region_implementation_pins=EAGER_ATEN_REGION_PINS,
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
