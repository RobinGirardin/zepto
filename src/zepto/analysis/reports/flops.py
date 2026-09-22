"""FLOP report types."""

from __future__ import annotations

from dataclasses import dataclass

from .attribution import AttributionSlice


@dataclass(frozen=True, slots=True)
class FlopReport:
    """FLOP accounting result for one invocation."""

    forward_flops: int
    backward_flops: int
    total_flops: int
    by_module: tuple[AttributionSlice, ...] = ()
    by_region: tuple[AttributionSlice, ...] = ()
    by_implementation: tuple[AttributionSlice, ...] = ()
    flop_policy: str = "zepto"
    zepto_forward_flops: int = 0
    zepto_backward_flops: int = 0
    zepto_total_flops: int = 0
    fcm_forward_flops: int = 0
    fcm_backward_flops: int = 0
    fcm_total_flops: int = 0
