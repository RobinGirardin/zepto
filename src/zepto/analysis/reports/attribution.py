"""Attribution rollup slices for memory and FLOP reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttributionSlice:
    """Per-key memory and flop rollup."""

    key: str
    forward_flops: int = 0
    backward_flops: int = 0
    allocated_bytes: int = 0
    peak_bytes: int = 0
    persistent_bytes: int = 0
    workspace_bytes: int = 0
