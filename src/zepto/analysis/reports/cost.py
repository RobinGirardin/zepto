"""Combined cost report types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..lowering.context import InvocationContext
from .flops import FlopReport
from .horizon import HorizonFlopReport, HorizonMemoryReport
from .memory import MemoryReport

if TYPE_CHECKING:
    from ..horizon.state import StateSnapshot


@dataclass(frozen=True, slots=True)
class CostReport:
    """Memory and FLOP accounting for one invocation."""

    memory: MemoryReport
    flops: FlopReport
    context: InvocationContext


@dataclass(frozen=True, slots=True)
class HorizonCostReport:
    """Combined horizon memory and FLOP accounting."""

    peak_vram: int
    total_flops: int
    per_step: tuple[CostReport, ...]
    state_final: StateSnapshot
    memory: HorizonMemoryReport
    flops: HorizonFlopReport
