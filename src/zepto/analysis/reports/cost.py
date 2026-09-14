"""Combined cost report types."""

from __future__ import annotations

from dataclasses import dataclass

from ..lowering.context import InvocationContext
from .flops import FlopReport
from .memory import MemoryReport


@dataclass(frozen=True, slots=True)
class CostReport:
    """Memory and FLOP accounting for one invocation."""

    memory: MemoryReport
    flops: FlopReport
    context: InvocationContext
