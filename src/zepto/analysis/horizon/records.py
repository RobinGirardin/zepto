"""Horizon simulation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zepto.graph.graph import Graph

if TYPE_CHECKING:
    from ..lowering.context import InvocationContext
    from ..lowered import LoweredGraph
    from .spec import HorizonSpec, HorizonStep
    from .state import StateSnapshot


@dataclass(frozen=True, slots=True)
class InvocationRecord:
    """One composed and lowered invocation within a horizon."""

    step: HorizonStep
    graph: Graph
    lowered: LoweredGraph
    state_after: StateSnapshot


@dataclass(frozen=True, slots=True)
class HorizonSimulation:
    """Full timeline of horizon invocations and carried state."""

    spec: HorizonSpec
    timeline: tuple[InvocationRecord, ...]
    state_initial: StateSnapshot
    state_final: StateSnapshot
    base_context: InvocationContext
