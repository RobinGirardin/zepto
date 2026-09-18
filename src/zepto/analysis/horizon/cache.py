"""Horizon structural cache helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..lowering.context import InvocationContext
from ..lowering.structural import StructuralLowering
from .spec import HorizonStep, StepKind
from zepto.graph.graph import Graph


@dataclass(frozen=True, slots=True)
class HorizonStructuralKey:
    """Fields that affect region discovery and implementation selection."""

    step_kind: StepKind
    seq_len: int
    batch: int
    phase: str
    attention_backend: str
    hardware: str
    backend: str
    allow_fallback: bool
    requested_capabilities: frozenset[str]
    implementation_pins: tuple[tuple[str, str], ...]
    module_implementation_pins: tuple[tuple[str, str], ...]
    region_implementation_pins: tuple[tuple[str, str], ...]
    precision_key: tuple[object, ...]


@dataclass
class StructuralCacheEntry:
    """One composed graph and its structural lowering artifacts."""

    graph: Graph
    structural: StructuralLowering


@dataclass
class HorizonStructuralCacheStats:
    """Per-run structural cache counters (for benchmarks and tests)."""

    cache_hits: int = 0
    cache_misses: int = 0
    compose_skipped: int = 0


@dataclass
class HorizonStepDurations:
    """Wall times for one horizon step (benchmark instrumentation)."""

    compose_seconds: float = 0.0
    discover_seconds: float = 0.0
    lower_seconds: float = 0.0
    step_total_seconds: float = 0.0


def _precision_key(context: InvocationContext) -> tuple[object, ...]:
    policy = context.precision
    return (
        policy.default_dtype,
        policy.by_role,
        policy.by_semantic_type,
        context.accounting,
    )


def make_structural_key(
    step: HorizonStep,
    context: InvocationContext,
) -> HorizonStructuralKey:
    """Build a cache key from step and merged context (state excluded)."""
    return HorizonStructuralKey(
        step_kind=step.kind,
        seq_len=step.seq_len,
        batch=step.batch,
        phase=context.phase,
        attention_backend=context.attention_backend,
        hardware=context.hardware,
        backend=context.backend,
        allow_fallback=context.allow_fallback,
        requested_capabilities=context.requested_capabilities,
        implementation_pins=tuple(sorted(context.implementation_pins.items())),
        module_implementation_pins=tuple(
            sorted(context.module_implementation_pins.items())
        ),
        region_implementation_pins=tuple(
            sorted(context.region_implementation_pins.items())
        ),
        precision_key=_precision_key(context),
    )
