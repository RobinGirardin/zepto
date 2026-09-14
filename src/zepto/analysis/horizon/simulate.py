"""Horizon simulation driver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from zepto.compose import compose_graph
from zepto.compose.context import Compose, Module
from zepto.compose.values import Tensor

from ..lowering import lower
from .records import HorizonSimulation, InvocationRecord
from .spec import HorizonSpec, HorizonStep, StepKind

if TYPE_CHECKING:
    from ..lowering.context import InvocationContext
    from ..lowering.registry import LoweringRegistry
    from .state import StatePortRegistry, StateSnapshot

if TYPE_CHECKING:
    ModuleFn = Module | Callable[[Compose], Module]
    InputsFn = Callable[
        [HorizonStep, InvocationContext, StateSnapshot],
        tuple[Tensor, ...],
    ]
else:
    ModuleFn = Callable[..., Module]
    InputsFn = Callable[..., tuple[Tensor, ...]]


def inputs_from_shape(hidden: int) -> InputsFn:
    """Build an inputs_fn for a single (batch, seq, hidden) root input."""

    def _inputs(
        step: HorizonStep,
        _ctx: InvocationContext,
        _state: StateSnapshot,
    ) -> tuple[Tensor, ...]:
        return (Tensor(shape=(step.batch, step.seq_len, hidden)),)

    return _inputs


def simulate_horizon(
    spec: HorizonSpec,
    module_fn: ModuleFn,
    inputs_fn: InputsFn,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> HorizonSimulation:
    """Compose and lower one graph per horizon step, advancing carried state."""
    port_registry = spec.build_state_registry()

    timeline: list[InvocationRecord] = []
    initial = port_registry.snapshot()

    for step in spec.steps:
        ctx = _merge_context(step, context, port_registry)
        snapshot = port_registry.snapshot()
        inputs = inputs_fn(step, ctx, snapshot)
        graph = compose_graph(module_fn, inputs)
        lowered = lower(graph, ctx, registry=registry)
        port_registry = port_registry.advance(step, lowered)
        timeline.append(
            InvocationRecord(
                step=step,
                graph=graph,
                lowered=lowered,
                state_after=port_registry.snapshot(),
            )
        )

    return HorizonSimulation(
        spec=spec,
        timeline=tuple(timeline),
        state_initial=initial,
        state_final=port_registry.snapshot(),
        base_context=context,
    )


def _merge_context(
    step: HorizonStep,
    base: InvocationContext,
    state: StatePortRegistry,
) -> InvocationContext:
    ctx = state.bind_to_context(base)
    phase = step.phase
    if phase not in ("forward", "backward", "full"):
        phase = "forward"
    ctx = replace(ctx, phase=phase)
    if step.kind == StepKind.DECODE and step.attention_backend is not None:
        ctx = replace(ctx, attention_backend=step.attention_backend)
    return ctx
