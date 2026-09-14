"""Horizon simulation driver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from zepto.compose import compose_graph
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph

from ..lowering import lower
from .records import HorizonSimulation, InvocationRecord
from .state import StatePortRegistry

if TYPE_CHECKING:
    from ..lowering.context import InvocationContext
    from ..lowering.registry import LoweringRegistry
    from .spec import HorizonSpec, HorizonStep


if TYPE_CHECKING:
    ComposeFn = Callable[
        [HorizonStep, StatePortRegistry],
        tuple[Callable[..., object] | object, tuple[Tensor, ...]],
    ]
else:
    ComposeFn = Callable[..., tuple[Callable[..., object] | object, tuple[Tensor, ...]]]


def simulate_horizon(
    compose_fn: ComposeFn,
    spec: HorizonSpec,
    base_context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
    state: StatePortRegistry | None = None,
) -> HorizonSimulation:
    """Compose and lower one graph per horizon step, advancing carried state."""
    port_registry = state if state is not None else StatePortRegistry.empty()
    if spec.optimizer_policy is not None:
        port_registry = replace(
            port_registry, optimizer_policy=spec.optimizer_policy
        )

    timeline: list[InvocationRecord] = []
    initial = port_registry.snapshot()

    for step in spec.steps:
        module_factory, inputs = compose_fn(step, port_registry)
        graph = _compose_step(module_factory, inputs)
        ctx = _merge_context(step, base_context, port_registry)
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
        base_context=base_context,
    )


def _compose_step(
    module_factory: Callable[..., object] | object,
    inputs: tuple[Tensor, ...],
) -> Graph:
    if callable(module_factory):
        return compose_graph(module_factory, inputs)
    return compose_graph(module_factory, inputs)


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
    for key, value in step.context_overrides.items():
        if key == "phase" and isinstance(value, str):
            ctx = replace(ctx, phase=value)
        elif key == "attention_backend" and isinstance(value, str):
            ctx = replace(ctx, attention_backend=value)
    return ctx
