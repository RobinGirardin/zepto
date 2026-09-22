"""Horizon simulation driver."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from zepto.compose import compose_graph
from zepto.compose.context import Compose, Module
from zepto.compose.values import Tensor

from ..lowering.defaults import DEFAULT_REGISTRY
from ..lowering.structural import discover_and_plan, execute_lowering_plan
from .cache import (
    HorizonStepDurations,
    HorizonStructuralCacheStats,
    StructuralCacheEntry,
    make_structural_key,
)
from .optimizer_step import (
    make_optimizer_stub_lowered,
    reference_lowered_for_optimizer,
)
from .records import HorizonSimulation, InvocationRecord
from .spec import HorizonSpec, HorizonStep, StepKind
from .state import trainable_parameter_elements
from .training_boundary import training_boundary_cost

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
    """Root input ``(step.batch, step.seq_len, hidden)`` for hidden-state modules.

    Always rank-3, including ``batch=1`` (not the ``(S, H)`` alias). Use this
    for decoder **blocks** and other modules whose first argument is a hidden
    activation. Full Embedding-first models need :func:`inputs_from_token_ids`.

    The leading dims **must** match ``step.batch`` and ``step.seq_len``. A
    mismatch is a silent caller bug: ``simulate_horizon`` will not error.
    """

    def _inputs(
        step: HorizonStep,
        _ctx: InvocationContext,
        _state: StateSnapshot,
    ) -> tuple[Tensor, ...]:
        return (Tensor(shape=(step.batch, step.seq_len, hidden)),)

    return _inputs


def inputs_from_token_ids() -> InputsFn:
    """Root input ``(step.batch, step.seq_len)`` for Embedding-first models.

    Always rank-2, including ``batch=1`` (not the ``(S,)`` alias). Pair with
    ``HorizonSpec.training(..., batch=B, micro_batches=1)`` for HuggingFace
    ``per_device_train_batch_size=B``.

    The leading dims **must** match ``step.batch`` and ``step.seq_len``. A
    mismatch is a silent caller bug: ``simulate_horizon`` will not error.
    """

    def _inputs(
        step: HorizonStep,
        _ctx: InvocationContext,
        _state: StateSnapshot,
    ) -> tuple[Tensor, ...]:
        return (
            Tensor(
                shape=(step.batch, step.seq_len),
                semantic_type="token_ids",
                requires_grad=False,
            ),
        )

    return _inputs


def simulate_horizon(
    spec: HorizonSpec,
    module_fn: ModuleFn,
    inputs_fn: InputsFn,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
    use_structural_cache: bool = True,
    cache_stats: HorizonStructuralCacheStats | None = None,
    step_durations: list[HorizonStepDurations] | None = None,
    profile_discover_on_first_step: bool = False,
) -> HorizonSimulation:
    """Compose and lower one graph per horizon step, advancing carried state."""
    port_registry = spec.build_state_registry()
    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    structural_cache: dict[object, StructuralCacheEntry] = {}
    stats = cache_stats if cache_stats is not None else HorizonStructuralCacheStats()

    timeline: list[InvocationRecord] = []
    initial = port_registry.snapshot()

    for step_index, step in enumerate(spec.steps):
        step_start = time.perf_counter()
        durations = HorizonStepDurations() if step_durations is not None else None

        if step.kind == StepKind.OPTIMIZER:
            reference = reference_lowered_for_optimizer(timeline)
            ctx = _merge_context(step, context, port_registry)
            lowered = make_optimizer_stub_lowered(reference, ctx)
            graph = timeline[-1].graph
            port_registry = port_registry.advance(step, lowered)
            timeline.append(
                InvocationRecord(
                    step=step,
                    graph=graph,
                    lowered=lowered,
                    state_after=port_registry.snapshot(),
                )
            )
            stats.state_only_steps += 1
            if step_durations is not None and durations is not None:
                durations.step_total_seconds = time.perf_counter() - step_start
                step_durations.append(durations)
            continue

        ctx = _merge_context(step, context, port_registry)
        snapshot = port_registry.snapshot()
        key = make_structural_key(step, ctx)

        if use_structural_cache and key in structural_cache:
            entry = structural_cache[key]
            stats.cache_hits += 1
            stats.compose_skipped += 1
            graph = entry.graph
            t_lower = time.perf_counter()
            lowered = execute_lowering_plan(
                graph, ctx, active_registry, entry.structural
            )
            if durations is not None:
                durations.lower_seconds = time.perf_counter() - t_lower
        else:
            inputs = inputs_fn(step, ctx, snapshot)
            t_compose = time.perf_counter()
            graph = compose_graph(module_fn, inputs)
            if durations is not None:
                durations.compose_seconds = time.perf_counter() - t_compose

            if use_structural_cache:
                t_discover = time.perf_counter()
                structural = discover_and_plan(graph, ctx, active_registry)
                discover_elapsed = time.perf_counter() - t_discover
                if (
                    durations is not None
                    and profile_discover_on_first_step
                    and step_index == 0
                ):
                    durations.discover_seconds = discover_elapsed
                t_lower = time.perf_counter()
                lowered = execute_lowering_plan(
                    graph, ctx, active_registry, structural
                )
                if durations is not None:
                    durations.lower_seconds = time.perf_counter() - t_lower
                structural_cache[key] = StructuralCacheEntry(
                    graph=graph, structural=structural
                )
                stats.cache_misses += 1
            else:
                from ..lowering import lower

                t_lower = time.perf_counter()
                lowered = lower(graph, ctx, registry=active_registry)
                if durations is not None:
                    durations.lower_seconds = time.perf_counter() - t_lower

        port_registry = port_registry.advance(step, lowered)
        if step.kind == StepKind.BACKWARD and spec.optimizer_policy is not None:
            trainable = trainable_parameter_elements(lowered)
            cost = training_boundary_cost(
                policy=spec.optimizer_policy,
                trainable_elements=trainable,
                context=ctx,
            )
            port_registry = port_registry.with_optimizer_state(
                spec.optimizer_policy, cost.optimizer_state_bytes
            )
        timeline.append(
            InvocationRecord(
                step=step,
                graph=graph,
                lowered=lowered,
                state_after=port_registry.snapshot(),
            )
        )
        if step_durations is not None and durations is not None:
            durations.step_total_seconds = time.perf_counter() - step_start
            step_durations.append(durations)

    if cache_stats is not None:
        cache_stats.cache_hits = stats.cache_hits
        cache_stats.cache_misses = stats.cache_misses
        cache_stats.compose_skipped = stats.compose_skipped
        cache_stats.state_only_steps = stats.state_only_steps

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
