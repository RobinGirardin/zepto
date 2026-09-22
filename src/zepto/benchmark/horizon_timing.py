"""Timing helpers for horizon simulation benchmarks."""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from zepto.analysis.flops import account_flops
from zepto.analysis.memory import account_memory

if TYPE_CHECKING:
    from zepto.analysis.horizon.records import HorizonSimulation
    from zepto.analysis.horizon.simulate import InputsFn, ModuleFn
    from zepto.analysis.horizon.spec import HorizonSpec
    from zepto.analysis.lowering import LoweringRegistry
    from zepto.analysis.lowering.context import InvocationContext


@dataclass(frozen=True, slots=True)
class HorizonStepTiming:
    """Wall time for one horizon step and its lowering substeps."""

    step_index: int
    step_name: str
    step_kind: str
    seq_len: int
    phase: str
    compose_seconds: float
    discover_seconds: float
    lower_seconds: float
    step_total_seconds: float


@dataclass(frozen=True, slots=True)
class HorizonSimulationTiming:
    """Aggregated timing for a full instrumented horizon run."""

    step_timings: tuple[HorizonStepTiming, ...]
    account_memory_seconds: float
    account_flops_seconds: float
    total_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            {
                "metadata": self.metadata,
                "total_seconds": self.total_seconds,
                "account_memory_seconds": self.account_memory_seconds,
                "account_flops_seconds": self.account_flops_seconds,
                "steps": [asdict(step) for step in self.step_timings],
            },
            indent=indent,
        )

    def format_summary(self) -> str:
        lines = [
            f"total={self.total_seconds:.3f}s "
            f"(memory={self.account_memory_seconds:.3f}s, "
            f"flops={self.account_flops_seconds:.3f}s)",
        ]
        meta = self.metadata
        if "cache_hits" in meta:
            state_only = meta.get("state_only_steps")
            cache_line = (
                f"cache: hits={meta.get('cache_hits', 0)} "
                f"misses={meta.get('cache_misses', 0)} "
                f"compose_skipped={meta.get('compose_skipped', 0)}"
            )
            if state_only is not None:
                cache_line += f" state_only={state_only}"
            lines.append(cache_line)
        compose = sum(s.compose_seconds for s in self.step_timings)
        discover = sum(s.discover_seconds for s in self.step_timings)
        lower = sum(s.lower_seconds for s in self.step_timings)
        lines.append(
            f"steps: compose={compose:.3f}s discover={discover:.3f}s "
            f"lower={lower:.3f}s (n={len(self.step_timings)})"
        )
        for step in self.step_timings:
            lines.append(
                f"  [{step.step_index}] {step.step_name}: "
                f"compose={step.compose_seconds:.3f}s "
                f"discover={step.discover_seconds:.3f}s "
                f"lower={step.lower_seconds:.3f}s "
                f"total={step.step_total_seconds:.3f}s"
            )
        return "\n".join(lines)


class WallClockLimitExceeded(TimeoutError):
    """Raised when a benchmark exceeds its wall-clock budget."""


def wall_clock_limit(seconds: float) -> Callable[[], None]:
    """Return a callable that raises if the budget is exceeded (POSIX SIGALRM)."""

    def _check() -> None:
        return None

    def _handler(_signum: int, _frame: object) -> None:
        raise WallClockLimitExceeded(
            f"Simulation exceeded wall-clock limit of {seconds:.0f}s"
        )

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    def _disarm() -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    _check.disarm = _disarm  # type: ignore[attr-defined]
    return _check


def simulate_horizon_timed(
    spec: HorizonSpec,
    module_fn: ModuleFn,
    inputs_fn: InputsFn,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
    metadata: dict[str, Any] | None = None,
    profile_discover_on_first_step: bool = True,
) -> tuple[HorizonSimulation, HorizonSimulationTiming]:
    """Like simulate_horizon, recording compose / lower per step."""
    from zepto.analysis.horizon.cache import HorizonStructuralCacheStats
    from zepto.analysis.horizon.simulate import simulate_horizon
    from zepto.analysis.lowering.defaults import DEFAULT_REGISTRY

    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    run_start = time.perf_counter()
    cache_stats = HorizonStructuralCacheStats()
    step_durations: list = []

    sim = simulate_horizon(
        spec,
        module_fn,
        inputs_fn,
        context,
        registry=active_registry,
        cache_stats=cache_stats,
        step_durations=step_durations,
        profile_discover_on_first_step=profile_discover_on_first_step,
    )

    step_timings = [
        HorizonStepTiming(
            step_index=index,
            step_name=step.name,
            step_kind=str(step.kind),
            seq_len=step.seq_len,
            phase=step.phase,
            compose_seconds=dur.compose_seconds,
            discover_seconds=dur.discover_seconds,
            lower_seconds=dur.lower_seconds,
            step_total_seconds=dur.step_total_seconds,
        )
        for index, (step, dur) in enumerate(zip(spec.steps, step_durations))
    ]

    meta = dict(metadata or {})
    meta.update(
        {
            "cache_hits": cache_stats.cache_hits,
            "cache_misses": cache_stats.cache_misses,
            "compose_skipped": cache_stats.compose_skipped,
            "state_only_steps": cache_stats.state_only_steps,
        }
    )

    t_mem = time.perf_counter()
    account_memory(sim)
    account_memory_seconds = time.perf_counter() - t_mem

    t_flops = time.perf_counter()
    account_flops(sim)
    account_flops_seconds = time.perf_counter() - t_flops

    timing = HorizonSimulationTiming(
        step_timings=tuple(step_timings),
        account_memory_seconds=account_memory_seconds,
        account_flops_seconds=account_flops_seconds,
        total_seconds=time.perf_counter() - run_start,
        metadata=meta,
    )
    return sim, timing
