"""Apertus training-horizon simulation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.modules.models.apertus import APERTUS_8B, Apertus
from zepto.analysis import HorizonSpec, reference_invocation
from zepto.analysis.horizon.records import HorizonSimulation
from zepto.analysis.optimizer import AdamW
from zepto.benchmark.horizon_timing import (
    HorizonSimulationTiming,
    WallClockLimitExceeded,
    simulate_horizon_timed,
    wall_clock_limit,
)
from zepto.compose import Tensor
from zepto.semantic.metadata import DType

DEFAULT_WALL_CLOCK_LIMIT_SECONDS = 300.0
TRAINING_SEQ_LEN = 2048
TRAINING_BATCH = 1
MICRO_BATCHES = 1


def flash_training_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
        "default_dtype": DType.FP16,
        "phase": "forward",
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def apertus_8b_module(_compose):
    return Apertus.apertus_8b(TRAINING_SEQ_LEN)


def apertus_8b_inputs(_step, _ctx, _state):
    return (Tensor(shape=(TRAINING_SEQ_LEN,)),)


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """One completed benchmark run."""

    sim: HorizonSimulation
    timing: HorizonSimulationTiming


def run_apertus_training_smoke(
    *,
    profile_discover_on_first_step: bool = True,
) -> BenchmarkRunResult:
    """Tiny two-layer Apertus; validates timing harness (~1s)."""
    seq_len = 64
    spec = HorizonSpec.training(seq_len=seq_len, micro_batches=1, optimizer=AdamW)
    ctx = flash_training_context()

    def module_fn(_compose):
        return Apertus(
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,
            vocab_size=1024,
            seq_len=seq_len,
        )

    def inputs_fn(_step, _ctx, _state):
        return (Tensor(shape=(seq_len,)),)

    sim, timing = simulate_horizon_timed(
        spec,
        module_fn,
        inputs_fn,
        ctx,
        metadata={"model": "Apertus-tiny", "num_layers": 2},
        profile_discover_on_first_step=profile_discover_on_first_step,
    )
    return BenchmarkRunResult(sim=sim, timing=timing)


def run_apertus_8b_training_horizon(
    *,
    wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
) -> BenchmarkRunResult:
    """Full Apertus 8B training horizon; raises on wall-clock timeout."""
    spec = HorizonSpec.training(
        seq_len=TRAINING_SEQ_LEN,
        micro_batches=MICRO_BATCHES,
        batch=TRAINING_BATCH,
        optimizer=AdamW,
    )
    ctx = flash_training_context()

    limit = wall_clock_limit(wall_clock_limit_seconds)
    try:
        sim, timing = simulate_horizon_timed(
            spec,
            apertus_8b_module,
            apertus_8b_inputs,
            ctx,
            metadata={
                "model": "Apertus-8B",
                "num_layers": APERTUS_8B.num_layers,
                "seq_len": TRAINING_SEQ_LEN,
                "micro_batches": MICRO_BATCHES,
                "horizon_steps": len(spec.steps),
            },
        )
    except WallClockLimitExceeded:
        raise
    finally:
        limit.disarm()  # type: ignore[attr-defined]

    return BenchmarkRunResult(sim=sim, timing=timing)


__all__ = [
    "BenchmarkRunResult",
    "DEFAULT_WALL_CLOCK_LIMIT_SECONDS",
    "WallClockLimitExceeded",
    "run_apertus_8b_training_horizon",
    "run_apertus_training_smoke",
]
