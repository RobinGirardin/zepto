"""Apertus training-horizon simulation benchmarks.

HuggingFace mapping: the ``batch`` argument is ``HorizonStep.batch``
(``per_device_train_batch_size``), not ``micro_batches``. Use
``HorizonSpec.training(seq_len=S, batch=B, micro_batches=1)`` for a single
parallel ``(B, S)`` training step. Token roots are always rank-2
``(step.batch, step.seq_len)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from zepto.modules.models.apertus import APERTUS_8B, Apertus
from zepto.analysis import HorizonSpec, inputs_from_token_ids, reference_invocation
from zepto.analysis.horizon.records import HorizonSimulation
from zepto.analysis.optimizer import AdamW
from zepto.benchmark.horizon_timing import (
    HorizonSimulationTiming,
    WallClockLimitExceeded,
    simulate_horizon_timed,
    wall_clock_limit,
)
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


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """One completed benchmark run."""

    sim: HorizonSimulation
    timing: HorizonSimulationTiming


def run_apertus_training_smoke(
    *,
    profile_discover_on_first_step: bool = True,
    batch: int = TRAINING_BATCH,
) -> BenchmarkRunResult:
    """Tiny two-layer Apertus; validates timing harness (~1s)."""
    seq_len = 64
    spec = HorizonSpec.training(
        seq_len=seq_len,
        micro_batches=1,
        batch=batch,
        optimizer=AdamW,
    )
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

    sim, timing = simulate_horizon_timed(
        spec,
        module_fn,
        inputs_from_token_ids(),
        ctx,
        metadata={"model": "Apertus-tiny", "num_layers": 2, "batch": batch},
        profile_discover_on_first_step=profile_discover_on_first_step,
    )
    return BenchmarkRunResult(sim=sim, timing=timing)


def run_apertus_8b_training_horizon(
    *,
    wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    batch: int = TRAINING_BATCH,
) -> BenchmarkRunResult:
    """Full Apertus 8B training horizon; raises on wall-clock timeout."""
    spec = HorizonSpec.training(
        seq_len=TRAINING_SEQ_LEN,
        micro_batches=MICRO_BATCHES,
        batch=batch,
        optimizer=AdamW,
    )
    ctx = flash_training_context()

    limit = wall_clock_limit(wall_clock_limit_seconds)
    try:
        sim, timing = simulate_horizon_timed(
            spec,
            apertus_8b_module,
            inputs_from_token_ids(),
            ctx,
            metadata={
                "model": "Apertus-8B",
                "num_layers": APERTUS_8B.num_layers,
                "seq_len": TRAINING_SEQ_LEN,
                "batch": batch,
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
    "TRAINING_BATCH",
    "WallClockLimitExceeded",
    "run_apertus_8b_training_horizon",
    "run_apertus_training_smoke",
]
