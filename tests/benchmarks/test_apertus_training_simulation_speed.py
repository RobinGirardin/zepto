"""Wall-clock benchmark: full Apertus 8B training-step cost simulation.

Prefer the CLI (no pytest flags):

    python -m zepto.benchmark smoke
    python -m zepto.benchmark apertus-8b-training --json /tmp/bench.json

Pytest (opt-in):

    pytest tests/benchmarks/test_apertus_training_simulation_speed.py -s -m benchmark -o addopts=
"""

from __future__ import annotations

import os

import pytest

from zepto.analysis.horizon.spec import StepKind
from zepto.benchmark.apertus_training import (
    DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    WallClockLimitExceeded,
    run_apertus_8b_training_horizon,
    run_apertus_training_smoke,
)


@pytest.mark.benchmark
def test_apertus_training_simulation_timing_harness_smoke() -> None:
    """Two-layer Apertus: validates the timing harness completes quickly."""
    result = run_apertus_training_smoke()
    timing = result.timing
    assert len(timing.step_timings) == 3
    assert timing.total_seconds > 0
    assert timing.step_timings[0].discover_seconds > 0
    assert timing.metadata.get("cache_misses") == 2
    opt_timing = timing.step_timings[2]
    assert opt_timing.compose_seconds == 0.0
    assert opt_timing.lower_seconds == 0.0


@pytest.mark.benchmark
def test_apertus_8b_training_horizon_simulation_speed(capfd) -> None:
    """End-to-end training horizon on official 8B widths; abort after 5 minutes."""
    try:
        result = run_apertus_8b_training_horizon(
            wall_clock_limit_seconds=DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
        )
    except WallClockLimitExceeded as exc:
        print(f"\n--- timed out: {exc} ---")
        pytest.fail(
            f"Apertus 8B training simulation exceeded "
            f"{DEFAULT_WALL_CLOCK_LIMIT_SECONDS:.0f}s"
        )

    sim = result.sim
    timing = result.timing

    assert len(sim.timeline) == len(sim.spec.steps)
    assert timing.total_seconds < DEFAULT_WALL_CLOCK_LIMIT_SECONDS

    last = sim.timeline[-1]
    assert last.step.kind == StepKind.OPTIMIZER
    assert len(last.lowered.nodes) == 0

    summary = timing.format_summary()
    print("\n--- Apertus 8B training simulation timing ---")
    print(summary)

    json_path = os.environ.get("ZEPTO_BENCH_JSON")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write(timing.to_json())

    captured = capfd.readouterr()
    assert "compose=" in captured.out
