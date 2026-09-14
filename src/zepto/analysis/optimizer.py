"""Optimizer policy contracts for horizon training steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    """Describe optimizer state and update-time workspace."""

    name: str
    state_bytes_per_parameter: int
    workspace_bytes: int = 0


AdamW = OptimizerPolicy(name="adamw", state_bytes_per_parameter=8)
SGD = OptimizerPolicy(name="sgd", state_bytes_per_parameter=0)
