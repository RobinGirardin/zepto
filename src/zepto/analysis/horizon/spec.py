"""Horizon step specification for multi-invocation simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..optimizer import OptimizerPolicy


@dataclass(frozen=True, slots=True)
class HorizonStep:
    """One invocation in a horizon timeline."""

    name: str
    seq_len: int
    batch: int = 1
    phase: str = "forward"
    context_overrides: Mapping[str, object] = field(default_factory=dict)


@dataclass
class HorizonSpec:
    """Builder for ordered horizon invocation steps."""

    steps: list[HorizonStep] = field(default_factory=list)
    optimizer_policy: OptimizerPolicy | None = field(default=None, repr=False)

    def prefill(self, seq_len: int, *, batch: int = 1) -> HorizonSpec:
        self.steps.append(HorizonStep("prefill", seq_len, batch, "forward"))
        return self

    def decode(self, steps: int, *, batch: int = 1) -> HorizonSpec:
        for i in range(steps):
            self.steps.append(HorizonStep(f"decode_{i}", 1, batch, "forward"))
        return self

    def grad_accum(
        self, micro_batches: int, seq_len: int, *, batch: int = 1
    ) -> HorizonSpec:
        for i in range(micro_batches):
            self.steps.append(
                HorizonStep(f"micro_forward_{i}", seq_len, batch, "forward")
            )
        return self

    def backward_step(self, seq_len: int, *, batch: int = 1) -> HorizonSpec:
        self.steps.append(HorizonStep("backward", seq_len, batch, "backward"))
        return self

    def training_step(
        self,
        seq_len: int,
        *,
        micro_batches: int = 1,
        batch: int = 1,
        optimizer: OptimizerPolicy | None = None,
    ) -> HorizonSpec:
        self.grad_accum(micro_batches, seq_len, batch=batch)
        self.backward_step(seq_len, batch=batch)
        if optimizer is not None:
            self.optimizer_policy = optimizer
            self.steps.append(
                HorizonStep("optimizer", seq_len, batch, "optimizer")
            )
        return self
