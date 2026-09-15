"""Horizon step specification for multi-invocation simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from zepto.semantic.metadata import DType

from ..optimizer import AdamW, OptimizerPolicy

if TYPE_CHECKING:
    from .state import StatePortRegistry


class StepKind(StrEnum):
    """Semantic kind of one horizon invocation step."""

    PREFILL = "prefill"
    DECODE = "decode"
    MICRO_FORWARD = "micro_forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    GENERIC = "generic"


@dataclass(frozen=True, slots=True)
class KVConfig:
    """KV cache geometry declared on a horizon spec."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class HorizonStep:
    """One invocation in a horizon timeline."""

    kind: StepKind
    seq_len: int
    batch: int = 1
    phase: str = "forward"
    name: str = ""
    attention_backend: str | None = None
    decode_index: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self._default_name())

    def _default_name(self) -> str:
        if self.kind == StepKind.DECODE and self.decode_index is not None:
            return f"decode_{self.decode_index}"
        return self.kind.value


def decode_step(
    index: int,
    *,
    batch: int = 1,
    attention_backend: str | None = None,
) -> HorizonStep:
    """Build one decode step with optional attention backend override."""
    return HorizonStep(
        kind=StepKind.DECODE,
        seq_len=1,
        batch=batch,
        phase="forward",
        attention_backend=attention_backend,
        decode_index=index,
    )


def optimizer_step(*, seq_len: int, batch: int = 1) -> HorizonStep:
    """Build one optimizer boundary step.

    .. deprecated::
        Optimizer costs are applied at the training boundary in reducers,
        not as a simulated horizon step. Prefer ``HorizonSpec.training()``
        without an explicit optimizer step.
    """
    return HorizonStep(
        kind=StepKind.OPTIMIZER,
        seq_len=seq_len,
        batch=batch,
        phase="forward",
        name="optimizer",
    )


@dataclass
class HorizonSpec:
    """Builder for ordered horizon invocation steps."""

    steps: list[HorizonStep] = field(default_factory=list)
    kv: KVConfig | None = None
    optimizer_policy: OptimizerPolicy | None = field(default=None, repr=False)

    @classmethod
    def inference(
        cls,
        *,
        prefill: int,
        decode_steps: int = 0,
        batch: int = 1,
        kv: KVConfig | None = None,
        decode_attention_backend: str | None = None,
    ) -> HorizonSpec:
        """Prefill plus optional decode timeline with optional KV state."""
        spec = cls(kv=kv)
        spec.prefill(prefill, batch=batch)
        if decode_steps > 0:
            spec.decode(
                decode_steps,
                batch=batch,
                attention_backend=decode_attention_backend,
            )
        return spec

    @classmethod
    def training(
        cls,
        *,
        seq_len: int,
        micro_batches: int = 1,
        batch: int = 1,
        optimizer: OptimizerPolicy | None = AdamW,
    ) -> HorizonSpec:
        """One training cycle: micro-batch forwards + backward (+ optimizer at reducer boundary)."""
        spec = cls(optimizer_policy=optimizer)
        spec.grad_accum(micro_batches, seq_len, batch=batch)
        spec.backward_step(seq_len, batch=batch)
        return spec

    @classmethod
    def repeat(
        cls,
        invocations: int,
        *,
        seq_len: int,
        batch: int = 1,
        phase: str = "forward",
    ) -> HorizonSpec:
        """Multiple invocations with no cross-step persistent state."""
        spec = cls()
        for index in range(invocations):
            spec.steps.append(
                HorizonStep(
                    kind=StepKind.GENERIC,
                    seq_len=seq_len,
                    batch=batch,
                    phase=phase,
                    name=f"invoke_{index}",
                )
            )
        return spec

    def build_state_registry(self) -> StatePortRegistry:
        """Construct the initial cross-step state registry from spec fields."""
        from .state import StatePortRegistry

        registry = StatePortRegistry.empty()
        if self.kv is not None:
            registry = registry.configure_kv(
                num_layers=self.kv.num_layers,
                num_kv_heads=self.kv.num_kv_heads,
                head_dim=self.kv.head_dim,
                dtype=self.kv.dtype,
            )
        if self.optimizer_policy is not None:
            registry = replace(registry, optimizer_policy=self.optimizer_policy)
        return registry

    def prefill(self, seq_len: int, *, batch: int = 1) -> HorizonSpec:
        self.steps.append(
            HorizonStep(
                kind=StepKind.PREFILL,
                seq_len=seq_len,
                batch=batch,
                phase="forward",
                name="prefill",
            )
        )
        return self

    def decode(
        self,
        steps: int,
        *,
        batch: int = 1,
        attention_backend: str | None = None,
    ) -> HorizonSpec:
        for index in range(steps):
            self.steps.append(
                decode_step(index, batch=batch, attention_backend=attention_backend)
            )
        return self

    def grad_accum(
        self, micro_batches: int, seq_len: int, *, batch: int = 1
    ) -> HorizonSpec:
        for index in range(micro_batches):
            self.steps.append(
                HorizonStep(
                    kind=StepKind.MICRO_FORWARD,
                    seq_len=seq_len,
                    batch=batch,
                    phase="forward",
                    name=f"micro_forward_{index}",
                )
            )
        return self

    def backward_step(self, seq_len: int, *, batch: int = 1) -> HorizonSpec:
        self.steps.append(
            HorizonStep(
                kind=StepKind.BACKWARD,
                seq_len=seq_len,
                batch=batch,
                phase="backward",
                name="backward",
            )
        )
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
        return self
