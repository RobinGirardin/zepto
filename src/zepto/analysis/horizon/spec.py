"""Horizon step specification for multi-invocation simulation.

Parallel batch vs gradient accumulation (normative):

| Symbol | Zepto field | HuggingFace analogue | Meaning |
|--------|-------------|----------------------|---------|
| **B** | ``HorizonStep.batch`` | ``per_device_train_batch_size`` (single pass) | Parallel batch width of one compose/lowering: tensor leading dim before ``S``. |
| **S** | ``HorizonStep.seq_len`` | sequence length in batch | Tokens per sequence in that step. |
| **G** | ``HorizonSpec.training(..., micro_batches=G)`` | ``gradient_accumulation_steps`` | Count of sequential ``TRAIN`` (``phase="full"``) micros on ``(b, S)`` before optimizer. |
| **b** | ``batch`` on each train micro | micro-batch size per accum step | Tensor width **per** ``TRAIN`` step. Activations stay width ``b``, not ``G × b``. |

HF single batched training step (no grad accum)::

    batch=B, micro_batches=1  →  one TRAIN step (phase="full") + optimizer

HF with gradient accumulation::

    batch=b, micro_batches=G  →  TRAIN × G on (b,S), then optimizer
    Peak VRAM is max over those TRAIN micros, not the sum.
    ``training()`` does not create ``GradAccumState``.
    Effective batch ≈ G × b (per device) is a FLOP / optimizer-frequency
    fact, not a VRAM fact.

The fluent :meth:`HorizonSpec.grad_accum` builder stays
``MICRO_FORWARD × N`` for custom timelines. It is **not** a HuggingFace
training peak.

Do not conflate ``batch=1, micro_batches=B`` with HF ``(B, S)`` VRAM: that is
``B`` sequential ``(1, S)`` forwards (``micro_sum``), not a parallel batch.

FLOPs sum over steps; **peak VRAM is max over steps**, not the sum of micro
peaks. Training timelines usually omit KV prefill; KV work is the
inference/decode path.

Decode steps should use the same ``batch`` as prefill
(``HorizonSpec.inference`` copies it). ``StatePortRegistry.advance`` on
``DECODE`` grows KV ``seq_len`` only and does not re-read ``step.batch``.
"""

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
    TRAIN = "train"
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
class ConvStateConfig:
    """Depthwise conv rolling-buffer geometry for horizon simulation."""

    layer_indices: tuple[int, ...]
    channels: int
    kernel_size: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class RecurrentStateConfig:
    """Recurrent scan state geometry for horizon simulation."""

    layers: tuple[tuple[int, str, int, int, int], ...]
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
        if self.batch < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch}")
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


def train_step(
    seq_len: int,
    *,
    batch: int = 1,
    index: int | None = None,
) -> HorizonStep:
    """Build one training graph step (forward + backward, ``phase="full"``).

    ``index is None`` names the step ``"train"`` (G=1). Otherwise
    ``"train_{index}"`` (G>1 micros). Do not name the G=1 step
    ``"train_0"``.
    """
    return HorizonStep(
        kind=StepKind.TRAIN,
        seq_len=seq_len,
        batch=batch,
        phase="full",
        name="train" if index is None else f"train_{index}",
    )


def optimizer_step(*, seq_len: int, batch: int = 1) -> HorizonStep:
    """Build one optimizer boundary step.

    Marks a timeline boundary for optimizer state ports (``StatePortRegistry``).
    Simulation does not compose or lower the training module on this step;
    compute FLOPs come from ``OptimizerPolicy.update_flops``. The
    ``phase="forward"`` field is a legacy placeholder—optimizer rows ignore
    model forward/backward phase for lowering and FLOP accounting.
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
    conv: ConvStateConfig | None = None
    recurrent: RecurrentStateConfig | None = None
    optimizer_policy: OptimizerPolicy | None = field(default=None, repr=False)

    @classmethod
    def inference(
        cls,
        *,
        prefill: int,
        decode_steps: int = 0,
        batch: int = 1,
        kv: KVConfig | None = None,
        conv: ConvStateConfig | None = None,
        recurrent: RecurrentStateConfig | None = None,
        decode_attention_backend: str | None = None,
    ) -> HorizonSpec:
        """Prefill plus optional decode timeline with optional KV state.

        Decode steps reuse the same ``batch`` as prefill. ``advance`` grows
        KV ``seq_len`` only; it does not re-apply decode ``step.batch``.
        """
        spec = cls(kv=kv, conv=conv, recurrent=recurrent)
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
        """HuggingFace training peak: ``TRAIN × G`` on ``(b, S)``, then optimizer.

        HuggingFace mapping: ``batch=B, micro_batches=1`` matches
        ``per_device_train_batch_size=B`` — one ``TRAIN`` step at
        ``phase="full"`` plus optimizer. ``batch=b, micro_batches=G``
        matches ``gradient_accumulation_steps=G``: ``G`` sequential
        ``TRAIN`` graphs at width ``b``, then optimizer. Peak VRAM is
        the max over those micros (same as G=1 at the same ``b``).
        Never set ``micro_batches=B`` with ``batch=1`` for HF VRAM
        parity unless explicitly documenting a ``micro_sum``
        sequential-forwards mode.

        This path never installs ``GradAccumState``. Inspect
        ``state_final.grad_accum`` only on timelines built with
        :meth:`grad_accum` (the forward-only escape hatch).
        """
        spec = cls(optimizer_policy=optimizer)
        spec._append_train_micros(
            seq_len, micro_batches=micro_batches, batch=batch
        )
        if optimizer is not None:
            spec.steps.append(optimizer_step(seq_len=seq_len, batch=batch))
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
        if self.conv is not None:
            registry = registry.configure_conv_layers(
                layer_indices=self.conv.layer_indices,
                channels=self.conv.channels,
                kernel_size=self.conv.kernel_size,
                dtype=self.conv.dtype,
            )
        if self.recurrent is not None:
            registry = registry.configure_recurrent_layers(
                specs=self.recurrent.layers,
                dtype=self.recurrent.dtype,
            )
        n_micro = sum(1 for s in self.steps if s.kind == StepKind.MICRO_FORWARD)
        registry = replace(registry, grad_accum_steps=n_micro)
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
        """Append sequential forward-only micros (not a training peak).

        Stays ``MICRO_FORWARD × N`` at ``phase="forward"`` for custom
        timelines. HuggingFace gradient-accumulation VRAM is
        :meth:`training` / :meth:`training_step` (``TRAIN × G`` on
        ``(b, S)``). Combining this builder with :meth:`backward_step`
        is the old split horizon and is not a HuggingFace training peak.
        """
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
        """Append ``TRAIN × G`` on ``(batch, seq_len)``, then optional optimizer.

        Same loop as :meth:`training`. ``micro_batches=1`` names the
        graph ``"train"``; ``G>1`` names them ``"train_{i}"``. Does not
        call :meth:`grad_accum` or :meth:`backward_step`.
        """
        self._append_train_micros(
            seq_len, micro_batches=micro_batches, batch=batch
        )
        if optimizer is not None:
            self.optimizer_policy = optimizer
            self.steps.append(optimizer_step(seq_len=seq_len, batch=batch))
        return self

    def _append_train_micros(
        self, seq_len: int, *, micro_batches: int, batch: int
    ) -> None:
        if micro_batches < 1:
            raise ValueError(f"micro_batches must be >= 1, got {micro_batches}")
        for index in range(micro_batches):
            self.steps.append(
                train_step(
                    seq_len,
                    batch=batch,
                    index=None if micro_batches == 1 else index,
                )
            )
