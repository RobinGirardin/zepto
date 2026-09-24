"""Cross-invocation state ports for horizon simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from zepto.semantic.metadata import DType

if TYPE_CHECKING:
    from ..lowered import LoweredGraph
    from ..lowering.context import InvocationContext
    from ..optimizer import OptimizerPolicy
    from .spec import HorizonStep


@dataclass(frozen=True, slots=True)
class KVCacheState:
    """KV cache footprint for one layer group."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    dtype: DType
    batch: int = 1

    @property
    def bytes(self) -> int:
        itemsize = self.dtype.itemsize or 0
        return (
            2
            * self.batch
            * self.num_layers
            * self.num_kv_heads
            * self.seq_len
            * self.head_dim
            * itemsize
        )


@dataclass(frozen=True, slots=True)
class Conv1DState:
    """Rolling depthwise conv buffer for one layer."""

    layer_index: int
    channels: int
    kernel_size: int
    dtype: DType
    batch: int = 1

    @property
    def bytes(self) -> int:
        itemsize = self.dtype.itemsize or 0
        return self.batch * self.channels * self.kernel_size * itemsize


@dataclass(frozen=True, slots=True)
class RecurrentScanState:
    """Recurrent scan hidden state for one layer (bytes scale with ``batch``)."""

    layer_index: int
    kind: Literal["gated_delta", "mamba2"]
    num_heads: int
    head_dim: int
    state_dim: int
    dtype: DType
    batch: int = 1

    @property
    def bytes(self) -> int:
        itemsize = self.dtype.itemsize or 0
        return (
            self.batch
            * self.num_heads
            * self.head_dim
            * self.state_dim
            * itemsize
        )


@dataclass(frozen=True, slots=True)
class GradAccumState:
    """Extra gradient buffer for the ``grad_accum()`` escape hatch.

    Not used by :meth:`~zepto.analysis.horizon.spec.HorizonSpec.training`
    or :meth:`~zepto.analysis.horizon.spec.HorizonSpec.training_step`:
    those timelines never install this buffer. Each ``TRAIN`` graph owns
    ``.grad`` via persist-grads. Created only when ``advance`` sees
    ``MICRO_FORWARD`` and ``grad_accum_steps > 1``.

    Bytes equal **gradient storage** for trainable params (gradient dtype
    × trainable elements) and are independent of parallel batch ``B``.

    ``parameter_bytes`` is the historical field name; the stored value is
    the gradient-dtype footprint from :func:`trainable_gradient_bytes`.
    """

    parameter_bytes: int
    micro_batches_seen: int

    @property
    def bytes(self) -> int:
        return self.parameter_bytes if self.micro_batches_seen > 0 else 0


@dataclass(frozen=True, slots=True)
class OptimizerState:
    """Optimizer moment buffers persisted after an optimizer step."""

    policy: OptimizerPolicy
    bytes: int


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable view of all carried state at one horizon boundary."""

    kv_caches: tuple[KVCacheState, ...] = ()
    grad_accum: GradAccumState | None = None
    optimizer: OptimizerState | None = None
    custom: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _KVTemplate:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class _ConvTemplate:
    layer_indices: tuple[int, ...]
    channels: int
    kernel_size: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class _RecurrentTemplate:
    layers: tuple[tuple[int, str, int, int, int], ...]
    dtype: DType


def state_object_bytes(value: object) -> int:
    """Return accounted bytes for a context.state entry value."""
    if isinstance(value, GradAccumState):
        return value.bytes
    if hasattr(value, "bytes"):
        raw = value.bytes  # type: ignore[attr-defined]
        if callable(raw):
            return int(raw())
        return int(raw)
    return 0


def snapshot_state_bytes(snapshot: StateSnapshot) -> int:
    """Sum accounted bytes for all entries in a state snapshot."""
    total = 0
    for kv in snapshot.kv_caches:
        total += kv.bytes
    if snapshot.grad_accum is not None:
        total += snapshot.grad_accum.bytes
    if snapshot.optimizer is not None:
        total += snapshot.optimizer.bytes
    for _name, value in snapshot.custom:
        total += state_object_bytes(value)
    return total


def trainable_parameter_elements(lowered: LoweredGraph) -> int:
    """Count trainable parameter elements in a lowered graph."""
    from ..accounting import numel

    return sum(
        numel(param.tensor.shape)
        for param in lowered.parameters.values()
        if param.trainable
    )


def parameter_bytes(lowered: LoweredGraph) -> int:
    """Total parameter storage bytes from a lowered graph."""
    from ..resolved import ResolvedValue

    accounting = lowered.context.accounting
    total = 0
    for param in lowered.parameters.values():
        total += accounting.bytes_for(
            ResolvedValue(
                tensor=param.tensor,
                role=param.role,
                dtype=param.tensor.dtype,
            )
        )
    return total


def trainable_gradient_bytes(lowered: LoweredGraph) -> int:
    """Persistent grad-accum footprint: trainable elements × gradient dtype.

    Resolves ``TensorRole.GRADIENT`` on a shape-only tensor so role policy
    wins over parameter semantic type and explicit param dtype (mixed
    precision: fp16 weights, fp32 grads).
    """
    from zepto.compose.values import Tensor
    from zepto.semantic.metadata import TensorRole

    from ..resolved import ResolvedValue

    accounting = lowered.context.accounting
    total = 0
    for param in lowered.parameters.values():
        if not param.trainable:
            continue
        grad_tensor = Tensor(shape=param.tensor.shape)
        dtype = accounting.resolve_dtype(grad_tensor, role=TensorRole.GRADIENT)
        total += accounting.bytes_for(
            ResolvedValue(
                tensor=grad_tensor,
                role=TensorRole.GRADIENT,
                dtype=dtype,
            )
        )
    return total


@dataclass
class StatePortRegistry:
    """Mutable registry of cross-step state injected into invocation contexts."""

    kv_template: _KVTemplate | None = None
    conv_template: _ConvTemplate | None = None
    recurrent_template: _RecurrentTemplate | None = None
    kv_caches: tuple[KVCacheState, ...] = ()
    grad_accum: GradAccumState | None = None
    optimizer: OptimizerState | None = None
    optimizer_policy: OptimizerPolicy | None = None
    grad_accum_steps: int = 1
    custom: tuple[tuple[str, object], ...] = ()

    @classmethod
    def empty(cls) -> StatePortRegistry:
        return cls()

    def configure_kv(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: DType,
    ) -> StatePortRegistry:
        return replace(
            self,
            kv_template=_KVTemplate(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            ),
        )

    def configure_conv_layers(
        self,
        *,
        layer_indices: tuple[int, ...],
        channels: int,
        kernel_size: int,
        dtype: DType,
    ) -> StatePortRegistry:
        return replace(
            self,
            conv_template=_ConvTemplate(
                layer_indices=layer_indices,
                channels=channels,
                kernel_size=kernel_size,
                dtype=dtype,
            ),
        )

    def configure_recurrent_layers(
        self,
        *,
        specs: tuple[tuple[int, str, int, int, int], ...],
        dtype: DType,
    ) -> StatePortRegistry:
        return replace(
            self,
            recurrent_template=_RecurrentTemplate(layers=specs, dtype=dtype),
        )

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            kv_caches=self.kv_caches,
            grad_accum=self.grad_accum,
            optimizer=self.optimizer,
            custom=self.custom,
        )

    def with_optimizer_state(
        self, policy: OptimizerPolicy, bytes: int
    ) -> StatePortRegistry:
        """Apply optimizer boundary: persist moments and clear grad accum."""
        return replace(
            self,
            optimizer=OptimizerState(policy=policy, bytes=bytes),
            grad_accum=None,
        )

    def bind_to_context(self, base: InvocationContext) -> InvocationContext:
        from ..lowering.context import InvocationContext as Ctx

        entries: list[tuple[str, object]] = []
        if self.kv_template is not None and not self.kv_caches:
            entries.append(("kv_template", self.kv_template))
        if self.conv_template is not None:
            entries.append(("conv_template", self.conv_template))
        if self.recurrent_template is not None:
            entries.append(("recurrent_template", self.recurrent_template))
        for index, kv in enumerate(self.kv_caches):
            entries.append((f"kv_cache:{index}", kv))
        if self.grad_accum is not None:
            entries.append(("grad_accum", self.grad_accum))
        if self.optimizer is not None:
            entries.append(("optimizer", self.optimizer))
        entries.extend(self.custom)
        return Ctx(
            phase=base.phase,
            hardware=base.hardware,
            backend=base.backend,
            precision=base.precision,
            accounting=base.accounting,
            state=tuple(entries),
            implementation_pins=base.implementation_pins,
            module_implementation_pins=base.module_implementation_pins,
            region_implementation_pins=base.region_implementation_pins,
            requested_capabilities=base.requested_capabilities,
            attention_backend=base.attention_backend,
            allow_fallback=base.allow_fallback,
            optim_prec=base.optim_prec,
            compute_capability=base.compute_capability,
            runtime_policy=base.runtime_policy,
            flop_policy=base.flop_policy,
        )

    def advance(self, step: HorizonStep, lowered: LoweredGraph) -> StatePortRegistry:
        from .spec import StepKind

        registry = self

        if step.kind == StepKind.PREFILL and registry.kv_template is not None:
            template = registry.kv_template
            kv = KVCacheState(
                num_layers=template.num_layers,
                num_kv_heads=template.num_kv_heads,
                head_dim=template.head_dim,
                seq_len=step.seq_len,
                dtype=template.dtype,
                batch=step.batch,
            )
            registry = replace(registry, kv_caches=(kv,))

        if step.kind == StepKind.PREFILL and registry.conv_template is not None:
            conv_t = registry.conv_template
            custom = list(registry.custom)
            existing = {name for name, _ in custom}
            for layer_index in conv_t.layer_indices:
                key = f"conv_state:{layer_index}"
                if key not in existing:
                    custom.append(
                        (
                            key,
                            Conv1DState(
                                layer_index=layer_index,
                                channels=conv_t.channels,
                                kernel_size=conv_t.kernel_size,
                                dtype=conv_t.dtype,
                                batch=step.batch,
                            ),
                        )
                    )
            registry = replace(registry, custom=tuple(custom))

        if step.kind == StepKind.PREFILL and registry.recurrent_template is not None:
            rec_t = registry.recurrent_template
            custom = list(registry.custom)
            existing = {name for name, _ in custom}
            for layer_index, kind, num_heads, head_dim, state_dim in rec_t.layers:
                key = f"scan_state:{layer_index}"
                if key not in existing:
                    kind_lit: Literal["gated_delta", "mamba2"] = (
                        "gated_delta" if kind == "gated_delta" else "mamba2"
                    )
                    custom.append(
                        (
                            key,
                            RecurrentScanState(
                                layer_index=layer_index,
                                kind=kind_lit,
                                num_heads=num_heads,
                                head_dim=head_dim,
                                state_dim=state_dim,
                                dtype=rec_t.dtype,
                                batch=step.batch,
                            ),
                        )
                    )
            registry = replace(registry, custom=tuple(custom))

        if step.kind == StepKind.DECODE and registry.kv_caches:
            current = registry.kv_caches[0]
            registry = replace(
                registry,
                kv_caches=(
                    replace(current, seq_len=current.seq_len + step.seq_len),
                ),
            )

        if step.kind == StepKind.MICRO_FORWARD and registry.grad_accum_steps > 1:
            # G>1 only. G=1 leaves grad_accum=None; backward owns weight grads.
            grad_bytes = trainable_gradient_bytes(lowered)
            accum = registry.grad_accum
            if accum is None:
                accum = GradAccumState(
                    parameter_bytes=grad_bytes,
                    micro_batches_seen=0,
                )
            registry = replace(
                registry,
                grad_accum=replace(
                    accum,
                    parameter_bytes=grad_bytes,
                    micro_batches_seen=accum.micro_batches_seen + 1,
                ),
            )

        if step.kind == StepKind.OPTIMIZER:
            policy = registry.optimizer_policy
            if policy is None:
                from ..optimizer import AdamW

                policy = AdamW
            trainable = trainable_parameter_elements(lowered)
            opt_bytes = policy.state_bytes(
                trainable_elements=trainable,
                context=lowered.context,
            ) + policy.update_workspace_bytes(context=lowered.context)
            registry = replace(
                registry,
                optimizer=OptimizerState(policy=policy, bytes=opt_bytes),
                grad_accum=None,
            )

        return registry
