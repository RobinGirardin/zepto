"""Model family plugin protocol for empirical cost studies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch import nn

    from zepto.analysis.horizon.spec import HorizonStep
    from zepto.compose import Tensor


ModuleFn = Callable[..., object]


@runtime_checkable
class ModelFamily(Protocol):
    model_id: str

    def option_keys(self) -> tuple[str, ...]: ...

    def validate_options(self, opts: dict[str, int]) -> None: ...

    def derive_options(self, sampled: dict[str, int]) -> dict[str, int]: ...

    def build_zepto_infer_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ) -> ModuleFn: ...

    def build_zepto_train_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ) -> ModuleFn: ...

    def zepto_infer_inputs(self, step: HorizonStep, ctx, state) -> tuple[Tensor, ...]: ...

    def zepto_train_inputs(
        self, step: HorizonStep, ctx, state
    ) -> tuple[Tensor, ...]: ...

    def build_hf_model(
        self,
        opts: dict[str, int],
        *,
        seq_len: int,
        precision: str,
        device: torch.device,
        twin_mode: str = "apertus_parity",
    ) -> nn.Module: ...

    def hf_forward_infer(self, model: nn.Module, input_ids: torch.Tensor): ...

    def hf_forward_train(
        self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ): ...
