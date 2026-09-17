"""Patch embedding modules for vision towers."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Conv2d, Conv3d, LinearMatMul, Reshape

from .linear import Linear
from .rms_norm import RMSNorm


class LinearPatchEmbed(Module):
    """Flatten spatiotemporal patches and project with a linear layer."""

    module_kind = "LinearPatchEmbed"

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        temporal_frames: int,
        out_dim: int,
        *,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        if min(patch_size, in_channels, temporal_frames, out_dim, grid_h, grid_w) <= 0:
            raise ValueError("patch embed dimensions must be positive")
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.temporal_frames = temporal_frames
        self.out_dim = out_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        flat_in = temporal_frames * in_channels * patch_size * patch_size
        self.proj = Linear(flat_in, out_dim)

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    def forward(self, pixels: Tensor) -> Tensor:
        expected = (
            self.temporal_frames,
            self.grid_h * self.patch_size,
            self.grid_w * self.patch_size,
            self.in_channels,
        )
        if pixels.shape != expected:
            raise ValueError(
                f"LinearPatchEmbed expected pixel shape {expected}, got {pixels.shape}"
            )
        flat = Reshape(
            shape=(self.num_patches, self.proj.weight.shape[0])
        )(pixels)
        return self.proj(flat)  # type: ignore[return-value]


class GemmaLinearPatchEmbed(Module):
    """Gemma 4: optional input RMSNorm then linear patch projection."""

    module_kind = "GemmaLinearPatchEmbed"

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        out_dim: int,
        *,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        self.inner = LinearPatchEmbed(
            patch_size,
            in_channels,
            temporal_frames=1,
            out_dim=out_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        flat_in = in_channels * patch_size * patch_size
        self.input_norm = RMSNorm(flat_in, elementwise_affine=False)

    def forward(self, pixels: Tensor) -> Tensor:
        expected = (
            1,
            self.inner.grid_h * self.inner.patch_size,
            self.inner.grid_w * self.inner.patch_size,
            self.inner.in_channels,
        )
        if pixels.shape != expected:
            raise ValueError(
                f"GemmaLinearPatchEmbed expected pixel shape {expected}, got {pixels.shape}"
            )
        patches = Reshape(
            shape=(
                self.inner.num_patches,
                self.inner.in_channels * self.inner.patch_size * self.inner.patch_size,
            )
        )(pixels)
        normed = self.input_norm(patches)
        return LinearMatMul()(normed, parameters=(self.inner.proj.weight,))  # type: ignore[return-value]


class Conv2dPatchEmbed(Module):
    """ViT-style conv patch embed with channel-last layout."""

    module_kind = "Conv2dPatchEmbed"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        *,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.weight = Parameter(
            shape=(patch_size, patch_size, in_channels, out_channels),
            semantic_type="weight",
        )

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    def forward(self, pixels: Tensor) -> Tensor:
        expected = (
            self.grid_h * self.patch_size,
            self.grid_w * self.patch_size,
            self.in_channels,
        )
        if pixels.shape != expected:
            raise ValueError(
                f"Conv2dPatchEmbed expected pixel shape {expected}, got {pixels.shape}"
            )
        conv_out = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )(pixels, parameters=(self.weight,))
        return Reshape(shape=(self.num_patches, self.out_channels))(conv_out)  # type: ignore[return-value]


class Conv3dPatchEmbed(Module):
    """Video tubulet patch embed (Qwen3-VL)."""

    module_kind = "Conv3dPatchEmbed"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        *,
        grid_t: int,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.grid_t = grid_t
        self.grid_h = grid_h
        self.grid_w = grid_w
        kt, kh, kw = kernel_size
        self.weight = Parameter(
            shape=(kt, kh, kw, in_channels, out_channels),
            semantic_type="weight",
        )

    @property
    def num_patches(self) -> int:
        return self.grid_t * self.grid_h * self.grid_w

    def forward(self, pixels: Tensor) -> Tensor:
        t_in = (self.grid_t - 1) * self.stride[0] + self.kernel_size[0]
        h_in = (self.grid_h - 1) * self.stride[1] + self.kernel_size[1]
        w_in = (self.grid_w - 1) * self.stride[2] + self.kernel_size[2]
        expected = (t_in, h_in, w_in, self.in_channels)
        if pixels.shape != expected:
            raise ValueError(
                f"Conv3dPatchEmbed expected pixel shape {expected}, got {pixels.shape}"
            )
        conv_out = Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )(pixels, parameters=(self.weight,))
        return Reshape(shape=(self.num_patches, self.out_channels))(conv_out)  # type: ignore[return-value]


def qwen3_vl_patch_embed(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
) -> Conv3dPatchEmbed:
    """Qwen3.8-27B vision tubulet embed: ``3→1152``, kernel/stride ``(2,16,16)``."""
    return Conv3dPatchEmbed(
        in_channels=3,
        out_channels=1152,
        kernel_size=(2, 16, 16),
        stride=(2, 16, 16),
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
    )


__all__ = [
    "Conv2dPatchEmbed",
    "Conv3dPatchEmbed",
    "GemmaLinearPatchEmbed",
    "LinearPatchEmbed",
    "qwen3_vl_patch_embed",
]
