"""Shared multimodal forward pattern for vision-language checkpoints."""

from __future__ import annotations

from zepto.compose import Module, Tensor


def embed_multimodal_sequence(
    model: Module,
    token_ids: Tensor,
    *,
    placeholder_indices: Tensor | None,
    vision_pixels: Tensor | None,
) -> Tensor:
    """Text-only ``Embedding`` or vision tower + ``MultimodalSequenceBuilder``.

    Expects ``model.embedding``, and when vision is enabled ``model.vision`` and
    ``model.sequence_builder``. Effective downstream sequence length for costing:
    ``S' = S_text - n_placeholders + S_visual`` when placeholders are replaced.
    """
    if vision_pixels is None:
        return model.embedding(token_ids)  # type: ignore[attr-defined,no-any-return]
    if placeholder_indices is None:
        raise ValueError("vision_pixels requires placeholder_indices")
    vision = model.vision  # type: ignore[attr-defined]
    builder = model.sequence_builder  # type: ignore[attr-defined]
    if vision is None or builder is None:
        raise ValueError("vision inputs require include_vision=True at construction")
    visual = vision(vision_pixels)
    return builder(token_ids, visual, placeholder_indices)  # type: ignore[misc,no-any-return]


__all__ = ["embed_multimodal_sequence"]
