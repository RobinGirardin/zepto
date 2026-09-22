"""Batch / sequence rank helpers for language-model modules."""

from __future__ import annotations

from zepto.compose import Tensor


def batch_seq_dims(value: Tensor) -> tuple[int, int]:
    """Return ``(batch, seq_len)`` for rank-2 ``(S, …)`` or rank-3 ``(B, S, …)``."""
    rank = len(value.shape)
    if rank == 2:
        return (1, value.shape[0])
    if rank == 3:
        return (value.shape[0], value.shape[1])
    raise ValueError(
        f"expected rank-2 (S, …) or rank-3 (B, S, …), got {value.shape}"
    )


def expect_token_ids_rank(token_ids: Tensor) -> None:
    if len(token_ids.shape) not in (1, 2):
        raise ValueError(
            f"token_ids must be rank-1 (S,) or rank-2 (B, S), got {token_ids.shape}"
        )


def expect_sequence_hidden_states(hidden_states: Tensor, hidden_size: int) -> int:
    """Validate ``(S, d)`` or ``(B, S, d)``; return tensor rank."""
    rank = len(hidden_states.shape)
    if rank not in (2, 3):
        raise ValueError(
            f"expected rank-2 (S, d) or rank-3 (B, S, d) hidden states, "
            f"got {hidden_states.shape}"
        )
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(
            f"hidden dim mismatch: expected {hidden_size}, "
            f"got {hidden_states.shape[-1]}"
        )
    return rank


def expect_logits_rank(logits: Tensor) -> int:
    rank = len(logits.shape)
    if rank not in (2, 3):
        raise ValueError(
            f"expected rank-2 (S, V) or rank-3 (B, S, V) logits, got {logits.shape}"
        )
    return rank


def expect_labels_for_hidden(labels: Tensor, hidden_states: Tensor) -> None:
    rank = len(hidden_states.shape)
    if rank == 2:
        if len(labels.shape) != 1:
            raise ValueError(
                f"expected rank-1 labels (S,) for (S, d) hidden states, "
                f"got {labels.shape}"
            )
        if labels.shape[0] != hidden_states.shape[0]:
            raise ValueError("labels and hidden states sequence length must match")
        return
    if len(labels.shape) != 2:
        raise ValueError(
            f"expected rank-2 labels (B, S) for (B, S, d) hidden states, "
            f"got {labels.shape}"
        )
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError("labels and hidden states batch/sequence dims must match")
