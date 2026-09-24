"""Subject schema for the Qwen3.8 hybrid catalog.

Evaluation and ledger columns stay on the shared validity_common schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from qwen38_validity.identities import Architecture, FreeKnobs
from validity_common.schema import (  # re-export
    EVALUATION_FIELDNAMES,
    LEDGER_FIELDNAMES,
    EvaluationRow,
    LedgerRow,
    PrecisionOutcome,
    input_seed,
    parse_evaluation_row,
    parse_ledger_row,
    subject_id,
)


def configuration_id(options: dict[str, int]) -> str:
    """Stable id for the architecture only (sorted option keys)."""
    payload = "\n".join(f"{key}={options[key]}" for key in sorted(options))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Subject:
    subject_id: str
    configuration_id: str
    knobs: FreeKnobs
    architecture: Architecture
    seq_len: int
    batch_size: int

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "subject_id": self.subject_id,
            "configuration_id": self.configuration_id,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "num_cycles": self.knobs.num_cycles,
            "hidden_size": self.knobs.hidden_size,
            "delta_qk_heads": self.knobs.delta_qk_heads,
            "delta_v_group": self.knobs.delta_v_group,
            "delta_head_dim": self.knobs.delta_head_dim,
            "attn_kv_heads": self.knobs.attn_kv_heads,
            "attn_gqa_group": self.knobs.attn_gqa_group,
            "attn_head_dim": self.knobs.attn_head_dim,
            "ffn_mult": self.knobs.ffn_mult,
            "num_layers": self.architecture.num_layers,
            "vocab_size": self.knobs.vocab_size,
            "delta_num_v_heads": self.architecture.delta_num_v_heads,
            "attn_num_q_heads": self.architecture.attn_num_q_heads,
            "intermediate_size": self.architecture.intermediate_size,
        }

    @classmethod
    def from_csv_row(cls, raw: dict[str, str]) -> Subject:
        knobs = FreeKnobs(
            num_cycles=int(raw["num_cycles"]),
            hidden_size=int(raw["hidden_size"]),
            delta_qk_heads=int(raw["delta_qk_heads"]),
            delta_v_group=int(raw["delta_v_group"]),
            delta_head_dim=int(raw["delta_head_dim"]),
            attn_kv_heads=int(raw["attn_kv_heads"]),
            attn_gqa_group=int(raw["attn_gqa_group"]),
            attn_head_dim=int(raw["attn_head_dim"]),
            ffn_mult=float(raw["ffn_mult"]),
            vocab_size=int(raw["vocab_size"]),
        )
        architecture = Architecture(
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_layers=int(raw["num_layers"]),
            vocab_size=int(raw["vocab_size"]),
            delta_num_qk_heads=int(raw["delta_qk_heads"]),
            delta_num_v_heads=int(raw["delta_num_v_heads"]),
            delta_head_dim=int(raw["delta_head_dim"]),
            attn_num_q_heads=int(raw["attn_num_q_heads"]),
            attn_num_kv_heads=int(raw["attn_kv_heads"]),
            attn_head_dim=int(raw["attn_head_dim"]),
        )
        return cls(
            subject_id=raw["subject_id"],
            configuration_id=raw["configuration_id"],
            knobs=knobs,
            architecture=architecture,
            seq_len=int(raw["seq_len"]),
            batch_size=int(raw["batch_size"]),
        )


SUBJECT_FIELDNAMES: tuple[str, ...] = (
    "subject_id",
    "configuration_id",
    "seq_len",
    "batch_size",
    "num_cycles",
    "hidden_size",
    "delta_qk_heads",
    "delta_v_group",
    "delta_head_dim",
    "attn_kv_heads",
    "attn_gqa_group",
    "attn_head_dim",
    "ffn_mult",
    "num_layers",
    "vocab_size",
    "delta_num_v_heads",
    "attn_num_q_heads",
    "intermediate_size",
)


__all__ = [
    "EVALUATION_FIELDNAMES",
    "LEDGER_FIELDNAMES",
    "SUBJECT_FIELDNAMES",
    "EvaluationRow",
    "LedgerRow",
    "PrecisionOutcome",
    "Subject",
    "configuration_id",
    "input_seed",
    "parse_evaluation_row",
    "parse_ledger_row",
    "subject_id",
]
