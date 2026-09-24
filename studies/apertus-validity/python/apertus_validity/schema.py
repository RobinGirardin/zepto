"""Subject, evaluation, and ledger rows. Subject id is the primary key.

Framework §4: a subject is one architecture together with one (B, S).
Seed is run metadata. It is not part of the subject key.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields

from apertus_validity.identities import Architecture, FreeKnobs


def configuration_id(options: dict[str, int]) -> str:
    """Stable id for the architecture only (sorted option keys)."""
    payload = "\n".join(f"{key}={options[key]}" for key in sorted(options))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def subject_id(configuration_id: str, *, seq_len: int, batch_size: int) -> str:
    """Stable id for one experimental unit: architecture, S, and B."""
    payload = f"{configuration_id}|S={seq_len}|B={batch_size}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def input_seed(master_seed: int, subject_id: str) -> int:
    """Reproducible twin input seed. Does not change n."""
    payload = f"{master_seed}|{subject_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


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
            "num_kv_heads": self.knobs.num_kv_heads,
            "gqa_group": self.knobs.gqa_group,
            "head_dim": self.knobs.head_dim,
            "ffn_mult": self.knobs.ffn_mult,
            "num_layers": self.knobs.num_layers,
            "vocab_size": self.knobs.vocab_size,
            "num_heads": self.architecture.num_heads,
            "hidden_size": self.architecture.hidden_size,
            "intermediate_size": self.architecture.intermediate_size,
        }

    @classmethod
    def from_csv_row(cls, raw: dict[str, str]) -> Subject:
        knobs = FreeKnobs(
            num_kv_heads=int(raw["num_kv_heads"]),
            gqa_group=int(raw["gqa_group"]),
            head_dim=int(raw["head_dim"]),
            ffn_mult=float(raw["ffn_mult"]),
            num_layers=int(raw["num_layers"]),
            vocab_size=int(raw["vocab_size"]),
        )
        architecture = Architecture(
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_heads=int(raw["num_heads"]),
            num_kv_heads=int(raw["num_kv_heads"]),
            num_layers=int(raw["num_layers"]),
            vocab_size=int(raw["vocab_size"]),
            head_dim=int(raw["head_dim"]),
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
    "num_kv_heads",
    "gqa_group",
    "head_dim",
    "ffn_mult",
    "num_layers",
    "vocab_size",
    "num_heads",
    "hidden_size",
    "intermediate_size",
)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One completed phase × precision observation for one subject.

    Primary H1 pair: y_vram, target_vram.
    Primary H2 pair: y_flop_fcm, target_flop.
    Remaining columns are diagnostics and must not enter the eight TOSTs.
    """

    subject_id: str
    configuration_id: str
    phase: str
    precision: str
    seq_len: int
    batch_size: int
    step: int
    y_vram: int
    target_vram: int
    target_vram_raw: int
    cublas_infer_correction_bytes: int
    y_flop_fcm: int
    target_flop: int
    y_flop: int
    y_runtime_workspace: int = 0
    y_activations: int = 0
    peak_minus_before: int = 0
    alloc_before: int = 0

    def to_csv_row(self) -> dict[str, str | int]:
        return asdict(self)


EVALUATION_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in fields(EvaluationRow))


def parse_evaluation_row(raw: dict[str, str]) -> EvaluationRow:
    return EvaluationRow(
        subject_id=raw["subject_id"],
        configuration_id=raw["configuration_id"],
        phase=raw["phase"],
        precision=raw["precision"],
        seq_len=int(raw["seq_len"]),
        batch_size=int(raw["batch_size"]),
        step=int(raw["step"]),
        y_vram=int(raw["y_vram"]),
        target_vram=int(raw["target_vram"]),
        target_vram_raw=int(raw["target_vram_raw"]),
        cublas_infer_correction_bytes=int(raw["cublas_infer_correction_bytes"]),
        y_flop_fcm=int(raw["y_flop_fcm"]),
        target_flop=int(raw["target_flop"]),
        y_flop=int(raw["y_flop"]),
        y_runtime_workspace=int(raw.get("y_runtime_workspace") or 0),
        y_activations=int(raw.get("y_activations") or 0),
        peak_minus_before=int(raw.get("peak_minus_before") or 0),
        alloc_before=int(raw.get("alloc_before") or 0),
    )


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One failed attempt. Framework §5.3: never drop silently."""

    subject_id: str
    configuration_id: str
    phase: str
    precision: str
    seq_len: int
    batch_size: int
    failure_kind: str
    message: str

    def to_csv_row(self) -> dict[str, str | int]:
        return asdict(self)


LEDGER_FIELDNAMES: tuple[str, ...] = tuple(f.name for f in fields(LedgerRow))


def parse_ledger_row(raw: dict[str, str]) -> LedgerRow:
    return LedgerRow(
        subject_id=raw["subject_id"],
        configuration_id=raw["configuration_id"],
        phase=raw["phase"],
        precision=raw["precision"],
        seq_len=int(raw["seq_len"]),
        batch_size=int(raw["batch_size"]),
        failure_kind=raw["failure_kind"],
        message=raw["message"],
    )


@dataclass(frozen=True, slots=True)
class PrecisionOutcome:
    """Completed rows and ledger rows from one subject × precision."""

    rows: tuple[EvaluationRow, ...]
    ledger: tuple[LedgerRow, ...]
