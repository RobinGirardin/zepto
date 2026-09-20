"""Dataset schema, stable IDs, and CSV row types."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9._=-]+")


def slug(text: str) -> str:
    """Stable, CSV-safe slug for option_id."""
    cleaned = _SLUG_UNSAFE.sub("_", text.strip())
    return cleaned or "empty"


def option_id(option_key: str, value: int) -> str:
    return slug(f"{option_key}={value}")


def configuration_id(_model_id: str, options: dict[str, int]) -> str:
    lines = [f"{key}={options[key]}" for key in sorted(options)]
    payload = "\n".join(lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def draw_id(
    configuration_id: str,
    *,
    seq_len: int,
    batch_size: int,
    precision: str,
    draw_seed: int,
) -> str:
    payload = (
        f"{configuration_id}|S={seq_len}|B={batch_size}|prec={precision}|seed={draw_seed}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class OptionRow:
    option_id: str
    option_key: str
    value: str

    @classmethod
    def from_int(cls, option_key: str, value: int) -> OptionRow:
        return cls(
            option_id=option_id(option_key, value),
            option_key=option_key,
            value=str(value),
        )


@dataclass(frozen=True, slots=True)
class ConfigurationLinkRow:
    configuration_id: str
    model_id: str
    option_id: str


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    model_id: str
    configuration_id: str
    draw_id: str
    phase: str
    precision: str
    seq_len: int
    batch_size: int
    step: int
    zepto_batch_representation: str
    y_flop: int
    y_vram: int
    target_flop: int
    target_vram: int
    target_vram_raw: int = 0
    cublas_infer_correction_bytes: int = 0
    y_runtime_workspace: int = 0
    y_activations: int = 0
    peak_minus_before: int = 0
    alloc_before: int = 0
    target_flop_no_opt: int = 0
    y_flop_no_opt: int = 0

    def to_csv_row(self) -> dict[str, str | int]:
        row = asdict(self)
        return {k: row[k] for k in EVALUATION_FIELDNAMES}


EVALUATION_FIELDNAMES: tuple[str, ...] = tuple(
    f.name for f in fields(EvaluationRow)
)

OPTION_FIELDNAMES: tuple[str, ...] = ("option_id", "option_key", "value")
CONFIG_LINK_FIELDNAMES: tuple[str, ...] = (
    "configuration_id",
    "model_id",
    "option_id",
)


def write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, str | int]],
    *,
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _parse_int(raw: dict[str, str], key: str, default: int = 0) -> int:
    if key not in raw or raw[key] == "":
        return default
    return int(raw[key])


def parse_evaluation_csv(path: Path) -> list[EvaluationRow]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out: list[EvaluationRow] = []
        for raw in reader:
            target_vram = int(raw["target_vram"])
            out.append(
                EvaluationRow(
                    model_id=raw["model_id"],
                    configuration_id=raw["configuration_id"],
                    draw_id=raw["draw_id"],
                    phase=raw["phase"],
                    precision=raw["precision"],
                    seq_len=int(raw["seq_len"]),
                    batch_size=int(raw["batch_size"]),
                    step=int(raw["step"]),
                    zepto_batch_representation=raw["zepto_batch_representation"],
                    y_flop=int(raw["y_flop"]),
                    y_vram=int(raw["y_vram"]),
                    target_flop=int(raw["target_flop"]),
                    target_vram=target_vram,
                    target_vram_raw=_parse_int(raw, "target_vram_raw", target_vram),
                    cublas_infer_correction_bytes=_parse_int(
                        raw, "cublas_infer_correction_bytes"
                    ),
                    y_runtime_workspace=_parse_int(raw, "y_runtime_workspace"),
                    y_activations=_parse_int(raw, "y_activations"),
                    peak_minus_before=_parse_int(raw, "peak_minus_before"),
                    alloc_before=_parse_int(raw, "alloc_before"),
                    target_flop_no_opt=_parse_int(raw, "target_flop_no_opt"),
                    y_flop_no_opt=_parse_int(raw, "y_flop_no_opt"),
                )
            )
        return out
