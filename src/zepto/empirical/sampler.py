"""Architecture and workload sampling for empirical cost studies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Literal

from zepto.empirical.schema import (
    ConfigurationLinkRow,
    OptionRow,
    configuration_id,
    draw_id,
    subject_id,
)

Precision = Literal["fp32", "fp16"]
_ALLOWED_PRECISIONS: frozenset[str] = frozenset(("fp32", "fp16"))


@dataclass(frozen=True, slots=True)
class FreeKnobCatalog:
    """Discrete supports for GQA dense decoder (Apertus / Granite) identity derivation."""

    gqa_group: tuple[int, ...] = (1, 2, 4)
    num_kv_heads: tuple[int, ...] = (2, 4, 8)
    head_dim: tuple[int, ...] = (64, 128)
    num_layers: tuple[int, ...] = (2, 4, 8)
    ffn_mult: tuple[float, ...] = (4.0, 5.25)
    vocab_size: tuple[int, ...] = (1024, 4096)

    def __post_init__(self) -> None:
        for name in (
            "gqa_group",
            "num_kv_heads",
            "head_dim",
            "num_layers",
            "ffn_mult",
            "vocab_size",
        ):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class WorkloadCatalog:
    """Discrete (B, S) support, optionally capped by a token budget."""

    seq_len: tuple[int, ...] = (32, 64, 128)
    batch_size: tuple[int, ...] = (1, 2, 4)
    min_seq_len: int = 32
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.seq_len:
            raise ValueError("seq_len must not be empty")
        if not self.batch_size:
            raise ValueError("batch_size must not be empty")
        if self.min_seq_len <= 0:
            raise ValueError("min_seq_len must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def feasible_pairs(self) -> list[tuple[int, int]]:
        pairs = [
            (batch, seq)
            for batch in self.batch_size
            for seq in self.seq_len
            if seq >= self.min_seq_len
            and (self.max_tokens is None or batch * seq <= self.max_tokens)
        ]
        if not pairs:
            raise ValueError(
                "workload catalog has no feasible (batch_size, seq_len) pairs"
            )
        return pairs


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    knobs: FreeKnobCatalog = field(default_factory=FreeKnobCatalog)
    workload: WorkloadCatalog = field(default_factory=WorkloadCatalog)
    precisions: tuple[Precision, ...] = ("fp32", "fp16")
    architecture_mins: dict[str, int] = field(default_factory=dict)
    max_reject_attempts: int = 10_000

    def __post_init__(self) -> None:
        if not self.precisions:
            raise ValueError("precisions must not be empty")
        for precision in self.precisions:
            if precision not in _ALLOWED_PRECISIONS:
                raise ValueError(f"unknown precision: {precision!r}")


@dataclass(frozen=True, slots=True)
class SampledConfiguration:
    model_id: str
    configuration_id: str
    options: dict[str, int]
    option_rows: tuple[OptionRow, ...]
    link_rows: tuple[ConfigurationLinkRow, ...]


@dataclass(frozen=True, slots=True)
class WorkloadDraw:
    configuration_id: str
    seq_len: int
    batch_size: int
    draw_seed: int
    draw_id: str
    subject_id: str


def _derive_draw_seed(master_seed: int, config_index: int, draw_index: int) -> int:
    mix = (master_seed & 0xFFFFFFFF) ^ ((config_index + 1) * 1_000_003)
    mix ^= (draw_index + 1) * 9_999_933
    return mix & 0xFFFFFFFF


def _sample_knobs(catalog: FreeKnobCatalog, rng: random.Random) -> dict[str, int | float]:
    return {
        "gqa_group": rng.choice(catalog.gqa_group),
        "num_kv_heads": rng.choice(catalog.num_kv_heads),
        "head_dim": rng.choice(catalog.head_dim),
        "ffn_mult": rng.choice(catalog.ffn_mult),
        "num_layers": rng.choice(catalog.num_layers),
        "vocab_size": rng.choice(catalog.vocab_size),
    }


def _sample_workload(catalog: WorkloadCatalog, rng: random.Random) -> tuple[int, int]:
    batch, seq = rng.choice(catalog.feasible_pairs())
    return batch, seq


def _configuration_from_options(family, opts: dict[str, int]) -> SampledConfiguration:
    cfg_id = configuration_id(family.model_id, opts)
    option_rows = tuple(OptionRow.from_int(k, opts[k]) for k in family.option_keys())
    links = tuple(
        ConfigurationLinkRow(
            configuration_id=cfg_id,
            model_id=family.model_id,
            option_id=row.option_id,
        )
        for row in option_rows
    )
    return SampledConfiguration(
        model_id=family.model_id,
        configuration_id=cfg_id,
        options=opts,
        option_rows=option_rows,
        link_rows=links,
    )


def sample_configurations(
    family,
    config: SamplerConfig,
    *,
    n_configs: int,
    master_seed: int,
) -> list[SampledConfiguration]:
    rng = random.Random(master_seed)
    mins = dict(config.architecture_mins)
    accepted: list[SampledConfiguration] = []
    seen: set[str] = set()
    attempts = 0

    while len(accepted) < n_configs:
        attempts += 1
        if attempts > config.max_reject_attempts:
            raise RuntimeError(
                f"could not sample {n_configs} valid configs after "
                f"{config.max_reject_attempts} attempts"
            )
        sampled = _sample_knobs(config.knobs, rng)
        opts = family.derive_options(sampled)
        try:
            family.validate_options(opts)
        except ValueError:
            continue
        if any(opts.get(key, 0) < minimum for key, minimum in mins.items()):
            continue
        cfg = _configuration_from_options(family, opts)
        if cfg.configuration_id in seen:
            continue
        seen.add(cfg.configuration_id)
        accepted.append(cfg)
    return accepted


def sample_draws(
    configuration: SampledConfiguration,
    config: SamplerConfig,
    *,
    m_draws: int,
    master_seed: int,
    config_index: int,
) -> list[WorkloadDraw]:
    rng = random.Random(_derive_draw_seed(master_seed, config_index, 0) ^ 0xA5A5_A5A5)
    draws: list[WorkloadDraw] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    draw_index = 0
    while len(draws) < m_draws:
        attempts += 1
        if attempts > config.max_reject_attempts:
            raise RuntimeError(
                f"could not sample {m_draws} unique (batch_size, seq_len) pairs after "
                f"{config.max_reject_attempts} attempts"
            )
        batch_size, seq_len = _sample_workload(config.workload, rng)
        key = (batch_size, seq_len)
        if key in seen:
            continue
        seen.add(key)
        seed = _derive_draw_seed(master_seed, config_index, draw_index)
        sid = subject_id(
            configuration.configuration_id,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        did = draw_id(
            configuration.configuration_id,
            seq_len=seq_len,
            batch_size=batch_size,
            draw_seed=seed,
        )
        draws.append(
            WorkloadDraw(
                configuration_id=configuration.configuration_id,
                seq_len=seq_len,
                batch_size=batch_size,
                draw_seed=seed,
                draw_id=did,
                subject_id=sid,
            )
        )
        draw_index += 1
    return draws


def options_dict_from_tables(
    configuration_id: str,
    option_rows: dict[str, OptionRow],
    link_rows: Iterable[ConfigurationLinkRow],
) -> dict[str, int]:
    """Rebuild option dict for one configuration from normalized dimension rows."""
    opts: dict[str, int] = {}
    for link in link_rows:
        if link.configuration_id != configuration_id:
            continue
        row = option_rows[link.option_id]
        opts[row.option_key] = int(row.value)
    return opts
