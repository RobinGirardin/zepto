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
)

Precision = Literal["fp32", "fp16"]
_ALLOWED_PRECISIONS: frozenset[str] = frozenset(("fp32", "fp16"))


@dataclass(frozen=True, slots=True)
class IntRange:
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"invalid range: {self.low} > {self.high}")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass(frozen=True, slots=True)
class ArchitectureRanges:
    head_dim: IntRange
    intermediate_size: IntRange
    num_heads: IntRange
    num_kv_heads: IntRange
    num_layers: IntRange
    vocab_size: IntRange


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    seq_len: IntRange
    batch_size: IntRange
    architecture: ArchitectureRanges
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
    precision: Precision
    draw_seed: int
    draw_id: str


def _derive_draw_seed(master_seed: int, config_index: int, draw_index: int) -> int:
    mix = (master_seed & 0xFFFFFFFF) ^ ((config_index + 1) * 1_000_003)
    mix ^= (draw_index + 1) * 9_999_933
    return mix & 0xFFFFFFFF


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
    attempts = 0
    arch = config.architecture

    while len(accepted) < n_configs:
        attempts += 1
        if attempts > config.max_reject_attempts:
            raise RuntimeError(
                f"could not sample {n_configs} valid configs after "
                f"{config.max_reject_attempts} attempts"
            )
        sampled = {
            "head_dim": arch.head_dim.sample(rng),
            "intermediate_size": arch.intermediate_size.sample(rng),
            "num_heads": arch.num_heads.sample(rng),
            "num_kv_heads": arch.num_kv_heads.sample(rng),
            "num_layers": arch.num_layers.sample(rng),
            "vocab_size": arch.vocab_size.sample(rng),
        }
        opts = family.derive_options(sampled)
        try:
            family.validate_options(opts)
        except ValueError:
            continue
        if any(opts.get(key, 0) < minimum for key, minimum in mins.items()):
            continue
        if opts["num_heads"] % opts["num_kv_heads"] != 0:
            continue
        cfg_id = configuration_id(family.model_id, opts)
        option_rows = tuple(
            OptionRow.from_int(k, opts[k]) for k in family.option_keys()
        )
        links = tuple(
            ConfigurationLinkRow(
                configuration_id=cfg_id,
                model_id=family.model_id,
                option_id=row.option_id,
            )
            for row in option_rows
        )
        accepted.append(
            SampledConfiguration(
                model_id=family.model_id,
                configuration_id=cfg_id,
                options=opts,
                option_rows=option_rows,
                link_rows=links,
            )
        )
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
    precisions = config.precisions
    draws: list[WorkloadDraw] = []
    for draw_index in range(m_draws):
        seq_len = config.seq_len.sample(rng)
        batch_size = config.batch_size.sample(rng)
        precision = precisions[rng.randrange(len(precisions))]
        seed = _derive_draw_seed(master_seed, config_index, draw_index)
        did = draw_id(
            configuration.configuration_id,
            seq_len=seq_len,
            batch_size=batch_size,
            precision=precision,
            draw_seed=seed,
        )
        draws.append(
            WorkloadDraw(
                configuration_id=configuration.configuration_id,
                seq_len=seq_len,
                batch_size=batch_size,
                precision=precision,
                draw_seed=seed,
                draw_id=did,
            )
        )
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
