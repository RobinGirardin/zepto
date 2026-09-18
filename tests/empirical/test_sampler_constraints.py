"""Architecture sampling rejection rules."""

from __future__ import annotations

from empirical.models.apertus_core import ApertusFamilyCore
from empirical.sampler import ArchitectureRanges, IntRange, SamplerConfig, sample_configurations


def _smoke_sampler() -> SamplerConfig:
    return SamplerConfig(
        seq_len=IntRange(8, 16),
        batch_size=IntRange(1, 2),
        architecture=ArchitectureRanges(
            head_dim=IntRange(4, 4),
            intermediate_size=IntRange(64, 64),
            num_heads=IntRange(8, 8),
            num_kv_heads=IntRange(2, 2),
            num_layers=IntRange(2, 2),
            vocab_size=IntRange(100, 100),
        ),
        precisions=("fp32",),
        architecture_mins={
            "hidden_size": 32,
            "intermediate_size": 64,
            "vocab_size": 100,
            "num_layers": 2,
        },
    )


def test_sample_golden_configuration() -> None:
    family = ApertusFamilyCore()
    configs = sample_configurations(
        family,
        _smoke_sampler(),
        n_configs=1,
        master_seed=0,
    )
    assert len(configs) == 1
    opts = configs[0].options
    assert opts["hidden_size"] == opts["num_heads"] * opts["head_dim"]


def test_rejects_indivisible_gqa() -> None:
    family = ApertusFamilyCore()
    cfg = SamplerConfig(
        seq_len=IntRange(8, 8),
        batch_size=IntRange(1, 1),
        architecture=ArchitectureRanges(
            head_dim=IntRange(4, 4),
            intermediate_size=IntRange(64, 64),
            num_heads=IntRange(7, 7),
            num_kv_heads=IntRange(4, 4),
            num_layers=IntRange(2, 2),
            vocab_size=IntRange(100, 100),
        ),
        precisions=("fp32",),
    )
    try:
        sample_configurations(family, cfg, n_configs=1, master_seed=1)
    except RuntimeError as exc:
        assert "could not sample" in str(exc)
    else:
        raise AssertionError("expected sampling to fail for invalid GQA ratio")
