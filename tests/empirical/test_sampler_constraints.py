"""Free-knob sampling produces valid Apertus identities by construction."""

from __future__ import annotations

from dataclasses import fields

import pytest

from zepto.empirical.models.apertus_core import ApertusFamilyCore, round_to_multiple
from zepto.empirical.sampler import (
    FreeKnobCatalog,
    SamplerConfig,
    WorkloadCatalog,
    WorkloadDraw,
    sample_configurations,
    sample_draws,
)


def _smoke_sampler() -> SamplerConfig:
    return SamplerConfig(
        knobs=FreeKnobCatalog(
            gqa_group=(4,),
            num_kv_heads=(2,),
            head_dim=(4,),
            num_layers=(2,),
            ffn_mult=(2.0,),
            vocab_size=(100,),
        ),
        workload=WorkloadCatalog(
            seq_len=(8, 16),
            batch_size=(1, 2),
            min_seq_len=8,
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
    assert opts["num_heads"] == 8
    assert opts["intermediate_size"] == 64


def test_free_knobs_always_satisfy_gqa_and_width() -> None:
    family = ApertusFamilyCore()
    cfg = SamplerConfig(
        knobs=FreeKnobCatalog(
            gqa_group=(1, 2, 4),
            num_kv_heads=(2, 4),
            head_dim=(4, 8),
            num_layers=(2,),
            ffn_mult=(2.0, 4.0),
            vocab_size=(100,),
        ),
        workload=WorkloadCatalog(seq_len=(8,), batch_size=(1,), min_seq_len=8),
        precisions=("fp32",),
    )
    configs = sample_configurations(family, cfg, n_configs=12, master_seed=3)
    assert len(configs) == 12
    assert len({c.configuration_id for c in configs}) == 12
    for sampled in configs:
        family.validate_options(sampled.options)
        opts = sampled.options
        assert opts["num_heads"] % opts["num_kv_heads"] == 0
        assert opts["hidden_size"] == opts["num_heads"] * opts["head_dim"]
        assert opts["head_dim"] % 2 == 0


def test_worked_example_derivation() -> None:
    family = ApertusFamilyCore()
    opts = family.derive_options(
        {
            "num_kv_heads": 4,
            "gqa_group": 2,
            "head_dim": 128,
            "ffn_mult": 5.25,
            "num_layers": 4,
            "vocab_size": 1024,
        }
    )
    assert opts["num_heads"] == 8
    assert opts["hidden_size"] == 1024
    assert opts["intermediate_size"] == round_to_multiple(5.25 * 1024)
    assert opts["intermediate_size"] == 5376
    family.validate_options(opts)


def test_sample_draws_unique_subjects_no_precision() -> None:
    family = ApertusFamilyCore()
    sampler = _smoke_sampler()
    configuration = sample_configurations(
        family, sampler, n_configs=1, master_seed=0
    )[0]
    draws = sample_draws(
        configuration,
        sampler,
        m_draws=2,
        master_seed=0,
        config_index=0,
    )
    assert len(draws) == 2
    pairs = {(d.batch_size, d.seq_len) for d in draws}
    assert len(pairs) == 2
    assert draws[0].subject_id != draws[1].subject_id
    assert not hasattr(WorkloadDraw, "precision")
    for draw in draws:
        assert not hasattr(draw, "precision")


def test_same_architecture_different_workload_are_distinct_subjects() -> None:
    family = ApertusFamilyCore()
    sampler = SamplerConfig(
        knobs=FreeKnobCatalog(
            gqa_group=(4,),
            num_kv_heads=(2,),
            head_dim=(4,),
            num_layers=(2,),
            ffn_mult=(2.0,),
            vocab_size=(100,),
        ),
        workload=WorkloadCatalog(seq_len=(8, 16), batch_size=(1,), min_seq_len=8),
        precisions=("fp32", "fp16"),
    )
    configuration = sample_configurations(
        family, sampler, n_configs=1, master_seed=1
    )[0]
    draws = sample_draws(
        configuration, sampler, m_draws=2, master_seed=1, config_index=0
    )
    assert draws[0].configuration_id == draws[1].configuration_id
    assert draws[0].subject_id != draws[1].subject_id


def test_default_precisions_are_fp32_and_fp16() -> None:
    field = next(f for f in fields(SamplerConfig) if f.name == "precisions")
    assert field.default == ("fp32", "fp16")


def test_sampler_config_rejects_mixed_precision() -> None:
    with pytest.raises(ValueError, match="unknown precision"):
        SamplerConfig(
            knobs=FreeKnobCatalog(
                gqa_group=(1,),
                num_kv_heads=(2,),
                head_dim=(4,),
                num_layers=(2,),
                ffn_mult=(2.0,),
                vocab_size=(100,),
            ),
            workload=WorkloadCatalog(seq_len=(8,), batch_size=(1,), min_seq_len=8),
            precisions=("mixed",),  # type: ignore[arg-type]
        )


def test_sampler_config_rejects_empty_precisions() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SamplerConfig(
            knobs=FreeKnobCatalog(
                gqa_group=(1,),
                num_kv_heads=(2,),
                head_dim=(4,),
                num_layers=(2,),
                ffn_mult=(2.0,),
                vocab_size=(100,),
            ),
            workload=WorkloadCatalog(seq_len=(8,), batch_size=(1,), min_seq_len=8),
            precisions=(),
        )
