"""Catalog identities match the locked hybrid worked example."""

from __future__ import annotations

from qwen38_validity import Catalog, FreeKnobs, derive_architecture
from qwen38_validity.identities import assert_identities, round_to_multiple


def test_derivation_matches_qwen38_family_core() -> None:
    from zepto.empirical.models.qwen38_core import Qwen38FamilyCore

    knobs = FreeKnobs(
        num_cycles=1,
        hidden_size=256,
        delta_qk_heads=4,
        delta_v_group=3,
        delta_head_dim=32,
        attn_kv_heads=2,
        attn_gqa_group=6,
        attn_head_dim=64,
        ffn_mult=5.25,
        vocab_size=1024,
    )
    architecture = derive_architecture(knobs)
    family = Qwen38FamilyCore().derive_options(
        {
            "num_cycles": 1,
            "hidden_size": 256,
            "delta_qk_heads": 4,
            "delta_v_group": 3,
            "delta_head_dim": 32,
            "attn_kv_heads": 2,
            "attn_gqa_group": 6,
            "attn_head_dim": 64,
            "ffn_mult": 5.25,
            "vocab_size": 1024,
        }
    )
    assert architecture.as_options() == family


def test_worked_example_derivation() -> None:
    knobs = FreeKnobs(
        num_cycles=1,
        hidden_size=256,
        delta_qk_heads=4,
        delta_v_group=3,
        delta_head_dim=32,
        attn_kv_heads=2,
        attn_gqa_group=6,
        attn_head_dim=64,
        ffn_mult=5.25,
        vocab_size=1024,
    )
    architecture = derive_architecture(knobs)
    assert architecture.num_layers == 4
    assert architecture.delta_num_v_heads == 12
    assert architecture.attn_num_q_heads == 12
    assert architecture.intermediate_size == 1344
    assert architecture.intermediate_size == round_to_multiple(5.25 * 256)
    assert architecture.hidden_size != architecture.attn_num_q_heads * architecture.attn_head_dim
    assert_identities(architecture)


def test_default_catalog_architectures_are_valid() -> None:
    catalog = Catalog()
    pairs = catalog.architectures()
    assert len(pairs) == 128
    for knobs, architecture in pairs:
        assert_identities(architecture)
        assert architecture.num_layers == 4 * knobs.num_cycles
        assert architecture.attn_num_q_heads == knobs.attn_kv_heads * knobs.attn_gqa_group
        assert architecture.hidden_size == knobs.hidden_size
        assert knobs.attn_gqa_group not in architecture.as_options()
        assert knobs.num_cycles not in architecture.as_options()
        assert knobs.ffn_mult not in architecture.as_options()


def test_large_vocab_is_off_by_default() -> None:
    catalog = Catalog()
    assert 248320 not in catalog.knobs.vocab_values()
    on = Catalog(knobs=type(catalog.knobs)(include_large_vocab=True, large_vocab=248320))
    assert 248320 in on.knobs.vocab_values()


def test_workload_respects_rope_window_and_token_cap() -> None:
    from qwen38_validity.catalog import WorkloadCatalog

    pairs = WorkloadCatalog(max_tokens=64).feasible_pairs()
    assert (1, 32) in pairs
    assert (2, 32) in pairs
    assert (4, 32) not in pairs
    assert all(seq >= 32 for _, seq in pairs)
