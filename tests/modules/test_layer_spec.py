from zepto.compose import Tensor, compose_graph
from zepto.modules.blocks.hybrid_decoder_block import HybridDecoderBlock
from zepto.modules.mixers.layer_spec import LayerSpec
from zepto.modules.mixers.mixer_config import Mamba2MixerConfig
from zepto.modules.mixers.mixer_presets import nemotron_h_layer_specs, qwen35_language_layer_specs
from zepto.modules.moe.moe_presets import nemotron_moe_block
from zepto.modules.blocks.qwen35_language_decoder_block import Qwen35LanguageDecoderBlock


def test_qwen_preset_length_and_ffn() -> None:
    specs = qwen35_language_layer_specs()
    assert len(specs) == 64
    assert all(s.ffn == "swiglu" for s in specs)


def test_nemotron_preset_counts() -> None:
    specs = nemotron_h_layer_specs()
    assert len(specs) == 52
    kinds = [s.mixer for s in specs]
    assert kinds.count("mamba2") == 23
    assert kinds.count("attention") == 6
    assert kinds.count("moe") == 23


def test_compose_mini_mamba_block() -> None:
    spec = LayerSpec(
        mixer="mamba2",
        mamba2=Mamba2MixerConfig(
            hidden_size=64,
            num_heads=2,
            head_dim=8,
            state_size=16,
            num_groups=2,
        ),
    )
    graph = compose_graph(
        lambda _ctx: HybridDecoderBlock(spec),
        (Tensor(shape=(4, 64)),),
    )
    assert len(graph.nodes) > 0


def test_compose_qwen_delta_block() -> None:
    spec = qwen35_language_layer_specs()[0]
    graph = compose_graph(
        lambda _ctx: Qwen35LanguageDecoderBlock(spec),
        (Tensor(shape=(4, 5120)),),
    )
    assert len(graph.nodes) > 0


def test_compose_nemotron_moe_hybrid() -> None:
    spec = nemotron_h_layer_specs()[1]
    assert spec.mixer == "moe"
    graph = compose_graph(
        lambda _ctx: HybridDecoderBlock(
            spec,
            moe_factory=lambda: nemotron_moe_block(hidden_size=2688, seq_len=4),
        ),
        (Tensor(shape=(4, 2688)),),
    )
    assert len(graph.nodes) > 0
