from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.l2_normalize import L2Normalize


def test_l2_normalize_compose() -> None:
    graph = compose_graph(
        lambda _ctx: L2Normalize(),
        (Tensor(shape=(4, 32), requires_grad=True),),
    )
    families = {graph.node(n).operation_family for n in graph.nodes}
    assert "divide" in families
    assert "inv_norm_size" not in {
        graph.edge(e).tensor.semantic_type for e in graph.edges
    }
