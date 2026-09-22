"""Shared assertions for batched region lowering tests."""

from __future__ import annotations

from zepto.semantic.operations.helpers import numel


def assert_flops_scales(batched_flops: int, single_flops: int, batch: int) -> None:
    assert batched_flops == batch * single_flops


def aux_numel(lowered, node) -> int:
    return sum(numel(lowered.edges[aux_id].tensor) for aux_id in node.auxiliary_edges)


def assert_aux_numel_scales(batched_lowered, batched_node, single_lowered, single_node, batch: int) -> None:
    assert aux_numel(batched_lowered, batched_node) == batch * aux_numel(
        single_lowered, single_node
    )


def aux_tensor(lowered, node, fragment: str):
    matches = [aux_id for aux_id in node.auxiliary_edges if fragment in aux_id]
    assert matches, f"no auxiliary edge matching {fragment!r} in {node.auxiliary_edges}"
    return lowered.edges[matches[0]].tensor
