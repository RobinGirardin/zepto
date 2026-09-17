"""Provenance registration stubs for Step 5 mixer regions (variants via kernel-full)."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph

from ...context import InvocationContext
from ...helpers import RegionEstimationContext
from ...region import ProvenanceMatchRule, Region
from ...registry import RegionImplementationDescriptor


def _prov(rule_id: str, kind: str, component: str) -> ProvenanceMatchRule:
    return ProvenanceMatchRule(
        id=rule_id,
        kind=kind,
        priority=10,
        component_type=component,
        require_contiguous_in_graph_order=True,
    )


MAMBA2_SCAN_PROVENANCE = _prov(
    "prov-mamba2-scan", "region/mamba2_scan", "SelectiveSSMScan"
)
GATED_RMS_NORM_PROVENANCE = _prov(
    "prov-gated-rms-norm", "region/gated_rms_norm", "GatedRMSNorm"
)
GATED_GROUPED_RMS_NORM_PROVENANCE = _prov(
    "prov-gated-grouped-rms-norm",
    "region/gated_grouped_rms_norm",
    "GatedGroupedRMSNorm",
)
GATED_DELTA_NET_PROVENANCE = _prov(
    "prov-gated-delta-net", "region/gated_delta_net", "GatedDeltaNet"
)
MAMBA2_MIXER_PROVENANCE = _prov(
    "prov-mamba2-mixer", "region/mamba2_mixer", "Mamba2Mixer"
)


@dataclass(frozen=True, slots=True)
class _StubMixerRegion:
    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        _ = (region, graph)
        if "fused" not in context.requested_capabilities:
            return "stub region requires fused capability"
        return "fused variant not implemented (kernel-full pending)"

    def lower(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
        edge_map: dict,
        *,
        estimation: RegionEstimationContext,
        lowered_edges: dict,
    ) -> LoweredNode:
        raise NotImplementedError(self.descriptor.id)


def _stub(kind: str, provenance: ProvenanceMatchRule) -> _StubMixerRegion:
    return _StubMixerRegion(
        descriptor=RegionImplementationDescriptor(
            id=kind,
            kind=kind,
            priority=0,
            capabilities=frozenset({"fused"}),
            provenance_rule=provenance,
        )
    )


MIXER_STUB_REGIONS: tuple[_StubMixerRegion, ...] = (
    _stub("region/mamba2_scan", MAMBA2_SCAN_PROVENANCE),
    _stub("region/gated_rms_norm", GATED_RMS_NORM_PROVENANCE),
    _stub("region/gated_grouped_rms_norm", GATED_GROUPED_RMS_NORM_PROVENANCE),
    _stub("region/gated_delta_net", GATED_DELTA_NET_PROVENANCE),
    _stub("region/mamba2_mixer", MAMBA2_MIXER_PROVENANCE),
)
