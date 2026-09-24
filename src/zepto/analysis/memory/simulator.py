"""Resource event simulator with Atto-aligned peak semantics."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis.lowered import LoweredEdge, LoweredGraph, LoweredNode
from zepto.analysis.resolved import ResolvedValue
from zepto.semantic.metadata import TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from .schedule import augment_timeline_with_scheduled_releases, build_last_use_map
from .types import (
    AttributionSlice,
    MemoryBreakdown,
    SimulationResult,
    StorageSlot,
    TimelineEvent,
    _AttributionAccumulator,
)


class ResourceEventSimulator:
    """Simulate logical resource lifetimes for one lowered invocation."""

    def __init__(self, lowered: LoweredGraph) -> None:
        self._lowered = lowered
        self._context = lowered.context
        self._slots: dict[str, StorageSlot] = {}
        self._timeline: list[TimelineEvent] = []
        self._sum_all = 0
        self._breakdown = MemoryBreakdown()
        self._edge_to_storage: dict[str, str] = {}
        self._node_workspace: dict[int, set[str]] = {}
        self._output_edges = set(lowered.output_edge_ids)
        self._module_rollups: dict[str, _AttributionAccumulator] = {}
        self._region_rollups: dict[str, _AttributionAccumulator] = {}
        self._implementation_rollups: dict[str, _AttributionAccumulator] = {}
        self._live = 0
        self._naive_peak = 0

    def run(self) -> SimulationResult:
        """Execute Pass 1 event collection and Pass 2 schedule augmentation."""
        self._bootstrap_parameters()
        self._bootstrap_inputs()
        self._bootstrap_state_ports()

        for node_index, node in enumerate(self._lowered.nodes):
            if not self._node_active(node):
                continue
            for event in node.resource_events:
                if not self._event_active(event):
                    continue
                self._apply_event(event, node_index, node)
            self._node_cleanup(node_index, node)
            self._record_node_attribution(node_index, node)

        naive_peak = self._naive_peak
        last_use = build_last_use_map(self._lowered)
        augmented = augment_timeline_with_scheduled_releases(
            self._timeline, last_use, self._lowered
        )
        peak = self._recompute_timeline_with_releases(augmented)

        live_timeline = tuple(
            (event.node_index, event.cumulative_live) for event in augmented
        )

        return SimulationResult(
            sum_all_bytes=self._sum_all,
            peak_live_bytes=peak,
            peak_live_bytes_naive=naive_peak,
            breakdown=self._breakdown,
            timeline=tuple(augmented),
            by_module=self._finalize_rollups(self._module_rollups),
            by_region=self._finalize_rollups(self._region_rollups),
            by_implementation=self._finalize_rollups(self._implementation_rollups),
            live_timeline=live_timeline,
        )

    def _event_active(self, event: ResourceEvent) -> bool:
        phase = self._context.phase
        if phase == "full":
            return True
        return event.phase == phase

    def _node_active(self, node: LoweredNode) -> bool:
        del node
        return True

    def _bytes_for_edge(self, edge_id: str) -> int:
        edge = self._lowered.edges[edge_id]
        return self._context.accounting.bytes_for(
            ResolvedValue(tensor=edge.tensor, role=edge.role, dtype=edge.tensor.dtype)
        )

    def _bytes_for_parameter(self, param_id: str) -> int:
        param = self._lowered.parameters[param_id]
        return self._context.accounting.bytes_for(
            ResolvedValue(
                tensor=param.tensor, role=param.role, dtype=param.tensor.dtype
            )
        )

    def _bootstrap_parameters(self) -> None:
        for param_id, param in self._lowered.parameters.items():
            byte_count = self._bytes_for_parameter(param_id)
            self._sum_all += byte_count
            self._breakdown = replace(
                self._breakdown, parameters=self._breakdown.parameters + byte_count
            )
            self._allocate_slot(
                storage_id=param.storage_id,
                edge_id=param_id,
                bytes=byte_count,
                kind="parameter",
                node_index=-1,
                phase="bootstrap",
                pinned=True,
            )
            self._persist_slot(param.storage_id)

    def _bootstrap_inputs(self) -> None:
        for edge_id, edge in self._lowered.edges.items():
            if edge.role is not TensorRole.INPUT:
                continue
            byte_count = self._bytes_for_edge(edge_id)
            self._edge_to_storage[edge_id] = edge.storage_id
            if edge.tensor.persistent:
                self._sum_all += byte_count
                self._breakdown = replace(
                    self._breakdown,
                    persistent_inputs=self._breakdown.persistent_inputs + byte_count,
                )
                self._allocate_slot(
                    storage_id=edge.storage_id,
                    edge_id=edge_id,
                    bytes=byte_count,
                    kind="persistent_input",
                    node_index=-1,
                    phase="bootstrap",
                    pinned=True,
                )
                self._persist_slot(edge.storage_id)
            else:
                self._sum_all += byte_count
                self._breakdown = replace(
                    self._breakdown,
                    activations=self._breakdown.activations + byte_count,
                )
                self._allocate_slot(
                    storage_id=edge.storage_id,
                    edge_id=edge_id,
                    bytes=byte_count,
                    kind="activation",
                    node_index=-1,
                    phase="bootstrap",
                )

    def _bootstrap_state_ports(self) -> None:
        from zepto.analysis.horizon.state import state_object_bytes

        for name, value in self._context.state:
            byte_count = state_object_bytes(value)
            if byte_count <= 0:
                continue
            storage_id = f"state:{name}"
            self._sum_all += byte_count
            self._breakdown = replace(
                self._breakdown, state=self._breakdown.state + byte_count
            )
            self._allocate_slot(
                storage_id=storage_id,
                edge_id=None,
                bytes=byte_count,
                kind="state",
                node_index=-1,
                phase="bootstrap",
                pinned=True,
            )
            self._persist_slot(storage_id)

    def _apply_event(
        self,
        event: ResourceEvent,
        node_index: int,
        node: LoweredNode,
    ) -> None:
        if event.kind is ResourceEventKind.ALLOCATE:
            self._handle_allocate(event, node_index, node)
        elif event.kind is ResourceEventKind.ALIAS:
            self._handle_alias(event, node_index, node)
        elif event.kind is ResourceEventKind.SAVE:
            self._handle_save(event, node_index)
        elif event.kind is ResourceEventKind.PERSIST:
            self._handle_persist(event, node_index)
        elif event.kind is ResourceEventKind.RELEASE:
            self._handle_release(event, node_index)
        elif event.kind is ResourceEventKind.WORKSPACE:
            self._handle_workspace(event, node_index, node)

    def _handle_allocate(
        self,
        event: ResourceEvent,
        node_index: int,
        node: LoweredNode,
    ) -> None:
        edge_id = event.value
        byte_count = self._resolve_bytes(edge_id)
        kind = self._classify_allocate(edge_id, node, event)
        storage_id = self._storage_id_for(edge_id)

        self._sum_all += byte_count
        self._breakdown = self._increment_breakdown(kind, byte_count)

        self._allocate_slot(
            storage_id=storage_id,
            edge_id=edge_id,
            bytes=byte_count,
            kind=kind,
            node_index=node_index,
            phase=event.phase,
            producer_node=node_index,
        )
        self._edge_to_storage[edge_id] = storage_id

        if kind == "workspace":
            self._node_workspace.setdefault(node_index, set()).add(storage_id)

    def _handle_alias(
        self,
        event: ResourceEvent,
        node_index: int,
        node: LoweredNode,
    ) -> None:
        alias_edge_id = event.value
        source_edge_id = event.storage or self._alias_source(node, alias_edge_id)
        source_storage = self._storage_id_for(source_edge_id)
        self._edge_to_storage[alias_edge_id] = source_storage
        slot = self._slots.get(source_storage)
        if slot is not None:
            slot.refcount += 1
        self._append_timeline(
            TimelineEvent(
                node_index=node_index,
                kind=ResourceEventKind.ALIAS,
                edge_id=alias_edge_id,
                storage_id=source_storage,
                delta_live=0,
                phase=event.phase,
            )
        )

    def _handle_save(self, event: ResourceEvent, node_index: int) -> None:
        storage_id = self._storage_id_for(event.value)
        slot = self._slots.get(storage_id)
        if slot is None:
            return
        if slot.kind != "saved":
            self._breakdown = replace(
                self._breakdown,
                saved_for_backward=self._breakdown.saved_for_backward + slot.bytes,
            )
        slot.pinned = True
        slot.kind = "saved"
        self._append_timeline(
            TimelineEvent(
                node_index=node_index,
                kind=ResourceEventKind.SAVE,
                edge_id=event.value,
                storage_id=storage_id,
                delta_live=0,
                phase=event.phase,
            )
        )

    def _handle_persist(self, event: ResourceEvent, node_index: int) -> None:
        storage_id = self._storage_id_for(event.value)
        self._persist_slot(storage_id)
        self._append_timeline(
            TimelineEvent(
                node_index=node_index,
                kind=ResourceEventKind.PERSIST,
                edge_id=event.value,
                storage_id=storage_id,
                delta_live=0,
                phase=event.phase,
            )
        )

    def _handle_release(self, event: ResourceEvent, node_index: int) -> None:
        storage_id = self._storage_id_for(event.value)
        delta = self._release_storage(storage_id)
        self._append_timeline(
            TimelineEvent(
                node_index=node_index,
                kind=ResourceEventKind.RELEASE,
                edge_id=event.value,
                storage_id=storage_id,
                delta_live=delta,
                phase=event.phase,
            )
        )

    def _handle_workspace(
        self,
        event: ResourceEvent,
        node_index: int,
        node: LoweredNode,
    ) -> None:
        allocate_event = ResourceEvent(ResourceEventKind.ALLOCATE, event.value)
        self._handle_allocate(allocate_event, node_index, node)

    def _node_cleanup(self, node_index: int, node: LoweredNode) -> None:
        for storage_id in self._node_workspace.get(node_index, ()):
            slot = self._slots.get(storage_id)
            if slot is None or slot.pinned:
                continue
            if slot.edge_id in self._output_edges:
                continue
            delta = self._release_storage(storage_id)
            if delta:
                self._append_timeline(
                    TimelineEvent(
                        node_index=node_index,
                        kind=ResourceEventKind.RELEASE,
                        edge_id=slot.edge_id,
                        storage_id=storage_id,
                        delta_live=delta,
                        phase="cleanup",
                    )
                )

        for edge_id in node.output_edges:
            if edge_id in self._output_edges:
                continue
            storage_id = self._storage_id_for(edge_id)
            slot = self._slots.get(storage_id)
            if slot is None or slot.pinned:
                continue
            if slot.producer_node != node_index:
                continue
            if slot.kind not in ("activation", "workspace"):
                continue
            if storage_id in self._node_workspace.get(node_index, ()):
                continue
            # Forward activations survive until Pass 2 schedule release.

    def _allocate_slot(
        self,
        *,
        storage_id: str,
        edge_id: str | None,
        bytes: int,
        kind: StorageSlot["kind"],
        node_index: int,
        phase: str,
        pinned: bool = False,
        producer_node: int | None = None,
    ) -> None:
        existing = self._slots.get(storage_id)
        if existing is not None and existing.live:
            existing.refcount += 1
            delta = 0
        else:
            self._slots[storage_id] = StorageSlot(
                storage_id=storage_id,
                bytes=bytes,
                live=True,
                refcount=1,
                kind=kind,
                pinned=pinned,
                producer_node=producer_node if producer_node is not None else node_index,
                edge_id=edge_id,
            )
            delta = bytes

        self._append_timeline(
            TimelineEvent(
                node_index=node_index,
                kind=ResourceEventKind.ALLOCATE,
                edge_id=edge_id,
                storage_id=storage_id,
                delta_live=delta,
                phase=phase,
            )
        )

    def _persist_slot(self, storage_id: str) -> None:
        slot = self._slots.get(storage_id)
        if slot is not None:
            slot.pinned = True

    def _release_storage(self, storage_id: str) -> int:
        slot = self._slots.get(storage_id)
        if slot is None or not slot.live:
            return 0
        if slot.pinned:
            return 0
        slot.refcount -= 1
        if slot.refcount > 0:
            return 0
        slot.live = False
        return -slot.bytes

    def _storage_id_for(self, edge_or_storage: str) -> str:
        if edge_or_storage in self._edge_to_storage:
            return self._edge_to_storage[edge_or_storage]
        if edge_or_storage in self._lowered.edges:
            return self._lowered.edges[edge_or_storage].storage_id
        if edge_or_storage in self._lowered.parameters:
            return self._lowered.parameters[edge_or_storage].storage_id
        return edge_or_storage

    def _resolve_bytes(self, edge_id: str) -> int:
        if edge_id in self._lowered.edges:
            return self._bytes_for_edge(edge_id)
        if edge_id in self._lowered.parameters:
            return self._bytes_for_parameter(edge_id)
        return 0

    def _classify_allocate(
        self,
        edge_id: str,
        node: LoweredNode,
        event: ResourceEvent,
    ) -> StorageSlot["kind"]:
        if edge_id in self._lowered.edges:
            edge = self._lowered.edges[edge_id]
            if edge.tensor.semantic_type == "kv_cache":
                return "state"
        if self._followed_by_persist(edge_id, node, event):
            return "weight_grad"
        if edge_id in self._lowered.edges:
            edge = self._lowered.edges[edge_id]
            if edge.role is TensorRole.GRADIENT:
                return "gradient_wavefront"
            if edge.role is TensorRole.WORKSPACE or edge.workspace:
                return "workspace"
        return "activation"

    def _followed_by_persist(
        self,
        edge_id: str,
        node: LoweredNode,
        event: ResourceEvent,
    ) -> bool:
        seen_allocate = False
        for node_event in node.resource_events:
            if node_event is event:
                seen_allocate = True
                continue
            if not seen_allocate:
                continue
            if node_event.value != edge_id:
                continue
            return node_event.kind is ResourceEventKind.PERSIST
        return False

    def _increment_breakdown(
        self, kind: StorageSlot["kind"], byte_count: int
    ) -> MemoryBreakdown:
        breakdown = self._breakdown
        if kind == "activation":
            return replace(breakdown, activations=breakdown.activations + byte_count)
        if kind == "workspace":
            return replace(breakdown, workspace=breakdown.workspace + byte_count)
        if kind == "gradient_wavefront":
            return replace(breakdown, gradients=breakdown.gradients + byte_count)
        if kind == "weight_grad":
            return replace(breakdown, weight_grads=breakdown.weight_grads + byte_count)
        if kind == "saved":
            return replace(
                breakdown, saved_for_backward=breakdown.saved_for_backward + byte_count
            )
        if kind == "state":
            return replace(breakdown, state=breakdown.state + byte_count)
        return breakdown

    def _alias_source(self, node: LoweredNode, alias_edge_id: str) -> str:
        if alias_edge_id in node.output_edges and node.input_edges:
            return node.input_edges[0]
        return alias_edge_id

    def _append_timeline(self, event: TimelineEvent) -> None:
        self._live += event.delta_live
        self._naive_peak = max(self._naive_peak, self._live)
        event = replace(event, cumulative_live=self._live)
        self._timeline.append(event)

    def _recompute_timeline_with_releases(
        self, timeline: list[TimelineEvent]
    ) -> int:
        slot_states: dict[str, tuple[int, int, bool, bool]] = {}
        live = 0
        peak = 0
        result_timeline: list[TimelineEvent] = []

        for event in timeline:
            if event.scheduled:
                delta = self._scheduled_release_delta(event.storage_id, slot_states)
            else:
                delta = event.delta_live
                self._update_replay_slot_state(event, slot_states)

            live += delta
            peak = max(peak, live)
            result_timeline.append(
                replace(event, delta_live=delta, cumulative_live=live)
            )

        self._timeline = result_timeline
        return peak

    def _scheduled_release_delta(
        self,
        storage_id: str,
        slot_states: dict[str, tuple[int, int, bool, bool]],
    ) -> int:
        state = slot_states.get(storage_id)
        if state is None or not state[2]:
            return 0
        byte_count, _refcount, _is_live, pinned = state
        slot = self._slots.get(storage_id)
        if slot is not None and slot.kind in ("parameter", "weight_grad", "state"):
            return 0
        if pinned and slot is not None and slot.kind == "persistent_input":
            return 0
        slot_states[storage_id] = (byte_count, 0, False, False)
        return -byte_count

    def _update_replay_slot_state(
        self,
        event: TimelineEvent,
        slot_states: dict[str, tuple[int, int, bool, bool]],
    ) -> None:
        storage_id = event.storage_id
        if event.kind is ResourceEventKind.ALLOCATE:
            byte_count = max(event.delta_live, 0)
            if byte_count == 0:
                state = slot_states.get(storage_id)
                if state is not None:
                    slot_states[storage_id] = (
                        state[0],
                        state[1] + 1,
                        state[2],
                        state[3],
                    )
                return
            slot_states[storage_id] = (byte_count, 1, True, False)
        elif event.kind is ResourceEventKind.ALIAS:
            state = slot_states.get(storage_id)
            if state is not None:
                slot_states[storage_id] = (
                    state[0],
                    state[1] + 1,
                    state[2],
                    state[3],
                )
        elif event.kind in (ResourceEventKind.PERSIST, ResourceEventKind.SAVE):
            state = slot_states.get(storage_id)
            if state is not None:
                slot_states[storage_id] = (state[0], state[1], state[2], True)
        elif event.kind is ResourceEventKind.RELEASE:
            state = slot_states.get(storage_id)
            if state is None:
                return
            byte_count, refcount, is_live, pinned = state
            if pinned or not is_live:
                return
            refcount -= 1
            slot_states[storage_id] = (
                byte_count,
                refcount,
                refcount > 0,
                pinned,
            )

    def _record_node_attribution(self, node_index: int, node: LoweredNode) -> None:
        live = self._timeline[-1].cumulative_live if self._timeline else 0
        module_key = "/".join(node.module_path) or "root"
        region_key = node.region_id or ""
        impl_key = node.implementation

        for key, rollups in (
            (module_key, self._module_rollups),
            (region_key, self._region_rollups),
            (impl_key, self._implementation_rollups),
        ):
            if not key:
                continue
            acc = rollups.setdefault(key, _AttributionAccumulator())
            acc.forward_flops += node.forward_flops
            acc.backward_flops += node.backward_flops
            acc.peak_live_at_node[node_index] = live

    def _finalize_rollups(
        self, rollups: dict[str, _AttributionAccumulator]
    ) -> tuple[AttributionSlice, ...]:
        slices: list[AttributionSlice] = []
        for key, acc in sorted(rollups.items()):
            peak = max(acc.peak_live_at_node.values()) if acc.peak_live_at_node else 0
            slices.append(
                AttributionSlice(
                    key=key,
                    forward_flops=acc.forward_flops,
                    backward_flops=acc.backward_flops,
                    allocated_bytes=acc.allocated_bytes,
                    peak_bytes=peak,
                    persistent_bytes=acc.persistent_bytes,
                    workspace_bytes=acc.workspace_bytes,
                )
            )
        return tuple(slices)
