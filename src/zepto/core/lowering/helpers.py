"""Shared helpers for the lowering pass."""

from __future__ import annotations

from dataclasses import replace

from ..graph import StructuralGraph
from ..ids import TensorId
from ..metadata import DType, TensorMetadata, TensorRole
from ..operation.records import EstimationContext, ResourceEvent, ResourceEventKind
from ..operation.structural import StructuralOperation
from ..tensor import Tensor
from .context import InvocationContext


def structural_storage_id(tensor: Tensor) -> str:
    """Return a string storage identity for accounting."""
    if tensor.storage_id is None:
        return f"anon:{tensor.id.index}"
    return f"s{tensor.storage_id.index}"


def resolve_tensor_metadata(
    metadata: TensorMetadata,
    context: InvocationContext,
    *,
    role: TensorRole | None = None,
) -> TensorMetadata:
    """Resolve dtype and optional role using the invocation policy."""
    resolved_dtype = context.accounting.resolve_dtype(metadata)
    resolved_role = role if role is not None else metadata.role
    if resolved_role is None:
        resolved_role = TensorRole.ACTIVATION
    return replace(
        metadata,
        dtype=resolved_dtype,
        role=resolved_role,
    )


def infer_tensor_role(
    tensor_id: TensorId,
    graph: StructuralGraph,
) -> TensorRole:
    """Infer the accounting role for a structural tensor."""
    if tensor_id in graph.inputs:
        return TensorRole.INPUT
    return TensorRole.ACTIVATION


def ensure_lowered_tensor(
    tensor_id: TensorId,
    graph: StructuralGraph,
    context: InvocationContext,
    lowered_tensors: dict[str, LoweredTensor],
    tensor_map: dict[TensorId, str],
) -> str:
    """Materialize one structural tensor in the lowered graph if needed."""
    from ..lowered import LoweredTensor

    if tensor_id in tensor_map:
        return tensor_map[tensor_id]

    structural = graph.tensor(tensor_id)
    lowered_id = f"t{tensor_id.index}"
    metadata = resolve_tensor_metadata(
        structural.metadata,
        context,
        role=infer_tensor_role(tensor_id, graph),
    )
    lowered_tensors[lowered_id] = LoweredTensor(
        id=lowered_id,
        structural_tensor_id=tensor_id,
        metadata=metadata,
        storage_id=structural_storage_id(structural),
    )
    tensor_map[tensor_id] = lowered_id
    return lowered_id


def build_estimation_context(
    structural: StructuralOperation,
    graph: StructuralGraph,
    context: InvocationContext,
) -> EstimationContext:
    """Build per-operation estimation metadata from bound port values."""
    port_metadata: list[tuple[str, TensorMetadata]] = []
    for port, tensor_id in zip(
        structural.input_ports, structural.input_tensors, strict=True
    ):
        metadata = resolve_tensor_metadata(
            graph.tensor(tensor_id).metadata,
            context,
        )
        port_metadata.append((port.name, metadata))
    assert structural.result is not None
    for port, metadata in zip(
        structural.output_ports, structural.result.outputs, strict=True
    ):
        resolved = resolve_tensor_metadata(metadata, context)
        port_metadata.append((port.name, resolved))
    return EstimationContext(
        phase=context.phase,
        port_metadata=tuple(port_metadata),
        precision=context.precision,
        state=context.state,
    )


def port_tensor_id(
    structural: StructuralOperation,
    port_name: str,
    role: str,
) -> TensorId:
    """Resolve a port name to its bound structural tensor id."""
    ports = structural.input_ports if role == "input" else structural.output_ports
    tensor_ids = (
        structural.input_tensors if role == "input" else structural.output_tensors
    )
    for port, tensor_id in zip(ports, tensor_ids, strict=True):
        if port.name == port_name:
            return tensor_id
    raise KeyError(f"Unknown {role} port {port_name!r}")


def remap_events(
    events: tuple[ResourceEvent, ...],
    structural: StructuralOperation,
    tensor_map: dict[TensorId, str],
) -> tuple[ResourceEvent, ...]:
    """Remap port-relative event targets to lowered tensor ids."""
    port_targets: dict[str, str] = {}
    for index, port in enumerate(structural.input_ports):
        port_targets[port.name] = tensor_map[structural.input_tensors[index]]
    for index, port in enumerate(structural.output_ports):
        lowered = tensor_map[structural.output_tensors[index]]
        port_targets[port.name] = lowered
        port_targets[f"output:{index}"] = lowered

    def remap(value: str | None) -> str | None:
        if value is None:
            return None
        if value in port_targets:
            return port_targets[value]
        if value.startswith("output:"):
            index = int(value.split(":", 1)[1])
            return tensor_map[structural.output_tensors[index]]
        return value

    return tuple(
        replace(event, value=remap(event.value), storage=remap(event.storage))
        for event in events
    )


def append_save_events(
    events: tuple[ResourceEvent, ...],
    structural: StructuralOperation,
    tensor_map: dict[TensorId, str],
) -> tuple[ResourceEvent, ...]:
    """Append SAVE events for structural saved-for-backward port refs."""
    save_events: list[ResourceEvent] = []
    for port_ref in structural.saved_for_backward:
        tensor_id = port_tensor_id(structural, port_ref.port_name, port_ref.role)
        save_events.append(
            ResourceEvent(ResourceEventKind.SAVE, tensor_map[tensor_id])
        )
    return (*events, *save_events)


def register_auxiliary_tensor(
    aux_id: str,
    metadata: TensorMetadata,
    *,
    storage_id: str | None = None,
    lowered_tensors: dict[str, LoweredTensor],
) -> None:
    """Register a lowering-local auxiliary tensor."""
    from ..lowered import LoweredTensor

    if aux_id in lowered_tensors:
        return
    lowered_tensors[aux_id] = LoweredTensor(
        id=aux_id,
        structural_tensor_id=None,
        metadata=metadata,
        storage_id=storage_id or aux_id,
        workspace=False,
    )


def is_zero_operand(tensor: Tensor, graph: StructuralGraph) -> bool:
    """Return whether a tensor is provably a constant-zero operand."""
    del graph
    return tensor.metadata.semantic_type == "constant_zero"
