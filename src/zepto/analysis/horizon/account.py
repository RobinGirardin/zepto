"""Horizon timeline reducers for memory and FLOP accounting."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis.reports.flops import FlopReport
from zepto.analysis.reports.memory import MemoryBreakdown, MemoryReport

from ..memory.simulator import ResourceEventSimulator
from ..optimizer import AdamW, optimizer_update_flops
from ..reports.attribution import AttributionSlice
from ..reports.horizon import HorizonFlopReport, HorizonMemoryReport
from ..resolved import ResolvedValue
from ..runtime import runtime_workspace_bytes
from .records import HorizonSimulation, InvocationRecord
from .spec import StepKind
from .state import parameter_bytes, snapshot_state_bytes, trainable_parameter_elements
from .training_boundary import training_boundary_cost


class HorizonMemoryReducer:
    """Merge per-step timelines with carried persistent storage."""

    def reduce(self, sim: HorizonSimulation) -> HorizonMemoryReport:
        per_step: list[MemoryReport] = []
        merged_breakdown = MemoryBreakdown()
        sum_all = 0
        merged_peak = 0
        param_bytes: int | None = None
        carry_state = snapshot_state_bytes(sim.state_initial)

        for index, record in enumerate(sim.timeline):
            step_kind = record.step.kind
            result = ResourceEventSimulator(record.lowered).run()
            runtime_ws = runtime_workspace_bytes(
                record.lowered.context, step_kind=step_kind
            )
            breakdown = replace(result.breakdown, runtime_workspace=runtime_ws)
            peak_with_runtime = result.peak_live_bytes + runtime_ws
            report = MemoryReport(
                sum_all_bytes=result.sum_all_bytes + runtime_ws,
                peak_live_bytes=peak_with_runtime,
                breakdown=breakdown,
                by_module=result.by_module,
                by_region=result.by_region,
                by_implementation=result.by_implementation,
                live_timeline=result.live_timeline,
            )
            per_step.append(report)

            step_params = result.breakdown.parameters
            if param_bytes is None:
                param_bytes = step_params

            step_state = result.breakdown.state
            transient_peak = peak_with_runtime - step_params - step_state
            step_peak = (param_bytes or 0) + carry_state + max(transient_peak, 0)
            after_state = snapshot_state_bytes(record.state_after)
            boundary_peak = (param_bytes or 0) + after_state
            if (
                record.step.kind == StepKind.BACKWARD
                and sim.spec.optimizer_policy is not None
            ):
                trainable = trainable_parameter_elements(record.lowered)
                boundary = training_boundary_cost(
                    policy=sim.spec.optimizer_policy,
                    trainable_elements=trainable,
                    context=record.lowered.context,
                )
                boundary_peak = max(
                    boundary_peak,
                    (param_bytes or 0) + after_state + boundary.optimizer_workspace_bytes,
                )
            merged_peak = max(merged_peak, step_peak, boundary_peak)

            if index == 0:
                sum_all += report.sum_all_bytes
                merged_breakdown = breakdown
            else:
                sum_all += report.sum_all_bytes - step_params
                if step_state > 0:
                    prev_state = snapshot_state_bytes(
                        sim.timeline[index - 1].state_after
                    )
                    sum_all -= prev_state
                merged_breakdown = _merge_breakdowns(
                    merged_breakdown, breakdown, step_params, step_state
                )

            carry_state = snapshot_state_bytes(record.state_after)

        persistent_carry = snapshot_state_bytes(sim.state_final)
        if param_bytes is not None:
            persistent_carry += param_bytes

        final_state_bytes = snapshot_state_bytes(sim.state_final)
        merged_breakdown = replace(
            merged_breakdown,
            state=final_state_bytes,
        )

        return HorizonMemoryReport(
            peak_live_bytes=merged_peak,
            sum_all_bytes=sum_all,
            persistent_carry_bytes=persistent_carry,
            per_step=tuple(per_step),
            breakdown=merged_breakdown,
        )


class HorizonFlopReducer:
    """Sum FLOPs across horizon steps."""

    def reduce(self, sim: HorizonSimulation) -> HorizonFlopReport:
        from ..flops.account import account_flops

        per_step = tuple(_flops_for_record(sim, record) for record in sim.timeline)
        total_forward = sum(report.forward_flops for report in per_step)
        total_backward = sum(report.backward_flops for report in per_step)
        total = sum(report.total_flops for report in per_step)
        return HorizonFlopReport(
            total_forward_flops=total_forward,
            total_backward_flops=total_backward,
            total_flops=total,
            per_step=per_step,
        )


def _bytes_per_element(lowered) -> int:
    from ..lowered import LoweredGraph

    if not isinstance(lowered, LoweredGraph):
        raise TypeError("expected LoweredGraph")
    if not lowered.parameters:
        return 0
    param = next(iter(lowered.parameters.values()))
    accounting = lowered.context.accounting
    total_bytes = accounting.bytes_for(
        ResolvedValue(
            tensor=param.tensor,
            role=param.role,
            dtype=param.tensor.dtype,
        )
    )
    numel = 1
    for dim in param.tensor.shape:
        numel *= dim
    if numel <= 0:
        return 0
    return total_bytes // numel


def _flops_for_record(
    sim: HorizonSimulation, record: InvocationRecord
) -> FlopReport:
    from ..flops.account import account_flops

    if record.step.kind != StepKind.OPTIMIZER:
        return account_flops(record.lowered)

    policy = sim.spec.optimizer_policy or AdamW
    param_bytes = parameter_bytes(record.lowered)
    bpe = _bytes_per_element(record.lowered)
    if bpe == 0:
        update = 0
    else:
        update = optimizer_update_flops(
            policy, param_bytes, bytes_per_element=bpe
        )

    return FlopReport(
        forward_flops=0,
        backward_flops=0,
        total_flops=update,
        by_implementation=(
            AttributionSlice(
                key=f"optimizer/{policy.name}",
                forward_flops=0,
                backward_flops=update,
            ),
        ),
    )


def account_horizon_memory(sim: HorizonSimulation) -> HorizonMemoryReport:
    return HorizonMemoryReducer().reduce(sim)


def account_horizon_flops(sim: HorizonSimulation) -> HorizonFlopReport:
    return HorizonFlopReducer().reduce(sim)


def _merge_breakdowns(
    left: MemoryBreakdown,
    right: MemoryBreakdown,
    duplicate_params: int,
    duplicate_state: int,
) -> MemoryBreakdown:
    # runtime_workspace is an ephemeral per-step handle pool: keep the max
    # (peak-relevant), not the sum of 1+2+0 handles across a train timeline.
    return MemoryBreakdown(
        activations=left.activations + right.activations,
        parameters=left.parameters,
        workspace=left.workspace + right.workspace,
        runtime_workspace=max(left.runtime_workspace, right.runtime_workspace),
        saved_for_backward=left.saved_for_backward + right.saved_for_backward,
        persistent_inputs=left.persistent_inputs + right.persistent_inputs,
        gradients=left.gradients + right.gradients,
        weight_grads=left.weight_grads + right.weight_grads,
        state=left.state + max(right.state - duplicate_state, 0),
    )
