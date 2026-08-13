from __future__ import annotations

from pathlib import Path
from collections import Counter

import pytest

from masp.domain import (
    DomainError,
    LoadState,
    SegmentKind,
    TaskState,
    TransportTask,
    Vehicle,
    VehicleState,
)
from masp.online import OnlineDispatchRuntime, run_online_scenario
from masp.scenario import build_simulator
from masp.topology import MapTopology

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def online_documents():
    return (
        read_json("generated/xiate-unified-map-model.json"),
        read_json("generated/xiate-conflict-resources.json"),
        read_json("generated/xiate-workstations.json"),
        read_json("config/robot-profiles.json"),
        read_json("config/scheduler.json"),
        read_json("config/traffic-zones.json"),
    )


def build_runtime(online_documents) -> tuple[OnlineDispatchRuntime, dict]:
    model, conflicts, workstations, profiles, scheduler, zones = online_documents
    scenario = read_json("scenarios/interactive-multi-fleet.json")
    runtime = OnlineDispatchRuntime(
        topology=MapTopology(model, conflicts, workstations, zones),
        model=model,
        profiles=profiles,
        scheduler=scheduler,
        traffic_zones=zones,
        vehicles=[Vehicle.from_dict(item) for item in scenario["vehicles"]],
        end_time_ms=int(scenario["endTimeMs"]),
        policy="congestion",
        seed=0,
    )
    return runtime, scenario


def task_from_row(row: dict, scheduler: dict) -> TransportTask:
    defaults = scheduler["serviceDefaults"]
    return TransportTask.from_dict(
        row,
        int(defaults["pickupServiceMs"]),
        int(defaults["dropoffServiceMs"]),
    )


def test_online_task_is_planned_only_after_submission_and_ack(online_documents) -> None:
    runtime, scenario = build_runtime(online_documents)
    scheduler = online_documents[4]
    task = task_from_row(scenario["tasks"][0], scheduler)

    assert runtime.plan_cycle() == ()
    runtime.submit_task(task)
    runtime.advance_to(task.release_time_ms)
    proposals = runtime.plan_cycle()

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.plan.created_at_ms == task.release_time_ms
    assert runtime.simulator.plans == {}
    acknowledgement = runtime.acknowledge_plan(
        proposal.proposal_id,
        proposal.plan.revision,
        accepted=True,
    )
    assert runtime.acknowledge_plan(
        proposal.proposal_id,
        proposal.plan.revision,
        accepted=True,
    ) is acknowledgement

    runtime.advance_to(task.release_time_ms)
    assert task.state is TaskState.EN_ROUTE_PICKUP
    assert proposal.plan.plan_id in runtime.simulator.plans
    assert all(
        item["endMs"] <= proposal.safe_until_ms
        for item in proposal.to_dict()["committedSegments"]
    )


def test_online_ack_rejects_plan_after_newer_telemetry(online_documents) -> None:
    runtime, scenario = build_runtime(online_documents)
    scheduler = online_documents[4]
    task = task_from_row(scenario["tasks"][0], scheduler)
    runtime.submit_task(task)
    runtime.advance_to(task.release_time_ms)
    proposal = runtime.plan_cycle()[0]
    vehicle = runtime.simulator.vehicles[proposal.plan.vehicle_id]

    assert runtime.update_idle_telemetry(
        vehicle.vehicle_id,
        vehicle_revision=vehicle.revision + 1,
        timestamp_ms=runtime.now_ms,
        current_node_id=vehicle.current_node_id or "",
        heading_rad=vehicle.heading_rad,
        load_state=LoadState.EMPTY,
    )
    with pytest.raises(DomainError) as caught:
        runtime.acknowledge_plan(
            proposal.proposal_id,
            proposal.plan.revision,
            accepted=True,
        )
    assert caught.value.code == "online.plan.vehicle_revision_stale"

    rejected = runtime.acknowledge_plan(
        proposal.proposal_id,
        proposal.plan.revision,
        accepted=False,
    )
    assert not rejected.accepted
    assert runtime.pending_proposal_ids == set()


def test_candidate_tail_enters_live_reservations_as_tentative_intent(
    online_documents,
) -> None:
    runtime, scenario = build_runtime(online_documents)
    scheduler = online_documents[4]

    for row in scenario["tasks"]:
        if int(row["releaseTimeMs"]) == 0:
            runtime.submit_task(task_from_row(row, scheduler))
    runtime.advance_to(0)
    for proposal in runtime.plan_cycle():
        runtime.acknowledge_plan(
            proposal.proposal_id,
            proposal.plan.revision,
            accepted=True,
        )
    runtime.advance_to(90_000)

    for row in scenario["tasks"]:
        if int(row["releaseTimeMs"]) == 90_000:
            runtime.submit_task(task_from_row(row, scheduler))
    runtime.advance_to(90_000)
    proposals = runtime.plan_cycle()
    proposal = next(
        item
        for item in proposals
        if item.candidate_plan is not None
        and len(item.candidate_plan.segments) > len(item.plan.segments)
    )
    candidate_plan = proposal.candidate_plan
    assert candidate_plan is not None
    tail_segment_ids = {
        segment.segment_id
        for segment in candidate_plan.segments[len(proposal.plan.segments) :]
    }

    assert tail_segment_ids
    assert proposal.plan.segments[-1].end_ms == proposal.safe_until_ms
    proposal_tail = [
        reservation
        for reservation in proposal.reservations
        if reservation.segment_id in tail_segment_ids
    ]
    assert proposal_tail
    assert all(not reservation.committed for reservation in proposal_tail)

    runtime.acknowledge_plan(
        proposal.proposal_id,
        proposal.plan.revision,
        accepted=True,
    )
    live_rows = runtime.reservations.for_vehicle(proposal.plan.vehicle_id)
    assert set(live_rows) == set(proposal.reservations)
    assert any(
        reservation.segment_id in tail_segment_ids
        and not reservation.committed
        for reservation in live_rows
    )
    assert any(
        reservation.segment_id == "idle-tail"
        and reservation.start_ms == proposal.safe_until_ms
        and reservation.end_ms == runtime.end_time_ms
        and reservation.committed
        for reservation in live_rows
    )
    assert any(
        reservation.segment_id == "tentative-idle-tail"
        and reservation.start_ms == candidate_plan.segments[-1].end_ms
        and reservation.end_ms == runtime.end_time_ms
        and not reservation.committed
        for reservation in live_rows
    )


def test_safe_wait_is_split_at_the_nominal_commitment_boundary(
    online_documents,
) -> None:
    runtime, scenario = build_runtime(online_documents)
    scheduler = online_documents[4]
    for row in scenario["tasks"]:
        if int(row["releaseTimeMs"]) == 0:
            runtime.submit_task(task_from_row(row, scheduler))
    runtime.advance_to(0)
    for proposal in runtime.plan_cycle():
        runtime.acknowledge_plan(
            proposal.proposal_id,
            proposal.plan.revision,
            accepted=True,
        )
    runtime.advance_to(90_000)
    for row in scenario["tasks"]:
        if int(row["releaseTimeMs"]) == 90_000:
            runtime.submit_task(task_from_row(row, scheduler))
    runtime.advance_to(90_000)

    proposal = next(
        item
        for item in runtime.plan_cycle()
        if item.candidate_plan is not None
        and item.candidate_plan.segments[0].kind is SegmentKind.WAIT
        and item.candidate_plan.segments[0].end_ms > item.nominal_until_ms
    )

    assert proposal.safe_until_ms == proposal.nominal_until_ms
    assert proposal.plan.segments[-1].kind is SegmentKind.WAIT
    assert proposal.plan.segments[-1].end_ms == proposal.nominal_until_ms
    assert proposal.candidate_plan is not None
    assert (
        proposal.candidate_plan.segments[0].end_ms
        > proposal.plan.segments[-1].end_ms
    )


def test_online_interactive_scenario_completes_without_conflicts(
    online_documents,
) -> None:
    scenario = read_json("scenarios/interactive-multi-fleet.json")
    runtime = run_online_scenario(
        scenario,
        *online_documents,
        policy="congestion",
        seed=0,
    )
    result = runtime.result()
    planning = runtime.planning_result().summary()

    assert result["metrics"]["completedTaskCount"] == len(scenario["tasks"])
    assert result["metrics"]["reservationConflictRejections"] == 0
    online = result["online"]
    assert online["taskSubmissionCount"] == len(scenario["tasks"])
    assert online["planProposalCount"] == len(runtime.accepted_plans)
    assert online["acknowledgedPlanCount"] == len(runtime.accepted_plans)
    assert online["continuationPlanCount"] > 0
    assert online["rejectedPlanCount"] == 0
    assert online["pendingPlanAckCount"] == 0
    assert online["telemetryUpdateCount"] == 0
    assert planning["planFragmentCount"] == len(runtime.accepted_plans)
    assert planning["plannedTaskCount"] == len(scenario["tasks"])
    fragments_by_task = Counter(plan.task_id for plan in runtime.accepted_plans)
    assert any(count > 1 for count in fragments_by_task.values())
    for plan in runtime.accepted_plans:
        assert plan.committed_until_ms == plan.segments[-1].end_ms
        assert runtime.topology.wait_allowed(
            plan.segments[-1].end_node_id or "",
            runtime.simulator.vehicles[plan.vehicle_id].robot_group,
        )
        if plan.continuation:
            previous = [
                item
                for item in runtime.accepted_plans
                if item.task_id == plan.task_id
                and item.created_at_ms < plan.created_at_ms
            ]
            assert previous
            assert previous[-1].vehicle_id == plan.vehicle_id
    assert planning["unplannedTaskCount"] == 0
    assert all(
        plan.created_at_ms >= runtime.simulator.tasks[plan.task_id].release_time_ms
        for plan in runtime.accepted_plans
    )

    replay = build_simulator(
        runtime.planned_scenario(scenario["scenarioId"], scenario["seed"]),
        *online_documents[:3],
        online_documents[4],
        ROOT / "schemas",
    ).run()
    assert replay["metrics"]["completedTaskCount"] == len(scenario["tasks"])
    assert replay["metrics"]["reservationConflictRejections"] == 0


def test_loaded_active_task_continues_without_repeating_pickup(
    online_documents,
) -> None:
    runtime, scenario = build_runtime(online_documents)
    scheduler = online_documents[4]
    task = task_from_row(scenario["tasks"][0], scheduler)
    runtime.submit_task(task)
    runtime.advance_to(task.release_time_ms)
    vehicle = next(
        item
        for item in runtime.simulator.vehicles.values()
        if item.robot_group == task.required_robot_group
    )
    task.assigned_vehicle_id = vehicle.vehicle_id
    task.state = TaskState.EN_ROUTE_DROPOFF
    vehicle.active_task_id = task.task_id
    vehicle.state = VehicleState.TO_DROPOFF
    vehicle.load_state = LoadState.LOADED
    vehicle.payload_id = task.payload_id or task.task_id
    vehicle.available_at_ms = runtime.now_ms

    proposal = runtime.plan_cycle()[0]

    assert proposal.plan.continuation
    assert proposal.plan.vehicle_id == vehicle.vehicle_id
    assert proposal.plan.task_id == task.task_id
    assert all(
        segment.kind is not SegmentKind.PICKUP
        for segment in proposal.candidate_plan.segments
    )
    runtime.acknowledge_plan(
        proposal.proposal_id,
        proposal.plan.revision,
        accepted=True,
    )
    runtime.advance_to(proposal.plan.committed_until_ms)
    assert task.state in {TaskState.EN_ROUTE_DROPOFF, TaskState.COMPLETED}


def test_online_runtime_accepts_local_rl_policy_with_safe_validation(
    online_documents,
) -> None:
    class ReverseLocalPolicy:
        candidate_count = 1

        def priority_orders(self, **kwargs):
            return (tuple(reversed(kwargs["proposals"])),)

    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    runtime = run_online_scenario(
        scenario,
        *online_documents,
        policy="rl",
        seed=0,
        priority_policy=ReverseLocalPolicy(),
        rl_candidate_count=1,
        rl_allow_deviation=True,
    )
    result = runtime.result()
    planning = runtime.planning_result().summary()

    assert result["metrics"]["completedTaskCount"] == len(scenario["tasks"])
    assert result["metrics"]["reservationConflictRejections"] == 0
    assert planning["rlInferenceCount"] > 0
    assert planning["rlFallbackCount"] == 0
