from __future__ import annotations

from pathlib import Path

import pytest

from masp.domain import DomainError, LoadState, TaskState, TransportTask, Vehicle
from masp.online import OnlineDispatchRuntime, run_online_scenario
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
    scenario = read_json("scenarios/phase3-realistic-multi-fleet-interactive.json")
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


def test_online_interactive_scenario_completes_without_conflicts(
    online_documents,
) -> None:
    scenario = read_json("scenarios/phase3-realistic-multi-fleet-interactive.json")
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
    assert result["online"] == {
        "taskSubmissionCount": len(scenario["tasks"]),
        "planProposalCount": len(scenario["tasks"]),
        "acknowledgedPlanCount": len(scenario["tasks"]),
        "rejectedPlanCount": 0,
        "pendingPlanAckCount": 0,
        "telemetryUpdateCount": 0,
    }
    assert planning["unplannedTaskCount"] == 0
    assert all(
        plan.created_at_ms >= runtime.simulator.tasks[plan.task_id].release_time_ms
        for plan in runtime.accepted_plans
    )
