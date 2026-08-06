from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from masp.reservations import ReservationConflict
from masp.scenario import build_simulator

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


def build_phase1(scenario: dict[str, Any]):
    return build_simulator(
        scenario,
        read_json("generated/xiate-unified-map-model.json"),
        read_json("generated/xiate-conflict-resources.json"),
        read_json("generated/xiate-workstations.json"),
        read_json("config/scheduler.json"),
        ROOT / "schemas",
    )


def test_single_vehicle_completes_pickup_and_dropoff_deterministically() -> None:
    scenario = read_json("scenarios/phase1-single-vehicle.json")

    first = build_phase1(deepcopy(scenario)).run()
    second = build_phase1(deepcopy(scenario)).run()

    assert first["eventDigestSha256"] == second["eventDigestSha256"]
    assert first["eventLog"] == second["eventLog"]
    assert first["metrics"]["completedTaskCount"] == 1
    assert first["metrics"]["reservationConflictRejections"] == 0
    assert first["tasks"][0]["state"] == "COMPLETED"
    assert first["vehicles"][0]["state"] == "IDLE"
    assert first["vehicles"][0]["loadState"] == "empty"
    assert first["vehicles"][0]["currentNodeId"] == "fork:PP1172"


def test_completion_events_precede_next_segment_entry_at_same_time() -> None:
    result = build_phase1(
        read_json("scenarios/phase1-single-vehicle.json")
    ).run()
    at_pickup_completion = [
        row["type"] for row in result["eventLog"] if row["timeMs"] == 16500
    ]

    assert at_pickup_completion == ["PICKUP_COMPLETED", "VEHICLE_ENTER_EDGE"]


def synthetic_documents() -> tuple[dict[str, Any], ...]:
    def node(node_id: str, node_type: str) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": node_type,
            "allowedRobotGroups": ["fork"],
            "headings": {"fork": 0.0},
            "waitPolicyByGroup": {
                "fork": {"allowed": node_type == "PP", "maxWaitMs": 1000}
            },
        }

    model = {
        "nodes": [
            node("fork:A", "PP"),
            node("fork:P1", "AP"),
            node("fork:D1", "AP"),
            node("fork:R1", "PP"),
            node("fork:C", "PP"),
            node("fork:P2", "AP"),
            node("fork:D2", "AP"),
            node("fork:R2", "PP"),
        ],
        "edges": [
            {"id": "fork:e1", "robotGroup": "fork", "start": "fork:A", "end": "fork:P1"},
            {"id": "fork:e2", "robotGroup": "fork", "start": "fork:P1", "end": "fork:D1"},
            {"id": "fork:e3", "robotGroup": "fork", "start": "fork:C", "end": "fork:P2"},
            {"id": "fork:e4", "robotGroup": "fork", "start": "fork:P2", "end": "fork:D2"},
            {"id": "fork:e5", "robotGroup": "fork", "start": "fork:D1", "end": "fork:R1"},
            {"id": "fork:e6", "robotGroup": "fork", "start": "fork:D2", "end": "fork:R2"},
        ],
    }
    conflicts = {
        "edgeResources": [
            {"edgeId": "fork:e1", "ownResource": "edge:fork:e1", "conflictResources": ["cross"]},
            {"edgeId": "fork:e2", "ownResource": "edge:fork:e2", "conflictResources": []},
            {"edgeId": "fork:e3", "ownResource": "edge:fork:e3", "conflictResources": ["cross"]},
            {"edgeId": "fork:e4", "ownResource": "edge:fork:e4", "conflictResources": []},
            {"edgeId": "fork:e5", "ownResource": "edge:fork:e5", "conflictResources": []},
            {"edgeId": "fork:e6", "ownResource": "edge:fork:e6", "conflictResources": []},
        ]
    }
    workstations = {
        "workstations": [
            {
                "id": f"station:{node_id}",
                "nodeId": node_id,
                "allowedRobotGroups": ["fork"],
                "pickupServiceMs": 10,
                "dropoffServiceMs": 10,
                "blocksTransitDuringService": True,
            }
            for node_id in ("fork:P1", "fork:D1", "fork:P2", "fork:D2")
        ]
    }
    scheduler = {"serviceDefaults": {"pickupServiceMs": 10, "dropoffServiceMs": 10}}
    return model, conflicts, workstations, scheduler


def synthetic_scenario(second_start_ms: int) -> dict[str, Any]:
    def task(number: int) -> dict[str, Any]:
        return {
            "taskId": f"task-{number}",
            "releaseTimeMs": 0,
            "pickupNodeId": f"fork:P{number}",
            "dropoffNodeId": f"fork:D{number}",
            "requiredRobotGroup": "fork",
            "payloadType": "test",
        }

    def plan(number: int, start_ms: int) -> dict[str, Any]:
        pickup_start = start_ms + 10
        second_edge_start = pickup_start + 10
        dropoff_start = second_edge_start + 10
        return {
            "id": f"plan-{number}",
            "revision": 0,
            "vehicleId": f"fork-{number}",
            "taskId": f"task-{number}",
            "basedOnVehicleRevision": 0,
            "basedOnWorldRevision": 0,
            "createdAtMs": 1,
            "horizonEndMs": 100,
            "committedUntilMs": dropoff_start + 20,
            "segments": [
                {
                    "id": f"v{number}-edge-1",
                    "kind": "traverse",
                    "startMs": start_ms,
                    "endMs": pickup_start,
                    "startNodeId": "fork:A" if number == 1 else "fork:C",
                    "endNodeId": f"fork:P{number}",
                    "edgeId": "fork:e1" if number == 1 else "fork:e3",
                    "expectedLoadState": "empty",
                },
                {
                    "id": f"v{number}-pickup",
                    "kind": "pickup",
                    "startMs": pickup_start,
                    "endMs": second_edge_start,
                    "startNodeId": f"fork:P{number}",
                    "endNodeId": f"fork:P{number}",
                    "expectedLoadState": "empty",
                },
                {
                    "id": f"v{number}-edge-2",
                    "kind": "traverse",
                    "startMs": second_edge_start,
                    "endMs": dropoff_start,
                    "startNodeId": f"fork:P{number}",
                    "endNodeId": f"fork:D{number}",
                    "edgeId": "fork:e2" if number == 1 else "fork:e4",
                    "expectedLoadState": "loaded",
                },
                {
                    "id": f"v{number}-dropoff",
                    "kind": "dropoff",
                    "startMs": dropoff_start,
                    "endMs": dropoff_start + 10,
                    "startNodeId": f"fork:D{number}",
                    "endNodeId": f"fork:D{number}",
                    "expectedLoadState": "loaded",
                },
                {
                    "id": f"v{number}-reposition",
                    "kind": "traverse",
                    "startMs": dropoff_start + 10,
                    "endMs": dropoff_start + 20,
                    "startNodeId": f"fork:D{number}",
                    "endNodeId": f"fork:R{number}",
                    "edgeId": "fork:e5" if number == 1 else "fork:e6",
                    "expectedLoadState": "empty",
                },
            ],
        }

    return {
        "schemaVersion": 1,
        "scenarioId": "two-vehicle-crossing",
        "seed": 0,
        "endTimeMs": 100,
        "vehicles": [
            {
                "vehicleId": "fork-1",
                "robotGroup": "fork",
                "initialNodeId": "fork:A",
                "initialHeadingRad": 0.0,
                "initialLoadState": "empty",
            },
            {
                "vehicleId": "fork-2",
                "robotGroup": "fork",
                "initialNodeId": "fork:C",
                "initialHeadingRad": 0.0,
                "initialLoadState": "empty",
            },
        ],
        "tasks": [task(1), task(2)],
        "plans": [plan(1, 10), plan(2, second_start_ms)],
    }


def test_overlapping_multi_vehicle_resource_is_rejected_atomically() -> None:
    model, conflicts, workstations, scheduler = synthetic_documents()

    with pytest.raises(ReservationConflict):
        build_simulator(
            synthetic_scenario(second_start_ms=15),
            model,
            conflicts,
            workstations,
            scheduler,
            ROOT / "schemas",
        )


def test_multi_vehicle_boundary_handoff_has_no_resource_conflict() -> None:
    model, conflicts, workstations, scheduler = synthetic_documents()
    simulator = build_simulator(
        synthetic_scenario(second_start_ms=20),
        model,
        conflicts,
        workstations,
        scheduler,
        ROOT / "schemas",
    )

    result = simulator.run()

    assert result["metrics"]["completedTaskCount"] == 2
    assert result["metrics"]["reservationConflictRejections"] == 0
