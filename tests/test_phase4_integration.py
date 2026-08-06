from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from masp.domain import DomainError, LoadState, TransportTask, Vehicle
from masp.motion import EdgeTravelTimeModel
from masp.phase3 import PriorityStrategy, RollingHorizonPlanner
from masp.phase4 import run_phase4_scenario, validate_phase4_scenario_document
from masp.recovery import RecoveryController, RecoveryPlanningError, RecoveryVehicle
from masp.reservations import ReservationTable
from masp.routing import RouteProvider
from masp.topology import MapTopology

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


def run_scenario(scenario, phase0_assets):
    return run_phase4_scenario(
        scenario,
        phase0_assets["model"],
        phase0_assets["conflicts"],
        phase0_assets["workstations"],
        phase0_assets["profiles"],
        phase0_assets["scheduler"],
        phase0_assets["traffic_zones"],
        ROOT / "schemas",
    )


def test_phase4_real_map_scenario_is_deterministic_and_accepted(
    phase0_assets,
) -> None:
    scenario = read_json("scenarios/phase4-deadlock-recovery.json")
    validate_phase4_scenario_document(scenario, ROOT / "schemas")

    first = run_scenario(scenario, phase0_assets)
    second = run_scenario(scenario, phase0_assets)

    assert first.accepted is True
    assert first.to_dict() == second.to_dict()
    assert first.zone_admission["firstEntryMs"] == 5_000
    assert first.recoverable_deadlock.decision.plan is not None
    assert (
        first.recoverable_deadlock.decision.plan.segments[0].source_edge_id
        == "fork:edge-323"
    )
    assert round(
        first.recoverable_deadlock.decision.plan.total_distance_m, 6
    ) == 4.76388
    assert first.recoverable_deadlock.projected_after_decision_report is not None
    assert first.recoverable_deadlock.projected_after_decision_report.cycles == ()
    assert first.unrecoverable_deadlock.decision.action == "safety_stop"
    assert (
        first.unrecoverable_deadlock.decision.reason_code
        == "deadlock.recovery_unavailable"
    )
    assert first.unrecoverable_deadlock.decision.freeze_reservation_ids
    assert first.zone_admission["terminalHoldReservationCount"] > 0


def test_unrecoverable_ring_shortest_backtracks_all_exceed_distance_limit(
    phase0_assets,
) -> None:
    scenario = read_json("scenarios/phase4-deadlock-recovery.json")
    case = scenario["unrecoverableDeadlock"]
    topology = MapTopology(
        phase0_assets["model"],
        phase0_assets["conflicts"],
        phase0_assets["workstations"],
        phase0_assets["traffic_zones"],
    )
    travel_times = EdgeTravelTimeModel(
        phase0_assets["model"],
        phase0_assets["profiles"],
        int(phase0_assets["scheduler"]["planner"].get("timeQuantumMs", 100)),
    )
    routes = RouteProvider(phase0_assets["model"], travel_times)
    controller = RecoveryController(
        topology,
        travel_times,
        phase0_assets["scheduler"],
        phase0_assets["traffic_zones"],
    )
    max_distance_m = float(
        phase0_assets["scheduler"]["traffic"]["reverse"]["maxDistanceM"]
    )
    rejection_codes: dict[str, str] = {}
    distances_m: dict[str, float] = {}

    for item in case["recoveryVehicles"]:
        load_state = LoadState(item["loadState"])
        shortest = routes.candidate_routes(
            item["robotGroup"],
            item["recoveryNodeId"],
            item["currentNodeId"],
            load_state,
            limit=1,
        )[0]
        backtrack_edge_ids = tuple(item["backtrackEdgeIds"])
        assert backtrack_edge_ids == tuple(reversed(shortest.edge_ids))
        distance_m = sum(
            float(topology.edges[edge_id]["length"])
            for edge_id in backtrack_edge_ids
        )
        distances_m[item["vehicleId"]] = round(distance_m, 3)
        assert distance_m > max_distance_m

        vehicle = RecoveryVehicle(
            vehicle_id=item["vehicleId"],
            robot_group=item["robotGroup"],
            load_state=load_state,
            recovery_node_id=item["recoveryNodeId"],
            wait_since_ms=int(item["waitSinceMs"]),
            priority_class=int(item["priorityClass"]),
            current_node_id=item["currentNodeId"],
            backtrack_edge_ids=backtrack_edge_ids,
            held_resource_ids=tuple(item["heldResourceIds"]),
        )
        with pytest.raises(RecoveryPlanningError) as caught:
            controller.plan_for_vehicle(
                vehicle,
                ReservationTable(),
                now_ms=int(scenario["nowMs"]),
                end_ms=int(scenario["endTimeMs"]),
            )
        rejection_codes[vehicle.vehicle_id] = caught.value.code

    assert distances_m == {
        "ring-0": 17.753,
        "ring-1": 19.445,
        "ring-2": 21.459,
        "ring-3": 19.767,
    }
    assert rejection_codes == {
        "ring-0": "recovery.distance.exceeded",
        "ring-1": "recovery.distance.exceeded",
        "ring-2": "recovery.distance.exceeded",
        "ring-3": "recovery.distance.exceeded",
    }
    result = run_scenario(scenario, phase0_assets)
    assert result.unrecoverable_deadlock.decision.action == "safety_stop"
    assert result.unrecoverable_deadlock.decision.plan is None
    assert (
        result.unrecoverable_deadlock.decision.reason_code
        == "deadlock.recovery_unavailable"
    )


def test_starvation_age_promotes_vehicle_in_task_age_order(phase0_assets) -> None:
    scenario = read_json("scenarios/phase3-rh-pp-benchmark.json")
    topology = MapTopology(
        phase0_assets["model"],
        phase0_assets["conflicts"],
        phase0_assets["workstations"],
        phase0_assets["traffic_zones"],
    )
    planner = RollingHorizonPlanner(
        topology,
        phase0_assets["model"],
        phase0_assets["profiles"],
        phase0_assets["scheduler"],
        phase0_assets["traffic_zones"],
    )
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    defaults = phase0_assets["scheduler"]["serviceDefaults"]
    tasks = [
        TransportTask.from_dict(
            item,
            int(defaults["pickupServiceMs"]),
            int(defaults["dropoffServiceMs"]),
        )
        for item in scenario["tasks"]
        if item["releaseTimeMs"] == 0
    ]
    proposals = planner.allocator.assign(vehicles, tasks, 0)
    assert len(proposals) >= 2
    promoted_vehicle_id = proposals[-1].vehicle_id
    planner.set_priority_ages({promoted_vehicle_id: 10_000})

    order = planner._order_for_strategy(
        PriorityStrategy.TASK_AGE,
        proposals,
        {item.task_id: item for item in tasks},
        ReservationTable(),
        {item.vehicle_id: item for item in vehicles},
        0,
        0,
        0,
        0,
    )

    assert order[0].vehicle_id == promoted_vehicle_id


def test_phase4_rejects_unknown_topology_evidence(phase0_assets) -> None:
    scenario = deepcopy(read_json("scenarios/phase4-deadlock-recovery.json"))
    scenario["recoverableDeadlock"]["evidenceEdgeIds"][0] = "fork:missing-edge"

    with pytest.raises(DomainError) as caught:
        run_scenario(scenario, phase0_assets)

    assert caught.value.code == "phase4.deadlock.evidence_edge"


def test_phase4_rejects_existing_but_unrelated_topology_evidence(
    phase0_assets,
) -> None:
    scenario = deepcopy(read_json("scenarios/phase4-deadlock-recovery.json"))
    scenario["recoverableDeadlock"]["evidenceEdgeIds"] = [
        "fork:edge-68",
        "fork:edge-370",
    ]

    with pytest.raises(DomainError) as caught:
        run_scenario(scenario, phase0_assets)

    assert caught.value.code == "phase4.deadlock.evidence_resource"


@pytest.mark.parametrize(
    ("vehicle_index", "position_field", "position_value", "expected_code"),
    [
        (
            0,
            "currentEdgeId",
            "fork:edge-68",
            "phase4.recovery_vehicle.evidence_edge",
        ),
        (
            1,
            "currentNodeId",
            "fork:PP1175",
            "phase4.recovery_vehicle.evidence_node",
        ),
    ],
)
def test_phase4_recovery_positions_must_belong_to_evidence_edges(
    phase0_assets,
    vehicle_index: int,
    position_field: str,
    position_value: str,
    expected_code: str,
) -> None:
    scenario = deepcopy(read_json("scenarios/phase4-deadlock-recovery.json"))
    vehicle = scenario["recoverableDeadlock"]["recoveryVehicles"][vehicle_index]
    vehicle[position_field] = position_value

    with pytest.raises(DomainError) as caught:
        run_scenario(scenario, phase0_assets)

    assert caught.value.code == expected_code
