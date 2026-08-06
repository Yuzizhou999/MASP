from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from masp.assignment import TaskAllocator
from masp.domain import (
    LoadState,
    SegmentKind,
    TransportTask,
    Vehicle,
    projected_vehicle_revision,
)
from masp.motion import EdgeTravelTimeModel
from masp.plans import PlanValidator
from masp.routing import RouteProvider
from masp.scenario import build_phase2_plans, build_simulator
from masp.topology import MapTopology

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


def phase2_documents():
    model = read_json("generated/xiate-unified-map-model.json")
    conflicts = read_json("generated/xiate-conflict-resources.json")
    workstations = read_json("generated/xiate-workstations.json")
    profiles = read_json("config/robot-profiles.json")
    scheduler = read_json("config/scheduler.json")
    traffic_zones = read_json("config/traffic-zones.json")
    return model, conflicts, workstations, profiles, scheduler, traffic_zones


def test_motion_time_is_positive_rounded_and_load_sensitive() -> None:
    model, _, _, profiles, scheduler, _ = phase2_documents()
    quantum = scheduler["planner"]["timeQuantumMs"]
    travel = EdgeTravelTimeModel(model, profiles, quantum)
    edge = next(item for item in model["edges"] if item["id"] == "fork:edge-218")

    empty_ms = travel.duration_ms(edge, LoadState.EMPTY)
    loaded_ms = travel.duration_ms(edge, LoadState.LOADED)

    assert empty_ms > 0
    assert empty_ms % quantum == 0
    assert loaded_ms % quantum == 0
    assert loaded_ms >= empty_ms


def test_candidate_routes_follow_the_vehicle_directed_subgraph() -> None:
    model, _, _, profiles, scheduler, _ = phase2_documents()
    travel = EdgeTravelTimeModel(model, profiles, scheduler["planner"]["timeQuantumMs"])
    routes = RouteProvider(model, travel)
    edges = {item["id"]: item for item in model["edges"]}

    candidates = routes.candidate_routes(
        "fork",
        "fork:PP1171",
        "fork:AP1123",
        LoadState.EMPTY,
        limit=3,
    )

    assert 1 <= len(candidates) <= 3
    for route in candidates:
        current = route.start_node_id
        for edge_id in route.edge_ids:
            edge = edges[edge_id]
            assert edge["robotGroup"] == "fork"
            assert edge["start"] == current
            current = edge["end"]
        assert current == route.end_node_id


def test_assignment_filters_robot_group_and_chooses_minimum_cost_vehicle() -> None:
    model, conflicts, workstations, profiles, scheduler, _ = phase2_documents()
    topology = MapTopology(model, conflicts, workstations)
    travel = EdgeTravelTimeModel(model, profiles, scheduler["planner"]["timeQuantumMs"])
    routes = RouteProvider(model, travel)
    allocator = TaskAllocator(topology, routes, scheduler["assignment"])
    vehicles = [
        Vehicle.from_dict(
            {
                "vehicleId": "fork-near",
                "robotGroup": "fork",
                "initialNodeId": "fork:PP1171",
                "initialHeadingRad": 0.0,
                "initialLoadState": "empty",
            }
        ),
        Vehicle.from_dict(
            {
                "vehicleId": "fork-far",
                "robotGroup": "fork",
                "initialNodeId": "fork:PP1175",
                "initialHeadingRad": 0.0,
                "initialLoadState": "empty",
            }
        ),
        Vehicle.from_dict(
            {
                "vehicleId": "jack-incompatible",
                "robotGroup": "jack",
                "initialNodeId": "jack:PP363",
                "initialHeadingRad": 0.0,
                "initialLoadState": "empty",
            }
        ),
    ]
    task = TransportTask(
        task_id="fork-task",
        release_time_ms=0,
        pickup_node_id="fork:AP1123",
        dropoff_node_id="fork:AP2121",
        required_robot_group="fork",
        payload_type="pallet",
        payload_id=None,
        pickup_service_ms=5000,
        dropoff_service_ms=5000,
    )
    costs = {
        vehicle.vehicle_id: allocator.compatible_cost(vehicle, task, 0)
        for vehicle in vehicles
    }

    proposals = allocator.assign(vehicles, [task], 0)

    compatible = {key: value for key, value in costs.items() if value is not None}
    expected = min(compatible, key=lambda key: (compatible[key].total_ms, key))
    assert len(proposals) == 1
    assert proposals[0].vehicle_id == expected
    assert proposals[0].vehicle_id != "jack-incompatible"

    alternatives = allocator.assign(
        vehicles,
        [task],
        0,
        frozenset({(proposals[0].vehicle_id, task.task_id)}),
    )
    assert len(alternatives) == 1
    assert alternatives[0].vehicle_id in compatible
    assert alternatives[0].vehicle_id != proposals[0].vehicle_id


def test_continuous_task_scenario_is_deterministic_and_policy_compliant() -> None:
    scenario = read_json("scenarios/phase2-continuous-tasks.json")
    documents = phase2_documents()

    first, first_planned = build_phase2_plans(
        deepcopy(scenario), *documents, ROOT / "schemas"
    )
    second, second_planned = build_phase2_plans(
        deepcopy(scenario), *documents, ROOT / "schemas"
    )

    assert first_planned == second_planned
    assert first.summary() == second.summary()
    assert first.unplanned_task_ids == ()
    assert len(first.plans) == 3
    assert Counter(plan.vehicle_id for plan in first.plans).most_common(1)[0][1] == 2

    model, conflicts, workstations, _, scheduler, traffic_zones = documents
    topology = MapTopology(model, conflicts, workstations)
    recovery_ids = {item["nodeId"] for item in traffic_zones["recoveryNodes"]}
    vehicles_by_id = {
        item["vehicleId"]: Vehicle.from_dict(item) for item in scenario["vehicles"]
    }
    plans_by_vehicle = defaultdict(list)
    for plan in first.plans:
        plans_by_vehicle[plan.vehicle_id].append(plan)
        assert plan.segments[-1].end_node_id in recovery_ids
        for segment in plan.segments:
            if segment.kind is SegmentKind.WAIT:
                assert topology.wait_allowed(
                    segment.start_node_id,
                    vehicles_by_id[plan.vehicle_id].robot_group,
                )

    for vehicle_id, plans in plans_by_vehicle.items():
        projected = vehicles_by_id[vehicle_id]
        for plan in sorted(plans, key=lambda item: item.created_at_ms):
            task_data = next(item for item in scenario["tasks"] if item["taskId"] == plan.task_id)
            task = TransportTask.from_dict(task_data, 5000, 5000)
            validated = PlanValidator(topology).validate(plan, projected, task)
            projected.current_node_id = validated.final_node_id
            projected.revision = projected_vehicle_revision(plan)

    simulation = build_simulator(
        first_planned,
        model,
        conflicts,
        workstations,
        scheduler,
        ROOT / "schemas",
    ).run()
    assert simulation["metrics"]["completedTaskCount"] == 3
    assert {task["state"] for task in simulation["tasks"]} == {"COMPLETED"}
    assert {vehicle["state"] for vehicle in simulation["vehicles"]} == {"IDLE"}


def test_phase2_schema_is_valid_and_example_matches() -> None:
    from jsonschema import Draft202012Validator

    from masp.scenario import validate_phase2_scenario_document

    schema = read_json("schemas/phase2-scenario.schema.json")
    Draft202012Validator.check_schema(schema)
    validate_phase2_scenario_document(
        read_json("scenarios/phase2-continuous-tasks.json"), ROOT / "schemas"
    )
