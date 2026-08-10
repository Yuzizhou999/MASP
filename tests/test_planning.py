from __future__ import annotations

import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from masp.assignment import TaskAllocator
from masp.domain import (
    LoadState,
    PlanSegment,
    SegmentKind,
    TransportTask,
    Vehicle,
    projected_vehicle_revision,
)
from masp.motion import EdgeTravelTimeModel
from masp.plans import PlanValidator
from masp.reservations import Reservation, ReservationTable
from masp.routing import RouteProvider
from masp.scenario import build_plans, build_simulator
from masp.sipp import ContinuousTimeSippPlanner, SippPlanningError
from masp.topology import MapTopology

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


def planning_documents():
    model = read_json("generated/xiate-unified-map-model.json")
    conflicts = read_json("generated/xiate-conflict-resources.json")
    workstations = read_json("generated/xiate-workstations.json")
    profiles = read_json("config/robot-profiles.json")
    scheduler = read_json("config/scheduler.json")
    traffic_zones = read_json("config/traffic-zones.json")
    return model, conflicts, workstations, profiles, scheduler, traffic_zones


def test_motion_time_is_positive_rounded_and_load_sensitive() -> None:
    model, _, _, profiles, scheduler, _ = planning_documents()
    quantum = scheduler["planner"]["timeQuantumMs"]
    travel = EdgeTravelTimeModel(model, profiles, quantum)
    edge = next(item for item in model["edges"] if item["id"] == "fork:edge-218")

    empty_ms = travel.duration_ms(edge, LoadState.EMPTY)
    loaded_ms = travel.duration_ms(edge, LoadState.LOADED)

    assert empty_ms > 0
    assert empty_ms % quantum == 0
    assert loaded_ms % quantum == 0
    assert loaded_ms >= empty_ms

    cache_size = len(travel._duration_cache)
    assert travel.duration_ms(edge, LoadState.EMPTY) == empty_ms
    assert len(travel._duration_cache) == cache_size

    reverse_probe = {
        **edge,
        "start": edge["end"],
        "end": edge["start"],
        "p0": edge["p3"],
        "p1": edge["p2"],
        "p2": edge["p1"],
        "p3": edge["p0"],
        "length": float(edge["length"]) / 2.0,
        "motionDirection": 1 - int(edge.get("motionDirection", 0)),
    }
    travel.duration_ms(reverse_probe, LoadState.EMPTY)
    assert len(travel._duration_cache) == cache_size + 1


def test_candidate_routes_follow_the_vehicle_directed_subgraph() -> None:
    model, _, _, profiles, scheduler, _ = planning_documents()
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


def test_candidate_routes_reuse_static_graph_and_route_cache() -> None:
    model, _, _, profiles, scheduler, _ = planning_documents()
    travel = EdgeTravelTimeModel(model, profiles, scheduler["planner"]["timeQuantumMs"])
    routes = RouteProvider(model, travel)

    first = routes.candidate_routes(
        "fork",
        "fork:PP1171",
        "fork:AP1123",
        LoadState.EMPTY,
        limit=3,
    )
    graph_count = len(routes._graphs)
    route_count = len(routes._route_cache)
    second = routes.candidate_routes(
        "fork",
        "fork:PP1171",
        "fork:AP1123",
        LoadState.EMPTY,
        limit=3,
    )

    assert second is first
    assert len(routes._graphs) == graph_count
    assert len(routes._route_cache) == route_count

    routes.candidate_routes(
        "fork",
        "fork:PP1171",
        "fork:AP1123",
        LoadState.EMPTY,
        limit=3,
        closed_edge_ids=frozenset({first[0].edge_ids[0]}),
    )
    assert len(routes._graphs) == graph_count + 1
    assert len(routes._route_cache) == route_count + 1


def test_occupied_wait_node_jumps_to_blocker_end() -> None:
    model, conflicts, workstations, profiles, scheduler, traffic_zones = (
        planning_documents()
    )
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    travel = EdgeTravelTimeModel(
        model, profiles, scheduler["planner"]["timeQuantumMs"]
    )
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel),
        travel,
        scheduler,
        (item["nodeId"] for item in traffic_zones["recoveryNodes"]),
    )
    reservations = ReservationTable()
    reservations.insert_batch(
        [
            Reservation(
                reservation_id="block-wait-node",
                resource_id="node:fork:PP1171",
                vehicle_id="other-vehicle",
                plan_id="other-plan",
                segment_id="hold",
                start_ms=1000,
                end_ms=5000,
                kind="wait",
                committed=True,
            )
        ]
    )

    with pytest.raises(SippPlanningError) as error:
        planner._append_wait(
            [],
            "fork-001",
            "fork",
            "fork:PP1171",
            0,
            2000,
            LoadState.EMPTY,
            reservations,
        )

    assert error.value.code == "sipp.wait.interval_occupied"
    assert error.value.suggested_delay_ms == 5000

    with pytest.raises(SippPlanningError) as error:
        planner._append_wait(
            [],
            "fork-001",
            "fork",
            "fork:PP1171",
            0,
            planner.max_wait_ms + 1000,
            LoadState.EMPTY,
            ReservationTable(),
        )

    assert error.value.code == "sipp.wait.too_long"
    assert error.value.suggested_delay_ms == 1000


def test_non_temporal_sipp_failures_are_not_retried_with_later_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, conflicts, workstations, profiles, scheduler, traffic_zones = (
        planning_documents()
    )
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    travel = EdgeTravelTimeModel(
        model, profiles, scheduler["planner"]["timeQuantumMs"]
    )
    routes = RouteProvider(model, travel)
    planner = ContinuousTimeSippPlanner(
        topology,
        routes,
        travel,
        scheduler,
        (item["nodeId"] for item in traffic_zones["recoveryNodes"]),
    )
    vehicle = Vehicle.from_dict(
        {
            "vehicleId": "fork-001",
            "robotGroup": "fork",
            "initialNodeId": "fork:PP1171",
            "initialHeadingRad": 0.0,
            "initialLoadState": "empty",
        }
    )
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
    empty_routes = routes.candidate_routes(
        "fork", vehicle.current_node_id or "", task.pickup_node_id, LoadState.EMPTY, 3
    )
    loaded_routes = routes.candidate_routes(
        "fork", task.pickup_node_id, task.dropoff_node_id, LoadState.LOADED, 3
    )
    recovery_routes = planner._recovery_routes("fork", task.dropoff_node_id)
    expected_combinations = len(empty_routes) * len(loaded_routes) * len(recovery_routes)
    attempts = 0

    def fail_permanently(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise SippPlanningError(
            "sipp.horizon.exceeded", "route exceeds planning horizon"
        )

    monkeypatch.setattr(planner, "_schedule_combination", fail_permanently)

    with pytest.raises(SippPlanningError) as error:
        planner.plan_task(vehicle, task, 0, 400000, ReservationTable(), 0)

    assert error.value.code == "sipp.no_schedule"
    assert attempts == expected_combinations
    assert attempts < expected_combinations * planner.max_schedule_attempts


def test_progressive_route_search_stops_after_fastest_feasible_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, conflicts, workstations, profiles, scheduler, traffic_zones = (
        planning_documents()
    )
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    travel = EdgeTravelTimeModel(
        model, profiles, scheduler["planner"]["timeQuantumMs"]
    )
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel),
        travel,
        scheduler,
        (item["nodeId"] for item in traffic_zones["recoveryNodes"]),
    )
    vehicle = Vehicle.from_dict(
        {
            "vehicleId": "fork-001",
            "robotGroup": "fork",
            "initialNodeId": "fork:PP1171",
            "initialHeadingRad": 0.0,
            "initialLoadState": "empty",
        }
    )
    task = TransportTask(
        task_id="progressive-task",
        release_time_ms=0,
        pickup_node_id="fork:AP1123",
        dropoff_node_id="fork:AP2121",
        required_robot_group="fork",
        payload_type="pallet",
        payload_id=None,
        pickup_service_ms=5000,
        dropoff_service_ms=5000,
    )
    attempts = 0

    def succeed(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return (
            PlanSegment(
                segment_id="synthetic",
                kind=SegmentKind.TRAVERSE,
                start_ms=0,
                end_ms=100,
                start_node_id=vehicle.current_node_id,
                end_node_id=task.dropoff_node_id,
                edge_id=None,
                expected_load_state=LoadState.EMPTY,
            ),
        )

    monkeypatch.setattr(planner, "_schedule_combination", succeed)
    planned = planner.plan_task(
        vehicle, task, 0, 400000, ReservationTable(), 0
    )

    assert attempts == 1
    assert planned.diagnostics.route_combinations_tried == 1
    assert planned.diagnostics.route_expansion_level == 1


def test_sipp_rejects_work_after_computation_deadline() -> None:
    model, conflicts, workstations, profiles, scheduler, traffic_zones = (
        planning_documents()
    )
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    travel = EdgeTravelTimeModel(
        model, profiles, scheduler["planner"]["timeQuantumMs"]
    )
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel),
        travel,
        scheduler,
        (item["nodeId"] for item in traffic_zones["recoveryNodes"]),
    )
    scenario = read_json("scenarios/continuous-task-planning.json")
    vehicle = Vehicle.from_dict(scenario["vehicles"][0])
    defaults = scheduler["serviceDefaults"]
    task = TransportTask.from_dict(
        scenario["tasks"][0],
        int(defaults["pickupServiceMs"]),
        int(defaults["dropoffServiceMs"]),
    )

    with pytest.raises(SippPlanningError) as error:
        planner.plan_task(
            vehicle,
            task,
            0,
            int(scenario["endTimeMs"]),
            ReservationTable(),
            0,
            deadline_ns=time.perf_counter_ns() - 1,
        )

    assert error.value.code == "sipp.deadline.exceeded"


def test_assignment_filters_robot_group_and_chooses_minimum_cost_vehicle() -> None:
    model, conflicts, workstations, profiles, scheduler, _ = planning_documents()
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
    scenario = read_json("scenarios/continuous-task-planning.json")
    documents = planning_documents()

    first, first_planned = build_plans(
        deepcopy(scenario), *documents, ROOT / "schemas"
    )
    second, second_planned = build_plans(
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


def test_planning_schema_is_valid_and_example_matches() -> None:
    from jsonschema import Draft202012Validator

    from masp.scenario import validate_planning_scenario_document

    schema = read_json("schemas/planning-scenario.schema.json")
    Draft202012Validator.check_schema(schema)
    validate_planning_scenario_document(
        read_json("scenarios/continuous-task-planning.json"), ROOT / "schemas"
    )
