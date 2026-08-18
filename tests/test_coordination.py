from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from masp.domain import (
    LoadState,
    PlanSegment,
    SegmentKind,
    TaskState,
    TransportTask,
    Vehicle,
    VehiclePlan,
    VehicleState,
)
from masp.motion import EdgeTravelTimeModel
from masp.coordination import CandidateScore, RollingHorizonPlanner
from masp.reservations import ReservationTable
from masp.routing import RouteProvider
from masp.scenario import (
    build_dispatch_plans,
    build_simulator,
    validate_dispatch_scenario_document,
)
from masp.topology import MapTopology
from masp.sipp import SippPlanningError

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dispatch_documents():
    return (
        read_json("generated/xiate-unified-map-model.json"),
        read_json("generated/xiate-conflict-resources.json"),
        read_json("generated/xiate-workstations.json"),
        read_json("config/robot-profiles.json"),
        read_json("config/scheduler.json"),
        read_json("config/traffic-zones.json"),
    )


@pytest.fixture(scope="module")
def dispatch_top_k_run(dispatch_documents):
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    planning, planned = build_dispatch_plans(
        scenario, *dispatch_documents, ROOT / "schemas"
    )
    return scenario, planning, planned


def test_candidate_score_uses_lexicographic_throughput_priority() -> None:
    more_dropoffs = CandidateScore(2, 2, 100_000, 100_000, 100_000, 100_000, 500_000)
    less_dropoffs = CandidateScore(1, 3, 0, 0, 0, 0, 1)

    assert more_dropoffs.ordering_key() < less_dropoffs.ordering_key()


def test_rh_pp_commitment_only_ends_after_a_stable_motion_boundary() -> None:
    plan = VehiclePlan(
        plan_id="rotation-boundary-plan",
        revision=1,
        vehicle_id="fork-001",
        task_id="task-001",
        based_on_vehicle_revision=0,
        based_on_world_revision=0,
        created_at_ms=0,
        horizon_end_ms=3000,
        committed_until_ms=3000,
        segments=(
            PlanSegment(
                segment_id="rotate-start",
                kind=SegmentKind.ROTATE,
                start_ms=0,
                end_ms=1000,
                start_node_id="fork:A",
                end_node_id="fork:A",
                edge_id=None,
                expected_load_state=LoadState.EMPTY,
                command_payload={"phase": "start"},
            ),
            PlanSegment(
                segment_id="traverse",
                kind=SegmentKind.TRAVERSE,
                start_ms=1000,
                end_ms=2000,
                start_node_id="fork:A",
                end_node_id="fork:B",
                edge_id="fork:edge",
                expected_load_state=LoadState.EMPTY,
            ),
            PlanSegment(
                segment_id="rotate-end",
                kind=SegmentKind.ROTATE,
                start_ms=2000,
                end_ms=3000,
                start_node_id="fork:B",
                end_node_id="fork:B",
                edge_id=None,
                expected_load_state=LoadState.EMPTY,
                command_payload={"phase": "end"},
            ),
        ),
    )

    assert not RollingHorizonPlanner._segment_has_stable_end(plan, 0)
    assert not RollingHorizonPlanner._segment_has_stable_end(plan, 1)
    assert RollingHorizonPlanner._segment_has_stable_end(plan, 2)


def test_random_priority_orders_are_reproducible(dispatch_documents) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = dispatch_documents
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    topology = MapTopology(model, conflicts, workstations)
    planner = RollingHorizonPlanner(
        topology, model, profiles, scheduler, zones, policy="random", seed=17
    )
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    defaults = scheduler["serviceDefaults"]
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

    assert planner._random_order(proposals, 0, 0, 0) == planner._random_order(
        proposals, 0, 0, 0
    )


def test_priority_candidates_only_permute_local_conflict_components(
    dispatch_documents, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = dispatch_documents
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    planner = RollingHorizonPlanner(
        MapTopology(model, conflicts, workstations, zones),
        model,
        profiles,
        scheduler,
        zones,
        policy="top_k",
        seed=0,
    )
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    defaults = scheduler["serviceDefaults"]
    tasks = [
        TransportTask.from_dict(
            item,
            int(defaults["pickupServiceMs"]),
            int(defaults["dropoffServiceMs"]),
        )
        for item in scenario["tasks"]
        if item["releaseTimeMs"] == 0
    ]
    projections, tasks_by_id = planner._validate_inputs(vehicles, tasks)
    proposals = planner.allocator.assign(vehicles, tasks, 0)
    assert len(proposals) >= 3
    coupled_ids = {proposals[0].vehicle_id, proposals[1].vehicle_id}

    def fake_resources(proposal, *_args):
        if proposal.vehicle_id in coupled_ids:
            return frozenset({"shared"})
        return frozenset({f"private:{proposal.vehicle_id}"})

    monkeypatch.setattr(planner, "_proposal_resource_ids", fake_resources)
    components = planner._conflict_components(
        proposals, tasks_by_id, projections
    )
    orders = planner._priority_orders(
        proposals,
        tasks_by_id,
        ReservationTable(),
        projections,
        0,
        0,
        0,
    )

    assert tuple(len(component) for component in components) == (
        2,
        *(1 for _ in proposals[2:]),
    )
    assert len(orders) <= 2
    for _, order in orders:
        assert tuple(item.vehicle_id for item in order[2:]) == tuple(
            item.vehicle_id for item in proposals[2:]
        )


def test_active_task_component_precedes_new_task_component(
    dispatch_documents,
) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = dispatch_documents
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    planner = RollingHorizonPlanner(
        MapTopology(model, conflicts, workstations, zones),
        model,
        profiles,
        scheduler,
        zones,
        policy="congestion",
        seed=0,
    )
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    defaults = scheduler["serviceDefaults"]
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
    projections, tasks_by_id = planner._validate_inputs(vehicles, tasks)
    queued = proposals[0]
    active = proposals[-1]
    active_task = tasks_by_id[active.task_id]
    active_vehicle = projections[active.vehicle_id]
    active_task.state = TaskState.EN_ROUTE_PICKUP
    active_task.assigned_vehicle_id = active.vehicle_id
    active_vehicle.state = VehicleState.TO_PICKUP
    active_vehicle.active_task_id = active.task_id

    order = planner._localized_order(
        planner.priority_strategies[0],
        ((queued,), (active,)),
        tasks_by_id,
        ReservationTable(),
        projections,
        0,
        0,
        0,
        0,
    )

    assert order == (active, queued)


def test_idle_cycles_jump_to_the_next_release_time(dispatch_documents) -> None:
    scenario = deepcopy(read_json("scenarios/rolling-dispatch-benchmark.json"))
    scenario["tasks"] = scenario["tasks"][:1]
    scenario["tasks"][0]["releaseTimeMs"] = 1234
    scenario["endTimeMs"] = 500000

    planning, _ = build_dispatch_plans(
        scenario, *dispatch_documents, ROOT / "schemas", policy="congestion", seed=0
    )

    assert planning.unplanned_task_ids == ()
    assert planning.cycles[0].decision_time_ms == 0
    assert planning.cycles[1].decision_time_ms == 1234


def test_failed_pairs_do_not_spin_without_a_future_state_change(
    dispatch_documents, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = dispatch_documents
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    topology = MapTopology(model, conflicts, workstations, zones)
    planner = RollingHorizonPlanner(
        topology, model, profiles, scheduler, zones, policy="congestion", seed=0
    )
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    defaults = scheduler["serviceDefaults"]
    task = TransportTask.from_dict(
        scenario["tasks"][0],
        int(defaults["pickupServiceMs"]),
        int(defaults["dropoffServiceMs"]),
    )
    attempts = 0

    def fail_plan(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise SippPlanningError("sipp.no_schedule", "forced failure")

    monkeypatch.setattr(planner.sipp, "plan_task", fail_plan)

    planning = planner.plan(vehicles, [task], 500000)

    assert planning.unplanned_task_ids == (task.task_id,)
    assert len(planning.cycles) == 1
    assert attempts == len(vehicles)


def test_failed_continuation_is_attempted_once_per_online_cycle(
    dispatch_documents, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = dispatch_documents
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    planner = RollingHorizonPlanner(
        MapTopology(model, conflicts, workstations, zones),
        model,
        profiles,
        scheduler,
        zones,
        policy="congestion",
        seed=0,
    )
    vehicle = Vehicle.from_dict(scenario["vehicles"][0])
    defaults = scheduler["serviceDefaults"]
    task = TransportTask.from_dict(
        scenario["tasks"][0],
        int(defaults["pickupServiceMs"]),
        int(defaults["dropoffServiceMs"]),
    )
    vehicle.active_task_id = task.task_id
    vehicle.state = VehicleState.TO_PICKUP
    task.assigned_vehicle_id = vehicle.vehicle_id
    task.state = TaskState.EN_ROUTE_PICKUP
    reservations = ReservationTable()
    reservations.insert_batch(
        (
            planner._hold(
                vehicle,
                plan_id="active-hold",
                node_id=vehicle.current_node_id or "",
                start_ms=0,
                end_ms=int(scenario["endTimeMs"]),
                label="idle-tail",
            ),
        )
    )
    attempts = 0

    def fail_plan(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise SippPlanningError("sipp.no_schedule", "forced continuation failure")

    monkeypatch.setattr(planner.sipp, "plan_remaining_task", fail_plan)

    proposal = planner.plan_cycle(
        [vehicle],
        [task],
        0,
        int(scenario["endTimeMs"]),
        reservations,
    )

    assert proposal.plans == ()
    assert attempts == 1
    assert len(proposal.cycle.candidates) == 1
    assert not proposal.cycle.deadline_exhausted

    attempts = 0
    cooled = planner.plan_cycle(
        [vehicle],
        [task],
        0,
        int(scenario["endTimeMs"]),
        reservations,
        excluded_pairs=frozenset({(vehicle.vehicle_id, task.task_id)}),
    )
    assert cooled.plans == ()
    assert attempts == 0


def test_top_k_rh_pp_is_safe_and_completes_stream(
    dispatch_documents, dispatch_top_k_run
) -> None:
    scenario, planning, planned = dispatch_top_k_run
    model, conflicts, workstations, _, scheduler, _ = dispatch_documents
    summary = planning.summary()
    first_decision = next(item for item in planning.cycles if item.candidates)

    assert planning.unplanned_task_ids == ()
    assert len(planning.plans) == len(scenario["tasks"])
    assert len(first_decision.candidates) == scheduler["coordination"][
        "priorityCandidateCount"
    ]
    assert sum(item.feasible for item in first_decision.candidates) >= 1
    assert all(
        item.safe_until_ms >= item.nominal_until_ms
        for cycle in planning.cycles
        for item in cycle.commitments
    )
    assert summary["planningLatencyMs"]["p95"] < scheduler["planner"][
        "planningPeriodMs"
    ]

    topology = MapTopology(model, conflicts, workstations)
    vehicles = {item["vehicleId"]: item for item in scenario["vehicles"]}
    for plan in planning.plans:
        for segment in plan.segments:
            if segment.kind is SegmentKind.WAIT:
                assert topology.wait_allowed(
                    segment.start_node_id, vehicles[plan.vehicle_id]["robotGroup"]
                )

    simulation = build_simulator(
        planned,
        model,
        conflicts,
        workstations,
        scheduler,
        ROOT / "schemas",
    ).run()
    assert simulation["metrics"]["completedTaskCount"] == len(scenario["tasks"])
    assert simulation["metrics"]["reservationConflictRejections"] == 0


def test_congestion_baseline_throughput_is_not_worse_than_random(
    dispatch_documents,
) -> None:
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    model, conflicts, workstations, _, scheduler, _ = dispatch_documents
    throughput = {}
    for policy in ("congestion", "random"):
        planning, planned = build_dispatch_plans(
            scenario,
            *dispatch_documents,
            ROOT / "schemas",
            policy=policy,
            seed=0,
        )
        assert planning.unplanned_task_ids == ()
        simulation = build_simulator(
            planned,
            model,
            conflicts,
            workstations,
            scheduler,
            ROOT / "schemas",
        ).run()
        throughput[policy] = simulation["metrics"]["completedDropoffsPerHour"]

    assert throughput["congestion"] >= throughput["random"]


def test_coordination_schema_is_valid_and_example_matches() -> None:
    schema = read_json("schemas/dispatch-scenario.schema.json")
    Draft202012Validator.check_schema(schema)
    validate_dispatch_scenario_document(
        read_json("scenarios/rolling-dispatch-benchmark.json"), ROOT / "schemas"
    )


@pytest.mark.parametrize(
    ("scenario_path", "vehicle_count", "task_count"),
    (
        ("scenarios/realistic-multi-fleet.json", 14, 32),
        ("scenarios/interactive-multi-fleet.json", 4, 6),
    ),
)
def test_realistic_multi_fleet_scenarios_are_valid(
    dispatch_documents,
    scenario_path: str,
    vehicle_count: int,
    task_count: int,
) -> None:
    model, conflicts, workstations, _, scheduler, zones = dispatch_documents
    scenario = read_json(scenario_path)
    validate_dispatch_scenario_document(scenario, ROOT / "schemas")
    assert len(scenario["vehicles"]) == vehicle_count
    assert len(scenario["tasks"]) == task_count

    topology = MapTopology(model, conflicts, workstations, zones)
    for item in scenario["vehicles"]:
        topology.validate_vehicle(Vehicle.from_dict(item))
    defaults = scheduler["serviceDefaults"]
    for item in scenario["tasks"]:
        topology.validate_task(
            TransportTask.from_dict(
                item,
                int(defaults["pickupServiceMs"]),
                int(defaults["dropoffServiceMs"]),
            )
        )


def test_realistic_pressure_tasks_are_individually_schedulable(
    dispatch_documents,
) -> None:
    scenario = read_json("scenarios/realistic-multi-fleet.json")

    for task in scenario["tasks"]:
        single_task = deepcopy(scenario)
        single_task["scenarioId"] = f"single-{task['taskId']}"
        single_task["tasks"] = [task]
        planning, _ = build_dispatch_plans(
            single_task,
            *dispatch_documents,
            ROOT / "schemas",
            policy="congestion",
            seed=0,
        )
        assert planning.unplanned_task_ids == (), task["taskId"]


def test_realistic_pressure_scenario_exercises_jack_shared_corridors(
    dispatch_documents,
) -> None:
    model, _, _, profiles, scheduler, _ = dispatch_documents
    scenario = read_json("scenarios/realistic-multi-fleet.json")
    routes = RouteProvider(
        model,
        EdgeTravelTimeModel(
            model,
            profiles,
            time_quantum_ms=int(scheduler["planner"]["timeQuantumMs"]),
        ),
    )
    edges = {item["id"]: item for item in model["edges"]}
    shared_task_count = 0
    shared_node_visits = 0

    for task in scenario["tasks"]:
        if task["requiredRobotGroup"] != "jack":
            continue
        shortest = routes.candidate_routes(
            "jack",
            task["pickupNodeId"],
            task["dropoffNodeId"],
            LoadState.LOADED,
            limit=1,
        )
        assert shortest, task["taskId"]
        shared_nodes = {
            node_id
            for edge_id in shortest[0].edge_ids
            for node_id in (edges[edge_id]["start"], edges[edge_id]["end"])
            if node_id.startswith("shared:")
        }
        if shared_nodes:
            shared_task_count += 1
            shared_node_visits += len(shared_nodes)

    assert shared_task_count >= 8
    assert shared_node_visits >= 40
