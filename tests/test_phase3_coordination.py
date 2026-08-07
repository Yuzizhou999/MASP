from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from masp.domain import SegmentKind, TransportTask, Vehicle
from masp.phase3 import CandidateScore, RollingHorizonPlanner
from masp.scenario import (
    build_phase3_plans,
    build_simulator,
    validate_phase3_scenario_document,
)
from masp.topology import MapTopology
from masp.sipp import SippPlanningError

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phase3_documents():
    return (
        read_json("generated/xiate-unified-map-model.json"),
        read_json("generated/xiate-conflict-resources.json"),
        read_json("generated/xiate-workstations.json"),
        read_json("config/robot-profiles.json"),
        read_json("config/scheduler.json"),
        read_json("config/traffic-zones.json"),
    )


@pytest.fixture(scope="module")
def phase3_top_k_run(phase3_documents):
    scenario = read_json("scenarios/phase3-rh-pp-benchmark.json")
    planning, planned = build_phase3_plans(
        scenario, *phase3_documents, ROOT / "schemas"
    )
    return scenario, planning, planned


def test_candidate_score_uses_lexicographic_throughput_priority() -> None:
    more_dropoffs = CandidateScore(2, 2, 100_000, 100_000, 100_000, 100_000, 500_000)
    less_dropoffs = CandidateScore(1, 3, 0, 0, 0, 0, 1)

    assert more_dropoffs.ordering_key() < less_dropoffs.ordering_key()


def test_random_priority_orders_are_reproducible(phase3_documents) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = phase3_documents
    scenario = read_json("scenarios/phase3-rh-pp-benchmark.json")
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


def test_idle_cycles_jump_to_the_next_release_time(phase3_documents) -> None:
    scenario = deepcopy(read_json("scenarios/phase3-rh-pp-benchmark.json"))
    scenario["tasks"] = scenario["tasks"][:1]
    scenario["tasks"][0]["releaseTimeMs"] = 1234
    scenario["endTimeMs"] = 500000

    planning, _ = build_phase3_plans(
        scenario, *phase3_documents, ROOT / "schemas", policy="congestion", seed=0
    )

    assert planning.unplanned_task_ids == ()
    assert planning.cycles[0].decision_time_ms == 0
    assert planning.cycles[1].decision_time_ms == 1234


def test_failed_pairs_do_not_spin_without_a_future_state_change(
    phase3_documents, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, conflicts, workstations, profiles, scheduler, zones = phase3_documents
    scenario = read_json("scenarios/phase3-rh-pp-benchmark.json")
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


def test_top_k_rh_pp_is_safe_and_completes_stream(
    phase3_documents, phase3_top_k_run
) -> None:
    scenario, planning, planned = phase3_top_k_run
    model, conflicts, workstations, _, scheduler, _ = phase3_documents
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
    phase3_documents,
) -> None:
    scenario = read_json("scenarios/phase3-rh-pp-benchmark.json")
    model, conflicts, workstations, _, scheduler, _ = phase3_documents
    throughput = {}
    for policy in ("congestion", "random"):
        planning, planned = build_phase3_plans(
            scenario,
            *phase3_documents,
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


def test_phase3_schema_is_valid_and_example_matches() -> None:
    schema = read_json("schemas/phase3-scenario.schema.json")
    Draft202012Validator.check_schema(schema)
    validate_phase3_scenario_document(
        read_json("scenarios/phase3-rh-pp-benchmark.json"), ROOT / "schemas"
    )


@pytest.mark.parametrize(
    ("scenario_path", "vehicle_count", "task_count"),
    (
        ("scenarios/phase3-realistic-multi-fleet.json", 14, 32),
        ("scenarios/phase3-realistic-multi-fleet-interactive.json", 4, 6),
    ),
)
def test_realistic_multi_fleet_scenarios_are_valid(
    phase3_documents,
    scenario_path: str,
    vehicle_count: int,
    task_count: int,
) -> None:
    model, conflicts, workstations, _, scheduler, zones = phase3_documents
    scenario = read_json(scenario_path)
    validate_phase3_scenario_document(scenario, ROOT / "schemas")
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
    phase3_documents,
) -> None:
    scenario = read_json("scenarios/phase3-realistic-multi-fleet.json")

    for task in scenario["tasks"]:
        single_task = deepcopy(scenario)
        single_task["scenarioId"] = f"single-{task['taskId']}"
        single_task["tasks"] = [task]
        planning, _ = build_phase3_plans(
            single_task,
            *phase3_documents,
            ROOT / "schemas",
            policy="congestion",
            seed=0,
        )
        assert planning.unplanned_task_ids == (), task["taskId"]
