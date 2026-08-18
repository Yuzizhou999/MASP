from __future__ import annotations

from collections import Counter
import json
from itertools import combinations
from pathlib import Path

import networkx as nx

from masp.domain import LoadState, TransportTask, Vehicle
from masp.motion import EdgeTravelTimeModel
from masp.online import run_online_scenario
from masp.routing import RouteProvider, SpatialRoute
from masp.scenario import validate_dispatch_scenario_document
from masp.topology import MapTopology


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "scenarios" / "rhpp-long-distance-conflict.json"
HIGH_VOLUME_SCENARIO_PATH = ROOT / "scenarios" / "rhpp-high-volume-long-distance.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def documents() -> tuple[dict, dict, dict, dict, dict, dict]:
    return (
        read_json(ROOT / "generated/xiate-unified-map-model.json"),
        read_json(ROOT / "generated/xiate-conflict-resources.json"),
        read_json(ROOT / "generated/xiate-workstations.json"),
        read_json(ROOT / "config/robot-profiles.json"),
        read_json(ROOT / "config/scheduler.json"),
        read_json(ROOT / "config/traffic-zones.json"),
    )


def shortest_task_routes(
    scenario: dict, model: dict, profiles: dict, scheduler: dict
) -> list[tuple[dict, SpatialRoute]]:
    routes = RouteProvider(
        model,
        EdgeTravelTimeModel(
            model,
            profiles,
            time_quantum_ms=int(scheduler["planner"]["timeQuantumMs"]),
        ),
    )
    result = []
    for task in scenario["tasks"]:
        candidates = routes.candidate_routes(
            task["requiredRobotGroup"],
            task["pickupNodeId"],
            task["dropoffNodeId"],
            LoadState.LOADED,
            limit=1,
        )
        assert candidates, task["taskId"]
        result.append((task, candidates[0]))
    return result


def test_long_distance_conflict_scenario_has_dense_route_overlap() -> None:
    scenario = read_json(SCENARIO_PATH)
    validate_dispatch_scenario_document(scenario, ROOT / "schemas")
    model, conflicts, workstations, profiles, scheduler, zones = documents()
    topology = MapTopology(model, conflicts, workstations, zones)
    defaults = scheduler["serviceDefaults"]

    for vehicle in scenario["vehicles"]:
        topology.validate_vehicle(Vehicle.from_dict(vehicle))
    for task in scenario["tasks"]:
        topology.validate_task(
            TransportTask.from_dict(
                task,
                int(defaults["pickupServiceMs"]),
                int(defaults["dropoffServiceMs"]),
            )
        )

    task_routes = shortest_task_routes(scenario, model, profiles, scheduler)
    edge_resources = {
        item["edgeId"]: item for item in conflicts["edgeResources"]
    }
    route_resources: list[set[str]] = []
    for _, route in task_routes:
        resources: set[str] = set()
        for edge_id in route.edge_ids:
            resource = edge_resources[edge_id]
            resources.add(resource["ownResource"])
            resources.update(resource["conflictResources"])
        route_resources.append(resources)

    conflict_graph = nx.Graph()
    conflict_graph.add_nodes_from(range(len(task_routes)))
    cross_group_overlaps = 0
    overlapping_pairs = 0
    for left, right in combinations(range(len(task_routes)), 2):
        if not route_resources[left].intersection(route_resources[right]):
            continue
        overlapping_pairs += 1
        conflict_graph.add_edge(left, right)
        if (
            task_routes[left][0]["requiredRobotGroup"]
            != task_routes[right][0]["requiredRobotGroup"]
        ):
            cross_group_overlaps += 1

    assert len(task_routes) == 24
    assert min(route.free_flow_travel_ms for _, route in task_routes) >= 200_000
    assert overlapping_pairs >= 250
    assert cross_group_overlaps >= 120
    assert max(map(len, nx.connected_components(conflict_graph))) == len(task_routes)


def test_high_volume_long_distance_scenario_has_two_reachable_balanced_waves() -> None:
    scenario = read_json(HIGH_VOLUME_SCENARIO_PATH)
    validate_dispatch_scenario_document(scenario, ROOT / "schemas")
    model, _, _, profiles, scheduler, _ = documents()

    tasks = scenario["tasks"]
    assert len(tasks) == 48
    assert Counter(task["releaseTimeMs"] for task in tasks) == {0: 24, 900_000: 24}
    assert Counter(task["requiredRobotGroup"] for task in tasks) == {
        "fork": 24,
        "jack": 24,
    }
    assert len({task["taskId"] for task in tasks}) == len(tasks)
    assert len({task["payloadId"] for task in tasks}) == len(tasks)

    route_signatures = {
        release_time_ms: Counter(
            (
                task["requiredRobotGroup"],
                task["pickupNodeId"],
                task["dropoffNodeId"],
            )
            for task in tasks
            if task["releaseTimeMs"] == release_time_ms
        )
        for release_time_ms in (0, 900_000)
    }
    assert route_signatures[0] == route_signatures[900_000]

    task_routes = shortest_task_routes(scenario, model, profiles, scheduler)
    assert len(task_routes) == 48
    assert min(route.free_flow_travel_ms for _, route in task_routes) >= 200_000


def test_long_distance_conflict_online_run_completes_without_collisions() -> None:
    scenario = read_json(SCENARIO_PATH)
    runtime = run_online_scenario(
        scenario,
        *documents(),
        policy="congestion",
        seed=int(scenario["seed"]),
    )
    result = runtime.result()
    planning = runtime.planning_result().summary()

    assert result["metrics"]["completedTaskCount"] == len(scenario["tasks"])
    assert result["metrics"]["reservationConflictRejections"] == 0
    assert planning["coupledConflictComponentCount"] > 0
    assert planning["largestConflictComponent"] >= 6
