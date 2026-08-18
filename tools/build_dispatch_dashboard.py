"""Build a standalone HTML dashboard from a MASP simulation run.

The dashboard intentionally has no runtime dependencies.  It embeds a compact
map/plan/event model in the generated HTML, so the result can be opened directly
from disk or served by any static HTTP server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from masp.domain import LoadState  # noqa: E402
from masp.motion import EdgeTravelTimeModel  # noqa: E402
from masp.routing import RouteProvider  # noqa: E402


DEFAULT_TEMPLATE = ROOT / "visualization" / "dispatch-dashboard.template.html"
DEFAULT_MAP = ROOT / "generated" / "xiate-unified-scene-model.json"
DEFAULT_MOTION_MAP = ROOT / "generated" / "xiate-unified-map-model.json"
DEFAULT_CONFLICTS = ROOT / "generated" / "xiate-conflict-resources.json"
DEFAULT_PROFILES = ROOT / "config" / "robot-profiles.json"
DEFAULT_SCHEDULER = ROOT / "config" / "scheduler.json"
PLACEHOLDER = "__MASP_DASHBOARD_JSON__"
PLANNING_METRIC_KEYS = (
    "policy",
    "plannedTaskCount",
    "unplannedTaskCount",
    "planningPeriodMs",
    "planningHorizonMs",
    "executionHorizonMs",
    "planningCycleCount",
    "decisionCycleCount",
    "routeCombinationsTried",
    "routeCombinationsPruned",
    "scheduleAttempts",
    "reservationConflictRejections",
    "planningLatencyMs",
    "planningTimeoutCount",
    "planningPeriodMissCount",
    "rlInferenceCount",
    "rlFallbackCount",
    "rlInferenceMs",
    "rlSafetyFallbackCount",
    "rlGuardianCandidateCount",
    "rlGuardianOverrideCount",
    "rlAllowDeviation",
)


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_map(model: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        {
            "id": node["id"],
            "type": node.get("type", "LM"),
            "x": node["x"],
            "y": node["y"],
            "groups": node.get("groups", []),
        }
        for node in model.get("nodes", [])
    ]
    edges = [
        {
            "id": edge["id"],
            "group": edge.get("group", "shared"),
            "start": edge["start"],
            "end": edge["end"],
            "p0": edge["p0"],
            "p1": edge["p1"],
            "p2": edge["p2"],
            "p3": edge["p3"],
            "length": float(edge.get("length", 0.0)),
            "motionDirection": int(edge.get("motionDirection", 0)),
            "shared": bool(edge.get("sharedMatch")),
        }
        for edge in model.get("edges", [])
    ]
    return {
        "bounds": model.get("metadata", {}).get("bounds", {}),
        "stats": model.get("stats", {}),
        "nodes": nodes,
        "edges": edges,
        "sharedOverlays": [
            {"p0": item["p0"], "p1": item["p1"], "p2": item["p2"], "p3": item["p3"]}
            for item in model.get("sharedOverlays", [])
        ],
    }


def compact_vehicle_profiles(profiles: dict[str, Any]) -> dict[str, Any]:
    return {
        group: {
            "length": float(profile["dimensions"]["length"]),
            "width": float(profile["dimensions"]["width"]),
        }
        for group, profile in profiles.get("robotGroups", {}).items()
    }


def compact_sweep_model(
    conflicts: dict[str, Any], profiles: dict[str, Any]
) -> dict[str, Any]:
    """Expose the exact footprint sampling parameters used by conflict generation."""

    metadata = conflicts.get("metadata", {})
    safety = profiles.get("simulationSafety", {})
    margin = float(
        metadata.get("footprintMargin", safety.get("footprintMargin", 0.0))
    )
    return {
        "sampleSpacing": float(metadata.get("sampleSpacing", 0.25)),
        "footprintMargin": margin,
        "baseGeometryOnly": bool(metadata.get("baseGeometryOnly", margin == 0.0)),
    }


def compact_planning(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in PLANNING_METRIC_KEYS
        if key in summary
    }


def _fit_phase_durations(values: list[int], target_ms: int) -> list[int]:
    total_ms = sum(values)
    if target_ms <= 0 or total_ms <= 0:
        return [0, max(0, target_ms), 0]
    scaled = [value * target_ms / total_ms for value in values]
    result = [int(value) for value in scaled]
    remainder = target_ms - sum(result)
    order = sorted(
        range(len(scaled)),
        key=lambda index: scaled[index] - result[index],
        reverse=True,
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def compact_plan(
    plan: dict[str, Any],
    *,
    edges: dict[str, dict[str, Any]] | None = None,
    travel_times: EdgeTravelTimeModel | None = None,
) -> dict[str, Any]:
    segments = []
    explicit_rotations = any(
        segment.get("kind") == "rotate" for segment in plan.get("segments", [])
    )
    for segment in plan.get("segments", []):
        item = {
            "id": segment.get("id"),
            "kind": segment.get("kind", "unknown"),
            "startMs": int(segment.get("startMs", 0)),
            "endMs": int(segment.get("endMs", 0)),
            "startNodeId": segment.get("startNodeId"),
            "endNodeId": segment.get("endNodeId"),
            "edgeId": segment.get("edgeId"),
            "expectedLoadState": segment.get("expectedLoadState", "empty"),
        }
        if segment.get("commandPayload"):
            item["commandPayload"] = dict(segment["commandPayload"])
        edge = edges.get(item["edgeId"]) if edges is not None and item["edgeId"] else None
        if travel_times is not None and edge is not None and item["kind"] == "traverse":
            phases = travel_times.motion_phases(
                edge, LoadState(item["expectedLoadState"])
            )
            phase_durations = (
                [0, item["endMs"] - item["startMs"], 0]
                if explicit_rotations
                else _fit_phase_durations(
                    [
                        phases.start_rotation_ms,
                        phases.linear_ms,
                        phases.end_rotation_ms,
                    ],
                    item["endMs"] - item["startMs"],
                )
            )
            item["motion"] = {
                "startRotationMs": phase_durations[0],
                "linearMs": phase_durations[1],
                "endRotationMs": phase_durations[2],
                "startHeadingRad": phases.start_heading_rad,
                "travelStartHeadingRad": phases.travel_start_heading_rad,
                "travelEndHeadingRad": phases.travel_end_heading_rad,
                "endHeadingRad": phases.end_heading_rad,
            }
        segments.append(item)
    return {
        "id": plan.get("id"),
        "vehicleId": plan.get("vehicleId"),
        "taskId": plan.get("taskId"),
        "createdAtMs": int(plan.get("createdAtMs", 0)),
        "committedUntilMs": int(plan.get("committedUntilMs", 0)),
        "segments": segments,
    }


def initial_global_route_times(
    planned: dict[str, Any],
    model: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
) -> dict[str, int]:
    """Compute assignment-time free-flow baselines with planner motion costs."""

    routing_model = dict(model)
    routing_model["edges"] = [
        {**edge, "robotGroup": edge.get("robotGroup", edge.get("group"))}
        for edge in model.get("edges", [])
    ]
    planner = scheduler.get("planner", {})
    defaults = scheduler.get("serviceDefaults", {})
    routes = RouteProvider(
        routing_model,
        EdgeTravelTimeModel(
            routing_model,
            profiles,
            time_quantum_ms=int(planner.get("timeQuantumMs", 100)),
        ),
    )
    vehicles = {item["vehicleId"]: item for item in planned.get("vehicles", [])}
    initial_plans: dict[str, list[dict[str, Any]]] = {}
    for plan in planned.get("plans", []):
        if plan.get("continuation") or not plan.get("taskId"):
            continue
        initial_plans.setdefault(plan["taskId"], []).append(plan)

    result: dict[str, int] = {}

    def route_duration(route, load_state: LoadState, entry_heading_rad: float | None = None) -> int:
        return routes.travel_times.route_duration_ms(
            (routes.edges[edge_id] for edge_id in route.edge_ids),
            load_state,
            entry_heading_rad=entry_heading_rad,
            terminal=True,
        )

    for task in planned.get("tasks", []):
        task_id = task["taskId"]
        plans = sorted(
            initial_plans.get(task_id, []),
            key=lambda item: (
                int(item.get("createdAtMs", 0)),
                int(item.get("revision", 0)),
            ),
        )
        if not plans:
            continue
        first_plan = plans[0]
        segments = first_plan.get("segments", [])
        vehicle = vehicles.get(first_plan.get("vehicleId"), {})
        start_node_id = (
            segments[0].get("startNodeId")
            if segments
            else vehicle.get("initialNodeId")
        )
        robot_group = task.get("requiredRobotGroup")
        if not start_node_id or not robot_group:
            continue
        empty_routes = routes.candidate_routes(
            robot_group,
            start_node_id,
            task.get("pickupNodeId"),
            LoadState.EMPTY,
            limit=1,
        )
        if not empty_routes:
            continue
        initial_heading_rad = vehicle.get("initialHeadingRad")
        empty_route = empty_routes[0]
        empty_ms = route_duration(
            empty_route,
            LoadState.EMPTY,
            float(initial_heading_rad) if initial_heading_rad is not None else None,
        )
        loaded_entry_heading_rad = (
            float(initial_heading_rad) if initial_heading_rad is not None else None
        )
        if empty_route.edge_ids:
            loaded_entry_heading_rad = routes.travel_times.motion_phases(
                routes.edges[empty_route.edge_ids[-1]], LoadState.EMPTY
            ).end_heading_rad
        loaded_routes = routes.candidate_routes(
            robot_group,
            task.get("pickupNodeId"),
            task.get("dropoffNodeId"),
            LoadState.LOADED,
            limit=1,
        )
        if not loaded_routes:
            continue
        loaded_ms = route_duration(
            loaded_routes[0], LoadState.LOADED, loaded_entry_heading_rad
        )
        result[task_id] = (
            empty_ms
            + int(task.get("pickupServiceMs", defaults.get("pickupServiceMs", 0)))
            + loaded_ms
            + int(task.get("dropoffServiceMs", defaults.get("dropoffServiceMs", 0)))
        )
    return result


def build_bundle(
    run_dir: Path,
    map_path: Path,
    *,
    motion_map_path: Path | None = None,
    scenario_path: Path | None = None,
    baseline_run_dir: Path | None = None,
    conflicts_path: Path = DEFAULT_CONFLICTS,
    profiles_path: Path = DEFAULT_PROFILES,
    scheduler_path: Path = DEFAULT_SCHEDULER,
) -> dict[str, Any]:
    result = load_json(run_dir / "result.json")
    planned = load_json(run_dir / "planned-scenario.json", required=False)
    planning = load_json(run_dir / "planning-summary.json", required=False)
    baseline_planning = (
        load_json(baseline_run_dir / "planning-summary.json", required=False)
        if baseline_run_dir is not None
        else {}
    )
    manifest = load_json(run_dir / "manifest.json", required=False)
    scenario = load_json(scenario_path, required=False) if scenario_path else {}
    model = load_json(map_path)
    motion_model = load_json(motion_map_path) if motion_map_path else model
    conflicts = load_json(conflicts_path, required=False)
    profiles = load_json(profiles_path)
    scheduler = load_json(scheduler_path)
    route_baselines = initial_global_route_times(
        planned, motion_model, profiles, scheduler
    )
    motion_model = dict(motion_model)
    motion_model["edges"] = [
        {**edge, "robotGroup": edge.get("robotGroup", edge.get("group"))}
        for edge in motion_model.get("edges", [])
    ]
    motion_edges = {edge["id"]: edge for edge in motion_model["edges"]}
    travel_times = EdgeTravelTimeModel(
        motion_model,
        profiles,
        time_quantum_ms=int(scheduler.get("planner", {}).get("timeQuantumMs", 100)),
    )

    result_tasks = {item["taskId"]: item for item in result.get("tasks", [])}
    planned_tasks = {item["taskId"]: item for item in planned.get("tasks", [])}
    tasks = []
    task_items = planned_tasks or result_tasks
    for task_id, task in task_items.items():
        merged = dict(task)
        merged.update(result_tasks.get(task_id, {}))
        tasks.append(
            {
                "taskId": task_id,
                "releaseTimeMs": int(merged.get("releaseTimeMs", 0)),
                "pickupNodeId": merged.get("pickupNodeId"),
                "dropoffNodeId": merged.get("dropoffNodeId"),
                "requiredRobotGroup": merged.get("requiredRobotGroup", "unknown"),
                "state": merged.get("state", "UNKNOWN"),
                "assignedVehicleId": merged.get("assignedVehicleId"),
                "assignedAtMs": merged.get("assignedAtMs"),
                "pickedAtMs": merged.get("pickedAtMs"),
                "completedAtMs": merged.get("completedAtMs"),
                "dueTimeMs": merged.get("dueTimeMs"),
                "initialGlobalRouteMs": route_baselines.get(task_id),
            }
        )

    result_vehicles = {item["vehicleId"]: item for item in result.get("vehicles", [])}
    planned_vehicles = {item["vehicleId"]: item for item in planned.get("vehicles", [])}
    vehicles = []
    vehicle_items = planned_vehicles or result_vehicles
    for vehicle_id, vehicle in vehicle_items.items():
        merged = dict(vehicle)
        merged.update(result_vehicles.get(vehicle_id, {}))
        vehicles.append(
            {
                "vehicleId": vehicle_id,
                "robotGroup": merged.get("robotGroup", "unknown"),
                "initialNodeId": merged.get("initialNodeId", merged.get("currentNodeId")),
                "initialHeadingRad": float(merged.get("initialHeadingRad", 0.0)),
                "state": merged.get("state", "UNKNOWN"),
                "loadState": merged.get("loadState", "empty"),
                "activeTaskId": merged.get("activeTaskId"),
                "availableAtMs": merged.get("availableAtMs"),
            }
        )

    plans = [
        compact_plan(item, edges=motion_edges, travel_times=travel_times)
        for item in planned.get("plans", [])
    ]
    end_time = int(result.get("endTimeMs") or planned.get("endTimeMs") or scenario.get("endTimeMs") or 0)
    return {
        "scenarioId": planned.get("scenarioId") or scenario.get("scenarioId") or run_dir.name,
        "seed": planned.get("seed", scenario.get("seed")),
        "endTimeMs": end_time,
        "replayMode": (
            "online"
            if result.get("online") or manifest.get("mode") == "online-simulation"
            else "offline"
        ),
        "map": compact_map(model),
        "sweepModel": compact_sweep_model(conflicts, profiles),
        "vehicleProfiles": compact_vehicle_profiles(profiles),
        "vehicles": vehicles,
        "tasks": tasks,
        "plans": plans,
        "events": result.get("eventLog", []),
        "metrics": result.get("metrics", {}),
        "online": result.get("online", {}),
        "planning": compact_planning(planning),
        "baselinePlanning": compact_planning(baseline_planning),
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone MASP dispatch dashboard")
    parser.add_argument("run_dir", type=Path, help="Run directory containing result.json")
    parser.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    parser.add_argument(
        "--motion-map",
        dest="motion_map_path",
        type=Path,
        default=None,
        help="Planner map containing node headings used to split rotation and travel phases",
    )
    parser.add_argument(
        "--conflicts",
        type=Path,
        default=DEFAULT_CONFLICTS,
        help="Conflict resources whose footprint sampling metadata is visualized",
    )
    parser.add_argument(
        "--profiles", type=Path, default=DEFAULT_PROFILES, help="Robot profiles with vehicle dimensions"
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--scenario", type=Path, default=None, help="Optional source scenario JSON")
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help="Optional baseline run directory used for planning metric comparisons",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.run_dir / "dispatch-dashboard.html"
    template = args.template.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"Missing placeholder {PLACEHOLDER} in {args.template}")
    motion_map_path = args.motion_map_path
    if motion_map_path is None and args.map_path.resolve() == DEFAULT_MAP.resolve():
        motion_map_path = DEFAULT_MOTION_MAP
    bundle = build_bundle(
        args.run_dir,
        args.map_path,
        motion_map_path=motion_map_path,
        scenario_path=args.scenario,
        baseline_run_dir=args.baseline_run,
        conflicts_path=args.conflicts,
        profiles_path=args.profiles,
        scheduler_path=args.scheduler,
    )
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "vehicles": len(bundle["vehicles"]), "tasks": len(bundle["tasks"]), "events": len(bundle["events"]), "bytes": output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
