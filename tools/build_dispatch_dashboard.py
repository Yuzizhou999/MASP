"""Build a standalone HTML dashboard from a MASP simulation run.

The dashboard intentionally has no runtime dependencies.  It embeds a compact
map/plan/event model in the generated HTML, so the result can be opened directly
from disk or served by any static HTTP server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "visualization" / "dispatch-dashboard.template.html"
DEFAULT_MAP = ROOT / "generated" / "xiate-unified-scene-model.json"
PLACEHOLDER = "__MASP_DASHBOARD_JSON__"


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


def compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    segments = []
    for segment in plan.get("segments", []):
        segments.append(
            {
                "id": segment.get("id"),
                "kind": segment.get("kind", "unknown"),
                "startMs": int(segment.get("startMs", 0)),
                "endMs": int(segment.get("endMs", 0)),
                "startNodeId": segment.get("startNodeId"),
                "endNodeId": segment.get("endNodeId"),
                "edgeId": segment.get("edgeId"),
                "expectedLoadState": segment.get("expectedLoadState", "empty"),
            }
        )
    return {
        "id": plan.get("id"),
        "vehicleId": plan.get("vehicleId"),
        "taskId": plan.get("taskId"),
        "createdAtMs": int(plan.get("createdAtMs", 0)),
        "committedUntilMs": int(plan.get("committedUntilMs", 0)),
        "segments": segments,
    }


def build_bundle(
    run_dir: Path,
    map_path: Path,
    *,
    scenario_path: Path | None = None,
) -> dict[str, Any]:
    result = load_json(run_dir / "result.json")
    planned = load_json(run_dir / "planned-scenario.json", required=False)
    planning = load_json(run_dir / "planning-summary.json", required=False)
    manifest = load_json(run_dir / "manifest.json", required=False)
    scenario = load_json(scenario_path, required=False) if scenario_path else {}

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
                "pickedAtMs": merged.get("pickedAtMs"),
                "completedAtMs": merged.get("completedAtMs"),
                "dueTimeMs": merged.get("dueTimeMs"),
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
                "state": merged.get("state", "UNKNOWN"),
                "loadState": merged.get("loadState", "empty"),
                "activeTaskId": merged.get("activeTaskId"),
                "availableAtMs": merged.get("availableAtMs"),
            }
        )

    plans = [compact_plan(item) for item in planned.get("plans", [])]
    end_time = int(result.get("endTimeMs") or planned.get("endTimeMs") or scenario.get("endTimeMs") or 0)
    return {
        "scenarioId": planned.get("scenarioId") or scenario.get("scenarioId") or run_dir.name,
        "seed": planned.get("seed", scenario.get("seed")),
        "endTimeMs": end_time,
        "map": compact_map(load_json(map_path)),
        "vehicles": vehicles,
        "tasks": tasks,
        "plans": plans,
        "events": result.get("eventLog", []),
        "metrics": result.get("metrics", {}),
        "planning": planning,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone MASP dispatch dashboard")
    parser.add_argument("run_dir", type=Path, help="Run directory containing result.json")
    parser.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--scenario", type=Path, default=None, help="Optional source scenario JSON")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.run_dir / "dispatch-dashboard.html"
    template = args.template.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"Missing placeholder {PLACEHOLDER} in {args.template}")
    bundle = build_bundle(args.run_dir, args.map_path, scenario_path=args.scenario)
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "vehicles": len(bundle["vehicles"]), "tasks": len(bundle["tasks"]), "events": len(bundle["events"]), "bytes": output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
