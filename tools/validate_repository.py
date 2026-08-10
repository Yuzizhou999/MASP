from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


ROBOT_GROUPS = {"fork", "jack"}
WAIT_FLAGS = {
    "LM": "allowOnLM",
    "AP": "allowOnAP",
    "PP": "allowOnPP",
    "CP": "allowOnCP",
}
REAL_MODE_SAFETY_FIELDS = (
    "footprintMarginM",
    "localizationErrorM",
    "communicationLatencyMs",
    "fixedClearanceM",
    "guaranteedDecelerationMps2",
    "reservationEntryBufferMs",
    "reservationExitBufferMs",
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def duplicate_values(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def json_path(parts: Iterable[Any]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def schema_issues(
    instance: dict[str, Any],
    schema: dict[str, Any],
    label: str,
) -> list[ValidationIssue]:
    validator = Draft202012Validator(schema)
    return [
        ValidationIssue(
            "error",
            "schema.invalid",
            f"{label} {json_path(error.absolute_path)}: {error.message}",
        )
        for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
    ]


def validate_task(
    task: dict[str, Any],
    task_schema: dict[str, Any],
    model: dict[str, Any],
    workstations: dict[str, Any],
) -> list[ValidationIssue]:
    issues = schema_issues(task, task_schema, "task")
    if issues:
        return issues

    nodes = {node["id"]: node for node in model["nodes"]}
    stations = {item["nodeId"]: item for item in workstations["workstations"]}
    group = task["requiredRobotGroup"]
    for field in ("pickupNodeId", "dropoffNodeId"):
        node_id = task[field]
        node = nodes.get(node_id)
        if node is None:
            issues.append(ValidationIssue("error", "task.node.missing", f"{field} {node_id!r} is unknown"))
            continue
        if node["type"] != "AP":
            issues.append(ValidationIssue("error", "task.node.not_ap", f"{field} {node_id!r} is not an AP"))
        if group not in node["allowedRobotGroups"]:
            issues.append(
                ValidationIssue(
                    "error",
                    "task.group.incompatible",
                    f"{field} {node_id!r} does not allow robot group {group!r}",
                )
            )
        station = stations.get(node_id)
        if station is None or group not in station["allowedRobotGroups"]:
            issues.append(
                ValidationIssue(
                    "error",
                    "task.workstation.incompatible",
                    f"{field} {node_id!r} has no compatible workstation",
                )
            )

    due_time = task.get("dueTimeMs")
    if due_time is not None and due_time < task["releaseTimeMs"]:
        issues.append(
            ValidationIssue(
                "error",
                "task.due_time.before_release",
                "dueTimeMs must not be earlier than releaseTimeMs",
            )
        )
    return issues


def validate_repository(
    model: dict[str, Any],
    conflicts: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    vehicles: dict[str, Any],
    workstations: dict[str, Any],
    traffic_zones: dict[str, Any],
) -> tuple[list[ValidationIssue], dict[str, int]]:
    issues: list[ValidationIssue] = []
    nodes = model.get("nodes", [])
    edges = model.get("edges", [])
    node_ids = [node["id"] for node in nodes]
    edge_ids = [edge["id"] for edge in edges]
    nodes_by_id = {node["id"]: node for node in nodes}
    edges_by_id = {edge["id"]: edge for edge in edges}

    for value in duplicate_values(node_ids):
        issues.append(ValidationIssue("error", "map.node.duplicate", f"duplicate node id {value!r}"))
    for value in duplicate_values(edge_ids):
        issues.append(ValidationIssue("error", "map.edge.duplicate", f"duplicate edge id {value!r}"))

    wait_config = scheduler["traffic"]["wait"]
    for node in nodes:
        groups = set(node["allowedRobotGroups"])
        if not groups or not groups <= ROBOT_GROUPS:
            issues.append(ValidationIssue("error", "map.node.groups", f"node {node['id']!r} has invalid groups"))
            continue
        if set(node.get("propertiesByGroup", {})) != groups:
            issues.append(
                ValidationIssue(
                    "error",
                    "map.node.properties",
                    f"node {node['id']!r} does not preserve properties for every group",
                )
            )
        policies = node.get("waitPolicyByGroup", {})
        if set(policies) != groups:
            issues.append(
                ValidationIssue(
                    "error",
                    "map.node.wait_policy",
                    f"node {node['id']!r} has incomplete wait policies",
                )
            )
            continue
        expected_allowed = bool(wait_config[WAIT_FLAGS[node["type"]]])
        for group in groups:
            policy = policies[group]
            if bool(policy["allowed"]) != expected_allowed:
                issues.append(
                    ValidationIssue(
                        "error",
                        "map.node.wait_policy_mismatch",
                        f"node {node['id']!r} group {group!r} does not match global wait policy",
                    )
                )
            expected_max = wait_config["maxPlannedWaitMs"] if expected_allowed else 0
            if policy["maxWaitMs"] != expected_max:
                issues.append(
                    ValidationIssue(
                        "error",
                        "map.node.max_wait_mismatch",
                        f"node {node['id']!r} group {group!r} has invalid maxWaitMs",
                    )
                )
            if node["allowWaitByGroup"].get(group) != expected_allowed:
                issues.append(
                    ValidationIssue(
                        "error",
                        "map.node.wait_compatibility",
                        f"node {node['id']!r} compatibility wait flag is stale",
                    )
                )

    for edge in edges:
        if edge["start"] not in nodes_by_id or edge["end"] not in nodes_by_id:
            issues.append(ValidationIssue("error", "map.edge.endpoint", f"edge {edge['id']!r} has an unknown endpoint"))
        group = edge.get("robotGroup")
        if group not in ROBOT_GROUPS or edge.get("allowedRobotGroups") != [group]:
            issues.append(ValidationIssue("error", "map.edge.group", f"edge {edge['id']!r} has invalid group metadata"))

    if set(profiles.get("robotGroups", {})) != ROBOT_GROUPS:
        issues.append(ValidationIssue("error", "profiles.groups", "robot profiles must define fork and jack"))

    conflict_node_ids = [item["nodeId"] for item in conflicts["nodeResources"]]
    if set(conflict_node_ids) != set(node_ids):
        issues.append(ValidationIssue("error", "conflicts.nodes", "node resources do not match map nodes"))
    edge_resources = {item["edgeId"]: item for item in conflicts["edgeResources"]}
    if set(edge_resources) != set(edge_ids):
        issues.append(ValidationIssue("error", "conflicts.edges", "edge resources do not match map edges"))

    pair_by_id = {item["resourceId"]: item for item in conflicts["conflictPairs"]}
    if len(pair_by_id) != len(conflicts["conflictPairs"]):
        issues.append(ValidationIssue("error", "conflicts.pair.duplicate", "duplicate conflict pair resource id"))
    references: dict[str, set[str]] = defaultdict(set)
    for edge_id, resource in edge_resources.items():
        for resource_id in resource["conflictResources"]:
            references[resource_id].add(edge_id)
    for resource_id, pair in pair_by_id.items():
        pair_edges = {pair["edgeA"], pair["edgeB"]}
        if not pair_edges <= set(edges_by_id):
            issues.append(ValidationIssue("error", "conflicts.pair.edge", f"{resource_id!r} references an unknown edge"))
        if references.get(resource_id, set()) != pair_edges:
            issues.append(ValidationIssue("error", "conflicts.pair.references", f"{resource_id!r} is not referenced by exactly its two edges"))

    configured_counts = scheduler["fleet"]["counts"]
    actual_counts = Counter(item["robotGroup"] for item in vehicles["vehicles"])
    if dict(actual_counts) != configured_counts:
        issues.append(
            ValidationIssue(
                "error",
                "fleet.counts",
                f"configured fleet counts {configured_counts!r} do not match vehicles {dict(actual_counts)!r}",
            )
        )
    if vehicles["fixedDuringRun"] != scheduler["fleet"]["fixedDuringRun"]:
        issues.append(ValidationIssue("error", "fleet.fixed", "fixedDuringRun differs between scheduler and vehicle config"))
    if not scheduler["fleet"]["fixedDuringRun"] or not vehicles["fixedDuringRun"]:
        issues.append(
            ValidationIssue(
                "error",
                "fleet.dynamic_unsupported",
                "the reference runtime requires a fleet that remains fixed during each run",
            )
        )
    for value in duplicate_values(item["vehicleId"] for item in vehicles["vehicles"]):
        issues.append(ValidationIssue("error", "fleet.vehicle.duplicate", f"duplicate vehicle id {value!r}"))
    for value in duplicate_values(item["initialNodeId"] for item in vehicles["vehicles"]):
        issues.append(ValidationIssue("error", "fleet.start.duplicate", f"multiple vehicles start at node {value!r}"))
    for vehicle in vehicles["vehicles"]:
        node = nodes_by_id.get(vehicle["initialNodeId"])
        if node is None:
            issues.append(ValidationIssue("error", "fleet.start.missing", f"vehicle {vehicle['vehicleId']!r} starts at an unknown node"))
            continue
        group = vehicle["robotGroup"]
        if group not in node["allowedRobotGroups"]:
            issues.append(ValidationIssue("error", "fleet.start.group", f"vehicle {vehicle['vehicleId']!r} cannot use its start node"))
        elif not node["waitPolicyByGroup"][group]["allowed"]:
            issues.append(ValidationIssue("error", "fleet.start.wait", f"vehicle {vehicle['vehicleId']!r} starts at a node that disallows waiting"))

    ap_nodes = {node["id"]: node for node in nodes if node["type"] == "AP"}
    for value in duplicate_values(item["id"] for item in workstations["workstations"]):
        issues.append(ValidationIssue("error", "workstations.id.duplicate", f"duplicate workstation id {value!r}"))
    for value in duplicate_values(item["nodeId"] for item in workstations["workstations"]):
        issues.append(ValidationIssue("error", "workstations.node.duplicate", f"multiple workstations use AP {value!r}"))
    station_by_node = {item["nodeId"]: item for item in workstations["workstations"]}
    if set(station_by_node) != set(ap_nodes):
        issues.append(ValidationIssue("error", "workstations.coverage", "workstations must cover every AP exactly once"))
    for node_id, station in station_by_node.items():
        node = ap_nodes.get(node_id)
        if node is None:
            continue
        if station["allowedRobotGroups"] != node["allowedRobotGroups"]:
            issues.append(ValidationIssue("error", "workstations.groups", f"workstation {station['id']!r} has stale groups"))
        if station["propertiesByGroup"] != node["propertiesByGroup"]:
            issues.append(ValidationIssue("error", "workstations.properties", f"workstation {station['id']!r} has stale properties"))
        if station["blocksTransitDuringService"] is not True:
            issues.append(ValidationIssue("error", "workstations.blocking", f"workstation {station['id']!r} must block its AP node"))

    recovery_node_ids: set[str] = set()
    for recovery in traffic_zones["recoveryNodes"]:
        node_id = recovery["nodeId"]
        if node_id in recovery_node_ids:
            issues.append(ValidationIssue("error", "zones.recovery.duplicate", f"duplicate recovery node {node_id!r}"))
        recovery_node_ids.add(node_id)
        node = nodes_by_id.get(node_id)
        if node is None:
            issues.append(ValidationIssue("error", "zones.recovery.missing", f"unknown recovery node {node_id!r}"))
            continue
        if not set(recovery["allowedRobotGroups"]) <= set(node["allowedRobotGroups"]):
            issues.append(ValidationIssue("error", "zones.recovery.groups", f"recovery node {node_id!r} has invalid groups"))
        for group in recovery["allowedRobotGroups"]:
            if not node["waitPolicyByGroup"][group]["allowed"]:
                issues.append(ValidationIssue("error", "zones.recovery.wait", f"recovery node {node_id!r} disallows waiting"))

    for value in duplicate_values(zone["id"] for zone in traffic_zones["zones"]):
        issues.append(ValidationIssue("error", "zones.duplicate", f"duplicate traffic zone id {value!r}"))
    claimed_zone_edges: dict[str, str] = {}
    claimed_zone_nodes: dict[str, str] = {}
    for zone in traffic_zones["zones"]:
        referenced_nodes = set(zone["memberNodeIds"]) | set(zone["recoveryNodeIds"])
        referenced_edges = set(zone["memberEdgeIds"]) | set(zone["entryEdgeIds"]) | set(zone["exitEdgeIds"])
        if not referenced_nodes <= set(nodes_by_id):
            issues.append(ValidationIssue("error", "zones.nodes", f"zone {zone['id']!r} references unknown nodes"))
        if not referenced_edges <= set(edges_by_id):
            issues.append(ValidationIssue("error", "zones.edges", f"zone {zone['id']!r} references unknown edges"))
        if not set(zone["recoveryNodeIds"]) <= recovery_node_ids:
            issues.append(ValidationIssue("error", "zones.recovery", f"zone {zone['id']!r} references undeclared recovery nodes"))
        if (
            zone["capacity"] != 1
            or zone["passingAllowed"] is not False
            or zone["directionalMode"] != "single_direction_at_a_time"
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "zones.unsupported_capacity",
                    f"zone {zone['id']!r} must be a single-capacity no-passing zone",
                )
            )
        member_nodes = set(zone["memberNodeIds"])
        controlled_edges = (
            set(zone["memberEdgeIds"])
            | set(zone["entryEdgeIds"])
            | set(zone["exitEdgeIds"])
        )
        for edge_id in sorted(controlled_edges):
            previous_zone = claimed_zone_edges.get(edge_id)
            if previous_zone is not None and previous_zone != zone["id"]:
                issues.append(
                    ValidationIssue(
                        "error",
                        "zones.edge.overlap",
                        f"edge {edge_id!r} belongs to zones {previous_zone!r} and {zone['id']!r}",
                    )
                )
            else:
                claimed_zone_edges[edge_id] = zone["id"]
        for node_id in sorted(member_nodes):
            previous_zone = claimed_zone_nodes.get(node_id)
            if previous_zone is not None and previous_zone != zone["id"]:
                issues.append(
                    ValidationIssue(
                        "error",
                        "zones.node.overlap",
                        f"node {node_id!r} belongs to zones {previous_zone!r} and {zone['id']!r}",
                    )
                )
            else:
                claimed_zone_nodes[node_id] = zone["id"]
        for edge_id in zone["memberEdgeIds"]:
            edge = edges_by_id.get(edge_id)
            if edge is not None and not {edge["start"], edge["end"]} & member_nodes:
                issues.append(
                    ValidationIssue(
                        "error",
                        "zones.member.direction",
                        f"zone {zone['id']!r} member edge {edge_id!r} does not touch a member node",
                    )
                )
        for edge_id in zone["entryEdgeIds"]:
            edge = edges_by_id.get(edge_id)
            if edge is not None and not (
                edge["start"] not in member_nodes and edge["end"] in member_nodes
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "zones.entry.direction",
                        f"zone {zone['id']!r} entry edge {edge_id!r} must point into the zone",
                    )
                )
        for edge_id in zone["exitEdgeIds"]:
            edge = edges_by_id.get(edge_id)
            if edge is not None and not (
                edge["start"] in member_nodes and edge["end"] not in member_nodes
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "zones.exit.direction",
                        f"zone {zone['id']!r} exit edge {edge_id!r} must point out of the zone",
                    )
                )
        expected_members = {
            edge_id
            for edge_id, edge in edges_by_id.items()
            if edge["start"] in member_nodes and edge["end"] in member_nodes
        }
        expected_entries = {
            edge_id
            for edge_id, edge in edges_by_id.items()
            if edge["start"] not in member_nodes and edge["end"] in member_nodes
        }
        expected_exits = {
            edge_id
            for edge_id, edge in edges_by_id.items()
            if edge["start"] in member_nodes and edge["end"] not in member_nodes
        }
        if (
            set(zone["memberEdgeIds"]) != expected_members
            or set(zone["entryEdgeIds"]) != expected_entries
            or set(zone["exitEdgeIds"]) != expected_exits
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "zones.boundary.incomplete",
                    f"zone {zone['id']!r} must classify every edge touching its member nodes",
                )
            )

    planner = scheduler["planner"]
    if planner["executionHorizonMs"] > planner["planningHorizonMs"]:
        issues.append(ValidationIssue("error", "planner.horizon", "execution horizon must not exceed planning horizon"))
    reverse = scheduler["traffic"]["reverse"]
    if not wait_config["shortTermOnly"]:
        issues.append(
            ValidationIssue(
                "error",
                "wait.long_term_unsupported",
                "the reference runtime permits waiting only as a bounded short-term action",
            )
        )
    if reverse["alongCurrentEdgeAllowed"] and reverse["mode"] not in {"recovery_only", "planned"}:
        issues.append(ValidationIssue("error", "reverse.mode", "current-edge reverse requires recovery_only or planned mode"))

    safety = scheduler["safety"]
    incomplete_safety = [field for field in REAL_MODE_SAFETY_FIELDS if safety[field] is None]
    if scheduler["mode"] == "real":
        if safety["provisional"] or incomplete_safety:
            issues.append(
                ValidationIssue(
                    "error",
                    "safety.real_mode_blocked",
                    f"real mode requires finalized safety values; missing={incomplete_safety!r}",
                )
            )
    elif incomplete_safety:
        issues.append(
            ValidationIssue(
                "warning",
                "safety.simulation_only",
                "safety values are provisional; configuration is simulation-only",
            )
        )

    stats = {
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "conflictPairCount": len(conflicts["conflictPairs"]),
        "workstationCount": len(workstations["workstations"]),
        "vehicleCount": len(vehicles["vehicles"]),
        "trafficZoneCount": len(traffic_zones["zones"]),
        "recoveryNodeCount": len(traffic_zones["recoveryNodes"]),
    }
    return issues, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MASP models and configuration")
    parser.add_argument("--map", type=Path, default=Path("generated/xiate-unified-map-model.json"))
    parser.add_argument("--conflicts", type=Path, default=Path("generated/xiate-conflict-resources.json"))
    parser.add_argument("--profiles", type=Path, default=Path("config/robot-profiles.json"))
    parser.add_argument("--scheduler", type=Path, default=Path("config/scheduler.json"))
    parser.add_argument("--vehicles", type=Path, default=Path("config/initial-vehicles.json"))
    parser.add_argument("--workstations", type=Path, default=Path("generated/xiate-workstations.json"))
    parser.add_argument("--traffic-zones", type=Path, default=Path("config/traffic-zones.json"))
    parser.add_argument("--schemas", type=Path, default=Path("schemas"))
    parser.add_argument("--task", action="append", type=Path, default=[])
    args = parser.parse_args()

    model = load_json(args.map)
    conflicts = load_json(args.conflicts)
    profiles = load_json(args.profiles)
    scheduler = load_json(args.scheduler)
    vehicles = load_json(args.vehicles)
    workstations = load_json(args.workstations)
    traffic_zones = load_json(args.traffic_zones)

    issues: list[ValidationIssue] = []
    for value, schema_name, label in (
        (scheduler, "scheduler.schema.json", "scheduler"),
        (vehicles, "vehicle.schema.json", "vehicles"),
        (workstations, "workstations.schema.json", "workstations"),
        (traffic_zones, "traffic-zones.schema.json", "traffic-zones"),
    ):
        issues.extend(schema_issues(value, load_json(args.schemas / schema_name), label))

    stats: dict[str, int] = {}
    if not any(issue.severity == "error" for issue in issues):
        semantic_issues, stats = validate_repository(
            model,
            conflicts,
            profiles,
            scheduler,
            vehicles,
            workstations,
            traffic_zones,
        )
        issues.extend(semantic_issues)

    task_schema = load_json(args.schemas / "task.schema.json")
    for task_path in args.task:
        issues.extend(validate_task(load_json(task_path), task_schema, model, workstations))

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    result = {
        "valid": not errors,
        "stats": stats,
        "errors": [asdict(issue) for issue in errors],
        "warnings": [asdict(issue) for issue in warnings],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
