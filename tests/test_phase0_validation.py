from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from tools.validate_phase0 import validate_repository, validate_task


ROOT = Path(__file__).resolve().parents[1]


def validate(assets: dict[str, dict[str, Any]]) -> tuple[list[Any], dict[str, int]]:
    return validate_repository(
        assets["model"],
        assets["conflicts"],
        assets["profiles"],
        assets["scheduler"],
        assets["vehicles"],
        assets["workstations"],
        assets["traffic_zones"],
    )


def error_codes(issues: list[Any]) -> set[str]:
    return {issue.code for issue in issues if issue.severity == "error"}


def test_phase0_repository_is_valid_for_simulation(
    phase0_assets: dict[str, dict[str, Any]],
) -> None:
    issues, stats = validate(phase0_assets)

    assert error_codes(issues) == set()
    assert {issue.code for issue in issues if issue.severity == "warning"} == {
        "safety.simulation_only"
    }
    assert stats == {
        "nodeCount": 552,
        "edgeCount": 1204,
        "conflictPairCount": 6266,
        "workstationCount": 133,
        "vehicleCount": 14,
        "trafficZoneCount": 0,
        "recoveryNodeCount": 15,
    }


def test_real_mode_is_blocked_until_safety_values_are_finalized(
    phase0_assets: dict[str, dict[str, Any]],
) -> None:
    assets = deepcopy(phase0_assets)
    assets["scheduler"]["mode"] = "real"

    issues, _ = validate(assets)

    assert "safety.real_mode_blocked" in error_codes(issues)


def test_fleet_counts_and_initial_positions_match_configuration(
    phase0_assets: dict[str, dict[str, Any]],
) -> None:
    scheduler = phase0_assets["scheduler"]
    vehicles = phase0_assets["vehicles"]["vehicles"]
    nodes = {node["id"]: node for node in phase0_assets["model"]["nodes"]}

    assert Counter(vehicle["robotGroup"] for vehicle in vehicles) == Counter(
        scheduler["fleet"]["counts"]
    )
    assert len({vehicle["initialNodeId"] for vehicle in vehicles}) == len(vehicles)
    for vehicle in vehicles:
        node = nodes[vehicle["initialNodeId"]]
        group = vehicle["robotGroup"]
        assert group in node["allowedRobotGroups"]
        assert node["waitPolicyByGroup"][group]["allowed"] is True


def test_dynamic_fleet_and_long_term_waiting_are_rejected(
    phase0_assets: dict[str, dict[str, Any]],
) -> None:
    assets = deepcopy(phase0_assets)
    assets["scheduler"]["fleet"]["fixedDuringRun"] = False
    assets["vehicles"]["fixedDuringRun"] = False
    assets["scheduler"]["traffic"]["wait"]["shortTermOnly"] = False

    issues, _ = validate(assets)

    assert "fleet.dynamic_unsupported" in error_codes(issues)
    assert "wait.long_term_unsupported" in error_codes(issues)


def test_workstations_cover_every_ap_and_block_during_service(
    phase0_assets: dict[str, dict[str, Any]],
) -> None:
    ap_ids = {
        node["id"]
        for node in phase0_assets["model"]["nodes"]
        if node["type"] == "AP"
    }
    stations = phase0_assets["workstations"]["workstations"]

    assert {station["nodeId"] for station in stations} == ap_ids
    assert all(station["blocksTransitDuringService"] is True for station in stations)

    assets = deepcopy(phase0_assets)
    assets["workstations"]["workstations"].pop()
    issues, _ = validate(assets)
    assert "workstations.coverage" in error_codes(issues)

    assets = deepcopy(phase0_assets)
    assets["workstations"]["workstations"].append(
        deepcopy(assets["workstations"]["workstations"][0])
    )
    issues, _ = validate(assets)
    assert "workstations.id.duplicate" in error_codes(issues)
    assert "workstations.node.duplicate" in error_codes(issues)


def test_task_group_must_be_compatible_with_both_ap_nodes(
    phase0_assets: dict[str, dict[str, Any]],
) -> None:
    task_schema = json.loads(
        (ROOT / "schemas/task.schema.json").read_text(encoding="utf-8")
    )
    group_only_station = next(
        station
        for station in phase0_assets["workstations"]["workstations"]
        if len(station["allowedRobotGroups"]) == 1
    )
    group = group_only_station["allowedRobotGroups"][0]
    task = {
        "taskId": "task-001",
        "releaseTimeMs": 0,
        "pickupNodeId": group_only_station["nodeId"],
        "dropoffNodeId": group_only_station["nodeId"],
        "requiredRobotGroup": group,
        "payloadType": "test-load",
    }

    assert validate_task(
        task,
        task_schema,
        phase0_assets["model"],
        phase0_assets["workstations"],
    ) == []

    task["requiredRobotGroup"] = "jack" if group == "fork" else "fork"
    issues = validate_task(
        task,
        task_schema,
        phase0_assets["model"],
        phase0_assets["workstations"],
    )
    assert "task.group.incompatible" in error_codes(issues)
    assert "task.workstation.incompatible" in error_codes(issues)
