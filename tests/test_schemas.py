from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_CASES = (
    ("schemas/scheduler.schema.json", "config/scheduler.json"),
    ("schemas/vehicle.schema.json", "config/initial-vehicles.json"),
    ("schemas/workstations.schema.json", "generated/xiate-workstations.json"),
    ("schemas/traffic-zones.schema.json", "config/traffic-zones.json"),
)


def read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("schema_path,instance_path", SCHEMA_CASES)
def test_repository_documents_match_their_schemas(
    schema_path: str,
    instance_path: str,
) -> None:
    schema = read_json(schema_path)
    Draft202012Validator.check_schema(schema)

    errors = list(Draft202012Validator(schema).iter_errors(read_json(instance_path)))
    assert errors == []


def test_task_schema_accepts_the_minimum_transport_task() -> None:
    schema = read_json("schemas/task.schema.json")
    Draft202012Validator.check_schema(schema)
    task = {
        "taskId": "task-001",
        "releaseTimeMs": 0,
        "pickupNodeId": "fork:AP1001",
        "dropoffNodeId": "fork:AP1002",
        "requiredRobotGroup": "fork",
        "payloadType": "pallet",
    }

    assert list(Draft202012Validator(schema).iter_errors(task)) == []


def test_schemas_reject_unconfirmed_runtime_policy_changes() -> None:
    scheduler_schema = read_json("schemas/scheduler.schema.json")
    scheduler = read_json("config/scheduler.json")
    scheduler["fleet"]["fixedDuringRun"] = False
    scheduler["traffic"]["wait"]["shortTermOnly"] = False

    scheduler_errors = list(
        Draft202012Validator(scheduler_schema).iter_errors(scheduler)
    )
    assert len(scheduler_errors) == 2

    vehicle_schema = read_json("schemas/vehicle.schema.json")
    vehicles = deepcopy(read_json("config/initial-vehicles.json"))
    vehicles["fixedDuringRun"] = False
    assert len(list(Draft202012Validator(vehicle_schema).iter_errors(vehicles))) == 1
