from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .domain import DomainError, TransportTask, Vehicle, VehiclePlan
from .planning import TaskPlanner, PlanningResult
from .coordination import DispatchPlanningResult, RollingHorizonPlanner
from .simulator import DeterministicSimulator
from .topology import MapTopology


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# 检查规范
def validate_scenario_document(
    scenario: dict[str, Any],
    schemas_dir: Path,
) -> None:
    task_schema = load_json(schemas_dir / "task.schema.json")
    plan_schema = load_json(schemas_dir / "plan.schema.json")
    scenario_schema = load_json(schemas_dir / "simulation-scenario.schema.json")
    registry = Registry().with_resources(
        [
            (task_schema["$id"], Resource.from_contents(task_schema)),
            (plan_schema["$id"], Resource.from_contents(plan_schema)),
        ]
    )
    errors = sorted(
        Draft202012Validator(scenario_schema, registry=registry).iter_errors(scenario),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise DomainError(
            "scenario.schema.invalid", f"scenario {path}: {error.message}"
        )


def validate_planning_scenario_document(
    scenario: dict[str, Any],
    schemas_dir: Path,
) -> None:
    task_schema = load_json(schemas_dir / "task.schema.json")
    scenario_schema = load_json(schemas_dir / "planning-scenario.schema.json")
    registry = Registry().with_resource(
        task_schema["$id"], Resource.from_contents(task_schema)
    )
    errors = sorted(
        Draft202012Validator(scenario_schema, registry=registry).iter_errors(scenario),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise DomainError(
            "planning.scenario.schema.invalid", f"scenario {path}: {error.message}"
        )


def validate_dispatch_scenario_document(
    scenario: dict[str, Any],
    schemas_dir: Path,
) -> None:
    task_schema = load_json(schemas_dir / "task.schema.json")
    scenario_schema = load_json(schemas_dir / "dispatch-scenario.schema.json")
    registry = Registry().with_resource(
        task_schema["$id"], Resource.from_contents(task_schema)
    )
    errors = sorted(
        Draft202012Validator(scenario_schema, registry=registry).iter_errors(scenario),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise DomainError(
            "coordination.scenario.schema.invalid", f"scenario {path}: {error.message}"
        )


def build_simulator(
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    scheduler: dict[str, Any],
    schemas_dir: Path,
    traffic_zones: dict[str, Any] | None = None,
) -> DeterministicSimulator:
    validate_scenario_document(scenario, schemas_dir)
    # 把场景里每个任务 JSON 变成 TransportTask 对象
    service_defaults = scheduler["serviceDefaults"]
    tasks = [
        TransportTask.from_dict(
            item,
            int(service_defaults["pickupServiceMs"]),
            int(service_defaults["dropoffServiceMs"]),
        )
        for item in scenario["tasks"]
    ]
    return DeterministicSimulator(
        topology=MapTopology(model, conflicts, workstations, traffic_zones),
        vehicles=[Vehicle.from_dict(item) for item in scenario["vehicles"]],
        tasks=tasks,
        plans=[VehiclePlan.from_dict(item) for item in scenario["plans"]],
        end_time_ms=int(scenario["endTimeMs"]),
    )


def build_plans(
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    schemas_dir: Path,
) -> tuple[PlanningResult, dict[str, Any]]:
    validate_planning_scenario_document(scenario, schemas_dir)
    defaults = scheduler["serviceDefaults"]
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    tasks = [
        TransportTask.from_dict(
            item,
            int(defaults["pickupServiceMs"]),
            int(defaults["dropoffServiceMs"]),
        )
        for item in scenario["tasks"]
    ]
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    planning = TaskPlanner(
        topology, model, profiles, scheduler, traffic_zones
    ).plan(vehicles, tasks, int(scenario["endTimeMs"]))
    planned_scenario = {
        "schemaVersion": 1,
        "scenarioId": f"{scenario['scenarioId']}-planned",
        "seed": scenario["seed"],
        "endTimeMs": scenario["endTimeMs"],
        "vehicles": scenario["vehicles"],
        "tasks": scenario["tasks"],
        "plans": [plan.to_dict() for plan in planning.plans],
    }
    validate_scenario_document(planned_scenario, schemas_dir)
    return planning, planned_scenario


def build_dispatch_plans(
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    schemas_dir: Path,
    *,
    policy: str | None = None,
    seed: int | None = None,
    priority_policy: Any | None = None,
    rl_checkpoint: str | Path | None = None,
    rl_candidate_count: int | None = None,
    rl_allow_deviation: bool = False,
) -> tuple[DispatchPlanningResult, dict[str, Any]]:
    validate_dispatch_scenario_document(scenario, schemas_dir)
    defaults = scheduler["serviceDefaults"]
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    tasks = [
        TransportTask.from_dict(
            item,
            int(defaults["pickupServiceMs"]),
            int(defaults["dropoffServiceMs"]),
        )
        for item in scenario["tasks"]
    ]
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    planning = RollingHorizonPlanner(
        topology,
        model,
        profiles,
        scheduler,
        traffic_zones,
        policy=policy,
        seed=int(scenario["seed"] if seed is None else seed),
        priority_policy=priority_policy,
        rl_checkpoint=str(rl_checkpoint) if rl_checkpoint is not None else None,
        rl_candidate_count=rl_candidate_count,
        rl_allow_deviation=rl_allow_deviation,
    ).plan(vehicles, tasks, int(scenario["endTimeMs"]))
    planned_scenario = {
        "schemaVersion": 1,
        "scenarioId": f"{scenario['scenarioId']}-{planning.policy}-planned",
        "seed": int(scenario["seed"] if seed is None else seed),
        "endTimeMs": scenario["endTimeMs"],
        "vehicles": scenario["vehicles"],
        "tasks": scenario["tasks"],
        "plans": [plan.to_dict() for plan in planning.plans],
    }
    validate_scenario_document(planned_scenario, schemas_dir)
    return planning, planned_scenario
