from __future__ import annotations

import json
from pathlib import Path

from masp.domain import LoadState
from masp.motion import EdgeTravelTimeModel
from tools.build_dispatch_dashboard import build_bundle, initial_global_route_times


TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "visualization" / "dispatch-dashboard.template.html"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_dashboard_compacts_planning_metrics_and_optional_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "optimized"
    baseline_dir = tmp_path / "baseline"
    map_path = tmp_path / "map.json"
    profiles_path = tmp_path / "profiles.json"
    scheduler_path = tmp_path / "scheduler.json"
    conflicts_path = tmp_path / "conflicts.json"
    motion_limits = {
        "maxForwardSpeed": 2.0,
        "maxReverseSpeed": 1.0,
        "maxAcceleration": 1.0,
        "maxDeceleration": 1.0,
        "maxRotationSpeed": 90.0,
        "maxRotationAcceleration": 120.0,
        "maxRotationDeceleration": 90.0,
    }
    write_json(
        profiles_path,
        {
            "robotGroups": {
                "fork": {
                    "dimensions": {"length": 2.0, "width": 1.0},
                    "unloaded": motion_limits,
                    "loaded": motion_limits,
                },
                "jack": {
                    "dimensions": {"length": 1.0, "width": 0.5},
                    "unloaded": motion_limits,
                    "loaded": motion_limits,
                },
            }
        },
    )
    write_json(
        scheduler_path,
        {
            "planner": {"timeQuantumMs": 100},
            "serviceDefaults": {"pickupServiceMs": 500, "dropoffServiceMs": 500},
        },
    )
    write_json(
        conflicts_path,
        {
            "metadata": {
                "sampleSpacing": 0.2,
                "footprintMargin": 0.15,
                "baseGeometryOnly": False,
            }
        },
    )
    write_json(
        map_path,
        {
            "metadata": {
                "bounds": {"minX": 0, "maxX": 1, "minY": 0, "maxY": 1}
            },
            "stats": {},
            "nodes": [
                {
                    "id": "shared:A",
                    "type": "LM",
                    "x": 0,
                    "y": 0,
                    "headings": {"fork": 0.0},
                },
                {
                    "id": "shared:B",
                    "type": "LM",
                    "x": 1,
                    "y": 0,
                    "headings": {"fork": 0.0},
                },
            ],
            "edges": [
                {
                    "id": "fork:e1",
                    "group": "fork",
                    "start": "shared:A",
                    "end": "shared:B",
                    "p0": [0, 0],
                    "p1": [0.3, 0],
                    "p2": [0.7, 0],
                    "p3": [1, 0],
                    "length": 1.0,
                    "motionDirection": 1,
                    "sharedMatch": "shared-path-1",
                }
            ],
            "sharedOverlays": [],
        },
    )
    write_json(
        run_dir / "result.json",
        {
            "endTimeMs": 1000,
            "tasks": [],
            "vehicles": [],
            "eventLog": [],
            "metrics": {"reservationConflictRejections": 0},
            "online": {},
        },
    )
    write_json(
        run_dir / "planning-summary.json",
        {
            "policy": "rl",
            "planningHorizonMs": 120000,
            "executionHorizonMs": 5000,
            "routeCombinationsTried": 25,
            "scheduleAttempts": 50,
            "planningLatencyMs": {"p95": 12.5, "max": 30.0},
            "planningTimeoutCount": 0,
            "rlInferenceCount": 3,
            "cycles": [{"large": "field excluded from dashboard"}],
        },
    )
    write_json(
        run_dir / "planned-scenario.json",
        {
            "scenarioId": "dashboard-fixture",
            "seed": 3,
            "endTimeMs": 1000,
            "tasks": [],
            "vehicles": [
                {
                    "vehicleId": "fork-001",
                    "robotGroup": "fork",
                    "initialNodeId": "shared:A",
                    "initialHeadingRad": 1.25,
                    "initialLoadState": "empty",
                }
            ],
            "plans": [
                {
                    "id": "plan-001",
                    "vehicleId": "fork-001",
                    "createdAtMs": 0,
                    "committedUntilMs": 1000,
                    "segments": [
                        {
                            "id": "segment-001",
                            "kind": "traverse",
                            "startMs": 0,
                            "endMs": 1000,
                            "startNodeId": "shared:A",
                            "endNodeId": "shared:B",
                            "edgeId": "fork:e1",
                            "expectedLoadState": "empty",
                        }
                    ],
                }
            ],
        },
    )
    write_json(
        baseline_dir / "planning-summary.json",
        {
            "policy": "congestion",
            "routeCombinationsTried": 100,
            "scheduleAttempts": 200,
            "planningLatencyMs": {"p95": 40.0, "max": 80.0},
            "planningTimeoutCount": 2,
        },
    )

    bundle = build_bundle(
        run_dir,
        map_path,
        baseline_run_dir=baseline_dir,
        conflicts_path=conflicts_path,
        profiles_path=profiles_path,
        scheduler_path=scheduler_path,
    )

    assert bundle["map"]["edges"][0]["shared"] is True
    assert bundle["map"]["edges"][0]["motionDirection"] == 1
    assert bundle["vehicleProfiles"]["fork"] == {"length": 2.0, "width": 1.0}
    assert bundle["sweepModel"] == {
        "sampleSpacing": 0.2,
        "footprintMargin": 0.15,
        "baseGeometryOnly": False,
    }
    assert bundle["vehicles"][0]["initialHeadingRad"] == 1.25
    assert bundle["planning"]["rlInferenceCount"] == 3
    assert bundle["planning"]["planningHorizonMs"] == 120000
    assert bundle["planning"]["executionHorizonMs"] == 5000
    assert bundle["planning"]["routeCombinationsTried"] == 25
    assert "cycles" not in bundle["planning"]
    assert bundle["baselinePlanning"]["routeCombinationsTried"] == 100
    motion = bundle["plans"][0]["segments"][0]["motion"]
    assert motion["startRotationMs"] > 0
    assert motion["linearMs"] > 0
    assert motion["endRotationMs"] > 0
    assert sum(motion[key] for key in ("startRotationMs", "linearMs", "endRotationMs")) == 1000


def test_dashboard_template_exposes_vehicle_footprints_and_live_commitments() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    for marker in (
        "vehicle-body",
        "vehicle-front",
        "commitment-sweep",
        "sampledFootprintPathD",
        "sampledFootprintPathD(edge,profile,startT=0,endT=1)",
        "geometryHeading",
        "traversalState",
        "start-rotation",
        "end-rotation",
        "shortestAngleDelta",
        "segment.motion",
        "原地转向",
        "sweepModel.sampleSpacing",
        "sweepModel.footprintMargin",
        "Math.max(3,Math.ceil(Number(edge.length||0)*(to-from)/spacing)+1)",
        "const localCorners=[[-halfLength,-halfWidth],[halfLength,-halfWidth],[halfLength,halfWidth],[-halfLength,halfWidth]]",
        "new Set(remaining.edgeIds)",
        "commitment-locks",
        "commitment-lock-fork",
        "commitment-lock-jack",
        "remaining.traversals.map",
        "窗口内实际扫掠范围",
        "同边完整锁定范围",
        "r:.32",
        "commitment-route",
        "currentPlanFor",
        "drawCommitments",
        "for(const vehicleId of selectedVehicles)",
        "selectedVehicles.has(vehicle.vehicleId)",
        "partialPathD(edge,startT=0,endT=1)",
        "remainingCommitment(plan,currentMs,windowEndMs)",
        "DATA.planning?.executionHorizonMs",
        "vehicle-hitbox",
        "selectVehicleFromTarget",
        "vehicleList.addEventListener('pointerdown'",
        "row.dataset.vehicle=vehicle.vehicleId",
        "const vehicleRows=new Map",
        "const vehicleGroups=new Map",
        "row.querySelector('.row-title').textContent",
        "toggle-vehicle-labels",
        "hide-vehicle-labels",
        "route-time-stats",
        "initialGlobalRouteMs",
        "renderRouteTimeStats",
        "collapsible-panel",
        "initialHeadingRad",
        "road-directions",
        "road-direction",
        "roadDirectionTransform(edge,t=.62)",
        "transform:roadDirectionTransform(edge)",
        "new Set(DATA.vehicles.map(vehicle => vehicle.vehicleId))",
        "id=\"zoom-in\"",
        "id=\"zoom-out\"",
        "svg.addEventListener('wheel'",
        "svg.setPointerCapture(event.pointerId)",
        "setMapViewBox",
        "if(expanded && wasFullView)zoomMap(.6)",
        '<body class="hide-vehicle-labels">',
        'id="toggle-vehicle-labels" type="button" aria-pressed="false">显示小车编号',
    ):
        assert marker in template
    assert "Math.hypot(Number(profile.length),Number(profile.width))" not in template
    assert "r:.7,class:`commitment-end" not in template


def test_initial_global_route_time_uses_free_flow_motion_and_service_costs() -> None:
    model = {
        "nodes": [
            {"id": "fork:A", "x": 0, "y": 0},
            {"id": "fork:B", "x": 1, "y": 0},
        ],
        "edges": [
            {
                "id": "fork:e1",
                "group": "fork",
                "start": "fork:A",
                "end": "fork:B",
                "p0": [0, 0],
                "p1": [0.3, 0],
                "p2": [0.7, 0],
                "p3": [1, 0],
                "length": 1.0,
                "motionDirection": 0,
            }
        ],
    }
    motion_limits = {
        "maxForwardSpeed": 2.0,
        "maxReverseSpeed": 1.0,
        "maxAcceleration": 0.5,
        "maxDeceleration": 0.5,
        "maxRotationSpeed": 90.0,
        "maxRotationAcceleration": 60.0,
        "maxRotationDeceleration": 60.0,
    }
    profiles = {
        "robotGroups": {
            "fork": {
                "dimensions": {"length": 2.0, "width": 1.0},
                "unloaded": motion_limits,
                "loaded": motion_limits,
            }
        }
    }
    planned = {
        "vehicles": [
            {"vehicleId": "fork-001", "robotGroup": "fork", "initialNodeId": "fork:A"}
        ],
        "tasks": [
            {
                "taskId": "task-001",
                "pickupNodeId": "fork:A",
                "dropoffNodeId": "fork:B",
                "requiredRobotGroup": "fork",
                "pickupServiceMs": 500,
                "dropoffServiceMs": 500,
            }
        ],
        "plans": [
            {
                "vehicleId": "fork-001",
                "taskId": "task-001",
                "createdAtMs": 0,
                "revision": 0,
                "continuation": False,
                "segments": [{"startNodeId": "fork:A"}],
            }
        ],
    }

    baselines = initial_global_route_times(
        planned,
        model,
        profiles,
        {"planner": {"timeQuantumMs": 100}, "serviceDefaults": {}},
    )

    assert baselines == {"task-001": 3900}


def test_initial_global_route_time_keeps_heading_between_edges() -> None:
    model = {
        "nodes": [
            {"id": "fork:A", "headings": {"fork": 0.0}},
            {"id": "fork:B", "headings": {"fork": 0.0}},
            {"id": "fork:C", "headings": {"fork": 3.141592653589793}},
            {"id": "fork:D", "headings": {"fork": 0.0}},
        ],
        "edges": [
            {
                "id": "fork:e1",
                "group": "fork",
                "start": "fork:A",
                "end": "fork:B",
                "p0": [0, 0],
                "p1": [0.3, 0],
                "p2": [0.7, 0],
                "p3": [1, 0],
                "length": 1.0,
                "motionDirection": 0,
            },
            {
                "id": "fork:e2",
                "group": "fork",
                "start": "fork:B",
                "end": "fork:C",
                "p0": [1, 0],
                "p1": [1.3, 0],
                "p2": [1.7, 0],
                "p3": [2, 0],
                "length": 1.0,
                "motionDirection": 0,
            },
            {
                "id": "fork:e3",
                "group": "fork",
                "start": "fork:C",
                "end": "fork:D",
                "p0": [2, 0],
                "p1": [2.3, 0],
                "p2": [2.7, 0],
                "p3": [3, 0],
                "length": 1.0,
                "motionDirection": 0,
            },
        ],
    }
    motion_limits = {
        "maxForwardSpeed": 2.0,
        "maxReverseSpeed": 1.0,
        "maxAcceleration": 0.5,
        "maxDeceleration": 0.5,
        "maxRotationSpeed": 90.0,
        "maxRotationAcceleration": 60.0,
        "maxRotationDeceleration": 60.0,
    }
    profiles = {
        "robotGroups": {
            "fork": {
                "dimensions": {"length": 2.0, "width": 1.0},
                "unloaded": motion_limits,
                "loaded": motion_limits,
            }
        }
    }
    planned = {
        "vehicles": [
            {
                "vehicleId": "fork-001",
                "robotGroup": "fork",
                "initialNodeId": "fork:A",
                "initialHeadingRad": 0.0,
            }
        ],
        "tasks": [
            {
                "taskId": "task-001",
                "pickupNodeId": "fork:B",
                "dropoffNodeId": "fork:D",
                "requiredRobotGroup": "fork",
                "pickupServiceMs": 500,
                "dropoffServiceMs": 500,
            }
        ],
        "plans": [
            {
                "vehicleId": "fork-001",
                "taskId": "task-001",
                "createdAtMs": 0,
                "revision": 0,
                "continuation": False,
                "segments": [{"startNodeId": "fork:A"}],
            }
        ],
    }
    scheduler = {"planner": {"timeQuantumMs": 100}, "serviceDefaults": {}}

    baselines = initial_global_route_times(planned, model, profiles, scheduler)
    normalized_edges = [
        {**edge, "robotGroup": "fork"} for edge in model["edges"]
    ]
    travel = EdgeTravelTimeModel(
        {**model, "edges": normalized_edges}, profiles, 100
    )
    expected = (
        travel.route_duration_ms(
            [normalized_edges[0]], LoadState.EMPTY, entry_heading_rad=0.0
        )
        + 500
        + travel.route_duration_ms(
            normalized_edges[1:], LoadState.LOADED, entry_heading_rad=0.0
        )
        + 500
    )
    legacy = (
        sum(travel.duration_ms(edge, LoadState.LOADED) for edge in normalized_edges)
        + 1000
    )

    assert baselines == {"task-001": expected}
    assert expected < legacy
