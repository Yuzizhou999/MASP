from __future__ import annotations

import json
from pathlib import Path

from tools.build_dispatch_dashboard import build_bundle


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_dashboard_compacts_planning_metrics_and_optional_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "optimized"
    baseline_dir = tmp_path / "baseline"
    map_path = tmp_path / "map.json"
    write_json(
        map_path,
        {
            "metadata": {
                "bounds": {"minX": 0, "maxX": 1, "minY": 0, "maxY": 1}
            },
            "stats": {},
            "nodes": [
                {"id": "shared:A", "type": "LM", "x": 0, "y": 0},
                {"id": "shared:B", "type": "LM", "x": 1, "y": 0},
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
            "routeCombinationsTried": 25,
            "scheduleAttempts": 50,
            "planningLatencyMs": {"p95": 12.5, "max": 30.0},
            "planningTimeoutCount": 0,
            "rlInferenceCount": 3,
            "cycles": [{"large": "field excluded from dashboard"}],
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
    )

    assert bundle["map"]["edges"][0]["shared"] is True
    assert bundle["planning"]["rlInferenceCount"] == 3
    assert bundle["planning"]["routeCombinationsTried"] == 25
    assert "cycles" not in bundle["planning"]
    assert bundle["baselinePlanning"]["routeCombinationsTried"] == 100
