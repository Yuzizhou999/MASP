from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from masp.scenario import build_simulator, load_json  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic MASP phase 1 scenario")
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=ROOT / "scenarios/phase1-single-vehicle.json",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=ROOT / "generated/xiate-unified-map-model.json",
    )
    parser.add_argument(
        "--conflicts",
        type=Path,
        default=ROOT / "generated/xiate-conflict-resources.json",
    )
    parser.add_argument(
        "--workstations",
        type=Path,
        default=ROOT / "generated/xiate-workstations.json",
    )
    parser.add_argument(
        "--scheduler",
        type=Path,
        default=ROOT / "config/scheduler.json",
    )
    parser.add_argument(
        "--traffic-zones",
        type=Path,
        default=ROOT / "config/traffic-zones.json",
    )
    parser.add_argument("--schemas", type=Path, default=ROOT / "schemas")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    scenario_path = args.scenario.resolve()
    scenario = load_json(scenario_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "runs" / scenario["scenarioId"]
    )
    scheduler = load_json(args.scheduler)
    traffic_zones = load_json(args.traffic_zones)
    simulator = build_simulator(
        scenario,
        load_json(args.map),
        load_json(args.conflicts),
        load_json(args.workstations),
        scheduler,
        args.schemas,
        traffic_zones=traffic_zones,
    )
    result = simulator.run()

    write_json(output_dir / "result.json", result)
    summary = {
        "schemaVersion": result["schemaVersion"],
        "scenarioId": scenario["scenarioId"],
        "endTimeMs": result["endTimeMs"],
        "eventDigestSha256": result["eventDigestSha256"],
        "metrics": result["metrics"],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "events.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in result["eventLog"]
        ),
        encoding="utf-8",
    )
    write_json(output_dir / "scheduler.snapshot.json", scheduler)
    manifest = {
        "schemaVersion": 1,
        "scenarioId": scenario["scenarioId"],
        "seed": scenario["seed"],
        "gitCommit": git_commit(),
        "inputs": {
            "scenarioSha256": sha256_file(scenario_path),
            "mapSha256": sha256_file(args.map),
            "conflictsSha256": sha256_file(args.conflicts),
            "workstationsSha256": sha256_file(args.workstations),
            "schedulerSha256": sha256_file(args.scheduler),
            "trafficZonesSha256": sha256_file(args.traffic_zones),
            "rlCheckpointSha256": None,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
