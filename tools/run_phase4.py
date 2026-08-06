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

from masp.phase4 import run_phase4_scenario  # noqa: E402
from masp.scenario import load_json  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MASP phase 4 deadlock supervision and reverse recovery"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=ROOT / "scenarios/phase4-deadlock-recovery.json",
    )
    parser.add_argument("--output-dir", type=Path)
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
        "--profiles", type=Path, default=ROOT / "config/robot-profiles.json"
    )
    parser.add_argument(
        "--scheduler", type=Path, default=ROOT / "config/scheduler.json"
    )
    parser.add_argument(
        "--traffic-zones", type=Path, default=ROOT / "config/traffic-zones.json"
    )
    parser.add_argument("--schemas", type=Path, default=ROOT / "schemas")
    args = parser.parse_args()

    scenario_path = args.scenario.resolve()
    scenario = load_json(scenario_path)
    result = run_phase4_scenario(
        scenario,
        load_json(args.map),
        load_json(args.conflicts),
        load_json(args.workstations),
        load_json(args.profiles),
        load_json(args.scheduler),
        load_json(args.traffic_zones),
        args.schemas,
    )
    document = result.to_dict()
    if not result.accepted:
        failed = [name for name, accepted in result.checks.items() if not accepted]
        raise SystemExit(f"phase 4 acceptance checks failed: {failed!r}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "runs" / scenario["scenarioId"]
    )
    write_json(output_dir / "summary.json", document)
    write_json(
        output_dir / "manifest.json",
        {
            "schemaVersion": 1,
            "scenarioId": scenario["scenarioId"],
            "gitCommit": git_commit(),
            "inputs": {
                "scenarioSha256": sha256_file(scenario_path),
                "mapSha256": sha256_file(args.map),
                "conflictsSha256": sha256_file(args.conflicts),
                "workstationsSha256": sha256_file(args.workstations),
                "profilesSha256": sha256_file(args.profiles),
                "schedulerSha256": sha256_file(args.scheduler),
                "trafficZonesSha256": sha256_file(args.traffic_zones),
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        },
    )
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
