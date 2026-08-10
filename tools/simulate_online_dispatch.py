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

from masp.online import run_online_scenario  # noqa: E402
from masp.scenario import load_json, validate_dispatch_scenario_document  # noqa: E402


POLICIES = (
    "top_k",
    "task_age",
    "shortest_remaining",
    "congestion",
    "previous_order",
    "random",
    "rl",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
        description="Run the MASP in-process online dispatch simulation"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=ROOT / "scenarios/interactive-multi-fleet.json",
    )
    parser.add_argument("--policy", choices=POLICIES, default="congestion")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rl-checkpoint", type=Path)
    parser.add_argument("--rl-candidates", type=int)
    parser.add_argument("--rl-allow-deviation", action="store_true")
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
        "--profiles",
        type=Path,
        default=ROOT / "config/robot-profiles.json",
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
    args = parser.parse_args()

    scenario_path = args.scenario.resolve()
    scenario = load_json(scenario_path)
    validate_dispatch_scenario_document(scenario, args.schemas)
    model = load_json(args.map)
    conflicts = load_json(args.conflicts)
    workstations = load_json(args.workstations)
    profiles = load_json(args.profiles)
    scheduler = load_json(args.scheduler)
    traffic_zones = load_json(args.traffic_zones)
    seed = int(scenario["seed"] if args.seed is None else args.seed)
    rl_checkpoint = args.rl_checkpoint.resolve() if args.rl_checkpoint else None
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "runs" / f"{scenario['scenarioId']}-online"
    )

    runtime = run_online_scenario(
        scenario,
        model,
        conflicts,
        workstations,
        profiles,
        scheduler,
        traffic_zones,
        policy=args.policy,
        seed=seed,
        rl_checkpoint=str(rl_checkpoint) if rl_checkpoint is not None else None,
        rl_candidate_count=args.rl_candidates,
        rl_allow_deviation=args.rl_allow_deviation,
    )
    planning = runtime.planning_result()
    planning_summary = planning.summary()
    simulation = runtime.result()
    planned_scenario = runtime.planned_scenario(scenario["scenarioId"], seed)
    compact_summary = {
        "schemaVersion": 1,
        "scenarioId": scenario["scenarioId"],
        "mode": "online-simulation",
        "policy": args.policy,
        "seed": seed,
        "planning": {
            key: planning_summary[key]
            for key in (
                "plannedTaskCount",
                "unplannedTaskCount",
                "planningCycleCount",
                "decisionCycleCount",
                "priorityCandidatesEvaluated",
                "insertedWaitMs",
                "routeCombinationsTried",
                "routeCombinationsPruned",
                "scheduleAttempts",
                "maxRouteExpansionLevel",
                "planningDeadlineExhaustedCount",
                "conflictComponentCount",
                "coupledConflictComponentCount",
                "largestConflictComponent",
                "planningLatencyMs",
                "planningTimeoutCount",
                "planningPeriodMissCount",
                "rlInferenceCount",
                "rlFallbackCount",
                "rlSafetyFallbackCount",
                "rlGuardianCandidateCount",
                "rlGuardianOverrideCount",
                "rlAllowDeviation",
                "rlInferenceMs",
            )
        },
        "online": simulation["online"],
        "simulation": simulation["metrics"],
        "eventDigestSha256": simulation["eventDigestSha256"],
    }

    write_json(output_dir / "planned-scenario.json", planned_scenario)
    write_json(output_dir / "planning-summary.json", planning_summary)
    write_json(output_dir / "result.json", simulation)
    write_json(output_dir / "summary.json", compact_summary)
    write_json(
        output_dir / "plan-acknowledgements.json",
        {
            "schemaVersion": 1,
            "acknowledgements": [
                item.to_dict()
                for item in runtime.acknowledgements.values()
            ],
        },
    )
    (output_dir / "events.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in simulation["eventLog"]
        ),
        encoding="utf-8",
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schemaVersion": 1,
            "scenarioId": scenario["scenarioId"],
            "mode": "online-simulation",
            "policy": args.policy,
            "seed": seed,
            "gitCommit": git_commit(),
            "inputs": {
                "scenarioSha256": sha256_file(scenario_path),
                "mapSha256": sha256_file(args.map),
                "conflictsSha256": sha256_file(args.conflicts),
                "workstationsSha256": sha256_file(args.workstations),
                "profilesSha256": sha256_file(args.profiles),
                "schedulerSha256": sha256_file(args.scheduler),
                "trafficZonesSha256": sha256_file(args.traffic_zones),
                "rlCheckpointSha256": (
                    sha256_file(rl_checkpoint)
                    if rl_checkpoint is not None
                    else None
                ),
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        },
    )
    print(json.dumps(compact_summary, ensure_ascii=False, indent=2))
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
