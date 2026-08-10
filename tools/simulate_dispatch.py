from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from masp.scenario import (  # noqa: E402
    build_dispatch_plans,
    build_simulator,
    load_json,
)


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


def execute_policy(
    *,
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    schemas: Path,
    policy: str,
    seed: int,
    rl_checkpoint: Path | None = None,
    rl_candidate_count: int | None = None,
    rl_allow_deviation: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    planning, planned_scenario = build_dispatch_plans(
        scenario,
        model,
        conflicts,
        workstations,
        profiles,
        scheduler,
        traffic_zones,
        schemas,
        policy=policy,
        seed=seed,
        rl_checkpoint=rl_checkpoint,
        rl_candidate_count=rl_candidate_count,
        rl_allow_deviation=rl_allow_deviation,
    )
    if planning.unplanned_task_ids:
        raise RuntimeError(
            f"{policy} seed {seed} left tasks unplanned: "
            + ", ".join(planning.unplanned_task_ids)
        )
    simulation = build_simulator(
        planned_scenario,
        model,
        conflicts,
        workstations,
        scheduler,
        schemas,
        traffic_zones=traffic_zones,
    ).run()
    return planning, planned_scenario, simulation


def run_benchmark(
    *,
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    schemas: Path,
    primary_planning_summary: dict[str, Any],
    rl_checkpoint: Path | None = None,
    rl_candidate_count: int | None = None,
    rl_allow_deviation: bool = False,
) -> dict[str, Any]:
    seeds = [int(value) for value in scheduler["coordination"]["benchmarkRandomSeeds"]]
    raw_runs: list[dict[str, Any]] = []
    benchmark_policies: list[tuple[str, list[int]]] = [
        ("congestion", [int(scenario["seed"])]),
        ("random", seeds),
    ]
    if rl_checkpoint is not None:
        benchmark_policies.append(("rl", seeds))
    for policy, policy_seeds in benchmark_policies:
        for seed in policy_seeds:
            planning, _, simulation = execute_policy(
                scenario=scenario,
                model=model,
                conflicts=conflicts,
                workstations=workstations,
                profiles=profiles,
                scheduler=scheduler,
                traffic_zones=traffic_zones,
                schemas=schemas,
                policy=policy,
                seed=seed,
                rl_checkpoint=rl_checkpoint,
                rl_candidate_count=rl_candidate_count,
                rl_allow_deviation=rl_allow_deviation,
            )
            planning_summary = planning.summary()
            metrics = simulation["metrics"]
            raw_runs.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "completedTaskCount": metrics["completedTaskCount"],
                    "completedDropoffsPerHour": metrics["completedDropoffsPerHour"],
                    "insertedWaitMs": planning_summary["insertedWaitMs"],
                    "planningP95Ms": planning_summary["planningLatencyMs"]["p95"],
                    "planningPeriodMissCount": planning_summary[
                        "planningPeriodMissCount"
                    ],
                    "planningTimeoutCount": planning_summary["planningTimeoutCount"],
                    "reservationConflictRejections": (
                        planning_summary["reservationConflictRejections"]
                        + metrics["reservationConflictRejections"]
                    ),
                    "rlInferenceCount": planning_summary["rlInferenceCount"],
                    "rlFallbackCount": planning_summary["rlFallbackCount"],
                    "rlSafetyFallbackCount": planning_summary[
                        "rlSafetyFallbackCount"
                    ],
                    "rlGuardianCandidateCount": planning_summary[
                        "rlGuardianCandidateCount"
                    ],
                    "rlGuardianOverrideCount": planning_summary[
                        "rlGuardianOverrideCount"
                    ],
                    "rlInferenceMs": planning_summary["rlInferenceMs"],
                    "eventDigestSha256": simulation["eventDigestSha256"],
                }
            )

    aggregates: dict[str, dict[str, Any]] = {}
    for policy, _ in benchmark_policies:
        rows = [item for item in raw_runs if item["policy"] == policy]
        aggregates[policy] = {
            "runCount": len(rows),
            "meanCompletedTaskCount": round(
                statistics.fmean(item["completedTaskCount"] for item in rows), 6
            ),
            "meanCompletedDropoffsPerHour": round(
                statistics.fmean(item["completedDropoffsPerHour"] for item in rows),
                6,
            ),
            "meanInsertedWaitMs": round(
                statistics.fmean(item["insertedWaitMs"] for item in rows), 3
            ),
            "maxPlanningP95Ms": max(item["planningP95Ms"] for item in rows),
            "planningPeriodMissCount": sum(
                item["planningPeriodMissCount"] for item in rows
            ),
            "planningTimeoutCount": sum(
                item["planningTimeoutCount"] for item in rows
            ),
            "reservationConflictRejections": sum(
                item["reservationConflictRejections"] for item in rows
            ),
            "rlInferenceCount": sum(item["rlInferenceCount"] for item in rows),
            "rlFallbackCount": sum(item["rlFallbackCount"] for item in rows),
            "rlSafetyFallbackCount": sum(
                item["rlSafetyFallbackCount"] for item in rows
            ),
            "rlGuardianCandidateCount": sum(
                item["rlGuardianCandidateCount"] for item in rows
            ),
            "rlGuardianOverrideCount": sum(
                item["rlGuardianOverrideCount"] for item in rows
            ),
            "meanRlInferenceMs": round(
                statistics.fmean(item["rlInferenceMs"] for item in rows), 3
            ),
        }

    congestion = aggregates["congestion"]
    random_baseline = aggregates["random"]
    checks = {
        "congestionThroughputNotWorseThanRandom": (
            congestion["meanCompletedDropoffsPerHour"]
            >= random_baseline["meanCompletedDropoffsPerHour"]
        ),
        "congestionCompletedTasksNotWorseThanRandom": (
            congestion["meanCompletedTaskCount"]
            >= random_baseline["meanCompletedTaskCount"]
        ),
        "allRunsConflictFree": all(
            item["reservationConflictRejections"] == 0 for item in raw_runs
        ),
        "topKPlanningP95WithinPeriod": (
            primary_planning_summary["planningLatencyMs"]["p95"]
            < scheduler["planner"]["planningPeriodMs"]
        ),
        "baselinePlanningP95WithinPeriod": all(
            item["maxPlanningP95Ms"] < scheduler["planner"]["planningPeriodMs"]
            for item in aggregates.values()
        ),
    }
    if "rl" in aggregates:
        checks["rlThroughputAboveBestBaseline"] = (
            aggregates["rl"]["meanCompletedDropoffsPerHour"]
            > max(
                aggregates["congestion"]["meanCompletedDropoffsPerHour"],
                aggregates["random"]["meanCompletedDropoffsPerHour"],
            )
        )
        checks["rlInferenceHadNoFallback"] = aggregates["rl"]["rlFallbackCount"] == 0
    acceptance_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"rlThroughputAboveBestBaseline"}
    }
    return {
        "schemaVersion": 1,
        "randomSeeds": seeds,
        "runs": raw_runs,
        "aggregates": aggregates,
        "checks": checks,
        "accepted": all(acceptance_checks.values()),
        "rlExitConditionMet": bool(
            checks.get("rlThroughputAboveBestBaseline", False)
            and checks.get("rlInferenceHadNoFallback", True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MASP rolling-horizon dispatch and priority coordination"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=ROOT / "scenarios/rolling-dispatch-benchmark.json",
    )
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        help="PPO priority checkpoint; invalid or missing checkpoints use congestion fallback",
    )
    parser.add_argument(
        "--rl-candidates",
        type=int,
        help="override the number of sampled RL permutations evaluated per decision",
    )
    parser.add_argument(
        "--rl-allow-deviation",
        action="store_true",
        help="experimental: allow RL to replace a feasible congestion guardian",
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
    model = load_json(args.map)
    conflicts = load_json(args.conflicts)
    workstations = load_json(args.workstations)
    profiles = load_json(args.profiles)
    scheduler = load_json(args.scheduler)
    traffic_zones = load_json(args.traffic_zones)
    policy = args.policy or str(scheduler["coordination"]["defaultPolicy"])
    seed = int(scenario["seed"] if args.seed is None else args.seed)
    rl_checkpoint = args.rl_checkpoint.resolve() if args.rl_checkpoint else None
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "runs" / scenario["scenarioId"]
    )

    planning, planned_scenario, simulation = execute_policy(
        scenario=scenario,
        model=model,
        conflicts=conflicts,
        workstations=workstations,
        profiles=profiles,
        scheduler=scheduler,
        traffic_zones=traffic_zones,
        schemas=args.schemas,
        policy=policy,
        seed=seed,
        rl_checkpoint=rl_checkpoint,
        rl_candidate_count=args.rl_candidates,
        rl_allow_deviation=args.rl_allow_deviation,
    )
    planning_summary = planning.summary()
    benchmark = None
    if not args.skip_benchmark:
        benchmark = run_benchmark(
            scenario=scenario,
            model=model,
            conflicts=conflicts,
            workstations=workstations,
            profiles=profiles,
            scheduler=scheduler,
            traffic_zones=traffic_zones,
            schemas=args.schemas,
            primary_planning_summary=planning_summary,
            rl_checkpoint=rl_checkpoint,
            rl_candidate_count=args.rl_candidates,
            rl_allow_deviation=args.rl_allow_deviation,
        )
        if not benchmark["accepted"]:
            raise SystemExit("dispatch benchmark acceptance checks failed")

    compact_summary = {
        "schemaVersion": 1,
        "scenarioId": scenario["scenarioId"],
        "policy": policy,
        "seed": seed,
        "planning": {
            key: planning_summary[key]
            for key in (
                "plannedTaskCount",
                "unplannedTaskCount",
                "planningCycleCount",
                "decisionCycleCount",
                "priorityCandidatesEvaluated",
                "feasiblePriorityCandidateCount",
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
        "simulation": simulation["metrics"],
        "eventDigestSha256": simulation["eventDigestSha256"],
        "benchmarkAccepted": benchmark["accepted"] if benchmark is not None else None,
        "rlExitConditionMet": (
            benchmark["rlExitConditionMet"] if benchmark is not None else None
        ),
    }
    write_json(output_dir / "planned-scenario.json", planned_scenario)
    write_json(output_dir / "planning-summary.json", planning_summary)
    write_json(output_dir / "result.json", simulation)
    write_json(output_dir / "summary.json", compact_summary)
    if benchmark is not None:
        write_json(output_dir / "benchmark.json", benchmark)
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
            "policy": policy,
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
                    if rl_checkpoint is not None and rl_checkpoint.is_file()
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
    if benchmark is not None:
        print(json.dumps(benchmark["aggregates"], ensure_ascii=False, indent=2))
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
