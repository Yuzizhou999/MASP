from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from masp.domain import TransportTask, Vehicle  # noqa: E402
from masp.phase3 import PriorityStrategy, RollingHorizonPlanner  # noqa: E402
from masp.reservations import ReservationTable  # noqa: E402
from masp.rl_priority import (  # noqa: E402
    PPOPriorityTrainer,
    PriorityObservationEncoder,
    PriorityOrderEnv,
    PriorityOrderNetwork,
    PriorityTrainingCase,
    reward_from_candidate,
)
from masp.scenario import load_json  # noqa: E402
from masp.topology import MapTopology  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_release_snapshot_cases(
    *,
    scenario_paths: list[Path],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    max_candidates: int,
    path_token_count: int,
) -> tuple[list[PriorityTrainingCase], dict[str, Any]]:
    cases: list[PriorityTrainingCase] = []
    encoder_config: dict[str, Any] | None = None
    for scenario_path in scenario_paths:
        scenario = load_json(scenario_path)
        topology = MapTopology(model, conflicts, workstations, traffic_zones)
        planner = RollingHorizonPlanner(
            topology,
            model,
            profiles,
            scheduler,
            traffic_zones,
            policy=PriorityStrategy.CONGESTION.value,
            seed=int(scenario["seed"]),
        )
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
        projections, tasks_by_id = planner._validate_inputs(vehicles, tasks)
        end_time_ms = int(scenario["endTimeMs"])
        reservations = ReservationTable()
        reservations.insert_batch(
            planner._hold(
                vehicle,
                plan_id=f"phase5-idle:{vehicle.vehicle_id}",
                node_id=vehicle.current_node_id or "",
                start_ms=0,
                end_ms=end_time_ms,
                label="idle-tail",
            )
            for vehicle in projections.values()
        )
        encoder = PriorityObservationEncoder(
            topology,
            planner.routes,
            planning_horizon_ms=planner.planning_horizon_ms,
            max_candidates=max_candidates,
            path_token_count=path_token_count,
        )
        encoder_config = encoder.config
        decision_times = sorted(
            {0, *(task.release_time_ms for task in tasks if task.release_time_ms < end_time_ms)}
        )
        for decision_time_ms in decision_times:
            pending = [
                task for task in tasks if task.release_time_ms <= decision_time_ms
            ]
            proposals = planner.allocator.assign(
                list(projections.values()), pending, decision_time_ms
            )
            if len(proposals) < 2 or len(proposals) > max_candidates:
                continue
            observation = encoder.encode(
                proposals,
                tasks_by_id,
                projections,
                reservations,
                decision_time_ms,
            )

            def evaluate(
                order_indices: tuple[int, ...],
                *,
                case_proposals=proposals,
                case_now_ms=decision_time_ms,
                case_tasks=tasks_by_id,
                case_projections=projections,
                case_reservations=reservations,
                case_planner=planner,
                case_end_ms=end_time_ms,
            ) -> tuple[float, dict[str, Any]]:
                order = tuple(case_proposals[index] for index in order_indices)
                outcome = case_planner._evaluate_candidate(
                    candidate_id="phase5-training",
                    strategy=PriorityStrategy.RL.value,
                    order=order,
                    now_ms=case_now_ms,
                    end_time_ms=case_end_ms,
                    tasks_by_id=case_tasks,
                    base_projections=case_projections,
                    base_reservations=case_reservations,
                    plan_counts=Counter(),
                )
                return reward_from_candidate(outcome)

            reference_order = planner._order_for_strategy(
                PriorityStrategy.CONGESTION,
                proposals,
                tasks_by_id,
                reservations,
                projections,
                decision_time_ms,
                0,
                0,
                0,
            )
            reference_action = tuple(proposals.index(item) for item in reference_order)

            cases.append(
                PriorityTrainingCase(
                    observation=observation,
                    evaluator=evaluate,
                    case_id=f"{scenario['scenarioId']}@{decision_time_ms}",
                    reference_action=reference_action,
                )
            )
    if not cases or encoder_config is None:
        raise RuntimeError(
            "no trainable decision state had at least two assigned vehicles"
        )
    return cases, encoder_config


def _clone_planner_state(
    planner: RollingHorizonPlanner,
    projections: dict[str, Vehicle],
    reservations: ReservationTable,
) -> tuple[dict[str, Vehicle], ReservationTable]:
    frozen_projections = {
        vehicle_id: Vehicle(
            vehicle_id=vehicle.vehicle_id,
            robot_group=vehicle.robot_group,
            current_node_id=vehicle.current_node_id,
            heading_rad=vehicle.heading_rad,
            load_state=vehicle.load_state,
            payload_id=vehicle.payload_id,
            capabilities=frozenset(vehicle.capabilities),
            revision=vehicle.revision,
            state=vehicle.state,
            current_edge_id=vehicle.current_edge_id,
            active_task_id=vehicle.active_task_id,
            plan_id=vehicle.plan_id,
            plan_revision=vehicle.plan_revision,
            committed_until_ms=vehicle.committed_until_ms,
            available_at_ms=vehicle.available_at_ms,
            fault_code=vehicle.fault_code,
            state_changed_at_ms=vehicle.state_changed_at_ms,
            state_durations_ms=Counter(vehicle.state_durations_ms),
            waiting_resume_state=vehicle.waiting_resume_state,
        )
        for vehicle_id, vehicle in projections.items()
    }
    return frozen_projections, planner._copy_reservations(reservations)


def _make_case_from_context(
    *,
    planner: RollingHorizonPlanner,
    encoder: PriorityObservationEncoder,
    scenario_id: str,
    end_time_ms: int,
    proposals: tuple[Any, ...],
    tasks_by_id: dict[str, TransportTask],
    projections: dict[str, Vehicle],
    reservations: ReservationTable,
    now_ms: int,
    context_index: int,
    reference_action: tuple[int, ...] | None = None,
) -> PriorityTrainingCase:
    observation = encoder.encode(
        proposals,
        tasks_by_id,
        projections,
        reservations,
        now_ms,
    )

    def evaluate(
        order_indices: tuple[int, ...],
        *,
        case_proposals=proposals,
        case_now_ms=now_ms,
        case_tasks=tasks_by_id,
        case_projections=projections,
        case_reservations=reservations,
        case_planner=planner,
        case_end_ms=end_time_ms,
    ) -> tuple[float, dict[str, Any]]:
        order = tuple(case_proposals[index] for index in order_indices)
        outcome = case_planner._evaluate_candidate(
            candidate_id="phase5-rolling-training",
            strategy=PriorityStrategy.RL.value,
            order=order,
            now_ms=case_now_ms,
            end_time_ms=case_end_ms,
            tasks_by_id=case_tasks,
            base_projections=case_projections,
            base_reservations=case_reservations,
            plan_counts=Counter(),
        )
        return reward_from_candidate(outcome)

    return PriorityTrainingCase(
        observation=observation,
        evaluator=evaluate,
        case_id=f"{scenario_id}@{now_ms}#{context_index}",
        reference_action=reference_action,
    )


def _build_rolling_training_cases(
    *,
    scenario_paths: list[Path],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    max_candidates: int,
    path_token_count: int,
) -> tuple[list[PriorityTrainingCase], dict[str, Any]]:
    cases: list[PriorityTrainingCase] = []
    encoder_config: dict[str, Any] | None = None
    for scenario_path in scenario_paths:
        scenario = load_json(scenario_path)
        topology = MapTopology(model, conflicts, workstations, traffic_zones)
        planner = RollingHorizonPlanner(
            topology,
            model,
            profiles,
            scheduler,
            traffic_zones,
            policy=PriorityStrategy.CONGESTION.value,
            seed=int(scenario["seed"]),
        )
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
        end_time_ms = int(scenario["endTimeMs"])
        encoder = PriorityObservationEncoder(
            topology,
            planner.routes,
            planning_horizon_ms=planner.planning_horizon_ms,
            max_candidates=max_candidates,
            path_token_count=path_token_count,
        )
        encoder_config = encoder.config
        captured: list[dict[str, Any]] = []
        original_priority_orders = planner._priority_orders

        def capture_priority_orders(*args: Any, **kwargs: Any):
            proposals = tuple(args[0]) if args else tuple(kwargs["proposals"])
            tasks_by_id = args[1] if len(args) > 1 else kwargs["tasks_by_id"]
            reservations = args[2] if len(args) > 2 else kwargs["reservations"]
            projections = args[3] if len(args) > 3 else kwargs["projections"]
            now_ms = args[4] if len(args) > 4 else kwargs["now_ms"]
            cycle_index = args[5] if len(args) > 5 else kwargs["cycle_index"]
            round_index = args[6] if len(args) > 6 else kwargs["round_index"]
            captured_context: dict[str, Any] | None = None
            if 2 <= len(proposals) <= max_candidates:
                frozen_projections, frozen_reservations = _clone_planner_state(
                    planner, projections, reservations
                )
                captured_context = {
                    "proposals": proposals,
                    "tasks": dict(tasks_by_id),
                    "projections": frozen_projections,
                    "reservations": frozen_reservations,
                    "now_ms": int(now_ms),
                }
                captured.append(captured_context)
            orders = original_priority_orders(*args, **kwargs)
            if captured_context is not None:
                baseline_order = planner._order_for_strategy(
                    PriorityStrategy.CONGESTION,
                    proposals,
                    tasks_by_id,
                    reservations,
                    projections,
                    int(now_ms),
                    int(cycle_index),
                    int(round_index),
                    0,
                )
                reference_action = tuple(proposals.index(item) for item in baseline_order)
                if len(reference_action) == len(proposals) and set(reference_action) == set(
                    range(len(proposals))
                ):
                    captured_context["reference_action"] = reference_action
            return orders

        planner._priority_orders = capture_priority_orders
        planner.plan(vehicles, tasks, end_time_ms)
        for index, context in enumerate(captured):
            cases.append(
                _make_case_from_context(
                    planner=planner,
                    encoder=encoder,
                    scenario_id=str(scenario["scenarioId"]),
                    end_time_ms=end_time_ms,
                    proposals=context["proposals"],
                    tasks_by_id=context["tasks"],
                    projections=context["projections"],
                    reservations=context["reservations"],
                    now_ms=context["now_ms"],
                    context_index=index,
                    reference_action=context.get("reference_action"),
                )
            )
    if not cases or encoder_config is None:
        raise RuntimeError("rolling baseline produced no trainable priority state")
    return cases, encoder_config


def build_training_cases(
    *,
    state_source: str,
    scenario_paths: list[Path],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    max_candidates: int,
    path_token_count: int,
) -> tuple[list[PriorityTrainingCase], dict[str, Any]]:
    builder = (
        _build_rolling_training_cases
        if state_source == "rolling"
        else _build_release_snapshot_cases
    )
    return builder(
        scenario_paths=scenario_paths,
        model=model,
        conflicts=conflicts,
        workstations=workstations,
        profiles=profiles,
        scheduler=scheduler,
        traffic_zones=traffic_zones,
        max_candidates=max_candidates,
        path_token_count=path_token_count,
    )


def evaluate_network(
    network: PriorityOrderNetwork,
    cases: list[PriorityTrainingCase],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    network.eval()
    for case in cases:
        observation = case.observation.as_dict()
        tensor_observation = {
            "agent_features": torch.as_tensor(
                observation["agent_features"], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "path_tokens": torch.as_tensor(
                observation["path_tokens"], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "path_mask": torch.as_tensor(
                observation["path_mask"], dtype=torch.bool, device=device
            ).unsqueeze(0),
            "mask": torch.as_tensor(
                observation["mask"], dtype=torch.bool, device=device
            ).unsqueeze(0),
        }
        with torch.no_grad():
            action, _, value = network.act(tensor_observation, deterministic=True)
        order = tuple(
            int(item)
            for item in action[0, : case.observation.candidate_count].tolist()
        )
        evaluated = case.evaluator(order)
        reward, info = evaluated if isinstance(evaluated, tuple) else (evaluated, {})
        rows.append(
            {
                "caseId": case.case_id,
                "candidateCount": case.observation.candidate_count,
                "order": list(order),
                "reward": round(float(reward), 6),
                "value": round(float(value.item()), 6),
                "evaluation": info,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the phase 5 PPO vehicle-priority policy"
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        type=Path,
        default=[ROOT / "scenarios/phase3-realistic-multi-fleet-interactive.json"],
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--behavior-clone-epochs",
        type=int,
        default=2,
        help="warm-start PPO from the congestion priority order",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-source",
        choices=("rolling", "release_snapshot"),
        default="rolling",
        help="collect real rolling planner states or the legacy release-time snapshots",
    )
    parser.add_argument("--max-training-cases", type=int, default=128)
    parser.add_argument("--evaluation-case-limit", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--path-tokens", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/phase5-rl-priority"
    )
    parser.add_argument(
        "--map", type=Path, default=ROOT / "generated/xiate-unified-map-model.json"
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_json(args.map)
    conflicts = load_json(args.conflicts)
    workstations = load_json(args.workstations)
    profiles = load_json(args.profiles)
    scheduler = load_json(args.scheduler)
    traffic_zones = load_json(args.traffic_zones)
    scenario_paths = [path.resolve() for path in args.scenarios]
    cases, encoder_config = build_training_cases(
        state_source=args.state_source,
        scenario_paths=scenario_paths,
        model=model,
        conflicts=conflicts,
        workstations=workstations,
        profiles=profiles,
        scheduler=scheduler,
        traffic_zones=traffic_zones,
        max_candidates=args.max_candidates,
        path_token_count=args.path_tokens,
    )
    captured_case_count = len(cases)
    if args.max_training_cases > 0 and len(cases) > args.max_training_cases:
        indices = np.linspace(
            0, len(cases) - 1, num=args.max_training_cases, dtype=np.int64
        )
        cases = [cases[int(index)] for index in indices]
    env = PriorityOrderEnv(cases, seed=args.seed)
    network = PriorityOrderNetwork(
        max_candidates=args.max_candidates,
        path_token_count=args.path_tokens,
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        transformer_layers=args.transformer_layers,
    )
    device = torch.device(args.device)
    trainer = PPOPriorityTrainer(env, network, device=device)
    behavior_clone_losses = trainer.behavior_clone(
        cases,
        epochs=args.behavior_clone_epochs,
        batch_size=args.batch_size,
    )
    metrics = trainer.train(
        args.steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    checkpoint = trainer.save_checkpoint(
        output_dir / "priority-policy.pt",
        encoder_config=encoder_config,
        metadata={
            "candidate_count": max(1, args.candidate_count),
            "training_scenarios": [str(path) for path in scenario_paths],
            "training_steps": args.steps,
            "seed": args.seed,
            "state_source": args.state_source,
            "captured_case_count": captured_case_count,
            "behavior_clone_epochs": args.behavior_clone_epochs,
        },
    )
    evaluation_cases = cases[: max(1, args.evaluation_case_limit)]
    evaluation = evaluate_network(network, evaluation_cases, device=device)
    summary = {
        "schemaVersion": 1,
        "checkpoint": str(checkpoint),
        "trainingCaseCount": len(cases),
        "capturedCaseCount": captured_case_count,
        "evaluationCaseCount": len(evaluation_cases),
        "stateSource": args.state_source,
        "trainingSteps": args.steps,
        "seed": args.seed,
        "lastUpdate": metrics[-1],
        "behaviorClone": {
            "epochs": args.behavior_clone_epochs,
            "meanLoss": round(
                float(np.mean(behavior_clone_losses)), 6
            )
            if behavior_clone_losses
            else None,
        },
        "meanGreedyReward": round(
            float(np.mean([row["reward"] for row in evaluation])), 6
        ),
        "trainingMetrics": metrics,
        "evaluation": evaluation,
        "safetyBoundary": (
            "RL emits priority permutations only; every reward and deployed order "
            "is evaluated by the unchanged SIPP/reservation/validator pipeline."
        ),
    }
    write_json(output_dir / "training-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
