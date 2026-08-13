from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from masp.domain import TransportTask, Vehicle  # noqa: E402
from masp.coordination import PriorityStrategy, RollingHorizonPlanner  # noqa: E402
from masp.reservations import ReservationTable  # noqa: E402
from masp.rl_priority import (  # noqa: E402
    PPOPriorityTrainer,
    PriorityObservationEncoder,
    PriorityOrderEnv,
    PriorityOrderNetwork,
    PriorityOracleLabel,
    PriorityTrainingCase,
    label_oracle_prefix,
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
                plan_id=f"training-idle:{vehicle.vehicle_id}",
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
            components = planner._conflict_components(
                proposals, tasks_by_id, projections
            )
            for component_index, component in enumerate(components):
                if not 2 <= len(component) <= max_candidates:
                    continue
                baseline_order = planner._order_for_strategy(
                    PriorityStrategy.CONGESTION,
                    component,
                    tasks_by_id,
                    reservations,
                    projections,
                    decision_time_ms,
                    0,
                    0,
                    0,
                )
                cases.append(
                    _make_case_from_context(
                        planner=planner,
                        encoder=encoder,
                        scenario_id=(
                            f"{scenario['scenarioId']}-component-{component_index}"
                        ),
                        end_time_ms=end_time_ms,
                        proposals=baseline_order,
                        tasks_by_id=tasks_by_id,
                        projections=projections,
                        reservations=reservations,
                        now_ms=decision_time_ms,
                        context_index=component_index,
                        reference_action=tuple(range(len(baseline_order))),
                        priority_age_ms=dict(planner.priority_age_ms),
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
    priority_age_ms: dict[str, int] | None = None,
) -> PriorityTrainingCase:
    observation = encoder.encode(
        proposals,
        tasks_by_id,
        projections,
        reservations,
        now_ms,
        priority_age_ms=priority_age_ms,
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
            candidate_id="rolling-training",
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
            components = planner._conflict_components(
                proposals, tasks_by_id, projections
            )
            for component in components:
                if not 2 <= len(component) <= max_candidates:
                    continue
                frozen_projections, frozen_reservations = _clone_planner_state(
                    planner, projections, reservations
                )
                baseline_order = planner._order_for_strategy(
                    PriorityStrategy.CONGESTION,
                    component,
                    tasks_by_id,
                    reservations,
                    projections,
                    int(now_ms),
                    int(cycle_index),
                    int(round_index),
                    0,
                )
                captured_context = {
                    "proposals": baseline_order,
                    "tasks": dict(tasks_by_id),
                    "projections": frozen_projections,
                    "reservations": frozen_reservations,
                    "now_ms": int(now_ms),
                    "reference_action": tuple(range(len(baseline_order))),
                    "priority_age_ms": dict(planner.priority_age_ms),
                }
                captured.append(captured_context)
            orders = original_priority_orders(*args, **kwargs)
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
                    priority_age_ms=context.get("priority_age_ms"),
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


def split_training_cases(
    cases: list[PriorityTrainingCase],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[PriorityTrainingCase], list[PriorityTrainingCase]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(cases) < 2:
        return list(cases), list(cases)
    ranked = sorted(
        cases,
        key=lambda case: hashlib.sha256(
            f"{seed}:{case.case_id}".encode("utf-8")
        ).digest(),
    )
    validation_count = min(
        len(ranked) - 1,
        max(1, round(len(ranked) * validation_fraction)),
    )
    validation_ids = {case.case_id for case in ranked[:validation_count]}
    training = [case for case in cases if case.case_id not in validation_ids]
    validation = [case for case in cases if case.case_id in validation_ids]
    return training, validation


def label_training_cases(
    cases: list[PriorityTrainingCase],
    *,
    prefix_count: int,
    max_evaluations: int,
) -> tuple[list[PriorityTrainingCase], dict[str, PriorityOracleLabel]]:
    labeled: list[PriorityTrainingCase] = []
    labels: dict[str, PriorityOracleLabel] = {}
    for index, case in enumerate(cases, start=1):
        label = label_oracle_prefix(
            case,
            prefix_count=prefix_count,
            max_evaluations=max_evaluations,
        )
        labels[case.case_id] = label
        labeled.append(replace(case, reference_action=label.action))
        if index % 10 == 0 or index == len(cases):
            print(
                json.dumps(
                    {
                        "oracleLabelProgress": index,
                        "oracleLabelTotal": len(cases),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
    return labeled, labels


def evaluate_network(
    network: PriorityOrderNetwork,
    cases: list[PriorityTrainingCase],
    *,
    device: torch.device,
    prefix_count: int | None = None,
    oracle_labels: dict[str, PriorityOracleLabel] | None = None,
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
            "conflict_matrix": torch.as_tensor(
                observation["conflict_matrix"],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0),
        }
        active_prefix = min(
            case.observation.candidate_count,
            case.observation.candidate_count
            if prefix_count is None
            else max(1, int(prefix_count)),
        )
        with torch.no_grad():
            action, _, value = network.act(
                tensor_observation,
                deterministic=True,
                max_steps=active_prefix,
            )
        prefix = tuple(
            int(item)
            for item in action[0, :active_prefix].tolist()
        )
        order = prefix + tuple(
            index
            for index in range(case.observation.candidate_count)
            if index not in set(prefix)
        )
        evaluated = case.evaluator(order)
        reward, info = evaluated if isinstance(evaluated, tuple) else (evaluated, {})
        row = {
            "caseId": case.case_id,
            "candidateCount": case.observation.candidate_count,
            "order": list(order),
            "reward": round(float(reward), 6),
            "value": round(float(value.item()), 6),
            "evaluation": info,
        }
        oracle = None if oracle_labels is None else oracle_labels.get(case.case_id)
        if oracle is not None:
            row["oracle"] = {
                "order": list(oracle.action),
                "reward": round(float(oracle.reward), 6),
                "feasible": oracle.feasible,
                "feasibleActionCount": oracle.feasible_action_count,
                "evaluatedActionCount": oracle.evaluated_action_count,
                "baselineReward": round(float(oracle.baseline_reward), 6),
                "matchesPrefix": (
                    tuple(order[:active_prefix])
                    == tuple(oracle.action[:active_prefix])
                ),
                "rewardDeltaVsBaseline": round(
                    float(reward) - float(oracle.baseline_reward), 6
                ),
            }
        rows.append(row)
    return rows


def summarize_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible_count = sum(
        bool(row["evaluation"].get("feasible", False)) for row in rows
    )
    oracle_feasible_count = sum(
        bool(row.get("oracle", {}).get("feasible", False)) for row in rows
    )
    oracle_match_count = sum(
        bool(row.get("oracle", {}).get("matchesPrefix", False)) for row in rows
    )
    oracle_feasible_rows = [
        row for row in rows if bool(row.get("oracle", {}).get("feasible", False))
    ]
    model_feasible_on_oracle_count = sum(
        bool(row["evaluation"].get("feasible", False))
        for row in oracle_feasible_rows
    )
    oracle_match_on_feasible_count = sum(
        bool(row.get("oracle", {}).get("matchesPrefix", False))
        for row in oracle_feasible_rows
    )
    reward_deltas = [
        float(row["oracle"]["rewardDeltaVsBaseline"])
        for row in rows
        if "oracle" in row
    ]
    return {
        "caseCount": len(rows),
        "modelFeasibleCount": feasible_count,
        "modelFeasibleRate": round(feasible_count / max(1, len(rows)), 6),
        "oracleFeasibleCount": oracle_feasible_count,
        "oracleFeasibleRate": round(
            oracle_feasible_count / max(1, len(rows)), 6
        ),
        "oraclePrefixMatchCount": oracle_match_count,
        "oraclePrefixMatchRate": round(
            oracle_match_count / max(1, len(rows)), 6
        ),
        "modelFeasibleOnOracleFeasibleCount": model_feasible_on_oracle_count,
        "modelFeasibleOnOracleFeasibleRate": round(
            model_feasible_on_oracle_count / max(1, len(oracle_feasible_rows)), 6
        ),
        "oraclePrefixMatchOnFeasibleCount": oracle_match_on_feasible_count,
        "oraclePrefixMatchOnFeasibleRate": round(
            oracle_match_on_feasible_count / max(1, len(oracle_feasible_rows)), 6
        ),
        "meanReward": round(
            float(np.mean([row["reward"] for row in rows])), 6
        )
        if rows
        else None,
        "meanRewardDeltaVsBaseline": round(float(np.mean(reward_deltas)), 6)
        if reward_deltas
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the learned vehicle-priority policy"
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        type=Path,
        default=[ROOT / "scenarios/interactive-multi-fleet.json"],
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="optional one-step PPO fine-tuning steps; 0 keeps oracle supervision",
    )
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--behavior-clone-epochs",
        type=int,
        default=20,
        help="supervised epochs over SIPP-enumerated oracle prefixes",
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
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--oracle-max-evaluations", type=int, default=720)
    parser.add_argument("--feasible-case-weight", type=int, default=4)
    parser.add_argument(
        "--capture-planning-timeout-ms",
        type=int,
        default=None,
        help="optional offline capture deadline; defaults to the deployment scheduler",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--path-tokens", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument(
        "--priority-prefix-count",
        type=int,
        default=2,
        help="number of vehicles learned per local conflict component",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/priority-policy-training"
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
    training_scheduler = copy.deepcopy(scheduler)
    if args.capture_planning_timeout_ms is not None:
        training_scheduler["planner"]["planningTimeoutMs"] = max(
            1_000, int(args.capture_planning_timeout_ms)
        )
    capture_planning_timeout_ms = int(
        training_scheduler["planner"]["planningTimeoutMs"]
    )
    traffic_zones = load_json(args.traffic_zones)
    scenario_paths = [path.resolve() for path in args.scenarios]
    cases, encoder_config = build_training_cases(
        state_source=args.state_source,
        scenario_paths=scenario_paths,
        model=model,
        conflicts=conflicts,
        workstations=workstations,
        profiles=profiles,
        scheduler=training_scheduler,
        traffic_zones=traffic_zones,
        max_candidates=args.max_candidates,
        path_token_count=args.path_tokens,
    )
    captured_case_count = len(cases)
    training_cases, validation_cases = split_training_cases(
        cases,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if (
        args.max_training_cases > 0
        and len(training_cases) > args.max_training_cases
    ):
        indices = np.linspace(
            0,
            len(training_cases) - 1,
            num=args.max_training_cases,
            dtype=np.int64,
        )
        training_cases = [training_cases[int(index)] for index in indices]
    if (
        args.evaluation_case_limit > 0
        and len(validation_cases) > args.evaluation_case_limit
    ):
        validation_cases = validation_cases[: args.evaluation_case_limit]

    training_cases, training_labels = label_training_cases(
        training_cases,
        prefix_count=args.priority_prefix_count,
        max_evaluations=args.oracle_max_evaluations,
    )
    validation_cases, validation_labels = label_training_cases(
        validation_cases,
        prefix_count=args.priority_prefix_count,
        max_evaluations=args.oracle_max_evaluations,
    )
    supervised_cases = list(training_cases)
    feasible_training_cases = [
        case for case in training_cases if training_labels[case.case_id].feasible
    ]
    optimization_cases = supervised_cases + [
        case
        for case in feasible_training_cases
        for _ in range(max(0, args.feasible_case_weight - 1))
    ]
    env = PriorityOrderEnv(
        optimization_cases,
        seed=args.seed,
        prefix_count=args.priority_prefix_count,
    )
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
        optimization_cases,
        epochs=args.behavior_clone_epochs,
        batch_size=args.batch_size,
    )
    metrics = (
        trainer.train(
            args.steps,
            rollout_steps=args.rollout_steps,
            update_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if args.steps > 0
        else []
    )
    evaluation = evaluate_network(
        network,
        validation_cases,
        device=device,
        prefix_count=args.priority_prefix_count,
        oracle_labels=validation_labels,
    )
    training_fit_rows = evaluate_network(
        network,
        supervised_cases[: max(1, args.evaluation_case_limit)],
        device=device,
        prefix_count=args.priority_prefix_count,
        oracle_labels=training_labels,
    )
    validation_summary = summarize_evaluation(evaluation)
    training_fit_summary = summarize_evaluation(training_fit_rows)
    output_dir = args.output_dir.resolve()
    checkpoint = trainer.save_checkpoint(
        output_dir / "priority-policy.pt",
        encoder_config=encoder_config,
        metadata={
            "candidate_count": max(1, args.candidate_count),
            "priority_prefix_count": max(1, args.priority_prefix_count),
            "training_scenarios": [str(path) for path in scenario_paths],
            "training_steps": args.steps,
            "training_method": "sipp_oracle_bc"
            + ("_ppo" if args.steps > 0 else ""),
            "seed": args.seed,
            "state_source": args.state_source,
            "captured_case_count": captured_case_count,
            "behavior_clone_epochs": args.behavior_clone_epochs,
            "validation_fraction": args.validation_fraction,
            "capture_planning_timeout_ms": capture_planning_timeout_ms,
        },
    )
    summary = {
        "schemaVersion": 2,
        "checkpoint": str(checkpoint),
        "trainingMethod": "sipp_oracle_bc" + ("_ppo" if args.steps > 0 else ""),
        "trainingCaseCount": len(supervised_cases),
        "optimizationCaseCount": len(optimization_cases),
        "trainingSplitCaseCount": len(training_cases),
        "fullyFeasibleTrainingCaseCount": len(feasible_training_cases),
        "capturedCaseCount": captured_case_count,
        "evaluationCaseCount": len(validation_cases),
        "stateSource": args.state_source,
        "trainingSteps": args.steps,
        "priorityPrefixCount": max(1, args.priority_prefix_count),
        "seed": args.seed,
        "lastUpdate": metrics[-1] if metrics else None,
        "behaviorClone": {
            "epochs": args.behavior_clone_epochs,
            "meanLoss": round(
                float(np.mean(behavior_clone_losses)), 6
            )
            if behavior_clone_losses
            else None,
        },
        "meanGreedyReward": validation_summary["meanReward"],
        "trainingFit": training_fit_summary,
        "validation": validation_summary,
        "oracleLabeling": {
            "trainingEvaluatedActionCount": sum(
                item.evaluated_action_count for item in training_labels.values()
            ),
            "trainingFeasibleActionCount": sum(
                item.feasible_action_count for item in training_labels.values()
            ),
            "validationEvaluatedActionCount": sum(
                item.evaluated_action_count for item in validation_labels.values()
            ),
            "validationFeasibleActionCount": sum(
                item.feasible_action_count for item in validation_labels.values()
            ),
        },
        "trainingMetrics": metrics,
        "evaluation": evaluation,
        "safetyBoundary": (
            "RL emits local priority prefixes only; every reward and deployed order "
            "is evaluated by the unchanged SIPP/reservation/validator pipeline."
        ),
    }
    write_json(output_dir / "training-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
