from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from masp.domain import TransportTask, Vehicle
from masp.coordination import RollingHorizonPlanner
from masp.reservations import ReservationTable
from masp.rl_priority import (
    EncodedPriorityObservation,
    PPOPriorityTrainer,
    PriorityObservationEncoder,
    PriorityOrderEnv,
    PriorityOrderNetwork,
    PriorityTrainingCase,
    load_checkpoint,
)
from masp.scenario import build_dispatch_plans, build_simulator
from masp.topology import MapTopology

from conftest import read_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rl_context():
    model = read_json("generated/xiate-unified-map-model.json")
    conflicts = read_json("generated/xiate-conflict-resources.json")
    workstations = read_json("generated/xiate-workstations.json")
    profiles = read_json("config/robot-profiles.json")
    scheduler = read_json("config/scheduler.json")
    zones = read_json("config/traffic-zones.json")
    scenario = read_json("scenarios/rolling-dispatch-benchmark.json")
    topology = MapTopology(model, conflicts, workstations, zones)
    planner = RollingHorizonPlanner(
        topology, model, profiles, scheduler, zones, policy="congestion", seed=0
    )
    vehicles = [Vehicle.from_dict(item) for item in scenario["vehicles"]]
    defaults = scheduler["serviceDefaults"]
    tasks = [
        TransportTask.from_dict(
            item,
            int(defaults["pickupServiceMs"]),
            int(defaults["dropoffServiceMs"]),
        )
        for item in scenario["tasks"]
    ]
    projections, tasks_by_id = planner._validate_inputs(vehicles, tasks)
    available_tasks = [task for task in tasks if task.release_time_ms == 0]
    proposals = planner.allocator.assign(list(projections.values()), available_tasks, 0)
    reservations = ReservationTable()
    reservations.insert_batch(
        planner._hold(
            vehicle,
            plan_id=f"test-idle:{vehicle.vehicle_id}",
            node_id=vehicle.current_node_id or "",
            start_ms=0,
            end_ms=int(scenario["endTimeMs"]),
            label="idle-tail",
        )
        for vehicle in projections.values()
    )
    return {
        "documents": (model, conflicts, workstations, profiles, scheduler, zones),
        "scenario": scenario,
        "planner": planner,
        "projections": projections,
        "tasks": tasks_by_id,
        "proposals": proposals,
        "reservations": reservations,
    }


def test_observation_encoder_emits_masked_graph_path_tokens(rl_context) -> None:
    planner = rl_context["planner"]
    encoder = PriorityObservationEncoder(
        planner.topology,
        planner.routes,
        planning_horizon_ms=planner.planning_horizon_ms,
        max_candidates=8,
        path_token_count=12,
    )
    observation = encoder.encode(
        rl_context["proposals"],
        rl_context["tasks"],
        rl_context["projections"],
        rl_context["reservations"],
        0,
    )

    assert observation.agent_features.shape == (8, 20)
    assert observation.path_tokens.shape == (8, 12, 10)
    assert observation.conflict_matrix is not None
    assert observation.conflict_matrix.shape == (8, 8)
    assert np.array_equal(
        observation.conflict_matrix, observation.conflict_matrix.T
    )
    assert observation.mask.sum() == len(rl_context["proposals"])
    assert observation.path_mask[: observation.candidate_count].sum() > 0
    assert not observation.mask[observation.candidate_count :].any()
    assert not observation.agent_features[observation.candidate_count :].any()


def test_pointer_decoder_returns_each_valid_candidate_once() -> None:
    network = PriorityOrderNetwork(
        max_candidates=6,
        path_token_count=4,
        hidden_dim=32,
        attention_heads=4,
        transformer_layers=1,
    )
    observation = {
        "agent_features": torch.randn(2, 6, 20),
        "path_tokens": torch.randn(2, 6, 4, 10),
        "path_mask": torch.tensor(
            [
                [[1, 1, 0, 0]] * 3 + [[0, 0, 0, 0]] * 3,
                [[1, 0, 0, 0]] * 5 + [[0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        ),
        "mask": torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool
        ),
    }

    action, log_prob, value = network.act(observation, deterministic=True)

    assert set(action[0, :3].tolist()) == {0, 1, 2}
    assert set(action[1, :5].tolist()) == {0, 1, 2, 3, 4}
    assert action[0, 3:].tolist() == [-1, -1, -1]
    assert action[1, 5:].tolist() == [-1]
    assert log_prob.shape == value.shape == (2,)


def test_local_priority_prefix_preserves_deterministic_baseline_tail() -> None:
    evaluated: list[tuple[int, ...]] = []
    observation = EncodedPriorityObservation(
        agent_features=np.zeros((4, 20), dtype=np.float32),
        path_tokens=np.zeros((4, 3, 10), dtype=np.float32),
        path_mask=np.ones((4, 3), dtype=np.int8),
        mask=np.ones((4,), dtype=np.int8),
        vehicle_ids=("v0", "v1", "v2", "v3"),
        task_ids=("t0", "t1", "t2", "t3"),
    )
    case = PriorityTrainingCase(
        observation=observation,
        evaluator=lambda order: evaluated.append(order) or 1.0,
    )
    env = PriorityOrderEnv([case], seed=0, prefix_count=2)
    env.reset(options={"case_index": 0})

    _, reward, terminated, truncated, info = env.step(
        np.asarray([2, 0, -1, -1], dtype=np.int64)
    )

    assert evaluated == [(2, 0, 1, 3)]
    assert reward == 1.0
    assert terminated and not truncated
    assert info["priorityPrefixCount"] == 2


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path: Path) -> None:
    mask = np.asarray([1, 1, 1, 0], dtype=np.int8)
    observation = {
        "agent_features": np.zeros((4, 20), dtype=np.float32),
        "path_tokens": np.zeros((4, 3, 10), dtype=np.float32),
        "path_mask": np.zeros((4, 3), dtype=np.int8),
        "mask": mask,
    }
    encoded = type("Observation", (), {"as_dict": lambda self: observation})()
    case = PriorityTrainingCase(
        observation=encoded,
        evaluator=lambda order: float(order[0] == 0),
    )
    env = PriorityOrderEnv.__new__(PriorityOrderEnv)
    # save_checkpoint only needs a trainer with an initialized optimizer.
    network = PriorityOrderNetwork(
        max_candidates=4,
        path_token_count=3,
        hidden_dim=32,
        attention_heads=4,
        transformer_layers=1,
    )
    trainer = PPOPriorityTrainer.__new__(PPOPriorityTrainer)
    trainer.network = network
    trainer.optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    checkpoint = PPOPriorityTrainer.save_checkpoint(
        trainer,
        tmp_path / "policy.pt",
        encoder_config={"max_candidates": 4, "path_token_count": 3},
    )
    payload = load_checkpoint(checkpoint)
    restored = PriorityOrderNetwork(**payload["network_config"])
    restored.load_state_dict(payload["state_dict"])
    tensor_observation = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in observation.items()
    }
    tensor_observation["agent_features"] = tensor_observation[
        "agent_features"
    ].float()
    tensor_observation["path_tokens"] = tensor_observation["path_tokens"].float()
    tensor_observation["path_mask"] = tensor_observation["path_mask"].bool()
    tensor_observation["mask"] = tensor_observation["mask"].bool()

    with torch.no_grad():
        before = network.act(tensor_observation, deterministic=True)[0]
        after = restored.act(tensor_observation, deterministic=True)[0]

    assert torch.equal(before, after)


def test_behavior_clone_handles_variable_candidate_counts() -> None:
    def make_case(count: int, reference: tuple[int, ...]) -> PriorityTrainingCase:
        observation = EncodedPriorityObservation(
            agent_features=np.zeros((4, 20), dtype=np.float32),
            path_tokens=np.zeros((4, 3, 10), dtype=np.float32),
            path_mask=np.asarray(
                [[1, 0, 0] if index < count else [0, 0, 0] for index in range(4)],
                dtype=np.int8,
            ),
            mask=np.asarray([1 if index < count else 0 for index in range(4)], dtype=np.int8),
            vehicle_ids=tuple(f"v{index}" for index in range(count)),
            task_ids=tuple(f"t{index}" for index in range(count)),
        )
        return PriorityTrainingCase(
            observation=observation,
            evaluator=lambda order: 0.0,
            reference_action=reference,
        )

    network = PriorityOrderNetwork(
        max_candidates=4,
        path_token_count=3,
        hidden_dim=32,
        attention_heads=4,
        transformer_layers=1,
    )
    trainer = PPOPriorityTrainer.__new__(PPOPriorityTrainer)
    trainer.network = network
    trainer.device = torch.device("cpu")
    trainer.optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)

    losses = trainer.behavior_clone(
        [make_case(2, (0, 1)), make_case(3, (2, 0, 1))], epochs=1, batch_size=2
    )

    assert losses and np.isfinite(losses[0])


def test_rl_policy_failure_falls_back_to_safe_congestion_planning(rl_context) -> None:
    class BrokenPolicy:
        def priority_orders(self, **kwargs):
            raise RuntimeError("forced inference failure")

    scenario = rl_context["scenario"]
    documents = rl_context["documents"]
    planning, planned = build_dispatch_plans(
        scenario,
        *documents,
        ROOT / "schemas",
        policy="rl",
        seed=0,
        priority_policy=BrokenPolicy(),
        rl_allow_deviation=True,
    )
    model, conflicts, workstations, _, scheduler, zones = documents
    simulation = build_simulator(
        planned,
        model,
        conflicts,
        workstations,
        scheduler,
        ROOT / "schemas",
        traffic_zones=zones,
    ).run()

    assert planning.unplanned_task_ids == ()
    assert planning.rl_inference_count > 0
    assert planning.rl_fallback_count == planning.rl_inference_count
    assert all(
        candidate.strategy == "congestion_fallback"
        for cycle in planning.cycles
        for candidate in cycle.candidates
    )
    assert simulation["metrics"]["completedTaskCount"] == len(scenario["tasks"])
    assert simulation["metrics"]["reservationConflictRejections"] == 0


def test_rl_orders_include_deterministic_guardian_candidate(rl_context) -> None:
    class ReversePolicy:
        candidate_count = 1

        def priority_orders(self, **kwargs):
            return (tuple(reversed(kwargs["proposals"])),)

    model, conflicts, workstations, profiles, scheduler, zones = rl_context[
        "documents"
    ]
    planner = RollingHorizonPlanner(
        MapTopology(model, conflicts, workstations, zones),
        model,
        profiles,
        scheduler,
        zones,
        policy="rl",
        priority_policy=ReversePolicy(),
        rl_allow_deviation=True,
    )
    orders = planner._priority_orders(
        rl_context["proposals"],
        rl_context["tasks"],
        rl_context["reservations"],
        rl_context["projections"],
        0,
        0,
        0,
    )

    assert orders[0][0] == "rl"
    assert any(strategy == "congestion_guardian" for strategy, _ in orders)
    assert planner.rl_guardian_candidate_count == 1
