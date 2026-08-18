from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical

from .assignment import AssignmentProposal
from .domain import LoadState, TaskState, TransportTask, Vehicle
from .reservations import ReservationTable


CHECKPOINT_VERSION = 2
OBSERVATION_VERSION = 1
ACTION_MODE = "local_priority_prefix"
REWARD_VERSION = 2
BASE_FEATURE_DIM = 20
PATH_FEATURE_DIM = 10


@dataclass(frozen=True)
class EncodedPriorityObservation:
    agent_features: np.ndarray
    path_tokens: np.ndarray
    path_mask: np.ndarray
    mask: np.ndarray
    vehicle_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    conflict_matrix: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        conflict_matrix = self.conflict_matrix
        if conflict_matrix is None:
            conflict_matrix = np.diag(self.mask.astype(np.int8))
        return {
            "agent_features": self.agent_features,
            "path_tokens": self.path_tokens,
            "path_mask": self.path_mask,
            "mask": self.mask,
            "conflict_matrix": conflict_matrix,
        }

    @property
    def candidate_count(self) -> int:
        return len(self.vehicle_ids)


class PriorityObservationEncoder:
    """Encode transferable graph/path features without absolute node embeddings."""

    def __init__(
        self,
        topology: Any,
        routes: Any,
        *,
        planning_horizon_ms: int,
        max_candidates: int = 64,
        path_token_count: int = 24,
    ) -> None:
        if max_candidates <= 0 or path_token_count <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.topology = topology
        self.routes = routes
        self.planning_horizon_ms = max(1, int(planning_horizon_ms))
        self.max_candidates = int(max_candidates)
        self.path_token_count = int(path_token_count)
        coordinates = [
            (float(node.get("x", 0.0)), float(node.get("y", 0.0)))
            for node in topology.nodes.values()
        ]
        xs = [item[0] for item in coordinates] or [0.0]
        ys = [item[1] for item in coordinates] or [0.0]
        self.map_scale_m = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        self._node_degree = self._build_node_degree()

    @property
    def config(self) -> dict[str, int]:
        return {
            "planning_horizon_ms": self.planning_horizon_ms,
            "max_candidates": self.max_candidates,
            "path_token_count": self.path_token_count,
            "base_feature_dim": BASE_FEATURE_DIM,
            "path_feature_dim": PATH_FEATURE_DIM,
        }

    def _build_node_degree(self) -> dict[str, int]:
        degree: dict[str, int] = {}
        for edge in self.routes.edges.values():
            degree[edge["start"]] = degree.get(edge["start"], 0) + 1
            degree[edge["end"]] = degree.get(edge["end"], 0) + 1
        return degree

    def encode(
        self,
        proposals: Sequence[AssignmentProposal],
        tasks_by_id: dict[str, TransportTask],
        projections: dict[str, Vehicle],
        reservations: ReservationTable,
        now_ms: int,
        *,
        priority_age_ms: dict[str, int] | None = None,
    ) -> EncodedPriorityObservation:
        proposals = tuple(proposals)
        if len(proposals) > self.max_candidates:
            raise ValueError(
                f"{len(proposals)} candidates exceed checkpoint limit "
                f"{self.max_candidates}"
            )
        features = np.zeros((self.max_candidates, BASE_FEATURE_DIM), dtype=np.float32)
        tokens = np.zeros(
            (self.max_candidates, self.path_token_count, PATH_FEATURE_DIM),
            dtype=np.float32,
        )
        path_mask = np.zeros(
            (self.max_candidates, self.path_token_count), dtype=np.int8
        )
        mask = np.zeros((self.max_candidates,), dtype=np.int8)
        conflict_matrix = np.zeros(
            (self.max_candidates, self.max_candidates), dtype=np.int8
        )
        vehicle_ids: list[str] = []
        task_ids: list[str] = []
        candidate_resources: list[set[str]] = []
        ages = priority_age_ms or {}
        snapshot = reservations.snapshot()

        for index, proposal in enumerate(proposals):
            vehicle = projections[proposal.vehicle_id]
            task = tasks_by_id[proposal.task_id]
            route_parts = self._routes_for(vehicle, task)
            flattened_edges = [
                (edge_id, load_state, phase)
                for phase, load_state, edge_ids in route_parts
                for edge_id in edge_ids
            ]
            resources = self._route_resources(item[0] for item in flattened_edges)
            for node_id in (task.pickup_node_id, task.dropoff_node_id):
                station = self.topology.workstations.get(node_id)
                if station is not None:
                    resources.add(f"workstation:{station.station_id}")
                    if station.blocks_transit_during_service:
                        resources.add(f"node:{node_id}")
                resources.update(
                    self.topology.traffic_zones.resource_ids_for_node(node_id)
                )
            candidate_resources.append(resources)
            occupied_ms, blocking_vehicles = self._reservation_pressure(
                resources, snapshot, now_ms, vehicle.vehicle_id
            )
            features[index] = self._base_features(
                proposal,
                vehicle,
                task,
                now_ms,
                occupied_ms,
                blocking_vehicles,
                len(flattened_edges),
                int(ages.get(vehicle.vehicle_id, 0)),
            )
            for token_index, (edge_id, load_state, phase) in enumerate(
                flattened_edges[: self.path_token_count]
            ):
                tokens[index, token_index] = self._path_token(
                    edge_id, load_state, phase, vehicle.heading_rad, token_index
                )
                path_mask[index, token_index] = 1
            mask[index] = 1
            vehicle_ids.append(vehicle.vehicle_id)
            task_ids.append(task.task_id)

        for left, left_resources in enumerate(candidate_resources):
            conflict_matrix[left, left] = 1
            for right in range(left + 1, len(candidate_resources)):
                if left_resources & candidate_resources[right]:
                    conflict_matrix[left, right] = 1
                    conflict_matrix[right, left] = 1

        return EncodedPriorityObservation(
            agent_features=features,
            path_tokens=tokens,
            path_mask=path_mask,
            mask=mask,
            vehicle_ids=tuple(vehicle_ids),
            task_ids=tuple(task_ids),
            conflict_matrix=conflict_matrix,
        )

    def _routes_for(
        self, vehicle: Vehicle, task: TransportTask
    ) -> tuple[tuple[float, LoadState, tuple[str, ...]], ...]:
        if vehicle.current_node_id is None:
            return ()
        if task.state is TaskState.EN_ROUTE_DROPOFF:
            loaded = self.routes.candidate_routes(
                task.required_robot_group,
                vehicle.current_node_id,
                task.dropoff_node_id,
                LoadState.LOADED,
                limit=1,
            )
            return (
                (1.0, LoadState.LOADED, loaded[0].edge_ids if loaded else ()),
            )
        empty = self.routes.candidate_routes(
            vehicle.robot_group,
            vehicle.current_node_id,
            task.pickup_node_id,
            LoadState.EMPTY,
            limit=1,
        )
        loaded = self.routes.candidate_routes(
            task.required_robot_group,
            task.pickup_node_id,
            task.dropoff_node_id,
            LoadState.LOADED,
            limit=1,
        )
        return (
            (0.0, LoadState.EMPTY, empty[0].edge_ids if empty else ()),
            (1.0, LoadState.LOADED, loaded[0].edge_ids if loaded else ()),
        )

    def _route_resources(self, edge_ids: Any) -> set[str]:
        resources: set[str] = set()
        for edge_id in edge_ids:
            if edge_id not in self.topology.edge_resources:
                continue
            resources.update(
                self.topology.prospective_motion_resources_for_edge(edge_id)
            )
            edge = self.routes.edges[edge_id]
            resources.add(f"node:{edge['start']}")
            resources.add(f"node:{edge['end']}")
            resources.update(
                self.topology.traffic_zones.resource_ids_for_edge(edge_id)
            )
        return resources

    def _reservation_pressure(
        self,
        resource_ids: set[str],
        reservations: Sequence[Any],
        now_ms: int,
        vehicle_id: str,
    ) -> tuple[int, int]:
        end_ms = now_ms + self.planning_horizon_ms
        occupied_ms = 0
        blockers: set[str] = set()
        for item in reservations:
            if item.resource_id not in resource_ids or item.vehicle_id == vehicle_id:
                continue
            overlap = max(0, min(item.end_ms, end_ms) - max(item.start_ms, now_ms))
            if overlap:
                occupied_ms += overlap
                blockers.add(item.vehicle_id)
        return occupied_ms, len(blockers)

    def _base_features(
        self,
        proposal: AssignmentProposal,
        vehicle: Vehicle,
        task: TransportTask,
        now_ms: int,
        occupied_ms: int,
        blocking_vehicles: int,
        route_edge_count: int,
        priority_age_ms: int,
    ) -> np.ndarray:
        horizon = float(self.planning_horizon_ms)
        due_slack = (
            (task.due_time_ms - now_ms) / horizon
            if task.due_time_ms is not None
            else 1.0
        )
        group_code = sum(ord(value) for value in vehicle.robot_group) % 31 / 30.0
        node_degree = self._node_degree.get(vehicle.current_node_id or "", 0)
        state_values = list(type(vehicle.state))
        state_index = state_values.index(vehicle.state) / max(1, len(state_values) - 1)
        cost = proposal.cost
        return np.asarray(
            [
                group_code,
                1.0 if vehicle.load_state is LoadState.LOADED else 0.0,
                state_index,
                math.sin(vehicle.heading_rad),
                math.cos(vehicle.heading_rad),
                np.clip((vehicle.available_at_ms - now_ms) / horizon, -2.0, 2.0),
                np.clip((now_ms - task.release_time_ms) / horizon, 0.0, 10.0),
                np.clip(task.priority_class / 10.0, -1.0, 1.0),
                np.clip(due_slack, -10.0, 10.0),
                1.0 if task.due_time_ms is not None else 0.0,
                np.clip(cost.empty_travel_ms / horizon, 0.0, 10.0),
                np.clip(cost.loaded_travel_ms / horizon, 0.0, 10.0),
                np.clip(cost.pickup_service_ms / horizon, 0.0, 10.0),
                np.clip(cost.dropoff_service_ms / horizon, 0.0, 10.0),
                np.clip(cost.total_ms / horizon, -10.0, 10.0),
                np.clip(occupied_ms / horizon, 0.0, 20.0),
                np.clip(blocking_vehicles / 16.0, 0.0, 1.0),
                np.clip(priority_age_ms / horizon, 0.0, 20.0),
                np.clip(node_degree / 12.0, 0.0, 1.0),
                np.clip(route_edge_count / self.path_token_count, 0.0, 4.0),
            ],
            dtype=np.float32,
        )

    def _path_token(
        self,
        edge_id: str,
        load_state: LoadState,
        phase: float,
        heading_rad: float,
        token_index: int,
    ) -> np.ndarray:
        edge = self.routes.edges[edge_id]
        start = self.topology.nodes[edge["start"]]
        end = self.topology.nodes[edge["end"]]
        dx = float(end.get("x", 0.0)) - float(start.get("x", 0.0))
        dy = float(end.get("y", 0.0)) - float(start.get("y", 0.0))
        angle = math.atan2(dy, dx)
        conflicts = self.topology.prospective_motion_resources_for_edge(edge_id)
        duration_ms = self.routes.travel_times.duration_ms(edge, load_state)
        narrow = self.topology.traffic_zones.zone_for_edge(edge_id) is not None
        return np.asarray(
            [
                phase,
                np.clip(float(edge.get("length", 0.0)) / 20.0, 0.0, 5.0),
                np.clip(duration_ms / self.planning_horizon_ms, 0.0, 5.0),
                np.clip((1 + len(conflicts)) / 64.0, 0.0, 1.0),
                np.clip(len(conflicts) / 64.0, 0.0, 1.0),
                np.clip(dx / self.map_scale_m, -1.0, 1.0),
                np.clip(dy / self.map_scale_m, -1.0, 1.0),
                math.cos(angle - heading_rad),
                math.sin(angle - heading_rad),
                1.0 if narrow else 0.0,
            ],
            dtype=np.float32,
        )


class PriorityOrderNetwork(nn.Module):
    """Temporal path encoder plus spatial attention and pointer decoder."""

    def __init__(
        self,
        *,
        max_candidates: int = 64,
        path_token_count: int = 24,
        base_feature_dim: int = BASE_FEATURE_DIM,
        path_feature_dim: int = PATH_FEATURE_DIM,
        hidden_dim: int = 96,
        attention_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.config = {
            "max_candidates": int(max_candidates),
            "path_token_count": int(path_token_count),
            "base_feature_dim": int(base_feature_dim),
            "path_feature_dim": int(path_feature_dim),
            "hidden_dim": int(hidden_dim),
            "attention_heads": int(attention_heads),
            "transformer_layers": int(transformer_layers),
        }
        self.path_projection = nn.Linear(path_feature_dim, hidden_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.agent_projection = nn.Linear(base_feature_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_score = nn.Linear(hidden_dim, 1, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def _encode(
        self,
        agent_features: torch.Tensor,
        path_tokens: torch.Tensor,
        path_mask: torch.Tensor,
        mask: torch.Tensor,
        conflict_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        path_mask = path_mask.bool()
        batch, candidates, token_count, _ = path_tokens.shape
        flat_paths = path_tokens.reshape(batch * candidates, token_count, -1)
        flat_path_mask = path_mask.reshape(batch * candidates, token_count)
        path_input = self.path_projection(flat_paths)
        safe_path_mask = flat_path_mask.clone()
        empty_rows = ~safe_path_mask.any(dim=1)
        safe_path_mask[empty_rows, 0] = True
        encoded_path = self.temporal_encoder(
            path_input, src_key_padding_mask=~safe_path_mask
        )
        weights = flat_path_mask.to(encoded_path.dtype).unsqueeze(-1)
        path_summary = (encoded_path * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        path_summary = path_summary.reshape(batch, candidates, -1)
        fused = self.fusion(
            torch.cat((self.agent_projection(agent_features), path_summary), dim=-1)
        )
        if conflict_matrix is not None:
            adjacency = conflict_matrix.to(fused.dtype)
            valid_pairs = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjacency = adjacency * valid_pairs.to(adjacency.dtype)
            identity = torch.eye(
                candidates, dtype=adjacency.dtype, device=adjacency.device
            ).unsqueeze(0)
            adjacency = torch.maximum(adjacency, identity * mask.unsqueeze(1))
            graph_message = torch.bmm(adjacency, fused) / adjacency.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            fused = 0.5 * (fused + graph_message)
        memory = self.spatial_encoder(fused, src_key_padding_mask=~mask)
        mask_weights = mask.to(memory.dtype).unsqueeze(-1)
        context = (memory * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp_min(1.0)
        return memory, context

    def _decode(
        self,
        memory: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        *,
        actions: torch.Tensor | None,
        deterministic: bool,
        max_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        batch, candidates, hidden = memory.shape
        counts = mask.sum(dim=1)
        selected = ~mask.clone()
        sequence = torch.full(
            (batch, candidates), -1, dtype=torch.long, device=memory.device
        )
        log_prob = torch.zeros(batch, device=memory.device)
        entropy = torch.zeros(batch, device=memory.device)
        state = context
        decoder_input = self.start_token.unsqueeze(0).expand(batch, hidden)
        keys = self.pointer_key(memory)

        step_count = candidates if max_steps is None else min(candidates, max(1, max_steps))
        for step in range(step_count):
            active = counts > step
            if not bool(active.any()):
                break
            state = self.decoder_cell(decoder_input, state)
            logits = self.pointer_score(
                torch.tanh(keys + self.pointer_query(state).unsqueeze(1))
            ).squeeze(-1)
            logits = logits.masked_fill(selected, -1.0e9)
            logits = torch.where(active.unsqueeze(1), logits, torch.zeros_like(logits))
            distribution = Categorical(logits=logits)
            if actions is None:
                choice = logits.argmax(dim=1) if deterministic else distribution.sample()
            else:
                choice = actions[:, step].clamp(min=0)
                invalid = active & (selected.gather(1, choice.unsqueeze(1)).squeeze(1))
                if bool(invalid.any()):
                    raise ValueError("actions must be a permutation of valid candidates")
            sequence[:, step] = torch.where(active, choice, sequence[:, step])
            log_prob += torch.where(active, distribution.log_prob(choice), 0.0)
            entropy += torch.where(active, distribution.entropy(), 0.0)
            chosen_memory = memory.gather(
                1, choice.view(batch, 1, 1).expand(batch, 1, hidden)
            ).squeeze(1)
            decoder_input = torch.where(active.unsqueeze(1), chosen_memory, decoder_input)
            chosen_mask = torch.zeros_like(selected)
            chosen_mask.scatter_(1, choice.unsqueeze(1), True)
            selected = selected | (chosen_mask & active.unsqueeze(1))
        return sequence, log_prob, entropy

    def act(
        self,
        observation: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        max_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, context = self._encode(**observation)
        sequence, log_prob, _ = self._decode(
            memory,
            context,
            observation["mask"],
            actions=None,
            deterministic=deterministic,
            max_steps=max_steps,
        )
        return sequence, log_prob, self.value_head(context).squeeze(-1)

    def evaluate_actions(
        self,
        observation: dict[str, torch.Tensor],
        actions: torch.Tensor,
        *,
        max_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, context = self._encode(**observation)
        _, log_prob, entropy = self._decode(
            memory,
            context,
            observation["mask"],
            actions=actions,
            deterministic=False,
            max_steps=max_steps,
        )
        return log_prob, entropy, self.value_head(context).squeeze(-1)


RewardEvaluator = Callable[[tuple[int, ...]], float | tuple[float, dict[str, Any]]]


@dataclass(frozen=True)
class PriorityTrainingCase:
    observation: EncodedPriorityObservation
    evaluator: RewardEvaluator
    case_id: str = "case"
    reference_action: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PriorityOracleLabel:
    action: tuple[int, ...]
    reward: float
    info: dict[str, Any]
    baseline_reward: float
    baseline_info: dict[str, Any]
    evaluated_action_count: int
    feasible_action_count: int

    @property
    def feasible(self) -> bool:
        return bool(self.info.get("feasible", False))


def _oracle_ordering_key(
    reward: float,
    info: dict[str, Any],
    order: tuple[int, ...],
) -> tuple[Any, ...]:
    score = info.get("score") or {}
    if bool(info.get("feasible", False)):
        return (
            0,
            -int(score.get("projectedDropoffs", 0)),
            -int(score.get("projectedPickups", 0)),
            int(score.get("latenessMs", 0)),
            int(score.get("queueTimeMs", 0)),
            int(score.get("waitMs", 0)),
            int(score.get("emptyTravelMs", 0)),
            int(score.get("completionTimeSumMs", 0)),
            int(info.get("scheduleAttempts", 0)),
            int(info.get("routeCombinationsTried", 0)),
            order,
        )
    return (
        1,
        -int(info.get("plannedTaskCount", 0)),
        -float(reward),
        int(info.get("scheduleAttempts", 0)),
        int(info.get("routeCombinationsTried", 0)),
        order,
    )


def label_oracle_prefix(
    case: PriorityTrainingCase,
    *,
    prefix_count: int,
    max_evaluations: int = 720,
) -> PriorityOracleLabel:
    """Enumerate a small local prefix and label it with the real safe evaluator."""

    count = case.observation.candidate_count
    active_prefix = min(count, max(1, int(prefix_count)))
    evaluation_count = math.perm(count, active_prefix)
    if evaluation_count > max_evaluations:
        raise ValueError(
            f"oracle prefix needs {evaluation_count} evaluations; "
            f"limit is {max_evaluations}"
        )

    evaluated: list[tuple[tuple[int, ...], float, dict[str, Any]]] = []
    baseline_order = tuple(range(count))
    baseline_result: tuple[float, dict[str, Any]] | None = None
    for prefix in permutations(range(count), active_prefix):
        prefix_set = set(prefix)
        order = tuple(prefix) + tuple(
            index for index in range(count) if index not in prefix_set
        )
        result = case.evaluator(order)
        if isinstance(result, tuple):
            reward, info = float(result[0]), dict(result[1])
        else:
            reward, info = float(result), {}
        evaluated.append((order, reward, info))
        if order == baseline_order:
            baseline_result = (reward, info)

    if baseline_result is None:
        raise RuntimeError("oracle enumeration did not evaluate the baseline order")
    feasible = [item for item in evaluated if bool(item[2].get("feasible", False))]
    candidates = feasible or evaluated
    action, reward, info = min(
        candidates,
        key=lambda item: _oracle_ordering_key(item[1], item[2], item[0]),
    )
    return PriorityOracleLabel(
        action=action,
        reward=reward,
        info=info,
        baseline_reward=baseline_result[0],
        baseline_info=baseline_result[1],
        evaluated_action_count=len(evaluated),
        feasible_action_count=len(feasible),
    )


class PriorityOrderEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """A safe evaluator for a learned local priority prefix."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cases: Sequence[PriorityTrainingCase],
        *,
        seed: int = 0,
        prefix_count: int | None = None,
    ) -> None:
        super().__init__()
        if not cases:
            raise ValueError("PriorityOrderEnv needs at least one training case")
        self.cases = tuple(cases)
        first = self.cases[0].observation
        max_candidates = first.mask.shape[0]
        for case in self.cases:
            observation = case.observation
            if observation.agent_features.shape != first.agent_features.shape:
                raise ValueError("all cases must use the same encoder dimensions")
        self.observation_space = spaces.Dict(
            {
                "agent_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=first.agent_features.shape,
                    dtype=np.float32,
                ),
                "path_tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=first.path_tokens.shape,
                    dtype=np.float32,
                ),
                "path_mask": spaces.MultiBinary(first.path_mask.shape),
                "mask": spaces.MultiBinary(first.mask.shape),
                "conflict_matrix": spaces.MultiBinary(
                    first.as_dict()["conflict_matrix"].shape
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(max_candidates, max_candidates, dtype=np.int64)
        )
        self._seed = int(seed)
        self.prefix_count = (
            max_candidates
            if prefix_count is None
            else max(1, min(max_candidates, int(prefix_count)))
        )
        self._current: PriorityTrainingCase | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=self._seed if seed is None else seed)
        requested = None if options is None else options.get("case_index")
        index = (
            int(requested) % len(self.cases)
            if requested is not None
            else int(self.np_random.integers(len(self.cases)))
        )
        self._current = self.cases[index]
        return self._current.observation.as_dict(), {"caseId": self._current.case_id}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._current is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        count = self._current.observation.candidate_count
        prefix_count = min(count, self.prefix_count)
        prefix = tuple(int(value) for value in action[:prefix_count])
        if (
            len(prefix) != prefix_count
            or len(set(prefix)) != prefix_count
            or any(value < 0 or value >= count for value in prefix)
        ):
            reward = -100.0
            info: dict[str, Any] = {"invalidPermutation": True}
        else:
            order = prefix + tuple(
                index for index in range(count) if index not in set(prefix)
            )
            evaluated = self._current.evaluator(order)
            if isinstance(evaluated, tuple):
                reward, info = float(evaluated[0]), dict(evaluated[1])
            else:
                reward, info = float(evaluated), {}
            info["invalidPermutation"] = False
            info["priorityPrefixCount"] = prefix_count
        observation = self._current.observation.as_dict()
        info["caseId"] = self._current.case_id
        self._current = None
        return observation, reward, True, False, info


def _tensor_observation(
    observation: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "agent_features": torch.as_tensor(
            observation["agent_features"], dtype=torch.float32, device=device
        ),
        "path_tokens": torch.as_tensor(
            observation["path_tokens"], dtype=torch.float32, device=device
        ),
        "path_mask": torch.as_tensor(
            observation["path_mask"], dtype=torch.bool, device=device
        ),
        "mask": torch.as_tensor(observation["mask"], dtype=torch.bool, device=device),
        "conflict_matrix": torch.as_tensor(
            observation.get(
                "conflict_matrix",
                np.diag(np.asarray(observation["mask"], dtype=np.int8)),
            ),
            dtype=torch.bool,
            device=device,
        ),
    }


class PPOPriorityTrainer:
    def __init__(
        self,
        env: PriorityOrderEnv,
        network: PriorityOrderNetwork,
        *,
        learning_rate: float = 3.0e-4,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.network = network
        self.device = torch.device(device)
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)

    def behavior_clone(
        self,
        cases: Sequence[PriorityTrainingCase],
        *,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float | None = None,
    ) -> list[float]:
        """Warm-start the pointer decoder from a safe deterministic priority baseline."""

        references = [
            case for case in cases if case.reference_action is not None
        ]
        if epochs <= 0 or not references:
            return []
        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = torch.optim.Adam(
                self.network.parameters(), lr=float(learning_rate)
            )
        losses: list[float] = []
        self.network.train()
        for _ in range(epochs):
            order = np.random.permutation(len(references))
            for start in range(0, len(order), max(1, batch_size)):
                selected = [references[int(index)] for index in order[start : start + batch_size]]
                observations = [
                    _tensor_observation(case.observation.as_dict(), self.device)
                    for case in selected
                ]
                batch_observation = {
                    key: torch.stack([item[key] for item in observations])
                    for key in observations[0]
                }
                max_candidates = batch_observation["mask"].shape[1]
                actions = torch.full(
                    (len(selected), max_candidates),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
                for row, case in enumerate(selected):
                    reference = case.reference_action or ()
                    actions[row, : len(reference)] = torch.as_tensor(
                        reference, dtype=torch.long, device=self.device
                    )
                log_prob, _, _ = self.network.evaluate_actions(
                    batch_observation,
                    actions,
                    max_steps=getattr(getattr(self, "env", None), "prefix_count", None),
                )
                loss = -log_prob.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        self.network.eval()
        return losses

    def train(
        self,
        total_steps: int,
        *,
        rollout_steps: int = 64,
        update_epochs: int = 4,
        batch_size: int = 32,
        seed: int = 0,
    ) -> list[dict[str, float]]:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        np.random.seed(seed)
        torch.manual_seed(seed)
        metrics: list[dict[str, float]] = []
        completed = 0
        update_index = 0
        while completed < total_steps:
            count = min(rollout_steps, total_steps - completed)
            observations: list[dict[str, np.ndarray]] = []
            actions: list[np.ndarray] = []
            old_log_probs: list[float] = []
            values: list[float] = []
            rewards: list[float] = []
            self.network.eval()
            for offset in range(count):
                observation, _ = self.env.reset(seed=seed + completed + offset)
                tensor_obs = {
                    key: value.unsqueeze(0)
                    for key, value in _tensor_observation(
                        observation, self.device
                    ).items()
                }
                with torch.no_grad():
                    action, log_prob, value = self.network.act(
                        tensor_obs, max_steps=self.env.prefix_count
                    )
                action_np = action[0].cpu().numpy()
                _, reward, _, _, _ = self.env.step(action_np)
                observations.append(observation)
                actions.append(action_np)
                old_log_probs.append(float(log_prob.item()))
                values.append(float(value.item()))
                rewards.append(float(reward))

            batch_observation = {
                key: torch.stack(
                    [
                        _tensor_observation(item, self.device)[key]
                        for item in observations
                    ]
                )
                for key in observations[0]
            }
            batch_actions = torch.as_tensor(
                np.stack(actions), dtype=torch.long, device=self.device
            )
            batch_old_log_probs = torch.as_tensor(
                old_log_probs, dtype=torch.float32, device=self.device
            )
            returns = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            advantages = returns - torch.as_tensor(
                values, dtype=torch.float32, device=self.device
            )
            if advantages.numel() > 1 and float(advantages.std()) > 1.0e-8:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1.0e-8
                )

            self.network.train()
            losses: list[float] = []
            indices = np.arange(count)
            for _ in range(update_epochs):
                np.random.shuffle(indices)
                for start in range(0, count, batch_size):
                    selected = torch.as_tensor(
                        indices[start : start + batch_size],
                        dtype=torch.long,
                        device=self.device,
                    )
                    obs = {
                        key: value.index_select(0, selected)
                        for key, value in batch_observation.items()
                    }
                    log_prob, entropy, predicted = self.network.evaluate_actions(
                        obs,
                        batch_actions.index_select(0, selected),
                        max_steps=self.env.prefix_count,
                    )
                    old = batch_old_log_probs.index_select(0, selected)
                    advantage = advantages.index_select(0, selected)
                    ratio = torch.exp(log_prob - old)
                    policy_loss = -torch.min(
                        ratio * advantage,
                        torch.clamp(
                            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                        )
                        * advantage,
                    ).mean()
                    value_loss = torch.square(
                        predicted - returns.index_select(0, selected)
                    ).mean()
                    loss = (
                        policy_loss
                        + self.value_coefficient * value_loss
                        - self.entropy_coefficient * entropy.mean()
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                    self.optimizer.step()
                    losses.append(float(loss.detach().cpu()))

            completed += count
            update_index += 1
            metrics.append(
                {
                    "update": float(update_index),
                    "steps": float(completed),
                    "meanReward": float(np.mean(rewards)),
                    "minReward": float(np.min(rewards)),
                    "maxReward": float(np.max(rewards)),
                    "meanLoss": float(np.mean(losses)) if losses else 0.0,
                }
            )
        self.network.eval()
        return metrics

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        encoder_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = dict(metadata or {})
        compatibility = {
            "observation_version": OBSERVATION_VERSION,
            "action_mode": ACTION_MODE,
            "reward_version": REWARD_VERSION,
        }
        for key, expected in compatibility.items():
            configured = checkpoint_metadata.get(key, expected)
            if configured != expected:
                raise ValueError(
                    f"checkpoint metadata {key}={configured!r} is incompatible "
                    f"with {expected!r}"
                )
            checkpoint_metadata[key] = expected
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "network_config": self.network.config,
                "encoder_config": dict(encoder_config),
                "state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": checkpoint_metadata,
            },
            path,
        )
        return path


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported RL priority checkpoint version")
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "observation_version": OBSERVATION_VERSION,
        "action_mode": ACTION_MODE,
        "reward_version": REWARD_VERSION,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"RL priority checkpoint {key} is incompatible: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    return payload


class RLPriorityPolicy:
    """Read-only priority plugin. It cannot access or mutate scheduler state."""

    def __init__(
        self,
        network: PriorityOrderNetwork,
        encoder: PriorityObservationEncoder,
        *,
        device: str | torch.device = "cpu",
        candidate_count: int = 1,
        prefix_count: int | None = None,
        seed: int = 0,
        torch_threads: int | None = None,
    ) -> None:
        self.network = network
        self.encoder = encoder
        self.device = torch.device(device)
        if self.device.type == "cpu":
            configured_threads = torch_threads
            if configured_threads is None:
                configured_threads = int(os.environ.get("MASP_RL_TORCH_THREADS", "1"))
            if configured_threads > 0:
                torch.set_num_threads(int(configured_threads))
        self.network.to(self.device).eval()
        self.candidate_count = max(1, int(candidate_count))
        self.prefix_count = (
            None if prefix_count is None else max(1, int(prefix_count))
        )
        self.seed = int(seed)
        self._call_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        topology: Any,
        routes: Any,
        planning_horizon_ms: int,
        device: str | torch.device = "cpu",
        candidate_count: int | None = None,
        prefix_count: int | None = None,
        seed: int = 0,
    ) -> RLPriorityPolicy:
        payload = load_checkpoint(path, device=device)
        network = PriorityOrderNetwork(**payload["network_config"])
        network.load_state_dict(payload["state_dict"])
        encoder_config = dict(payload["encoder_config"])
        trained_horizon_ms = int(
            encoder_config.get("planning_horizon_ms", planning_horizon_ms)
        )
        if trained_horizon_ms != int(planning_horizon_ms):
            raise ValueError(
                "RL priority checkpoint planning horizon does not match runtime"
            )
        encoder = PriorityObservationEncoder(
            topology,
            routes,
            planning_horizon_ms=planning_horizon_ms,
            max_candidates=int(encoder_config["max_candidates"]),
            path_token_count=int(encoder_config["path_token_count"]),
        )
        checkpoint_candidates = int(payload.get("metadata", {}).get("candidate_count", 1))
        checkpoint_prefix = payload.get("metadata", {}).get("priority_prefix_count")
        if checkpoint_prefix is None:
            raise ValueError("RL priority checkpoint is missing priority_prefix_count")
        requested_prefix = (
            int(checkpoint_prefix) if prefix_count is None else int(prefix_count)
        )
        if requested_prefix != int(checkpoint_prefix):
            raise ValueError(
                "RL priority checkpoint prefix length does not match runtime"
            )
        return cls(
            network,
            encoder,
            device=device,
            candidate_count=(
                checkpoint_candidates if candidate_count is None else candidate_count
            ),
            prefix_count=requested_prefix,
            seed=seed,
        )

    def priority_orders(
        self,
        *,
        proposals: Sequence[AssignmentProposal],
        tasks_by_id: dict[str, TransportTask],
        projections: dict[str, Vehicle],
        reservations: ReservationTable,
        now_ms: int,
        priority_age_ms: dict[str, int] | None = None,
        count: int | None = None,
        prefix_count: int | None = None,
    ) -> tuple[tuple[AssignmentProposal, ...], ...]:
        proposals = tuple(proposals)
        if len(proposals) <= 1:
            return (proposals,)
        observation = self.encoder.encode(
            proposals,
            tasks_by_id,
            projections,
            reservations,
            now_ms,
            priority_age_ms=priority_age_ms,
        )
        tensor_obs = {
            key: value.unsqueeze(0)
            for key, value in _tensor_observation(
                observation.as_dict(), self.device
            ).items()
        }
        requested = max(1, int(self.candidate_count if count is None else count))
        requested_prefix = min(
            len(proposals),
            max(
                1,
                int(
                    len(proposals)
                    if prefix_count is None and self.prefix_count is None
                    else (
                        self.prefix_count if prefix_count is None else prefix_count
                    )
                ),
            ),
        )
        generated: list[tuple[AssignmentProposal, ...]] = []
        signatures: set[tuple[tuple[str, str], ...]] = set()
        devices = [] if self.device.type == "cpu" else [self.device.index or 0]
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self._call_count * 104729)
            for index in range(max(requested * 5, 1)):
                action, _, _ = self.network.act(
                    tensor_obs,
                    deterministic=(index == 0),
                    max_steps=requested_prefix,
                )
                prefix_indices = tuple(
                    int(value)
                    for value in action[0, :requested_prefix].tolist()
                )
                if (
                    len(set(prefix_indices)) != requested_prefix
                    or any(value < 0 or value >= len(proposals) for value in prefix_indices)
                ):
                    continue
                remaining_indices = tuple(
                    value
                    for value in range(len(proposals))
                    if value not in set(prefix_indices)
                )
                order = tuple(
                    proposals[value]
                    for value in prefix_indices + remaining_indices
                )
                signature = tuple((item.vehicle_id, item.task_id) for item in order)
                if signature in signatures:
                    continue
                signatures.add(signature)
                generated.append(order)
                if len(generated) >= requested:
                    break
        self._call_count += 1
        if not generated:
            raise RuntimeError("RL decoder returned no valid priority order")
        return tuple(generated)


def reward_from_candidate(outcome: Any) -> tuple[float, dict[str, Any]]:
    """Dense PPO reward derived only after the safe candidate evaluator runs."""

    record = outcome.record
    schedule_attempts = sum(item.schedule_attempts for item in outcome.records)
    route_combinations = sum(
        item.route_combinations_tried for item in outcome.records
    )
    computation_penalty = 0.02 * schedule_attempts + 0.05 * route_combinations
    if not record.feasible or record.score is None:
        missing = max(0, len(record.order) - int(record.planned_task_count))
        return -20.0 - 5.0 * missing - computation_penalty, {
            "feasible": False,
            "plannedTaskCount": record.planned_task_count,
            "failureCode": record.failure_code,
            "scheduleAttempts": schedule_attempts,
            "routeCombinationsTried": route_combinations,
        }
    score = record.score
    reward = (
        20.0 * score.projected_dropoffs
        + 4.0 * score.projected_pickups
        + 2.0 * record.planned_task_count
        - score.lateness_ms / 10_000.0
        - score.queue_time_ms / 100_000.0
        - score.wait_ms / 20_000.0
        - score.empty_travel_ms / 100_000.0
        - score.completion_time_sum_ms / 1_000_000.0
        - computation_penalty
    )
    return float(reward), {
        "feasible": True,
        "plannedTaskCount": record.planned_task_count,
        "score": score.to_dict(),
        "scheduleAttempts": schedule_attempts,
        "routeCombinationsTried": route_combinations,
    }


def benchmark_inference_ms(
    policy: RLPriorityPolicy,
    kwargs: dict[str, Any],
    *,
    repeats: int = 10,
) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter_ns()
        policy.priority_orders(**kwargs)
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": float(np.mean(durations)),
        "p95": float(ordered[p95_index]),
        "max": float(max(durations)),
    }
