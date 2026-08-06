from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable

from .assignment import AssignmentProposal
from .domain import (
    DomainError,
    LoadState,
    SegmentKind,
    TaskState,
    TransportTask,
    Vehicle,
    VehiclePlan,
)
from .phase2 import Phase2Planner, PlanningRecord
from .reservations import Reservation, ReservationConflict, ReservationTable
from .sipp import PlannedTask, SippPlanningError


class PriorityStrategy(str, Enum):
    TASK_AGE = "task_age"
    SHORTEST_REMAINING = "shortest_remaining"
    CONGESTION = "congestion"
    PREVIOUS_ORDER = "previous_order"
    RANDOM = "random"


@dataclass(frozen=True)
class CandidateScore:
    projected_dropoffs: int
    projected_pickups: int
    lateness_ms: int
    queue_time_ms: int
    wait_ms: int
    empty_travel_ms: int
    completion_time_sum_ms: int

    def ordering_key(self) -> tuple[int, ...]:
        # 安全可行性在进入评分前已作为硬约束处理。
        return (
            -self.projected_dropoffs,
            -self.projected_pickups,
            self.lateness_ms,
            self.queue_time_ms,
            self.wait_ms,
            self.empty_travel_ms,
            self.completion_time_sum_ms,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "projectedDropoffs": self.projected_dropoffs,
            "projectedPickups": self.projected_pickups,
            "latenessMs": self.lateness_ms,
            "queueTimeMs": self.queue_time_ms,
            "waitMs": self.wait_ms,
            "emptyTravelMs": self.empty_travel_ms,
            "completionTimeSumMs": self.completion_time_sum_ms,
        }


@dataclass(frozen=True)
class PriorityCandidateRecord:
    candidate_id: str
    strategy: str
    order: tuple[tuple[str, str], ...]
    feasible: bool
    planned_task_count: int
    score: CandidateScore | None
    failure_pair: tuple[str, str] | None = None
    failure_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidateId": self.candidate_id,
            "strategy": self.strategy,
            "order": [
                {"vehicleId": vehicle_id, "taskId": task_id}
                for vehicle_id, task_id in self.order
            ],
            "feasible": self.feasible,
            "plannedTaskCount": self.planned_task_count,
        }
        if self.score is not None:
            result["score"] = self.score.to_dict()
        if self.failure_pair is not None:
            result["failurePair"] = {
                "vehicleId": self.failure_pair[0],
                "taskId": self.failure_pair[1],
            }
        if self.failure_code is not None:
            result["failureCode"] = self.failure_code
        return result


@dataclass(frozen=True)
class SafeCommitment:
    vehicle_id: str
    task_id: str
    nominal_until_ms: int
    safe_until_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicleId": self.vehicle_id,
            "taskId": self.task_id,
            "nominalUntilMs": self.nominal_until_ms,
            "safeUntilMs": self.safe_until_ms,
            "extendedToSafeNode": self.safe_until_ms > self.nominal_until_ms,
        }


@dataclass(frozen=True)
class RollingCycleRecord:
    cycle_index: int
    decision_time_ms: int
    pending_task_count: int
    available_vehicle_count: int
    candidates: tuple[PriorityCandidateRecord, ...]
    selected_candidate_ids: tuple[str, ...]
    commitments: tuple[SafeCommitment, ...]
    planning_duration_ms: float

    def to_dict(self, include_timing: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cycleIndex": self.cycle_index,
            "decisionTimeMs": self.decision_time_ms,
            "pendingTaskCount": self.pending_task_count,
            "availableVehicleCount": self.available_vehicle_count,
            "candidateCount": len(self.candidates),
            "feasibleCandidateCount": sum(item.feasible for item in self.candidates),
            "selectedCandidateIds": list(self.selected_candidate_ids),
            "commitments": [item.to_dict() for item in self.commitments],
            "candidates": [item.to_dict() for item in self.candidates],
        }
        if include_timing:
            result["planningDurationMs"] = round(self.planning_duration_ms, 3)
        return result


@dataclass(frozen=True)
class Phase3PlanningResult:
    policy: str
    plans: tuple[VehiclePlan, ...]
    records: tuple[PlanningRecord, ...]
    cycles: tuple[RollingCycleRecord, ...]
    unplanned_task_ids: tuple[str, ...]
    reservation_conflict_rejections: int
    planning_period_ms: int
    planning_horizon_ms: int
    execution_horizon_ms: int
    planning_timeout_ms: int

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return round(ordered[index], 3)

    def summary(self, include_timing: bool = True) -> dict[str, Any]:
        durations = [item.planning_duration_ms for item in self.cycles]
        result: dict[str, Any] = {
            "policy": self.policy,
            "plannedTaskCount": len(self.plans),
            "unplannedTaskCount": len(self.unplanned_task_ids),
            "unplannedTaskIds": list(self.unplanned_task_ids),
            "planningPeriodMs": self.planning_period_ms,
            "planningHorizonMs": self.planning_horizon_ms,
            "executionHorizonMs": self.execution_horizon_ms,
            "planningCycleCount": len(self.cycles),
            "decisionCycleCount": sum(bool(item.candidates) for item in self.cycles),
            "priorityCandidatesEvaluated": sum(
                len(item.candidates) for item in self.cycles
            ),
            "feasiblePriorityCandidateCount": sum(
                candidate.feasible
                for cycle in self.cycles
                for candidate in cycle.candidates
            ),
            "insertedWaitMs": sum(item.inserted_wait_ms for item in self.records),
            "routeCombinationsTried": sum(
                item.route_combinations_tried for item in self.records
            ),
            "scheduleAttempts": sum(item.schedule_attempts for item in self.records),
            "reservationConflictRejections": self.reservation_conflict_rejections,
            "assignments": [
                {
                    "decisionTimeMs": item.decision_time_ms,
                    "vehicleId": item.vehicle_id,
                    "taskId": item.task_id,
                    "assignmentCostMs": item.assignment_cost_ms,
                    "completionTimeMs": item.completion_time_ms,
                    "insertedWaitMs": item.inserted_wait_ms,
                }
                for item in self.records
            ],
            "cycles": [item.to_dict(include_timing) for item in self.cycles],
        }
        if include_timing:
            result["planningLatencyMs"] = {
                "p50": self._percentile(durations, 0.50),
                "p95": self._percentile(durations, 0.95),
                "p99": self._percentile(durations, 0.99),
                "max": round(max(durations, default=0.0), 3),
            }
            result["planningTimeoutCount"] = sum(
                value > self.planning_timeout_ms for value in durations
            )
            result["planningPeriodMissCount"] = sum(
                value > self.planning_period_ms for value in durations
            )
        return result


@dataclass
class _CandidateOutcome:
    record: PriorityCandidateRecord
    projections: dict[str, Vehicle]
    reservations: ReservationTable
    plans: tuple[VehiclePlan, ...]
    records: tuple[PlanningRecord, ...]


class RollingHorizonPlanner(Phase2Planner):
    """周期性生成优先级候选，并以安全预留为硬约束进行 RH-PP 选优。"""

    def __init__(
        self,
        topology,
        model: dict[str, Any],
        profiles: dict[str, Any],
        scheduler: dict[str, Any],
        traffic_zones: dict[str, Any],
        *,
        policy: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(topology, model, profiles, scheduler, traffic_zones)
        planner = scheduler["planner"]
        coordination = scheduler["coordination"]
        self.planning_period_ms = int(planner["planningPeriodMs"])
        self.planning_horizon_ms = int(planner["planningHorizonMs"])
        self.execution_horizon_ms = int(planner["executionHorizonMs"])
        self.planning_timeout_ms = int(planner["planningTimeoutMs"])
        self.priority_candidate_count = int(coordination["priorityCandidateCount"])
        self.priority_strategies = tuple(
            PriorityStrategy(value) for value in coordination["priorityStrategies"]
        )
        self.policy = policy or str(coordination["defaultPolicy"])
        allowed_policies = {"top_k", *(item.value for item in PriorityStrategy)}
        if self.policy not in allowed_policies:
            raise ValueError(f"unknown phase 3 priority policy {self.policy!r}")
        self.seed = int(seed)
        self.previous_order: tuple[str, ...] = ()
        self.priority_age_ms: dict[str, int] = {}

    def set_priority_ages(self, ages_ms: dict[str, int]) -> None:
        """Apply deadlock-supervisor starvation ages to later priority rounds."""

        self.priority_age_ms = {
            vehicle_id: max(0, int(age_ms))
            for vehicle_id, age_ms in ages_ms.items()
        }

    def plan(
        self,
        vehicles: list[Vehicle],
        tasks: list[TransportTask],
        end_time_ms: int,
    ) -> Phase3PlanningResult:
        projections, tasks_by_id = self._validate_inputs(vehicles, tasks)
        reservations = ReservationTable()
        reservations.insert_batch(
            self._hold(
                vehicle,
                plan_id=f"idle:{vehicle.vehicle_id}",
                node_id=vehicle.current_node_id or "",
                start_ms=0,
                end_ms=end_time_ms,
                label="idle-tail",
            )
            for vehicle in projections.values()
        )
        plans: list[VehiclePlan] = []
        records: list[PlanningRecord] = []
        cycles: list[RollingCycleRecord] = []
        planned_tasks: set[str] = set()
        plan_counts: Counter[str] = Counter()
        now_ms = 0
        cycle_index = 0

        while now_ms <= end_time_ms and len(planned_tasks) < len(tasks):
            cycle_started = time.perf_counter_ns()
            pending_at_start = [
                task
                for task in tasks
                if task.task_id not in planned_tasks and task.release_time_ms <= now_ms
            ]
            available_at_start = [
                vehicle
                for vehicle in projections.values()
                if vehicle.available_at_ms <= now_ms
            ]
            candidate_records: list[PriorityCandidateRecord] = []
            selected_ids: list[str] = []
            commitments: list[SafeCommitment] = []
            excluded_pairs: set[tuple[str, str]] = set()
            round_index = 0

            while True:
                pending = [
                    task
                    for task in tasks
                    if task.task_id not in planned_tasks
                    and task.release_time_ms <= now_ms
                ]
                available = [
                    vehicle
                    for vehicle in projections.values()
                    if vehicle.available_at_ms <= now_ms
                ]
                proposals = self.allocator.assign(
                    available, pending, now_ms, frozenset(excluded_pairs)
                )
                if not proposals:
                    break
                orders = self._priority_orders(
                    proposals,
                    tasks_by_id,
                    reservations,
                    projections,
                    now_ms,
                    cycle_index,
                    round_index,
                )
                outcomes = [
                    self._evaluate_candidate(
                        candidate_id=f"cycle-{cycle_index:04d}-round-{round_index:02d}-candidate-{index:02d}",
                        strategy=strategy,
                        order=order,
                        now_ms=now_ms,
                        end_time_ms=end_time_ms,
                        tasks_by_id=tasks_by_id,
                        base_projections=projections,
                        base_reservations=reservations,
                        plan_counts=plan_counts,
                    )
                    for index, (strategy, order) in enumerate(orders)
                ]
                candidate_records.extend(item.record for item in outcomes)
                feasible = [item for item in outcomes if item.record.feasible]
                if not feasible:
                    failed = [
                        item.record
                        for item in outcomes
                        if item.record.failure_pair is not None
                    ]
                    if not failed:
                        break
                    failure = min(
                        failed,
                        key=lambda item: (
                            -item.planned_task_count,
                            item.failure_pair or ("", ""),
                            item.candidate_id,
                        ),
                    )
                    excluded_pairs.add(failure.failure_pair or ("", ""))
                    round_index += 1
                    continue

                selected = min(
                    feasible,
                    key=lambda item: (
                        item.record.score.ordering_key()
                        if item.record.score is not None
                        else (math.inf,),
                        item.record.strategy,
                        item.record.order,
                    ),
                )
                projections = selected.projections
                reservations = selected.reservations
                plans.extend(selected.plans)
                records.extend(selected.records)
                selected_ids.append(selected.record.candidate_id)
                for plan in selected.plans:
                    planned_tasks.add(plan.task_id)
                    plan_counts[plan.vehicle_id] += 1
                    commitments.append(self._safe_commitment(plan, projections[plan.vehicle_id]))
                self.previous_order = tuple(
                    vehicle_id for vehicle_id, _ in selected.record.order
                )
                round_index += 1

            duration_ms = (time.perf_counter_ns() - cycle_started) / 1_000_000
            cycles.append(
                RollingCycleRecord(
                    cycle_index=cycle_index,
                    decision_time_ms=now_ms,
                    pending_task_count=len(pending_at_start),
                    available_vehicle_count=len(available_at_start),
                    candidates=tuple(candidate_records),
                    selected_candidate_ids=tuple(selected_ids),
                    commitments=tuple(commitments),
                    planning_duration_ms=duration_ms,
                )
            )
            now_ms += self.planning_period_ms
            cycle_index += 1

        return Phase3PlanningResult(
            policy=self.policy,
            plans=tuple(
                sorted(plans, key=lambda item: (item.created_at_ms, item.vehicle_id))
            ),
            records=tuple(records),
            cycles=tuple(cycles),
            unplanned_task_ids=tuple(sorted(set(tasks_by_id) - planned_tasks)),
            reservation_conflict_rejections=reservations.conflict_rejections,
            planning_period_ms=self.planning_period_ms,
            planning_horizon_ms=self.planning_horizon_ms,
            execution_horizon_ms=self.execution_horizon_ms,
            planning_timeout_ms=self.planning_timeout_ms,
        )

    def _validate_inputs(
        self, vehicles: list[Vehicle], tasks: list[TransportTask]
    ) -> tuple[dict[str, Vehicle], dict[str, TransportTask]]:
        # 复用阶段 2 的输入语义，但不提前修改调用方对象。
        projections = {
            vehicle.vehicle_id: replace(
                vehicle, state_durations_ms=Counter(vehicle.state_durations_ms)
            )
            for vehicle in vehicles
        }
        tasks_by_id = {task.task_id: task for task in tasks}
        if len(projections) != len(vehicles) or len(tasks_by_id) != len(tasks):
            raise ValueError("vehicle and task ids must be unique")
        for vehicle in projections.values():
            self.topology.validate_vehicle(vehicle)
            if not self.topology.wait_allowed(
                vehicle.current_node_id or "", vehicle.robot_group
            ):
                raise DomainError(
                    "phase3.vehicle.initial_wait_disallowed",
                    f"vehicle {vehicle.vehicle_id!r} must start at a waitable node"
                )
        for task in tasks:
            if task.state is not TaskState.QUEUED or task.assigned_vehicle_id is not None:
                raise DomainError(
                    "phase3.task.initial_state",
                    f"task {task.task_id!r} must start QUEUED and unassigned"
                )
            self.topology.validate_task(task)
        return projections, tasks_by_id

    def _priority_orders(
        self,
        proposals: tuple[AssignmentProposal, ...],
        tasks_by_id: dict[str, TransportTask],
        reservations: ReservationTable,
        projections: dict[str, Vehicle],
        now_ms: int,
        cycle_index: int,
        round_index: int,
    ) -> tuple[tuple[str, tuple[AssignmentProposal, ...]], ...]:
        if self.policy == "top_k":
            strategies = self.priority_strategies
            limit = self.priority_candidate_count
        else:
            strategies = (PriorityStrategy(self.policy),)
            limit = 1

        generated: list[tuple[str, tuple[AssignmentProposal, ...]]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for variant, strategy in enumerate(strategies):
            order = self._order_for_strategy(
                strategy,
                proposals,
                tasks_by_id,
                reservations,
                projections,
                now_ms,
                cycle_index,
                round_index,
                variant,
            )
            signature = tuple((item.vehicle_id, item.task_id) for item in order)
            if signature in seen:
                continue
            seen.add(signature)
            generated.append((strategy.value, order))
            if len(generated) >= limit:
                return tuple(generated)

        # 启发式顺序重复时，用确定性随机扰动补足可用的不同顺序。
        attempts = 0
        while len(generated) < limit and attempts < max(20, limit * 20):
            order = self._random_order(
                proposals, cycle_index, round_index, len(strategies) + attempts
            )
            signature = tuple((item.vehicle_id, item.task_id) for item in order)
            attempts += 1
            if signature in seen:
                continue
            seen.add(signature)
            generated.append((PriorityStrategy.RANDOM.value, order))
        return tuple(generated)

    def _order_for_strategy(
        self,
        strategy: PriorityStrategy,
        proposals: tuple[AssignmentProposal, ...],
        tasks_by_id: dict[str, TransportTask],
        reservations: ReservationTable,
        projections: dict[str, Vehicle],
        now_ms: int,
        cycle_index: int,
        round_index: int,
        variant: int,
    ) -> tuple[AssignmentProposal, ...]:
        if strategy is PriorityStrategy.RANDOM:
            return self._random_order(proposals, cycle_index, round_index, variant)
        if strategy is PriorityStrategy.PREVIOUS_ORDER:
            ranks = {vehicle_id: index for index, vehicle_id in enumerate(self.previous_order)}
            return tuple(
                sorted(
                    proposals,
                    key=lambda item: (
                        ranks.get(item.vehicle_id, len(ranks)),
                        item.vehicle_id,
                        item.task_id,
                    ),
                )
            )
        if strategy is PriorityStrategy.SHORTEST_REMAINING:
            return tuple(
                sorted(
                    proposals,
                    key=lambda item: (
                        item.cost.empty_travel_ms
                        + item.cost.loaded_travel_ms
                        + item.cost.pickup_service_ms
                        + item.cost.dropoff_service_ms,
                        item.vehicle_id,
                        item.task_id,
                    ),
                )
            )
        if strategy is PriorityStrategy.CONGESTION:
            pressure = {
                (item.vehicle_id, item.task_id): self._congestion_pressure(
                    projections[item.vehicle_id],
                    tasks_by_id[item.task_id],
                    reservations,
                    now_ms,
                )
                for item in proposals
            }
            return tuple(
                sorted(
                    proposals,
                    key=lambda item: (
                        -pressure[(item.vehicle_id, item.task_id)],
                        item.cost.total_ms,
                        item.vehicle_id,
                        item.task_id,
                    ),
                )
            )
        return tuple(
            sorted(
                proposals,
                key=lambda item: (
                    -self.priority_age_ms.get(item.vehicle_id, 0),
                    -tasks_by_id[item.task_id].priority_class,
                    tasks_by_id[item.task_id].release_time_ms,
                    item.task_id,
                    item.vehicle_id,
                ),
            )
        )

    def _random_order(
        self,
        proposals: tuple[AssignmentProposal, ...],
        cycle_index: int,
        round_index: int,
        variant: int,
    ) -> tuple[AssignmentProposal, ...]:
        values = list(sorted(proposals, key=lambda item: (item.vehicle_id, item.task_id)))
        rng = random.Random(
            self.seed * 1_000_003
            + cycle_index * 10_007
            + round_index * 1_009
            + variant * 101
        )
        rng.shuffle(values)
        return tuple(values)

    def _congestion_pressure(
        self,
        vehicle: Vehicle,
        task: TransportTask,
        reservations: ReservationTable,
        now_ms: int,
    ) -> int:
        vehicle_group = task.required_robot_group
        vehicle_node = vehicle.current_node_id
        if vehicle_node is None:
            return 0
        route_sets = (
            self.routes.candidate_routes(
                vehicle_group,
                vehicle_node,
                task.pickup_node_id,
                LoadState.EMPTY,
                limit=1,
            ),
            self.routes.candidate_routes(
                vehicle_group,
                task.pickup_node_id,
                task.dropoff_node_id,
                LoadState.LOADED,
                limit=1,
            ),
        )
        edge_ids = {
            edge_id
            for routes in route_sets
            for route in routes
            for edge_id in route.edge_ids
        }
        resource_ids: set[str] = set()
        for edge_id in edge_ids:
            edge_resource = self.topology.edge_resources[edge_id]
            resource_ids.add(edge_resource["ownResource"])
            resource_ids.update(edge_resource["conflictResources"])
        window_start = now_ms
        window_end = window_start + self.planning_horizon_ms
        return sum(
            max(0, min(item.end_ms, window_end) - max(item.start_ms, window_start))
            for item in reservations.snapshot()
            if item.resource_id in resource_ids
            and item.start_ms < window_end
            and window_start < item.end_ms
        )

    def _evaluate_candidate(
        self,
        *,
        candidate_id: str,
        strategy: str,
        order: tuple[AssignmentProposal, ...],
        now_ms: int,
        end_time_ms: int,
        tasks_by_id: dict[str, TransportTask],
        base_projections: dict[str, Vehicle],
        base_reservations: ReservationTable,
        plan_counts: Counter[str],
    ) -> _CandidateOutcome:
        projections = {
            vehicle_id: replace(
                vehicle, state_durations_ms=Counter(vehicle.state_durations_ms)
            )
            for vehicle_id, vehicle in base_projections.items()
        }
        reservations = self._copy_reservations(base_reservations)
        candidate_plans: list[VehiclePlan] = []
        candidate_records: list[PlanningRecord] = []
        local_counts = Counter(plan_counts)
        order_signature = tuple((item.vehicle_id, item.task_id) for item in order)
        for proposal in order:
            vehicle = projections[proposal.vehicle_id]
            task = tasks_by_id[proposal.task_id]
            try:
                planned = self.sipp.plan_task(
                    vehicle,
                    task,
                    now_ms,
                    end_time_ms,
                    reservations,
                    local_counts[vehicle.vehicle_id],
                )
                validated = self.validator.validate(planned.plan, vehicle, task)
                replacement = self._replace_tail(
                    list(reservations.for_vehicle(vehicle.vehicle_id)),
                    vehicle,
                    validated,
                    end_time_ms,
                )
                reservations.replace_vehicle(vehicle.vehicle_id, replacement)
            except (SippPlanningError, ReservationConflict) as error:
                return _CandidateOutcome(
                    record=PriorityCandidateRecord(
                        candidate_id=candidate_id,
                        strategy=strategy,
                        order=order_signature,
                        feasible=False,
                        planned_task_count=len(candidate_plans),
                        score=None,
                        failure_pair=(proposal.vehicle_id, proposal.task_id),
                        failure_code=getattr(error, "code", type(error).__name__),
                    ),
                    projections=projections,
                    reservations=reservations,
                    plans=tuple(candidate_plans),
                    records=tuple(candidate_records),
                )
            candidate_plans.append(planned.plan)
            candidate_records.append(self._record(now_ms, proposal, planned))
            local_counts[vehicle.vehicle_id] += 1
            self._project_vehicle(vehicle, validated)

        score = self._score_candidate(candidate_plans, tasks_by_id, now_ms)
        return _CandidateOutcome(
            record=PriorityCandidateRecord(
                candidate_id=candidate_id,
                strategy=strategy,
                order=order_signature,
                feasible=True,
                planned_task_count=len(candidate_plans),
                score=score,
            ),
            projections=projections,
            reservations=reservations,
            plans=tuple(candidate_plans),
            records=tuple(candidate_records),
        )

    def _score_candidate(
        self,
        plans: list[VehiclePlan],
        tasks_by_id: dict[str, TransportTask],
        now_ms: int,
    ) -> CandidateScore:
        horizon_end = now_ms + self.planning_horizon_ms
        pickups = 0
        dropoffs = 0
        lateness = 0
        queue_time = 0
        wait_ms = 0
        empty_travel_ms = 0
        completion_sum = 0
        for plan in plans:
            task = tasks_by_id[plan.task_id]
            completion_sum += plan.segments[-1].end_ms
            queue_time += max(0, now_ms - task.release_time_ms)
            for segment in plan.segments:
                duration = segment.end_ms - segment.start_ms
                if segment.kind is SegmentKind.PICKUP and segment.end_ms <= horizon_end:
                    pickups += 1
                elif segment.kind is SegmentKind.DROPOFF:
                    if segment.end_ms <= horizon_end:
                        dropoffs += 1
                    if task.due_time_ms is not None:
                        lateness += max(0, segment.end_ms - task.due_time_ms)
                elif segment.kind is SegmentKind.WAIT:
                    wait_ms += duration
                elif (
                    segment.kind is SegmentKind.TRAVERSE
                    and segment.expected_load_state.value == "empty"
                ):
                    empty_travel_ms += duration
        return CandidateScore(
            projected_dropoffs=dropoffs,
            projected_pickups=pickups,
            lateness_ms=lateness,
            queue_time_ms=queue_time,
            wait_ms=wait_ms,
            empty_travel_ms=empty_travel_ms,
            completion_time_sum_ms=completion_sum,
        )

    def _safe_commitment(self, plan: VehiclePlan, projected_vehicle: Vehicle) -> SafeCommitment:
        nominal = min(
            plan.segments[-1].end_ms,
            plan.created_at_ms + self.execution_horizon_ms,
        )
        safe_until = plan.segments[-1].end_ms
        for segment in plan.segments:
            if segment.end_ms < nominal or segment.end_node_id is None:
                continue
            if self.topology.wait_allowed(
                segment.end_node_id, projected_vehicle.robot_group
            ):
                safe_until = segment.end_ms
                break
        return SafeCommitment(
            vehicle_id=plan.vehicle_id,
            task_id=plan.task_id,
            nominal_until_ms=nominal,
            safe_until_ms=safe_until,
        )

    @staticmethod
    def _copy_reservations(source: ReservationTable) -> ReservationTable:
        result = ReservationTable()
        result.insert_batch(source.snapshot())
        result.conflict_rejections = source.conflict_rejections
        return result
