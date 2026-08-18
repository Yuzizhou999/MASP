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
from .planning import TaskPlanner, PlanningRecord
from .reservations import Reservation, ReservationConflict, ReservationTable
from .sipp import PlannedTask, SippPlanningError


class PriorityStrategy(str, Enum):
    TASK_AGE = "task_age"
    SHORTEST_REMAINING = "shortest_remaining"
    CONGESTION = "congestion"
    PREVIOUS_ORDER = "previous_order"
    RANDOM = "random"
    RL = "rl"


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
    timed_out: bool = False

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
        if self.timed_out:
            result["timedOut"] = True
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
    conflict_component_sizes: tuple[int, ...] = ()
    deadline_exhausted: bool = False

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
            "conflictComponentSizes": list(self.conflict_component_sizes),
            "conflictComponentCount": len(self.conflict_component_sizes),
            "largestConflictComponent": max(
                self.conflict_component_sizes, default=0
            ),
            "deadlineExhausted": self.deadline_exhausted,
        }
        if include_timing:
            result["planningDurationMs"] = round(self.planning_duration_ms, 3)
        return result


@dataclass(frozen=True)
class DispatchPlanningResult:
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
    rl_inference_count: int = 0
    rl_fallback_count: int = 0
    rl_inference_ms: float = 0.0
    rl_safety_fallback_count: int = 0
    rl_guardian_candidate_count: int = 0
    rl_guardian_override_count: int = 0
    rl_allow_deviation: bool = False

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return round(ordered[index], 3)

    def summary(self, include_timing: bool = True) -> dict[str, Any]:
        durations = [item.planning_duration_ms for item in self.cycles]
        planned_task_ids = {plan.task_id for plan in self.plans}
        result: dict[str, Any] = {
            "policy": self.policy,
            "plannedTaskCount": len(planned_task_ids),
            "planFragmentCount": len(self.plans),
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
            "routeCombinationsPruned": sum(
                item.route_combinations_pruned for item in self.records
            ),
            "scheduleAttempts": sum(item.schedule_attempts for item in self.records),
            "maxRouteExpansionLevel": max(
                (item.route_expansion_level for item in self.records), default=0
            ),
            "sippDeadlineExhaustedCount": sum(
                item.deadline_exhausted for item in self.records
            ),
            "planningDeadlineExhaustedCount": sum(
                item.deadline_exhausted for item in self.cycles
            ),
            "conflictComponentCount": sum(
                len(item.conflict_component_sizes) for item in self.cycles
            ),
            "coupledConflictComponentCount": sum(
                size > 1
                for item in self.cycles
                for size in item.conflict_component_sizes
            ),
            "largestConflictComponent": max(
                (
                    size
                    for item in self.cycles
                    for size in item.conflict_component_sizes
                ),
                default=0,
            ),
            "reservationConflictRejections": self.reservation_conflict_rejections,
            "rlInferenceCount": self.rl_inference_count,
            "rlFallbackCount": self.rl_fallback_count,
            "rlInferenceMs": round(self.rl_inference_ms, 3),
            "rlSafetyFallbackCount": self.rl_safety_fallback_count,
            "rlGuardianCandidateCount": self.rl_guardian_candidate_count,
            "rlGuardianOverrideCount": self.rl_guardian_override_count,
            "rlAllowDeviation": self.rl_allow_deviation,
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


@dataclass(frozen=True)
class DispatchCycleProposal:
    plans: tuple[VehiclePlan, ...]
    records: tuple[PlanningRecord, ...]
    cycle: RollingCycleRecord
    proposed_reservations: ReservationTable


@dataclass
class _CandidateOutcome:
    record: PriorityCandidateRecord
    projections: dict[str, Vehicle]
    reservations: ReservationTable
    plans: tuple[VehiclePlan, ...]
    records: tuple[PlanningRecord, ...]


class RollingHorizonPlanner(TaskPlanner):
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
        priority_policy: Any | None = None,
        rl_checkpoint: str | None = None,
        rl_candidate_count: int | None = None,
        rl_allow_deviation: bool = False,
    ) -> None:
        super().__init__(topology, model, profiles, scheduler, traffic_zones)
        planner = scheduler["planner"]
        coordination = scheduler["coordination"]
        self.planning_period_ms = int(planner["planningPeriodMs"])
        self.planning_horizon_ms = int(planner["planningHorizonMs"])
        self.execution_horizon_ms = int(planner["executionHorizonMs"])
        self.planning_timeout_ms = int(planner["planningTimeoutMs"])
        self.planning_deadline_guard_ms = min(
            max(0, int(planner.get("planningDeadlineGuardMs", 50))),
            max(0, self.planning_timeout_ms - 1),
        )
        self.priority_candidate_count = int(coordination["priorityCandidateCount"])
        self.local_conflict_planning = bool(
            coordination.get("localConflictPlanning", True)
        )
        self.conflict_route_lookahead_count = max(
            1, int(coordination.get("conflictRouteLookaheadCount", 1))
        )
        self.rl_priority_prefix_count = max(
            1, int(coordination.get("rlPriorityPrefixCount", 2))
        )
        self.priority_strategies = tuple(
            PriorityStrategy(value) for value in coordination["priorityStrategies"]
        )
        self.policy = policy or str(coordination["defaultPolicy"])
        allowed_policies = {"top_k", *(item.value for item in PriorityStrategy)}
        if self.policy not in allowed_policies:
            raise ValueError(f"unknown dispatch priority policy {self.policy!r}")
        self.seed = int(seed)
        self.previous_order: tuple[str, ...] = ()
        self.priority_age_ms: dict[str, int] = {}
        self.rl_policy = priority_policy
        self.rl_inference_timeout_ms = int(
            coordination.get("rlInferenceTimeoutMs", self.planning_timeout_ms)
        )
        self.rl_inference_count = 0
        self.rl_fallback_count = 0
        self.rl_inference_ms = 0.0
        self.rl_safety_fallback_count = 0
        self.rl_guardian_candidate_count = 0
        self.rl_guardian_override_count = 0
        self.rl_allow_deviation = bool(rl_allow_deviation)
        self._last_conflict_component_sizes: tuple[int, ...] = ()
        self._proposal_resource_cache: dict[
            tuple[str, str, str, str, str, int], frozenset[str]
        ] = {}
        if (
            self.policy == PriorityStrategy.RL.value
            and self.rl_allow_deviation
            and self.rl_policy is None
            and rl_checkpoint
        ):
            try:
                from .rl_priority import RLPriorityPolicy

                self.rl_policy = RLPriorityPolicy.from_checkpoint(
                    rl_checkpoint,
                    topology=self.topology,
                    routes=self.routes,
                    planning_horizon_ms=self.planning_horizon_ms,
                    candidate_count=rl_candidate_count,
                    prefix_count=self.rl_priority_prefix_count,
                    seed=self.seed,
                )
            except (OSError, RuntimeError, ValueError, KeyError, ImportError):
                # Checkpoint problems are handled by the deterministic fallback.
                self.rl_policy = None

    def set_priority_ages(self, ages_ms: dict[str, int]) -> None:
        """Apply deadlock-supervisor starvation ages to later priority rounds."""

        self.priority_age_ms = {
            vehicle_id: max(0, int(age_ms))
            for vehicle_id, age_ms in ages_ms.items()
        }

    def plan_cycle(
        self,
        vehicles: list[Vehicle],
        tasks: list[TransportTask],
        now_ms: int,
        end_time_ms: int,
        reservations: ReservationTable,
        *,
        cycle_index: int = 0,
        plan_counts: Counter[str] | None = None,
        excluded_pairs: frozenset[tuple[str, str]] = frozenset(),
    ) -> DispatchCycleProposal:
        """Plan one online decision cycle without mutating the live reservation table."""

        if now_ms < 0 or end_time_ms <= now_ms:
            raise ValueError("online planning requires 0 <= now_ms < end_time_ms")
        projections, tasks_by_id = self._validate_inputs(vehicles, tasks)
        working_reservations = self._copy_reservations(reservations)
        local_plan_counts = Counter(plan_counts or {})
        planned_task_ids: set[str] = set()
        plans: list[VehiclePlan] = []
        records: list[PlanningRecord] = []
        candidate_records: list[PriorityCandidateRecord] = []
        selected_ids: list[str] = []
        commitments: list[SafeCommitment] = []
        excluded_pairs_this_cycle: set[tuple[str, str]] = set(excluded_pairs)
        pending_at_start = [
            task for task in tasks if task.release_time_ms <= now_ms
        ]
        available_at_start = [
            vehicle
            for vehicle in projections.values()
            if vehicle.available_at_ms <= now_ms
        ]
        round_index = 0
        cycle_started = time.perf_counter_ns()
        cycle_deadline_ns = cycle_started + (
            self.planning_timeout_ms - self.planning_deadline_guard_ms
        ) * 1_000_000
        conflict_component_sizes: list[int] = []
        deadline_exhausted = False

        while True:
            if time.perf_counter_ns() >= cycle_deadline_ns:
                deadline_exhausted = True
                break
            pending = [
                task
                for task in tasks
                if task.task_id not in planned_task_ids
                and task.release_time_ms <= now_ms
            ]
            available = [
                vehicle
                for vehicle in projections.values()
                if vehicle.available_at_ms <= now_ms
            ]
            proposals = self.allocator.assign(
                available,
                pending,
                now_ms,
                frozenset(excluded_pairs_this_cycle),
            )
            continuations: list[AssignmentProposal] = []
            for vehicle in sorted(available, key=lambda item: item.vehicle_id):
                task_id = vehicle.active_task_id
                if task_id is None or task_id in planned_task_ids:
                    continue
                if (vehicle.vehicle_id, task_id) in excluded_pairs_this_cycle:
                    continue
                task = tasks_by_id.get(task_id)
                if task is None:
                    continue
                cost = self.allocator.continuation_cost(vehicle, task, now_ms)
                if cost is not None:
                    continuations.append(
                        AssignmentProposal(vehicle.vehicle_id, task.task_id, cost)
                    )
            proposals = tuple(
                sorted(
                    (*proposals, *continuations),
                    key=lambda item: (item.vehicle_id, item.task_id),
                )
            )
            if not proposals:
                break

            orders = self._priority_orders(
                proposals,
                tasks_by_id,
                working_reservations,
                projections,
                now_ms,
                cycle_index,
                round_index,
            )
            conflict_component_sizes.extend(self._last_conflict_component_sizes)
            outcomes: list[_CandidateOutcome] = []
            for index, (strategy, order) in enumerate(orders):
                if time.perf_counter_ns() >= cycle_deadline_ns:
                    deadline_exhausted = True
                    break
                outcome = self._evaluate_candidate(
                    candidate_id=(
                        f"online-cycle-{cycle_index:04d}-round-{round_index:02d}"
                        f"-candidate-{index:02d}"
                    ),
                    strategy=strategy,
                    order=order,
                    now_ms=now_ms,
                    end_time_ms=end_time_ms,
                    tasks_by_id=tasks_by_id,
                    base_projections=projections,
                    base_reservations=working_reservations,
                    plan_counts=local_plan_counts,
                    deadline_ns=cycle_deadline_ns,
                )
                outcomes.append(outcome)
                if outcome.record.timed_out:
                    deadline_exhausted = True
                    break
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
                excluded_pairs_this_cycle.add(failure.failure_pair or ("", ""))
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
            working_reservations = selected.reservations
            plans.extend(selected.plans)
            records.extend(selected.records)
            selected_ids.append(selected.record.candidate_id)
            for plan in selected.plans:
                planned_task_ids.add(plan.task_id)
                local_plan_counts[plan.vehicle_id] += 1
                commitments.append(
                    self._safe_commitment(plan, projections[plan.vehicle_id])
                )
            self.previous_order = tuple(
                vehicle_id for vehicle_id, _ in selected.record.order
            )
            round_index += 1
            if deadline_exhausted:
                break

        duration_ms = (time.perf_counter_ns() - cycle_started) / 1_000_000
        return DispatchCycleProposal(
            plans=tuple(plans),
            records=tuple(records),
            cycle=RollingCycleRecord(
                cycle_index=cycle_index,
                decision_time_ms=now_ms,
                pending_task_count=len(pending_at_start),
                available_vehicle_count=len(available_at_start),
                candidates=tuple(candidate_records),
                selected_candidate_ids=tuple(selected_ids),
                commitments=tuple(commitments),
                planning_duration_ms=duration_ms,
                conflict_component_sizes=tuple(conflict_component_sizes),
                deadline_exhausted=deadline_exhausted,
            ),
            proposed_reservations=working_reservations,
        )

    def plan(
        self,
        vehicles: list[Vehicle],
        tasks: list[TransportTask],
        end_time_ms: int,
    ) -> DispatchPlanningResult:
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
        pair_retry_until: dict[tuple[str, str], int] = {}
        now_ms = 0
        cycle_index = 0

        while now_ms <= end_time_ms and len(planned_tasks) < len(tasks):
            cycle_started = time.perf_counter_ns()
            cycle_deadline_ns = (
                cycle_started
                + (self.planning_timeout_ms - self.planning_deadline_guard_ms)
                * 1_000_000
            )
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
            conflict_component_sizes: list[int] = []
            deadline_exhausted = False

            while True:
                if time.perf_counter_ns() >= cycle_deadline_ns:
                    deadline_exhausted = True
                    break
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
                cooled_down_pairs = {
                    pair
                    for pair, retry_at_ms in pair_retry_until.items()
                    if retry_at_ms > now_ms
                }
                proposals = self.allocator.assign(
                    available,
                    pending,
                    now_ms,
                    frozenset(excluded_pairs | cooled_down_pairs),
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
                conflict_component_sizes.extend(
                    self._last_conflict_component_sizes
                )
                outcomes: list[_CandidateOutcome] = []
                for index, (strategy, order) in enumerate(orders):
                    if time.perf_counter_ns() >= cycle_deadline_ns:
                        deadline_exhausted = True
                        break
                    outcome = self._evaluate_candidate(
                        candidate_id=f"cycle-{cycle_index:04d}-round-{round_index:02d}-candidate-{index:02d}",
                        strategy=strategy,
                        order=order,
                        now_ms=now_ms,
                        end_time_ms=end_time_ms,
                        tasks_by_id=tasks_by_id,
                        base_projections=projections,
                        base_reservations=reservations,
                        plan_counts=plan_counts,
                        deadline_ns=cycle_deadline_ns,
                    )
                    outcomes.append(outcome)
                    if outcome.record.timed_out:
                        deadline_exhausted = True
                        break
                candidate_records.extend(item.record for item in outcomes)
                feasible = [item for item in outcomes if item.record.feasible]
                if (
                    not feasible
                    and self.policy == PriorityStrategy.RL.value
                    and self.rl_policy is not None
                    and any(item.record.strategy == PriorityStrategy.RL.value for item in outcomes)
                ):
                    # A legal learned permutation can still be locally infeasible.
                    # Give the deterministic safety baseline one chance before
                    # excluding a vehicle-task pair for the next round.
                    fallback_order = self._localized_order(
                        PriorityStrategy.CONGESTION,
                        self._conflict_components(
                            proposals, tasks_by_id, projections
                        ),
                        tasks_by_id,
                        reservations,
                        projections,
                        now_ms,
                        cycle_index,
                        round_index,
                        0,
                    )
                    fallback_signature = tuple(
                        (item.vehicle_id, item.task_id) for item in fallback_order
                    )
                    if not any(
                        item.record.order == fallback_signature for item in outcomes
                    ):
                        fallback = self._evaluate_candidate(
                            candidate_id=(
                                f"cycle-{cycle_index:04d}-round-{round_index:02d}"
                                "-candidate-rl-fallback"
                            ),
                            strategy="congestion_fallback",
                            order=fallback_order,
                            now_ms=now_ms,
                            end_time_ms=end_time_ms,
                            tasks_by_id=tasks_by_id,
                            base_projections=projections,
                            base_reservations=reservations,
                            plan_counts=plan_counts,
                            deadline_ns=cycle_deadline_ns,
                        )
                        outcomes.append(fallback)
                        candidate_records.append(fallback.record)
                        self.rl_safety_fallback_count += 1
                        feasible = [item for item in outcomes if item.record.feasible]
                        if fallback.record.timed_out:
                            deadline_exhausted = True
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
                    failure_pair = failure.failure_pair or ("", "")
                    excluded_pairs.add(failure_pair)
                    pair_retry_until[failure_pair] = (
                        now_ms
                        + (
                            self.planning_period_ms
                            if failure.timed_out
                            else self.planning_horizon_ms
                        )
                    )
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
                    pair_retry_until.pop((plan.vehicle_id, plan.task_id), None)
                    commitments.append(self._safe_commitment(plan, projections[plan.vehicle_id]))
                self.previous_order = tuple(
                    vehicle_id for vehicle_id, _ in selected.record.order
                )
                round_index += 1
                if deadline_exhausted:
                    break

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
                    conflict_component_sizes=tuple(conflict_component_sizes),
                    deadline_exhausted=deadline_exhausted,
                )
            )
            if selected_ids:
                now_ms += self.planning_period_ms
                cycle_index += 1
                continue

            if deadline_exhausted and now_ms + self.planning_period_ms <= end_time_ms:
                now_ms += self.planning_period_ms
                cycle_index += 1
                continue

            # When no candidate was selected, a fixed-period retry cannot change
            # the state. A moving vehicle's reservations are covered by its next
            # availability time; idle-tail reservations do not change before the
            # scenario ends. Jump only to a release or vehicle availability event.
            next_times = [
                task.release_time_ms
                for task in tasks
                if task.task_id not in planned_tasks
                and now_ms < task.release_time_ms < end_time_ms
            ] + [
                vehicle.available_at_ms
                for vehicle in projections.values()
                if now_ms < vehicle.available_at_ms < end_time_ms
            ]
            if not next_times:
                break
            now_ms = min(next_times)
            cycle_index += 1

        return DispatchPlanningResult(
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
            rl_inference_count=self.rl_inference_count,
            rl_fallback_count=self.rl_fallback_count,
            rl_inference_ms=self.rl_inference_ms,
            rl_safety_fallback_count=self.rl_safety_fallback_count,
            rl_guardian_candidate_count=self.rl_guardian_candidate_count,
            rl_guardian_override_count=self.rl_guardian_override_count,
            rl_allow_deviation=self.rl_allow_deviation,
        )

    def _validate_inputs(
        self, vehicles: list[Vehicle], tasks: list[TransportTask]
    ) -> tuple[dict[str, Vehicle], dict[str, TransportTask]]:
        # Reuse the base planner input semantics without mutating caller objects.
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
                    "coordination.vehicle.initial_wait_disallowed",
                    f"vehicle {vehicle.vehicle_id!r} must start at a waitable node"
                )
        for task in tasks:
            is_queued = (
                task.state is TaskState.QUEUED
                and task.assigned_vehicle_id is None
            )
            is_continuation = task.state in {
                TaskState.EN_ROUTE_PICKUP,
                TaskState.EN_ROUTE_DROPOFF,
            } and task.assigned_vehicle_id in projections
            if not (is_queued or is_continuation):
                raise DomainError(
                    "coordination.task.initial_state",
                    f"task {task.task_id!r} is not queued or safely continuable"
                )
            if is_continuation:
                vehicle = projections[task.assigned_vehicle_id or ""]
                if vehicle.active_task_id != task.task_id:
                    raise DomainError(
                        "coordination.task.continuation_mismatch",
                        f"task {task.task_id!r} is not active on its assigned vehicle",
                    )
            self.topology.validate_task(task)
        return projections, tasks_by_id

    def _proposal_resource_ids(
        self,
        proposal: AssignmentProposal,
        tasks_by_id: dict[str, TransportTask],
        projections: dict[str, Vehicle],
    ) -> frozenset[str]:
        vehicle = projections[proposal.vehicle_id]
        task = tasks_by_id[proposal.task_id]
        current_node_id = vehicle.current_node_id or ""
        cache_key = (
            vehicle.robot_group,
            current_node_id,
            task.pickup_node_id,
            task.dropoff_node_id,
            task.required_robot_group,
            self.conflict_route_lookahead_count,
        )
        cached = self._proposal_resource_cache.get(cache_key)
        if cached is not None:
            return cached

        if task.state is TaskState.EN_ROUTE_DROPOFF:
            routes = (
                self.routes.candidate_routes(
                    task.required_robot_group,
                    current_node_id,
                    task.dropoff_node_id,
                    LoadState.LOADED,
                    limit=self.conflict_route_lookahead_count,
                ),
                self.sipp._recovery_routes(
                    task.required_robot_group, task.dropoff_node_id
                )[: self.conflict_route_lookahead_count],
            )
        else:
            routes = (
                self.routes.candidate_routes(
                    vehicle.robot_group,
                    current_node_id,
                    task.pickup_node_id,
                    LoadState.EMPTY,
                    limit=self.conflict_route_lookahead_count,
                ),
                self.routes.candidate_routes(
                    task.required_robot_group,
                    task.pickup_node_id,
                    task.dropoff_node_id,
                    LoadState.LOADED,
                    limit=self.conflict_route_lookahead_count,
                ),
                self.sipp._recovery_routes(
                    task.required_robot_group, task.dropoff_node_id
                )[: self.conflict_route_lookahead_count],
            )
        resources: set[str] = set()
        for route_group in routes:
            for route in route_group:
                for edge_id in route.edge_ids:
                    edge = self.topology.edges[edge_id]
                    resources.update(
                        self.topology.prospective_motion_resources_for_edge(edge_id)
                    )
                    resources.add(f"node:{edge['start']}")
                    resources.add(f"node:{edge['end']}")
                    resources.update(
                        self.topology.traffic_zones.resource_ids_for_edge(edge_id)
                    )
        for node_id in (task.pickup_node_id, task.dropoff_node_id):
            station = self.topology.workstations[node_id]
            resources.add(f"workstation:{station.station_id}")
            if station.blocks_transit_during_service:
                resources.add(f"node:{node_id}")
            resources.update(
                self.topology.traffic_zones.resource_ids_for_node(node_id)
            )
        result = frozenset(resources)
        self._proposal_resource_cache[cache_key] = result
        return result

    def _conflict_components(
        self,
        proposals: tuple[AssignmentProposal, ...],
        tasks_by_id: dict[str, TransportTask],
        projections: dict[str, Vehicle],
    ) -> tuple[tuple[AssignmentProposal, ...], ...]:
        if len(proposals) <= 1 or not self.local_conflict_planning:
            return (proposals,) if proposals else ()
        resources = [
            self._proposal_resource_ids(item, tasks_by_id, projections)
            for item in proposals
        ]
        parent = list(range(len(proposals)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(proposals)):
            for right in range(left + 1, len(proposals)):
                if resources[left] & resources[right]:
                    union(left, right)

        grouped: dict[int, list[tuple[int, AssignmentProposal]]] = {}
        for index, proposal in enumerate(proposals):
            grouped.setdefault(find(index), []).append((index, proposal))
        ordered = sorted(grouped.values(), key=lambda rows: rows[0][0])
        return tuple(
            tuple(proposal for _, proposal in rows)
            for rows in ordered
        )

    def _localized_order(
        self,
        strategy: PriorityStrategy,
        components: tuple[tuple[AssignmentProposal, ...], ...],
        tasks_by_id: dict[str, TransportTask],
        reservations: ReservationTable,
        projections: dict[str, Vehicle],
        now_ms: int,
        cycle_index: int,
        round_index: int,
        variant: int,
    ) -> tuple[AssignmentProposal, ...]:
        def continuation_rank(proposal: AssignmentProposal) -> tuple[bool, int]:
            task = tasks_by_id[proposal.task_id]
            active = task.state in {
                TaskState.EN_ROUTE_PICKUP,
                TaskState.EN_ROUTE_DROPOFF,
            }
            if not active:
                # Preserve the selected strategy's relative order for queued work.
                return True, 0
            return False, (
                task.assigned_at_ms
                if task.assigned_at_ms is not None
                else task.release_time_ms
            )

        prioritized_components = tuple(
            sorted(
                enumerate(components),
                key=lambda item: (
                    min(continuation_rank(proposal) for proposal in item[1]),
                    item[0],
                ),
            )
        )
        ordered = tuple(
            proposal
            for component_index, (_, component) in enumerate(prioritized_components)
            for proposal in self._order_for_strategy(
                strategy,
                component,
                tasks_by_id,
                reservations,
                projections,
                now_ms,
                cycle_index,
                round_index,
                variant * 1009 + component_index,
            )
        )
        # An active task already owns the vehicle and its payload.  It must
        # retain first claim on shared resources when its commitment expires;
        # otherwise a newly released task can reserve the continuation path
        # and leave the active vehicle rolling through WAIT fragments.
        return tuple(
            sorted(
                ordered,
                key=continuation_rank,
            )
        )

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
        generated: list[tuple[str, tuple[AssignmentProposal, ...]]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        components = self._conflict_components(
            proposals, tasks_by_id, projections
        )
        self._last_conflict_component_sizes = tuple(
            len(component) for component in components
        )

        def heuristic_order(
            strategy: PriorityStrategy, variant: int = 0
        ) -> tuple[AssignmentProposal, ...]:
            return self._localized_order(
                strategy,
                components,
                tasks_by_id,
                reservations,
                projections,
                now_ms,
                cycle_index,
                round_index,
                variant,
            )

        if self.policy == "top_k":
            strategies = self.priority_strategies
            limit = self.priority_candidate_count
        elif self.policy == PriorityStrategy.RL.value:
            if not self.rl_allow_deviation:
                return (
                    (
                        "congestion_fallback",
                        heuristic_order(PriorityStrategy.CONGESTION),
                    ),
                )
            limit = self.priority_candidate_count

            def append_unique(
                strategy_name: str, order: tuple[AssignmentProposal, ...]
            ) -> bool:
                signature = tuple(
                    (item.vehicle_id, item.task_id) for item in order
                )
                if signature in seen:
                    if strategy_name == "congestion_guardian":
                        for index, (_, existing) in enumerate(generated):
                            existing_signature = tuple(
                                (item.vehicle_id, item.task_id)
                                for item in existing
                            )
                            if existing_signature == signature:
                                generated[index] = (strategy_name, existing)
                                return True
                    return False
                seen.add(signature)
                generated.append((strategy_name, tuple(order)))
                return True

            # Keep the useful deterministic Top-K candidates and reserve one
            # slot for learned exploration. Random remains a final filler only.
            heuristic_target = max(1, limit - 1)
            for variant, strategy in enumerate(self.priority_strategies):
                if strategy is PriorityStrategy.RANDOM:
                    continue
                strategy_name = (
                    "congestion_guardian"
                    if strategy is PriorityStrategy.CONGESTION
                    else strategy.value
                )
                if append_unique(strategy_name, heuristic_order(strategy, variant)):
                    if strategy is PriorityStrategy.CONGESTION:
                        self.rl_guardian_candidate_count += 1
                if len(generated) >= heuristic_target:
                    break

            can_infer = (
                self.rl_policy is not None
                and len(generated) < limit
                and any(len(component) > 1 for component in components)
            )
            if can_infer:
                rl_limit = min(
                    limit - len(generated),
                    max(1, int(getattr(self.rl_policy, "candidate_count", 1))),
                )
                started = time.perf_counter_ns()
                self.rl_inference_count += 1
                inference_recorded = False
                try:
                    component_orders: list[
                        tuple[tuple[AssignmentProposal, ...], ...]
                    ] = []
                    for component in components:
                        baseline = self._order_for_strategy(
                            PriorityStrategy.CONGESTION,
                            component,
                            tasks_by_id,
                            reservations,
                            projections,
                            now_ms,
                            cycle_index,
                            round_index,
                            0,
                        )
                        if len(component) <= 1:
                            component_orders.append((baseline,))
                            continue
                        local_orders = self.rl_policy.priority_orders(
                            proposals=baseline,
                            tasks_by_id=tasks_by_id,
                            projections=projections,
                            reservations=reservations,
                            now_ms=now_ms,
                            priority_age_ms=self.priority_age_ms,
                            count=rl_limit,
                            prefix_count=min(
                                self.rl_priority_prefix_count, len(component)
                            ),
                        )
                        expected = {
                            (item.vehicle_id, item.task_id) for item in component
                        }
                        validated_orders: list[tuple[AssignmentProposal, ...]] = []
                        for order in local_orders:
                            signature = tuple(
                                (item.vehicle_id, item.task_id) for item in order
                            )
                            if len(signature) != len(expected) or set(signature) != expected:
                                raise ValueError(
                                    "RL local priority output is not a valid permutation"
                                )
                            validated_orders.append(tuple(order))
                        if not validated_orders:
                            raise ValueError("RL priority policy returned no local candidates")
                        component_orders.append(tuple(validated_orders))
                    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                    self.rl_inference_ms += elapsed_ms
                    inference_recorded = True
                    if elapsed_ms > self.rl_inference_timeout_ms:
                        raise TimeoutError(
                            f"RL priority inference exceeded {self.rl_inference_timeout_ms} ms"
                        )
                    for variant in range(rl_limit):
                        order = tuple(
                            proposal
                            for local_orders in component_orders
                            for proposal in local_orders[
                                min(variant, len(local_orders) - 1)
                            ]
                        )
                        append_unique(PriorityStrategy.RL.value, tuple(order))
                        if len(generated) >= limit:
                            break
                except Exception:
                    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                    if not inference_recorded:
                        self.rl_inference_ms += elapsed_ms
                    self.rl_fallback_count += 1
            else:
                if self.rl_policy is None:
                    self.rl_fallback_count += 1

            attempts = 0
            while len(generated) < limit and attempts < max(20, limit * 20):
                order = heuristic_order(
                    PriorityStrategy.RANDOM,
                    len(self.priority_strategies) + attempts,
                )
                attempts += 1
                append_unique(PriorityStrategy.RANDOM.value, order)
            return tuple(generated)
        else:
            strategies = (PriorityStrategy(self.policy),)
            limit = 1

        for variant, strategy in enumerate(strategies):
            order = heuristic_order(strategy, variant)
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
            order = heuristic_order(
                PriorityStrategy.RANDOM, len(strategies) + attempts
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
            resource_ids.update(
                self.topology.prospective_motion_resources_for_edge(edge_id)
            )
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
        deadline_ns: int | None = None,
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

        def deadline_outcome(
            failure_pair: tuple[str, str] | None = None,
        ) -> _CandidateOutcome:
            feasible = bool(candidate_plans)
            return _CandidateOutcome(
                record=PriorityCandidateRecord(
                    candidate_id=candidate_id,
                    strategy=strategy,
                    order=order_signature,
                    feasible=feasible,
                    planned_task_count=len(candidate_plans),
                    score=(
                        self._score_candidate(candidate_plans, tasks_by_id, now_ms)
                        if feasible
                        else None
                    ),
                    failure_pair=None if feasible else failure_pair,
                    failure_code="sipp.deadline.exceeded",
                    timed_out=True,
                ),
                projections=projections,
                reservations=reservations,
                plans=tuple(candidate_plans),
                records=tuple(candidate_records),
            )

        for proposal in order:
            if (
                deadline_ns is not None
                and time.perf_counter_ns() >= deadline_ns
            ):
                return deadline_outcome(
                    (proposal.vehicle_id, proposal.task_id)
                )
            vehicle = projections[proposal.vehicle_id]
            task = tasks_by_id[proposal.task_id]
            try:
                continuing = task.state in {
                    TaskState.EN_ROUTE_PICKUP,
                    TaskState.EN_ROUTE_DROPOFF,
                }
                plan_method = (
                    self.sipp.plan_remaining_task
                    if continuing
                    else self.sipp.plan_task
                )
                planned = plan_method(
                    vehicle,
                    task,
                    now_ms,
                    end_time_ms,
                    reservations,
                    local_counts[vehicle.vehicle_id],
                    deadline_ns=deadline_ns,
                )
                validated = self.validator.validate(
                    planned.plan,
                    vehicle,
                    task,
                    allow_continuation=continuing,
                )
                replacement = self._replace_tail(
                    list(reservations.for_vehicle(vehicle.vehicle_id)),
                    vehicle,
                    validated,
                    end_time_ms,
                )
                reservations.replace_vehicle(vehicle.vehicle_id, replacement)
            except (SippPlanningError, ReservationConflict) as error:
                if getattr(error, "code", None) == "sipp.deadline.exceeded":
                    return deadline_outcome(
                        (proposal.vehicle_id, proposal.task_id)
                    )
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
            if planned.diagnostics.deadline_exhausted:
                return deadline_outcome()

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
                    segment.kind in {SegmentKind.ROTATE, SegmentKind.TRAVERSE}
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
        for index, segment in enumerate(plan.segments):
            if segment.end_ms < nominal or segment.end_node_id is None:
                continue
            if (
                segment.kind is SegmentKind.WAIT
                and segment.start_ms < nominal < segment.end_ms
                and self.topology.wait_allowed(
                    segment.end_node_id, projected_vehicle.robot_group
                )
            ):
                safe_until = nominal
                break
            if self.topology.wait_allowed(
                segment.end_node_id, projected_vehicle.robot_group
            ) and self._segment_has_stable_end(plan, index):
                safe_until = segment.end_ms
                break
        return SafeCommitment(
            vehicle_id=plan.vehicle_id,
            task_id=plan.task_id,
            nominal_until_ms=nominal,
            safe_until_ms=safe_until,
        )

    @staticmethod
    def _segment_has_stable_end(plan: VehiclePlan, index: int) -> bool:
        segment = plan.segments[index]
        if segment.kind is SegmentKind.ROTATE:
            return segment.command_payload.get("phase") == "end"
        if segment.kind is not SegmentKind.TRAVERSE:
            return True
        if index + 1 >= len(plan.segments):
            return True
        following = plan.segments[index + 1]
        return not (
            following.kind is SegmentKind.ROTATE
            and following.command_payload.get("phase") == "end"
            and following.start_ms == segment.end_ms
        )

    @staticmethod
    def _copy_reservations(source: ReservationTable) -> ReservationTable:
        return source.clone()
