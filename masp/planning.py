from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from .assignment import AssignmentProposal, TaskAllocator
from .domain import (
    DomainError,
    LoadState,
    SegmentKind,
    TaskState,
    TransportTask,
    Vehicle,
    VehiclePlan,
    VehicleState,
    projected_vehicle_revision,
)
from .motion import EdgeTravelTimeModel
from .plans import PlanValidator, ValidatedPlan
from .reservations import Reservation, ReservationTable
from .routing import RouteProvider
from .sipp import ContinuousTimeSippPlanner, PlannedTask, SippPlanningError
from .topology import MapTopology


@dataclass(frozen=True)
class PlanningRecord:
    decision_time_ms: int
    vehicle_id: str
    task_id: str
    assignment_cost_ms: int
    completion_time_ms: int
    inserted_wait_ms: int
    route_combinations_tried: int
    route_combinations_pruned: int
    schedule_attempts: int
    route_expansion_level: int
    deadline_exhausted: bool


@dataclass(frozen=True)
class PlanningResult:
    plans: tuple[VehiclePlan, ...]
    records: tuple[PlanningRecord, ...]
    unplanned_task_ids: tuple[str, ...]
    reservation_conflict_rejections: int

    def summary(self) -> dict[str, Any]:
        return {
            "plannedTaskCount": len(self.plans),
            "unplannedTaskCount": len(self.unplanned_task_ids),
            "unplannedTaskIds": list(self.unplanned_task_ids),
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
        }


class TaskPlanner:
    def __init__(
        self,
        topology: MapTopology,
        model: dict[str, Any],
        profiles: dict[str, Any],
        scheduler: dict[str, Any],
        traffic_zones: dict[str, Any],
    ) -> None:
        planner_config = scheduler["planner"]
        self.topology = topology
        self.scheduler = scheduler
        self.travel_times = EdgeTravelTimeModel(
            model,
            profiles,
            time_quantum_ms=int(planner_config.get("timeQuantumMs", 100)),
        )
        self.routes = RouteProvider(model, self.travel_times)
        self.allocator = TaskAllocator(
            topology, self.routes, scheduler.get("assignment", {})
        )
        self.sipp = ContinuousTimeSippPlanner(
            topology,
            self.routes,
            self.travel_times,
            scheduler,
            (item["nodeId"] for item in traffic_zones["recoveryNodes"]),
        )
        self.validator = PlanValidator(topology)

    def plan(
        self,
        vehicles: list[Vehicle],
        tasks: list[TransportTask],
        end_time_ms: int,
    ) -> PlanningResult:
        projections = {
            vehicle.vehicle_id: replace(vehicle, state_durations_ms=Counter())
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
                    "planning.vehicle.initial_wait_disallowed",
                    f"vehicle {vehicle.vehicle_id!r} must start at a waitable node",
                )
        for task in tasks:
            if task.state is not TaskState.QUEUED or task.assigned_vehicle_id is not None:
                raise DomainError(
                    "planning.task.initial_state",
                    f"task {task.task_id!r} must start QUEUED and unassigned",
                )
            self.topology.validate_task(task)

        reservations = ReservationTable()
        initial_holds = [
            self._hold(
                vehicle,
                plan_id=f"idle:{vehicle.vehicle_id}",
                node_id=vehicle.current_node_id or "",
                start_ms=0,
                end_ms=end_time_ms,
                label="idle-tail",
            )
            for vehicle in projections.values()
        ]
        reservations.insert_batch(initial_holds)
        reservations_by_vehicle = {
            vehicle_id: list(reservations.for_vehicle(vehicle_id))
            for vehicle_id in projections
        }
        plans: list[VehiclePlan] = []
        records: list[PlanningRecord] = []
        planned_tasks: set[str] = set()
        plan_counts: Counter[str] = Counter()
        now_ms = 0

        while now_ms <= end_time_ms and len(planned_tasks) < len(tasks):
            excluded_pairs: set[tuple[str, str]] = set()
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
                round_success = False
                for proposal in proposals:
                    vehicle = projections[proposal.vehicle_id]
                    task = tasks_by_id[proposal.task_id]
                    try:
                        planned = self.sipp.plan_task(
                            vehicle,
                            task,
                            now_ms,
                            end_time_ms,
                            reservations,
                            plan_counts[vehicle.vehicle_id],
                        )
                        validated = self.validator.validate(planned.plan, vehicle, task)
                        replacement = self._replace_tail(
                            reservations_by_vehicle[vehicle.vehicle_id],
                            vehicle,
                            validated,
                            end_time_ms,
                        )
                        reservations.replace_vehicle(vehicle.vehicle_id, replacement)
                    except SippPlanningError:
                        excluded_pairs.add((proposal.vehicle_id, proposal.task_id))
                        continue

                    reservations_by_vehicle[vehicle.vehicle_id] = replacement
                    plans.append(planned.plan)
                    planned_tasks.add(task.task_id)
                    plan_counts[vehicle.vehicle_id] += 1
                    records.append(self._record(now_ms, proposal, planned))
                    self._project_vehicle(vehicle, validated)
                    round_success = True
                if not round_success and all(
                    (proposal.vehicle_id, proposal.task_id) in excluded_pairs
                    for proposal in proposals
                ):
                    continue

            next_times = [
                task.release_time_ms
                for task in tasks
                if task.task_id not in planned_tasks and task.release_time_ms > now_ms
            ] + [
                vehicle.available_at_ms
                for vehicle in projections.values()
                if vehicle.available_at_ms > now_ms
            ] + [
                item.end_ms
                for item in reservations.snapshot()
                if now_ms < item.end_ms < end_time_ms
            ]
            if not next_times:
                break
            next_now = min(next_times)
            if next_now <= now_ms:
                break
            now_ms = next_now

        return PlanningResult(
            plans=tuple(sorted(plans, key=lambda item: (item.created_at_ms, item.vehicle_id))),
            records=tuple(records),
            unplanned_task_ids=tuple(sorted(set(tasks_by_id) - planned_tasks)),
            reservation_conflict_rejections=reservations.conflict_rejections,
        )

    @staticmethod
    def _record(
        now_ms: int,
        proposal: AssignmentProposal,
        planned: PlannedTask,
    ) -> PlanningRecord:
        diagnostics = planned.diagnostics
        return PlanningRecord(
            decision_time_ms=now_ms,
            vehicle_id=proposal.vehicle_id,
            task_id=proposal.task_id,
            assignment_cost_ms=proposal.cost.total_ms,
            completion_time_ms=diagnostics.completion_time_ms,
            inserted_wait_ms=diagnostics.inserted_wait_ms,
            route_combinations_tried=diagnostics.route_combinations_tried,
            route_combinations_pruned=diagnostics.route_combinations_pruned,
            schedule_attempts=diagnostics.schedule_attempts,
            route_expansion_level=diagnostics.route_expansion_level,
            deadline_exhausted=diagnostics.deadline_exhausted,
        )

    @staticmethod
    def _project_vehicle(vehicle: Vehicle, validated: ValidatedPlan) -> None:
        plan = validated.plan
        vehicle.current_node_id = validated.final_node_id
        vehicle.current_edge_id = None
        vehicle.load_state = validated.final_load_state
        vehicle.payload_id = None
        vehicle.state = VehicleState.IDLE
        vehicle.revision = projected_vehicle_revision(plan)
        vehicle.available_at_ms = plan.segments[-1].end_ms
        for segment in reversed(plan.segments):
            if segment.kind in {SegmentKind.ROTATE, SegmentKind.TRAVERSE}:
                try:
                    vehicle.heading_rad = float(
                        segment.command_payload["endHeadingRad"]
                    )
                except (KeyError, TypeError, ValueError):
                    pass
                break

    def _replace_tail(
        self,
        previous: list[Reservation],
        vehicle: Vehicle,
        validated: ValidatedPlan,
        end_time_ms: int,
    ) -> list[Reservation]:
        retained = [item for item in previous if item.segment_id != "idle-tail"]
        plan = validated.plan
        first = plan.segments[0]
        last = plan.segments[-1]
        previous_end_ms = vehicle.available_at_ms
        if first.start_ms > previous_end_ms:
            retained.append(
                self._hold(
                    vehicle,
                    plan_id=plan.plan_id,
                    node_id=vehicle.current_node_id or "",
                    start_ms=previous_end_ms,
                    end_ms=first.start_ms,
                    label=f"pre-plan-{plan.revision}",
                )
            )
        retained.extend(validated.reservations())
        if last.end_ms < end_time_ms:
            retained.append(
                self._hold(
                    vehicle,
                    plan_id=plan.plan_id,
                    node_id=validated.final_node_id,
                    start_ms=last.end_ms,
                    end_ms=end_time_ms,
                    label="idle-tail",
                )
            )
        return retained

    @staticmethod
    def _hold(
        vehicle: Vehicle,
        plan_id: str,
        node_id: str,
        start_ms: int,
        end_ms: int,
        label: str,
    ) -> Reservation:
        resource_id = f"node:{node_id}"
        return Reservation(
            reservation_id=f"reservation:{plan_id}:{label}:{resource_id}",
            resource_id=resource_id,
            vehicle_id=vehicle.vehicle_id,
            plan_id=plan_id,
            segment_id=label,
            start_ms=start_ms,
            end_ms=end_ms,
            kind="safety_hold",
            committed=True,
        )
