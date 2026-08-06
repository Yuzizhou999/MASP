from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import Any

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
from .events import DeterministicEventQueue, EventType, SimulationEvent
from .plans import PlanValidator, ValidatedPlan
from .reservations import Reservation, ReservationTable
from .topology import MapTopology


class DeterministicSimulator:
    def __init__(
        self,
        topology: MapTopology,
        vehicles: list[Vehicle],
        tasks: list[TransportTask],
        plans: list[VehiclePlan],
        end_time_ms: int,
    ) -> None:
        if end_time_ms <= 0:
            raise DomainError("scenario.end_time.invalid", "endTimeMs must be positive")
        self.topology = topology
        self.end_time_ms = end_time_ms
        self.now_ms = 0
        self.event_queue = DeterministicEventQueue()
        self.reservations = ReservationTable()
        self.event_log: list[dict[str, Any]] = []
        self.released_task_ids: set[str] = set()
        self.vehicles = self._unique_by_id(vehicles, "vehicle_id", "vehicle")
        self.tasks = self._unique_by_id(tasks, "task_id", "task")
        self.plans = self._unique_by_id(plans, "plan_id", "plan")
        self._plans_by_vehicle: dict[str, list[ValidatedPlan]] = {}
        self._prepare()

    @staticmethod
    def _unique_by_id(items: list[Any], field_name: str, label: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in items:
            item_id = getattr(item, field_name)
            if item_id in result:
                raise DomainError(
                    f"scenario.{label}.duplicate", f"duplicate {label} id {item_id!r}"
                )
            result[item_id] = item
        return result

    def _prepare(self) -> None:
        for vehicle in self.vehicles.values():
            self.topology.validate_vehicle(vehicle)
        for task in sorted(
            self.tasks.values(), key=lambda item: (item.release_time_ms, item.task_id)
        ):
            if task.state is not TaskState.QUEUED or task.assigned_vehicle_id is not None:
                raise DomainError(
                    "scenario.task.initial_state",
                    "phase 1 tasks must enter in QUEUED state without an assigned vehicle",
                )
            self.topology.validate_task(task)
            self.event_queue.schedule(
                task.release_time_ms,
                EventType.TASK_RELEASED,
                {"taskId": task.task_id},
            )

        planned_tasks: set[str] = set()
        validator = PlanValidator(self.topology)
        raw_plans_by_vehicle: dict[str, list[VehiclePlan]] = {}
        for plan in self.plans.values():
            raw_plans_by_vehicle.setdefault(plan.vehicle_id, []).append(plan)

        for vehicle_id in sorted(raw_plans_by_vehicle):
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle is None:
                raise DomainError(
                    "scenario.plan.reference",
                    f"plans reference unknown vehicle {vehicle_id!r}",
                )
            projected = replace(vehicle, state_durations_ms=Counter())
            previous_end_ms = 0
            validated_for_vehicle: list[ValidatedPlan] = []
            for plan in sorted(
                raw_plans_by_vehicle[vehicle_id],
                key=lambda item: (item.created_at_ms, item.plan_id),
            ):
                task = self.tasks.get(plan.task_id)
                if task is None:
                    raise DomainError(
                        "scenario.plan.reference",
                        f"plan {plan.plan_id!r} references an unknown task",
                    )
                if plan.created_at_ms < previous_end_ms:
                    raise DomainError(
                        "scenario.vehicle.plan_overlap",
                        f"vehicle {vehicle_id!r} receives a new task before becoming idle",
                    )
                if plan.task_id in planned_tasks:
                    raise DomainError(
                        "scenario.task.multiple_plans",
                        "a task cannot be assigned to more than one vehicle",
                    )
                validated = validator.validate(plan, projected, task)
                if plan.horizon_end_ms > self.end_time_ms:
                    raise DomainError(
                        "scenario.plan.after_end",
                        f"plan {plan.plan_id!r} exceeds scenario endTimeMs",
                    )
                planned_tasks.add(plan.task_id)
                validated_for_vehicle.append(validated)
                previous_end_ms = plan.segments[-1].end_ms
                projected.current_node_id = validated.final_node_id
                projected.current_edge_id = None
                projected.load_state = validated.final_load_state
                projected.state = VehicleState.IDLE
                projected.revision = projected_vehicle_revision(plan)
                projected.available_at_ms = previous_end_ms
            self._plans_by_vehicle[vehicle_id] = validated_for_vehicle

        for plan in sorted(
            self.plans.values(), key=lambda item: (item.created_at_ms, item.vehicle_id, item.plan_id)
        ):
            vehicle = self.vehicles.get(plan.vehicle_id)
            task = self.tasks.get(plan.task_id)
            if vehicle is None or task is None:
                raise DomainError(
                    "scenario.plan.reference",
                    f"plan {plan.plan_id!r} references an unknown vehicle or task",
                )
            self._schedule_plan(plan)

        reservation_batch: list[Reservation] = []
        for vehicle_id in sorted(self.vehicles):
            vehicle = self.vehicles[vehicle_id]
            validated_plans = self._plans_by_vehicle.get(vehicle_id, [])
            if not validated_plans:
                reservation_batch.extend(
                    self._occupancy_reservations(
                        vehicle,
                        plan_id=f"idle:{vehicle_id}",
                        node_id=vehicle.current_node_id or "",
                        start_ms=0,
                        end_ms=self.end_time_ms,
                        label="idle",
                    )
                )
                continue

            cursor_ms = 0
            cursor_node_id = vehicle.current_node_id or ""
            for index, validated in enumerate(validated_plans):
                plan = validated.plan
                first = plan.segments[0]
                last = plan.segments[-1]
                reservation_batch.extend(
                    self._occupancy_reservations(
                        vehicle,
                        plan_id=plan.plan_id,
                        node_id=cursor_node_id,
                        start_ms=cursor_ms,
                        end_ms=first.start_ms,
                        label=f"pre-plan-{index}",
                    )
                )
                reservation_batch.extend(validated.reservations())
                cursor_ms = last.end_ms
                cursor_node_id = validated.final_node_id
            reservation_batch.extend(
                self._occupancy_reservations(
                    vehicle,
                    plan_id=validated_plans[-1].plan.plan_id,
                    node_id=cursor_node_id,
                    start_ms=cursor_ms,
                    end_ms=self.end_time_ms,
                    label="final-hold",
                )
            )

        self.reservations.insert_batch(reservation_batch)

    @staticmethod
    def _occupancy_reservations(
        vehicle: Vehicle,
        plan_id: str,
        node_id: str,
        start_ms: int,
        end_ms: int,
        label: str,
    ) -> tuple[Reservation, ...]:
        if end_ms <= start_ms:
            return ()
        resource_id = f"node:{node_id}"
        return (
            Reservation(
                reservation_id=f"reservation:{plan_id}:{label}:{resource_id}",
                resource_id=resource_id,
                vehicle_id=vehicle.vehicle_id,
                plan_id=plan_id,
                segment_id=label,
                start_ms=start_ms,
                end_ms=end_ms,
                kind="safety_hold",
                committed=True,
            ),
        )

    def _schedule_plan(self, plan: VehiclePlan) -> None:
        lifecycle_payload = {
            "planId": plan.plan_id,
            "taskId": plan.task_id,
            "vehicleId": plan.vehicle_id,
        }
        self.event_queue.schedule(
            plan.created_at_ms, EventType.TASK_ASSIGNED, lifecycle_payload
        )
        self.event_queue.schedule(
            plan.created_at_ms, EventType.PLAN_COMPUTED, lifecycle_payload
        )
        self.event_queue.schedule(
            plan.created_at_ms, EventType.PLAN_COMMITTED, lifecycle_payload
        )
        for segment in plan.segments:
            payload = {
                **lifecycle_payload,
                "segmentId": segment.segment_id,
            }
            start_event, end_event = {
                SegmentKind.TRAVERSE: (
                    EventType.VEHICLE_ENTER_EDGE,
                    EventType.VEHICLE_EXIT_EDGE,
                ),
                SegmentKind.WAIT: (
                    EventType.VEHICLE_WAIT_STARTED,
                    EventType.VEHICLE_WAIT_ENDED,
                ),
                SegmentKind.PICKUP: (
                    EventType.PICKUP_STARTED,
                    EventType.PICKUP_COMPLETED,
                ),
                SegmentKind.DROPOFF: (
                    EventType.DROPOFF_STARTED,
                    EventType.DROPOFF_COMPLETED,
                ),
            }[segment.kind]
            self.event_queue.schedule(segment.start_ms, start_event, payload)
            self.event_queue.schedule(segment.end_ms, end_event, payload)

    def run(self) -> dict[str, Any]:
        while self.event_queue and self.event_queue.peek().time_ms <= self.end_time_ms:
            event = self.event_queue.pop()
            self.now_ms = event.time_ms
            self._apply(event)
            self._record(event)
        self.now_ms = self.end_time_ms
        return self.result()

    def _apply(self, event: SimulationEvent) -> None:
        payload = event.payload
        task = self.tasks.get(payload.get("taskId"))
        vehicle = self.vehicles.get(payload.get("vehicleId"))
        plan = self.plans.get(payload.get("planId"))
        segment = None
        if plan is not None and "segmentId" in payload:
            segment = next(
                item for item in plan.segments if item.segment_id == payload["segmentId"]
            )

        if event.event_type is EventType.TASK_RELEASED:
            self.released_task_ids.add(payload["taskId"])
            return
        if task is None or vehicle is None:
            raise DomainError(
                "event.reference.invalid",
                f"event {event.event_type.value} has invalid task or vehicle references",
            )
        if event.event_type is EventType.TASK_ASSIGNED:
            if task.task_id not in self.released_task_ids:
                raise DomainError(
                    "task.assignment.before_release", "task was assigned before release"
                )
            task.assigned_vehicle_id = vehicle.vehicle_id
            task.transition(TaskState.ASSIGNED, event.time_ms)
            return
        if event.event_type is EventType.PLAN_COMPUTED:
            return
        if event.event_type is EventType.PLAN_COMMITTED:
            if plan.based_on_vehicle_revision != vehicle.revision:
                raise DomainError(
                    "execution.plan.vehicle_revision_stale",
                    f"plan {plan.plan_id!r} expects vehicle revision "
                    f"{plan.based_on_vehicle_revision}, actual revision is {vehicle.revision}",
                )
            task.transition(TaskState.EN_ROUTE_PICKUP, event.time_ms)
            vehicle.transition(VehicleState.TO_PICKUP, event.time_ms)
            vehicle.active_task_id = task.task_id
            vehicle.plan_id = plan.plan_id
            vehicle.plan_revision = plan.revision
            vehicle.committed_until_ms = plan.committed_until_ms
            vehicle.available_at_ms = plan.segments[-1].end_ms
            return
        if segment is None:
            raise DomainError("event.segment.missing", "action event has no plan segment")
        if vehicle.load_state is not segment.expected_load_state:
            raise DomainError(
                "execution.load_mismatch",
                f"vehicle {vehicle.vehicle_id!r} load state differs from its plan",
            )

        if event.event_type is EventType.VEHICLE_ENTER_EDGE:
            if vehicle.current_node_id != segment.start_node_id:
                raise DomainError(
                    "execution.edge.start_mismatch", "vehicle is not at edge start"
                )
            vehicle.current_node_id = None
            vehicle.current_edge_id = segment.edge_id
            vehicle.revision += 1
        elif event.event_type is EventType.VEHICLE_EXIT_EDGE:
            if vehicle.current_edge_id != segment.edge_id:
                raise DomainError(
                    "execution.edge.exit_mismatch", "vehicle is not on the expected edge"
                )
            vehicle.current_edge_id = None
            vehicle.current_node_id = segment.end_node_id or ""
            node = self.topology.nodes[vehicle.current_node_id]
            vehicle.heading_rad = float(
                node.get("headings", {}).get(vehicle.robot_group, vehicle.heading_rad)
            )
            vehicle.revision += 1
            if (
                vehicle.state is VehicleState.REPOSITIONING
                and plan is not None
                and segment is plan.segments[-1]
            ):
                vehicle.transition(VehicleState.IDLE, event.time_ms)
                vehicle.plan_id = None
                vehicle.plan_revision = None
                vehicle.committed_until_ms = event.time_ms
        elif event.event_type is EventType.VEHICLE_WAIT_STARTED:
            vehicle.waiting_resume_state = vehicle.state
            vehicle.transition(VehicleState.WAITING, event.time_ms)
        elif event.event_type is EventType.VEHICLE_WAIT_ENDED:
            resume = vehicle.waiting_resume_state
            if resume not in {
                VehicleState.TO_PICKUP,
                VehicleState.TO_DROPOFF,
                VehicleState.REPOSITIONING,
            }:
                raise DomainError(
                    "execution.wait.resume_invalid", "wait has no valid resume state"
                )
            vehicle.transition(resume, event.time_ms)
            vehicle.waiting_resume_state = None
        elif event.event_type is EventType.PICKUP_STARTED:
            task.transition(TaskState.PICKUP_SERVICE, event.time_ms)
            vehicle.transition(VehicleState.PICKING, event.time_ms)
        elif event.event_type is EventType.PICKUP_COMPLETED:
            vehicle.load_state = LoadState.LOADED
            vehicle.payload_id = task.payload_id or task.task_id
            task.transition(TaskState.EN_ROUTE_DROPOFF, event.time_ms)
            vehicle.transition(VehicleState.TO_DROPOFF, event.time_ms)
        elif event.event_type is EventType.DROPOFF_STARTED:
            task.transition(TaskState.DROPOFF_SERVICE, event.time_ms)
            vehicle.transition(VehicleState.DROPPING, event.time_ms)
        elif event.event_type is EventType.DROPOFF_COMPLETED:
            vehicle.load_state = LoadState.EMPTY
            vehicle.payload_id = None
            task.transition(TaskState.COMPLETED, event.time_ms)
            vehicle.active_task_id = None
            has_repositioning = plan is not None and segment is not plan.segments[-1]
            if has_repositioning:
                vehicle.transition(VehicleState.REPOSITIONING, event.time_ms)
            else:
                vehicle.transition(VehicleState.IDLE, event.time_ms)
                vehicle.plan_id = None
                vehicle.plan_revision = None
                vehicle.committed_until_ms = event.time_ms

    def _record(self, event: SimulationEvent) -> None:
        payload = dict(sorted(event.payload.items()))
        row: dict[str, Any] = {
            "timeMs": event.time_ms,
            "priority": event.sort_key[1],
            "sequence": event.sequence,
            "type": event.event_type.value,
            "payload": payload,
        }
        vehicle_id = payload.get("vehicleId")
        task_id = payload.get("taskId")
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            row["vehicleState"] = vehicle.state.value
            row["vehicleRevision"] = vehicle.revision
        if task_id in self.tasks:
            task = self.tasks[task_id]
            row["taskState"] = task.state.value
            row["taskRevision"] = task.revision
        self.event_log.append(row)

    def result(self) -> dict[str, Any]:
        event_json = json.dumps(
            self.event_log,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        completed = [
            task for task in self.tasks.values() if task.state is TaskState.COMPLETED
        ]
        cycle_times = [
            task.completed_at_ms - task.release_time_ms
            for task in completed
            if task.completed_at_ms is not None
        ]
        queue_times = [
            task.assigned_at_ms - task.release_time_ms
            for task in self.tasks.values()
            if task.assigned_at_ms is not None
        ]
        delivery_times = [
            task.completed_at_ms - task.picked_at_ms
            for task in completed
            if task.completed_at_ms is not None and task.picked_at_ms is not None
        ]
        task_states = Counter(task.state.value for task in self.tasks.values())
        return {
            "schemaVersion": 1,
            "endTimeMs": self.end_time_ms,
            "eventDigestSha256": hashlib.sha256(event_json.encode("utf-8")).hexdigest(),
            "eventLog": self.event_log,
            "tasks": [
                self.tasks[item_id].to_dict() for item_id in sorted(self.tasks)
            ],
            "vehicles": [
                self.vehicles[item_id].to_dict(self.end_time_ms)
                for item_id in sorted(self.vehicles)
            ],
            "metrics": {
                "releasedTaskCount": len(self.released_task_ids),
                "completedTaskCount": len(completed),
                "taskStateCounts": dict(sorted(task_states.items())),
                "completedDropoffsPerHour": round(
                    len(completed) * 3_600_000 / self.end_time_ms, 6
                ),
                "meanTaskCycleTimeMs": self._mean_or_none(cycle_times),
                "meanTaskQueueTimeMs": self._mean_or_none(queue_times),
                "meanPickupToDropoffTimeMs": self._mean_or_none(delivery_times),
                "reservationCount": len(self.reservations.snapshot()),
                "reservationConflictRejections": self.reservations.conflict_rejections,
                "eventCount": len(self.event_log),
            },
        }

    @staticmethod
    def _mean_or_none(values: list[int]) -> float | None:
        return round(mean(values), 6) if values else None
