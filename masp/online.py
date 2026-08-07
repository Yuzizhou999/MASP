from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .domain import (
    DomainError,
    LoadState,
    TaskState,
    TransportTask,
    Vehicle,
    VehiclePlan,
    VehicleState,
)
from .phase2 import PlanningRecord
from .phase3 import (
    Phase3CycleProposal,
    Phase3PlanningResult,
    RollingCycleRecord,
    RollingHorizonPlanner,
)
from .reservations import Reservation, ReservationTable
from .simulator import DeterministicSimulator
from .topology import MapTopology


@dataclass(frozen=True)
class OnlinePlanProposal:
    proposal_id: str
    plan: VehiclePlan
    nominal_until_ms: int
    safe_until_ms: int
    reservations: tuple[Reservation, ...]

    def to_dict(self) -> dict[str, Any]:
        committed_segments = [
            segment.to_dict()
            for segment in self.plan.segments
            if segment.end_ms <= self.safe_until_ms
        ]
        return {
            "proposalId": self.proposal_id,
            "status": "PROPOSED",
            "planId": self.plan.plan_id,
            "planRevision": self.plan.revision,
            "vehicleId": self.plan.vehicle_id,
            "taskId": self.plan.task_id,
            "basedOnVehicleRevision": self.plan.based_on_vehicle_revision,
            "createdAtMs": self.plan.created_at_ms,
            "nominalUntilMs": self.nominal_until_ms,
            "safeUntilMs": self.safe_until_ms,
            "extendedToSafeNode": self.safe_until_ms > self.nominal_until_ms,
            "committedSegments": committed_segments,
            "plan": self.plan.to_dict(),
        }


@dataclass(frozen=True)
class PlanAcknowledgement:
    proposal_id: str
    plan_id: str
    plan_revision: int
    accepted: bool
    acknowledged_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposalId": self.proposal_id,
            "planId": self.plan_id,
            "planRevision": self.plan_revision,
            "accepted": self.accepted,
            "acknowledgedAtMs": self.acknowledged_at_ms,
        }


class OnlineDispatchRuntime:
    """In-process online dispatcher with a simulated clock and explicit plan ACKs."""

    def __init__(
        self,
        topology: MapTopology,
        model: dict[str, Any],
        profiles: dict[str, Any],
        scheduler: dict[str, Any],
        traffic_zones: dict[str, Any],
        vehicles: Iterable[Vehicle],
        end_time_ms: int,
        *,
        policy: str | None = None,
        seed: int = 0,
    ) -> None:
        vehicle_rows = list(vehicles)
        self.topology = topology
        self.scheduler = scheduler
        self.end_time_ms = int(end_time_ms)
        self._initial_vehicle_documents = tuple(
            {
                "vehicleId": vehicle.vehicle_id,
                "robotGroup": vehicle.robot_group,
                "initialNodeId": vehicle.current_node_id,
                "initialHeadingRad": vehicle.heading_rad,
                "initialLoadState": vehicle.load_state.value,
                "capabilities": sorted(vehicle.capabilities),
            }
            for vehicle in vehicle_rows
        )
        self._task_documents: dict[str, dict[str, Any]] = {}
        self.planner = RollingHorizonPlanner(
            topology,
            model,
            profiles,
            scheduler,
            traffic_zones,
            policy=policy,
            seed=seed,
        )
        self.simulator = DeterministicSimulator(
            topology=topology,
            vehicles=vehicle_rows,
            tasks=[],
            plans=[],
            end_time_ms=self.end_time_ms,
        )
        self.reservations = ReservationTable()
        self.reservations.insert_batch(
            self.planner._hold(
                vehicle,
                plan_id=f"online-idle:{vehicle.vehicle_id}",
                node_id=vehicle.current_node_id or "",
                start_ms=0,
                end_ms=self.end_time_ms,
                label="idle-tail",
            )
            for vehicle in vehicle_rows
        )
        self.plan_counts: Counter[str] = Counter()
        self.pair_retry_until: dict[tuple[str, str], int] = {}
        self.cycles: list[RollingCycleRecord] = []
        self.records: list[PlanningRecord] = []
        self.proposals: dict[str, OnlinePlanProposal] = {}
        self.pending_proposal_ids: set[str] = set()
        self.acknowledgements: dict[str, PlanAcknowledgement] = {}
        self.accepted_plans: list[VehiclePlan] = []
        self.task_submission_count = 0
        self.telemetry_update_count = 0

    @property
    def now_ms(self) -> int:
        return self.simulator.now_ms

    def submit_task(self, task: TransportTask) -> TransportTask:
        before = task.task_id in self.simulator.tasks
        submitted = self.simulator.submit_task(task)
        if not before:
            self.task_submission_count += 1
            self._task_documents[task.task_id] = self._task_document(task)
        return submitted

    def advance_to(self, target_time_ms: int) -> None:
        self.simulator.run_until(target_time_ms)

    def plan_cycle(self) -> tuple[OnlinePlanProposal, ...]:
        if self.pending_proposal_ids:
            raise DomainError(
                "online.plan.ack_pending",
                "all pending plan proposals must be acknowledged before replanning",
            )
        if self.now_ms >= self.end_time_ms:
            return ()

        vehicles = [
            vehicle
            for vehicle in self.simulator.vehicles.values()
            if vehicle.state is VehicleState.IDLE
            and vehicle.current_node_id is not None
            and vehicle.active_task_id is None
            and vehicle.available_at_ms <= self.now_ms
        ]
        tasks = [
            task
            for task in self.simulator.tasks.values()
            if task.task_id in self.simulator.released_task_ids
            and task.state is TaskState.QUEUED
            and task.assigned_vehicle_id is None
        ]
        planned: Phase3CycleProposal = self.planner.plan_cycle(
            vehicles,
            tasks,
            self.now_ms,
            self.end_time_ms,
            self.reservations,
            cycle_index=len(self.cycles),
            plan_counts=self.plan_counts,
            excluded_pairs=frozenset(
                pair
                for pair, retry_at_ms in self.pair_retry_until.items()
                if retry_at_ms > self.now_ms
            ),
        )
        self.cycles.append(planned.cycle)
        self.records.extend(planned.records)
        for candidate in planned.cycle.candidates:
            if not candidate.feasible and candidate.failure_pair is not None:
                self.pair_retry_until[candidate.failure_pair] = (
                    self.now_ms + self.planner.planning_horizon_ms
                )
        commitments = {
            (item.vehicle_id, item.task_id): item
            for item in planned.cycle.commitments
        }
        proposals: list[OnlinePlanProposal] = []
        for plan in planned.plans:
            commitment = commitments[(plan.vehicle_id, plan.task_id)]
            proposal_id = self._proposal_id(plan)
            proposal = OnlinePlanProposal(
                proposal_id=proposal_id,
                plan=plan,
                nominal_until_ms=commitment.nominal_until_ms,
                safe_until_ms=commitment.safe_until_ms,
                reservations=planned.proposed_reservations.for_vehicle(
                    plan.vehicle_id
                ),
            )
            self.proposals[proposal_id] = proposal
            self.pending_proposal_ids.add(proposal_id)
            proposals.append(proposal)
        return tuple(proposals)

    def acknowledge_plan(
        self,
        proposal_id: str,
        plan_revision: int,
        *,
        accepted: bool,
    ) -> PlanAcknowledgement:
        existing = self.acknowledgements.get(proposal_id)
        if existing is not None:
            if existing.plan_revision == plan_revision and existing.accepted is accepted:
                return existing
            raise DomainError(
                "online.plan.ack_idempotency_conflict",
                f"proposal {proposal_id!r} received a conflicting acknowledgement",
            )
        proposal = self.proposals.get(proposal_id)
        if proposal is None or proposal_id not in self.pending_proposal_ids:
            raise DomainError(
                "online.plan.ack_unknown",
                f"proposal {proposal_id!r} is not pending",
            )
        plan = proposal.plan
        if plan.revision != plan_revision:
            raise DomainError(
                "online.plan.ack_revision_mismatch",
                f"proposal {proposal_id!r} expects plan revision {plan.revision}",
            )
        if accepted:
            vehicle = self.simulator.vehicles[plan.vehicle_id]
            if vehicle.revision != plan.based_on_vehicle_revision:
                raise DomainError(
                    "online.plan.vehicle_revision_stale",
                    f"plan {plan.plan_id!r} expects vehicle revision "
                    f"{plan.based_on_vehicle_revision}, actual revision is {vehicle.revision}",
                )
            self.simulator.add_plan(plan)
            self.reservations.replace_vehicle(
                plan.vehicle_id,
                proposal.reservations,
            )
            self.plan_counts[plan.vehicle_id] += 1
            self.pair_retry_until.pop((plan.vehicle_id, plan.task_id), None)
            self.accepted_plans.append(plan)

        acknowledgement = PlanAcknowledgement(
            proposal_id=proposal_id,
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            accepted=accepted,
            acknowledged_at_ms=self.now_ms,
        )
        self.pending_proposal_ids.remove(proposal_id)
        self.acknowledgements[proposal_id] = acknowledgement
        return acknowledgement

    def update_idle_telemetry(
        self,
        vehicle_id: str,
        *,
        vehicle_revision: int,
        timestamp_ms: int,
        current_node_id: str,
        heading_rad: float,
        load_state: LoadState = LoadState.EMPTY,
    ) -> bool:
        if timestamp_ms != self.now_ms:
            raise DomainError(
                "online.telemetry.timestamp",
                "telemetry must match the current dispatcher clock",
            )
        vehicle = self.simulator.vehicles.get(vehicle_id)
        if vehicle is None:
            raise DomainError(
                "online.telemetry.vehicle_unknown",
                f"unknown vehicle {vehicle_id!r}",
            )
        if vehicle_revision < vehicle.revision:
            raise DomainError(
                "online.telemetry.revision_stale",
                f"telemetry revision {vehicle_revision} is older than {vehicle.revision}",
            )
        if vehicle_revision == vehicle.revision:
            if (
                vehicle.current_node_id == current_node_id
                and vehicle.current_edge_id is None
                and vehicle.heading_rad == float(heading_rad)
                and vehicle.load_state is load_state
            ):
                return False
            raise DomainError(
                "online.telemetry.idempotency_conflict",
                f"vehicle revision {vehicle_revision} was reused with different telemetry",
            )
        if vehicle.state is not VehicleState.IDLE or vehicle.active_task_id is not None:
            raise DomainError(
                "online.telemetry.active_plan_replacement_unsupported",
                "the MVP only accepts corrective telemetry for idle vehicles",
            )
        if load_state is not LoadState.EMPTY:
            raise DomainError(
                "online.telemetry.loaded_idle_unsupported",
                "idle telemetry must report an empty vehicle",
            )
        node = self.topology.nodes.get(current_node_id)
        if node is None or vehicle.robot_group not in node["allowedRobotGroups"]:
            raise DomainError(
                "online.telemetry.node_invalid",
                f"vehicle {vehicle_id!r} cannot occupy node {current_node_id!r}",
            )
        if not self.topology.wait_allowed(current_node_id, vehicle.robot_group):
            raise DomainError(
                "online.telemetry.wait_disallowed",
                f"telemetry node {current_node_id!r} is not a safe waiting node",
            )

        vehicle.current_node_id = current_node_id
        vehicle.current_edge_id = None
        vehicle.heading_rad = float(heading_rad)
        vehicle.load_state = load_state
        vehicle.revision = int(vehicle_revision)
        vehicle.available_at_ms = self.now_ms
        hold = self.planner._hold(
            vehicle,
            plan_id=f"telemetry:{vehicle_id}:{vehicle_revision}",
            node_id=current_node_id,
            start_ms=self.now_ms,
            end_ms=self.end_time_ms,
            label="idle-tail",
        )
        self.reservations.replace_vehicle(vehicle_id, (hold,))
        self.simulator.reservations.replace_vehicle(vehicle_id, (hold,))
        self.telemetry_update_count += 1
        return True

    def planning_result(self) -> Phase3PlanningResult:
        unplanned = tuple(
            sorted(
                task.task_id
                for task in self.simulator.tasks.values()
                if task.state not in {TaskState.COMPLETED, TaskState.CANCELLED}
            )
        )
        return Phase3PlanningResult(
            policy=self.planner.policy,
            plans=tuple(self.accepted_plans),
            records=tuple(self.records),
            cycles=tuple(self.cycles),
            unplanned_task_ids=unplanned,
            reservation_conflict_rejections=self.reservations.conflict_rejections,
            planning_period_ms=self.planner.planning_period_ms,
            planning_horizon_ms=self.planner.planning_horizon_ms,
            execution_horizon_ms=self.planner.execution_horizon_ms,
            planning_timeout_ms=self.planner.planning_timeout_ms,
            rl_inference_count=self.planner.rl_inference_count,
            rl_fallback_count=self.planner.rl_fallback_count,
            rl_inference_ms=self.planner.rl_inference_ms,
            rl_safety_fallback_count=self.planner.rl_safety_fallback_count,
            rl_guardian_candidate_count=self.planner.rl_guardian_candidate_count,
            rl_guardian_override_count=self.planner.rl_guardian_override_count,
            rl_allow_deviation=self.planner.rl_allow_deviation,
        )

    def result(self) -> dict[str, Any]:
        result = self.simulator.result()
        result["online"] = {
            "taskSubmissionCount": self.task_submission_count,
            "planProposalCount": len(self.proposals),
            "acknowledgedPlanCount": sum(
                item.accepted for item in self.acknowledgements.values()
            ),
            "rejectedPlanCount": sum(
                not item.accepted for item in self.acknowledgements.values()
            ),
            "pendingPlanAckCount": len(self.pending_proposal_ids),
            "telemetryUpdateCount": self.telemetry_update_count,
        }
        return result

    def planned_scenario(self, scenario_id: str, seed: int) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "scenarioId": f"{scenario_id}-{self.planner.policy}-online",
            "seed": int(seed),
            "endTimeMs": self.end_time_ms,
            "vehicles": list(self._initial_vehicle_documents),
            "tasks": [
                self._task_documents[task_id]
                for task_id in sorted(self._task_documents)
            ],
            "plans": [plan.to_dict() for plan in self.accepted_plans],
        }

    @staticmethod
    def _task_document(task: TransportTask) -> dict[str, Any]:
        return {
            "taskId": task.task_id,
            "releaseTimeMs": task.release_time_ms,
            "pickupNodeId": task.pickup_node_id,
            "dropoffNodeId": task.dropoff_node_id,
            "requiredRobotGroup": task.required_robot_group,
            "payloadType": task.payload_type,
            "payloadId": task.payload_id,
            "pickupServiceMs": task.pickup_service_ms,
            "dropoffServiceMs": task.dropoff_service_ms,
            "priorityClass": task.priority_class,
            "dueTimeMs": task.due_time_ms,
        }

    @staticmethod
    def _proposal_id(plan: VehiclePlan) -> str:
        value = (
            f"{plan.plan_id}:{plan.revision}:{plan.based_on_vehicle_revision}:"
            f"{plan.created_at_ms}"
        )
        return "proposal-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def run_online_scenario(
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    *,
    policy: str | None = None,
    seed: int | None = None,
) -> OnlineDispatchRuntime:
    defaults = scheduler["serviceDefaults"]
    task_rows = sorted(
        scenario["tasks"],
        key=lambda item: (int(item["releaseTimeMs"]), item["taskId"]),
    )
    runtime = OnlineDispatchRuntime(
        topology=MapTopology(model, conflicts, workstations, traffic_zones),
        model=model,
        profiles=profiles,
        scheduler=scheduler,
        traffic_zones=traffic_zones,
        vehicles=[Vehicle.from_dict(item) for item in scenario["vehicles"]],
        end_time_ms=int(scenario["endTimeMs"]),
        policy=policy,
        seed=int(scenario["seed"] if seed is None else seed),
    )
    planning_period_ms = int(scheduler["planner"]["planningPeriodMs"])
    next_cycle_ms = 0
    task_index = 0

    while runtime.now_ms < runtime.end_time_ms:
        next_release_ms = (
            int(task_rows[task_index]["releaseTimeMs"])
            if task_index < len(task_rows)
            else runtime.end_time_ms
        )
        next_time_ms = min(next_cycle_ms, next_release_ms, runtime.end_time_ms)
        runtime.advance_to(next_time_ms)

        submitted = False
        while (
            task_index < len(task_rows)
            and int(task_rows[task_index]["releaseTimeMs"]) == next_time_ms
        ):
            runtime.submit_task(
                TransportTask.from_dict(
                    task_rows[task_index],
                    int(defaults["pickupServiceMs"]),
                    int(defaults["dropoffServiceMs"]),
                )
            )
            task_index += 1
            submitted = True
        if submitted:
            runtime.advance_to(next_time_ms)

        if next_time_ms == next_cycle_ms or submitted:
            proposals = runtime.plan_cycle()
            for proposal in proposals:
                runtime.acknowledge_plan(
                    proposal.proposal_id,
                    proposal.plan.revision,
                    accepted=True,
                )
            runtime.advance_to(next_time_ms)
        if next_time_ms == next_cycle_ms:
            next_cycle_ms = min(
                runtime.end_time_ms,
                next_cycle_ms + planning_period_ms,
            )

    return runtime
