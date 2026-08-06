from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from .deadlock import DeadlockReport
from .domain import LoadState, PlanSegment, SegmentKind
from .motion import EdgeTravelTimeModel
from .reservations import Reservation, ReservationConflict, ReservationTable
from .topology import MapTopology


class RecoveryPlanningError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RecoveryVehicle:
    vehicle_id: str
    robot_group: str
    load_state: LoadState
    recovery_node_id: str
    wait_since_ms: int
    priority_class: int = 0
    current_node_id: str | None = None
    current_edge_id: str | None = None
    edge_progress: float | None = None
    backtrack_edge_ids: tuple[str, ...] = ()
    held_resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.current_node_id is None) == (self.current_edge_id is None):
            raise ValueError("recovery vehicle must be either on one node or one edge")
        if self.current_edge_id is not None:
            if self.edge_progress is None or not 0.0 < self.edge_progress < 1.0:
                raise ValueError("on-edge recovery requires edge_progress in (0, 1)")
        elif self.edge_progress is not None:
            raise ValueError("node recovery cannot declare edge_progress")
        if self.wait_since_ms < 0:
            raise ValueError("wait_since_ms must be non-negative")
        if self.priority_class < 0:
            raise ValueError("priority_class must be non-negative")


@dataclass(frozen=True)
class RecoverySegment:
    segment_id: str
    start_ms: int
    end_ms: int
    start_node_id: str | None
    end_node_id: str
    source_edge_id: str
    distance_m: float
    resource_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.segment_id,
            "kind": "reverse",
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "startNodeId": self.start_node_id,
            "endNodeId": self.end_node_id,
            "sourceEdgeId": self.source_edge_id,
            "distanceM": round(self.distance_m, 6),
            "resourceIds": list(self.resource_ids),
        }


@dataclass(frozen=True)
class RecoveryPlan:
    plan_id: str
    vehicle_id: str
    recovery_node_id: str
    created_at_ms: int
    completed_at_ms: int
    total_distance_m: float
    segments: tuple[RecoverySegment, ...]
    reservations: tuple[Reservation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.plan_id,
            "vehicleId": self.vehicle_id,
            "recoveryNodeId": self.recovery_node_id,
            "createdAtMs": self.created_at_ms,
            "completedAtMs": self.completed_at_ms,
            "totalDistanceM": round(self.total_distance_m, 6),
            "segments": [item.to_dict() for item in self.segments],
            "reservationCount": len(self.reservations),
        }


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason_code: str
    cycle_vehicle_ids: tuple[str, ...]
    frozen_resource_ids: tuple[str, ...]
    freeze_reservation_ids: tuple[str, ...] = ()
    plan: RecoveryPlan | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "reasonCode": self.reason_code,
            "cycleVehicleIds": list(self.cycle_vehicle_ids),
            "frozenResourceIds": list(self.frozen_resource_ids),
            "freezeReservationIds": list(self.freeze_reservation_ids),
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


class RecoveryController:
    """Choose and atomically reserve one deterministic reverse recovery action."""

    def __init__(
        self,
        topology: MapTopology,
        travel_times: EdgeTravelTimeModel,
        scheduler: dict[str, Any],
        traffic_zones: dict[str, Any],
    ) -> None:
        self.topology = topology
        self.travel_times = travel_times
        self.reverse = scheduler["traffic"]["reverse"]
        self.deadlock = scheduler["traffic"]["deadlock"]
        self.recovery_nodes = {
            item["nodeId"]: frozenset(item["allowedRobotGroups"])
            for item in traffic_zones["recoveryNodes"]
        }
        self.max_attempts = int(self.deadlock["maxRecoveryAttempts"])
        self.recovery_timeout_ms = int(self.deadlock["recoveryTimeoutMs"])
        self._attempts_by_signature: dict[tuple[str, ...], int] = {}
        self._active_decisions: dict[tuple[str, ...], RecoveryDecision] = {}
        self._active_table_versions: dict[tuple[str, ...], int] = {}

    def mark_progress(self, vehicle_ids: tuple[str, ...]) -> None:
        """Clear a cycle only after execution confirms that the cycle has broken."""

        progressed = set(vehicle_ids)
        signatures = set(self._attempts_by_signature) | set(self._active_decisions)
        for signature in tuple(signatures):
            if progressed & set(signature):
                self._attempts_by_signature.pop(signature, None)
                self._active_decisions.pop(signature, None)
                self._active_table_versions.pop(signature, None)

    def mark_recovery_failed(
        self,
        cycle_vehicle_ids: tuple[str, ...],
        reservations: ReservationTable,
    ) -> int:
        """Cancel one unmodified recovery transaction before allowing a retry."""

        signature = tuple(sorted(cycle_vehicle_ids))
        active = self._active_decisions.get(signature)
        expected_version = self._active_table_versions.get(signature)
        if active is None or active.plan is None or expected_version is None:
            return 0
        if reservations.version != expected_version:
            raise RecoveryPlanningError(
                "recovery.transaction.stale",
                "reservation table changed after the recovery transaction",
            )
        removed = reservations.remove_plan(
            active.plan.plan_id, include_committed=True
        )
        self._active_decisions.pop(signature, None)
        self._active_table_versions.pop(signature, None)
        return removed

    def resolve(
        self,
        report: DeadlockReport,
        vehicles: tuple[RecoveryVehicle, ...],
        reservations: ReservationTable,
        now_ms: int,
        end_ms: int,
    ) -> RecoveryDecision:
        if end_ms <= now_ms:
            raise RecoveryPlanningError(
                "recovery.horizon.invalid", "recovery horizon must end after now_ms"
            )
        if not report.cycles:
            self._require_current_report(report, reservations)
            self._attempts_by_signature.clear()
            self._active_decisions.clear()
            self._active_table_versions.clear()
            return RecoveryDecision(
                action="none",
                reason_code="deadlock.none",
                cycle_vehicle_ids=(),
                frozen_resource_ids=(),
            )
        cycle = report.cycles[0]
        if len({item.vehicle_id for item in vehicles}) != len(vehicles):
            raise RecoveryPlanningError(
                "recovery.vehicle.duplicate", "duplicate recovery vehicle id"
            )
        by_id = {item.vehicle_id: item for item in vehicles}
        frozen_resources = tuple(
            sorted(
                {
                    item.resource_id
                    for item in report.dependencies
                    if item.waiting_vehicle_id in cycle
                }
                | {
                    resource_id
                    for vehicle_id in cycle
                    if vehicle_id in by_id
                    for resource_id in by_id[vehicle_id].held_resource_ids
                }
            )
        )
        active = self._active_decisions.get(cycle)
        expired_reason: str | None = None
        if active is not None:
            if self._decision_is_active(active, reservations, now_ms):
                return active
            self._active_decisions.pop(cycle, None)
            self._active_table_versions.pop(cycle, None)
            expired_reason = (
                active.reason_code
                if active.action == "safety_stop"
                else "deadlock.recovery_timeout"
            )

        persisted = self._persisted_decision(
            report,
            cycle,
            frozen_resources,
            by_id,
            reservations,
            now_ms,
        )
        if persisted is not None:
            self._active_decisions[cycle] = persisted
            self._active_table_versions[cycle] = reservations.version
            return persisted

        self._require_current_report(report, reservations)
        if expired_reason is not None:
            return self._safety_stop(
                report,
                cycle,
                frozen_resources,
                reservations,
                now_ms,
                end_ms,
                reason_code=expired_reason,
            )
        attempts = self._attempts_by_signature.get(cycle, 0)
        if attempts >= self.max_attempts:
            return self._safety_stop(
                report,
                cycle,
                frozen_resources,
                reservations,
                now_ms,
                end_ms,
                reason_code="deadlock.livelock_detected",
            )

        candidates: list[
            tuple[
                tuple[object, ...],
                RecoveryPlan,
                tuple[Reservation, ...],
            ]
        ] = []
        for vehicle_id in cycle:
            vehicle = by_id.get(vehicle_id)
            if vehicle is None:
                continue
            try:
                plan_id = self._recovery_plan_id(
                    report, cycle, attempts + 1, vehicle.vehicle_id, now_ms
                )
                plan = self._build_plan(
                    vehicle,
                    now_ms=now_ms,
                    end_ms=end_ms,
                    plan_id=plan_id,
                )
                previous = reservations.for_vehicle(vehicle.vehicle_id)
                replacement = self._recovery_replacement(
                    vehicle,
                    previous,
                    plan,
                    now_ms,
                )
                trial = ReservationTable()
                trial.insert_batch(reservations.snapshot())
                trial.freeze_resources(
                    plan.plan_id,
                    frozen_resources,
                    now_ms,
                    plan.completed_at_ms,
                    exempt_vehicle_id=vehicle.vehicle_id,
                    plan_id=plan.plan_id,
                )
                trial.replace_vehicle(vehicle.vehicle_id, replacement)
            except (RecoveryPlanningError, ReservationConflict):
                continue
            waited_ms = max(0, now_ms - vehicle.wait_since_ms)
            score = (
                int(vehicle.load_state is LoadState.LOADED),
                vehicle.priority_class,
                waited_ms,
                plan.total_distance_m,
                plan.completed_at_ms,
                -len(vehicle.held_resource_ids),
                vehicle.vehicle_id,
            )
            candidates.append((score, plan, replacement))

        if not candidates:
            return self._safety_stop(
                report,
                cycle,
                frozen_resources,
                reservations,
                now_ms,
                end_ms,
                reason_code="deadlock.recovery_unavailable",
            )
        _, selected, replacement = min(candidates, key=lambda item: item[0])
        freeze = reservations.freeze_resources(
            selected.plan_id,
            frozen_resources,
            now_ms,
            selected.completed_at_ms,
            exempt_vehicle_id=selected.vehicle_id,
            plan_id=selected.plan_id,
        )
        reservations.replace_vehicle(selected.vehicle_id, replacement)
        self._attempts_by_signature[cycle] = attempts + 1
        decision = RecoveryDecision(
            action="reverse",
            reason_code="deadlock.reverse_reserved",
            cycle_vehicle_ids=cycle,
            frozen_resource_ids=frozen_resources,
            freeze_reservation_ids=tuple(
                item.reservation_id for item in freeze.reservations
            ),
            plan=selected,
        )
        self._active_decisions[cycle] = decision
        self._active_table_versions[cycle] = reservations.version
        return decision

    def _safety_stop(
        self,
        report: DeadlockReport,
        cycle: tuple[str, ...],
        frozen_resources: tuple[str, ...],
        reservations: ReservationTable,
        now_ms: int,
        end_ms: int,
        *,
        reason_code: str,
    ) -> RecoveryDecision:
        freeze_id = self._safety_stop_id(
            report, cycle, now_ms, reason_code
        )
        freeze = reservations.freeze_resources(
            freeze_id,
            frozen_resources,
            now_ms,
            end_ms,
            plan_id=freeze_id,
        )
        decision = RecoveryDecision(
            action="safety_stop",
            reason_code=reason_code,
            cycle_vehicle_ids=cycle,
            frozen_resource_ids=frozen_resources,
            freeze_reservation_ids=tuple(
                item.reservation_id for item in freeze.reservations
            ),
        )
        self._active_decisions[cycle] = decision
        self._active_table_versions[cycle] = reservations.version
        return decision

    @staticmethod
    def _require_current_report(
        report: DeadlockReport, reservations: ReservationTable
    ) -> None:
        if report.reservation_version != reservations.version:
            raise RecoveryPlanningError(
                "recovery.report.stale",
                "deadlock report was built from an older reservation table version",
            )

    @staticmethod
    def _cycle_token(cycle: tuple[str, ...]) -> str:
        payload = "\0".join(cycle).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _recovery_plan_id(
        self,
        report: DeadlockReport,
        cycle: tuple[str, ...],
        attempt: int,
        vehicle_id: str,
        now_ms: int,
    ) -> str:
        return (
            f"recovery:{self._cycle_token(cycle)}:{report.reservation_version}:"
            f"{now_ms}:{attempt}:{vehicle_id}"
        )

    def _safety_stop_id(
        self,
        report: DeadlockReport,
        cycle: tuple[str, ...],
        now_ms: int,
        reason_code: str,
    ) -> str:
        return (
            f"safety-stop:{self._cycle_token(cycle)}:{report.reservation_version}:"
            f"{now_ms}:{reason_code}"
        )

    def _persisted_decision(
        self,
        report: DeadlockReport,
        cycle: tuple[str, ...],
        frozen_resources: tuple[str, ...],
        vehicles_by_id: dict[str, RecoveryVehicle],
        reservations: ReservationTable,
        now_ms: int,
    ) -> RecoveryDecision | None:
        snapshot = reservations.snapshot()
        rows_by_id = {item.reservation_id: item for item in snapshot}
        token = self._cycle_token(cycle)
        reverse_prefix = f"recovery:{token}:"
        reverse_plan_ids = sorted(
            {
                item.plan_id
                for item in snapshot
                if item.plan_id.startswith(reverse_prefix)
            },
            reverse=True,
        )
        for plan_id in reverse_plan_ids:
            parts = plan_id.split(":", 5)
            if len(parts) != 6:
                continue
            try:
                created_at_ms = int(parts[3])
                attempt = int(parts[4])
            except ValueError:
                continue
            vehicle = vehicles_by_id.get(parts[5])
            if vehicle is None:
                continue
            plan_rows = [item for item in snapshot if item.plan_id == plan_id]
            non_freeze_rows = [
                item for item in plan_rows if item.kind != "safety_freeze"
            ]
            if not non_freeze_rows:
                continue
            try:
                plan = self._build_plan(
                    vehicle,
                    now_ms=created_at_ms,
                    end_ms=max(item.end_ms for item in non_freeze_rows),
                    plan_id=plan_id,
                )
            except RecoveryPlanningError:
                continue
            active_rows = [item for item in plan.reservations if item.end_ms > now_ms]
            if not active_rows or any(
                rows_by_id.get(item.reservation_id) != item for item in active_rows
            ):
                continue
            decision = RecoveryDecision(
                action="reverse",
                reason_code="deadlock.reverse_reserved",
                cycle_vehicle_ids=cycle,
                frozen_resource_ids=frozen_resources,
                freeze_reservation_ids=tuple(
                    f"reservation:{plan_id}:{resource_id}"
                    for resource_id in frozen_resources
                ),
                plan=plan,
            )
            if self._decision_is_active(decision, reservations, now_ms):
                self._attempts_by_signature[cycle] = max(
                    self._attempts_by_signature.get(cycle, 0), attempt
                )
                return decision

        stop_prefix = f"safety-stop:{token}:"
        stop_plan_ids = sorted(
            {
                item.plan_id
                for item in snapshot
                if item.kind == "safety_freeze"
                and item.plan_id.startswith(stop_prefix)
            },
            reverse=True,
        )
        for plan_id in stop_plan_ids:
            parts = plan_id.split(":", 4)
            if len(parts) != 5:
                continue
            freeze_rows = tuple(
                sorted(
                    (
                        item
                        for item in snapshot
                        if item.plan_id == plan_id and item.kind == "safety_freeze"
                    ),
                    key=lambda item: item.reservation_id,
                )
            )
            decision = RecoveryDecision(
                action="safety_stop",
                reason_code=parts[4],
                cycle_vehicle_ids=cycle,
                frozen_resource_ids=tuple(
                    sorted(item.resource_id for item in freeze_rows)
                ),
                freeze_reservation_ids=tuple(
                    item.reservation_id for item in freeze_rows
                ),
            )
            if self._decision_is_active(decision, reservations, now_ms):
                return decision
        return None

    @staticmethod
    def _decision_is_active(
        decision: RecoveryDecision,
        reservations: ReservationTable,
        now_ms: int,
    ) -> bool:
        rows_by_id = {item.reservation_id: item for item in reservations.snapshot()}
        if decision.plan is not None:
            active_rows = [
                item for item in decision.plan.reservations if item.end_ms > now_ms
            ]
            if not active_rows or any(
                rows_by_id.get(item.reservation_id) != item for item in active_rows
            ):
                return False
            if now_ms < decision.plan.completed_at_ms:
                if not decision.freeze_reservation_ids:
                    return False
                for reservation_id in decision.freeze_reservation_ids:
                    freeze = rows_by_id.get(reservation_id)
                    if (
                        freeze is None
                        or freeze.kind != "safety_freeze"
                        or freeze.end_ms <= now_ms
                    ):
                        return False
            return True
        if not decision.freeze_reservation_ids:
            return False
        freezes = [rows_by_id.get(item) for item in decision.freeze_reservation_ids]
        return all(
            item is not None
            and item.kind == "safety_freeze"
            and item.end_ms > now_ms
            for item in freezes
        )

    @staticmethod
    def _recovery_replacement(
        vehicle: RecoveryVehicle,
        previous: tuple[Reservation, ...],
        plan: RecoveryPlan,
        now_ms: int,
    ) -> tuple[Reservation, ...]:
        active = tuple(
            item for item in previous if item.start_ms <= now_ms < item.end_ms
        )
        active_resource_ids = {item.resource_id for item in active}
        missing = set(vehicle.held_resource_ids) - active_resource_ids
        if missing:
            raise RecoveryPlanningError(
                "recovery.held_resource.missing",
                f"vehicle {vehicle.vehicle_id!r} does not hold {sorted(missing)!r}",
            )

        replacement_rows: list[Reservation] = []
        for item in previous:
            if item.end_ms <= now_ms:
                replacement_rows.append(item)
                continue
            if item.start_ms <= now_ms < item.end_ms:
                release_ms = min(item.end_ms, plan.completed_at_ms)
                if release_ms > item.start_ms:
                    replacement_rows.append(replace(item, end_ms=release_ms))
        replacement_rows.extend(plan.reservations)
        return tuple(replacement_rows)

    def plan_for_vehicle(
        self,
        vehicle: RecoveryVehicle,
        reservations: ReservationTable,
        *,
        now_ms: int,
        end_ms: int,
        plan_id: str | None = None,
    ) -> RecoveryPlan:
        candidate = self._build_plan(
            vehicle,
            now_ms=now_ms,
            end_ms=end_ms,
            plan_id=plan_id or f"recovery:{now_ms}:{vehicle.vehicle_id}",
        )
        trial = ReservationTable()
        trial.insert_batch(reservations.snapshot())
        trial.insert_batch(candidate.reservations)
        return candidate

    def _build_plan(
        self,
        vehicle: RecoveryVehicle,
        *,
        now_ms: int,
        end_ms: int,
        plan_id: str,
    ) -> RecoveryPlan:
        mode = str(self.reverse["mode"])
        if mode not in {"recovery_only", "planned"}:
            raise RecoveryPlanningError(
                "recovery.reverse.disabled", "dynamic reverse is not enabled"
            )
        if vehicle.load_state is LoadState.LOADED and not self.reverse["loadedAllowed"]:
            raise RecoveryPlanningError(
                "recovery.reverse.loaded_disallowed",
                "loaded reverse recovery is disabled",
            )
        allowed_groups = self.recovery_nodes.get(vehicle.recovery_node_id)
        if allowed_groups is None or vehicle.robot_group not in allowed_groups:
            raise RecoveryPlanningError(
                "recovery.node.invalid", "recovery node is not declared for this group"
            )
        if not self.topology.wait_allowed(
            vehicle.recovery_node_id, vehicle.robot_group
        ):
            raise RecoveryPlanningError(
                "recovery.node.wait_disallowed", "recovery node cannot hold the vehicle"
            )

        segments: list[RecoverySegment] = []
        cursor_ms = now_ms
        total_distance_m = 0.0
        if vehicle.current_edge_id is not None:
            if not self.reverse["alongCurrentEdgeAllowed"]:
                raise RecoveryPlanningError(
                    "recovery.current_edge.disabled",
                    "along-current-edge reverse is disabled",
                )
            edge = self._edge(vehicle.current_edge_id, vehicle.robot_group)
            if edge["start"] != vehicle.recovery_node_id:
                raise RecoveryPlanningError(
                    "recovery.current_edge.target",
                    "current edge does not backtrack to the selected recovery node",
                )
            distance_m = float(edge["length"]) * float(vehicle.edge_progress or 0.0)
            cursor_ms = self._append_reverse_segment(
                segments,
                vehicle,
                edge,
                start_node_id=None,
                end_node_id=edge["start"],
                start_ms=cursor_ms,
                distance_m=distance_m,
            )
            total_distance_m += distance_m
        else:
            if not self.reverse["alongCurrentEdgeAllowed"]:
                raise RecoveryPlanningError(
                    "recovery.backtrack.disabled",
                    "dynamic backtrack along prior edges is disabled",
                )
            current_node_id = vehicle.current_node_id or ""
            if not vehicle.backtrack_edge_ids:
                raise RecoveryPlanningError(
                    "recovery.path.missing", "no deterministic backtrack path is available"
                )
            for edge_id in vehicle.backtrack_edge_ids:
                edge = self._edge(edge_id, vehicle.robot_group)
                if edge["end"] != current_node_id:
                    raise RecoveryPlanningError(
                        "recovery.path.discontinuous", "backtrack edges are not continuous"
                    )
                distance_m = float(edge["length"])
                cursor_ms = self._append_reverse_segment(
                    segments,
                    vehicle,
                    edge,
                    start_node_id=current_node_id,
                    end_node_id=edge["start"],
                    start_ms=cursor_ms,
                    distance_m=distance_m,
                )
                total_distance_m += distance_m
                current_node_id = edge["start"]
            if current_node_id != vehicle.recovery_node_id:
                raise RecoveryPlanningError(
                    "recovery.path.target", "backtrack path misses the recovery node"
                )

        if total_distance_m > float(self.reverse["maxDistanceM"]) + 1e-9:
            raise RecoveryPlanningError(
                "recovery.distance.exceeded", "reverse distance exceeds maxDistanceM"
            )
        duration_ms = cursor_ms - now_ms
        if duration_ms > int(self.reverse["maxDurationMs"]):
            raise RecoveryPlanningError(
                "recovery.duration.exceeded", "reverse duration exceeds maxDurationMs"
            )
        if cursor_ms > end_ms:
            raise RecoveryPlanningError(
                "recovery.horizon.exceeded", "recovery completes after the control horizon"
            )
        hold_end_ms = cursor_ms + self.recovery_timeout_ms
        if hold_end_ms > end_ms:
            raise RecoveryPlanningError(
                "recovery.hold_horizon.exceeded",
                "control horizon cannot preserve the full recovery-point hold",
            )

        rows: list[Reservation] = []
        for segment in segments:
            for resource_id in segment.resource_ids:
                rows.append(
                    Reservation(
                        reservation_id=(
                            f"reservation:{plan_id}:{segment.segment_id}:{resource_id}"
                        ),
                        resource_id=resource_id,
                        vehicle_id=vehicle.vehicle_id,
                        plan_id=plan_id,
                        segment_id=segment.segment_id,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        kind="reverse",
                        committed=True,
                    )
                )
        resource_id = f"node:{vehicle.recovery_node_id}"
        rows.append(
            Reservation(
                reservation_id=f"reservation:{plan_id}:recovery-hold:{resource_id}",
                resource_id=resource_id,
                vehicle_id=vehicle.vehicle_id,
                plan_id=plan_id,
                segment_id="recovery-hold",
                start_ms=cursor_ms,
                end_ms=hold_end_ms,
                kind="safety_hold",
                committed=True,
            )
        )

        return RecoveryPlan(
            plan_id=plan_id,
            vehicle_id=vehicle.vehicle_id,
            recovery_node_id=vehicle.recovery_node_id,
            created_at_ms=now_ms,
            completed_at_ms=cursor_ms,
            total_distance_m=total_distance_m,
            segments=tuple(segments),
            reservations=tuple(rows),
        )

    def _append_reverse_segment(
        self,
        segments: list[RecoverySegment],
        vehicle: RecoveryVehicle,
        edge: dict[str, Any],
        *,
        start_node_id: str | None,
        end_node_id: str,
        start_ms: int,
        distance_m: float,
    ) -> int:
        reverse_edge = {
            **edge,
            "start": edge["end"],
            "end": edge["start"],
            "p0": edge["p3"],
            "p1": edge["p2"],
            "p2": edge["p1"],
            "p3": edge["p0"],
            "length": distance_m,
            "motionDirection": 1 - int(edge.get("motionDirection", 0)),
        }
        duration_ms = self.travel_times.duration_ms(reverse_edge, vehicle.load_state)
        end_ms = start_ms + duration_ms
        probe = PlanSegment(
            segment_id="recovery-probe",
            kind=SegmentKind.TRAVERSE,
            start_ms=start_ms,
            end_ms=end_ms,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            edge_id=edge["id"],
            expected_load_state=vehicle.load_state,
        )
        resources = self.topology.derived_resources(probe)
        segments.append(
            RecoverySegment(
                segment_id=f"reverse-{len(segments):04d}",
                start_ms=start_ms,
                end_ms=end_ms,
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                source_edge_id=edge["id"],
                distance_m=distance_m,
                resource_ids=resources,
            )
        )
        return end_ms

    def _edge(self, edge_id: str, robot_group: str) -> dict[str, Any]:
        edge = self.topology.edges.get(edge_id)
        if edge is None or edge["robotGroup"] != robot_group:
            raise RecoveryPlanningError(
                "recovery.edge.invalid", f"invalid recovery edge {edge_id!r}"
            )
        return edge
