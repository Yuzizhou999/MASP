from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .deadlock import BlockedRequirement, DeadlockReport, DeadlockSupervisor
from .domain import DomainError, LoadState, PlanSegment, SegmentKind, Vehicle
from .motion import EdgeTravelTimeModel
from .recovery import RecoveryController, RecoveryDecision, RecoveryVehicle
from .reservations import Reservation, ReservationTable
from .routing import RouteProvider, SpatialRoute
from .sipp import ContinuousTimeSippPlanner
from .topology import MapTopology


@dataclass(frozen=True)
class DeadlockCaseResult:
    case_id: str
    report: DeadlockReport
    decision: RecoveryDecision
    projected_after_decision_report: DeadlockReport | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "caseId": self.case_id,
            "waitGraph": self.report.to_dict(),
            "decision": self.decision.to_dict(),
            "projectedAfterDecisionWaitGraph": (
                self.projected_after_decision_report.to_dict()
                if self.projected_after_decision_report is not None
                else None
            ),
        }


@dataclass(frozen=True)
class RecoveryScenarioResult:
    scenario_id: str
    zone_admission: dict[str, object]
    recoverable_deadlock: DeadlockCaseResult
    unrecoverable_deadlock: DeadlockCaseResult
    checks: dict[str, bool]

    @property
    def accepted(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        recovery_plan = self.recoverable_deadlock.decision.plan
        projected_recovery = (
            self.recoverable_deadlock.projected_after_decision_report
        )
        recovery_clears_cycle = (
            recovery_plan is not None
            and projected_recovery is not None
            and not projected_recovery.cycles
        )
        reports = (
            self.recoverable_deadlock.report,
            self.unrecoverable_deadlock.report,
        )
        return {
            "schemaVersion": 1,
            "scenarioId": self.scenario_id,
            "zoneAdmission": self.zone_admission,
            "recoverableDeadlock": self.recoverable_deadlock.to_dict(),
            "unrecoverableDeadlock": self.unrecoverable_deadlock.to_dict(),
            "metrics": {
                "deadlockRiskCount": sum(len(item.cycles) for item in reports),
                "waitGraphCycleCount": sum(len(item.cycles) for item in reports),
                "maxWaitGraphCycleLength": max(
                    (item.max_cycle_length for item in reports), default=0
                ),
                "starvationPromotionCount": sum(
                    age_ms > 0
                    for item in reports
                    for age_ms in item.priority_age_ms.values()
                ),
                "atomicZoneEntryDeferralCount": int(
                    bool(self.zone_admission["entryDeferredOutside"])
                ),
                "recoverySuccessCount": int(recovery_clears_cycle),
                "reverseRecoveryCount": int(recovery_plan is not None),
                "reverseDistanceM": (
                    round(recovery_plan.total_distance_m, 6)
                    if recovery_plan is not None
                    else 0.0
                ),
                "safeStopCount": int(
                    self.unrecoverable_deadlock.decision.action == "safety_stop"
                ),
                "livelockDetectedCount": int(
                    self.recoverable_deadlock.decision.reason_code
                    == "deadlock.livelock_detected"
                    or self.unrecoverable_deadlock.decision.reason_code
                    == "deadlock.livelock_detected"
                ),
            },
            "checks": dict(sorted(self.checks.items())),
            "accepted": self.accepted,
        }


def validate_recovery_scenario_document(
    scenario: dict[str, Any], schemas_dir: Path
) -> None:
    schema = _load_json(schemas_dir / "recovery-scenario.schema.json")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(scenario),
        key=lambda item: list(item.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )
    raise DomainError(
        "recovery.scenario.schema.invalid", f"scenario {path}: {error.message}"
    )


def run_recovery_scenario(
    scenario: dict[str, Any],
    model: dict[str, Any],
    conflicts: dict[str, Any],
    workstations: dict[str, Any],
    profiles: dict[str, Any],
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    schemas_dir: Path,
) -> RecoveryScenarioResult:
    validate_recovery_scenario_document(scenario, schemas_dir)
    now_ms = int(scenario["nowMs"])
    end_time_ms = int(scenario["endTimeMs"])
    if end_time_ms <= now_ms:
        raise DomainError(
            "recovery.scenario.horizon",
            "endTimeMs must be greater than nowMs",
        )
    topology = MapTopology(model, conflicts, workstations, traffic_zones)
    travel_times = EdgeTravelTimeModel(
        model,
        profiles,
        int(scheduler["planner"].get("timeQuantumMs", 100)),
    )
    zone_admission = _run_zone_admission(
        scenario["zoneAdmission"],
        topology,
        model,
        travel_times,
        scheduler,
        traffic_zones,
        end_time_ms,
    )
    recoverable = _run_deadlock_case(
        scenario["recoverableDeadlock"],
        topology,
        travel_times,
        scheduler,
        traffic_zones,
        now_ms,
        end_time_ms,
    )
    unrecoverable = _run_deadlock_case(
        scenario["unrecoverableDeadlock"],
        topology,
        travel_times,
        scheduler,
        traffic_zones,
        now_ms,
        end_time_ms,
    )
    recovery_plan = recoverable.decision.plan
    checks = {
        "zoneEntryDeferredOutside": bool(zone_admission["entryDeferredOutside"]),
        "zoneReservationContinuous": bool(
            zone_admission["zoneReservationContinuous"]
        ),
        "zoneInternalWaitCountIsZero": zone_admission["internalWaitCount"] == 0,
        "recoverableCycleDetected": recoverable.report.max_cycle_length == 2,
        "reverseRecoveryReserved": (
            recoverable.decision.action == "reverse"
            and recovery_plan is not None
            and recovery_plan.recovery_node_id == "fork:PP1173"
        ),
        "recoverableCycleCleared": (
            recoverable.projected_after_decision_report is not None
            and not recoverable.projected_after_decision_report.cycles
        ),
        "unrecoverableRingDetected": unrecoverable.report.max_cycle_length == 4,
        "unrecoverableRingSafeStopped": (
            unrecoverable.decision.action == "safety_stop"
            and unrecoverable.decision.plan is None
            and bool(unrecoverable.decision.freeze_reservation_ids)
        ),
    }
    return RecoveryScenarioResult(
        scenario_id=str(scenario["scenarioId"]),
        zone_admission=zone_admission,
        recoverable_deadlock=recoverable,
        unrecoverable_deadlock=unrecoverable,
        checks=checks,
    )


def _run_zone_admission(
    value: dict[str, Any],
    topology: MapTopology,
    model: dict[str, Any],
    travel_times: EdgeTravelTimeModel,
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    end_time_ms: int,
) -> dict[str, object]:
    zone = topology.traffic_zones.zones_by_id.get(value["zoneId"])
    if zone is None:
        raise DomainError(
            "recovery.zone.missing", f"unknown recovery zone {value['zoneId']!r}"
        )
    if value["blockingReservation"]["resourceId"] != zone.resource_id:
        raise DomainError(
            "recovery.zone.blocker_resource",
            "zone admission blocker must occupy the configured zone resource",
        )
    edge_ids = tuple(value["edgeIds"])
    edges = [topology.edges[edge_id] for edge_id in edge_ids]
    current_node_id = value["startNodeId"]
    for edge in edges:
        if edge["start"] != current_node_id:
            raise DomainError(
                "recovery.zone.route_discontinuous", "zone intent route is discontinuous"
            )
        current_node_id = edge["end"]
    load_state = LoadState(value["loadState"])
    route = SpatialRoute(
        start_node_id=value["startNodeId"],
        end_node_id=current_node_id,
        edge_ids=edge_ids,
        free_flow_travel_ms=sum(
            travel_times.duration_ms(edge, load_state) for edge in edges
        ),
    )
    routes = RouteProvider(model, travel_times)
    planner = ContinuousTimeSippPlanner(
        topology,
        routes,
        travel_times,
        scheduler,
        (item["nodeId"] for item in traffic_zones["recoveryNodes"]),
    )
    vehicle = Vehicle(
        vehicle_id=value["vehicleId"],
        robot_group=value["robotGroup"],
        current_node_id=value["startNodeId"],
        heading_rad=float(
            topology.nodes[value["startNodeId"]]
            .get("headings", {})
            .get(value["robotGroup"], 0.0)
        ),
        load_state=load_state,
    )
    table = ReservationTable()
    blocker = _reservation_from_dict(value["blockingReservation"])
    table.insert_batch((blocker,))
    segments = planner.schedule_route_intent(
        vehicle,
        route,
        ready_ms=0,
        load_state=load_state,
        reservations=table,
        horizon_end_ms=end_time_ms,
    )
    traversals = [item for item in segments if item.kind is SegmentKind.TRAVERSE]
    waits = [item for item in segments if item.kind is SegmentKind.WAIT]
    if traversals[-1].end_ms >= end_time_ms:
        raise DomainError(
            "recovery.zone.terminal_hold_horizon",
            "zone intent must leave time to hold its terminal safe node",
        )
    planned_rows = _segment_reservations(
        value["vehicleId"],
        "recovery-zone-intent",
        segments,
        topology,
        terminal_hold_node_id=route.end_node_id,
        terminal_hold_end_ms=end_time_ms,
    )
    table.insert_batch(planned_rows)
    zone_intervals = [
        (item.start_ms, item.end_ms)
        for item in traversals
        if zone.resource_id in topology.required_resources(item)
    ]
    continuous = (
        len(zone_intervals) == len(traversals)
        and bool(zone_intervals)
        and zone_intervals[0][0] == traversals[0].start_ms
        and zone_intervals[-1][1] == traversals[-1].end_ms
        and all(
            left[1] == right[0]
            for left, right in zip(zone_intervals, zone_intervals[1:])
        )
    )
    first_entry_ms = traversals[0].start_ms
    internal_wait_count = sum(
        topology.traffic_zones.zone_for_node(item.start_node_id or "") is not None
        for item in waits
    )
    return {
        "zoneId": zone.zone_id,
        "vehicleId": value["vehicleId"],
        "blockingVehicleId": blocker.vehicle_id,
        "blockingUntilMs": blocker.end_ms,
        "firstEntryMs": first_entry_ms,
        "completedAtMs": traversals[-1].end_ms,
        "entryDeferredOutside": (
            first_entry_ms >= blocker.end_ms
            and all(item.start_node_id == value["startNodeId"] for item in waits)
        ),
        "zoneReservationContinuous": continuous,
        "internalWaitCount": internal_wait_count,
        "waitMs": sum(item.end_ms - item.start_ms for item in waits),
        "segmentCount": len(segments),
        "reservationCount": len(planned_rows),
        "terminalHoldUntilMs": end_time_ms,
        "terminalHoldReservationCount": sum(
            item.segment_id == "terminal-hold" for item in planned_rows
        ),
    }


def _run_deadlock_case(
    value: dict[str, Any],
    topology: MapTopology,
    travel_times: EdgeTravelTimeModel,
    scheduler: dict[str, Any],
    traffic_zones: dict[str, Any],
    now_ms: int,
    end_ms: int,
) -> DeadlockCaseResult:
    evidence_edge_ids = tuple(value.get("evidenceEdgeIds", ()))
    unknown_evidence = sorted(set(evidence_edge_ids) - topology.edges.keys())
    if unknown_evidence:
        raise DomainError(
            "recovery.deadlock.evidence_edge",
            f"deadlock case references unknown evidence edges {unknown_evidence!r}",
        )
    evidence_resources, evidence_nodes_by_group = _deadlock_evidence(
        topology, evidence_edge_ids
    )
    known_resources = _known_resource_ids(topology)
    referenced_resources = {
        item["resourceId"] for item in value["reservations"]
    } | {
        resource_id
        for item in value["requirements"]
        for resource_id in item["resourceIds"]
    } | {
        resource_id
        for item in value["recoveryVehicles"]
        for resource_id in item["heldResourceIds"]
    }
    unknown_resources = sorted(referenced_resources - known_resources)
    if unknown_resources:
        raise DomainError(
            "recovery.deadlock.resource",
            f"deadlock case references unknown resources {unknown_resources!r}",
        )
    uncovered_resources = sorted(referenced_resources - evidence_resources)
    if uncovered_resources:
        raise DomainError(
            "recovery.deadlock.evidence_resource",
            "deadlock evidence edges do not cover referenced resources "
            f"{uncovered_resources!r}",
        )
    table = ReservationTable()
    table.insert_batch(_reservation_from_dict(item) for item in value["reservations"])
    requirements = tuple(
        BlockedRequirement(
            vehicle_id=item["vehicleId"],
            resource_ids=tuple(item["resourceIds"]),
            start_ms=int(item["startMs"]),
            end_ms=int(item["endMs"]),
            blocked_since_ms=int(item["blockedSinceMs"]),
            has_alternative=bool(item["hasAlternative"]),
        )
        for item in value["requirements"]
    )
    supervisor = DeadlockSupervisor(
        int(scheduler["traffic"]["deadlock"]["starvationAgeStepMs"])
    )
    report = supervisor.analyze(now_ms, requirements, table)
    controller = RecoveryController(
        topology, travel_times, scheduler, traffic_zones
    )
    vehicles = tuple(_recovery_vehicle_from_dict(item) for item in value["recoveryVehicles"])
    if len({item.vehicle_id for item in vehicles}) != len(vehicles):
        raise DomainError(
            "recovery.recovery_vehicle.duplicate",
            "deadlock case contains duplicate recovery vehicle ids",
        )
    for vehicle in vehicles:
        if vehicle.current_node_id is not None:
            node = topology.nodes.get(vehicle.current_node_id)
            if node is None or vehicle.robot_group not in node["allowedRobotGroups"]:
                raise DomainError(
                    "recovery.recovery_vehicle.node",
                    f"invalid recovery node position for {vehicle.vehicle_id!r}",
                )
            if vehicle.current_node_id not in evidence_nodes_by_group.get(
                vehicle.robot_group, set()
            ):
                raise DomainError(
                    "recovery.recovery_vehicle.evidence_node",
                    "recovery vehicle current node must be an endpoint of an "
                    f"evidence edge for {vehicle.vehicle_id!r}",
                )
        else:
            edge = topology.edges.get(vehicle.current_edge_id or "")
            if edge is None or edge["robotGroup"] != vehicle.robot_group:
                raise DomainError(
                    "recovery.recovery_vehicle.edge",
                    f"invalid recovery edge position for {vehicle.vehicle_id!r}",
                )
            if vehicle.current_edge_id not in evidence_edge_ids:
                raise DomainError(
                    "recovery.recovery_vehicle.evidence_edge",
                    "recovery vehicle current edge must be one of the evidence "
                    f"edges for {vehicle.vehicle_id!r}",
                )
    decision = controller.resolve(report, vehicles, table, now_ms, end_ms)
    projected_report = None
    if decision.action == "reverse" and decision.plan is not None:
        completion_ms = decision.plan.completed_at_ms
        table.expire_before(completion_ms)
        projected_requirements = tuple(
            replace(
                item,
                start_ms=completion_ms,
                end_ms=completion_ms + (item.end_ms - item.start_ms),
            )
            for item in requirements
            if item.vehicle_id != decision.plan.vehicle_id
        )
        projected_report = supervisor.analyze(
            completion_ms,
            projected_requirements,
            table,
        )
    return DeadlockCaseResult(
        str(value["id"]),
        report,
        decision,
        projected_report,
    )


def _deadlock_evidence(
    topology: MapTopology,
    evidence_edge_ids: tuple[str, ...],
) -> tuple[set[str], dict[str, set[str]]]:
    resources: set[str] = set()
    nodes_by_group: dict[str, set[str]] = {}
    for edge_id in evidence_edge_ids:
        edge = topology.edges[edge_id]
        edge_resource = topology.edge_resources.get(edge_id)
        if edge_resource is None:
            raise DomainError(
                "recovery.deadlock.evidence_resource",
                f"evidence edge {edge_id!r} has no conflict resource record",
            )
        resources.add(edge_resource["ownResource"])
        resources.update(edge_resource["conflictResources"])
        resources.add(f"node:{edge['start']}")
        resources.add(f"node:{edge['end']}")
        resources.update(topology.traffic_zones.resource_ids_for_edge(edge_id))
        nodes_by_group.setdefault(edge["robotGroup"], set()).update(
            (edge["start"], edge["end"])
        )
    return resources, nodes_by_group


def _known_resource_ids(topology: MapTopology) -> set[str]:
    resources = {f"node:{node_id}" for node_id in topology.nodes}
    for item in topology.edge_resources.values():
        resources.add(item["ownResource"])
        resources.update(item["conflictResources"])
    resources.update(
        f"workstation:{station.station_id}"
        for station in topology.workstations.values()
    )
    resources.update(zone.resource_id for zone in topology.traffic_zones.zones())
    return resources


def _reservation_from_dict(value: dict[str, Any]) -> Reservation:
    return Reservation(
        reservation_id=value["id"],
        resource_id=value["resourceId"],
        vehicle_id=value["vehicleId"],
        plan_id=f"runtime:{value['vehicleId']}",
        segment_id="runtime-hold",
        start_ms=int(value["startMs"]),
        end_ms=int(value["endMs"]),
        kind="safety_hold",
        committed=True,
    )


def _recovery_vehicle_from_dict(value: dict[str, Any]) -> RecoveryVehicle:
    return RecoveryVehicle(
        vehicle_id=value["vehicleId"],
        robot_group=value["robotGroup"],
        load_state=LoadState(value["loadState"]),
        recovery_node_id=value["recoveryNodeId"],
        wait_since_ms=int(value["waitSinceMs"]),
        priority_class=int(value["priorityClass"]),
        current_node_id=value.get("currentNodeId"),
        current_edge_id=value.get("currentEdgeId"),
        edge_progress=(
            float(value["edgeProgress"]) if "edgeProgress" in value else None
        ),
        backtrack_edge_ids=tuple(value.get("backtrackEdgeIds", ())),
        held_resource_ids=tuple(value["heldResourceIds"]),
    )


def _segment_reservations(
    vehicle_id: str,
    plan_id: str,
    segments: tuple[PlanSegment, ...],
    topology: MapTopology,
    *,
    terminal_hold_node_id: str | None = None,
    terminal_hold_end_ms: int | None = None,
) -> tuple[Reservation, ...]:
    rows: list[Reservation] = []
    for segment in segments:
        kind = "wait" if segment.kind is SegmentKind.WAIT else "transit"
        for resource_id in topology.required_resources(segment):
            rows.append(
                Reservation(
                    reservation_id=(
                        f"reservation:{plan_id}:{segment.segment_id}:{resource_id}"
                    ),
                    resource_id=resource_id,
                    vehicle_id=vehicle_id,
                    plan_id=plan_id,
                    segment_id=segment.segment_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    kind=kind,
                    committed=True,
                )
            )
    if terminal_hold_node_id is not None or terminal_hold_end_ms is not None:
        if (
            terminal_hold_node_id is None
            or terminal_hold_end_ms is None
            or not segments
        ):
            raise ValueError("terminal hold requires a node, horizon and route segments")
        hold_start_ms = segments[-1].end_ms
        if terminal_hold_end_ms <= hold_start_ms:
            raise ValueError("terminal hold requires time after route completion")
        probe = PlanSegment(
            segment_id="terminal-hold",
            kind=SegmentKind.WAIT,
            start_ms=hold_start_ms,
            end_ms=terminal_hold_end_ms,
            start_node_id=terminal_hold_node_id,
            end_node_id=terminal_hold_node_id,
            edge_id=None,
            expected_load_state=segments[-1].expected_load_state,
        )
        for resource_id in topology.required_resources(probe):
            rows.append(
                Reservation(
                    reservation_id=(
                        f"reservation:{plan_id}:terminal-hold:{resource_id}"
                    ),
                    resource_id=resource_id,
                    vehicle_id=vehicle_id,
                    plan_id=plan_id,
                    segment_id="terminal-hold",
                    start_ms=hold_start_ms,
                    end_ms=terminal_hold_end_ms,
                    kind="safety_hold",
                    committed=True,
                )
            )
    return tuple(rows)


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
