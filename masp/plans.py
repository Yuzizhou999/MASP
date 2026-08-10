from __future__ import annotations

from dataclasses import dataclass

from .domain import (
    DomainError,
    LoadState,
    SegmentKind,
    TransportTask,
    Vehicle,
    VehiclePlan,
)
from .reservations import Reservation
from .topology import MapTopology

# 验证通过后，把"计划 + 每段资源 + 终点状态"打包成一份结果，供调度循环/提交管理器使用
@dataclass(frozen=True)
class ValidatedPlan:
    plan: VehiclePlan
    resources_by_segment: dict[str, tuple[str, ...]]
    final_node_id: str
    final_load_state: LoadState

    # 把计划变成"资源预留清单"
    def reservations(self) -> tuple[Reservation, ...]:
        rows: list[Reservation] = []
        for segment in self.plan.segments:
            kind = {
                SegmentKind.TRAVERSE: "transit",
                SegmentKind.WAIT: "wait",
                SegmentKind.PICKUP: "service",
                SegmentKind.DROPOFF: "service",
            }[segment.kind]
            for resource_id in self.resources_by_segment[segment.segment_id]:
                rows.append(
                    Reservation(
                        reservation_id=(
                            f"reservation:{self.plan.plan_id}:{segment.segment_id}:"
                            f"{resource_id}"
                        ),
                        resource_id=resource_id,
                        vehicle_id=self.plan.vehicle_id,
                        plan_id=self.plan.plan_id,
                        segment_id=segment.segment_id,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        kind=kind,
                        committed=True,
                    )
                )
        return tuple(rows)


class PlanValidator:
    def __init__(self, topology: MapTopology) -> None:
        self.topology = topology

    def validate(
        self,
        plan: VehiclePlan,
        vehicle: Vehicle,
        task: TransportTask,
    ) -> ValidatedPlan:
        # 计划与对象对得上号
        if plan.vehicle_id != vehicle.vehicle_id:
            raise DomainError("plan.vehicle.mismatch", "plan vehicle does not match")
        if plan.task_id != task.task_id:
            raise DomainError("plan.task.mismatch", "plan task does not match")
        if task.required_robot_group != vehicle.robot_group:
            raise DomainError(
                "plan.group.mismatch", "task and vehicle robot groups do not match"
            )
        if vehicle.load_state is not LoadState.EMPTY:
            raise DomainError(
                "plan.initial_load.invalid",
                "a pickup/dropoff transport plan must start with an empty vehicle",
            )
        if plan.based_on_vehicle_revision != vehicle.revision:
            raise DomainError(
                "plan.vehicle_revision.stale",
                f"plan expects vehicle revision {plan.based_on_vehicle_revision}, "
                f"actual revision is {vehicle.revision}",
            )
        if plan.based_on_world_revision != 0:
            raise DomainError(
                "plan.world_revision.stale",
                "explicit simulation scenarios start from world revision 0",
            )
        if not plan.segments:
            raise DomainError("plan.segments.empty", "plan must contain segments")

        segments = plan.segments
        # 时间对不对
        if plan.created_at_ms < task.release_time_ms:
            raise DomainError(
                "plan.created.before_release", "plan cannot be created before task release"
            )
        if plan.created_at_ms > segments[0].start_ms:
            raise DomainError(
                "plan.commit.not_before_execution",
                "plan must be committed no later than its first segment start",
            )
        if plan.committed_until_ms < segments[-1].end_ms:
            raise DomainError(
                "plan.commit.incomplete",
                "explicit simulation requires the complete pickup/dropoff plan to be committed",
            )
        if plan.horizon_end_ms < plan.committed_until_ms:
            raise DomainError(
                "plan.horizon.invalid", "planning horizon ends before committed prefix"
            )

        # 段 ID 不能重复
        segment_ids = [segment.segment_id for segment in segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise DomainError("plan.segment.duplicate", "segment ids must be unique")

        # 车必须停在节点上才能开始
        current_node_id = vehicle.current_node_id
        if current_node_id is None:
            raise DomainError(
                "plan.vehicle.on_edge",
                "explicit plans must start while the vehicle is parked at a node",
            )
        load_state = vehicle.load_state
        service_phase = "before_pickup"
        previous_end_ms: int | None = None
        resources_by_segment: dict[str, tuple[str, ...]] = {}
        # 逐段体检
        for segment in segments:
            if segment.start_ms < 0 or segment.end_ms <= segment.start_ms:
                raise DomainError(
                    "plan.segment.interval",
                    f"segment {segment.segment_id!r} has an invalid interval",
                )
            if previous_end_ms is not None and segment.start_ms != previous_end_ms:
                raise DomainError(
                    "plan.segment.not_contiguous",
                    "explicit plans must use wait segments and have no time gaps",
                )
            previous_end_ms = segment.end_ms
            if segment.expected_load_state is not load_state:
                raise DomainError(
                    "plan.segment.load_mismatch",
                    f"segment {segment.segment_id!r} expects the wrong load state",
                )

            if segment.kind is SegmentKind.TRAVERSE:
                edge = self.topology.edges.get(segment.edge_id or "")
                if edge is None:
                    raise DomainError(
                        "plan.edge.missing",
                        f"segment {segment.segment_id!r} references an unknown edge",
                    )
                if edge["robotGroup"] != vehicle.robot_group:
                    raise DomainError(
                        "plan.edge.group",
                        f"vehicle {vehicle.vehicle_id!r} cannot use edge {edge['id']!r}",
                    )
                if (
                    segment.start_node_id != current_node_id
                    or segment.start_node_id != edge["start"]
                    or segment.end_node_id != edge["end"]
                ):
                    raise DomainError(
                        "plan.edge.continuity",
                        f"segment {segment.segment_id!r} is not continuous with the route",
                    )
                current_node_id = edge["end"]

            elif segment.kind is SegmentKind.WAIT:
                if (
                    segment.start_node_id != current_node_id
                    or segment.end_node_id != current_node_id
                ):
                    raise DomainError(
                        "plan.wait.node",
                        f"wait segment {segment.segment_id!r} must stay at the current node",
                    )
                if not self.topology.wait_allowed(current_node_id, vehicle.robot_group):
                    raise DomainError(
                        "plan.wait.disallowed",
                        f"node {current_node_id!r} does not allow planned waiting",
                    )

            elif segment.kind is SegmentKind.PICKUP:
                if service_phase != "before_pickup":
                    raise DomainError(
                        "plan.pickup.order", "a plan must contain exactly one pickup"
                    )
                self._validate_service_node(
                    segment.start_node_id,
                    segment.end_node_id,
                    current_node_id,
                    task.pickup_node_id,
                    "pickup",
                )
                if segment.end_ms - segment.start_ms != task.pickup_service_ms:
                    raise DomainError(
                        "plan.pickup.duration",
                        "pickup segment duration does not match the task service time",
                    )
                load_state = LoadState.LOADED
                service_phase = "after_pickup"

            elif segment.kind is SegmentKind.DROPOFF:
                if service_phase != "after_pickup":
                    raise DomainError(
                        "plan.dropoff.order", "dropoff must occur once after pickup"
                    )
                self._validate_service_node(
                    segment.start_node_id,
                    segment.end_node_id,
                    current_node_id,
                    task.dropoff_node_id,
                    "dropoff",
                )
                if segment.end_ms - segment.start_ms != task.dropoff_service_ms:
                    raise DomainError(
                        "plan.dropoff.duration",
                        "dropoff segment duration does not match the task service time",
                    )
                load_state = LoadState.EMPTY
                service_phase = "after_dropoff"

            derived = set(self.topology.derived_resources(segment))
            supplied = set(segment.resource_ids)
            if supplied and not derived <= supplied:
                missing = sorted(derived - supplied)
                raise DomainError(
                    "plan.resources.incomplete",
                    f"segment {segment.segment_id!r} omits required resources {missing!r}",
                )
            resources_by_segment[segment.segment_id] = self.topology.required_resources(
                segment
            )

        if service_phase != "after_dropoff":
            raise DomainError(
                "plan.service.incomplete", "plan must finish one pickup and one dropoff"
            )
        if not self.topology.wait_allowed(current_node_id, vehicle.robot_group):
            raise DomainError(
                "plan.final_node.wait_disallowed",
                f"plan must end at a waitable node, got {current_node_id!r}",
            )
        return ValidatedPlan(
            plan=plan,
            resources_by_segment=resources_by_segment,
            final_node_id=current_node_id,
            final_load_state=load_state,
        )

    @staticmethod
    def _validate_service_node(
        start_node_id: str | None,
        end_node_id: str | None,
        current_node_id: str,
        expected_node_id: str,
        phase: str,
    ) -> None:
        if (
            start_node_id != current_node_id
            or end_node_id != current_node_id
            or current_node_id != expected_node_id
        ):
            raise DomainError(
                f"plan.{phase}.node",
                f"{phase} service must remain at task AP {expected_node_id!r}",
            )
