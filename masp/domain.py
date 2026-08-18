###定义任务、车辆、计划、计划段这四类核心对象，以及它们各自的"合法状态机"——不是任何状态都能随便乱切，必须按规则来

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# 带"错误码"的异常
class DomainError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

# 任务状态机
class TaskState(str, Enum):
    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"
    EN_ROUTE_PICKUP = "EN_ROUTE_PICKUP"
    PICKUP_SERVICE = "PICKUP_SERVICE"
    EN_ROUTE_DROPOFF = "EN_ROUTE_DROPOFF"
    DROPOFF_SERVICE = "DROPOFF_SERVICE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

# 车辆状态机
class VehicleState(str, Enum):
    IDLE = "IDLE"
    TO_PICKUP = "TO_PICKUP"
    PICKING = "PICKING"
    TO_DROPOFF = "TO_DROPOFF"
    DROPPING = "DROPPING"
    REPOSITIONING = "REPOSITIONING"
    WAITING = "WAITING"
    ROTATING = "ROTATING"
    REVERSING = "REVERSING"
    CHARGING = "CHARGING"
    FAULT = "FAULT"
    STOPPED = "STOPPED"

# 载荷
class LoadState(str, Enum):
    EMPTY = "empty"
    LOADED = "loaded"

# 计划段
class SegmentKind(str, Enum):
    ROTATE = "rotate"
    TRAVERSE = "traverse"
    WAIT = "wait"
    PICKUP = "pickup"
    DROPOFF = "dropoff"

# 任务状态转移表
TASK_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.QUEUED: {TaskState.ASSIGNED, TaskState.CANCELLED},
    TaskState.ASSIGNED: {
        TaskState.QUEUED,
        TaskState.EN_ROUTE_PICKUP,
        TaskState.CANCELLED,
    },
    TaskState.EN_ROUTE_PICKUP: {TaskState.PICKUP_SERVICE, TaskState.CANCELLED},
    TaskState.PICKUP_SERVICE: {TaskState.EN_ROUTE_DROPOFF, TaskState.FAILED},
    TaskState.EN_ROUTE_DROPOFF: {TaskState.DROPOFF_SERVICE, TaskState.FAILED},
    TaskState.DROPOFF_SERVICE: {TaskState.COMPLETED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.CANCELLED: set(),
    TaskState.FAILED: set(),
}

# 车辆状态转移表
VEHICLE_TRANSITIONS: dict[VehicleState, set[VehicleState]] = {
    VehicleState.IDLE: {
        VehicleState.TO_PICKUP,
        VehicleState.CHARGING,
        VehicleState.FAULT,
    },
    VehicleState.TO_PICKUP: {
        VehicleState.PICKING,
        VehicleState.WAITING,
        VehicleState.ROTATING,
        VehicleState.REVERSING,
        VehicleState.FAULT,
    },
    VehicleState.PICKING: {VehicleState.TO_DROPOFF, VehicleState.FAULT},
    VehicleState.TO_DROPOFF: {
        VehicleState.DROPPING,
        VehicleState.WAITING,
        VehicleState.ROTATING,
        VehicleState.REVERSING,
        VehicleState.FAULT,
    },
    VehicleState.DROPPING: {
        VehicleState.IDLE,
        VehicleState.REPOSITIONING,
        VehicleState.FAULT,
    },
    VehicleState.REPOSITIONING: {
        VehicleState.IDLE,
        VehicleState.WAITING,
        VehicleState.ROTATING,
        VehicleState.FAULT,
    },
    VehicleState.WAITING: {
        VehicleState.TO_PICKUP,
        VehicleState.TO_DROPOFF,
        VehicleState.REPOSITIONING,
        VehicleState.FAULT,
    },
    VehicleState.ROTATING: {
        VehicleState.TO_PICKUP,
        VehicleState.TO_DROPOFF,
        VehicleState.REPOSITIONING,
        VehicleState.FAULT,
    },
    VehicleState.REVERSING: {VehicleState.WAITING, VehicleState.FAULT},
    VehicleState.CHARGING: {VehicleState.IDLE, VehicleState.FAULT},
    VehicleState.FAULT: {VehicleState.STOPPED},
    VehicleState.STOPPED: set(),
}

# 任务对象
@dataclass
class TransportTask:
    task_id: str
    release_time_ms: int
    pickup_node_id: str
    dropoff_node_id: str
    required_robot_group: str
    payload_type: str
    payload_id: str | None
    pickup_service_ms: int
    dropoff_service_ms: int
    priority_class: int = 0
    due_time_ms: int | None = None
    state: TaskState = TaskState.QUEUED
    revision: int = 0
    assigned_vehicle_id: str | None = None
    assigned_at_ms: int | None = None
    picked_at_ms: int | None = None
    completed_at_ms: int | None = None
    failure_reason: str | None = None

    # 受约束的状态切换
    def transition(self, new_state: TaskState, at_ms: int) -> None:
        if new_state not in TASK_TRANSITIONS[self.state]:
            raise DomainError(
                "task.transition.invalid",
                f"task {self.task_id!r} cannot transition from {self.state.value} "
                f"to {new_state.value}",
            )
        self.state = new_state
        self.revision += 1
        if new_state is TaskState.ASSIGNED:
            self.assigned_at_ms = at_ms    # 记录"何时分配"
        elif new_state is TaskState.EN_ROUTE_DROPOFF:
            self.picked_at_ms = at_ms      # 记录"何时取完货"
        elif new_state is TaskState.COMPLETED:
            self.completed_at_ms = at_ms   # 记录"何时完成"

    # 与 JSON 互转
    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        default_pickup_service_ms: int,
        default_dropoff_service_ms: int,
    ) -> TransportTask:
        return cls(
            task_id=value["taskId"],
            revision=int(value.get("revision", 0)),
            release_time_ms=int(value["releaseTimeMs"]),
            pickup_node_id=value["pickupNodeId"],
            dropoff_node_id=value["dropoffNodeId"],
            required_robot_group=value["requiredRobotGroup"],
            payload_type=value["payloadType"],
            payload_id=value.get("payloadId"),
            pickup_service_ms=int(
                value.get("pickupServiceMs") or default_pickup_service_ms
            ),
            dropoff_service_ms=int(
                value.get("dropoffServiceMs") or default_dropoff_service_ms
            ),
            priority_class=int(value.get("priorityClass", 0)),
            due_time_ms=value.get("dueTimeMs"),
            state=TaskState(value.get("state", TaskState.QUEUED.value)),
            assigned_vehicle_id=value.get("assignedVehicleId"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "revision": self.revision,
            "releaseTimeMs": self.release_time_ms,
            "pickupNodeId": self.pickup_node_id,
            "dropoffNodeId": self.dropoff_node_id,
            "requiredRobotGroup": self.required_robot_group,
            "payloadType": self.payload_type,
            "payloadId": self.payload_id,
            "state": self.state.value,
            "assignedVehicleId": self.assigned_vehicle_id,
            "assignedAtMs": self.assigned_at_ms,
            "pickedAtMs": self.picked_at_ms,
            "completedAtMs": self.completed_at_ms,
            "failureReason": self.failure_reason,
        }

# 车辆对象
@dataclass
class Vehicle:
    vehicle_id: str
    robot_group: str
    current_node_id: str | None
    heading_rad: float
    load_state: LoadState
    payload_id: str | None = None
    capabilities: frozenset[str] = frozenset()
    revision: int = 0
    state: VehicleState = VehicleState.IDLE
    current_edge_id: str | None = None
    active_task_id: str | None = None
    plan_id: str | None = None
    plan_revision: int | None = None
    committed_until_ms: int = 0
    available_at_ms: int = 0
    fault_code: str | None = None
    state_changed_at_ms: int = 0
    state_durations_ms: Counter[str] = field(default_factory=Counter)
    waiting_resume_state: VehicleState | None = None
    rotation_resume_state: VehicleState | None = None

    def transition(self, new_state: VehicleState, at_ms: int) -> None:
        if at_ms < self.state_changed_at_ms:
            raise DomainError(
                "vehicle.transition.time_reversed",
                f"vehicle {self.vehicle_id!r} transition time moved backwards",
            )
        if new_state not in VEHICLE_TRANSITIONS[self.state]:
            raise DomainError(
                "vehicle.transition.invalid",
                f"vehicle {self.vehicle_id!r} cannot transition from {self.state.value} "
                f"to {new_state.value}",
            )
        self.state_durations_ms[self.state.value] += at_ms - self.state_changed_at_ms
        self.state = new_state
        self.state_changed_at_ms = at_ms
        self.revision += 1

    def durations_at(self, at_ms: int) -> dict[str, int]:
        result = Counter(self.state_durations_ms)
        result[self.state.value] += at_ms - self.state_changed_at_ms
        return dict(sorted(result.items()))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Vehicle:
        return cls(
            vehicle_id=value["vehicleId"],
            robot_group=value["robotGroup"],
            current_node_id=value["initialNodeId"],
            heading_rad=float(value["initialHeadingRad"]),
            load_state=LoadState(value["initialLoadState"]),
            payload_id=value.get("payloadId"),
            capabilities=frozenset(value.get("capabilities", [])),
        )

    def to_dict(self, at_ms: int) -> dict[str, Any]:
        return {
            "vehicleId": self.vehicle_id,
            "revision": self.revision,
            "robotGroup": self.robot_group,
            "state": self.state.value,
            "currentNodeId": self.current_node_id,
            "currentEdgeId": self.current_edge_id,
            "headingRad": self.heading_rad,
            "loadState": self.load_state.value,
            "payloadId": self.payload_id,
            "activeTaskId": self.active_task_id,
            "planId": self.plan_id,
            "planRevision": self.plan_revision,
            "committedUntilMs": self.committed_until_ms,
            "availableAtMs": self.available_at_ms,
            "faultCode": self.fault_code,
            "stateDurationsMs": self.durations_at(at_ms),
        }

# PlanSegment
@dataclass(frozen=True)
class PlanSegment:
    segment_id: str
    kind: SegmentKind
    start_ms: int
    end_ms: int
    start_node_id: str | None
    end_node_id: str | None
    edge_id: str | None
    expected_load_state: LoadState
    resource_ids: tuple[str, ...] = ()
    command_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlanSegment:
        return cls(
            segment_id=value["id"],
            kind=SegmentKind(value["kind"]),
            start_ms=int(value["startMs"]),
            end_ms=int(value["endMs"]),
            start_node_id=value.get("startNodeId"),
            end_node_id=value.get("endNodeId"),
            edge_id=value.get("edgeId"),
            expected_load_state=LoadState(value["expectedLoadState"]),
            resource_ids=tuple(sorted(set(value.get("resourceIds", [])))),
            command_payload=dict(value.get("commandPayload", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.segment_id,
            "kind": self.kind.value,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "startNodeId": self.start_node_id,
            "endNodeId": self.end_node_id,
            "expectedLoadState": self.expected_load_state.value,
            "resourceIds": list(self.resource_ids),
        }
        if self.edge_id is not None:
            result["edgeId"] = self.edge_id
        if self.command_payload:
            result["commandPayload"] = self.command_payload
        return result

# 一辆车的完整计划
@dataclass(frozen=True)
class VehiclePlan:
    plan_id: str
    revision: int
    vehicle_id: str
    task_id: str
    based_on_vehicle_revision: int
    based_on_world_revision: int
    created_at_ms: int
    horizon_end_ms: int
    committed_until_ms: int
    segments: tuple[PlanSegment, ...]
    continuation: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VehiclePlan:
        return cls(
            plan_id=value["id"],
            revision=int(value["revision"]),
            vehicle_id=value["vehicleId"],
            task_id=value["taskId"],
            based_on_vehicle_revision=int(value["basedOnVehicleRevision"]),
            based_on_world_revision=int(value["basedOnWorldRevision"]),
            created_at_ms=int(value["createdAtMs"]),
            horizon_end_ms=int(value["horizonEndMs"]),
            committed_until_ms=int(value["committedUntilMs"]),
            segments=tuple(PlanSegment.from_dict(item) for item in value["segments"]),
            continuation=bool(value.get("continuation", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.plan_id,
            "revision": self.revision,
            "vehicleId": self.vehicle_id,
            "taskId": self.task_id,
            "basedOnVehicleRevision": self.based_on_vehicle_revision,
            "basedOnWorldRevision": self.based_on_world_revision,
            "createdAtMs": self.created_at_ms,
            "horizonEndMs": self.horizon_end_ms,
            "committedUntilMs": self.committed_until_ms,
            "continuation": self.continuation,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def projected_vehicle_revision(plan: VehiclePlan) -> int:
    """Return the vehicle revision after deterministic execution of a full plan."""
    dropoff_index = next(
        (
            index
            for index, segment in enumerate(plan.segments)
            if segment.kind is SegmentKind.DROPOFF
        ),
        len(plan.segments) - 1,
    )
    reposition_completion = int(dropoff_index < len(plan.segments) - 1)
    return (
        plan.based_on_vehicle_revision
        + int(not plan.continuation)
        + 2 * len(plan.segments)
        + reposition_completion
    )
