from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .domain import LoadState, TaskState, TransportTask, Vehicle, VehicleState
from .routing import RouteProvider
from .topology import MapTopology


@dataclass(frozen=True)
class AssignmentCost:
    empty_travel_ms: int
    loaded_travel_ms: int
    pickup_service_ms: int
    dropoff_service_ms: int
    due_time_penalty_ms: int
    task_age_credit_ms: int
    priority_credit_ms: int

    # 总耗时 = 空载行驶 + 载货行驶 + 取货服务 + 放货服务 + 逾期惩罚 - 任务老化奖励 - 优先级奖励
    @property
    def total_ms(self) -> int:
        return (
            self.empty_travel_ms
            + self.loaded_travel_ms
            + self.pickup_service_ms
            + self.dropoff_service_ms
            + self.due_time_penalty_ms
            - self.task_age_credit_ms
            - self.priority_credit_ms
        )

# 分配记录
@dataclass(frozen=True)
class AssignmentProposal:
    vehicle_id: str
    task_id: str
    cost: AssignmentCost


class TaskAllocator:
    def __init__(
        self,
        topology: MapTopology,
        routes: RouteProvider,
        config: dict[str, Any],
    ) -> None:
        self.topology = topology
        self.routes = routes
        self.age_credit_per_second_ms = int(
            config.get("taskAgeCreditPerSecondMs", 100)
        )
        self.priority_credit_ms = int(config.get("priorityClassCreditMs", 5000))
        self.due_time_penalty_factor = int(config.get("dueTimePenaltyFactor", 2))

    def compatible_cost(
        self,
        vehicle: Vehicle,
        task: TransportTask,
        now_ms: int,
    ) -> AssignmentCost | None:
        # 兼容性过滤
        if (
            vehicle.state is not VehicleState.IDLE
            or vehicle.load_state is not LoadState.EMPTY
            or vehicle.current_node_id is None
            or vehicle.fault_code is not None
            or vehicle.robot_group != task.required_robot_group
            or task.state is not TaskState.QUEUED
            or task.release_time_ms > now_ms
        ):
            return None
        try:
            self.topology.validate_task(task)
        except ValueError:
            return None
        # 车必须能开到取货点、且从取货点能开到放货点，否则没资格
        empty_travel = self.routes.shortest_travel_ms(
            vehicle.robot_group,
            vehicle.current_node_id,
            task.pickup_node_id,
            LoadState.EMPTY,
        )
        loaded_travel = self.routes.shortest_travel_ms(
            vehicle.robot_group,
            task.pickup_node_id,
            task.dropoff_node_id,
            LoadState.LOADED,
        )
        if empty_travel is None or loaded_travel is None:
            return None
        # 算超时罚分
        estimated_completion = (
            now_ms
            + empty_travel
            + task.pickup_service_ms
            + loaded_travel
            + task.dropoff_service_ms
        )
        lateness = (
            max(0, estimated_completion - task.due_time_ms)
            if task.due_time_ms is not None
            else 0
        )
        # 打包成本
        return AssignmentCost(
            empty_travel_ms=empty_travel,
            loaded_travel_ms=loaded_travel,
            pickup_service_ms=task.pickup_service_ms,
            dropoff_service_ms=task.dropoff_service_ms,
            due_time_penalty_ms=lateness * self.due_time_penalty_factor,
            task_age_credit_ms=(
                max(0, now_ms - task.release_time_ms)
                // 1000
                * self.age_credit_per_second_ms
            ),
            priority_credit_ms=task.priority_class * self.priority_credit_ms,
        )

    def continuation_cost(
        self,
        vehicle: Vehicle,
        task: TransportTask,
        now_ms: int,
    ) -> AssignmentCost | None:
        """Estimate the remaining cost without changing an active assignment."""

        if (
            vehicle.current_node_id is None
            or vehicle.fault_code is not None
            or vehicle.active_task_id != task.task_id
            or task.assigned_vehicle_id != vehicle.vehicle_id
            or vehicle.robot_group != task.required_robot_group
        ):
            return None
        if task.state is TaskState.EN_ROUTE_PICKUP:
            if vehicle.load_state is not LoadState.EMPTY:
                return None
            empty_travel = self.routes.shortest_travel_ms(
                vehicle.robot_group,
                vehicle.current_node_id,
                task.pickup_node_id,
                LoadState.EMPTY,
            )
            loaded_travel = self.routes.shortest_travel_ms(
                vehicle.robot_group,
                task.pickup_node_id,
                task.dropoff_node_id,
                LoadState.LOADED,
            )
            pickup_service_ms = task.pickup_service_ms
        elif task.state is TaskState.EN_ROUTE_DROPOFF:
            if vehicle.load_state is not LoadState.LOADED:
                return None
            empty_travel = 0
            loaded_travel = self.routes.shortest_travel_ms(
                vehicle.robot_group,
                vehicle.current_node_id,
                task.dropoff_node_id,
                LoadState.LOADED,
            )
            pickup_service_ms = 0
        else:
            return None
        if empty_travel is None or loaded_travel is None:
            return None
        estimated_completion = (
            now_ms
            + empty_travel
            + pickup_service_ms
            + loaded_travel
            + task.dropoff_service_ms
        )
        lateness = (
            max(0, estimated_completion - task.due_time_ms)
            if task.due_time_ms is not None
            else 0
        )
        return AssignmentCost(
            empty_travel_ms=empty_travel,
            loaded_travel_ms=loaded_travel,
            pickup_service_ms=pickup_service_ms,
            dropoff_service_ms=task.dropoff_service_ms,
            due_time_penalty_ms=lateness * self.due_time_penalty_factor,
            task_age_credit_ms=(
                max(0, now_ms - task.release_time_ms)
                // 1000
                * self.age_credit_per_second_ms
            ),
            priority_credit_ms=task.priority_class * self.priority_credit_ms,
        )

    # 全局最优配对
    def assign(
        self,
        vehicles: list[Vehicle],
        tasks: list[TransportTask],
        now_ms: int,
        excluded_pairs: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[AssignmentProposal, ...]:
        candidates: dict[tuple[str, str], AssignmentCost] = {}
        # 算出所有合格配对
        for vehicle in sorted(vehicles, key=lambda item: item.vehicle_id):
            for task in sorted(tasks, key=lambda item: item.task_id):
                if (vehicle.vehicle_id, task.task_id) in excluded_pairs:
                    continue
                cost = self.compatible_cost(vehicle, task, now_ms)
                if cost is not None:
                    candidates[(vehicle.vehicle_id, task.task_id)] = cost
        if not candidates:
            return ()

        # 建立二分图，变成流量问题
        minimum = min(cost.total_ms for cost in candidates.values())
        offset = max(0, -minimum)
        source, sink = ("source",), ("sink",)
        graph = nx.DiGraph()
        for vehicle_id in sorted({key[0] for key in candidates}):
            vehicle_node = ("vehicle", vehicle_id)
            graph.add_edge(source, vehicle_node, capacity=1, weight=0)
        for task_id in sorted({key[1] for key in candidates}):
            task_node = ("task", task_id)
            graph.add_edge(task_node, sink, capacity=1, weight=0)
        for (vehicle_id, task_id), cost in sorted(candidates.items()):
            graph.add_edge(
                ("vehicle", vehicle_id),
                ("task", task_id),
                capacity=1,
                weight=cost.total_ms + offset,
            )

        # 用 networkx 的最小费用最大流算法
        flow = nx.max_flow_min_cost(graph, source, sink, capacity="capacity", weight="weight")
        proposals = [
            AssignmentProposal(vehicle_id, task_id, candidates[(vehicle_id, task_id)])
            for vehicle_id, task_id in sorted(candidates)
            if flow.get(("vehicle", vehicle_id), {}).get(("task", task_id), 0) == 1
        ]
        return tuple(sorted(proposals, key=lambda item: (item.vehicle_id, item.task_id)))
