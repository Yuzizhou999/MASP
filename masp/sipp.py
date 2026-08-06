from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from .domain import (
    DomainError,
    LoadState,
    PlanSegment,
    SegmentKind,
    TransportTask,
    Vehicle,
    VehiclePlan,
)
from .motion import EdgeTravelTimeModel
from .reservations import RelativeReservationRequest, ReservationTable
from .routing import RouteProvider, SpatialRoute
from .topology import MapTopology
from .zones import TrafficZone

# SIPP 规划器的异常类，带有一个可选的建议延迟时间（毫秒）
class SippPlanningError(DomainError):
    def __init__(self, code: str, message: str, suggested_delay_ms: int = 0) -> None:
        super().__init__(code, message)
        self.suggested_delay_ms = suggested_delay_ms

# SIPP 规划器的诊断信息
@dataclass(frozen=True)
class SippDiagnostics:
    route_combinations_tried: int
    schedule_attempts: int
    inserted_wait_ms: int
    completion_time_ms: int

# 规划结果 = 计划 + 诊断
@dataclass(frozen=True)
class PlannedTask:
    plan: VehiclePlan
    diagnostics: SippDiagnostics


class ContinuousTimeSippPlanner:
    def __init__(
        self,
        topology: MapTopology,
        routes: RouteProvider,
        travel_times: EdgeTravelTimeModel,
        scheduler: dict[str, Any],
        recovery_node_ids: Iterable[str],
    ) -> None:
        planner = scheduler["planner"]
        wait = scheduler["traffic"]["wait"]
        self.topology = topology
        self.routes = routes
        self.travel_times = travel_times
        self.route_limit = int(planner.get("candidateRouteCount", 3))
        self.time_quantum_ms = int(planner.get("timeQuantumMs", 100))
        self.max_schedule_attempts = int(planner.get("maxSippScheduleAttempts", 200))
        self.max_wait_ms = int(wait["maxPlannedWaitMs"])
        self.recovery_node_ids = tuple(sorted(set(recovery_node_ids)))

    def plan_task(
        self,
        vehicle: Vehicle,
        task: TransportTask,
        now_ms: int,
        horizon_end_ms: int,
        reservations: ReservationTable,
        plan_number: int,
    ) -> PlannedTask:
        # 要求车辆停在节点上，而不是在边上
        if vehicle.current_node_id is None:
            raise SippPlanningError(
                "sipp.vehicle.on_edge", "SIPP requires a vehicle parked at a node"
            )
        # 分别为空载、载货、恢复路线生成候选路线
        empty_routes = self.routes.candidate_routes(
            vehicle.robot_group,
            vehicle.current_node_id,
            task.pickup_node_id,
            LoadState.EMPTY,
            self.route_limit,
        )
        loaded_routes = self.routes.candidate_routes(
            vehicle.robot_group,
            task.pickup_node_id,
            task.dropoff_node_id,
            LoadState.LOADED,
            self.route_limit,
        )
        recovery_routes = self._recovery_routes(vehicle.robot_group, task.dropoff_node_id)
        if not empty_routes or not loaded_routes or not recovery_routes:
            raise SippPlanningError(
                "sipp.spatial_route.missing",
                f"task {task.task_id!r} has no complete pickup/dropoff/recovery route",
            )

        # 路线和时间组合尝试，找出最优的计划
        best: tuple[tuple[Any, ...], tuple[PlanSegment, ...], int, int] | None = None
        combinations_tried = 0
        total_attempts = 0
        for empty_route, loaded_route, recovery_route in product(
            empty_routes, loaded_routes, recovery_routes
        ):
            combinations_tried += 1
            shift_ms = 0
            for _ in range(self.max_schedule_attempts):
                total_attempts += 1
                try:
                    segments = self._schedule_combination(
                        vehicle,
                        task,
                        now_ms,
                        shift_ms,
                        horizon_end_ms,
                        reservations,
                        empty_route,
                        loaded_route,
                        recovery_route,
                    )
                    completion = segments[-1].end_ms
                    wait_ms = sum(
                        segment.end_ms - segment.start_ms
                        for segment in segments
                        if segment.kind is SegmentKind.WAIT
                    )
                    key = (
                        completion,
                        wait_ms,
                        empty_route.edge_ids,
                        loaded_route.edge_ids,
                        recovery_route.edge_ids,
                    )
                    if best is None or key < best[0]:
                        best = (key, segments, wait_ms, completion)
                    break
                except SippPlanningError as error:
                    delay = max(self.time_quantum_ms, error.suggested_delay_ms)
                    shift_ms += (
                        (delay + self.time_quantum_ms - 1)
                        // self.time_quantum_ms
                        * self.time_quantum_ms
                    )
                    if (
                        shift_ms > self.max_wait_ms
                        or not self.topology.wait_allowed(
                            vehicle.current_node_id, vehicle.robot_group
                        )
                    ):
                        break

        if best is None:
            raise SippPlanningError(
                "sipp.no_schedule",
                f"task {task.task_id!r} has no conflict-free schedule within the horizon",
            )
        _, segments, wait_ms, completion = best
        plan = VehiclePlan(
            plan_id=f"auto-plan:{vehicle.vehicle_id}:{plan_number:04d}:{task.task_id}",
            revision=plan_number,
            vehicle_id=vehicle.vehicle_id,
            task_id=task.task_id,
            based_on_vehicle_revision=vehicle.revision,
            based_on_world_revision=0,
            created_at_ms=now_ms,
            horizon_end_ms=horizon_end_ms,
            committed_until_ms=completion,
            segments=segments,
        )
        return PlannedTask(
            plan=plan,
            diagnostics=SippDiagnostics(
                route_combinations_tried=combinations_tried,
                schedule_attempts=total_attempts,
                inserted_wait_ms=wait_ms,
                completion_time_ms=completion,
            ),
        )

    def schedule_route_intent(
        self,
        vehicle: Vehicle,
        route: SpatialRoute,
        ready_ms: int,
        load_state: LoadState,
        reservations: ReservationTable,
        horizon_end_ms: int,
    ) -> tuple[PlanSegment, ...]:
        """Schedule a fixed spatial route with the same safety rules as task planning."""
        if vehicle.current_node_id is None:
            raise SippPlanningError(
                "sipp.vehicle.on_edge", "route intent requires a vehicle parked at a node"
            )
        if route.start_node_id != vehicle.current_node_id:
            raise SippPlanningError(
                "sipp.route_intent.start",
                f"route starts at {route.start_node_id!r}, vehicle is at "
                f"{vehicle.current_node_id!r}",
            )
        expected_node_id = route.start_node_id
        for edge_id in route.edge_ids:
            edge = self.topology.edges.get(edge_id)
            if edge is None:
                raise SippPlanningError(
                    "sipp.route_intent.edge_missing",
                    f"route references unknown edge {edge_id!r}",
                )
            if edge["robotGroup"] != vehicle.robot_group:
                raise SippPlanningError(
                    "sipp.route_intent.robot_group",
                    f"edge {edge_id!r} belongs to robot group {edge['robotGroup']!r}, "
                    f"vehicle belongs to {vehicle.robot_group!r}",
                )
            if edge["start"] != expected_node_id:
                raise SippPlanningError(
                    "sipp.route_intent.discontinuous",
                    f"route is discontinuous at edge {edge_id!r}: expected start "
                    f"{expected_node_id!r}, got {edge['start']!r}",
                )
            expected_node_id = edge["end"]
        if expected_node_id != route.end_node_id:
            raise SippPlanningError(
                "sipp.route_intent.end",
                f"route ends at {expected_node_id!r}, declared end is {route.end_node_id!r}",
            )
        segments: list[PlanSegment] = []
        self._schedule_route(
            segments,
            vehicle,
            route,
            ready_ms,
            load_state,
            reservations,
            horizon_end_ms,
        )
        return tuple(segments)

    # 生成恢复路线
    def _recovery_routes(
        self, robot_group: str, dropoff_node_id: str
    ) -> tuple[SpatialRoute, ...]:
        candidates: list[SpatialRoute] = []
        for node_id in self.recovery_node_ids:
            node = self.topology.nodes.get(node_id)
            if node is None or robot_group not in node["allowedRobotGroups"]:
                continue
            candidates.extend(
                self.routes.candidate_routes(
                    robot_group,
                    dropoff_node_id,
                    node_id,
                    LoadState.EMPTY,
                    limit=1,
                )
            )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (item.free_flow_travel_ms, item.edge_ids, item.end_node_id),
            )[: self.route_limit]
        )

    # 规划一组路线和时间组合，返回计划段
    def _schedule_combination(
        self,
        vehicle: Vehicle,
        task: TransportTask,
        now_ms: int,
        shift_ms: int,
        horizon_end_ms: int,
        reservations: ReservationTable,
        empty_route: SpatialRoute,
        loaded_route: SpatialRoute,
        recovery_route: SpatialRoute,
    ) -> tuple[PlanSegment, ...]:
        segments: list[PlanSegment] = []
        current_ms = now_ms
        if shift_ms:
            shifted = now_ms + shift_ms
            self._append_wait(
                segments,
                vehicle.vehicle_id,
                vehicle.robot_group,
                vehicle.current_node_id or "",
                current_ms,
                shifted,
                LoadState.EMPTY,
                reservations,
            )
            current_ms = shifted

        current_ms = self._schedule_route(
            segments,
            vehicle,
            empty_route,
            current_ms,
            LoadState.EMPTY,
            reservations,
            horizon_end_ms,
        )
        current_ms = self._append_service(
            segments,
            vehicle,
            task.pickup_node_id,
            current_ms,
            task.pickup_service_ms,
            SegmentKind.PICKUP,
            LoadState.EMPTY,
            reservations,
            horizon_end_ms,
        )
        current_ms = self._schedule_route(
            segments,
            vehicle,
            loaded_route,
            current_ms,
            LoadState.LOADED,
            reservations,
            horizon_end_ms,
        )
        current_ms = self._append_service(
            segments,
            vehicle,
            task.dropoff_node_id,
            current_ms,
            task.dropoff_service_ms,
            SegmentKind.DROPOFF,
            LoadState.LOADED,
            reservations,
            horizon_end_ms,
        )
        self._schedule_route(
            segments,
            vehicle,
            recovery_route,
            current_ms,
            LoadState.EMPTY,
            reservations,
            horizon_end_ms,
        )
        return tuple(segments)

    # 先问这条边及它关联的冲突资源啥时候全空闲，再决定几点出发
    def _schedule_route(
        self,
        segments: list[PlanSegment],
        vehicle: Vehicle,
        route: SpatialRoute,
        ready_ms: int,
        load_state: LoadState,
        reservations: ReservationTable,
        horizon_end_ms: int,
    ) -> int:
        current_node_id = route.start_node_id
        current_ms = ready_ms
        edge_ids = route.edge_ids
        edge_index = 0
        while edge_index < len(edge_ids):
            edge_id = edge_ids[edge_index]
            entry_zone = self.topology.traffic_zones.entry_zone_for_edge(edge_id)
            if entry_zone is not None:
                atomic_end = self._zone_atomic_end(
                    edge_ids,
                    edge_index,
                    entry_zone,
                    vehicle.robot_group,
                )
                atomic_edge_ids = edge_ids[edge_index : atomic_end + 1]
                current_ms = self._schedule_atomic_edges(
                    segments,
                    vehicle,
                    atomic_edge_ids,
                    current_node_id,
                    current_ms,
                    load_state,
                    reservations,
                    horizon_end_ms,
                )
                current_node_id = self.topology.edges[atomic_edge_ids[-1]]["end"]
                edge_index = atomic_end + 1
                continue

            controlled_zone = self.topology.traffic_zones.zone_for_edge(edge_id)
            if controlled_zone is not None:
                raise SippPlanningError(
                    "sipp.zone.entry_missing",
                    f"route reaches zone {controlled_zone.zone_id!r} through "
                    f"non-entry edge {edge_id!r}",
                )
            current_ms = self._schedule_single_edge(
                segments,
                vehicle,
                edge_id,
                current_node_id,
                current_ms,
                load_state,
                reservations,
                horizon_end_ms,
            )
            current_node_id = self.topology.edges[edge_id]["end"]
            edge_index += 1
        return current_ms

    def _zone_atomic_end(
        self,
        edge_ids: tuple[str, ...],
        entry_index: int,
        zone: TrafficZone,
        robot_group: str,
    ) -> int:
        exited = False
        for index in range(entry_index, len(edge_ids)):
            edge_id = edge_ids[index]
            edge = self.topology.edges[edge_id]
            if not exited:
                if not self.topology.traffic_zones.edge_is_controlled_by(edge_id, zone):
                    raise SippPlanningError(
                        "sipp.zone.exit_missing",
                        f"route leaves zone {zone.zone_id!r} without a configured exit edge",
                    )
                exited = edge_id in zone.exit_edge_ids
            elif self.topology.traffic_zones.entry_zone_for_edge(edge_id) is not None:
                raise SippPlanningError(
                    "sipp.zone.safe_node_missing",
                    f"route enters another zone before finding a safe node after {zone.zone_id!r}",
                )

            end_node_id = edge["end"]
            if (
                exited
                and self.topology.wait_allowed(end_node_id, robot_group)
                and self.topology.traffic_zones.zone_for_node(end_node_id) is None
            ):
                return index
        code = "sipp.zone.safe_node_missing" if exited else "sipp.zone.exit_missing"
        raise SippPlanningError(
            code,
            f"route through zone {zone.zone_id!r} has no reachable outside safe waiting node",
        )

    def _schedule_single_edge(
        self,
        segments: list[PlanSegment],
        vehicle: Vehicle,
        edge_id: str,
        current_node_id: str,
        current_ms: int,
        load_state: LoadState,
        reservations: ReservationTable,
        horizon_end_ms: int,
    ) -> int:
        edge = self.topology.edges[edge_id]
        duration_ms = self.travel_times.duration_ms(edge, load_state)
        probe = PlanSegment(
            segment_id="probe",
            kind=SegmentKind.TRAVERSE,
            start_ms=current_ms,
            end_ms=current_ms + duration_ms,
            start_node_id=edge["start"],
            end_node_id=edge["end"],
            edge_id=edge_id,
            expected_load_state=load_state,
        )
        resources = self.topology.derived_resources(probe)
        departure_ms = self._first_common_start(
            resources,
            current_ms,
            duration_ms,
            vehicle.vehicle_id,
            reservations,
        )
        if departure_ms > current_ms:
            self._append_wait(
                segments,
                vehicle.vehicle_id,
                vehicle.robot_group,
                current_node_id,
                current_ms,
                departure_ms,
                load_state,
                reservations,
            )
        arrival_ms = departure_ms + duration_ms
        if arrival_ms > horizon_end_ms:
            raise SippPlanningError(
                "sipp.horizon.exceeded", "route exceeds planning horizon"
            )
        segments.append(
            PlanSegment(
                segment_id=f"segment-{len(segments):04d}",
                kind=SegmentKind.TRAVERSE,
                start_ms=departure_ms,
                end_ms=arrival_ms,
                start_node_id=edge["start"],
                end_node_id=edge["end"],
                edge_id=edge_id,
                expected_load_state=load_state,
                resource_ids=resources,
            )
        )
        return arrival_ms

    def _schedule_atomic_edges(
        self,
        segments: list[PlanSegment],
        vehicle: Vehicle,
        edge_ids: tuple[str, ...],
        current_node_id: str,
        ready_ms: int,
        load_state: LoadState,
        reservations: ReservationTable,
        horizon_end_ms: int,
    ) -> int:
        offset_ms = 0
        scheduled: list[tuple[dict[str, Any], int, int, tuple[str, ...]]] = []
        requests: list[RelativeReservationRequest] = []
        expected_node_id = current_node_id
        for edge_id in edge_ids:
            edge = self.topology.edges[edge_id]
            if edge["start"] != expected_node_id:
                raise SippPlanningError(
                    "sipp.zone.route_discontinuous",
                    f"atomic zone route is discontinuous at edge {edge_id!r}",
                )
            duration_ms = self.travel_times.duration_ms(edge, load_state)
            probe = PlanSegment(
                segment_id="probe",
                kind=SegmentKind.TRAVERSE,
                start_ms=offset_ms,
                end_ms=offset_ms + duration_ms,
                start_node_id=edge["start"],
                end_node_id=edge["end"],
                edge_id=edge_id,
                expected_load_state=load_state,
            )
            resources = self.topology.derived_resources(probe)
            scheduled.append((edge, offset_ms, offset_ms + duration_ms, resources))
            requests.extend(
                RelativeReservationRequest(resource_id, offset_ms, offset_ms + duration_ms)
                for resource_id in resources
            )
            offset_ms += duration_ms
            expected_node_id = edge["end"]

        availability = reservations.first_available_bundle_start(
            requests,
            ready_ms,
            vehicle_id=vehicle.vehicle_id,
        )
        if availability.start_ms > ready_ms:
            self._append_wait(
                segments,
                vehicle.vehicle_id,
                vehicle.robot_group,
                current_node_id,
                ready_ms,
                availability.start_ms,
                load_state,
                reservations,
            )
        completion_ms = availability.start_ms + offset_ms
        if completion_ms > horizon_end_ms:
            raise SippPlanningError(
                "sipp.horizon.exceeded", "atomic zone traversal exceeds planning horizon"
            )
        for edge, start_offset, end_offset, resources in scheduled:
            segments.append(
                PlanSegment(
                    segment_id=f"segment-{len(segments):04d}",
                    kind=SegmentKind.TRAVERSE,
                    start_ms=availability.start_ms + start_offset,
                    end_ms=availability.start_ms + end_offset,
                    start_node_id=edge["start"],
                    end_node_id=edge["end"],
                    edge_id=edge["id"],
                    expected_load_state=load_state,
                    resource_ids=resources,
                )
            )
        return completion_ms

    # 排服务，先找服务资源的空闲时刻。
    # 如果"到达时服务点还被占着"——不在 AP 上等，直接抛错并建议延迟
    def _append_service(
        self,
        segments: list[PlanSegment],
        vehicle: Vehicle,
        node_id: str,
        ready_ms: int,
        duration_ms: int,
        kind: SegmentKind,
        load_state: LoadState,
        reservations: ReservationTable,
        horizon_end_ms: int,
    ) -> int:
        probe = PlanSegment(
            segment_id="probe",
            kind=kind,
            start_ms=ready_ms,
            end_ms=ready_ms + duration_ms,
            start_node_id=node_id,
            end_node_id=node_id,
            edge_id=None,
            expected_load_state=load_state,
        )
        resources = self.topology.derived_resources(probe)
        start_ms = self._first_common_start(
            resources,
            ready_ms,
            duration_ms,
            vehicle.vehicle_id,
            reservations,
        )
        if start_ms != ready_ms:
            raise SippPlanningError(
                "sipp.ap_wait.disallowed",
                f"service at {node_id!r} is not available on arrival",
                suggested_delay_ms=start_ms - ready_ms,
            )
        end_ms = start_ms + duration_ms
        if end_ms > horizon_end_ms:
            raise SippPlanningError(
                "sipp.horizon.exceeded", "service exceeds planning horizon"
            )
        segments.append(
            PlanSegment(
                segment_id=f"segment-{len(segments):04d}",
                kind=kind,
                start_ms=start_ms,
                end_ms=end_ms,
                start_node_id=node_id,
                end_node_id=node_id,
                edge_id=None,
                expected_load_state=load_state,
                resource_ids=resources,
            )
        )
        return end_ms

    # 插入一个合法等待段
    def _append_wait(
        self,
        segments: list[PlanSegment],
        vehicle_id: str,
        robot_group: str,
        node_id: str,
        start_ms: int,
        end_ms: int,
        load_state: LoadState,
        reservations: ReservationTable,
    ) -> None:
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            return
        if duration_ms > self.max_wait_ms:
            raise SippPlanningError(
                "sipp.wait.too_long", "planned wait exceeds maxPlannedWaitMs"
            )
        if not self.topology.wait_allowed(node_id, robot_group):
            raise SippPlanningError(
                "sipp.wait.disallowed",
                f"node {node_id!r} does not permit planned waiting",
                suggested_delay_ms=duration_ms,
            )
        resource_id = f"node:{node_id}"
        if not reservations.is_available(
            resource_id, start_ms, end_ms, vehicle_id=vehicle_id
        ):
            raise SippPlanningError(
                "sipp.wait.interval_occupied",
                f"node {node_id!r} is unavailable for the required wait",
                suggested_delay_ms=self.time_quantum_ms,
            )
        segments.append(
            PlanSegment(
                segment_id=f"segment-{len(segments):04d}",
                kind=SegmentKind.WAIT,
                start_ms=start_ms,
                end_ms=end_ms,
                start_node_id=node_id,
                end_node_id=node_id,
                edge_id=None,
                expected_load_state=load_state,
                resource_ids=(resource_id,),
            )
        )

    # 给定"一批资源 + 要占的时长"，返回一个让所有资源都能同时容纳的最早开始时刻
    @staticmethod
    def _first_common_start(
        resource_ids: Iterable[str],
        not_before_ms: int,
        duration_ms: int,
        vehicle_id: str,
        reservations: ReservationTable,
    ) -> int:
        candidate = not_before_ms
        resources = tuple(resource_ids)
        for _ in range(10_000):
            next_candidate = max(
                (
                    reservations.first_available_start(
                        resource_id,
                        candidate,
                        duration_ms,
                        vehicle_id=vehicle_id,
                    )
                    for resource_id in resources
                ),
                default=candidate,
            )
            if next_candidate == candidate:
                return candidate
            candidate = next_candidate
        raise SippPlanningError(
            "sipp.common_interval.limit", "common resource interval search did not converge"
        )
