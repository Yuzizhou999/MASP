from __future__ import annotations

import time
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
    route_combinations_pruned: int
    schedule_attempts: int
    inserted_wait_ms: int
    completion_time_ms: int
    route_expansion_level: int
    deadline_exhausted: bool

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
        self.progressive_route_search = bool(
            planner.get("progressiveRouteSearch", True)
        )
        self.route_expansion_wait_threshold_ms = max(
            0, int(planner.get("routeExpansionWaitThresholdMs", 5000))
        )
        self.branch_and_bound_enabled = bool(
            planner.get("routeBranchAndBound", True)
        )
        self.time_quantum_ms = int(planner.get("timeQuantumMs", 100))
        self.max_schedule_attempts = int(planner.get("maxSippScheduleAttempts", 200))
        self.max_wait_ms = int(wait["maxPlannedWaitMs"])
        self.recovery_node_ids = tuple(sorted(set(recovery_node_ids)))
        self._recovery_route_cache: dict[
            tuple[str, str, int], tuple[SpatialRoute, ...]
        ] = {}

    def plan_task(
        self,
        vehicle: Vehicle,
        task: TransportTask,
        now_ms: int,
        horizon_end_ms: int,
        reservations: ReservationTable,
        plan_number: int,
        *,
        deadline_ns: int | None = None,
    ) -> PlannedTask:
        # 要求车辆停在节点上，而不是在边上
        if vehicle.current_node_id is None:
            raise SippPlanningError(
                "sipp.vehicle.on_edge", "SIPP requires a vehicle parked at a node"
            )
        self._check_deadline(deadline_ns)
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
        self._check_deadline(deadline_ns)
        if not empty_routes or not loaded_routes or not recovery_routes:
            raise SippPlanningError(
                "sipp.spatial_route.missing",
                f"task {task.task_id!r} has no complete pickup/dropoff/recovery route",
            )

        # 路线和时间组合尝试，找出最优的计划
        best: tuple[tuple[Any, ...], tuple[PlanSegment, ...], int, int] | None = None
        combinations_tried = 0
        combinations_pruned = 0
        total_attempts = 0
        expansion_level = 0
        deadline_exhausted = False
        indexed_combinations = [
            (
                empty_index,
                loaded_index,
                recovery_index,
                empty_route,
                loaded_route,
                recovery_route,
            )
            for (empty_index, empty_route), (loaded_index, loaded_route), (
                recovery_index,
                recovery_route,
            ) in product(
                enumerate(empty_routes),
                enumerate(loaded_routes),
                enumerate(recovery_routes),
            )
        ]
        max_level = max(len(empty_routes), len(loaded_routes), len(recovery_routes))
        levels = range(1, max_level + 1) if self.progressive_route_search else (max_level,)
        stop_search = False
        for level in levels:
            if (
                self.progressive_route_search
                and best is not None
                and best[2] <= self.route_expansion_wait_threshold_ms
            ):
                break
            expansion_level = level
            level_combinations = [
                item
                for item in indexed_combinations
                if (
                    max(item[0], item[1], item[2]) + 1 == level
                    if self.progressive_route_search
                    else True
                )
            ]
            level_combinations.sort(
                key=lambda item: (
                    item[3].free_flow_travel_ms
                    + item[4].free_flow_travel_ms
                    + item[5].free_flow_travel_ms,
                    item[3].edge_ids,
                    item[4].edge_ids,
                    item[5].edge_ids,
                )
            )
            for _, _, _, empty_route, loaded_route, recovery_route in level_combinations:
                if self._deadline_reached(deadline_ns):
                    deadline_exhausted = True
                    stop_search = True
                    break
                lower_bound = (
                    now_ms
                    + empty_route.free_flow_travel_ms
                    + task.pickup_service_ms
                    + loaded_route.free_flow_travel_ms
                    + task.dropoff_service_ms
                    + recovery_route.free_flow_travel_ms
                )
                if (
                    self.branch_and_bound_enabled
                    and best is not None
                    and best[3] < lower_bound
                ):
                    combinations_pruned += 1
                    continue
                combinations_tried += 1
                shift_ms = 0
                observed_shifts: set[int] = set()
                for _ in range(self.max_schedule_attempts):
                    if self._deadline_reached(deadline_ns):
                        deadline_exhausted = True
                        stop_search = True
                        break
                    total_attempts += 1
                    observed_shifts.add(shift_ms)
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
                            deadline_ns=deadline_ns,
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
                        if error.code == "sipp.deadline.exceeded":
                            deadline_exhausted = True
                            stop_search = True
                            break
                        if error.suggested_delay_ms <= 0:
                            break
                        delay = max(self.time_quantum_ms, error.suggested_delay_ms)
                        next_shift_ms = shift_ms + (
                            (delay + self.time_quantum_ms - 1)
                            // self.time_quantum_ms
                            * self.time_quantum_ms
                        )
                        if next_shift_ms in observed_shifts:
                            break
                        shift_ms = next_shift_ms
                        if (
                            shift_ms > self.max_wait_ms
                            or not self.topology.wait_allowed(
                                vehicle.current_node_id, vehicle.robot_group
                            )
                        ):
                            break
                if stop_search:
                    break
            if stop_search:
                break

        if best is None:
            if deadline_exhausted:
                raise SippPlanningError(
                    "sipp.deadline.exceeded",
                    f"task {task.task_id!r} planning exceeded its computation deadline",
                )
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
                route_combinations_pruned=combinations_pruned,
                schedule_attempts=total_attempts,
                inserted_wait_ms=wait_ms,
                completion_time_ms=completion,
                route_expansion_level=expansion_level,
                deadline_exhausted=deadline_exhausted,
            ),
        )

    @staticmethod
    def _deadline_reached(deadline_ns: int | None) -> bool:
        return deadline_ns is not None and time.perf_counter_ns() >= deadline_ns

    @classmethod
    def _check_deadline(cls, deadline_ns: int | None) -> None:
        if cls._deadline_reached(deadline_ns):
            raise SippPlanningError(
                "sipp.deadline.exceeded", "planning computation deadline exceeded"
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
        cache_key = (robot_group, dropoff_node_id, self.route_limit)
        cached = self._recovery_route_cache.get(cache_key)
        if cached is not None:
            return cached
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
        result = tuple(
            sorted(
                candidates,
                key=lambda item: (item.free_flow_travel_ms, item.edge_ids, item.end_node_id),
            )[: self.route_limit]
        )
        self._recovery_route_cache[cache_key] = result
        return result

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
        *,
        deadline_ns: int | None = None,
    ) -> tuple[PlanSegment, ...]:
        self._check_deadline(deadline_ns)
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
            deadline_ns=deadline_ns,
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
            deadline_ns=deadline_ns,
        )
        current_ms = self._schedule_route(
            segments,
            vehicle,
            loaded_route,
            current_ms,
            LoadState.LOADED,
            reservations,
            horizon_end_ms,
            deadline_ns=deadline_ns,
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
            deadline_ns=deadline_ns,
        )
        self._schedule_route(
            segments,
            vehicle,
            recovery_route,
            current_ms,
            LoadState.EMPTY,
            reservations,
            horizon_end_ms,
            deadline_ns=deadline_ns,
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
        *,
        deadline_ns: int | None = None,
    ) -> int:
        current_node_id = route.start_node_id
        current_ms = ready_ms
        edge_ids = route.edge_ids
        edge_index = 0
        while edge_index < len(edge_ids):
            self._check_deadline(deadline_ns)
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
                    deadline_ns=deadline_ns,
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
                deadline_ns=deadline_ns,
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
        exit_index: int | None = None
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
                if exited:
                    exit_index = index
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
        # A task route can end at an AP before its later recovery route reaches
        # a PP/CP. Schedule the complete zone traversal atomically here; normal
        # edge scheduling still rejects any wait on the following LM/AP, while
        # RH-PP extends the committed prefix to the recovery route's safe node.
        if exit_index is not None:
            return exit_index
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
        *,
        deadline_ns: int | None = None,
    ) -> int:
        self._check_deadline(deadline_ns)
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
            deadline_ns=deadline_ns,
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
        *,
        deadline_ns: int | None = None,
    ) -> int:
        self._check_deadline(deadline_ns)
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
        *,
        deadline_ns: int | None = None,
    ) -> int:
        self._check_deadline(deadline_ns)
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
            deadline_ns=deadline_ns,
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
                "sipp.wait.too_long",
                "planned wait exceeds maxPlannedWaitMs",
                suggested_delay_ms=duration_ms - self.max_wait_ms,
            )
        if not self.topology.wait_allowed(node_id, robot_group):
            raise SippPlanningError(
                "sipp.wait.disallowed",
                f"node {node_id!r} does not permit planned waiting",
                suggested_delay_ms=duration_ms,
            )
        resource_id = f"node:{node_id}"
        blockers = reservations.overlapping(
            resource_id,
            start_ms,
            end_ms,
            vehicle_id=vehicle_id,
        )
        if blockers:
            raise SippPlanningError(
                "sipp.wait.interval_occupied",
                f"node {node_id!r} is unavailable for the required wait",
                suggested_delay_ms=max(
                    self.time_quantum_ms,
                    max(item.end_ms - start_ms for item in blockers),
                ),
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
        *,
        deadline_ns: int | None = None,
    ) -> int:
        candidate = not_before_ms
        resources = tuple(resource_ids)
        for _ in range(10_000):
            ContinuousTimeSippPlanner._check_deadline(deadline_ns)
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
