from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import DomainError, PlanSegment, SegmentKind, TransportTask, Vehicle
from .zones import TrafficZoneIndex


@dataclass(frozen=True)
class Workstation:
    station_id: str
    node_id: str
    allowed_robot_groups: frozenset[str]
    pickup_service_ms: int
    dropoff_service_ms: int
    blocks_transit_during_service: bool


class MapTopology:
    def __init__(
        self,
        model: dict[str, Any],
        conflicts: dict[str, Any],
        workstations: dict[str, Any],
        traffic_zones: dict[str, Any] | None = None,
    ) -> None:
        self.nodes = {item["id"]: item for item in model["nodes"]}
        self.edges = {item["id"]: item for item in model["edges"]}
        self.edge_resources = {
            item["edgeId"]: item for item in conflicts["edgeResources"]
        }
        self.workstations = {
            item["nodeId"]: Workstation(
                station_id=item["id"],
                node_id=item["nodeId"],
                allowed_robot_groups=frozenset(item["allowedRobotGroups"]),
                pickup_service_ms=int(item["pickupServiceMs"]),
                dropoff_service_ms=int(item["dropoffServiceMs"]),
                blocks_transit_during_service=bool(
                    item["blocksTransitDuringService"]
                ),
            )
            for item in workstations["workstations"]
        }
        self.traffic_zones = TrafficZoneIndex(model, traffic_zones)

    def validate_vehicle(self, vehicle: Vehicle) -> None:
        node = self.nodes.get(vehicle.current_node_id or "")
        if node is None:
            raise DomainError(
                "vehicle.initial_node.missing",
                f"vehicle {vehicle.vehicle_id!r} starts at an unknown node",
            )
        if vehicle.robot_group not in node["allowedRobotGroups"]:
            raise DomainError(
                "vehicle.initial_node.incompatible",
                f"vehicle {vehicle.vehicle_id!r} cannot use {vehicle.current_node_id!r}",
            )

    def validate_task(self, task: TransportTask) -> None:
        if task.pickup_service_ms <= 0 or task.dropoff_service_ms <= 0:
            raise DomainError(
                "task.service.duration",
                f"task {task.task_id!r} service durations must be positive",
            )
        if task.due_time_ms is not None and task.due_time_ms < task.release_time_ms:
            raise DomainError(
                "task.due_time.before_release",
                f"task {task.task_id!r} due time is earlier than its release time",
            )
        for role, node_id in (
            ("pickup", task.pickup_node_id),
            ("dropoff", task.dropoff_node_id),
        ):
            node = self.nodes.get(node_id)
            if node is None or node["type"] != "AP":
                raise DomainError(
                    "task.ap.invalid",
                    f"task {task.task_id!r} {role} node {node_id!r} is not an AP",
                )
            station = self.workstations.get(node_id)
            if (
                task.required_robot_group not in node["allowedRobotGroups"]
                or station is None
                or task.required_robot_group not in station.allowed_robot_groups
            ):
                raise DomainError(
                    "task.ap.incompatible",
                    f"task {task.task_id!r} {role} AP does not allow group "
                    f"{task.required_robot_group!r}",
                )

    def derived_resources(self, segment: PlanSegment) -> tuple[str, ...]:
        resources: set[str] = set()
        # 走一条路，要把这条路和所有会撞的路、以及两头路口都锁住
        if segment.kind is SegmentKind.TRAVERSE:
            if segment.edge_id not in self.edge_resources:
                raise DomainError(
                    "plan.edge.resources_missing",
                    f"edge {segment.edge_id!r} has no conflict resource record",
                )
            edge_resource = self.edge_resources[segment.edge_id]
            resources.add(edge_resource["ownResource"])
            resources.update(edge_resource["conflictResources"])
            if segment.start_node_id:
                resources.add(f"node:{segment.start_node_id}")
            if segment.end_node_id:
                resources.add(f"node:{segment.end_node_id}")
            resources.update(
                self.traffic_zones.resource_ids_for_edge(segment.edge_id or "")
            )
        # 等车不动，只占当前节点即可
        elif segment.kind is SegmentKind.WAIT:
            if segment.start_node_id:
                resources.add(f"node:{segment.start_node_id}")
                resources.update(
                    self.traffic_zones.resource_ids_for_node(segment.start_node_id)
                )
        # 装卸货时占着工位；如果这工位在路中间，就把路也封了
        elif segment.kind in {SegmentKind.PICKUP, SegmentKind.DROPOFF}:
            node_id = segment.start_node_id
            station = self.workstations.get(node_id or "")
            if station is None:
                raise DomainError(
                    "plan.service.workstation_missing",
                    f"service segment {segment.segment_id!r} has no workstation",
                )
            resources.add(f"workstation:{station.station_id}")
            if station.blocks_transit_during_service:
                resources.add(f"node:{station.node_id}")
            resources.update(
                self.traffic_zones.resource_ids_for_node(station.node_id)
            )
        return tuple(sorted(resources))

    def required_resources(self, segment: PlanSegment) -> tuple[str, ...]:
        return tuple(sorted(set(self.derived_resources(segment)) | set(segment.resource_ids)))

    def wait_allowed(self, node_id: str, robot_group: str) -> bool:
        if self.traffic_zones.zone_for_node(node_id) is not None:
            return False
        node = self.nodes[node_id]
        return bool(node["waitPolicyByGroup"][robot_group]["allowed"])
