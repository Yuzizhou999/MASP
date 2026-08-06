from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .domain import DomainError


@dataclass(frozen=True)
class TrafficZone:
    zone_id: str
    member_node_ids: frozenset[str]
    member_edge_ids: frozenset[str]
    entry_edge_ids: frozenset[str]
    exit_edge_ids: frozenset[str]
    capacity: int
    passing_allowed: bool
    directional_mode: str
    recovery_node_ids: frozenset[str]

    @property
    def resource_id(self) -> str:
        return f"zone:{self.zone_id}"

    @property
    def controlled_edge_ids(self) -> frozenset[str]:
        return frozenset(
            self.member_edge_ids | self.entry_edge_ids | self.exit_edge_ids
        )


class TrafficZoneIndex:
    """Validated indexes for the phase-4 single-capacity traffic-zone MVP."""

    def __init__(
        self,
        model: dict[str, Any],
        traffic_zones: dict[str, Any] | None = None,
    ) -> None:
        self.nodes = {item["id"]: item for item in model["nodes"]}
        self.edges = {item["id"]: item for item in model["edges"]}
        self.zones_by_id: dict[str, TrafficZone] = {}
        self._zones_by_edge: dict[str, TrafficZone] = {}
        self._entry_zones_by_edge: dict[str, TrafficZone] = {}
        self._zones_by_node: dict[str, TrafficZone] = {}

        for raw in (traffic_zones or {}).get("zones", []):
            zone = self._parse_zone(raw)
            if zone.zone_id in self.zones_by_id:
                raise DomainError(
                    "zone.id.duplicate", f"duplicate traffic zone {zone.zone_id!r}"
                )
            self._validate_topology(zone)
            self.zones_by_id[zone.zone_id] = zone
            self._index_zone(zone)

    @staticmethod
    def _parse_zone(raw: dict[str, Any]) -> TrafficZone:
        zone_id = str(raw["id"])
        capacity = int(raw["capacity"])
        passing_allowed = bool(raw["passingAllowed"])
        directional_mode = str(raw["directionalMode"])
        if capacity != 1:
            raise DomainError(
                "zone.mvp.capacity",
                f"traffic zone {zone_id!r} must use capacity=1 in the phase-4 MVP",
            )
        if passing_allowed:
            raise DomainError(
                "zone.mvp.passing",
                f"traffic zone {zone_id!r} must set passingAllowed=false in the phase-4 MVP",
            )
        if directional_mode != "single_direction_at_a_time":
            raise DomainError(
                "zone.mvp.directional_mode",
                f"traffic zone {zone_id!r} must use single_direction_at_a_time",
            )
        return TrafficZone(
            zone_id=zone_id,
            member_node_ids=frozenset(raw["memberNodeIds"]),
            member_edge_ids=frozenset(raw["memberEdgeIds"]),
            entry_edge_ids=frozenset(raw["entryEdgeIds"]),
            exit_edge_ids=frozenset(raw["exitEdgeIds"]),
            capacity=capacity,
            passing_allowed=passing_allowed,
            directional_mode=directional_mode,
            recovery_node_ids=frozenset(raw["recoveryNodeIds"]),
        )

    def _validate_topology(self, zone: TrafficZone) -> None:
        if not zone.member_node_ids:
            raise DomainError(
                "zone.nodes.empty", f"traffic zone {zone.zone_id!r} has no member nodes"
            )
        if not zone.member_edge_ids:
            raise DomainError(
                "zone.edges.empty", f"traffic zone {zone.zone_id!r} has no member edges"
            )
        if not zone.entry_edge_ids or not zone.exit_edge_ids:
            raise DomainError(
                "zone.boundary.empty",
                f"traffic zone {zone.zone_id!r} requires entry and exit edges",
            )
        if zone.entry_edge_ids & zone.exit_edge_ids:
            raise DomainError(
                "zone.edges.overlap",
                f"traffic zone {zone.zone_id!r} cannot use one edge as both entry and exit",
            )

        unknown_nodes = zone.member_node_ids - self.nodes.keys()
        unknown_edges = zone.controlled_edge_ids - self.edges.keys()
        unknown_recovery = zone.recovery_node_ids - self.nodes.keys()
        if unknown_nodes or unknown_edges or unknown_recovery:
            raise DomainError(
                "zone.reference.missing",
                f"traffic zone {zone.zone_id!r} has missing references: "
                f"nodes={sorted(unknown_nodes)!r}, edges={sorted(unknown_edges)!r}, "
                f"recovery={sorted(unknown_recovery)!r}",
            )

        internal_edge_ids = (
            zone.member_edge_ids - zone.entry_edge_ids - zone.exit_edge_ids
        )
        for edge_id in sorted(internal_edge_ids):
            edge = self.edges[edge_id]
            if (
                edge["start"] not in zone.member_node_ids
                or edge["end"] not in zone.member_node_ids
            ):
                raise DomainError(
                    "zone.member_edge.boundary",
                    f"member edge {edge_id!r} must remain inside zone {zone.zone_id!r}",
                )
        for edge_id in sorted(zone.entry_edge_ids):
            edge = self.edges[edge_id]
            if (
                edge["start"] in zone.member_node_ids
                or edge["end"] not in zone.member_node_ids
            ):
                raise DomainError(
                    "zone.entry.direction",
                    f"entry edge {edge_id!r} must point from outside into zone {zone.zone_id!r}",
                )
        for edge_id in sorted(zone.exit_edge_ids):
            edge = self.edges[edge_id]
            if (
                edge["start"] not in zone.member_node_ids
                or edge["end"] in zone.member_node_ids
            ):
                raise DomainError(
                    "zone.exit.direction",
                    f"exit edge {edge_id!r} must point from zone {zone.zone_id!r} to outside",
                )

        expected_members = frozenset(
            edge_id
            for edge_id, edge in self.edges.items()
            if edge["start"] in zone.member_node_ids
            and edge["end"] in zone.member_node_ids
        )
        expected_entries = frozenset(
            edge_id
            for edge_id, edge in self.edges.items()
            if edge["start"] not in zone.member_node_ids
            and edge["end"] in zone.member_node_ids
        )
        expected_exits = frozenset(
            edge_id
            for edge_id, edge in self.edges.items()
            if edge["start"] in zone.member_node_ids
            and edge["end"] not in zone.member_node_ids
        )
        if (
            zone.member_edge_ids != expected_members
            or zone.entry_edge_ids != expected_entries
            or zone.exit_edge_ids != expected_exits
        ):
            raise DomainError(
                "zone.boundary.incomplete",
                f"traffic zone {zone.zone_id!r} must classify every edge touching a member node; "
                f"missingMembers={sorted(expected_members - zone.member_edge_ids)!r}, "
                f"missingEntries={sorted(expected_entries - zone.entry_edge_ids)!r}, "
                f"missingExits={sorted(expected_exits - zone.exit_edge_ids)!r}",
            )

    def _index_zone(self, zone: TrafficZone) -> None:
        for edge_id in sorted(zone.controlled_edge_ids):
            existing = self._zones_by_edge.get(edge_id)
            if existing is not None:
                raise DomainError(
                    "zone.edge.multiple",
                    f"edge {edge_id!r} belongs to zones {existing.zone_id!r} and {zone.zone_id!r}",
                )
            self._zones_by_edge[edge_id] = zone
        for edge_id in sorted(zone.entry_edge_ids):
            self._entry_zones_by_edge[edge_id] = zone
        for node_id in sorted(zone.member_node_ids):
            existing = self._zones_by_node.get(node_id)
            if existing is not None:
                raise DomainError(
                    "zone.node.multiple",
                    f"node {node_id!r} belongs to zones {existing.zone_id!r} and {zone.zone_id!r}",
                )
            self._zones_by_node[node_id] = zone

    def __bool__(self) -> bool:
        return bool(self.zones_by_id)

    def zones(self) -> tuple[TrafficZone, ...]:
        return tuple(self.zones_by_id[item] for item in sorted(self.zones_by_id))

    def zone_for_edge(self, edge_id: str) -> TrafficZone | None:
        return self._zones_by_edge.get(edge_id)

    def entry_zone_for_edge(self, edge_id: str) -> TrafficZone | None:
        return self._entry_zones_by_edge.get(edge_id)

    def zone_for_node(self, node_id: str) -> TrafficZone | None:
        return self._zones_by_node.get(node_id)

    def resource_ids_for_edge(self, edge_id: str) -> tuple[str, ...]:
        zone = self.zone_for_edge(edge_id)
        return (zone.resource_id,) if zone is not None else ()

    def resource_ids_for_node(self, node_id: str | None) -> tuple[str, ...]:
        if node_id is None:
            return ()
        zone = self.zone_for_node(node_id)
        return (zone.resource_id,) if zone is not None else ()

    def edge_is_controlled_by(
        self, edge_id: str, zone: TrafficZone
    ) -> bool:
        return edge_id in zone.controlled_edge_ids

    def first_zone_edge(self, edge_ids: Iterable[str]) -> TrafficZone | None:
        for edge_id in edge_ids:
            zone = self.zone_for_edge(edge_id)
            if zone is not None:
                return zone
        return None
