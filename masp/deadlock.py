from __future__ import annotations

from dataclasses import dataclass

from .reservations import ReservationTable


@dataclass(frozen=True)
class BlockedRequirement:
    """A vehicle's next atomic resource request at a runtime decision point."""

    vehicle_id: str
    resource_ids: tuple[str, ...]
    start_ms: int
    end_ms: int
    blocked_since_ms: int
    has_alternative: bool = False

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("blocked requirement must use a non-empty interval")
        if self.blocked_since_ms > self.start_ms:
            raise ValueError("blocked_since_ms cannot be later than start_ms")


@dataclass(frozen=True, order=True)
class WaitDependency:
    waiting_vehicle_id: str
    blocking_vehicle_id: str
    resource_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "waitingVehicleId": self.waiting_vehicle_id,
            "blockingVehicleId": self.blocking_vehicle_id,
            "resourceId": self.resource_id,
        }


@dataclass(frozen=True)
class DeadlockReport:
    analyzed_at_ms: int
    reservation_version: int
    dependencies: tuple[WaitDependency, ...]
    cycles: tuple[tuple[str, ...], ...]
    blocked_vehicle_ids: tuple[str, ...]
    priority_age_ms: dict[str, int]

    @property
    def max_cycle_length(self) -> int:
        return max((len(item) for item in self.cycles), default=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "analyzedAtMs": self.analyzed_at_ms,
            "reservationVersion": self.reservation_version,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "cycles": [list(item) for item in self.cycles],
            "blockedVehicleIds": list(self.blocked_vehicle_ids),
            "priorityAgeMs": dict(sorted(self.priority_age_ms.items())),
            "maxCycleLength": self.max_cycle_length,
        }


class DeadlockSupervisor:
    """Build a hard wait graph and maintain deterministic starvation ages."""

    def __init__(self, starvation_age_step_ms: int = 5_000) -> None:
        if starvation_age_step_ms <= 0:
            raise ValueError("starvation_age_step_ms must be positive")
        self.starvation_age_step_ms = starvation_age_step_ms
        self._blocked_since_ms: dict[str, int] = {}

    def analyze(
        self,
        now_ms: int,
        requirements: tuple[BlockedRequirement, ...],
        reservations: ReservationTable,
    ) -> DeadlockReport:
        dependencies: set[WaitDependency] = set()
        explicit_since: dict[str, int] = {}
        for requirement in requirements:
            if requirement.has_alternative:
                continue
            for resource_id in requirement.resource_ids:
                for blocker in reservations.overlapping(
                    resource_id,
                    requirement.start_ms,
                    requirement.end_ms,
                    vehicle_id=requirement.vehicle_id,
                ):
                    if not blocker.committed:
                        continue
                    dependencies.add(
                        WaitDependency(
                            waiting_vehicle_id=requirement.vehicle_id,
                            blocking_vehicle_id=blocker.vehicle_id,
                            resource_id=resource_id,
                        )
                    )
                    explicit_since[requirement.vehicle_id] = min(
                        explicit_since.get(
                            requirement.vehicle_id, requirement.blocked_since_ms
                        ),
                        requirement.blocked_since_ms,
                    )

        blocked_vehicle_ids = tuple(
            sorted({item.waiting_vehicle_id for item in dependencies})
        )
        blocked_set = set(blocked_vehicle_ids)
        for vehicle_id in tuple(self._blocked_since_ms):
            if vehicle_id not in blocked_set:
                del self._blocked_since_ms[vehicle_id]
        for vehicle_id in blocked_set:
            observed_since = explicit_since.get(vehicle_id, now_ms)
            previous = self._blocked_since_ms.get(vehicle_id, observed_since)
            self._blocked_since_ms[vehicle_id] = min(previous, observed_since)

        priority_age_ms = {
            vehicle_id: (
                max(0, now_ms - blocked_since)
                // self.starvation_age_step_ms
                * self.starvation_age_step_ms
            )
            for vehicle_id, blocked_since in self._blocked_since_ms.items()
        }
        ordered_dependencies = tuple(sorted(dependencies))
        return DeadlockReport(
            analyzed_at_ms=now_ms,
            reservation_version=reservations.version,
            dependencies=ordered_dependencies,
            cycles=self._strongly_connected_cycles(ordered_dependencies),
            blocked_vehicle_ids=blocked_vehicle_ids,
            priority_age_ms=dict(sorted(priority_age_ms.items())),
        )

    @staticmethod
    def _strongly_connected_cycles(
        dependencies: tuple[WaitDependency, ...],
    ) -> tuple[tuple[str, ...], ...]:
        adjacency: dict[str, set[str]] = {}
        for item in dependencies:
            adjacency.setdefault(item.waiting_vehicle_id, set()).add(
                item.blocking_vehicle_id
            )
            adjacency.setdefault(item.blocking_vehicle_id, set())

        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        low_links: dict[str, int] = {}
        components: list[tuple[str, ...]] = []

        def visit(vehicle_id: str) -> None:
            nonlocal index
            indices[vehicle_id] = index
            low_links[vehicle_id] = index
            index += 1
            stack.append(vehicle_id)
            on_stack.add(vehicle_id)
            for blocker_id in sorted(adjacency[vehicle_id]):
                if blocker_id not in indices:
                    visit(blocker_id)
                    low_links[vehicle_id] = min(
                        low_links[vehicle_id], low_links[blocker_id]
                    )
                elif blocker_id in on_stack:
                    low_links[vehicle_id] = min(
                        low_links[vehicle_id], indices[blocker_id]
                    )
            if low_links[vehicle_id] != indices[vehicle_id]:
                return
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == vehicle_id:
                    break
            normalized = tuple(sorted(component))
            if len(normalized) > 1 or vehicle_id in adjacency[vehicle_id]:
                components.append(normalized)

        for vehicle_id in sorted(adjacency):
            if vehicle_id not in indices:
                visit(vehicle_id)
        return tuple(sorted(set(components)))
