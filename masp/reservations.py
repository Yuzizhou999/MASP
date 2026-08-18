from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Iterator


SAFETY_FREEZE_KIND = "safety_freeze"

# 单条预留记录
@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    resource_id: str
    vehicle_id: str
    plan_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    kind: str
    committed: bool = False
    priority: int = 0
    exempt_vehicle_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError(
                f"reservation {self.reservation_id!r} must use a non-empty "
                "half-open interval"
            )
        if self.exempt_vehicle_id is not None:
            if self.kind != SAFETY_FREEZE_KIND:
                raise ValueError(
                    "exempt_vehicle_id is only valid for safety-freeze reservations"
                )
            if not self.exempt_vehicle_id:
                raise ValueError("exempt_vehicle_id must be non-empty when provided")

    def overlaps(self, other: Reservation) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms


@dataclass(frozen=True)
class RelativeReservationRequest:
    resource_id: str
    start_offset_ms: int
    end_offset_ms: int

    def __post_init__(self) -> None:
        if self.start_offset_ms < 0 or self.end_offset_ms <= self.start_offset_ms:
            raise ValueError("relative reservation request requires a non-empty interval")


@dataclass(frozen=True)
class ReservationBlocker:
    resource_id: str
    requested_start_ms: int
    requested_end_ms: int
    reservation: Reservation


@dataclass(frozen=True)
class BundleAvailability:
    start_ms: int
    blockers: tuple[ReservationBlocker, ...]


@dataclass(frozen=True)
class FreezeResult:
    reservations: tuple[Reservation, ...]
    cancelled_reservation_ids: tuple[str, ...]

    def __iter__(self) -> Iterator[Reservation]:
        """Keep existing callers that iterate over the created freezes working."""

        return iter(self.reservations)

# 冲突异常
class ReservationConflict(ValueError):
    def __init__(self, candidate: Reservation, existing: Reservation) -> None:
        self.candidate = candidate
        self.existing = existing
        super().__init__(
            f"reservation {candidate.reservation_id!r} conflicts with "
            f"{existing.reservation_id!r} on resource {candidate.resource_id!r}"
        )

# 记录表
class ReservationTable:
    def __init__(self) -> None:
        self._by_resource: dict[str, list[Reservation]] = defaultdict(list)
        self._by_id: dict[str, Reservation] = {}
        self.version = 0
        self.conflict_rejections = 0

    @staticmethod
    def _sort_key(item: Reservation) -> tuple[int, int, str]:
        return item.start_ms, item.end_ms, item.reservation_id

    @staticmethod
    def _is_exempt_for(item: Reservation, vehicle_id: str | None) -> bool:
        if vehicle_id is None:
            return False
        if item.kind == SAFETY_FREEZE_KIND:
            return item.exempt_vehicle_id == vehicle_id
        return item.vehicle_id == vehicle_id

    @staticmethod
    def _reservations_conflict(left: Reservation, right: Reservation) -> bool:
        if left.resource_id != right.resource_id or not left.overlaps(right):
            return False

        left_is_freeze = left.kind == SAFETY_FREEZE_KIND
        right_is_freeze = right.kind == SAFETY_FREEZE_KIND
        if left_is_freeze and right_is_freeze:
            return False
        if not left_is_freeze and not right_is_freeze:
            return left.vehicle_id != right.vehicle_id

        freeze = left if left_is_freeze else right
        ordinary = right if left_is_freeze else left
        if freeze.exempt_vehicle_id == ordinary.vehicle_id:
            return False

        # A gate may be overlaid on an occupant that was already executing at
        # the instant the freeze began.  A reservation starting at that instant
        # is future work and remains blocked by the gate.
        was_executing = (
            ordinary.start_ms < freeze.start_ms < ordinary.end_ms
        )
        return not was_executing

    # 把整张表导出成排序确定的元组
    def snapshot(self) -> tuple[Reservation, ...]:
        return tuple(
            sorted(self._by_id.values(), key=lambda item: (item.resource_id, *self._sort_key(item)))
        )

    def clone(self) -> ReservationTable:
        """Copy an already validated table without replaying conflict checks."""

        result = ReservationTable()
        result._by_id = dict(self._by_id)
        result._by_resource = defaultdict(
            list,
            {
                resource_id: list(rows)
                for resource_id, rows in self._by_resource.items()
            },
        )
        result.version = self.version
        result.conflict_rejections = self.conflict_rejections
        return result

    def for_vehicle(self, vehicle_id: str) -> tuple[Reservation, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._by_id.values()
                    if item.kind != SAFETY_FREEZE_KIND
                    and item.vehicle_id == vehicle_id
                ),
                key=lambda item: (item.resource_id, *self._sort_key(item)),
            )
        )

    def overlapping(
        self,
        resource_id: str,
        start_ms: int,
        end_ms: int,
        vehicle_id: str | None = None,
    ) -> tuple[Reservation, ...]:
        if end_ms <= start_ms:
            raise ValueError("overlap query requires end_ms > start_ms")
        return tuple(
            item
            for item in self._by_resource.get(resource_id, ())
            if item.start_ms < end_ms
            and start_ms < item.end_ms
            and not self._is_exempt_for(item, vehicle_id)
        )

    # 找出所有和候选预留冲突的已有预留
    def conflicts_for(self, candidate: Reservation) -> tuple[Reservation, ...]:
        return tuple(
            item
            for item in self.overlapping(
                candidate.resource_id,
                candidate.start_ms,
                candidate.end_ms,
                vehicle_id=candidate.vehicle_id,
            )
            if self._reservations_conflict(candidate, item)
        )

    # 查询函数
    def is_available(
        self,
        resource_id: str,
        start_ms: int,
        end_ms: int,
        vehicle_id: str | None = None,
    ) -> bool:
        if end_ms <= start_ms:
            raise ValueError("availability query requires end_ms > start_ms")
        return not self.overlapping(resource_id, start_ms, end_ms, vehicle_id)

    # 返回最早能开始的时刻
    def first_available_start(
        self,
        resource_id: str,
        not_before_ms: int,
        duration_ms: int,
        vehicle_id: str | None = None,
    ) -> int:
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        candidate = not_before_ms
        for item in self._by_resource.get(resource_id, ()):
            if self._is_exempt_for(item, vehicle_id):
                continue
            if candidate + duration_ms <= item.start_ms:
                break
            if candidate < item.end_ms and item.start_ms < candidate + duration_ms:
                candidate = item.end_ms
        return candidate

    def first_available_bundle_start(
        self,
        requests: Iterable[RelativeReservationRequest],
        not_before_ms: int,
        vehicle_id: str | None = None,
    ) -> BundleAvailability:
        if not_before_ms < 0:
            raise ValueError("not_before_ms must be non-negative")
        ordered = tuple(requests)
        candidate = not_before_ms
        observed: dict[tuple[str, int, int, str], ReservationBlocker] = {}
        while True:
            conflicts: list[tuple[RelativeReservationRequest, Reservation]] = []
            for request in ordered:
                requested_start = candidate + request.start_offset_ms
                requested_end = candidate + request.end_offset_ms
                for reservation in self.overlapping(
                    request.resource_id,
                    requested_start,
                    requested_end,
                    vehicle_id=vehicle_id,
                ):
                    conflicts.append((request, reservation))
                    key = (
                        request.resource_id,
                        requested_start,
                        requested_end,
                        reservation.reservation_id,
                    )
                    observed[key] = ReservationBlocker(
                        resource_id=request.resource_id,
                        requested_start_ms=requested_start,
                        requested_end_ms=requested_end,
                        reservation=reservation,
                    )
            if not conflicts:
                return BundleAvailability(
                    start_ms=candidate,
                    blockers=tuple(
                        sorted(
                            observed.values(),
                            key=lambda item: (
                                item.requested_start_ms,
                                item.requested_end_ms,
                                item.resource_id,
                                item.reservation.start_ms,
                                item.reservation.end_ms,
                                item.reservation.reservation_id,
                            ),
                        )
                    ),
                )
            next_candidate = max(
                reservation.end_ms - request.start_offset_ms
                for request, reservation in conflicts
            )
            if next_candidate <= candidate:
                raise RuntimeError("bundle availability search did not advance")
            candidate = next_candidate

    def freeze_resources(
        self,
        freeze_id: str,
        resource_ids: Iterable[str],
        start_ms: int,
        end_ms: int,
        *,
        exempt_vehicle_id: str | None = None,
        plan_id: str | None = None,
    ) -> FreezeResult:
        """Install safety gates and atomically invalidate affected plan tails.

        A freeze can overlap reservations that were already executing at
        ``start_ms`` and reservations belonging to ``exempt_vehicle_id``.  It
        blocks every other reservation that enters the resource during the
        freeze interval.
        """

        if not freeze_id:
            raise ValueError("freeze_id must be non-empty")
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("safety freeze requires a non-empty interval")
        if exempt_vehicle_id is not None and not exempt_vehicle_id:
            raise ValueError("exempt_vehicle_id must be non-empty when provided")
        freeze_plan_id = freeze_id if plan_id is None else plan_id
        if not freeze_plan_id:
            raise ValueError("plan_id must be non-empty when provided")
        ordered_resources = tuple(sorted(set(resource_ids)))
        if not ordered_resources:
            raise ValueError("safety freeze requires at least one resource")
        vehicle_id = f"safety-stop:{freeze_id}"
        batch = tuple(
            Reservation(
                reservation_id=f"reservation:{freeze_id}:{resource_id}",
                resource_id=resource_id,
                vehicle_id=vehicle_id,
                plan_id=freeze_plan_id,
                segment_id="safety-freeze",
                start_ms=start_ms,
                end_ms=end_ms,
                kind=SAFETY_FREEZE_KIND,
                committed=True,
                exempt_vehicle_id=exempt_vehicle_id,
            )
            for resource_id in ordered_resources
        )
        for candidate in batch:
            if candidate.reservation_id in self._by_id:
                raise ValueError(
                    f"duplicate reservation id {candidate.reservation_id!r}"
                )

        frozen_resource_ids = set(ordered_resources)
        affected_plan_ids = {
            item.plan_id
            for item in self._by_id.values()
            if item.kind != SAFETY_FREEZE_KIND
            and item.vehicle_id != exempt_vehicle_id
            and item.resource_id in frozen_resource_ids
            and item.start_ms < end_ms
            and start_ms < item.end_ms
        }
        cancelled = tuple(
            sorted(
                (
                    item
                    for item in self._by_id.values()
                    if item.kind != SAFETY_FREEZE_KIND
                    and item.vehicle_id != exempt_vehicle_id
                    and item.plan_id in affected_plan_ids
                    and item.start_ms >= start_ms
                ),
                key=lambda item: (item.resource_id, *self._sort_key(item)),
            )
        )
        cancelled_ids = {item.reservation_id for item in cancelled}

        remaining = tuple(
            item
            for item in self._by_id.values()
            if item.reservation_id not in cancelled_ids
        )
        for candidate in batch:
            for existing in remaining:
                if self._reservations_conflict(candidate, existing):
                    raise ReservationConflict(candidate, existing)
        for index, candidate in enumerate(batch):
            for other in batch[:index]:
                if self._reservations_conflict(candidate, other):
                    raise ReservationConflict(candidate, other)

        next_by_id = {
            item.reservation_id: item
            for item in remaining
        }
        for candidate in batch:
            next_by_id[candidate.reservation_id] = candidate
        next_by_resource: dict[str, list[Reservation]] = defaultdict(list)
        for item in next_by_id.values():
            next_by_resource[item.resource_id].append(item)
        for rows in next_by_resource.values():
            rows.sort(key=self._sort_key)

        self._by_id = next_by_id
        self._by_resource = next_by_resource
        self.version += 1
        return FreezeResult(
            reservations=batch,
            cancelled_reservation_ids=tuple(
                item.reservation_id for item in cancelled
            ),
        )

    # 原子提交插入
    def insert_batch(self, reservations: Iterable[Reservation]) -> None:
        batch = tuple(reservations)
        seen_ids: set[str] = set()
        candidates_by_resource: dict[str, list[Reservation]] = defaultdict(list)
        for candidate in batch:
            if candidate.reservation_id in self._by_id or candidate.reservation_id in seen_ids:
                raise ValueError(f"duplicate reservation id {candidate.reservation_id!r}")
            seen_ids.add(candidate.reservation_id)
            conflicts = self.conflicts_for(candidate)
            if conflicts:
                self.conflict_rejections += 1
                raise ReservationConflict(candidate, conflicts[0])
            for other in candidates_by_resource[candidate.resource_id]:
                if self._reservations_conflict(candidate, other):
                    self.conflict_rejections += 1
                    raise ReservationConflict(candidate, other)
            candidates_by_resource[candidate.resource_id].append(candidate)

        for candidate in batch:
            self._by_id[candidate.reservation_id] = candidate
            rows = self._by_resource[candidate.resource_id]
            rows.append(candidate)
            rows.sort(key=self._sort_key)
        if batch:
            self.version += 1

    def replace_vehicle(self, vehicle_id: str, reservations: Iterable[Reservation]) -> None:
        batch = tuple(reservations)
        if any(item.vehicle_id != vehicle_id for item in batch):
            raise ValueError("replacement batch must belong to one vehicle")
        seen_ids: set[str] = set()
        candidates_by_resource: dict[str, list[Reservation]] = defaultdict(list)
        for candidate in batch:
            existing_by_id = self._by_id.get(candidate.reservation_id)
            if (
                candidate.reservation_id in seen_ids
                or existing_by_id is not None
                and (
                    existing_by_id.vehicle_id != vehicle_id
                    or existing_by_id.kind == SAFETY_FREEZE_KIND
                )
            ):
                raise ValueError(f"duplicate reservation id {candidate.reservation_id!r}")
            seen_ids.add(candidate.reservation_id)
            conflicts = self.conflicts_for(candidate)
            if conflicts:
                self.conflict_rejections += 1
                raise ReservationConflict(candidate, conflicts[0])
            candidates_by_resource[candidate.resource_id].append(candidate)

        previous = list(self.for_vehicle(vehicle_id))
        for item in previous:
            self._remove(item)
        for candidate in batch:
            self._by_id[candidate.reservation_id] = candidate
            self._by_resource[candidate.resource_id].append(candidate)
        for rows in self._by_resource.values():
            rows.sort(key=self._sort_key)
        if previous or batch:
            self.version += 1

    # 把某计划的预留从"候选"标记为"已提交"
    def commit_plan(self, plan_id: str) -> int:
        matching = [item for item in self._by_id.values() if item.plan_id == plan_id]
        if not matching:
            raise KeyError(f"plan {plan_id!r} has no reservations")
        changed = 0
        for item in matching:
            if item.committed:
                continue
            committed = replace(item, committed=True)
            self._replace(committed)
            changed += 1
        if changed:
            self.version += 1
        return changed

    # 取消某计划的预留，默认只撤未提交的
    def remove_plan(self, plan_id: str, include_committed: bool = False) -> int:
        removable = [
            item
            for item in self._by_id.values()
            if item.plan_id == plan_id and (include_committed or not item.committed)
        ]
        for item in removable:
            self._remove(item)
        if removable:
            self.version += 1
        return len(removable)

    # 把已经结束的预留清掉，防止表无限膨胀
    def expire_before(self, time_ms: int) -> int:
        expired = [item for item in self._by_id.values() if item.end_ms <= time_ms]
        for item in expired:
            self._remove(item)
        if expired:
            self.version += 1
        return len(expired)

    def _replace(self, replacement: Reservation) -> None:
        previous = self._by_id[replacement.reservation_id]
        rows = self._by_resource[previous.resource_id]
        rows[rows.index(previous)] = replacement
        rows.sort(key=self._sort_key)
        self._by_id[replacement.reservation_id] = replacement

    def _remove(self, item: Reservation) -> None:
        del self._by_id[item.reservation_id]
        rows = self._by_resource[item.resource_id]
        rows.remove(item)
        if not rows:
            del self._by_resource[item.resource_id]
