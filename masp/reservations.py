from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

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

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError(
                f"reservation {self.reservation_id!r} must use a non-empty "
                "half-open interval"
            )

    def overlaps(self, other: Reservation) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

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

    # 把整张表导出成排序确定的元组
    def snapshot(self) -> tuple[Reservation, ...]:
        return tuple(
            sorted(self._by_id.values(), key=lambda item: (item.resource_id, *self._sort_key(item)))
        )

    def for_vehicle(self, vehicle_id: str) -> tuple[Reservation, ...]:
        return tuple(
            sorted(
                (item for item in self._by_id.values() if item.vehicle_id == vehicle_id),
                key=lambda item: (item.resource_id, *self._sort_key(item)),
            )
        )

    # 找出所有和候选预留冲突的已有预留
    def conflicts_for(self, candidate: Reservation) -> tuple[Reservation, ...]:
        return tuple(
            item
            for item in self._by_resource.get(candidate.resource_id, ())
            if item.vehicle_id != candidate.vehicle_id and item.overlaps(candidate)
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
        return not any(
            item.start_ms < end_ms
            and start_ms < item.end_ms
            and (vehicle_id is None or item.vehicle_id != vehicle_id)
            for item in self._by_resource.get(resource_id, ())
        )

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
            if vehicle_id is not None and item.vehicle_id == vehicle_id:
                continue
            if candidate + duration_ms <= item.start_ms:
                break
            if candidate < item.end_ms and item.start_ms < candidate + duration_ms:
                candidate = item.end_ms
        return candidate

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
                if other.vehicle_id != candidate.vehicle_id and other.overlaps(candidate):
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
                and existing_by_id.vehicle_id != vehicle_id
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
