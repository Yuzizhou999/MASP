###定义"仿真里会发生哪些事件"，以及"同一时刻多个事件发生时，谁先谁后"的确定性规则——并用一个高效优先队列把它们按序排好。

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any

# 事件类型清单
class EventType(str, Enum):
    VEHICLE_FAULTED = "VEHICLE_FAULTED"
    VEHICLE_EXIT_EDGE = "VEHICLE_EXIT_EDGE"
    VEHICLE_WAIT_ENDED = "VEHICLE_WAIT_ENDED"
    PICKUP_COMPLETED = "PICKUP_COMPLETED"
    DROPOFF_COMPLETED = "DROPOFF_COMPLETED"
    VEHICLE_ENTER_EDGE = "VEHICLE_ENTER_EDGE"
    VEHICLE_WAIT_STARTED = "VEHICLE_WAIT_STARTED"
    PICKUP_STARTED = "PICKUP_STARTED"
    DROPOFF_STARTED = "DROPOFF_STARTED"
    TASK_RELEASED = "TASK_RELEASED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    PLAN_COMPUTED = "PLAN_COMPUTED"
    PLAN_COMMITTED = "PLAN_COMMITTED"
    METRICS_SAMPLED = "METRICS_SAMPLED"


EVENT_PRIORITY: dict[EventType, int] = {
    EventType.VEHICLE_FAULTED: 0,
    EventType.VEHICLE_EXIT_EDGE: 10,
    EventType.VEHICLE_WAIT_ENDED: 10,
    EventType.PICKUP_COMPLETED: 20,
    EventType.DROPOFF_COMPLETED: 20,
    EventType.VEHICLE_ENTER_EDGE: 30,
    EventType.VEHICLE_WAIT_STARTED: 30,
    EventType.PICKUP_STARTED: 30,
    EventType.DROPOFF_STARTED: 30,
    EventType.TASK_RELEASED: 40,
    EventType.TASK_ASSIGNED: 50,
    EventType.PLAN_COMPUTED: 60,
    EventType.PLAN_COMMITTED: 61,
    EventType.METRICS_SAMPLED: 70,
}

# 定义一个事件 = 时间 + 类型 + 序号 + 数据
@dataclass(order=True, frozen=True)
class SimulationEvent:
    sort_key: tuple[int, int, int] = field(init=False, repr=False)
    time_ms: int = field(compare=False)
    event_type: EventType = field(compare=False)
    sequence: int = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.time_ms < 0:
            raise ValueError("event time must be non-negative")
        object.__setattr__(
            self,
            "sort_key",
            (self.time_ms, EVENT_PRIORITY[self.event_type], self.sequence),
        )

# 优先队列实现
class DeterministicEventQueue:
    def __init__(self) -> None:
        self._events: list[SimulationEvent] = []
        self._sequence = count()

    def schedule(
        self,
        time_ms: int,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        event = SimulationEvent(
            time_ms=time_ms,
            event_type=event_type,
            sequence=next(self._sequence),
            payload=dict(payload or {}),
        )
        heapq.heappush(self._events, event)
        return event

    def peek(self) -> SimulationEvent | None:
        return self._events[0] if self._events else None

    def pop(self) -> SimulationEvent:
        return heapq.heappop(self._events)

    def __bool__(self) -> bool:
        return bool(self._events)

    def __len__(self) -> int:
        return len(self._events)
