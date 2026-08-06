from __future__ import annotations

from masp.events import DeterministicEventQueue, EventType


def test_same_timestamp_events_follow_fixed_safety_order() -> None:
    queue = DeterministicEventQueue()
    for event_type in (
        EventType.PLAN_COMMITTED,
        EventType.TASK_RELEASED,
        EventType.VEHICLE_ENTER_EDGE,
        EventType.PICKUP_COMPLETED,
        EventType.VEHICLE_EXIT_EDGE,
        EventType.VEHICLE_FAULTED,
    ):
        queue.schedule(100, event_type)

    assert [queue.pop().event_type for _ in range(6)] == [
        EventType.VEHICLE_FAULTED,
        EventType.VEHICLE_EXIT_EDGE,
        EventType.PICKUP_COMPLETED,
        EventType.TASK_RELEASED,
        EventType.PLAN_COMMITTED,
        EventType.VEHICLE_ENTER_EDGE,
    ]


def test_equal_type_events_keep_insertion_order() -> None:
    queue = DeterministicEventQueue()
    first = queue.schedule(100, EventType.TASK_RELEASED, {"taskId": "first"})
    second = queue.schedule(100, EventType.TASK_RELEASED, {"taskId": "second"})

    assert queue.pop() == first
    assert queue.pop() == second
