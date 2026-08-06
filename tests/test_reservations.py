from __future__ import annotations

import pytest

from masp.reservations import Reservation, ReservationConflict, ReservationTable


def reservation(
    reservation_id: str,
    vehicle_id: str,
    start_ms: int,
    end_ms: int,
    *,
    committed: bool = True,
) -> Reservation:
    return Reservation(
        reservation_id=reservation_id,
        resource_id="edge-conflict:crossing",
        vehicle_id=vehicle_id,
        plan_id=f"plan:{vehicle_id}",
        segment_id=f"segment:{reservation_id}",
        start_ms=start_ms,
        end_ms=end_ms,
        kind="transit",
        committed=committed,
    )


def test_half_open_intervals_allow_touching_boundaries() -> None:
    table = ReservationTable()
    table.insert_batch(
        [
            reservation("r1", "vehicle-1", 0, 100),
            reservation("r2", "vehicle-2", 100, 200),
        ]
    )

    assert len(table.snapshot()) == 2
    assert table.is_available("edge-conflict:crossing", 200, 300)
    assert table.first_available_start("edge-conflict:crossing", 50, 100) == 200


def test_conflicting_batch_is_rejected_without_partial_insert() -> None:
    table = ReservationTable()

    with pytest.raises(ReservationConflict):
        table.insert_batch(
            [
                reservation("r1", "vehicle-1", 0, 100),
                reservation("r2", "vehicle-2", 50, 150),
            ]
        )

    assert table.snapshot() == ()
    assert table.version == 0
    assert table.conflict_rejections == 1


def test_tentative_plan_can_be_committed_and_removed_by_policy() -> None:
    table = ReservationTable()
    table.insert_batch([reservation("r1", "vehicle-1", 0, 100, committed=False)])

    assert table.commit_plan("plan:vehicle-1") == 1
    assert table.remove_plan("plan:vehicle-1") == 0
    assert table.remove_plan("plan:vehicle-1", include_committed=True) == 1
    assert table.snapshot() == ()


def test_vehicle_reservations_are_replaced_atomically() -> None:
    table = ReservationTable()
    old = reservation("old", "vehicle-1", 0, 100)
    blocker = reservation("blocker", "vehicle-2", 100, 200)
    table.insert_batch([old, blocker])

    replacement = reservation("new", "vehicle-1", 200, 300)
    table.replace_vehicle("vehicle-1", [replacement])

    assert {item.reservation_id for item in table.for_vehicle("vehicle-1")} == {"new"}
    with pytest.raises(ReservationConflict):
        table.replace_vehicle("vehicle-1", [reservation("bad", "vehicle-1", 150, 250)])
    assert {item.reservation_id for item in table.for_vehicle("vehicle-1")} == {"new"}
