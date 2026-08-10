from __future__ import annotations

import pytest

from masp.reservations import (
    FreezeResult,
    Reservation,
    ReservationConflict,
    ReservationTable,
)


def reservation(
    reservation_id: str,
    vehicle_id: str,
    start_ms: int,
    end_ms: int,
    *,
    committed: bool = True,
    resource_id: str = "edge-conflict:crossing",
    plan_id: str | None = None,
) -> Reservation:
    return Reservation(
        reservation_id=reservation_id,
        resource_id=resource_id,
        vehicle_id=vehicle_id,
        plan_id=plan_id or f"plan:{vehicle_id}",
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


def test_clone_preserves_snapshot_and_mutates_independently() -> None:
    table = ReservationTable()
    table.insert_batch([reservation("r1", "vehicle-1", 0, 100)])
    table.conflict_rejections = 3

    cloned = table.clone()
    cloned.insert_batch([reservation("r2", "vehicle-2", 100, 200)])

    assert table.snapshot() == (reservation("r1", "vehicle-1", 0, 100),)
    assert len(cloned.snapshot()) == 2
    assert cloned.version == table.version + 1
    assert cloned.conflict_rejections == table.conflict_rejections


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


def test_safety_freeze_overlay_snapshot_can_be_rebuilt() -> None:
    table = ReservationTable()
    occupant = reservation("occupant", "vehicle-1", 0, 100)
    table.insert_batch([occupant])

    result = table.freeze_resources("freeze-1", [occupant.resource_id], 50, 200)

    assert isinstance(result, FreezeResult)
    assert result.cancelled_reservation_ids == ()
    assert tuple(result) == result.reservations
    assert table.version == 2
    rebuilt = ReservationTable()
    rebuilt.insert_batch(table.snapshot())
    assert rebuilt.snapshot() == table.snapshot()


def test_safety_freeze_rejects_a_new_ordinary_entry() -> None:
    table = ReservationTable()
    resource_id = "edge-conflict:frozen"
    table.freeze_resources("freeze-1", [resource_id], 50, 200)

    with pytest.raises(ReservationConflict) as error:
        table.insert_batch(
            [
                reservation(
                    "new-entry",
                    "vehicle-2",
                    100,
                    150,
                    resource_id=resource_id,
                )
            ]
        )

    assert error.value.existing.kind == "safety_freeze"
    assert table.version == 1


def test_safety_freeze_cancels_the_entire_committed_future_plan_tail() -> None:
    table = ReservationTable()
    frozen_resource = "edge-conflict:frozen"
    plan_id = "plan:vehicle-1:committed"
    active = reservation(
        "active",
        "vehicle-1",
        0,
        100,
        resource_id=frozen_resource,
        plan_id=plan_id,
    )
    future_on_frozen_resource = reservation(
        "future-frozen",
        "vehicle-1",
        100,
        150,
        resource_id=frozen_resource,
        plan_id=plan_id,
    )
    future_tail_elsewhere = reservation(
        "future-tail",
        "vehicle-1",
        150,
        250,
        resource_id="node:future-tail",
        plan_id=plan_id,
    )
    table.insert_batch([active, future_on_frozen_resource, future_tail_elsewhere])

    result = table.freeze_resources("freeze-1", [frozen_resource], 50, 200)

    assert result.cancelled_reservation_ids == ("future-frozen", "future-tail")
    assert active in table.snapshot()
    assert future_on_frozen_resource not in table.snapshot()
    assert future_tail_elsewhere not in table.snapshot()
    assert table.version == 2


def test_exempt_vehicle_plan_survives_and_coexists_with_safety_freeze() -> None:
    table = ReservationTable()
    frozen_resource = "edge-conflict:frozen"
    exempt = reservation(
        "exempt-future",
        "vehicle-exempt",
        75,
        100,
        resource_id=frozen_resource,
    )
    cancelled = reservation(
        "ordinary-future",
        "vehicle-ordinary",
        125,
        150,
        resource_id=frozen_resource,
    )
    table.insert_batch([exempt, cancelled])

    result = table.freeze_resources(
        "freeze-1",
        [frozen_resource],
        50,
        200,
        exempt_vehicle_id="vehicle-exempt",
        plan_id="plan:safety-gate",
    )

    assert result.cancelled_reservation_ids == ("ordinary-future",)
    assert exempt in table.snapshot()
    assert result.reservations[0].plan_id == "plan:safety-gate"
    assert result.reservations[0].exempt_vehicle_id == "vehicle-exempt"
    assert table.for_vehicle(result.reservations[0].vehicle_id) == ()
    assert table.is_available(frozen_resource, 75, 100, "vehicle-exempt")
    replacement = reservation(
        "exempt-replacement",
        "vehicle-exempt",
        100,
        125,
        resource_id=frozen_resource,
    )
    table.replace_vehicle("vehicle-exempt", [replacement])
    assert result.reservations[0] in table.snapshot()
    assert table.for_vehicle("vehicle-exempt") == (replacement,)
    rebuilt = ReservationTable()
    rebuilt.insert_batch(table.snapshot())
    assert rebuilt.snapshot() == table.snapshot()


def test_failed_safety_freeze_does_not_cancel_or_increment_version() -> None:
    table = ReservationTable()
    resource_id = "edge-conflict:frozen"
    table.freeze_resources("freeze-1", [resource_id], 50, 200)
    future = reservation(
        "future",
        "vehicle-1",
        250,
        300,
        resource_id=resource_id,
    )
    table.insert_batch([future])
    before = table.snapshot()
    version = table.version

    with pytest.raises(ValueError, match="duplicate reservation id"):
        table.freeze_resources("freeze-1", [resource_id], 250, 400)

    assert table.snapshot() == before
    assert table.version == version
