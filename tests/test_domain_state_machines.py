from __future__ import annotations

import pytest

from masp.domain import (
    DomainError,
    LoadState,
    TaskState,
    TransportTask,
    Vehicle,
    VehicleState,
)


def make_task() -> TransportTask:
    return TransportTask(
        task_id="task-1",
        release_time_ms=0,
        pickup_node_id="fork:pickup",
        dropoff_node_id="fork:dropoff",
        required_robot_group="fork",
        payload_type="pallet",
        payload_id="payload-1",
        pickup_service_ms=100,
        dropoff_service_ms=100,
    )


def test_task_state_machine_records_business_timestamps() -> None:
    task = make_task()
    task.assigned_vehicle_id = "fork-1"

    task.transition(TaskState.ASSIGNED, 10)
    task.transition(TaskState.EN_ROUTE_PICKUP, 10)
    task.transition(TaskState.PICKUP_SERVICE, 20)
    task.transition(TaskState.EN_ROUTE_DROPOFF, 120)
    task.transition(TaskState.DROPOFF_SERVICE, 150)
    task.transition(TaskState.COMPLETED, 250)

    assert task.revision == 6
    assert task.assigned_at_ms == 10
    assert task.picked_at_ms == 120
    assert task.completed_at_ms == 250


def test_task_state_machine_rejects_skipping_pickup() -> None:
    task = make_task()

    with pytest.raises(DomainError, match="cannot transition") as error:
        task.transition(TaskState.COMPLETED, 10)

    assert error.value.code == "task.transition.invalid"


def test_vehicle_state_machine_tracks_time_in_each_state() -> None:
    vehicle = Vehicle(
        vehicle_id="fork-1",
        robot_group="fork",
        current_node_id="fork:start",
        heading_rad=0.0,
        load_state=LoadState.EMPTY,
    )

    vehicle.transition(VehicleState.TO_PICKUP, 10)
    vehicle.transition(VehicleState.PICKING, 30)
    vehicle.transition(VehicleState.TO_DROPOFF, 50)
    vehicle.transition(VehicleState.DROPPING, 80)
    vehicle.transition(VehicleState.IDLE, 100)

    assert vehicle.durations_at(120) == {
        "DROPPING": 20,
        "IDLE": 30,
        "PICKING": 20,
        "TO_DROPOFF": 30,
        "TO_PICKUP": 20,
    }


def test_vehicle_state_machine_rejects_invalid_transition() -> None:
    vehicle = Vehicle(
        vehicle_id="fork-1",
        robot_group="fork",
        current_node_id="fork:start",
        heading_rad=0.0,
        load_state=LoadState.EMPTY,
    )

    with pytest.raises(DomainError) as error:
        vehicle.transition(VehicleState.DROPPING, 0)

    assert error.value.code == "vehicle.transition.invalid"
