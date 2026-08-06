from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from masp.deadlock import BlockedRequirement, DeadlockSupervisor
from masp.domain import LoadState
from masp.motion import EdgeTravelTimeModel
from masp.recovery import (
    RecoveryController,
    RecoveryPlanningError,
    RecoveryVehicle,
)
from masp.reservations import Reservation, ReservationConflict, ReservationTable
from masp.topology import MapTopology


def reservation(
    reservation_id: str,
    resource_id: str,
    vehicle_id: str,
    *,
    start_ms: int = 0,
    end_ms: int = 30_000,
) -> Reservation:
    return Reservation(
        reservation_id=reservation_id,
        resource_id=resource_id,
        vehicle_id=vehicle_id,
        plan_id=f"runtime:{vehicle_id}",
        segment_id="runtime-hold",
        start_ms=start_ms,
        end_ms=end_ms,
        kind="safety_hold",
        committed=True,
    )


def two_vehicle_deadlock() -> tuple[ReservationTable, tuple[BlockedRequirement, ...]]:
    table = ReservationTable()
    table.insert_batch(
        (
            reservation("hold-a", "edge-conflict:2423", "vehicle-a"),
            reservation("hold-b", "node:shared:LM1254", "vehicle-b"),
        )
    )
    requirements = (
        BlockedRequirement(
            vehicle_id="vehicle-a",
            resource_ids=("node:shared:LM1254",),
            start_ms=10_000,
            end_ms=11_000,
            blocked_since_ms=0,
        ),
        BlockedRequirement(
            vehicle_id="vehicle-b",
            resource_ids=("edge-conflict:2423",),
            start_ms=10_000,
            end_ms=11_000,
            blocked_since_ms=0,
        ),
    )
    return table, requirements


def requirements_at(
    requirements: tuple[BlockedRequirement, ...], now_ms: int
) -> tuple[BlockedRequirement, ...]:
    return tuple(
        replace(
            item,
            start_ms=now_ms,
            end_ms=now_ms + (item.end_ms - item.start_ms),
        )
        for item in requirements
    )


def recovery_controller(phase0_assets, scheduler=None) -> RecoveryController:
    scheduler = scheduler or phase0_assets["scheduler"]
    topology = MapTopology(
        phase0_assets["model"],
        phase0_assets["conflicts"],
        phase0_assets["workstations"],
        phase0_assets["traffic_zones"],
    )
    return RecoveryController(
        topology,
        EdgeTravelTimeModel(
            phase0_assets["model"],
            phase0_assets["profiles"],
            int(scheduler["planner"]["timeQuantumMs"]),
        ),
        scheduler,
        phase0_assets["traffic_zones"],
    )


def recovery_vehicles() -> tuple[RecoveryVehicle, ...]:
    return (
        RecoveryVehicle(
            vehicle_id="vehicle-a",
            robot_group="fork",
            load_state=LoadState.EMPTY,
            recovery_node_id="fork:PP1173",
            wait_since_ms=0,
            current_edge_id="fork:edge-323",
            edge_progress=0.99,
            held_resource_ids=("edge-conflict:2423",),
        ),
        RecoveryVehicle(
            vehicle_id="vehicle-b",
            robot_group="fork",
            load_state=LoadState.EMPTY,
            recovery_node_id="fork:PP1173",
            wait_since_ms=0,
            current_node_id="shared:LM1254",
        ),
    )


def test_wait_graph_detects_cycle_and_ages_starvation() -> None:
    table, requirements = two_vehicle_deadlock()
    supervisor = DeadlockSupervisor(starvation_age_step_ms=5_000)

    report = supervisor.analyze(10_000, requirements, table)

    assert report.cycles == (("vehicle-a", "vehicle-b"),)
    assert report.max_cycle_length == 2
    assert report.priority_age_ms == {"vehicle-a": 10_000, "vehicle-b": 10_000}
    assert {
        (item.waiting_vehicle_id, item.blocking_vehicle_id, item.resource_id)
        for item in report.dependencies
    } == {
        ("vehicle-a", "vehicle-b", "node:shared:LM1254"),
        ("vehicle-b", "vehicle-a", "edge-conflict:2423"),
    }


def test_wait_graph_ignores_requests_with_a_legal_alternative() -> None:
    table, requirements = two_vehicle_deadlock()
    supervisor = DeadlockSupervisor()
    alternatives = tuple(
        BlockedRequirement(
            **{
                **item.__dict__,
                "has_alternative": item.vehicle_id == "vehicle-a",
            }
        )
        for item in requirements
    )

    report = supervisor.analyze(10_000, alternatives, table)

    assert report.cycles == ()
    assert report.blocked_vehicle_ids == ("vehicle-b",)


def test_wait_graph_ignores_uncommitted_candidate_reservations() -> None:
    table, requirements = two_vehicle_deadlock()
    candidate_rows = tuple(
        Reservation(
            **{
                **item.__dict__,
                "reservation_id": f"candidate-{item.reservation_id}",
                "vehicle_id": f"candidate-{item.vehicle_id}",
                "committed": False,
            }
        )
        for item in table.snapshot()
    )
    candidate_only = ReservationTable()
    candidate_only.insert_batch(candidate_rows)

    report = DeadlockSupervisor().analyze(10_000, requirements, candidate_only)

    assert report.dependencies == ()
    assert report.cycles == ()
    assert report.blocked_vehicle_ids == ()


def test_four_vehicle_ring_is_reported_as_one_deterministic_cycle() -> None:
    vehicle_ids = tuple(f"vehicle-{index}" for index in range(4))
    table = ReservationTable()
    table.insert_batch(
        reservation(f"hold-{index}", f"ring-resource-{index}", vehicle_id)
        for index, vehicle_id in enumerate(vehicle_ids)
    )
    requirements = tuple(
        BlockedRequirement(
            vehicle_id=vehicle_id,
            resource_ids=(f"ring-resource-{(index + 1) % 4}",),
            start_ms=5_000,
            end_ms=6_000,
            blocked_since_ms=0,
        )
        for index, vehicle_id in enumerate(vehicle_ids)
    )

    report = DeadlockSupervisor().analyze(5_000, requirements, table)

    assert report.cycles == (vehicle_ids,)
    assert report.max_cycle_length == 4


def test_current_edge_reverse_is_reserved_atomically(phase0_assets) -> None:
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)
    controller = recovery_controller(phase0_assets)
    before = table.snapshot()

    decision = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )
    repeated = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )

    assert decision.action == "reverse"
    assert repeated == decision
    assert decision.plan is not None
    assert decision.plan.vehicle_id == "vehicle-a"
    assert decision.plan.recovery_node_id == "fork:PP1173"
    assert decision.plan.total_distance_m == pytest.approx(4.76388)
    assert [item.source_edge_id for item in decision.plan.segments] == [
        "fork:edge-323"
    ]
    assert all(item.kind == "reverse" for item in decision.plan.reservations[:-1])
    assert decision.freeze_reservation_ids
    assert len(table.snapshot()) == (
        len(before)
        + len(decision.plan.reservations)
        + len(decision.freeze_reservation_ids)
    )
    replaced_hold = next(
        item for item in table.snapshot() if item.reservation_id == "hold-a"
    )
    assert replaced_hold.end_ms == decision.plan.completed_at_ms
    terminal_hold = decision.plan.reservations[-1]
    assert terminal_hold.kind == "safety_hold"
    assert terminal_hold.end_ms - terminal_hold.start_ms == 30_000
    assert any(
        item.resource_id == "edge-conflict:2423"
        for item in decision.plan.reservations
    )
    with pytest.raises(ReservationConflict, match="conflicts with"):
        table.insert_batch(
            (
                reservation(
                    "third-vehicle-during-recovery",
                    decision.frozen_resource_ids[0],
                    "vehicle-c",
                    start_ms=12_000,
                    end_ms=17_000,
                ),
            )
        )


@pytest.mark.parametrize("mode", ["disabled", "map_edges_only"])
def test_dynamic_reverse_modes_are_enforced(phase0_assets, mode: str) -> None:
    scheduler = deepcopy(phase0_assets["scheduler"])
    scheduler["traffic"]["reverse"]["mode"] = mode
    controller = recovery_controller(phase0_assets, scheduler)
    table, _ = two_vehicle_deadlock()

    with pytest.raises(RecoveryPlanningError, match="dynamic reverse"):
        controller.plan_for_vehicle(
            recovery_vehicles()[0], table, now_ms=10_000, end_ms=50_000
        )


def test_loaded_reverse_policy_is_enforced(phase0_assets) -> None:
    scheduler = deepcopy(phase0_assets["scheduler"])
    scheduler["traffic"]["reverse"]["loadedAllowed"] = False
    controller = recovery_controller(phase0_assets, scheduler)
    table, _ = two_vehicle_deadlock()
    loaded = RecoveryVehicle(
        **{
            **recovery_vehicles()[0].__dict__,
            "load_state": LoadState.LOADED,
        }
    )

    with pytest.raises(RecoveryPlanningError, match="loaded reverse"):
        controller.plan_for_vehicle(loaded, table, now_ms=10_000, end_ms=50_000)


def test_unavailable_recovery_point_causes_stable_safety_stop(phase0_assets) -> None:
    table, requirements = two_vehicle_deadlock()
    table.insert_batch(
        (reservation("blocked-recovery", "node:fork:PP1173", "vehicle-c"),)
    )
    report = DeadlockSupervisor().analyze(10_000, requirements, table)
    controller = recovery_controller(phase0_assets)
    before = table.snapshot()

    decision = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )
    repeated = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )

    assert decision.action == "safety_stop"
    assert decision.reason_code == "deadlock.recovery_unavailable"
    assert repeated == decision
    assert decision.plan is None
    assert decision.freeze_reservation_ids
    assert len(table.snapshot()) == len(before) + len(decision.frozen_resource_ids)
    assert all(
        item.kind == "safety_freeze"
        for item in table.snapshot()
        if item.reservation_id in decision.freeze_reservation_ids
    )
    with pytest.raises(ReservationConflict, match="conflicts with"):
        table.insert_batch(
            (
                reservation(
                    "third-vehicle-entry",
                    decision.frozen_resource_ids[0],
                    "vehicle-d",
                    start_ms=30_000,
                    end_ms=31_000,
                ),
            )
        )


def test_repeated_cycle_is_stopped_as_livelock(phase0_assets) -> None:
    scheduler = deepcopy(phase0_assets["scheduler"])
    scheduler["traffic"]["deadlock"]["maxRecoveryAttempts"] = 1
    controller = recovery_controller(phase0_assets, scheduler)
    first_table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, first_table)

    first = controller.resolve(
        report,
        recovery_vehicles(),
        first_table,
        now_ms=10_000,
        end_ms=50_000,
    )
    repeated_while_active = controller.resolve(
        report,
        recovery_vehicles(),
        first_table,
        now_ms=10_000,
        end_ms=50_000,
    )
    first_plan_id = first.plan.plan_id if first.plan is not None else ""
    removed_count = controller.mark_recovery_failed(
        report.cycles[0], first_table
    )
    retry_report = DeadlockSupervisor().analyze(
        15_000,
        requirements_at(requirements, 15_000),
        first_table,
    )
    second = controller.resolve(
        retry_report,
        recovery_vehicles(),
        first_table,
        now_ms=15_000,
        end_ms=55_000,
    )

    assert first.action == "reverse"
    assert repeated_while_active == first
    assert removed_count == (
        len(first.plan.reservations if first.plan else ())
        + len(first.freeze_reservation_ids)
    )
    assert all(item.plan_id != first_plan_id for item in first_table.snapshot())
    assert second.action == "safety_stop"
    assert second.reason_code == "deadlock.livelock_detected"
    assert second.freeze_reservation_ids


def test_failed_recovery_can_retry_on_the_same_reservation_table(phase0_assets) -> None:
    controller = recovery_controller(phase0_assets)
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)

    first = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )
    assert first.plan is not None
    controller.mark_recovery_failed(report.cycles[0], table)
    retry_report = DeadlockSupervisor().analyze(
        15_000,
        requirements_at(requirements, 15_000),
        table,
    )
    second = controller.resolve(
        retry_report,
        recovery_vehicles(),
        table,
        now_ms=15_000,
        end_ms=60_000,
    )

    assert second.action == "reverse"
    assert second.plan is not None
    assert second.plan.plan_id != first.plan.plan_id
    assert all(item.plan_id != first.plan.plan_id for item in table.snapshot())


def test_recovery_cancels_conflicting_future_plans_without_restoring_them(
    phase0_assets,
) -> None:
    controller = recovery_controller(phase0_assets)
    table, requirements = two_vehicle_deadlock()
    hold_a = next(item for item in table.snapshot() if item.reservation_id == "hold-a")
    table.replace_vehicle("vehicle-a", (replace(hold_a, end_ms=12_000),))
    future_rows = (
        reservation(
            "future-entry",
            "edge-conflict:2423",
            "vehicle-c",
            start_ms=12_000,
            end_ms=17_000,
        ),
        reservation(
            "future-tail",
            "node:fork:PP1175",
            "vehicle-c",
            start_ms=17_000,
            end_ms=20_000,
        ),
    )
    table.insert_batch(future_rows)
    report = DeadlockSupervisor().analyze(10_000, requirements, table)

    decision = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )

    assert decision.action == "reverse"
    assert not {item.reservation_id for item in future_rows} & {
        item.reservation_id for item in table.snapshot()
    }
    controller.mark_recovery_failed(report.cycles[0], table)
    assert not {item.reservation_id for item in future_rows} & {
        item.reservation_id for item in table.snapshot()
    }


def test_expired_reverse_rows_do_not_invalidate_the_active_recovery_hold(
    phase0_assets,
) -> None:
    controller = recovery_controller(phase0_assets)
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)

    first = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )
    assert first.plan is not None
    after_reverse_ms = first.plan.completed_at_ms + 1
    table.expire_before(after_reverse_ms)
    repeated = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=after_reverse_ms,
        end_ms=60_000,
    )

    assert repeated == first


def test_recovery_decision_is_recovered_after_controller_restart(phase0_assets) -> None:
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)
    first = recovery_controller(phase0_assets).resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )
    assert first.plan is not None
    after_reverse_ms = first.plan.completed_at_ms + 1
    table.expire_before(after_reverse_ms)

    repeated = recovery_controller(phase0_assets).resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=after_reverse_ms,
        end_ms=60_000,
    )

    assert repeated == first


def test_stale_wait_graph_cannot_commit_a_recovery(phase0_assets) -> None:
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)
    table.expire_before(30_000)

    with pytest.raises(RecoveryPlanningError) as caught:
        recovery_controller(phase0_assets).resolve(
            report,
            recovery_vehicles(),
            table,
            now_ms=30_000,
            end_ms=70_000,
        )

    assert caught.value.code == "recovery.report.stale"
    assert table.snapshot() == ()


def test_failed_recovery_rejects_a_changed_reservation_table(phase0_assets) -> None:
    controller = recovery_controller(phase0_assets)
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)
    decision = controller.resolve(
        report,
        recovery_vehicles(),
        table,
        now_ms=10_000,
        end_ms=50_000,
    )
    assert decision.plan is not None
    table.insert_batch(
        (
            reservation(
                "unrelated-update",
                "node:fork:PP1175",
                "vehicle-z",
                start_ms=40_000,
                end_ms=41_000,
            ),
        )
    )

    with pytest.raises(RecoveryPlanningError) as caught:
        controller.mark_recovery_failed(report.cycles[0], table)

    assert caught.value.code == "recovery.transaction.stale"
    assert any(item.plan_id == decision.plan.plan_id for item in table.snapshot())


def test_reverse_of_a_map_reverse_edge_uses_forward_motion_limits(
    phase0_assets,
) -> None:
    controller = recovery_controller(phase0_assets)
    edge_value = controller.topology.edges["fork:edge-71"]
    vehicle = RecoveryVehicle(
        vehicle_id="reverse-edge-vehicle",
        robot_group="fork",
        load_state=LoadState.EMPTY,
        recovery_node_id="fork:CP1039",
        wait_since_ms=0,
        current_edge_id=edge_value["id"],
        edge_progress=0.5,
    )

    plan = controller.plan_for_vehicle(
        vehicle,
        ReservationTable(),
        now_ms=0,
        end_ms=40_000,
    )
    reverse_edge = {
        **edge_value,
        "start": edge_value["end"],
        "end": edge_value["start"],
        "p0": edge_value["p3"],
        "p1": edge_value["p2"],
        "p2": edge_value["p1"],
        "p3": edge_value["p0"],
        "length": float(edge_value["length"]) * 0.5,
        "motionDirection": 0,
    }

    assert plan.segments[0].end_ms - plan.segments[0].start_ms == (
        controller.travel_times.duration_ms(reverse_edge, LoadState.EMPTY)
    )


def test_recovery_requires_full_terminal_hold_horizon(phase0_assets) -> None:
    controller = recovery_controller(phase0_assets)
    table, _ = two_vehicle_deadlock()

    with pytest.raises(RecoveryPlanningError) as caught:
        controller.plan_for_vehicle(
            recovery_vehicles()[0],
            table,
            now_ms=10_000,
            end_ms=25_000,
        )

    assert caught.value.code == "recovery.hold_horizon.exceeded"


def test_on_edge_recovery_rejects_endpoint_progress() -> None:
    with pytest.raises(ValueError, match=r"edge_progress in \(0, 1\)"):
        RecoveryVehicle(
            vehicle_id="endpoint-vehicle",
            robot_group="fork",
            load_state=LoadState.EMPTY,
            recovery_node_id="fork:PP1173",
            wait_since_ms=0,
            current_edge_id="fork:edge-323",
            edge_progress=1.0,
        )


def test_duplicate_recovery_vehicle_ids_are_rejected(phase0_assets) -> None:
    table, requirements = two_vehicle_deadlock()
    report = DeadlockSupervisor().analyze(10_000, requirements, table)
    controller = recovery_controller(phase0_assets)
    duplicate = recovery_vehicles()[0]

    with pytest.raises(RecoveryPlanningError) as caught:
        controller.resolve(
            report,
            (duplicate, duplicate),
            table,
            now_ms=10_000,
            end_ms=50_000,
        )

    assert caught.value.code == "recovery.vehicle.duplicate"
