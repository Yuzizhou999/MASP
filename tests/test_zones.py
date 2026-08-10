from __future__ import annotations

from copy import deepcopy

import pytest

from masp.domain import DomainError, LoadState, PlanSegment, SegmentKind, Vehicle
from masp.motion import EdgeTravelTimeModel
from masp.reservations import (
    RelativeReservationRequest,
    Reservation,
    ReservationTable,
)
from masp.routing import RouteProvider, SpatialRoute
from masp.sipp import ContinuousTimeSippPlanner, SippPlanningError
from masp.topology import MapTopology


def node(node_id: str, node_type: str, x: float, wait_allowed: bool) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "x": x,
        "y": 0.0,
        "allowedRobotGroups": ["fork"],
        "headings": {"fork": 0.0},
        "waitPolicyByGroup": {
            "fork": {
                "allowed": wait_allowed,
                "maxWaitMs": 60_000 if wait_allowed else 0,
            }
        },
    }


def edge(edge_id: str, start: str, end: str, x0: float, x1: float) -> dict:
    one_third = (x1 - x0) / 3.0
    return {
        "id": edge_id,
        "start": start,
        "end": end,
        "p0": [x0, 0.0],
        "p1": [x0 + one_third, 0.0],
        "p2": [x0 + 2.0 * one_third, 0.0],
        "p3": [x1, 0.0],
        "length": abs(x1 - x0),
        "motionDirection": 0,
        "maxSpeed": None,
        "loadMaxSpeed": None,
        "robotGroup": "fork",
    }


def documents() -> tuple[dict, dict, dict, dict, dict, dict]:
    model = {
        "nodes": [
            node("A", "PP", 0.0, True),
            # Even a globally waitable member node is unsafe for ordinary zone waiting.
            node("I1", "PP", 1.0, True),
            node("I2", "LM", 2.0, False),
            node("Y", "LM", 3.0, False),
            node("X", "PP", 4.0, True),
        ],
        "edges": [
            edge("e-entry", "A", "I1", 0.0, 1.0),
            edge("e-member", "I1", "I2", 1.0, 2.0),
            edge("e-exit", "I2", "Y", 2.0, 3.0),
            edge("e-after", "Y", "X", 3.0, 4.0),
        ],
    }
    conflicts = {
        "edgeResources": [
            {
                "edgeId": item["id"],
                "ownResource": f"edge:{item['id']}",
                "conflictResources": [],
            }
            for item in model["edges"]
        ]
    }
    workstations = {"workstations": []}
    state_profile = {
        "maxForwardSpeed": 1.0,
        "maxReverseSpeed": 1.0,
        "maxAcceleration": 1.0,
        "maxDeceleration": 1.0,
        "maxRotationSpeed": 90.0,
        "maxRotationAcceleration": 90.0,
        "maxRotationDeceleration": 90.0,
    }
    profiles = {
        "robotGroups": {
            "fork": {
                "unloaded": dict(state_profile),
                "loaded": dict(state_profile),
            }
        }
    }
    scheduler = {
        "planner": {
            "candidateRouteCount": 3,
            "timeQuantumMs": 100,
            "maxSippScheduleAttempts": 200,
        },
        "traffic": {"wait": {"maxPlannedWaitMs": 60_000}},
    }
    zones = {
        "recoveryNodes": [{"nodeId": "A", "allowedRobotGroups": ["fork"]}],
        "zones": [
            {
                "id": "narrow",
                "memberNodeIds": ["I1", "I2"],
                "memberEdgeIds": ["e-member"],
                "entryEdgeIds": ["e-entry"],
                "exitEdgeIds": ["e-exit"],
                "capacity": 1,
                "passingAllowed": False,
                "directionalMode": "single_direction_at_a_time",
                "recoveryNodeIds": ["A"],
            }
        ],
    }
    return model, conflicts, workstations, profiles, scheduler, zones


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("capacity", 2, "zone.mvp.capacity"),
        ("passingAllowed", True, "zone.mvp.passing"),
        ("directionalMode", "unrestricted", "zone.mvp.directional_mode"),
    ],
)
def test_zone_mvp_rejects_unsupported_modes(field: str, value: object, code: str) -> None:
    model, conflicts, workstations, _, _, zones = documents()
    invalid = deepcopy(zones)
    invalid["zones"][0][field] = value

    with pytest.raises(DomainError) as caught:
        MapTopology(model, conflicts, workstations, invalid)

    assert caught.value.code == code


def test_zone_resource_covers_boundary_member_and_internal_node_actions() -> None:
    model, conflicts, workstations, _, _, zones = documents()
    topology = MapTopology(model, conflicts, workstations, zones)

    for edge_id in ("e-entry", "e-member", "e-exit"):
        edge_value = topology.edges[edge_id]
        segment = PlanSegment(
            segment_id=edge_id,
            kind=SegmentKind.TRAVERSE,
            start_ms=0,
            end_ms=100,
            start_node_id=edge_value["start"],
            end_node_id=edge_value["end"],
            edge_id=edge_id,
            expected_load_state=LoadState.EMPTY,
        )
        assert "zone:narrow" in topology.derived_resources(segment)

    outside = topology.edges["e-after"]
    outside_segment = PlanSegment(
        segment_id="outside",
        kind=SegmentKind.TRAVERSE,
        start_ms=0,
        end_ms=100,
        start_node_id=outside["start"],
        end_node_id=outside["end"],
        edge_id=outside["id"],
        expected_load_state=LoadState.EMPTY,
    )
    assert "zone:narrow" not in topology.derived_resources(outside_segment)
    assert topology.wait_allowed("I1", "fork") is False


def test_relative_bundle_search_reports_deterministic_blockers() -> None:
    table = ReservationTable()
    table.insert_batch(
        [
            Reservation(
                "zone-blocker",
                "zone:narrow",
                "vehicle-b",
                "plan-b",
                "segment-b",
                100,
                300,
                "transit",
                True,
            ),
            Reservation(
                "exit-blocker",
                "node:X",
                "vehicle-c",
                "plan-c",
                "segment-c",
                500,
                700,
                "safety_hold",
                True,
            ),
        ]
    )

    result = table.first_available_bundle_start(
        [
            RelativeReservationRequest("zone:narrow", 0, 200),
            RelativeReservationRequest("node:X", 200, 300),
        ],
        not_before_ms=0,
        vehicle_id="vehicle-a",
    )

    assert result.start_ms == 500
    assert [item.reservation.reservation_id for item in result.blockers] == [
        "zone-blocker",
        "exit-blocker",
    ]
    assert [item.reservation_id for item in table.overlapping("zone:narrow", 0, 400)] == [
        "zone-blocker"
    ]


def test_route_intent_rejects_edge_for_another_robot_group() -> None:
    model, conflicts, workstations, profiles, scheduler, zones = documents()
    foreign_edge = edge("e-jack", "A", "X", 0.0, 4.0)
    foreign_edge["robotGroup"] = "jack"
    model["edges"].append(foreign_edge)
    conflicts["edgeResources"].append(
        {
            "edgeId": foreign_edge["id"],
            "ownResource": f"edge:{foreign_edge['id']}",
            "conflictResources": [],
        }
    )
    topology = MapTopology(model, conflicts, workstations, zones)
    travel_times = EdgeTravelTimeModel(model, profiles, time_quantum_ms=100)
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel_times),
        travel_times,
        scheduler,
        recovery_node_ids=("A", "X"),
    )
    vehicle = Vehicle(
        vehicle_id="vehicle-a",
        robot_group="fork",
        current_node_id="A",
        heading_rad=0.0,
        load_state=LoadState.EMPTY,
    )

    with pytest.raises(SippPlanningError) as caught:
        planner.schedule_route_intent(
            vehicle,
            SpatialRoute("A", "X", ("e-jack",), 0),
            ready_ms=0,
            load_state=LoadState.EMPTY,
            reservations=ReservationTable(),
            horizon_end_ms=100_000,
        )

    assert caught.value.code == "sipp.route_intent.robot_group"


def test_route_intent_rejects_discontinuous_non_zone_edges() -> None:
    model, conflicts, workstations, profiles, scheduler, zones = documents()
    model["nodes"].extend(
        [
            node("B", "PP", 5.0, True),
            node("C", "PP", 6.0, True),
        ]
    )
    outside_edges = [
        edge("e-outside-ab", "A", "B", 0.0, 5.0),
        edge("e-outside-cx", "C", "X", 6.0, 4.0),
    ]
    model["edges"].extend(outside_edges)
    conflicts["edgeResources"].extend(
        {
            "edgeId": item["id"],
            "ownResource": f"edge:{item['id']}",
            "conflictResources": [],
        }
        for item in outside_edges
    )
    topology = MapTopology(model, conflicts, workstations, zones)
    travel_times = EdgeTravelTimeModel(model, profiles, time_quantum_ms=100)
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel_times),
        travel_times,
        scheduler,
        recovery_node_ids=("A", "X"),
    )
    vehicle = Vehicle(
        vehicle_id="vehicle-a",
        robot_group="fork",
        current_node_id="A",
        heading_rad=0.0,
        load_state=LoadState.EMPTY,
    )

    with pytest.raises(SippPlanningError) as caught:
        planner.schedule_route_intent(
            vehicle,
            SpatialRoute(
                "A",
                "X",
                ("e-outside-ab", "e-outside-cx"),
                0,
            ),
            ready_ms=0,
            load_state=LoadState.EMPTY,
            reservations=ReservationTable(),
            horizon_end_ms=100_000,
        )

    assert caught.value.code == "sipp.route_intent.discontinuous"


def test_sipp_delays_whole_zone_section_at_the_outside_entry_node() -> None:
    model, conflicts, workstations, profiles, scheduler, zones = documents()
    topology = MapTopology(model, conflicts, workstations, zones)
    travel_times = EdgeTravelTimeModel(model, profiles, time_quantum_ms=100)
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel_times),
        travel_times,
        scheduler,
        recovery_node_ids=("A", "X"),
    )
    vehicle = Vehicle(
        vehicle_id="vehicle-a",
        robot_group="fork",
        current_node_id="A",
        heading_rad=0.0,
        load_state=LoadState.EMPTY,
    )
    route = SpatialRoute(
        start_node_id="A",
        end_node_id="X",
        edge_ids=("e-entry", "e-member", "e-exit", "e-after"),
        free_flow_travel_ms=0,
    )
    durations = [
        travel_times.duration_ms(topology.edges[edge_id], LoadState.EMPTY)
        for edge_id in route.edge_ids
    ]
    after_offset = sum(durations[:3])
    reservations = ReservationTable()
    reservations.insert_batch(
        [
            Reservation(
                reservation_id="block-after-exit",
                resource_id="edge:e-after",
                vehicle_id="vehicle-b",
                plan_id="plan-b",
                segment_id="segment-b",
                start_ms=after_offset,
                end_ms=after_offset + 5_000,
                kind="transit",
                committed=True,
            )
        ]
    )

    segments = planner.schedule_route_intent(
        vehicle,
        route,
        ready_ms=0,
        load_state=LoadState.EMPTY,
        reservations=reservations,
        horizon_end_ms=100_000,
    )

    waits = [item for item in segments if item.kind is SegmentKind.WAIT]
    traversals = [item for item in segments if item.kind is SegmentKind.TRAVERSE]
    expected_delay = 5_000
    assert [(item.start_node_id, item.start_ms, item.end_ms) for item in waits] == [
        ("A", 0, expected_delay)
    ]
    assert [item.edge_id for item in traversals] == list(route.edge_ids)
    assert traversals[0].start_ms == expected_delay
    assert all(
        left.end_ms == right.start_ms for left, right in zip(traversals, traversals[1:])
    )
    assert all(
        "zone:narrow" in item.resource_ids for item in traversals[:3]
    )
    assert "zone:narrow" not in traversals[3].resource_ids
    assert segments[-1].end_ms == expected_delay + sum(durations)


def test_sipp_allows_zone_route_to_end_before_an_outside_safe_node() -> None:
    model, conflicts, workstations, profiles, scheduler, zones = documents()
    topology = MapTopology(model, conflicts, workstations, zones)
    travel_times = EdgeTravelTimeModel(model, profiles, time_quantum_ms=100)
    planner = ContinuousTimeSippPlanner(
        topology,
        RouteProvider(model, travel_times),
        travel_times,
        scheduler,
        recovery_node_ids=("A", "X"),
    )
    vehicle = Vehicle(
        vehicle_id="vehicle-a",
        robot_group="fork",
        current_node_id="A",
        heading_rad=0.0,
        load_state=LoadState.EMPTY,
    )
    route = SpatialRoute(
        start_node_id="A",
        end_node_id="Y",
        edge_ids=("e-entry", "e-member", "e-exit"),
        free_flow_travel_ms=0,
    )

    segments = planner.schedule_route_intent(
        vehicle,
        route,
        ready_ms=0,
        load_state=LoadState.EMPTY,
        reservations=ReservationTable(),
        horizon_end_ms=100_000,
    )

    traversals = [item for item in segments if item.kind is SegmentKind.TRAVERSE]
    assert [item.edge_id for item in traversals] == list(route.edge_ids)
    assert all("zone:narrow" in item.resource_ids for item in traversals)
    assert all(
        left.end_ms == right.start_ms for left, right in zip(traversals, traversals[1:])
    )
