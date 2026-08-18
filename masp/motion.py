from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .domain import DomainError, LoadState


ROTATION_EPSILON_DEGREES = 0.001


def _profile_time(distance: float, maximum: float, acceleration: float, deceleration: float) -> float:
    if distance <= 0:
        return 0.0
    if maximum <= 0 or acceleration <= 0 or deceleration <= 0:
        raise DomainError("motion.profile.invalid", "motion limits must be positive")
    acceleration_distance = maximum * maximum / (2.0 * acceleration) # 加速段要走多远
    deceleration_distance = maximum * maximum / (2.0 * deceleration) # 减速段要走多远
    if distance >= acceleration_distance + deceleration_distance:
        # 距离够 → 梯形曲线：加速 → 匀速 → 减速
        cruise_distance = distance - acceleration_distance - deceleration_distance
        return maximum / acceleration + cruise_distance / maximum + maximum / deceleration
    # 距离不够 → 三角曲线：加速 → 减速
    peak = math.sqrt(2.0 * distance / (1.0 / acceleration + 1.0 / deceleration))
    return peak / acceleration + peak / deceleration

# 算两个角度（弧度）的差，并归一化到 [-180°, 180°] 再取绝对值。
def _angle_difference_degrees(left_rad: float, right_rad: float) -> float:
    difference = (right_rad - left_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(difference))


def _headings_equal(left_rad: float, right_rad: float) -> bool:
    return _angle_difference_degrees(left_rad, right_rad) < ROTATION_EPSILON_DEGREES

# 算出"进入这条边时车头该朝哪"和"离开这条边时车头朝哪"（用贝塞尔曲线的切线方向）
def _edge_tangent(edge: dict[str, Any], at_end: bool) -> float:
    p0, p1, p2, p3 = edge["p0"], edge["p1"], edge["p2"], edge["p3"]
    if at_end:
        dx, dy = p3[0] - p2[0], p3[1] - p2[1]
    else:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if abs(dx) + abs(dy) < 1e-12:
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    heading = math.atan2(dy, dx)
    return heading + (math.pi if int(edge.get("motionDirection", 0)) == 1 else 0.0)

# 用微分几何的曲率公式，算出这条边最弯的地方曲率多大
def _maximum_curvature(edge: dict[str, Any], samples: int = 16) -> float:
    # 沿曲线采样 16 个点，算每个点的曲率 k = |dx*ddy - dy*ddx| / (dx²+dy²)^1.5
    # 取最大值
    p0, p1, p2, p3 = edge["p0"], edge["p1"], edge["p2"], edge["p3"]
    maximum = 0.0
    for index in range(samples + 1):
        t = index / samples
        u = 1.0 - t
        dx = (
            3.0 * u * u * (p1[0] - p0[0])
            + 6.0 * u * t * (p2[0] - p1[0])
            + 3.0 * t * t * (p3[0] - p2[0])
        )
        dy = (
            3.0 * u * u * (p1[1] - p0[1])
            + 6.0 * u * t * (p2[1] - p1[1])
            + 3.0 * t * t * (p3[1] - p2[1])
        )
        ddx = 6.0 * u * (p2[0] - 2.0 * p1[0] + p0[0]) + 6.0 * t * (
            p3[0] - 2.0 * p2[0] + p1[0]
        )
        ddy = 6.0 * u * (p2[1] - 2.0 * p1[1] + p0[1]) + 6.0 * t * (
            p3[1] - 2.0 * p2[1] + p1[1]
        )
        denominator = (dx * dx + dy * dy) ** 1.5
        if denominator > 1e-12:
            maximum = max(maximum, abs(dx * ddy - dy * ddx) / denominator)
    return maximum


@dataclass(frozen=True)
class EdgeMotionPhases:
    start_rotation_ms: int
    linear_ms: int
    end_rotation_ms: int
    start_heading_rad: float
    travel_start_heading_rad: float
    travel_end_heading_rad: float
    end_heading_rad: float

    @property
    def duration_ms(self) -> int:
        return self.start_rotation_ms + self.linear_ms + self.end_rotation_ms


def _allocate_phase_durations(
    raw_ms: tuple[float, float, float], total_ms: int
) -> tuple[int, int, int]:
    """Scale raw phase durations to the quantized edge duration without drift."""

    raw_total = sum(raw_ms)
    if raw_total <= 0.0:
        return 0, total_ms, 0
    scaled = [value * total_ms / raw_total for value in raw_ms]
    result = [math.floor(value) for value in scaled]
    remainder = total_ms - sum(result)
    order = sorted(
        range(len(scaled)),
        key=lambda index: scaled[index] - result[index],
        reverse=True,
    )
    for index in order[:remainder]:
        result[index] += 1
    return result[0], result[1], result[2]

# 算一条边总耗时
class EdgeTravelTimeModel:
    def __init__(
        self,
        model: dict[str, Any],
        profiles: dict[str, Any],
        time_quantum_ms: int = 100,
    ) -> None:
        if time_quantum_ms <= 0:
            raise ValueError("time_quantum_ms must be positive")
        self.nodes = {item["id"]: item for item in model["nodes"]}
        self.profiles = profiles["robotGroups"]
        self.time_quantum_ms = time_quantum_ms
        self._duration_cache: dict[tuple[Any, ...], int] = {}
        self._phase_cache: dict[tuple[Any, ...], EdgeMotionPhases] = {}
        self._rotation_duration_cache: dict[
            tuple[str, LoadState, float, float], int
        ] = {}

    @staticmethod
    def _duration_key(
        edge: dict[str, Any], load_state: LoadState
    ) -> tuple[Any, ...]:
        return (
            edge["id"],
            edge["robotGroup"],
            edge["start"],
            edge["end"],
            float(edge["length"]),
            int(edge.get("motionDirection", 0)),
            edge.get("maxSpeed"),
            edge.get("loadMaxSpeed"),
            tuple(float(value) for point in ("p0", "p1", "p2", "p3") for value in edge[point]),
            load_state,
        )

    def duration_ms(self, edge: dict[str, Any], load_state: LoadState) -> int:
        cache_key = self._duration_key(edge, load_state)
        cached = self._duration_cache.get(cache_key)
        if cached is not None:
            return cached
        phases = self.motion_phases(edge, load_state)
        return phases.duration_ms

    def rotation_duration_ms(
        self,
        robot_group: str,
        load_state: LoadState,
        start_heading_rad: float,
        end_heading_rad: float,
    ) -> int:
        cache_key = (
            robot_group,
            load_state,
            round(float(start_heading_rad), 9),
            round(float(end_heading_rad), 9),
        )
        cached = self._rotation_duration_cache.get(cache_key)
        if cached is not None:
            return cached
        angle_degrees = _angle_difference_degrees(
            start_heading_rad, end_heading_rad
        )
        if angle_degrees < ROTATION_EPSILON_DEGREES:
            self._rotation_duration_cache[cache_key] = 0
            return 0
        state_profile = self.profiles[robot_group][
            "loaded" if load_state is LoadState.LOADED else "unloaded"
        ]
        seconds = _profile_time(
            angle_degrees,
            float(state_profile["maxRotationSpeed"]),
            float(state_profile["maxRotationAcceleration"]),
            float(state_profile["maxRotationDeceleration"]),
        )
        raw_ms = max(1, math.ceil(seconds * 1000.0))
        duration_ms = (
            math.ceil(raw_ms / self.time_quantum_ms) * self.time_quantum_ms
        )
        self._rotation_duration_cache[cache_key] = duration_ms
        return duration_ms

    def route_duration_ms(
        self,
        edges: Iterable[dict[str, Any]],
        load_state: LoadState,
        *,
        entry_heading_rad: float | None = None,
        terminal: bool = True,
    ) -> int:
        """Return the free-flow duration using continuous route headings.

        A route rotates into its first edge once, uses direct rotations between
        adjacent edges, and only rotates to the node heading when it is a
        terminal route. This is the same motion accounting used by SIPP.
        """

        items = tuple(edges)
        if not items:
            return 0
        phases = tuple(self.motion_phases(edge, load_state) for edge in items)
        entry_heading = (
            phases[0].start_heading_rad
            if entry_heading_rad is None
            else float(entry_heading_rad)
        )
        if _headings_equal(entry_heading, phases[0].start_heading_rad):
            total = phases[0].start_rotation_ms
        else:
            total = self.rotation_duration_ms(
                items[0]["robotGroup"],
                load_state,
                entry_heading,
                phases[0].travel_start_heading_rad,
            )
        total += sum(phase.linear_ms for phase in phases)
        for previous, following, edge in zip(
            phases, phases[1:], items[1:]
        ):
            total += self.rotation_duration_ms(
                edge["robotGroup"],
                load_state,
                previous.travel_end_heading_rad,
                following.travel_start_heading_rad,
            )
        if terminal:
            total += phases[-1].end_rotation_ms
        return total

    def motion_phases(self, edge: dict[str, Any], load_state: LoadState) -> EdgeMotionPhases:
        cache_key = self._duration_key(edge, load_state)
        cached = self._phase_cache.get(cache_key)
        if cached is not None:
            return cached
        group = edge["robotGroup"]
        state_profile = self.profiles[group][
            "loaded" if load_state is LoadState.LOADED else "unloaded"
        ]
        reverse = int(edge.get("motionDirection", 0)) == 1
        # 定"有效最高速度"（取最小值）
        profile_speed = float(
            state_profile["maxReverseSpeed" if reverse else "maxForwardSpeed"]
        )
        speed_limits = [profile_speed]
        edge_limit = (
            edge.get("loadMaxSpeed")
            if load_state is LoadState.LOADED
            else edge.get("maxSpeed")
        )
        if edge_limit is None and load_state is LoadState.LOADED:
            edge_limit = edge.get("maxSpeed")
        if edge_limit is not None:
            speed_limits.append(float(edge_limit))

        curvature = _maximum_curvature(edge)
        if curvature > 1e-9:
            lateral_acceleration = float(state_profile["maxAcceleration"])
            speed_limits.append(math.sqrt(lateral_acceleration / curvature))
        maximum_speed = min(speed_limits)
        # 算直线（沿路）时间
        linear_seconds = _profile_time(
            float(edge["length"]),
            maximum_speed,
            float(state_profile["maxAcceleration"]),
            float(state_profile["maxDeceleration"]),
        )

        # 算两端"转头"时间
        start_heading = float(
            self.nodes[edge["start"]].get("headings", {}).get(group, _edge_tangent(edge, False))
        )
        end_heading = float(
            self.nodes[edge["end"]].get("headings", {}).get(group, _edge_tangent(edge, True))
        )
        start_rotation = _angle_difference_degrees(start_heading, _edge_tangent(edge, False))
        end_rotation = _angle_difference_degrees(_edge_tangent(edge, True), end_heading)
        if start_rotation < ROTATION_EPSILON_DEGREES:
            start_rotation = 0.0
        if end_rotation < ROTATION_EPSILON_DEGREES:
            end_rotation = 0.0
        start_rotation_seconds = _profile_time(
            start_rotation,
            float(state_profile["maxRotationSpeed"]),
            float(state_profile["maxRotationAcceleration"]),
            float(state_profile["maxRotationDeceleration"]),
        )
        end_rotation_seconds = _profile_time(
            end_rotation,
            float(state_profile["maxRotationSpeed"]),
            float(state_profile["maxRotationAcceleration"]),
            float(state_profile["maxRotationDeceleration"]),
        )
        # 总耗时 = 沿路走 + 两端转头，这里向上取整到 100ms 的倍数
        raw_phase_ms = (
            start_rotation_seconds * 1000.0,
            linear_seconds * 1000.0,
            end_rotation_seconds * 1000.0,
        )
        raw_ms = math.ceil(sum(raw_phase_ms))
        duration_ms = max(
            self.time_quantum_ms,
            math.ceil(raw_ms / self.time_quantum_ms) * self.time_quantum_ms,
        )
        start_rotation_ms, linear_ms, end_rotation_ms = _allocate_phase_durations(
            raw_phase_ms, duration_ms
        )
        phases = EdgeMotionPhases(
            start_rotation_ms=start_rotation_ms,
            linear_ms=linear_ms,
            end_rotation_ms=end_rotation_ms,
            start_heading_rad=start_heading,
            travel_start_heading_rad=_edge_tangent(edge, False),
            travel_end_heading_rad=_edge_tangent(edge, True),
            end_heading_rad=end_heading,
        )
        self._phase_cache[cache_key] = phases
        self._duration_cache[cache_key] = duration_ms
        return phases
