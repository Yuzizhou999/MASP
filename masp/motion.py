from __future__ import annotations

import math
from typing import Any

from .domain import DomainError, LoadState


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

    def duration_ms(self, edge: dict[str, Any], load_state: LoadState) -> int:
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
        angular_seconds = _profile_time(
            start_rotation,
            float(state_profile["maxRotationSpeed"]),
            float(state_profile["maxRotationAcceleration"]),
            float(state_profile["maxRotationDeceleration"]),
        ) + _profile_time(
            end_rotation,
            float(state_profile["maxRotationSpeed"]),
            float(state_profile["maxRotationAcceleration"]),
            float(state_profile["maxRotationDeceleration"]),
        )
        # 总耗时 = 沿路走 + 两端转头，这里向上取整到 100ms 的倍数
        raw_ms = math.ceil((linear_seconds + angular_seconds) * 1000.0)
        return max(
            self.time_quantum_ms,
            math.ceil(raw_ms / self.time_quantum_ms) * self.time_quantum_ms,
        )
