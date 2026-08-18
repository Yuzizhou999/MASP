from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree


ROTATION_EPSILON_RAD = math.radians(0.001)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# 曲线上取点，三次贝塞尔曲线
def cubic_point(edge: dict[str, Any], t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = edge["p0"], edge["p1"], edge["p2"], edge["p3"]
    u = 1.0 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )

# 曲线上的朝向
def cubic_heading(edge: dict[str, Any], t: float) -> float:
    p0, p1, p2, p3 = edge["p0"], edge["p1"], edge["p2"], edge["p3"]
    u = 1.0 - t
    dx = (
        3 * u * u * (p1[0] - p0[0])
        + 6 * u * t * (p2[0] - p1[0])
        + 3 * t * t * (p3[0] - p2[0])
    )
    dy = (
        3 * u * u * (p1[1] - p0[1])
        + 6 * u * t * (p2[1] - p1[1])
        + 3 * t * t * (p3[1] - p2[1])
    )
    if abs(dx) + abs(dy) < 1e-9:
        dx = p3[0] - p0[0]
        dy = p3[1] - p0[1]
    return math.degrees(math.atan2(dy, dx))


def edge_travel_heading_rad(edge: dict[str, Any], *, at_end: bool) -> float:
    heading = math.radians(cubic_heading(edge, 1.0 if at_end else 0.0))
    if int(edge.get("motionDirection", 0)) == 1:
        heading += math.pi
    return heading


def shortest_angle_delta(start_rad: float, end_rad: float) -> float:
    return (end_rad - start_rad + math.pi) % (2.0 * math.pi) - math.pi


def normalized_heading(heading_rad: float) -> float:
    result = (float(heading_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return 0.0 if abs(result) < ROTATION_EPSILON_RAD else result


def rotation_swept_polygon(
    node: dict[str, Any],
    length: float,
    width: float,
    margin: float,
    start_heading_rad: float,
    end_heading_rad: float,
    sample_spacing: float,
):
    footprint = box(
        -(length / 2.0 + margin),
        -(width / 2.0 + margin),
        length / 2.0 + margin,
        width / 2.0 + margin,
    )
    delta = shortest_angle_delta(start_heading_rad, end_heading_rad)
    corner_radius = math.hypot(length / 2.0 + margin, width / 2.0 + margin)
    angular_step = sample_spacing / max(corner_radius, 1e-9)
    sample_count = max(2, math.ceil(abs(delta) / max(angular_step, math.radians(1.0))) + 1)
    placements = []
    for index in range(sample_count):
        progress = index / (sample_count - 1)
        heading = start_heading_rad + delta * progress
        rotated = affinity.rotate(
            footprint, heading, origin=(0, 0), use_radians=True
        )
        placements.append(
            affinity.translate(rotated, xoff=float(node["x"]), yoff=float(node["y"]))
        )
    return unary_union(placements)


def rotation_action_specs(model: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = {node["id"]: node for node in model["nodes"]}
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    def register(
        node_id: str,
        group: str,
        start_heading: float,
        end_heading: float,
        reference_kind: str,
        reference: dict[str, str],
    ) -> None:
        start_heading = normalized_heading(start_heading)
        end_heading = normalized_heading(end_heading)
        if (
            abs(shortest_angle_delta(start_heading, end_heading))
            < ROTATION_EPSILON_RAD
        ):
            return
        key = (
            node_id,
            group,
            round(start_heading, 9),
            round(end_heading, 9),
        )
        item = by_key.setdefault(
            key,
            {
                "nodeId": node_id,
                "robotGroup": group,
                "startHeadingRad": start_heading,
                "endHeadingRad": end_heading,
                "edgePhases": [],
                "edgeTransitions": [],
            },
        )
        item[reference_kind].append(reference)

    for edge in sorted(model["edges"], key=lambda item: item["id"]):
        group = edge["robotGroup"]
        for phase, node_id, start_heading, end_heading in (
            (
                "start",
                edge["start"],
                float(
                    nodes[edge["start"]]
                    .get("headings", {})
                    .get(group, edge_travel_heading_rad(edge, at_end=False))
                ),
                edge_travel_heading_rad(edge, at_end=False),
            ),
            (
                "end",
                edge["end"],
                edge_travel_heading_rad(edge, at_end=True),
                float(
                    nodes[edge["end"]]
                    .get("headings", {})
                    .get(group, edge_travel_heading_rad(edge, at_end=True))
                ),
            ),
        ):
            register(
                node_id,
                group,
                start_heading,
                end_heading,
                "edgePhases",
                {"edgeId": edge["id"], "phase": phase},
            )

    incoming: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    outgoing: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in sorted(model["edges"], key=lambda item: item["id"]):
        incoming[(edge["end"], edge["robotGroup"])].append(edge)
        outgoing[(edge["start"], edge["robotGroup"])].append(edge)
    for node_group in sorted(set(incoming) & set(outgoing)):
        node_id, group = node_group
        for incoming_edge in incoming[node_group]:
            for outgoing_edge in outgoing[node_group]:
                register(
                    node_id,
                    group,
                    edge_travel_heading_rad(incoming_edge, at_end=True),
                    edge_travel_heading_rad(outgoing_edge, at_end=False),
                    "edgeTransitions",
                    {
                        "incomingEdgeId": incoming_edge["id"],
                        "outgoingEdgeId": outgoing_edge["id"],
                    },
                )
    result = []
    for index, item in enumerate(by_key.values()):
        result.append(
            {
                "rotationId": f"rotation:{index}",
                **item,
                "edgePhases": sorted(
                    item["edgePhases"], key=lambda value: (value["edgeId"], value["phase"])
                ),
                "edgeTransitions": sorted(
                    item["edgeTransitions"],
                    key=lambda value: (
                        value["incomingEdgeId"],
                        value["outgoingEdgeId"],
                    ),
                ),
            }
        )
    fallback_index = len(result)
    for node in sorted(model["nodes"], key=lambda item: item["id"]):
        for group in sorted(node["allowedRobotGroups"]):
            if not node["waitPolicyByGroup"][group]["allowed"]:
                continue
            result.append(
                {
                    "rotationId": f"rotation:{fallback_index}",
                    "nodeId": node["id"],
                    "robotGroup": group,
                    "startHeadingRad": 0.0,
                    "endHeadingRad": math.pi,
                    "edgePhases": [],
                    "edgeTransitions": [],
                    "arbitraryHeadingFallback": True,
                }
            )
            fallback_index += 1
    return result

# 车辆在曲线上行驶时可能占据的空间，即扫掠区域
def swept_polygon(
    edge: dict[str, Any],
    length: float,
    width: float,
    margin: float,
    sample_spacing: float,
):
    #车辆外形 = 一个矩形
    footprint = box(
        -(length / 2.0 + margin), # 长 + 两边预留的margin
        -(width / 2.0 + margin),
        length / 2.0 + margin,
        width / 2.0 + margin,
    )
    # 计算在曲线上采样的点数，至少采样3个点
    sample_count = max(3, math.ceil(edge["length"] / sample_spacing) + 1)
    placements = []
    for index in range(sample_count):
        t = index / (sample_count - 1)
        x, y = cubic_point(edge, t)
        rotated = affinity.rotate(footprint, cubic_heading(edge, t), origin=(0, 0), use_radians=False)
        placements.append(affinity.translate(rotated, xoff=x, yoff=y))
    # 把所有采样点的车辆外形合并成一个多边形，表示车辆在该曲线上行驶时可能占据的空间
    return unary_union(placements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate edge conflict resources from robot footprints")
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-spacing", type=float, default=0.25)
    parser.add_argument("--margin", type=float)
    args = parser.parse_args()

    # 加载地图模型和机器人配置文件
    model = load_json(args.map)
    profiles = load_json(args.profiles)
    margin = (
        float(args.margin)
        if args.margin is not None
        else float(profiles["simulationSafety"]["footprintMargin"])
    )

    # 为每条边和原地旋转动作生成扫掠多边形
    edges = model["edges"]
    edge_polygons = []
    for edge in edges:
        dimensions = profiles["robotGroups"][edge["robotGroup"]]["dimensions"]
        edge_polygons.append(
            swept_polygon(
                edge,
                float(dimensions["length"]),
                float(dimensions["width"]),
                margin,
                args.sample_spacing,
            )
        )

    nodes = {node["id"]: node for node in model["nodes"]}
    rotation_actions = rotation_action_specs(model)
    rotation_polygons = []
    for action in rotation_actions:
        dimensions = profiles["robotGroups"][action["robotGroup"]]["dimensions"]
        rotation_polygons.append(
            rotation_swept_polygon(
                nodes[action["nodeId"]],
                float(dimensions["length"]),
                float(dimensions["width"]),
                margin,
                float(action["startHeadingRad"]),
                float(action["endHeadingRad"]),
                args.sample_spacing,
            )
        )

    tree = STRtree(edge_polygons)
    pair_rows: list[dict[str, Any]] = []
    resources_by_edge: dict[str, list[str]] = defaultdict(list)
    type_counts: Counter[str] = Counter()

    # 查找所有可能的冲突对
    for left_index, left_polygon in enumerate(edge_polygons):
        # 只检查右边的索引大于左边的索引，避免重复计算
        for right_index in tree.query(left_polygon, predicate="intersects"):
            right_index = int(right_index)
            if right_index <= left_index:
                continue
            left_edge = edges[left_index]
            right_edge = edges[right_index]
            # 精确相交检查
            intersection = left_polygon.intersection(edge_polygons[right_index])
            if intersection.is_empty:
                continue
            pair_type = "-".join(sorted((left_edge["robotGroup"], right_edge["robotGroup"]))) # fork-fork / jack-jack / fork-jack
            resource_id = f"edge-conflict:{len(pair_rows)}"
            resources_by_edge[left_edge["id"]].append(resource_id)
            resources_by_edge[right_edge["id"]].append(resource_id)
            type_counts[pair_type] += 1
            pair_rows.append(
                {
                    "resourceId": resource_id,
                    "edgeA": left_edge["id"],
                    "edgeB": right_edge["id"],
                    "groupA": left_edge["robotGroup"],
                    "groupB": right_edge["robotGroup"],
                    "sharedCanonicalEndpoint": bool(
                        {left_edge["start"], left_edge["end"]}
                        & {right_edge["start"], right_edge["end"]}
                    ),
                    "intersectionArea": round(float(intersection.area), 6),
                }
            )

    motion_pair_rows: list[dict[str, Any]] = []
    resources_by_rotation: dict[str, list[str]] = defaultdict(list)
    rotation_tree = STRtree(rotation_polygons) if rotation_polygons else None
    for edge_index, edge_polygon in enumerate(edge_polygons):
        rotation_matches = (
            rotation_tree.query(edge_polygon, predicate="intersects")
            if rotation_tree is not None
            else ()
        )
        for rotation_index in rotation_matches:
            rotation_index = int(rotation_index)
            intersection = edge_polygon.intersection(rotation_polygons[rotation_index])
            if intersection.is_empty:
                continue
            edge = edges[edge_index]
            rotation = rotation_actions[rotation_index]
            resource_id = f"motion-conflict:{len(motion_pair_rows)}"
            resources_by_edge[edge["id"]].append(resource_id)
            resources_by_rotation[rotation["rotationId"]].append(resource_id)
            motion_pair_rows.append(
                {
                    "resourceId": resource_id,
                    "kind": "edge-rotation",
                    "edgeId": edge["id"],
                    "rotationId": rotation["rotationId"],
                    "intersectionArea": round(float(intersection.area), 6),
                }
            )

    for left_index, left_polygon in enumerate(rotation_polygons):
        rotation_matches = (
            rotation_tree.query(left_polygon, predicate="intersects")
            if rotation_tree is not None
            else ()
        )
        for right_index in rotation_matches:
            right_index = int(right_index)
            if right_index <= left_index:
                continue
            left = rotation_actions[left_index]
            right = rotation_actions[right_index]
            if left["nodeId"] == right["nodeId"]:
                continue
            intersection = left_polygon.intersection(rotation_polygons[right_index])
            if intersection.is_empty:
                continue
            resource_id = f"motion-conflict:{len(motion_pair_rows)}"
            resources_by_rotation[left["rotationId"]].append(resource_id)
            resources_by_rotation[right["rotationId"]].append(resource_id)
            motion_pair_rows.append(
                {
                    "resourceId": resource_id,
                    "kind": "rotation-rotation",
                    "rotationA": left["rotationId"],
                    "rotationB": right["rotationId"],
                    "intersectionArea": round(float(intersection.area), 6),
                }
            )

    # 给每条边汇总它涉及的所有冲突资源
    edge_resources = []
    for edge in edges:
        own_resource = f"edge:{edge['id']}"
        edge_resources.append(
            {
                "edgeId": edge["id"],
                "robotGroup": edge["robotGroup"],
                "ownResource": own_resource,
                "conflictResources": sorted(resources_by_edge[edge["id"]]),
            }
        )

    rotation_resources = [
        {
            **action,
            "ownResource": action["rotationId"],
            "conflictResources": sorted(resources_by_rotation[action["rotationId"]]),
        }
        for action in rotation_actions
    ]

    result = {
        "metadata": {
            "map": args.map.name,
            "profiles": args.profiles.name,
            "sampleSpacing": args.sample_spacing,
            "footprintMargin": margin,
            "baseGeometryOnly": margin == 0.0,
        },
        "stats": {
            "edgeCount": len(edges),
            "conflictPairCount": len(pair_rows),
            "rotationResourceCount": len(rotation_resources),
            "motionConflictPairCount": len(motion_pair_rows),
            "conflictTypeCounts": dict(sorted(type_counts.items())),
            "nodeResourceCount": len(model["nodes"]),
        },
        "nodeResources": [
            {
                "resourceId": f"node:{node['id']}",
                "nodeId": node["id"],
                "allowedRobotGroups": node["allowedRobotGroups"],
            }
            for node in model["nodes"]
        ],
        "edgeResources": edge_resources,
        "rotationResources": rotation_resources,
        "conflictPairs": pair_rows,
        "motionConflictPairs": motion_pair_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(result["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
