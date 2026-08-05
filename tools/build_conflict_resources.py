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

    # 为每条边生成扫掠多边形
    edges = model["edges"]
    polygons = []
    for edge in edges:
        dimensions = profiles["robotGroups"][edge["robotGroup"]]["dimensions"]
        polygons.append(
            swept_polygon(
                edge,
                float(dimensions["length"]),
                float(dimensions["width"]),
                margin,
                args.sample_spacing,
            )
        )

    tree = STRtree(polygons)
    pair_rows: list[dict[str, Any]] = []
    resources_by_edge: dict[str, list[str]] = defaultdict(list)
    type_counts: Counter[str] = Counter()

    # 查找所有可能的冲突对
    for left_index, left_polygon in enumerate(polygons):
        # 只检查右边的索引大于左边的索引，避免重复计算
        for right_index in tree.query(left_polygon, predicate="intersects"):
            right_index = int(right_index)
            if right_index <= left_index:
                continue
            left_edge = edges[left_index]
            right_edge = edges[right_index]
            # 精确相交检查
            intersection = left_polygon.intersection(polygons[right_index])
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
        "conflictPairs": pair_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(result["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
