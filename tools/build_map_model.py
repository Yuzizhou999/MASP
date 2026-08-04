from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


POINT_TYPES = {
    "LocationMark": "LM",
    "ActionPoint": "AP",
    "ParkPoint": "PP",
    "ChargePoint": "CP",
}


def typed_properties(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    value_fields = (
        "boolValue",
        "int32Value",
        "int64Value",
        "doubleValue",
        "stringValue",
    )
    for item in items or []:
        value = None
        for field in value_fields:
            if field in item:
                value = item[field]
                break
        result[item["key"]] = value
    return result


def cubic_point(
    p0: dict[str, float],
    p1: dict[str, float],
    p2: dict[str, float],
    p3: dict[str, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = (
        u * u * u * p0["x"]
        + 3.0 * u * u * t * p1["x"]
        + 3.0 * u * t * t * p2["x"]
        + t * t * t * p3["x"]
    )
    y = (
        u * u * u * p0["y"]
        + 3.0 * u * u * t * p1["y"]
        + 3.0 * u * t * t * p2["y"]
        + t * t * t * p3["y"]
    )
    return x, y


def cubic_length(points: list[dict[str, float]], samples: int = 32) -> float:
    previous = cubic_point(*points, 0.0)
    total = 0.0
    for index in range(1, samples + 1):
        current = cubic_point(*points, index / samples)
        total += math.dist(previous, current)
        previous = current
    return total


def reachable(graph: dict[str, list[str]], start: str) -> set[str]:
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def build_model(source: Path) -> dict[str, Any]:
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    type_counts: Counter[str] = Counter()

    for point in raw["advancedPointList"]:
        node_id = point["instanceName"]
        node_type = POINT_TYPES.get(point["className"], point["className"])
        props = typed_properties(point.get("property", []))
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "className": point["className"],
                "x": round(float(point["pos"]["x"]), 4),
                "y": round(float(point["pos"]["y"]), 4),
                "heading": round(float(point.get("dir", 0.0)), 6),
                "allowWait": node_type in {"PP", "CP"},
                "properties": props,
            }
        )
        node_ids.add(node_id)
        type_counts[node_type] += 1

    graph = {node_id: [] for node_id in node_ids}
    reverse_graph = {node_id: [] for node_id in node_ids}
    edges: list[dict[str, Any]] = []
    missing_endpoints: list[str] = []

    for index, curve in enumerate(raw["advancedCurveList"]):
        start = curve["startPos"]["instanceName"]
        end = curve["endPos"]["instanceName"]
        if start not in node_ids or end not in node_ids:
            missing_endpoints.append(curve.get("instanceName", str(index)))
            continue

        props = typed_properties(curve.get("property", []))
        start_pos = curve["startPos"]["pos"]
        end_pos = curve["endPos"]["pos"]
        if curve.get("className") == "StraightPath":
            control_pos1 = {
                "x": start_pos["x"] + (end_pos["x"] - start_pos["x"]) / 3.0,
                "y": start_pos["y"] + (end_pos["y"] - start_pos["y"]) / 3.0,
            }
            control_pos2 = {
                "x": start_pos["x"] + 2.0 * (end_pos["x"] - start_pos["x"]) / 3.0,
                "y": start_pos["y"] + 2.0 * (end_pos["y"] - start_pos["y"]) / 3.0,
            }
        else:
            control_pos1 = curve["controlPos1"]
            control_pos2 = curve["controlPos2"]

        control_points = [start_pos, control_pos1, control_pos2, end_pos]
        graph[start].append(end)
        reverse_graph[end].append(start)
        edges.append(
            {
                "id": f"edge-{index}",
                "name": curve.get("instanceName", f"{start}-{end}"),
                "pathClass": curve.get("className", "UnknownPath"),
                "start": start,
                "end": end,
                "p0": [round(control_points[0]["x"], 4), round(control_points[0]["y"], 4)],
                "p1": [round(control_points[1]["x"], 4), round(control_points[1]["y"], 4)],
                "p2": [round(control_points[2]["x"], 4), round(control_points[2]["y"], 4)],
                "p3": [round(control_points[3]["x"], 4), round(control_points[3]["y"], 4)],
                "length": round(cubic_length(control_points), 3),
                "motionDirection": props.get("direction", 0),
                "moveStyle": props.get("movestyle", 0),
                "maxSpeed": props.get("maxspeed"),
                "loadMaxSpeed": props.get("loadMaxSpeed"),
            }
        )

    if not nodes:
        raise ValueError("The map contains no routing nodes")

    first = nodes[0]["id"]
    forward_count = len(reachable(graph, first))
    reverse_count = len(reachable(reverse_graph, first))
    xs = [node["x"] for node in nodes]
    ys = [node["y"] for node in nodes]
    background_step = 0.25
    background_cells = sorted(
        {
            (
                round(float(point["x"]) / background_step),
                round(float(point["y"]) / background_step),
            )
            for point in raw.get("normalPosList", [])
            if "x" in point and "y" in point
        }
    )
    header = raw.get("header", {})
    min_pos = header.get("minPos", {})
    max_pos = header.get("maxPos", {})

    return {
        "metadata": {
            "source": source.name,
            "mapName": header.get("mapName", source.stem),
            "mapType": header.get("mapType"),
            "version": header.get("version"),
            "resolution": header.get("resolution"),
            "bounds": {
                "minX": float(min_pos.get("x", min(xs))),
                "maxX": float(max_pos.get("x", max(xs))),
                "minY": float(min_pos.get("y", min(ys))),
                "maxY": float(max_pos.get("y", max(ys))),
            },
            "routeBounds": {
                "minX": min(xs),
                "maxX": max(xs),
                "minY": min(ys),
                "maxY": max(ys),
            },
        },
        "stats": {
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "typeCounts": dict(sorted(type_counts.items())),
            "missingEndpointCount": len(missing_endpoints),
            "forwardReachable": forward_count,
            "reverseReachable": reverse_count,
            "stronglyConnected": forward_count == len(nodes) and reverse_count == len(nodes),
            "rawBackgroundPointCount": len(raw.get("normalPosList", [])),
            "displayBackgroundPointCount": len(background_cells),
        },
        "background": {
            "step": background_step,
            "cells": background_cells,
        },
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a .smap file to a scheduling graph model")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = build_model(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, separators=(",", ":"))

    stats = model["stats"]
    print(
        f"built {stats['nodeCount']} nodes and {stats['edgeCount']} directed edges; "
        f"stronglyConnected={stats['stronglyConnected']}"
    )


if __name__ == "__main__":
    main()
