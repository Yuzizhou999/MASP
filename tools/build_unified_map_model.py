from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.hypot(left["x"] - right["x"], left["y"] - right["y"])


def cubic_point(edge: dict[str, Any], t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = edge["p0"], edge["p1"], edge["p2"], edge["p3"]
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
    return x, y


def path_deviation(
    left: dict[str, Any], right: dict[str, Any], reverse_right: bool, samples: int = 24
) -> tuple[float, float]:
    distances: list[float] = []
    for index in range(samples + 1):
        t = index / samples
        right_t = 1.0 - t if reverse_right else t
        distances.append(math.dist(cubic_point(left, t), cubic_point(right, right_t)))
    return max(distances), sum(distances) / len(distances)


def canonical_node(
    canonical_id: str,
    group_nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    positions = {
        group: [node["x"], node["y"]]
        for group, node in sorted(group_nodes.items())
    }
    headings = {
        group: node["heading"]
        for group, node in sorted(group_nodes.items())
    }
    coordinates = list(positions.values())
    return {
        "id": canonical_id,
        "type": next(iter(group_nodes.values()))["type"],
        "x": round(sum(pos[0] for pos in coordinates) / len(coordinates), 4),
        "y": round(sum(pos[1] for pos in coordinates) / len(coordinates), 4),
        "allowedRobotGroups": sorted(group_nodes),
        "aliases": {
            group: node["id"]
            for group, node in sorted(group_nodes.items())
        },
        "positions": positions,
        "headings": headings,
        "allowWaitByGroup": {
            group: node["allowWait"]
            for group, node in sorted(group_nodes.items())
        },
    }


def build_unified_model(
    fork: dict[str, Any],
    jack: dict[str, Any],
    same_id_tolerance: float,
    alias_tolerance: float,
    path_tolerance: float,
) -> dict[str, Any]:
    group_models = {"fork": fork, "jack": jack}
    group_nodes = {
        group: {node["id"]: node for node in model["nodes"]}
        for group, model in group_models.items()
    }
    fork_nodes = group_nodes["fork"]
    jack_nodes = group_nodes["jack"]
    matched_fork: set[str] = set()
    matched_jack: set[str] = set()
    node_map: dict[tuple[str, str], str] = {}
    nodes: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []

    for node_id in sorted(set(fork_nodes) & set(jack_nodes)):
        fork_node = fork_nodes[node_id]
        jack_node = jack_nodes[node_id]
        distance = node_distance(fork_node, jack_node)
        if fork_node["type"] == jack_node["type"] and distance <= same_id_tolerance:
            canonical_id = f"shared:{node_id}"
            nodes.append(canonical_node(canonical_id, {"fork": fork_node, "jack": jack_node}))
            node_map[("fork", node_id)] = canonical_id
            node_map[("jack", node_id)] = canonical_id
            matched_fork.add(node_id)
            matched_jack.add(node_id)
            match_records.append(
                {
                    "canonicalNode": canonical_id,
                    "forkNode": node_id,
                    "jackNode": node_id,
                    "matchReason": "same-id",
                    "distance": round(distance, 4),
                }
            )

    alias_candidates: list[tuple[float, str, str]] = []
    for fork_id, fork_node in fork_nodes.items():
        if fork_id in matched_fork:
            continue
        for jack_id, jack_node in jack_nodes.items():
            if jack_id in matched_jack or fork_node["type"] != jack_node["type"]:
                continue
            distance = node_distance(fork_node, jack_node)
            if distance <= alias_tolerance:
                alias_candidates.append((distance, fork_id, jack_id))

    for distance, fork_id, jack_id in sorted(alias_candidates):
        if fork_id in matched_fork or jack_id in matched_jack:
            continue
        canonical_id = f"shared:{fork_id}|{jack_id}"
        nodes.append(
            canonical_node(
                canonical_id,
                {"fork": fork_nodes[fork_id], "jack": jack_nodes[jack_id]},
            )
        )
        node_map[("fork", fork_id)] = canonical_id
        node_map[("jack", jack_id)] = canonical_id
        matched_fork.add(fork_id)
        matched_jack.add(jack_id)
        match_records.append(
            {
                "canonicalNode": canonical_id,
                "forkNode": fork_id,
                "jackNode": jack_id,
                "matchReason": "coordinate-alias",
                "distance": round(distance, 4),
            }
        )

    for group, source_nodes, matched in (
        ("fork", fork_nodes, matched_fork),
        ("jack", jack_nodes, matched_jack),
    ):
        for node_id in sorted(set(source_nodes) - matched):
            canonical_id = f"{group}:{node_id}"
            nodes.append(canonical_node(canonical_id, {group: source_nodes[node_id]}))
            node_map[(group, node_id)] = canonical_id

    edges: list[dict[str, Any]] = []
    edges_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group, model in group_models.items():
        for edge in model["edges"]:
            unified_edge = {
                **edge,
                "id": f"{group}:{edge['id']}",
                "robotGroup": group,
                "localStart": edge["start"],
                "localEnd": edge["end"],
                "start": node_map[(group, edge["start"])],
                "end": node_map[(group, edge["end"])],
                "allowedRobotGroups": [group],
            }
            edges.append(unified_edge)
            edges_by_group[group].append(unified_edge)

    fork_buckets: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    jack_buckets: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges_by_group["fork"]:
        fork_buckets[frozenset((edge["start"], edge["end"]))].append(edge)
    for edge in edges_by_group["jack"]:
        jack_buckets[frozenset((edge["start"], edge["end"]))].append(edge)

    shared_path_matches: list[dict[str, Any]] = []
    for key in sorted(set(fork_buckets) & set(jack_buckets), key=lambda value: sorted(value)):
        candidates: list[tuple[float, float, dict[str, Any], dict[str, Any], str]] = []
        for fork_edge in fork_buckets[key]:
            for jack_edge in jack_buckets[key]:
                same_direction = (
                    fork_edge["start"] == jack_edge["start"]
                    and fork_edge["end"] == jack_edge["end"]
                )
                reverse_direction = (
                    fork_edge["start"] == jack_edge["end"]
                    and fork_edge["end"] == jack_edge["start"]
                )
                if not same_direction and not reverse_direction:
                    continue
                maximum, mean = path_deviation(fork_edge, jack_edge, reverse_direction)
                if maximum <= path_tolerance:
                    relation = "same-direction" if same_direction else "opposite-direction"
                    candidates.append((maximum, mean, fork_edge, jack_edge, relation))

        used_fork_edges: set[str] = set()
        used_jack_edges: set[str] = set()
        for maximum, mean, fork_edge, jack_edge, relation in sorted(candidates, key=lambda item: item[:2]):
            if fork_edge["id"] in used_fork_edges or jack_edge["id"] in used_jack_edges:
                continue
            used_fork_edges.add(fork_edge["id"])
            used_jack_edges.add(jack_edge["id"])
            shared_path_matches.append(
                {
                    "id": f"shared-path-{len(shared_path_matches)}",
                    "forkEdge": fork_edge["id"],
                    "jackEdge": jack_edge["id"],
                    "start": fork_edge["start"],
                    "end": fork_edge["end"],
                    "directionRelation": relation,
                    "maxDeviation": round(maximum, 4),
                    "meanDeviation": round(mean, 4),
                }
            )

    all_x = [node["x"] for node in nodes]
    all_y = [node["y"] for node in nodes]
    shared_node_count = sum(len(node["allowedRobotGroups"]) == 2 for node in nodes)
    return {
        "metadata": {
            "modelType": "multi-robot-layered-map",
            "robotGroups": {
                "fork": {"source": fork["metadata"]["source"]},
                "jack": {"source": jack["metadata"]["source"]},
            },
            "mergePolicy": {
                "sameIdTolerance": same_id_tolerance,
                "coordinateAliasTolerance": alias_tolerance,
                "sharedPathTolerance": path_tolerance,
                "note": "Routes remain robot-group specific; shared physical resources are matched separately.",
            },
            "bounds": {
                "minX": min(all_x),
                "maxX": max(all_x),
                "minY": min(all_y),
                "maxY": max(all_y),
            },
        },
        "stats": {
            "canonicalNodeCount": len(nodes),
            "sharedNodeCount": shared_node_count,
            "forkOnlyNodeCount": sum(node["allowedRobotGroups"] == ["fork"] for node in nodes),
            "jackOnlyNodeCount": sum(node["allowedRobotGroups"] == ["jack"] for node in nodes),
            "edgeCount": len(edges),
            "forkEdgeCount": len(edges_by_group["fork"]),
            "jackEdgeCount": len(edges_by_group["jack"]),
            "sharedPathMatchCount": len(shared_path_matches),
        },
        "nodeMatches": sorted(match_records, key=lambda item: item["canonicalNode"]),
        "nodes": sorted(nodes, key=lambda node: node["id"]),
        "edges": edges,
        "sharedPathMatches": shared_path_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified layered map for fork and jack robot groups")
    parser.add_argument("--fork", required=True, type=Path)
    parser.add_argument("--jack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--same-id-tolerance", type=float, default=0.15)
    parser.add_argument("--alias-tolerance", type=float, default=0.02)
    parser.add_argument("--path-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    model = build_unified_model(
        load_model(args.fork),
        load_model(args.jack),
        args.same_id_tolerance,
        args.alias_tolerance,
        args.path_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(model["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
