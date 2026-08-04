from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact visualization data for the unified fork/jack map")
    parser.add_argument("--unified", required=True, type=Path)
    parser.add_argument("--fork", required=True, type=Path)
    parser.add_argument("--jack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    unified = load(args.unified)
    fork = load(args.fork)
    jack = load(args.jack)

    shared_by_edge: dict[str, str] = {}
    for match in unified["sharedPathMatches"]:
        shared_by_edge[match["forkEdge"]] = match["id"]
        shared_by_edge[match["jackEdge"]] = match["id"]

    nodes = [
        {
            "id": node["id"],
            "type": node["type"],
            "x": node["x"],
            "y": node["y"],
            "groups": node["allowedRobotGroups"],
            "aliases": node["aliases"],
            "positions": node["positions"],
        }
        for node in unified["nodes"]
    ]

    edges = [
        {
            "id": edge["id"],
            "group": edge["robotGroup"],
            "start": edge["start"],
            "end": edge["end"],
            "localStart": edge["localStart"],
            "localEnd": edge["localEnd"],
            "p0": edge["p0"],
            "p1": edge["p1"],
            "p2": edge["p2"],
            "p3": edge["p3"],
            "length": edge["length"],
            "motionDirection": edge["motionDirection"],
            "pathClass": edge.get("pathClass", "UnknownPath"),
            "sharedMatch": shared_by_edge.get(edge["id"]),
        }
        for edge in unified["edges"]
    ]

    shared_edges = {edge["id"]: edge for edge in edges}
    shared_overlays = []
    for match in unified["sharedPathMatches"]:
        source = shared_edges[match["forkEdge"]]
        shared_overlays.append(
            {
                "id": match["id"],
                "p0": source["p0"],
                "p1": source["p1"],
                "p2": source["p2"],
                "p3": source["p3"],
                "forkEdge": match["forkEdge"],
                "jackEdge": match["jackEdge"],
                "directionRelation": match["directionRelation"],
                "maxDeviation": match["maxDeviation"],
            }
        )

    bounds = {
        "minX": min(fork["metadata"]["bounds"]["minX"], jack["metadata"]["bounds"]["minX"]),
        "maxX": max(fork["metadata"]["bounds"]["maxX"], jack["metadata"]["bounds"]["maxX"]),
        "minY": min(fork["metadata"]["bounds"]["minY"], jack["metadata"]["bounds"]["minY"]),
        "maxY": max(fork["metadata"]["bounds"]["maxY"], jack["metadata"]["bounds"]["maxY"]),
    }

    model = {
        "metadata": {"bounds": bounds},
        "stats": unified["stats"],
        "backgrounds": {
            "fork": fork["background"],
            "jack": jack["background"],
        },
        "nodes": nodes,
        "edges": edges,
        "sharedOverlays": shared_overlays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        json.dumps(
            {
                "nodes": len(nodes),
                "edges": len(edges),
                "sharedOverlays": len(shared_overlays),
                "bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
