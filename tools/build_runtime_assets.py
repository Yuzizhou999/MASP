from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# 地图上每个 AP 节点 → 生成一条工位记录
def build_workstations(
    model: dict[str, Any],
    scheduler_config: dict[str, Any],
    map_name: str,
) -> dict[str, Any]:
    defaults = scheduler_config["serviceDefaults"]
    workstations = []
    for node in model["nodes"]:
        if node["type"] != "AP":
            continue
        workstations.append(
            {
                "id": f"station:{node['id']}",
                "nodeId": node["id"],
                "capabilities": ["pickup", "dropoff"],
                "allowedRobotGroups": node["allowedRobotGroups"],
                "capacity": int(defaults["workstationCapacity"]),
                "pickupServiceMs": int(defaults["pickupServiceMs"]),
                "dropoffServiceMs": int(defaults["dropoffServiceMs"]),
                "blocksTransitDuringService": bool(defaults["blocksTransitDuringService"]),
                "propertiesByGroup": node.get("propertiesByGroup", {}),
            }
        )

    return {
        "schemaVersion": 1,
        "map": map_name,
        "workstations": sorted(workstations, key=lambda item: item["nodeId"]),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime workstation assets")
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--scheduler-config", required=True, type=Path)
    parser.add_argument("--workstations-output", required=True, type=Path)
    args = parser.parse_args()

    model = load_json(args.map)
    scheduler_config = load_json(args.scheduler_config)
    workstations = build_workstations(model, scheduler_config, args.map.name)
    write_json(args.workstations_output, workstations)
    print(
        json.dumps(
            {"workstationCount": len(workstations["workstations"])},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

