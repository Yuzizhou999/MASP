from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_runtime_assets import build_workstations, write_json
from build_unified_map_model import build_unified_model, load_model
from validate_repository import validate_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate MASP runtime assets")
    parser.add_argument("--fork", type=Path, default=Path("generated/xiate-fork-map-model.json"))
    parser.add_argument("--jack", type=Path, default=Path("generated/xiate-jack-map-model.json"))
    parser.add_argument("--unified-output", type=Path, default=Path("generated/xiate-unified-map-model.json"))
    parser.add_argument("--conflicts", type=Path, default=Path("generated/xiate-conflict-resources.json"))
    parser.add_argument("--profiles", type=Path, default=Path("config/robot-profiles.json"))
    parser.add_argument("--scheduler", type=Path, default=Path("config/scheduler.json"))
    parser.add_argument("--vehicles", type=Path, default=Path("config/initial-vehicles.json"))
    parser.add_argument("--traffic-zones", type=Path, default=Path("config/traffic-zones.json"))
    parser.add_argument("--workstations-output", type=Path, default=Path("generated/xiate-workstations.json"))
    parser.add_argument("--same-id-tolerance", type=float, default=0.15)
    parser.add_argument("--alias-tolerance", type=float, default=0.02)
    parser.add_argument("--path-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    scheduler = load_model(args.scheduler)
    model = build_unified_model(
        load_model(args.fork),
        load_model(args.jack),
        args.same_id_tolerance,
        args.alias_tolerance,
        args.path_tolerance,
        scheduler,
    )
    write_json(args.unified_output, model)

    workstations = build_workstations(model, scheduler, args.unified_output.name)
    write_json(args.workstations_output, workstations)

    issues, stats = validate_repository(
        model,
        load_model(args.conflicts),
        load_model(args.profiles),
        scheduler,
        load_model(args.vehicles),
        workstations,
        load_model(args.traffic_zones),
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    result: dict[str, Any] = {
        "valid": not errors,
        "stats": stats,
        "warnings": [issue.message for issue in issues if issue.severity == "warning"],
        "errors": [issue.message for issue in errors],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
