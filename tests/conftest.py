from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def phase0_assets() -> dict[str, dict[str, Any]]:
    return {
        "model": read_json("generated/xiate-unified-map-model.json"),
        "conflicts": read_json("generated/xiate-conflict-resources.json"),
        "profiles": read_json("config/robot-profiles.json"),
        "scheduler": read_json("config/scheduler.json"),
        "vehicles": read_json("config/initial-vehicles.json"),
        "workstations": read_json("generated/xiate-workstations.json"),
        "traffic_zones": read_json("config/traffic-zones.json"),
    }
