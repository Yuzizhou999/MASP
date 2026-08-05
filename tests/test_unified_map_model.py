from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.build_unified_map_model import build_unified_model


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def build_model() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    source_models = {
        "fork": read_json("generated/xiate-fork-map-model.json"),
        "jack": read_json("generated/xiate-jack-map-model.json"),
    }
    scheduler = read_json("config/scheduler.json")
    model = build_unified_model(
        source_models["fork"],
        source_models["jack"],
        same_id_tolerance=0.15,
        alias_tolerance=0.02,
        path_tolerance=0.15,
        scheduler_config=scheduler,
    )
    return model, source_models, scheduler


def test_unified_map_preserves_properties_for_every_robot_group() -> None:
    model, source_models, _ = build_model()
    source_nodes = {
        group: {node["id"]: node for node in source["nodes"]}
        for group, source in source_models.items()
    }

    for node in model["nodes"]:
        assert set(node["propertiesByGroup"]) == set(node["allowedRobotGroups"])
        for group, local_id in node["aliases"].items():
            assert node["propertiesByGroup"][group] == source_nodes[group][local_id].get(
                "properties", {}
            )


def test_global_wait_switches_are_resolved_into_every_node() -> None:
    model, _, scheduler = build_model()
    wait = scheduler["traffic"]["wait"]
    flag_by_type = {
        "LM": "allowOnLM",
        "AP": "allowOnAP",
        "PP": "allowOnPP",
        "CP": "allowOnCP",
    }

    for node in model["nodes"]:
        expected = wait[flag_by_type[node["type"]]]
        for group in node["allowedRobotGroups"]:
            policy = node["waitPolicyByGroup"][group]
            assert policy["allowed"] is expected
            assert policy["maxWaitMs"] == (wait["maxPlannedWaitMs"] if expected else 0)
            assert policy["source"] == f"global:{flag_by_type[node['type']]}"
            assert node["allowWaitByGroup"][group] is expected


def test_generated_model_matches_a_fresh_build() -> None:
    rebuilt, _, _ = build_model()
    generated = read_json("generated/xiate-unified-map-model.json")

    assert generated == rebuilt
