from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from masp.scenario import validate_scenario_document


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_schemas_are_valid_and_example_matches() -> None:
    for filename in ("plan.schema.json", "simulation-scenario.schema.json"):
        schema = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    scenario = json.loads(
        (ROOT / "scenarios/phase1-single-vehicle.json").read_text(encoding="utf-8")
    )
    validate_scenario_document(scenario, ROOT / "schemas")
