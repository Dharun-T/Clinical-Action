"""Run the Phase 1 target and leakage validation."""

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    definition = json.loads((project_root / "results" / "target_definition.json").read_text(encoding="utf-8"))
    audit = json.loads((project_root / "results" / "target_audit.json").read_text(encoding="utf-8"))
    assert definition["target_column"] == "clinical_action_proxy"
    assert definition["observation_window_hours"] == 24
    assert definition["source_table"] == "hosp.poe"
    assert audit["target_recomputed_from_source"] is True
    assert audit["positive_rows"] > 0
    assert audit["negative_rows"] > 0
    assert audit["leakage_audit"]["duplicate_prediction_units"] == 0
    assert audit["leakage_audit"]["negative_time_since_previous_count"] == 0
    print("Verified target definition, source recomputation, class coverage, and temporal leakage checks.")


if __name__ == "__main__":
    main()
