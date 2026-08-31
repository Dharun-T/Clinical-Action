"""Verify Phase 4 calibrated CAS outputs."""

import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    predictions = pd.read_csv(project_root / "results" / "predictions.csv")
    metrics = json.loads((project_root / "results" / "calibration_metrics.json").read_text(encoding="utf-8"))
    assert not predictions.empty
    assert predictions["cas"].between(0, 1).all()
    assert predictions["raw_probability"].between(0, 1).all()
    assert predictions["calibrated_probability"].between(0, 1).all()
    assert metrics["calibration_patients"] > 0
    assert metrics["test_patients"] > 0
    assert metrics["cas_probability_source"] in {"raw", "calibrated"}
    assert joblib.load(project_root / "models" / "phase4" / "calibrated_model.joblib") is not None
    assert (project_root / "figures" / "calibration_curve.png").is_file()
    assert (project_root / "figures" / "cas_distribution.png").is_file()
    print("Verified calibrated model, CAS bounds, calibration metrics, and CAS figures.")


if __name__ == "__main__":
    main()