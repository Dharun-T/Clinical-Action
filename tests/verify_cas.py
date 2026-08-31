"""Verify Step 5 CAS and explainability artifacts."""

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report_directory = project_root / "outputs" / "reports"
    predictions = pd.read_csv(report_directory / "cas_test_predictions.csv")
    importance = pd.read_csv(report_directory / "permutation_importance.csv")
    metadata = json.loads((report_directory / "cas_metadata.json").read_text(encoding="utf-8"))
    assert not predictions.empty
    assert predictions["cas"].between(0, 1).all()
    assert "valuenum" not in predictions.columns
    assert not importance.empty
    assert {"feature", "importance_mean", "importance_std"}.issubset(importance.columns)
    assert 0 <= metadata["cas_min"] <= metadata["cas_max"] <= 1
    print("Verified CAS bounds, held-out predictions, and permutation importance artifacts.")


if __name__ == "__main__":
    main()
