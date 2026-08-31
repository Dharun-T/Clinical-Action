"""Verify Phase 2 temporal and medication features."""

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = pd.read_csv(project_root / "data" / "processed" / "clinical_actionability_demo_temporal_v2.csv")
    new_features = {
        "time_between_previous_values_hours",
        "rate_of_change_per_hour",
        "recent_lab_std_7d",
        "recent_prescription_count_7d",
    }
    assert new_features.issubset(dataset.columns)
    assert "valuenum" not in dataset.columns
    assert (dataset["time_between_previous_values_hours"].dropna() >= 0).all()
    assert (dataset["recent_prescription_count_7d"] >= 0).all()
    assert not dataset.duplicated(
        subset=["subject_id", "hadm_id", "test_name", "charttime"]
    ).any()
    assert dataset["clinical_action_proxy"].isin([0, 1]).all()
    print(f"Verified Phase 2 rows: {len(dataset)}")
    print("Verified historical rate, variability, medication-frequency features, and no current result column.")


if __name__ == "__main__":
    main()