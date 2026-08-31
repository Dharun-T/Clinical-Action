"""Verify the generated Step 3 dataset without printing patient records."""

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal.csv"
    dataset = pd.read_csv(dataset_path)

    required_columns = {
        "subject_id",
        "hadm_id",
        "test_name",
        "charttime",
        "previous_lab_value",
        "clinical_action_proxy",
    }
    missing_columns = required_columns.difference(dataset.columns)
    assert not missing_columns, f"Missing columns: {sorted(missing_columns)}"
    assert "valuenum" not in dataset.columns
    assert "value" not in dataset.columns
    assert set(dataset["clinical_action_proxy"].dropna().unique()).issubset({0, 1})
    assert not dataset.duplicated(
        subset=["subject_id", "hadm_id", "test_name", "charttime"]
    ).any()
    print(f"Verified rows: {len(dataset)}")
    print(f"Verified patients: {dataset['subject_id'].nunique()}")
    print("Verified binary target, duplicate handling, and no current-result columns.")


if __name__ == "__main__":
    main()
