"""Create an auditable target definition and validate temporal leakage controls."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.check_mimic_access import find_dataset_root


TARGET_COLUMN = "clinical_action_proxy"
WINDOW_HOURS = 24
FEATURE_COLUMNS = {
    "test_name",
    "gender",
    "anchor_age",
    "admission_type",
    "previous_lab_value",
    "time_since_previous_hours",
    "change_from_previous",
    "prior_lab_count",
    "recent_test_count_7d",
    "prior_prescription_count",
}


def recompute_target(processed: pd.DataFrame, dataset_root: Path) -> pd.Series:
    """Recompute target labels directly from the provider-order source table."""
    orders = pd.read_csv(
        dataset_root / "hosp" / "poe.csv.gz",
        usecols=["subject_id", "hadm_id", "ordertime", "order_type", "transaction_type"],
        parse_dates=["ordertime"],
    ).dropna(subset=["ordertime", "hadm_id"])
    orders = orders[
        orders["order_type"].eq("Lab") & orders["transaction_type"].eq("New")
    ]
    candidates = processed.reset_index(names="event_id")
    joined = candidates[["event_id", "subject_id", "hadm_id", "charttime"]].dropna(
        subset=["hadm_id"]
    ).merge(orders, on=["subject_id", "hadm_id"], how="left")
    end_time = joined["charttime"] + pd.to_timedelta(WINDOW_HOURS, unit="h")
    positive_ids = joined.loc[
        joined["ordertime"].gt(joined["charttime"])
        & joined["ordertime"].le(end_time),
        "event_id",
    ].unique()
    return candidates["event_id"].isin(positive_ids).astype(int)


def audit_temporal_features(processed: pd.DataFrame) -> dict[str, int | bool]:
    """Check that generated feature fields do not contain the future lab value."""
    forbidden_columns = {"value", "valuenum", "valueuom", "clinical_action"}
    forbidden_feature_columns = forbidden_columns.intersection(FEATURE_COLUMNS)
    assert not forbidden_feature_columns, sorted(forbidden_feature_columns)
    assert "charttime" in processed.columns
    assert TARGET_COLUMN in processed.columns
    assert set(processed[TARGET_COLUMN].dropna().unique()).issubset({0, 1})
    assert not processed.duplicated(
        subset=["subject_id", "hadm_id", "test_name", "charttime"]
    ).any()
    assert (processed["time_since_previous_hours"].dropna() >= 0).all()
    assert (processed["prior_lab_count"] >= 0).all()
    assert (processed["recent_test_count_7d"] >= 0).all()
    assert (processed["prior_prescription_count"] >= 0).all()
    return {
        "forbidden_current_result_columns_present": bool(
            forbidden_columns.intersection(processed.columns)
        ),
        "duplicate_prediction_units": int(
            processed.duplicated(
                subset=["subject_id", "hadm_id", "test_name", "charttime"]
            ).sum()
        ),
        "negative_time_since_previous_count": int(
            (processed["time_since_previous_hours"].dropna() < 0).sum()
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = find_dataset_root(project_root)
    processed_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal.csv"
    results_directory = project_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    processed = pd.read_csv(processed_path, parse_dates=["charttime"])

    recomputed = recompute_target(processed, dataset_root)
    stored = processed[TARGET_COLUMN].astype(int).reset_index(drop=True)
    assert recomputed.equals(stored), "Stored target does not match source-table recomputation."
    leakage_audit = audit_temporal_features(processed)
    assert not leakage_audit["forbidden_current_result_columns_present"]

    definition = {
        "target_column": TARGET_COLUMN,
        "positive_class": "A new provider order with order_type='Lab' and transaction_type='New' occurs strictly after the candidate lab charttime and within 24 hours for the same hadm_id.",
        "negative_class": "No qualifying new laboratory provider order occurs in that 24-hour window for the same admission.",
        "prediction_time": "candidate laboratory charttime",
        "observation_window_hours": WINDOW_HOURS,
        "source_table": "hosp.poe",
        "join_keys": ["subject_id", "hadm_id"],
        "timestamp_columns": {"candidate_lab": "charttime", "action_event": "ordertime"},
        "causal_limitation": "Temporal proximity does not prove that the laboratory report caused the order. This is a measurable workflow proxy for the MIMIC-IV Demo 2.2 development dataset.",
        "development_only": True,
    }
    audit = {
        "rows": int(len(processed)),
        "patients": int(processed["subject_id"].nunique()),
        "positive_rows": int(stored.sum()),
        "negative_rows": int((stored == 0).sum()),
        "positive_rate": float(stored.mean()),
        "target_recomputed_from_source": True,
        "leakage_audit": leakage_audit,
        "development_only": True,
    }
    (results_directory / "target_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    (results_directory / "target_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Validated target rows: {len(processed)}")
    print(f"Positive/negative rows: {stored.sum()}/{(stored == 0).sum()}")
    print("Target recomputation matches the source POE table.")
    print("Temporal leakage audit passed; current laboratory result columns are absent.")


if __name__ == "__main__":
    main()
