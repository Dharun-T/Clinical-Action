"""Build a leakage-aware temporal dataset from the MIMIC-IV demo.

The target is a development proxy: a new laboratory order after a candidate
laboratory report and within the following 24 hours. It does not prove that
the laboratory report caused the order.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.check_mimic_access import find_dataset_root


LABORATORY_RULES = {
    "glucose": "Glucose",
    "creatinine": "Creatinine",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "hemoglobin": "Hemoglobin",
}
OBSERVATION_WINDOW_HOURS = 24
HISTORY_WINDOW_DAYS = 7


def load_laboratory_events(dataset_root: Path) -> pd.DataFrame:
    labitems = pd.read_csv(dataset_root / "hosp" / "d_labitems.csv.gz")
    labevents = pd.read_csv(
        dataset_root / "hosp" / "labevents.csv.gz",
        usecols=[
            "labevent_id",
            "subject_id",
            "hadm_id",
            "itemid",
            "charttime",
            "storetime",
            "valuenum",
            "valueuom",
        ],
        parse_dates=["charttime", "storetime"],
    )

    selected_frames = []
    for test_key, label in LABORATORY_RULES.items():
        selected_items = labitems[
            labitems["label"].eq(label)
            & labitems["fluid"].eq("Blood")
        ]
        selected_events = labevents[labevents["itemid"].isin(selected_items["itemid"])].copy()
        selected_events["test_name"] = test_key
        selected_frames.append(selected_events)

    laboratory_events = pd.concat(selected_frames, ignore_index=True)
    laboratory_events = laboratory_events.dropna(subset=["charttime", "valuenum"])
    laboratory_events = laboratory_events.drop_duplicates(
        subset=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    )
    laboratory_events = laboratory_events.sort_values(
        ["subject_id", "test_name", "charttime", "labevent_id"]
    ).drop_duplicates(
        subset=["subject_id", "hadm_id", "test_name", "charttime"],
        keep="last",
    ).reset_index(drop=True)
    return laboratory_events


def add_history_features(laboratory_events: pd.DataFrame) -> pd.DataFrame:
    events = laboratory_events.copy()
    grouped = events.groupby(["subject_id", "test_name"], sort=False)
    events["previous_lab_value"] = grouped["valuenum"].shift(1)
    events["previous_previous_lab_value"] = grouped["valuenum"].shift(2)
    events["previous_charttime"] = grouped["charttime"].shift(1)
    events["previous_previous_charttime"] = grouped["charttime"].shift(2)
    events["time_since_previous_hours"] = (
        grouped["charttime"].diff().dt.total_seconds() / 3600
    )
    events["time_between_previous_values_hours"] = (
        events["previous_charttime"] - events["previous_previous_charttime"]
    ).dt.total_seconds() / 3600
    events["change_from_previous"] = (
        events["previous_lab_value"] - events["previous_previous_lab_value"]
    )
    events["rate_of_change_per_hour"] = (
        events["change_from_previous"] / events["time_between_previous_values_hours"]
    )
    events["prior_lab_count"] = grouped.cumcount()

    recent_counts = np.zeros(len(events), dtype=int)
    recent_stds = np.full(len(events), np.nan)
    for _, group_indices in events.groupby(["subject_id", "test_name"], sort=False).groups.items():
        indices = np.asarray(group_indices)
        timestamps = events.loc[indices, "charttime"].astype("int64").to_numpy()
        window_ns = HISTORY_WINDOW_DAYS * 24 * 60 * 60 * 1_000_000_000
        starts = np.searchsorted(timestamps, timestamps - window_ns, side="left")
        recent_counts[indices] = np.arange(len(indices)) - starts
        values = events.loc[indices, "valuenum"].to_numpy()
        for position, (row_index, start) in enumerate(zip(indices, starts)):
            prior_values = values[start:position]
            if len(prior_values) >= 2:
                recent_stds[row_index] = float(np.std(prior_values, ddof=1))
    events["recent_test_count_7d"] = recent_counts
    events["recent_lab_std_7d"] = recent_stds
    return events.drop(columns=["previous_charttime", "previous_previous_charttime"])


def add_patient_context(events: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    patients = pd.read_csv(
        dataset_root / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = pd.read_csv(
        dataset_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admission_type"],
    )
    enriched = events.merge(patients, on="subject_id", how="left")
    return enriched.merge(admissions, on=["subject_id", "hadm_id"], how="left")


def add_prior_prescription_count(events: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    prescriptions = pd.read_csv(
        dataset_root / "hosp" / "prescriptions.csv.gz",
        usecols=["subject_id", "hadm_id", "starttime"],
        parse_dates=["starttime"],
    ).dropna(subset=["starttime"])
    prescription_times: dict[tuple[int, int], np.ndarray] = {}
    for key, group in prescriptions.groupby(["subject_id", "hadm_id"], sort=False):
        prescription_times[key] = np.sort(group["starttime"].astype("int64").to_numpy())

    prior_counts = []
    recent_counts = []
    recent_window_ns = HISTORY_WINDOW_DAYS * 24 * 60 * 60 * 1_000_000_000
    for event in events.itertuples(index=False):
        key = (event.subject_id, event.hadm_id)
        times = prescription_times.get(key)
        if times is None or pd.isna(event.hadm_id):
            prior_counts.append(0)
            recent_counts.append(0)
            continue
        event_time = event.charttime.value
        prior_counts.append(int(np.searchsorted(times, event_time, side="left")))
        recent_counts.append(
            int(
                np.searchsorted(times, event_time, side="left")
                - np.searchsorted(times, event_time - recent_window_ns, side="left")
            )
        )
    enriched = events.copy()
    enriched["prior_prescription_count"] = prior_counts
    enriched["recent_prescription_count_7d"] = recent_counts
    return enriched


def add_action_proxy(events: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    provider_orders = pd.read_csv(
        dataset_root / "hosp" / "poe.csv.gz",
        usecols=["subject_id", "hadm_id", "ordertime", "order_type", "transaction_type"],
        parse_dates=["ordertime"],
    ).dropna(subset=["ordertime"])
    provider_orders = provider_orders[
        provider_orders["order_type"].eq("Lab")
        & provider_orders["transaction_type"].eq("New")
    ].dropna(subset=["ordertime"])
    candidate_events = events.reset_index(names="event_id")
    joined = candidate_events[
        ["event_id", "subject_id", "hadm_id", "charttime"]
    ].dropna(subset=["hadm_id"]).merge(
        provider_orders,
        on=["subject_id", "hadm_id"],
        how="left",
    )
    window_end = joined["charttime"] + pd.to_timedelta(
        OBSERVATION_WINDOW_HOURS, unit="h"
    )
    qualifying_events = joined.loc[
        joined["ordertime"].gt(joined["charttime"])
        & joined["ordertime"].le(window_end),
        "event_id",
    ].unique()
    action_labels = candidate_events["event_id"].isin(qualifying_events).astype(int)
    enriched = events.copy()
    enriched["clinical_action_proxy"] = action_labels.to_numpy()
    return enriched


def select_output_columns(events: pd.DataFrame) -> pd.DataFrame:
    output_columns: Iterable[str] = (
        "subject_id",
        "hadm_id",
        "test_name",
        "charttime",
        "gender",
        "anchor_age",
        "admission_type",
        "previous_lab_value",
        "time_since_previous_hours",
        "change_from_previous",
        "time_between_previous_values_hours",
        "rate_of_change_per_hour",
        "recent_lab_std_7d",
        "prior_lab_count",
        "recent_test_count_7d",
        "prior_prescription_count",
        "recent_prescription_count_7d",
        "clinical_action_proxy",
    )
    return events.loc[:, list(output_columns)]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = find_dataset_root(project_root)
    output_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal_v2.csv"

    events = load_laboratory_events(dataset_root)
    events = add_history_features(events)
    events = add_patient_context(events, dataset_root)
    events = add_prior_prescription_count(events, dataset_root)
    events = add_action_proxy(events, dataset_root)
    output = select_output_columns(events)
    output.to_csv(output_path, index=False)

    print(f"Saved development dataset: {output_path}")
    print(f"Rows: {len(output)}")
    print(f"Patients: {output['subject_id'].nunique()}")
    print(f"New lab-order proxy positives: {int(output['clinical_action_proxy'].sum())}")
    print("The current laboratory result is excluded from features.")
    print("This target is a temporal new-lab-order proxy, not clinical causation.")


if __name__ == "__main__":
    main()
