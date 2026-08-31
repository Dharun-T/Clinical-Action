"""Run Phase 5 feature ablation and laboratory-wise evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.models.train_models import make_models, make_preprocessor


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "clinical_action_proxy"
GROUP_COLUMN = "subject_id"
METADATA_COLUMNS = {"subject_id", "hadm_id", "charttime", TARGET_COLUMN}
LABORATORY_FEATURES = [
    "test_name",
    "previous_lab_value",
    "time_since_previous_hours",
    "change_from_previous",
    "time_between_previous_values_hours",
    "rate_of_change_per_hour",
    "recent_lab_std_7d",
    "prior_lab_count",
    "recent_test_count_7d",
]
CLINICAL_FEATURES = [
    "gender",
    "anchor_age",
    "admission_type",
    "prior_prescription_count",
    "recent_prescription_count_7d",
]


def fit_and_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
            ("model", make_models()["xgboost"]),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline.predict_proba(x_test)[:, 1]


def score(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal_v2.csv"
    results_directory = project_root / "results"
    results_directory.mkdir(exist_ok=True)
    dataset = pd.read_csv(dataset_path, parse_dates=["charttime"])
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    features = dataset.drop(columns=list(METADATA_COLUMNS))
    target = dataset[TARGET_COLUMN]
    groups = dataset[GROUP_COLUMN]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(features, target, groups=groups))

    feature_sets = {
        "Laboratory + Temporal Features": LABORATORY_FEATURES,
        "Clinical Context Features": CLINICAL_FEATURES,
        "Combined Features": LABORATORY_FEATURES + CLINICAL_FEATURES,
    }
    ablation_rows = []
    for feature_set_name, selected_columns in feature_sets.items():
        x_train = features.iloc[train_indices][selected_columns]
        x_test = features.iloc[test_indices][selected_columns]
        numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [column for column in selected_columns if column not in numeric_columns]
        probability = fit_and_predict(
            x_train,
            target.iloc[train_indices],
            x_test,
            numeric_columns,
            categorical_columns,
        )
        scores = score(target.iloc[test_indices], probability)
        ablation_rows.append({"feature_set": feature_set_name, **scores})
    pd.DataFrame(ablation_rows).to_csv(results_directory / "ablation_results.csv", index=False)

    combined_features = features[LABORATORY_FEATURES + CLINICAL_FEATURES]
    laboratory_rows = []
    for test_name in sorted(dataset["test_name"].unique()):
        test_mask = dataset["test_name"].eq(test_name)
        test_train_indices = [index for index in train_indices if test_mask.iloc[index]]
        test_test_indices = [index for index in test_indices if test_mask.iloc[index]]
        y_train = target.iloc[test_train_indices]
        y_test = target.iloc[test_test_indices]
        row: dict[str, str | int | float | None] = {
            "test_name": test_name,
            "test_samples": len(test_test_indices),
            "positive_test_samples": int(y_test.sum()),
            "negative_test_samples": int((y_test == 0).sum()),
        }
        if len(y_train) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
            row.update({"roc_auc": None, "pr_auc": None, "f1": None, "status": "insufficient class coverage"})
        else:
            x_train = combined_features.iloc[test_train_indices]
            x_test = combined_features.iloc[test_test_indices]
            numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
            categorical_columns = [column for column in combined_features.columns if column not in numeric_columns]
            probability = fit_and_predict(x_train, y_train, x_test, numeric_columns, categorical_columns)
            row.update({**score(y_test, probability), "status": "evaluated"})
        laboratory_rows.append(row)
    pd.DataFrame(laboratory_rows).to_csv(results_directory / "laboratory_results.csv", index=False)

    metadata = {
        "dataset": str(dataset_path),
        "model": "XGBoost",
        "split": "same seeded GroupShuffleSplit by subject_id for ablation experiments",
        "random_state": RANDOM_STATE,
        "train_patients": int(groups.iloc[train_indices].nunique()),
        "test_patients": int(groups.iloc[test_indices].nunique()),
        "development_only": True,
    }
    (results_directory / "phase5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Ablation experiments completed: {len(ablation_rows)}")
    print(f"Laboratory-wise evaluations completed: {len(laboratory_rows)}")
    print("All Phase 5 metrics use the same held-out patient-group test split.")


if __name__ == "__main__":
    main()
