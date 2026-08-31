"""Run Phase 3 model comparison and evaluation on the Phase 2 dataset."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.models.train_models import make_models, make_preprocessor


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "clinical_action_proxy"
GROUP_COLUMN = "subject_id"
METADATA_COLUMNS = {"subject_id", "hadm_id", "charttime", TARGET_COLUMN}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal_v2.csv"
    results_directory = project_root / "results"
    figures_directory = project_root / "figures"
    models_directory = project_root / "models" / "phase3"
    results_directory.mkdir(exist_ok=True)
    figures_directory.mkdir(exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_path, parse_dates=["charttime"])
    feature_columns = [column for column in dataset.columns if column not in METADATA_COLUMNS]
    features = dataset[feature_columns]
    target = dataset[TARGET_COLUMN]
    groups = dataset[GROUP_COLUMN]
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(features, target, groups=groups))
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]
    train_patients = set(groups.iloc[train_indices])
    test_patients = set(groups.iloc[test_indices])
    assert train_patients.isdisjoint(test_patients)
    assert y_train.nunique() == 2 and y_test.nunique() == 2

    metrics = []
    probabilities = {}
    fitted_models = {}
    for model_name, estimator in make_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predicted_probability = pipeline.predict_proba(x_test)[:, 1]
        predicted_class = (predicted_probability >= 0.5).astype(int)
        probabilities[model_name] = predicted_probability
        fitted_models[model_name] = pipeline
        metrics.append({
            "model": model_name,
            "roc_auc": roc_auc_score(y_test, predicted_probability),
            "pr_auc": average_precision_score(y_test, predicted_probability),
            "precision": precision_score(y_test, predicted_class, zero_division=0),
            "recall": recall_score(y_test, predicted_class, zero_division=0),
            "f1": f1_score(y_test, predicted_class, zero_division=0),
            "brier_score": brier_score_loss(y_test, predicted_probability),
        })
        joblib.dump(pipeline, models_directory / f"{model_name}.joblib")

    comparison = pd.DataFrame(metrics).sort_values(
        ["pr_auc", "roc_auc", "brier_score"], ascending=[False, False, True]
    )
    comparison.to_csv(results_directory / "model_comparison.csv", index=False)
    best_model_name = str(comparison.iloc[0]["model"])
    joblib.dump(fitted_models[best_model_name], models_directory / "best_model.joblib")

    split = dataset[[GROUP_COLUMN]].copy()
    split["split"] = "test"
    split.loc[split.index.isin(train_indices), "split"] = "train"
    split.to_csv(results_directory / "phase3_split_assignments.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, predicted_probability in probabilities.items():
        RocCurveDisplay.from_predictions(y_test, predicted_probability, name=model_name, ax=axis)
    axis.set_title("ROC curves: Phase 3 patient-group test set")
    figure.tight_layout()
    figure.savefig(figures_directory / "roc_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, predicted_probability in probabilities.items():
        PrecisionRecallDisplay.from_predictions(y_test, predicted_probability, name=model_name, ax=axis)
    axis.set_title("Precision-recall curves: Phase 3 patient-group test set")
    figure.tight_layout()
    figure.savefig(figures_directory / "precision_recall_curve.png", dpi=180)
    plt.close(figure)

    best_probability = probabilities[best_model_name]
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, (best_probability >= 0.5).astype(int)),
        display_labels=["No action proxy", "Action proxy"],
    ).plot(ax=axis, colorbar=False, cmap="Blues")
    axis.set_title(f"Confusion matrix: {best_model_name}")
    figure.tight_layout()
    figure.savefig(figures_directory / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    metadata = {
        "dataset": str(dataset_path),
        "feature_set": "Phase 2 laboratory, temporal, patient, admission, and prescription-frequency features",
        "split": "GroupShuffleSplit with subject_id groups; 80/20 seeded split",
        "random_state": RANDOM_STATE,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "best_model": best_model_name,
        "development_only": True,
    }
    (results_directory / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Phase 3 rows train/test: {len(train_indices)}/{len(test_indices)}")
    print(f"Phase 3 patients train/test: {len(train_patients)}/{len(test_patients)}")
    print(f"Best model: {best_model_name}")
    print("Saved model comparison, ROC, precision-recall, and confusion-matrix artifacts.")


if __name__ == "__main__":
    main()
