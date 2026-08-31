"""Verify Step 4 model artifacts and evaluation outputs."""

import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_directory = project_root / "outputs" / "models"
    report_directory = project_root / "outputs" / "reports"
    metrics = pd.read_csv(report_directory / "model_metrics.csv")
    selection = json.loads((report_directory / "model_selection.json").read_text(encoding="utf-8"))
    assert set(metrics["model"]) == {"logistic_regression", "random_forest", "xgboost"}
    score_columns = ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score"]
    assert ((metrics[score_columns] >= 0) & (metrics[score_columns] <= 1)).all().all()
    assert selection["best_model"] in set(metrics["model"])
    assert (model_directory / "best_model.joblib").is_file()
    assert joblib.load(model_directory / "best_model.joblib") is not None
    split_assignments = pd.read_csv(report_directory / "split_assignments.csv")
    train_patients = set(split_assignments.loc[split_assignments["split"] == "train", "subject_id"])
    test_patients = set(split_assignments.loc[split_assignments["split"] == "test", "subject_id"])
    assert train_patients.isdisjoint(test_patients)
    print("Verified three model metrics, saved pipelines, best-model selection, and patient-level split.")


if __name__ == "__main__":
    main()
