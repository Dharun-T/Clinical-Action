"""Verify Phase 3 evaluation artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results = project_root / "results"
    comparison = pd.read_csv(results / "model_comparison.csv")
    metadata = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    assert set(comparison["model"]) == {"logistic_regression", "random_forest", "xgboost"}
    metric_columns = ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score"]
    assert ((comparison[metric_columns] >= 0) & (comparison[metric_columns] <= 1)).all().all()
    assert metadata["train_patients"] > 0 and metadata["test_patients"] > 0
    assert metadata["best_model"] in set(comparison["model"])
    assert joblib.load(project_root / "models" / "phase3" / "best_model.joblib") is not None
    for filename in ["roc_curve.png", "precision_recall_curve.png", "confusion_matrix.png"]:
        assert (project_root / "figures" / filename).is_file()
    print("Verified Phase 3 model comparison, metrics, patient split, saved model, and evaluation figures.")


if __name__ == "__main__":
    main()