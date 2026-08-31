"""Verify Phase 5 ablation and laboratory-wise results."""

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results = project_root / "results"
    ablation = pd.read_csv(results / "ablation_results.csv")
    laboratory = pd.read_csv(results / "laboratory_results.csv")
    metadata = json.loads((results / "phase5_metadata.json").read_text(encoding="utf-8"))
    assert len(ablation) == 3
    assert set(ablation["feature_set"]) == {"Laboratory + Temporal Features", "Clinical Context Features", "Combined Features"}
    assert ((ablation[["roc_auc", "pr_auc", "f1"]] >= 0) & (ablation[["roc_auc", "pr_auc", "f1"]] <= 1)).all().all()
    assert set(laboratory["test_name"]) == {"glucose", "creatinine", "potassium", "sodium", "hemoglobin"}
    evaluated = laboratory[laboratory["status"] == "evaluated"]
    assert not evaluated.empty
    assert ((evaluated[["roc_auc", "pr_auc", "f1"]] >= 0) & (evaluated[["roc_auc", "pr_auc", "f1"]] <= 1)).all().all()
    assert metadata["train_patients"] > 0 and metadata["test_patients"] > 0
    print("Verified ablation results, five laboratory evaluations, patient split, and metric bounds.")


if __name__ == "__main__":
    main()