"""Create Phase 6 research visualizations from actual Phase 3 artifacts."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "clinical_action_proxy"
GROUP_COLUMN = "subject_id"
METADATA_COLUMNS = {"subject_id", "hadm_id", "charttime", TARGET_COLUMN}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal_v2.csv"
    model_path = project_root / "models" / "phase3" / "best_model.joblib"
    results_directory = project_root / "results"
    figures_directory = project_root / "figures"
    dataset = pd.read_csv(dataset_path, parse_dates=["charttime"])
    feature_columns = [column for column in dataset.columns if column not in METADATA_COLUMNS]
    features = dataset[feature_columns]
    target = dataset[TARGET_COLUMN]
    groups = dataset[GROUP_COLUMN]
    _, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE).split(
            features, target, groups=groups
        )
    )
    model = joblib.load(model_path)
    importance = permutation_importance(
        model,
        features.iloc[test_indices],
        target.iloc[test_indices],
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(results_directory / "feature_importance.csv", index=False)

    top = importance_frame.head(10).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top["feature"].str.replace("_", " ").str.title(), top["importance_mean"], xerr=top["importance_std"], color="#176b87")
    axis.set_xlabel("Decrease in held-out PR-AUC after permutation")
    axis.set_title("Global feature importance: Phase 3 selected model")
    figure.tight_layout()
    figure.savefig(figures_directory / "feature_importance.png", dpi=180)
    plt.close(figure)

    manifest = {
        "feature_importance_method": "Permutation importance on the held-out patient-group test partition",
        "model": str(model_path),
        "dataset": str(dataset_path),
        "development_only": True,
        "artifacts": {
            "model_comparison": "results/model_comparison.csv",
            "ablation_results": "results/ablation_results.csv",
            "laboratory_results": "results/laboratory_results.csv",
            "calibration_metrics": "results/calibration_metrics.json",
            "feature_importance": "results/feature_importance.csv",
            "feature_importance_figure": "figures/feature_importance.png",
        },
    }
    (results_directory / "research_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved global feature importance for {len(feature_columns)} features.")
    print("Phase 6 research manifest and feature-importance figure generated.")


if __name__ == "__main__":
    main()
