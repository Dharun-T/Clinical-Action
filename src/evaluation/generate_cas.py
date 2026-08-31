"""Generate CAS probabilities and held-out permutation importance."""

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
    input_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal.csv"
    model_path = project_root / "outputs" / "models" / "best_model.joblib"
    report_directory = project_root / "outputs" / "reports"
    figure_directory = project_root / "outputs" / "figures"
    dataset = pd.read_csv(input_path, parse_dates=["charttime"])
    model = joblib.load(model_path)

    feature_columns = [column for column in dataset.columns if column not in METADATA_COLUMNS]
    features = dataset[feature_columns]
    target = dataset[TARGET_COLUMN]
    groups = dataset[GROUP_COLUMN]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, test_indices = next(splitter.split(features, target, groups=groups))
    x_test = features.iloc[test_indices]
    y_test = target.iloc[test_indices]

    cas_values = model.predict_proba(x_test)[:, 1]
    predictions = dataset.iloc[test_indices][["subject_id", "hadm_id", "test_name", "charttime"]].copy()
    predictions["clinical_action_proxy"] = y_test.to_numpy()
    predictions["cas"] = cas_values
    predictions.to_csv(report_directory / "cas_test_predictions.csv", index=False)

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(report_directory / "permutation_importance.csv", index=False)

    top_importance = importance_frame.head(10).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top_importance["feature"], top_importance["importance_mean"], xerr=top_importance["importance_std"])
    axis.set_xlabel("Decrease in held-out PR-AUC after permutation")
    axis.set_title("Permutation importance for selected model")
    figure.tight_layout()
    figure.savefig(figure_directory / "permutation_importance.png", dpi=150)
    plt.close(figure)

    metadata = {
        "cas_definition": "P(Clinical Action = 1) from the selected model",
        "model_path": str(model_path),
        "calibration": "Raw model probability; no recalibration fitted on the demo.",
        "explainability": "Permutation importance on the held-out patient-level test partition.",
        "test_rows": len(test_indices),
        "test_patients": int(groups.iloc[test_indices].nunique()),
        "cas_min": float(cas_values.min()),
        "cas_max": float(cas_values.max()),
        "development_only": True,
    }
    (report_directory / "cas_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated CAS predictions: {len(predictions)}")
    print(f"CAS range: {cas_values.min():.6f} to {cas_values.max():.6f}")
    print("Permutation importance calculated on the held-out test partition.")
    print("CAS is probability of the defined proxy outcome, not treatment advice.")


if __name__ == "__main__":
    main()
