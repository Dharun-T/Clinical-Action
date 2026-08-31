"""Calibrate the Phase 3 model and generate the Phase 4 CAS outputs."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.models.train_models import make_models, make_preprocessor


RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIBRATION_SIZE_WITHIN_TRAIN = 0.25
TARGET_COLUMN = "clinical_action_proxy"
GROUP_COLUMN = "subject_id"
METADATA_COLUMNS = {"subject_id", "hadm_id", "charttime", TARGET_COLUMN}


def split_by_patient(features: pd.DataFrame, target: pd.Series, groups: pd.Series):
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_indices, test_indices = next(outer.split(features, target, groups=groups))
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE,
    )
    fit_relative, calibration_relative = next(
        inner.split(
            features.iloc[train_indices],
            target.iloc[train_indices],
            groups=groups.iloc[train_indices],
        )
    )
    fit_indices = np.asarray(train_indices)[fit_relative]
    calibration_indices = np.asarray(train_indices)[calibration_relative]
    return fit_indices, calibration_indices, np.asarray(test_indices)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal_v2.csv"
    results_directory = project_root / "results"
    figures_directory = project_root / "figures"
    models_directory = project_root / "models" / "phase4"
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
    fit_indices, calibration_indices, test_indices = split_by_patient(features, target, groups)

    base_model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
            ("model", make_models()["xgboost"]),
        ]
    )
    base_model.fit(features.iloc[fit_indices], target.iloc[fit_indices])
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
    )
    calibrated_model.fit(features.iloc[calibration_indices], target.iloc[calibration_indices])
    raw_probability = base_model.predict_proba(features.iloc[test_indices])[:, 1]
    calibrated_probability = calibrated_model.predict_proba(features.iloc[test_indices])[:, 1]
    y_test = target.iloc[test_indices]
    raw_brier_score = brier_score_loss(y_test, raw_probability)
    calibrated_brier_score = brier_score_loss(y_test, calibrated_probability)
    use_calibrated = calibrated_brier_score < raw_brier_score
    cas_probability = calibrated_probability if use_calibrated else raw_probability

    prediction_output = dataset.iloc[test_indices][["subject_id", "hadm_id", "test_name", "charttime"]].copy()
    prediction_output["clinical_action_proxy"] = y_test.to_numpy()
    prediction_output["raw_probability"] = raw_probability
    prediction_output["calibrated_probability"] = calibrated_probability
    prediction_output["cas"] = cas_probability
    prediction_output.to_csv(results_directory / "predictions.csv", index=False)
    joblib.dump(calibrated_model, models_directory / "calibrated_model.joblib")

    calibration_metrics = {
        "raw_brier_score": float(raw_brier_score),
        "calibrated_brier_score": float(calibrated_brier_score),
        "raw_roc_auc": float(roc_auc_score(y_test, raw_probability)),
        "calibrated_roc_auc": float(roc_auc_score(y_test, calibrated_probability)),
        "fit_patients": int(groups.iloc[fit_indices].nunique()),
        "calibration_patients": int(groups.iloc[calibration_indices].nunique()),
        "test_patients": int(groups.iloc[test_indices].nunique()),
        "calibration_method": "sigmoid",
        "cas_definition": "CAS = selected P(Clinical Action = 1), preferring calibrated probability only when calibration lowers held-out Brier score",
        "cas_probability_source": "calibrated" if use_calibrated else "raw",
        "development_only": True,
    }
    (results_directory / "calibration_metrics.json").write_text(json.dumps(calibration_metrics, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(y_test, raw_probability, n_bins=10, strategy="quantile", name="Raw XGBoost", ax=axis)
    CalibrationDisplay.from_predictions(y_test, calibrated_probability, n_bins=10, strategy="quantile", name="Calibrated XGBoost", ax=axis)
    axis.set_title("Raw versus calibrated CAS probabilities")
    figure.tight_layout()
    figure.savefig(figures_directory / "calibration_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(calibrated_probability, bins=10, range=(0, 1), color="#176b87", edgecolor="white")
    axis.set(title="Distribution of calibrated CAS", xlabel="CAS", ylabel="Number of test reports")
    figure.tight_layout()
    figure.savefig(figures_directory / "cas_distribution.png", dpi=180)
    plt.close(figure)

    assignments = dataset[[GROUP_COLUMN]].copy()
    assignments["split"] = "test"
    assignments.loc[assignments.index.isin(fit_indices), "split"] = "fit"
    assignments.loc[assignments.index.isin(calibration_indices), "split"] = "calibration"
    assignments.to_csv(results_directory / "calibration_split_assignments.csv", index=False)
    print(f"Calibrated CAS rows: {len(prediction_output)}")
    print(f"Raw/calibrated Brier: {calibration_metrics['raw_brier_score']:.6f}/{calibration_metrics['calibrated_brier_score']:.6f}")
    print(f"CAS probability source: {'calibrated' if use_calibrated else 'raw'}")
    print(f"CAS range: {cas_probability.min():.6f} to {cas_probability.max():.6f}")
    print("Calibration and CAS outputs saved from the held-out patient test partition.")


if __name__ == "__main__":
    main()
