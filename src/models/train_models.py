"""Train and compare binary classifiers on the Step 3 temporal dataset."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "clinical_action_proxy"
GROUP_COLUMN = "subject_id"
METADATA_COLUMNS = {"subject_id", "hadm_id", "charttime", TARGET_COLUMN}


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def evaluate_predictions(model_name: str, y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "model": model_name,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def save_evaluation_plots(
    y_test: pd.Series,
    probabilities_by_model: dict[str, np.ndarray],
    output_directory: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, probabilities in probabilities_by_model.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_test, probabilities, n_bins=10, strategy="quantile"
        )
        axes[0].plot(mean_predicted, fraction_positive, marker="o", label=model_name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axes[0].set(title="Calibration curves", xlabel="Mean predicted probability", ylabel="Observed fraction positive")
    axes[0].legend()

    for model_name, probabilities in probabilities_by_model.items():
        matrix = confusion_matrix(y_test, (probabilities >= 0.5).astype(int))
        axes[1].plot([], [], label=f"{model_name}: {matrix.ravel().tolist()}")
    axes[1].axis("off")
    axes[1].set_title("Confusion matrices at threshold 0.5")
    axes[1].legend(loc="center", fontsize="small")

    figure.tight_layout()
    figure.savefig(output_directory / "calibration_and_confusion_matrices.png", dpi=150)
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "clinical_actionability_demo_temporal.csv"
    model_directory = project_root / "outputs" / "models"
    report_directory = project_root / "outputs" / "reports"
    figure_directory = project_root / "outputs" / "figures"
    model_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(input_path, parse_dates=["charttime"])
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
    train_subjects = set(groups.iloc[train_indices])
    test_subjects = set(groups.iloc[test_indices])
    if train_subjects.intersection(test_subjects):
        raise RuntimeError("Patient leakage detected between train and test groups.")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("Both train and test partitions must contain both target classes.")

    metrics = []
    probabilities_by_model = {}
    for model_name, estimator in make_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics.append(evaluate_predictions(model_name, y_test, probabilities))
        probabilities_by_model[model_name] = probabilities
        joblib.dump(pipeline, model_directory / f"{model_name}.joblib")

    metrics_frame = pd.DataFrame(metrics).sort_values(
        ["pr_auc", "roc_auc", "brier_score"], ascending=[False, False, True]
    )
    metrics_frame.to_csv(report_directory / "model_metrics.csv", index=False)
    best_model_name = str(metrics_frame.iloc[0]["model"])
    joblib.dump(
        joblib.load(model_directory / f"{best_model_name}.joblib"),
        model_directory / "best_model.joblib",
    )
    split_assignments = dataset[[GROUP_COLUMN]].copy()
    split_assignments["split"] = np.where(split_assignments.index.isin(train_indices), "train", "test")
    split_assignments.to_csv(report_directory / "split_assignments.csv", index=False)
    save_evaluation_plots(y_test, probabilities_by_model, figure_directory)

    selection = {
        "best_model": best_model_name,
        "selection_rule": "highest PR-AUC, then highest ROC-AUC, then lowest Brier score",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_patients": len(train_subjects),
        "test_patients": len(test_subjects),
        "target": TARGET_COLUMN,
        "development_only": True,
    }
    (report_directory / "model_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Trained models: {', '.join(make_models())}")
    print(f"Train rows/patients: {len(train_indices)}/{len(train_subjects)}")
    print(f"Test rows/patients: {len(test_indices)}/{len(test_subjects)}")
    print(f"Best model by predefined rule: {best_model_name}")
    print("Metrics are calculated from the development demo test partition only.")


if __name__ == "__main__":
    main()
