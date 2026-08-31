"""FastAPI service for the CAS research application."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.joblib"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_actionability_demo_temporal.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "reports" / "permutation_importance.csv"
RESEARCH_IMPORTANCE_PATH = PROJECT_ROOT / "results" / "feature_importance.csv"
RAW_ROOT = PROJECT_ROOT / "mimic-iv-clinical-database-demo-2.2" / "hosp"
TEST_LABELS = {"glucose": "Glucose", "creatinine": "Creatinine", "potassium": "Potassium", "sodium": "Sodium", "hemoglobin": "Hemoglobin"}
TESTS = list(TEST_LABELS)


class PredictionRequest(BaseModel):
    patient_id: int | None = None
    test_name: str
    age: int = Field(ge=0, le=120)
    gender: str
    admission_type: str
    previous_value: float | None = None
    previous_previous_value: float | None = None
    value_change: float | None = None
    hours_since_previous_test: float = Field(ge=0)
    recent_test_count: int = Field(ge=0)
    prior_lab_count: int = Field(ge=0)
    prior_prescription_count: int = Field(ge=0)
    clinical_context: str | None = None
    medication_information: str | None = None


class ExplanationRequest(BaseModel):
    test_name: str


def _read_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH, parse_dates=["charttime"])


def _read_patients() -> pd.DataFrame:
    return pd.read_csv(RAW_ROOT / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])


def _read_admissions() -> pd.DataFrame:
    return pd.read_csv(RAW_ROOT / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "admission_type"])


def _read_labs() -> pd.DataFrame:
    items = pd.read_csv(RAW_ROOT / "d_labitems.csv.gz")
    selected = items[items["label"].isin(TEST_LABELS.values()) & items["fluid"].eq("Blood")]
    labs = pd.read_csv(
        RAW_ROOT / "labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
        parse_dates=["charttime"],
    )
    labs = labs[labs["itemid"].isin(selected["itemid"])].dropna(subset=["charttime", "valuenum"])
    labs = labs.merge(selected[["itemid", "label"]], on="itemid", how="left")
    labs["test_name"] = labs["label"].map({value: key for key, value in TEST_LABELS.items()})
    return labs.drop_duplicates(["subject_id", "hadm_id", "test_name", "charttime"])


def _level(cas: float) -> str:
    if cas < 0.30:
        return "Low"
    if cas < 0.70:
        return "Moderate"
    return "High"


def _feature_frame(request: PredictionRequest) -> pd.DataFrame:
    if request.test_name not in TESTS:
        raise HTTPException(status_code=422, detail=f"Unsupported laboratory test: {request.test_name}")
    previous = request.previous_value
    previous_previous = request.previous_previous_value
    change = request.value_change
    if change is None and previous is not None and previous_previous is not None:
        change = previous - previous_previous
    return pd.DataFrame([{
        "test_name": request.test_name,
        "gender": request.gender,
        "anchor_age": request.age,
        "admission_type": request.admission_type,
        "previous_lab_value": previous,
        "time_since_previous_hours": request.hours_since_previous_test,
        "change_from_previous": change,
        "prior_lab_count": request.prior_lab_count,
        "recent_test_count_7d": request.recent_test_count,
        "prior_prescription_count": request.prior_prescription_count,
    }])


def _importance() -> list[dict[str, Any]]:
    if not IMPORTANCE_PATH.is_file():
        return []
    frame = pd.read_csv(IMPORTANCE_PATH).head(5)
    return [
        {"feature": row.feature.replace("_", " ").title(), "importance": round(float(row.importance_mean), 6)}
        for row in frame.itertuples()
    ]


app = FastAPI(title="Clinical Actionability Score API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "available" if MODEL_PATH.is_file() else "missing"}


@app.get("/api/tests")
def tests() -> list[dict[str, str]]:
    return [{"value": key, "label": label} for key, label in TEST_LABELS.items()]


@app.get("/api/patients")
def patients() -> list[int]:
    return sorted(_read_patients()["subject_id"].astype(int).tolist())


@app.get("/api/patient/{patient_id}")
def patient(patient_id: int) -> dict[str, Any]:
    people = _read_patients()
    person = people[people["subject_id"].eq(patient_id)]
    if person.empty:
        raise HTTPException(status_code=404, detail="Patient was not found in the demo dataset.")
    admissions = _read_admissions()
    patient_admissions = admissions[admissions["subject_id"].eq(patient_id)]
    return {
        "patient_id": patient_id,
        "age": int(person.iloc[0]["anchor_age"]),
        "gender": str(person.iloc[0]["gender"]),
        "admission_types": sorted(patient_admissions["admission_type"].dropna().unique().tolist()),
        "admission_count": int(patient_admissions["hadm_id"].nunique()),
    }


@app.get("/api/labs/{patient_id}")
def labs(patient_id: int, test_name: str | None = None) -> list[dict[str, Any]]:
    frame = _read_labs()
    frame = frame[frame["subject_id"].eq(patient_id)]
    if test_name:
        if test_name not in TESTS:
            raise HTTPException(status_code=422, detail="Unsupported laboratory test.")
        frame = frame[frame["test_name"].eq(test_name)]
    return [
        {"test_name": row.test_name, "timestamp": row.charttime.isoformat(), "value": float(row.valuenum), "unit": row.valueuom or ""}
        for row in frame.sort_values("charttime").tail(20).itertuples()
    ]


@app.get("/api/analytics")
def analytics() -> dict[str, Any]:
    processed = _read_processed()
    prediction_path = PROJECT_ROOT / "outputs" / "reports" / "cas_test_predictions.csv"
    predictions = pd.read_csv(prediction_path) if prediction_path.is_file() else pd.DataFrame()
    test_counts = processed["test_name"].value_counts().reindex(TESTS, fill_value=0)
    response: dict[str, Any] = {
        "total_patients": int(processed["subject_id"].nunique()),
        "laboratory_tests": len(TESTS),
        "prediction_ready_reports": int(len(processed)),
        "predictions_generated": int(len(predictions)),
        "average_cas": round(float(predictions["cas"].mean()), 4) if not predictions.empty else None,
        "test_distribution": [{"test": TEST_LABELS[key], "count": int(test_counts[key])} for key in TESTS],
        "test_frequency": [{"test": TEST_LABELS[key], "count": int(test_counts[key])} for key in TESTS],
    }
    if not predictions.empty:
        predictions["level"] = predictions["cas"].map(_level)
        response["cas_distribution"] = [{"range": label, "count": int(((predictions["cas"] >= low) & (predictions["cas"] < high)).sum())} for label, low, high in [("0.00–0.29", 0, 0.30), ("0.30–0.69", 0.30, 0.70), ("0.70–1.00", 0.70, 1.01)]]
        response["level_distribution"] = [{"level": level, "count": int((predictions["level"] == level).sum())} for level in ["Low", "Moderate", "High"]]
        grouped = predictions.groupby("test_name")["cas"].mean().reindex(TESTS)
        response["average_cas_by_test"] = [{"test": TEST_LABELS[key], "average_cas": round(float(grouped[key]), 4)} for key in TESTS]
    else:
        response["cas_distribution"] = []
        response["level_distribution"] = []
        response["average_cas_by_test"] = []
    return response


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Saved model is unavailable.")
    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba(_feature_frame(request))[0, 1])
    factors = _importance()
    return {"test": TEST_LABELS[request.test_name], "probability": probability, "cas": probability, "actionability_level": _level(probability), "contributing_factors": factors}


@app.post("/api/explain")
def explain(request: ExplanationRequest) -> dict[str, Any]:
    if request.test_name not in TESTS:
        raise HTTPException(status_code=422, detail="Unsupported laboratory test.")
    return {"test": TEST_LABELS[request.test_name], "method": "Held-out permutation importance", "factors": _importance()}


@app.get("/api/research-results")
def research_results() -> dict[str, Any]:
    def read_csv(filename: str) -> list[dict[str, Any]]:
        path = PROJECT_ROOT / "results" / filename
        if not path.is_file():
            return []
        return pd.read_csv(path).where(pd.notna(pd.read_csv(path)), None).to_dict(orient="records")

    importance_path = RESEARCH_IMPORTANCE_PATH if RESEARCH_IMPORTANCE_PATH.is_file() else IMPORTANCE_PATH
    importance = []
    if importance_path.is_file():
        frame = pd.read_csv(importance_path).head(10)
        importance = [{"feature": row.feature.replace("_", " ").title(), "importance": float(row.importance_mean)} for row in frame.itertuples()]
    return {
        "model_comparison": read_csv("model_comparison.csv"),
        "ablation_results": read_csv("ablation_results.csv"),
        "laboratory_results": read_csv("laboratory_results.csv"),
        "feature_importance": importance,
        "development_only": True,
    }