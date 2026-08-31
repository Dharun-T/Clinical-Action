# CAS Project Handoff Prompt for Another AI

You are continuing an existing college research prototype. Read this entire document before changing anything. Do not rebuild the project from scratch.

## Project

Title: A Patient-Centric Framework for Predicting Clinical Actionability of Future Laboratory Reports.

Research question: Given patient-specific information available before a future laboratory report, can supervised machine learning estimate whether a measurable downstream clinical workflow action will occur after that report?

The system does not predict the future laboratory value. It predicts a probability of a defined measurable action.

CAS means Clinical Actionability Score:

```text
CAS = P(Clinical Action = 1)
0 <= CAS <= 1
```

This is a research and decision-support prototype. It is not a diagnostic system, treatment recommendation system, hospital deployment, or replacement for clinical judgment. Never recommend treatment, stopping tests, or changing medication.

## Workspace and environment

Project root:

```text
C:\Users\Vinothini\OneDrive\Desktop\projects\CAS
```

Operating system: Windows.

Verified environment:

```text
Python 3.14.3
Git 2.53.0
VS Code 1.133.0
Node 24.12.0
npm 11.6.2
```

Python virtual environment: `.venv\`.

## Dataset

Local dataset:

```text
mimic-iv-clinical-database-demo-2.2\
```

This is the openly available MIMIC-IV Clinical Database Demo 2.2, approximately 100 patients. It is not the full credentialed MIMIC-IV research dataset. All outputs from it must be labeled development-only.

Important dataset limitations:

- MIMIC-IV does not contain a ready-made label saying that a particular lab caused a clinical action.
- Dates are deidentified and shifted consistently within each patient.
- Calendar dates must not be compared across patients.
- Temporal proximity is not proof of causation.
- Never invent patient values, outcomes, target labels, metrics, or clinical conclusions.

Relevant source tables:

- `hosp.labevents`: lab observations with `subject_id`, `hadm_id`, `itemid`, `charttime`, `storetime`, `value`, `valuenum`, and `valueuom`.
- `hosp.d_labitems`: lab dictionary with `itemid`, `label`, `fluid`, and `category`.
- `hosp.patients`: `subject_id`, `gender`, and `anchor_age`.
- `hosp.admissions`: `subject_id`, `hadm_id`, `admission_type`, and admission timestamps.
- `hosp.poe`: provider orders with `ordertime`, `order_type`, and `transaction_type`.
- `hosp.prescriptions`: prescription starts used for prior medication context.
- `hosp.emar`: medication administration records available for future sensitivity analysis.
- `hosp.diagnoses_icd`: diagnoses, but no event timestamp in the core table.
- `hosp.procedures_icd`: procedures with `chartdate`, which can be too coarse for causal attribution.
- ICU tables include `icustays`, `chartevents`, `inputevents`, and `procedureevents`.

Official documentation used:

```text
https://physionet.org/content/mimiciv/
https://mimic.mit.edu/docs/iv/modules/hosp/
https://mimic.mit.edu/docs/iv/modules/icu/
```

## Required laboratory tests

The five tests are:

1. Glucose
2. Creatinine
3. Potassium
4. Sodium
5. Hemoglobin

Mapping is based on exact labels with `fluid = Blood`. Multiple item IDs and category variants exist. Do not guess IDs. The mapping uses `d_labitems` and joins to `labevents` by `itemid`.

## Target definition

Current target column:

```text
clinical_action_proxy
```

It is called a proxy because MIMIC-IV does not establish causal attribution.

Target = 1 when all of these are true:

```text
hosp.poe.order_type = "Lab"
hosp.poe.transaction_type = "New"
ordertime is strictly after candidate laboratory charttime
ordertime is no later than 24 hours after charttime
subject_id and hadm_id match the candidate laboratory event
```

Target = 0 when no qualifying new laboratory provider order occurs in the 24-hour window for the same admission.

Prediction time is the candidate lab `charttime`. The future laboratory value and all information recorded after that time must be excluded from features.

Target artifacts:

```text
results/target_definition.json
results/target_audit.json
src/targets/validate_target.py
tests/verify_target.py
```

Actual target audit from the demo:

```text
Prediction units: 14,904
Patients: 100
Positive rows: 10,795
Negative rows: 4,109
Positive rate: 0.7243022007514761
```

Stored labels were independently recomputed from `hosp.poe` and matched.

## Feature engineering

Original dataset:

```text
data/processed/clinical_actionability_demo_temporal.csv
```

Phase 2 dataset:

```text
data/processed/clinical_actionability_demo_temporal_v2.csv
```

Phase 2 model features:

```text
test_name
gender
anchor_age
admission_type
previous_lab_value
time_since_previous_hours
change_from_previous
time_between_previous_values_hours
rate_of_change_per_hour
recent_lab_std_7d
prior_lab_count
recent_test_count_7d
prior_prescription_count
recent_prescription_count_7d
```

Metadata, not model inputs:

```text
subject_id
hadm_id
charttime
clinical_action_proxy
```

The current candidate `valuenum` is absent from the modeling dataset. The historical change feature uses the two earlier values, not the candidate result. Counts, intervals, rates, and variability use records before the candidate `charttime`.

Feature builder:

```text
src/features/build_temporal_dataset.py
```

Feature dictionary:

```text
data/processed/feature_dictionary.md
```

## Existing research phases

### Phase 1

Target and leakage audit:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.targets.validate_target
.\.venv\Scripts\python.exe tests\verify_target.py
```

### Phase 2

Expanded features:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.features.build_temporal_dataset
.\.venv\Scripts\python.exe tests\verify_feature_engineering.py
```

### Phase 3

File: `src/evaluation/run_phase3_evaluation.py`.

Models:

- Logistic Regression
- Random Forest
- XGBoost

Preprocessing:

- Numeric median imputation
- Missing indicators
- Standardization
- Categorical most-frequent imputation
- One-hot encoding

Split: seeded `GroupShuffleSplit`, random state 42, test size 0.2, grouped by `subject_id`. Training and testing patients are disjoint.

Actual Phase 3 results:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9233 | 0.9649 | 0.8882 | 0.9950 | 0.9386 | 0.0755 |
| Random Forest | 0.9079 | 0.9563 | 0.8920 | 0.9804 | 0.9341 | 0.0815 |
| Logistic Regression | 0.8831 | 0.9309 | 0.9160 | 0.8390 | 0.8758 | 0.1186 |

These are demo test results only, not clinical validation.

Outputs:

```text
results/model_comparison.csv
results/metrics.json
results/phase3_split_assignments.csv
models/phase3/logistic_regression.joblib
models/phase3/random_forest.joblib
models/phase3/xgboost.joblib
models/phase3/best_model.joblib
figures/roc_curve.png
figures/precision_recall_curve.png
figures/confusion_matrix.png
```

### Phase 4

File: `src/evaluation/run_phase4_calibration.py`.

Patient groups:

```text
60 model-fit patients
20 calibration patients
20 final-test patients
```

Sigmoid calibration was evaluated:

```text
Raw Brier score: 0.078848
Calibrated Brier score: 0.079723
```

Calibration worsened held-out Brier, so raw XGBoost probability remains the selected CAS source. The calibrated model is still saved for comparison.

Outputs:

```text
results/predictions.csv
results/calibration_metrics.json
results/calibration_split_assignments.csv
models/phase4/calibrated_model.joblib
figures/calibration_curve.png
figures/cas_distribution.png
```

### Phase 5

File: `src/evaluation/run_phase5_ablation.py`.

Ablation feature sets:

1. Laboratory + Temporal Features
2. Clinical Context Features
3. Combined Features

Actual ablation results:

| Feature Set | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Laboratory + Temporal | 0.8618 | 0.9336 | 0.9092 |
| Clinical Context | 0.8820 | 0.9338 | 0.9385 |
| Combined | 0.9233 | 0.9649 | 0.9392 |

Actual laboratory-wise results:

| Test | Samples | Positive | Negative | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Creatinine | 545 | 389 | 156 | 0.9177 | 0.9555 | 0.9335 |
| Glucose | 533 | 418 | 115 | 0.9010 | 0.9646 | 0.9464 |
| Hemoglobin | 504 | 370 | 134 | 0.9322 | 0.9697 | 0.9363 |
| Potassium | 561 | 406 | 155 | 0.9193 | 0.9592 | 0.9372 |
| Sodium | 560 | 405 | 155 | 0.9213 | 0.9609 | 0.9382 |

Outputs:

```text
results/ablation_results.csv
results/laboratory_results.csv
results/phase5_metadata.json
```

### Phase 6

Files:

```text
src/evaluation/run_phase6_research_outputs.py
run_experiments.py
tests/verify_phase6_outputs.py
```

The complete runner executes target validation, feature building, Phase 3,
Phase 4, Phase 5, and Phase 6 in sequence:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe run_experiments.py
```

Phase 6 outputs:

```text
results/feature_importance.csv
results/research_manifest.json
figures/feature_importance.png
```

Feature importance is held-out permutation importance. It is not a causal or
patient-specific explanation.

## FastAPI backend

Entry point:

```text
backend/main.py
```

Dependencies:

```text
backend/requirements.txt
```

Run from the project root:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Endpoints:

```text
GET  /api/health
GET  /api/tests
GET  /api/patients
GET  /api/patient/{patient_id}
GET  /api/labs/{patient_id}?test_name=glucose
GET  /api/analytics
GET  /api/research-results
POST /api/predict
POST /api/explain
```

The API currently loads:

```text
outputs/models/best_model.joblib
```

The current API prediction path uses the original feature schema. The newer
Phase 3 and Phase 4 models use the Phase 2 v2 feature dataset and are stored
separately under `models/phase3` and `models/phase4`. Do not silently switch
models or schemas without a deliberate migration and tests.

The API predicts with `model.predict_proba(...)[0, 1]` and returns that value as
both `probability` and `cas`. The API uses actual saved permutation importance
for contributing-factor output.

## React frontend

Frontend directory:

```text
frontend/
```

Important files:

```text
frontend/package.json
frontend/vite.config.js
frontend/index.html
frontend/src/main.jsx
frontend/src/App.jsx
frontend/src/services/api.js
frontend/src/styles.css
```

Stack:

- React
- Vite
- Tailwind CSS through `@tailwindcss/vite`
- Recharts
- lucide-react

Views:

- Dashboard
- Prediction
- Patient Cases
- Analytics
- About

The frontend loads real analytics, automatically selects the first real demo
patient, fetches that patient's laboratory history, renders a Recharts trend,
calls FastAPI for prediction, shows CAS, and shows recent session predictions.
The Analytics view displays actual model comparison, ablation, laboratory-wise,
and feature-importance tables from `/api/research-results`.

Frontend API base URL:

```text
VITE_API_URL if configured, otherwise http://127.0.0.1:8000/api
```

## Run full web application

Use two PowerShell terminals from the project root.

Terminal 1:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Both terminals must stay running. If the frontend says `Failed to fetch`, the
FastAPI process is not available on port 8000. Start Terminal 1 and refresh.

Production build:

```powershell
cd frontend
npm run build
```

## Existing Streamlit application

The original Streamlit frontend remains at:

```text
app/app.py
```

It is separate from the React/FastAPI application and should not be removed
unless explicitly requested.

## Tests

Important verification scripts:

```text
tests/verify_environment.py
tests/verify_temporal_dataset.py
tests/verify_model_training.py
tests/verify_cas.py
tests/verify_app.py
tests/verify_target.py
tests/verify_feature_engineering.py
tests/verify_phase3_evaluation.py
tests/verify_phase4_calibration.py
tests/verify_phase5_ablation.py
tests/verify_phase6_outputs.py
```

Run the research validations:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe tests\verify_target.py
.\.venv\Scripts\python.exe tests\verify_feature_engineering.py
.\.venv\Scripts\python.exe tests\verify_phase3_evaluation.py
.\.venv\Scripts\python.exe tests\verify_phase4_calibration.py
.\.venv\Scripts\python.exe tests\verify_phase5_ablation.py
.\.venv\Scripts\python.exe tests\verify_phase6_outputs.py
```

## Critical risks and rules

1. The target is a new-lab-order workflow proxy, not a direct causal clinical-action label.
2. The local dataset is MIMIC-IV Demo 2.2, not full MIMIC-IV.
3. Never report a result that was not generated by the current code and data.
4. Never invent data, patients, outcomes, performance, CAS, explanations, or clinical conclusions.
5. Never use future laboratory results or post-prediction information as model features.
6. Do not claim clinical validation, treatment benefit, reduced testing, or replacement of doctors.
7. Permutation importance is not a causal or patient-specific explanation.
8. Sample cases are input demonstrations only; outputs must come from the actual model.
9. Phase 4 calibration did not improve Brier score, so raw probability is currently selected CAS.
10. The current API model schema differs from the newer Phase 2/3 research model schema.
11. Preserve the existing Streamlit app and original model artifacts unless a migration is explicitly requested.
12. Use the smallest change, run a focused test after editing, and do not undo unrelated user changes.

## Recommended next work

The next major research improvement is an intentional temporal train/test split. After that, decide whether to migrate the FastAPI prediction endpoint to the Phase 2/3 feature schema. That migration must include synchronized frontend inputs, API validation, model loading, tests, and updated documentation. Do not silently mix the original API model with the Phase 2/3 experiment results.
