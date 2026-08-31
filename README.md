# Patient-Centric Clinical Actionability Framework

This repository is a research prototype for estimating the probability that a
future report for one of five laboratory tests will be followed by a predefined
measurable clinical action:

- Glucose
- Creatinine
- Potassium
- Sodium
- Hemoglobin

The model output is called the Clinical Actionability Score (CAS). CAS is the
model probability `P(Clinical Action = 1)`, so it is between 0 and 1.

## Important scope

This project is a decision-support research demonstration. It is not a medical
diagnostic system, treatment recommendation system, hospital deployment, or
replacement for clinical judgment. It will not recommend stopping tests or
changing treatment.

No dataset, patient outcome, clinical action, model metric, or research result
has been invented. Step 2 verifies the supplied MIMIC-IV demo schema before any
real-data pipeline is built. The demo is development-only and is not the full
credentialed MIMIC-IV research dataset.

## Step 1: environment setup

Create and activate a virtual environment from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tests\verify_environment.py
```

If PowerShell blocks activation, run this once in the current PowerShell
window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Planned steps

1. Environment and project foundation
2. Dataset selection, access, and data understanding
3. Data processing, feature engineering, and target creation
4. Supervised machine learning
5. CAS, explainability, and validation
6. Simple Streamlit application and final integration

The later steps must use only data and measurable downstream events that are
actually available and authorized. If the selected dataset cannot support the
original research question, that limitation will be documented rather than
hidden.

## Step 2 status: MIMIC-IV access and schema review

The workspace contains the openly available MIMIC-IV Clinical Database Demo
2.2, a 100-patient development subset. The schema review is recorded in
[data/mimic_iv_data_dictionary.md](data/mimic_iv_data_dictionary.md).
MIMIC-IV is credentialed-access data. Before obtaining it, use the official
PhysioNet page to create an account, complete the required training, accept the
data-use agreement, and follow your college supervisor's ethics process:

https://physionet.org/content/mimiciv/

After authorization, place the files in this layout:

```text
data/raw/mimic_iv/hosp/
data/raw/mimic_iv/icu/
```

Then run the filename-only access check:

```powershell
.\.venv\Scripts\python.exe src\data\check_mimic_access.py
```

The check must report all expected files as `FOUND` before real-data
inspection begins. It never opens, downloads, or displays restricted data.

## React and FastAPI application

The project also includes a separate web application. The React frontend is in
`frontend/` and communicates with the Python API in `backend/` over REST. The
existing Streamlit application remains available, but it is not used by this
web application.

Open two PowerShell terminals from the project root.

Terminal 1: install backend packages and start FastAPI:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2: install frontend packages and start Vite:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

Both terminals must remain running while using the website. If the frontend
shows `Failed to fetch`, the FastAPI terminal is not running or port 8000 is
blocked. Start the backend command above first, then refresh the frontend.

The API loads the existing `outputs/models/best_model.joblib` pipeline and
returns model probabilities from `POST /api/predict`. Analytics and laboratory
history are read from the local MIMIC-IV Demo 2.2 files and processed outputs.
The demo remains a development dataset; its statistics and predictions must
not be presented as clinical validation.

## Phase 1 research audit

Phase 1 formalizes the target as a measurable workflow proxy. A positive row
means a new `Lab` provider order (`transaction_type = New`) occurs strictly
after the candidate laboratory `charttime` and within 24 hours for the same
admission. A negative row has no such order in that window. This temporal
relationship does not prove causation.

Run the independent target and leakage audit with:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.targets.validate_target
.\.venv\Scripts\python.exe tests\verify_target.py
```

The audit writes `results/target_definition.json` and
`results/target_audit.json`. It recomputes the stored labels from `hosp.poe`,
checks duplicate prediction units, checks nonnegative historical intervals,
and confirms that the current laboratory result is not a feature.

## Phase 2 feature engineering

Phase 2 adds historical time gap, rate of change, 7-day laboratory variability,
and 7-day prescription frequency. These features are calculated only from
records earlier than the candidate laboratory `charttime`. The new dataset is
written to `data/processed/clinical_actionability_demo_temporal_v2.csv`; the
original model-compatible dataset is retained until Phase 3 retraining is
complete.

Run and verify Phase 2 with:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.features.build_temporal_dataset
.\.venv\Scripts\python.exe tests\verify_feature_engineering.py
```

## Phase 3 model comparison and evaluation

Phase 3 trains Logistic Regression, Random Forest, and XGBoost on the versioned
Phase 2 dataset using the same seeded patient-group split. It writes actual
metrics to `results/model_comparison.csv`, metadata to `results/metrics.json`,
models to `models/phase3/`, and publication-ready ROC, precision-recall, and
confusion-matrix figures to `figures/`.

Run and verify Phase 3 with:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.evaluation.run_phase3_evaluation
.\.venv\Scripts\python.exe tests\verify_phase3_evaluation.py
```

## Phase 4 calibration and CAS

Phase 4 fits a sigmoid calibration layer using a separate patient-group
calibration partition inside the training data. The untouched patient-group
test partition is used to compare raw and calibrated Brier scores. Calibrated
probability is saved as CAS in `results/predictions.csv`; the calibrated model
is saved under `models/phase4/`.

Run and verify Phase 4 with:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.evaluation.run_phase4_calibration
.\.venv\Scripts\python.exe tests\verify_phase4_calibration.py
```

## Phase 5 ablation and laboratory-wise evaluation

Phase 5 trains the same XGBoost configuration on three feature sets using the
same patient-group split: Laboratory + Temporal, Clinical Context, and
Combined. It also evaluates each of the five laboratory tests separately.
Actual outputs are written to `results/ablation_results.csv`,
`results/laboratory_results.csv`, and `results/phase5_metadata.json`.

Run and verify Phase 5 with:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.evaluation.run_phase5_ablation
.\.venv\Scripts\python.exe tests\verify_phase5_ablation.py
```
