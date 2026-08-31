"""Streamlit dashboard for the Clinical Actionability Score."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.joblib"
IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "reports" / "permutation_importance.csv"
TEST_NAMES = ["glucose", "creatinine", "potassium", "sodium", "hemoglobin"]
TEST_LABELS = {
    "glucose": "Glucose",
    "creatinine": "Creatinine",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "hemoglobin": "Hemoglobin",
}
ADMISSION_TYPES = ["EMERGENCY", "ELECTIVE", "URGENT", "NEWBORN", "OTHER"]
GENDERS = ["F", "M", "Unknown"]

SAMPLE_CASES = {
    "No sample selected": {},
    "Sample Case 1 — Stable Glucose Monitoring": {
        "test_name": "glucose", "gender": "F", "anchor_age": 50,
        "admission_type": "EMERGENCY", "previous_previous_lab_value": 118.0,
        "previous_lab_value": 120.0, "time_since_previous_hours": 24.0,
        "prior_lab_count": 4, "recent_test_count_7d": 2, "prior_prescription_count": 3,
    },
    "Sample Case 2 — Changing Creatinine Trend": {
        "test_name": "creatinine", "gender": "M", "anchor_age": 67,
        "admission_type": "URGENT", "previous_previous_lab_value": 1.1,
        "previous_lab_value": 1.5, "time_since_previous_hours": 18.0,
        "prior_lab_count": 5, "recent_test_count_7d": 3, "prior_prescription_count": 6,
    },
    "Sample Case 3 — Recent Potassium Monitoring": {
        "test_name": "potassium", "gender": "F", "anchor_age": 72,
        "admission_type": "EMERGENCY", "previous_previous_lab_value": 4.2,
        "previous_lab_value": 5.0, "time_since_previous_hours": 8.0,
        "prior_lab_count": 7, "recent_test_count_7d": 5, "prior_prescription_count": 8,
    },
    "Sample Case 4 — Sodium Monitoring": {
        "test_name": "sodium", "gender": "M", "anchor_age": 59,
        "admission_type": "ELECTIVE", "previous_previous_lab_value": 139.0,
        "previous_lab_value": 138.0, "time_since_previous_hours": 36.0,
        "prior_lab_count": 3, "recent_test_count_7d": 1, "prior_prescription_count": 2,
    },
    "Sample Case 5 — Hemoglobin Monitoring": {
        "test_name": "hemoglobin", "gender": "F", "anchor_age": 44,
        "admission_type": "URGENT", "previous_previous_lab_value": 10.2,
        "previous_lab_value": 9.4, "time_since_previous_hours": 20.0,
        "prior_lab_count": 4, "recent_test_count_7d": 2, "prior_prescription_count": 4,
    },
}


def apply_sample_case() -> None:
    case = SAMPLE_CASES[st.session_state.get("sample_case", "No sample selected")]
    for field, value in case.items():
        st.session_state[field] = value


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The saved model could not be found. Run the training step first.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_importance() -> pd.DataFrame:
    if not IMPORTANCE_PATH.is_file():
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    return pd.read_csv(IMPORTANCE_PATH).head(5)


def create_input_frame(
    test_name: str,
    gender: str,
    anchor_age: int,
    admission_type: str,
    previous_lab_value: float | None,
    time_since_previous_hours: float,
    change_from_previous: float | None,
    prior_lab_count: int,
    recent_test_count_7d: int,
    prior_prescription_count: int,
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "test_name": test_name,
            "gender": gender,
            "anchor_age": anchor_age,
            "admission_type": admission_type,
            "previous_lab_value": previous_lab_value,
            "time_since_previous_hours": time_since_previous_hours,
            "change_from_previous": change_from_previous,
            "prior_lab_count": prior_lab_count,
            "recent_test_count_7d": recent_test_count_7d,
            "prior_prescription_count": prior_prescription_count,
        }
    ])


def actionability_level(cas: float) -> tuple[str, str]:
    if cas < 0.30:
        return "Lower", "level-low"
    if cas < 0.70:
        return "Moderate", "level-medium"
    return "Higher", "level-high"


def render_trend(test_name: str, previous_previous: float | None, previous: float | None) -> None:
    chart_data = pd.DataFrame(
        {"Position": ["Previous-previous", "Previous", "Future report"],
         "Value": [previous_previous, previous, None]}
    )
    figure, axis = plt.subplots(figsize=(8, 2.8))
    axis.plot(chart_data["Position"], chart_data["Value"], marker="o", color="#176b87", linewidth=2)
    axis.scatter(["Future report"], [0], facecolors="white", edgecolors="#d28b32", s=80, zorder=3)
    axis.set_title(f"{TEST_LABELS[test_name]} history and prediction point")
    axis.set_ylabel("Recorded value")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption("The future report has no value because CAS is predicted before that result is known.")


def initialize_state() -> None:
    defaults = {
        "test_name": "glucose", "gender": "F", "anchor_age": 50,
        "admission_type": "EMERGENCY", "previous_previous_lab_value": -1.0,
        "previous_lab_value": -1.0, "time_since_previous_hours": 24.0,
        "prior_lab_count": 0, "recent_test_count_7d": 0, "prior_prescription_count": 0,
        "recent_predictions": [],
    }
    for field, value in defaults.items():
        st.session_state.setdefault(field, value)


def main() -> None:
    st.set_page_config(page_title="Clinical Actionability Score", layout="wide")
    initialize_state()
    st.markdown("""
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem; }
        .hero { border-bottom: 1px solid #dbe5e8; padding-bottom: 1.4rem; margin-bottom: 1.5rem; }
        .hero h1 { color: #123b4a; font-size: 2.25rem; margin-bottom: .2rem; }
        .hero p { color: #55717a; font-size: 1.05rem; margin: 0; }
        .section-title { color: #123b4a; font-size: 1.25rem; font-weight: 700; margin: 1.2rem 0 .65rem; }
        .cas-card { background: #f2f7f8; border: 1px solid #c9dfe3; border-radius: 8px; padding: 1.4rem; text-align: center; }
        .cas-card .value { color: #0e6179; font-size: 3.8rem; font-weight: 750; line-height: 1; }
        .cas-card .label { color: #55717a; font-size: .9rem; margin-top: .5rem; }
        .level-low { color: #3c7862; font-weight: 700; }
        .level-medium { color: #a36d20; font-weight: 700; }
        .level-high { color: #a44f4f; font-weight: 700; }
        .footer { border-top: 1px solid #dbe5e8; color: #71858a; font-size: .78rem; margin-top: 2rem; padding-top: .8rem; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='hero'><h1>Clinical Actionability Score</h1><p>Patient-Centric Prediction of Future Laboratory Test Actionability</p><p>Estimate the likelihood that the next laboratory report will lead to a measurable clinical action.</p></div>", unsafe_allow_html=True)
    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.markdown("### CAS Dashboard")
        st.radio("Navigate", ["Dashboard", "Prediction", "Example Cases", "About Project"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Research prototype — decision support only.")

    st.markdown("<div class='section-title'>Patient and test input</div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("**Patient information**")
        patient_one, patient_two = st.columns(2)
        with patient_one:
            st.number_input("Age", min_value=0, max_value=120, step=1, key="anchor_age")
            st.selectbox("Gender", GENDERS, key="gender")
        with patient_two:
            st.selectbox("Admission type", ADMISSION_TYPES, key="admission_type")
            st.text_input("Patient context", value="Previous laboratory history", disabled=True)
    with right:
        st.markdown("**Future laboratory test**")
        st.selectbox("Select test", TEST_NAMES, format_func=lambda value: TEST_LABELS[value], key="test_name")
        st.caption("The selected test is the future report whose actionability will be estimated.")

    st.markdown("<div class='section-title'>Previous laboratory history</div>", unsafe_allow_html=True)
    history_one, history_two, history_three = st.columns(3)
    with history_one:
        st.number_input("Previous-previous value", step=0.1, key="previous_previous_lab_value", help="Use -1 when unavailable.")
        st.number_input("Previous value", step=0.1, key="previous_lab_value", help="Use -1 when unavailable.")
    with history_two:
        st.number_input("Hours since previous test", min_value=0.0, step=1.0, key="time_since_previous_hours")
        st.number_input("Previous test count", min_value=0, step=1, key="prior_lab_count")
    with history_three:
        st.number_input("Recent tests in 7 days", min_value=0, step=1, key="recent_test_count_7d")
        st.number_input("Prescription starts before report", min_value=0, step=1, key="prior_prescription_count")

    previous_previous = None if st.session_state.previous_previous_lab_value < 0 else st.session_state.previous_previous_lab_value
    previous = None if st.session_state.previous_lab_value < 0 else st.session_state.previous_lab_value
    change = None if previous_previous is None or previous is None else previous - previous_previous
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Previous value", "Unavailable" if previous is None else f"{previous:.2f}")
    metric_two.metric("Change from previous", "Unavailable" if change is None else f"{change:+.2f}")
    metric_three.metric("History available", f"{st.session_state.prior_lab_count} earlier tests")
    render_trend(st.session_state.test_name, previous_previous, previous)

    st.markdown("<div class='section-title'>Example patient cases</div>", unsafe_allow_html=True)
    st.selectbox("Load a sample input", list(SAMPLE_CASES), key="sample_case", on_change=apply_sample_case)
    st.caption("Sample Case values are demonstration inputs. Their outputs are generated by the saved model after prediction.")

    st.markdown("<div class='section-title'>CAS prediction</div>", unsafe_allow_html=True)
    if st.button("Predict Clinical Actionability", type="primary", use_container_width=True):
        input_frame = create_input_frame(
            st.session_state.test_name,
            st.session_state.gender,
            int(st.session_state.anchor_age),
            st.session_state.admission_type,
            previous,
            float(st.session_state.time_since_previous_hours),
            change,
            int(st.session_state.prior_lab_count),
            int(st.session_state.recent_test_count_7d),
            int(st.session_state.prior_prescription_count),
        )
        try:
            cas = float(model.predict_proba(input_frame)[0, 1])
            level, level_class = actionability_level(cas)
            st.session_state.last_prediction = {"cas": cas, "level": level, "test": TEST_LABELS[st.session_state.test_name]}
            st.session_state.recent_predictions.insert(0, {
                "Test": TEST_LABELS[st.session_state.test_name],
                "CAS": f"{cas:.3f}", "Actionability Level": level,
                "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
            })
            st.session_state.recent_predictions = st.session_state.recent_predictions[:8]
        except (ValueError, TypeError, KeyError):
            st.error("The entered values could not be processed. Please review the input fields.")

    if "last_prediction" in st.session_state:
        prediction = st.session_state.last_prediction
        result, factors = st.columns([1, 1.35], gap="large")
        with result:
            level, level_class = actionability_level(prediction["cas"])
            st.markdown(f"<div class='cas-card'><div class='label'>Clinical Actionability Score</div><div class='value'>{prediction['cas']:.2f}</div><div class='{level_class}'>Predicted Actionability Level: {level}</div><div class='label'>{prediction['cas']:.1%} predicted probability</div></div>", unsafe_allow_html=True)
        with factors:
            st.markdown("**Top contributing factors**")
            importance = load_importance()
            if importance.empty:
                st.info("Explainability results are not available yet.")
            else:
                display_importance = importance[["feature", "importance_mean"]].copy()
                display_importance["feature"] = display_importance["feature"].str.replace("_", " ").str.title()
                display_importance = display_importance.rename(columns={"feature": "Factor", "importance_mean": "Permutation importance"})
                st.dataframe(display_importance, hide_index=True, width="stretch")

    st.markdown("<div class='section-title'>Recent predictions</div>", unsafe_allow_html=True)
    if st.session_state.recent_predictions:
        st.dataframe(pd.DataFrame(st.session_state.recent_predictions), hide_index=True, width="stretch")
    else:
        st.caption("Predictions generated during this session will appear here.")

    st.markdown("<div class='section-title'>About the project</div>", unsafe_allow_html=True)
    st.markdown("**Project:** A Patient-Centric Framework for Predicting Clinical Actionability of Future Laboratory Reports  \n**Input:** Patient-specific context and previous laboratory trends  \n**Model:** Supervised machine learning  \n**Output:** CAS, a 0–1 probability of measurable clinical action  \n**Tests:** Glucose, Creatinine, Potassium, Sodium, Hemoglobin")
    st.markdown("<div class='footer'>Research prototype — decision support only.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
