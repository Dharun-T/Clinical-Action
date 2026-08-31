"""Verify the Streamlit app can load the saved model and create valid input."""

from pathlib import Path

import joblib

from app.app import create_input_frame


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = joblib.load(project_root / "outputs" / "models" / "best_model.joblib")
    input_frame = create_input_frame(
        "glucose", "F", 50, "EMERGENCY", None, 24.0, None, 0, 0, 0
    )
    cas = float(model.predict_proba(input_frame)[0, 1])
    assert 0 <= cas <= 1
    print(f"Verified Streamlit input and saved model prediction; CAS={cas:.6f}.")


if __name__ == "__main__":
    main()
