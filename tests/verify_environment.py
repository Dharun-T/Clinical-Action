"""Small Step 1 smoke test for the project dependencies."""

from importlib.metadata import version

import matplotlib
import numpy
import pandas
import sklearn
import streamlit
import xgboost


REQUIRED_PACKAGES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "streamlit",
)


def main() -> None:
    print("Python environment verification")
    imported_modules = (pandas, numpy, sklearn, xgboost, matplotlib, streamlit)
    for package_name in REQUIRED_PACKAGES:
        print(f"{package_name}: {version(package_name)}")
    print(f"Successfully imported {len(imported_modules)} required packages.")


if __name__ == "__main__":
    main()
