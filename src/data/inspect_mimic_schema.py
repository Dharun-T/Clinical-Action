"""Audit the supplied MIMIC-IV demo schema without printing patient records."""

from pathlib import Path

import pandas as pd

from check_mimic_access import EXPECTED_FILES, find_dataset_root


LABORATORY_TESTS = {
    "glucose": r"glucose",
    "creatinine": r"creatinine",
    "potassium": r"potassium",
    "sodium": r"sodium",
    "hemoglobin": r"hemoglobin|haemoglobin",
}


def read_headers(file_path: Path) -> list[str]:
    """Read only a CSV header so no patient rows are displayed."""
    return list(pd.read_csv(file_path, compression="gzip", nrows=0).columns)


def audit_tables(dataset_root: Path) -> None:
    for module_name, filenames in EXPECTED_FILES.items():
        print(f"[{module_name} table schemas]")
        for filename in filenames:
            file_path = dataset_root / module_name / filename
            columns = read_headers(file_path)
            print(f"{filename}: {len(columns)} columns")
            print(f"  {', '.join(columns)}")


def audit_laboratory_dictionary(dataset_root: Path) -> None:
    dictionary_path = dataset_root / "hosp" / "d_labitems.csv.gz"
    labitems = pd.read_csv(dictionary_path, compression="gzip")
    required_columns = {"itemid", "label", "fluid", "category"}
    missing_columns = required_columns.difference(labitems.columns)
    if missing_columns:
        raise ValueError(f"d_labitems is missing required columns: {sorted(missing_columns)}")

    print("[required laboratory label matches]")
    for test_name, pattern in LABORATORY_TESTS.items():
        matches = labitems[labitems["label"].str.contains(pattern, case=False, na=False, regex=True)]
        print(f"{test_name}: {len(matches)} matching dictionary rows")
        for _, row in matches.iterrows():
            print(
                f"  itemid={row['itemid']}, label={row['label']!r}, "
                f"fluid={row['fluid']!r}, category={row['category']!r}"
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = find_dataset_root(project_root)
    print(f"Auditing schema under: {dataset_root}")
    audit_tables(dataset_root)
    audit_laboratory_dictionary(dataset_root)
    print("Schema audit completed. No patient rows were printed.")


if __name__ == "__main__":
    main()
