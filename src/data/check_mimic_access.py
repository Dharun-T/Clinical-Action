"""Check whether expected MIMIC-IV files are locally available.

This script checks filenames only. It does not download, open, or display
restricted MIMIC-IV data.
"""

from pathlib import Path


EXPECTED_FILES = {
    "hosp": (
        "patients.csv.gz",
        "admissions.csv.gz",
        "labevents.csv.gz",
        "d_labitems.csv.gz",
        "diagnoses_icd.csv.gz",
        "prescriptions.csv.gz",
        "emar.csv.gz",
        "poe.csv.gz",
        "poe_detail.csv.gz",
        "procedures_icd.csv.gz",
    ),
    "icu": (
        "icustays.csv.gz",
        "chartevents.csv.gz",
        "inputevents.csv.gz",
        "procedureevents.csv.gz",
    ),
}


def find_dataset_root(project_root: Path) -> Path:
    """Find the supplied demo first, then the documented raw-data location."""
    demo_root = project_root / "mimic-iv-clinical-database-demo-2.2"
    raw_root = project_root / "data" / "raw" / "mimic_iv"
    if demo_root.is_dir():
        return demo_root
    return raw_root


def check_directory(data_root: Path) -> bool:
    """Report the presence of expected files without reading their contents."""
    all_files_present = True
    for module_name, filenames in EXPECTED_FILES.items():
        module_directory = data_root / module_name
        print(f"[{module_name}]")
        for filename in filenames:
            file_path = module_directory / filename
            status = "FOUND" if file_path.is_file() else "MISSING"
            print(f"{status}: {file_path}")
            all_files_present = all_files_present and file_path.is_file()
    return all_files_present


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_root = find_dataset_root(project_root)
    print(f"Checking MIMIC-IV file access under: {data_root}")
    files_available = check_directory(data_root)
    if files_available:
        if data_root.name == "mimic-iv-clinical-database-demo-2.2":
            print("MIMIC-IV demo files are present. Treat all results as development-only.")
        else:
            print("All expected files are present. Do not process them until access is authorized.")
    else:
        print("MIMIC-IV files are not available locally; real-data processing is stopped.")


if __name__ == "__main__":
    main()
