"""Run the reproducible CAS research workflow in phase order."""

import subprocess
import sys


PHASES = [
    "src.targets.validate_target",
    "src.features.build_temporal_dataset",
    "src.evaluation.run_phase3_evaluation",
    "src.evaluation.run_phase4_calibration",
    "src.evaluation.run_phase5_ablation",
    "src.evaluation.run_phase6_research_outputs",
]


def main() -> None:
    for module in PHASES:
        print(f"\nRunning {module}")
        subprocess.run([sys.executable, "-m", module], check=True)
    print("\nAll reproducible research phases completed.")


if __name__ == "__main__":
    main()
