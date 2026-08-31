"""Verify Phase 6 research manifest and feature-importance artifacts."""

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results = project_root / "results"
    manifest = json.loads((results / "research_manifest.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(results / "feature_importance.csv")
    assert manifest["development_only"] is True
    assert manifest["feature_importance_method"].startswith("Permutation importance")
    assert not importance.empty
    assert {"feature", "importance_mean", "importance_std"}.issubset(importance.columns)
    assert (project_root / "figures" / "feature_importance.png").is_file()
    assert (results / "model_comparison.csv").is_file()
    assert (results / "ablation_results.csv").is_file()
    assert (results / "laboratory_results.csv").is_file()
    print("Verified Phase 6 research manifest, feature importance, and linked result artifacts.")


if __name__ == "__main__":
    main()