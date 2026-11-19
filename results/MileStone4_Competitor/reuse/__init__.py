"""Curated reusable competitor operators, dataset adapters, and reference scripts for MileStone_After."""

from pathlib import Path
from typing import Any, Dict

from scripts.phase2.common import RESULTS_DIR, load_json_file


def resolve_v5_frozen_dir(results_root: Path = RESULTS_DIR) -> Path:
    """Resolve V5_FROZEN directory across both cloud (`MileStone2`) and local renamed layout."""
    for rel in (
        Path("MileStone2") / "V5_FROZEN",
        Path("MileStone2_AblationExperiment_I") / "V5_FROZEN",
    ):
        cand = results_root / rel
        if (cand / "SHA256SUMS.txt").exists():
            return cand
    return results_root / "MileStone2" / "V5_FROZEN"


def resolve_milestone3_dir(results_root: Path = RESULTS_DIR) -> Path:
    """Resolve Milestone 3 directory across cloud (`MileStone3`) and local renamed layout."""
    for name in ("MileStone3", "MileStone3_AblationExperiment_II"):
        cand = results_root / name
        if cand.exists():
            return cand
    return results_root / "MileStone3"


def load_v5_params_compat(preset: str = "balanced_optimal", results_root: Path = RESULTS_DIR) -> Dict[str, Any]:
    """Load frozen V5 grid-searched parameter set from whichever MileStone2 directory exists."""
    v5_dir = resolve_v5_frozen_dir(results_root=results_root)
    grid_path = v5_dir / "results" / "grid_search_results.json"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing frozen grid search file: {grid_path}")
    data = load_json_file(grid_path)
    rec = data.get("recommended_parameter_sets", {})
    if preset not in rec:
        raise KeyError(f"Preset '{preset}' not found in {list(rec.keys())}")
    return dict(rec[preset])
