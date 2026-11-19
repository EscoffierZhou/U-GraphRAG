"""
merge_phase3_kernels.py
=======================
Merge Kernel 1 (`core3`: qwen3b, qwen7b, llama3_8b) and Kernel 2 (`extra2`: mistral7b, gemma2)
Kaggle / AutoDL Milestone 3 outputs (zipped archives or unpacked directories) into one unified
`results/MileStone3/` directory and rebuild WP11 figures (Figs 2-7), LaTeX/MD tables, and report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3 import wp11_build_paper_package
from scripts.phase3.phase3_common import (
    MILE3_DIR,
    assert_safe_output_path,
    atomic_write_json,
    load_json_file,
    merge_model_entries_dict,
    setup_phase2_logger,
    verify_v5_frozen_checksums,
)

logger = setup_phase2_logger("phase3.merge_kernels")


def _locate_m3_root(src: Path, work_tmp: Path, idx: int) -> Path:
    """Resolve directory containing WP6_CrossFamily/ etc. from zip or directory."""
    if src.is_file() and src.suffix.lower() == ".zip":
        dest = work_tmp / f"unpacked_{idx}"
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest)
        for cand in [dest] + [p for p in dest.rglob("MileStone3") if p.is_dir()]:
            if (cand / "WP6_CrossFamily").exists() or (cand / "WP10_Efficiency").exists():
                return cand
        return dest
    if (src / "WP6_CrossFamily").exists() or (src / "WP10_Efficiency").exists():
        return src
    for cand in src.rglob("MileStone3"):
        if cand.is_dir() and ((cand / "WP6_CrossFamily").exists() or (cand / "WP10_Efficiency").exists()):
            return cand
    return src


def merge_kernel_outputs(sources: List[Path], out_root: Path) -> Dict[str, Any]:
    """Merge WP6/WP7/WP8/WP9/WP10 outputs cleanly across dual Kaggle runs and rebuild WP11."""
    v5_info = verify_v5_frozen_checksums()
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in (
        "WP6_CrossFamily/predictions",
        "WP7_CrossDataset",
        "WP8_MethodValidity",
        "WP9_Robustness",
        "WP10_Efficiency",
        "WP11_PaperPackage/figures",
    ):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    merged_wp6: Dict[str, Any] = {}
    wp6_template: Dict[str, Any] = {}
    merged_wp10: Dict[str, Any] = {}
    wp10_template: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        roots = [_locate_m3_root(s, Path(tmpdir), i) for i, s in enumerate(sources)]

        for rt in roots:
            # Copy WP7 / WP8 / WP9 canonical single-run files when present
            for rel in (
                Path("WP8_MethodValidity/wp8_method_validity_results.json"),
                Path("WP7_CrossDataset/wp7_cross_dataset_transfer.json"),
                Path("WP9_Robustness/wp9_stress_curves.json"),
            ):
                src_f = rt / rel
                if src_f.is_file():
                    dst_f = assert_safe_output_path(out_root / rel)
                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_f, dst_f)

            # Copy per-sample WP6 prediction JSONL traces
            pred_src = rt / "WP6_CrossFamily" / "predictions"
            if pred_src.is_dir():
                for jf in pred_src.glob("*.jsonl"):
                    shutil.copy2(jf, out_root / "WP6_CrossFamily" / "predictions" / jf.name)

            # Deep-merge WP6 matrix by model key
            wp6_f = rt / "WP6_CrossFamily" / "wp6_cross_family_matrix.json"
            if wp6_f.is_file():
                d6 = load_json_file(wp6_f)
                wp6_template = {**d6}
                merged_wp6 = merge_model_entries_dict(merged_wp6, d6.get("matrix", {}))

            # Deep-merge WP10 efficiency models by model key
            wp10_f = rt / "WP10_Efficiency" / "wp10_efficiency_pareto.json"
            if wp10_f.is_file():
                d10 = load_json_file(wp10_f)
                wp10_template = {**d10}
                merged_wp10 = merge_model_entries_dict(merged_wp10, d10.get("models", {}))

    if merged_wp6:
        meta6 = dict(wp6_template.get("metadata", {}))
        meta6["models"] = list(merged_wp6.keys())
        meta6["model_group"] = "all5_merged" if len(merged_wp6) >= 5 else meta6.get("model_group")
        wp6_template["metadata"] = meta6
        wp6_template["v5_integrity"] = v5_info
        wp6_template["matrix"] = merged_wp6
        atomic_write_json(out_root / "WP6_CrossFamily" / "wp6_cross_family_matrix.json", wp6_template)

    if merged_wp10:
        meta10 = dict(wp10_template.get("metadata", {}))
        meta10["models"] = list(merged_wp10.keys())
        meta10["model_group"] = "all5_merged" if len(merged_wp10) >= 5 else meta10.get("model_group")
        wp10_template["metadata"] = meta10
        wp10_template["v5_integrity"] = v5_info
        wp10_template["models"] = merged_wp10
        atomic_write_json(out_root / "WP10_Efficiency" / "wp10_efficiency_pareto.json", wp10_template)

    # Rebuild WP11 paper package on merged artifacts
    prev_argv = sys.argv
    try:
        sys.argv = [
            "wp11_build_paper_package.py",
            "--wp6-dir", str(out_root / "WP6_CrossFamily"),
            "--wp7-dir", str(out_root / "WP7_CrossDataset"),
            "--wp8-dir", str(out_root / "WP8_MethodValidity"),
            "--wp9-dir", str(out_root / "WP9_Robustness"),
            "--wp10-dir", str(out_root / "WP10_Efficiency"),
            "--out-dir", str(out_root / "WP11_PaperPackage"),
        ]
        wp11_build_paper_package.main()
    finally:
        sys.argv = prev_argv

    return {
        "wp6_models": list(merged_wp6.keys()),
        "wp10_models": list(merged_wp10.keys()),
        "out_root": str(out_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Kernel 1 + Kernel 2 Milestone 3 outputs & rebuild WP11.")
    parser.add_argument("--sources", nargs="+", required=True, help="Paths to Kernel 1 / Kernel 2 zips or folders")
    parser.add_argument("--out-dir", type=str, default=str(MILE3_DIR))
    args = parser.parse_args()
    summary = merge_kernel_outputs([Path(s) for s in args.sources], Path(args.out_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
