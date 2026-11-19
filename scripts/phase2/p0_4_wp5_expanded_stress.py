"""
p0_4_wp5_expanded_stress.py
===========================
MileStone 2 P0-4 Closure Experiment:
  Expands WP5 3-Axis Stress Evaluation (`Retrieval Noise`, `Factual Conflict Injection`,
  and `Gold Evidence Drop`) from N=60 to N=150 unseen HotpotQA test items under the
  unified `v5_proxy` M1 signal alignment (`P0-2`), reporting:
    - Mean metrics (EM, F1, Effective F1, Conflict Retention, ECE, Brier, AUROC, AURC, Abstention)
    - Paired Bootstrap 95% CIs (`Full_UGraphRAG` vs `B0_StandardRAG` and `Full_HardPrune`)
    - Holm-Bonferroni & Benjamini-Hochberg FDR adjusted p-values across stress intensities.

Writes strictly to `results/MileStone2/P0_Closure_Fixes/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import load_or_build_cache
from scripts.phase2.common import (
    WP_DIRS,
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    get_vram_snapshot_mb,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    sequential_load_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from scripts.phase2.wp5_stress_and_profile import run_stress_suites

logger = setup_phase2_logger("phase2.p0_4")


def _build_stress_sig_tests(
    suite_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute paired statistical tests (with Holm & BH-FDR) across stress levels for one axis."""
    f1_vs_b0: Dict[str, Any] = {}
    brier_vs_b0: Dict[str, Any] = {}
    f1_vs_prune: Dict[str, Any] = {}

    for lvl, payload in suite_map.items():
        recs = payload.get("records", {})
        r_full = recs.get("Full_UGraphRAG", [])
        r_b0 = recs.get("B0_StandardRAG", [])
        r_prune = recs.get("Full_HardPrune", [])
        if r_full and len(r_full) == len(r_b0):
            f1_vs_b0[lvl] = paired_bootstrap_with_tests(
                [r["raw_f1"] * 100.0 for r in r_full],
                [r["raw_f1"] * 100.0 for r in r_b0],
            )
            brier_vs_b0[lvl] = paired_bootstrap_with_tests(
                [(r["confidence"] - r["raw_f1"]) ** 2 for r in r_full],
                [(r["confidence"] - r["raw_f1"]) ** 2 for r in r_b0],
            )
        if r_full and len(r_full) == len(r_prune):
            f1_vs_prune[lvl] = paired_bootstrap_with_tests(
                [r["raw_f1"] * 100.0 for r in r_full],
                [r["raw_f1"] * 100.0 for r in r_prune],
            )

    return {
        "Full_vs_B0_f1_pct": apply_multiple_comparison_correction(f1_vs_b0),
        "Full_vs_B0_brier": apply_multiple_comparison_correction(brier_vs_b0),
        "Full_vs_HardPrune_f1_pct": apply_multiple_comparison_correction(f1_vs_prune),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-4: Expanded WP5 3-Axis Stress Evaluation (N >= 150).")
    parser.add_argument("--model", type=str, default="qwen3b")
    parser.add_argument("--stress-limit", type=int, default=150)
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["P0"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.p0_4", out_dir / "p0_4_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()

    cache = load_or_build_cache("hotpotqa", pool_k=16, allow_synthetic_fallback=args.dry_run)
    test_items = [it for it in cache.get("items", []) if it.get("split") == "test"]
    if not test_items:
        test_items = list(cache.get("items", []))
    stress_items = test_items[: (12 if args.dry_run else args.stress_limit)]

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_p0_4_{args.model}.json", flush_every=25)
    reset_cuda_peak()
    generator = sequential_load_generator(args.model, dry_run=args.dry_run, max_new_tokens=48)
    t0 = time.time()

    try:
        logger.info(f"Running P0-4 Expanded 3-Axis Stress Suite (N={len(stress_items)}, eu_mode=v5_proxy)...")
        raw_suites = run_stress_suites(
            stress_items,
            generator,
            params,
            ckpt,
            eu_mode="v5_proxy",
            noise_rates=(0.0, 0.4, 0.6),
            conflict_rates=(0.2, 0.4),
            drop_rates=(0.5, 1.0),
            return_per_sample=True,
        )

        # Link clean 0% baseline cleanly across conflict & gold drop comparisons so zero redundant passes occur
        clean_0_payload = raw_suites["retrieval_noise_suite"]["noise_0pct"]
        raw_suites["conflict_injection_suite"] = {
            "conflict_0pct": clean_0_payload,
            **raw_suites["conflict_injection_suite"],
        }
        raw_suites["gold_evidence_drop_suite"] = {
            "gold_drop_0pct": clean_0_payload,
            **raw_suites["gold_evidence_drop_suite"],
        }

        summaries: Dict[str, Any] = {}
        stats: Dict[str, Any] = {}
        for axis_name, suite_map in raw_suites.items():
            summaries[axis_name] = {lvl: pl["summary"] for lvl, pl in suite_map.items()}
            stats[axis_name] = _build_stress_sig_tests(suite_map)

        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "eu_mode": "v5_proxy",
                "n_tested": len(stress_items),
                "dry_run": args.dry_run,
                "elapsed_sec": round(time.time() - t0, 2),
                "vram_peak_mb": get_vram_snapshot_mb(),
            },
            "v5_integrity": v5_info,
            "stress_suites": summaries,
            "paired_significance": stats,
        }
        out_file = out_dir / f"p0_4_wp5_expanded_stress_{args.model}.json"
        atomic_write_json(out_file, payload)
        logger.info(f"Saved P0-4 expanded stress results -> {out_file}")
    finally:
        teardown_generator(generator)


if __name__ == "__main__":
    main()
