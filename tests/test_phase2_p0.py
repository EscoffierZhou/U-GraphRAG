"""
test_phase2_p0.py
=================
Unit & isolated `--dry-run` verification suite for MileStone 2 P0 Closure fixes
(P0-1 .. P0-4):
  - Holm-Bonferroni & Benjamini-Hochberg FDR multiple comparison adjustments
  - P0-1 Dev Pareto lambda_conflict selection & overlap-gated conflict diagnostic
  - P0-2 Unified M1 signal registry & WP4/WP5 `v5_proxy` alignment
  - P0-3 Dev Risk-Constrained Coverage Thresholding & adaptive gating
  - P0-4 End-to-end isolated dry-run across p0_1 -> p0_2 -> p0_3 -> p0_4
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2 import (
    cache_builder,
    common,
    p0_1_m2_conflict_and_scale,
    p0_2_m1_signal_unification,
    p0_3_m3_adaptive_gating,
    p0_4_wp5_expanded_stress,
    plot_p0_closure,
)
from scripts.phase2.wp1_eu_validation import _paraphrase_queries  # noqa: F401
from src.m2_evidence_optimizer import EvidenceCandidate


class TestP0ClosureMathAndPolicies(unittest.TestCase):
    def test_holm_and_bh_fdr_multiple_comparison_correction(self) -> None:
        raw_p = [0.001, 0.04, 0.03, 0.50]
        res = common.adjust_p_values_multiple_comparisons(raw_p)
        holm = res["holm_p_values"]
        bh = res["bh_fdr_q_values"]
        # Sorted p: 0.001 (m=4 -> 0.004), 0.03 (m=3 -> 0.09), 0.04 (m=2 -> 0.08 -> mono 0.09), 0.50 (m=1 -> 0.50)
        self.assertAlmostEqual(holm[0], 0.004, places=5)
        self.assertAlmostEqual(holm[2], 0.09, places=5)
        self.assertAlmostEqual(holm[1], 0.09, places=5)
        self.assertAlmostEqual(holm[3], 0.50, places=5)
        # BH FDR: rank4=0.50, rank3=0.04*4/3=0.053333, rank2=min(0.053333, 0.03*4/2=0.06)=0.053333, rank1=0.004
        self.assertAlmostEqual(bh[0], 0.004, places=5)
        self.assertAlmostEqual(bh[2], 0.053333, places=5)
        self.assertAlmostEqual(bh[1], 0.053333, places=5)
        self.assertAlmostEqual(bh[3], 0.50, places=5)

    def test_overlap_gated_conflict_isolates_clone_vs_bridge(self) -> None:
        anchor = EvidenceCandidate(
            doc_id="d_anchor",
            text="The Treaty of Ghent was signed in 1814 in Belgium ending the war.",
            score=0.9,
            entities={"treaty", "ghent", "1814", "belgium"},
        )
        clone_conf = EvidenceCandidate(
            doc_id="d_clone_conf",
            text="The Treaty of Ghent was signed in 1831 in Belgium ending the war.",
            score=0.88,
            entities={"treaty", "ghent", "1831", "belgium"},
        )
        bridge = EvidenceCandidate(
            doc_id="d_bridge",
            text="Belgium was formally established as an independent constitutional monarchy after the 1830 revolution.",
            score=0.7,
            entities={"belgium", "1830", "monarchy"},
        )
        dists = {
            "d_anchor": {"1814": 0.9, "1831": 0.1},
            "d_clone_conf": {"1814": 0.1, "1831": 0.9},
            "d_bridge": {"monarchy": 0.85, "1814": 0.15},
        }
        mat_orig, _ = common.build_interaction_matrix_for_docs([anchor, clone_conf, bridge], dists, graph_mode="entity_js")
        mat_gated, _ = common.build_interaction_matrix_for_docs(
            [anchor, clone_conf, bridge], dists, graph_mode="entity_js", overlap_conflict_shingle_gate=0.20
        )
        # In un-gated original, disjoint bridge pair STILL incurs substantial JS conflict
        self.assertGreater(mat_orig.conflict[0][2], 0.02)
        # In overlap-gated, disjoint bridge conflict drops to 0.0 while overlapping conflict clone stays strong
        self.assertEqual(mat_gated.conflict[0][2], 0.0)
        self.assertGreater(mat_gated.conflict[0][1], 0.25)

    def test_p01_dev_pareto_selector_detects_recall_collapse(self) -> None:
        # Case A: Higher lambdas collapse recall without clean Pareto trade-off -> pareto_gate_passed=False
        dev_clean_bad = {
            "M2_orig_lc1.0": {"f1": 55.0, "sp_recall": 90.0},
            "M2_orig_lc2.5": {"f1": 52.0, "sp_recall": 82.0},
            "M2_orig_lc5.0": {"f1": 45.0, "sp_recall": 62.0},
            "M2_orig_lc10.0": {"f1": 40.0, "sp_recall": 50.0},
        }
        dev_conf_bad = {
            "M2_orig_lc1.0": {"f1": 53.0, "sp_recall": 80.0, "conflict_retention_pct": 98.0},
            "M2_orig_lc2.5": {"f1": 51.0, "sp_recall": 72.0, "conflict_retention_pct": 94.0},
            "M2_orig_lc5.0": {"f1": 47.0, "sp_recall": 58.0, "conflict_retention_pct": 72.0},
            "M2_orig_lc10.0": {"f1": 44.0, "sp_recall": 50.0, "conflict_retention_pct": 66.0},
        }
        res_bad = p0_1_m2_conflict_and_scale.select_dev_pareto_lambda(dev_clean_bad, dev_conf_bad)
        self.assertFalse(res_bad["pareto_gate_passed"])

        # Case B: Valid Pareto improvement at lambda=2.5 passes gate cleanly
        dev_clean_good = {**dev_clean_bad, "M2_orig_lc2.5": {"f1": 56.2, "sp_recall": 88.5}}
        dev_conf_good = {**dev_conf_bad, "M2_orig_lc2.5": {"f1": 55.0, "sp_recall": 79.0, "conflict_retention_pct": 75.0}}
        res_good = p0_1_m2_conflict_and_scale.select_dev_pareto_lambda(dev_clean_good, dev_conf_good)
        self.assertTrue(res_good["pareto_gate_passed"])
        self.assertEqual(res_good["selected_lambda_conflict"], 2.5)

    def test_p03_dev_risk_constrained_threshold_avoids_hard_task_collapse(self) -> None:
        cal = {"T": 1.0, "w_tok": 1.0, "platt_a": 1.0, "platt_b": 0.0}
        # Hard low-confidence regime (e.g., MuSiQue): confidences around 0.15..0.35 (would collapse under 0.38)
        hard_traces = [
            {"raw_confidence": 0.12 + 0.01 * i, "ctx_probs_by_temp": {"1.0": 0.12 + 0.01 * i}, "max_ev_rel": 0.8, "mean_ev_rel": 0.7, "f1": 1.0 if i >= 16 else 0.0}
            for i in range(20)
        ]
        pol_hard = p0_3_m3_adaptive_gating.select_dev_risk_constrained_threshold(hard_traces, cal)
        self.assertEqual(pol_hard["regime"], "risk_constrained_coverage")
        self.assertGreaterEqual(pol_hard["dev_selected_coverage_pct"], 75.0)
        self.assertLess(pol_hard["adaptive_tau_conf"], 0.38)

        # Easy high-accuracy regime (e.g., PopQA): retains >= 90% coverage
        easy_traces = [
            {"raw_confidence": 0.90 if i > 0 else 0.10, "ctx_probs_by_temp": {"1.0": 0.90 if i > 0 else 0.10}, "max_ev_rel": 0.9, "mean_ev_rel": 0.9, "f1": 1.0 if i > 0 else 0.0}
            for i in range(20)
        ]
        pol_easy = p0_3_m3_adaptive_gating.select_dev_risk_constrained_threshold(easy_traces, cal)
        self.assertEqual(pol_easy["regime"], "low_risk_outlier_guard")
        self.assertGreaterEqual(pol_easy["dev_selected_coverage_pct"], 90.0)


class TestP0ClosureEndToEndDryRun(unittest.TestCase):
    def test_p0_1_through_p0_4_isolated_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_data = tmp_root / "data_p2"
            tmp_cache = tmp_data / "caches"
            tmp_out = tmp_root / "p0_out"
            tmp_out.mkdir(parents=True, exist_ok=True)

            try:
                with patch.object(cache_builder, "PHASE2_DATA_DIR", tmp_data), \
                     patch.object(cache_builder, "PHASE2_CACHE_DIR", tmp_cache):

                    with patch.object(sys, "argv", [
                        "cache_builder.py",
                        "--datasets", "hotpotqa", "popqa", "2wikimultihop", "musique",
                        "--limit", "30", "--hotpot-limit", "300", "--pool-k", "16", "--dry-run",
                    ]):
                        cache_builder.main()

                    # P0-1 dry run
                    with patch.object(sys, "argv", [
                        "p0_1_m2_conflict_and_scale.py",
                        "--datasets", "hotpotqa", "2wikimultihop",
                        "--out-dir", str(tmp_out),
                        "--dry-run",
                    ]):
                        p0_1_m2_conflict_and_scale.main()
                    p01_res = json.loads((tmp_out / "p0_1_m2_conflict_and_scale.json").read_text(encoding="utf-8"))
                    self.assertIn("hotpotqa", p01_res["datasets"])
                    self.assertIn("test_clean_significance", p01_res["datasets"]["hotpotqa"])

                    # P0-2 dry run
                    with patch.object(sys, "argv", [
                        "p0_2_m1_signal_unification.py",
                        "--models", "qwen3b",
                        "--out-dir", str(tmp_out),
                        "--dry-run",
                    ]):
                        p0_2_m1_signal_unification.main()
                    p02_res = json.loads((tmp_out / "p0_2_m1_signal_unification.json").read_text(encoding="utf-8"))
                    self.assertIn("v5_eu_missing_minus_gold", p02_res["expanded_replacement_interventions"]["qwen3b"]["paired_interventions"])

                    # P0-3 dry run
                    with patch.object(sys, "argv", [
                        "p0_3_m3_adaptive_gating.py",
                        "--models", "qwen3b",
                        "--datasets", "popqa", "hotpotqa", "2wikimultihop", "musique",
                        "--out-dir", str(tmp_out),
                        "--dry-run",
                    ]):
                        p0_3_m3_adaptive_gating.main()
                    p03_res = json.loads((tmp_out / "p0_3_m3_adaptive_gating.json").read_text(encoding="utf-8"))
                    self.assertIn("Gated_Beta__DevAdaptiveTau", p03_res["models"]["qwen3b"]["datasets"]["musique"]["test_policies"])

                    # P0-4 dry run
                    with patch.object(sys, "argv", [
                        "p0_4_wp5_expanded_stress.py",
                        "--model", "qwen3b",
                        "--out-dir", str(tmp_out),
                        "--dry-run",
                    ]):
                        p0_4_wp5_expanded_stress.main()
                    p04_res = json.loads((tmp_out / "p0_4_wp5_expanded_stress_qwen3b.json").read_text(encoding="utf-8"))
                    self.assertIn("Full_vs_B0_f1_pct", p04_res["paired_significance"]["conflict_injection_suite"])

                    # Plotter smoke checks on isolated outputs
                    fig_dir = tmp_out / "figures"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    p1_fig = plot_p0_closure.plot_p0_1_figures(tmp_out, fig_dir)
                    p2_fig = plot_p0_closure.plot_p0_2_figures(tmp_out, fig_dir)
                    p3_fig = plot_p0_closure.plot_p0_3_figures(tmp_out, fig_dir)
                    p4_fig = plot_p0_closure.plot_p0_4_figures(tmp_out, fig_dir)
                    for fp in (p1_fig, p2_fig, p3_fig, p4_fig):
                        self.assertTrue(fp.exists() and fp.stat().st_size > 1000)
            finally:
                common.close_phase2_log_handlers()


if __name__ == "__main__":
    unittest.main()
