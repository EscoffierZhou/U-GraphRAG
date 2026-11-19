"""
test_phase2_wp.py
=================
CPU unit & end-to-end `--dry-run` verification suite for MileStone 2 (`scripts/phase2/`).
Strictly isolates all writes into `tempfile.TemporaryDirectory()` and never modifies
`V5_FROZEN`, `MileStone1`, or production `WP1..WP5` directories.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2 import cache_builder, common
from scripts.phase2 import (
    wp1_eu_validation,
    wp2_m2_ablation,
    wp3_m3_calibration,
    wp4_generalization,
    wp5_stress_and_profile,
)


class TestPhase2Infrastructure(unittest.TestCase):
    def test_v5_frozen_checksums_and_write_guard(self) -> None:
        info = common.verify_v5_frozen_checksums()
        self.assertEqual(info["status"], "ok")
        self.assertGreaterEqual(info["verified_count"], 60)

        with self.assertRaises(ValueError):
            common.assert_safe_output_path(common.V5_FROZEN_DIR / "results" / "grid_search_results.json")
        with self.assertRaises(ValueError):
            common.assert_safe_output_path(common.MILE1_DIR / "should_fail.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            v5_copy = Path(tmpdir) / "V5_FROZEN_copy"
            shutil.copytree(common.V5_FROZEN_DIR, v5_copy)
            # Corrupt one JSON copy inside tmpdir and ensure verifier detects mismatch
            target = v5_copy / "results" / "grid_search_results.json"
            target.write_text('{"corrupted": true}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                common.verify_v5_frozen_checksums(v5_copy)

    def test_load_v5_balanced_params(self) -> None:
        params = common.load_v5_balanced_params()
        self.assertIn("m1_tau_eu", params)
        self.assertIn("m3_confidence_platt_a", params)
        self.assertAlmostEqual(float(params["m1_tau_eu"]), 0.18, places=4)

    def test_aurc_auprc_abstention_and_paired_stats(self) -> None:
        confs = [0.95, 0.90, 0.80, 0.65, 0.40, 0.25, 0.15, 0.10]
        accs = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        abstained = [False, False, False, False, True, True, True, True]

        suite = common.compute_calibration_and_risk_suite(confs, accs, abstained_flags=abstained)
        self.assertGreaterEqual(suite["auroc_confidence"], 0.99)
        self.assertGreaterEqual(suite["auprc_error"], 0.99)
        self.assertEqual(suite["harmful_abstention_rate_pct"], 0.0)
        self.assertEqual(suite["valid_abstention_precision_pct"], 100.0)
        self.assertAlmostEqual(suite["e_aurc"], 0.0, places=5)

        a = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        sig = common.paired_bootstrap_with_tests(a, b, n_boot=500, seed=7)
        self.assertGreater(sig["mean_delta"], 0.4)
        self.assertGreater(sig["ci_95_low"], 0.0)
        self.assertLess(sig["sign_flip_p_value"], 0.1)

    def test_checkpoint_manager_resume_and_conflict_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.json"
            mgr = common.Phase2CheckpointManager(ckpt_path, flush_every=2)
            mgr.record("item_1", {"val": 1})
            self.assertFalse(ckpt_path.exists())
            mgr.record("item_2", {"val": 2})
            self.assertTrue(ckpt_path.exists())

            mgr2 = common.Phase2CheckpointManager(ckpt_path, flush_every=2)
            self.assertTrue(mgr2.has("item_1"))
            self.assertEqual(mgr2.get("item_2")["val"], 2)

        doc = {"doc_id": "d1", "title": "Treaty of Ghent", "text": "The Treaty of Ghent was signed in 1814 in Ghent."}
        conf = cache_builder.inject_numeric_or_entity_conflict(doc, seed_key="test_seed")
        self.assertTrue(conf["is_synthetic_conflict"])
        self.assertNotEqual(conf["text"], doc["text"])
        self.assertNotIn("1814", conf["text"])


class TestPhase2EndToEndDryRun(unittest.TestCase):
    def test_wp0_through_wp5_isolated_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_data = tmp_root / "data_p2"
            tmp_cache = tmp_data / "caches"
            tmp_out = tmp_root / "results"
            for sub in ("wp1", "wp2", "wp3", "wp4", "wp5"):
                (tmp_out / sub).mkdir(parents=True, exist_ok=True)

            try:
                # Isolate cache writes into tmp_cache across every module holding PHASE2_CACHE_DIR
                with patch.object(cache_builder, "PHASE2_DATA_DIR", tmp_data), \
                     patch.object(cache_builder, "PHASE2_CACHE_DIR", tmp_cache), \
                     patch.object(wp1_eu_validation, "PHASE2_CACHE_DIR", tmp_cache):

                    # 1. WP0: build caches on real hotpotqa + popqa + dryrun synthetic 2wiki/musique
                    with patch.object(sys, "argv", [
                        "cache_builder.py",
                        "--datasets", "hotpotqa", "popqa", "2wikimultihop", "musique",
                        "--limit", "35", "--hotpot-limit", "300", "--pool-k", "16", "--dry-run",
                    ]):
                        cache_builder.main()

                    self.assertTrue((tmp_cache / "hotpotqa_pool_k16.json").exists())
                    self.assertTrue((tmp_cache / "popqa_pool_k16.json").exists())
                    self.assertTrue((tmp_cache / "2wikimultihop_pool_k16.json").exists())
                    self.assertTrue((tmp_cache / "musique_pool_k16.json").exists())
                    self.assertTrue((tmp_cache / "wp1_four_quadrant_probe.json").exists())

                    # 2. WP1: dry-run uncertainty validation + threshold transfer
                    with patch.object(sys, "argv", [
                        "wp1_eu_validation.py",
                        "--models", "qwen3b", "qwen7b",
                        "--datasets", "hotpotqa", "popqa",
                        "--limit", "16",
                        "--out-dir", str(tmp_out / "wp1"),
                        "--dry-run",
                    ]):
                        wp1_eu_validation.main()
                    wp1_res = json.loads((tmp_out / "wp1" / "wp1_eu_validation_results.json").read_text(encoding="utf-8"))
                    self.assertIn("qwen3b", wp1_res["runs"])
                    self.assertIn("qwen7b", wp1_res["runs"])
                    self.assertIn("Q1_known_consistent", wp1_res["runs"]["qwen3b"]["four_quadrant_summary"])
                    self.assertIn("qwen3b_hotpot_to_qwen7b_hotpotqa", wp1_res["threshold_transfer"])

                    # 3. WP2: M2 mechanism & graph topology ablation
                    with patch.object(sys, "argv", [
                        "wp2_m2_ablation.py",
                        "--dataset", "hotpotqa", "--split", "test",
                        "--limit", "10", "--conflict-limit", "6",
                        "--out-dir", str(tmp_out / "wp2"),
                        "--dry-run",
                    ]):
                        wp2_m2_ablation.main()
                    wp2_res = json.loads((tmp_out / "wp2" / "wp2_m2_ablation_hotpotqa_test.json").read_text(encoding="utf-8"))
                    for req_var in ("A0_top3", "A1_exact_no_graph", "A4_full_m2_mst_dfs", "A5_mmr_baseline", "G_dense_cosine", "P_hard_conflict_prune"):
                        self.assertIn(req_var, wp2_res["standard_ablation_summary"])
                    self.assertIn("conflict_injected_diagnostic_summary", wp2_res)

                    # 4. WP3: M3 independent calibration + AURC/E-AURC/Risk-Coverage
                    with patch.object(sys, "argv", [
                        "wp3_m3_calibration.py",
                        "--dataset", "hotpotqa", "--dev-limit", "10", "--test-limit", "10",
                        "--out-dir", str(tmp_out / "wp3"),
                        "--dry-run",
                    ]):
                        wp3_m3_calibration.main()
                    wp3_res = json.loads((tmp_out / "wp3" / "wp3_m3_calibration_hotpotqa.json").read_text(encoding="utf-8"))
                    self.assertIn("M3_0_uncalibrated", wp3_res["unseen_test_variants"])
                    self.assertIn("M3_5_contrastive_b008", wp3_res["unseen_test_variants"])
                    self.assertIn("aurc", wp3_res["unseen_test_variants"]["M3_3_full_platt"]["metrics"])

                    # 5. WP4: Multi-dataset x multi-backbone generalization matrix
                    with patch.object(sys, "argv", [
                        "wp4_generalization.py",
                        "--models", "qwen3b", "qwen7b",
                        "--datasets", "hotpotqa", "2wikimultihop", "musique",
                        "--split", "test", "--limit", "6",
                        "--out-dir", str(tmp_out / "wp4"),
                        "--dry-run",
                    ]):
                        wp4_generalization.main()
                    wp4_res = json.loads((tmp_out / "wp4" / "wp4_generalization_matrix.json").read_text(encoding="utf-8"))
                    self.assertEqual(wp4_res["matrix"]["qwen3b"]["status"], "completed")
                    self.assertIn("musique", wp4_res["matrix"]["qwen7b"]["datasets"])

                    # 6. WP5: Profiling + 3-axis stress suites
                    with patch.object(sys, "argv", [
                        "wp5_stress_and_profile.py",
                        "--model", "qwen3b", "--profile-limit", "5", "--stress-limit", "6",
                        "--out-dir", str(tmp_out / "wp5"),
                        "--dry-run",
                    ]):
                        wp5_stress_and_profile.main()
                    wp5_res = json.loads((tmp_out / "wp5" / "wp5_stress_and_profile_qwen3b.json").read_text(encoding="utf-8"))
                    self.assertIn("stage_latency_ms", wp5_res["efficiency_profile"])
                    self.assertIn("gold_evidence_drop_suite", wp5_res["stress_suites"])
                    self.assertEqual(len(wp5_res["efficiency_profile"]["scaling_matrix"]), 6)
            finally:
                common.close_phase2_log_handlers()


if __name__ == "__main__":
    unittest.main()
