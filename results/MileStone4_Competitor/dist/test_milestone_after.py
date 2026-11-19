"""
test_milestone_after.py
=======================
CPU unit & --dry-run verification suite for all `MileStone4_Competitor/reuse` operators,
dataset adapters (`RAMDocs`, `NQ-Swap`, `S4 All-Irrelevant`), post-hoc Split Conformal Risk Control,
and end-to-end `run_milestone_after_horserace.py`. Writes strictly into `tempfile.TemporaryDirectory()`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.MileStone4_Competitor.dist.milestone_after_posthoc_conformal import run_posthoc_conformal_audit
from results.MileStone4_Competitor.dist.run_milestone_after_horserace import ALL_HORSERACE_METHODS, run_horserace
from results.MileStone4_Competitor.reuse.datasets.dataset_adapters import (
    build_all_irrelevant_s4_slice,
    load_nqswap_dataset,
    load_ramdocs_dataset,
)
from results.MileStone4_Competitor.reuse.operators.bge_reranker_operator import IsolatedBGEReranker
from results.MileStone4_Competitor.reuse.operators.cad_operator import generate_with_cad
from results.MileStone4_Competitor.reuse.operators.ck_plug_operator import (
    compute_ck_plug_step_logits,
    generate_with_ck_plug,
    relative_top_filter_pair,
)
from results.MileStone4_Competitor.reuse.operators.conformal_risk_operator import (
    evaluate_conformal_abstention,
    fit_conformal_risk_control,
    fit_split_conformal_abstention,
)
from results.MileStone4_Competitor.reuse.operators.instruct_rag_icl_operator import (
    extract_instructrag_short_answer,
    generate_with_instructrag_icl,
)
from results.MileStone4_Competitor.reuse.operators.relevance_gate_operator import check_query_evidence_relevance_gate
from scripts.phase2 import cache_builder, common
from scripts.phase3 import phase3_common
from src.baseline_rag import QwenNF4Generator


class TestMileStoneAfterOperatorsAndAdapters(unittest.TestCase):
    def test_dataset_adapters_ramdocs_nqswap_and_s4(self) -> None:
        ram = load_ramdocs_dataset(max_items=5)
        self.assertEqual(len(ram), 5)
        self.assertTrue(ram[0]["question"])
        self.assertTrue(ram[0]["possible_answers"])
        self.assertTrue(any("is_synthetic_conflict" in d for d in ram[0]["pool"]))

        nqs_dual = load_nqswap_dataset(max_items=5, mode="dual_conflict")
        self.assertEqual(len(nqs_dual), 5)
        self.assertGreaterEqual(len(nqs_dual[0]["pool"]), 2)
        self.assertTrue(nqs_dual[0]["pool"][0]["is_gold"])
        self.assertTrue(nqs_dual[0]["pool"][1]["is_synthetic_conflict"])

        s4 = build_all_irrelevant_s4_slice(ram, max_items=4)
        self.assertEqual(len(s4), 4)
        self.assertTrue(all(not d["is_gold"] for it in s4 for d in it["pool"]))

    def test_ck_plug_cad_instructrag_and_bge_operators(self) -> None:
        # CK-PLUG relative_top_filter + adaptive Confidence Gain modulation
        logits_ctx = torch.tensor([[5.0, 4.0, 1.0, -4.0, -5.0]])
        logits_stu = torch.tensor([[1.0, 4.5, 4.4, -3.0, -5.0]])
        cf, sf, mask = relative_top_filter_pair(logits_ctx, logits_stu, relative_top=0.05, min_tokens_to_keep=2)
        self.assertTrue(bool(mask[0, -1].item()))
        adj, info = compute_ck_plug_step_logits(logits_ctx, logits_stu, alpha=0.5, adaptive=True, select_top=2)
        self.assertEqual(adj.shape, logits_ctx.shape)
        self.assertIn("ig", info)

        gen = QwenNF4Generator(dry_run=True)
        docs = [{"doc_id": "d1", "title": "France", "text": "Paris is the capital of France.", "score": 0.9}]
        ck_res = generate_with_ck_plug(gen, "Capital of France?", docs)
        cad_res = generate_with_cad(gen, "Capital of France?", docs, alpha=0.5)
        ir_res = generate_with_instructrag_icl(gen, "Capital of France?", docs)
        self.assertTrue(ck_res.raw_answer)
        self.assertTrue(cad_res.raw_answer)
        self.assertTrue(ir_res.raw_answer)
        self.assertEqual(extract_instructrag_short_answer("Reasoning...\nShort Answer: Paris.", gen), "Paris")

        reranker = IsolatedBGEReranker(dry_run=True)
        sel = reranker.select_reranked("Capital of France?", docs + [{"doc_id": "d2", "title": "Unrelated", "text": "Cats sleep.", "score": 0.95}], top_k=1)
        self.assertEqual(len(sel), 1)
        reranker.teardown()

    def test_split_conformal_and_relevance_lower_bound_gate(self) -> None:
        dev_confs = [0.95, 0.90, 0.85, 0.70, 0.65, 0.60, 0.40, 0.30, 0.20, 0.15]
        dev_f1s = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sc_tight = fit_split_conformal_abstention(dev_confs, dev_f1s, alpha_harmful=0.10)
        sc_loose = fit_split_conformal_abstention(dev_confs, dev_f1s, alpha_harmful=0.35)
        self.assertLessEqual(sc_tight["tau_conf"], sc_loose["tau_conf"])

        rc = fit_conformal_risk_control(dev_confs, dev_f1s, target_risk=0.40)
        self.assertIn("tau_conf", rc)
        ev = evaluate_conformal_abstention(dev_confs, dev_f1s, sc_tight["tau_conf"])
        self.assertLessEqual(ev["harmful_abstention_rate_pct"], 25.0)

        rel_ok = check_query_evidence_relevance_gate(
            "Who directed Inception?",
            [{"title": "Inception", "text": "Inception is a film directed by Christopher Nolan.", "score": 0.85}],
        )
        self.assertTrue(rel_ok["passes_relevance_gate"])

        rel_bad = check_query_evidence_relevance_gate(
            "Who directed Inception?",
            [{"title": "Photosynthesis", "text": "Chlorophyll absorbs sunlight in plants.", "score": 0.68}],
        )
        self.assertFalse(rel_bad["passes_relevance_gate"])


class TestMileStoneAfterEndToEndDryRun(unittest.TestCase):
    def test_horserace_and_posthoc_conformal_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            p2_data = tmp_root / "data_p2"
            p2_cache = p2_data / "caches"
            p3_data = tmp_root / "data_p3"
            p3_cache = p3_data / "caches"
            out_dir = tmp_root / "after_out"
            common.atomic_write_json(
                p2_data / "hotpotqa_dev_500.json",
                cache_builder._generate_synthetic_multihop("hotpotqa", 40),
            )

            try:
                with patch.object(cache_builder, "PHASE2_DATA_DIR", p2_data), \
                     patch.object(cache_builder, "PHASE2_CACHE_DIR", p2_cache), \
                     patch.object(phase3_common, "PHASE3_DATA_DIR", p3_data), \
                     patch.object(phase3_common, "PHASE3_CACHE_DIR", p3_cache):
                    payload = run_horserace(
                        model_key="llama3_8b",
                        scale="smoke",
                        out_dir=out_dir,
                        dry_run=True,
                        seed=42,
                    )
                    self.assertTrue((out_dir / "milestone_after_horserace_results.json").exists())
                    self.assertTrue((out_dir / "TABLES_MILESTONE_AFTER.md").exists())
                    self.assertTrue((out_dir / "tables_milestone_after.tex").exists())
                    s4_diag = payload["m1_s4_relevance_gate_diagnostic"]
                    self.assertLessEqual(s4_diag["relevance_gated_early_exit_rate_pct"], s4_diag["raw_v5_early_exit_rate_pct"])
                    self.assertLess(s4_diag["relevance_gate_pass_rate_pct"], 50.0)
                    for b in ("ramdocs", "nqswap_dual", "hotpotqa_conflict", "hotpotqa_s4_irrelevant", "hotpotqa_clean"):
                        self.assertIn(b, payload["horserace_results"])
                        self.assertEqual(set(payload["horserace_results"][b].keys()), set(ALL_HORSERACE_METHODS))

                    # Verify offline Split Conformal post-hoc audit against a mock WP8 ckpt
                    wp8_ckpt = tmp_root / "ckpt_wp8_qwen3b_smoke.json"
                    mock_ckpt = common.Phase2CheckpointManager(wp8_ckpt, flush_every=10)
                    for i in range(10):
                        f1v = 1.0 if i < 6 else 0.0
                        cp = 0.85 if i < 6 else 0.25
                        tr = {
                            "id": f"id_{i}", "split": "dev", "raw_answer": "Ans", "em": f1v, "f1": f1v,
                            "ctx_probs_by_temp": {"0.8": cp, "1.0": cp, "1.25": cp, "1.5": cp, "2.0": cp, "2.5": cp, "3.0": cp},
                            "max_ev_rel": 0.9, "mean_ev_rel": 0.7, "is_faithful": True, "has_unanchored": False, "raw_confidence": cp,
                        }
                        mock_ckpt.record(f"wp8_3::hotpotqa::trace::dev_b0_v2::d_{i}", tr)
                        mock_ckpt.record(f"wp8_3::hotpotqa::trace::test_b0_v2::t_{i}", tr)
                    mock_ckpt.flush(force=True)
                    posthoc = run_posthoc_conformal_audit(wp8_ckpt_path=wp8_ckpt, out_dir=out_dir / "posthoc")
                    self.assertIn("hotpotqa", posthoc["wp8_conformal"])
                    self.assertTrue((out_dir / "posthoc" / "CONFORMAL_POSTHOC_TABLE.md").exists())
            finally:
                common.close_phase2_log_handlers("phase2.")
                common.close_phase2_log_handlers("phase4.")


if __name__ == "__main__":
    unittest.main()
