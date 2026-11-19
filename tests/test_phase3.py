"""
test_phase3.py
==============
CPU Unit & End-to-End `--dry-run` Verification Suite for MileStone 3 (WP6-WP11, AutoDL Setup,
Cross-Family 5-Model 3+2 Kaggle Anti-Disk-Overflow Grouping, Gemma-2 EOS, and Safe Mode Resolution).
Strictly isolates all writes into `tempfile.TemporaryDirectory()`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2 import cache_builder, common
from scripts.phase3 import (
    autodl_setup,
    merge_phase3_kernels,
    phase3_common,
    wp6_cross_family,
    wp7_cross_dataset,
    wp8_method_validity,
    wp9_wp10_stress_efficiency,
    wp11_build_paper_package,
)
from src.baseline_rag import QwenNF4Generator, collect_eos_and_pad_ids


class _DummyMistralTokenizer:
    eos_token_id = [2, 3]
    pad_token_id = None
    unk_token_id = 0
    chat_template = "dummy"

    def convert_tokens_to_ids(self, tok: str) -> int:
        return {"<|eot_id|>": 128009, "</s>": 2, "<end_of_turn>": 107}.get(tok, -1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(m.get("role") == "system" for m in messages):
            raise RuntimeError("System role not supported in dummy Mistral template")
        return f"[INST] {messages[0]['content']} [/INST]"


class _DummyGemmaTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3
    chat_template = "dummy"

    def convert_tokens_to_ids(self, tok: str) -> int:
        return {"<end_of_turn>": 107, "<|eot_id|>": 128009, "</s>": 2}.get(tok, -1)


class TestPhase3ModelRegistryAndGrouping(unittest.TestCase):
    """
    Validates model registry configuration, 3+2 Kaggle anti-disk-overflow
    groups (core3, extra2, all5), and model alias deduplication.
    """

    def test_model_registry_contains_gemma2(self) -> None:
        self.assertIn("gemma2", phase3_common.PHASE3_MODEL_REGISTRY)
        gemma = phase3_common.PHASE3_MODEL_REGISTRY["gemma2"]
        self.assertEqual(gemma["family"], "Gemma-2")
        self.assertIn("gemma-2-9b-it", gemma["hf_repo"].lower())
        self.assertIn("gemma-2-9b-it", gemma["local_dirs"])

    def test_model_groups_satisfy_3_plus_2_layout(self) -> None:
        self.assertIn("core3", phase3_common.PHASE3_MODEL_GROUPS)
        self.assertIn("extra2", phase3_common.PHASE3_MODEL_GROUPS)
        self.assertIn("all5", phase3_common.PHASE3_MODEL_GROUPS)

        core3 = phase3_common.PHASE3_MODEL_GROUPS["core3"]
        extra2 = phase3_common.PHASE3_MODEL_GROUPS["extra2"]
        all5 = phase3_common.PHASE3_MODEL_GROUPS["all5"]

        self.assertEqual(len(core3), 3, "core3 group must have exactly 3 models")
        self.assertEqual(len(extra2), 2, "extra2 group must have exactly 2 models")
        self.assertEqual(len(all5), 5, "all5 group must have exactly 5 models")

        self.assertEqual(core3, ["qwen3b", "qwen7b", "llama3_8b"])
        self.assertEqual(extra2, ["mistral7b", "gemma2"])
        self.assertEqual(all5, ["qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"])

    def test_resolve_phase3_model_list_groups_and_aliases(self) -> None:
        # 1. Group resolution
        self.assertEqual(
            phase3_common.resolve_phase3_model_list(model_group="core3"),
            ["qwen3b", "qwen7b", "llama3_8b"],
        )
        self.assertEqual(
            phase3_common.resolve_phase3_model_list(model_group="extra2"),
            ["mistral7b", "gemma2"],
        )
        self.assertEqual(
            phase3_common.resolve_phase3_model_list(model_group="all5"),
            ["qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"],
        )

        # 2. Alias mapping and deduplication ('gemma2_9b' -> 'gemma2')
        res = phase3_common.resolve_phase3_model_list(["gemma2_9b", "qwen3b", "gemma2"])
        self.assertEqual(res, ["gemma2", "qwen3b"])

        # 3. Merging explicit models with group
        res_merge = phase3_common.resolve_phase3_model_list(
            models=["qwen3b", "gemma2"],
            model_group="core3",
        )
        self.assertEqual(res_merge, ["qwen3b", "qwen7b", "llama3_8b", "gemma2"])


class TestPhase3TokenizerAndGraphMode(unittest.TestCase):
    """
    Validates tokenization stop-ids (including Gemma <end_of_turn>),
    adaptive graph mode resolution, and lambda_conflict defaults.
    """

    def test_collect_eos_recognizes_gemma_end_of_turn(self) -> None:
        tok = _DummyGemmaTokenizer()
        eos_ids, pad_id = collect_eos_and_pad_ids(tok)
        self.assertIn(107, eos_ids, "collect_eos_and_pad_ids must recognize Gemma '<end_of_turn>'")
        self.assertIn(1, eos_ids)
        self.assertIn(128009, eos_ids)
        self.assertEqual(pad_id, 0)

    def test_collect_eos_handles_list_and_llama_eot(self) -> None:
        tok = _DummyMistralTokenizer()
        eos_ids, pad_id = collect_eos_and_pad_ids(tok)
        self.assertIn(2, eos_ids)
        self.assertIn(3, eos_ids)
        self.assertIn(128009, eos_ids)
        self.assertIn(107, eos_ids)
        self.assertIsInstance(pad_id, int)

    def test_build_prompt_falls_back_when_system_role_rejected(self) -> None:
        gen = QwenNF4Generator(dry_run=True)
        gen.tokenizer = _DummyMistralTokenizer()
        prompt = gen.build_prompt("Who created Python?", [{"title": "Doc", "text": "Guido van Rossum created Python."}])
        self.assertTrue(prompt.startswith("[INST]"))
        self.assertIn("Who created Python?", prompt)

    def test_select_m2_by_graph_mode_default_and_override_lambda(self) -> None:
        pool = [
            {"id": "doc1", "title": "Doc1", "text": "Python is a high-level programming language."},
            {"id": "doc2", "title": "Doc2", "text": "Python was created by Guido van Rossum."},
        ]
        with patch("scripts.phase3.phase3_common.TwoStageM2Optimizer") as mock_opt_cls:
            mock_inst = mock_opt_cls.return_value
            mock_inst.optimize.return_value = SimpleNamespace(
                ordered_docs=[],
                m2a_result=SimpleNamespace(evaluated_subsets_count=1),
                total_elapsed_ms=0.1,
            )

            # 1. overlap_gated_js default lambda_conflict == 5.0
            phase3_common.select_m2_by_graph_mode(
                question="Who created Python?",
                pool=pool,
                per_doc_dists=None,
                params={"gamma_relevance": 1.0},
                graph_mode="overlap_gated_js",
                lambda_conflict=None,
            )
            self.assertEqual(mock_opt_cls.call_args.kwargs["lambda_conflict"], 5.0)

            # 2. entity_js default lambda_conflict == 1.0
            phase3_common.select_m2_by_graph_mode(
                question="Who created Python?",
                pool=pool,
                per_doc_dists=None,
                params={"gamma_relevance": 1.0},
                graph_mode="entity_js",
                lambda_conflict=None,
            )
            self.assertEqual(mock_opt_cls.call_args.kwargs["lambda_conflict"], 1.0)

            # 3. entity_cooccur default lambda_conflict == 0.0
            phase3_common.select_m2_by_graph_mode(
                question="Who created Python?",
                pool=pool,
                per_doc_dists=None,
                params={"gamma_relevance": 1.0},
                graph_mode="entity_cooccur",
                lambda_conflict=None,
            )
            self.assertEqual(mock_opt_cls.call_args.kwargs["lambda_conflict"], 0.0)

            # 4. Explicit override lambda_conflict
            phase3_common.select_m2_by_graph_mode(
                question="Who created Python?",
                pool=pool,
                per_doc_dists=None,
                params={"gamma_relevance": 1.0},
                graph_mode="overlap_gated_js",
                lambda_conflict=2.75,
            )
            self.assertEqual(mock_opt_cls.call_args.kwargs["lambda_conflict"], 2.75)

    def test_resolve_wp8_recommended_graph_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Non-existent path returns default fallback
            mode, src = phase3_common.resolve_wp8_recommended_graph_mode(
                default="overlap_gated_js",
                wp8_path=tmp_path / "missing_wp8.json",
            )
            self.assertEqual(mode, "overlap_gated_js")
            self.assertEqual(src, "fallback_default")

            # 2. Valid WP8 result returns wp8_decision
            valid_file = tmp_path / "wp8_valid.json"
            valid_file.write_text(
                json.dumps({
                    "wp8_2_m2_validity": {
                        "validity_gate_decision": {
                            "recommended_m2_graph_mode": "entity_js"
                        }
                    }
                }),
                encoding="utf-8",
            )
            mode, src = phase3_common.resolve_wp8_recommended_graph_mode(wp8_path=valid_file)
            self.assertEqual(mode, "entity_js")
            self.assertEqual(src, "wp8_decision")

            # 3. Corrupted or invalid JSON returns safe fallback
            corrupt_file = tmp_path / "wp8_corrupt.json"
            corrupt_file.write_text("not a json document", encoding="utf-8")
            mode, src = phase3_common.resolve_wp8_recommended_graph_mode(
                default="overlap_gated_js",
                wp8_path=corrupt_file,
            )
            self.assertEqual(mode, "overlap_gated_js")
            self.assertEqual(src, "fallback_default")

    def test_failure_attribution_classifier(self) -> None:
        self.assertEqual(phase3_common.classify_failure_case(1.0, False, 1.0, 1.0), "correct")
        self.assertEqual(phase3_common.classify_failure_case(0.0, True, 0.5, 0.5), "valid_abstention_guard")
        self.assertEqual(phase3_common.classify_failure_case(1.0, True, 1.0, 1.0), "harmful_abstention")
        self.assertEqual(phase3_common.classify_failure_case(0.0, False, 0.0, 0.0), "retrieval_miss")
        self.assertEqual(phase3_common.classify_failure_case(0.0, False, 1.0, 0.5), "selection_loss")
        self.assertEqual(phase3_common.classify_failure_case(0.0, False, 1.0, 1.0), "generation_or_reasoning_error")

    def test_zero_leakage_overlap_conflict_guard_and_ungated_registry(self) -> None:
        # 1. Registry uses ungated mirrors for cross-family downloads on overseas Kaggle
        for mk in ("qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"):
            reg = phase3_common.PHASE3_MODEL_REGISTRY[mk]
            self.assertFalse(reg["gated_hf"], f"{mk} default HF repo must be ungated")

        # 2. Overlap guard uses lexical shingles ONLY (never peeking at is_synthetic_conflict)
        clean_bridge = [
            {"doc_id": "a", "text": "The Treaty of Ghent was signed in 1814 ending the War of 1812."},
            {"doc_id": "b", "text": "John Quincy Adams served as chief American negotiator during peace talks."},
        ]
        self.assertFalse(phase3_common.should_bypass_m1_early_exit(clean_bridge, "overlap_gated_js"))

        overlapping_unlabeled = [
            {"doc_id": "a", "text": "The Treaty of Ghent was signed in December 1814 in Belgium ending hostilities.", "is_synthetic_conflict": False},
            {"doc_id": "c", "text": "The Treaty of Ghent was signed in December 1831 in Belgium ending hostilities.", "is_synthetic_conflict": False},
        ]
        self.assertTrue(phase3_common.should_bypass_m1_early_exit(overlapping_unlabeled, "overlap_gated_js"))
        self.assertFalse(phase3_common.should_bypass_m1_early_exit(overlapping_unlabeled, "entity_js"))

        # 3. WP9 Axis B1 conflict slice actively uses overlap_gated_js when m2_graph_mode='overlap_gated_js'
        gen = QwenNF4Generator(dry_run=True)
        params = phase3_common.load_v5_balanced_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = phase3_common.Phase2CheckpointManager(Path(tmpdir) / "wp9_leak_test.json")
            item = {
                "id": "leak_check_1",
                "question": "When was the Treaty of Ghent signed?",
                "possible_answers": ["1814"],
                "perturbed_pool": [
                    {
                        "doc_id": "g1",
                        "title": "Treaty of Ghent",
                        "text": "The Treaty of Ghent was signed in 1814 in Belgium.",
                        "score": 0.99,
                        "is_gold": True,
                        "is_synthetic_conflict": False,
                    },
                    {
                        "doc_id": "c1",
                        "title": "Treaty Conf",
                        "text": "The Treaty of Ghent was signed in 1831 in Belgium.",
                        "score": 0.97,
                        "is_gold": False,
                        "is_synthetic_conflict": True,
                    },
                    {
                        "doc_id": "d2",
                        "title": "War of 1812",
                        "text": "The War of 1812 ended following ratified peace agreements in Europe.",
                        "score": 0.65,
                        "is_gold": False,
                        "is_synthetic_conflict": False,
                    },
                ],
            }
            res = wp9_wp10_stress_efficiency._eval_wp9_slice(
                "axis_b1_conflict", "conflict_100pct", [item], gen, params, "overlap_gated_js", ckpt
            )
            self.assertIn("Full_UGraphRAG", res)

    def test_download_with_hf_and_m3_fallback_logger(self) -> None:
        import os
        import src.m3_calibrated_decoder as m3_mod

        self.assertTrue(hasattr(m3_mod, "logger"), "m3_calibrated_decoder must define module logger for contrastive fallback")

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "bge"

            def _fake_snapshot(repo_id: str, local_dir: str, **kwargs: Any) -> str:
                p = Path(local_dir)
                p.mkdir(parents=True, exist_ok=True)
                (p / "model.safetensors").write_bytes(b"ok")
                return str(p)

            with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot), \
                 patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, clear=False):
                ok = autodl_setup._download_with_hf("BAAI/bge-small-en-v1.5", target)
                self.assertTrue(ok)

    def test_abstention_metrics_llama_hedging_sentencepiece_nl_and_calib_b0(self) -> None:
        import src.m3_calibrated_decoder as m3_mod

        # 1. evaluate_qa_prediction must preserve raw_em/raw_f1 when abstaining (only zeroing effective_em/effective_f1)
        qa_ab_right = common.evaluate_qa_prediction("Paris", ["Paris"], is_abstained=True)
        self.assertEqual(qa_ab_right["em"], 1.0)
        self.assertEqual(qa_ab_right["f1"], 1.0)
        self.assertEqual(qa_ab_right["effective_em"], 0.0)
        self.assertEqual(qa_ab_right["effective_f1"], 0.0)

        suite = common.compute_calibration_and_risk_suite(
            [0.9, 0.2, 0.8],
            [1.0, 0.0, 1.0],
            abstained_flags=[False, True, True],
        )
        self.assertGreater(suite["harmful_abstention_rate_pct"], 0.0)
        self.assertLess(suite["valid_abstention_precision_pct"], 100.0)

        # 2. Llama-3.1-8B single-line hedging strip preserves valid answers & bracket prefixes
        gen = QwenNF4Generator(dry_run=True)
        self.assertEqual(
            gen._postprocess_answer("Yes.  However, the context does not confirm both."),
            "Yes",
        )
        self.assertEqual(
            gen._postprocess_answer("Scott Derrickson. Answer: Scott Derrickson"),
            "Scott Derrickson",
        )
        self.assertEqual(
            gen._postprocess_answer("1995 (Note: based on Document 1)"),
            "1995",
        )
        self.assertEqual(
            gen._postprocess_answer("[1] Short Answer: Scott Derrickson"),
            "Scott Derrickson",
        )
        self.assertEqual(
            gen._postprocess_answer("James Burke [1] [5] Connections (TV series): documentary"),
            "James Burke",
        )

        # 3. collect_newline_boundary_ids must exclude SentencePiece plain space token when encoding " \n"
        class _DummySpTokenizer:
            def __init__(self) -> None:
                self._id_to_tok = {10: " ", 11: "\u2581", 13: "\n", 14: "\n\n", 15: " \n"}
                self._vocab = {"\n": 13, "\n\n": 14}
            def convert_tokens_to_ids(self, tok: str) -> int:
                return self._vocab.get(tok, -1)
            def encode(self, txt: str, add_special_tokens: bool = False):
                if txt == " \n":
                    return [10, 13]
                if txt == "\n":
                    return [13]
                if txt == "\n\n":
                    return [14]
                return []
            def decode(self, ids, skip_special_tokens: bool = False) -> str:
                return "".join(self._id_to_tok.get(int(i), "") for i in ids)

        nl_ids = m3_mod.collect_newline_boundary_ids(_DummySpTokenizer())
        self.assertIn(13, nl_ids)
        self.assertIn(14, nl_ids)
        self.assertNotIn(10, nl_ids)
        self.assertNotIn(11, nl_ids)

        # 4. calibrate_dev_m1_tau_eu & resolve_doc_sp_labels
        params = phase3_common.load_v5_balanced_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = phase3_common.Phase2CheckpointManager(Path(tmpdir) / "cal_m1_test.json")
            dev_items = [
                {
                    "id": "d1",
                    "question": "Who directed Inception?",
                    "pool": [{"doc_id": "p1", "title": "Inception", "text": "Inception is directed by Christopher Nolan.", "score": 0.9}],
                },
                {
                    "id": "d2",
                    "question": "What is the capital of France?",
                    "pool": [{"doc_id": "p2", "title": "France", "text": "Paris is the capital of France.", "score": 0.8}],
                },
            ]
            tau = phase3_common.calibrate_dev_m1_tau_eu(dev_items, gen, params, ckpt=ckpt, model_tag="qwen3b", dataset_name="hotpotqa")
            self.assertGreaterEqual(tau, 0.05)
            self.assertLessEqual(tau, 0.90)
            self.assertTrue(ckpt.has("m1_tau_eu::qwen3b::hotpotqa::q60"))

            # Check calib_from_b0 replaces shifted b012 distribution with b0 distribution
            ev_docs = [{"doc_id": "d0", "title": "France", "text": "Paris is the capital of France.", "score": 0.9, "calibrated_reliability": 0.9}]
            ordered = [m3_mod.M3CalibratedDecoder.normalize_evidence_item(d) for d in ev_docs]
            import hashlib
            p_hash = hashlib.md5(phase3_common.build_prompt("Capital of France?", ordered, generator=gen).encode("utf-8")).hexdigest()[:16]
            ckpt.record(f"test_pfx::{p_hash}::b0.000", {"raw_answer": "Paris", "ctx_probs_by_temp": {"1.0": 0.95}, "is_faithful": True, "has_unanchored": False})
            ckpt.record(f"test_pfx::{p_hash}::b0.120", {"raw_answer": "Paris", "ctx_probs_by_temp": {"1.0": 0.42}, "is_faithful": True, "has_unanchored": False})
            res_b0 = phase3_common.run_open_decode_cached(gen, "Capital of France?", ev_docs, 0.12, params, ckpt=ckpt, cache_prefix="test_pfx", calib_from_b0=True)
            self.assertEqual(res_b0["ctx_probs_by_temp"]["1.0"], 0.95)

        self.assertEqual(
            phase3_common.resolve_doc_sp_labels(
                [{"doc_id": "kg_rel_123", "title": "Entity A"}, {"doc_id": "kg_rel_999", "title": "Entity B"}],
                ["kg_rel_123"],
            ),
            ["kg_rel_123", "Entity B"],
        )


class TestPhase3TwoBatchGroupingAndWP11(unittest.TestCase):
    """
    Validates end-to-end dry-run execution of WP6 and WP9/WP10 in two batches
    (core3 followed by extra2 with --skip-stress), automatic incremental merging
    of all 5 models, preservation of WP9 stress curves, and WP11 5-model plotting.
    """

    def test_wp6_and_wp9_wp10_core3_extra2_merge_and_wp11_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_data_p2 = tmp_root / "data_p2"
            tmp_cache_p2 = tmp_data_p2 / "caches"
            tmp_data_p3 = tmp_root / "data_p3"
            tmp_cache_p3 = tmp_data_p3 / "caches"
            tmp_res = tmp_root / "results_m3"
            for sub in ("wp6", "wp7", "wp8", "wp9", "wp10", "wp11"):
                (tmp_res / sub).mkdir(parents=True, exist_ok=True)

            try:
                with patch.object(cache_builder, "PHASE2_DATA_DIR", tmp_data_p2), \
                     patch.object(cache_builder, "PHASE2_CACHE_DIR", tmp_cache_p2), \
                     patch.object(phase3_common, "PHASE3_DATA_DIR", tmp_data_p3), \
                     patch.object(phase3_common, "PHASE3_CACHE_DIR", tmp_cache_p3):

                    # 1. WP6 Batch 1: core3
                    with patch.object(sys, "argv", [
                        "wp6_cross_family.py",
                        "--model-group", "core3",
                        "--datasets", "hotpotqa", "popqa",
                        "--scale", "smoke",
                        "--seeds", "42",
                        "--out-dir", str(tmp_res / "wp6"),
                        "--dry-run",
                    ]):
                        wp6_cross_family.main()
                    wp6_res1 = json.loads((tmp_res / "wp6" / "wp6_cross_family_matrix.json").read_text(encoding="utf-8"))
                    self.assertEqual(set(wp6_res1["matrix"].keys()), {"qwen3b", "qwen7b", "llama3_8b"})

                    # 2. WP6 Batch 2: extra2 (incremental merge without overwrite)
                    with patch.object(sys, "argv", [
                        "wp6_cross_family.py",
                        "--model-group", "extra2",
                        "--datasets", "hotpotqa", "popqa",
                        "--scale", "smoke",
                        "--seeds", "42",
                        "--out-dir", str(tmp_res / "wp6"),
                        "--dry-run",
                    ]):
                        wp6_cross_family.main()
                    wp6_res2 = json.loads((tmp_res / "wp6" / "wp6_cross_family_matrix.json").read_text(encoding="utf-8"))
                    all5_expected = {"qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"}
                    self.assertEqual(set(wp6_res2["matrix"].keys()), all5_expected)
                    for mk in all5_expected:
                        self.assertEqual(wp6_res2["matrix"][mk]["status"], "completed")

                    # 3. WP9 & WP10 Batch 1: core3
                    with patch.object(sys, "argv", [
                        "wp9_wp10_stress_efficiency.py",
                        "--model-group", "core3",
                        "--scale", "smoke",
                        "--wp9-out-dir", str(tmp_res / "wp9"),
                        "--wp10-out-dir", str(tmp_res / "wp10"),
                        "--dry-run",
                    ]):
                        wp9_wp10_stress_efficiency.main()
                    wp9_res1 = json.loads((tmp_res / "wp9" / "wp9_stress_curves.json").read_text(encoding="utf-8"))
                    wp10_res1 = json.loads((tmp_res / "wp10" / "wp10_efficiency_pareto.json").read_text(encoding="utf-8"))
                    self.assertIn("curves", wp9_res1["stress"])
                    saved_stress_curves = wp9_res1["stress"]["curves"]
                    self.assertEqual(set(wp10_res1["models"].keys()), {"qwen3b", "qwen7b", "llama3_8b"})

                    # 4. WP9 & WP10 Batch 2: extra2 with --skip-stress (incremental merge)
                    with patch.object(sys, "argv", [
                        "wp9_wp10_stress_efficiency.py",
                        "--model-group", "extra2",
                        "--skip-stress",
                        "--scale", "smoke",
                        "--wp9-out-dir", str(tmp_res / "wp9"),
                        "--wp10-out-dir", str(tmp_res / "wp10"),
                        "--dry-run",
                    ]):
                        wp9_wp10_stress_efficiency.main()
                    wp9_res2 = json.loads((tmp_res / "wp9" / "wp9_stress_curves.json").read_text(encoding="utf-8"))
                    wp10_res2 = json.loads((tmp_res / "wp10" / "wp10_efficiency_pareto.json").read_text(encoding="utf-8"))

                    # Stress curves from Batch 1 are preserved
                    self.assertEqual(wp9_res2["stress"]["curves"], saved_stress_curves)
                    # WP10 has all 5 models completed
                    self.assertEqual(set(wp10_res2["models"].keys()), all5_expected)
                    for mk in all5_expected:
                        self.assertEqual(wp10_res2["models"][mk]["status"], "completed")

                    # 5. WP11 _plot_fig4_cross_family verification with all 5 models
                    fig_out_dir = tmp_res / "wp11" / "figures"
                    fig_out_dir.mkdir(parents=True, exist_ok=True)
                    fig4_path = wp11_build_paper_package._plot_fig4_cross_family(wp6_res2, fig_out_dir)
                    self.assertIsNotNone(fig4_path)
                    self.assertTrue(fig4_path.exists())
                    self.assertGreater(fig4_path.stat().st_size, 1000)

                    # 6. WP8 & WP7 smoke runs to feed WP11 end-to-end report generation
                    with patch.object(sys, "argv", [
                        "wp8_method_validity.py",
                        "--model", "qwen3b",
                        "--scale", "smoke",
                        "--out-dir", str(tmp_res / "wp8"),
                        "--dry-run",
                    ]):
                        wp8_method_validity.main()
                    wp8_res = json.loads((tmp_res / "wp8" / "wp8_method_validity_results.json").read_text(encoding="utf-8"))
                    ckpt_wp8 = json.loads((tmp_res / "wp8" / "ckpt_wp8_qwen3b_smoke.json").read_text(encoding="utf-8"))
                    self.assertIn("calibrated_tau_eu", wp8_res["metadata"])
                    self.assertEqual(ckpt_wp8["metadata"].get("model"), "qwen3b")
                    self.assertEqual(ckpt_wp8["metadata"].get("schema_version"), "wp8_v2")
                    self.assertEqual(wp8_res["wp8_3_m3_validity"]["hotpotqa"]["dev_adaptive_tau"]["eval_split"], "test")
                    self.assertIn("calibrated_tau_eu", wp6_res2["matrix"]["qwen3b"])

                    with patch.object(sys, "argv", [
                        "wp7_cross_dataset.py",
                        "--model", "llama3_8b",
                        "--datasets", "popqa", "hotpotqa",
                        "--scale", "smoke",
                        "--out-dir", str(tmp_res / "wp7"),
                        "--dry-run",
                    ]):
                        wp7_cross_dataset.main()

                    with patch.object(sys, "argv", [
                        "wp11_build_paper_package.py",
                        "--wp6-dir", str(tmp_res / "wp6"),
                        "--wp7-dir", str(tmp_res / "wp7"),
                        "--wp8-dir", str(tmp_res / "wp8"),
                        "--wp9-dir", str(tmp_res / "wp9"),
                        "--wp10-dir", str(tmp_res / "wp10"),
                        "--out-dir", str(tmp_res / "wp11"),
                    ]):
                        wp11_build_paper_package.main()

                    for fig_name in (
                        "fig2_wp8_m1_uncertainty.png",
                        "fig3_wp8_m2_optimization.png",
                        "fig4_wp6_cross_family_main.png",
                        "fig5_wp8_calibration_rc.png",
                        "fig6_wp9_stress_curves.png",
                        "fig7_wp10_quality_latency_vram.png",
                    ):
                        fig_p = tmp_res / "wp11" / "figures" / fig_name
                        self.assertTrue(fig_p.exists() and fig_p.stat().st_size > 1000, f"Missing figure {fig_p}")
                    self.assertTrue((tmp_res / "wp11" / "MILESTONE3_EXPERIMENT_REPORT.md").exists())
                    self.assertTrue((tmp_res / "wp11" / "tables_phase3.tex").exists())

                    # 7. Merge helper verification combining two separate kernel output dirs -> all 5 models
                    k1_dir = tmp_root / "kernel1_out"
                    k2_dir = tmp_root / "kernel2_out"
                    (k1_dir / "WP6_CrossFamily").mkdir(parents=True, exist_ok=True)
                    (k1_dir / "WP10_Efficiency").mkdir(parents=True, exist_ok=True)
                    (k1_dir / "WP8_MethodValidity").mkdir(parents=True, exist_ok=True)
                    (k2_dir / "WP6_CrossFamily").mkdir(parents=True, exist_ok=True)
                    (k2_dir / "WP10_Efficiency").mkdir(parents=True, exist_ok=True)

                    wp6_k1 = json.loads(json.dumps(wp6_res2))
                    wp6_k1["matrix"] = {m: wp6_res2["matrix"][m] for m in ("qwen3b", "qwen7b", "llama3_8b")}
                    wp6_k2 = json.loads(json.dumps(wp6_res2))
                    wp6_k2["matrix"] = {m: wp6_res2["matrix"][m] for m in ("mistral7b", "gemma2")}
                    (k1_dir / "WP6_CrossFamily" / "wp6_cross_family_matrix.json").write_text(json.dumps(wp6_k1), encoding="utf-8")
                    (k2_dir / "WP6_CrossFamily" / "wp6_cross_family_matrix.json").write_text(json.dumps(wp6_k2), encoding="utf-8")

                    wp10_k1 = json.loads(json.dumps(wp10_res2))
                    wp10_k1["models"] = {m: wp10_res2["models"][m] for m in ("qwen3b", "qwen7b", "llama3_8b")}
                    wp10_k2 = json.loads(json.dumps(wp10_res2))
                    wp10_k2["models"] = {m: wp10_res2["models"][m] for m in ("mistral7b", "gemma2")}
                    (k1_dir / "WP10_Efficiency" / "wp10_efficiency_pareto.json").write_text(json.dumps(wp10_k1), encoding="utf-8")
                    (k2_dir / "WP10_Efficiency" / "wp10_efficiency_pareto.json").write_text(json.dumps(wp10_k2), encoding="utf-8")
                    (k1_dir / "WP8_MethodValidity" / "wp8_method_validity_results.json").write_text(
                        (tmp_res / "wp8" / "wp8_method_validity_results.json").read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )

                    merged_out = tmp_root / "merged_all5"
                    m_info = merge_phase3_kernels.merge_kernel_outputs([k1_dir, k2_dir], merged_out)
                    self.assertEqual(set(m_info["wp6_models"]), all5_expected)
                    self.assertEqual(set(m_info["wp10_models"]), all5_expected)
                    self.assertTrue((merged_out / "WP11_PaperPackage" / "figures" / "fig4_wp6_cross_family_main.png").exists())
            finally:
                common.close_phase2_log_handlers("phase2.")
                common.close_phase2_log_handlers("phase3.")


if __name__ == "__main__":
    unittest.main()
