import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.plot_ablation_charts import (
    _compute_paired_bootstrap_stats,
    generate_tables_latex,
    plot_fig4_m3_calibration_and_risk_coverage,
)
from scripts.run_overnight_ablation import select_popqa_m2_evidence
from scripts.run_parameter_grid_search import _fit_platt_regularized
from src.m1_adaptive_retriever import MinHashGraphExpander, UncertaintyDecoupler
from src.m2_evidence_optimizer import (
    EvidenceCandidate,
    MSTEvidenceOrganizer,
    TwoStageM2Optimizer,
)


class TestV4M2ForwardOrderAndMinDocs(unittest.TestCase):
    """Validates V4 M2 fixes: Forward Root-First MST-DFS and multi-hop min_k=2 floor."""

    def test_mst_default_is_forward_root_first(self):
        # Graph chain: q_anchor (high query entity overlap) -- bridge -- leaf_distractor
        c_anchor = EvidenceCandidate(
            doc_id="d_anchor",
            text="Scott Derrickson directed the 2016 film.",
            score=0.90,
            calibrated_reliability=0.80,
            entities={"scott derrickson", "doctor strange"},
        )
        c_bridge = EvidenceCandidate(
            doc_id="d_bridge",
            text="Doctor Strange premiered in Hong Kong before October 2016.",
            score=0.82,
            calibrated_reliability=0.85,
            entities={"doctor strange", "hong kong"},
        )
        c_leaf = EvidenceCandidate(
            doc_id="d_leaf",
            text="Hong Kong cinema hosts major international premieres.",
            score=0.60,
            calibrated_reliability=0.40,
            entities={"hong kong"},
        )
        explicit_edges = {
            ("d_anchor", "d_bridge"): 0.90,
            ("d_bridge", "d_leaf"): 0.75,
        }

        organizer_default = MSTEvidenceOrganizer()
        self.assertFalse(organizer_default.recency_aware)

        res_fwd = organizer_default.organize(
            selected_docs=[c_leaf, c_bridge, c_anchor],
            explicit_edges=explicit_edges,
            query_entities={"scott derrickson"},
        )
        # Forward Root-First places query anchor FIRST (index 0) and peripheral leaf LAST
        self.assertEqual(res_fwd.roots[0], "d_anchor")
        self.assertEqual(res_fwd.ordered_ids[0], "d_anchor")
        self.assertEqual(res_fwd.ordered_ids[-1], "d_leaf")

        res_rev = organizer_default.organize(
            selected_docs=[c_leaf, c_bridge, c_anchor],
            explicit_edges=explicit_edges,
            recency_aware=True,
            query_entities={"scott derrickson"},
        )
        # Recency reversal places peripheral leaf first when explicitly requested
        self.assertEqual(res_rev.ordered_ids[0], "d_leaf")
        self.assertEqual(res_rev.ordered_ids[-1], "d_anchor")

    def test_min_k_2_prevents_multihop_k1_overpruning(self):
        # Negative calibrated_reliability after large offset subtracting baseline cost:
        # doc1 covers Q entity, doc2 is needed 2-hop bridge with negative reliability (-0.12).
        c1 = EvidenceCandidate(
            doc_id="hop1",
            text="The Oberoi family owns Oberoi Hotels.",
            score=0.85,
            calibrated_reliability=0.45,
            entities={"oberoi family", "oberoi hotels"},
        )
        c2 = EvidenceCandidate(
            doc_id="hop2",
            text="Oberoi Hotels is headquartered in Delhi, India.",
            score=0.55,
            calibrated_reliability=-0.12,
            entities={"oberoi hotels", "delhi"},
        )
        c3 = EvidenceCandidate(
            doc_id="noise",
            text="Unrelated tourist travel blog.",
            score=0.30,
            calibrated_reliability=-0.30,
            entities={"travel"},
        )

        for algo in ("exact_enum", "greedy"):
            opt = TwoStageM2Optimizer(
                lambda_coverage=0.5,
                lambda_complementarity=0.3,
                lambda_redundancy=0.4,
                lambda_conflict=1.0,
                selection_algorithm=algo,
            )
            self.assertFalse(opt.organizer.recency_aware)

            out_min1 = opt.optimize(
                candidates=[c1, c2, c3],
                max_k=5,
                query_entities={"oberoi family"},
                calibrate=False,
                min_k=1,
            )
            # Negative second hop causes min_k=1 to collapse to single doc
            self.assertEqual(len(out_min1.selected_docs), 1)

            out_min2 = opt.optimize(
                candidates=[c1, c2, c3],
                max_k=5,
                query_entities={"oberoi family"},
                calibrate=False,
                min_k=2,
            )
            # min_k=2 preserves the 2-hop chain while still pruning irrelevant 3rd distractor
            self.assertEqual(len(out_min2.selected_docs), 2)
            self.assertEqual({d.doc_id for d in out_min2.selected_docs}, {"hop1", "hop2"})


class TestV4PlattShrinkageAndM1Ranking(unittest.TestCase):
    """Validates V4 Platt zero-intercept shrinkage and MinHash LSH vs entity fallback ranking."""

    def test_platt_zero_intercept_prevents_base_rate_shift(self):
        # Reproduce V3 dev shift scenario (N=45 hard dev slice: mean conf ~0.68, pos rate ~0.42)
        raw_confs = [0.68] * 26 + [0.64] * 19
        targets = [1.0] * 14 + [0.0] * 12 + [1.0] * 5 + [0.0] * 14
        a, b = _fit_platt_regularized(raw_confs, targets)
        self.assertGreaterEqual(b, -0.28)
        self.assertLessEqual(b, 0.35)
        self.assertGreaterEqual(a, 0.55)
        self.assertLessEqual(a, 2.20)

    def test_lsh_collision_outranks_entity_bridge_fallback(self):
        anchor_text = "Givron archival dossier verified encyclopedia factual record linkage."
        corpus = [
            {"id": "seed_bg", "title": "Givron overview", "text": f"{anchor_text} Survey of Givron country context.", "entities": ["Givron"]},
            {"id": "gold_rel", "title": "Givron Fact Sheet", "text": f"{anchor_text} Sovereign territory of Givron is France.", "entities": ["Givron", "France"]},
            {"id": "fallback_noise", "title": "Givron index", "text": "Unrelated external directory mentioning random keywords.", "entities": ["Givron"]},
        ]
        expander = MinHashGraphExpander(num_perm=64, num_bands=32, shingle_k=3, similarity_threshold=0.15, max_candidate_pool=5)
        expander.build_corpus_index(corpus)
        res = expander.expand(
            query="In what country is Givron?",
            initial_docs=[corpus[0]],
            epistemic_uncertainty=0.9,
            tau_eu=0.2,
            gold_ids={"gold_rel"},
        )
        ids = [d["id"] for d in res.expanded_candidates]
        self.assertIn("gold_rel", ids)
        self.assertIn("fallback_noise", ids)
        self.assertLess(ids.index("gold_rel"), ids.index("fallback_noise"))


class TestV4ReportingConsistency(unittest.TestCase):
    """Validates V4 Table 2 metrics and Table 3 exact signed Delta vs DFS."""

    def test_table3_uses_signed_mean_delta_and_table2_includes_calibration(self):
        step1_data = {
            "3B": {"em": 40.0, "f1": 50.0, "latency": 1.0, "peak_res": 3000.0, "headroom": 5000.0},
            "7B": {"em": 60.0, "f1": 70.0, "latency": 1.5, "peak_res": 5200.0, "headroom": 2800.0},
        }
        ablation_data = {
            "B0_Standard_RAG": {
                "name": "B0: Standard RAG", "EM": 50.0, "F1": 64.68,
                "sp_prec": 60.0, "sp_rec": 70.0, "sp_f1": 64.6,
                "token_saving": 0.0, "ece": 0.11, "auroc": 0.65,
                "coverage": 100.0, "selective_risk": 35.0, "latency_s": 0.5,
            },
            "Full_UGraphRAG": {
                "name": "Full U-GraphRAG", "EM": 54.0, "F1": 67.20,
                "sp_prec": 72.0, "sp_rec": 71.0, "sp_f1": 71.5,
                "token_saving": 28.0, "ece": 0.06, "auroc": 0.72,
                "coverage": 92.0, "selective_risk": 24.0, "latency_s": 0.9,
            },
        }
        perm_data = {
            "dfs": {"mean": 66.61, "intra_sample_std": 0.0, "cross_sample_std": 39.2},
            "reverse": {"mean": 63.19, "intra_sample_std": 0.0, "cross_sample_std": 41.3},
            "shuffles": {"mean": 64.46, "intra_sample_std": 2.1, "cross_sample_std": 40.1},
            "max_drop": 13.59,
        }
        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp) / "results"
            out_dir = Path(tmp) / "figures"
            res_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (res_dir / "step3_hotpotqa_ablations.json").write_text(json.dumps(ablation_data), encoding="utf-8")
            (res_dir / "step4_m2_permutation.json").write_text(json.dumps(perm_data), encoding="utf-8")
            generate_tables_latex(res_dir, out_dir)
            md_txt = (out_dir / "TABLES_SUMMARY.md").read_text(encoding="utf-8")
            tex_txt = (out_dir / "tables_latex.tex").read_text(encoding="utf-8")

            # Table 3 must display -3.42 (63.19 - 66.61), NEVER -13.59 (worst per-sample max_drop)
            self.assertIn("-3.42%", md_txt)
            self.assertIn("-3.42", tex_txt)
            self.assertNotIn("-13.59", md_txt)
            self.assertNotIn("-13.59", tex_txt)

            # Table 2 must contain AUROC, Coverage, Selective Risk
            self.assertIn("AUROC", md_txt)
            self.assertIn("0.720", md_txt)
            self.assertIn("92.0%", md_txt)


class TestV5PopQAM2FixAndStatisticalReporting(unittest.TestCase):
    """Validates V5 PopQA zero-leakage M2 selection fix, Bootstrap 95% CI, and Fig 4 rendering."""

    def test_select_popqa_m2_evidence_rescues_opaque_factual_leaf_without_leakage(self):
        # Use completely opaque IDs (no 'kg_rel' string anywhere) to prove zero label/ID leakage
        d0_docs = [
            {
                "id": "hub_overview_a",
                "title": "George Alexander Eugene Douglas Haig, 2nd Earl Haig - mother",
                "text": "George Haig factual dossier canonical hub. Overview examining the mother of George Haig.",
                "entities": ["George Haig"],
                "score": 0.90,
            },
            {
                "id": "distractor_lexicon_b",
                "title": "George Haig - lexicon (mother)",
                "text": "George Haig mother directory: catalog entry and archives.",
                "entities": ["George Haig"],
                "score": 0.84,
            },
        ]
        # Structural LSH bridge has stale corpus score=0.65 + graph_similarity=0.42
        lsh_factual = {
            "id": "opaque_factual_passage_77",
            "title": "George Haig - Structured Infobox",
            "text": "George Haig factual dossier canonical hub. Recorded maternal lineage for George Haig traces to Dorothy Maud Vivian.",
            "entities": ["George Haig", "Dorothy Maud Vivian"],
            "score": 0.65,
            "graph_similarity": 0.42,
        }
        outer_distractor = {
            "id": "distractor_chronicle_c",
            "title": "George Haig chronicle",
            "text": "George Haig chronicle without direct answer.",
            "entities": ["George Haig"],
            "score": 0.72,
        }
        expanded = list(d0_docs) + [lsh_factual]
        ranked_pool = list(d0_docs) + [outer_distractor]

        m2_opt = TwoStageM2Optimizer(
            lambda_coverage=0.8,
            lambda_complementarity=0.2,
            lambda_redundancy=0.4,
            lambda_conflict=1.0,
            selection_algorithm="exact_enum",
            recency_aware=False,
        )
        decoupler = UncertaintyDecoupler()

        def mock_model_fn(question, ctx):
            joined = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in ctx)
            if "Dorothy Maud Vivian" in joined:
                # Sharp peaked distribution when true factual leaf is present
                return {"Dorothy": 0.88, "Other": 0.06, "Unknown": 0.06}
            # Flat uncertain distribution on hubs/distractors alone
            return {f"tok_{i}": 1.0 / 12.0 for i in range(12)}

        d0_metrics = decoupler.decompose("Who was George Haig's mother?", d0_docs, model_fn=mock_model_fn)
        selected = select_popqa_m2_evidence(
            question="Who was George Haig's mother?",
            subj="George Haig",
            s_wiki="George Haig",
            d0_docs=d0_docs,
            expanded_candidates=expanded,
            ranked_pool=ranked_pool,
            d0_epistemic_uncertainty=d0_metrics.epistemic_uncertainty,
            m2_opt=m2_opt,
            decoupler=decoupler,
            model_fn=mock_model_fn,
            pool_limit=8,
            m2_gamma_1=0.8,
            m2_gamma_2=0.2,
            m2_offset=0.22,
            max_k=3,
            min_k=2,
        )
        selected_ids = [d["id"] for d in selected]
        self.assertIn("opaque_factual_passage_77", selected_ids)
        self.assertGreaterEqual(len(selected), 2)
        factual_item = next(d for d in selected if d["id"] == "opaque_factual_passage_77")
        # Calibrated reliability must be restored above offset shift and epistemic reduction > 0
        self.assertGreater(factual_item["calibrated_reliability"], 0.5)
        self.assertGreater(factual_item["epistemic_reduction"], 0.05)

    def test_bootstrap_stats_and_fig4_rendering(self):
        # Missing values -> graceful None (preserves summary-only callers)
        self.assertIsNone(_compute_paired_bootstrap_stats({"dfs": {"mean": 66.0}, "reverse": {"mean": 63.0}, "shuffles": {"mean": 64.0}}))

        # With per-sample arrays -> valid 95% CI and sensitive subset counts
        n = 20
        dfs_vals = [100.0 if i < 14 else 0.0 for i in range(n)]
        rev_vals = [100.0 if i < 12 else 0.0 for i in range(n)]
        shuf_vals = [80.0 if i < 14 else 0.0 for i in range(n)]
        boot = _compute_paired_bootstrap_stats(
            {"dfs": {"values": dfs_vals}, "reverse": {"values": rev_vals}, "shuffles": {"values": shuf_vals}},
            n_boot=500,
            seed=42,
        )
        self.assertIsNotNone(boot)
        self.assertEqual(boot["dfs_vs_reverse"]["wins"], 2)
        self.assertEqual(boot["dfs_vs_reverse"]["ties"], 18)

        # Fig 4 renders cleanly from synthetic per-sample records
        recs_full = [{"confidence": 0.35 + 0.03 * i, "f1": 1.0 if i >= 6 else 0.0, "em": 1.0 if i >= 8 else 0.0} for i in range(20)]
        recs_b0 = [{"confidence": 0.65 + 0.01 * i, "f1": 1.0 if i % 2 == 0 else 0.0, "em": 1.0 if i % 3 == 0 else 0.0} for i in range(20)]
        payload = {
            "ablations_7b": {
                "Full_UGraphRAG": {"ece": 0.09, "brier_score": 0.18, "auroc": 0.80, "coverage": 95.0, "selective_risk": 25.0, "sample_records": recs_full},
                "B0_Standard_RAG": {"ece": 0.10, "brier_score": 0.22, "auroc": 0.52, "coverage": 100.0, "selective_risk": 50.0, "sample_records": recs_b0},
                "M2_Full": {"ece": 0.14, "brier_score": 0.23, "auroc": 0.55, "coverage": 100.0, "selective_risk": 45.0, "sample_records": recs_b0},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            rdir = Path(tmp) / "results"
            fdir = Path(tmp) / "figures"
            rdir.mkdir(parents=True, exist_ok=True)
            fdir.mkdir(parents=True, exist_ok=True)
            (rdir / "step3_step4_ablation_hotpot100.json").write_text(json.dumps(payload), encoding="utf-8")
            out_fig4 = plot_fig4_m3_calibration_and_risk_coverage(rdir, fdir)
            self.assertIsNotNone(out_fig4)
            self.assertTrue(out_fig4.exists())


if __name__ == "__main__":
    unittest.main()
