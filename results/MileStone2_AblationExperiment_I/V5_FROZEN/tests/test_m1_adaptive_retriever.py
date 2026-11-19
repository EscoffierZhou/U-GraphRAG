"""
test_m1_adaptive_retriever.py
=============================
Comprehensive Pure-CPU Unit Tests for M1 Adaptive Retriever:
1. Uncertainty non-negative decomposition (H >= 0, U_AU >= 0, U_EU >= 0)
2. Jensen-Shannon divergence (JS_Div) symmetry, non-negativity, and bounds
3. Marginal epistemic reduction Delta U_EU(d_i) >= 0 and first-step distribution
4. MinHash signatures and Jaccard collision graph construction
5. Bloom Filter insertion, membership query, and false positive characteristics
6. Explicit Bloom Filter community pruning and recall drop tracking (Recall Drop under Bloom Pruning)
7. Adaptive gating router: early exit, max capacity <= 15, and token saving rate calculation
8. Offline calibration probing and parameter persistence (tau_EU, aleatoric_scale, etc.)
9. Full integration compatibility with M2 EvidenceCandidate specification
"""

import sys
import os
import math
import tempfile
import unittest
from typing import List, Dict, Any, Set

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.m1_adaptive_retriever import (
    compute_entropy,
    js_divergence,
    minhash_signature,
    estimate_jaccard_similarity,
    BloomFilter,
    MinHashGraphExpander,
    UncertaintyDecoupler,
    M1AdaptiveRetriever,
    UncertaintyMetrics,
    ExpansionResult,
)


class TestUncertaintyMathematicalProperties(unittest.TestCase):
    """Verifies fundamental mathematical axioms of entropy and uncertainty decomposition."""

    def test_entropy_properties(self):
        # 1. Deterministic peak distribution -> entropy must be 0
        p_peak = {"a": 1.0, "b": 0.0, "c": 0.0}
        self.assertAlmostEqual(compute_entropy(p_peak), 0.0, places=5)

        # 2. Uniform 4-way distribution -> entropy must be ln(4)
        p_uniform = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        self.assertAlmostEqual(compute_entropy(p_uniform), math.log(4), places=5)

        # 3. Empty distribution -> 0.0
        self.assertEqual(compute_entropy({}), 0.0)

        # 4. Unnormalized distribution should auto-normalize
        p_unnorm = {"x": 2.0, "y": 2.0}
        self.assertAlmostEqual(compute_entropy(p_unnorm), math.log(2), places=5)

    def test_js_divergence_properties(self):
        p = {"token_a": 0.7, "token_b": 0.3}
        q = {"token_a": 0.2, "token_b": 0.8}
        r = {"token_c": 1.0}  # disjoint support

        # Identity: D_JS(p, p) == 0
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=6)

        # Symmetry: D_JS(p, q) == D_JS(q, p)
        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)
        self.assertAlmostEqual(js_pq, js_qp, places=6)
        self.assertGreater(js_pq, 0.0)

        # Non-negativity and upper bound: 0 <= D_JS <= ln(2) approx 0.693147
        self.assertGreaterEqual(js_pq, 0.0)
        self.assertLessEqual(js_pq, math.log(2.0) + 1e-6)

        # Disjoint support divergence: max divergence is ln(2)
        js_disjoint = js_divergence(p, r)
        self.assertAlmostEqual(js_disjoint, math.log(2.0), places=4)

        # Empty distribution fallback
        self.assertEqual(js_divergence({}, {}), 0.0)

    def test_non_negative_uncertainty_decomposition(self):
        """Validates that U_EU = max(0, H_total - U_AU) is strictly non-negative."""
        decoupler = UncertaintyDecoupler(tau_eu=0.45, aleatoric_scale=1.0)

        # Test case 1: Standard query and passages
        query = "What is the capital of France?"
        docs = [
            {"id": "doc1", "text": "Paris is the capital and most populous city of France."},
            {"id": "doc2", "text": "Lyon is a major city in France."},
        ]
        metrics = decoupler.compute_uncertainty(query, docs)

        self.assertGreaterEqual(metrics.total_entropy, 0.0)
        self.assertGreaterEqual(metrics.aleatoric_uncertainty, 0.0)
        self.assertGreaterEqual(metrics.epistemic_uncertainty, 0.0)
        self.assertEqual(
            metrics.epistemic_uncertainty,
            max(0.0, metrics.total_entropy - metrics.aleatoric_uncertainty),
        )

        # Test case 2: Simulated high aleatoric noise (aleatoric_scale >> 1)
        decoupler_high_noise = UncertaintyDecoupler(tau_eu=0.45, aleatoric_scale=5.0)
        metrics_clamped = decoupler_high_noise.compute_uncertainty(query, docs)
        # Even if AU > H_total, EU must never be negative
        self.assertGreaterEqual(metrics_clamped.epistemic_uncertainty, 0.0)

    def test_marginal_epistemic_reduction(self):
        """Tests marginal epistemic reduction Delta U_EU(d_i) >= 0 and downstream M2 formatting."""
        decoupler = UncertaintyDecoupler(tau_eu=0.40)
        query = "Which director won an Academy Award for Inception?"
        initial_docs = [
            {"id": "d1", "text": "Christopher Nolan directed Inception, released in 2010."}
        ]
        candidate_docs = [
            {"id": "d2", "text": "Inception won four Academy Awards at the 83rd Academy Awards."},
            {"id": "d3", "text": "Irrelevant document about deep sea exploration and corals."},
        ]

        gains = decoupler.compute_marginal_epistemic_reduction(
            query=query,
            initial_docs=initial_docs,
            candidate_docs=candidate_docs,
        )

        self.assertIn("d2", gains)
        self.assertIn("d3", gains)
        for doc_id, gain in gains.items():
            self.assertGreaterEqual(gain, 0.0, f"Gain for {doc_id} must be non-negative")


class TestBloomFilterAndRecallDrop(unittest.TestCase):
    """Tests BloomFilter bit-array behavior, entity hashing, and Recall Drop under pruning."""

    def test_bloom_filter_membership(self):
        bf = BloomFilter(expected_elements=50, false_positive_rate=0.01)
        test_entities = ["Ed Wood", "Scott Derrickson", "Doctor Strange", "Plan 9"]

        for ent in test_entities:
            bf.add(ent)

        # 1. Zero False Negative Guarantee
        for ent in test_entities:
            self.assertTrue(ent in bf, f"Expected {ent} to be present in BloomFilter")

        # 2. Non-existent entity usually not found
        self.assertFalse("Completely Unrelated Novel Entity 123456" in bf)

    def test_bloom_filter_recall_drop_tracking(self):
        """
        Explicitly verifies the tracking of Gold Passage Recall Drop under Bloom Pruning.
        Ensures recall_before, recall_after, and recall_drop are correctly computed.
        """
        expander = MinHashGraphExpander()

        # Suppose candidate pool has 4 documents
        candidates = [
            {"id": "gold_1", "text": "Information about Scott Derrickson and Sinister.", "entities": ["Scott Derrickson", "Sinister"]},
            {"id": "gold_2", "text": "Ed Wood produced Plan 9 in Hollywood.", "entities": ["Ed Wood", "Plan 9"]},
            {"id": "irrelevant_3", "text": "Bananas are yellow tropical fruits.", "entities": ["Bananas", "Fruit"]},
            {"id": "irrelevant_4", "text": "Quantum mechanics describes microscopic particles.", "entities": ["Quantum"]},
        ]
        gold_ids = {"gold_1", "gold_2"}

        # Query filter only contains entities for gold_1 and irrelevant_3
        # gold_2 will be pruned because its entities ("Ed Wood", "Plan 9") are not in the query bloom filter
        bf = BloomFilter(expected_elements=10)
        bf.add("Scott Derrickson")
        bf.add("Sinister")

        pruned, stats = expander.prune_candidates(candidates, bf, gold_ids=gold_ids)

        # Before pruning: both gold_1 and gold_2 were in candidates -> recall_before = 2 / 2 = 1.0
        self.assertEqual(stats["before_pruning_count"], 4)
        self.assertEqual(stats["before_pruning_gold_count"], 2)
        self.assertAlmostEqual(stats["recall_before"], 1.0, places=4)

        # After pruning: only gold_1 survived (gold_2 had no overlapping entities)
        # -> recall_after = 1 / 2 = 0.5
        self.assertEqual(stats["after_pruning_gold_count"], 1)
        self.assertAlmostEqual(stats["recall_after"], 0.5, places=4)

        # Recall drop must be strictly 0.5
        self.assertAlmostEqual(stats["recall_drop"], 0.5, places=4)
        self.assertGreater(stats["recall_drop"], 0.0)


class TestMinHashGraphAndAdaptiveRouting(unittest.TestCase):
    """Tests MinHash indexing, hash collision graphs, early exit, and token saving rate."""

    def setUp(self):
        self.corpus = [
            {"id": "c1", "text": "Scott Derrickson directed Doctor Strange in the Marvel Cinematic Universe.", "entities": ["Scott Derrickson", "Doctor Strange"]},
            {"id": "c2", "text": "Ed Wood directed cult sci-fi films including Plan 9 from Outer Space.", "entities": ["Ed Wood", "Plan 9"]},
            {"id": "c3", "text": "Benedict Cumberbatch starred as Doctor Strange directed by Scott Derrickson.", "entities": ["Benedict Cumberbatch", "Doctor Strange", "Scott Derrickson"]},
            {"id": "c4", "text": "Plan 9 from Outer Space is often called the worst movie ever made.", "entities": ["Plan 9", "Outer Space"]},
            {"id": "c5", "text": "Marvel Studios produces superhero blockbusters in Hollywood.", "entities": ["Marvel Studios", "Hollywood"]},
            {"id": "c6", "text": "Bela Lugosi appeared in several movies directed by Ed Wood.", "entities": ["Bela Lugosi", "Ed Wood"]},
        ]
        self.expander = MinHashGraphExpander(num_perm=64, num_bands=16, jaccard_threshold=0.1)
        self.expander.build_corpus_index(self.corpus)

    def test_minhash_signature_and_jaccard(self):
        sig1 = minhash_signature("Marvel Studios produces superhero blockbusters in Hollywood.", num_perm=64)
        sig2 = minhash_signature("Marvel Studios produces superhero blockbusters in Hollywood.", num_perm=64)
        sig3 = minhash_signature("Completely unrelated botanical plant biology text.", num_perm=64)

        # Identical text -> Jaccard == 1.0
        self.assertAlmostEqual(estimate_jaccard_similarity(sig1, sig2), 1.0, places=4)
        # Disjoint text -> Jaccard < 0.2
        self.assertLess(estimate_jaccard_similarity(sig1, sig3), 0.2)

    def test_adaptive_routing_early_exit(self):
        """When U_EU <= tau_EU, Early Exit triggers: retain D_0, record token_saving_rate."""
        initial_docs = [self.corpus[0], self.corpus[1]]  # 2 docs
        low_eu = 0.20
        tau_eu = 0.50  # low_eu <= tau_eu -> early exit

        result = self.expander.expand(
            query="Who directed Doctor Strange?",
            initial_docs=initial_docs,
            epistemic_uncertainty=low_eu,
            tau_eu=tau_eu,
            max_candidates=15,
        )

        self.assertTrue(result.early_exit)
        self.assertEqual(len(result.expanded_candidates), 2)
        # Token saving rate = (15 - 2) / 15 = 13 / 15 approx 0.8667
        expected_saving = (15 - 2) / 15.0
        self.assertAlmostEqual(result.token_saving_rate, expected_saving, places=4)
        self.assertIn("Early Exit triggered", result.routing_decision)

    def test_adaptive_routing_expansion_bounded(self):
        """When U_EU > tau_EU, expands along MinHash graph, strictly bounded by max_candidates <= 15."""
        initial_docs = [self.corpus[0]]  # Doctor Strange / Scott Derrickson
        high_eu = 0.85
        tau_eu = 0.40  # high_eu > tau_eu -> expand

        result = self.expander.expand(
            query="Scott Derrickson Doctor Strange Benedict Cumberbatch",
            initial_docs=initial_docs,
            epistemic_uncertainty=high_eu,
            tau_eu=tau_eu,
            max_candidates=15,
        )

        self.assertFalse(result.early_exit)
        # Expanded candidates should contain seed doc plus connected neighbors (e.g. c3)
        self.assertGreaterEqual(len(result.expanded_candidates), 1)
        self.assertLessEqual(len(result.expanded_candidates), 15)
        # Token saving rate should be non-negative and properly tracked
        self.assertGreaterEqual(result.token_saving_rate, 0.0)


class TestOfflineCalibrationAndPersistence(unittest.TestCase):
    """Tests offline calibration probing and serialization to JSON."""

    def test_calibration_roundtrip(self):
        decoupler = UncertaintyDecoupler()

        # Synthetic calibration set
        calibration_samples = [
            {
                "question": "What country is Ed Wood from?",
                "passages": ["Ed Wood was an American filmmaker born in Poughkeepsie."],
            },
            {
                "question": "Who composed the music for Inception?",
                "passages": ["Hans Zimmer composed the score for the film Inception."],
            },
            {
                "question": "Ambiguous query with multiple candidate meanings?",
                "passages": ["Text with very conflicting or diverse vocabulary."],
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            calib_res = decoupler.calibrate_offline(
                calibration_samples=calibration_samples,
                save_path=tmp_path,
            )

            self.assertIn("tau_eu", calib_res)
            self.assertIn("aleatoric_scale", calib_res)
            self.assertIn("attention_layer_weights", calib_res)
            self.assertTrue(decoupler.is_calibrated)

            # Reload into a fresh decoupler instance
            new_decoupler = UncertaintyDecoupler(calibration_path=tmp_path)
            self.assertTrue(new_decoupler.is_calibrated)
            self.assertAlmostEqual(new_decoupler.tau_eu, calib_res["tau_eu"], places=5)
            self.assertAlmostEqual(new_decoupler.aleatoric_scale, calib_res["aleatoric_scale"], places=5)
            self.assertEqual(
                new_decoupler.attention_layer_weights,
                calib_res["attention_layer_weights"],
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestIntegrationM2Compatibility(unittest.TestCase):
    """Verifies that M1 outputs can be seamlessly ingested by M2 EvidenceCandidate."""

    def test_m2_candidate_conversion(self):
        decoupler = UncertaintyDecoupler(tau_eu=0.1)  # force expansion
        expander = MinHashGraphExpander()
        corpus = [
            {"id": "doc_A", "text": "Alpha statement with reliable evidence.", "score": 0.95, "entities": ["Alpha"]},
            {"id": "doc_B", "text": "Beta statement confirming Alpha.", "score": 0.90, "entities": ["Beta", "Alpha"]},
        ]
        expander.build_corpus_index(corpus)

        retriever = M1AdaptiveRetriever(decoupler=decoupler, expander=expander)
        res = retriever.retrieve_and_expand(
            query="Alpha Beta relationship",
            initial_docs=[corpus[0]],
            corpus_docs=corpus,
        )

        self.assertTrue(hasattr(res, "m2_candidates"))
        m2_cands = res.m2_candidates
        self.assertGreaterEqual(len(m2_cands), 1)

        # Check required fields for M2 evidence optimizer
        for cand in m2_cands:
            self.assertIn("doc_id", cand)
            self.assertIn("text", cand)
            self.assertIn("score", cand)
            self.assertIn("entities", cand)
            self.assertIn("aleatoric_uncertainty", cand)
            self.assertIn("epistemic_reduction", cand)
            self.assertGreaterEqual(cand["epistemic_reduction"], 0.0)

        # Check pairwise JS divergence compatibility between first-step distributions
        per_doc_dists = res.per_doc_distributions
        if len(per_doc_dists) >= 2:
            d_ids = list(per_doc_dists.keys())
            js_div = js_divergence(per_doc_dists[d_ids[0]], per_doc_dists[d_ids[1]])
            self.assertGreaterEqual(js_div, 0.0)
            self.assertLessEqual(js_div, math.log(2.0) + 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
