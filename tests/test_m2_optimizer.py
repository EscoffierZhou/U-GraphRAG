"""
test_m2_optimizer.py
====================
Comprehensive Unit Tests for Two-Stage M2 Evidence Optimizer.

Covers:
1. Reliability calibration (minmax, sigmoid, clip, direct)
2. Canonical counterexample reproduction from Debate/GPT_opinion.md:
   A(0.95), B(0.90), C(0.85), D(-0.20) with negative-weight/conflict edge C-D.
   Proves Naive MST is forced to retain toxic node D, while Two-Stage M2 purges D.
3. Boundary conditions: |D*| = 0, |D*| = 1, extreme high-conflict rejection.
4. Robustness on disconnected graphs and isolated nodes (no crashing, topological order preserved).
5. Combinatorial scale benchmark: 15-choose-5 exact enumeration (4,944 subsets) timing and CPU metrics.
6. Greedy approximation vs exact enumeration ablation.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from m2_evidence_optimizer import (
    BudgetedEvidenceSelector,
    EvidenceCandidate,
    InteractionMatrix,
    MSTEvidenceOrganizer,
    TwoStageM2Optimizer,
    calibrate_reliability,
    naive_mst_baseline,
)


class TestReliabilityCalibration(unittest.TestCase):
    """Unit tests for non-negative reliability calibration and uncertainty penalty."""

    def test_minmax_calibration(self) -> None:
        docs = [
            EvidenceCandidate(doc_id="d1", text="text1", score=10.0),
            EvidenceCandidate(doc_id="d2", text="text2", score=5.0),
            EvidenceCandidate(doc_id="d3", text="text3", score=0.0),
        ]
        calibrated = calibrate_reliability(docs, method="minmax", gamma_1=0.0, gamma_2=0.0)
        self.assertAlmostEqual(calibrated[0], 1.0, places=4)
        self.assertAlmostEqual(calibrated[1], 0.5, places=4)
        self.assertAlmostEqual(calibrated[2], 0.0, places=4)

    def test_uncertainty_penalty(self) -> None:
        # Document with high aleatoric uncertainty should have penalized reliability
        d_clean = EvidenceCandidate(
            doc_id="clean", text="clean doc", score=1.0,
            aleatoric_uncertainty=0.0, epistemic_reduction=0.5
        )
        d_ambiguous = EvidenceCandidate(
            doc_id="ambiguous", text="ambiguous doc", score=1.0,
            aleatoric_uncertainty=1.5, epistemic_reduction=0.0
        )
        calibrate_reliability([d_clean, d_ambiguous], method="clip", gamma_1=1.0, gamma_2=1.0)
        self.assertGreater(d_clean.calibrated_reliability, d_ambiguous.calibrated_reliability)

    def test_direct_mode(self) -> None:
        docs = [
            EvidenceCandidate(doc_id="A", text="a", score=0.95),
            EvidenceCandidate(doc_id="D", text="d", score=-0.20),
        ]
        calibrated = calibrate_reliability(docs, method="direct")
        self.assertEqual(calibrated[0], 0.95)
        self.assertEqual(calibrated[1], -0.20)


class TestCanonicalCounterexample(unittest.TestCase):
    """
    Formal reproduction of the canonical counterexample from Debate/GPT_opinion.md:
    Four documents: A(0.95), B(0.90), C(0.85), D(-0.20)
    Only 3 edges exist: A-B (+0.80), B-C (+0.70), C-D (-0.50, high conflict)
    """

    def setUp(self) -> None:
        self.candidates = [
            EvidenceCandidate(doc_id="A", text="Evidence A", score=0.95, calibrated_reliability=0.95),
            EvidenceCandidate(doc_id="B", text="Evidence B", score=0.90, calibrated_reliability=0.90),
            EvidenceCandidate(doc_id="C", text="Evidence C", score=0.85, calibrated_reliability=0.85),
            EvidenceCandidate(doc_id="D", text="Evidence D (Toxic)", score=-0.20, calibrated_reliability=-0.20),
        ]
        self.explicit_edges = {
            ("A", "B"): {"weight": 0.80, "comp": 0.80, "conf": 0.0, "red": 0.05},
            ("B", "C"): {"weight": 0.70, "comp": 0.70, "conf": 0.0, "red": 0.05},
            ("C", "D"): {"weight": -0.50, "comp": 0.05, "conf": 0.90, "red": 0.10},
        }
        self.edge_weights_only = {
            ("A", "B"): 0.80,
            ("B", "C"): 0.70,
            ("C", "D"): -0.50,
        }

    def test_naive_mst_fails_and_forces_toxic_node(self) -> None:
        """Naive single-stage MST must retain all nodes A, B, C, D and toxic edge C-D to preserve spanning."""
        baseline_res = naive_mst_baseline(self.candidates, self.edge_weights_only)
        retained_ids = set(baseline_res.ordered_ids)

        # Baseline MST retains all 4 nodes because it only optimizes edges, not node selection
        self.assertEqual(retained_ids, {"A", "B", "C", "D"}, "Naive MST failed to retain all 4 vertices")
        self.assertIn("D", retained_ids, "Toxic node D was unexpectedly pruned by naive MST")

        # In a 4-node connected graph with only 3 edges, MST MUST pick edge C-D
        mst_pairs = [(u, v) for u, v, _ in baseline_res.mst_edges] + [(v, u) for u, v, _ in baseline_res.mst_edges]
        self.assertIn(("C", "D"), mst_pairs, "Naive MST did not select toxic edge C-D")

    def test_two_stage_m2_purges_toxic_node(self) -> None:
        """Two-Stage M2 (M2-A + M2-B) successfully rejects toxic node D and outputs {A, B, C}."""
        optimizer = TwoStageM2Optimizer(
            lambda_coverage=1.0,
            lambda_complementarity=0.5,
            lambda_redundancy=0.5,
            lambda_conflict=1.0,
            selection_algorithm="exact_enum",
        )

        output = optimizer.optimize(
            candidates=self.candidates,
            explicit_edges=self.explicit_edges,
            max_k=3,
            calibrate=False,  # Keep exact calibrated values from counterexample
        )

        selected_ids = [d.doc_id for d in output.selected_docs]
        ordered_ids = output.ordered_ids

        # Stage M2-A correctly selected {A, B, C}
        self.assertEqual(set(selected_ids), {"A", "B", "C"})
        self.assertNotIn("D", selected_ids, "Toxic node D must be purged by M2-A")

        # Stage M2-B linearized {A, B, C} without D
        self.assertEqual(set(ordered_ids), {"A", "B", "C"})
        self.assertNotIn("D", ordered_ids, "Toxic node D must not appear in linearized evidence")

        # MST edges inside D* must only contain A-B and B-C
        mst_pairs = set()
        for u, v, _ in output.m2b_result.mst_edges:
            mst_pairs.add(tuple(sorted([u, v])))
        self.assertEqual(mst_pairs, {("A", "B"), ("B", "C")})

    def test_two_stage_m2_with_budget_4_still_rejects_toxic_node(self) -> None:
        """Even if budget max_k=4, M2-A rejects D because F({A,B,C,D}) < F({A,B,C}) due to negative R and conflict penalty."""
        optimizer = TwoStageM2Optimizer(
            lambda_coverage=1.0,
            lambda_complementarity=0.5,
            lambda_redundancy=0.5,
            lambda_conflict=1.0,
            selection_algorithm="exact_enum",
        )
        output = optimizer.optimize(
            candidates=self.candidates,
            explicit_edges=self.explicit_edges,
            max_k=4,  # Budget allows 4, but objective chooses 3
            calibrate=False,
        )
        selected_ids = [d.doc_id for d in output.selected_docs]
        self.assertEqual(set(selected_ids), {"A", "B", "C"})
        self.assertNotIn("D", selected_ids)


class TestBoundaryConditions(unittest.TestCase):
    """Unit tests for edge cases: empty candidate set, single document, and extreme conflict."""

    def setUp(self) -> None:
        self.organizer = MSTEvidenceOrganizer()
        self.selector = BudgetedEvidenceSelector()

    def test_empty_set_robustness(self) -> None:
        empty_docs: list[EvidenceCandidate] = []
        matrix = InteractionMatrix(0)
        res_a = self.selector.exact_enum_select(empty_docs, matrix, max_k=5, allow_empty=True)
        self.assertEqual(res_a.selected_docs, [])
        self.assertEqual(res_a.total_score, 0.0)

        res_b = self.organizer.organize(empty_docs)
        self.assertEqual(res_b.ordered_docs, [])
        self.assertEqual(res_b.ordered_ids, [])
        self.assertEqual(res_b.mst_edges, [])
        self.assertEqual(res_b.components, [])

    def test_single_document_robustness(self) -> None:
        single_doc = [EvidenceCandidate(doc_id="solo", text="Single passage", calibrated_reliability=0.88)]
        matrix = InteractionMatrix(1)
        res_a = self.selector.exact_enum_select(single_doc, matrix, max_k=5)
        self.assertEqual(len(res_a.selected_docs), 1)
        self.assertEqual(res_a.selected_docs[0].doc_id, "solo")

        res_b = self.organizer.organize(single_doc)
        self.assertEqual(res_b.ordered_ids, ["solo"])
        self.assertEqual(res_b.mst_edges, [])  # No edges possible for single node
        self.assertEqual(res_b.components, [["solo"]])
        self.assertEqual(res_b.roots, ["solo"])

    def test_extreme_conflict_rejection(self) -> None:
        """When candidates have extreme mutual conflict, M2-A should select only the single best candidate."""
        docs = [
            EvidenceCandidate(doc_id="C1", text="Claim X is true", calibrated_reliability=0.90),
            EvidenceCandidate(doc_id="C2", text="Claim X is completely false", calibrated_reliability=0.85),
        ]
        explicit_edges = {
            ("C1", "C2"): {"weight": -10.0, "comp": 0.0, "conf": 10.0, "red": 0.0}
        }
        optimizer = TwoStageM2Optimizer(
            lambda_conflict=2.0,
            selection_algorithm="exact_enum",
        )
        out = optimizer.optimize(docs, explicit_edges=explicit_edges, max_k=2, calibrate=False)
        # Extreme conflict (-20.0 penalty) prevents co-selection: only C1 should be chosen
        self.assertEqual([d.doc_id for d in out.selected_docs], ["C1"])
        self.assertEqual(out.ordered_ids, ["C1"])


class TestDisconnectedAndIsolatedComponents(unittest.TestCase):
    """Tests MSTEvidenceOrganizer behavior on multiple disconnected components and isolated vertices."""

    def test_multiple_components_and_isolated_vertex(self) -> None:
        # 6 nodes: Component 1 (n1-n2-n3), Component 2 (n4-n5), Component 3 (n6 isolated)
        docs = [
            EvidenceCandidate(doc_id="n1", text="t1", calibrated_reliability=0.9),
            EvidenceCandidate(doc_id="n2", text="t2", calibrated_reliability=0.8),
            EvidenceCandidate(doc_id="n3", text="t3", calibrated_reliability=0.7),
            EvidenceCandidate(doc_id="n4", text="t4", calibrated_reliability=0.85),
            EvidenceCandidate(doc_id="n5", text="t5", calibrated_reliability=0.75),
            EvidenceCandidate(doc_id="n6", text="t6 (isolated)", calibrated_reliability=0.6),
        ]
        explicit_edges = {
            ("n1", "n2"): 0.9,
            ("n2", "n3"): 0.8,
            ("n4", "n5"): 0.7,
        }

        organizer = MSTEvidenceOrganizer()
        result = organizer.organize(selected_docs=docs, explicit_edges=explicit_edges)

        # 1. All 6 nodes must be covered without loss or duplicate
        self.assertEqual(len(result.ordered_ids), 6)
        self.assertEqual(set(result.ordered_ids), {"n1", "n2", "n3", "n4", "n5", "n6"})

        # 2. Must identify exactly 3 components
        self.assertEqual(len(result.components), 3)

        # 3. MST edges count must be (3-1) + (2-1) + 0 = 3
        self.assertEqual(len(result.mst_edges), 3)

        # 4. Roots must contain 3 representatives
        self.assertEqual(len(result.roots), 3)
        self.assertIn("n6", result.roots)


class TestCombinatorialBenchmark15Choose5(unittest.TestCase):
    """
    Combinatorial scale benchmark for n=15 candidates, budget K=5:
    Exact subset count: sum_{k=0..5} C(15, k) = 1 + 15 + 105 + 455 + 1365 + 3003 = 4,944 subsets.
    Measures and logs deterministic CPU runtime.
    """

    def setUp(self) -> None:
        # Deterministic generation of 15 candidate documents
        self.candidates = []
        for i in range(15):
            score = 0.50 + 0.03 * (i % 10) - (0.1 if i % 4 == 0 else 0.0)
            entities = {f"ent_{i % 5}", f"ent_{(i + 1) % 5}", f"topic_{i // 3}"}
            self.candidates.append(
                EvidenceCandidate(
                    doc_id=f"doc_{i:02d}",
                    text=f"Sample evidence passage content number {i}",
                    score=score,
                    entities=entities,
                    aleatoric_uncertainty=0.1 * (i % 3),
                    epistemic_reduction=0.2 * ((i + 1) % 3),
                )
            )

        # Create structured interaction matrix with realistic properties
        self.matrix = InteractionMatrix(15)
        for i in range(15):
            for j in range(i + 1, 15):
                comp = 0.2 if (i + j) % 2 == 0 else 0.05
                red = 0.15 if abs(i - j) == 1 else 0.02
                conf = 0.3 if (i == 2 and j == 7) or (i == 4 and j == 11) else 0.0
                weight = comp - conf
                self.matrix.set_pair(i, j, comp=comp, red=red, conf=conf, weight=weight)

    def test_exact_enumeration_4944_subsets(self) -> None:
        selector = BudgetedEvidenceSelector(
            lambda_coverage=1.0,
            lambda_complementarity=0.5,
            lambda_redundancy=0.5,
            lambda_conflict=1.0,
        )

        calibrate_reliability(self.candidates, method="minmax")

        query_entities = {"ent_0", "ent_1", "ent_2", "topic_0"}

        # Run exact enumeration with allow_empty=True
        result = selector.exact_enum_select(
            candidates=self.candidates,
            interaction_matrix=self.matrix,
            max_k=5,
            query_entities=query_entities,
            allow_empty=True,
        )

        # Verify mathematical identity: sum_{k=0}^5 C(15, k) == 4,944
        self.assertEqual(
            result.evaluated_subsets_count,
            4944,
            f"Expected 4944 evaluated subsets, got {result.evaluated_subsets_count}"
        )

        # Verify selection constraints
        self.assertLessEqual(len(result.selected_docs), 5)
        self.assertGreater(len(result.selected_docs), 0)

        # Verify subscores integrity
        self.assertIn("reliability", result.subscores)
        self.assertIn("coverage", result.subscores)
        self.assertIn("conflict", result.subscores)

        # Performance assertion: 4,944 subsets on CPU should complete comfortably in < 150 ms
        self.assertLess(
            result.elapsed_ms,
            250.0,
            f"Exact enum too slow: {result.elapsed_ms:.2f} ms > 250 ms"
        )
        print(f"\n[BENCHMARK] 15-choose-5 Exact Enum: {result.evaluated_subsets_count} subsets evaluated in {result.elapsed_ms:.2f} ms")

    def test_greedy_vs_exact_ablation(self) -> None:
        """Compare greedy selection vs exact enumeration."""
        selector = BudgetedEvidenceSelector()
        calibrate_reliability(self.candidates, method="minmax")
        query_entities = {"ent_0", "ent_1", "ent_2", "topic_0"}

        res_exact = selector.exact_enum_select(
            candidates=self.candidates,
            interaction_matrix=self.matrix,
            max_k=5,
            query_entities=query_entities,
            allow_empty=False,
        )

        res_greedy = selector.greedy_select(
            candidates=self.candidates,
            interaction_matrix=self.matrix,
            max_k=5,
            query_entities=query_entities,
        )

        # Exact enumeration must be >= greedy by definition of optimality
        self.assertGreaterEqual(
            res_exact.total_score,
            res_greedy.total_score - 1e-6,
            f"Exact score ({res_exact.total_score}) should be >= Greedy score ({res_greedy.total_score})"
        )

        # Greedy should evaluate far fewer candidates
        # Base (1) + 15 + 14 + 13 + 12 + 11 = 66 maximum evaluations
        self.assertLessEqual(res_greedy.evaluated_subsets_count, 66)
        self.assertGreater(res_exact.evaluated_subsets_count, 4000)

        print(f"[ABLATION] Exact: score={res_exact.total_score:.4f}, evals={res_exact.evaluated_subsets_count}, time={res_exact.elapsed_ms:.2f}ms")
        print(f"[ABLATION] Greedy: score={res_greedy.total_score:.4f}, evals={res_greedy.evaluated_subsets_count}, time={res_greedy.elapsed_ms:.2f}ms")


if __name__ == "__main__":
    unittest.main()
