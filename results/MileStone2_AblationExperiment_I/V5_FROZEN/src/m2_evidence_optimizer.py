"""
m2_evidence_optimizer.py
========================
Two-Stage M2 Evidence Optimizer for U-GraphRAG (V1.1 Frozen Formulation)

This module implements the two-stage evidence optimization pipeline:
- M2-A: Budgeted Evidence Set Selection (BudgetedEvidenceSelector)
        Optimizes combinatorial objective:
        F(S) = sum_{d_i in S} R_tilde_i + lambda_1 * Coverage(S)
               + lambda_2 * Complementarity(S) - lambda_3 * Redundancy(S)
               - lambda_4 * Conflict(S)
        Supports both deterministic exact enumeration (sum_{k=0}^K C(n, k))
        and greedy approximation.
- M2-B: Maximum-Spanning-Tree Guided Evidence Organization (MSTEvidenceOrganizer)
        Constructs the backbone relational graph solely on the selected subset D*,
        computes MST/MSF using Kruskal's algorithm, and linearizes evidence via
        maximum-weight-edge rooted DFS with robust handling of |D*|=0, |D*|=1,
        and multiple disconnected components.

Pure CPU, Training-Free, Deterministic Implementation.
SDUFE GeminiOutOfToken Team.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class EvidenceCandidate:
    """Represents a candidate evidence passage retrieved for query processing."""

    doc_id: str
    text: str
    score: float = 0.0  # Raw retrieval relevance score S_rel
    entities: Set[str] = field(default_factory=set)
    aleatoric_uncertainty: float = 0.0  # U_AU >= 0 (e.g., token entropy / semantic ambiguity)
    epistemic_reduction: float = 0.0    # Delta U_EU >= 0 (information gain / query entropy drop)
    calibrated_reliability: float = 0.0 # R_tilde in [0, 1] or calibrated scale
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.entities, list):
            self.entities = set(self.entities)


@dataclass
class SelectionResult:
    """Encapsulates the result of M2-A budgeted evidence selection."""

    selected_docs: List[EvidenceCandidate]
    selected_indices: List[int]
    total_score: float
    subscores: Dict[str, float]
    evaluated_subsets_count: int
    elapsed_ms: float
    algorithm: str


@dataclass
class OrganizationResult:
    """Encapsulates the result of M2-B MST evidence organization."""

    ordered_docs: List[EvidenceCandidate]
    ordered_ids: List[str]
    mst_edges: List[Tuple[str, str, float]]  # (u, v, weight)
    components: List[List[str]]              # Connected components
    roots: List[str]                         # Root node per component
    elapsed_ms: float


@dataclass
class M2OptimizationOutput:
    """Complete output of the Two-Stage M2 pipeline."""

    selected_docs: List[EvidenceCandidate]
    ordered_docs: List[EvidenceCandidate]
    ordered_ids: List[str]
    m2a_result: SelectionResult
    m2b_result: OrganizationResult
    total_elapsed_ms: float


# -----------------------------------------------------------------------------
# Disjoint Set Union (Union-Find) for Kruskal's MST
# -----------------------------------------------------------------------------
class DisjointSetUnion:
    """Disjoint Set Union with path compression and union-by-rank."""

    def __init__(self, elements: List[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in elements}
        self.rank: Dict[str, int] = {x: 0 for x in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


# -----------------------------------------------------------------------------
# Reliability Calibration
# -----------------------------------------------------------------------------
def calibrate_reliability(
    candidates: List[EvidenceCandidate],
    method: str = "minmax",
    gamma_1: float = 1.0,
    gamma_2: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> List[float]:
    """
    Safely calibrate raw retrieval scores S_rel into non-negative reliability R_tilde in [0, 1].

    Formula:
      S_norm = Normalized(S_rel) in [0, 1]
      R_tilde = clip(S_norm * exp(gamma_1 * Delta U_EU - gamma_2 * U_AU), clip_min, clip_max)

    If gamma_1 == 0 and gamma_2 == 0, R_tilde reduces to calibrated relevance score.
    """
    if not candidates:
        return []

    raw_scores = [c.score for c in candidates]
    n = len(candidates)

    if method == "minmax":
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        if abs(max_s - min_s) < 1e-9:
            norm_scores = [1.0 if max_s > 0 else 0.5 for _ in raw_scores]
        else:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in raw_scores]
    elif method == "sigmoid":
        norm_scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
    elif method == "clip":
        norm_scores = [max(0.0, min(1.0, s)) for s in raw_scores]
    elif method == "direct":
        # Keep raw score directly (can be negative for counterexample testing)
        norm_scores = list(raw_scores)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibrated: List[float] = []
    for i, c in enumerate(candidates):
        s_norm = norm_scores[i]
        if method == "direct":
            # Direct mode preserves negative values if specified for adversarial audit
            r_val = s_norm
        else:
            # Multiplicative uncertainty modulation
            unc_factor = math.exp(gamma_1 * c.epistemic_reduction - gamma_2 * c.aleatoric_uncertainty)
            r_val = max(clip_min, min(clip_max, s_norm * unc_factor))

        c.calibrated_reliability = r_val
        calibrated.append(r_val)

    return calibrated


# -----------------------------------------------------------------------------
# Interaction Matrix Cache
# -----------------------------------------------------------------------------
class InteractionMatrix:
    """Precomputed pairwise interaction matrices for complementary, redundancy, and conflict."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.complementarity: List[List[float]] = [[0.0] * size for _ in range(size)]
        self.redundancy: List[List[float]] = [[0.0] * size for _ in range(size)]
        self.conflict: List[List[float]] = [[0.0] * size for _ in range(size)]
        self.edge_weights: List[List[float]] = [[0.0] * size for _ in range(size)]

    def set_pair(
        self,
        i: int,
        j: int,
        comp: float = 0.0,
        red: float = 0.0,
        conf: float = 0.0,
        weight: Optional[float] = None,
    ) -> None:
        self.complementarity[i][j] = self.complementarity[j][i] = comp
        self.redundancy[i][j] = self.redundancy[j][i] = red
        self.conflict[i][j] = self.conflict[j][i] = conf
        if weight is None:
            weight = comp - conf
        self.edge_weights[i][j] = self.edge_weights[j][i] = weight

    @classmethod
    def from_candidates_and_edges(
        cls,
        candidates: List[EvidenceCandidate],
        edge_data: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    ) -> InteractionMatrix:
        """Construct interaction matrix from explicit edge data dictionary."""
        n = len(candidates)
        mat = cls(n)
        id2idx = {c.doc_id: i for i, c in enumerate(candidates)}

        # Natural fallback: entity overlap if no edge_data supplied
        if edge_data is None:
            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = candidates[i], candidates[j]
                    if ci.entities and cj.entities:
                        inter = len(ci.entities & cj.entities)
                        union = len(ci.entities | cj.entities)
                        jaccard = inter / union if union > 0 else 0.0
                        mat.set_pair(i, j, comp=jaccard, red=jaccard * 0.5, conf=0.0, weight=jaccard)
            return mat

        for (u, v), props in edge_data.items():
            if u in id2idx and v in id2idx:
                i, j = id2idx[u], id2idx[v]
                comp = props.get("complementarity", props.get("comp", 0.0))
                red = props.get("redundancy", props.get("red", 0.0))
                conf = props.get("conflict", props.get("conf", 0.0))
                weight = props.get("weight", comp - conf)
                mat.set_pair(i, j, comp=comp, red=red, conf=conf, weight=weight)

        return mat


# -----------------------------------------------------------------------------
# Stage M2-A: Budgeted Evidence Set Selector
# -----------------------------------------------------------------------------
class BudgetedEvidenceSelector:
    """
    M2-A: Budgeted Evidence Set Selection.

    Solves:
      D* = argmax_{S subset D_cand, |S| <= K} F(S)

    Where:
      F(S) = sum_{d_i in S} R_tilde_i + lambda_1 * Coverage(S)
             + lambda_2 * Complementarity(S) - lambda_3 * Redundancy(S)
             - lambda_4 * Conflict(S)
    """

    def __init__(
        self,
        lambda_coverage: float = 1.0,
        lambda_complementarity: float = 0.5,
        lambda_redundancy: float = 0.5,
        lambda_conflict: float = 1.0,
    ) -> None:
        self.lambda_1 = lambda_coverage
        self.lambda_2 = lambda_complementarity
        self.lambda_3 = lambda_redundancy
        self.lambda_4 = lambda_conflict

    def compute_coverage(
        self,
        subset_indices: Tuple[int, ...],
        candidates: List[EvidenceCandidate],
        query_entities: Optional[Set[str]] = None,
    ) -> float:
        """Compute query entity/concept coverage in [0, 1]."""
        if not subset_indices:
            return 0.0
        if not query_entities:
            # Without explicit query entities, coverage is normalized count of covered entities
            all_ent = set()
            for idx in subset_indices:
                all_ent.update(candidates[idx].entities)
            # Saturated logarithmic or ratio coverage
            return min(1.0, len(all_ent) / 5.0) if all_ent else 0.5

        covered = set()
        for idx in subset_indices:
            covered.update(candidates[idx].entities & query_entities)
        return len(covered) / len(query_entities)

    def evaluate_subset(
        self,
        subset_indices: Tuple[int, ...],
        candidates: List[EvidenceCandidate],
        interaction_matrix: InteractionMatrix,
        query_entities: Optional[Set[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the objective F(S) for a specific candidate subset."""
        if not subset_indices:
            return 0.0, {
                "reliability": 0.0,
                "coverage": 0.0,
                "complementarity": 0.0,
                "redundancy": 0.0,
                "conflict": 0.0,
            }

        # 1. Sum of calibrated reliability
        rel_sum = sum(candidates[idx].calibrated_reliability for idx in subset_indices)

        # 2. Query coverage
        cov = self.compute_coverage(subset_indices, candidates, query_entities)

        # 3. Pairwise interactions
        comp_sum = 0.0
        red_sum = 0.0
        conf_sum = 0.0

        k = len(subset_indices)
        for a in range(k):
            i = subset_indices[a]
            for b in range(a + 1, k):
                j = subset_indices[b]
                comp_sum += interaction_matrix.complementarity[i][j]
                red_sum += interaction_matrix.redundancy[i][j]
                conf_sum += interaction_matrix.conflict[i][j]

        total = (
            rel_sum
            + self.lambda_1 * cov
            + self.lambda_2 * comp_sum
            - self.lambda_3 * red_sum
            - self.lambda_4 * conf_sum
        )

        subscores = {
            "reliability": rel_sum,
            "coverage": cov,
            "complementarity": comp_sum,
            "redundancy": red_sum,
            "conflict": conf_sum,
        }
        return total, subscores

    def exact_enum_select(
        self,
        candidates: List[EvidenceCandidate],
        interaction_matrix: InteractionMatrix,
        max_k: int = 5,
        query_entities: Optional[Set[str]] = None,
        allow_empty: bool = False,
        min_k: int = 1,
    ) -> SelectionResult:
        """
        Exact enumeration of all subsets of size <= max_k:
          Total evaluated subsets = sum_{k=0..max_k} C(n, k)
        For n=15, k=5, sum is 1 + 15 + 105 + 455 + 1365 + 3003 = 4,944 subsets.
        """
        t0 = time.perf_counter()
        n = len(candidates)
        effective_k = min(n, max_k)

        best_score = -float("inf") if not allow_empty else 0.0
        best_subset: Tuple[int, ...] = ()
        best_subscores: Dict[str, float] = {}
        eval_count = 0

        if allow_empty or effective_k == 0:
            min_size = 0
        else:
            min_size = min(max(1, int(min_k)), effective_k)
        # When candidates are completely empty, return empty selection safely
        if n == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SelectionResult(
                selected_docs=[],
                selected_indices=[],
                total_score=0.0,
                subscores={"reliability": 0.0, "coverage": 0.0, "complementarity": 0.0, "redundancy": 0.0, "conflict": 0.0},
                evaluated_subsets_count=1 if allow_empty else 0,
                elapsed_ms=elapsed_ms,
                algorithm="exact_enum",
            )

        for size in range(min_size, effective_k + 1):
            for subset in combinations(range(n), size):
                eval_count += 1
                score, subscores = self.evaluate_subset(
                    subset, candidates, interaction_matrix, query_entities
                )
                if score > best_score:
                    best_score = score
                    best_subset = subset
                    best_subscores = subscores

        # Edge case: if allow_empty is False but all subsets produced negative scores,
        # fallback to the single highest reliability document
        if not best_subset and not allow_empty and n > 0:
            best_idx = max(range(n), key=lambda idx: candidates[idx].calibrated_reliability)
            best_subset = (best_idx,)
            best_score, best_subscores = self.evaluate_subset(
                best_subset, candidates, interaction_matrix, query_entities
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        selected_docs = [candidates[i] for i in best_subset]

        return SelectionResult(
            selected_docs=selected_docs,
            selected_indices=list(best_subset),
            total_score=best_score,
            subscores=best_subscores,
            evaluated_subsets_count=eval_count,
            elapsed_ms=elapsed_ms,
            algorithm="exact_enum",
        )

    def greedy_select(
        self,
        candidates: List[EvidenceCandidate],
        interaction_matrix: InteractionMatrix,
        max_k: int = 5,
        query_entities: Optional[Set[str]] = None,
        min_k: int = 1,
    ) -> SelectionResult:
        """
        Greedy forward step-by-step selection for ablation & speed benchmark.
        Iteratively adds candidate doc that yields highest marginal gain Delta F(d).
        Once min_k documents are selected, stops early if max marginal gain <= 0.
        """
        t0 = time.perf_counter()
        n = len(candidates)
        effective_k = min(n, max_k)
        target_min_k = min(max(1, int(min_k)), effective_k)

        current_subset: List[int] = []
        best_score, best_subscores = self.evaluate_subset(
            tuple(current_subset), candidates, interaction_matrix, query_entities
        )
        eval_count = 1  # Base empty set

        if n == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SelectionResult(
                selected_docs=[],
                selected_indices=[],
                total_score=0.0,
                subscores=best_subscores,
                evaluated_subsets_count=eval_count,
                elapsed_ms=elapsed_ms,
                algorithm="greedy",
            )

        remaining = set(range(n))
        for _ in range(effective_k):
            # When min_k > 1 (e.g., multi-hop QA), enforce reaching target_min_k before early stopping
            below_min_k = (len(current_subset) < target_min_k) if target_min_k > 1 else False
            best_cand = None
            best_cand_score = -float("inf") if below_min_k else best_score
            best_cand_subscores = best_subscores

            for cand in remaining:
                eval_count += 1
                trial_subset = tuple(sorted(current_subset + [cand]))
                score, subscores = self.evaluate_subset(
                    trial_subset, candidates, interaction_matrix, query_entities
                )
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand = cand
                    best_cand_subscores = subscores

            if best_cand is not None and (below_min_k or best_cand_score > best_score):
                current_subset.append(best_cand)
                remaining.remove(best_cand)
                best_score = best_cand_score
                best_subscores = best_cand_subscores
            else:
                break

        # If greedy ended with empty set, pick single best doc if candidates exist
        if not current_subset and n > 0:
            best_single = max(range(n), key=lambda idx: candidates[idx].calibrated_reliability)
            current_subset = [best_single]
            best_score, best_subscores = self.evaluate_subset(
                tuple(current_subset), candidates, interaction_matrix, query_entities
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        selected_docs = [candidates[i] for i in current_subset]

        return SelectionResult(
            selected_docs=selected_docs,
            selected_indices=current_subset,
            total_score=best_score,
            subscores=best_subscores,
            evaluated_subsets_count=eval_count,
            elapsed_ms=elapsed_ms,
            algorithm="greedy",
        )


# -----------------------------------------------------------------------------
# Stage M2-B: Maximum-Spanning-Tree Guided Evidence Organizer
# -----------------------------------------------------------------------------
class MSTEvidenceOrganizer:
    """
    M2-B: Maximum-Spanning-Tree Guided Evidence Organizer.

    Operates strictly on the pre-selected subset D* from M2-A:
    1. Extracts subgraph G[D*] with edge weights W(u, v).
    2. Computes Maximum Spanning Tree / Forest via Kruskal's algorithm.
    3. Handles boundary cases: |D*| = 0, |D*| = 1, isolated nodes, multiple components.
    4. Performs deterministic DFS linearization rooted at the highest-weight edge.
       When recency_aware=True (default for causal decoder LLMs), places supporting
       components and peripheral leaves first so the primary anchor root sits adjacent
       to the question prompt (bottom-up evidence aggregation).
    """

    def __init__(self, recency_aware: bool = False) -> None:
        self.recency_aware = bool(recency_aware)

    def organize(
        self,
        selected_docs: List[EvidenceCandidate],
        interaction_matrix: Optional[InteractionMatrix] = None,
        explicit_edges: Optional[Dict[Tuple[str, str], float]] = None,
        all_candidates: Optional[List[EvidenceCandidate]] = None,
        recency_aware: Optional[bool] = None,
        query_entities: Optional[Set[str]] = None,
    ) -> OrganizationResult:
        """
        Organize selected evidence documents into a linearized sequence.

        Parameters:
          selected_docs: Subset D* selected by M2-A.
          interaction_matrix: Precomputed matrix covering all_candidates.
          explicit_edges: Optional explicit (u, v) -> weight dictionary.
          all_candidates: Original candidate list corresponding to interaction_matrix.
          recency_aware: Override recency-aware bottom-up leaf-to-root linearization
                         (defaults to False -> Forward Root-First MST-DFS primacy order).
          query_entities: Optional normalized entities mentioned in the query for root priority.
        """
        use_recency = self.recency_aware if recency_aware is None else bool(recency_aware)
        q_ents = {str(e).lower() for e in query_entities if e} if query_entities else set()
        t0 = time.perf_counter()
        k = len(selected_docs)

        # Boundary Case 1: Empty set
        if k == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return OrganizationResult(
                ordered_docs=[],
                ordered_ids=[],
                mst_edges=[],
                components=[],
                roots=[],
                elapsed_ms=elapsed_ms,
            )

        # Boundary Case 2: Single document
        if k == 1:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            doc = selected_docs[0]
            return OrganizationResult(
                ordered_docs=[doc],
                ordered_ids=[doc.doc_id],
                mst_edges=[],
                components=[[doc.doc_id]],
                roots=[doc.doc_id],
                elapsed_ms=elapsed_ms,
            )

        # Build local index mapping for D*
        doc_ids = [d.doc_id for d in selected_docs]
        doc_map = {d.doc_id: d for d in selected_docs}

        # Collect candidate edges between nodes in D*
        edges: List[Tuple[str, str, float]] = []

        if explicit_edges is not None:
            # Use provided explicit edge dictionary
            for (u, v), w in explicit_edges.items():
                if u in doc_map and v in doc_map and u != v:
                    edges.append((u, v, float(w)))
        elif interaction_matrix is not None and all_candidates is not None:
            # Map back to global candidate indices
            cand_id2idx = {c.doc_id: i for i, c in enumerate(all_candidates)}
            for i in range(k):
                u_id = doc_ids[i]
                u_global = cand_id2idx.get(u_id)
                if u_global is None:
                    continue
                for j in range(i + 1, k):
                    v_id = doc_ids[j]
                    v_global = cand_id2idx.get(v_id)
                    if v_global is None:
                        continue
                    w = interaction_matrix.edge_weights[u_global][v_global]
                    edges.append((u_id, v_id, float(w)))
        else:
            # Natural fallback: entity jaccard similarity
            for i in range(k):
                for j in range(i + 1, k):
                    di, dj = selected_docs[i], selected_docs[j]
                    if di.entities and dj.entities:
                        inter = len(di.entities & dj.entities)
                        union = len(di.entities | dj.entities)
                        w = inter / union if union > 0 else 0.0
                    else:
                        w = 0.0
                    edges.append((di.doc_id, dj.doc_id, w))

        # Sort edges descending by weight for Maximum Spanning Tree
        edges.sort(key=lambda e: e[2], reverse=True)

        # Kruskal's algorithm
        dsu = DisjointSetUnion(doc_ids)
        mst_edges: List[Tuple[str, str, float]] = []
        adj: Dict[str, List[Tuple[str, float]]] = {doc_id: [] for doc_id in doc_ids}

        for u, v, w in edges:
            if dsu.union(u, v):
                mst_edges.append((u, v, w))
                adj[u].append((v, w))
                adj[v].append((u, w))

        # Identify connected components in the resulting MST / MSF
        comp_map: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            root = dsu.find(doc_id)
            comp_map.setdefault(root, []).append(doc_id)

        # Sort neighbors in adj by edge weight descending for deterministic DFS
        for u in adj:
            adj[u].sort(key=lambda item: item[1], reverse=True)

        # Linearize each connected component via Rooted DFS
        components_list: List[List[str]] = []
        ordered_ids: List[str] = []
        component_roots: List[str] = []

        # Sort components by query-entity overlap, internal edge weight, and max document reliability
        def _query_overlap_count(nid: str) -> int:
            if not q_ents or not doc_map[nid].entities:
                return 0
            return len({str(e).lower() for e in doc_map[nid].entities} & q_ents)

        def component_priority(comp_nodes: List[str]) -> Tuple[int, float, float]:
            comp_set = set(comp_nodes)
            comp_edge_weights = [w for (u, v, w) in mst_edges if u in comp_set and v in comp_set]
            max_w = max(comp_edge_weights) if comp_edge_weights else -float("inf")
            max_q_ov = max(_query_overlap_count(nid) for nid in comp_nodes)
            max_rel = max(doc_map[nid].calibrated_reliability for nid in comp_nodes)
            return (max_q_ov, max_w, max_rel)

        # Forward Root-First (recency_aware=False): primary anchor component & root come FIRST (primacy order).
        # When recency_aware=True, secondary components precede the primary highest-priority component.
        sorted_components = sorted(comp_map.values(), key=component_priority, reverse=not use_recency)

        for comp_nodes in sorted_components:
            comp_set = set(comp_nodes)
            components_list.append(comp_nodes)

            # Determine root for this component
            if len(comp_nodes) == 1:
                # Isolated single node
                root_node = comp_nodes[0]
            else:
                # Find maximum-weight edge inside this component
                internal_edges = [e for e in mst_edges if e[0] in comp_set and e[1] in comp_set]
                if internal_edges:
                    best_edge = max(internal_edges, key=lambda e: e[2])
                    u_cand, v_cand, _ = best_edge
                    # Break tie by query-entity overlap, then calibrated reliability, then doc_id
                    key_u = (_query_overlap_count(u_cand), float(doc_map[u_cand].calibrated_reliability))
                    key_v = (_query_overlap_count(v_cand), float(doc_map[v_cand].calibrated_reliability))
                    if key_u > key_v:
                        root_node = u_cand
                    elif key_v > key_u:
                        root_node = v_cand
                    else:
                        root_node = min(u_cand, v_cand)
                else:
                    # Multi-node component without edges (unlikely, but safe fallback)
                    root_node = max(
                        comp_nodes,
                        key=lambda nid: (_query_overlap_count(nid), doc_map[nid].calibrated_reliability, nid),
                    )

            component_roots.append(root_node)

            # DFS traversal within component
            visited: Set[str] = set()
            dfs_order: List[str] = []

            def dfs(node: str) -> None:
                visited.add(node)
                dfs_order.append(node)
                for neighbor, _ in adj[node]:
                    if neighbor in comp_set and neighbor not in visited:
                        dfs(neighbor)

            dfs(root_node)

            # Safety check: if any disconnected isolated node exists in the component
            for node in comp_nodes:
                if node not in visited:
                    dfs_order.append(node)
                    visited.add(node)

            if use_recency:
                # Bottom-up MST linearization: peripheral leaves -> bridge -> primary root right before Question:
                dfs_order.reverse()

            ordered_ids.extend(dfs_order)

        ordered_docs = [doc_map[nid] for nid in ordered_ids]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return OrganizationResult(
            ordered_docs=ordered_docs,
            ordered_ids=ordered_ids,
            mst_edges=mst_edges,
            components=components_list,
            roots=component_roots,
            elapsed_ms=elapsed_ms,
        )


# -----------------------------------------------------------------------------
# End-to-End Two-Stage Optimizer
# -----------------------------------------------------------------------------
class TwoStageM2Optimizer:
    """
    Unified end-to-end interface for Two-Stage M2 Evidence Optimizer:
      Stage 1 (M2-A): Budgeted Evidence Set Selection -> D*
      Stage 2 (M2-B): MST-Guided Organization on D* -> Linearized Evidence Sequence
    """

    def __init__(
        self,
        lambda_coverage: float = 1.0,
        lambda_complementarity: float = 0.5,
        lambda_redundancy: float = 0.5,
        lambda_conflict: float = 1.0,
        selection_algorithm: str = "exact_enum",
        recency_aware: bool = False,
    ) -> None:
        self.selector = BudgetedEvidenceSelector(
            lambda_coverage=lambda_coverage,
            lambda_complementarity=lambda_complementarity,
            lambda_redundancy=lambda_redundancy,
            lambda_conflict=lambda_conflict,
        )
        self.organizer = MSTEvidenceOrganizer(recency_aware=recency_aware)
        self.selection_algorithm = selection_algorithm

    def optimize(
        self,
        candidates: List[EvidenceCandidate],
        interaction_matrix: Optional[InteractionMatrix] = None,
        explicit_edges: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
        max_k: int = 5,
        query_entities: Optional[Set[str]] = None,
        calibrate: bool = True,
        calibration_method: str = "minmax",
        min_k: int = 1,
    ) -> M2OptimizationOutput:
        """
        Execute full Two-Stage pipeline.

        1. Calibrates candidate reliability if needed.
        2. Solves M2-A budgeted selection.
        3. Solves M2-B MST evidence organization solely on D*.
        """
        t0 = time.perf_counter()

        if calibrate:
            calibrate_reliability(candidates, method=calibration_method)

        if interaction_matrix is None:
            interaction_matrix = InteractionMatrix.from_candidates_and_edges(
                candidates, explicit_edges
            )

        # Stage M2-A
        if self.selection_algorithm == "exact_enum":
            m2a_res = self.selector.exact_enum_select(
                candidates=candidates,
                interaction_matrix=interaction_matrix,
                max_k=max_k,
                query_entities=query_entities,
                allow_empty=False,
                min_k=min_k,
            )
        elif self.selection_algorithm == "greedy":
            m2a_res = self.selector.greedy_select(
                candidates=candidates,
                interaction_matrix=interaction_matrix,
                max_k=max_k,
                query_entities=query_entities,
                min_k=min_k,
            )
        else:
            raise ValueError(f"Unknown selection algorithm: {self.selection_algorithm}")

        # Stage M2-B
        # If explicit_edges supplied, format edge weights for organizer
        m2b_explicit_weights: Optional[Dict[Tuple[str, str], float]] = None
        if explicit_edges is not None:
            m2b_explicit_weights = {
                pair: data.get("weight", data.get("comp", 0.0) - data.get("conf", 0.0))
                for pair, data in explicit_edges.items()
            }

        m2b_res = self.organizer.organize(
            selected_docs=m2a_res.selected_docs,
            interaction_matrix=interaction_matrix,
            explicit_edges=m2b_explicit_weights,
            all_candidates=candidates,
            query_entities=query_entities,
        )

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return M2OptimizationOutput(
            selected_docs=m2a_res.selected_docs,
            ordered_docs=m2b_res.ordered_docs,
            ordered_ids=m2b_res.ordered_ids,
            m2a_result=m2a_res,
            m2b_result=m2b_res,
            total_elapsed_ms=total_elapsed_ms,
        )


# -----------------------------------------------------------------------------
# Baseline: Naive MST (for formal audit and counterexample comparison)
# -----------------------------------------------------------------------------
def naive_mst_baseline(
    candidates: List[EvidenceCandidate],
    edges: Dict[Tuple[str, str], float],
) -> OrganizationResult:
    """
    Direct Naive MST baseline without Stage M2-A node selection.
    Forces all candidate nodes into the spanning tree regardless of reliability.
    """
    organizer = MSTEvidenceOrganizer()
    return organizer.organize(
        selected_docs=candidates,
        explicit_edges=edges,
    )
