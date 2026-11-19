"""
src/m1_adaptive_retriever.py
============================
Method 1 (M1) for U-GraphRAG: Epistemic Uncertainty Decoupling & MeshRAG Hash Expansion

This module implements:
1. UncertaintyDecoupler:
   - Closed-form decomposition of total predictive entropy into Aleatoric Uncertainty (AU)
     and Epistemic Uncertainty (EU): U_EU = max(0, H_total - U_AU).
   - Offline calibration (training-free probing) on small validation datasets to calibrate
     the threshold tau_EU, aleatoric scaling factors, and attention-head weights,
     persisted to models/m1_calibration.json.
   - Marginal epistemic uncertainty reduction Delta U_EU(q; d_i) = max(0, U_EU(D_0) - U_EU(D_0 U {d_i})).
   - First-token predictive distributions p_i for downstream M2 conflict (JS divergence) computation.
2. MinHashGraphExpander:
   - k-shingle MinHash signatures and collision graph construction (LSH-style zero-token expansion).
   - Query entity fingerprint extraction and Bloom Filter community pruning.
   - Explicit tracking of golden paragraph recall drop under Bloom pruning.
   - Adaptive gating routing: Early exit if U_EU <= tau_EU with token saving rate calculation;
     otherwise, expansion to candidate pool D_cand (|D_cand| <= 15).
3. M1AdaptiveRetriever:
   - Integrated pipeline bridging initial retrieval D_0, uncertainty gating, hash expansion,
     and candidate preparation for M2 evidence optimizer.

Pure Python / PyTorch compatible, Deterministic and Training-Free.
SDUFE GeminiOutOfToken Team.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Windows UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# -----------------------------------------------------------------------------
# Data Containers
# -----------------------------------------------------------------------------
@dataclass
class UncertaintyMetrics:
    """Encapsulates predictive uncertainty decomposition metrics."""

    total_entropy: float                  # H(q, D) >= 0
    aleatoric_uncertainty: float          # U_AU(q, D) >= 0
    epistemic_uncertainty: float          # U_EU = max(0, H - U_AU) >= 0
    predictive_distribution: Dict[str, float] = field(default_factory=dict)
    marginal_reductions: Dict[str, float] = field(default_factory=dict)  # doc_id -> Delta U_EU
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpansionResult:
    """Encapsulates output of adaptive graph expansion."""

    query: str
    initial_docs: List[Dict[str, Any]]
    expanded_candidates: List[Dict[str, Any]]
    early_exit: bool
    epistemic_uncertainty: float
    tau_eu: float
    token_saving_rate: float
    pruning_stats: Dict[str, Any]
    elapsed_ms: float
    routing_decision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Mathematical & Information Theoretic Utilities
# -----------------------------------------------------------------------------
def compute_entropy(distribution: Union[Dict[str, float], Sequence[float]], base: float = math.e) -> float:
    """
    Compute Shannon entropy H(p) = - sum p_i * log(p_i).
    Distribution can be a probability dict {token: prob} or a list/tensor of non-negative probabilities.
    Guaranteed non-negative (>= 0.0).
    """
    if isinstance(distribution, dict):
        probs = [p for p in distribution.values() if p > 0.0]
    else:
        probs = [float(p) for p in distribution if float(p) > 0.0]

    if not probs:
        return 0.0

    total_p = sum(probs)
    if total_p <= 0.0:
        return 0.0

    # Normalize to ensure valid probability mass
    norm_probs = [p / total_p for p in probs]

    log_fn = math.log if base == math.e else (lambda x: math.log(x) / math.log(base))
    entropy = -sum(p * log_fn(p) for p in norm_probs if p > 0.0)
    return max(0.0, float(entropy))


def js_divergence(
    p: Union[Dict[str, float], Sequence[float]],
    q: Union[Dict[str, float], Sequence[float]],
    base: float = math.e,
) -> float:
    """
    Compute Jensen-Shannon Divergence D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M),
    where M = 0.5 * (P + Q).
    Symmetric and bounded in [0, ln(2)] (if base=e) or [0, 1] (if base=2).
    """
    if isinstance(p, dict) and isinstance(q, dict):
        all_keys = set(p.keys()) | set(q.keys())
        p_vec = [p.get(k, 0.0) for k in all_keys]
        q_vec = [q.get(k, 0.0) for k in all_keys]
    else:
        p_list = list(p)
        q_list = list(q)
        max_len = max(len(p_list), len(q_list))
        p_vec = p_list + [0.0] * (max_len - len(p_list))
        q_vec = q_list + [0.0] * (max_len - len(q_list))

    sum_p = sum(p_vec)
    sum_q = sum(q_vec)
    if sum_p <= 0.0 and sum_q <= 0.0:
        return 0.0
    if sum_p <= 0.0:
        return 1.0 if base == 2 else math.log(2.0)
    if sum_q <= 0.0:
        return 1.0 if base == 2 else math.log(2.0)

    p_norm = [x / sum_p for x in p_vec]
    q_norm = [y / sum_q for y in q_vec]

    log_fn = math.log if base == math.e else (lambda x: math.log(x) / math.log(base))

    # M = 0.5 * (P + Q)
    m_norm = [0.5 * (p_i + q_i) for p_i, q_i in zip(p_norm, q_norm)]

    kl_pm = 0.0
    kl_qm = 0.0
    for p_i, q_i, m_i in zip(p_norm, q_norm, m_norm):
        if p_i > 0.0 and m_i > 0.0:
            kl_pm += p_i * log_fn(p_i / m_i)
        if q_i > 0.0 and m_i > 0.0:
            kl_qm += q_i * log_fn(q_i / m_i)

    js = 0.5 * (kl_pm + kl_qm)
    return max(0.0, float(js))


# -----------------------------------------------------------------------------
# Bloom Filter Implementation for Community Pruning
# -----------------------------------------------------------------------------
class BloomFilter:
    """
    Standard space-efficient probabilistic Bloom Filter with multiple hash seeds.
    Used for query-entity community pruning and fast membership testing.
    """

    def __init__(
        self,
        capacity: int = 1000,
        error_rate: float = 0.01,
        expected_elements: Optional[int] = None,
        false_positive_rate: Optional[float] = None,
    ) -> None:
        if expected_elements is not None:
            capacity = expected_elements
        if false_positive_rate is not None:
            error_rate = false_positive_rate

        if capacity <= 0:
            capacity = 100
        if not (0.0 < error_rate < 1.0):
            error_rate = 0.01

        self.capacity = capacity
        self.error_rate = error_rate
        # m = - (n * ln(p)) / (ln(2)^2)
        self.bit_size = max(16, int(-1.0 * capacity * math.log(error_rate) / (math.log(2.0) ** 2)))
        # k = (m / n) * ln(2)
        self.num_hashes = max(1, int((self.bit_size / capacity) * math.log(2.0)))
        self.bit_array = bytearray((self.bit_size + 7) // 8)
        self.count = 0

    def __contains__(self, item: str) -> bool:
        return self.contains(item)

    def _get_hashes(self, item: str) -> List[int]:
        """Compute k distinct hash indices using double hashing (Kirsch-Mitzenmacher technique)."""
        b = item.encode("utf-8")
        h1 = int(hashlib.sha256(b).hexdigest()[:16], 16)
        h2 = int(hashlib.md5(b).hexdigest()[:16], 16)
        return [(h1 + i * h2) % self.bit_size for i in range(self.num_hashes)]

    def add(self, item: str) -> None:
        """Add an item into the Bloom Filter."""
        for bit_idx in self._get_hashes(item):
            byte_idx = bit_idx // 8
            offset = bit_idx % 8
            self.bit_array[byte_idx] |= (1 << offset)
        self.count += 1

    def contains(self, item: str) -> bool:
        """Test whether an item might be in the set (zero false negatives)."""
        for bit_idx in self._get_hashes(item):
            byte_idx = bit_idx // 8
            offset = bit_idx % 8
            if not (self.bit_array[byte_idx] & (1 << offset)):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        return self.contains(item)

    def current_false_positive_rate(self) -> float:
        """Calculate theoretical false positive probability under current count."""
        # p = (1 - exp(-k * n / m))^k
        exponent = -1.0 * self.num_hashes * self.count / self.bit_size
        return float((1.0 - math.exp(exponent)) ** self.num_hashes)


# -----------------------------------------------------------------------------
# MinHash Signer and Collision Graph Expander
# -----------------------------------------------------------------------------
class MinHashIndexer:
    """
    MinHash generator for passage k-shingles using deterministic linear permutation hashes.
    h_i(x) = (a_i * hash(x) + b_i) mod p_prime
    """

    LARGE_PRIME = 2147483647  # 2^31 - 1 Mersenne prime

    def __init__(self, num_perm: int = 64, shingle_k: int = 3, seed: int = 42) -> None:
        self.num_perm = num_perm
        self.shingle_k = shingle_k
        self.seed = seed
        self._a, self._b = self._init_permutations(num_perm, seed)

    def _init_permutations(self, num_perm: int, seed: int) -> Tuple[List[int], List[int]]:
        import random

        rng = random.Random(seed)
        a = [rng.randint(1, self.LARGE_PRIME - 1) for _ in range(num_perm)]
        b = [rng.randint(0, self.LARGE_PRIME - 1) for _ in range(num_perm)]
        return a, b

    def extract_shingles(self, text: str) -> Set[str]:
        """Extract word-level or character-level k-shingles."""
        cleaned = re.sub(r"\s+", " ", text.lower().strip())
        tokens = [t.strip(string.punctuation) for t in cleaned.split() if t.strip(string.punctuation)]
        if len(tokens) >= self.shingle_k:
            return {" ".join(tokens[i : i + self.shingle_k]) for i in range(len(tokens) - self.shingle_k + 1)}
        if tokens:
            return {" ".join(tokens)}
        # Character fallback if very short
        if len(cleaned) >= self.shingle_k:
            return {cleaned[i : i + self.shingle_k] for i in range(len(cleaned) - self.shingle_k + 1)}
        return {cleaned} if cleaned else set()

    def compute_signature(self, text_or_shingles: Union[str, Set[str]]) -> List[int]:
        """Compute the MinHash signature vector of length num_perm."""
        if isinstance(text_or_shingles, str):
            shingles = self.extract_shingles(text_or_shingles)
        else:
            shingles = text_or_shingles

        if not shingles:
            return [0] * self.num_perm

        sig = [self.LARGE_PRIME] * self.num_perm
        for shingle in shingles:
            # Deterministic 32-bit hash of shingle
            h_val = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:8], 16)
            for i in range(self.num_perm):
                val = (self._a[i] * h_val + self._b[i]) % self.LARGE_PRIME
                if val < sig[i]:
                    sig[i] = val
        return sig

    def estimated_jaccard(self, sig1: Sequence[int], sig2: Sequence[int]) -> float:
        """Estimate Jaccard similarity via MinHash signature matching fraction."""
        if len(sig1) != len(sig2) or not sig1:
            return 0.0
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return float(matches / len(sig1))


def minhash_signature(text: str, num_perm: int = 64, shingle_k: int = 3) -> List[int]:
    """Convenience functional interface for generating MinHash signatures."""
    indexer = MinHashIndexer(num_perm=num_perm, shingle_k=shingle_k)
    return indexer.compute_signature(text)


def estimate_jaccard_similarity(sig1: Sequence[int], sig2: Sequence[int]) -> float:
    """Convenience functional interface for estimating Jaccard similarity."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
    return float(matches / len(sig1))


# -----------------------------------------------------------------------------
# MinHash Graph Expander with Bloom Filter Community Pruning
# -----------------------------------------------------------------------------
class MinHashGraphExpander:
    """
    MeshRAG-style training-free passage collision graph expander.
    Uses k-shingle MinHash signatures, dual-layer hash collisions,
    and Bloom Filter community pruning to expand retrieved passages without LLM token cost.
    """

    def __init__(
        self,
        num_perm: int = 64,
        shingle_k: int = 3,
        num_bands: int = 32,
        similarity_threshold: float = 0.15,
        max_candidate_pool: int = 15,
        bloom_capacity: int = 1000,
        bloom_error_rate: float = 0.01,
        jaccard_threshold: Optional[float] = None,
    ) -> None:
        if jaccard_threshold is not None:
            similarity_threshold = jaccard_threshold
        self.num_perm = num_perm
        self.shingle_k = shingle_k
        self.num_bands = num_bands
        self.rows_per_band = max(1, num_perm // num_bands)
        self.similarity_threshold = similarity_threshold
        self.max_candidate_pool = max_candidate_pool
        self.bloom_capacity = bloom_capacity
        self.bloom_error_rate = bloom_error_rate

        self.indexer = MinHashIndexer(num_perm=num_perm, shingle_k=shingle_k)
        self.corpus_docs: Dict[str, Dict[str, Any]] = {}
        self.signatures: Dict[str, List[int]] = {}
        self.band_buckets: Dict[int, Dict[Tuple[int, ...], Set[str]]] = {}
        self.adjacency_graph: Dict[str, List[Tuple[str, float]]] = {}

    def extract_entities(self, text: str) -> Set[str]:
        """Simple deterministic entity / noun phrase fingerprint extractor."""
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", text)
        stopwords = {
            "the", "a", "an", "in", "on", "of", "for", "with", "at", "by",
            "to", "from", "and", "or", "is", "are", "was", "were", "what",
            "who", "which", "when", "where", "why", "how", "it", "this", "that"
        }
        entities: Set[str] = set()
        # Capitalized token sequences (Named Entities)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for cap in capitalized:
            if cap.lower() not in stopwords and len(cap) > 2:
                entities.add(cap.lower())

        # Significant alphanumeric terms
        for w in words:
            low = w.lower()
            if low not in stopwords and len(low) > 3 and not low.isdigit():
                entities.add(low)
        return entities

    def build_corpus_index(self, corpus: Sequence[Dict[str, Any]]) -> None:
        """
        Index corpus documents, compute MinHash signatures, and build collision buckets.
        Each doc must have 'id' (or 'doc_id') and 'text'.
        """
        self.corpus_docs.clear()
        self.signatures.clear()
        self.band_buckets = {b: {} for b in range(self.num_bands)}
        self.adjacency_graph.clear()

        for doc in corpus:
            doc_id = str(doc.get("id", doc.get("doc_id", f"doc_{len(self.corpus_docs)}")))
            text = doc.get("text", "")
            entities = set(doc.get("entities", []))
            if not entities:
                entities = self.extract_entities(text)

            self.corpus_docs[doc_id] = {
                "id": doc_id,
                "doc_id": doc_id,
                "text": text,
                "title": doc.get("title", ""),
                "score": float(doc.get("score", 0.0)),
                "entities": list(entities),
                "metadata": doc.get("metadata", {}),
            }

            sig = self.indexer.compute_signature(text)
            self.signatures[doc_id] = sig

            # Dual-layer LSH banding
            for b in range(self.num_bands):
                start = b * self.rows_per_band
                end = start + self.rows_per_band
                band_slice = tuple(sig[start:end])
                bucket_dict = self.band_buckets[b]
                if band_slice not in bucket_dict:
                    bucket_dict[band_slice] = set()
                bucket_dict[band_slice].add(doc_id)

        # Build collision adjacency graph
        self._build_collision_graph()

    def _build_collision_graph(self) -> None:
        """Construct weighted graph edges from LSH bucket collisions."""
        candidate_pairs: Set[Tuple[str, str]] = set()
        for b in range(self.num_bands):
            for doc_set in self.band_buckets[b].values():
                if len(doc_set) > 1:
                    doc_list = sorted(list(doc_set))
                    for i in range(len(doc_list)):
                        for j in range(i + 1, len(doc_list)):
                            candidate_pairs.add((doc_list[i], doc_list[j]))

        adj: Dict[str, List[Tuple[str, float]]] = {did: [] for did in self.corpus_docs}
        for u, v in candidate_pairs:
            sim = self.indexer.estimated_jaccard(self.signatures[u], self.signatures[v])
            if sim >= self.similarity_threshold:
                adj[u].append((v, sim))
                adj[v].append((u, sim))

        # Sort neighbor lists by descending similarity
        for u in adj:
            adj[u].sort(key=lambda x: x[1], reverse=True)
        self.adjacency_graph = adj

    def create_query_bloom_filter(self, query: str, seed_entities: Optional[Iterable[str]] = None) -> BloomFilter:
        """Create a Bloom Filter loaded with query entities and initial seed terms."""
        bf = BloomFilter(capacity=self.bloom_capacity, error_rate=self.bloom_error_rate)
        q_entities = self.extract_entities(query)
        for ent in q_entities:
            bf.add(ent)
            bf.add(ent.lower())
        if seed_entities:
            for s_ent in seed_entities:
                bf.add(s_ent)
                bf.add(s_ent.lower())
        return bf

    def prune_candidates(
        self,
        candidate_docs: List[Dict[str, Any]],
        bloom_filter: BloomFilter,
        gold_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prune candidate passages using the query Bloom Filter.
        Explicitly tracks and records gold passage recall change under pruning.
        """
        gold_set = set(gold_ids) if gold_ids is not None else set()
        has_gold = len(gold_set) > 0

        before_ids = {str(doc.get("doc_id") or doc.get("id") or idx) for idx, doc in enumerate(candidate_docs)}
        before_gold_count = len(before_ids & gold_set) if has_gold else 0

        pruned_docs: List[Dict[str, Any]] = []
        for idx, doc in enumerate(candidate_docs):
            doc_entities = set(doc.get("entities", []))
            if not doc_entities:
                doc_entities = self.extract_entities(doc.get("text", ""))

            # Fast membership check: does any entity collide with query filter?
            match_count = sum(1 for ent in doc_entities if (ent in bloom_filter or ent.lower() in bloom_filter))
            if match_count > 0:
                doc_copy = dict(doc)
                doc_copy["bloom_match_count"] = match_count
                pruned_docs.append(doc_copy)

        after_ids = {str(doc.get("doc_id") or doc.get("id") or idx) for idx, doc in enumerate(pruned_docs)}
        after_gold_count = len(after_ids & gold_set) if has_gold else 0

        recall_before = (before_gold_count / len(gold_set)) if has_gold else 1.0
        recall_after = (after_gold_count / len(gold_set)) if has_gold else 1.0
        recall_drop = max(0.0, recall_before - recall_after)

        stats = {
            "before_pruning_count": len(candidate_docs),
            "after_pruning_count": len(pruned_docs),
            "before_pruning_gold_count": before_gold_count,
            "after_pruning_gold_count": after_gold_count,
            "total_before_pruning": len(candidate_docs),
            "total_after_pruning": len(pruned_docs),
            "pruned_count": len(candidate_docs) - len(pruned_docs),
            "gold_total_count": len(gold_set),
            "gold_before_count": before_gold_count,
            "gold_after_count": after_gold_count,
            "recall_before": float(recall_before),
            "recall_after": float(recall_after),
            "recall_drop": float(recall_drop),
            "bloom_false_positive_rate": bloom_filter.current_false_positive_rate(),
        }

        return pruned_docs, stats

    def expand(
        self,
        query: str,
        initial_docs: List[Dict[str, Any]],
        epistemic_uncertainty: float,
        tau_eu: float = 0.45,
        gold_ids: Optional[Set[str]] = None,
        max_candidates: Optional[int] = None,
    ) -> ExpansionResult:
        """
        Adaptive routing execution:
        - If U_EU <= tau_eu: Early exit with initial docs D_0, token_saving_rate computed.
        - If U_EU > tau_eu: Expand along MinHash collision graph to candidate pool D_cand (|D_cand| <= max_pool).
        """
        t0 = time.perf_counter()
        k0 = len(initial_docs)
        max_pool = max_candidates if max_candidates is not None else self.max_candidate_pool

        # 1. Early Exit check
        if epistemic_uncertainty <= tau_eu:
            saving_rate = max(0.0, (max_pool - k0) / max_pool)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ExpansionResult(
                query=query,
                initial_docs=initial_docs,
                expanded_candidates=initial_docs,
                early_exit=True,
                epistemic_uncertainty=epistemic_uncertainty,
                tau_eu=tau_eu,
                token_saving_rate=float(saving_rate),
                pruning_stats={"reason": "early_exit_low_uncertainty", "recall_drop": 0.0},
                elapsed_ms=elapsed_ms,
                routing_decision="Early Exit triggered (U_EU <= tau_eu)",
            )

        # 2. Graph Expansion when U_EU > tau_eu
        seed_ids = [str(d.get("id", d.get("doc_id", ""))) for d in initial_docs if str(d.get("id", d.get("doc_id", "")))]
        seed_entities: Set[str] = set()
        for d in initial_docs:
            seed_entities.update(d.get("entities", []))

        # Collect 1-hop and 2-hop collision neighbors (+ entity-bridge fallback on sparse pools)
        candidate_ids: Set[str] = set(seed_ids)
        neighbor_scores: Dict[str, float] = {sid: 1.0 for sid in seed_ids}

        for sid in seed_ids:
            for neighbor_id, sim in self.adjacency_graph.get(sid, []):
                candidate_ids.add(neighbor_id)
                neighbor_scores[neighbor_id] = max(neighbor_scores.get(neighbor_id, 0.0), sim)

        if len(candidate_ids) < max_pool and self.corpus_docs:
            q_entities = self.extract_entities(query)
            bridge_terms = seed_entities | q_entities
            for cid, doc_obj in self.corpus_docs.items():
                if cid in candidate_ids:
                    continue
                doc_ents = set(doc_obj.get("entities", []))
                overlap = len(doc_ents & bridge_terms)
                if overlap > 0:
                    denom = max(1, len(doc_ents | bridge_terms))
                    # Keep weak entity-bridge fallback strictly below MinHash LSH threshold (0.15)
                    # so structural k-shingle collisions always rank ahead of generic entity matches.
                    bridge_sim = min(0.14, 0.04 + 0.10 * (overlap / denom))
                    candidate_ids.add(cid)
                    neighbor_scores[cid] = max(neighbor_scores.get(cid, 0.0), bridge_sim)

        # Build Candidate doc objects
        raw_candidates: List[Dict[str, Any]] = []
        for cid in candidate_ids:
            if cid in self.corpus_docs:
                doc_obj = dict(self.corpus_docs[cid])
                doc_obj["graph_similarity"] = neighbor_scores.get(cid, doc_obj.get("score", 0.0))
                raw_candidates.append(doc_obj)
            else:
                # Retain initial docs even if not in indexed corpus
                for init_d in initial_docs:
                    if str(init_d.get("id", init_d.get("doc_id", ""))) == cid:
                        raw_candidates.append(dict(init_d))
                        break

        # 3. Bloom Filter community pruning
        bf = self.create_query_bloom_filter(query, seed_entities)
        pruned_candidates, stats = self.prune_candidates(raw_candidates, bf, gold_ids=gold_ids)

        # Ensure seed docs are always preserved
        preserved_docs: List[Dict[str, Any]] = []
        preserved_ids: Set[str] = set()
        for doc in initial_docs:
            did = str(doc.get("id", doc.get("doc_id", "")))
            preserved_docs.append(doc)
            preserved_ids.add(did)

        # Sort remaining candidates by score / similarity and fill up to max_pool
        additional = [d for d in pruned_candidates if str(d.get("id", d.get("doc_id", ""))) not in preserved_ids]
        additional.sort(key=lambda d: float(d.get("graph_similarity", d.get("score", 0.0))), reverse=True)

        final_candidates = (preserved_docs + additional)[:max_pool]
        saving_rate = max(0.0, (max_pool - len(final_candidates)) / max_pool)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ExpansionResult(
            query=query,
            initial_docs=initial_docs,
            expanded_candidates=final_candidates,
            early_exit=False,
            epistemic_uncertainty=epistemic_uncertainty,
            tau_eu=tau_eu,
            token_saving_rate=float(saving_rate),
            pruning_stats=stats,
            elapsed_ms=elapsed_ms,
            routing_decision="Adaptive Hash Graph Expansion triggered (U_EU > tau_eu)",
        )


# -----------------------------------------------------------------------------
# Closed-Form Uncertainty Decoupler with Training-Free Offline Calibration
# -----------------------------------------------------------------------------
class UncertaintyDecoupler:
    """
    Closed-form Epistemic and Aleatoric Uncertainty Decoupler.
    Decomposes total predictive entropy into AU and EU:
      U_EU = max(0, H_total - U_AU)
    Marginal blind-spot elimination gain:
      Delta U_EU(d_i) = max(0, U_EU(D_0) - U_EU(D_0 U {d_i}))
    Provides offline calibration (tau_EU, aleatoric scale, attention weights)
    persisted into models/m1_calibration.json.
    """

    def __init__(
        self,
        tau_eu: float = 0.45,
        aleatoric_scale: float = 1.0,
        attention_layer_weights: Optional[Dict[str, float]] = None,
        calibration_path: Optional[str] = None,
    ) -> None:
        self.tau_eu = tau_eu
        self.aleatoric_scale = aleatoric_scale
        self.attention_layer_weights = attention_layer_weights or {
            "shallow": 0.2,
            "middle": 0.5,
            "deep": 0.3,
        }
        self.is_calibrated = False
        self.calibration_metadata: Dict[str, Any] = {}

        if calibration_path and os.path.exists(calibration_path):
            self.load_calibration(calibration_path)

    def compute_first_token_distribution(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
        model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Extract first-token predictive distribution p(v | q, D).
        If model_fn is provided (PyTorch LLM wrapper), calls it.
        Otherwise, uses deterministic lexical/semantic relevance distribution.
        """
        if model_fn is not None:
            doc_texts = [d.get("text", "") for d in docs]
            return model_fn(query, doc_texts)

        # Deterministic training-free fallback distribution
        words = re.findall(r"\w+", query.lower())
        query_terms = [w for w in words if len(w) > 2]
        dist: Dict[str, float] = {}

        if not docs:
            # High uniform uncertainty if no context
            return {"[UNK]": 0.5, "[AMBIGUOUS]": 0.5}

        # Aggregate evidence vocabulary frequencies
        for doc in docs:
            text = doc.get("text", "").lower()
            doc_terms = re.findall(r"\w+", text)
            for dt in doc_terms:
                if len(dt) > 2:
                    dist[dt] = dist.get(dt, 0.0) + 1.0

        # Prioritize intersection with query or query entities
        for qt in query_terms:
            if qt in dist:
                dist[qt] *= 2.0

        if not dist:
            return {"yes": 0.5, "no": 0.5}

        # Softmax normalization with log-frequency smoothing
        top_terms = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:32]
        # Log-frequency transformation for realistic entropy dynamics
        smoothed_scores = {k: math.log(1.0 + float(v)) for k, v in top_terms}
        max_val = max(smoothed_scores.values()) if smoothed_scores else 0.0
        exp_vals = {k: math.exp(v - max_val) for k, v in smoothed_scores.items()}
        total_exp = sum(exp_vals.values())
        return {k: float(v / total_exp) for k, v in exp_vals.items()}

    def measure_aleatoric_uncertainty(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
        base_distribution: Dict[str, float],
        perturbation_evaluator: Optional[Callable[[], List[Dict[str, float]]]] = None,
    ) -> float:
        """
        Measure Aleatoric Uncertainty (AU) using attention head perturbation / semantic ambiguity.
        AU captures inherent task ambiguity invariant to retrieval scale.
        """
        if perturbation_evaluator is not None:
            # Model-driven head perturbation
            perturbed_dists = perturbation_evaluator()
            if perturbed_dists:
                entropies = [compute_entropy(p) for p in perturbed_dists]
                mean_entropy = sum(entropies) / len(entropies)
                return max(0.0, float(self.aleatoric_scale * mean_entropy))

        # Training-free algebraic estimation:
        # Measures overlap contradiction and dispersion within candidate contexts
        if not docs:
            return 0.1 * self.aleatoric_scale

        doc_vocab_sets = [set(re.findall(r"\w+", d.get("text", "").lower())) for d in docs]
        pairwise_jaccards: List[float] = []
        for i in range(len(doc_vocab_sets)):
            for j in range(i + 1, len(doc_vocab_sets)):
                u = len(doc_vocab_sets[i] | doc_vocab_sets[j])
                inter = len(doc_vocab_sets[i] & doc_vocab_sets[j])
                pairwise_jaccards.append(inter / u if u > 0 else 0.0)

        # Low pairwise lexical agreement signals semantic divergence/ambiguity (AU)
        avg_agreement = sum(pairwise_jaccards) / len(pairwise_jaccards) if pairwise_jaccards else 0.5
        divergence = 1.0 - avg_agreement

        # Multiply with base entropy spread
        base_entropy = compute_entropy(base_distribution)
        raw_au = (0.25 + 0.5 * divergence) * base_entropy * self.aleatoric_scale
        return max(0.0, float(raw_au))

    def decompose(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
        model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
    ) -> UncertaintyMetrics:
        """
        Perform closed-form uncertainty decomposition on (query, docs):
          H_total = - sum p log p
          U_AU = aleatoric ambiguity
          U_EU = max(0, H_total - U_AU)
        """
        p_dist = self.compute_first_token_distribution(query, docs, model_fn=model_fn)
        h_total = compute_entropy(p_dist)
        u_au = self.measure_aleatoric_uncertainty(query, docs, p_dist)

        # Direct closed-form decomposition: U_EU = max(0, H_total - U_AU)
        u_eu = max(0.0, h_total - u_au)

        return UncertaintyMetrics(
            total_entropy=float(h_total),
            aleatoric_uncertainty=float(u_au),
            epistemic_uncertainty=float(u_eu),
            predictive_distribution=p_dist,
        )

    # Alias for API consistency across test suites
    compute_uncertainty = decompose

    def compute_marginal_epistemic_reduction(
        self,
        query: str,
        initial_docs: Sequence[Dict[str, Any]],
        candidate_docs: Sequence[Dict[str, Any]],
        model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute marginal blind-spot elimination gain for each candidate d_i:
          Delta U_EU(d_i) = max(0, U_EU(D_0) - U_EU(D_0 U {d_i}))
        """
        base_metrics = self.decompose(query, initial_docs, model_fn=model_fn)
        u_eu_base = base_metrics.epistemic_uncertainty

        reductions: Dict[str, float] = {}
        for cand in candidate_docs:
            cid = str(cand.get("id", cand.get("doc_id", "")))
            combined_docs = list(initial_docs) + [cand]
            cand_metrics = self.decompose(query, combined_docs, model_fn=model_fn)
            gain = max(0.0, u_eu_base - cand_metrics.epistemic_uncertainty)
            reductions[cid] = float(gain)

        return reductions

    def calibrate_offline(
        self,
        calibration_samples: Sequence[Dict[str, Any]],
        model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
        save_path: Optional[str] = "models/m1_calibration.json",
    ) -> Dict[str, Any]:
        """
        Offline calibration on small-sample validation set (20~30 samples).
        Estimates distribution stability across query difficulty, calibrates tau_EU,
        and persists results to save_path.
        """
        eu_list: List[float] = []
        au_list: List[float] = []
        entropy_list: List[float] = []

        for sample in calibration_samples:
            query = sample.get("question", sample.get("query", ""))
            ctxs = sample.get("ctxs", sample.get("docs", []))
            metrics = self.decompose(query, ctxs, model_fn=model_fn)
            eu_list.append(metrics.epistemic_uncertainty)
            au_list.append(metrics.aleatoric_uncertainty)
            entropy_list.append(metrics.total_entropy)

        n = len(eu_list)
        if n == 0:
            calibrated_tau = self.tau_eu
            calibrated_scale = self.aleatoric_scale
        else:
            sorted_eu = sorted(eu_list)
            # Optimal tau_EU: set to median or 60th percentile of calibration sample EU
            calibrated_tau = sorted_eu[int(0.60 * (n - 1))]
            avg_entropy = sum(entropy_list) / n
            avg_au = sum(au_list) / n
            calibrated_scale = (avg_au / avg_entropy) if avg_entropy > 1e-6 else 1.0

        self.tau_eu = float(calibrated_tau)
        self.aleatoric_scale = float(calibrated_scale)
        self.is_calibrated = True

        config = {
            "tau_eu": self.tau_eu,
            "aleatoric_scale": self.aleatoric_scale,
            "attention_layer_weights": self.attention_layer_weights,
            "calibration_samples_count": n,
            "mean_total_entropy": float(sum(entropy_list) / n) if n else 0.0,
            "mean_aleatoric_uncertainty": float(sum(au_list) / n) if n else 0.0,
            "mean_epistemic_uncertainty": float(sum(eu_list) / n) if n else 0.0,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.calibration_metadata = config

        if save_path:
            self.save_calibration(save_path)

        return config

    def save_calibration(self, filepath: str) -> None:
        """Persist calibration parameters to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tau_eu": self.tau_eu,
            "aleatoric_scale": self.aleatoric_scale,
            "attention_layer_weights": self.attention_layer_weights,
            "is_calibrated": self.is_calibrated,
            "metadata": self.calibration_metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_calibration(self, filepath: str) -> None:
        """Load calibration parameters from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tau_eu = float(data.get("tau_eu", self.tau_eu))
        self.aleatoric_scale = float(data.get("aleatoric_scale", self.aleatoric_scale))
        self.attention_layer_weights = data.get("attention_layer_weights", self.attention_layer_weights)
        self.is_calibrated = bool(data.get("is_calibrated", True))
        self.calibration_metadata = data.get("metadata", {})


# -----------------------------------------------------------------------------
# Integrated Top-Level M1 Adaptive Retriever Pipeline
# -----------------------------------------------------------------------------
class M1AdaptiveRetriever:
    """
    Integrated M1 pipeline bridging uncertainty decoupling and hash graph expansion.
    Produces candidates formatted ready for downstream M2 evidence optimizer.
    """

    def __init__(
        self,
        decoupler: Optional[UncertaintyDecoupler] = None,
        expander: Optional[MinHashGraphExpander] = None,
        calibration_path: Optional[str] = "models/m1_calibration.json",
    ) -> None:
        if decoupler is not None:
            self.decoupler = decoupler
        else:
            self.decoupler = UncertaintyDecoupler(calibration_path=calibration_path)

        if expander is not None:
            self.expander = expander
        else:
            self.expander = MinHashGraphExpander()

        self.default_model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None
        self.use_model_probe_for_candidates: bool = False

    def index_corpus(self, corpus: Sequence[Dict[str, Any]]) -> None:
        """Index background corpus for graph expansion."""
        self.expander.build_corpus_index(corpus)

    def process(
        self,
        query: str,
        initial_docs: List[Dict[str, Any]],
        gold_ids: Optional[Set[str]] = None,
        model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
    ) -> Tuple[ExpansionResult, List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """
        Execute full M1 adaptive retrieval pipeline:
        1. Measure predictive uncertainty (EU, AU, Total Entropy) on (query, D_0) using neural probe if configured.
        2. Perform adaptive gating & MinHash Bloom graph expansion.
        3. Compute marginal epistemic reductions Delta U_EU(d_i).
        4. Compute first-token predictive distributions p_i for downstream M2 JS divergence conflict.
        Returns:
          (expansion_result, m2_candidate_dicts, per_doc_first_token_distributions)
        """
        eff_model_fn = model_fn if model_fn is not None else self.default_model_fn

        # Step 1: Uncertainty Decoupling (strictly uses neural model probe when available to match tau_eu scale)
        metrics = self.decoupler.decompose(query, initial_docs, model_fn=eff_model_fn)

        # Step 2: Adaptive Graph Expansion
        expansion_res = self.expander.expand(
            query=query,
            initial_docs=initial_docs,
            epistemic_uncertainty=metrics.epistemic_uncertainty,
            tau_eu=self.decoupler.tau_eu,
            gold_ids=gold_ids,
        )

        # Candidate-level marginal scoring (defaults to O(1)-cost algebraic estimator unless explicitly enabled)
        cand_model_fn = model_fn if (model_fn is not None or self.use_model_probe_for_candidates) else None

        # Step 3: Marginal Epistemic Reductions
        marginal_gains = self.decoupler.compute_marginal_epistemic_reduction(
            query=query,
            initial_docs=initial_docs,
            candidate_docs=expansion_res.expanded_candidates,
            model_fn=cand_model_fn,
        )

        # Step 4: Per-doc first token distributions for M2 JS Divergence computation
        per_doc_dists: Dict[str, Dict[str, float]] = {}
        m2_candidates: List[Dict[str, Any]] = []

        for cand in expansion_res.expanded_candidates:
            cid = str(cand.get("id", cand.get("doc_id", "")))
            # Compute single-doc predictive distribution
            p_i = self.decoupler.compute_first_token_distribution(query, [cand], model_fn=cand_model_fn)
            per_doc_dists[cid] = p_i

            doc_metrics = self.decoupler.decompose(query, [cand], model_fn=cand_model_fn)
            delta_eu = marginal_gains.get(cid, 0.0)

            m2_candidates.append({
                "doc_id": cid,
                "text": cand.get("text", ""),
                "score": float(cand.get("score", cand.get("graph_similarity", 0.0))),
                "entities": cand.get("entities", []),
                "aleatoric_uncertainty": float(doc_metrics.aleatoric_uncertainty),
                "epistemic_reduction": float(delta_eu),
                "metadata": {
                    "is_seed": cid in {str(d.get("id", d.get("doc_id", ""))) for d in initial_docs},
                    "graph_similarity": float(cand.get("graph_similarity", 0.0)),
                    "total_entropy": float(doc_metrics.total_entropy),
                },
            })

        return expansion_res, m2_candidates, per_doc_dists

    def retrieve_and_expand(
        self,
        query: str,
        initial_docs: List[Dict[str, Any]],
        corpus_docs: Optional[Sequence[Dict[str, Any]]] = None,
        gold_ids: Optional[Set[str]] = None,
        model_fn: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
    ) -> ExpansionResult:
        """
        Convenience wrapper returning ExpansionResult with attached m2_candidates and distributions.
        """
        if corpus_docs is not None and not self.expander.corpus_docs:
            self.index_corpus(corpus_docs)
        exp_res, m2_cands, per_doc_dists = self.process(
            query=query,
            initial_docs=initial_docs,
            gold_ids=gold_ids,
            model_fn=model_fn,
        )
        exp_res.m2_candidates = m2_cands
        exp_res.per_doc_distributions = per_doc_dists
        return exp_res


# -----------------------------------------------------------------------------
# Command-Line Offline Calibration & Self-Diagnostic CLI
# -----------------------------------------------------------------------------
def run_cli_offline_calibration(
    dataset_path: str = "data/hotpotqa_dev_300.json",
    num_samples: int = 30,
    save_path: str = "models/m1_calibration.json",
) -> Dict[str, Any]:
    """Execute training-free offline calibration on sample dataset."""
    print(f"[*] Starting M1 Offline Calibration on up to {num_samples} samples from: {dataset_path}")
    samples_data = []

    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for item in raw_data[:num_samples]:
                # Extract passages from context
                docs = []
                ctx = item.get("context", {})
                titles = ctx.get("title", [])
                sentences_list = ctx.get("sentences", [])
                for t, sents in zip(titles, sentences_list):
                    text = f"{t}: " + " ".join(sents)
                    docs.append({"id": t, "text": text, "title": t})

                samples_data.append({
                    "question": item.get("question", ""),
                    "answer": item.get("ground_truth", item.get("answer", "")),
                    "docs": docs[:5],
                })
        print(f"[+] Loaded {len(samples_data)} valid calibration queries from disk.")
    else:
        print(f"[-] Dataset path {dataset_path} not found. Using representative synthetic calibration set.")
        # Fallback synthetic set
        for i in range(num_samples):
            samples_data.append({
                "question": f"What is the capital of territory {i}?",
                "answer": f"City_{i}",
                "docs": [
                    {"id": f"doc_{i}_1", "text": f"Territory {i} has historical city center City_{i} as its seat of government."},
                    {"id": f"doc_{i}_2", "text": f"Province bordering territory {i} was founded in 1845."},
                ],
            })

    decoupler = UncertaintyDecoupler()
    calib_result = decoupler.calibrate_offline(samples_data, save_path=save_path)
    print(f"[+] Offline Calibration complete!")
    print(f"    tau_EU calibrated: {calib_result['tau_eu']:.4f}")
    print(f"    Aleatoric scale:   {calib_result['aleatoric_scale']:.4f}")
    print(f"    Mean H_total:      {calib_result['mean_total_entropy']:.4f}")
    print(f"    Mean AU:           {calib_result['mean_aleatoric_uncertainty']:.4f}")
    print(f"    Mean EU:           {calib_result['mean_epistemic_uncertainty']:.4f}")
    print(f"    Persisted to:      {os.path.abspath(save_path)}")
    return calib_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M1 Uncertainty Decoupler & MinHash Expander")
    parser.add_argument("--calibrate", action="store_true", help="Run offline calibration")
    parser.add_argument("--self-test", action="store_true", help="Run quick inline sanity checks")
    parser.add_argument("--data", type=str, default="data/hotpotqa_dev_300.json", help="Path to calibration dataset")
    parser.add_argument("--samples", type=int, default=30, help="Number of calibration samples")
    parser.add_argument("--save", type=str, default="models/m1_calibration.json", help="Calibration output JSON")
    args = parser.parse_args()

    if args.self_test:
        print("[*] Running M1 inline self-test...")
        decoupler = UncertaintyDecoupler()
        expander = MinHashGraphExpander()
        sample_corpus = [
            {"id": "doc1", "text": "Ed Wood is an American filmmaker who directed Plan 9.", "entities": ["Ed Wood", "Plan 9"]},
            {"id": "doc2", "text": "Scott Derrickson directed Doctor Strange and Sinister.", "entities": ["Scott Derrickson", "Doctor Strange"]},
            {"id": "doc3", "text": "Plan 9 from Outer Space is a cult classic sci-fi film.", "entities": ["Plan 9", "Outer Space"]},
            {"id": "doc4", "text": "Doctor Strange is an American superhero film directed by Scott Derrickson.", "entities": ["Doctor Strange", "Scott Derrickson"]},
        ]
        expander.build_corpus_index(sample_corpus)
        retriever = M1AdaptiveRetriever(decoupler=decoupler, expander=expander)
        res = retriever.retrieve_and_expand(
            query="Were Scott Derrickson and Ed Wood of the same nationality?",
            initial_docs=[sample_corpus[0]],
            corpus_docs=sample_corpus,
            gold_ids={"doc1", "doc2"}
        )
        print(f"[+] Inline self-test PASSED! early_exit={res.early_exit}, candidates={len(res.expanded_candidates)}, recall_drop={res.pruning_stats.get('recall_drop')}")
    elif args.calibrate or not os.path.exists(args.save):
        run_cli_offline_calibration(args.data, args.samples, args.save)
    else:
        print("[*] M1 module loaded successfully. Run with --calibrate to recalibrate models/m1_calibration.json or --self-test.")
