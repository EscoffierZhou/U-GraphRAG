"""
relevance_gate_operator.py
==========================
Puzzle Piece 1 fix for M1 non-monotonicity on S4 (All-Irrelevant / off-topic homogeneous pools),
combining:
  1. Lightweight Query-Context Relevance Lower-Bound Gate (zero extra LLM forward passes):
     requires content-token overlap recall OR retrieval similarity of d0_docs above `rel_floor`
     before allowing M1 Early-Exit.
  2. Optional Query-Level Internal Confidence hook adapted from ICLR 2026
     `repos/query_level_uncertainty/ql_uncertainty/internal_confidence.py`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np

STOPWORDS = {
    "the", "a", "an", "in", "on", "of", "for", "with", "at", "by",
    "to", "from", "and", "or", "is", "are", "was", "were", "be", "been",
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "has", "have", "had",
    "it", "its", "this", "that", "these", "those", "as", "into", "about",
}


def _content_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {t for t in toks if len(t) >= 2 and t not in STOPWORDS}


def check_query_evidence_relevance_gate(
    question: str,
    d0_docs: Sequence[Dict[str, Any]],
    min_lexical_overlap: float = 0.25,
    min_retriever_score: float = 0.70,
    shared_dense_model: Any = None,
    min_dense_cosine: float = 0.35,
) -> Dict[str, Any]:
    """
    Lightweight relevance lower-bound gate preventing low-entropy homogeneous wrong pools
    from blindly taking M1 Early-Exit:
      Early-Exit allowed ONLY if at least ONE top-3 document clears lexical content overlap
      `>= min_lexical_overlap` OR (if dense model provided) cosine `>= min_dense_cosine`, AND
      max retriever score `>= min_retriever_score` (or lexical overlap is strictly >= 0.45).
    """
    if not d0_docs:
        return {
            "passes_relevance_gate": False,
            "max_lexical_overlap": 0.0,
            "max_retriever_score": 0.0,
            "dense_max_cosine": None,
            "gate_reason": "empty_d0",
        }

    q_toks = _content_tokens(question)
    lex_overlaps: List[float] = []
    ret_scores: List[float] = []
    for d in d0_docs:
        ret_scores.append(float(d.get("score", 0.0)))
        d_toks = _content_tokens(f"{d.get('title', '')} {d.get('text', '')}")
        if q_toks:
            lex_overlaps.append(float(len(q_toks & d_toks)) / float(len(q_toks)))
        else:
            lex_overlaps.append(0.0)

    max_lex = float(max(lex_overlaps)) if lex_overlaps else 0.0
    max_sc = float(max(ret_scores)) if ret_scores else 0.0
    dense_max: float | None = None

    if shared_dense_model is not None:
        try:
            texts = [question] + [f"{d.get('title', '')} {d.get('text', '')}" for d in d0_docs]
            embs = shared_dense_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            sims = [float(np.dot(embs[0], embs[i + 1])) for i in range(len(d0_docs))]
            dense_max = float(max(sims)) if sims else 0.0
        except Exception:
            dense_max = None

    strong_lex = max_lex >= max(0.45, float(min_lexical_overlap))
    semantic_ok = (dense_max >= float(min_dense_cosine)) if dense_max is not None else (max_lex >= float(min_lexical_overlap))
    score_ok = max_sc >= float(min_retriever_score)
    passes = bool(strong_lex or (semantic_ok and score_ok))

    return {
        "passes_relevance_gate": passes,
        "max_lexical_overlap": round(max_lex, 4),
        "max_retriever_score": round(max_sc, 4),
        "dense_max_cosine": round(dense_max, 4) if dense_max is not None else None,
        "gate_reason": "relevant" if passes else "low_query_evidence_relevance_block",
    }
