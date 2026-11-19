"""
dataset_adapters.py
===================
Normalizes cloned public conflict benchmarks (`RAMDocs`, `NQ-Swap`) and synthetic
`S4 All-Irrelevant` starvation slices into the canonical U-GraphRAG schema:
  {
    "id": str,
    "question": str,
    "possible_answers": List[str],
    "wrong_answers": List[str],
    "pool": [{"doc_id", "title", "text", "score", "type", "is_gold", "is_synthetic_conflict"}],
    "gold_titles": List[str],
    "benchmark": str,
  }
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DATASETS_DIR = Path(__file__).resolve().parent


def _to_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    return [str(val).strip()]


def load_ramdocs_dataset(
    path: Optional[Path] = None,
    max_items: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load RAMDocs_test.jsonl (arXiv:2504.13079, 500 multi-doc conflict + ambiguity + noise queries).
    Assigns realistic descending mock retriever scores slightly randomized deterministically,
    while marking misinfo docs with `is_synthetic_conflict=True` ONLY for offline metrics.
    """
    src = Path(path) if path else (DATASETS_DIR / "RAMDocs_test.jsonl")
    rows: List[Dict[str, Any]] = []
    with open(src, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            raw = json.loads(line)
            rng = random.Random(int(hashlib.md5(f"ram|{seed}|{idx}".encode()).hexdigest()[:8], 16))
            raw_docs = list(raw.get("documents", []))
            rng.shuffle(raw_docs)
            pool: List[Dict[str, Any]] = []
            gold_titles: List[str] = []
            for j, doc in enumerate(raw_docs):
                dtype = str(doc.get("type", "noise")).lower()
                is_gold = dtype == "correct"
                is_conf = dtype == "misinfo"
                # Misinfo often scores competitively high in dense retrieval before conflict gating
                base_sc = 0.88 - 0.04 * j
                if is_conf:
                    base_sc = max(base_sc, 0.85 - 0.02 * j)
                title = f"RAMDoc_{idx}_{j}"
                if is_gold:
                    gold_titles.append(title)
                pool.append({
                    "doc_id": f"ram_{idx}_{j}",
                    "title": title,
                    "text": str(doc.get("text", "")),
                    "score": round(max(0.15, min(0.99, base_sc)), 4),
                    "type": dtype,
                    "is_gold": is_gold,
                    "is_synthetic_conflict": is_conf,
                })
            # Sort descending by initial retriever score (simulating standard retrieval order)
            pool.sort(key=lambda d: d["score"], reverse=True)
            rows.append({
                "id": f"ramdocs_{idx}",
                "question": str(raw["question"]),
                "possible_answers": _to_str_list(raw.get("gold_answers")) or [""],
                "wrong_answers": _to_str_list(raw.get("wrong_answers")),
                "pool": pool,
                "gold_titles": gold_titles,
                "benchmark": "ramdocs",
            })
            if max_items is not None and len(rows) >= max_items:
                break
    return rows


def load_nqswap_dataset(
    path: Optional[Path] = None,
    max_items: Optional[int] = None,
    mode: str = "dual_conflict",
    donor_distractors: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load DeCoRe's NQ-Swap parquet (`nqswap_dev.parquet`, 4746 counterfactual queries).
    Modes:
      - `sub_only`: only substituted counterfactual context `sub_context` -> gold = `sub_answer`
        (canonical parametric-vs-context loyalty test).
      - `dual_conflict`: pool contains `sub_context` (counterfactual target, gold=`sub_answer`),
        `org_context` (or parametric conflict doc), plus optional noise distractors -> tests
        whether RAG systems resist noisy/conflicting pools while adhering to target evidence.
    """
    import pandas as pd

    src = Path(path) if path else (DATASETS_DIR / "nqswap_dev.parquet")
    df = pd.read_parquet(src)
    limit = min(len(df), int(max_items)) if max_items is not None else len(df)
    distractor_texts = list(donor_distractors) if donor_distractors else [str(df.iloc[k]["org_context"]).replace("<P>", "").replace("</P>", "").strip() for k in range(min(len(df), 200))]

    rows: List[Dict[str, Any]] = []
    for idx in range(limit):
        rec = df.iloc[idx]
        q = str(rec["question"]).strip()
        if not q.endswith("?"):
            q = q + "?"
        sub_ctx = str(rec["sub_context"]).replace("<P>", "").replace("</P>", "").strip()
        org_ctx = str(rec["org_context"]).replace("<P>", "").replace("</P>", "").strip()
        sub_ans = _to_str_list(rec["sub_answer"]) or [""]
        org_ans = _to_str_list(rec["org_answer"])

        sub_doc = {
            "doc_id": f"nqswap_{idx}_sub",
            "title": f"NQSwap_Sub_{idx}",
            "text": sub_ctx,
            "score": 0.92,
            "type": "substituted_context",
            "is_gold": True,
            "is_synthetic_conflict": False,
        }
        if mode == "sub_only":
            pool = [sub_doc]
        else:
            org_doc = {
                "doc_id": f"nqswap_{idx}_org",
                "title": f"NQSwap_Orig_{idx}",
                "text": org_ctx,
                "score": 0.89,
                "type": "original_conflict",
                "is_gold": False,
                "is_synthetic_conflict": True,
            }
            rng = random.Random(int(hashlib.md5(f"nqswap|{seed}|{idx}".encode()).hexdigest()[:8], 16))
            noise_indices = [k for k in range(len(distractor_texts)) if k != idx]
            picked = rng.sample(noise_indices, k=min(3, len(noise_indices)))
            noise_docs = [
                {
                    "doc_id": f"nqswap_{idx}_n{j}",
                    "title": f"NQSwap_Noise_{k}",
                    "text": distractor_texts[k],
                    "score": round(0.76 - 0.06 * j, 4),
                    "type": "noise",
                    "is_gold": False,
                    "is_synthetic_conflict": False,
                }
                for j, k in enumerate(picked)
            ]
            pool = [sub_doc, org_doc] + noise_docs

        rows.append({
            "id": f"nqswap_{idx}",
            "question": q,
            "possible_answers": sub_ans,
            "wrong_answers": org_ans,
            "pool": pool,
            "gold_titles": [sub_doc["title"]],
            "benchmark": f"nqswap_{mode}",
        })
    return rows


def build_all_irrelevant_s4_slice(
    items: Sequence[Dict[str, Any]],
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Construct Knowledgeable-R1 S4 (All-Irrelevant / Missing Evidence) stress slice:
    replaces each item's candidate pool with topically homogeneous off-target passages
    from unrelated questions while keeping retriever scores moderately plausible (0.62..0.72),
    specifically testing whether M1 avoids low-entropy blind Early-Exit.
    """
    src = list(items[:max_items]) if max_items is not None else list(items)
    n = len(src)
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(src):
        if n <= 1:
            donor_pool = list(it.get("pool", []))
        else:
            # Pick two distant unrelated items to form a coherent but off-topic pool
            donor = src[(idx + max(1, n // 2)) % n]
            donor_pool = list(donor.get("pool", []))
        irrel_pool = []
        for j, d in enumerate(donor_pool[:8]):
            irrel_pool.append({
                **dict(d),
                "doc_id": f"s4_irrel_{it['id']}_{j}",
                "score": round(0.68 - 0.03 * j, 4),
                "is_gold": False,
                "is_synthetic_conflict": False,
                "type": "s4_all_irrelevant",
            })
        out.append({
            **dict(it),
            "id": f"s4_{it['id']}",
            "pool": irrel_pool,
            "benchmark": "s4_all_irrelevant",
            "is_unanswerable_slice": True,
        })
    return out
