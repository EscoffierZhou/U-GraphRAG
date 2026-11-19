"""
bge_reranker_operator.py
========================
Isolated 568M Cross-Encoder reranker (`BAAI/bge-reranker-v2-m3`) wrapper for Stage-1
scoring & full VRAM teardown prior to loading the 7B/8B generator on RTX 4090 (24 GB).
Supports ModelScope / HuggingFace mirror fallback and fast deterministic fallback under `dry_run`.
"""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from scripts.phase2.common import MODELS_DIR
from scripts.phase2.wp2_m2_ablation import _select_mmr


def _resolve_reranker_source() -> str:
    """Resolve `BAAI/bge-reranker-v2-m3` locally, via ModelScope inside China, or HuggingFace."""
    candidates = [MODELS_DIR / "bge-reranker-v2-m3", Path("/root/autodl-tmp/models/bge-reranker-v2-m3")]
    for cand in candidates:
        if cand.exists() and (any(cand.glob("*.safetensors")) or any(cand.glob("*.bin"))):
            return str(cand)

    local_dir = MODELS_DIR / "bge-reranker-v2-m3"
    is_overseas = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    if not is_overseas:
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_dl

            local_dir.mkdir(parents=True, exist_ok=True)
            for ms_repo in ("BAAI/bge-reranker-v2-m3", "Xorbits/bge-reranker-v2-m3"):
                try:
                    ms_dl(ms_repo, local_dir=str(local_dir))
                    if any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin")):
                        return str(local_dir)
                except Exception:
                    continue
        except Exception:
            pass

    try:
        from huggingface_hub import snapshot_download as hf_dl

        local_dir.mkdir(parents=True, exist_ok=True)
        hf_dl(repo_id="BAAI/bge-reranker-v2-m3", local_dir=str(local_dir), tqdm_class=None)
        if any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin")):
            return str(local_dir)
    except Exception:
        pass
    return "BAAI/bge-reranker-v2-m3"


class IsolatedBGEReranker:
    """Loads `BAAI/bge-reranker-v2-m3` cleanly in FP16 and explicitly releases VRAM after scoring."""

    def __init__(self, dry_run: bool = False, device: Optional[str] = None) -> None:
        self.dry_run = bool(dry_run)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        if self.dry_run or self.model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        src = _resolve_reranker_source()
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(src, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def score_pool(self, question: str, docs: Sequence[Dict[str, Any]], batch_size: int = 16) -> List[float]:
        if not docs:
            return []
        if self.dry_run or self.model is None:
            q_toks = set(re.findall(r"[a-z0-9]+", question.lower()))
            out = []
            for d in docs:
                dt = set(re.findall(r"[a-z0-9]+", f"{d.get('title', '')} {d.get('text', '')}".lower()))
                lex = len(q_toks & dt) / max(1, len(q_toks))
                out.append(round(0.65 * float(d.get("score", 0.5)) + 0.35 * lex, 6))
            return out

        pairs = [[question, f"{d.get('title', '')}: {d.get('text', '')}"] for d in docs]
        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                toks = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**toks).logits.view(-1).float()
                probs = torch.sigmoid(logits).cpu().tolist()
                scores.extend(float(p) for p in probs)
        return scores

    def select_reranked(
        self,
        question: str,
        docs: Sequence[Dict[str, Any]],
        top_k: int = 5,
        use_mmr: bool = False,
        precomputed_scores: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []
        sc = list(precomputed_scores) if precomputed_scores is not None else self.score_pool(question, docs)
        reranked = sorted(
            [{**dict(d), "rerank_score": float(s), "score": float(s), "calibrated_reliability": float(s)} for d, s in zip(docs, sc)],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        if not use_mmr:
            return reranked[: min(top_k, len(reranked))]
        return _select_mmr(reranked, k_target=min(top_k, len(reranked)))

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
