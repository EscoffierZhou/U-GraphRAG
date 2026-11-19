"""
src/u_graphrag_pipeline.py
==========================
End-to-end evaluation pipeline for U-GraphRAG and comprehensive ablations.

Supports 8 standardized experimental configurations:
1. B0_Standard_RAG: Standard hybrid retrieval Top-K, sorted by retrieval score, greedy decoding.
2. B1_Naive_MST: Single-stage MST baseline without M2-A subset filtering (KG2RAG / Naive MST).
3. M1_Only: M1 epistemic uncertainty gated adaptive retrieval, then Top-K truncation by score.
4. M2_Greedy: M1 + M2-A greedy approximation subset selection + M2-B MST topological ordering.
5. M2_Exact_NoMST: M1 + M2-A exact combinatorial subset selection, but without MST (ordered by individual reliability).
6. M2_Full: M1 + M2-A exact combinatorial subset selection + M2-B MST linearization from max-weight edge root.
7. M2_Permutation_Test: Based on M2_Full gold subset, evaluates DFS order, Reverse order, and 5x random shuffles.
8. Full_UGraphRAG: M1 + M2_Full + M3 (Reliability-weighted contrastive calibrated decoding and abstention gate).

All metrics conform strictly to the research proposal schema:
EM, F1, sp_precision, sp_recall, sp_f1, ece, abstention_rate, token_saving_rate,
peak_cuda_allocated_mb, peak_cuda_reserved_mb, avg_latency_sec, avg_forward_calls.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

# Ensure local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from metrics_profiler import MetricsProfiler

try:
    from evaluation_toolkit import (
        compute_full_sample_metrics,
        compute_detailed_calibration,
        compute_selective_prediction_metrics,
        CheckpointManager,
        ExperimentTensorBoardLogger,
        DetailedReportGenerator,
    )
except ImportError:
    try:
        from src.evaluation_toolkit import (
            compute_full_sample_metrics,
            compute_detailed_calibration,
            compute_selective_prediction_metrics,
            CheckpointManager,
            ExperimentTensorBoardLogger,
            DetailedReportGenerator,
        )
    except ImportError:
        compute_full_sample_metrics = None
        compute_detailed_calibration = None
        compute_selective_prediction_metrics = None
        CheckpointManager = None
        ExperimentTensorBoardLogger = None
        DetailedReportGenerator = None

try:
    from baseline_rag import (
        exact_match_score as compute_exact_match,
        f1_score as compute_f1_score,
        normalize_answer as normalize_text,
    )
except ImportError:
    try:
        from src.baseline_rag import (
            exact_match_score as compute_exact_match,
            f1_score as compute_f1_score,
            normalize_answer as normalize_text,
        )
    except ImportError:
        def normalize_text(s: str) -> str:
            import re, string
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def compute_exact_match(prediction: str, ground_truth: str) -> float:
            return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

        def compute_f1_score(prediction: str, ground_truth: str) -> float:
            pred_tokens = normalize_text(prediction).split()
            gold_tokens = normalize_text(ground_truth).split()
            if len(pred_tokens) == 0 or len(gold_tokens) == 0:
                return float(pred_tokens == gold_tokens)
            common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return (2 * precision * recall) / (precision + recall)

# Import M2 module (established and tested)
try:
    from m2_evidence_optimizer import (
        BudgetedEvidenceSelector,
        EvidenceCandidate,
        InteractionMatrix,
        M2OptimizationOutput,
        MSTEvidenceOrganizer,
        TwoStageM2Optimizer,
        calibrate_reliability,
        naive_mst_baseline,
    )
except ImportError:
    from src.m2_evidence_optimizer import (
        BudgetedEvidenceSelector,
        EvidenceCandidate,
        InteractionMatrix,
        M2OptimizationOutput,
        MSTEvidenceOrganizer,
        TwoStageM2Optimizer,
        calibrate_reliability,
        naive_mst_baseline,
    )

# Try dynamic import for M1 and M3
HAS_M1 = False
HAS_M3 = False
try:
    import m1_adaptive_retriever
    from m1_adaptive_retriever import M1AdaptiveRetriever, js_divergence
    HAS_M1 = True
except ImportError:
    try:
        from src import m1_adaptive_retriever
        from src.m1_adaptive_retriever import M1AdaptiveRetriever, js_divergence
        HAS_M1 = True
    except ImportError:
        HAS_M1 = False
        js_divergence = None

try:
    import m3_calibrated_decoder
    from m3_calibrated_decoder import M3CalibratedDecoder, CalibratedDecoderConfig, compute_auroc
    HAS_M3 = True
except ImportError:
    try:
        from src import m3_calibrated_decoder
        from src.m3_calibrated_decoder import M3CalibratedDecoder, CalibratedDecoderConfig, compute_auroc
        HAS_M3 = True
    except ImportError:
        HAS_M3 = False
        compute_auroc = None

# Import Baseline RAG components
try:
    from baseline_rag import (
        BM25Retriever,
        DenseRetriever,
        HybridRetriever,
        QwenNF4Generator,
        reciprocal_rank_fusion,
    )
except ImportError:
    from src.baseline_rag import (
        BM25Retriever,
        DenseRetriever,
        HybridRetriever,
        QwenNF4Generator,
        reciprocal_rank_fusion,
    )

logger = logging.getLogger("u_graphrag_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [UGraphRAG] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Section 1: Evaluation Metrics & Helpers ---

def compute_supporting_facts_metrics(
    retrieved_titles: List[str],
    gold_supporting_facts: Any
) -> Dict[str, float]:
    """
    Computes supporting facts precision, recall, and F1 on document/title level.
    Supports:
    - dict {'title': [...], 'sent_id': [...]}
    - list/tuple of [title, sent_idx] or {'title': ...} or title str
    """
    if not gold_supporting_facts:
        return {"sp_precision": 0.0, "sp_recall": 0.0, "sp_f1": 0.0}

    gold_titles: Set[str] = set()
    if isinstance(gold_supporting_facts, dict):
        for t in gold_supporting_facts.get("title", []):
            if t:
                gold_titles.add(str(t).strip())
    elif isinstance(gold_supporting_facts, (list, tuple, set)):
        for item in gold_supporting_facts:
            if isinstance(item, dict) and item.get("title"):
                gold_titles.add(str(item["title"]).strip())
            elif isinstance(item, (list, tuple)) and len(item) > 0 and item[0]:
                gold_titles.add(str(item[0]).strip())
            elif isinstance(item, str) and item.strip():
                gold_titles.add(item.strip())

    pred_titles = {str(t).strip() for t in retrieved_titles if str(t).strip()}

    if not gold_titles:
        return {"sp_precision": 0.0, "sp_recall": 0.0, "sp_f1": 0.0}

    overlap = len(gold_titles.intersection(pred_titles))
    precision = overlap / len(pred_titles) if pred_titles else 0.0
    recall = overlap / len(gold_titles) if gold_titles else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "sp_precision": round(float(precision), 4),
        "sp_recall": round(float(recall), 4),
        "sp_f1": round(float(f1), 4),
    }


def compute_ece(
    confidences: List[float],
    accuracies: List[float],
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE).
    confidences: predicted confidence in [0, 1]
    accuracies: binary accuracy in {0, 1}
    """
    if not confidences or not accuracies or len(confidences) != len(accuracies):
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


# --- Section 2: Built-in Resilient M1 & M3 Implementations (Dynamic Native Fallback) ---

class BuiltinEpistemicDecoupler:
    """
    Fallback epistemic uncertainty decoupler when M1 module is not yet attached.
    Provides standard closed-form AU/EU decoupling proxy via query ambiguity and doc divergence.
    """
    def __init__(self, tau_eu: float = 0.5):
        self.tau_eu = tau_eu

    def estimate_uncertainty(self, query: str, initial_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        # High epistemic uncertainty when query is long/specific or document retrieval scores have high variance
        scores = [d.get("score", 0.5) for d in initial_docs]
        if not scores:
            return {"u_total": 0.8, "u_au": 0.4, "u_eu": 0.4, "trigger_expansion": True}
        
        variance = float(np.var(scores)) if len(scores) > 1 else 0.1
        mean_score = float(np.mean(scores))
        
        u_au = max(0.05, min(0.95, 1.0 - mean_score))
        u_eu = max(0.05, min(0.95, variance * 2.0 + (0.3 if len(query.split()) > 10 else 0.1)))
        u_total = min(1.0, u_au + u_eu)
        trigger_expansion = u_eu > self.tau_eu

        return {
            "u_total": round(u_total, 4),
            "u_au": round(u_au, 4),
            "u_eu": round(u_eu, 4),
            "trigger_expansion": trigger_expansion
        }


class BuiltinCalibratedDecoder:
    """
    Fallback reliability-weighted contrastive calibrated decoder and abstention gate.
    Used when m3_calibrated_decoder is loading or running dry-run.
    """
    def __init__(
        self,
        tau_abstain: float = 0.25,
        abstain_message: str = "Insufficient evidence to answer confidently."
    ):
        self.tau_abstain = tau_abstain
        self.abstain_message = abstain_message

    def decode_and_calibrate(
        self,
        generator: Any,
        prompt: str,
        evidence_docs: List[Dict[str, Any]],
        sample_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, bool, int]:
        """
        Returns:
            answer: generated answer or abstention message
            confidence: calibrated confidence score [0, 1]
            abstained: boolean flag
            forward_calls: number of forward calls
        """
        # Call standard generator first
        pred, calls = generator.generate(prompt, evidence_docs)
        
        # Estimate confidence from evidence reliability weights
        if evidence_docs:
            scores = [d.get("calibrated_reliability", d.get("score", 0.5)) for d in evidence_docs]
            conf = float(np.mean(scores)) if scores else 0.5
        else:
            conf = 0.1
            
        conf = max(0.05, min(0.98, conf))
        abstained = False
        
        # Abstention decision
        if conf < self.tau_abstain:
            abstained = True
            pred = self.abstain_message
            
        return pred, conf, abstained, calls


# --- Section 3: UGraphRAGPipeline Core Orchestrator ---

class UGraphRAGPipeline:
    """
    Unified Experimental Engine for U-GraphRAG and all Ablation Studies.
    Supports all 8 ablation configurations with strict interface contracts.
    """

    SUPPORTED_MODES = [
        "B0_Standard_RAG",
        "B1_Naive_MST",
        "M1_Only",
        "M2_Greedy",
        "M2_Exact_NoMST",
        "M2_Full",
        "M2_Permutation_Test",
        "Full_UGraphRAG",
    ]

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[QwenNF4Generator] = None,
        corpus: Optional[List[Dict[str, str]]] = None,
        token_budget: int = 512,
        top_k_standard: int = 5,
        top_k_candidate_pool: int = 10,
        profiler: Optional[MetricsProfiler] = None,
        m1_module: Optional[Any] = None,
        m3_module: Optional[Any] = None,
        m1_tau_eu: Optional[float] = None,
        m1_use_model_probe: bool = True,
        m2_gamma_1: float = 0.5,
        m2_gamma_2: float = 0.3,
        m2_lambda_div: float = 0.4,
        m2_reliability_offset: float = 0.22,
        m2_min_docs: int = 2,
        m3_beta: float = 0.12,
        m3_alpha_trunc: float = 0.10,
        m3_abstain_threshold: float = 0.25,
        m3_confidence_threshold: float = 0.50,
        m3_confidence_temperature: float = 1.5,
        m3_confidence_tok_weight: float = 0.50,
        m3_confidence_platt_a: float = 1.0,
        m3_confidence_platt_b: float = 0.0,
        m3_contrastive_conf_guard: float = 0.85,
        m3_protect_eos: bool = True,
        dense_model_path: Optional[str] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.corpus = corpus or []
        self.token_budget = token_budget
        self.top_k_standard = top_k_standard
        self.top_k_final = top_k_standard
        self.top_k_candidate_pool = top_k_candidate_pool
        self.profiler = profiler or MetricsProfiler()
        self._m1_corpus_indexed = False
        self.m1_use_model_probe = bool(m1_use_model_probe)
        self._m1_probe_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]] = {}
        self.m2_reliability_offset = float(m2_reliability_offset)
        self.m2_min_docs = max(1, int(m2_min_docs))

        # Resolve and share local dense encoder on CPU (keeps full 8GB GPU VRAM for Qwen)
        if dense_model_path:
            self.dense_model_path = dense_model_path
        elif os.path.isdir("models/bge-small-en-v1.5"):
            self.dense_model_path = "models/bge-small-en-v1.5"
        else:
            self.dense_model_path = "BAAI/bge-small-en-v1.5"

        self.shared_dense_model = None
        if self.retriever is not None and getattr(getattr(self.retriever, "dense_retriever", None), "st_model", None) is not None:
            self.shared_dense_model = self.retriever.dense_retriever.st_model
        elif os.path.isdir(self.dense_model_path) or dense_model_path is not None:
            try:
                from sentence_transformers import SentenceTransformer
                self.shared_dense_model = SentenceTransformer(self.dense_model_path, device="cpu")
            except Exception as e:
                logger.warning(f"Shared SentenceTransformer load fallback ({e}); HybridRetriever will use TF-IDF dense.")
                self.shared_dense_model = None

        if self.retriever is None and self.corpus:
            try:
                self.retriever = HybridRetriever(
                    self.corpus,
                    dense_model_path=self.dense_model_path,
                    shared_dense_model=self.shared_dense_model,
                )
            except Exception as e:
                logger.warning(f"HybridRetriever init warning on corpus: {e}")

        # M1 module initialization (prefer genuine M1AdaptiveRetriever)
        if m1_module is not None:
            self.m1 = m1_module
        elif HAS_M1 and hasattr(m1_adaptive_retriever, "M1AdaptiveRetriever"):
            cal_path = "models/m1_calibration.json" if os.path.exists("models/m1_calibration.json") else None
            self.m1 = m1_adaptive_retriever.M1AdaptiveRetriever(calibration_path=cal_path)
            if self.corpus:
                self.m1.index_corpus(self.corpus)
                self._m1_corpus_indexed = True
        elif HAS_M1 and hasattr(m1_adaptive_retriever, "AdaptiveRetriever"):
            self.m1 = m1_adaptive_retriever.AdaptiveRetriever()
        else:
            self.m1 = BuiltinEpistemicDecoupler()

        if m1_tau_eu is not None and hasattr(self.m1, "decoupler"):
            self.m1.decoupler.tau_eu = float(m1_tau_eu)

        # Align M1 inference uncertainty scale with m1_calibration.json model-probe scale
        if hasattr(self.m1, "default_model_fn"):
            self.m1.default_model_fn = self._build_m1_model_fn()

        # M2 module initialization
        self.m2_gamma_1 = m2_gamma_1
        self.m2_gamma_2 = m2_gamma_2
        self.m2_lambda_div = m2_lambda_div
        self.m2_optimizer_exact = TwoStageM2Optimizer(
            lambda_coverage=m2_gamma_1,
            lambda_complementarity=m2_gamma_2,
            lambda_redundancy=m2_lambda_div,
            lambda_conflict=1.0,
            selection_algorithm="exact_enum",
            recency_aware=False,
        )
        self.m2_optimizer_greedy = TwoStageM2Optimizer(
            lambda_coverage=m2_gamma_1,
            lambda_complementarity=m2_gamma_2,
            lambda_redundancy=m2_lambda_div,
            lambda_conflict=1.0,
            selection_algorithm="greedy",
            recency_aware=False,
        )

        # M3 module initialization (prefer genuine M3CalibratedDecoder)
        if m3_module is not None:
            self.m3 = m3_module
        elif HAS_M3 and hasattr(m3_calibrated_decoder, "M3CalibratedDecoder"):
            m3_cfg = m3_calibrated_decoder.CalibratedDecoderConfig(
                abstain_threshold=float(m3_abstain_threshold),
                confidence_threshold=float(m3_confidence_threshold),
                beta=float(m3_beta),
                alpha_trunc=float(m3_alpha_trunc),
                max_new_tokens=48,
                preserve_evidence_order=True,
                confidence_temperature=float(m3_confidence_temperature),
                confidence_tok_weight=float(m3_confidence_tok_weight),
                confidence_platt_a=float(m3_confidence_platt_a),
                confidence_platt_b=float(m3_confidence_platt_b),
                contrastive_conf_guard=float(m3_contrastive_conf_guard),
                protect_eos=bool(m3_protect_eos),
            )
            self.m3 = m3_calibrated_decoder.M3CalibratedDecoder(config=m3_cfg)
        elif HAS_M3 and hasattr(m3_calibrated_decoder, "CalibratedDecoder"):
            self.m3 = m3_calibrated_decoder.CalibratedDecoder()
        else:
            self.m3 = BuiltinCalibratedDecoder()

        logger.info(
            f"UGraphRAGPipeline initialized | Token Budget: {self.token_budget} | "
            f"M1: {type(self.m1).__name__} (probe={getattr(self.m1, 'default_model_fn', None) is not None}, "
            f"tau_eu={getattr(getattr(self.m1, 'decoupler', None), 'tau_eu', 'N/A')}) | "
            f"M2: TwoStageM2Optimizer | M3: {type(self.m3).__name__}"
        )

    def _build_m1_model_fn(self) -> Optional[Callable[[str, List[str]], Dict[str, float]]]:
        """Construct cached next-token softmax probe matching offline calibration scale."""
        if not self.m1_use_model_probe or self.generator is None:
            return None
        if getattr(self.generator, "dry_run", True) or getattr(self.generator, "model", None) is None:
            return None

        gen = self.generator
        cache = self._m1_probe_cache

        def _probe(q: str, doc_texts: List[str]) -> Dict[str, float]:
            cache_key = (q, tuple(str(t)[:256] for t in doc_texts))
            if cache_key in cache:
                return cache[cache_key]
            ctx_list = [{"title": f"Evidence {idx+1}", "text": str(t)} for idx, t in enumerate(doc_texts)]
            prompt = self.format_prompt(q, ctx_list)
            inputs = gen.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(gen.model.device)
            with torch.no_grad():
                outputs = gen.model(**inputs)
                logits = outputs.logits[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                top16 = torch.topk(probs, 16)
                res: Dict[str, float] = {}
                for idx_t, val in zip(top16.indices, top16.values):
                    tid = int(idx_t.item())
                    tok_str = gen.tokenizer.decode([tid]).strip() or "_"
                    res[f"t_{tid}:{tok_str}"] = float(val.item())
            total_p = sum(res.values())
            if total_p > 0.0:
                res = {k: v / total_p for k, v in res.items()}
            if len(cache) < 4096:
                cache[cache_key] = res
            return res

        return _probe

    def configure_parameters(
        self,
        m1_tau_eu: Optional[float] = None,
        m2_gamma_1: Optional[float] = None,
        m2_gamma_2: Optional[float] = None,
        m2_lambda_div: Optional[float] = None,
        m2_reliability_offset: Optional[float] = None,
        m2_min_docs: Optional[int] = None,
        top_k_candidate_pool: Optional[int] = None,
        m3_beta: Optional[float] = None,
        m3_alpha_trunc: Optional[float] = None,
        m3_abstain_threshold: Optional[float] = None,
        m3_confidence_threshold: Optional[float] = None,
        m3_confidence_temperature: Optional[float] = None,
        m3_confidence_tok_weight: Optional[float] = None,
        m3_confidence_platt_a: Optional[float] = None,
        m3_confidence_platt_b: Optional[float] = None,
        m3_contrastive_conf_guard: Optional[float] = None,
        m3_protect_eos: Optional[bool] = None,
    ) -> None:
        """Dynamically reconfigure M1/M2/M3 hyperparameters without reloading LLM weights."""
        if m1_tau_eu is not None and hasattr(self.m1, "decoupler"):
            self.m1.decoupler.tau_eu = float(m1_tau_eu)
        if top_k_candidate_pool is not None:
            self.top_k_candidate_pool = int(top_k_candidate_pool)
        if m2_reliability_offset is not None:
            self.m2_reliability_offset = float(m2_reliability_offset)
        if m2_min_docs is not None:
            self.m2_min_docs = max(1, int(m2_min_docs))

        reinit_m2 = any(v is not None for v in (m2_gamma_1, m2_gamma_2, m2_lambda_div))
        if m2_gamma_1 is not None:
            self.m2_gamma_1 = float(m2_gamma_1)
        if m2_gamma_2 is not None:
            self.m2_gamma_2 = float(m2_gamma_2)
        if m2_lambda_div is not None:
            self.m2_lambda_div = float(m2_lambda_div)
        if reinit_m2:
            self.m2_optimizer_exact = TwoStageM2Optimizer(
                lambda_coverage=self.m2_gamma_1,
                lambda_complementarity=self.m2_gamma_2,
                lambda_redundancy=self.m2_lambda_div,
                lambda_conflict=1.0,
                selection_algorithm="exact_enum",
                recency_aware=False,
            )
            self.m2_optimizer_greedy = TwoStageM2Optimizer(
                lambda_coverage=self.m2_gamma_1,
                lambda_complementarity=self.m2_gamma_2,
                lambda_redundancy=self.m2_lambda_div,
                lambda_conflict=1.0,
                selection_algorithm="greedy",
                recency_aware=False,
            )

        if hasattr(self.m3, "config"):
            cfg = self.m3.config
            if m3_beta is not None:
                cfg.beta = float(m3_beta)
            if m3_alpha_trunc is not None:
                cfg.alpha_trunc = float(m3_alpha_trunc)
            if m3_abstain_threshold is not None:
                cfg.abstain_threshold = float(m3_abstain_threshold)
            if m3_confidence_threshold is not None:
                cfg.confidence_threshold = float(m3_confidence_threshold)
            if m3_confidence_temperature is not None:
                cfg.confidence_temperature = float(m3_confidence_temperature)
            if m3_confidence_tok_weight is not None:
                cfg.confidence_tok_weight = float(m3_confidence_tok_weight)
            if m3_confidence_platt_a is not None:
                cfg.confidence_platt_a = float(m3_confidence_platt_a)
            if m3_confidence_platt_b is not None:
                cfg.confidence_platt_b = float(m3_confidence_platt_b)
            if m3_contrastive_conf_guard is not None:
                cfg.contrastive_conf_guard = float(m3_contrastive_conf_guard)
            if m3_protect_eos is not None:
                cfg.protect_eos = bool(m3_protect_eos)

    @staticmethod
    def _extract_entities(text: str, title: str = "") -> Set[str]:
        """Extract normalized named entities, title mentions, and 4-digit years for M2 graph edges."""
        stop_caps = {
            "the", "a", "an", "in", "on", "at", "for", "of", "to", "by", "with", "from", "and", "or",
            "when", "what", "who", "where", "which", "why", "how", "is", "was", "are", "were",
            "yes", "no", "it", "this", "that", "these", "those", "he", "she", "they", "his", "her",
            "document", "title", "context", "question", "answer",
        }
        ents: Set[str] = set()
        if title:
            t_norm = title.strip().lower()
            if t_norm and t_norm not in stop_caps:
                ents.add(t_norm)
            for tok in re.findall(r"[A-Za-z0-9]{4,}", t_norm):
                if tok not in stop_caps:
                    ents.add(tok)

        for phrase in re.findall(r"\b[A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+){0,3}\b", text or ""):
            low = phrase.strip().lower()
            if len(low) >= 3 and low not in stop_caps:
                ents.add(low)
                for sub in low.split():
                    if len(sub) >= 4 and sub not in stop_caps:
                        ents.add(sub)

        for yr in re.findall(r"\b(?:1[0-9]{3}|20[0-2][0-9])\b", text or ""):
            ents.add(yr)

        return ents

    def _convert_to_candidates(
        self,
        docs: List[Dict[str, Any]],
        m2_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[EvidenceCandidate]:
        """Convert dictionary documents to EvidenceCandidate dataclasses with entities & uncertainty metadata."""
        candidates = []
        meta_map = m2_meta_map or {}
        for i, doc in enumerate(docs):
            doc_id = str(doc.get("id", doc.get("title", f"doc_{i}")))
            title = str(doc.get("title", doc_id))
            text = str(doc.get("text", doc.get("content", "")))
            meta = meta_map.get(doc_id, {})

            raw_score = doc.get("score", meta.get("score", doc.get("fusion_score", 1.0 / (i + 1))))
            score = float(raw_score)
            au = float(doc.get("aleatoric_uncertainty", meta.get("aleatoric_uncertainty", 0.0)))
            er = float(doc.get("epistemic_reduction", meta.get("epistemic_reduction", 0.0)))
            token_count = int(doc.get("token_count", max(1, len(text.split()))))

            raw_ents = doc.get("entities") or meta.get("entities")
            if raw_ents:
                entities = {str(e).strip().lower() for e in raw_ents if str(e).strip()}
                entities |= self._extract_entities(text, title=title)
            else:
                entities = self._extract_entities(text, title=title)

            candidates.append(
                EvidenceCandidate(
                    doc_id=doc_id,
                    text=text,
                    score=score,
                    entities=entities,
                    aleatoric_uncertainty=au,
                    epistemic_reduction=er,
                    metadata={"title": title, "original_idx": i, "token_count": token_count},
                )
            )
        return candidates

    def _extract_all_local_docs(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all normalized local context documents from HotpotQA dict or list context formats."""
        ctx = sample.get("context")
        docs: List[Dict[str, Any]] = []
        if isinstance(ctx, dict):
            titles = ctx.get("title", []) or []
            sents_list = ctx.get("sentences", []) or []
            for idx, (title, sents) in enumerate(zip(titles, sents_list)):
                t_str = str(title)
                text = " ".join(sents) if isinstance(sents, list) else str(sents)
                docs.append({
                    "id": t_str or f"doc_{idx}",
                    "title": t_str or f"doc_{idx}",
                    "text": text,
                })
        elif isinstance(ctx, list):
            for idx, item in enumerate(ctx):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    t_str = str(item[0])
                    sents = item[1]
                    text = " ".join(sents) if isinstance(sents, list) else str(sents)
                    docs.append({"id": t_str or f"doc_{idx}", "title": t_str or f"doc_{idx}", "text": text})
                elif isinstance(item, dict):
                    d = dict(item)
                    t_str = str(d.get("title", d.get("id", f"doc_{idx}")))
                    d.setdefault("id", t_str)
                    d.setdefault("title", t_str)
                    d.setdefault("text", str(d.get("content", "")))
                    docs.append(d)
        return docs

    def _rank_docs(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rank local candidate docs via HybridRetriever (BM25 + Dense RRF) and normalize scores to [0.15, 0.98]."""
        if not docs:
            return []
        try:
            local_ret = HybridRetriever(
                docs,
                dense_model_path=self.dense_model_path,
                shared_dense_model=self.shared_dense_model,
            )
            ranked = local_ret.retrieve(query, top_k=min(top_k, len(docs)))
            if ranked:
                raw_scores = [float(d.get("fusion_score", 0.0)) for d in ranked]
                s_min, s_max = min(raw_scores), max(raw_scores)
                span = (s_max - s_min) if (s_max - s_min) > 1e-9 else 1.0
                for rank_idx, d in enumerate(ranked):
                    norm_s = (float(d.get("fusion_score", 0.0)) - s_min) / span if len(ranked) > 1 else 0.95
                    d["score"] = round(0.20 + 0.78 * norm_s if s_max > s_min else max(0.20, 0.95 - 0.07 * rank_idx), 4)
                    d["entities"] = list(self._extract_entities(d.get("text", ""), title=str(d.get("title", ""))))
                return ranked[:top_k]
        except Exception as e:
            logger.debug(f"Hybrid ranking fallback to lexical overlap: {e}")

        # Pure lexical + title overlap fallback
        q_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for idx, d in enumerate(docs):
            txt = f"{d.get('title', '')} {d.get('text', '')}".lower()
            d_terms = set(re.findall(r"\w+", txt))
            overlap = len(q_terms & d_terms) / max(1, len(q_terms))
            d_copy = dict(d)
            d_copy["score"] = round(min(0.98, 0.20 + 0.78 * overlap), 4)
            d_copy["entities"] = list(self._extract_entities(d_copy.get("text", ""), title=str(d_copy.get("title", ""))))
            scored.append((overlap, -idx, d_copy))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]

    def _build_explicit_edges(
        self,
        candidates: List[EvidenceCandidate],
        per_doc_dists: Dict[str, Dict[str, float]],
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Build explicit pairwise edges combining entity-bridge complementarity, MinHash redundancy, and M1 JS conflict."""
        edges: Dict[Tuple[str, str], Dict[str, float]] = {}
        shingles: Dict[str, Set[str]] = {}
        for c in candidates:
            toks = re.findall(r"\w+", (c.text or "").lower())
            shingles[c.doc_id] = {" ".join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))} if len(toks) >= 3 else set(toks)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                ci, cj = candidates[i], candidates[j]
                if ci.entities and cj.entities:
                    inter_e = len(ci.entities & cj.entities)
                    union_e = len(ci.entities | cj.entities)
                    comp = inter_e / union_e if union_e > 0 else 0.0
                else:
                    comp = 0.0

                si, sj = shingles.get(ci.doc_id, set()), shingles.get(cj.doc_id, set())
                union_s = len(si | sj)
                red = (len(si & sj) / union_s) if union_s > 0 else 0.0

                pi = per_doc_dists.get(ci.doc_id)
                pj = per_doc_dists.get(cj.doc_id)
                if pi and pj and HAS_M1 and hasattr(m1_adaptive_retriever, "js_divergence"):
                    raw_js = float(m1_adaptive_retriever.js_divergence(pi, pj))
                    # Only treat JS divergence as factual conflict when docs overlap on subject entities
                    conf = raw_js * (0.08 + 0.35 * comp) if comp > 0.0 else raw_js * 0.05
                else:
                    conf = 0.0

                # Bridge edge weight favors multi-hop entity connections while penalizing strong divergence
                weight = comp + 0.15 * red - 0.35 * conf
                edges[(ci.doc_id, cj.doc_id)] = {
                    "complementarity": float(comp),
                    "redundancy": float(red),
                    "conflict": float(conf),
                    "weight": float(weight),
                }
        return edges

    def _retrieve_pool(
        self,
        query: str,
        sample: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieves and ranks candidate pool for query, supporting both HotpotQA dict/list local contexts and open corpus.
        """
        local_docs = self._extract_all_local_docs(sample)
        if local_docs:
            return self._rank_docs(query, local_docs, top_k=top_k)

        if self.retriever is not None:
            results = self.retriever.retrieve(query, top_k=top_k)
            docs: List[Dict[str, Any]] = []
            for item in results:
                if isinstance(item, dict):
                    d = dict(item)
                    if "score" not in d:
                        d["score"] = float(d.get("fusion_score", 0.5))
                    d.setdefault("entities", list(self._extract_entities(d.get("text", ""), title=str(d.get("title", "")))))
                    docs.append(d)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    doc_idx, score = item[0], item[1]
                    if 0 <= doc_idx < len(self.corpus):
                        d = dict(self.corpus[doc_idx])
                        d["score"] = float(score)
                        d.setdefault("entities", list(self._extract_entities(d.get("text", ""), title=str(d.get("title", "")))))
                        docs.append(d)
            return docs

        return []

    def format_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Format RAG prompt using loaded Qwen ChatML template when available, else structured fallback."""
        if self.generator is not None and hasattr(self.generator, "build_prompt"):
            try:
                return self.generator.build_prompt(query, context_docs)
            except Exception:
                pass

        context_str = "\n\n".join(
            [f"Document [{i+1}] (Title: {d.get('title', d.get('id', 'N/A'))}):\n{d.get('text', '')}"
             for i, d in enumerate(context_docs)]
        )
        prompt = (
            f"You are a factual, accurate assistant. Answer the question concisely based on the provided documents.\n"
            f"If the documents do not provide sufficient information, say I cannot answer.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Short Answer:"
        )
        return prompt

    def run_pipeline_single(
        self,
        sample: Dict[str, Any],
        mode: str = "Full_UGraphRAG",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline on a single query according to specified ablation mode.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Must be one of {self.SUPPORTED_MODES}")

        query = sample.get("question", sample.get("query", ""))
        ground_truth = sample.get("answer", sample.get("answers", [""]))
        gold_supporting_facts = sample.get("supporting_facts", [])

        # Start timer & profiler
        t_start = time.perf_counter()
        forward_calls = 0
        mem_before = self.profiler.get_current_cuda_mem_mb()
        allocated_before = float(mem_before.get("allocated_mb", mem_before.get("allocated", 0.0)))
        reserved_before = float(mem_before.get("reserved_mb", mem_before.get("reserved", 0.0)))

        # Step 1: Candidate Pool Retrieval
        local_docs_all = self._extract_all_local_docs(sample)
        ranked_pool = (
            self._rank_docs(query, local_docs_all, top_k=max(self.top_k_candidate_pool, len(local_docs_all)))
            if local_docs_all
            else self._retrieve_pool(query, sample, top_k=self.top_k_candidate_pool)
        )
        pool_k = self.top_k_candidate_pool if mode != "B0_Standard_RAG" else self.top_k_standard
        candidates_raw = ranked_pool[:pool_k]
        initial_k = max(2, min(3, self.top_k_standard))
        initial_docs = ranked_pool[:initial_k]

        # M1 Uncertainty Gated Check & MinHash-Bloom Graph Expansion
        m1_info: Dict[str, Any] = {"u_total": 0.0, "u_au": 0.0, "u_eu": 0.0, "trigger_expansion": False}
        m2_meta_map: Dict[str, Dict[str, Any]] = {}
        per_doc_dists: Dict[str, Dict[str, float]] = {}
        m1_expanded_docs: List[Dict[str, Any]] = list(candidates_raw)

        if mode in ["M1_Only", "M2_Greedy", "M2_Exact_NoMST", "M2_Full", "M2_Permutation_Test", "Full_UGraphRAG"]:
            if hasattr(self.m1, "process"):
                if local_docs_all:
                    self.m1.index_corpus(ranked_pool)
                elif self.corpus and not self._m1_corpus_indexed:
                    self.m1.index_corpus(self.corpus)
                    self._m1_corpus_indexed = True

                exp_res, m2_cands, per_doc_dists = self.m1.process(query=query, initial_docs=initial_docs)
                if initial_docs and hasattr(self.m1, "decoupler"):
                    eff_probe = getattr(self.m1, "default_model_fn", None)
                    u_m = self.m1.decoupler.decompose(query, initial_docs, model_fn=eff_probe)
                    m1_info = {
                        "u_total": round(float(u_m.total_entropy), 4),
                        "u_au": round(float(u_m.aleatoric_uncertainty), 4),
                        "u_eu": round(float(u_m.epistemic_uncertainty), 4),
                        "trigger_expansion": not bool(exp_res.early_exit),
                        "expanded_count": int(len(exp_res.expanded_candidates)),
                    }
                for mc in m2_cands:
                    m2_meta_map[str(mc.get("doc_id", ""))] = mc
                score_by_id = {str(d.get("id", d.get("title", ""))): float(d.get("score", 0.5)) for d in ranked_pool}
                init_id_set = {str(d.get("id", d.get("title", ""))) for d in initial_docs}
                # M1 early-exit skips MinHash expansion hops; M2 always optimizes over the unified pool
                if exp_res.expanded_candidates:
                    merged: List[Dict[str, Any]] = []
                    seen_ids: Set[str] = set()
                    for d in exp_res.expanded_candidates:
                        did = str(d.get("id", d.get("title", "")))
                        if did and did not in seen_ids:
                            dc = dict(d)
                            dc["score"] = score_by_id.get(did, float(dc.get("score", 0.5)))
                            merged.append(dc)
                            seen_ids.add(did)
                    if mode in ["M2_Greedy", "M2_Exact_NoMST", "M2_Full", "M2_Permutation_Test", "Full_UGraphRAG"]:
                        for d in candidates_raw:
                            did = str(d.get("id", d.get("title", "")))
                            if did and did not in seen_ids:
                                merged.append(dict(d))
                                seen_ids.add(did)
                    # Preserve initial anchor quality while promoting novel docs with genuine epistemic gain
                    merged.sort(
                        key=lambda d: (
                            1 if str(d.get("id", d.get("title", ""))) in init_id_set else 0,
                            float(d.get("score", 0.5))
                            + 0.15 * float(m2_meta_map.get(str(d.get("id", d.get("title", ""))), {}).get("epistemic_reduction", 0.0)),
                        ),
                        reverse=True,
                    )
                    m1_expanded_docs = merged[:self.top_k_candidate_pool]
            elif hasattr(self.m1, "estimate_uncertainty"):
                m1_info = self.m1.estimate_uncertainty(query, initial_docs)
            elif hasattr(self.m1, "decouple"):
                m1_info = self.m1.decouple(query, initial_docs)

        # Step 2: Route by Mode
        permutation_stats: Dict[str, Any] = {}
        selected_docs: List[Dict[str, Any]] = []

        if mode == "B0_Standard_RAG":
            # Baseline 0: Standard Hybrid RAG Top-K
            selected_docs = candidates_raw[:self.top_k_standard]

        elif mode == "B1_Naive_MST":
            # Baseline 1: Reviewer Naive MST baseline (no M2-A filtering)
            candidates_objs = self._convert_to_candidates(candidates_raw)
            query_entities = self._extract_entities(query)
            organizer = MSTEvidenceOrganizer(recency_aware=False)
            org_res = organizer.organize(
                selected_docs=candidates_objs,
                query_entities=query_entities if query_entities else None,
            )
            mst_ordered = org_res.ordered_docs[:self.top_k_standard]
            selected_docs = [
                {"id": c.doc_id, "title": c.metadata.get("title", c.doc_id), "text": c.text, "score": c.score}
                for c in mst_ordered
            ]

        elif mode == "M1_Only":
            # M1 only: early exit on confident queries (initial_docs), or MinHash-expanded docs up to top_k_standard
            if m1_info.get("trigger_expansion", False):
                selected_docs = m1_expanded_docs[:self.top_k_standard]
            else:
                selected_docs = initial_docs

        elif mode in ["M2_Greedy", "M2_Exact_NoMST", "M2_Full", "M2_Permutation_Test", "Full_UGraphRAG"]:
            pool_for_m2 = m1_expanded_docs if m1_expanded_docs else candidates_raw
            candidates_objs = self._convert_to_candidates(pool_for_m2, m2_meta_map=m2_meta_map)
            query_entities = self._extract_entities(query)
            explicit_edges = self._build_explicit_edges(candidates_objs, per_doc_dists)
            optimizer = self.m2_optimizer_greedy if mode == "M2_Greedy" else self.m2_optimizer_exact

            # Calibrate reliability and subtract token/noise baseline cost so low-utility distractors are pruned
            calibrate_reliability(
                candidates_objs,
                method="minmax",
                gamma_1=self.m2_gamma_1,
                gamma_2=self.m2_gamma_2,
            )
            orig_rel = {c.doc_id: float(c.calibrated_reliability) for c in candidates_objs}
            for cand in candidates_objs:
                cand.calibrated_reliability -= self.m2_reliability_offset

            # Run M2 Two-Stage Optimization with genuine entities, AU/ER metadata, and explicit edges
            m2_out = optimizer.optimize(
                candidates=candidates_objs,
                explicit_edges=explicit_edges,
                max_k=self.top_k_final,
                query_entities=query_entities if query_entities else None,
                calibrate=False,
                min_k=min(self.m2_min_docs, len(candidates_objs)) if candidates_objs else 1,
            )
            for cand in candidates_objs:
                if cand.doc_id in orig_rel:
                    cand.calibrated_reliability = orig_rel[cand.doc_id]

            if mode == "M2_Exact_NoMST":
                # Subset optimization done, but NO MST topological ordering: sort by calibrated reliability
                docs_sorted = sorted(m2_out.selected_docs, key=lambda c: getattr(c, "calibrated_reliability", c.score), reverse=True)
                selected_docs = [
                    {"id": c.doc_id, "title": c.metadata.get("title", c.doc_id), "text": c.text,
                     "score": c.score, "calibrated_reliability": getattr(c, "calibrated_reliability", c.score),
                     "aleatoric_uncertainty": c.aleatoric_uncertainty}
                    for c in docs_sorted
                ]
            else:
                # MST Topological Linearization applied
                selected_docs = [
                    {"id": c.doc_id, "title": c.metadata.get("title", c.doc_id), "text": c.text,
                     "score": c.score, "calibrated_reliability": getattr(c, "calibrated_reliability", c.score),
                     "aleatoric_uncertainty": c.aleatoric_uncertainty}
                    for c in m2_out.ordered_docs
                ]

        # Calculate token savings rate relative to Standard Top-K RAG baseline (B0)
        b0_baseline_docs = ranked_pool[:self.top_k_standard] if ranked_pool else candidates_raw
        b0_baseline_tokens = sum(max(1, len(d.get("text", "").split())) for d in b0_baseline_docs) or 1
        selected_tokens = sum(max(1, len(d.get("text", "").split())) for d in selected_docs)
        token_saving_rate = max(0.0, 1.0 - (selected_tokens / b0_baseline_tokens))

        # Supporting facts precision/recall/F1
        retrieved_titles = [str(d.get("title", d.get("id", ""))) for d in selected_docs]
        sp_metrics = compute_supporting_facts_metrics(retrieved_titles, gold_supporting_facts)

        # Step 3: Generation & Permutation Test Execution
        predicted_answer = ""
        confidence = 0.5
        abstained = False

        m3_telemetry: Dict[str, Any] = {}
        if mode == "M2_Permutation_Test":
            # Run Permutation Test: DFS vs Reverse vs 5x Random Shuffles
            # 1. DFS Order
            prompt_dfs = self.format_prompt(query, selected_docs)
            pred_dfs, calls_dfs = self.generator.generate(prompt_dfs, selected_docs) if self.generator else ("pred_dfs", 1)
            forward_calls += calls_dfs
            em_dfs = compute_exact_match(pred_dfs, ground_truth)
            f1_dfs = compute_f1_score(pred_dfs, ground_truth)

            # 2. Reverse Order
            docs_rev = list(reversed(selected_docs))
            prompt_rev = self.format_prompt(query, docs_rev)
            pred_rev, calls_rev = self.generator.generate(prompt_rev, docs_rev) if self.generator else ("pred_rev", 1)
            forward_calls += calls_rev
            em_rev = compute_exact_match(pred_rev, ground_truth)
            f1_rev = compute_f1_score(pred_rev, ground_truth)

            # 3. 5x Random Shuffles
            shuffle_seeds = [42, 43, 44, 45, 46]
            shuffled_f1s = []
            shuffled_ems = []
            for s in shuffle_seeds:
                docs_shuf = list(selected_docs)
                rng = random.Random(s)
                rng.shuffle(docs_shuf)
                prompt_shuf = self.format_prompt(query, docs_shuf)
                pred_shuf, c_shuf = self.generator.generate(prompt_shuf, docs_shuf) if self.generator else (f"pred_{s}", 1)
                forward_calls += c_shuf
                shuffled_ems.append(compute_exact_match(pred_shuf, ground_truth))
                shuffled_f1s.append(compute_f1_score(pred_shuf, ground_truth))

            f1_shuffles_mean = float(np.mean(shuffled_f1s))
            f1_shuffles_std = float(np.std(shuffled_f1s))
            max_drop = max(0.0, f1_dfs - min([f1_rev] + shuffled_f1s))

            permutation_stats = {
                "dfs_em": round(float(em_dfs), 4),
                "dfs_f1": round(float(f1_dfs), 4),
                "reverse_em": round(float(em_rev), 4),
                "reverse_f1": round(float(f1_rev), 4),
                "shuffles_f1_mean": round(f1_shuffles_mean, 4),
                "shuffles_f1_std": round(f1_shuffles_std, 4),
                "max_performance_drop": round(float(max_drop), 4),
            }

            predicted_answer = pred_dfs
            confidence = 0.85

        elif mode == "Full_UGraphRAG":
            # Calibrated decoding with M3 reliability contrast & abstention gate
            if hasattr(self.m3, "generate_calibrated") and self.generator is not None:
                m3_res = self.m3.generate_calibrated(
                    generator=self.generator,
                    question=query,
                    evidence_items=selected_docs,
                )
                predicted_answer = m3_res.answer
                confidence = float(m3_res.confidence)
                abstained = bool(m3_res.is_abstained)
                forward_calls += int(m3_res.forward_calls)
                m3_telemetry = dict(m3_res.metadata.get("telemetry", {})) if isinstance(m3_res.metadata, dict) else {}
                m3_telemetry["raw_answer"] = getattr(m3_res, "raw_answer", predicted_answer)
            elif hasattr(self.m3, "decode_and_calibrate"):
                prompt = self.format_prompt(query, selected_docs)
                pred, conf, abst, calls = self.m3.decode_and_calibrate(
                    generator=self.generator,
                    prompt=prompt,
                    evidence_docs=selected_docs,
                    sample_context=sample,
                )
                predicted_answer = pred
                confidence = conf
                abstained = abst
                forward_calls += calls
            else:
                prompt = self.format_prompt(query, selected_docs)
                pred, calls = self.generator.generate(prompt, selected_docs) if self.generator else ("pred", 1)
                predicted_answer = pred
                confidence = 0.85
                forward_calls += calls

        else:
            # Standard Greedy Decoding for B0, B1, M1, M2_Greedy, M2_Exact_NoMST, M2_Full
            prompt = self.format_prompt(query, selected_docs)
            if self.generator is not None:
                predicted_answer, calls = self.generator.generate(prompt, selected_docs)
                forward_calls += calls
            else:
                predicted_answer = f"Mock prediction for {query}"
                forward_calls += 1
            confidence = float(np.mean([d.get("calibrated_reliability", d.get("score", 0.5)) for d in selected_docs])) if selected_docs else 0.5

        # Timing and memory
        t_elapsed = time.perf_counter() - t_start
        mem_after = self.profiler.get_current_cuda_mem_mb()
        allocated_after = float(mem_after.get("max_allocated_mb", mem_after.get("allocated_mb", mem_after.get("allocated", 0.0))))
        reserved_after = float(mem_after.get("max_reserved_mb", mem_after.get("reserved_mb", mem_after.get("reserved", 0.0))))
        peak_alloc = max(allocated_before, allocated_after)
        peak_res = max(reserved_before, reserved_after)

        # Core accuracy & extended NLP metrics
        if compute_full_sample_metrics is not None:
            nlp_m = compute_full_sample_metrics(predicted_answer, sample.get("possible_answers", [ground_truth]) or [ground_truth])
            em = nlp_m["em"]
            f1 = nlp_m["f1"]
            r1 = nlp_m["rouge_1"]
            r2 = nlp_m["rouge_2"]
            rl = nlp_m["rouge_l"]
            b1 = nlp_m["bleu_1"]
            b4 = nlp_m["bleu_4"]
            gen_toks = nlp_m["generated_tokens"]
        else:
            em = compute_exact_match(predicted_answer, ground_truth)
            f1 = compute_f1_score(predicted_answer, ground_truth)
            r1, r2, rl, b1, b4 = f1, f1, f1, f1, 0.0
            gen_toks = len(predicted_answer.split())

        record = {
            "query": query,
            "mode": mode,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "em": float(em),
            "f1": float(f1),
            "rouge_1": float(r1),
            "rouge_2": float(r2),
            "rouge_l": float(rl),
            "bleu_1": float(b1),
            "bleu_4": float(b4),
            "generated_tokens": int(gen_toks),
            "sp_precision": sp_metrics["sp_precision"],
            "sp_recall": sp_metrics["sp_recall"],
            "sp_f1": sp_metrics["sp_f1"],
            "confidence": round(float(confidence), 4),
            "abstained": abstained,
            "token_saving_rate": round(float(token_saving_rate), 4),
            "latency_sec": round(float(t_elapsed), 4),
            "forward_calls": forward_calls,
            "peak_cuda_allocated_mb": round(float(peak_alloc), 2),
            "peak_cuda_reserved_mb": round(float(peak_res), 2),
            "selected_doc_count": len(selected_docs),
            "selected_titles": retrieved_titles,
            "m1_uncertainty": m1_info,
            "m3_telemetry": m3_telemetry,
            "permutation_stats": permutation_stats,
        }

        return record

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        mode: str = "Full_UGraphRAG",
        max_samples: Optional[int] = None,
        log_interval: int = 5,
        output_json: Optional[Union[str, Path]] = None,
        tb_logger: Optional[Any] = None,
        checkpoint_frequency: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluates a dataset batch with Checkpoint Recovery & Extended Metrics:
        calculates macro-averaged metrics, ECE/MCE/Brier scores, and logs to TensorBoard.
        """
        samples_to_eval = dataset[:max_samples] if max_samples else dataset
        n_samples = len(samples_to_eval)

        # Initialize CheckpointManager for soft-interrupt recovery
        ckpt_mgr = None
        records: List[Dict[str, Any]] = []
        completed_keys: Set[str] = set()
        if output_json and CheckpointManager is not None:
            ckpt_mgr = CheckpointManager(output_json, save_frequency=checkpoint_frequency)
            records, completed_keys = ckpt_mgr.load_checkpoint()

        logger.info(f"Starting evaluation on {n_samples} samples | Mode: {mode} (Already completed: {len(records)})")
        t_start = time.perf_counter()

        try:
            for i, sample in enumerate(samples_to_eval):
                sample_key = str(sample.get("id", sample.get("sample_id", sample.get("question", i))))
                if sample_key in completed_keys:
                    continue

                rec = self.run_pipeline_single(sample, mode=mode, seed=42 + i)
                records.append(rec)
                completed_keys.add(sample_key)

                # TensorBoard Step Logging
                if tb_logger is not None:
                    tb_logger.log_step_metrics(
                        len(records),
                        {
                            f"{mode}/Running_EM": float(np.mean([r["em"] for r in records])) * 100.0,
                            f"{mode}/Running_F1": float(np.mean([r["f1"] for r in records])) * 100.0,
                            f"{mode}/Running_ROUGE_L": float(np.mean([r.get("rouge_l", 0.0) for r in records])) * 100.0,
                            f"{mode}/Running_SP_F1": float(np.mean([r["sp_f1"] for r in records])),
                            f"{mode}/Running_Confidence": float(np.mean([r["confidence"] for r in records])),
                            f"Resource/Latency_Sec": rec["latency_sec"],
                            f"Resource/CUDA_Allocated_MB": rec["peak_cuda_allocated_mb"],
                        }
                    )

                # Periodic Checkpoint Saving
                if ckpt_mgr is not None:
                    ckpt_mgr.save_checkpoint(records, completed_keys=completed_keys)

                if (len(records)) % log_interval == 0 or len(records) == n_samples:
                    logger.info(
                        f"[{mode}] Progress {len(records)}/{n_samples} | "
                        f"EM: {np.mean([r['em'] for r in records]):.4f} | "
                        f"F1: {np.mean([r['f1'] for r in records]):.4f} | "
                        f"ROUGE-L: {np.mean([r.get('rouge_l', 0.0) for r in records]):.4f} | "
                        f"Sp-F1: {np.mean([r['sp_f1'] for r in records]):.4f}"
                    )

        except (KeyboardInterrupt, SystemExit) as interrupt_err:
            if ckpt_mgr is not None:
                ckpt_mgr.save_checkpoint(records, completed_keys=completed_keys, force=True)
                logger.warning(f"\n[!] Soft Interrupt caught! Checkpoint saved ({len(records)}/{n_samples} items). Resume anytime by rerunning.")
            raise interrupt_err

        t_total = time.perf_counter() - t_start

        # Aggregate metrics
        avg_em = float(np.mean([r["em"] for r in records])) if records else 0.0
        avg_f1 = float(np.mean([r["f1"] for r in records])) if records else 0.0
        avg_r1 = float(np.mean([r.get("rouge_1", 0.0) for r in records])) if records else 0.0
        avg_r2 = float(np.mean([r.get("rouge_2", 0.0) for r in records])) if records else 0.0
        avg_rl = float(np.mean([r.get("rouge_l", 0.0) for r in records])) if records else 0.0
        avg_b1 = float(np.mean([r.get("bleu_1", 0.0) for r in records])) if records else 0.0
        avg_b4 = float(np.mean([r.get("bleu_4", 0.0) for r in records])) if records else 0.0
        avg_sp_p = float(np.mean([r["sp_precision"] for r in records])) if records else 0.0
        avg_sp_r = float(np.mean([r["sp_recall"] for r in records])) if records else 0.0
        avg_sp_f1 = float(np.mean([r["sp_f1"] for r in records])) if records else 0.0
        avg_token_saving = float(np.mean([r["token_saving_rate"] for r in records])) if records else 0.0
        avg_latency = float(np.mean([r["latency_sec"] for r in records])) if records else 0.0
        avg_calls = float(np.mean([r["forward_calls"] for r in records])) if records else 0.0
        abstention_rate = float(np.mean([1.0 if r["abstained"] else 0.0 for r in records])) if records else 0.0

        confidences = [r["confidence"] for r in records]
        accuracies = [1.0 if r["f1"] >= 0.5 else 0.0 for r in records]
        abstained_flags = [bool(r["abstained"]) for r in records]

        # Advanced calibration and selective metrics
        if compute_detailed_calibration is not None:
            calib_res = compute_detailed_calibration(confidences, accuracies, n_bins=10)
            ece = calib_res["ece"]
            mce = calib_res["mce"]
            brier_score = calib_res["brier_score"]
        else:
            ece = compute_ece(confidences, accuracies)
            mce = 0.0
            brier_score = float(np.mean([(c - a)**2 for c, a in zip(confidences, accuracies)])) if confidences else 0.0

        if compute_selective_prediction_metrics is not None:
            sel_res = compute_selective_prediction_metrics(abstained_flags, accuracies)
            selective_coverage = sel_res["coverage"]
            selective_risk = sel_res["selective_risk"]
        else:
            selective_coverage = 1.0 - abstention_rate
            selective_risk = 0.0

        auroc = (
            compute_auroc(confidences, [1 if a >= 0.5 else 0 for a in accuracies])
            if (compute_auroc is not None and confidences)
            else 0.5
        )

        peak_allocated = float(max([r["peak_cuda_allocated_mb"] for r in records], default=0.0))
        peak_reserved = float(max([r["peak_cuda_reserved_mb"] for r in records], default=0.0))

        # Aggregate permutation metrics if applicable
        permutation_summary = {}
        if mode == "M2_Permutation_Test":
            perm_dfs_f1 = [r["permutation_stats"].get("dfs_f1", 0.0) for r in records if r.get("permutation_stats")]
            perm_rev_f1 = [r["permutation_stats"].get("reverse_f1", 0.0) for r in records if r.get("permutation_stats")]
            perm_shuf_mean = [r["permutation_stats"].get("shuffles_f1_mean", 0.0) for r in records if r.get("permutation_stats")]
            perm_shuf_std = [r["permutation_stats"].get("shuffles_f1_std", 0.0) for r in records if r.get("permutation_stats")]
            perm_drop = [r["permutation_stats"].get("max_performance_drop", 0.0) for r in records if r.get("permutation_stats")]

            permutation_summary = {
                "mean_dfs_f1": round(float(np.mean(perm_dfs_f1)), 4) if perm_dfs_f1 else 0.0,
                "mean_reverse_f1": round(float(np.mean(perm_rev_f1)), 4) if perm_rev_f1 else 0.0,
                "mean_shuffles_f1": round(float(np.mean(perm_shuf_mean)), 4) if perm_shuf_mean else 0.0,
                "mean_shuffles_std": round(float(np.mean(perm_shuf_std)), 4) if perm_shuf_std else 0.0,
                "avg_max_performance_drop": round(float(np.mean(perm_drop)), 4) if perm_drop else 0.0,
                "dfs": {
                    "mean": round(float(np.mean(perm_dfs_f1)), 4) if perm_dfs_f1 else 0.0,
                    "std": round(float(np.std(perm_dfs_f1)), 4) if perm_dfs_f1 else 0.0,
                    "values": perm_dfs_f1,
                },
                "reverse": {
                    "mean": round(float(np.mean(perm_rev_f1)), 4) if perm_rev_f1 else 0.0,
                    "std": round(float(np.std(perm_rev_f1)), 4) if perm_rev_f1 else 0.0,
                    "values": perm_rev_f1,
                },
                "shuffles": {
                    "mean": round(float(np.mean(perm_shuf_mean)), 4) if perm_shuf_mean else 0.0,
                    "std": round(float(np.std(perm_shuf_mean)), 4) if perm_shuf_mean else 0.0,
                    "values": perm_shuf_mean,
                },
                "max_drop": round(float(np.mean(perm_drop)), 4) if perm_drop else 0.0,
            }

        summary = {
            "mode": mode,
            "total_samples": n_samples,
            "total_wall_time_sec": round(t_total, 2),
            "exact_match": round(avg_em * 100.0, 2),
            "f1_score": round(avg_f1 * 100.0, 2),
            "sp_precision": round(avg_sp_p * 100.0, 2),
            "sp_recall": round(avg_sp_r * 100.0, 2),
            "sp_f1": round(avg_sp_f1 * 100.0, 2),
            "rouge_1": round(avg_r1 * 100.0, 2),
            "rouge_2": round(avg_r2 * 100.0, 2),
            "rouge_l": round(avg_rl * 100.0, 2),
            "bleu_1": round(avg_b1 * 100.0, 2),
            "bleu_4": round(avg_b4 * 100.0, 2),
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier_score, 4),
            "auroc": round(float(auroc), 4),
            "coverage": round(selective_coverage * 100.0, 2),
            "selective_coverage": round(selective_coverage * 100.0, 2),
            "selective_risk": round(selective_risk * 100.0, 2),
            "abstention_rate": round(abstention_rate * 100.0, 2),
            "token_saving_rate": round(avg_token_saving * 100.0, 2),
            "peak_cuda_allocated_mb": round(peak_allocated, 2),
            "peak_cuda_reserved_mb": round(peak_reserved, 2),
            "avg_latency_sec": round(avg_latency, 4),
            "avg_forward_calls": round(avg_calls, 2),
            "permutation_summary": permutation_summary,
            "sample_records": records,
        }

        # Finalize checkpoint & Log summary to TensorBoard
        if ckpt_mgr is not None and len(records) >= n_samples:
            ckpt_mgr.finalize(summary)

        if tb_logger is not None:
            tb_logger.log_summary_metrics(mode, summary)

        return summary


# --- Section 4: Standalone CLI & Self-Test Runner ---

def build_pipeline_from_args(
    model_name_or_path: Optional[str] = None,
    dense_model_path: Optional[str] = None,
    dry_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    budget: int = 512,
) -> UGraphRAGPipeline:
    """Helper to construct UGraphRAGPipeline with generators and retrievers."""
    generator = None
    retriever = None

    if dry_run or model_name_or_path is None:
        generator = QwenNF4Generator(model_name_or_path=None, device=device, dry_run=True)
    else:
        try:
            generator = QwenNF4Generator(model_name_or_path=model_name_or_path, device=device, dry_run=False)
        except Exception as e:
            logger.warning(f"Failed to load generator ({e}). Falling back to dry-run generator.")
            generator = QwenNF4Generator(model_name_or_path=None, device=device, dry_run=True)

    if dense_model_path and not dry_run:
        try:
            retriever = HybridRetriever(corpus=[], dense_model_path=dense_model_path)
        except Exception as e:
            logger.warning(f"Failed to load hybrid retriever ({e}). Proceeding without global corpus retriever.")
            retriever = None

    return UGraphRAGPipeline(
        retriever=retriever,
        generator=generator,
        token_budget=budget,
        dense_model_path=dense_model_path,
    )


def run_pipeline_smoke_test():
    """Runs a complete 8-mode smoke test using mock multi-hop contexts."""
    logger.info("Running UGraphRAGPipeline 8-mode ablation self-check...")
    pipeline = build_pipeline_from_args(dry_run=True)

    test_sample = {
        "id": "smoke_001",
        "question": "Which award did the film directed by Christopher Nolan in 2010 receive at the Oscars?",
        "answer": "Academy Award for Best Cinematography",
        "supporting_facts": [["Inception (film)", 0], ["83rd Academy Awards", 1]],
        "context": [
            ["Inception (film)", ["Inception is a 2010 science fiction film written and directed by Christopher Nolan.", "The film won four Oscars at the 83rd Academy Awards."]],
            ["Christopher Nolan", ["Christopher Edward Nolan is a British-American filmmaker known for Hollywood blockbusters."]],
            ["83rd Academy Awards", ["The 83rd Academy Awards ceremony honored the best films of 2010 in Los Angeles.", "Inception won Best Cinematography."]],
            ["The Dark Knight", ["The Dark Knight is a 2008 superhero film directed by Christopher Nolan."]],
            ["Leonardo DiCaprio", ["Leonardo Wilhelm DiCaprio is an American actor who starred in Inception."]],
            ["Distractor One", ["Random irrelevant text that has nothing to do with Nolan or Oscars."]],
            ["Distractor Two", ["Another noise paragraph designed to test evidence selection."]],
        ],
    }

    all_passed = True
    results_summary = {}

    for mode in UGraphRAGPipeline.SUPPORTED_MODES:
        logger.info(f"--- Verifying mode: {mode} ---")
        out = pipeline.run_pipeline_single(test_sample, mode=mode)
        assert "em" in out and "f1" in out and "sp_f1" in out, f"Missing core keys in {mode}"
        assert "token_saving_rate" in out and "latency_sec" in out, f"Missing profile keys in {mode}"
        
        if mode == "M2_Permutation_Test":
            assert "permutation_stats" in out and "shuffles_f1_mean" in out["permutation_stats"]
            logger.info(f"Permutation Stats verified: {out['permutation_stats']}")

        results_summary[mode] = {
            "em": out["em"],
            "f1": out["f1"],
            "sp_f1": out["sp_f1"],
            "saving": out["token_saving_rate"],
        }
        logger.info(f"Mode {mode} PASSED. Selected docs: {out['selected_doc_count']}, Sp-F1: {out['sp_f1']}")

    logger.info("UGraphRAGPipeline 8-mode verification completed successfully!")
    return results_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="U-GraphRAG Pipeline Evaluation Engine")
    parser.add_argument("--mode", type=str, default="Full_UGraphRAG", choices=UGraphRAGPipeline.SUPPORTED_MODES + ["all_smoke_test"])
    parser.add_argument("--dataset-path", type=str, default="data/hotpotqa_dev_300.json")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--dense-path", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "all_smoke_test":
        run_pipeline_smoke_test()
    else:
        pipeline = build_pipeline_from_args(
            model_name_or_path=args.model_path if args.model_path else None,
            dense_model_path=args.dense_path if args.dense_path else None,
            dry_run=args.dry_run,
            budget=args.token_budget,
        )

        if os.path.exists(args.dataset_path):
            with open(args.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items from {args.dataset_path}")
            summary = pipeline.evaluate_dataset(data, mode=args.mode, max_samples=args.max_samples)
            
            if args.output_json:
                os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved evaluation report to {args.output_json}")
            else:
                print(json.dumps({k: v for k, v in summary.items() if k != "sample_records"}, indent=2))
        else:
            logger.warning(f"Dataset path {args.dataset_path} not found. Running smoke test instead.")
            run_pipeline_smoke_test()
