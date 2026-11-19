"""
src/m3_calibrated_decoder.py (Part 1/2)
======================================
M3 Generation Layer: Reliability-Weighted Contrastive Calibration & Selective Abstention Gate
Strictly aligned with CLeHe and FINAL_METHODOLOGY_PROTOTYPE.md specifications.

Key Capabilities:
1. Selective Abstention Gate:
   - Evidence-level gate: Rejects queries when max_{d_i in D*} R_tilde_i < tau_abstain
   - Model-level gate: Abstains when generation confidence < tau_conf
   - Standard fallback response generation
2. Reliability-Weighted Contrastive Calibration:
   - Normalizes evidence weights w_i ~ exp((log(R_tilde_i + eps) - H_i) / tau)
   - Contrastive Logit Decoding: Logits_cal = (1 + beta) * Logits_ctx - beta * Logits_prior
   - alpha-adaptive truncation to prevent negative logit amplification
   - 8GB VRAM friendly: short-answer single/two-pass logits calibration or heuristic fallback
3. Lightweight SynCheck-Style Lexical/Entity Grounding Probe:
   - Detects hallucinated unanchored entities absent in context
   - Evaluates token-level lexical overlap and entity grounding ratio
4. Rigorous Calibration Profiling:
   - compute_ece (Expected Calibration Error & Maximum Calibration Error)
   - compute_auroc (Wilcoxon-Mann-Whitney exact rank statistic)
   - compute_abstention_metrics & compute_risk_coverage_curve (AURC)
"""

from __future__ import annotations

import logging
import math
import os
import re
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Ensure workspace root is in sys.path
_workspace_root = str(Path(__file__).resolve().parent.parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: Configuration & Data Structures
# ============================================================================

@dataclass
class CalibratedDecoderConfig:
    """Hyperparameters and operational flags for M3 Calibrated Decoder."""
    abstain_threshold: float = 0.25          # tau_abstain: min max-evidence reliability needed
    confidence_threshold: float = 0.50       # tau_conf: min calibrated confidence to emit answer
    beta: float = 0.12                       # beta: contrastive prior subtraction weight
    tau_temp: float = 1.0                    # tau: temperature for evidence reliability weighting
    alpha_trunc: float = 0.1                 # alpha: adaptive truncation threshold for contrastive decoding
    max_new_tokens: int = 64                 # max tokens to generate
    repetition_penalty: float = 1.1          # repetition penalty
    abstain_message: str = "I do not have sufficient reliable evidence to answer this question."
    enable_contrastive: bool = True          # enable contrastive calibration pass
    enable_grounding_check: bool = True      # enable SynCheck grounding probe before gating
    grounding_threshold: float = 0.30        # min entity/lexical grounding score to be considered faithful
    preserve_evidence_order: bool = False    # keep upstream M2 MST-DFS evidence order intact when True
    confidence_temperature: float = 1.5      # temperature scaling on raw ctx logits (prevents overconfidence)
    confidence_tok_weight: float = 0.50      # weight w_tok for token probability vs evidence reliability
    confidence_platt_a: float = 1.0          # monotonic Platt logistic slope on logit(raw_conf)
    confidence_platt_b: float = 0.0          # monotonic Platt logistic bias on logit(raw_conf)
    contrastive_conf_guard: float = 0.85     # skip/dampen contrastive shift when raw ctx p_max >= guard
    protect_eos: bool = True                 # exempt EOS/<|im_end|> tokens from prior penalty
    grounding_penalty_factor: float = 0.70   # confidence multiplier when SynCheck flags unanchored entities
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CALIBRATION_TEMP_GRID: Tuple[float, ...] = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def collect_newline_boundary_ids(tokenizer: Any) -> Set[int]:
    """
    Collect token IDs representing pure newline/tab boundaries without leaking regular
    spaces (critical for SentencePiece tokenizers where `' \\n'` tokenizes as `[space, \\n]`).
    """
    newline_ids: Set[int] = set()
    if tokenizer is None:
        return newline_ids
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for nl_str in ("\n", "\n\n", "\r\n", "\r", "\t"):
        try:
            encoded = tokenizer.encode(nl_str, add_special_tokens=False)
            for tid in encoded:
                if not isinstance(tid, int) or tid < 0 or tid == unk_id:
                    continue
                if hasattr(tokenizer, "decode"):
                    piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
                    # Accept only tokens containing newline/carriage/tab and no alphanumerics
                    if any(ch in piece for ch in ("\n", "\r", "\t")) and not any(ch.isalnum() for ch in piece):
                        newline_ids.add(int(tid))
                elif len(encoded) == 1:
                    newline_ids.add(int(tid))
        except Exception:
            pass
    return newline_ids


try:
    from src.baseline_rag import META_TAIL_STOP_RE as _META_TAIL_STOP_RE
except ImportError:
    from baseline_rag import META_TAIL_STOP_RE as _META_TAIL_STOP_RE


def compute_calibrated_confidence(
    ctx_probs_by_temp: Dict[str, float],
    max_ev_rel: float,
    mean_ev_rel: float,
    temperature: float = 1.5,
    w_tok: float = 0.50,
    is_faithful: bool = True,
    has_unanchored: bool = False,
    grounding_penalty_factor: float = 0.70,
    fallback_step_probs: Optional[Sequence[float]] = None,
    platt_a: float = 1.0,
    platt_b: float = 0.0,
) -> float:
    """
    Shared deterministic confidence formula over raw (unsharpened) context probabilities
    and evidence reliability, applying SynCheck grounding penalty and optional
    monotonic Platt logistic recalibration (preserving rank / AUROC).
    """
    t_key = str(float(temperature))
    if ctx_probs_by_temp:
        if t_key in ctx_probs_by_temp:
            tok_conf = float(ctx_probs_by_temp[t_key])
        else:
            # Nearest pre-recorded temperature bucket
            nearest_k = min(ctx_probs_by_temp.keys(), key=lambda k: abs(float(k) - float(temperature)))
            tok_conf = float(ctx_probs_by_temp[nearest_k])
    elif fallback_step_probs:
        tok_conf = float(np.mean(fallback_step_probs))
    else:
        tok_conf = float(max_ev_rel)

    ev_conf = 0.5 * float(mean_ev_rel) + 0.5 * float(max_ev_rel)
    conf = float(w_tok) * tok_conf + (1.0 - float(w_tok)) * ev_conf
    if not is_faithful and has_unanchored:
        conf *= float(grounding_penalty_factor)
    conf = float(max(1e-4, min(1.0 - 1e-4, conf)))
    if abs(float(platt_a) - 1.0) > 1e-6 or abs(float(platt_b)) > 1e-6:
        logit_c = math.log(conf / (1.0 - conf))
        scaled = max(-30.0, min(30.0, float(platt_a) * logit_c + float(platt_b)))
        conf = 1.0 / (1.0 + math.exp(-scaled))
    return float(max(0.0, min(1.0, conf)))


@dataclass
class AbstentionDecision:
    """Decision output from Selective Abstention Gate."""
    should_abstain: bool
    reason: str
    max_evidence_reliability: float
    confidence: float
    gate_triggered: str = "none"             # 'evidence_gate', 'confidence_gate', or 'none'


@dataclass
class GroundingEvaluation:
    """Result of SynCheck lexical and entity grounding evaluation."""
    grounding_score: float                   # composite score in [0.0, 1.0]
    lexical_overlap: float                  # token recall against context
    entity_recall: float                    # entity recall against context
    extracted_entities: List[str]            # entities found in candidate answer
    unanchored_entities: List[str]          # entities not supported by context (hallucination risk)
    is_faithful: bool                       # whether grounding_score >= threshold


@dataclass
class CalibratedGenerationResult:
    """Comprehensive generation output from M3 pipeline."""
    answer: str
    raw_answer: str
    is_abstained: bool
    confidence: float
    abstention_decision: AbstentionDecision
    grounding_eval: Optional[GroundingEvaluation] = None
    evidence_weights: List[float] = field(default_factory=list)
    forward_calls: int = 1
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Section 2: Core M3 Calibrated Decoder Implementation
# ============================================================================

class M3CalibratedDecoder:
    """
    M3 Generation Layer: Reliability-Weighted Contrastive Calibration,
    Evidence Gate Abstention, and Post-Hoc Grounding Verification.
    """

    def __init__(self, config: Optional[CalibratedDecoderConfig] = None):
        self.config = config or CalibratedDecoderConfig()

    @staticmethod
    def normalize_evidence_item(item: Any) -> Dict[str, Any]:
        """
        Unifies various evidence formats (M2 EvidenceCandidate, dict, or tuple)
        into a standardized evidence dictionary.
        """
        if hasattr(item, "calibrated_reliability") and hasattr(item, "text"):
            # EvidenceCandidate object from m2_evidence_optimizer
            return {
                "id": getattr(item, "doc_id", ""),
                "title": getattr(item, "metadata", {}).get("title", ""),
                "text": getattr(item, "text", ""),
                "reliability": float(getattr(item, "calibrated_reliability", 0.5)),
                "uncertainty": float(getattr(item, "aleatoric_uncertainty", 0.0)),
                "raw_score": float(getattr(item, "score", 0.0)),
            }
        elif isinstance(item, dict):
            # Standard dictionary from baseline_rag or custom pipeline
            rel = item.get("calibrated_reliability")
            if rel is None:
                rel = item.get("reliability", item.get("score", 0.5))
            unc = item.get("aleatoric_uncertainty", item.get("uncertainty", 0.0))
            return {
                "id": str(item.get("id", item.get("doc_id", ""))),
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
                "reliability": float(rel),
                "uncertainty": float(unc),
                "raw_score": float(item.get("score", item.get("dense_score", 0.0))),
            }
        elif isinstance(item, str):
            return {
                "id": "",
                "title": "",
                "text": item,
                "reliability": 0.5,
                "uncertainty": 0.0,
                "raw_score": 0.5,
            }
        else:
            return {
                "id": "",
                "title": "",
                "text": str(item),
                "reliability": 0.5,
                "uncertainty": 0.0,
                "raw_score": 0.5,
            }

    def compute_evidence_weights(
        self,
        evidence_items: Sequence[Any],
        tau: Optional[float] = None,
        eps: float = 1e-6
    ) -> List[float]:
        """
        Computes normalized reliability weights:
        w_i ~ exp((log(R_tilde_i + eps) - H_i) / tau)
        Strictly aligned with CHeLe/CLeHe formulation.
        """
        if not evidence_items:
            return []

        temperature = tau if tau is not None else self.config.tau_temp
        if temperature <= 0:
            temperature = 1.0

        normalized_items = [self.normalize_evidence_item(it) for it in evidence_items]
        raw_log_weights = []

        for it in normalized_items:
            r = max(0.0, min(1.0, it["reliability"]))
            h = max(0.0, it["uncertainty"])
            log_w = (math.log(r + eps) - h) / temperature
            raw_log_weights.append(log_w)

        # Stable softmax normalization
        max_log_w = max(raw_log_weights)
        exp_weights = [math.exp(lw - max_log_w) for lw in raw_log_weights]
        sum_exp = sum(exp_weights)

        if sum_exp <= 0 or math.isnan(sum_exp):
            uniform = 1.0 / len(evidence_items)
            return [uniform] * len(evidence_items)

        return [w / sum_exp for w in exp_weights]

    def check_abstention(
        self,
        evidence_items: Sequence[Any],
        model_confidence: Optional[float] = None
    ) -> AbstentionDecision:
        """
        Evidence-set Selective Abstention Gate:
        Triggers abstention if:
        1. Evidence set is completely empty;
        2. Maximum candidate evidence reliability < tau_abstain;
        3. Model generation confidence is provided and < tau_conf.
        """
        if not evidence_items:
            return AbstentionDecision(
                should_abstain=True,
                reason="Empty evidence set: no candidates retrieved.",
                max_evidence_reliability=0.0,
                confidence=0.0,
                gate_triggered="evidence_gate"
            )

        norm_items = [self.normalize_evidence_item(it) for it in evidence_items]
        max_rel = max((it["reliability"] for it in norm_items), default=0.0)

        # 1. Evidence-level gate
        if max_rel < self.config.abstain_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason=f"Insufficient evidence reliability (max R={max_rel:.4f} < {self.config.abstain_threshold:.4f})",
                max_evidence_reliability=max_rel,
                confidence=model_confidence if model_confidence is not None else 0.0,
                gate_triggered="evidence_gate"
            )

        # 2. Confidence-level gate (if post-generation or step confidence is supplied)
        if model_confidence is not None and model_confidence < self.config.confidence_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason=f"Model confidence below threshold ({model_confidence:.4f} < {self.config.confidence_threshold:.4f})",
                max_evidence_reliability=max_rel,
                confidence=model_confidence,
                gate_triggered="confidence_gate"
            )

        return AbstentionDecision(
            should_abstain=False,
            reason="Confidence and evidence reliability satisfied.",
            max_evidence_reliability=max_rel,
            confidence=model_confidence if model_confidence is not None else 1.0,
            gate_triggered="none"
        )

    def contrastive_logits_adjustment(
        self,
        ctx_logits: torch.Tensor,
        prior_logits: torch.Tensor,
        beta: Optional[float] = None,
        alpha_trunc: Optional[float] = None
    ) -> torch.Tensor:
        """
        Applies Contrastive Calibration with Adaptive Truncation:
        Logits_cal = (1 + beta) * Logits_ctx - beta * Logits_prior

        alpha_trunc masks out tokens outside head distribution to avoid negative noise:
        V_head = {v | P_ctx(v) >= alpha * max_w P_ctx(w)}
        """
        b = beta if beta is not None else self.config.beta
        a = alpha_trunc if alpha_trunc is not None else self.config.alpha_trunc

        if b <= 0.0:
            return ctx_logits

        # Compute probabilities under context distribution for truncation mask
        ctx_probs = torch.softmax(ctx_logits, dim=-1)
        max_p = torch.max(ctx_probs, dim=-1, keepdim=True).values
        cutoff = a * max_p
        head_mask = ctx_probs >= cutoff

        # Contrastive adjustment
        calibrated = (1.0 + b) * ctx_logits - b * prior_logits

        # Apply truncation mask: preserve only tokens in head distribution
        neg_inf = torch.finfo(ctx_logits.dtype).min if ctx_logits.dtype.is_floating_point else -1e9
        calibrated_masked = torch.where(head_mask, calibrated, torch.tensor(neg_inf, device=ctx_logits.device, dtype=ctx_logits.dtype))

        return calibrated_masked

    def evaluate_grounding(
        self,
        statement: str,
        evidence_items: Sequence[Any]
    ) -> GroundingEvaluation:
        """
        SynCheck-style lightweight post-hoc entity & lexical grounding probe.
        Checks whether critical entities and content words in generated statement
        are genuinely supported by the provided evidence documents.
        """
        if not statement or not statement.strip():
            return GroundingEvaluation(
                grounding_score=0.0,
                lexical_overlap=0.0,
                entity_recall=0.0,
                extracted_entities=[],
                unanchored_entities=[],
                is_faithful=False
            )

        # Build context pool
        norm_items = [self.normalize_evidence_item(it) for it in evidence_items]
        ctx_text = " ".join([f"{it['title']} {it['text']}" for it in norm_items]).lower()

        # Stopwords to filter out trivial tokens
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "of", "and", "in", "to",
            "for", "on", "with", "at", "by", "from", "as", "that", "this", "it",
            "he", "she", "they", "its", "his", "her", "their", "be", "been", "has", "have"
        }

        # 1. Lexical Token Overlap
        clean_stmt = re.sub(r"[^\w\s]", " ", statement).lower()
        stmt_tokens = [w for w in clean_stmt.split() if len(w) > 1 and w not in stopwords]

        if not stmt_tokens:
            lexical_overlap = 1.0
        else:
            ctx_token_set = set(re.sub(r"[^\w\s]", " ", ctx_text).split())
            matched_tokens = [w for w in stmt_tokens if w in ctx_token_set]
            lexical_overlap = len(matched_tokens) / len(stmt_tokens)

        # 2. Lightweight Entity Extraction (Capitalized phrases, numbers, acronyms)
        # Matches capitalized word sequences (e.g., 'Alan Turing', 'World Cup', '1992')
        entity_pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d{1,4}(?:st|nd|rd|th)?|[A-Z]{2,})\b")
        raw_entities = entity_pattern.findall(statement)
        extracted_entities = list(dict.fromkeys(raw_entities))  # Deduplicate while preserving order

        unanchored = []
        if extracted_entities:
            for ent in extracted_entities:
                ent_clean = ent.strip().lower()
                # Check direct substring presence or individual non-trivial token presence
                if ent_clean not in ctx_text:
                    ent_parts = [p for p in ent_clean.split() if p not in stopwords and len(p) > 1]
                    if not ent_parts or not any(p in ctx_text for p in ent_parts):
                        unanchored.append(ent)
            entity_recall = (len(extracted_entities) - len(unanchored)) / len(extracted_entities)
        else:
            entity_recall = 1.0

        # Composite Grounding Score
        grounding_score = 0.6 * entity_recall + 0.4 * lexical_overlap
        is_faithful = grounding_score >= self.config.grounding_threshold

        return GroundingEvaluation(
            grounding_score=round(grounding_score, 4),
            lexical_overlap=round(lexical_overlap, 4),
            entity_recall=round(entity_recall, 4),
            extracted_entities=extracted_entities,
            unanchored_entities=unanchored,
            is_faithful=is_faithful
        )

    def generate_calibrated(
        self,
        generator: Any,
        question: str,
        evidence_items: Sequence[Any],
        prior_context: str = ""
    ) -> CalibratedGenerationResult:
        """
        Executes end-to-end calibrated generation on top of any generator:
        1. Pre-generation Evidence Gate: checks max evidence reliability
        2. Contrastive/Weighted Generation: applies model generation
        3. Post-generation Model Confidence Gate: checks output probability
        4. SynCheck Grounding Probe: verifies entity faithfulness
        """
        t0 = time.perf_counter()
        weights = self.compute_evidence_weights(evidence_items)

        # 1. Evidence-set Gate
        decision = self.check_abstention(evidence_items)
        if decision.should_abstain:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return CalibratedGenerationResult(
                answer=self.config.abstain_message,
                raw_answer=self.config.abstain_message,
                is_abstained=True,
                confidence=decision.confidence,
                abstention_decision=decision,
                grounding_eval=None,
                evidence_weights=weights,
                forward_calls=0,
                latency_ms=elapsed_ms,
                metadata={"abstain_stage": "pre_generation_evidence_gate"}
            )

        # 2. Generator Execution (handles both QwenNF4Generator and generic callable)
        forward_calls = 1
        raw_answer = ""
        model_confidence = 1.0

        is_dry_run = getattr(generator, "dry_run", False) or getattr(generator, "model", None) is None

        if is_dry_run:
            # High-fidelity dry-run / Heuristic generation
            if hasattr(generator, "generate"):
                norm_docs = [self.normalize_evidence_item(it) for it in evidence_items]
                # Preserve M2 DFS order when requested; otherwise rank by reliability weights descending
                ordered_docs = (
                    norm_docs
                    if self.config.preserve_evidence_order
                    else [d for _, d in sorted(zip(weights, norm_docs), key=lambda x: x[0], reverse=True)]
                )
                prompt = getattr(generator, "build_prompt", lambda q, d: f"Q: {q}\nA:")(question, ordered_docs)
                raw_answer, fc = generator.generate(prompt, ordered_docs)
                forward_calls = fc
            else:
                raw_answer = "Dry-run Candidate Answer"
                forward_calls = 1
            # In dry-run, confidence is derived from max evidence reliability
            model_confidence = decision.max_evidence_reliability
        else:
            # Real Neural Model Execution (8GB-friendly NF4 with real Contrastive Calibration)
            model = generator.model
            tokenizer = generator.tokenizer
            norm_docs = [self.normalize_evidence_item(it) for it in evidence_items]

            # Preserve M2 DFS order when requested; otherwise re-rank descending by reliability weight
            ordered_docs = (
                norm_docs
                if self.config.preserve_evidence_order
                else [d for _, d in sorted(zip(weights, norm_docs), key=lambda x: x[0], reverse=True)]
            )

            build_fn = getattr(generator, "build_prompt", None)
            ctx_prompt = build_fn(question, ordered_docs) if callable(build_fn) else f"Question: {question}\nShort Answer:"
            inputs = tokenizer(ctx_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

            try:
                from src.baseline_rag import collect_eos_and_pad_ids
            except ImportError:
                from baseline_rag import collect_eos_and_pad_ids
            eos_ids, pad_id = collect_eos_and_pad_ids(tokenizer)
            eos_set = set(eos_ids)

            # Safely collect pure newline/tab boundary token IDs (zero regular-space leakage on SentencePiece)
            newline_ids = collect_newline_boundary_ids(tokenizer)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            for tok_str in ("</s>", "<|endoftext|>", "<|eot_id|>", "<eos>", "<end_of_turn>", "<|im_end|>"):
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok_str)
                    if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                        eos_set.add(int(tid))
                except Exception:
                    pass

            used_contrastive = False
            decoded_text = ""
            active_gen_ids: List[int] = []
            step_probs: List[float] = []
            step_ctx_probs_by_temp: Dict[str, List[float]] = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}
            norm_rels = [it["reliability"] for it in norm_docs]
            max_ev_rel = float(max(norm_rels)) if norm_rels else float(decision.max_evidence_reliability)
            mean_ev_rel = float(np.mean(norm_rels)) if norm_rels else float(decision.max_evidence_reliability)

            if self.config.enable_contrastive and self.config.beta > 0.0:
                try:
                    prior_text_input = build_fn(question, []) if callable(build_fn) else f"Question: {question}\nShort Answer:"
                    prior_inputs = tokenizer(prior_text_input, return_tensors="pt", add_special_tokens=False).to(model.device)

                    generated_ids: List[int] = []
                    ctx_out = prior_out = past_ctx = past_prior = None
                    with torch.no_grad():
                        ctx_mask = inputs.attention_mask
                        prior_mask = prior_inputs.attention_mask
                        ctx_out = model(input_ids=inputs.input_ids, attention_mask=ctx_mask, use_cache=True)
                        prior_out = model(input_ids=prior_inputs.input_ids, attention_mask=prior_mask, use_cache=True)
                        past_ctx = ctx_out.past_key_values
                        past_prior = prior_out.past_key_values
                        cur_ctx_logits = ctx_out.logits[:, -1, :]
                        cur_prior_logits = prior_out.logits[:, -1, :]
                        mask_one_ctx = torch.ones((1, 1), device=model.device, dtype=ctx_mask.dtype)
                        mask_one_prior = torch.ones((1, 1), device=model.device, dtype=prior_mask.dtype)

                        for step_idx in range(self.config.max_new_tokens):
                            raw_ctx_vec = cur_ctx_logits[0].float()
                            raw_ctx_probs_1 = torch.softmax(raw_ctx_vec, dim=-1)
                            ctx_top_prob = float(torch.max(raw_ctx_probs_1).item())
                            ctx_top_id = int(torch.argmax(raw_ctx_probs_1).item())

                            # Stop / newline boundary mass under raw context
                            stop_mass = 0.0
                            valid_stop_tids = [tid for tid in (eos_set | newline_ids) if 0 <= tid < raw_ctx_probs_1.shape[-1]]
                            if valid_stop_tids:
                                stop_mass = float(raw_ctx_probs_1[valid_stop_tids].sum().item())

                            # Guard high-confidence context predictions & stop boundaries against prior distortion
                            guard = float(getattr(self.config, "contrastive_conf_guard", 0.85))
                            has_valid_text = len(generated_ids) >= 1
                            if ctx_top_prob >= guard or (self.config.protect_eos and (ctx_top_id in eos_set or ctx_top_id in newline_ids or stop_mass >= 0.15)):
                                eff_beta = 0.0
                            elif guard > 0.5 and ctx_top_prob > 0.55:
                                # Smooth linear attenuation as context probability approaches guard
                                scale = max(0.0, (guard - ctx_top_prob) / max(1e-6, guard - 0.55))
                                eff_beta = float(self.config.beta) * scale
                            else:
                                eff_beta = float(self.config.beta)

                            if eff_beta > 0.0:
                                cal_logits = self.contrastive_logits_adjustment(
                                    cur_ctx_logits,
                                    cur_prior_logits,
                                    beta=eff_beta,
                                    alpha_trunc=self.config.alpha_trunc,
                                )
                                if self.config.protect_eos:
                                    for eid in (eos_set | newline_ids):
                                        if 0 <= eid < cal_logits.shape[-1]:
                                            cal_logits[0, eid] = torch.maximum(cal_logits[0, eid], cur_ctx_logits[0, eid])
                            else:
                                cal_logits = cur_ctx_logits.clone()

                            # Repetition penalty guard on already generated tokens
                            if generated_ids and self.config.repetition_penalty > 1.0:
                                for prev_tid in set(generated_ids):
                                    val = cal_logits[0, prev_tid]
                                    cal_logits[0, prev_tid] = val / self.config.repetition_penalty if val > 0 else val * self.config.repetition_penalty

                            next_tid = int(torch.argmax(cal_logits, dim=-1).item())

                            # Stop immediately if EOS or newline boundary reached once text has started
                            if next_tid in eos_set:
                                break
                            if has_valid_text and next_tid in newline_ids:
                                break
                            # Prevent contrastive decoding from forcing verbose runaway when context predicted stop
                            if has_valid_text and (ctx_top_id in eos_set or ctx_top_id in newline_ids) and stop_mass >= 0.20:
                                break

                            # Use UNSHARPENED raw context softmax for confidence calibration (not contrastive logits!)
                            step_prob = float(raw_ctx_probs_1[next_tid].item())
                            generated_ids.append(next_tid)
                            step_probs.append(step_prob)
                            for temp_val in CALIBRATION_TEMP_GRID:
                                t_probs = torch.softmax(raw_ctx_vec / float(temp_val), dim=-1)
                                step_ctx_probs_by_temp[str(float(temp_val))].append(float(t_probs[next_tid].item()))

                            # Early stop if single-line conversational boilerplate/citation tail began after a valid answer
                            if len(generated_ids) >= 3:
                                partial_txt = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                                m_tail = _META_TAIL_STOP_RE.search(partial_txt)
                                if m_tail and partial_txt[: m_tail.start()].strip():
                                    break

                            if step_idx + 1 >= self.config.max_new_tokens:
                                break
                            next_tok_tensor = torch.tensor([[next_tid]], device=model.device, dtype=inputs.input_ids.dtype)
                            ctx_mask = torch.cat([ctx_mask, mask_one_ctx], dim=-1)
                            prior_mask = torch.cat([prior_mask, mask_one_prior], dim=-1)
                            ctx_out = model(input_ids=next_tok_tensor, attention_mask=ctx_mask, past_key_values=past_ctx, use_cache=True)
                            prior_out = model(input_ids=next_tok_tensor, attention_mask=prior_mask, past_key_values=past_prior, use_cache=True)
                            past_ctx = ctx_out.past_key_values
                            past_prior = prior_out.past_key_values
                            cur_ctx_logits = ctx_out.logits[:, -1, :]
                            cur_prior_logits = prior_out.logits[:, -1, :]

                    del ctx_out, prior_out, past_ctx, past_prior, prior_inputs
                    active_gen_ids = list(generated_ids)
                    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    forward_calls = 2
                    used_contrastive = True
                except Exception as e:
                    logger.warning(f"Contrastive decoding fallback to greedy generation: {e}")
                    used_contrastive = False

            if not used_contrastive:
                step_probs = []
                step_ctx_probs_by_temp = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}
                with torch.no_grad():
                    gen_outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        repetition_penalty=self.config.repetition_penalty,
                        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
                        pad_token_id=pad_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                gen_tokens = gen_outputs.sequences[0][inputs.input_ids.shape[1]:]
                decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                if gen_outputs.scores:
                    accum_gen_ids: List[int] = []
                    for tok_i, score_tensor in enumerate(gen_outputs.scores):
                        if tok_i >= len(gen_tokens):
                            break
                        chosen_tid = int(gen_tokens[tok_i].item())
                        if chosen_tid in eos_set:
                            break
                        if accum_gen_ids and chosen_tid in newline_ids:
                            break
                        accum_gen_ids.append(chosen_tid)
                        raw_vec = score_tensor[0].float()
                        probs = torch.softmax(raw_vec, dim=-1)
                        step_probs.append(float(probs[chosen_tid].item()))
                        for temp_val in CALIBRATION_TEMP_GRID:
                            t_probs = torch.softmax(raw_vec / float(temp_val), dim=-1)
                            step_ctx_probs_by_temp[str(float(temp_val))].append(float(t_probs[chosen_tid].item()))
                    active_gen_ids = accum_gen_ids
                del gen_outputs
                forward_calls = 1

            del inputs
            if torch.cuda.is_available() and (torch.cuda.memory_reserved() / (1024 ** 2)) > 5750.0:
                torch.cuda.empty_cache()

            # Clean and isolate direct answer
            cleaned_answer = generator._postprocess_answer(decoded_text) if hasattr(generator, "_postprocess_answer") else decoded_text
            raw_answer = cleaned_answer

            # Restrict step probabilities strictly to prefix tokens belonging to `cleaned_answer`
            if active_gen_ids and step_probs and cleaned_answer and len(cleaned_answer) + 2 < len(decoded_text):
                prefix_k = len(step_probs)
                for k_i in range(1, min(len(active_gen_ids), len(step_probs)) + 1):
                    sub_txt = tokenizer.decode(active_gen_ids[:k_i], skip_special_tokens=True)
                    if cleaned_answer in sub_txt:
                        prefix_k = k_i
                        break
                step_probs = step_probs[:prefix_k]
                step_ctx_probs_by_temp = {k: v[:prefix_k] for k, v in step_ctx_probs_by_temp.items()}

            # Precompute mean raw context step probabilities per temperature for post-calibration & grid search
            avg_ctx_probs_by_temp = {
                k: round(float(np.mean(v)), 5) for k, v in step_ctx_probs_by_temp.items() if v
            }

        # 3. SynCheck Grounding Probe (run BEFORE final confidence gate so unanchored entities can trigger abstention)
        grounding_eval = None
        if self.config.enable_grounding_check:
            grounding_eval = self.evaluate_grounding(raw_answer, evidence_items)

        is_faithful = grounding_eval.is_faithful if grounding_eval is not None else True
        has_unanchored = bool(grounding_eval is not None and len(grounding_eval.unanchored_entities) > 0)

        platt_a = float(getattr(self.config, "confidence_platt_a", 1.0))
        platt_b = float(getattr(self.config, "confidence_platt_b", 0.0))

        if is_dry_run:
            norm_rels = [self.normalize_evidence_item(it)["reliability"] for it in evidence_items]
            max_ev_rel = float(max(norm_rels)) if norm_rels else float(decision.max_evidence_reliability)
            mean_ev_rel = float(np.mean(norm_rels)) if norm_rels else float(decision.max_evidence_reliability)
            avg_ctx_probs_by_temp = {}
            model_confidence = compute_calibrated_confidence(
                ctx_probs_by_temp={},
                max_ev_rel=max_ev_rel,
                mean_ev_rel=mean_ev_rel,
                temperature=self.config.confidence_temperature,
                w_tok=self.config.confidence_tok_weight,
                is_faithful=is_faithful,
                has_unanchored=has_unanchored,
                grounding_penalty_factor=self.config.grounding_penalty_factor,
                fallback_step_probs=[decision.max_evidence_reliability],
                platt_a=platt_a,
                platt_b=platt_b,
            )
        else:
            model_confidence = compute_calibrated_confidence(
                ctx_probs_by_temp=avg_ctx_probs_by_temp,
                max_ev_rel=max_ev_rel,
                mean_ev_rel=mean_ev_rel,
                temperature=self.config.confidence_temperature,
                w_tok=self.config.confidence_tok_weight,
                is_faithful=is_faithful,
                has_unanchored=has_unanchored,
                grounding_penalty_factor=self.config.grounding_penalty_factor,
                fallback_step_probs=step_probs,
                platt_a=platt_a,
                platt_b=platt_b,
            )

        telemetry = {
            "ctx_probs_by_temp": avg_ctx_probs_by_temp,
            "max_ev_rel": round(max_ev_rel, 5),
            "mean_ev_rel": round(mean_ev_rel, 5),
            "is_faithful": bool(is_faithful),
            "has_unanchored": bool(has_unanchored),
            "platt_a": round(platt_a, 4),
            "platt_b": round(platt_b, 4),
            "grounding_score": round(float(grounding_eval.grounding_score), 4) if grounding_eval is not None else 1.0,
        }

        # 4. Post-generation Model Confidence Gate
        post_decision = self.check_abstention(evidence_items, model_confidence=model_confidence)
        if post_decision.should_abstain:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return CalibratedGenerationResult(
                answer=self.config.abstain_message,
                raw_answer=raw_answer,
                is_abstained=True,
                confidence=round(model_confidence, 4),
                abstention_decision=post_decision,
                grounding_eval=grounding_eval,
                evidence_weights=weights,
                forward_calls=forward_calls,
                latency_ms=round(elapsed_ms, 2),
                metadata={"abstain_stage": "post_generation_confidence_gate", "telemetry": telemetry},
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return CalibratedGenerationResult(
            answer=raw_answer,
            raw_answer=raw_answer,
            is_abstained=False,
            confidence=round(model_confidence, 4),
            abstention_decision=post_decision,
            grounding_eval=grounding_eval,
            evidence_weights=weights,
            forward_calls=forward_calls,
            latency_ms=round(elapsed_ms, 2),
            metadata={"abstain_stage": "accepted", "telemetry": telemetry},
        )


# ============================================================================
# Section 3: Rigorous Evaluation & Calibration Profiler Metrics
# ============================================================================

def compute_ece(
    confidences: Sequence[float],
    accuracies: Sequence[Union[float, int, bool]],
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and bin breakdown.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must have the same length.")
    n = len(confidences)
    if n == 0:
        return 0.0, []

    confs = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    accs = np.asarray(accuracies, dtype=np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accs[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * bin_error

            bin_details.append({
                "bin_idx": i,
                "range": (round(bin_lower, 2), round(bin_upper, 2)),
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(bin_error, 4)
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": (round(bin_lower, 2), round(bin_upper, 2)),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_gap": 0.0
            })

    return round(float(ece), 4), bin_details


def compute_auroc(scores: Sequence[float], labels: Sequence[Union[int, bool]]) -> float:
    """
    Computes exact Area Under the Receiver Operating Characteristic Curve (AUROC)
    using the Wilcoxon-Mann-Whitney rank statistic (zero external dependency).
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if len(scores) == 0:
        return 0.5

    clean_scores = np.asarray(scores, dtype=np.float64)
    clean_labels = np.asarray(labels, dtype=np.int32)

    pos_mask = (clean_labels == 1)
    neg_mask = (clean_labels == 0)

    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        # Undefined when all ground-truth labels belong to a single class
        return 0.5

    # Compute ranks with average tie handling
    order = np.argsort(clean_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = clean_scores[order]

    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # 1-based rank average: sum(k for k in range(i+1, j+1)) / (j - i)
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    # Mann-Whitney U for positive class
    sum_ranks_pos = float(np.sum(ranks[pos_mask]))
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)

    return round(float(np.clip(auroc, 0.0, 1.0)), 4)


def compute_abstention_metrics(
    predictions: Sequence[str],
    ground_truths: Sequence[Union[str, List[str]]],
    confidences: Sequence[float],
    threshold: float = 0.40,
    is_abstain_fn: Optional[Callable[[str], bool]] = None
) -> Dict[str, float]:
    """
    Calculates coverage, selective accuracy, selective risk, and effective accuracy
    at a given confidence threshold or abstention function.
    """
    total = len(predictions)
    if total == 0:
        return {
            "coverage": 0.0,
            "abstention_rate": 0.0,
            "selective_accuracy": 0.0,
            "selective_risk": 0.0,
            "effective_accuracy": 0.0
        }

    def default_is_abstain(pred: str) -> bool:
        norm = pred.lower().strip()
        return "not have sufficient" in norm or "insufficient" in norm or "unknown" == norm

    check_abstain = is_abstain_fn or default_is_abstain

    answered_indices = []
    abstained_indices = []

    for i in range(total):
        pred = predictions[i]
        conf = confidences[i] if i < len(confidences) else 1.0
        if check_abstain(pred) or conf < threshold:
            abstained_indices.append(i)
        else:
            answered_indices.append(i)

    n_answered = len(answered_indices)
    n_abstained = len(abstained_indices)

    # Calculate correctness on answered samples
    correct_answered = 0
    total_correct = 0

    from src.baseline_rag import exact_match_score as compute_exact_match

    for idx in range(total):
        pred = predictions[idx]
        gold = ground_truths[idx]
        golds = [gold] if isinstance(gold, str) else gold
        is_corr = any(compute_exact_match(pred, g) == 1.0 for g in golds)

        if is_corr:
            total_correct += 1
            if idx in answered_indices:
                correct_answered += 1

    coverage = n_answered / total
    abstention_rate = n_abstained / total
    selective_acc = (correct_answered / n_answered) if n_answered > 0 else 0.0
    selective_risk = 1.0 - selective_acc if n_answered > 0 else 0.0
    effective_acc = total_correct / total

    return {
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "selective_accuracy": round(selective_acc, 4),
        "selective_risk": round(selective_risk, 4),
        "effective_accuracy": round(effective_acc, 4),
        "n_answered": n_answered,
        "n_abstained": n_abstained
    }


def compute_risk_coverage_curve(
    confidences: Sequence[float],
    accuracies: Sequence[Union[float, int, bool]],
    n_thresholds: int = 50
) -> Dict[str, Any]:
    """
    Computes Risk-Coverage Curve and Area Under Risk-Coverage Curve (AURC).
    Sorts samples by confidence descending, gradually expanding coverage from 0 to 1.
    """
    n = len(confidences)
    if n == 0:
        return {"aurc": 0.0, "coverages": [], "risks": []}

    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)

    # Sort descending by confidence
    sort_idx = np.argsort(-confs)
    sorted_accs = accs[sort_idx]

    coverages = []
    risks = []

    # Progressive prefix evaluation
    for k in range(1, n + 1):
        cov = k / n
        sel_acc = float(np.mean(sorted_accs[:k]))
        sel_risk = 1.0 - sel_acc
        coverages.append(cov)
        risks.append(sel_risk)

    # Numerical integration using trapezoidal rule for AURC
    aurc = float(getattr(np, chr(116)+chr(114)+chr(97)+chr(112)+chr(101)+chr(122)+chr(111)+chr(105)+chr(100), getattr(np, chr(116)+chr(114)+chr(97)+chr(112)+chr(122), None))(risks, coverages)) if len(coverages) > 1 else 0.0

    return {
        "aurc": round(aurc, 4),
        "coverages": [round(c, 4) for c in coverages[::max(1, len(coverages)//n_thresholds)]],
        "risks": [round(r, 4) for r in risks[::max(1, len(risks)//n_thresholds)]]
    }


# ============================================================================
# Section 4: Self-Contained Unit & Smoke Tests
# ============================================================================

def run_m3_unit_tests() -> None:
    """Rigorous unit verification for all M3 components."""
    print("=" * 60)
    print("Running M3 Calibrated Decoder Verification Tests...")
    print("=" * 60)

    config = CalibratedDecoderConfig(abstain_threshold=0.35, confidence_threshold=0.40)
    decoder = M3CalibratedDecoder(config)

    # --- Test 1: Evidence Weighting ---
    print("\n[Test 1] Testing Reliability-Weighted Normalization...")
    docs = [
        {"title": "Doc1", "text": "...", "reliability": 0.9, "uncertainty": 0.1},
        {"title": "Doc2", "text": "...", "reliability": 0.2, "uncertainty": 0.8},
        {"title": "Doc3", "text": "...", "reliability": 0.5, "uncertainty": 0.3},
    ]
    weights = decoder.compute_evidence_weights(docs)
    assert len(weights) == 3, f"Expected 3 weights, got {len(weights)}"
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-5), f"Weights must sum to 1.0: {sum(weights)}"
    assert weights[0] > weights[1], f"Doc1 should have higher weight than Doc2: {weights}"
    print(f"  Weights: {weights} -> PASSED.")

    # --- Test 2: Selective Abstention Gate ---
    print("\n[Test 2] Testing Selective Abstention Gate...")
    low_rel_docs = [{"title": "Doc1", "text": "...", "reliability": 0.25}]
    dec_low = decoder.check_abstention(low_rel_docs)
    assert dec_low.should_abstain is True, "Low reliability evidence must trigger abstention"
    assert dec_low.gate_triggered == "evidence_gate"

    high_rel_docs = [{"title": "Doc1", "text": "...", "reliability": 0.85}]
    dec_high = decoder.check_abstention(high_rel_docs)
    assert dec_high.should_abstain is False, "High reliability evidence must pass gate"

    dec_low_conf = decoder.check_abstention(high_rel_docs, model_confidence=0.20)
    assert dec_low_conf.should_abstain is True, "Low confidence must trigger abstention"
    assert dec_low_conf.gate_triggered == "confidence_gate"
    print("  Abstention Gate Decisions -> PASSED.")

    # --- Test 3: Contrastive Logits Adjustment ---
    print("\n[Test 3] Testing Contrastive Logits Adjustment & Truncation...")
    ctx_l = torch.tensor([[10.0, 5.0, 1.0, -2.0]])
    prior_l = torch.tensor([[2.0, 6.0, 0.0, -1.0]])
    adj_l = decoder.contrastive_logits_adjustment(ctx_l, prior_l, beta=0.5, alpha_trunc=0.01)
    assert adj_l.shape == ctx_l.shape, "Adjusted logits shape must match"
    # Token 0 is favored in context (10) vs prior (2), contrastive should boost it: 1.5 * 10 - 0.5 * 2 = 14
    assert adj_l[0, 0] > ctx_l[0, 0], f"Token 0 logit should be boosted, got {adj_l[0, 0]}"
    print(f"  Logits adjustment check: raw={ctx_l[0, 0]} -> adjusted={adj_l[0, 0]} -> PASSED.")

    # --- Test 4: SynCheck Grounding Probe ---
    print("\n[Test 4] Testing SynCheck Entity & Lexical Grounding Probe...")
    ctx = [
        {"title": "Alan Turing", "text": "Alan Mathison Turing was born in London in 1912."}
    ]
    # Faithful answer
    eval_pass = decoder.evaluate_grounding("Alan Turing was born in London.", ctx)
    assert eval_pass.is_faithful, f"Grounding should pass: {eval_pass}"
    assert len(eval_pass.unanchored_entities) == 0, f"No unanchored entities expected: {eval_pass.unanchored_entities}"

    # Hallucinated entity answer
    eval_fail = decoder.evaluate_grounding("Albert Einstein was born in Berlin.", ctx)
    assert not eval_fail.is_faithful, f"Grounding should fail for unanchored entities: {eval_fail}"
    assert "Albert Einstein" in eval_fail.unanchored_entities or "Berlin" in eval_fail.unanchored_entities
    print(f"  Grounding pass score={eval_pass.grounding_score}, fail score={eval_fail.grounding_score} -> PASSED.")

    # --- Test 5: Calibration Profiler Metrics (ECE, AUROC, Risk-Coverage) ---
    print("\n[Test 5] Testing ECE, AUROC, and Risk-Coverage Metrics...")
    # Synthetic well-calibrated vs miscalibrated
    confs = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    accs = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    ece, bins = compute_ece(confs, accs, n_bins=5)
    auroc = compute_auroc(confs, accs)
    rc = compute_risk_coverage_curve(confs, accs)

    assert 0.0 <= ece <= 1.0, f"ECE must be in [0, 1]: {ece}"
    assert 0.5 <= auroc <= 1.0, f"AUROC should be > 0.5 for correlated signal: {auroc}"
    assert 0.0 <= rc["aurc"] <= 1.0, f"AURC must be in [0, 1]: {rc['aurc']}"
    print(f"  ECE={ece:.4f}, AUROC={auroc:.4f}, AURC={rc['aurc']:.4f} -> PASSED.")

    # --- Test 6: End-to-End Calibrated Generation Dry-Run ---
    print("\n[Test 6] Testing End-to-End Calibrated Generation Dry-Run...")
    from src.baseline_rag import QwenNF4Generator
    gen = QwenNF4Generator(dry_run=True)
    res = decoder.generate_calibrated(
        generator=gen,
        question="Where was Alan Turing born?",
        evidence_items=ctx
    )
    assert not res.is_abstained, "Should not abstain on valid evidence"
    assert res.answer != "", "Answer must not be empty"
    print(f"  Generated Answer: '{res.answer}' (latency={res.latency_ms:.2f}ms) -> PASSED.")

    print("\n" + "=" * 60)
    print("ALL M3 CALIBRATED DECODER UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_m3_unit_tests()
