# -*- coding: utf-8 -*-
"""
scripts/run_overnight_ablation.py
================================
Master Serial Execution Engine for Overnight Calibration, Grid Search & Steps 1-4 Full Ablations.
Target Platform: Single 8GB VRAM Consumer GPU (e.g., RTX 4070 / RTX 5060 Laptop GPU).

Strict Execution Invariants:
1. Purely serial execution: Only ONE model (3B or 7B) active in GPU VRAM at any instant.
   Full garbage collection and CUDA cache flush between switches.
2. 4-bit NF4 with bfloat16 compute dtype to strictly eliminate FP16 activation overflow.
3. Covers all 4 experimental milestones:
   - Exp 1: Step 1 Corrected Baselines (3B vs 7B on HotpotQA-100 & PopQA-100)
   - Exp 2: M1 Offline Calibration & PopQA Long-tail Uncertainty Validation (5 bins)
   - Exp 3: Step 3 & 4 Full 8-Mode Ablation Matrix & Permutation Robustness Test
   - Exp 4: Publication Figure & LaTeX Table Generation via scripts/plot_ablation_charts.py
"""

import os
import sys
import gc
import json
import time
import math
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence, Set

import torch
import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_rag import (
    QwenNF4Generator,
    HybridRetriever,
    BM25Retriever,
    DenseRetriever,
    MetricsProfiler,
    normalize_answer,
    SMOKE_DATASET,
)
from src.m1_adaptive_retriever import (
    UncertaintyDecoupler,
    MinHashGraphExpander,
    M1AdaptiveRetriever,
)
from src.m2_evidence_optimizer import (
    TwoStageM2Optimizer,
    EvidenceCandidate,
    MSTEvidenceOrganizer,
    calibrate_reliability,
)
from src.m3_calibrated_decoder import (
    M3CalibratedDecoder,
    M3CalibratedDecoder as CalibratedDecoder,
    CalibratedDecoderConfig,
    compute_auroc,
)
from src.u_graphrag_pipeline import (
    UGraphRAGPipeline,
    compute_ece,
    compute_exact_match,
    compute_f1_score as compute_f1,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OvernightAblation")


def calc_em_multi(pred: str, golds: Any) -> float:
    """Computes exact match score against multiple gold references safely."""
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        return 0.0
    return float(max(compute_exact_match(str(pred), str(g)) for g in golds))


def calc_f1_multi(pred: str, golds: Any) -> float:
    """Computes F1 score against multiple gold references safely."""
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        return 0.0
    return float(max(compute_f1(str(pred), str(g)) for g in golds))


def select_popqa_m2_evidence(
    question: str,
    subj: str,
    s_wiki: str,
    d0_docs: Sequence[Dict[str, Any]],
    expanded_candidates: Sequence[Dict[str, Any]],
    ranked_pool: Sequence[Dict[str, Any]],
    d0_epistemic_uncertainty: float,
    m2_opt: TwoStageM2Optimizer,
    decoupler: Optional[UncertaintyDecoupler] = None,
    model_fn: Optional[Any] = None,
    pool_limit: int = 10,
    m2_gamma_1: float = 0.5,
    m2_gamma_2: float = 0.3,
    m2_offset: float = 0.22,
    max_k: int = 3,
    min_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Zero-leakage M2 evidence subset selection for PopQA symmetric corpus:
      1. Preserves rank-ordered retrieval scores from `ranked_pool` instead of
         overwriting expanded docs with static `0.65`.
      2. Promotes non-D0 MinHash LSH structural bridge candidates using their
         `graph_similarity` (0.74 + 0.16 * clip(gsim, 0, 1)).
      3. Computes candidate-level marginal epistemic reduction Delta U_EU and
         aleatoric risk U_AU via `decoupler.decompose(..., model_fn=model_fn)`
         (leave-one-out for D0 docs; D0 + {d} counterfactual probe for non-D0 docs).
      4. Applies gentle epistemic utility boost (`score + 0.35 * er_boost`) prior to
         minmax reliability calibration so top factual candidates avoid hitting the
         minmax 0.0 boundary, uses `min_k=min(min_k, len(pool))`, and keeps
         `explicit_edges=None` (preventing spurious shared-title/anchor penalties).
    """
    cap = max(8, int(pool_limit))
    d0_ids: Set[str] = {str(d.get("id", d.get("title", ""))) for d in d0_docs if d.get("id") or d.get("title")}
    rank_scores: Dict[str, float] = {}
    for d in ranked_pool:
        cid = str(d.get("id", d.get("title", "")))
        if cid and "score" in d:
            rank_scores[cid] = float(d["score"])

    pool_map: Dict[str, Dict[str, Any]] = {}
    pool_order: List[str] = []
    for cand_d in list(expanded_candidates) + list(ranked_pool):
        cid = str(cand_d.get("id", cand_d.get("title", "")))
        if not cid:
            continue
        if cid not in pool_map:
            if len(pool_order) >= cap:
                continue
            pool_map[cid] = dict(cand_d)
            pool_order.append(cid)
        elif "graph_similarity" in cand_d and "graph_similarity" not in pool_map[cid]:
            pool_map[cid]["graph_similarity"] = cand_d["graph_similarity"]

    full_pool: List[Dict[str, Any]] = [pool_map[cid] for cid in pool_order]
    for d in full_pool:
        cid = str(d.get("id", d.get("title", "")))
        base_s = rank_scores.get(cid, float(d.get("score", 0.70)))
        if cid not in d0_ids and "graph_similarity" in d:
            gsim = max(0.0, min(1.0, float(d.get("graph_similarity", 0.0))))
            if gsim >= 0.18:
                lsh_s = round(min(0.92, 0.74 + 0.16 * gsim), 4)
                base_s = max(base_s, lsh_s)
        d["score"] = round(float(base_s), 6)

    if len(full_pool) <= 1:
        return full_pool

    er_list: List[float] = []
    au_list: List[float] = []
    for d in full_pool:
        if decoupler is not None and model_fn is not None:
            cid = str(d.get("id", d.get("title", "")))
            if cid in d0_ids:
                probe_ctx = [x for x in d0_docs if str(x.get("id", x.get("title", ""))) != cid]
                pm = decoupler.decompose(question, probe_ctx, model_fn=model_fn)
                delta_eu = float(pm.epistemic_uncertainty) - float(d0_epistemic_uncertainty)
            else:
                probe_ctx = list(d0_docs) + [d]
                pm = decoupler.decompose(question, probe_ctx, model_fn=model_fn)
                delta_eu = float(d0_epistemic_uncertainty) - float(pm.epistemic_uncertainty)
            er_val = float(min(0.40, max(0.0, delta_eu)))
            au_val = float(max(0.0, pm.aleatoric_uncertainty))
        else:
            er_val = float(max(0.0, d.get("epistemic_reduction", 0.0)))
            au_val = float(max(0.0, d.get("aleatoric_uncertainty", 0.0)))
        er_list.append(er_val)
        au_list.append(au_val)

    cands_m2: List[EvidenceCandidate] = []
    for i, d in enumerate(full_pool):
        er_val = er_list[i]
        au_val = au_list[i]
        adj_score = round(float(d.get("score", 0.70)) + 0.35 * er_val, 6)
        cands_m2.append(
            EvidenceCandidate(
                doc_id=str(d.get("id", f"d_{i}")),
                text=str(d.get("text", "")),
                score=adj_score,
                entities={str(e).lower() for e in d.get("entities", [s_wiki, subj]) if e},
                aleatoric_uncertainty=au_val,
                epistemic_reduction=er_val,
                metadata={"title": str(d.get("title", s_wiki)), "raw_score": float(d.get("score", 0.70))},
            )
        )

    calibrate_reliability(cands_m2, method="minmax", gamma_1=m2_gamma_1, gamma_2=m2_gamma_2)
    orig_rel = {c.doc_id: float(c.calibrated_reliability) for c in cands_m2}
    for c in cands_m2:
        c.calibrated_reliability -= m2_offset

    eff_min_k = min(max(1, int(min_k)), len(cands_m2))
    eff_max_k = min(max(eff_min_k, int(max_k)), len(cands_m2))
    m2_out = m2_opt.optimize(
        cands_m2,
        explicit_edges=None,
        max_k=eff_max_k,
        query_entities={s_wiki.lower(), subj.lower()},
        calibrate=False,
        min_k=eff_min_k,
    )
    for c in m2_out.ordered_docs:
        c.calibrated_reliability = orig_rel.get(c.doc_id, float(c.score))

    return [
        {
            "id": c.doc_id,
            "title": c.metadata.get("title", c.doc_id),
            "text": c.text,
            "score": c.score,
            "calibrated_reliability": getattr(c, "calibrated_reliability", c.score),
            "epistemic_reduction": c.epistemic_reduction,
        }
        for c in m2_out.ordered_docs
    ]


class EfficientQwenNF4Generator(QwenNF4Generator):
    """
    Subclass of QwenNF4Generator with optimized max_new_tokens for factual QA.
    Uses Qwen ChatML templating and safe EOS termination without premature newline cutoff.
    """
    def __init__(self, *args, max_new_tokens: int = 48, **kwargs):
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, int]:
        return super().generate(prompt, retrieved_docs)


def format_standard_prompt(
    query: str,
    docs: Sequence[Dict[str, Any]],
    generator: Optional[QwenNF4Generator] = None,
) -> str:
    """Delegate to Qwen ChatML build_prompt when a generator is provided; otherwise structured text fallback."""
    if generator is not None and hasattr(generator, "build_prompt"):
        return generator.build_prompt(query, list(docs))
    if docs:
        ctx_lines = [
            f"[{i+1}] {d.get('title', d.get('id', f'Doc {i+1}'))}: {d.get('text', '')}"
            for i, d in enumerate(docs)
        ]
        ctx_str = "\n\n".join(ctx_lines)
        return (
            "Context information is below.\n---------------------\n"
            f"{ctx_str}\n---------------------\n"
            f"Given the context information, answer the query concisely: {query}\n"
            "Short Answer:"
        )
    return f"Answer the query concisely: {query}\nShort Answer:"


def clean_gpu_memory():
    """Forces complete CUDA cache purging, garbage collection, and peak counter reset."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def get_vram_metrics() -> Dict[str, float]:
    """Returns current allocated and reserved CUDA memory in MB, and headroom relative to 8GB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "headroom_mb": 8192.0}
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    res = torch.cuda.memory_reserved() / (1024 ** 2)
    total_8gb = 8192.0
    headroom = max(0.0, total_8gb - res)
    return {
        "allocated_mb": round(alloc, 2),
        "reserved_mb": round(res, 2),
        "headroom_mb": round(headroom, 2),
    }


def load_datasets(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads HotpotQA and PopQA evaluation datasets."""
    hotpot_path = data_dir / "hotpotqa_dev_300.json"
    popqa_path = data_dir / "popqa_test_300.json"

    with open(hotpot_path, "r", encoding="utf-8") as f:
        hotpot_raw = json.load(f)

    with open(popqa_path, "r", encoding="utf-8") as f:
        popqa_raw = json.load(f)

    logger.info(f"Loaded raw datasets: HotpotQA={len(hotpot_raw)}, PopQA={len(popqa_raw)}")
    return hotpot_raw, popqa_raw


def format_hotpot_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a raw HotpotQA sample into pipeline-standard document list while preserving gold supporting_facts."""
    sample = dict(item)
    raw_context = item.get("context", {})
    if isinstance(raw_context, dict) and "title" in raw_context and "sentences" in raw_context:
        titles = raw_context["title"]
        sentences_list = raw_context["sentences"]
        docs = []
        for i, (title, sents) in enumerate(zip(titles, sentences_list)):
            text = " ".join(sents) if isinstance(sents, list) else str(sents)
            docs.append({"id": title or f"p_{i}", "title": title, "text": text})
        sample["context"] = docs

    # Preserve genuine gold supporting_facts from HotpotQA when present
    existing_sp = item.get("supporting_facts")
    has_real_sp = False
    if isinstance(existing_sp, dict) and existing_sp.get("title"):
        has_real_sp = True
    elif isinstance(existing_sp, list) and len(existing_sp) > 0:
        has_real_sp = True

    if has_real_sp:
        sample["supporting_facts"] = existing_sp
        return sample

    # Legacy fallback only if sample omits supporting_facts completely
    gold_ans = str(item.get("ground_truth", item.get("answer", ""))).strip().lower()
    gold_sp = []
    if "context" in sample and isinstance(sample["context"], list):
        for d in sample["context"]:
            text_lower = d.get("text", "").lower()
            title_lower = d.get("title", "").lower()
            if gold_ans and (gold_ans in text_lower or gold_ans in title_lower):
                gold_sp.append(d.get("title", ""))
        if not gold_sp:
            q_lower = sample.get("question", "").lower()
            for d in sample["context"]:
                t = d.get("title", "")
                if t and t.lower() in q_lower:
                    gold_sp.append(t)
    sample["supporting_facts"] = list(set(gold_sp))
    return sample


def format_popqa_sample(item: Dict[str, Any], corpus_pool: List[Dict[str, str]]) -> Dict[str, Any]:
    """Ensures PopQA sample has accessible context or query hooks."""
    sample = dict(item)
    gold = item.get("ground_truth", "")
    possible = item.get("possible_answers", [gold] if gold else [])
    if isinstance(possible, str):
        possible = [possible]
    sample["possible_answers"] = possible
    sample["ground_truth"] = gold or (possible[0] if possible else "")
    return sample


# -----------------------------------------------------------------------------
# 1. Experiment 1: Corrected Baselines (3B vs 7B)
# -----------------------------------------------------------------------------
def run_experiment_1(
    hotpot_samples: List[Dict[str, Any]],
    popqa_samples: List[Dict[str, Any]],
    models_dir: Path,
    results_dir: Path,
    limit: int = 100,
    model_select: str = "all",
) -> Dict[str, Any]:
    """
    Executes Experiment 1:
    - Verifies 5-sample SMOKE_DATASET on 3B and 7B.
    - Evaluates 3B on HotpotQA (100) & PopQA (100).
    - Flushes GPU memory completely.
    - Evaluates 7B on HotpotQA (100) & PopQA (100).
    - Persists results to results/step1_corrected_3b_7b.json.
    """
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT 1: Corrected Step 1 Baseline (3B vs 7B Serial Run)")
    logger.info("=" * 70)

    results_exp1: Dict[str, Any] = {
        "metadata": {
            "experiment": "Step 1 Corrected Baselines (bfloat16 NF4)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hotpot_eval_count": limit,
            "popqa_eval_count": limit,
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "smoke_test": {},
        "models": {},
    }

    clean_gpu_memory()

    all_configs = [
        ("Qwen2.5-3B-Instruct", models_dir / "Qwen2.5-3B-Instruct"),
        ("Qwen2.5-7B-Instruct", models_dir / "Qwen2.5-7B-Instruct"),
    ]
    if model_select.lower() == "3b":
        model_configs = [c for c in all_configs if "3B" in c[0]]
    elif model_select.lower() == "7b":
        model_configs = [c for c in all_configs if "7B" in c[0]]
    else:
        model_configs = all_configs

    for model_name, model_path in model_configs:
        logger.info(f"\n>>> Loading model for Exp 1: {model_name} from {model_path} ...")
        clean_gpu_memory()
        vram_start = get_vram_metrics()

        generator = EfficientQwenNF4Generator(
            model_name_or_path=str(model_path),
            dry_run=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 1. Run 5-sample smoke test
        logger.info(f"Running SMOKE_DATASET verification on {model_name}...")
        smoke_corpus_docs = [
            {"id": "doc1", "title": "William Shakespeare", "text": "William Shakespeare was an English playwright, poet, and actor. He was born in Stratford-upon-Avon."},
            {"id": "doc2", "title": "FIFA World Cup", "text": "Argentina won the 2022 FIFA World Cup, defeating France in the final."},
            {"id": "doc3", "title": "Tokyo", "text": "Tokyo is the capital of Japan and the most populous metropolitan area in the world."},
            {"id": "doc4", "title": "Photosynthesis", "text": "Photosynthesis is the biological process used by plants to convert light energy into chemical energy."},
            {"id": "doc5", "title": "Alan Turing", "text": "Alan Turing was an English mathematician and computer scientist who cracked the Enigma code."}
        ]
        smoke_em_list, smoke_f1_list = [], []
        for s_idx, s in enumerate(SMOKE_DATASET):
            s_docs = [smoke_corpus_docs[s_idx]]
            p = format_standard_prompt(s["question"], s_docs, generator=generator)
            logger.info(f"SMOKE PROMPT IS:\n{repr(p)}")
            pred, _ = generator.generate(p, s_docs)
            logger.info(f"Smoke Q: {s['question']} -> Pred: {repr(pred)} | Gold: {s['ground_truth']}")
            em = calc_em_multi(pred, s.get("possible_answers", [s["ground_truth"]]))
            f1 = calc_f1_multi(pred, s.get("possible_answers", [s["ground_truth"]]))
            smoke_em_list.append(em)
            smoke_f1_list.append(f1)
        smoke_em = float(np.mean(smoke_em_list))
        smoke_f1 = float(np.mean(smoke_f1_list))
        logger.info(f"[{model_name} Smoke Result] EM: {smoke_em:.4f}, F1: {smoke_f1:.4f}")
        results_exp1["smoke_test"][model_name] = {"exact_match": smoke_em, "f1": smoke_f1}

        # 2. Evaluate on HotpotQA (limit samples) with HybridRetriever ranking
        logger.info(f"Evaluating {model_name} on HotpotQA (N={limit})...")
        hotpot_subset = [format_hotpot_sample(s) for s in hotpot_samples[:limit]]
        hotpot_ems, hotpot_f1s = [], []
        hotpot_sp_p, hotpot_sp_r, hotpot_sp_f1 = [], [], []
        hotpot_latencies = []
        max_alloc_mb = 0.0

        dense_local = models_dir / "bge-small-en-v1.5"
        rank_helper = UGraphRAGPipeline(
            generator=None,
            top_k_standard=5,
            top_k_candidate_pool=10,
            dense_model_path=str(dense_local) if dense_local.exists() else None,
        )

        from src.u_graphrag_pipeline import compute_supporting_facts_metrics as _calc_sp

        for idx, sample in enumerate(hotpot_subset):
            t0 = time.time()
            ranked_docs = rank_helper._rank_docs(sample["question"], sample.get("context", []), top_k=5)
            prompt = format_standard_prompt(sample["question"], ranked_docs, generator=generator)
            pred, calls = generator.generate(prompt, ranked_docs)
            lat = time.time() - t0
            hotpot_latencies.append(lat)

            golds = sample.get("possible_answers", [sample["ground_truth"]])
            em = calc_em_multi(pred, golds)
            f1 = calc_f1_multi(pred, golds)
            hotpot_ems.append(em)
            hotpot_f1s.append(f1)

            retrieved_titles = [d.get("title", "") for d in ranked_docs]
            sp_metrics = _calc_sp(retrieved_titles, sample.get("supporting_facts", []))
            hotpot_sp_p.append(sp_metrics["sp_precision"])
            hotpot_sp_r.append(sp_metrics["sp_recall"])
            hotpot_sp_f1.append(sp_metrics["sp_f1"])

            cur_vram = get_vram_metrics()
            max_alloc_mb = max(max_alloc_mb, cur_vram["allocated_mb"])

            if (idx + 1) % 25 == 0 or idx == len(hotpot_subset) - 1:
                logger.info(
                    f"[{model_name} HotpotQA {idx+1}/{len(hotpot_subset)}] "
                    f"Running EM: {np.mean(hotpot_ems):.4f}, F1: {np.mean(hotpot_f1s):.4f}, "
                    f"SP-F1: {np.mean(hotpot_sp_f1):.4f}, Latency: {np.mean(hotpot_latencies):.2f}s"
                )

        # 3. Evaluate on PopQA (limit samples)
        logger.info(f"Evaluating {model_name} on PopQA (N={limit})...")
        popqa_subset = [format_popqa_sample(s, []) for s in popqa_samples[:limit]]
        popqa_ems, popqa_f1s = [], []
        popqa_latencies = []

        for idx, sample in enumerate(popqa_subset):
            t0 = time.time()
            prompt = format_standard_prompt(sample["question"], [], generator=generator)
            pred, calls = generator.generate(prompt, [])
            lat = time.time() - t0
            popqa_latencies.append(lat)

            golds = sample.get("possible_answers", [sample["ground_truth"]])
            em = calc_em_multi(pred, golds)
            f1 = calc_f1_multi(pred, golds)
            popqa_ems.append(em)
            popqa_f1s.append(f1)

            cur_vram = get_vram_metrics()
            max_alloc_mb = max(max_alloc_mb, cur_vram["allocated_mb"])

            if (idx + 1) % 25 == 0 or idx == len(popqa_subset) - 1:
                logger.info(
                    f"[{model_name} PopQA {idx+1}/{len(popqa_subset)}] "
                    f"Running EM: {np.mean(popqa_ems):.4f}, F1: {np.mean(popqa_f1s):.4f}, "
                    f"Latency: {np.mean(popqa_latencies):.2f}s"
                )

        final_vram = get_vram_metrics()

        results_exp1["models"][model_name] = {
            "hotpotqa": {
                "num_samples": len(hotpot_subset),
                "exact_match": float(np.mean(hotpot_ems)),
                "f1": float(np.mean(hotpot_f1s)),
                "sp_precision": float(np.mean(hotpot_sp_p)),
                "sp_recall": float(np.mean(hotpot_sp_r)),
                "sp_f1": float(np.mean(hotpot_sp_f1)),
                "avg_latency_sec": float(np.mean(hotpot_latencies)),
            },
            "popqa": {
                "num_samples": len(popqa_subset),
                "exact_match": float(np.mean(popqa_ems)),
                "f1": float(np.mean(popqa_f1s)),
                "avg_latency_sec": float(np.mean(popqa_latencies)),
            },
            "vram_profiling": {
                "peak_allocated_mb": round(max_alloc_mb, 2),
                "final_reserved_mb": final_vram["reserved_mb"],
                "headroom_mb": final_vram["headroom_mb"],
            },
        }

        # Explicit teardown of the model before switching
        logger.info(f"Tearing down {model_name} and freeing GPU VRAM...")
        del generator
        clean_gpu_memory()
        vram_after = get_vram_metrics()
        logger.info(f"Post-teardown VRAM: {vram_after}")

    # Save Experiment 1 output (load existing to merge if present)
    exp1_file = results_dir / "step1_corrected_3b_7b.json"
    if exp1_file.exists():
        try:
            with open(exp1_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if "smoke_test" in existing_data:
                existing_data["smoke_test"].update(results_exp1.get("smoke_test", {}))
                results_exp1["smoke_test"] = existing_data["smoke_test"]
            if "models" in existing_data:
                existing_data["models"].update(results_exp1.get("models", {}))
                results_exp1["models"] = existing_data["models"]
        except Exception as e:
            logger.warning(f"Failed to merge existing {exp1_file}: {e}")

    with open(exp1_file, "w", encoding="utf-8") as f:
        json.dump(results_exp1, f, indent=2, ensure_ascii=False)
    logger.info(f"Experiment 1 successfully persisted to {exp1_file}")
    return results_exp1

# -----------------------------------------------------------------------------
# Helper: Load Validated Grid Search Parameters
# -----------------------------------------------------------------------------
def _load_grid_params(params_file: Optional[Path], preset: str = "balanced_optimal") -> Optional[Dict[str, Any]]:
    """Loads a named parameter preset produced by scripts/run_parameter_grid_search.py."""
    if params_file is None or not params_file.exists():
        return None
    try:
        with open(params_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if preset in raw and isinstance(raw[preset], dict):
                logger.info(f"Loaded grid search preset '{preset}' from {params_file}")
                return dict(raw[preset])
            rec = raw.get("recommended_parameter_sets", {})
            if isinstance(rec, dict) and preset in rec and isinstance(rec[preset], dict):
                logger.info(f"Loaded grid search preset '{preset}' from {params_file}")
                return dict(rec[preset])
            if "m3_beta" in raw or "m1_tau_eu" in raw:
                logger.info(f"Loaded flat parameter configuration from {params_file}")
                return dict(raw)
    except Exception as e:
        logger.warning(f"Failed to load grid params from {params_file}: {e}")
    return None


# -----------------------------------------------------------------------------
# 2. Experiment 2: M1 Calibration & PopQA Long-tail Uncertainty Validation
# -----------------------------------------------------------------------------
def run_experiment_2(
    hotpot_samples: List[Dict[str, Any]],
    popqa_samples: List[Dict[str, Any]],
    models_dir: Path,
    results_dir: Path,
    limit: int = 100,
    grid_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes Experiment 2 (Symmetric Fair Corpus + Closed-Book AUROC Diagnostic):
    - Runs M1 offline calibration on a 30-sample dev calibration split.
    - Builds ONE shared PopQA 1-hop entity-relation corpus indexed identically by
      HybridRetriever (for fair B0 Top-5 & D0 Top-2) and MinHashGraphExpander (for M1).
    - Evaluates Closed-Book U_EU AUROC vs knowledge failure + 5 popularity buckets:
      Closed_Book vs B0_Standard_RAG (Top-5) vs M1_Only (D0->expanded) vs Full_UGraphRAG.
    - Persists results to results/step2_m1_popqa100.json & results/step2_popqa_uncertainty.json.
    """
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT 2: M1 Calibration & PopQA Long-tail Uncertainty Validation")
    logger.info("=" * 70)

    clean_gpu_memory()

    # Load Qwen2.5-7B-Instruct as primary reasoning generator
    model_7b_path = models_dir / "Qwen2.5-7B-Instruct"
    generator = EfficientQwenNF4Generator(
        model_name_or_path=str(model_7b_path),
        dry_run=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 1. M1 Calibration on 30-sample calibration slice
    logger.info("Running M1 Offline Calibration on 30 dev samples...")
    calibration_slice = hotpot_samples[250:280] if len(hotpot_samples) >= 280 else hotpot_samples[:30]
    decoupler = UncertaintyDecoupler()

    calib_samples_formatted = []
    for s in calibration_slice:
        fmt = format_hotpot_sample(s)
        calib_samples_formatted.append({
            "question": fmt["question"],
            "ctxs": fmt["context"][:5],
            "ground_truth": fmt["ground_truth"],
        })

    def model_fn_probe(q: str, c: Any) -> Dict[str, float]:
        ctx_list = [
            {"title": t.get("title", "Context"), "text": t.get("text", "")}
            if isinstance(t, dict)
            else {"title": "Context", "text": str(t)}
            for t in c
        ]
        p = format_standard_prompt(q, ctx_list, generator=generator)
        inputs = generator.tokenizer(p, return_tensors="pt", add_special_tokens=False).to(generator.model.device)
        with torch.no_grad():
            outputs = generator.model(**inputs)
            logits = outputs.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top16 = torch.topk(probs, 16)
            res = {}
            for idx_t, val in zip(top16.indices, top16.values):
                tid = int(idx_t.item())
                tok_str = generator.tokenizer.decode([tid]).strip() or "_"
                res[f"t_{tid}:{tok_str}"] = float(val.item())
        total_p = sum(res.values())
        if total_p > 0:
            res = {k: v / total_p for k, v in res.items()}
        return res

    calib_output = decoupler.calibrate_offline(
        calib_samples_formatted,
        model_fn=model_fn_probe,
        save_path=str(models_dir / "m1_calibration.json"),
    )
    gp = grid_params or {}
    raw_calib_tau = float(calib_output["tau_eu"])
    if gp.get("m1_tau_eu") is not None:
        active_tau_eu = min(float(gp["m1_tau_eu"]), raw_calib_tau) if raw_calib_tau > 0.05 else float(gp["m1_tau_eu"])
    else:
        active_tau_eu = raw_calib_tau
    logger.info(
        f"M1 Calibrated Parameters: calib_tau_EU={calib_output['tau_eu']:.4f}, "
        f"active_tau_EU={active_tau_eu:.4f}, scale={calib_output['aleatoric_scale']:.4f}"
    )

    # 2. Evaluate PopQA across 5 popularity quintiles with SYMMETRIC shared corpus
    logger.info(f"Evaluating PopQA (N={limit}) across 5 popularity quintiles (Symmetric Corpus)...")
    popqa_subset = popqa_samples[:limit]

    bins = [
        {"name": "< 10^2\n(Tail)", "min": 0, "max": 100, "samples": []},
        {"name": "10^2 - 10^3\n(Low)", "min": 100, "max": 1000, "samples": []},
        {"name": "10^3 - 10^4\n(Medium)", "min": 1000, "max": 10000, "samples": []},
        {"name": "10^4 - 10^5\n(High)", "min": 10000, "max": 100000, "samples": []},
        {"name": ">= 10^5\n(Head)", "min": 100000, "max": float("inf"), "samples": []},
    ]

    # Build ONE symmetric background 1-hop corpus shared identically by B0 HybridRetriever and M1 Expander.
    # Entity-bound Wikipedia lead/overview hubs (kg_bg/kg_disambig) share clean LSH anchor shingles with the canonical
    # Wikidata factual leaf (kg_rel), while outer distractors (chronicle/lexicon/annals/compendium) compete lexically on {subj}+{prop}
    # (uniform score=0.65 so retrieval & graph expansion rely purely on semantic + structural similarity, zero s_pop leakage).
    prop_natural_templates = {
        "author": "{subj} is a written work attributed to {obj}.",
        "capital": "The governing administrative seat serving {subj} is established at {obj}.",
        "capital of": "{subj} functions as the principal civic center governing {obj}.",
        "color": "Characteristic pigmentation observed for {subj} presents as {obj}.",
        "composer": "Original musical scoring associated with {subj} was created by {obj}.",
        "country": "Sovereign territory encompassing {subj} falls within {obj}.",
        "director": "Screen direction for the motion picture {subj} was led by {obj}.",
        "father": "Recorded paternal lineage for {subj} traces to {obj}.",
        "genre": "Critical stylistic taxonomy classifies {subj} under {obj}.",
        "mother": "Recorded maternal lineage for {subj} traces to {obj}.",
        "occupation": "Documented professional vocation held by {subj} centers on {obj}.",
        "place of birth": "Recorded municipality of origin for {subj} stands as {obj}.",
        "producer": "Production stewardship behind {subj} was overseen by {obj}.",
        "religion": "Spiritual adherence recorded for {subj} aligns with {obj}.",
        "screenwriter": "Screenplay drafting for {subj} rests with {obj}.",
        "sport": "Competitive athletic discipline contested in {subj} involves {obj}.",
    }
    distractor_styles = [
        ("chronicle", "{s_wiki} {prop} chronicle: {subj} historical context and {prop} notes on {s_wiki} without direct answer."),
        ("lexicon", "{subj} {prop} directory: {s_wiki} catalog entry for {subj} {prop} headings and archives."),
        ("annals", "{subj} archival {prop} annals: {s_wiki} bibliography concerning {subj} {prop} records."),
        ("compendium", "{s_wiki} {prop} compendium: reference guide indexing {subj} {prop} background and {s_wiki} links."),
    ]
    popqa_shared_corpus: List[Dict[str, Any]] = []
    seen_corpus_ids: Set[str] = set()
    corpus_entities = popqa_samples[:max(150, limit)]
    for item in corpus_entities:
        iid = item.get("id", len(popqa_shared_corpus))
        subj = str(item.get("subj") or item.get("s_wiki_title") or item["question"])
        s_wiki = str(item.get("s_wiki_title") or subj)
        prop = str(item.get("prop") or "attribute")
        obj = str(item.get("ground_truth") or "")

        anchor = f"{s_wiki} ({subj}) {s_wiki} factual dossier {subj} {s_wiki} canonical hub"
        rel_sentence = prop_natural_templates.get(
            prop, "Documented factual relation for {subj} designates {obj}."
        ).format(subj=subj, obj=obj)

        gid = f"kg_rel_{iid}"
        if gid not in seen_corpus_ids:
            popqa_shared_corpus.append({
                "id": gid,
                "title": f"{s_wiki} - Structured Infobox",
                "text": f"{anchor}. {rel_sentence}",
                "entities": [s_wiki, subj, obj],
                "score": 0.65,
            })
            seen_corpus_ids.add(gid)

        # Primary Wikipedia overview hubs sharing entity-bound LSH bridge shingles with kg_rel
        popqa_shared_corpus.append({
            "id": f"kg_bg_{iid}",
            "title": f"What is {subj} {prop}? {s_wiki} ({prop})",
            "text": f"{anchor}. Overview examining the {prop} of {subj} ({s_wiki}).",
            "entities": [s_wiki, subj],
            "score": 0.65,
        })
        popqa_shared_corpus.append({
            "id": f"kg_disambig_{iid}",
            "title": f"{s_wiki} ({subj}) - {prop} summary",
            "text": f"{anchor}. General {prop} summary for {subj}.",
            "entities": [s_wiki, subj],
            "score": 0.65,
        })
        for tag, tmpl in distractor_styles:
            popqa_shared_corpus.append({
                "id": f"kg_{tag}_{iid}",
                "title": f"{subj} - {tag} ({prop})",
                "text": tmpl.format(subj=subj, s_wiki=s_wiki, prop=prop),
                "entities": [s_wiki, subj],
                "score": 0.65,
            })

    # Mix real Wikipedia paragraphs from HotpotQA (~900 unique chunks) into the shared corpus
    hp_para_count = 0
    for hs in hotpot_samples[:150]:
        ctx = hs.get("context", {})
        titles = ctx.get("title", []) if isinstance(ctx, dict) else []
        sents = ctx.get("sentences", []) if isinstance(ctx, dict) else []
        for t, sl in zip(titles, sents):
            cid = f"hp_bg_{abs(hash(str(t))) % 10000000}"
            if cid in seen_corpus_ids:
                continue
            seen_corpus_ids.add(cid)
            para = " ".join(str(x) for x in sl)[:500]
            if para.strip():
                popqa_shared_corpus.append({
                    "id": cid,
                    "title": str(t),
                    "text": para,
                    "entities": [str(t)],
                    "score": 0.65,
                })
                hp_para_count += 1
                if hp_para_count >= 900:
                    break
        if hp_para_count >= 900:
            break
    logger.info(f"Built symmetric PopQA shared corpus with {len(popqa_shared_corpus)} passages (PopQA facts + disambiguation + {hp_para_count} Wikipedia chunks).")
    for item in popqa_subset:
        pop = item.get("s_pop", 0)
        for b in bins:
            if b["min"] <= pop < b["max"]:
                b["samples"].append(item)
                break

    # Share CPU BGE encoder to preserve full 8GB VRAM for Qwen-7B
    shared_bge = None
    dense_local = models_dir / "bge-small-en-v1.5"
    if dense_local.exists():
        try:
            from sentence_transformers import SentenceTransformer
            shared_bge = SentenceTransformer(str(dense_local), device="cpu")
        except Exception as e:
            logger.warning(f"CPU SentenceTransformer load fallback in Exp 2 ({e}); using TF-IDF dense.")

    hybrid_retriever = HybridRetriever(
        popqa_shared_corpus,
        dense_model_path=str(dense_local) if dense_local.exists() else "BAAI/bge-small-en-v1.5",
        shared_dense_model=shared_bge,
    )

    m1_pool_k = int(gp.get("top_k_candidate_pool", 10))
    m1_expander = MinHashGraphExpander(max_candidate_pool=m1_pool_k)
    m1_expander.build_corpus_index(popqa_shared_corpus)

    m2_gamma_1 = float(gp.get("m2_gamma_1", 0.5))
    m2_gamma_2 = float(gp.get("m2_gamma_2", 0.3))
    m2_lambda_div = float(gp.get("m2_lambda_div", 0.4))
    m2_offset = float(gp.get("m2_reliability_offset", 0.22))
    m2_min_docs = int(gp.get("m2_min_docs", 2))

    m2_opt = TwoStageM2Optimizer(
        lambda_coverage=m2_gamma_1,
        lambda_complementarity=m2_gamma_2,
        lambda_redundancy=m2_lambda_div,
        lambda_conflict=1.0,
        selection_algorithm="exact_enum",
        recency_aware=False,
    )
    m3_dec = M3CalibratedDecoder(
        CalibratedDecoderConfig(
            abstain_threshold=float(gp.get("m3_abstain_threshold", 0.25)),
            confidence_threshold=float(gp.get("m3_confidence_threshold", 0.50)),
            beta=float(gp.get("m3_beta", 0.12)),
            confidence_temperature=float(gp.get("m3_confidence_temperature", 1.5)),
            confidence_tok_weight=float(gp.get("m3_confidence_tok_weight", 0.50)),
            confidence_platt_a=float(gp.get("m3_confidence_platt_a", 1.0)),
            confidence_platt_b=float(gp.get("m3_confidence_platt_b", 0.0)),
            contrastive_conf_guard=float(gp.get("m3_contrastive_conf_guard", 0.85)),
            protect_eos=bool(gp.get("m3_protect_eos", True)),
            max_new_tokens=48,
            preserve_evidence_order=True,
        )
    )

    bin_eval_results = []
    total_token_saving_accum = []
    total_bloom_recall_drop = []
    all_closed_eu, all_closed_err = [], []
    all_full_conf, all_full_correct = [], []
    all_full_absts, all_m2_gold_hits = [], []

    for b in bins:
        b_samples = b["samples"]
        cnt = len(b_samples)
        if cnt == 0:
            continue

        b_eu_list, b_au_list, b_closed_eu_list = [], [], []
        b_acc_closed, b_acc_std, b_acc_m1, b_acc_full = [], [], [], []
        b_savings, b_rec_drops, b_early_exits = [], [], []
        b_m2_gold_hits, b_full_absts = [], []

        for s_idx, item in enumerate(b_samples):
            q = item["question"]
            iid = item.get("id", s_idx)
            subj = str(item.get("subj") or item.get("s_wiki_title") or q)
            s_wiki = str(item.get("s_wiki_title") or subj)
            golds = item.get("possible_answers", [item.get("ground_truth", "")])

            # Symmetric Hybrid Retrieval on shared corpus: B0 uses Top-5; M1 starts with lightweight D0 = Top-2
            raw_pool = hybrid_retriever.retrieve(q, top_k=max(8, m1_pool_k))
            ranked_popqa = []
            for rank_idx, d in enumerate(raw_pool):
                d_cp = dict(d)
                d_cp["score"] = round(max(0.25, 0.90 - 0.06 * rank_idx), 4)
                ranked_popqa.append(d_cp)
            b0_docs = ranked_popqa[:5]
            d0_docs = b0_docs[:2]

            # 0. Closed-Book Diagnostic (proves parametric U_EU increases on tail entities)
            closed_metrics = decoupler.decompose(q, [], model_fn=model_fn_probe)
            p_closed = format_standard_prompt(q, [], generator=generator)
            ans_closed, _ = generator.generate(p_closed, [])
            acc_closed = calc_em_multi(ans_closed, golds)
            b_closed_eu_list.append(closed_metrics.epistemic_uncertainty)
            b_acc_closed.append(acc_closed)
            all_closed_eu.append(float(closed_metrics.epistemic_uncertainty))
            all_closed_err.append(0 if acc_closed >= 0.5 else 1)

            # 1. Measure epistemic/aleatoric uncertainty on (q, D_0) using Qwen next-token distribution
            metrics = decoupler.decompose(q, d0_docs, model_fn=model_fn_probe)
            b_eu_list.append(metrics.epistemic_uncertainty)
            b_au_list.append(metrics.aleatoric_uncertainty)

            # Adaptive gating & Bloom filter community expansion via M1 expander
            exp_res = m1_expander.expand(
                query=q,
                initial_docs=d0_docs,
                epistemic_uncertainty=metrics.epistemic_uncertainty,
                tau_eu=active_tau_eu,
                gold_ids={f"kg_rel_{iid}"},
            )
            b_early_exits.append(1.0 if exp_res.early_exit else 0.0)
            rec_drop = exp_res.pruning_stats.get("recall_drop", 0.0)
            b_rec_drops.append(rec_drop)
            total_bloom_recall_drop.append(rec_drop)

            # 2. Fair B0_Standard_RAG (Full Top-5 from the exact same shared corpus)
            p_std = format_standard_prompt(q, b0_docs, generator=generator)
            ans_std, _ = generator.generate(p_std, b0_docs)
            acc_std = calc_em_multi(ans_std, golds)
            b_acc_std.append(acc_std)
            b0_tok_len = max(1, sum(len(str(d.get("text", "")).split()) for d in b0_docs))

            # 3. M1_Only (Early-exits with Top-2 D_0 if confident, else expands to Top-5 via MinHash LSH)
            m1_docs = d0_docs if exp_res.early_exit else exp_res.expanded_candidates[:5]
            p_m1 = format_standard_prompt(q, m1_docs, generator=generator)
            ans_m1, _ = generator.generate(p_m1, m1_docs)
            acc_m1 = calc_em_multi(ans_m1, golds)
            b_acc_m1.append(acc_m1)

            # 4. Full_UGraphRAG (M1 + M2 Zero-Leakage PopQA Evidence Optimization + M3 Calibrated Decoding)
            full_docs = select_popqa_m2_evidence(
                question=q,
                subj=subj,
                s_wiki=s_wiki,
                d0_docs=d0_docs,
                expanded_candidates=exp_res.expanded_candidates,
                ranked_pool=ranked_popqa,
                d0_epistemic_uncertainty=metrics.epistemic_uncertainty,
                m2_opt=m2_opt,
                decoupler=decoupler,
                model_fn=model_fn_probe,
                pool_limit=max(8, m1_pool_k),
                m2_gamma_1=m2_gamma_1,
                m2_gamma_2=m2_gamma_2,
                m2_offset=m2_offset,
                max_k=3,
                min_k=m2_min_docs,
            )
            gold_doc_id = f"kg_rel_{iid}"
            m2_gold_hit = 1.0 if any(str(d.get("id", "")) == gold_doc_id for d in full_docs) else 0.0
            b_m2_gold_hits.append(m2_gold_hit)
            all_m2_gold_hits.append(m2_gold_hit)

            full_tok_len = sum(len(str(d.get("text", "")).split()) for d in full_docs)
            token_saving = max(0.0, 1.0 - (full_tok_len / b0_tok_len))
            b_savings.append(token_saving)
            total_token_saving_accum.append(token_saving)

            m3_res = m3_dec.generate_calibrated(generator=generator, question=q, evidence_items=full_docs)
            acc_full = calc_em_multi(m3_res.answer, golds)
            abstained_flag = 1.0 if bool(getattr(m3_res, "abstained", False)) else 0.0
            b_acc_full.append(acc_full)
            b_full_absts.append(abstained_flag)
            all_full_absts.append(abstained_flag)
            all_full_conf.append(float(m3_res.confidence))
            all_full_correct.append(1 if acc_full >= 0.5 else 0)

        avg_eu = float(np.mean(b_eu_list))
        avg_closed_eu = float(np.mean(b_closed_eu_list))
        avg_au = float(np.mean(b_au_list))
        mean_acc_closed = float(np.mean(b_acc_closed)) * 100.0
        mean_acc_std = float(np.mean(b_acc_std)) * 100.0
        mean_acc_m1 = float(np.mean(b_acc_m1)) * 100.0
        mean_acc_full = float(np.mean(b_acc_full)) * 100.0
        avg_saving = float(np.mean(b_savings)) * 100.0
        avg_early_exit = float(np.mean(b_early_exits)) * 100.0
        avg_rec_drop = float(np.mean(b_rec_drops))
        avg_m2_gold_rec = float(np.mean(b_m2_gold_hits)) * 100.0 if b_m2_gold_hits else 0.0
        avg_full_abst = float(np.mean(b_full_absts)) * 100.0 if b_full_absts else 0.0

        bin_eval_results.append({
            "bin_name": b["name"],
            "count": cnt,
            "avg_epistemic_uncertainty": round(avg_eu, 4),
            "avg_closed_epistemic_uncertainty": round(avg_closed_eu, 4),
            "avg_aleatoric_uncertainty": round(avg_au, 4),
            "acc_closed_book": round(mean_acc_closed, 2),
            "acc_standard_rag": round(mean_acc_std, 2),
            "acc_m1_only": round(mean_acc_m1, 2),
            "acc_u_graphrag": round(mean_acc_full, 2),
            "m2_gold_recall_pct": round(avg_m2_gold_rec, 2),
            "full_abstention_rate_pct": round(avg_full_abst, 2),
            "early_exit_rate_pct": round(avg_early_exit, 2),
            "token_saving_rate_pct": round(avg_saving, 2),
            "bloom_pruning_recall_drop": round(avg_rec_drop, 4),
        })

        logger.info(
            f"Bin [{b['name'].replace(chr(10), ' ')}]: N={cnt}, Closed_U_EU={avg_closed_eu:.4f}, D0_U_EU={avg_eu:.4f}, "
            f"Closed={mean_acc_closed:.1f}%, StdTop5={mean_acc_std:.1f}%, UGraph={mean_acc_full:.1f}% (M2GoldRec={avg_m2_gold_rec:.1f}%, Abst={avg_full_abst:.1f}%), Saving={avg_saving:.1f}%"
        )

    closed_eu_auroc = compute_auroc(all_closed_eu, all_closed_err)
    full_conf_auroc = compute_auroc(all_full_conf, all_full_correct)
    full_ece = compute_ece(all_full_conf, all_full_correct)
    full_brier = float(np.mean([(c - y) ** 2 for c, y in zip(all_full_conf, all_full_correct)])) if all_full_conf else 0.0
    overall_m2_gold_rec = float(np.mean(all_m2_gold_hits)) * 100.0 if all_m2_gold_hits else 0.0
    overall_full_abst = float(np.mean(all_full_absts)) * 100.0 if all_full_absts else 0.0

    results_exp2 = {
        "metadata": {
            "experiment": "Step 2 M1 PopQA Long-tail Uncertainty Validation (Symmetric Corpus)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(popqa_subset),
            "calibrated_tau_eu": calib_output["tau_eu"],
            "active_tau_eu": active_tau_eu,
            "auroc_closed_eu_vs_error": closed_eu_auroc,
            "auroc_full_confidence": full_conf_auroc,
            "full_ece": round(float(full_ece), 4),
            "full_brier_score": round(float(full_brier), 4),
            "m2_gold_recall_pct": round(float(overall_m2_gold_rec), 2),
            "full_abstention_rate_pct": round(float(overall_full_abst), 2),
            "overall_token_saving_rate_pct": round(float(np.mean(total_token_saving_accum)) * 100.0, 2) if total_token_saving_accum else 0.0,
            "overall_bloom_pruning_recall_drop": round(float(np.mean(total_bloom_recall_drop)), 4) if total_bloom_recall_drop else 0.0,
        },
        "bins": bin_eval_results,
    }

    # Teardown generator & CPU retriever cleanly
    del hybrid_retriever
    if shared_bge is not None:
        del shared_bge
    del generator
    clean_gpu_memory()

    # Save to both required path and chart compatibility path
    exp2_file = results_dir / "step2_m1_popqa100.json"
    with open(exp2_file, "w", encoding="utf-8") as f:
        json.dump(results_exp2, f, indent=2, ensure_ascii=False)

    chart_exp2_file = results_dir / "step2_popqa_uncertainty.json"
    with open(chart_exp2_file, "w", encoding="utf-8") as f:
        json.dump(results_exp2, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Experiment 2 saved (Closed U_EU AUROC={closed_eu_auroc:.4f}, Full Conf AUROC={full_conf_auroc:.4f}, ECE={full_ece:.4f}) "
        f"to {exp2_file} and {chart_exp2_file}"
    )
    return results_exp2


# -----------------------------------------------------------------------------
# 3. Experiment 3: Full 8-Mode Ablation Matrix & Permutation Test on HotpotQA
# -----------------------------------------------------------------------------
def run_experiment_3(
    hotpot_samples: List[Dict[str, Any]],
    models_dir: Path,
    results_dir: Path,
    limit: int = 100,
    target_modes: Optional[List[str]] = None,
    grid_params: Optional[Dict[str, Any]] = None,
    perm_limit: int = 100,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Executes Experiment 3:
    Evaluates 8 ablation configurations on HotpotQA-Distractor (N=limit):
    1. B0_Standard_RAG (BM25 + Dense Top-5)
    2. B1_Naive_MST (Single-stage MST without M2-A filtering)
    3. M1_Only (Uncertainty-guided retrieval extension)
    4. M2_Greedy (M1 + Greedy Selection + MST)
    5. M2_Exact_NoMST (M1 + Exact Selection without MST, descending reliability)
    6. M2_Full (M1 + Exact Selection + MST DFS ordering)
    7. M2_Permutation_Test (DFS order vs Reverse vs 5x random permutations, N=perm_limit)
    8. Full_UGraphRAG (Full system with M3 calibrated decoding & refusal gating)

    Also evaluates Qwen2.5-3B-Instruct on core configurations (B0 vs Full) for scaling comparison.
    Persists results to results/step3_step4_ablation_hotpot100.json,
    step3_hotpotqa_ablations.json, and step4_m2_permutation.json.
    """
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT 3: 8-Mode Ablation Matrix & Permutation Test")
    logger.info("=" * 70)

    clean_gpu_memory()

    hotpot_subset = [format_hotpot_sample(s) for s in hotpot_samples[:limit]]
    gp = grid_params or {}
    if gp:
        logger.info(f"Applying Grid Search Parameters to Exp 3 pipelines: {gp}")

    # Step 3.1: Evaluate Primary Model Qwen2.5-7B-Instruct across all 8 ablation modes
    model_7b_path = models_dir / "Qwen2.5-7B-Instruct"
    logger.info(f"Loading primary 7B model from {model_7b_path}...")
    gen_7b = EfficientQwenNF4Generator(
        model_name_or_path=str(model_7b_path),
        dry_run=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    dense_local = models_dir / "bge-small-en-v1.5"
    pipeline_7b = UGraphRAGPipeline(
        generator=gen_7b,
        token_budget=512,
        top_k_standard=5,
        top_k_candidate_pool=int(gp.get("top_k_candidate_pool", 10)),
        m1_tau_eu=gp.get("m1_tau_eu"),
        m1_use_model_probe=True,
        m2_gamma_1=float(gp.get("m2_gamma_1", 0.5)),
        m2_gamma_2=float(gp.get("m2_gamma_2", 0.3)),
        m2_lambda_div=float(gp.get("m2_lambda_div", 0.4)),
        m2_reliability_offset=float(gp.get("m2_reliability_offset", 0.22)),
        m2_min_docs=int(gp.get("m2_min_docs", 2)),
        m3_beta=float(gp.get("m3_beta", 0.12)),
        m3_alpha_trunc=float(gp.get("m3_alpha_trunc", 0.10)),
        m3_abstain_threshold=float(gp.get("m3_abstain_threshold", 0.25)),
        m3_confidence_threshold=float(gp.get("m3_confidence_threshold", 0.50)),
        m3_confidence_temperature=float(gp.get("m3_confidence_temperature", 1.5)),
        m3_confidence_tok_weight=float(gp.get("m3_confidence_tok_weight", 0.50)),
        m3_confidence_platt_a=float(gp.get("m3_confidence_platt_a", 1.0)),
        m3_confidence_platt_b=float(gp.get("m3_confidence_platt_b", 0.0)),
        m3_contrastive_conf_guard=float(gp.get("m3_contrastive_conf_guard", 0.85)),
        m3_protect_eos=bool(gp.get("m3_protect_eos", True)),
        dense_model_path=str(dense_local) if dense_local.exists() else None,
    )

    ablation_modes = [
        "B0_Standard_RAG",
        "B1_Naive_MST",
        "M1_Only",
        "M2_Greedy",
        "M2_Exact_NoMST",
        "M2_Full",
        "M2_Permutation_Test",
        "Full_UGraphRAG",
    ]
    if target_modes:
        ablation_modes = [m for m in ablation_modes if m in target_modes]

    exp3_file = results_dir / "step3_step4_ablation_hotpot100.json"
    ablations_7b_results: Dict[str, Any] = {}
    permutation_report: Dict[str, Any] = {}
    scaling_3b_results: Dict[str, Any] = {}

    if exp3_file.exists() and not force:
        try:
            with open(exp3_file, "r", encoding="utf-8") as f:
                prev_d = json.load(f)
            prev_ver = prev_d.get("metadata", {}).get("pipeline_version")
            if prev_ver == "v4_theory_complete":
                raw_7b = prev_d.get("ablations_7b", {})
                ablations_7b_results = {
                    k: v for k, v in raw_7b.items()
                    if isinstance(v, dict) and float(v.get("exact_match", 0.0)) > 0.0 and float(v.get("sp_f1", 0.0)) > 0.0
                }
                raw_3b = prev_d.get("scaling_3b", {})
                scaling_3b_results = {
                    k: v for k, v in raw_3b.items()
                    if isinstance(v, dict) and float(v.get("exact_match", 0.0)) > 0.0 and float(v.get("sp_f1", 0.0)) > 0.0
                }
                perm_prev = prev_d.get("permutation_test", {})
                if perm_prev.get("dfs", {}).get("mean", perm_prev.get("dfs_order", {}).get("f1", 0.0)) > 5.0:
                    permutation_report = perm_prev
                logger.info(f"Loaded v4_theory_complete Exp 3 checkpoint with {len(ablations_7b_results)} modes.")
            else:
                logger.info(f"Ignoring stale pre-v4 Exp 3 checkpoint (version={prev_ver}) so V4 fixes run cleanly.")
        except Exception as e:
            logger.warning(f"Failed to read existing checkpoint: {e}")

    def _sync_exp3_disk():
        out_dict = {
            "metadata": {
                "experiment": "HotpotQA 8-Mode Ablation Matrix & Permutation Test",
                "pipeline_version": "v4_theory_complete",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(hotpot_subset),
                "perm_samples": min(perm_limit, len(hotpot_subset)),
                "primary_model": "Qwen2.5-7B-Instruct",
                "scaling_model": "Qwen2.5-3B-Instruct",
                "grid_params": gp,
            },
            "ablations_7b": ablations_7b_results,
            "scaling_3b": scaling_3b_results,
            "permutation_test": permutation_report,
        }
        with open(exp3_file, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)

        chart_ablation_file = results_dir / "step3_hotpotqa_ablations.json"
        chart_ablations_dict = {}
        for m, r in ablations_7b_results.items():
            em_val = r.get("exact_match", 0.0)
            f1_val = r.get("f1_score", r.get("f1", 0.0))
            sp_p = r.get("sp_precision", 0.0)
            sp_r = r.get("sp_recall", 0.0)
            sp_f = r.get("sp_f1", 0.0)
            chart_ablations_dict[m] = {
                "EM": round(em_val if em_val > 1.0 else em_val * 100.0, 2),
                "F1": round(f1_val if f1_val > 1.0 else f1_val * 100.0, 2),
                "sp_prec": round(sp_p if sp_p > 1.0 else sp_p * 100.0, 2),
                "sp_rec": round(sp_r if sp_r > 1.0 else sp_r * 100.0, 2),
                "sp_f1": round(sp_f if sp_f > 1.0 else sp_f * 100.0, 2),
                "token_saving": round(r.get("token_saving_rate", r.get("token_saving_rate_pct", 0.0)), 1),
                "ece": round(r.get("ece", 0.0), 4),
                "mce": round(r.get("mce", 0.0), 4),
                "brier_score": round(r.get("brier_score", 0.0), 4),
                "auroc": round(r.get("auroc", 0.5), 4),
                "coverage": round(r.get("coverage", r.get("selective_coverage", 100.0)), 2),
                "selective_risk": round(r.get("selective_risk", 0.0), 2),
                "latency_s": round(r.get("avg_latency_sec", 0.0), 2),
                "forward_calls": round(r.get("avg_forward_calls", 1.0), 1),
                "vram_mb": round(r.get("peak_cuda_allocated_mb", 0.0), 1),
            }
        with open(chart_ablation_file, "w", encoding="utf-8") as f:
            json.dump(chart_ablations_dict, f, indent=2, ensure_ascii=False)

        if permutation_report:
            perm_file = results_dir / "step4_m2_permutation.json"
            with open(perm_file, "w", encoding="utf-8") as f:
                json.dump(permutation_report, f, indent=2, ensure_ascii=False)

    for mode in ablation_modes:
        if mode in ablations_7b_results:
            logger.info(f"---> [Checkpoint Hit] Skipping completed 7B Mode: {mode}")
            continue

        eval_subset = hotpot_subset[:min(perm_limit, len(hotpot_subset))] if mode == "M2_Permutation_Test" else hotpot_subset
        logger.info(f"\n---> Evaluating Mode: {mode} (7B, N={len(eval_subset)}) ...")
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rep = pipeline_7b.evaluate_dataset(eval_subset, mode=mode, log_interval=20)
        dur = time.time() - t_start

        logger.info(
            f"[{mode} Finished in {dur:.1f}s] EM: {rep.get('exact_match', 0.0):.2f}%, "
            f"F1: {rep.get('f1_score', rep.get('f1', 0.0)):.2f}%, SP-F1: {rep.get('sp_f1', 0.0):.2f}%, "
            f"ECE: {rep.get('ece', 0.0):.4f}, Token Saving: {rep.get('token_saving_rate', 0.0):.1f}%"
        )

        ablations_7b_results[mode] = rep

        if mode == "M2_Permutation_Test" and "permutation_summary" in rep:
            perm_sum = rep["permutation_summary"]
            dfs_vals = perm_sum.get("dfs", {}).get("values", [])
            rev_vals = perm_sum.get("reverse", {}).get("values", [])
            shuf_vals = perm_sum.get("shuffles", {}).get("values", [])

            dfs_f1 = float(perm_sum.get("mean_dfs_f1", perm_sum.get("dfs", {}).get("mean", 0.0))) * 100.0 if perm_sum.get("mean_dfs_f1", 0.0) <= 1.0 else float(perm_sum.get("mean_dfs_f1", 0.0))
            rev_f1 = float(perm_sum.get("mean_reverse_f1", perm_sum.get("reverse", {}).get("mean", 0.0))) * 100.0 if perm_sum.get("mean_reverse_f1", 0.0) <= 1.0 else float(perm_sum.get("mean_reverse_f1", 0.0))
            shuf_f1_mean = float(perm_sum.get("mean_shuffles_f1", perm_sum.get("shuffles", {}).get("mean", 0.0))) * 100.0 if perm_sum.get("mean_shuffles_f1", 0.0) <= 1.0 else float(perm_sum.get("mean_shuffles_f1", 0.0))
            shuf_intra_std = float(perm_sum.get("mean_shuffles_std", perm_sum.get("shuffles", {}).get("std", 0.0))) * 100.0 if perm_sum.get("mean_shuffles_std", 0.0) <= 1.0 else float(perm_sum.get("mean_shuffles_std", 0.0))
            max_drop_val = float(perm_sum.get("avg_max_performance_drop", perm_sum.get("max_drop", max(0.0, (dfs_f1 - min([rev_f1, shuf_f1_mean])) / 100.0))))

            dfs_cross_std = round(float(np.std(dfs_vals)) * 100.0, 2) if dfs_vals else 0.0
            rev_cross_std = round(float(np.std(rev_vals)) * 100.0, 2) if rev_vals else 0.0
            shuf_cross_std = round(float(np.std(shuf_vals)) * 100.0, 2) if shuf_vals else 0.0

            permutation_report = {
                "dfs": {
                    "mean": round(dfs_f1, 2),
                    "std": 0.0,
                    "intra_sample_std": 0.0,
                    "cross_sample_std": dfs_cross_std,
                    "values": [round(float(v) * 100.0, 2) for v in dfs_vals],
                },
                "reverse": {
                    "mean": round(rev_f1, 2),
                    "std": 0.0,
                    "intra_sample_std": 0.0,
                    "cross_sample_std": rev_cross_std,
                    "values": [round(float(v) * 100.0, 2) for v in rev_vals],
                },
                "shuffles": {
                    "mean": round(shuf_f1_mean, 2),
                    "std": round(shuf_intra_std, 2),
                    "intra_sample_std": round(shuf_intra_std, 2),
                    "cross_sample_std": shuf_cross_std,
                    "values": [round(float(v) * 100.0, 2) for v in shuf_vals],
                },
                "max_drop": round(max_drop_val * 100.0 if max_drop_val <= 1.0 else max_drop_val, 2),
                "dfs_order": {"f1": round(dfs_f1, 2), "intra_sample_std": 0.0, "cross_sample_std": dfs_cross_std},
                "reverse_order": {"f1": round(rev_f1, 2), "intra_sample_std": 0.0, "cross_sample_std": rev_cross_std},
                "random_shuffles": {
                    "f1_mean": round(shuf_f1_mean, 2),
                    "f1_std": round(shuf_intra_std, 2),
                    "intra_sample_std": round(shuf_intra_std, 2),
                    "cross_sample_std": shuf_cross_std,
                },
                "order_robustness_drop": {
                    "f1_drop": round(max(0.0, dfs_f1 - shuf_f1_mean), 2),
                    "relative_drop_pct": round(max(0.0, dfs_f1 - shuf_f1_mean) / max(1e-4, dfs_f1) * 100.0, 2),
                },
                "num_samples_evaluated": len(eval_subset),
            }

        _sync_exp3_disk()
        logger.info(f"Checkpointed Exp 3 mode {mode} to disk.")

    if target_modes is not None:
        logger.info("Target modes completed. Skipping 3B scaling run.")
        del pipeline_7b
        del gen_7b
        clean_gpu_memory()
        return {
            "metadata": {"experiment": "Step 3 Ablations & Step 4 Permutation Test"},
            "ablations_7b": ablations_7b_results,
            "scaling_3b": scaling_3b_results,
            "permutation_test": permutation_report,
        }

    # Step 3.2: Teardown 7B, purge GPU cache, then load 3B for scaling law comparison
    logger.info("Tearing down 7B model and purging CUDA memory before 3B scaling test...")
    del pipeline_7b
    del gen_7b
    clean_gpu_memory()

    model_3b_path = models_dir / "Qwen2.5-3B-Instruct"
    logger.info(f"Loading 3B model from {model_3b_path} for scaling law comparison...")
    gen_3b = EfficientQwenNF4Generator(
        model_name_or_path=str(model_3b_path),
        dry_run=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pipeline_3b = UGraphRAGPipeline(
        generator=gen_3b,
        token_budget=512,
        top_k_standard=5,
        top_k_candidate_pool=int(gp.get("top_k_candidate_pool", 10)),
        m1_tau_eu=gp.get("m1_tau_eu"),
        m1_use_model_probe=True,
        m2_gamma_1=float(gp.get("m2_gamma_1", 0.5)),
        m2_gamma_2=float(gp.get("m2_gamma_2", 0.3)),
        m2_lambda_div=float(gp.get("m2_lambda_div", 0.4)),
        m2_reliability_offset=float(gp.get("m2_reliability_offset", 0.22)),
        m2_min_docs=int(gp.get("m2_min_docs", 2)),
        m3_beta=float(gp.get("m3_beta", 0.12)),
        m3_alpha_trunc=float(gp.get("m3_alpha_trunc", 0.10)),
        m3_abstain_threshold=float(gp.get("m3_abstain_threshold", 0.25)),
        m3_confidence_threshold=float(gp.get("m3_confidence_threshold", 0.50)),
        m3_confidence_temperature=float(gp.get("m3_confidence_temperature", 1.5)),
        m3_confidence_tok_weight=float(gp.get("m3_confidence_tok_weight", 0.50)),
        m3_confidence_platt_a=float(gp.get("m3_confidence_platt_a", 1.0)),
        m3_confidence_platt_b=float(gp.get("m3_confidence_platt_b", 0.0)),
        m3_contrastive_conf_guard=float(gp.get("m3_contrastive_conf_guard", 0.85)),
        m3_protect_eos=bool(gp.get("m3_protect_eos", True)),
        dense_model_path=str(dense_local) if dense_local.exists() else None,
    )

    for mode in ["B0_Standard_RAG", "Full_UGraphRAG"]:
        if mode in scaling_3b_results:
            logger.info(f"---> [Checkpoint Hit] Skipping completed 3B Mode: {mode}")
            continue
        logger.info(f"---> Evaluating 3B on {mode} (N={len(hotpot_subset)}) ...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rep = pipeline_3b.evaluate_dataset(hotpot_subset, mode=mode, log_interval=25)
        scaling_3b_results[mode] = rep
        logger.info(
            f"[3B {mode}] EM: {rep['exact_match']:.4f}, F1: {rep.get('f1_score', rep.get('f1', 0.0)):.4f}, "
            f"PeakVRAM: {rep.get('peak_cuda_allocated_mb', 0.0):.1f} MB"
        )
        _sync_exp3_disk()
        logger.info(f"Checkpointed 3B mode {mode} to disk.")

    del pipeline_3b
    del gen_3b
    clean_gpu_memory()

    _sync_exp3_disk()
    logger.info(f"Experiment 3 completed and saved to {exp3_file}, {results_dir / 'step3_hotpotqa_ablations.json'}, and {results_dir / 'step4_m2_permutation.json'}")
    return {
        "metadata": {
            "experiment": "HotpotQA 8-Mode Ablation Matrix & Permutation Test",
            "pipeline_version": "v4_theory_complete",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(hotpot_subset),
        },
        "ablations_7b": ablations_7b_results,
        "scaling_3b": scaling_3b_results,
        "permutation_test": permutation_report,
    }


# -----------------------------------------------------------------------------
# 4. Experiment 4: Plotting Figures & LaTeX Table Generation
# -----------------------------------------------------------------------------
def run_experiment_4(results_dir: Path, figures_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Executes Experiment 4:
    Invokes scripts/plot_ablation_charts.py to generate figures and tables_latex.tex.
    """
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT 4: Publication Chart & LaTeX Table Generation")
    logger.info("=" * 70)

    import subprocess
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_ablation_charts.py"),
        "--results-dir", str(results_dir),
        "--figures-dir", str(figures_dir),
        "--data-dir", str(data_dir),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"Plotting script failed with code {proc.returncode}:\n{proc.stderr}")
    else:
        logger.info(f"Plotting script succeeded:\n{proc.stdout}")

    expected_files = [
        figures_dir / "fig1_popqa_eu_vs_popularity.png",
        figures_dir / "fig2_hotpotqa_ablation_progression.png",
        figures_dir / "fig3_permutation_test_variance.png",
        figures_dir / "tables_latex.tex",
    ]

    file_status = {}
    for f in expected_files:
        exists = f.exists()
        size_kb = f.stat().st_size / 1024.0 if exists else 0.0
        file_status[f.name] = {"exists": exists, "size_kb": round(size_kb, 2)}
        logger.info(f"Generated asset check: {f.name} -> Exists: {exists} ({size_kb:.1f} KB)")

    return file_status


# -----------------------------------------------------------------------------
# Main Orchestration Loop
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Overnight Ablation Master Script")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset JSONs")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing Qwen models")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save experimental results")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory for figures and latex")
    parser.add_argument("--hotpot-limit", type=int, default=100, help="Number of HotpotQA samples for ablations")
    parser.add_argument("--popqa-limit", type=int, default=100, help="Number of PopQA samples")
    parser.add_argument("--perm-limit", type=int, default=100, help="Number of HotpotQA samples for Step 4 Permutation Test")
    parser.add_argument("--grid-search", action="store_true", help="Run unified single-pass M1/M2/M3 grid search before Exp 2/3")
    parser.add_argument("--grid-offset", type=int, default=100, help="Dev slice offset for grid search (avoids [0:100] test leak)")
    parser.add_argument("--grid-limit", type=int, default=80, help="Number of validation samples for grid search")
    parser.add_argument("--grid-preset", type=str, default="balanced", choices=["balanced", "exhaustive"], help="Grid sweep granularity")
    parser.add_argument("--params-file", type=str, default="models/best_grid_params.json", help="Path to grid-searched parameter JSON")
    parser.add_argument("--params-preset", type=str, default="balanced_optimal", choices=["balanced_optimal", "low_ece_high_precision", "full_coverage_calibrated"], help="Preset key inside params-file")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip Experiment 1")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip Experiment 2")
    parser.add_argument("--skip-exp3", action="store_true", help="Skip Experiment 3")
    parser.add_argument("--skip-exp4", action="store_true", help="Skip Experiment 4")
    parser.add_argument("--force", action="store_true", help="Ignore prior checkpoints and rerun selected Exp 3 modes cleanly")
    parser.add_argument("--model-select", type=str, default="all", choices=["all", "3b", "7b"], help="Select model for Exp 1")
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated modes for Exp 3")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    models_dir = PROJECT_ROOT / args.models_dir
    results_dir = PROJECT_ROOT / args.results_dir
    figures_dir = PROJECT_ROOT / args.figures_dir
    params_path = PROJECT_ROOT / args.params_file

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== INITIALIZING OVERNIGHT ABLATION SUITE ===")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Initial VRAM: {get_vram_metrics()}")

    hotpot_samples, popqa_samples = load_datasets(data_dir)

    t_all_start = time.time()

    # Optional Stage 0: Unified Single-Pass M1/M2/M3 Grid Search (outputs multiple valid presets)
    if args.grid_search:
        import importlib.util
        gs_path = PROJECT_ROOT / "scripts" / "run_parameter_grid_search.py"
        gs_spec = importlib.util.spec_from_file_location("ugraphrag_grid_search", str(gs_path))
        gs_mod = importlib.util.module_from_spec(gs_spec)
        gs_spec.loader.exec_module(gs_mod)
        run_full_grid_search = gs_mod.run_full_grid_search
        logger.info(">>> STAGE 0: Running Unified Single-Pass Multi-Parameter Grid Search...")
        run_full_grid_search(
            data_dir=data_dir,
            models_dir=models_dir,
            results_dir=results_dir,
            offset=args.grid_offset,
            limit=args.grid_limit,
            preset=args.grid_preset,
            dry_run=not torch.cuda.is_available(),
            save_best_to=params_path,
        )
        clean_gpu_memory()

    grid_params = _load_grid_params(params_path, preset=args.params_preset)

    # Exp 1
    if not args.skip_exp1:
        run_experiment_1(hotpot_samples, popqa_samples, models_dir, results_dir, limit=args.hotpot_limit, model_select=args.model_select)

    # Exp 2
    if not args.skip_exp2:
        run_experiment_2(hotpot_samples, popqa_samples, models_dir, results_dir, limit=args.popqa_limit, grid_params=grid_params)

    # Exp 3
    if not args.skip_exp3:
        target_modes = args.modes.split(",") if args.modes else None
        run_experiment_3(
            hotpot_samples,
            models_dir,
            results_dir,
            limit=args.hotpot_limit,
            target_modes=target_modes,
            grid_params=grid_params,
            perm_limit=args.perm_limit,
            force=args.force,
        )

    # Exp 4
    if not args.skip_exp4:
        run_experiment_4(results_dir, figures_dir, data_dir)

    total_time = time.time() - t_all_start
    logger.info(f"=== ALL EXPERIMENTAL STAGES COMPLETED IN {total_time/60.0:.2f} MINUTES ===")


if __name__ == "__main__":
    main()
