"""
scripts/run_parameter_grid_search.py
====================================
Unified Single-Pass Multi-Parameter Grid Search Engine for U-GraphRAG (M1 + M2 + M3).

Why Grid Search in One Experiment?
  Traditional serial sweeps reload 4-bit LLM weights and recompute identical prompts
  repeatedly. This engine executes shared caching across all 3 modules so that ONE
  GPU experiment simultaneously evaluates dozens of M1/M2/M3 parameter combinations and
  outputs a validated Pareto leaderboard of multiple valid parameter sets:

  - Stage 1 (M1): Runs ONE cached next-token probe + D0/expanded generation per sample,
    then sweeps 10+ tau_EU thresholds (fixed grid + empirical percentiles) in 1ms.
  - Stage 2 (M2): Runs an exhaustive 54+ CPU graph optimization grid across
    (gamma_1, gamma_2, lambda_div, reliability_offset, top_k_candidate_pool), then
    evaluates Top-K Pareto candidates with prompt-tuple LLM caching.
  - Stage 3 (M3): Runs a compact decoding sweep over (beta, contrastive_conf_guard),
    captures raw unsharpened context logits across 7 temperatures, and evaluates 2,000+
    (confidence_temperature, w_tok, abstain_threshold, confidence_threshold)
    post-calibration & gating combinations on CPU in <1 second.

Usage:
  # Local fast smoke verification (no GPU weights required)
  python scripts/run_parameter_grid_search.py --dry-run --limit 15

  # Real RTX 5060 (8GB) tuning on disjoint dev slice [100:145]
  python scripts/run_parameter_grid_search.py --offset 100 --limit 45 --preset balanced
"""

import os
import sys
import gc
import copy
import json
import math
import time
import logging
import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.baseline_rag import QwenNF4Generator
from src.m1_adaptive_retriever import compute_entropy
from src.m2_evidence_optimizer import (
    InteractionMatrix,
    MSTEvidenceOrganizer,
    TwoStageM2Optimizer,
    calibrate_reliability,
)
from src.m3_calibrated_decoder import (
    CALIBRATION_TEMP_GRID,
    CalibratedDecoderConfig,
    M3CalibratedDecoder,
    compute_calibrated_confidence,
)
from src.evaluation_toolkit import (
    compute_detailed_calibration,
    compute_selective_prediction_metrics,
)
from src.u_graphrag_pipeline import (
    UGraphRAGPipeline,
    compute_exact_match,
    compute_f1_score,
    compute_supporting_facts_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [GridSearch] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parameter_grid_search")


def _extract_gold_titles(sample: Dict[str, Any]) -> Set[str]:
    sp_obj = sample.get("supporting_facts")
    titles: Set[str] = set()
    if isinstance(sp_obj, dict):
        titles = {str(t).strip() for t in sp_obj.get("title", []) if str(t).strip()}
    elif isinstance(sp_obj, list):
        for entry in sp_obj:
            if isinstance(entry, dict) and "title" in entry:
                titles.add(str(entry["title"]).strip())
            elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                titles.add(str(entry[0]).strip())
    return titles


def _load_hotpot_slice(data_dir: Path, offset: int, limit: int) -> List[Dict[str, Any]]:
    path = data_dir / "hotpotqa_dev_300.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    end = min(len(raw), offset + limit)
    if offset >= len(raw):
        offset = 0
        end = min(len(raw), limit)
    return raw[offset:end]


def _count_tokens(docs: Sequence[Dict[str, Any]]) -> int:
    return sum(max(1, len(str(d.get("text", "")).split())) for d in docs)


def run_m1_grid_search(
    pipeline: UGraphRAGPipeline,
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Stage 1 Grid Search:
    Computes U_EU(q, D_0), early-exit outcome (D_0), and expanded outcome once per sample,
    then evaluates an entire threshold grid (fixed + empirical percentiles) in 1ms.
    """
    logger.info("=" * 68)
    logger.info(f"STAGE 1: M1 Uncertainty Routing Threshold Grid (N={len(samples)})")
    logger.info("=" * 68)

    cached_items: List[Dict[str, Any]] = []
    answer_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[str, float, float]] = {}

    def _eval_docs(idx: int, sample: Dict[str, Any], docs: List[Dict[str, Any]]) -> Tuple[str, float, float]:
        key = (idx, tuple(str(d.get("title", d.get("id", ""))) for d in docs))
        if key in answer_cache:
            return answer_cache[key]
        query = sample["question"]
        gt = str(sample.get("ground_truth") or sample.get("answer", ""))
        prompt = pipeline.format_prompt(query, docs)
        if pipeline.generator is not None:
            pred, _ = pipeline.generator.generate(prompt, docs)
        else:
            pred = f"Mock answer for {query}"
        em = compute_exact_match(pred, gt)
        f1 = compute_f1_score(pred, gt)
        answer_cache[key] = (pred, float(em), float(f1))
        return answer_cache[key]

    eff_probe = getattr(pipeline.m1, "default_model_fn", None)

    for idx, sample in enumerate(samples):
        query = sample["question"]
        gold_sp = _extract_gold_titles(sample)
        ranked_pool = pipeline._retrieve_pool(query, sample, top_k=max(10, pipeline.top_k_candidate_pool))
        b0_docs = ranked_pool[:pipeline.top_k_standard]
        # Initial seed D_0 uses top-3 to let M1 earn genuine token savings when confident
        d0_docs = ranked_pool[:min(3, pipeline.top_k_standard)]

        if hasattr(pipeline.m1, "decoupler"):
            if hasattr(pipeline.m1, "index_corpus") and ranked_pool:
                pipeline.m1.index_corpus(ranked_pool)
            metrics = pipeline.m1.decoupler.decompose(query, d0_docs, model_fn=eff_probe)
            u_eu = float(metrics.epistemic_uncertainty)
            u_au = float(metrics.aleatoric_uncertainty)
            h_tot = float(metrics.total_entropy)
            exp_res = pipeline.m1.expander.expand(
                query=query,
                initial_docs=d0_docs,
                epistemic_uncertainty=u_eu,
                tau_eu=-1.0,  # force expansion once so we cache the expanded pool
                gold_ids=None,
            )
            score_map = {str(d.get("id", d.get("title", ""))): float(d.get("score", 0.5)) for d in ranked_pool}
            init_ids = {str(d.get("id", d.get("title", ""))) for d in d0_docs}
            seen: Set[str] = set()
            merged: List[Dict[str, Any]] = []
            for d in list(exp_res.expanded_candidates) + list(ranked_pool):
                did = str(d.get("id", d.get("title", "")))
                if did and did not in seen:
                    dc = dict(d)
                    dc["score"] = score_map.get(did, float(dc.get("score", 0.5)))
                    merged.append(dc)
                    seen.add(did)
            merged.sort(
                key=lambda d: (1 if str(d.get("id", d.get("title", ""))) in init_ids else 0, float(d.get("score", 0.5))),
                reverse=True,
            )
            exp_docs = merged[:pipeline.top_k_standard]
        else:
            u_eu, u_au, h_tot = 0.25, 0.20, 0.45
            exp_docs = list(b0_docs)

        sp_b0 = compute_supporting_facts_metrics([str(d.get("title", d.get("id", ""))) for d in b0_docs], gold_sp)
        sp_d0 = compute_supporting_facts_metrics([str(d.get("title", d.get("id", ""))) for d in d0_docs], gold_sp)
        sp_exp = compute_supporting_facts_metrics([str(d.get("title", d.get("id", ""))) for d in exp_docs], gold_sp)

        _, em_b0, f1_b0 = _eval_docs(idx, sample, b0_docs)
        _, em_d0, f1_d0 = _eval_docs(idx, sample, d0_docs)
        _, em_exp, f1_exp = _eval_docs(idx, sample, exp_docs)

        b0_toks = max(1, _count_tokens(b0_docs))
        cached_items.append({
            "u_eu": u_eu,
            "u_au": u_au,
            "h_tot": h_tot,
            "b0": {"em": em_b0, "f1": f1_b0, "sp_f1": sp_b0["sp_f1"], "sp_rec": sp_b0["sp_recall"], "tokens": b0_toks},
            "early": {"em": em_d0, "f1": f1_d0, "sp_f1": sp_d0["sp_f1"], "sp_rec": sp_d0["sp_recall"], "tokens": _count_tokens(d0_docs)},
            "expanded": {"em": em_exp, "f1": f1_exp, "sp_f1": sp_exp["sp_f1"], "sp_rec": sp_exp["sp_recall"], "tokens": _count_tokens(exp_docs)},
        })

    eu_vals = np.array([it["u_eu"] for it in cached_items], dtype=np.float64)
    pct_thresholds = [float(np.quantile(eu_vals, q)) for q in (0.20, 0.35, 0.50, 0.65, 0.80)] if len(eu_vals) > 0 else []
    fixed_thresholds = [0.06, 0.10, 0.12, 0.1528, 0.18, 0.22, 0.28, 0.35, 0.50, 1.00, 1.50]
    candidate_taus = sorted({round(t, 4) for t in (fixed_thresholds + pct_thresholds) if t > 0.0})

    b0_em = float(np.mean([it["b0"]["em"] for it in cached_items])) * 100.0
    b0_f1 = float(np.mean([it["b0"]["f1"] for it in cached_items])) * 100.0
    b0_spf1 = float(np.mean([it["b0"]["sp_f1"] for it in cached_items])) * 100.0

    grid_results: List[Dict[str, Any]] = []
    for tau in candidate_taus:
        ems, f1s, sp_f1s, sp_recs, savings = [], [], [], [], []
        early_cnt = 0
        for it in cached_items:
            is_early = it["u_eu"] <= tau
            if is_early:
                early_cnt += 1
            chosen = it["early"] if is_early else it["expanded"]
            ems.append(chosen["em"])
            f1s.append(chosen["f1"])
            sp_f1s.append(chosen["sp_f1"])
            sp_recs.append(chosen["sp_rec"])
            sav = max(0.0, 1.0 - chosen["tokens"] / it["b0"]["tokens"])
            savings.append(sav)

        early_rate = (early_cnt / len(cached_items)) * 100.0 if cached_items else 0.0
        avg_em = float(np.mean(ems)) * 100.0
        avg_f1 = float(np.mean(f1s)) * 100.0
        avg_spf1 = float(np.mean(sp_f1s)) * 100.0
        avg_sprec = float(np.mean(sp_recs)) * 100.0
        avg_sav = float(np.mean(savings)) * 100.0

        # Valid adaptive M1 config: genuinely branches (15%..85% early exit) AND preserves quality vs B0
        is_valid = (15.0 <= early_rate <= 85.0) and (avg_f1 >= b0_f1 - 1.0)
        score = (avg_f1 - b0_f1) + 0.20 * avg_sav + (2.0 if is_valid else -5.0)

        grid_results.append({
            "tau_eu": tau,
            "early_exit_rate_pct": round(early_rate, 2),
            "expansion_rate_pct": round(100.0 - early_rate, 2),
            "em": round(avg_em, 2),
            "f1": round(avg_f1, 2),
            "delta_f1_vs_b0": round(avg_f1 - b0_f1, 2),
            "sp_f1": round(avg_spf1, 2),
            "sp_recall": round(avg_sprec, 2),
            "token_saving_pct": round(avg_sav, 2),
            "is_valid": bool(is_valid),
            "composite_score": round(score, 3),
        })

    grid_results.sort(key=lambda r: (r["is_valid"], r["composite_score"], r["f1"]), reverse=True)
    valid_list = [r for r in grid_results if r["is_valid"]]

    logger.info(
        f"[Stage 1 Done] B0 Baseline: EM={b0_em:.2f}%, F1={b0_f1:.2f}%, Sp-F1={b0_spf1:.2f}% | "
        f"U_EU quantiles (min/p50/max): {float(np.min(eu_vals)):.4f} / {float(np.median(eu_vals)):.4f} / {float(np.max(eu_vals)):.4f}"
    )
    logger.info(f"[Stage 1] Found {len(valid_list)} valid tau_eu parameters out of {len(grid_results)} tested.")
    return {
        "b0_baseline": {"em": round(b0_em, 2), "f1": round(b0_f1, 2), "sp_f1": round(b0_spf1, 2)},
        "u_eu_distribution": {
            "min": round(float(np.min(eu_vals)), 4),
            "p25": round(float(np.quantile(eu_vals, 0.25)), 4),
            "median": round(float(np.median(eu_vals)), 4),
            "p75": round(float(np.quantile(eu_vals, 0.75)), 4),
            "max": round(float(np.max(eu_vals)), 4),
        },
        "valid_configs": valid_list[:8],
        "all_grid": grid_results,
    }


def run_m2_grid_search(
    pipeline: UGraphRAGPipeline,
    samples: List[Dict[str, Any]],
    preset: str = "balanced",
) -> Dict[str, Any]:
    """
    Stage 2 Grid Search:
    Phase 2a: Pre-computes candidate pools & edges ONCE per sample, then sweeps 36-72 M2
              weight/offset/pool combinations on CPU in ~1-2s (tracks Exact vs Greedy separation!).
    Phase 2b: Evaluates Top diverse Pareto configs + baseline default with real LLM generation,
              reusing exact prompt-sequence cache so identical evidence chains never run twice.
    """
    logger.info("=" * 68)
    logger.info(f"STAGE 2: M2 Combinatorial Optimization & Topology Grid (N={len(samples)})")
    logger.info("=" * 68)

    # Pre-cache pool & graph edges up to max_pool=15 once per sample
    max_pool = 15
    precomputed: List[Dict[str, Any]] = []
    for sample in samples:
        query = sample["question"]
        gold_sp = _extract_gold_titles(sample)
        ranked_pool = pipeline._retrieve_pool(query, sample, top_k=max_pool)
        b0_tokens = max(1, _count_tokens(ranked_pool[:pipeline.top_k_standard]))

        # Attach lexical marginal stats via M1 in O(1) CPU mode if available
        per_doc_dists: Dict[str, Dict[str, float]] = {}
        m2_meta_map: Dict[str, Dict[str, Any]] = {}
        if hasattr(pipeline.m1, "index_corpus") and ranked_pool:
            pipeline.m1.index_corpus(ranked_pool)
        if hasattr(pipeline.m1, "process"):
            try:
                _, m2_cands, per_doc_dists = pipeline.m1.process(
                    query, ranked_pool[:min(5, len(ranked_pool))], model_fn=None
                )
                for mc in m2_cands:
                    m2_meta_map[str(mc.get("doc_id", ""))] = mc
            except Exception:
                pass

        cands_all = pipeline._convert_to_candidates(ranked_pool, m2_meta_map=m2_meta_map)
        query_ents = pipeline._extract_entities(query)
        edges_all = pipeline._build_explicit_edges(cands_all, per_doc_dists)

        precomputed.append({
            "sample": sample,
            "query": query,
            "gt": str(sample.get("ground_truth") or sample.get("answer", "")),
            "gold_sp": gold_sp,
            "b0_tokens": b0_tokens,
            "cands_all": cands_all,
            "edges_all": edges_all,
            "query_ents": query_ents,
        })

    if preset == "exhaustive":
        gamma1_grid = [0.3, 0.5, 0.8]
        gamma2_grid = [0.2, 0.3, 0.5]
        lmb_div_grid = [0.2, 0.4, 0.6]
        offset_grid = [0.16, 0.22, 0.28, 0.34]
        pool_grid = [8, 10]
    else:
        gamma1_grid = [0.3, 0.5, 0.8]
        gamma2_grid = [0.2, 0.3, 0.5]
        lmb_div_grid = [0.2, 0.4]
        offset_grid = [0.16, 0.22, 0.28, 0.34]
        pool_grid = [8, 10]

    cpu_grid_summary: List[Dict[str, Any]] = []
    organizer = MSTEvidenceOrganizer(recency_aware=False)

    t_cpu_0 = time.perf_counter()
    for pool_k, offset, g1, g2, ldiv in product(pool_grid, offset_grid, gamma1_grid, gamma2_grid, lmb_div_grid):
        opt_exact = TwoStageM2Optimizer(
            lambda_coverage=g1,
            lambda_complementarity=g2,
            lambda_redundancy=ldiv,
            lambda_conflict=1.0,
            selection_algorithm="exact_enum",
            recency_aware=False,
        )
        opt_greedy = TwoStageM2Optimizer(
            lambda_coverage=g1,
            lambda_complementarity=g2,
            lambda_redundancy=ldiv,
            lambda_conflict=1.0,
            selection_algorithm="greedy",
            recency_aware=False,
        )

        sp_ps, sp_rs, sp_f1s, savings, doc_cnts = [], [], [], [], []
        diff_greedy_exact = 0
        per_sample_ordered_ids: List[Tuple[str, ...]] = []
        per_sample_ordered_docs: List[List[Dict[str, Any]]] = []

        for item in precomputed:
            sub_cands = [copy.copy(c) for c in item["cands_all"][:pool_k]]
            sub_ids = {c.doc_id for c in sub_cands}
            sub_edges = {k: v for k, v in item["edges_all"].items() if k[0] in sub_ids and k[1] in sub_ids}
            imat = InteractionMatrix.from_candidates_and_edges(sub_cands, sub_edges)

            calibrate_reliability(sub_cands, method="minmax", gamma_1=g1, gamma_2=g2)
            for c in sub_cands:
                c.calibrated_reliability -= offset

            hop_min_k = min(2, len(sub_cands)) if sub_cands else 1
            res_exact = opt_exact.optimize(
                candidates=sub_cands,
                interaction_matrix=imat,
                explicit_edges=sub_edges,
                max_k=pipeline.top_k_final,
                query_entities=item["query_ents"] or None,
                calibrate=False,
                min_k=hop_min_k,
            )
            res_greedy = opt_greedy.selector.greedy_select(
                candidates=sub_cands,
                interaction_matrix=imat,
                max_k=pipeline.top_k_final,
                query_entities=item["query_ents"] or None,
                min_k=hop_min_k,
            )

            exact_set = {c.doc_id for c in res_exact.selected_docs}
            greedy_set = {c.doc_id for c in res_greedy.selected_docs}
            if exact_set != greedy_set:
                diff_greedy_exact += 1

            ordered_dicts = [
                {
                    "id": c.doc_id,
                    "title": c.metadata.get("title", c.doc_id),
                    "text": c.text,
                    "score": c.score,
                    "calibrated_reliability": float(c.calibrated_reliability + offset),
                    "aleatoric_uncertainty": c.aleatoric_uncertainty,
                }
                for c in res_exact.ordered_docs
            ]
            titles = [str(d["title"]) for d in ordered_dicts]
            sp = compute_supporting_facts_metrics(titles, item["gold_sp"])
            sp_ps.append(sp["sp_precision"])
            sp_rs.append(sp["sp_recall"])
            sp_f1s.append(sp["sp_f1"])
            doc_cnts.append(len(ordered_dicts))
            savings.append(max(0.0, 1.0 - _count_tokens(ordered_dicts) / item["b0_tokens"]))
            per_sample_ordered_ids.append(tuple(titles))
            per_sample_ordered_docs.append(ordered_dicts)

        avg_sp_p = float(np.mean(sp_ps)) * 100.0
        avg_sp_r = float(np.mean(sp_rs)) * 100.0
        avg_sp_f1 = float(np.mean(sp_f1s)) * 100.0
        avg_sav = float(np.mean(savings)) * 100.0
        avg_docs = float(np.mean(doc_cnts))
        diff_pct = (diff_greedy_exact / len(precomputed)) * 100.0 if precomputed else 0.0

        # Structural Pareto score balancing supporting-fact F1, recall, and token savings
        struct_score = avg_sp_f1 + 0.15 * avg_sav + 0.10 * avg_sp_r + (0.5 if diff_pct > 0 else 0.0)
        cpu_grid_summary.append({
            "params": {
                "m2_gamma_1": g1,
                "m2_gamma_2": g2,
                "m2_lambda_div": ldiv,
                "m2_reliability_offset": offset,
                "m2_min_docs": 2,
                "top_k_candidate_pool": pool_k,
            },
            "sp_precision": round(avg_sp_p, 2),
            "sp_recall": round(avg_sp_r, 2),
            "sp_f1": round(avg_sp_f1, 2),
            "token_saving_pct": round(avg_sav, 2),
            "avg_selected_docs": round(avg_docs, 2),
            "exact_vs_greedy_diff_pct": round(diff_pct, 2),
            "structural_score": round(struct_score, 3),
            "_per_sample_ordered_docs": per_sample_ordered_docs,
            "_signature": tuple(per_sample_ordered_ids),
        })

    t_cpu_dur = time.perf_counter() - t_cpu_0
    logger.info(f"[Stage 2a] Evaluated {len(cpu_grid_summary)} CPU M2 configs in {t_cpu_dur:.2f}s.")

    cpu_grid_summary.sort(key=lambda x: x["structural_score"], reverse=True)

    # Offset-stratified selection: prevent aggressive pruning (0.34) from starving gentle offsets on CPU proxy
    selected_for_llm: List[Dict[str, Any]] = []
    seen_sigs: Set[Tuple[Tuple[str, ...], ...]] = set()
    for target_off in offset_grid:
        bucket = [c for c in cpu_grid_summary if abs(c["params"]["m2_reliability_offset"] - target_off) < 1e-6]
        picked_in_bucket = 0
        for cfg in bucket:
            sig = cfg["_signature"]
            if sig not in seen_sigs:
                selected_for_llm.append(cfg)
                seen_sigs.add(sig)
                picked_in_bucket += 1
                if picked_in_bucket >= 2:
                    break
    for cfg in cpu_grid_summary:
        p = cfg["params"]
        is_default = (
            p.get("m2_gamma_1") == 0.5
            and p.get("m2_gamma_2") == 0.3
            and p.get("m2_lambda_div") == 0.4
            and p.get("m2_reliability_offset") == 0.22
            and p.get("top_k_candidate_pool") == 10
        )
        sig = cfg["_signature"]
        if (sig not in seen_sigs and len(selected_for_llm) < 9) or is_default:
            if not any(x["params"] == p for x in selected_for_llm):
                selected_for_llm.append(cfg)
                seen_sigs.add(sig)

    # Stage 2b: Evaluate Top Pareto M2 configs with real LLM + prompt cache
    llm_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[str, float, float]] = {}
    llm_evaluated: List[Dict[str, Any]] = []

    for cfg in selected_for_llm:
        ems, f1s = [], []
        for idx, item in enumerate(precomputed):
            ordered_docs = cfg["_per_sample_ordered_docs"][idx]
            key = (idx, tuple(str(d["title"]) for d in ordered_docs))
            if key in llm_cache:
                _, em, f1 = llm_cache[key]
            else:
                prompt = pipeline.format_prompt(item["query"], ordered_docs)
                if pipeline.generator is not None:
                    pred, _ = pipeline.generator.generate(prompt, ordered_docs)
                else:
                    pred = f"Mock answer for {item['query']}"
                em = float(compute_exact_match(pred, item["gt"]))
                f1 = float(compute_f1_score(pred, item["gt"]))
                llm_cache[key] = (pred, em, f1)
            ems.append(em)
            f1s.append(f1)

        avg_em = float(np.mean(ems)) * 100.0
        avg_f1 = float(np.mean(f1s)) * 100.0
        out_entry = {k: v for k, v in cfg.items() if not k.startswith("_")}
        out_entry["em"] = round(avg_em, 2)
        out_entry["f1"] = round(avg_f1, 2)
        llm_evaluated.append(out_entry)

    best_llm_f1 = max((r["f1"] for r in llm_evaluated), default=0.0)
    best_llm_em = max((r["em"] for r in llm_evaluated), default=0.0)
    for r in llm_evaluated:
        within_floor = (r["f1"] >= best_llm_f1 - 1.5) and (r["em"] >= best_llm_em - 2.0)
        r["within_acc_floor"] = bool(within_floor)
        r["joint_score"] = round(
            r["f1"]
            + 0.5 * r["em"]
            + 0.06 * r["sp_f1"]
            + 0.04 * r["token_saving_pct"]
            + (5.0 if within_floor else -10.0),
            3,
        )

    llm_evaluated.sort(key=lambda r: r["joint_score"], reverse=True)
    # Strip private fields from full CPU table
    clean_cpu_table = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cpu_grid_summary[:20]]

    best_cfg = llm_evaluated[0]
    best_ordered_docs = next(
        c["_per_sample_ordered_docs"] for c in selected_for_llm if c["params"] == best_cfg["params"]
    )
    logger.info(
        f"[Stage 2b Done] Unique LLM prompts executed: {len(llm_cache)} (saved massive duplicate decoding) | "
        f"Best M2 -> EM={best_cfg['em']:.2f}%, F1={best_cfg['f1']:.2f}%, "
        f"Sp-F1={best_cfg['sp_f1']:.2f}%, Saving={best_cfg['token_saving_pct']:.1f}% | "
        f"Params={best_cfg['params']}"
    )

    return {
        "evaluated_llm_configs": llm_evaluated,
        "top20_cpu_structural_grid": clean_cpu_table,
        "best_params": best_cfg["params"],
        "_best_ordered_docs": best_ordered_docs,
        "_precomputed": precomputed,
    }


def _fit_platt_regularized(raw_confs: Sequence[float], targets: Sequence[float]) -> Tuple[float, float]:
    """Fit regularized 1D Platt logistic recalibration shrinking smoothly toward identity (a=1, b=0)."""
    n = len(raw_confs)
    if n < 8 or len(targets) != n:
        return 1.0, 0.0
    y = np.asarray(targets, dtype=np.float64)
    pos = float(np.sum(y))
    if pos < 2.0 or (n - pos) < 2.0:
        return 1.0, 0.0

    c = np.clip(np.asarray(raw_confs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    z = np.log(c / (1.0 - c))
    # Strong zero-intercept shrinkage prevents dev-slice base-rate shift from biasing test calibration
    lam_a, lam_b = 4.0, 14.0
    a, b = 1.0, 0.0

    for _ in range(20):
        s = np.clip(a * z + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-s))
        ga = float(np.sum((p - y) * z) + lam_a * (a - 1.0))
        gb = float(np.sum(p - y) + lam_b * b)
        w = np.maximum(p * (1.0 - p), 1e-6)
        haa = float(np.sum(w * z * z) + lam_a)
        hbb = float(np.sum(w) + lam_b)
        hab = float(np.sum(w * z))
        det = haa * hbb - hab * hab
        if abs(det) < 1e-10:
            break
        da = max(-0.5, min(0.5, (hbb * ga - hab * gb) / det))
        db = max(-0.5, min(0.5, (-hab * ga + haa * gb) / det))
        a = float(np.clip(a - da, 0.55, 2.20))
        b = float(np.clip(b - db, -0.28, 0.35))
        if abs(da) < 1e-5 and abs(db) < 1e-5:
            break

    return round(a, 4), round(b, 4)


def _apply_platt_scores(raw_confs: Sequence[float], a: float, b: float) -> List[float]:
    if abs(a - 1.0) <= 1e-6 and abs(b) <= 1e-6:
        return [float(c) for c in raw_confs]
    c = np.clip(np.asarray(raw_confs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    z = np.log(c / (1.0 - c))
    s = np.clip(a * z + b, -30.0, 30.0)
    return [float(v) for v in (1.0 / (1.0 + np.exp(-s)))]


def _cross_validated_platt_confs(raw_confs: Sequence[float], targets: Sequence[float]) -> Tuple[float, float, List[float]]:
    """Compute honest out-of-fold Platt predictions on dev slice plus full-fit parameters."""
    n = len(raw_confs)
    full_a, full_b = _fit_platt_regularized(raw_confs, targets)
    if n < 12 or (abs(full_a - 1.0) <= 0.02 and abs(full_b) <= 0.02):
        return full_a, full_b, _apply_platt_scores(raw_confs, full_a, full_b)

    n_folds = 5 if n >= 25 else 3
    oof = [float(c) for c in raw_confs]
    for f in range(n_folds):
        tr_c = [raw_confs[i] for i in range(n) if i % n_folds != f]
        tr_y = [targets[i] for i in range(n) if i % n_folds != f]
        fa, fb = _fit_platt_regularized(tr_c, tr_y)
        val_idx = [i for i in range(n) if i % n_folds == f]
        val_scaled = _apply_platt_scores([raw_confs[i] for i in val_idx], fa, fb)
        for pos_idx, orig_idx in enumerate(val_idx):
            oof[orig_idx] = val_scaled[pos_idx]
    return full_a, full_b, oof


def run_m3_grid_search(
    pipeline: UGraphRAGPipeline,
    precomputed: List[Dict[str, Any]],
    ordered_docs_per_sample: List[List[Dict[str, Any]]],
    m2_baseline_metrics: Dict[str, float],
    preset: str = "balanced",
) -> Dict[str, Any]:
    """
    Stage 3 Grid Search:
    Sweeps (beta, contrastive_conf_guard) decoding passes ONCE per config on GPU, records
    multi-temperature raw context probabilities & grounding telemetry, then evaluates
    ~1,500-2,000 post-calibration (T_conf, w_tok, abstain_threshold, confidence_threshold)
    combinations on CPU in <1 second.
    """
    logger.info("=" * 68)
    logger.info(f"STAGE 3: M3 Decoding & Calibration/Gate Grid (N={len(precomputed)})")
    logger.info("=" * 68)

    if preset == "exhaustive":
        decode_grid = [
            {"beta": 0.0, "guard": 1.0},
            {"beta": 0.05, "guard": 0.88},
            {"beta": 0.10, "guard": 0.88},
            {"beta": 0.15, "guard": 0.85},
            {"beta": 0.20, "guard": 0.85},
            {"beta": 0.30, "guard": 0.85},
        ]
    else:
        decode_grid = [
            {"beta": 0.0, "guard": 1.0},
            {"beta": 0.04, "guard": 0.90},
            {"beta": 0.10, "guard": 0.88},
            {"beta": 0.18, "guard": 0.85},
        ]

    temp_grid = [0.8, 1.0, 1.25, 1.5, 2.0]
    w_tok_grid = [0.30, 0.45, 0.60, 0.75]
    abstain_rel_grid = [0.05, 0.15, 0.25]
    conf_thresh_grid = [0.15, 0.22, 0.30, 0.38, 0.45, 0.52]

    all_evaluated_configs: List[Dict[str, Any]] = []
    decode_pass_summaries: List[Dict[str, Any]] = []

    m2_em = float(m2_baseline_metrics.get("em", 0.0))
    m2_f1 = float(m2_baseline_metrics.get("f1", 0.0))

    for d_cfg in decode_grid:
        beta = d_cfg["beta"]
        guard = d_cfg["guard"]
        m3 = M3CalibratedDecoder(
            config=CalibratedDecoderConfig(
                beta=beta,
                enable_contrastive=(beta > 0.0),
                contrastive_conf_guard=guard,
                protect_eos=True,
                preserve_evidence_order=True,
                max_new_tokens=48,
                abstain_threshold=0.0,      # collect raw answer cleanly; sweeping thresholds post-hoc!
                confidence_threshold=0.0,
            )
        )

        raw_traces: List[Dict[str, Any]] = []
        for idx, item in enumerate(precomputed):
            ev_docs = ordered_docs_per_sample[idx]
            res = m3.generate_calibrated(
                generator=pipeline.generator,
                question=item["query"],
                evidence_items=ev_docs,
            )
            tel = dict(res.metadata.get("telemetry", {})) if isinstance(res.metadata, dict) else {}
            raw_ans = res.raw_answer or res.answer
            em_raw = float(compute_exact_match(raw_ans, item["gt"]))
            f1_raw = float(compute_f1_score(raw_ans, item["gt"]))
            raw_traces.append({
                "raw_answer": raw_ans,
                "em_raw": em_raw,
                "f1_raw": f1_raw,
                "acc_bin": 1.0 if f1_raw >= 0.5 else 0.0,
                "ctx_probs_by_temp": tel.get("ctx_probs_by_temp", {}),
                "max_ev_rel": float(tel.get("max_ev_rel", max((d.get("calibrated_reliability", 0.5) for d in ev_docs), default=0.5))),
                "mean_ev_rel": float(tel.get("mean_ev_rel", float(np.mean([d.get("calibrated_reliability", 0.5) for d in ev_docs])) if ev_docs else 0.5)),
                "is_faithful": bool(tel.get("is_faithful", True)),
                "has_unanchored": bool(tel.get("has_unanchored", False)),
            })

        decode_em = float(np.mean([t["em_raw"] for t in raw_traces])) * 100.0
        decode_f1 = float(np.mean([t["f1_raw"] for t in raw_traces])) * 100.0
        decode_pass_summaries.append({
            "beta": beta,
            "contrastive_conf_guard": guard,
            "raw_em": round(decode_em, 2),
            "raw_f1": round(decode_f1, 2),
            "delta_em_vs_m2": round(decode_em - m2_em, 2),
            "delta_f1_vs_m2": round(decode_f1 - m2_f1, 2),
        })

        # Exhaustive CPU calibration & abstention gate grid on captured telemetry (with 5-fold CV Platt calibration)
        accs_f1: List[float] = [tr["acc_bin"] for tr in raw_traces]
        base_risk = (1.0 - float(np.mean(accs_f1))) * 100.0 if accs_f1 else 0.0

        for T_conf, w_tok in product(temp_grid, w_tok_grid):
            raw_confs = [
                compute_calibrated_confidence(
                    ctx_probs_by_temp=tr["ctx_probs_by_temp"],
                    max_ev_rel=tr["max_ev_rel"],
                    mean_ev_rel=tr["mean_ev_rel"],
                    temperature=T_conf,
                    w_tok=w_tok,
                    is_faithful=tr["is_faithful"],
                    has_unanchored=tr["has_unanchored"],
                )
                for tr in raw_traces
            ]

            full_a, full_b, oof_confs = _cross_validated_platt_confs(raw_confs, accs_f1)
            calib_variants: List[Tuple[float, float, List[float]]] = [(1.0, 0.0, raw_confs)]
            if abs(full_a - 1.0) > 0.02 or abs(full_b) > 0.02:
                calib_variants.append((full_a, full_b, oof_confs))

            for platt_a, platt_b, eval_confs in calib_variants:
                # Scale histogram bins dynamically on small dev slices (N=45 -> 5 bins = 9/bin) to avoid Poisson noise
                dev_bins = min(10, max(5, len(eval_confs) // 8))
                cal_m = compute_detailed_calibration(eval_confs, accs_f1, n_bins=dev_bins)
                ece = cal_m["ece"]
                brier = cal_m["brier_score"]
                ece_thresh = 0.095 if len(eval_confs) >= 80 else 0.145

                for tau_abs, tau_conf in product(abstain_rel_grid, conf_thresh_grid):
                    abstained: List[bool] = []
                    ems_eff: List[float] = []
                    f1s_eff: List[float] = []

                    for tr, conf in zip(raw_traces, eval_confs):
                        is_abs = (tr["max_ev_rel"] < tau_abs) or (conf < tau_conf)
                        abstained.append(is_abs)
                        ems_eff.append(0.0 if is_abs else tr["em_raw"])
                        f1s_eff.append(0.0 if is_abs else tr["f1_raw"])

                    sel_m = compute_selective_prediction_metrics(abstained, accs_f1)
                    cov_pct = sel_m["coverage"] * 100.0
                    sel_acc_pct = sel_m["effective_accuracy"] * 100.0
                    sel_risk_pct = sel_m["selective_risk"] * 100.0
                    abs_pct = (1.0 - sel_m["coverage"]) * 100.0
                    eff_em = float(np.mean(ems_eff)) * 100.0
                    eff_f1 = float(np.mean(f1s_eff)) * 100.0

                    # High-coverage selective validity: preserve >=84% coverage so EM/F1 stays competitive while reducing risk & calibration error
                    is_valid = (
                        (decode_f1 >= m2_f1 - 0.8)
                        and (ece <= ece_thresh)
                        and (84.0 <= cov_pct <= 98.5)
                        and (eff_f1 >= decode_f1 - 3.5)
                        and (sel_risk_pct < base_risk)
                    )
                    # Near-full coverage calibrated operating point (>=97.5% coverage)
                    is_valid_full_cov = (decode_f1 >= m2_f1 - 0.8) and (ece <= ece_thresh) and (cov_pct >= 97.5)

                    risk_reduction = base_risk - sel_risk_pct
                    balanced_score = (
                        eff_f1
                        + 0.45 * eff_em
                        + 0.15 * decode_f1
                        - 90.0 * brier
                        - 55.0 * ece
                        + 0.35 * max(0.0, risk_reduction)
                        - 45.0 * max(0.0, 0.85 - cov_pct / 100.0)
                    )

                    all_evaluated_configs.append({
                        "params": {
                            "m3_beta": beta,
                            "m3_contrastive_conf_guard": guard,
                            "m3_confidence_temperature": T_conf,
                            "m3_confidence_tok_weight": w_tok,
                            "m3_confidence_platt_a": platt_a,
                            "m3_confidence_platt_b": platt_b,
                            "m3_abstain_threshold": tau_abs,
                            "m3_confidence_threshold": tau_conf,
                            "m3_protect_eos": True,
                        },
                        "raw_em": round(decode_em, 2),
                        "raw_f1": round(decode_f1, 2),
                        "effective_em": round(eff_em, 2),
                        "effective_f1": round(eff_f1, 2),
                        "ece": round(ece, 4),
                        "mce": round(cal_m["mce"], 4),
                        "brier_score": round(cal_m["brier_score"], 4),
                        "mean_confidence": round(float(np.mean(eval_confs)), 4),
                        "coverage_pct": round(cov_pct, 2),
                        "abstention_rate_pct": round(abs_pct, 2),
                        "selective_accuracy_pct": round(sel_acc_pct, 2),
                        "selective_risk_pct": round(sel_risk_pct, 2),
                        "base_risk_pct": round(base_risk, 2),
                        "risk_reduction_pct": round(risk_reduction, 2),
                        "is_valid_selective": bool(is_valid),
                        "is_valid_full_cov": bool(is_valid_full_cov),
                        "balanced_score": round(balanced_score, 3),
                    })

    valid_selective = [c for c in all_evaluated_configs if c["is_valid_selective"]]
    valid_selective.sort(key=lambda c: c["balanced_score"], reverse=True)

    valid_full_cov = [c for c in all_evaluated_configs if c["is_valid_full_cov"]]
    valid_full_cov.sort(key=lambda c: (c["effective_f1"] - 100.0 * c["ece"], c["coverage_pct"]), reverse=True)

    all_evaluated_configs.sort(key=lambda c: c["balanced_score"], reverse=True)

    # Safe high-coverage fallbacks so non-empty dev/dry-run runs never default to <85% coverage
    high_cov_fallback = sorted(
        [c for c in all_evaluated_configs if c["coverage_pct"] >= 82.0] or all_evaluated_configs,
        key=lambda c: c["balanced_score"],
        reverse=True,
    )
    full_cov_fallback = sorted(
        [c for c in all_evaluated_configs if c["coverage_pct"] >= 97.0] or all_evaluated_configs,
        key=lambda c: (-c["coverage_pct"], c["ece"], -c["effective_f1"]),
    )

    # Deduplicate top diverse valid configs so user gets multiple distinct operating points
    def _diverse_top(pool: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        chosen: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Any, ...]] = set()
        for c in pool:
            p = c["params"]
            sig = (
                p["m3_beta"],
                p["m3_confidence_temperature"],
                p["m3_confidence_tok_weight"],
                p.get("m3_confidence_platt_a", 1.0),
                p["m3_confidence_threshold"],
            )
            if sig not in seen_signatures:
                chosen.append(c)
                seen_signatures.add(sig)
                if len(chosen) >= k:
                    break
        return chosen

    top_valid_selective = _diverse_top(valid_selective if valid_selective else high_cov_fallback, k=10)
    top_valid_full_cov = _diverse_top(valid_full_cov if valid_full_cov else full_cov_fallback, k=5)

    logger.info(
        f"[Stage 3 Done] Tested {len(all_evaluated_configs)} M3 combinations across {len(decode_grid)} beta passes | "
        f"Valid Selective Configs: {len(valid_selective)} | Valid Full-Cov Configs: {len(valid_full_cov)}"
    )
    return {
        "decode_pass_summaries": decode_pass_summaries,
        "total_combinations_tested": len(all_evaluated_configs),
        "valid_selective_count": len(valid_selective),
        "valid_full_cov_count": len(valid_full_cov),
        "top_valid_selective_configs": top_valid_selective,
        "top_valid_full_cov_configs": top_valid_full_cov,
        "best_balanced_params": (top_valid_selective[0]["params"] if top_valid_selective else all_evaluated_configs[0]["params"]),
    }


def write_grid_search_markdown(report: Dict[str, Any], out_md: Path) -> None:
    s1 = report["stage1_m1"]
    s2 = report["stage2_m2"]
    s3 = report["stage3_m3"]
    rec = report["recommended_parameter_sets"]

    lines = [
        "# U-GraphRAG 单次多参数网格搜索验证报告 (Grid Search Report)",
        f"- **生成时间**: {report['metadata']['timestamp']}",
        f"- **硬件设备**: {report['metadata']['device']}",
        f"- **测试模式**: {'Dry-Run 逻辑验证' if report['metadata']['dry_run'] else 'Real GPU 量化推理'}",
        f"- **调参独立切片**: HotpotQA `[{report['metadata']['offset']}:{report['metadata']['offset'] + report['metadata']['num_samples']}]` (与主测试集隔离)",
        "",
        "## 1. Stage 1: M1 不确定度早退阈值 $\\tau_{\\mathrm{EU}}$ 有效参数组",
        f"基准 B0: `EM={s1['b0_baseline']['em']:.2f}%`, `F1={s1['b0_baseline']['f1']:.2f}%`, `Sp-F1={s1['b0_baseline']['sp_f1']:.2f}%`",
        "",
        "| $\\tau_{\\mathrm{EU}}$ | Early Exit (%) | EM (%) | F1 (%) | $\\Delta$F1 vs B0 | Sp-F1 (%) | Token Saving (%) | 有效性 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in s1["all_grid"][:10]:
        tag = "VALID" if r["is_valid"] else "-"
        lines.append(
            f"| {r['tau_eu']:.4f} | {r['early_exit_rate_pct']:.1f}% | {r['em']:.2f}% | {r['f1']:.2f}% | "
            f"{r['delta_f1_vs_b0']:+.2f}% | {r['sp_f1']:.2f}% | {r['token_saving_pct']:.1f}% | {tag} |"
        )

    lines += [
        "",
        "## 2. Stage 2: M2 组合优化与候选池权重有效参数组 (Pareto Top)",
        "| pool_k | offset | $(\\gamma_1, \\gamma_2, \\lambda_{div})$ | EM (%) | F1 (%) | Sp-F1 (%) | Token Saving (%) | Exact vs Greedy Diff (%) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in s2["evaluated_llm_configs"]:
        p = r["params"]
        lines.append(
            f"| {p['top_k_candidate_pool']} | {p['m2_reliability_offset']:.2f} | "
            f"({p['m2_gamma_1']}, {p['m2_gamma_2']}, {p['m2_lambda_div']}) | "
            f"{r['em']:.2f}% | {r['f1']:.2f}% | {r['sp_f1']:.2f}% | {r['token_saving_pct']:.1f}% | {r['exact_vs_greedy_diff_pct']:.1f}% |"
        )

    lines += [
        "",
        "## 3. Stage 3: M3 对比解码与温度校准/门控有效参数组",
        f"共扫描解码配置 {len(s3['decode_pass_summaries'])} 组 x 后校准/拒答门控组合 = **{s3['total_combinations_tested']}** 组参数。",
        "",
        "| $\\beta$ | $T_{\\text{conf}}$ | $w_{\\text{tok}}$ | Platt $(a, b)$ | $(\\tau_{\\text{abs}}, \\tau_{\\text{conf}})$ | Raw / Eff EM/F1 (%) | ECE | Brier | Coverage (%) | Selective Risk (%) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in s3["top_valid_selective_configs"][:8]:
        p = r["params"]
        lines.append(
            f"| {p['m3_beta']:.2f} | {p['m3_confidence_temperature']:.2f} | {p['m3_confidence_tok_weight']:.2f} | "
            f"({p.get('m3_confidence_platt_a', 1.0):.2f}, {p.get('m3_confidence_platt_b', 0.0):.2f}) | "
            f"({p['m3_abstain_threshold']:.2f}, {p['m3_confidence_threshold']:.2f}) | "
            f"{r['raw_em']:.2f} / {r['effective_f1']:.2f} | {r['ece']:.4f} | {r['brier_score']:.4f} | "
            f"{r['coverage_pct']:.1f}% | {r['selective_risk_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 4. 自动沉淀的多套推荐有效参数组合 (`models/best_grid_params.json`)",
        "```json",
        json.dumps(rec, indent=2, ensure_ascii=False),
        "```",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Written Markdown grid search report: {out_md}")


def run_full_grid_search(
    data_dir: Path,
    models_dir: Path,
    results_dir: Path,
    offset: int = 100,
    limit: int = 80,
    preset: str = "balanced",
    dry_run: bool = False,
    save_best_to: Optional[Path] = None,
) -> Dict[str, Any]:
    """Master single-experiment grid search entrypoint across M1, M2, and M3."""
    t0 = time.time()
    samples = _load_hotpot_slice(data_dir, offset=offset, limit=limit)
    model_7b = models_dir / "Qwen2.5-7B-Instruct"
    dense_path = models_dir / "bge-small-en-v1.5"

    if not dry_run and model_7b.exists():
        generator = QwenNF4Generator(
            model_name_or_path=str(model_7b),
            dry_run=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    else:
        generator = QwenNF4Generator(model_name_or_path=None, dry_run=True, device="cpu")
        device_name = "DryRun-CPU"

    pipeline = UGraphRAGPipeline(
        generator=generator,
        token_budget=512,
        top_k_standard=5,
        top_k_candidate_pool=10,
        m1_use_model_probe=not dry_run,
        dense_model_path=str(dense_path) if dense_path.exists() else None,
    )

    s1_res = run_m1_grid_search(pipeline, samples)
    s2_res = run_m2_grid_search(pipeline, samples, preset=preset)

    best_m2_row = s2_res["evaluated_llm_configs"][0]
    s3_res = run_m3_grid_search(
        pipeline=pipeline,
        precomputed=s2_res["_precomputed"],
        ordered_docs_per_sample=s2_res["_best_ordered_docs"],
        m2_baseline_metrics={"em": best_m2_row["em"], "f1": best_m2_row["f1"]},
        preset=preset,
    )

    # Compile multiple valid parameter presets (Max-Balanced, High-Precision/Low-ECE, Conservative-FullCov)
    best_tau = (
        s1_res["valid_configs"][0]["tau_eu"]
        if s1_res["valid_configs"]
        else s1_res["all_grid"][0]["tau_eu"]
    )
    alt_tau = (
        s1_res["valid_configs"][min(1, len(s1_res["valid_configs"]) - 1)]["tau_eu"]
        if s1_res["valid_configs"]
        else best_tau
    )

    best_m2_params = s2_res["best_params"]
    alt_m2_params = (
        s2_res["evaluated_llm_configs"][min(1, len(s2_res["evaluated_llm_configs"]) - 1)]["params"]
    )

    m3_balanced = s3_res["best_balanced_params"]
    m3_low_ece = sorted(
        s3_res["top_valid_selective_configs"],
        key=lambda c: (c["ece"], -c["raw_f1"]),
    )[0]["params"]
    m3_full_cov = s3_res["top_valid_full_cov_configs"][0]["params"]

    recommended_sets = {
        "balanced_optimal": {
            "m1_tau_eu": best_tau,
            **best_m2_params,
            **m3_balanced,
        },
        "low_ece_high_precision": {
            "m1_tau_eu": alt_tau,
            **best_m2_params,
            **m3_low_ece,
        },
        "full_coverage_calibrated": {
            "m1_tau_eu": best_tau,
            **alt_m2_params,
            **m3_full_cov,
        },
    }

    full_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device_name,
            "dry_run": bool(dry_run or not model_7b.exists()),
            "offset": offset,
            "num_samples": len(samples),
            "preset": preset,
            "wall_time_sec": round(time.time() - t0, 2),
        },
        "stage1_m1": s1_res,
        "stage2_m2": {
            "evaluated_llm_configs": s2_res["evaluated_llm_configs"],
            "top20_cpu_structural_grid": s2_res["top20_cpu_structural_grid"],
            "best_params": s2_res["best_params"],
        },
        "stage3_m3": s3_res,
        "recommended_parameter_sets": recommended_sets,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "grid_search_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved full grid search results JSON: {out_json}")

    write_grid_search_markdown(full_report, results_dir / "GRID_SEARCH_REPORT.md")

    if save_best_to is not None:
        save_best_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_best_to, "w", encoding="utf-8") as f:
            json.dump(recommended_sets, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved top multi-objective parameter presets to: {save_best_to}")

    del pipeline, generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Single-Pass Multi-Parameter Grid Search for U-GraphRAG")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--offset", type=int, default=100, help="Start index in hotpotqa_dev_300.json (default 100 avoids [0:100] test leak)")
    parser.add_argument("--limit", type=int, default=80, help="Number of validation samples for grid search")
    parser.add_argument("--preset", type=str, default="balanced", choices=["balanced", "exhaustive"])
    parser.add_argument("--dry-run", action="store_true", help="Run dry-run logic check without GPU model weights")
    parser.add_argument("--save-best-to", type=str, default="models/best_grid_params.json")
    args = parser.parse_args()

    run_full_grid_search(
        data_dir=PROJECT_ROOT / args.data_dir,
        models_dir=PROJECT_ROOT / args.models_dir,
        results_dir=PROJECT_ROOT / args.results_dir,
        offset=args.offset,
        limit=args.limit,
        preset=args.preset,
        dry_run=args.dry_run,
        save_best_to=PROJECT_ROOT / args.save_best_to if args.save_best_to else None,
    )


if __name__ == "__main__":
    main()
