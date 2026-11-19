"""
wp2_m2_ablation.py
==================
Phase II Work Package 2 (WP2): Complete M2 Combinatorial Selection & MST Organization
Mechanism Ablation Suite on Frozen Identical Top-K Candidate Pools.

Compares:
  - A0_top3 / A0_top5: Rank-ordered top-K baselines
  - A1_exact_no_graph: Exact selection without pairwise interaction graph
  - A2_exact_minhash_no_js: Exact + MinHash redundancy + entity bridge (no JS conflict)
  - A3_exact_full_orig_order: Exact + MinHash + JS conflict (retrieval rank order)
  - A4_full_m2_mst_dfs: Full M2 Exact + MST-DFS organization
  - A5_mmr_baseline: Classic Maximal Marginal Relevance greedy baseline
  - A6_greedy_m2: Greedy solver over the full M2 objective
  - Graph Controls: G_dense_cosine, G_entity_cooccur, G_random
  - Pruning & Ordering Controls: P_hard_conflict_prune, O_reverse_dfs, O_random_order
Plus Conflict-Injected Diagnostic Slice to measure Conflict Retention Rate directly.

Writes strictly to `results/MileStone2/WP2_M2_Optimization/`.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import load_or_build_cache
from scripts.phase2.common import (
    WP_DIRS,
    Phase2CheckpointManager,
    atomic_write_json,
    build_interaction_matrix_for_docs,
    build_prompt,
    compute_next_token_distribution,
    compute_token_count,
    dict_docs_to_evidence_candidates,
    evaluate_qa_prediction,
    extract_entities,
    get_vram_snapshot_mb,
    jaccard_sim,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    sequential_load_generator,
    setup_phase2_logger,
    teardown_generator,
    text_shingles,
    verify_v5_frozen_checksums,
)
from src.m2_evidence_optimizer import EvidenceCandidate, TwoStageM2Optimizer
from src.u_graphrag_pipeline import compute_supporting_facts_metrics

logger = setup_phase2_logger("phase2.wp2")


def _candidates_to_doc_dicts(
    ordered_candidates: Sequence[EvidenceCandidate],
    raw_doc_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in ordered_candidates:
        base = dict(raw_doc_map.get(c.doc_id, {}))
        base["doc_id"] = c.doc_id
        base["title"] = c.metadata.get("title", base.get("title", c.doc_id))
        base["text"] = c.text
        base["calibrated_reliability"] = round(float(c.calibrated_reliability), 4)
        base["token_count"] = int(c.metadata.get("token_count", compute_token_count(c.text)))
        out.append(base)
    return out


def _select_mmr(
    pool_docs: Sequence[Dict[str, Any]],
    k_target: int,
    lambda_mmr: float = 0.65,
) -> List[Dict[str, Any]]:
    """Standard Maximal Marginal Relevance greedy baseline using 3-shingle Jaccard similarity."""
    if not pool_docs or k_target <= 0:
        return []
    shingles = [text_shingles(str(d.get("text", "")), k=3) for d in pool_docs]
    scores = [float(d.get("score", 0.5)) for d in pool_docs]
    s_min, s_max = min(scores), max(scores)
    span = (s_max - s_min) if (s_max - s_min) > 1e-9 else 1.0
    rel = [(s - s_min) / span for s in scores]

    selected_idx: List[int] = []
    remaining = set(range(len(pool_docs)))
    while remaining and len(selected_idx) < min(k_target, len(pool_docs)):
        best_i = -1
        best_val = -float("inf")
        for i in sorted(remaining):
            max_sim = max((jaccard_sim(shingles[i], shingles[j]) for j in selected_idx), default=0.0)
            val = lambda_mmr * rel[i] - (1.0 - lambda_mmr) * max_sim
            if val > best_val:
                best_val = val
                best_i = i
        if best_i < 0:
            break
        selected_idx.append(best_i)
        remaining.remove(best_i)
    return [dict(pool_docs[i]) for i in selected_idx]


def _compute_mean_pairwise_redundancy(docs: Sequence[Dict[str, Any]]) -> float:
    if len(docs) < 2:
        return 0.0
    sh = [text_shingles(str(d.get("text", "")), k=3) for d in docs]
    vals: List[float] = []
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            vals.append(jaccard_sim(sh[i], sh[j]))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _select_variant_docs(
    variant_name: str,
    item_id: str,
    question: str,
    pool_docs: Sequence[Dict[str, Any]],
    per_doc_dists: Dict[str, Dict[str, float]],
    params: Dict[str, Any],
    max_k: int,
    min_k: int,
) -> List[Dict[str, Any]]:
    """Run one WP2 selection/organization variant cleanly over the same candidate pool."""
    raw_map = {str(d["doc_id"]): d for d in pool_docs}
    q_ents = extract_entities(question)
    g1 = float(params.get("m2_gamma_1", 0.8))
    g2 = float(params.get("m2_gamma_2", 0.2))
    lam_div = float(params.get("m2_lambda_div", 0.4))
    rel_off = float(params.get("m2_reliability_offset", 0.22))

    if variant_name == "A0_top3":
        return [dict(d) for d in pool_docs[: min(3, len(pool_docs))]]
    if variant_name == "A0_top5":
        return [dict(d) for d in pool_docs[: min(max_k, len(pool_docs))]]
    if variant_name == "A5_mmr_baseline":
        return _select_mmr(pool_docs, k_target=min(max_k, len(pool_docs)))

    cands = dict_docs_to_evidence_candidates(
        pool_docs, reliability_offset=rel_off, gamma_1=g1, gamma_2=g2, calibrate=True
    )

    algo = "greedy" if variant_name == "A6_greedy_m2" else "exact_enum"
    recency = variant_name == "O_reverse_dfs"

    if variant_name == "A1_exact_no_graph":
        imat, active_cands = build_interaction_matrix_for_docs(cands, per_doc_dists, graph_mode="no_graph")
        opt = TwoStageM2Optimizer(g1, 0.0, 0.0, 0.0, selection_algorithm="exact_enum", recency_aware=False)
        res = opt.optimize(active_cands, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.selected_docs, raw_map)

    if variant_name == "A2_exact_minhash_no_js":
        imat, active_cands = build_interaction_matrix_for_docs(
            cands, per_doc_dists, use_minhash_redundancy=True, use_js_conflict=False, graph_mode="entity_js"
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 0.0, selection_algorithm="exact_enum", recency_aware=False)
        res = opt.optimize(active_cands, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    if variant_name in ("A3_exact_full_orig_order", "A4_full_m2_mst_dfs", "A6_greedy_m2", "O_reverse_dfs", "O_random_order"):
        imat, active_cands = build_interaction_matrix_for_docs(
            cands, per_doc_dists, use_minhash_redundancy=True, use_js_conflict=True, graph_mode="entity_js"
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 1.0, selection_algorithm=algo, recency_aware=recency)
        res = opt.optimize(active_cands, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        if variant_name == "A3_exact_full_orig_order":
            ordered = sorted(res.selected_docs, key=lambda c: int(c.metadata.get("original_idx", 0)))
            return _candidates_to_doc_dicts(ordered, raw_map)
        if variant_name == "O_random_order":
            ordered = list(res.selected_docs)
            seed = int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16)
            random.Random(seed).shuffle(ordered)
            return _candidates_to_doc_dicts(ordered, raw_map)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    if variant_name == "P_hard_conflict_prune":
        imat, active_cands = build_interaction_matrix_for_docs(
            cands,
            per_doc_dists,
            use_minhash_redundancy=True,
            use_js_conflict=True,
            graph_mode="entity_js",
            conflict_prune_threshold=0.05,
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 1.0, selection_algorithm="exact_enum", recency_aware=False)
        res = opt.optimize(active_cands, interaction_matrix=imat, max_k=min(max_k, len(active_cands)), min_k=min(min_k, len(active_cands)), query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    if variant_name in ("G_dense_cosine", "G_entity_cooccur", "G_random"):
        g_mode = {"G_dense_cosine": "dense_cosine", "G_entity_cooccur": "entity_cooccur", "G_random": "random"}[variant_name]
        seed = int(hashlib.md5(f"{variant_name}|{item_id}".encode("utf-8")).hexdigest()[:8], 16)
        imat, active_cands = build_interaction_matrix_for_docs(
            cands,
            per_doc_dists,
            use_minhash_redundancy=(g_mode != "entity_cooccur"),
            use_js_conflict=(g_mode != "entity_cooccur"),
            graph_mode=g_mode,
            rng=random.Random(seed),
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 1.0 if g_mode != "entity_cooccur" else 0.0, selection_algorithm="exact_enum")
        res = opt.optimize(active_cands, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    raise ValueError(f"Unknown WP2 variant: {variant_name}")


ALL_WP2_VARIANTS: Tuple[str, ...] = (
    "A0_top3",
    "A0_top5",
    "A1_exact_no_graph",
    "A2_exact_minhash_no_js",
    "A3_exact_full_orig_order",
    "A4_full_m2_mst_dfs",
    "A5_mmr_baseline",
    "A6_greedy_m2",
    "G_dense_cosine",
    "G_entity_cooccur",
    "G_random",
    "P_hard_conflict_prune",
    "O_reverse_dfs",
    "O_random_order",
)


def _summarize_variant(records: Sequence[Dict[str, Any]], total_pool_tokens: Sequence[int]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    em = 100.0 * sum(r["em"] for r in records) / n
    f1 = 100.0 * sum(r["f1"] for r in records) / n
    sp_p = 100.0 * sum(r["sp_precision"] for r in records) / n
    sp_r = 100.0 * sum(r["sp_recall"] for r in records) / n
    sp_f1 = 100.0 * sum(r["sp_f1"] for r in records) / n
    red = 100.0 * sum(r["redundancy_rate"] for r in records) / n
    tok_used = sum(r["selected_tokens"] for r in records) / n
    tok_pool = sum(total_pool_tokens) / max(1, len(total_pool_tokens))
    tok_save = max(0.0, (tok_pool - tok_used) / max(1.0, tok_pool)) * 100.0
    conf_ret = 100.0 * sum(r.get("conflict_retained", 0) for r in records) / n
    return {
        "n": n,
        "em": round(em, 2),
        "f1": round(f1, 2),
        "sp_precision": round(sp_p, 2),
        "sp_recall": round(sp_r, 2),
        "sp_f1": round(sp_f1, 2),
        "mean_k": round(sum(r["selected_k"] for r in records) / n, 2),
        "redundancy_rate_pct": round(red, 2),
        "conflict_retention_pct": round(conf_ret, 2),
        "mean_selected_tokens": round(tok_used, 1),
        "token_saving_vs_pool_pct": round(tok_save, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP2: M2 Combinatorial Selection & MST Ablation Suite.")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "2wikimultihop", "musique"])
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--model", type=str, default="qwen3b")
    parser.add_argument("--pool-k", type=int, default=8, help="Shared candidate pool size trimmed from WP0 k16 cache")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--conflict-limit", type=int, default=60, help="Conflict-injected diagnostic slice size")
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["WP2"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.wp2", out_dir / "wp2_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    min_k = int(params.get("m2_min_docs", 2))

    cache = load_or_build_cache(args.dataset, pool_k=16, allow_synthetic_fallback=args.dry_run)
    items = [it for it in cache.get("items", []) if args.split == "all" or it.get("split") == args.split]
    if not items:
        items = list(cache.get("items", []))
    eff_limit = 20 if args.dry_run else args.limit
    items = items[:eff_limit]

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp2_{args.dataset}_{args.split}_{args.model}.json", flush_every=25)
    reset_cuda_peak()
    generator = sequential_load_generator(args.model, dry_run=args.dry_run, max_new_tokens=48)
    t0 = time.time()

    try:
        # 1. Standard Ablation Suite on identical cached pool[:pool_k]
        variant_records: Dict[str, List[Dict[str, Any]]] = {v: [] for v in ALL_WP2_VARIANTS}
        pool_tokens_per_item: List[int] = []

        for item in items:
            iid = str(item["id"])
            pool = list(item.get("pool", []))[: args.pool_k]
            if not pool:
                continue
            pool_tokens_per_item.append(sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in pool))

            # Precompute single-doc distributions once per sample so every variant sees identical M1 signals
            dist_key = f"dists::{iid}"
            if ckpt.has(dist_key):
                per_doc_dists = ckpt.get(dist_key)
            else:
                per_doc_dists = {}
                for d in pool:
                    pr = compute_next_token_distribution(generator, item["question"], [d["text"]], variant_seed=d["doc_id"])
                    per_doc_dists[str(d["doc_id"])] = pr["dist"]
                ckpt.record(dist_key, per_doc_dists)

            gold_titles = set(item.get("gold_titles", []))
            golds = item["possible_answers"]

            for vname in ALL_WP2_VARIANTS:
                rkey = f"std::{vname}::{iid}"
                if ckpt.has(rkey):
                    variant_records[vname].append(ckpt.get(rkey))
                    continue

                selected = _select_variant_docs(
                    vname, iid, item["question"], pool, per_doc_dists, params, max_k=args.max_k, min_k=min_k
                )
                prompt = build_prompt(item["question"], selected, generator=generator)
                pred, _ = generator.generate(prompt, selected, max_new_tokens=48)
                qa = evaluate_qa_prediction(pred, golds)
                sel_titles = [str(d.get("title", d.get("doc_id", ""))) for d in selected]
                sp = compute_supporting_facts_metrics(sel_titles, list(gold_titles))
                rec = {
                    "id": iid,
                    "em": qa["em"],
                    "f1": qa["f1"],
                    "sp_precision": sp["sp_precision"],
                    "sp_recall": sp["sp_recall"],
                    "sp_f1": sp["sp_f1"],
                    "selected_k": len(selected),
                    "redundancy_rate": _compute_mean_pairwise_redundancy(selected),
                    "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in selected),
                    "conflict_retained": 0,
                }
                ckpt.record(rkey, rec)
                variant_records[vname].append(rec)

        ckpt.flush(force=True)

        # 2. Conflict-Injected Diagnostic Slice (tests conflict rejection across A0/A1/A2/A4/A5/P_hard)
        diag_variants = ("A0_top5", "A1_exact_no_graph", "A2_exact_minhash_no_js", "A4_full_m2_mst_dfs", "A5_mmr_baseline", "P_hard_conflict_prune")
        diag_records: Dict[str, List[Dict[str, Any]]] = {v: [] for v in diag_variants}
        conf_items = items[: (15 if args.dry_run else args.conflict_limit)]
        diag_pool_tokens: List[int] = []

        for item in conf_items:
            iid = str(item["id"])
            conf_doc = item.get("synthetic_conflict_doc")
            if not conf_doc:
                continue
            base_pool = list(item.get("pool", []))[: max(1, args.pool_k - 1)]
            # Place conflicting doc at rank 2 (high retrieval rank!) to test whether M2 rejects it
            inj_pool = [base_pool[0], dict(conf_doc)] + base_pool[1:]
            inj_pool[1]["score"] = round(float(base_pool[0].get("score", 0.9)) * 0.98, 4)
            diag_pool_tokens.append(sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in inj_pool))

            dkey = f"diag_dists::{iid}"
            if ckpt.has(dkey):
                per_doc_dists = ckpt.get(dkey)
            else:
                per_doc_dists = {}
                for d in inj_pool:
                    pr = compute_next_token_distribution(generator, item["question"], [d["text"]], variant_seed=d["doc_id"])
                    per_doc_dists[str(d["doc_id"])] = pr["dist"]
                ckpt.record(dkey, per_doc_dists)

            gold_titles = set(item.get("gold_titles", []))
            golds = item["possible_answers"]
            conf_id = str(conf_doc["doc_id"])

            for vname in diag_variants:
                rkey = f"diag::{vname}::{iid}"
                if ckpt.has(rkey):
                    diag_records[vname].append(ckpt.get(rkey))
                    continue

                selected = _select_variant_docs(
                    vname, iid, item["question"], inj_pool, per_doc_dists, params, max_k=args.max_k, min_k=min_k
                )
                prompt = build_prompt(item["question"], selected, generator=generator)
                pred, _ = generator.generate(prompt, selected, max_new_tokens=48)
                qa = evaluate_qa_prediction(pred, golds)
                sel_titles = [str(d.get("title", d.get("doc_id", ""))) for d in selected]
                sp = compute_supporting_facts_metrics(sel_titles, list(gold_titles))
                retained_conf = int(any(str(d.get("doc_id")) == conf_id or d.get("is_synthetic_conflict") for d in selected))
                rec = {
                    "id": iid,
                    "em": qa["em"],
                    "f1": qa["f1"],
                    "sp_precision": sp["sp_precision"],
                    "sp_recall": sp["sp_recall"],
                    "sp_f1": sp["sp_f1"],
                    "selected_k": len(selected),
                    "redundancy_rate": _compute_mean_pairwise_redundancy(selected),
                    "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in selected),
                    "conflict_retained": retained_conf,
                }
                ckpt.record(rkey, rec)
                diag_records[vname].append(rec)

        ckpt.flush(force=True)

        std_summary = {v: _summarize_variant(recs, pool_tokens_per_item) for v, recs in variant_records.items()}
        diag_summary = {v: _summarize_variant(recs, diag_pool_tokens) for v, recs in diag_records.items()}

        # Paired significance tests of Full M2 (A4) against key comparators
        sig_tests: Dict[str, Any] = {}
        a4_f1 = [r["f1"] * 100.0 for r in variant_records["A4_full_m2_mst_dfs"]]
        a4_em = [r["em"] * 100.0 for r in variant_records["A4_full_m2_mst_dfs"]]
        for comp in ("A0_top5", "A1_exact_no_graph", "A2_exact_minhash_no_js", "A3_exact_full_orig_order", "A5_mmr_baseline", "G_random", "P_hard_conflict_prune"):
            c_recs = variant_records[comp]
            if len(c_recs) == len(a4_f1) and len(c_recs) > 0:
                sig_tests[f"A4_vs_{comp}"] = {
                    "f1_pct": paired_bootstrap_with_tests(a4_f1, [r["f1"] * 100.0 for r in c_recs]),
                    "em_pct": paired_bootstrap_with_tests(a4_em, [r["em"] * 100.0 for r in c_recs]),
                }

        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": args.dataset,
                "split": args.split,
                "model": args.model,
                "pool_k": args.pool_k,
                "max_k": args.max_k,
                "min_k": min_k,
                "n_items": len(items),
                "dry_run": args.dry_run,
                "elapsed_sec": round(time.time() - t0, 2),
                "vram_peak_mb": get_vram_snapshot_mb(),
            },
            "v5_integrity": v5_info,
            "standard_ablation_summary": std_summary,
            "conflict_injected_diagnostic_summary": diag_summary,
            "paired_significance_vs_a4": sig_tests,
        }
        out_file = out_dir / f"wp2_m2_ablation_{args.dataset}_{args.split}.json"
        atomic_write_json(out_file, payload)
        logger.info(f"Saved WP2 results -> {out_file}")
    finally:
        teardown_generator(generator)


if __name__ == "__main__":
    main()
