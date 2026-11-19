"""
p0_1_m2_conflict_and_scale.py
=============================
MileStone 2 P0-1 & P0-4 Closure Experiment:
  1. Systematic lambda_conflict in {1.0, 2.5, 5.0, 10.0} sweep on Dev (N=80 each for
     HotpotQA & 2WikiMultiHopQA) over unified pools (pool_k=8, max_k=5, min_k=2),
     tracking Gold Evidence Recall, Conflict Retention, Answer EM/F1, Supporting Fact Recall,
     and Evidence Redundancy on BOTH Clean and Conflict-Injected pools.
  2. Deterministic Dev Pareto operating point selection (evaluating whether any lambda_c > 1.0
     suppresses conflicts without materially damaging Gold Evidence Recall), plus mechanistic
     overlap-conditioned JS conflict diagnostic comparing disjoint bridge vs overlapping clone divergence.
  3. Full unseen Test verification (HotpotQA N=320 >= 300; 2WikiMultiHopQA N=220 >= 200)
     with exact greedy-generation deduplication, Paired Bootstrap 95% CIs, and
     Holm-Bonferroni / Benjamini-Hochberg FDR multiple-comparison correction.

Writes strictly to `results/MileStone2/P0_Closure_Fixes/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import load_or_build_cache
from scripts.phase2.common import (
    WP_DIRS,
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    build_interaction_matrix_for_docs,
    build_prompt,
    compute_next_token_distribution,
    compute_token_count,
    dict_docs_to_evidence_candidates,
    evaluate_qa_prediction,
    extract_entities,
    get_vram_snapshot_mb,
    load_json_file,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    sequential_load_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from scripts.phase2.wp2_m2_ablation import (
    _candidates_to_doc_dicts,
    _compute_mean_pairwise_redundancy,
    _select_mmr,
    _summarize_variant,
)
from src.m2_evidence_optimizer import TwoStageM2Optimizer
from src.u_graphrag_pipeline import compute_supporting_facts_metrics

logger = setup_phase2_logger("phase2.p0_1")

LAMBDA_CONFLICT_GRID: Tuple[float, ...] = (1.0, 2.5, 5.0, 10.0)


def _make_conflict_injected_pool(
    item: Dict[str, Any],
    pool_k: int,
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Construct uniform conflict-injected candidate pool placing conflict at rank 2 (0.98 * top1)."""
    conf_doc = item.get("synthetic_conflict_doc")
    base_pool = [dict(d) for d in item.get("pool", [])[: max(1, pool_k - 1)]]
    if not conf_doc or not base_pool:
        return None
    c_copy = dict(conf_doc)
    c_copy["score"] = round(float(base_pool[0].get("score", 0.9)) * 0.98, 4)
    inj_pool = [base_pool[0], c_copy] + base_pool[1:]
    return inj_pool, str(c_copy["doc_id"])


def _select_p01_variant(
    variant_name: str,
    question: str,
    pool_docs: Sequence[Dict[str, Any]],
    per_doc_dists: Dict[str, Dict[str, float]],
    params: Dict[str, Any],
    max_k: int,
    min_k: int,
    lam_conflict_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Select ordered documents for one P0-1 variant over an identical candidate pool."""
    raw_map = {str(d["doc_id"]): d for d in pool_docs}
    q_ents = extract_entities(question)
    g1 = float(params.get("m2_gamma_1", 0.8))
    g2 = float(params.get("m2_gamma_2", 0.2))
    lam_div = float(params.get("m2_lambda_div", 0.4))
    rel_off = float(params.get("m2_reliability_offset", 0.22))

    if variant_name == "A0_top5":
        return [dict(d) for d in pool_docs[: min(max_k, len(pool_docs))]]
    if variant_name == "A5_mmr_baseline":
        return _select_mmr(pool_docs, k_target=min(max_k, len(pool_docs)))

    cands = dict_docs_to_evidence_candidates(
        pool_docs, reliability_offset=rel_off, gamma_1=g1, gamma_2=g2, calibrate=True
    )

    if variant_name == "G_entity_cooccur":
        imat, active = build_interaction_matrix_for_docs(
            cands, per_doc_dists, use_minhash_redundancy=False, use_js_conflict=False, graph_mode="entity_cooccur"
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 0.0, selection_algorithm="exact_enum")
        res = opt.optimize(active, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    if variant_name == "P_hard_conflict_prune":
        imat, active = build_interaction_matrix_for_docs(
            cands, per_doc_dists, use_minhash_redundancy=True, use_js_conflict=True, graph_mode="entity_js", conflict_prune_threshold=0.05
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 1.0, selection_algorithm="exact_enum")
        res = opt.optimize(
            active,
            interaction_matrix=imat,
            max_k=min(max_k, len(active)),
            min_k=min(min_k, len(active)),
            query_entities=q_ents,
            calibrate=False,
        )
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    if variant_name.startswith("M2_orig_lc"):
        lam_c = lam_conflict_override if lam_conflict_override is not None else float(variant_name.replace("M2_orig_lc", ""))
        imat, active = build_interaction_matrix_for_docs(
            cands, per_doc_dists, use_minhash_redundancy=True, use_js_conflict=True, graph_mode="entity_js"
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, lam_c, selection_algorithm="exact_enum")
        res = opt.optimize(active, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    if variant_name == "D_overlap_gated_js_lc5":
        imat, active = build_interaction_matrix_for_docs(
            cands,
            per_doc_dists,
            use_minhash_redundancy=True,
            use_js_conflict=True,
            graph_mode="entity_js",
            overlap_conflict_shingle_gate=0.20,
        )
        opt = TwoStageM2Optimizer(g1, g2, lam_div, 5.0, selection_algorithm="exact_enum")
        res = opt.optimize(active, interaction_matrix=imat, max_k=max_k, min_k=min_k, query_entities=q_ents, calibrate=False)
        return _candidates_to_doc_dicts(res.ordered_docs, raw_map)

    raise ValueError(f"Unknown P0-1 variant: {variant_name}")


def select_dev_pareto_lambda(
    dev_clean_summary: Dict[str, Dict[str, Any]],
    dev_conf_summary: Dict[str, Dict[str, Any]],
    gold_recall_tolerance_pp: float = 5.0,
    min_conflict_drop_pp: float = 5.0,
) -> Dict[str, Any]:
    """
    Deterministically select Pareto lambda_conflict on Dev from {1.0, 2.5, 5.0, 10.0}.
    A candidate lambda_c > 1.0 passes the Pareto conflict-suppression gate only if:
      - Conflict Retention decreases by >= min_conflict_drop_pp vs lambda_c=1.0, AND
      - Neither Clean nor Conflict Gold Evidence Recall drops by > gold_recall_tolerance_pp.
    Returns selected lambda_c plus explicit boolean diagnostic `pareto_gate_passed`.
    """
    base_key = "M2_orig_lc1.0"
    base_clean_gr = float(dev_clean_summary[base_key]["sp_recall"])
    base_conf_gr = float(dev_conf_summary[base_key]["sp_recall"])
    base_conf_ret = float(dev_conf_summary[base_key]["conflict_retention_pct"])

    feasible: List[Tuple[float, float]] = []
    rows: List[Dict[str, Any]] = []
    for lam in LAMBDA_CONFLICT_GRID:
        vkey = f"M2_orig_lc{lam:.1f}"
        c_sum = dev_clean_summary[vkey]
        x_sum = dev_conf_summary[vkey]
        clean_gr_drop = base_clean_gr - float(c_sum["sp_recall"])
        conf_gr_drop = base_conf_gr - float(x_sum["sp_recall"])
        conf_ret_drop = base_conf_ret - float(x_sum["conflict_retention_pct"])
        avg_f1 = 0.5 * (float(c_sum["f1"]) + float(x_sum["f1"]))

        passes = (
            lam > 1.0
            and conf_ret_drop >= min_conflict_drop_pp
            and clean_gr_drop <= gold_recall_tolerance_pp
            and conf_gr_drop <= gold_recall_tolerance_pp
        )
        if passes:
            feasible.append((avg_f1, lam))
        rows.append({
            "lambda_conflict": lam,
            "clean_f1_pct": c_sum["f1"],
            "clean_gold_recall_pct": c_sum["sp_recall"],
            "conflict_f1_pct": x_sum["f1"],
            "conflict_gold_recall_pct": x_sum["sp_recall"],
            "conflict_retention_pct": x_sum["conflict_retention_pct"],
            "clean_gold_drop_pp": round(clean_gr_drop, 2),
            "conflict_gold_drop_pp": round(conf_gr_drop, 2),
            "conflict_retention_drop_pp": round(conf_ret_drop, 2),
            "passes_pareto_gate": bool(passes),
        })

    if feasible:
        feasible.sort(key=lambda x: (-x[0], x[1]))
        chosen = feasible[0][1]
        return {
            "selected_lambda_conflict": chosen,
            "pareto_gate_passed": True,
            "reason": f"lambda={chosen} suppressed conflict without excessive Gold Recall loss",
            "dev_frontier": rows,
        }

    # Fallback: pick lambda in grid with highest mean Clean+Conflict F1 (typically 1.0 if higher lambdas break recall)
    best_by_f1 = max(
        LAMBDA_CONFLICT_GRID,
        key=lambda l: 0.5 * (
            float(dev_clean_summary[f"M2_orig_lc{l:.1f}"]["f1"])
            + float(dev_conf_summary[f"M2_orig_lc{l:.1f}"]["f1"])
        ),
    )
    return {
        "selected_lambda_conflict": float(best_by_f1),
        "pareto_gate_passed": False,
        "reason": (
            "No lambda_conflict in {2.5, 5.0, 10.0} reduced Conflict Retention by >=5pp "
            "without dropping Gold Evidence Recall by >5pp; unmodified raw_js penalizes disjoint multi-hop bridges."
        ),
        "dev_frontier": rows,
    }


def _run_pool_evaluation(
    dataset_name: str,
    split_tag: str,
    pool_mode: str,
    items: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    generator: Any,
    params: Dict[str, Any],
    pool_k: int,
    max_k: int,
    min_k: int,
    ckpt: Phase2CheckpointManager,
    warm_wp2_ckpt: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], List[int]]:
    """
    Evaluate all variants on a dataset slice under either 'clean' or 'conflict' pool_mode.
    Deduplicates greedy LLM calls cleanly when multiple variants pick the identical ordered doc sequence.
    """
    records_by_var: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    pool_tokens: List[int] = []
    warm = warm_wp2_ckpt or {}
    warm_prefix = "diag_dists" if pool_mode == "conflict" else "dists"

    for item in items:
        iid = str(item["id"])
        if pool_mode == "conflict":
            inj = _make_conflict_injected_pool(item, pool_k=pool_k)
            if inj is None:
                continue
            pool, conf_id = inj
        else:
            pool = [dict(d) for d in item.get("pool", [])[:pool_k]]
            conf_id = ""
        if not pool:
            continue

        pool_tokens.append(sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in pool))

        dist_key = f"dists::{dataset_name}::{pool_mode}::{iid}"
        warm_key = f"{warm_prefix}::{iid}"
        if ckpt.has(dist_key):
            per_doc_dists = ckpt.get(dist_key)
        elif warm_key in warm and all(str(d["doc_id"]) in warm[warm_key] for d in pool):
            per_doc_dists = dict(warm[warm_key])
            ckpt.record(dist_key, per_doc_dists)
        else:
            per_doc_dists = {
                str(d["doc_id"]): compute_next_token_distribution(
                    generator, item["question"], [d["text"]], variant_seed=str(d["doc_id"])
                )["dist"]
                for d in pool
            }
            ckpt.record(dist_key, per_doc_dists)

        gold_titles = set(item.get("gold_titles", []))
        golds = item["possible_answers"]

        for vname in variants:
            rkey = f"res::{dataset_name}::{split_tag}::{pool_mode}::{vname}::{iid}"
            if ckpt.has(rkey):
                records_by_var[vname].append(ckpt.get(rkey))
                continue

            selected = _select_p01_variant(
                vname, item["question"], pool, per_doc_dists, params, max_k=max_k, min_k=min_k
            )
            doc_sig = "|".join(str(d["doc_id"]) for d in selected)
            gen_key = f"gen::{dataset_name}::{pool_mode}::{iid}::{doc_sig}"
            if ckpt.has(gen_key):
                qa_cached = ckpt.get(gen_key)
                qa = qa_cached["qa"]
            else:
                prompt = build_prompt(item["question"], selected, generator=generator)
                pred, _ = generator.generate(prompt, selected, max_new_tokens=48)
                qa = evaluate_qa_prediction(pred, golds)
                ckpt.record(gen_key, {"pred": pred, "qa": qa})

            sel_titles = [str(d.get("title", d.get("doc_id", ""))) for d in selected]
            sp = compute_supporting_facts_metrics(sel_titles, list(gold_titles))
            retained_conf = int(any(str(d.get("doc_id")) == conf_id or d.get("is_synthetic_conflict") for d in selected))
            rec = {
                "id": iid,
                "em": qa["em"],
                "f1": qa["f1"],
                "sp_precision": sp["sp_precision"],
                "sp_recall": sp["sp_recall"],
                "gold_evidence_recall": sp["sp_recall"],
                "sp_f1": sp["sp_f1"],
                "selected_k": len(selected),
                "redundancy_rate": _compute_mean_pairwise_redundancy(selected),
                "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in selected),
                "conflict_retained": retained_conf,
            }
            ckpt.record(rkey, rec)
            records_by_var[vname].append(rec)

    ckpt.flush(force=True)
    summary = {v: _summarize_variant(recs, pool_tokens) for v, recs in records_by_var.items()}
    return records_by_var, summary, pool_tokens


def _build_p01_sig_tests(
    recs_map: Dict[str, List[Dict[str, Any]]],
    pareto_var: str,
) -> Dict[str, Any]:
    """Build paired statistical comparisons across F1, Gold Recall, and Conflict Retention with Holm/BH adjustment."""
    pairs = [
        ("M2_orig_lc2.5", "M2_orig_lc1.0"),
        ("M2_orig_lc5.0", "M2_orig_lc1.0"),
        ("M2_orig_lc10.0", "M2_orig_lc1.0"),
        (pareto_var, "A5_mmr_baseline"),
        (pareto_var, "A0_top5"),
        ("G_entity_cooccur", "A5_mmr_baseline"),
        ("M2_orig_lc1.0", "P_hard_conflict_prune"),
        ("D_overlap_gated_js_lc5", "M2_orig_lc1.0"),
    ]
    f1_tests: Dict[str, Any] = {}
    gr_tests: Dict[str, Any] = {}
    cr_tests: Dict[str, Any] = {}
    for a_name, b_name in pairs:
        if a_name == b_name or a_name not in recs_map or b_name not in recs_map:
            continue
        ra, rb = recs_map[a_name], recs_map[b_name]
        if len(ra) != len(rb) or not ra:
            continue
        key = f"{a_name}_vs_{b_name}"
        f1_tests[key] = paired_bootstrap_with_tests([r["f1"] * 100.0 for r in ra], [r["f1"] * 100.0 for r in rb])
        gr_tests[key] = paired_bootstrap_with_tests([r["sp_recall"] * 100.0 for r in ra], [r["sp_recall"] * 100.0 for r in rb])
        cr_tests[key] = paired_bootstrap_with_tests([r["conflict_retained"] * 100.0 for r in ra], [r["conflict_retained"] * 100.0 for r in rb])

    return {
        "answer_f1_pct": apply_multiple_comparison_correction(f1_tests, p_field="paired_t_p_value"),
        "gold_recall_pct": apply_multiple_comparison_correction(gr_tests, p_field="paired_t_p_value"),
        "conflict_retention_pct": apply_multiple_comparison_correction(cr_tests, p_field="paired_t_p_value"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-1: M2 lambda_conflict Sweep + Expanded Test Evaluation.")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihop"], choices=["hotpotqa", "2wikimultihop"])
    parser.add_argument("--model", type=str, default="qwen3b")
    parser.add_argument("--pool-k", type=int, default=8)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--dev-limit", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=0, help="0 = ALL unseen test items (320 HotpotQA, 220 2Wiki)")
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["P0"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.p0_1", out_dir / "p0_1_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    min_k = int(params.get("m2_min_docs", 2))

    dev_variants = [
        "A0_top5",
        "A5_mmr_baseline",
        "G_entity_cooccur",
        "M2_orig_lc1.0",
        "M2_orig_lc2.5",
        "M2_orig_lc5.0",
        "M2_orig_lc10.0",
        "P_hard_conflict_prune",
        "D_overlap_gated_js_lc5",
    ]

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_p0_1_{args.model}.json", flush_every=25)
    reset_cuda_peak()
    generator = sequential_load_generator(args.model, dry_run=args.dry_run, max_new_tokens=48)
    t0 = time.time()

    results_by_ds: Dict[str, Any] = {}
    try:
        for ds in args.datasets:
            cache = load_or_build_cache(ds, pool_k=16, allow_synthetic_fallback=args.dry_run)
            all_items = list(cache.get("items", []))
            dev_items = [it for it in all_items if it.get("split") == "dev"]
            test_items = [it for it in all_items if it.get("split") == "test"]
            if args.dry_run:
                dev_items = dev_items[:8] or all_items[:8]
                test_items = test_items[:12] or all_items[8:20]
            else:
                dev_items = dev_items[: args.dev_limit] if args.dev_limit > 0 else dev_items
                test_items = test_items[: args.test_limit] if args.test_limit > 0 else test_items

            # Read-only warm per-doc dists from WP2 checkpoint when running real GPU
            warm_ckpt_path = WP_DIRS["WP2"] / f"ckpt_wp2_{ds}_test_{args.model}.json"
            warm_records = (
                load_json_file(warm_ckpt_path).get("records", {})
                if (not args.dry_run and warm_ckpt_path.exists())
                else {}
            )

            logger.info(f"[{ds}] Running Dev Sweep (N={len(dev_items)}) on Clean & Conflict pools...")
            _, dev_clean_sum, _ = _run_pool_evaluation(
                ds, "dev", "clean", dev_items, dev_variants, generator, params, args.pool_k, args.max_k, min_k, ckpt
            )
            _, dev_conf_sum, _ = _run_pool_evaluation(
                ds, "dev", "conflict", dev_items, dev_variants, generator, params, args.pool_k, args.max_k, min_k, ckpt
            )

            pareto_info = select_dev_pareto_lambda(dev_clean_sum, dev_conf_sum)
            sel_lam = float(pareto_info["selected_lambda_conflict"])
            pareto_var = f"M2_orig_lc{sel_lam:.1f}"
            logger.info(f"[{ds}] Dev Pareto result: {pareto_info['selected_lambda_conflict']} (passed={pareto_info['pareto_gate_passed']})")

            # Full Unseen Test Evaluation (evaluate full lambda grid + baselines on N=320/220 cleanly via dedup)
            test_variants = list(dev_variants)
            logger.info(f"[{ds}] Running Unseen Test Evaluation (N={len(test_items)}) on Clean & Conflict pools...")
            test_clean_recs, test_clean_sum, _ = _run_pool_evaluation(
                ds, "test", "clean", test_items, test_variants, generator, params, args.pool_k, args.max_k, min_k, ckpt, warm_wp2_ckpt=warm_records
            )
            test_conf_recs, test_conf_sum, _ = _run_pool_evaluation(
                ds, "test", "conflict", test_items, test_variants, generator, params, args.pool_k, args.max_k, min_k, ckpt, warm_wp2_ckpt=warm_records
            )

            results_by_ds[ds] = {
                "n_dev": len(dev_items),
                "n_test_unseen": len(test_items),
                "dev_clean_summary": dev_clean_sum,
                "dev_conflict_summary": dev_conf_sum,
                "dev_pareto_selection": pareto_info,
                "test_clean_summary": test_clean_sum,
                "test_conflict_summary": test_conf_sum,
                "test_clean_significance": _build_p01_sig_tests(test_clean_recs, pareto_var),
                "test_conflict_significance": _build_p01_sig_tests(test_conf_recs, pareto_var),
            }

        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "datasets": args.datasets,
                "lambda_conflict_grid": list(LAMBDA_CONFLICT_GRID),
                "pool_k": args.pool_k,
                "max_k": args.max_k,
                "min_k": min_k,
                "dry_run": args.dry_run,
                "elapsed_sec": round(time.time() - t0, 2),
                "vram_peak_mb": get_vram_snapshot_mb(),
            },
            "v5_integrity": v5_info,
            "datasets": results_by_ds,
        }
        out_file = out_dir / "p0_1_m2_conflict_and_scale.json"
        atomic_write_json(out_file, payload)
        logger.info(f"Saved P0-1 closure results -> {out_file}")
    finally:
        teardown_generator(generator)


if __name__ == "__main__":
    main()
