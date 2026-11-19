"""
wp6_cross_family.py
===================
Milestone 3 Phase B (WP6): Cross-Family Generalization & Component Attribution.

Evaluates sequentially across distinct LLM families (`qwen3b`, `qwen7b`, `llama3_8b`,
plus optional `mistral7b`) and all 4 benchmarks (`hotpotqa`, `popqa`, `2wikimultihop`, `musique`)
comparing ALL 9 methods specified in AboutMileStone_3.md Sec 6.3:
  1. B0_StandardRAG (Hybrid Top-3 fixed baseline)
  2. B1_BM25_RAG    (Strong sparse BM25 Top-5 comparator)
  3. B2_Dense_RAG   (Strong dense BGE Top-5 comparator)
  4. B3_MMR_RAG     (Strong diversity MMR Top-5 comparator)
  5. M1_Only        (Adaptive V5 Proxy early-exit/expansion)
  6. M2_Only        (Always expand + M2 subset selection & MST-DFS ordering)
  7. M3_Only        (Fixed Top-3 + Dev-calibrated M3 adaptive decoding & abstention)
  8. M1_M2          (M1 adaptive gating + M2 selection/ordering, uncalibrated decode)
  9. Full_UGraphRAG (M1 + M2 + M3 unified system)

Supports multi-seed execution (`--seeds 42 1024 2026`), per-sample JSONL exports,
automatic failure attribution (`retrieval_miss` / `selection_loss` / `generation_error`),
paired Bootstrap 95% CIs, and Holm-Bonferroni / BH-FDR multiple comparison correction.

Writes strictly to `results/MileStone3/WP6_CrossFamily/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import _load_cpu_dense_model
from scripts.phase2.wp2_m2_ablation import _select_mmr
from scripts.phase2.wp3_m3_calibration import _collect_decode_traces, _fit_dev_calibration_configs, _score_trace_with_cal_params
from scripts.phase3.phase3_common import (
    WP3_DIRS,
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    calibrate_dev_m1_tau_eu,
    classify_failure_case,
    cleanup_cached_model_weights,
    collect_hardware_manifest,
    compute_calibration_and_risk_suite,
    compute_multiseed_aggregate,
    compute_next_token_distribution,
    compute_token_count,
    compute_unified_m1_signal,
    evaluate_qa_prediction,
    get_vram_snapshot_mb,
    is_phase3_backbone_available,
    load_phase3_dataset_cache,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    resolve_doc_sp_labels,
    resolve_phase3_model_list,
    resolve_wp8_recommended_graph_mode,
    run_open_decode_cached,
    select_dev_risk_constrained_threshold,
    select_m2_by_graph_mode,
    sequential_load_phase3_generator,
    setup_phase2_logger,
    should_bypass_m1_early_exit,
    summarize_failure_taxonomy,
    teardown_generator,
    verify_v5_frozen_checksums,
    write_jsonl_records,
)
from src.baseline_rag import BM25Retriever, DenseRetriever
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.u_graphrag_pipeline import compute_supporting_facts_metrics

logger = setup_phase2_logger("phase3.wp6")

ALL_WP6_METHODS: Tuple[str, ...] = (
    "B0_StandardRAG",
    "B1_BM25_RAG",
    "B2_Dense_RAG",
    "B3_MMR_RAG",
    "M1_Only",
    "M2_Only",
    "M3_Only",
    "M1_M2",
    "Full_UGraphRAG",
)


def _select_bm25_or_dense_docs(
    item: Dict[str, Any],
    mode: str,
    pool: Sequence[Dict[str, Any]],
    k: int = 5,
    shared_dense_model: Any = None,
) -> List[Dict[str, Any]]:
    id_key = "bm25_top5_ids" if mode == "bm25" else "dense_top5_ids"
    id_map = {str(d["doc_id"]): d for d in pool}
    cached_ids = item.get(id_key) or []
    chosen = [dict(id_map[did]) for did in cached_ids if did in id_map][:k]
    if chosen:
        return chosen
    if not pool:
        return []
    corpus_docs = [{"id": d["doc_id"], "title": d["title"], "text": d["text"]} for d in pool]
    if mode == "bm25":
        bm_ret = BM25Retriever(corpus_docs)
        hits = bm_ret.retrieve(item["question"], top_k=min(k, len(pool)))
        return [dict(pool[idx]) for idx, _ in hits]
    dense_ret = DenseRetriever(corpus_docs, shared_model=shared_dense_model)
    hits = dense_ret.retrieve(item["question"], top_k=min(k, len(pool)))
    return [dict(pool[idx]) for idx, _ in hits]


def _permute_pool_for_seed(pool: Sequence[Dict[str, Any]], seed: int, item_id: str) -> List[Dict[str, Any]]:
    """Seed=42 preserves canonical ranked order; alternate seeds apply bounded tie-order perturbation."""
    out = [dict(d) for d in pool]
    if seed == 42 or len(out) <= 2:
        return out
    rng = random.Random(int(hashlib.md5(f"{seed}|{item_id}".encode()).hexdigest()[:8], 16))
    # Preserve rank-0..2 relative head with mild tail/tie shuffle to capture rank sensitivity safely
    tail = out[2:]
    rng.shuffle(tail)
    return out[:2] + tail


def evaluate_model_dataset_seed(
    model_key: str,
    dataset_name: str,
    seed: int,
    dev_items: Sequence[Dict[str, Any]],
    test_items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    m2_graph_mode: str,
    out_dir: Path,
    ckpt: Phase2CheckpointManager,
    methods: Sequence[str] = ALL_WP6_METHODS,
    shared_dense_model: Any = None,
    m2_lambda_conflict: Optional[float] = None,
    tau_eu_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Run all 9 WP6 methods on one (model, dataset, seed) slice and save JSONL traces."""
    decoupler = UncertaintyDecoupler()
    tau_eu = float(tau_eu_override if tau_eu_override is not None else params.get("m1_tau_eu", 0.18))
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    rel_off = float(params.get("m2_reliability_offset", 0.22))

    # Fit Dev calibration parameters and adaptive abstention threshold strictly on Dev (once per model/ds)
    dev_cal_key = f"wp6_dev_cal_v2::{dataset_name}"
    if ckpt.has(dev_cal_key):
        cal_bundle = ckpt.get(dev_cal_key)
    else:
        class _DevNs:
            def has(self, k: str) -> bool:
                return ckpt.has(f"wp6_dev_v2::{dataset_name}::{k}")
            def get(self, k: str, d: Any = None) -> Any:
                return ckpt.get(f"wp6_dev_v2::{dataset_name}::{k}", d)
            def record(self, k: str, v: Any) -> None:
                ckpt.record(f"wp6_dev_v2::{dataset_name}::{k}", v)
            def flush(self, force: bool = True) -> None:
                ckpt.flush(force=force)

        dev_b0 = _collect_decode_traces(dev_items, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0", ckpt=_DevNs())
        full_platt = _fit_dev_calibration_configs(dev_b0)["full_platt"]
        thresh_info = select_dev_risk_constrained_threshold(dev_b0, full_platt, abstain_threshold=ab_tau)
        cal_bundle = {"full_platt": full_platt, "adaptive_tau_conf": float(thresh_info["adaptive_tau_conf"])}
        ckpt.record(dev_cal_key, cal_bundle)
        ckpt.flush(force=True)

    platt = cal_bundle["full_platt"]
    adaptive_tau = float(cal_bundle["adaptive_tau_conf"])

    records_by_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    pool_tokens: List[int] = []
    decode_mem: Dict[str, Dict[str, Any]] = {}

    for item in test_items:
        iid = str(item["id"])
        pool_full = _permute_pool_for_seed(item.get("pool", []), seed=seed, item_id=iid)
        # Use full candidate pool (up to 16) consistently across B1/B2/B3/M2/Full to eliminate artificial recall handicaps
        pool = pool_full[:16]
        if not pool:
            continue
        pool_tokens.append(sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in pool))

        gold_titles = list(item.get("gold_titles", []))
        golds = item["possible_answers"]
        pool_sp = compute_supporting_facts_metrics(resolve_doc_sp_labels(pool_full, gold_titles), gold_titles)
        pool_gold_recall = float(pool_sp["sp_recall"])
        d0_docs = pool[: min(3, len(pool))]
        d0_sig = "|".join(str(d.get("doc_id", "")) for d in d0_docs)

        # Shared V5 Proxy M1 routing signal (seed-invariant when d0 docs repeat)
        eu_ckey = f"wp6_eu::{dataset_name}::{iid}::{d0_sig}"
        if ckpt.has(eu_ckey):
            u_eu = float(ckpt.get(eu_ckey)["u_eu"])
        else:
            u_eu = float(compute_unified_m1_signal(generator, item["question"], d0_docs, eu_mode="v5_proxy", decoupler=decoupler)["u_eu"])
            ckpt.record(eu_ckey, {"u_eu": u_eu})
        raw_early_exit = bool(u_eu <= tau_eu)
        conflict_bypass = should_bypass_m1_early_exit(pool[: min(5, len(pool))], m2_graph_mode)
        effective_early_exit = bool(raw_early_exit and not conflict_bypass)

        # Shared per-doc distributions cached lazily per unique (item, doc_id) across seeds
        per_doc_dists: Optional[Dict[str, Dict[str, float]]] = None
        m2_selected_cache: Optional[List[Dict[str, Any]]] = None

        def _get_m2_docs() -> List[Dict[str, Any]]:
            nonlocal per_doc_dists, m2_selected_cache
            if m2_selected_cache is not None:
                return m2_selected_cache
            per_doc_dists = {}
            for d in pool:
                did = str(d["doc_id"])
                doc_dkey = f"wp6_dist::{dataset_name}::{iid}::{did}"
                if ckpt.has(doc_dkey):
                    per_doc_dists[did] = ckpt.get(doc_dkey)
                else:
                    dist_val = compute_next_token_distribution(
                        generator, item["question"], [d["text"]], variant_seed=did
                    )["dist"]
                    per_doc_dists[did] = dist_val
                    ckpt.record(doc_dkey, dist_val)
            m2_selected_cache, _ = select_m2_by_graph_mode(
                item["question"],
                pool,
                per_doc_dists,
                params,
                graph_mode=m2_graph_mode,
                pool_k=len(pool),
                max_k=5,
                lambda_conflict=m2_lambda_conflict,
            )
            return m2_selected_cache

        for method in methods:
            rkey = f"wp6_res_v2::{dataset_name}::{seed}::{method}::{iid}"
            if ckpt.has(rkey):
                records_by_method[method].append(ckpt.get(rkey))
                continue

            # Select evidence and decoder configuration cleanly per method over identical `pool`
            use_m3 = method in ("M3_Only", "Full_UGraphRAG")
            if method == "B0_StandardRAG":
                ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in d0_docs]
            elif method == "B1_BM25_RAG":
                ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in _select_bm25_or_dense_docs(item, "bm25", pool, k=5)]
            elif method == "B2_Dense_RAG":
                ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in _select_bm25_or_dense_docs(item, "dense", pool, k=5, shared_dense_model=shared_dense_model)]
            elif method == "B3_MMR_RAG":
                ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in _select_mmr(pool, k_target=min(5, len(pool)))]
            elif method == "M1_Only":
                chosen = d0_docs if raw_early_exit else pool[: min(5, len(pool))]
                ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in chosen]
            elif method == "M2_Only":
                ev = _get_m2_docs()
            elif method == "M3_Only":
                ev = [{**d, "calibrated_reliability": float(min(1.0, float(d.get("score", 0.5)) + rel_off))} for d in d0_docs]
            elif method in ("M1_M2", "Full_UGraphRAG"):
                if effective_early_exit:
                    ev = [{**d, "calibrated_reliability": float(min(1.0, float(d.get("score", 0.5)) + rel_off))} for d in d0_docs]
                else:
                    ev = _get_m2_docs()
            else:
                raise ValueError(f"Unknown WP6 method: {method}")

            eff_beta = 0.12 if (method == "Full_UGraphRAG" and not effective_early_exit) else 0.0
            trace = run_open_decode_cached(
                generator=generator,
                question=item["question"],
                evidence_docs=ev,
                eff_beta=eff_beta,
                params=params,
                ckpt=ckpt,
                cache_prefix=f"wp6_gen_v2::{dataset_name}",
                memory_cache=decode_mem,
                max_new_tokens=48,
                calib_from_b0=True,
            )
            raw_answer = trace["raw_answer"]
            if use_m3:
                conf = float(
                    _score_trace_with_cal_params(
                        trace,
                        platt["T"],
                        platt["w_tok"],
                        platt["platt_a"],
                        platt["platt_b"],
                    )
                )
                is_ab = (float(trace["max_ev_rel"]) < ab_tau) or (conf < adaptive_tau)
            else:
                conf = float(trace["raw_confidence"])
                is_ab = False

            qa = evaluate_qa_prediction(raw_answer, golds, is_abstained=is_ab)
            sel_titles = resolve_doc_sp_labels(ev, gold_titles)
            sp = compute_supporting_facts_metrics(sel_titles, gold_titles)
            fail_cat = classify_failure_case(qa["f1"], is_ab, pool_gold_recall, float(sp["sp_recall"]))

            rec = {
                "id": iid,
                "question": item["question"],
                "gold_answers": list(golds),
                "prediction": raw_answer,
                "raw_em": qa["em"],
                "raw_f1": qa["f1"],
                "effective_em": qa["effective_em"],
                "effective_f1": qa["effective_f1"],
                "confidence": round(conf, 4),
                "is_abstained": bool(is_ab),
                "u_eu": round(u_eu, 4),
                "tau_eu_used": round(tau_eu, 6),
                "early_exit": bool(raw_early_exit if method == "M1_Only" else (effective_early_exit if method in ("M1_M2", "Full_UGraphRAG") else True)),
                "sp_precision": sp["sp_precision"],
                "sp_recall": sp["sp_recall"],
                "sp_f1": sp["sp_f1"],
                "pool_gold_recall": pool_gold_recall,
                "selected_doc_ids": [str(d["doc_id"]) for d in ev],
                "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in ev),
                "failure_category": fail_cat,
            }
            ckpt.record(rkey, rec)
            records_by_method[method].append(rec)

    ckpt.flush(force=True)

    # Export JSONL traces & summarize
    pred_dir = out_dir / "predictions"
    method_summaries: Dict[str, Any] = {}
    for method, recs in records_by_method.items():
        write_jsonl_records(pred_dir / f"{model_key}_{dataset_name}_{method}_seed{seed}.jsonl", recs)
        n = max(1, len(recs))
        cal = compute_calibration_and_risk_suite(
            [r["confidence"] for r in recs],
            [r["raw_f1"] for r in recs],
            abstained_flags=[r["is_abstained"] for r in recs],
        )
        mean_tok = sum(r["selected_tokens"] for r in recs) / n
        mean_pool_tok = sum(pool_tokens) / max(1, len(pool_tokens))
        method_summaries[method] = {
            "n": len(recs),
            "em_pct": round(100.0 * sum(r["raw_em"] for r in recs) / n, 2),
            "f1_pct": round(100.0 * sum(r["raw_f1"] for r in recs) / n, 2),
            "effective_em_pct": round(100.0 * sum(r["effective_em"] for r in recs) / n, 2),
            "effective_f1_pct": round(100.0 * sum(r["effective_f1"] for r in recs) / n, 2),
            "sp_precision_pct": round(100.0 * sum(r["sp_precision"] for r in recs) / n, 2),
            "sp_recall_pct": round(100.0 * sum(r["sp_recall"] for r in recs) / n, 2),
            "sp_f1_pct": round(100.0 * sum(r["sp_f1"] for r in recs) / n, 2),
            "early_exit_rate_pct": round(100.0 * sum(1 for r in recs if r["early_exit"]) / n, 2),
            "mean_selected_tokens": round(mean_tok, 1),
            "token_saving_vs_pool_pct": round(max(0.0, (mean_pool_tok - mean_tok) / max(1.0, mean_pool_tok)) * 100.0, 2),
            "ece": cal["ece"],
            "brier_score": cal["brier_score"],
            "auroc_confidence": cal["auroc_confidence"],
            "aurc": cal["aurc"],
            "e_aurc": cal["e_aurc"],
            "coverage_pct": cal["selective_coverage_pct"],
            "selective_risk_pct": cal["selective_risk_pct"],
            "harmful_abstention_rate_pct": cal["harmful_abstention_rate_pct"],
            "failure_taxonomy": summarize_failure_taxonomy(recs),
        }

    # Paired bootstrap & Holm-Bonferroni comparing Full_UGraphRAG vs baselines (first seed primary audit)
    full_recs = records_by_method["Full_UGraphRAG"]
    sig_f1: Dict[str, Any] = {}
    sig_brier: Dict[str, Any] = {}
    for base_m in ("B0_StandardRAG", "B1_BM25_RAG", "B2_Dense_RAG", "B3_MMR_RAG"):
        b_recs = records_by_method[base_m]
        if full_recs and len(full_recs) == len(b_recs):
            sig_f1[f"Full_vs_{base_m}"] = paired_bootstrap_with_tests(
                [r["raw_f1"] * 100.0 for r in full_recs], [r["raw_f1"] * 100.0 for r in b_recs]
            )
            sig_brier[f"Full_vs_{base_m}"] = paired_bootstrap_with_tests(
                [(r["confidence"] - r["raw_f1"]) ** 2 for r in full_recs],
                [(r["confidence"] - r["raw_f1"]) ** 2 for r in b_recs],
            )

    return {
        "seed": seed,
        "adaptive_tau_conf": adaptive_tau,
        "methods": method_summaries,
        "paired_significance": {
            "f1_pct": apply_multiple_comparison_correction(sig_f1),
            "brier_score": apply_multiple_comparison_correction(sig_brier),
        },
    }


DEFAULT_PHASE3_MODELS = ["qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="WP6: Cross-Family Generalization & 9-Method Comparison.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_PHASE3_MODELS, help="Backbone models to evaluate")
    parser.add_argument("--model-group", type=str, default=None, choices=["core3", "extra2", "all5"], help="3+2 Kaggle anti-disk-overflow model group")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "popqa", "2wikimultihop", "musique"])
    parser.add_argument("--scale", type=str, default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--m2-graph-mode", type=str, default="auto", choices=["auto", "entity_js", "entity_cooccur", "overlap_gated_js"])
    parser.add_argument("--m2-lambda-conflict", type=float, default=None, help="Explicit M2 lambda_conflict weight; overrides default/learned setting")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "bf16"])
    parser.add_argument("--allow-hf-online", action="store_true")
    parser.add_argument("--cleanup-after-model", action="store_true", help="Delete cached model weights after teardown to free disk space")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing wp6_cross_family_matrix.json instead of merging non-rerun models")
    parser.add_argument("--merge-from", nargs="+", default=[], help="External JSON files to merge matrix records from")
    parser.add_argument("--out-dir", type=str, default=str(WP3_DIRS["WP6"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase3.wp6", out_dir / "wp6_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    t0 = time.time()

    # Determine actual model list supporting 3+2 Kaggle anti-disk-overflow
    models_arg = args.models
    if args.model_group is not None and args.models == DEFAULT_PHASE3_MODELS:
        models_arg = None
    actual_models = resolve_phase3_model_list(models_arg, args.model_group)

    # Resolve M2 graph mode
    if args.m2_graph_mode == "auto":
        effective_graph_mode, graph_mode_source = resolve_wp8_recommended_graph_mode()
    else:
        effective_graph_mode = args.m2_graph_mode
        graph_mode_source = "cli_override"
    logger.info(
        f"WP6 execution setup: models={actual_models} (group={args.model_group}), "
        f"m2_graph_mode={effective_graph_mode} (source={graph_mode_source}, lambda_conflict={args.m2_lambda_conflict})"
    )

    slices: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {
        ds: load_phase3_dataset_cache(ds, scale=args.scale, allow_synthetic_fallback=args.dry_run)
        for ds in args.datasets
    }
    shared_dense = _load_cpu_dense_model(dry_run=args.dry_run)

    matrix: Dict[str, Any] = {}
    for mk in actual_models:
        if not args.dry_run and not is_phase3_backbone_available(mk, allow_hf_online=args.allow_hf_online):
            logger.warning(f"Skipping '{mk}': not downloaded locally and --allow-hf-online not set.")
            matrix[mk] = {"status": "skipped_unavailable"}
            continue

        ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp6_{mk}_{args.scale}.json", flush_every=25)
        ckpt.metadata.update({
            "model": mk,
            "scale": args.scale,
            "quantization": args.quantization,
            "m2_graph_mode": effective_graph_mode,
            "m2_graph_mode_source": graph_mode_source,
            "m2_lambda_conflict": args.m2_lambda_conflict,
            "datasets": list(args.datasets),
            "seeds": list(args.seeds),
            "schema_version": "wp6_v2",
        })
        reset_cuda_peak()
        generator = sequential_load_phase3_generator(mk, dry_run=args.dry_run, max_new_tokens=48, quantization=args.quantization)
        per_ds: Dict[str, Any] = {}
        try:
            if "hotpotqa" in slices:
                anchor_ds = "hotpotqa"
                anchor_dev_items = slices["hotpotqa"][0]
            else:
                try:
                    anchor_dev_items, _ = load_phase3_dataset_cache("hotpotqa", scale=args.scale, allow_synthetic_fallback=False)
                    anchor_ds = "hotpotqa"
                except Exception:
                    anchor_ds = args.datasets[0]
                    anchor_dev_items = slices[anchor_ds][0]

            model_tau_eu = calibrate_dev_m1_tau_eu(
                anchor_dev_items,
                generator,
                params,
                ckpt=ckpt,
                target_quantile=0.60,
                model_tag=mk,
                dataset_name=anchor_ds,
            )
            ckpt.metadata["calibrated_tau_eu"] = model_tau_eu
            ckpt.metadata["anchor_tau_dataset"] = anchor_ds
            ckpt.flush(force=True)
            logger.info(
                f"[WP6] model={mk} calibrated anchor M1 tau_eu={model_tau_eu:.4f} "
                f"on {anchor_ds} Dev (q=0.60, V5 default={params.get('m1_tau_eu', 0.18)})"
            )

            for ds in args.datasets:
                dev_items, test_items = slices[ds]
                per_seed_runs = []
                for sd in args.seeds:
                    logger.info(f"[WP6] model={mk} | dataset={ds} | seed={sd} | dev={len(dev_items)}, test={len(test_items)}")
                    per_seed_runs.append(
                        evaluate_model_dataset_seed(
                            mk,
                            ds,
                            sd,
                            dev_items,
                            test_items,
                            generator,
                            params,
                            effective_graph_mode,
                            out_dir,
                            ckpt,
                            shared_dense_model=shared_dense,
                            m2_lambda_conflict=args.m2_lambda_conflict,
                            tau_eu_override=model_tau_eu,
                        )
                    )
                multiseed = {
                    m: compute_multiseed_aggregate([r["methods"][m] for r in per_seed_runs])
                    for m in ALL_WP6_METHODS
                }
                per_ds[ds] = {
                    "n_dev": len(dev_items),
                    "n_test": len(test_items),
                    "primary_seed_run": per_seed_runs[0],
                    "multiseed_aggregate": multiseed,
                    "all_seed_runs": per_seed_runs,
                }
            matrix[mk] = {
                "status": "completed",
                "calibrated_tau_eu": model_tau_eu,
                "anchor_tau_dataset": anchor_ds,
                "vram_peak_mb": get_vram_snapshot_mb(),
                "datasets": per_ds,
            }
        finally:
            teardown_generator(generator)
            if args.cleanup_after_model:
                cleanup_cached_model_weights(mk)

    out_file = out_dir / "wp6_cross_family_matrix.json"
    merged_matrix: Dict[str, Any] = {}

    # Merge from existing file if not overwrite
    if not args.overwrite and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            for mk_key, mk_val in existing_payload.get("matrix", {}).items():
                merged_matrix[mk_key] = mk_val
        except Exception as e:
            logger.warning(f"Failed to read existing matrix from {out_file}: {e}")

    # Merge from external JSONs specified via --merge-from
    for mf in args.merge_from:
        mf_path = Path(mf)
        if mf_path.exists():
            try:
                with open(mf_path, "r", encoding="utf-8") as f:
                    ext_payload = json.load(f)
                for mk_key, mk_val in ext_payload.get("matrix", {}).items():
                    merged_matrix[mk_key] = mk_val
            except Exception as e:
                logger.warning(f"Failed to merge external matrix from {mf_path}: {e}")

    # Current run overwrites/updates current models
    merged_matrix.update(matrix)

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": list(merged_matrix.keys()),
            "active_models": actual_models,
            "model_group": args.model_group,
            "datasets": args.datasets,
            "scale": args.scale,
            "seeds": args.seeds,
            "m2_graph_mode": effective_graph_mode,
            "m2_graph_mode_source": graph_mode_source,
            "m2_lambda_conflict": args.m2_lambda_conflict,
            "quantization": args.quantization,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "hardware_manifest": collect_hardware_manifest(),
        "v5_integrity": v5_info,
        "matrix": merged_matrix,
    }
    atomic_write_json(out_file, payload)
    logger.info(f"Saved WP6 cross-family generalization matrix ({len(merged_matrix)} models) -> {out_file}")


if __name__ == "__main__":
    main()
