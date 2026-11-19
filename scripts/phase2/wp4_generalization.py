"""
wp4_generalization.py
=====================
Phase II Work Package 4 (WP4): Cross-Dataset & Cross-Backbone Generalization Matrix.

Runs strictly ONE 4-bit NF4 backbone at a time (8GB VRAM sequential discipline) across:
  Datasets: HotpotQA (unseen test), PopQA (unseen test), 2WikiMultiHopQA (test), MuSiQue (test)
  Backbones: qwen3b, qwen7b (+ optional mistral7b, llama3_8b when cached/enabled)
  Systems: ClosedBook, B0_StandardRAG, M1_NoM2, Full_UGraphRAG

Writes strictly to `results/MileStone2/WP4_Generalization/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import load_or_build_cache
from scripts.phase2.common import (
    MODELS_DIR,
    WP_DIRS,
    BACKBONE_ALIASES,
    Phase2CheckpointManager,
    atomic_write_json,
    build_calibrated_decoder,
    build_prompt,
    compute_calibration_and_risk_suite,
    compute_next_token_distribution,
    compute_token_count,
    compute_unified_m1_signal,
    evaluate_qa_prediction,
    get_vram_snapshot_mb,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    select_m2_evidence_from_pool,
    sequential_load_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m3_calibrated_decoder import CalibratedDecoderConfig, M3CalibratedDecoder
from src.u_graphrag_pipeline import compute_supporting_facts_metrics

logger = setup_phase2_logger("phase2.wp4")


def _is_backbone_available(model_key: str, allow_hf_online: bool) -> bool:
    if allow_hf_online:
        return True
    low = model_key.lower()
    if low in BACKBONE_ALIASES:
        local_name, _ = BACKBONE_ALIASES[low]
        return (MODELS_DIR / local_name).exists()
    p = Path(model_key)
    return p.exists() or (MODELS_DIR / model_key).exists()


def _summarize_mode_records(records: Sequence[Dict[str, Any]], pool_tokens: Sequence[int]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    cal_suite = compute_calibration_and_risk_suite(
        [r["confidence"] for r in records],
        [r["raw_f1"] for r in records],
        abstained_flags=[bool(r["is_abstained"]) for r in records],
    )
    mean_used_tok = sum(r["selected_tokens"] for r in records) / n
    mean_pool_tok = sum(pool_tokens) / max(1, len(pool_tokens))
    tok_save = max(0.0, (mean_pool_tok - mean_used_tok) / max(1.0, mean_pool_tok)) * 100.0
    return {
        "n": n,
        "em_pct": round(100.0 * sum(r["raw_em"] for r in records) / n, 2),
        "f1_pct": round(100.0 * sum(r["raw_f1"] for r in records) / n, 2),
        "effective_em_pct": round(100.0 * sum(r["effective_em"] for r in records) / n, 2),
        "effective_f1_pct": round(100.0 * sum(r["effective_f1"] for r in records) / n, 2),
        "sp_precision_pct": round(100.0 * sum(r["sp_precision"] for r in records) / n, 2),
        "sp_recall_pct": round(100.0 * sum(r["sp_recall"] for r in records) / n, 2),
        "sp_f1_pct": round(100.0 * sum(r["sp_f1"] for r in records) / n, 2),
        "early_exit_rate_pct": round(100.0 * sum(1 for r in records if r.get("early_exit")) / n, 2),
        "mean_selected_tokens": round(mean_used_tok, 1),
        "token_saving_vs_pool_pct": round(tok_save, 2),
        "calibration_and_risk": {k: v for k, v in cal_suite.items() if k not in ("bin_breakdown", "curve_points")},
    }


def _evaluate_dataset_with_model(
    model_key: str,
    dataset_name: str,
    items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    eu_mode: str,
    ckpt: Phase2CheckpointManager,
    pool_k: int = 8,
    max_k: int = 5,
) -> Dict[str, Any]:
    """Run all 4 modes sequentially on one (model, dataset) slice."""
    decoupler = UncertaintyDecoupler()
    tau_eu = float(params.get("m1_tau_eu", 0.18))
    uncal_decoder = M3CalibratedDecoder(
        CalibratedDecoderConfig(
            abstain_threshold=0.0,
            confidence_threshold=0.0,
            beta=0.0,
            enable_contrastive=False,
            preserve_evidence_order=True,
            max_new_tokens=48,
            confidence_temperature=1.0,
            confidence_tok_weight=1.0,
            confidence_platt_a=1.0,
            confidence_platt_b=0.0,
        )
    )
    full_decoder = build_calibrated_decoder(params, max_new_tokens=48)

    modes = ("ClosedBook", "B0_StandardRAG", "M1_NoM2", "Full_UGraphRAG")
    records_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    pool_tokens: List[int] = []

    for item in items:
        iid = str(item["id"])
        pool = list(item.get("pool", []))[:pool_k]
        if not pool:
            continue
        pool_tokens.append(sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in pool))
        d0_docs = pool[: min(3, len(pool))]
        gold_titles = list(item.get("gold_titles", []))
        golds = item["possible_answers"]

        # Shared M1 routing decision for M1_NoM2 and Full_UGraphRAG
        eu_key = f"eu::{dataset_name}::{iid}"
        if ckpt.has(eu_key):
            eu_info = ckpt.get(eu_key)
        else:
            m1_sig = compute_unified_m1_signal(
                generator, item["question"], d0_docs, eu_mode=eu_mode, decoupler=decoupler
            )
            eu_val = float(m1_sig["u_eu"])
            eu_info = {"u_eu": eu_val, "early_exit": bool(eu_val <= tau_eu), "eu_mode": m1_sig["mode"]}
            ckpt.record(eu_key, eu_info)

        early_exit = bool(eu_info["early_exit"])

        # Precompute M2 selection only when M1 triggers expansion
        m2_docs_cache: Optional[List[Dict[str, Any]]] = None

        for mode in modes:
            rkey = f"run::{dataset_name}::{mode}::{iid}"
            if ckpt.has(rkey):
                records_by_mode[mode].append(ckpt.get(rkey))
                continue

            if mode == "ClosedBook":
                prompt = build_prompt(item["question"], [], generator=generator)
                pred, _ = generator.generate(prompt, [], max_new_tokens=32)
                qa = evaluate_qa_prediction(pred, golds, is_abstained=False)
                # Closed-book confidence from base next-token probability
                prob_info = compute_next_token_distribution(generator, item["question"], [], variant_seed=f"cb_{iid}")
                rec = {
                    "id": iid,
                    "raw_em": qa["em"],
                    "raw_f1": qa["f1"],
                    "effective_em": qa["effective_em"],
                    "effective_f1": qa["effective_f1"],
                    "confidence": float(prob_info["max_prob"]),
                    "is_abstained": False,
                    "early_exit": True,
                    "sp_precision": 0.0,
                    "sp_recall": 0.0,
                    "sp_f1": 0.0,
                    "selected_tokens": 0,
                }
            elif mode == "B0_StandardRAG":
                ev = [
                    {**d, "reliability": float(d.get("score", 0.5)), "uncertainty": 0.0}
                    for d in d0_docs
                ]
                dec_res = uncal_decoder.generate_calibrated(generator, item["question"], ev)
                qa = evaluate_qa_prediction(dec_res.raw_answer, golds, is_abstained=dec_res.is_abstained)
                sp = compute_supporting_facts_metrics([str(d.get("title", d.get("doc_id", ""))) for d in ev], gold_titles)
                rec = {
                    "id": iid,
                    "raw_em": qa["em"],
                    "raw_f1": qa["f1"],
                    "effective_em": qa["effective_em"],
                    "effective_f1": qa["effective_f1"],
                    "confidence": float(dec_res.confidence),
                    "is_abstained": bool(dec_res.is_abstained),
                    "early_exit": True,
                    "sp_precision": sp["sp_precision"],
                    "sp_recall": sp["sp_recall"],
                    "sp_f1": sp["sp_f1"],
                    "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in ev),
                }
            elif mode == "M1_NoM2":
                chosen = d0_docs if early_exit else pool[:max_k]
                ev = [{**d, "reliability": float(d.get("score", 0.5)), "uncertainty": 0.0} for d in chosen]
                dec_res = uncal_decoder.generate_calibrated(generator, item["question"], ev)
                qa = evaluate_qa_prediction(dec_res.raw_answer, golds, is_abstained=dec_res.is_abstained)
                sp = compute_supporting_facts_metrics([str(d.get("title", d.get("doc_id", ""))) for d in ev], gold_titles)
                rec = {
                    "id": iid,
                    "raw_em": qa["em"],
                    "raw_f1": qa["f1"],
                    "effective_em": qa["effective_em"],
                    "effective_f1": qa["effective_f1"],
                    "confidence": float(dec_res.confidence),
                    "is_abstained": bool(dec_res.is_abstained),
                    "early_exit": early_exit,
                    "sp_precision": sp["sp_precision"],
                    "sp_recall": sp["sp_recall"],
                    "sp_f1": sp["sp_f1"],
                    "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in ev),
                }
            else:
                if early_exit:
                    ev = [
                        {
                            **d,
                            "reliability": float(min(1.0, float(d.get("score", 0.5)) + float(params.get("m2_reliability_offset", 0.22)))),
                            "uncertainty": 0.0,
                        }
                        for d in d0_docs
                    ]
                else:
                    if m2_docs_cache is None:
                        per_doc_dists = {
                            str(d["doc_id"]): compute_next_token_distribution(
                                generator, item["question"], [d["text"]], variant_seed=str(d["doc_id"])
                            )["dist"]
                            for d in pool
                        }
                        m2_docs_cache, _ = select_m2_evidence_from_pool(
                            item["question"], pool, per_doc_dists, params, pool_k=pool_k, max_k=max_k
                        )
                    ev = m2_docs_cache

                dec_res = full_decoder.generate_calibrated(generator, item["question"], ev)
                qa = evaluate_qa_prediction(dec_res.raw_answer, golds, is_abstained=dec_res.is_abstained)
                sp = compute_supporting_facts_metrics([str(d.get("title", d.get("doc_id", ""))) for d in ev], gold_titles)
                rec = {
                    "id": iid,
                    "raw_em": qa["em"],
                    "raw_f1": qa["f1"],
                    "effective_em": qa["effective_em"],
                    "effective_f1": qa["effective_f1"],
                    "confidence": float(dec_res.confidence),
                    "is_abstained": bool(dec_res.is_abstained),
                    "early_exit": early_exit,
                    "sp_precision": sp["sp_precision"],
                    "sp_recall": sp["sp_recall"],
                    "sp_f1": sp["sp_f1"],
                    "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in ev),
                }

            ckpt.record(rkey, rec)
            records_by_mode[mode].append(rec)

    ckpt.flush(force=True)
    summary = {m: _summarize_mode_records(recs, pool_tokens) for m, recs in records_by_mode.items()}

    sig = {}
    full_recs = records_by_mode["Full_UGraphRAG"]
    if full_recs:
        for base_m in ("B0_StandardRAG", "M1_NoM2"):
            b_recs = records_by_mode[base_m]
            if len(b_recs) == len(full_recs):
                sig[f"Full_vs_{base_m}"] = {
                    "raw_f1_pct": paired_bootstrap_with_tests([r["raw_f1"] * 100.0 for r in full_recs], [r["raw_f1"] * 100.0 for r in b_recs]),
                    "raw_em_pct": paired_bootstrap_with_tests([r["raw_em"] * 100.0 for r in full_recs], [r["raw_em"] * 100.0 for r in b_recs]),
                    "brier_diff": paired_bootstrap_with_tests(
                        [(r["confidence"] - r["raw_f1"]) ** 2 for r in full_recs],
                        [(r["confidence"] - r["raw_f1"]) ** 2 for r in b_recs],
                    ),
                }

    return {"modes": summary, "paired_significance": sig}


def main() -> None:
    parser = argparse.ArgumentParser(description="WP4: Cross-Dataset & Cross-Backbone Sequential Matrix.")
    parser.add_argument("--models", nargs="+", default=["qwen3b", "qwen7b"], help="Seq order (e.g. qwen3b qwen7b mistral7b)")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "popqa", "2wikimultihop", "musique"])
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=120, help="Samples per unseen test slice")
    parser.add_argument("--eu-mode", type=str, default="v5_proxy", choices=["v5_proxy", "paraphrase_disagreement", "bald_mi"])
    parser.add_argument("--allow-hf-online", action="store_true", help="Allow loading remote HF backbones not yet cached locally")
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["WP4"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.wp4", out_dir / "wp4_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    t0 = time.time()

    # Load all dataset caches on CPU first so GPU model holds are minimal
    eff_limit = 15 if args.dry_run else args.limit
    dataset_slices: Dict[str, List[Dict[str, Any]]] = {}
    for dname in args.datasets:
        cache = load_or_build_cache(dname, pool_k=16, allow_synthetic_fallback=args.dry_run)
        items = [it for it in cache.get("items", []) if args.split == "all" or it.get("split") == args.split]
        if not items:
            items = list(cache.get("items", []))
        dataset_slices[dname] = items[:eff_limit]

    matrix_results: Dict[str, Any] = {}
    for m_key in args.models:
        if not args.dry_run and not _is_backbone_available(m_key, args.allow_hf_online):
            logger.warning(f"Skipping backbone '{m_key}': not cached locally and --allow-hf-online not passed.")
            matrix_results[m_key] = {"status": "skipped_unavailable"}
            continue

        ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp4_{m_key}.json", flush_every=25)
        reset_cuda_peak()
        generator = sequential_load_generator(m_key, dry_run=args.dry_run, max_new_tokens=48)
        per_ds: Dict[str, Any] = {}
        try:
            for dname, slice_items in dataset_slices.items():
                logger.info(f"Evaluating model={m_key} on dataset={dname} (N={len(slice_items)})...")
                per_ds[dname] = _evaluate_dataset_with_model(
                    model_key=m_key,
                    dataset_name=dname,
                    items=slice_items,
                    generator=generator,
                    params=params,
                    eu_mode=args.eu_mode,
                    ckpt=ckpt,
                )
            matrix_results[m_key] = {
                "status": "completed",
                "vram_peak_mb": get_vram_snapshot_mb(),
                "datasets": per_ds,
            }
        finally:
            teardown_generator(generator)

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": args.models,
            "datasets": args.datasets,
            "split": args.split,
            "eu_mode": args.eu_mode,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "v5_integrity": v5_info,
        "matrix": matrix_results,
    }
    out_file = out_dir / "wp4_generalization_matrix.json"
    atomic_write_json(out_file, payload)
    logger.info(f"Saved WP4 generalization matrix -> {out_file}")


if __name__ == "__main__":
    main()
