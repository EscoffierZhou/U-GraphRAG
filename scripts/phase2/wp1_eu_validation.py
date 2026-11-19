"""
wp1_eu_validation.py
====================
Phase II Work Package 1 (WP1): Model-Driven Epistemic / Aleatoric Uncertainty
Decomposition (BALD/MI via Prompt + Leave-One-Out Perturbations), Closed-Book
AUROC/AUPRC Benchmark, 4-Quadrant Controlled Validation, Evidence Replacement
Counterfactual Test, and Cross-Model / Cross-Dataset Threshold Transfer.

Writes strictly to `results/MileStone2/WP1_M1_EU/`.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import build_four_quadrant_probe, load_or_build_cache
from scripts.phase2.common import (
    PHASE2_CACHE_DIR,
    WP_DIRS,
    Phase2CheckpointManager,
    atomic_write_json,
    build_prompt,
    compute_auprc,
    compute_bald_uncertainty,
    compute_next_token_distribution,
    evaluate_qa_prediction,
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
from src.baseline_rag import normalize_answer
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m3_calibrated_decoder import compute_auroc

logger = setup_phase2_logger("phase2.wp1")


def _paraphrase_queries(question: str) -> List[str]:
    q = question.strip()
    return [
        q,
        f"Provide the short factual answer for: {q}",
        f"Concisely answer the following question: {q}",
        f"Identify the exact answer to: {q}",
    ]


def compute_all_uncertainty_metrics(
    generator: Any,
    question: str,
    docs: Sequence[Dict[str, Any]],
    decoupler: UncertaintyDecoupler,
    n_sc_samples: int = 5,
    include_self_consistency: bool = True,
) -> Dict[str, float]:
    """
    Compute all 6 uncertainty estimators on an identical (question, docs) input:
      1. raw_entropy_h_total: base single-pass token entropy
      2. max_prob_uncertainty: 1 - max p(y | x, D)
      3. length_norm_entropy: normalized entropy H / log|V|
      4. v5_u_eu_proxy: V5 algebraic Jaccard-proxy UncertaintyDecoupler.decompose()
      5. self_consistency_5x: 1 - consensus ratio across 5 stochastic/paraphrase passes
      6. bald_u_eu / bald_u_au: true perturbation Mutual Information (prompt paraphrases + LOO doc subsets)
    """
    doc_texts = [str(d.get("text", "")) for d in docs]
    base_probe = compute_next_token_distribution(generator, question, doc_texts, variant_seed="base")
    raw_h = float(base_probe["entropy"])
    max_p_unc = float(1.0 - base_probe["max_prob"])
    len_norm_h = float(base_probe["norm_entropy"])

    # V5 proxy baseline via existing src module
    v5_res = decoupler.decompose(
        question,
        [{"id": d.get("doc_id", str(i)), "text": str(d.get("text", "")), "entities": d.get("entities", [])} for i, d in enumerate(docs)],
        model_fn=lambda q, txts: compute_next_token_distribution(generator, q, txts, variant_seed="v5")["dist"],
    )

    # Perturbation ensemble for BALD MI via shared common helper
    bald_vals = compute_bald_uncertainty(generator, question, docs)
    bald_h_total = bald_vals["bald_h_total"]
    bald_u_eu = bald_vals["bald_u_eu"]
    bald_u_au = bald_vals["bald_u_au"]
    paras = _paraphrase_queries(question)

    # 5x Self-consistency answer dispersion (works on both real GPU & deterministic dry-run)
    sc_uncertainty = 0.0
    if include_self_consistency:
        sc_answers: List[str] = []
        if generator is not None and not getattr(generator, "dry_run", True) and getattr(generator, "model", None) is not None:
            import torch
            from src.baseline_rag import collect_eos_and_pad_ids

            eos_ids, pad_id = collect_eos_and_pad_ids(generator.tokenizer)
            for k_idx in range(n_sc_samples):
                q_var = paras[k_idx % len(paras)]
                prompt = build_prompt(q_var, docs, generator=generator, reliability_weighted=False)
                inputs = generator.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(generator.model.device)
                with torch.no_grad():
                    out = generator.model.generate(
                        **inputs,
                        max_new_tokens=24,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
                        pad_token_id=pad_id,
                    )
                gen_toks = out[0][inputs.input_ids.shape[1] :]
                ans_raw = generator.tokenizer.decode(gen_toks, skip_special_tokens=True)
                sc_answers.append(normalize_answer(generator._postprocess_answer(ans_raw)))
                del inputs, out
                if torch.cuda.is_available() and (torch.cuda.memory_reserved() / (1024 ** 2)) > 5150.0:
                    torch.cuda.empty_cache()
        else:
            for k_idx in range(n_sc_samples):
                p_k = compute_next_token_distribution(
                    generator,
                    paras[k_idx % len(paras)],
                    doc_texts if k_idx % 2 == 0 else doc_texts[: max(1, len(doc_texts) - 1)],
                    variant_seed=f"sc_{k_idx}",
                )
                top_tok = max(p_k["dist"].items(), key=lambda kv: kv[1])[0] if p_k["dist"] else f"unk_{k_idx}"
                sc_answers.append(top_tok)

        counts = Counter(sc_answers)
        top_count = counts.most_common(1)[0][1] if counts else 1
        sc_uncertainty = float(1.0 - (top_count / max(1, len(sc_answers))))

    return {
        "raw_entropy_h_total": round(raw_h, 6),
        "max_prob_uncertainty": round(max_p_unc, 6),
        "length_norm_entropy": round(len_norm_h, 6),
        "v5_u_eu_proxy": round(float(v5_res.epistemic_uncertainty), 6),
        "v5_u_au_proxy": round(float(v5_res.aleatoric_uncertainty), 6),
        "self_consistency_5x": round(sc_uncertainty, 6),
        "bald_h_total": round(bald_h_total, 6),
        "bald_u_eu": round(bald_u_eu, 6),
        "bald_u_au": round(bald_u_au, 6),
    }


def _find_best_youden_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Find threshold maximizing TPR - FPR on Dev without peeking at Test."""
    if not scores or len(set(labels)) < 2:
        return 0.18
    cands = sorted(set(float(s) for s in scores))
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    best_j = -1.0
    best_tau = cands[len(cands) // 2]
    for tau in cands:
        tp = sum(1 for s, y in zip(scores, labels) if s > tau and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s > tau and y == 0)
        tpr = tp / max(1, n_pos)
        fpr = fp / max(1, n_neg)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_tau = tau
    return round(float(best_tau), 6)


def _rank_metrics(records: Sequence[Dict[str, Any]], label_key: str, metric_keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    labels = [int(r[label_key]) for r in records]
    out: Dict[str, Dict[str, float]] = {}
    for mk in metric_keys:
        scores = [float(r[mk]) for r in records]
        out[mk] = {
            "auroc": round(float(compute_auroc(scores, labels)), 4),
            "auprc": round(float(compute_auprc(labels, scores)), 4),
            "mean_pos": round(float(np.mean([s for s, y in zip(scores, labels) if y == 1])) if any(labels) else 0.0, 4),
            "mean_neg": round(float(np.mean([s for s, y in zip(scores, labels) if y == 0])) if any(1 - y for y in labels) else 0.0, 4),
        }
    return out


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr

        val = spearmanr(x, y).statistic
        return round(float(val), 4) if not math.isnan(float(val)) else 0.0
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        denom = np.std(rx) * np.std(ry)
        if denom < 1e-12:
            return 0.0
        return round(float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))) / denom), 4)


def run_wp1_for_model(
    model_key: str,
    datasets: Sequence[str],
    limit: int,
    dry_run: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    """Execute WP1 uncertainty validation for one backbone using shared WP0 caches."""
    decoupler = UncertaintyDecoupler()
    v5_params = load_v5_balanced_params()
    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp1_{model_key}.json", flush_every=25)
    reset_cuda_peak()
    generator = sequential_load_generator(model_key, dry_run=dry_run, max_new_tokens=32)

    metric_names = [
        "bald_u_eu",
        "v5_u_eu_proxy",
        "raw_entropy_h_total",
        "max_prob_uncertainty",
        "length_norm_entropy",
        "self_consistency_5x",
    ]
    dataset_results: Dict[str, Any] = {}

    try:
        # 1. Closed-book vs RAG error predictability across datasets
        for dname in datasets:
            cache = load_or_build_cache(dname, pool_k=16, allow_synthetic_fallback=dry_run)
            items = list(cache.get("items", []))
            if limit > 0:
                items = items[:limit]

            per_sample: List[Dict[str, Any]] = []
            for item in items:
                ckey = f"cb::{dname}::{item['id']}"
                if ckpt.has(ckey):
                    per_sample.append(ckpt.get(ckey))
                    continue

                q = item["question"]
                golds = item["possible_answers"]
                d0_docs = item["pool"][:3]

                # Closed-book uncertainty metrics & generation
                cb_metrics = compute_all_uncertainty_metrics(generator, q, [], decoupler)
                cb_prompt = build_prompt(q, [], generator=generator)
                cb_pred, _ = generator.generate(cb_prompt, [], max_new_tokens=24)
                cb_eval = evaluate_qa_prediction(cb_pred, golds)

                # RAG (D0) generation to verify whether retrieval helps this query
                rag_prompt = build_prompt(q, d0_docs, generator=generator)
                rag_pred, _ = generator.generate(rag_prompt, d0_docs, max_new_tokens=24)
                rag_eval = evaluate_qa_prediction(rag_pred, golds)

                rec = {
                    "id": item["id"],
                    "split": item.get("split", "test"),
                    "s_pop": item.get("s_pop"),
                    "closed_em": cb_eval["em"],
                    "closed_f1": cb_eval["f1"],
                    "closed_is_error": int(cb_eval["f1"] < 0.5),
                    "rag_em": rag_eval["em"],
                    "rag_f1": rag_eval["f1"],
                    "retrieval_helped": int((rag_eval["f1"] - cb_eval["f1"]) > 0.2),
                    **cb_metrics,
                }
                ckpt.record(ckey, rec)
                per_sample.append(rec)

            ckpt.flush(force=True)
            auroc_err = _rank_metrics(per_sample, "closed_is_error", metric_names)
            extra_stats: Dict[str, Any] = {}
            if dname == "popqa":
                pops = [math.log10(max(1, int(r["s_pop"] or 1))) for r in per_sample]
                extra_stats["spearman_bald_eu_vs_log_spop"] = _spearman_corr(
                    [r["bald_u_eu"] for r in per_sample], pops
                )
                extra_stats["spearman_v5_eu_vs_log_spop"] = _spearman_corr(
                    [r["v5_u_eu_proxy"] for r in per_sample], pops
                )

            dataset_results[dname] = {
                "n_samples": len(per_sample),
                "closed_book_em_pct": round(100.0 * sum(r["closed_em"] for r in per_sample) / max(1, len(per_sample)), 2),
                "d0_rag_em_pct": round(100.0 * sum(r["rag_em"] for r in per_sample) / max(1, len(per_sample)), 2),
                "closed_error_detection": auroc_err,
                "extra": extra_stats,
                "records": per_sample,
            }

        # 2. Four-Quadrant Controlled Probe (Q1-Q4)
        probe_file = PHASE2_CACHE_DIR / "wp1_four_quadrant_probe.json"
        if not probe_file.exists():
            hp_cache = load_or_build_cache("hotpotqa", pool_k=16, allow_synthetic_fallback=dry_run)
            probe_data = build_four_quadrant_probe(hp_cache, max_per_quadrant=15 if dry_run else 40)
        else:
            probe_data = load_json_file(probe_file)

        quad_records: List[Dict[str, Any]] = []
        probe_items = probe_data.get("items", [])
        if dry_run:
            probe_items = probe_items[:40]
        for pitem in probe_items:
            p_key = f"q4::{pitem['probe_id']}"
            if ckpt.has(p_key):
                quad_records.append(ckpt.get(p_key))
                continue
            m = compute_all_uncertainty_metrics(generator, pitem["question"], pitem["docs"], decoupler)
            rec = {
                "probe_id": pitem["probe_id"],
                "quadrant": pitem["quadrant"],
                "expected_eu_high": int(bool(pitem["expected_eu_high"])),
                "has_conflict": int(bool(pitem["has_conflict"])),
                "evidence_missing": int(pitem["quadrant"] == "Q4_unknown_missing"),
                **m,
            }
            ckpt.record(p_key, rec)
            quad_records.append(rec)
        ckpt.flush(force=True)

        quadrant_summary: Dict[str, Any] = {}
        for q_name in ("Q1_known_consistent", "Q2_unknown_sufficient", "Q3_known_conflicting", "Q4_unknown_missing"):
            sub = [r for r in quad_records if r["quadrant"] == q_name]
            if not sub:
                continue
            quadrant_summary[q_name] = {
                "count": len(sub),
                "mean_bald_u_eu": round(float(np.mean([r["bald_u_eu"] for r in sub])), 4),
                "mean_bald_u_au": round(float(np.mean([r["bald_u_au"] for r in sub])), 4),
                "mean_v5_u_eu_proxy": round(float(np.mean([r["v5_u_eu_proxy"] for r in sub])), 4),
                "mean_raw_entropy": round(float(np.mean([r["raw_entropy_h_total"] for r in sub])), 4),
            }

        # Distinguishability AUROC across quadrants (Insufficient/Conflict vs Sufficient/Consistent)
        quadrant_separation = _rank_metrics(quad_records, "expected_eu_high", metric_names)
        q3_q4 = [r for r in quad_records if r["quadrant"] in ("Q3_known_conflicting", "Q4_unknown_missing")]
        conflict_vs_missing_au_auroc = (
            round(float(compute_auroc([r["bald_u_au"] for r in q3_q4], [r["has_conflict"] for r in q3_q4])), 4)
            if len({r["has_conflict"] for r in q3_q4}) == 2
            else 0.5
        )

        # 3. Evidence Replacement Sensitivity Test on HotpotQA
        hp_cache = load_or_build_cache("hotpotqa", pool_k=16, allow_synthetic_fallback=dry_run)
        hp_test = [it for it in hp_cache["items"] if it.get("split") in ("dev", "test")][: (20 if dry_run else 60)]
        repl_rows: List[Dict[str, Any]] = []
        for it in hp_test:
            rkey = f"repl::{it['id']}"
            if ckpt.has(rkey):
                repl_rows.append(ckpt.get(rkey))
                continue
            pool = it["pool"]
            gold = [d for d in pool if d.get("is_gold")]
            dist = [d for d in pool if not d.get("is_gold")]
            if not gold or not dist:
                continue
            conf_doc = it.get("synthetic_conflict_doc")
            m_gold = compute_all_uncertainty_metrics(
                generator, it["question"], gold[:2] + dist[:1], decoupler, include_self_consistency=False
            )
            m_miss = compute_all_uncertainty_metrics(
                generator, it["question"], dist[:3], decoupler, include_self_consistency=False
            )
            m_conf = compute_all_uncertainty_metrics(
                generator,
                it["question"],
                ([gold[0], conf_doc] + dist[:1]) if conf_doc else dist[:3],
                decoupler,
                include_self_consistency=False,
            )
            r_rec = {
                "id": it["id"],
                "gold_bald_eu": m_gold["bald_u_eu"],
                "missing_bald_eu": m_miss["bald_u_eu"],
                "conflict_bald_eu": m_conf["bald_u_eu"],
                "gold_bald_au": m_gold["bald_u_au"],
                "missing_bald_au": m_miss["bald_u_au"],
                "conflict_bald_au": m_conf["bald_u_au"],
            }
            ckpt.record(rkey, r_rec)
            repl_rows.append(r_rec)
        ckpt.flush(force=True)

        eu_missing_vs_gold = paired_bootstrap_with_tests(
            [r["missing_bald_eu"] for r in repl_rows],
            [r["gold_bald_eu"] for r in repl_rows],
        )
        au_conflict_vs_gold = paired_bootstrap_with_tests(
            [r["conflict_bald_au"] for r in repl_rows],
            [r["gold_bald_au"] for r in repl_rows],
        )

        # Dev Youden threshold extraction for transfer verification
        hp_dev = [r for r in dataset_results.get("hotpotqa", {}).get("records", []) if r.get("split") == "dev"]
        if not hp_dev:
            hp_dev = dataset_results.get("hotpotqa", {}).get("records", [])
        dev_tau_bald = _find_best_youden_threshold(
            [r["bald_u_eu"] for r in hp_dev],
            [r["closed_is_error"] for r in hp_dev],
        )
        dev_tau_v5 = float(v5_params.get("m1_tau_eu", 0.18))

        vram_peak = get_vram_snapshot_mb()
        return {
            "model": model_key,
            "vram_peak_mb": vram_peak,
            "learned_dev_tau_bald_eu": dev_tau_bald,
            "v5_frozen_tau_eu": dev_tau_v5,
            "datasets": {
                k: {kk: vv for kk, vv in v.items() if kk != "records"}
                for k, v in dataset_results.items()
            },
            "raw_records_by_dataset": {k: v["records"] for k, v in dataset_results.items()},
            "four_quadrant_summary": quadrant_summary,
            "four_quadrant_separation_auroc": quadrant_separation,
            "conflict_vs_missing_au_auroc": conflict_vs_missing_au_auroc,
            "replacement_sensitivity": {
                "n_tested": len(repl_rows),
                "eu_missing_minus_gold": eu_missing_vs_gold,
                "au_conflict_minus_gold": au_conflict_vs_gold,
            },
        }
    finally:
        teardown_generator(generator)


def _evaluate_threshold_transfer(
    dev_tau: float,
    records: Sequence[Dict[str, Any]],
    score_key: str = "bald_u_eu",
    quantile_ref_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Evaluate raw vs quantile-matched threshold transfer across models or datasets."""
    if not records:
        return {}
    scores = [float(r[score_key]) for r in records]
    labels = [int(r["closed_is_error"]) for r in records]

    def _stats_at(tau: float) -> Dict[str, float]:
        preds = [1 if s > tau else 0 for s in scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        prec = tp / max(1, tp + fp)
        f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
        return {
            "tau": round(float(tau), 6),
            "trigger_rate_pct": round(100.0 * sum(preds) / len(preds), 2),
            "tpr_pct": round(100.0 * tpr, 2),
            "fpr_pct": round(100.0 * fpr, 2),
            "youden_j": round(tpr - fpr, 4),
            "f1": round(f1, 4),
        }

    out = {"raw_threshold": _stats_at(dev_tau)}
    if quantile_ref_scores:
        q_rank = float(np.mean([1.0 if x <= dev_tau else 0.0 for x in quantile_ref_scores]))
        q_tau = float(np.quantile(scores, q_rank))
        out["quantile_matched"] = _stats_at(q_tau)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="WP1: Model-driven EU/AU Validation & Transfer.")
    parser.add_argument("--models", nargs="+", default=["qwen3b", "qwen7b"], help="Backbones to evaluate sequentially")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "popqa"])
    parser.add_argument("--limit", type=int, default=300, help="Max samples per dataset (0 = all)")
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["WP1"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.wp1", out_dir / "wp1_run.log")
    v5_info = verify_v5_frozen_checksums()
    logger.info(f"V5 checksum verified: {v5_info}")

    t0 = time.time()
    all_runs: Dict[str, Any] = {}
    for m_key in args.models:
        logger.info(f"Running WP1 sequentially for backbone={m_key} (dry_run={args.dry_run})...")
        all_runs[m_key] = run_wp1_for_model(
            model_key=m_key,
            datasets=args.datasets,
            limit=40 if (args.dry_run and args.limit > 40) else args.limit,
            dry_run=args.dry_run,
            out_dir=out_dir,
        )

    # Cross-model and cross-dataset threshold transfer summary
    transfer_summary: Dict[str, Any] = {}
    primary = args.models[0]
    p_run = all_runs[primary]
    dev_tau = p_run["learned_dev_tau_bald_eu"]
    ref_hp = [r["bald_u_eu"] for r in p_run["raw_records_by_dataset"].get("hotpotqa", []) if r.get("split") == "dev"]

    if "popqa" in p_run["raw_records_by_dataset"]:
        transfer_summary[f"{primary}_hotpot_to_popqa"] = _evaluate_threshold_transfer(
            dev_tau, p_run["raw_records_by_dataset"]["popqa"], quantile_ref_scores=ref_hp
        )
    for sec in args.models[1:]:
        for dname, recs in all_runs[sec]["raw_records_by_dataset"].items():
            transfer_summary[f"{primary}_hotpot_to_{sec}_{dname}"] = _evaluate_threshold_transfer(
                dev_tau, recs, quantile_ref_scores=ref_hp
            )

    summary_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": args.models,
            "datasets": args.datasets,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "v5_integrity": v5_info,
        "runs": {
            m: {k: v for k, v in r.items() if k != "raw_records_by_dataset"}
            for m, r in all_runs.items()
        },
        "threshold_transfer": transfer_summary,
    }
    atomic_write_json(out_dir / "wp1_eu_validation_results.json", summary_payload)
    logger.info(f"Saved WP1 results -> {out_dir / 'wp1_eu_validation_results.json'}")


if __name__ == "__main__":
    main()
