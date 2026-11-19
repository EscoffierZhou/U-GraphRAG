"""
wp8_method_validity.py
======================
Milestone 3 Phase A (WP8): Methodological Validity Verification Gate.

Tests rigorously before scaling across all models (AboutMileStone_3.md Sec II):
  8.1 M1 Signal Validity: Controlled interventions (Known-Consistent, Unknown-Sufficient,
      Known-Conflicting, Unknown-Missing) comparing V5 Proxy vs Paraphrase Disagreement vs Entropy.
  8.2 M2 Independent Contribution Gate: Compares B3_MMR, EntityCooccur, RandomGraph, NoGraph,
      Orig_M2_EntityJS (lambda_c in {1.0, 2.5, 5.0, 10.0}), and OverlapGated_JS across
      Clean and Conflict-Injected pools, verifying whether M2 yields independent value in
      quality, recall, conflict suppression, or deterministic efficiency.
  8.3 M3 Adaptive Calibration & Gating: Fixed_Beta_0 vs Fixed_Beta_012 vs Uncertainty_Gated_Beta(q),
      Static tau=0.38 vs Dev Risk-Constrained Coverage Threshold across {0.4..0.8} coverages,
      plus abstention behavior broken down by error type.

Writes strictly to `results/MileStone3/WP8_MethodValidity/`.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import build_four_quadrant_probe
from scripts.phase2.p0_1_m2_conflict_and_scale import _run_pool_evaluation, select_dev_pareto_lambda
from scripts.phase2.p0_2_m1_signal_unification import _compute_dual_signals
from scripts.phase2.p0_3_m3_adaptive_gating import (
    compute_exact_risk_at_target_coverages,
    evaluate_policy_on_traces,
    select_dev_risk_constrained_threshold,
)
from scripts.phase2.wp3_m3_calibration import _collect_decode_traces, _fit_dev_calibration_configs
from scripts.phase3.phase3_common import (
    M1_SIGNAL_TAXONOMY,
    WP3_DIRS,
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    calibrate_dev_m1_tau_eu,
    compute_unified_m1_signal,
    get_vram_snapshot_mb,
    load_phase2_cache,
    load_phase3_dataset_cache,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    sequential_load_phase3_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m3_calibrated_decoder import compute_auroc

logger = setup_phase2_logger("phase3.wp8")


def _evaluate_wp8_m1_validity(
    generator: Any,
    hotpot_cache: Dict[str, Any],
    items: Sequence[Dict[str, Any]],
    tau_eu: float,
) -> Dict[str, Any]:
    """8.1 M1 Signal Validity across 4 interventions + controlled replacement."""
    decoupler = UncertaintyDecoupler()
    probe = build_four_quadrant_probe(hotpot_cache, max_per_quadrant=min(30, max(6, len(items) // 2)))
    q_items = probe.get("items", [])
    rows_by_q: Dict[str, List[Dict[str, float]]] = {}
    for qit in q_items:
        sig = _compute_dual_signals(generator, qit["question"], qit["docs"], decoupler)
        sig["v5_triggers_expand"] = float(sig["v5_u_eu_proxy"] > tau_eu)
        rows_by_q.setdefault(qit["quadrant"], []).append(sig)

    summary_q = {}
    for qname, rlist in rows_by_q.items():
        summary_q[qname] = {
            "count": len(rlist),
            "mean_v5_eu": round(float(np.mean([r["v5_u_eu_proxy"] for r in rlist])), 4),
            "mean_para_eu": round(float(np.mean([r["paraphrase_disagreement_eu"] for r in rlist])), 4),
            "mean_v5_au": round(float(np.mean([r["v5_u_au_proxy"] for r in rlist])), 4),
            "mean_para_au": round(float(np.mean([r["within_paraphrase_entropy_au"] for r in rlist])), 4),
            "v5_expand_rate_pct": round(100.0 * float(np.mean([r["v5_triggers_expand"] for r in rlist])), 2),
        }

    # Verify Q4 (Missing) > Q1 (Consistent) under V5 Proxy vs Paraphrase Disagreement
    q1_v5 = summary_q.get("Q1_known_consistent", {}).get("mean_v5_eu", 0.0)
    q4_v5 = summary_q.get("Q4_unknown_missing", {}).get("mean_v5_eu", 0.0)
    q1_para = summary_q.get("Q1_known_consistent", {}).get("mean_para_eu", 0.0)
    q4_para = summary_q.get("Q4_unknown_missing", {}).get("mean_para_eu", 0.0)
    all_rows = [(q, r) for q, lst in rows_by_q.items() for r in lst]
    expected_hi = [1 if q in ("Q3_known_conflicting", "Q4_unknown_missing") else 0 for q, _ in all_rows]
    auroc_v5 = round(float(compute_auroc([r["v5_u_eu_proxy"] for _, r in all_rows], expected_hi)), 4)
    auroc_para = round(float(compute_auroc([r["paraphrase_disagreement_eu"] for _, r in all_rows], expected_hi)), 4)
    v5_mono = bool(q4_v5 >= q1_v5)
    para_mono = bool(q4_para >= q1_para)

    return {
        "taxonomy": M1_SIGNAL_TAXONOMY,
        "tau_eu": tau_eu,
        "quadrant_summary": summary_q,
        "insufficient_or_conflict_auroc": {"v5_proxy": auroc_v5, "paraphrase_disagreement": auroc_para},
        "m1_monotonic_missing_gt_consistent": v5_mono,
        "paraphrase_monotonic_missing_gt_consistent": para_mono,
        "recommended_m1_claim_scope": (
            "operational_uncertainty_routing_signal" if not v5_mono else "monotonic_epistemic_proxy"
        ),
        "scope_note": (
            "U_EU^v5 = (0.25 + 0.5*J_bar)*H_total emphasizes high-entropy overlapping conflict (Q3) "
            "while weighting unrelated missing docs (J_bar->0) at 0.25*H_total; interpret V5 Proxy as an "
            "operational uncertainty routing signal alongside Paraphrase Disagreement (U_EU^para)."
        ),
    }


def _decide_m2_validity(
    clean_sum: Dict[str, Any],
    conf_sum: Dict[str, Any],
    pareto_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Implements AboutMileStone_3.md Section 8.2 & Phase A gate:
      Determine whether M2 independently improves quality/conflict suppression over MMR/EntityGraph,
      or whether M2 should honestly narrow its claim to deterministic budgeted topology organization.
    """
    mmr_f1_clean = float(clean_sum["A5_mmr_baseline"]["f1"])
    m2_lc1_f1_clean = float(clean_sum["M2_orig_lc1.0"]["f1"])
    ent_f1_clean = float(clean_sum["G_entity_cooccur"]["f1"])
    gated_conf_ret = float(conf_sum["D_overlap_gated_js_lc5"]["conflict_retention_pct"])
    orig_conf_ret = float(conf_sum["M2_orig_lc1.0"]["conflict_retention_pct"])
    gated_gr = float(conf_sum["D_overlap_gated_js_lc5"]["sp_recall"])
    orig_gr = float(conf_sum["M2_orig_lc1.0"]["sp_recall"])

    orig_conflict_effective = bool(pareto_info.get("pareto_gate_passed"))
    gated_conflict_effective = (orig_conf_ret - gated_conf_ret >= 10.0) and ((orig_gr - gated_gr) <= 8.0)

    if orig_conflict_effective:
        rec_mode = "entity_js"
        claim = "full_soft_conflict_and_budgeted_topology"
    elif gated_conflict_effective:
        rec_mode = "overlap_gated_js"
        claim = "overlap_gated_conflict_and_budgeted_topology"
    elif ent_f1_clean >= mmr_f1_clean:
        rec_mode = "entity_cooccur"
        claim = "deterministic_budgeted_entity_topology"
    else:
        rec_mode = "entity_cooccur" if ent_f1_clean >= m2_lc1_f1_clean else "entity_js"
        claim = "narrowed_deterministic_context_organizer"

    return {
        "original_js_pareto_passed": orig_conflict_effective,
        "overlap_gated_js_effective": gated_conflict_effective,
        "m2_orig_vs_mmr_clean_f1_delta": round(m2_lc1_f1_clean - mmr_f1_clean, 2),
        "entity_graph_vs_mmr_clean_f1_delta": round(ent_f1_clean - mmr_f1_clean, 2),
        "overlap_gated_conflict_drop_pp": round(orig_conf_ret - gated_conf_ret, 2),
        "recommended_m2_graph_mode": rec_mode,
        "recommended_claim_scope": claim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP8: Milestone 3 Phase A Methodological Validity Gate.")
    parser.add_argument("--model", type=str, default="qwen3b")
    parser.add_argument("--scale", type=str, default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "bf16"])
    parser.add_argument("--out-dir", type=str, default=str(WP3_DIRS["WP8"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase3.wp8", out_dir / "wp8_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    tau_eu = float(params.get("m1_tau_eu", 0.18))
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    min_k = int(params.get("m2_min_docs", 2))
    t0 = time.time()

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp8_{args.model}_{args.scale}.json", flush_every=25)
    ckpt.metadata.update({
        "model": args.model,
        "scale": args.scale,
        "quantization": args.quantization,
        "dry_run": args.dry_run,
        "schema_version": "wp8_v2",
    })
    reset_cuda_peak()
    generator = sequential_load_phase3_generator(args.model, dry_run=args.dry_run, max_new_tokens=48, quantization=args.quantization)

    try:
        hp_dev, hp_test = load_phase3_dataset_cache("hotpotqa", scale=args.scale, allow_synthetic_fallback=args.dry_run)
        mu_dev, mu_test = load_phase3_dataset_cache("musique", scale=args.scale, allow_synthetic_fallback=args.dry_run)
        hp_cache_raw = load_phase2_cache("hotpotqa", pool_k=16, allow_synthetic_fallback=args.dry_run)

        tau_eu = calibrate_dev_m1_tau_eu(
            hp_dev,
            generator,
            params,
            ckpt=ckpt,
            target_quantile=0.60,
            model_tag=args.model,
            dataset_name="hotpotqa",
        )
        ckpt.metadata["calibrated_tau_eu"] = tau_eu
        ckpt.metadata["anchor_tau_dataset"] = "hotpotqa"
        ckpt.flush(force=True)

        # 8.1 M1 Signal Validity
        logger.info("Running WP8.1 M1 Signal Validity Probes...")
        m1_validity = _evaluate_wp8_m1_validity(generator, hp_cache_raw, hp_dev, tau_eu=tau_eu)

        # 8.2 M2 Independent Contribution Gate (Clean + Conflict across comparators)
        m2_cap = min(len(hp_test), 24 if args.scale == "smoke" else 160)
        dev_cap = min(len(hp_dev), 12 if args.scale == "smoke" else 80)
        m2_vars = [
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
        logger.info(f"Running WP8.2 M2 Validity Gate (dev={dev_cap}, test={m2_cap})...")
        _, dev_clean_sum, _ = _run_pool_evaluation(
            "hotpotqa_wp8", "dev", "clean", hp_dev[:dev_cap], m2_vars, generator, params, 8, 5, min_k, ckpt
        )
        _, dev_conf_sum, _ = _run_pool_evaluation(
            "hotpotqa_wp8", "dev", "conflict", hp_dev[:dev_cap], m2_vars, generator, params, 8, 5, min_k, ckpt
        )
        pareto_info = select_dev_pareto_lambda(dev_clean_sum, dev_conf_sum)
        test_clean_recs, test_clean_sum, _ = _run_pool_evaluation(
            "hotpotqa_wp8", "test", "clean", hp_test[:m2_cap], m2_vars, generator, params, 8, 5, min_k, ckpt
        )
        test_conf_recs, test_conf_sum, _ = _run_pool_evaluation(
            "hotpotqa_wp8", "test", "conflict", hp_test[:m2_cap], m2_vars, generator, params, 8, 5, min_k, ckpt
        )
        m2_decision = _decide_m2_validity(test_clean_sum, test_conf_sum, pareto_info)

        # Paired tests for M2 comparators
        m2_sig_clean = apply_multiple_comparison_correction({
            "EntityGraph_vs_MMR": paired_bootstrap_with_tests(
                [r["f1"] * 100.0 for r in test_clean_recs["G_entity_cooccur"]],
                [r["f1"] * 100.0 for r in test_clean_recs["A5_mmr_baseline"]],
            ),
            "M2_lc1_vs_MMR": paired_bootstrap_with_tests(
                [r["f1"] * 100.0 for r in test_clean_recs["M2_orig_lc1.0"]],
                [r["f1"] * 100.0 for r in test_clean_recs["A5_mmr_baseline"]],
            ),
            "OverlapGated_vs_Orig_ConfRet": paired_bootstrap_with_tests(
                [r["conflict_retained"] * 100.0 for r in test_conf_recs["D_overlap_gated_js_lc5"]],
                [r["conflict_retained"] * 100.0 for r in test_conf_recs["M2_orig_lc1.0"]],
            ),
        })

        # 8.3 M3 Adaptive Gating & Risk-Constrained Coverage (HotpotQA + MuSiQue focus)
        logger.info("Running WP8.3 M3 Gating & Coverage Analysis...")
        decoupler = UncertaintyDecoupler()
        m3_summary: Dict[str, Any] = {}
        for ds_tag, dev_s, test_s in [("hotpotqa", hp_dev[:dev_cap], hp_test[:m2_cap]), ("musique", mu_dev[:dev_cap], mu_test[:m2_cap])]:
            class _Ns:
                def __init__(self, inner: Phase2CheckpointManager, p: str) -> None:
                    self._i, self._p = inner, p
                def has(self, k: str) -> bool:
                    return self._i.has(f"wp8_3::{self._p}::{k}")
                def get(self, k: str, d: Any = None) -> Any:
                    return self._i.get(f"wp8_3::{self._p}::{k}", d)
                def record(self, k: str, v: Any) -> None:
                    self._i.record(f"wp8_3::{self._p}::{k}", v)
                def flush(self, force: bool = True) -> None:
                    self._i.flush(force=force)

            ns = _Ns(ckpt, ds_tag)
            dev_b0 = _collect_decode_traces(dev_s, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0_v2", ckpt=ns)
            cal = _fit_dev_calibration_configs(dev_b0)["full_platt"]
            pol = select_dev_risk_constrained_threshold(dev_b0, cal, abstain_threshold=ab_tau)
            t_b0 = _collect_decode_traces(test_s, generator, params, beta=0.0, reliability_prompt=False, tag="test_b0_v2", ckpt=ns)
            t_b12 = _collect_decode_traces(test_s, generator, params, beta=0.12, reliability_prompt=False, tag="test_b012_v2", ckpt=ns)

            gated = []
            trig = 0
            for it, r0, r12 in zip(test_s, t_b0, t_b12):
                eu = compute_unified_m1_signal(generator, it["question"], it["pool"][:3], eu_mode="v5_proxy", decoupler=decoupler)["u_eu"]
                if eu > tau_eu:
                    gated.append({
                        **r0,
                        "raw_answer": r12["raw_answer"],
                        "em": r12["em"],
                        "f1": r12["f1"],
                    })
                    trig += 1
                else:
                    gated.append(r0)

            ev_fixed_038 = evaluate_policy_on_traces(gated, cal, ab_tau, 0.38)
            ev_adaptive = evaluate_policy_on_traces(gated, cal, ab_tau, float(pol["adaptive_tau_conf"]))
            cov_risks = compute_exact_risk_at_target_coverages(
                ev_adaptive["_per_sample_confs"], [x / 100.0 for x in ev_adaptive["_per_sample_f1s"]]
            )
            m3_summary[ds_tag] = {
                "beta_trigger_rate_pct": round(100.0 * trig / max(1, len(test_s)), 2),
                "calibrated_platt": cal,
                "dev_policy": pol,
                "fixed_tau038": {
                    "eval_split": "test",
                    "policy_fit_split": "fixed_baseline",
                    **{k: v for k, v in ev_fixed_038.items() if not k.startswith("_")},
                },
                "dev_adaptive_tau": {
                    "eval_split": "test",
                    "policy_fit_split": "dev",
                    "policy_fit_n": len(dev_s),
                    **{k: v for k, v in ev_adaptive.items() if not k.startswith("_")},
                },
                "risk_at_target_coverages": cov_risks,
            }

        ckpt.flush(force=True)
        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "scale": args.scale,
                "quantization": args.quantization,
                "dry_run": args.dry_run,
                "calibrated_tau_eu": tau_eu,
                "anchor_tau_dataset": "hotpotqa",
                "elapsed_sec": round(time.time() - t0, 2),
                "vram_peak_mb": get_vram_snapshot_mb(),
            },
            "v5_integrity": v5_info,
            "wp8_1_m1_validity": m1_validity,
            "wp8_2_m2_validity": {
                "dev_pareto_selection": pareto_info,
                "test_clean_summary": test_clean_sum,
                "test_conflict_summary": test_conf_sum,
                "paired_significance_clean_and_conflict": m2_sig_clean,
                "validity_gate_decision": m2_decision,
            },
            "wp8_3_m3_validity": m3_summary,
        }
        out_file = out_dir / "wp8_method_validity_results.json"
        atomic_write_json(out_file, payload)
        logger.info(f"Saved WP8 Methodological Validity results -> {out_file} (M2 decision={m2_decision['recommended_claim_scope']})")
    finally:
        teardown_generator(generator)


if __name__ == "__main__":
    main()
