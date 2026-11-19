"""
p0_3_m3_adaptive_gating.py
==========================
MileStone 2 P0-3 & P0-4 Closure Experiment:
  1. Uncertainty-Gated Contrastive Decoding:
       beta(q) = 0.0  if U_EU^v5(q) <= tau_EU
               = 0.12 if U_EU^v5(q) >  tau_EU
     using unified V5 Proxy M1 routing signal (P0-2 alignment), compared strictly
     against Fixed_Beta_0 and Fixed_Beta_012 on the exact same M2-selected pools.
  2. Dev Risk-Constrained Coverage Calibration (eliminating MuSiQue 61.7%-82.5%
     over-abstention caused by a single static threshold tau_conf=0.38):
       - Learns temperature/Platt calibration strictly on Dev (N=80)
       - Evaluates Risk across target coverages {0.4, 0.5, 0.6, 0.7, 0.8}
       - Selects adaptive Dev Risk-Constrained Threshold (zero test leakage)
  3. Expands unseen test sample sizes across all 4 benchmarks (P0-4):
       PopQA (N=120), HotpotQA (N=320 >= 300), 2WikiMultiHopQA (N=220 >= 200),
       MuSiQue (N=220 >= 200)
     and attaches Paired Bootstrap 95% CIs + Holm-Bonferroni / BH-FDR correction.

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
    compute_calibration_and_risk_suite,
    compute_unified_m1_signal,
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
from scripts.phase2.wp3_m3_calibration import (
    _collect_decode_traces,
    _fit_dev_calibration_configs,
    _score_trace_with_cal_params,
)
from src.m1_adaptive_retriever import UncertaintyDecoupler

logger = setup_phase2_logger("phase2.p0_3")

TARGET_COVERAGES: Tuple[float, ...] = (0.80, 0.70, 0.60, 0.50, 0.40)


def select_dev_risk_constrained_threshold(
    dev_traces: Sequence[Dict[str, Any]],
    cal_params: Dict[str, float],
    abstain_threshold: float = 0.05,
    min_target_coverage: float = 0.75,
    low_risk_regime_cap: float = 0.10,
) -> Dict[str, Any]:
    """
    Learn a domain-adaptive abstention threshold tau_conf* strictly from Dev predictions
    without dataset-specific hardcoding:
      - In high-accuracy regimes (un-abstained Dev risk <= low_risk_regime_cap), preserve
        coverage >= 0.90 and pick threshold in low tail maximizing net selective utility
        (Valid Abstention - 1.5 * Harmful Abstention) so accurate tasks avoid false abstention.
      - In harder multi-hop regimes, scan candidate Dev quantiles yielding Dev Coverage
        in [min_target_coverage, 0.95] (plus 0.38 if feasible within coverage floor) and
        select threshold minimizing Dev Selective Risk (tie-broken by highest Coverage).
    Also returns Dev quantile thresholds for fixed target coverages {0.80, 0.70, 0.60, 0.50, 0.40}.
    """
    if not dev_traces:
        return {
            "adaptive_tau_conf": 0.38,
            "regime": "empty_fallback",
            "dev_base_risk_pct": 0.0,
            "dev_selected_coverage_pct": 100.0,
            "dev_selected_risk_pct": 0.0,
            "coverage_quantile_taus": {f"cov_{int(c*100)}": 0.38 for c in TARGET_COVERAGES},
        }

    dev_confs = [
        _score_trace_with_cal_params(
            tr,
            cal_params["T"],
            cal_params["w_tok"],
            cal_params["platt_a"],
            cal_params["platt_b"],
        )
        for tr in dev_traces
    ]
    dev_f1s = [float(tr["f1"]) for tr in dev_traces]
    dev_rels = [float(tr.get("max_ev_rel", 0.5)) for tr in dev_traces]
    n = len(dev_traces)

    cov_taus = {
        f"cov_{int(round(c * 100))}": round(float(np.quantile(dev_confs, max(0.0, min(1.0, 1.0 - c)))), 6)
        for c in TARGET_COVERAGES
    }

    base_ab = [rel < abstain_threshold for rel in dev_rels]
    base_suite = compute_calibration_and_risk_suite(dev_confs, dev_f1s, abstained_flags=base_ab, n_bins=10)
    base_risk = float(base_suite["selective_risk_pct"]) / 100.0

    def _eval_tau(tau: float) -> Tuple[Dict[str, Any], float]:
        ab = [(rel < abstain_threshold) or (c < tau) for rel, c in zip(dev_rels, dev_confs)]
        s = compute_calibration_and_risk_suite(dev_confs, dev_f1s, abstained_flags=ab, n_bins=10)
        n_valid = sum(1 for a, f in zip(ab, dev_f1s) if a and f < 0.5)
        n_harm = sum(1 for a, f in zip(ab, dev_f1s) if a and f >= 0.5)
        net_util = (n_valid - 1.5 * n_harm) / max(1, n)
        return s, net_util

    if base_risk <= low_risk_regime_cap:
        # High-accuracy domain (e.g., single-hop QA): only abstain on extreme low-confidence outliers
        cand_taus = sorted(set([0.05, 0.15, 0.25, 0.35] + [float(np.quantile(dev_confs, q)) for q in (0.02, 0.05, 0.08, 0.10)]))
        best_tau = 0.05
        best_score = -1e9
        best_suite = base_suite
        for tau in cand_taus:
            s, net_u = _eval_tau(tau)
            if (s["selective_coverage_pct"] / 100.0) >= 0.90:
                if net_u > best_score or (abs(net_u - best_score) < 1e-9 and s["selective_coverage_pct"] > best_suite["selective_coverage_pct"]):
                    best_score = net_u
                    best_tau = float(tau)
                    best_suite = s
        regime = "low_risk_outlier_guard"
    else:
        # Complex/hard domain: search coverage band [min_target_coverage, 0.95] on Dev
        q_cands = [float(np.quantile(dev_confs, q)) for q in np.linspace(0.05, max(0.05, 1.0 - min_target_coverage), 9)]
        cand_taus = sorted(set(q_cands + [0.38]))
        best_tau = q_cands[len(q_cands) // 2]
        best_risk = 1e9
        best_suite = base_suite
        found = False
        for tau in cand_taus:
            s, _ = _eval_tau(tau)
            cov = float(s["selective_coverage_pct"]) / 100.0
            risk = float(s["selective_risk_pct"])
            if min_target_coverage - 1e-6 <= cov <= 0.96:
                found = True
                if risk < best_risk - 1e-6 or (abs(risk - best_risk) <= 1e-6 and cov > (best_suite["selective_coverage_pct"] / 100.0)):
                    best_risk = risk
                    best_tau = float(tau)
                    best_suite = s
        if not found:
            best_tau = cov_taus["cov_80"]
            best_suite, _ = _eval_tau(best_tau)
        regime = "risk_constrained_coverage"

    return {
        "adaptive_tau_conf": round(float(best_tau), 6),
        "regime": regime,
        "dev_base_risk_pct": round(base_risk * 100.0, 2),
        "dev_selected_coverage_pct": best_suite["selective_coverage_pct"],
        "dev_selected_risk_pct": best_suite["selective_risk_pct"],
        "coverage_quantile_taus": cov_taus,
    }


def evaluate_policy_on_traces(
    traces: Sequence[Dict[str, Any]],
    cal_params: Dict[str, float],
    abstain_threshold: float,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Evaluate one (decoding trace sequence, calibration, tau_conf) configuration."""
    confs: List[float] = []
    raw_f1s: List[float] = []
    raw_ems: List[float] = []
    abstained: List[bool] = []
    eff_f1s: List[float] = []
    eff_ems: List[float] = []

    for tr in traces:
        conf = _score_trace_with_cal_params(
            tr,
            cal_params["T"],
            cal_params["w_tok"],
            cal_params["platt_a"],
            cal_params["platt_b"],
        )
        is_ab = (float(tr.get("max_ev_rel", 0.5)) < abstain_threshold) or (conf < confidence_threshold)
        confs.append(conf)
        raw_f1s.append(float(tr["f1"]))
        raw_ems.append(float(tr["em"]))
        abstained.append(bool(is_ab))
        eff_f1s.append(0.0 if is_ab else float(tr["f1"]))
        eff_ems.append(0.0 if is_ab else float(tr["em"]))

    suite = compute_calibration_and_risk_suite(confs, raw_f1s, abstained_flags=abstained, n_bins=10)
    n = max(1, len(traces))
    return {
        "n": len(traces),
        "tau_conf": round(float(confidence_threshold), 6),
        "raw_em_pct": round(100.0 * sum(raw_ems) / n, 2),
        "raw_f1_pct": round(100.0 * sum(raw_f1s) / n, 2),
        "effective_em_pct": round(100.0 * sum(eff_ems) / n, 2),
        "effective_f1_pct": round(100.0 * sum(eff_f1s) / n, 2),
        "ece": suite["ece"],
        "mce": suite["mce"],
        "brier_score": suite["brier_score"],
        "auroc_confidence": suite["auroc_confidence"],
        "auprc_error": suite["auprc_error"],
        "aurc": suite["aurc"],
        "e_aurc": suite["e_aurc"],
        "coverage_pct": suite["selective_coverage_pct"],
        "abstention_rate_pct": suite["abstention_rate_pct"],
        "selective_accuracy_pct": suite["selective_accuracy_pct"],
        "selective_risk_pct": suite["selective_risk_pct"],
        "valid_abstention_precision_pct": suite["valid_abstention_precision_pct"],
        "harmful_abstention_rate_pct": suite["harmful_abstention_rate_pct"],
        "mean_confidence": suite["mean_confidence"],
        "_per_sample_f1s": [x * 100.0 for x in raw_f1s],
        "_per_sample_ems": [x * 100.0 for x in raw_ems],
        "_per_sample_briers": [(c - f) ** 2 for c, f in zip(confs, raw_f1s)],
        "_per_sample_confs": confs,
    }


def compute_exact_risk_at_target_coverages(
    confs: Sequence[float],
    f1s: Sequence[float],
    targets: Sequence[float] = TARGET_COVERAGES,
) -> Dict[str, float]:
    """Compute prefix selective risk at exact coverage fractions {0.8, 0.7, 0.6, 0.5, 0.4}."""
    n = len(confs)
    if n == 0:
        return {}
    pairs = sorted(zip(confs, f1s), key=lambda p: p[0], reverse=True)
    out: Dict[str, float] = {}
    for cov in sorted(targets):
        k = max(1, min(n, int(round(cov * n))))
        sel_f1 = float(np.mean([f for _, f in pairs[:k]]))
        out[f"risk_at_cov_{int(round(cov * 100))}"] = round((1.0 - sel_f1) * 100.0, 2)
    return out


def _warm_from_existing_wp3_and_wp4(
    dataset_name: str,
    model_key: str,
    ckpt: Phase2CheckpointManager,
) -> None:
    """Safely seed read-only matching traces/EU signals from WP3/WP4 checkpoints if present."""
    wp3_ckpt_path = WP_DIRS["WP3"] / f"ckpt_wp3_{dataset_name}_{model_key}.json"
    if wp3_ckpt_path.exists():
        wp3_recs = load_json_file(wp3_ckpt_path).get("records", {})
        for k, v in wp3_recs.items():
            if k.startswith(("trace::dev_b0::", "trace::test_b0::", "trace::test_b012::")):
                ns_key = f"{dataset_name}::{k}"
                if not ckpt.has(ns_key):
                    ckpt.record(ns_key, v)

    wp4_ckpt_path = WP_DIRS["WP4"] / f"ckpt_wp4_{model_key}.json"
    if wp4_ckpt_path.exists():
        wp4_recs = load_json_file(wp4_ckpt_path).get("records", {})
        prefix = f"eu::{dataset_name}::"
        for k, v in wp4_recs.items():
            if k.startswith(prefix):
                if not ckpt.has(k):
                    ckpt.record(k, v)


def _get_or_compute_v5_eu(
    dataset_name: str,
    items: Sequence[Dict[str, Any]],
    generator: Any,
    decoupler: UncertaintyDecoupler,
    ckpt: Phase2CheckpointManager,
    pool_k: int = 8,
) -> Dict[str, float]:
    """Retrieve or compute V5 Proxy U_EU(q, D_0) for items consistently with WP4."""
    eu_map: Dict[str, float] = {}
    for item in items:
        iid = str(item["id"])
        ckey = f"eu::{dataset_name}::{iid}"
        if ckpt.has(ckey):
            eu_map[iid] = float(ckpt.get(ckey)["u_eu"])
            continue
        d0_docs = list(item.get("pool", []))[: min(3, pool_k)]
        sig = compute_unified_m1_signal(generator, item["question"], d0_docs, eu_mode="v5_proxy", decoupler=decoupler)
        eu_map[iid] = float(sig["u_eu"])
        ckpt.record(ckey, {"u_eu": float(sig["u_eu"]), "u_au": float(sig["u_au"]), "eu_mode": "v5_proxy"})
    ckpt.flush(force=True)
    return eu_map


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-3: M3 Adaptive Gating & Risk-Constrained Coverage Evaluation.")
    parser.add_argument("--models", nargs="+", default=["qwen3b"])
    parser.add_argument("--datasets", nargs="+", default=["popqa", "hotpotqa", "2wikimultihop", "musique"])
    parser.add_argument("--dev-limit", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=0, help="0 = ALL unseen test items (120 PopQA, 320 Hotpot, 220 2Wiki, 220 MuSiQue)")
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["P0"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.p0_3", out_dir / "p0_3_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    tau_eu = float(params.get("m1_tau_eu", 0.18))
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    t0 = time.time()

    all_models_out: Dict[str, Any] = {}
    decoupler = UncertaintyDecoupler()

    for m_key in args.models:
        ckpt = Phase2CheckpointManager(out_dir / f"ckpt_p0_3_{m_key}.json", flush_every=25)
        reset_cuda_peak()
        generator = sequential_load_generator(m_key, dry_run=args.dry_run, max_new_tokens=48)
        per_ds_out: Dict[str, Any] = {}
        cross_ds_gated_vs_b0_f1: Dict[str, Any] = {}
        cross_ds_gated_vs_b012_f1: Dict[str, Any] = {}
        cross_ds_gated_vs_b0_brier: Dict[str, Any] = {}

        try:
            for dname in args.datasets:
                if not args.dry_run:
                    _warm_from_existing_wp3_and_wp4(dname, m_key, ckpt)
                cache = load_or_build_cache(dname, pool_k=16, allow_synthetic_fallback=args.dry_run)
                all_items = list(cache.get("items", []))
                dev_items = [it for it in all_items if it.get("split") == "dev"]
                test_items = [it for it in all_items if it.get("split") == "test"]
                if args.dry_run:
                    dev_items = dev_items[:12] or all_items[:12]
                    test_items = test_items[:16] or all_items[12:28]
                else:
                    dev_items = dev_items[: args.dev_limit] if args.dev_limit > 0 else dev_items
                    test_items = test_items[: args.test_limit] if args.test_limit > 0 else test_items

                logger.info(f"[{m_key} | {dname}] Dev N={len(dev_items)}, Unseen Test N={len(test_items)}")

                # Build a dataset-namespaced checkpoint wrapper view so reused trace keys never collide
                class _NsCkpt:
                    def __init__(self, inner: Phase2CheckpointManager, prefix: str) -> None:
                        self._inner = inner
                        self._prefix = prefix
                    def has(self, key: str) -> bool:
                        return self._inner.has(f"{self._prefix}::{key}")
                    def get(self, key: str, default: Any = None) -> Any:
                        return self._inner.get(f"{self._prefix}::{key}", default)
                    def record(self, key: str, value: Any) -> None:
                        self._inner.record(f"{self._prefix}::{key}", value)
                    def flush(self, force: bool = True) -> None:
                        self._inner.flush(force=force)

                ns_ckpt = _NsCkpt(ckpt, dname)

                # Step 1: Dev traces & Dev calibration + Dev risk-constrained threshold (Zero test leakage)
                dev_traces_b0 = _collect_decode_traces(dev_items, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0", ckpt=ns_ckpt)
                learned_cal = _fit_dev_calibration_configs(dev_traces_b0)
                full_platt = learned_cal["full_platt"]
                dev_thresh_info = select_dev_risk_constrained_threshold(dev_traces_b0, full_platt, abstain_threshold=ab_tau)
                adaptive_tau = float(dev_thresh_info["adaptive_tau_conf"])

                # Step 2: Unseen Test traces (beta=0.0 & beta=0.12) + Unified M1 V5 Proxy U_EU(q)
                test_traces_b0 = _collect_decode_traces(test_items, generator, params, beta=0.0, reliability_prompt=False, tag="test_b0", ckpt=ns_ckpt)
                test_traces_b012 = _collect_decode_traces(test_items, generator, params, beta=0.12, reliability_prompt=False, tag="test_b012", ckpt=ns_ckpt)
                eu_map = _get_or_compute_v5_eu(dname, test_items, generator, decoupler, ckpt)

                gated_traces: List[Dict[str, Any]] = []
                triggered_count = 0
                for tr0, tr12 in zip(test_traces_b0, test_traces_b012):
                    iid = str(tr0["id"])
                    if eu_map.get(iid, 0.0) > tau_eu:
                        gated_traces.append(tr12)
                        triggered_count += 1
                    else:
                        gated_traces.append(tr0)

                # Step 3: Evaluate combinations strictly on unseen Test
                evals = {
                    "Fixed_B0__Tau038": evaluate_policy_on_traces(test_traces_b0, full_platt, ab_tau, 0.38),
                    "Fixed_B012__Tau038": evaluate_policy_on_traces(test_traces_b012, full_platt, ab_tau, 0.38),
                    "Gated_Beta__Tau038": evaluate_policy_on_traces(gated_traces, full_platt, ab_tau, 0.38),
                    "Fixed_B0__DevAdaptiveTau": evaluate_policy_on_traces(test_traces_b0, full_platt, ab_tau, adaptive_tau),
                    "Fixed_B012__DevAdaptiveTau": evaluate_policy_on_traces(test_traces_b012, full_platt, ab_tau, adaptive_tau),
                    "Gated_Beta__DevAdaptiveTau": evaluate_policy_on_traces(gated_traces, full_platt, ab_tau, adaptive_tau),
                }

                # Target Coverage Sweep table {0.8, 0.7, 0.6, 0.5, 0.4} using Dev-fitted quantile taus & prefix risk
                cov_quantile_eval = {}
                for c_lbl, q_tau in dev_thresh_info["coverage_quantile_taus"].items():
                    cov_quantile_eval[c_lbl] = {
                        k: v
                        for k, v in evaluate_policy_on_traces(gated_traces, full_platt, ab_tau, q_tau).items()
                        if not k.startswith("_")
                    }

                exact_prefix_risk = {
                    mode_tag: compute_exact_risk_at_target_coverages(
                        evals[f"{mode_tag}__DevAdaptiveTau"]["_per_sample_confs"],
                        [x / 100.0 for x in evals[f"{mode_tag}__DevAdaptiveTau"]["_per_sample_f1s"]],
                    )
                    for mode_tag in ("Fixed_B0", "Fixed_B012", "Gated_Beta")
                }

                g_adp = evals["Gated_Beta__DevAdaptiveTau"]
                b0_adp = evals["Fixed_B0__DevAdaptiveTau"]
                b12_adp = evals["Fixed_B012__DevAdaptiveTau"]
                cross_ds_gated_vs_b0_f1[dname] = paired_bootstrap_with_tests(g_adp["_per_sample_f1s"], b0_adp["_per_sample_f1s"])
                cross_ds_gated_vs_b012_f1[dname] = paired_bootstrap_with_tests(g_adp["_per_sample_f1s"], b12_adp["_per_sample_f1s"])
                cross_ds_gated_vs_b0_brier[dname] = paired_bootstrap_with_tests(g_adp["_per_sample_briers"], b0_adp["_per_sample_briers"])

                per_ds_out[dname] = {
                    "n_dev": len(dev_items),
                    "n_test_unseen": len(test_items),
                    "tau_eu": tau_eu,
                    "beta_trigger_rate_pct": round(100.0 * triggered_count / max(1, len(test_items)), 2),
                    "dev_full_platt_params": full_platt,
                    "dev_threshold_policy": dev_thresh_info,
                    "test_policies": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in evals.items()},
                    "test_dev_quantile_coverage_sweep": cov_quantile_eval,
                    "exact_risk_at_target_coverages": exact_prefix_risk,
                }

            all_models_out[m_key] = {
                "vram_peak_mb": get_vram_snapshot_mb(),
                "datasets": per_ds_out,
                "paired_significance_across_datasets": {
                    "Gated_Beta_vs_Fixed_B0_f1_pct": apply_multiple_comparison_correction(cross_ds_gated_vs_b0_f1),
                    "Gated_Beta_vs_Fixed_B012_f1_pct": apply_multiple_comparison_correction(cross_ds_gated_vs_b012_f1),
                    "Gated_Beta_vs_Fixed_B0_brier": apply_multiple_comparison_correction(cross_ds_gated_vs_b0_brier),
                },
            }
        finally:
            teardown_generator(generator)

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": args.models,
            "datasets": args.datasets,
            "eu_mode": "v5_proxy",
            "tau_eu": tau_eu,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "v5_integrity": v5_info,
        "models": all_models_out,
    }
    out_file = out_dir / "p0_3_m3_adaptive_gating.json"
    atomic_write_json(out_file, payload)
    logger.info(f"Saved P0-3 adaptive gating results -> {out_file}")


if __name__ == "__main__":
    main()
