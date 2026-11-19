"""
wp3_m3_calibration.py
=====================
Phase II Work Package 3 (WP3): Independent M3 Calibration, Contrastive Decoding,
Reliability-Weighted Prompting, and Risk-Coverage (AURC / E-AURC / Abstention)
Evaluation with Zero-Leakage Dev -> Unseen Test Parameter Transfer.

Evaluates cleanly:
  - M3_0_uncalibrated: T=1.0, w_tok=1.0, beta=0.0, raw logit prob
  - M3_1_temp_only: Dev-fitted T, w_tok=1.0, platt=(1.0, 0.0)
  - M3_2_zero_intercept_platt: Dev-fitted (platt_a, b=0.0)
  - M3_3_full_platt: Dev-fitted (T, w_tok, platt_a, platt_b)
  - M3_4_v5_frozen_platt: Frozen V5 grid_search parameters evaluated on unseen test
  - M3_5_contrastive_b008: Contrastive decoding beta=0.08 + Full Platt
  - M3_6_contrastive_b012: Contrastive decoding beta=0.12 + Full Platt
  - M3_7_reliability_prompt: Reliability-tagged prompt formatting + Full Platt

Writes strictly to `results/MileStone2/WP3_M3_Calibration/`.
"""

from __future__ import annotations

import argparse
import math
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
    atomic_write_json,
    compute_calibration_and_risk_suite,
    compute_next_token_distribution,
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
from src.m3_calibrated_decoder import (
    CALIBRATION_TEMP_GRID,
    CalibratedDecoderConfig,
    M3CalibratedDecoder,
    compute_calibrated_confidence,
)

logger = setup_phase2_logger("phase2.wp3")


def _logit(p: float) -> float:
    pc = max(1e-4, min(1.0 - 1e-4, float(p)))
    return math.log(pc / (1.0 - pc))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(40.0, x)))
    ex = math.exp(max(-40.0, x))
    return ex / (1.0 + ex)


def _select_m2_docs_for_item(
    item: Dict[str, Any],
    generator: Any,
    params: Dict[str, Any],
    pool_k: int = 8,
    max_k: int = 5,
) -> List[Dict[str, Any]]:
    pool = list(item.get("pool", []))[:pool_k]
    if not pool:
        return []
    per_doc_dists = {
        str(d["doc_id"]): compute_next_token_distribution(
            generator, item["question"], [d["text"]], variant_seed=str(d["doc_id"])
        )["dist"]
        for d in pool
    }
    ordered, _ = select_m2_evidence_from_pool(
        item["question"], pool, per_doc_dists, params, pool_k=pool_k, max_k=max_k
    )
    return ordered


def _collect_decode_traces(
    items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    beta: float,
    reliability_prompt: bool,
    tag: str,
    ckpt: Phase2CheckpointManager,
    max_new_tokens: int = 48,
) -> List[Dict[str, Any]]:
    """
    Run M3 decoding with gates open (`threshold=0.0`) so raw answer, QA scores,
    and per-temperature raw context probabilities can be evaluated across calibration configs
    without re-running LLM inference.
    """
    cfg = CalibratedDecoderConfig(
        abstain_threshold=0.0,
        confidence_threshold=0.0,
        beta=float(beta),
        enable_contrastive=(float(beta) > 0.0),
        preserve_evidence_order=True,
        max_new_tokens=max_new_tokens,
        confidence_temperature=1.0,
        confidence_tok_weight=1.0,
        confidence_platt_a=1.0,
        confidence_platt_b=0.0,
        contrastive_conf_guard=float(params.get("m3_contrastive_conf_guard", 0.90)),
        protect_eos=bool(params.get("m3_protect_eos", True)),
    )
    decoder = M3CalibratedDecoder(config=cfg)
    traces: List[Dict[str, Any]] = []

    for item in items:
        iid = str(item["id"])
        ckey = f"trace::{tag}::{iid}"
        if ckpt.has(ckey):
            traces.append(ckpt.get(ckey))
            continue

        ev_docs = _select_m2_docs_for_item(item, generator, params)
        if reliability_prompt:
            ev_input = [
                {
                    **d,
                    "title": f"{d['title']} [Reliability={d['reliability']:.2f}]",
                }
                for d in ev_docs
            ]
        else:
            ev_input = ev_docs

        res = decoder.generate_calibrated(
            generator=generator,
            question=item["question"],
            evidence_items=ev_input,
        )
        qa = evaluate_qa_prediction(res.raw_answer, item["possible_answers"], is_abstained=False)
        tel = res.metadata.get("telemetry", {})
        rec = {
            "id": iid,
            "split": item.get("split", "test"),
            "raw_answer": res.raw_answer,
            "em": qa["em"],
            "f1": qa["f1"],
            "ctx_probs_by_temp": tel.get("ctx_probs_by_temp", {}),
            "max_ev_rel": float(tel.get("max_ev_rel", 0.5)),
            "mean_ev_rel": float(tel.get("mean_ev_rel", 0.5)),
            "is_faithful": bool(tel.get("is_faithful", True)),
            "has_unanchored": bool(tel.get("has_unanchored", False)),
            "raw_confidence": float(res.confidence),
        }
        ckpt.record(ckey, rec)
        traces.append(rec)

    ckpt.flush(force=True)
    return traces


def _score_trace_with_cal_params(
    trace: Dict[str, Any],
    temperature: float,
    w_tok: float,
    platt_a: float,
    platt_b: float,
) -> float:
    return float(
        compute_calibrated_confidence(
            ctx_probs_by_temp=trace.get("ctx_probs_by_temp", {}),
            max_ev_rel=float(trace.get("max_ev_rel", 0.5)),
            mean_ev_rel=float(trace.get("mean_ev_rel", 0.5)),
            temperature=temperature,
            w_tok=w_tok,
            is_faithful=bool(trace.get("is_faithful", True)),
            has_unanchored=bool(trace.get("has_unanchored", False)),
            grounding_penalty_factor=0.70,
            fallback_step_probs=[float(trace.get("raw_confidence", 0.5))],
            platt_a=platt_a,
            platt_b=platt_b,
        )
    )


def _fit_platt_nll(raw_confs: Sequence[float], labels: Sequence[int], zero_intercept: bool = False) -> Tuple[float, float]:
    """Fit logistic calibration (platt_a, platt_b) on Dev split minimizing NLL via grid + L2-regularized Newton."""
    if len(raw_confs) < 4 or len(set(labels)) < 2:
        return (1.0, 0.0)

    logits = np.asarray([_logit(p) for p in raw_confs], dtype=float)
    y = np.asarray(list(labels), dtype=float)

    best_nll = float("inf")
    best_pair = (1.0, 0.0)
    a_grid = np.linspace(0.2, 2.5, 47)
    b_grid = [0.0] if zero_intercept else np.linspace(-1.5, 1.5, 61)

    for a in a_grid:
        for b in b_grid:
            z = np.clip(a * logits + b, -30.0, 30.0)
            preds = 1.0 / (1.0 + np.exp(-z))
            nll = -float(np.mean(y * np.log(preds + 1e-9) + (1.0 - y) * np.log(1.0 - preds + 1e-9)))
            # Small regularization preventing degenerate extreme slopes on small Dev sets
            nll += 1e-3 * ((a - 1.0) ** 2 + b ** 2)
            if nll < best_nll:
                best_nll = nll
                best_pair = (round(float(a), 4), round(float(b), 4))
    return best_pair


def _fit_dev_calibration_configs(dev_traces: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Learn T, w_tok, Zero-Intercept Platt, and Full Platt strictly on Dev split."""
    if not dev_traces:
        return {
            "temp_only": {"T": 1.5, "w_tok": 1.0, "platt_a": 1.0, "platt_b": 0.0},
            "zero_intercept_platt": {"T": 2.0, "w_tok": 0.6, "platt_a": 1.0, "platt_b": 0.0},
            "full_platt": {"T": 2.0, "w_tok": 0.6, "platt_a": 1.0, "platt_b": 0.0},
        }

    labels = [1 if t["f1"] >= 0.5 else 0 for t in dev_traces]
    accs = [float(t["f1"]) for t in dev_traces]

    # 1. Fit Temp Only (w_tok=1.0, platt=(1.0, 0.0)) minimizing ECE + Brier on Dev
    best_t = 1.5
    best_score_t = float("inf")
    for temp in CALIBRATION_TEMP_GRID:
        confs = [_score_trace_with_cal_params(tr, temp, 1.0, 1.0, 0.0) for tr in dev_traces]
        suite = compute_calibration_and_risk_suite(confs, accs)
        obj = suite["ece"] + 0.5 * suite["brier_score"]
        if obj < best_score_t:
            best_score_t = obj
            best_t = float(temp)

    # 2. Fit (T, w_tok) jointly with post-Platt Brier/ECE/AUROC (parsimony favors w_tok=1.0
    # unless retrieval reliability genuinely adds discrimination without cross-branch drift).
    best_tw = (best_t, 1.0)
    best_obj_tw = float("inf")
    full_a, full_b = 1.0, 0.0
    for temp in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        for w_tok in (1.0, 0.8, 0.6):
            raw_c = [_score_trace_with_cal_params(tr, temp, w_tok, 1.0, 0.0) for tr in dev_traces]
            pa, pb = _fit_platt_nll(raw_c, labels, zero_intercept=False)
            post_c = [_score_trace_with_cal_params(tr, temp, w_tok, pa, pb) for tr in dev_traces]
            suite = compute_calibration_and_risk_suite(post_c, accs)
            obj = (
                float(suite["brier_score"])
                + 0.5 * float(suite["ece"])
                - 0.05 * float(suite["auroc_confidence"])
                + 0.01 * (1.0 - float(w_tok))
            )
            if obj < best_obj_tw:
                best_obj_tw = obj
                best_tw = (float(temp), float(w_tok))
                full_a, full_b = pa, pb

    raw_best_tw = [_score_trace_with_cal_params(tr, best_tw[0], best_tw[1], 1.0, 0.0) for tr in dev_traces]
    zi_a, zi_b = _fit_platt_nll(raw_best_tw, labels, zero_intercept=True)

    return {
        "temp_only": {"T": best_t, "w_tok": 1.0, "platt_a": 1.0, "platt_b": 0.0},
        "zero_intercept_platt": {"T": best_tw[0], "w_tok": best_tw[1], "platt_a": zi_a, "platt_b": zi_b},
        "full_platt": {"T": best_tw[0], "w_tok": best_tw[1], "platt_a": full_a, "platt_b": full_b},
    }


def _evaluate_variant_on_traces(
    traces: Sequence[Dict[str, Any]],
    temperature: float,
    w_tok: float,
    platt_a: float,
    platt_b: float,
    abstain_threshold: float,
    confidence_threshold: float,
) -> Dict[str, Any]:
    confs: List[float] = []
    raw_f1s: List[float] = []
    raw_ems: List[float] = []
    abstained: List[bool] = []
    eff_f1s: List[float] = []
    eff_ems: List[float] = []

    for tr in traces:
        conf = _score_trace_with_cal_params(tr, temperature, w_tok, platt_a, platt_b)
        is_ab = (float(tr.get("max_ev_rel", 0.5)) < abstain_threshold) or (conf < confidence_threshold)
        confs.append(conf)
        raw_f1s.append(float(tr["f1"]))
        raw_ems.append(float(tr["em"]))
        abstained.append(bool(is_ab))
        eff_f1s.append(0.0 if is_ab else float(tr["f1"]))
        eff_ems.append(0.0 if is_ab else float(tr["em"]))

    suite = compute_calibration_and_risk_suite(confs, raw_f1s, abstained_flags=abstained, n_bins=10)

    # Explicit tau_conf threshold sweep for paper Risk-Coverage table
    tau_sweep: List[Dict[str, float]] = []
    for tau in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        ab_t = [(float(tr.get("max_ev_rel", 0.5)) < abstain_threshold) or (c < tau) for tr, c in zip(traces, confs)]
        s_t = compute_calibration_and_risk_suite(confs, raw_f1s, abstained_flags=ab_t, n_bins=10)
        tau_sweep.append({
            "tau_conf": tau,
            "coverage_pct": s_t["selective_coverage_pct"],
            "abstention_rate_pct": s_t["abstention_rate_pct"],
            "selective_accuracy_pct": s_t["selective_accuracy_pct"],
            "selective_risk_pct": s_t["selective_risk_pct"],
            "harmful_abstention_rate_pct": s_t["harmful_abstention_rate_pct"],
            "valid_abstention_precision_pct": s_t["valid_abstention_precision_pct"],
        })

    n = max(1, len(traces))
    return {
        "cal_params": {
            "temperature": temperature,
            "w_tok": w_tok,
            "platt_a": platt_a,
            "platt_b": platt_b,
            "tau_abstain": abstain_threshold,
            "tau_conf": confidence_threshold,
        },
        "raw_em_pct": round(100.0 * sum(raw_ems) / n, 2),
        "raw_f1_pct": round(100.0 * sum(raw_f1s) / n, 2),
        "effective_em_pct": round(100.0 * sum(eff_ems) / n, 2),
        "effective_f1_pct": round(100.0 * sum(eff_f1s) / n, 2),
        "metrics": suite,
        "tau_conf_sweep": tau_sweep,
        "_per_sample_confs": confs,
        "_per_sample_f1s": raw_f1s,
        "_per_sample_briers": [(c - f) ** 2 for c, f in zip(confs, raw_f1s)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP3: Independent M3 Calibration & Risk-Coverage Evaluation.")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "popqa", "2wikimultihop", "musique"])
    parser.add_argument("--model", type=str, default="qwen3b")
    parser.add_argument("--dev-limit", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=120)
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["WP3"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.wp3", out_dir / "wp3_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()

    cache = load_or_build_cache(args.dataset, pool_k=16, allow_synthetic_fallback=args.dry_run)
    all_items = list(cache.get("items", []))
    dev_items = [it for it in all_items if it.get("split") == "dev"][: (15 if args.dry_run else args.dev_limit)]
    test_items = [it for it in all_items if it.get("split") == "test"][: (20 if args.dry_run else args.test_limit)]
    if not dev_items:
        dev_items = all_items[:15]
    if not test_items:
        test_items = all_items[15:35]

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp3_{args.dataset}_{args.model}.json", flush_every=25)
    reset_cuda_peak()
    generator = sequential_load_generator(args.model, dry_run=args.dry_run, max_new_tokens=48)
    t0 = time.time()

    try:
        # Step 1: Fit calibration hyperparameters strictly on Dev traces (beta=0.0 base pass)
        dev_traces_b0 = _collect_decode_traces(dev_items, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0", ckpt=ckpt)
        learned = _fit_dev_calibration_configs(dev_traces_b0)
        logger.info(f"Dev-fitted calibration params (zero test leakage): {learned}")

        # Step 2: Collect Unseen Test decode traces for distinct decoding interventions
        test_traces_b0 = _collect_decode_traces(test_items, generator, params, beta=0.0, reliability_prompt=False, tag="test_b0", ckpt=ckpt)
        test_traces_b008 = _collect_decode_traces(test_items, generator, params, beta=0.08, reliability_prompt=False, tag="test_b008", ckpt=ckpt)
        test_traces_b012 = _collect_decode_traces(test_items, generator, params, beta=0.12, reliability_prompt=False, tag="test_b012", ckpt=ckpt)
        test_traces_rel = _collect_decode_traces(test_items, generator, params, beta=0.0, reliability_prompt=True, tag="test_rel_prompt", ckpt=ckpt)

        ab_tau = float(params.get("m3_abstain_threshold", 0.05))
        conf_tau = float(params.get("m3_confidence_threshold", 0.38))

        variants_eval: Dict[str, Dict[str, Any]] = {
            "M3_0_uncalibrated": _evaluate_variant_on_traces(
                test_traces_b0, 1.0, 1.0, 1.0, 0.0, abstain_threshold=0.0, confidence_threshold=0.0
            ),
            "M3_1_temp_only": _evaluate_variant_on_traces(
                test_traces_b0, learned["temp_only"]["T"], 1.0, 1.0, 0.0, abstain_threshold=ab_tau, confidence_threshold=conf_tau
            ),
            "M3_2_zero_intercept_platt": _evaluate_variant_on_traces(
                test_traces_b0,
                learned["zero_intercept_platt"]["T"],
                learned["zero_intercept_platt"]["w_tok"],
                learned["zero_intercept_platt"]["platt_a"],
                0.0,
                abstain_threshold=ab_tau,
                confidence_threshold=conf_tau,
            ),
            "M3_3_full_platt": _evaluate_variant_on_traces(
                test_traces_b0,
                learned["full_platt"]["T"],
                learned["full_platt"]["w_tok"],
                learned["full_platt"]["platt_a"],
                learned["full_platt"]["platt_b"],
                abstain_threshold=ab_tau,
                confidence_threshold=conf_tau,
            ),
            "M3_4_v5_frozen_platt": _evaluate_variant_on_traces(
                test_traces_b0,
                float(params.get("m3_confidence_temperature", 2.0)),
                float(params.get("m3_confidence_tok_weight", 0.6)),
                float(params.get("m3_confidence_platt_a", 0.8174)),
                float(params.get("m3_confidence_platt_b", -0.2053)),
                abstain_threshold=ab_tau,
                confidence_threshold=conf_tau,
            ),
            "M3_5_contrastive_b008": _evaluate_variant_on_traces(
                test_traces_b008,
                learned["full_platt"]["T"],
                learned["full_platt"]["w_tok"],
                learned["full_platt"]["platt_a"],
                learned["full_platt"]["platt_b"],
                abstain_threshold=ab_tau,
                confidence_threshold=conf_tau,
            ),
            "M3_6_contrastive_b012": _evaluate_variant_on_traces(
                test_traces_b012,
                learned["full_platt"]["T"],
                learned["full_platt"]["w_tok"],
                learned["full_platt"]["platt_a"],
                learned["full_platt"]["platt_b"],
                abstain_threshold=ab_tau,
                confidence_threshold=conf_tau,
            ),
            "M3_7_reliability_prompt": _evaluate_variant_on_traces(
                test_traces_rel,
                learned["full_platt"]["T"],
                learned["full_platt"]["w_tok"],
                learned["full_platt"]["platt_a"],
                learned["full_platt"]["platt_b"],
                abstain_threshold=ab_tau,
                confidence_threshold=conf_tau,
            ),
        }

        # Paired tests comparing Full Platt & Contrastive vs Uncalibrated baseline
        sig_tests: Dict[str, Any] = {}
        u_briers = variants_eval["M3_0_uncalibrated"]["_per_sample_briers"]
        u_f1s = [x * 100.0 for x in variants_eval["M3_0_uncalibrated"]["_per_sample_f1s"]]
        for key in ("M3_3_full_platt", "M3_4_v5_frozen_platt", "M3_5_contrastive_b008", "M3_6_contrastive_b012", "M3_7_reliability_prompt"):
            v_briers = variants_eval[key]["_per_sample_briers"]
            v_f1s = [x * 100.0 for x in variants_eval[key]["_per_sample_f1s"]]
            sig_tests[f"{key}_vs_uncalibrated"] = {
                "brier_delta": paired_bootstrap_with_tests(v_briers, u_briers),
                "raw_f1_pct_delta": paired_bootstrap_with_tests(v_f1s, u_f1s),
            }

        clean_variants = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in variants_eval.items()
        }

        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": args.dataset,
                "model": args.model,
                "n_dev": len(dev_items),
                "n_test_unseen": len(test_items),
                "dry_run": args.dry_run,
                "elapsed_sec": round(time.time() - t0, 2),
                "vram_peak_mb": get_vram_snapshot_mb(),
            },
            "v5_integrity": v5_info,
            "dev_learned_calibration": learned,
            "unseen_test_variants": clean_variants,
            "paired_significance_vs_uncalibrated": sig_tests,
        }
        out_file = out_dir / f"wp3_m3_calibration_{args.dataset}.json"
        atomic_write_json(out_file, payload)
        logger.info(f"Saved WP3 results -> {out_file}")
    finally:
        teardown_generator(generator)


if __name__ == "__main__":
    main()
